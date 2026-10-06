include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: ZDirection
using Oceananigans.TimeSteppers: implicit_step!
using Oceananigans.TurbulenceClosures: VerticallyImplicitDiffusionLowerDiagonal,
                                       VerticallyImplicitDiffusionDiagonal,
                                       VerticallyImplicitDiffusionUpperDiagonal
using Test

@testset "Vertically implicit diffusion correctness [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 32
    Lz = FT(100)
    grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100), y=(0, 100), z=(0, Lz))
    vitd = VerticallyImplicitTimeDiscretization()
    etd = Oceananigans.TurbulenceClosures.ExplicitTimeDiscretization()

    # Cosine profile: c(z) = cos(π z / Lz)
    # Analytical solution for diffusion: c(z,t) = cos(π z / Lz) * exp(-κ (π/Lz)² t)
    # This satisfies zero-flux BCs at z=0 and z=Lz
    k = FT(π) / Lz
    cosine(z) = cos(k * z)

    # Analytical decay factor after time t with diffusivity κ
    analytical_decay(κ, t) = exp(-κ * k^2 * t)

    @testset "Implicit scalar diffusion matches analytical solution" begin
        κ = FT(10)
        Δt = FT(1)
        Nt = 10
        t_final = Δt * Nt

        closure = VerticalScalarDiffusivity(vitd; κ)
        model = AtmosphereModel(grid; closure, advection=nothing, tracers=:ρc)

        set!(model; ρc = (x, y, z) -> cosine(z))
        ρc₀ = sum(interior(model.tracers.ρc) .^ 2)

        for _ in 1:Nt
            time_step!(model, Δt)
        end

        # Compare numerical decay to analytical decay
        ρc₁ = sum(interior(model.tracers.ρc) .^ 2)
        numerical_decay = sqrt(ρc₁ / ρc₀)
        expected_decay = analytical_decay(κ, t_final)

        @test isapprox(numerical_decay, expected_decay, rtol=0.05)
    end

    @testset "Implicit and explicit diffusion match analytical solution" begin
        κ = FT(1)
        Δt = FT(0.5)
        Nt = 10
        t_final = Δt * Nt

        implicit_closure = VerticalScalarDiffusivity(vitd; κ)
        explicit_closure = VerticalScalarDiffusivity(etd; κ)

        implicit_model = AtmosphereModel(grid; closure=implicit_closure, advection=nothing, tracers=:ρc)
        explicit_model = AtmosphereModel(grid; closure=explicit_closure, advection=nothing, tracers=:ρc)

        set!(implicit_model; ρc = (x, y, z) -> cosine(z))
        set!(explicit_model; ρc = (x, y, z) -> cosine(z))

        ρc₀_implicit = sum(interior(implicit_model.tracers.ρc) .^ 2)
        ρc₀_explicit = sum(interior(explicit_model.tracers.ρc) .^ 2)

        for _ in 1:Nt
            time_step!(implicit_model, Δt)
            time_step!(explicit_model, Δt)
        end

        ρc₁_implicit = sum(interior(implicit_model.tracers.ρc) .^ 2)
        ρc₁_explicit = sum(interior(explicit_model.tracers.ρc) .^ 2)

        numerical_decay_implicit = sqrt(ρc₁_implicit / ρc₀_implicit)
        numerical_decay_explicit = sqrt(ρc₁_explicit / ρc₀_explicit)
        expected_decay = analytical_decay(κ, t_final)

        # Both should match analytical solution
        @test isapprox(numerical_decay_implicit, expected_decay, rtol=0.05)
        @test isapprox(numerical_decay_explicit, expected_decay, rtol=0.05)

        # And they should match each other closely
        @test isapprox(numerical_decay_implicit, numerical_decay_explicit, rtol=0.01)
    end

    @testset "Implicit viscosity matches analytical solution" begin
        ν = FT(10)
        Δt = FT(1)
        Nt = 10
        t_final = Δt * Nt

        closure = VerticalScalarDiffusivity(vitd; ν)
        model = AtmosphereModel(grid; closure, advection=nothing)

        set!(model; ρu = (x, y, z) -> cosine(z))
        ρu₀ = sum(interior(model.momentum.ρu) .^ 2)

        for _ in 1:Nt
            time_step!(model, Δt)
        end

        ρu₁ = sum(interior(model.momentum.ρu) .^ 2)
        numerical_decay = sqrt(ρu₁ / ρu₀)
        expected_decay = analytical_decay(ν, t_final)

        @test isapprox(numerical_decay, expected_decay, rtol=0.05)
    end

    @testset "Implicit diffusion with both ν and κ matches analytical solutions" begin
        ν = FT(5)
        κ = FT(10)
        Δt = FT(1)
        Nt = 10
        t_final = Δt * Nt

        closure = VerticalScalarDiffusivity(vitd; ν, κ)
        model = AtmosphereModel(grid; closure, advection=nothing, tracers=:ρc)

        set!(model; ρu = (x, y, z) -> cosine(z), ρc = (x, y, z) -> cosine(z))

        ρu₀ = sum(interior(model.momentum.ρu) .^ 2)
        ρc₀ = sum(interior(model.tracers.ρc) .^ 2)

        for _ in 1:Nt
            time_step!(model, Δt)
        end

        ρu₁ = sum(interior(model.momentum.ρu) .^ 2)
        ρc₁ = sum(interior(model.tracers.ρc) .^ 2)

        numerical_decay_u = sqrt(ρu₁ / ρu₀)
        numerical_decay_c = sqrt(ρc₁ / ρc₀)
        expected_decay_u = analytical_decay(ν, t_final)
        expected_decay_c = analytical_decay(κ, t_final)

        @test isapprox(numerical_decay_u, expected_decay_u, rtol=0.05)
        @test isapprox(numerical_decay_c, expected_decay_c, rtol=0.05)
    end
end

#####
##### Density weighting of the implicit solve
#####
##### The prognostics are density weighted (`ρc`), and the explicit flux divergence forms
##### `∂z(ρ⁰ κ ∂z c)` on the specific variable. The implicit solve must form the same operator.
##### Solving the unweighted `∂t(ρ⁰c) = ∂z(κ ∂z(ρ⁰c))` instead relaxes to `ρ⁰c = const`, i.e.
##### `c ∝ 1/ρ⁰`, rather than to a well-mixed `c = const` — worth `≈ +20 K/km` on a deep column.
##### The tests above run on `Lz = 100 m`, over which `ρ⁰` varies < 1%, which is why they pass
##### either way; these use a 4 km column, over which it varies ≈ 40%.
#####

@testset "Density weighting of the vertically implicit solve [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 32
    Lz = FT(4000)
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 100), y=(0, 100), z=(0, Lz),
                           topology=(Periodic, Periodic, Bounded))

    # A second grid stretched in z, for the tests that do not need explicit stability, so the
    # coefficients are exercised with location-dependent Δz as well as location-dependent ρ⁰.
    stretched_grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 100), y=(0, 100),
                                     z = [Lz * (FT(k - 1) / Nz)^FT(1.3) for k = 1:Nz+1],
                                     topology=(Periodic, Periodic, Bounded))

    vitd = VerticallyImplicitTimeDiscretization()
    etd = Oceananigans.TurbulenceClosures.ExplicitTimeDiscretization()

    ρ⁰(model) = Array(interior(Breeze.AtmosphereModels.total_density(model.dynamics), 1, 1, :))

    # A profile with structure to relax, and zero flux through both boundaries by default.
    initial(z) = 1 + cos(FT(π) * z / Lz)

    # κ t must be large enough for the diffusion length √(2κt) ≈ 1 km to be a decent fraction of
    # the density scale height (≈ 8.5 km) — that ratio is the size of the effect under test.
    @testset "Deep-column implicit matches explicit" begin
        κ = FT(200)
        Δt = FT(25)   # explicit stability needs Δt < Δz²/2κ = 39 s
        Nt = 100

        implicit_model = AtmosphereModel(grid; closure = VerticalScalarDiffusivity(vitd; κ),
                                         advection = nothing, tracers = :ρc)
        explicit_model = AtmosphereModel(grid; closure = VerticalScalarDiffusivity(etd; κ),
                                         advection = nothing, tracers = :ρc)

        for model in (implicit_model, explicit_model)
            set!(model; ρc = (x, y, z) -> initial(z))
        end

        for _ in 1:Nt
            time_step!(implicit_model, Δt)
            time_step!(explicit_model, Δt)
        end

        ρcⁱ = Array(interior(implicit_model.tracers.ρc, 1, 1, :))
        ρcᵉ = Array(interior(explicit_model.tracers.ρc, 1, 1, :))

        # The initial profile is a single smooth mode, so backward and forward Euler agree to
        # O(κ k² Δt) ≈ 0.3% here; the unweighted solve is wrong by ≈ 10%.
        @test maximum(abs, ρcⁱ .- ρcᵉ) / maximum(abs, ρcᵉ) < 0.02
    end

    @testset "Deep-column steady state is well mixed in the specific variable" begin
        κ = FT(200)
        Δt = FT(2000)
        Nt = 200      # κ t / Lz² ≈ 25: far past equilibration

        model = AtmosphereModel(stretched_grid; closure = VerticalScalarDiffusivity(vitd; κ),
                                advection = nothing, tracers = :ρc)
        set!(model; ρc = (x, y, z) -> initial(z))

        for _ in 1:Nt
            time_step!(model, Δt)
        end

        ρc = Array(interior(model.tracers.ρc, 1, 1, :))
        c  = ρc ./ ρ⁰(model)

        # The specific variable is well mixed ...
        @test (maximum(c) - minimum(c)) / abs(sum(c) / Nz) < 1e-3

        # ... which the unweighted operator cannot do: it relaxes ρc, not c, to a constant, and
        # ρ⁰ varies by ≈ 40% over this column.
        @test (maximum(ρc) - minimum(ρc)) / abs(sum(ρc) / Nz) > 0.1
    end

    @testset "Deep-column mass conservation" begin
        κ = FT(50)
        Δt = FT(500)
        Nt = 50

        model = AtmosphereModel(stretched_grid; closure = VerticalScalarDiffusivity(vitd; κ),
                                advection = nothing, tracers = :ρc)
        set!(model; ρc = (x, y, z) -> initial(z))

        ## One bulk copy of the face positions rather than `Nz` calls to a grid operator: on a
        ## stretched grid `Δz` is an array, so the comprehension was a scalar read per cell and
        ## errored on a GPU. `diff` of the faces is exactly `Δzᶜᶜᶜ`, and needs no `@allowscalar`
        ## — which would only have permitted the reads, leaving the same host-side loop.
        Δzᶜ = diff(Array(znodes(stretched_grid, Face())))
        mass(m) = sum(Δzᶜ .* Array(interior(m.tracers.ρc, 1, 1, :)))
        mass₀ = mass(model)

        for _ in 1:Nt
            time_step!(model, Δt)
        end

        # Σ Δz ρ⁰c is conserved under zero surface flux only if the tridiagonal diagonal is
        # written out explicitly; `1 - du - dl` leaks mass once the off-diagonals carry a
        # location-dependent density prefactor.
        @test isapprox(mass(model), mass₀, rtol = 100 * eps(FT))
    end

    @testset "Deep-column implicit viscosity matches explicit" begin
        ν = FT(200)
        Δt = FT(25)
        Nt = 100

        implicit_model = AtmosphereModel(grid; closure = VerticalScalarDiffusivity(vitd; ν),
                                         advection = nothing)
        explicit_model = AtmosphereModel(grid; closure = VerticalScalarDiffusivity(etd; ν),
                                         advection = nothing)

        for model in (implicit_model, explicit_model)
            set!(model; ρu = (x, y, z) -> initial(z))
        end

        for _ in 1:Nt
            time_step!(implicit_model, Δt)
            time_step!(explicit_model, Δt)
        end

        ρuⁱ = Array(interior(implicit_model.momentum.ρu, 1, 1, :))
        ρuᵉ = Array(interior(explicit_model.momentum.ρu, 1, 1, :))

        @test maximum(abs, ρuⁱ .- ρuᵉ) / maximum(abs, ρuᵉ) < 0.02
    end
end

#####
##### Density weighting at z-Faces (`ρw`)
#####
##### `ρw` is the one prognostic whose specific variable `w = ρw / ρᶠ` is reconstructed at faces
##### while its stress `ρ ν ∂z w` lives at centers — the opposite of a tracer — so its rows carry
##### `ρᶜ / ρᶠ` where the z-Center rows carry `ρᶠ / ρᶜ`. These exercise the coefficients directly
##### rather than through a time step: a single-column anelastic model projects `ρw` to zero, so
##### a trajectory comparison would say nothing about the operator.
#####

@testset "Density weighting of the z-Face implicit solve [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 32
    Lz = FT(4000)

    uniform_grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 100), y=(0, 100), z=(0, Lz),
                                   topology=(Periodic, Periodic, Bounded))

    stretched_grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 100), y=(0, 100),
                                     z = [Lz * (FT(k - 1) / Nz)^FT(1.3) for k = 1:Nz+1],
                                     topology=(Periodic, Periodic, Bounded))

    vitd = VerticallyImplicitTimeDiscretization()
    Δt = FT(25)

    # Constant, and decreasing by a factor of 5 over the column: the two profiles the operator
    # error was measured on. `ν` is passed to the closure as a function of position and read back
    # at cell centers by the reference below, so both see the same numbers.
    #
    # `ν₀` is converted here rather than inside the profiles: a closure that calls `FT(200)`
    # captures `FT` itself, and `Type{Float64}` is not `isbits`, so the whole closure — and with
    # it the `VerticalScalarDiffusivity` holding it — fails to upload to a GPU.
    ν₀ = FT(200)
    ν_profiles = ("constant ν"          => z -> ν₀,
                  "ν decreasing with z" => z -> ν₀ * (1 - 4 * z / (5 * Lz)))

    # The tridiagonal row of `1 - Δt ∂z(ρ ν ∂z (q/ρᶠ))` at an interior face k, written out from the
    # physics: the stress ρᶜₖ νₖ (wₖ₊₁ - wₖ)/Δzᶜₖ sits at center k, and its divergence over Δzᶠₖ
    # drives face k. `ℑzᵃᵃᶠ` is the plain two-point mean, on stretched grids too.
    #
    # Returns du(k), dl(k-1) and d(k). Note dl(k-1) divides by ρᶠₖ₋₁ — where qₖ₋₁ is reconstructed
    # as wₖ₋₁ — while both terms of d(k) divide by this row's ρᶠₖ. That mismatch is why the
    # diagonal cannot be formed as `1 - du - dl`.
    function reference_row(k, zᶠ, zᶜ, ρᶜ, ν)
        Δzᶜᵏ   = zᶠ[k+1] - zᶠ[k]
        Δzᶜᵏ⁻¹ = zᶠ[k]   - zᶠ[k-1]
        Δzᶠᵏ   = zᶜ[k]   - zᶜ[k-1]

        ρᶠᵏ⁻¹ = (ρᶜ[k-1] + ρᶜ[k-2]) / 2
        ρᶠᵏ   = (ρᶜ[k]   + ρᶜ[k-1]) / 2
        ρᶠᵏ⁺¹ = (ρᶜ[k+1] + ρᶜ[k])   / 2

        νᵏ   = ν(zᶜ[k])
        νᵏ⁻¹ = ν(zᶜ[k-1])

        du = -Δt * νᵏ   * ρᶜ[k]   / (Δzᶜᵏ   * Δzᶠᵏ * ρᶠᵏ⁺¹)
        dl = -Δt * νᵏ⁻¹ * ρᶜ[k-1] / (Δzᶜᵏ⁻¹ * Δzᶠᵏ * ρᶠᵏ⁻¹)
        d  = 1 + Δt * (νᵏ * ρᶜ[k] / Δzᶜᵏ + νᵏ⁻¹ * ρᶜ[k-1] / Δzᶜᵏ⁻¹) / (Δzᶠᵏ * ρᶠᵏ)

        return du, dl, d
    end

    for (grid_name, grid) in ("uniform grid" => uniform_grid, "stretched grid" => stretched_grid),
        (ν_name, ν) in ν_profiles

        @testset "z-Face coefficients match the hand-derived operator [$(grid_name), $(ν_name)]" begin
            closure = VerticalScalarDiffusivity(vitd; ν = (x, y, z, t) -> ν(z))
            model = AtmosphereModel(grid; closure, advection = nothing)
            ρ = Breeze.AtmosphereModels.total_density(model.dynamics)

            zᶠ = Array(znodes(grid, Face()))
            zᶜ = Array(znodes(grid, Center()))
            ρᶜ = Array(interior(ρ, 1, 1, :))

            # ρ must vary enough over the column for the weighting to be the effect under test.
            @test ρᶜ[1] / ρᶜ[Nz] > 1.4

            scheme = Breeze.AtmosphereModels.implicit_step_scheme(nothing)
            ρw = model.momentum.ρw
            bcs = (ρw.boundary_conditions.top, ρw.boundary_conditions.bottom, ρw.boundary_conditions.immersed)

            coefficient(marker, k, advection) = @allowscalar Oceananigans.Solvers.get_coefficient(
                1, 1, k, grid, marker, nothing, ZDirection(),
                model.closure, model.closure_fields, nothing, Center(), Center(), Face(),
                Δt, model.clock, Oceananigans.fields(model), advection, nothing, ρ, bcs...)

            du(k, advection=scheme) = coefficient(VerticallyImplicitDiffusionUpperDiagonal(), k, advection)
            dl(k, advection=scheme) = coefficient(VerticallyImplicitDiffusionLowerDiagonal(), k, advection)
            d(k,  advection=scheme) = coefficient(VerticallyImplicitDiffusionDiagonal(),      k, advection)

            # Interior rows only: k = 3 … Nz-1 keeps the reference off every halo value.
            for k = 3:Nz-1
                du_ref, dl_ref, d_ref = reference_row(k, zᶠ, zᶜ, ρᶜ, ν)
                @test du(k)   ≈ du_ref atol = 64 * eps(FT) * abs(du_ref)
                @test dl(k-1) ≈ dl_ref atol = 64 * eps(FT) * abs(dl_ref)
                @test d(k)    ≈ d_ref  atol = 64 * eps(FT) * abs(d_ref)
            end

            # And the weighting is what makes them agree. What the unweighted coefficients drop is
            # a ratio across a *half* cell — the density where the stress lives over the density
            # where `w` is reconstructed — not the ≈ 40% ρ varies by over the whole column, so the
            # gap per row is only the departure of ρᶜ/ρᶠ from 1, ≈ Δz/2H, or 0.6–0.8% here. The
            # operator those rows build is a second difference in which they very nearly cancel,
            # and the dropped ratio survives that cancellation an order of magnitude larger:
            # applying it to a smooth profile leaves 5.4–6.4% across the four cases. That is the
            # error under test, so measure it there rather than row by row. `nothing` in the
            # advection slot is Oceananigans' own, unweighted method.
            q = [sinpi(zᶠ[k] / Lz)^2 for k = 1:Nz+1]
            Lq(advection) = [dl(k-1, advection) * q[k-1] +
                             (d(k, advection) - 1) * q[k] +
                             du(k, advection) * q[k+1] for k = 3:Nz-1]

            weighted = Lq(scheme)
            @test maximum(abs, Lq(nothing) .- weighted) / maximum(abs, weighted) > 0.02

            # Momentum conservation. `Σ Δzᶠ q` is unchanged by the solve when every Δzᶠ-weighted
            # column sum equals Δzᶠₖ, which holds only because the diagonal is written out: the
            # two off-diagonals reconstruct qₖ at ρᶠₖ, but the diagonal's copies of them divide by
            # this row's ρᶠ, and forming it as `1 - du - dl` would leave a residual ∝ 1/ρᶠₖ₊₁ - 1/ρᶠₖ.
            for k = 3:Nz-1
                Δzᶠᵏ⁻¹ = zᶜ[k-1] - zᶜ[k-2]
                Δzᶠᵏ   = zᶜ[k]   - zᶜ[k-1]
                Δzᶠᵏ⁺¹ = zᶜ[k+1] - zᶜ[k]
                column_sum = Δzᶠᵏ * d(k) + Δzᶠᵏ⁻¹ * du(k-1) + Δzᶠᵏ⁺¹ * dl(k)
                @test column_sum ≈ Δzᶠᵏ atol = 64 * eps(FT) * Δzᶠᵏ
            end

            # Every row is finite, boundary rows included. Row 1 divides by ρᶠ₁, which reads ρ's
            # bottom halo — a value no mask zeroes.
            @test all(isfinite, [du(k) for k = 1:Nz-1])
            @test all(isfinite, [dl(k) for k = 1:Nz-1])
            @test all(isfinite, [d(k)  for k = 1:Nz])
        end
    end

    @testset "The z-Face solve inverts the hand-derived operator" begin
        ν = last(first(ν_profiles))
        closure = VerticalScalarDiffusivity(vitd; ν = (x, y, z, t) -> ν(z))
        model = AtmosphereModel(stretched_grid; closure, advection = nothing)
        ρ = Breeze.AtmosphereModels.total_density(model.dynamics)

        zᶠ = Array(znodes(stretched_grid, Face()))
        zᶜ = Array(znodes(stretched_grid, Center()))
        ρᶜ = Array(interior(ρ, 1, 1, :))

        q = Field{Center, Center, Face}(stretched_grid)
        set!(q, (x, y, z) -> sinpi(z / Lz)^2)
        fill_halo_regions!(q)
        before = Array(interior(q, 1, 1, :))

        implicit_step!(q, model.timestepper.implicit_solver, model.closure, model.closure_fields,
                       nothing, model.clock, Oceananigans.fields(model), Δt,
                       Breeze.AtmosphereModels.implicit_step_scheme(nothing), nothing, ρ)

        after = Array(interior(q, 1, 1, :))
        @test after != before
        @test all(isfinite, after)

        # Backward Euler: the solve returns the qⁿ⁺¹ satisfying (I - Δt L) qⁿ⁺¹ = qⁿ row by row.
        for k = 3:Nz-1
            du_ref, dl_ref, d_ref = reference_row(k, zᶠ, zᶜ, ρᶜ, ν)
            residual = dl_ref * after[k-1] + d_ref * after[k] + du_ref * after[k+1]
            @test residual ≈ before[k] atol = 1e3 * eps(FT) * abs(before[k])
        end
    end
end

@testset "Density-weighted get_coefficient wins dispatch [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    grid = RectilinearGrid(default_arch; size=(1, 1, 16), x=(0, 100), y=(0, 100), z=(0, 4000),
                           topology=(Periodic, Periodic, Bounded))
    closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ = 100, ν = 100)
    model = AtmosphereModel(grid; closure, advection=nothing, tracers=:ρc)
    set!(model; θ = 300, ρc = (x, y, z) -> cospi(z / 4000))

    ρ = Breeze.AtmosphereModels.total_density(model.dynamics)
    scheme = Breeze.AtmosphereModels.implicit_step_scheme(nothing)
    w = model.velocities.w
    clock = model.clock
    mf = Oceananigans.fields(model)
    c, f = Center(), Face()

    ρc, ρw = model.tracers.ρc, model.momentum.ρw
    field_bcs(q) = (q.boundary_conditions.top, q.boundary_conditions.bottom, q.boundary_conditions.immersed)

    # The argument list `implicit_step!` calls the solver with, `(advection, w, density)` and the
    # field's three boundary conditions last.
    coefficient_arguments(marker, ℓz, id, bcs) =
        (1, 1, 8, grid, marker, nothing, ZDirection(),
         model.closure, model.closure_fields, id, c, c, ℓz, 1.0, clock, mf, scheme, w, ρ, bcs...)

    # `ρc` at z-Centers and `ρw` at z-Faces take the same seam and differ only in `ℓz`, which
    # selects the mirrored coefficients.
    cases = ((c, Breeze.AtmosphereModels.closure_scalar_index(model, :ρc), field_bcs(ρc)),
             (f, nothing,                                                  field_bcs(ρw)))

    for (ℓz, id, bcs) in cases,
        marker in (VerticallyImplicitDiffusionUpperDiagonal(),
                   VerticallyImplicitDiffusionLowerDiagonal(),
                   VerticallyImplicitDiffusionDiagonal())

        call = coefficient_arguments(marker, ℓz, id, bcs)
        coefficient = @allowscalar Oceananigans.Solvers.get_coefficient(call...)

        # A method must exist for the full argument list — `which` throws otherwise — and it must
        # be the weighted one rather than Oceananigans' fallback. Asserted on the method rather
        # than on the value the two return: under uniform ν and Δz the z-Face diagonal's two
        # half-cell ratios cancel exactly, `(du ρᶜₖ + dl ρᶜₖ₋₁) / ρᶠₖ = du + dl`, so there the
        # weighted coefficient is bit-identical to the unweighted one that must not win.
        @test which(Oceananigans.Solvers.get_coefficient, typeof.(call)).module === Breeze.AtmosphereModels
        @test isfinite(coefficient)
    end
end

#####
##### Routing of the implicit vertical solve by prognostic name
#####
##### Momentum takes the solve with the closure's viscosity, every scalar with the diffusivity of
##### its position among the closure's scalar names, and the dynamics-specific prognostics (the
##### compressible dry density) sit it out. Keyed on names rather than positions, so that a
##### prognostic tuple that starts with `ρᵈ` does not shift every scalar's diffusivity by one.
#####

@testset "Implicit-solve routing by prognostic name [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 8), x=(0, 100), y=(0, 100), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded))
    closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ = 1)

    closure_scalar_names = Breeze.AtmosphereModels.closure_scalar_names
    closure_scalar_index = Breeze.AtmosphereModels.closure_scalar_index
    skip_vertical_diffusion = Breeze.AtmosphereModels.skip_vertical_diffusion

    anelastic = AtmosphereModel(grid; closure, tracers = (:ρa, :ρb))
    compressible = AtmosphereModel(grid; closure, tracers = (:ρa, :ρb),
                                   dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = 300))

    for model in (anelastic, compressible)
        names = keys(Oceananigans.prognostic_fields(model))
        scalar_names = closure_scalar_names(model)

        # The thermodynamic density, moisture and the user tracers, in that order
        @test scalar_names == (:ρθ, :ρqᵛ, :ρa, :ρb)

        for (i, name) in enumerate(scalar_names)
            @test name ∈ names
            @test closure_scalar_index(model, name) === Val(i)
            @test !skip_vertical_diffusion(model, name)
        end

        for name in (:ρu, :ρv, :ρw)
            @test closure_scalar_index(model, name) === nothing
            @test !skip_vertical_diffusion(model, name)
        end
    end

    # The compressible dry density leads the prognostic tuple and is the one prognostic that is
    # not vertically diffused
    names = keys(Oceananigans.prognostic_fields(compressible))
    @test names[1] === :ρᵈ
    @test skip_vertical_diffusion(compressible, :ρᵈ)
    @test findfirst(==(:ρθ), names) == 5
    @test Breeze.TimeSteppers.acoustic_prognostic_names(compressible) == (:ρᵈ, :ρu, :ρv, :ρw, :ρθ)

    # Both steppers take a step through the routed implicit solve
    set!(anelastic; θ = 300, ρa = 1, ρb = 2)
    set!(compressible; θ = 300, ρ = compressible.dynamics.reference_state.density, ρa = 1, ρb = 2)
    for model in (anelastic, compressible)
        time_step!(model, 1)
        @test model.clock.iteration == 1
        @test all(isfinite, Array(interior(model.tracers.ρa)))
        @test all(isfinite, Array(interior(model.tracers.ρb)))
    end
end
