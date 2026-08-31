include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Grids: ZDirection
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
##### Mass-flux weighting of the implicit solve
#####
##### The prognostics are density weighted (`ρc`), and the explicit flux divergence forms
##### `∂z(ρ⁰ κ ∂z c)` on the specific variable. The implicit solve must form the same operator.
##### Solving the unweighted `∂t(ρ⁰c) = ∂z(κ ∂z(ρ⁰c))` instead relaxes to `ρ⁰c = const`, i.e.
##### `c ∝ 1/ρ⁰`, rather than to a well-mixed `c = const` — worth `≈ +20 K/km` on a deep column.
##### The tests above run on `Lz = 100 m`, over which `ρ⁰` varies < 1%, which is why they pass
##### either way; these use a 4 km column, over which it varies ≈ 40%.
#####

@testset "Mass-flux weighting of the vertically implicit solve [$(FT)]" for FT in test_float_types()
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

@testset "Mass-weighted get_coefficient wins dispatch [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    grid = RectilinearGrid(default_arch; size=(1, 1, 16), x=(0, 100), y=(0, 100), z=(0, 4000),
                           topology=(Periodic, Periodic, Bounded))
    closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ = 100, ν = 100)
    model = AtmosphereModel(grid; closure, advection=nothing, tracers=:ρc)
    set!(model; θ = 300, ρc = (x, y, z) -> cospi(z / 4000))

    ρ = Breeze.AtmosphereModels.total_density(model.dynamics)
    scheme = Breeze.AtmosphereModels.implicit_step_scheme(nothing, :ρc)
    w = model.velocities.w
    clock = model.clock
    mf = Oceananigans.fields(model)
    c, f = Center(), Face()
    id = Breeze.AtmosphereModels.closure_scalar_index(model, :ρc)

    ρc = model.tracers.ρc
    bcs = (ρc.boundary_conditions.top, ρc.boundary_conditions.bottom, ρc.boundary_conditions.immersed)

    coefficient(marker, trailing...) = @allowscalar Oceananigans.Solvers.get_coefficient(
        1, 1, 8, grid, marker, nothing, ZDirection(),
        model.closure, model.closure_fields, id, c, c, c, 1.0, clock, mf, trailing...)

    for marker in (VerticallyImplicitDiffusionUpperDiagonal(),
                   VerticallyImplicitDiffusionLowerDiagonal(),
                   VerticallyImplicitDiffusionDiagonal())

        weighted   = coefficient(marker, scheme, w, ρ, bcs...)   # what implicit_step! passes
        unweighted = coefficient(marker)                          # the diffusion-only fallback

        # A method must exist for the full argument list, and it must be the weighted one: over a
        # 4 km column ρ varies ~40%, so the weighted coefficient cannot coincide with the unweighted.
        @test isfinite(weighted)
        @test weighted != unweighted
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
