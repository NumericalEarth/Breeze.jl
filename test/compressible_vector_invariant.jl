include(joinpath(@__DIR__, "setup.jl"))

#####
##### CompressibleVectorInvariant: the discrete identities the scheme is built on — exact conservation
##### of the face-density-weighted kinetic energy by the centered core (and by an upwind-biased
##### vorticity), agreement with the flux form at second order, the full three-dimensional divergence
##### of the unsplit form — and the compressible AtmosphereModel stepping with it on the three grids.
#####

using Breeze
using Breeze.Advection: x_divergence_term, mass_flux_divergenceᶜᶜᶜ
using Breeze.AtmosphereModels: x_momentum_flux_divergence, y_momentum_flux_divergence, z_momentum_flux_divergence
using Oceananigans
using Oceananigans.Advection: EnergyConserving, materialize_advection
using Oceananigans.Grids: halo_size
using Oceananigans.MultiRegion: YinYangGrid
using Oceananigans.Operators: Vᶠᶜᶜ, Vᶜᶠᶜ, Vᶜᶜᶠ, Vᶜᶜᶜ, divᶜᶜᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ
using Oceananigans.Units
using Random
using Test

#####
##### Models and fields
#####

function compressible_model(grid, momentum_advection; time_discretization = ExplicitTimeStepping(), kw...)
    dynamics = CompressibleDynamics(time_discretization; reference_potential_temperature = 300)
    return AtmosphereModel(grid; dynamics, momentum_advection, kw...)
end

# A random state: positive density, velocities of order one, halos filled by `set!`
function set_random_state!(model; seed = 42)
    Random.seed!(seed)
    grid = model.grid
    ρ = CenterField(grid); u = XFaceField(grid); v = YFaceField(grid); w = ZFaceField(grid)
    set!(ρ, (x, y, z) -> 1 + rand() / 2)
    set!(u, (x, y, z) -> randn())
    set!(v, (x, y, z) -> randn())
    set!(w, (x, y, z) -> randn())
    set!(model; ρ, u, v, w)
    return nothing
end

#####
##### Kinetic energy production of the advective tendency: with the model's velocity diagnosis
##### u = ρu / ℑˣρ the kinetic energy is Σ V ℑˣρ u²/2 and dE/dt = Σ V (u ∂t(ρu) − u²/2 ∂t(ℑˣρ)),
##### with ∂t ρ = −∇·(ρ𝐮). Without pressure, gravity or Coriolis the advection must give dE/dt = 0.
#####

@inline u_work(i, j, k, grid, scheme, momentum, velocities, dynamics) =
    @inbounds - velocities.u[i, j, k] * Vᶠᶜᶜ(i, j, k, grid) * x_momentum_flux_divergence(i, j, k, grid, scheme, momentum, velocities, dynamics)

@inline v_work(i, j, k, grid, scheme, momentum, velocities, dynamics) =
    @inbounds - velocities.v[i, j, k] * Vᶜᶠᶜ(i, j, k, grid) * y_momentum_flux_divergence(i, j, k, grid, scheme, momentum, velocities, dynamics)

@inline w_work(i, j, k, grid, scheme, momentum, velocities, dynamics) =
    @inbounds - velocities.w[i, j, k] * Vᶜᶜᶠ(i, j, k, grid) * z_momentum_flux_divergence(i, j, k, grid, scheme, momentum, velocities, dynamics)

@inline u_continuity_work(i, j, k, grid, momentum, velocities) =
    @inbounds velocities.u[i, j, k]^2 / 2 * Vᶠᶜᶜ(i, j, k, grid) * ℑxᶠᵃᵃ(i, j, k, grid, divᶜᶜᶜ, momentum.ρu, momentum.ρv, momentum.ρw)

@inline v_continuity_work(i, j, k, grid, momentum, velocities) =
    @inbounds velocities.v[i, j, k]^2 / 2 * Vᶜᶠᶜ(i, j, k, grid) * ℑyᵃᶠᵃ(i, j, k, grid, divᶜᶜᶜ, momentum.ρu, momentum.ρv, momentum.ρw)

@inline w_continuity_work(i, j, k, grid, momentum, velocities) =
    @inbounds velocities.w[i, j, k]^2 / 2 * Vᶜᶜᶠ(i, j, k, grid) * ℑzᵃᵃᶠ(i, j, k, grid, divᶜᶜᶜ, momentum.ρu, momentum.ρv, momentum.ρw)

total(op) = sum(compute!(Field(op)))
total_abs(op) = sum(abs, compute!(Field(op)))

# The kinetic energy production, relative to the sum of the magnitudes of its contributions
function kinetic_energy_production(model, scheme)
    grid = model.grid
    args = (materialize_advection(scheme, grid), model.momentum, model.velocities, model.dynamics)
    continuity_args = (model.momentum, model.velocities)

    ops = (KernelFunctionOperation{Face, Center, Center}(u_work, grid, args...),
           KernelFunctionOperation{Center, Face, Center}(v_work, grid, args...),
           KernelFunctionOperation{Center, Center, Face}(w_work, grid, args...),
           KernelFunctionOperation{Face, Center, Center}(u_continuity_work, grid, continuity_args...),
           KernelFunctionOperation{Center, Face, Center}(v_continuity_work, grid, continuity_args...),
           KernelFunctionOperation{Center, Center, Face}(w_continuity_work, grid, continuity_args...))

    P = sum(total(op) for op in ops)
    scale = sum(total_abs(op) for op in ops)
    return P / scale
end

#####
##### Advective tendencies of a scheme on the model's state, as fields
#####

function momentum_flux_divergences(model, scheme)
    grid = model.grid
    args = (materialize_advection(scheme, grid), model.momentum, model.velocities, model.dynamics)
    Gu = compute!(Field(KernelFunctionOperation{Face, Center, Center}(x_momentum_flux_divergence, grid, args...)))
    Gv = compute!(Field(KernelFunctionOperation{Center, Face, Center}(y_momentum_flux_divergence, grid, args...)))
    Gw = compute!(Field(KernelFunctionOperation{Center, Center, Face}(z_momentum_flux_divergence, grid, args...)))
    return Gu, Gv, Gw
end

# Maximum difference between the tendencies of two schemes relative to the maximum tendency, over
# the interior cells at least `margin` cells away from the bounded walls
function relative_difference(model, scheme, reference; margin = 0)
    G = momentum_flux_divergences(model, scheme)
    Gᵣ = momentum_flux_divergences(model, reference)
    Nx, Ny, Nz = size(model.grid)
    js = 1 + margin : Ny - margin
    difference = maximum(maximum(abs, interior(g, :, js, :) .- interior(gᵣ, :, js, :)) for (g, gᵣ) in zip(G, Gᵣ))
    scale = maximum(maximum(abs, interior(gᵣ, :, js, :)) for gᵣ in Gᵣ)
    return difference / scale
end

#####
##### Grids. Spacings are stretched wherever the grid allows it.
#####

stretched(N, L; power = 1.3) = L .* (collect(0:N) ./ N) .^ power

function rectilinear_grid(arch, FT, N; stretched_z = true, L = 1000)
    Nx, Ny, Nz = N
    z = stretched_z ? stretched(Nz, L) : (0, L)
    return RectilinearGrid(arch, FT; size = N, halo = (6, 6, 4), x = (0, L), y = (0, L), z,
                           topology = (Periodic, Periodic, Bounded))
end

latitude_longitude_grid(arch, FT, N; L = 10kilometers) =
    LatitudeLongitudeGrid(arch, FT; size = N, halo = (6, 6, 4), longitude = (0, 360), latitude = (-60, 60),
                          z = stretched(N[3], L), topology = (Periodic, Bounded, Bounded))

# Smooth fields that vanish with their normal derivative at the walls
smooth_u(x, y, z) = 10 * sin(2π * x / 1000) * cos(2π * y / 1000) * (1 + z / 1000)
smooth_v(x, y, z) = 10 * cos(2π * x / 1000) * sin(4π * y / 1000) * (1 + z / 2000)
smooth_w(x, y, z) = 0.5 * sin(2π * x / 1000) * sin(2π * y / 1000) * sin(π * z / 1000)^2
smooth_ρ(x, y, z) = 1.2 * (1 + 0.2 * cos(2π * x / 1000) * sin(2π * y / 1000)) * exp(-z / 2000)

spherical_u(λ, φ, z) = 20 * cosd(2λ) * cosd(1.5φ)^2 * (1 + z / 10kilometers)
spherical_v(λ, φ, z) = 10 * sind(3λ) * sind(3φ) * cosd(1.5φ)^2
spherical_w(λ, φ, z) = 0.2 * sind(λ) * cosd(3φ) * cosd(1.5φ)^2 * sin(π * z / 10kilometers)^2
spherical_ρ(λ, φ, z) = 1.2 * (1 + 0.1 * cosd(λ) * cosd(φ)) * exp(-z / 8kilometers)

function set_smooth_state!(model; spherical = false)
    if spherical
        set!(model; ρ = spherical_ρ, u = spherical_u, v = spherical_v, w = spherical_w)
    else
        set!(model; ρ = smooth_ρ, u = smooth_u, v = smooth_v, w = smooth_w)
    end
    return nothing
end

#####
##### Schemes
#####

# `core` is the order of the centered core; the two-point core (2) is exact next to closed walls too
centered(core = 2) = CompressibleVectorInvariant(core_order = core)
unsplit(core = 2) = CompressibleVectorInvariant(vertical_formulation = UnsplitVerticalAdvection(), core_order = core)
weno_vorticity(core = 2) = CompressibleVectorInvariant(vorticity_scheme = WENO(order = 9), core_order = core)
averaged_weno_vorticity(core = 2) = CompressibleVectorInvariant(vorticity_scheme = WENO(order = 7), vorticity_pairing = AveragedVorticityPairing(), core_order = core)
corner_weno_vorticity(core = 2) = CompressibleVectorInvariant(vorticity_scheme = WENO(order = 7), vorticity_pairing = CornerVorticityPairing(), core_order = core)
averaged_centered_vorticity(core = 2) = CompressibleVectorInvariant(vorticity_scheme = Centered(order = 2), vorticity_pairing = AveragedVorticityPairing(), core_order = core)
weno_divergence() = CompressibleVectorInvariant(divergence_scheme = WENO(order = 5))
weno_kinetic_energy() = CompressibleVectorInvariant(kinetic_energy_gradient_scheme = WENO(order = 5))
weno_vertical() = CompressibleVectorInvariant(vertical_advection_scheme = WENO(order = 5))
flux_form() = Centered(order = 2)

@testset "CompressibleVectorInvariant [$(FT)]" for FT in test_float_types()
    previous_float_type = Oceananigans.defaults.FloatType
    Oceananigans.defaults.FloatType = FT
    roundoff = 1000 * eps(FT)

    try
        @testset "Construction" begin
            scheme = CompressibleVectorInvariant()
            @test scheme.vorticity_scheme isa EnergyConserving
            @test scheme.vertical_formulation isa SplitVerticalAdvection
            @test scheme.core_scheme isa Centered{2}
            @test CompressibleVectorInvariant(core_order = 2).core_scheme isa Centered{1}
            @test halo_size(rectilinear_grid(CPU(), FT, (8, 8, 8))) == (6, 6, 4)

            weno = CompressibleWENOVectorInvariant()
            @test weno.vorticity_scheme isa WENO{5}
            @test weno.divergence_scheme isa WENO{3}
            @test weno.vorticity_pairing isa UpwindVorticityPairing
            @test weno.centered_vorticity_scheme isa Centered{5}
            @test isnothing(scheme.centered_vorticity_scheme)
            @test Oceananigans.Grids.required_halo_size_x(weno) == 6
            @test Oceananigans.Grids.required_halo_size_x(averaged_weno_vorticity(4)) == 6  # WENO(7) with the fourth-order core
            @test Oceananigans.Grids.required_halo_size_x(CompressibleWENOVectorInvariant(vorticity_pairing = AveragedVorticityPairing())) == 7
            @test Oceananigans.Grids.required_halo_size_z(weno) == 4

            weno5 = CompressibleWENOVectorInvariant(order = 5)
            @test Oceananigans.Grids.required_halo_size_x(weno5) == 4

            @test_throws ArgumentError CompressibleVectorInvariant(vorticity_scheme = WENO(),
                                                                   vertical_formulation = UnsplitVerticalAdvection())
            @test sprint(show, scheme) isa String
            @test sprint(show, weno) isa String
        end

        @testset "Kinetic energy production vanishes for the centered core" begin
            for (name, grid) in (("rectilinear", rectilinear_grid(default_arch, FT, (12, 10, 8))),
                                 ("latitude-longitude", latitude_longitude_grid(default_arch, FT, (16, 12, 6))))
                model = compressible_model(grid, centered())
                set_random_state!(model)

                # Exact (to roundoff) for the centered schemes in both vertical formulations, and for an
                # upwind-biased vorticity, whose symmetric pairings do no work for any reconstruction.
                # The fourth-order core relies on the transposition of its interpolations, which the
                # reduced-order stencils next to closed walls break, so it is exact on the periodic grid only.
                cores = name == "rectilinear" ? (2, 4) : (2,)
                for core in cores, build in (centered, unsplit, weno_vorticity, corner_weno_vorticity, averaged_weno_vorticity, averaged_centered_vorticity)
                    @test abs(kinetic_energy_production(model, build(core))) < roundoff
                end

                # The flux form conserves the same energy only on a uniform grid; here the vertical
                # spacing is stretched, so it does not.
                @test abs(kinetic_energy_production(model, flux_form())) > roundoff

                # Reconstruction of the self-transported kinetic energy gradient and divergence, and of
                # the vertical fluxes, breaks the identity: on these random (grid-scale) fields the
                # production is a sizeable fraction of the work done, and it is not zero.
                for scheme in (weno_divergence(), weno_kinetic_energy(), weno_vertical())
                    P = kinetic_energy_production(model, scheme)
                    @test roundoff < abs(P) < 1
                end
            end
        end

        @testset "Identity with the flux form on $(name)" for (name, grid_builder, spherical, margin) in
                (("RectilinearGrid", rectilinear_grid, false, 0), ("LatitudeLongitudeGrid", latitude_longitude_grid, true, 2))

            ratios = FT[]
            differences = FT[]
            for N in ((24, 16, 8), (48, 32, 16))
                grid = grid_builder(default_arch, FT, N)
                model = compressible_model(grid, centered())
                set_smooth_state!(model; spherical)
                push!(differences, relative_difference(model, centered(), flux_form(); margin))
            end

            # Both are second-order discretizations of the same operator
            @test differences[1] < 0.2
            @test 3 < differences[1] / differences[2] < 5
        end

        @testset "Corner pairing of a centered reconstruction is the corner vorticity" begin
            grid = rectilinear_grid(default_arch, FT, (16, 12, 6))
            model = compressible_model(grid, centered())
            set_random_state!(model)
            for pairing in (UpwindVorticityPairing(), CornerVorticityPairing()), core in (2, 4)
                corner = CompressibleVectorInvariant(vorticity_scheme = Centered(order = 4), vorticity_pairing = pairing, core_order = core)
                @test relative_difference(model, corner, centered(core)) < roundoff
            end
            @test relative_difference(model, averaged_centered_vorticity(), centered()) > 1e-3
        end

        @testset "Split and unsplit vertical formulations" begin
            grid = rectilinear_grid(default_arch, FT, (24, 16, 8))
            model = compressible_model(grid, centered())

            # Without vertical velocity the horizontal tendencies of the two forms are the same discrete expression
            set!(model; ρ = smooth_ρ, u = smooth_u, v = smooth_v, w = 0)
            Gu, Gv, _ = momentum_flux_divergences(model, centered())
            Gu′, Gv′, _ = momentum_flux_divergences(model, unsplit())
            @test maximum(abs, interior(Gu) .- interior(Gu′)) ≤ roundoff * maximum(abs, interior(Gu))
            @test maximum(abs, interior(Gv) .- interior(Gv′)) ≤ roundoff * maximum(abs, interior(Gv))

            # With vertical velocity they differ at second order on a uniform grid. (On the power-law
            # stretched grid the difference is dominated by the lowest cells, where the spacing ratio
            # does not approach one under refinement, and the ratio is 2^1.3 instead of 4.)
            differences = FT[]
            for N in ((24, 16, 8), (48, 32, 16))
                grid = rectilinear_grid(default_arch, FT, N; stretched_z = false)
                model = compressible_model(grid, centered())
                set_smooth_state!(model)
                push!(differences, relative_difference(model, unsplit(), centered()))
            end
            @test 3 < differences[1] / differences[2] < 5
        end

        @testset "Full three-dimensional divergence of the unsplit form" begin
            grid = rectilinear_grid(default_arch, FT, (16, 16, 8); stretched_z = false)
            model = compressible_model(grid, unsplit())

            # Horizontally nondivergent momentum from a streamfunction (ρ uniform), with ∂z(ρw) ≠ 0
            ψ = Field{Face, Face, Center}(grid)
            set!(ψ, (x, y, z) -> 100 * sin(2π * x / 1000) * sin(2π * y / 1000))
            u = compute!(Field(-∂y(ψ)))
            v = compute!(Field(∂x(ψ)))
            set!(model; ρ = 1.2, u, v, w = (x, y, z) -> sin(π * z / 1000))

            # The divergence term of the x momentum equation is u ℑˣ(∇·ρ𝐮) with the full divergence
            grid = model.grid
            full = compute!(Field(KernelFunctionOperation{Face, Center, Center}(x_divergence_term, grid, unsplit(), UnsplitVerticalAdvection(),
                                                                                model.velocities.u, model.momentum...)))
            split = compute!(Field(KernelFunctionOperation{Face, Center, Center}(x_divergence_term, grid, centered(), SplitVerticalAdvection(),
                                                                                 model.velocities.u, model.momentum...)))
            @test maximum(abs, interior(full)) > 1e-3
            @test maximum(abs, interior(split)) < roundoff * maximum(abs, interior(full))

            D = compute!(Field(KernelFunctionOperation{Center, Center, Center}(divᶜᶜᶜ, grid, model.momentum...)))
            Dh = compute!(Field(KernelFunctionOperation{Center, Center, Center}(mass_flux_divergenceᶜᶜᶜ, grid, SplitVerticalAdvection(), model.momentum...)))
            @test maximum(abs, interior(D)) > 1e-3
            @test maximum(abs, interior(Dh)) < roundoff * maximum(abs, interior(D))
        end

        @testset "Time stepping on $(name)" for (name, grid) in
                (("RectilinearGrid", rectilinear_grid(default_arch, FT, (12, 10, 8))),
                 ("LatitudeLongitudeGrid", latitude_longitude_grid(default_arch, FT, (24, 12, 6))))

            spherical = grid isa LatitudeLongitudeGrid
            for scheme in (centered(), centered(4), unsplit(), CompressibleWENOVectorInvariant())
                coriolis = spherical ? SphericalCoriolis() : FPlane(f = 1e-4)
                model = compressible_model(grid, scheme; coriolis, time_discretization = SplitExplicitTimeDiscretization())
                set!(model; θ = 300, ρ = model.dynamics.reference_state.density)
                if spherical
                    set!(model; u = (λ, φ, z) -> 10 * cosd(φ), v = (λ, φ, z) -> sind(2λ) * cosd(φ))
                else
                    set!(model; u = (x, y, z) -> sin(2π * x / 1000), v = (x, y, z) -> cos(2π * x / 1000))
                end

                ρV = KernelFunctionOperation{Center, Center, Center}((i, j, k, grid, ρ) -> ρ[i, j, k] * Vᶜᶜᶜ(i, j, k, grid), grid,
                                                                      model.dynamics.dry_density)
                mass₀ = total(ρV)
                Δt = spherical ? 60 : 0.5
                for _ in 1:10
                    time_step!(model, Δt)
                end
                @test all(isfinite, interior(model.momentum.ρu))
                @test abs(total(ρV) - mass₀) ≤ 100 * eps(FT) * mass₀
            end
        end

        @testset "Time stepping on YinYangGrid" begin
            grid = YinYangGrid(default_arch, FT; size = (72, 24, 4), z = (0, 10kilometers), halo = (4, 4, 4))
            scheme = CompressibleWENOVectorInvariant(order = 5)
            model = compressible_model(grid, scheme; coriolis = SphericalCoriolis(), time_discretization = SplitExplicitTimeDiscretization())
            set!(model; θ = 300, ρ = model.dynamics.reference_state.density,
                 u = (λ, φ, z) -> 10 * cosd(φ), v = (λ, φ, z) -> sind(2λ) * cosd(φ))
            for _ in 1:5
                time_step!(model, 60)
            end
            @test isfinite(maximum(abs, model.momentum.ρu))
            @test isfinite(maximum(abs, model.momentum.ρv))
        end
    finally
        Oceananigans.defaults.FloatType = previous_float_type
    end
end
