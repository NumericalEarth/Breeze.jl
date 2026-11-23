using Breeze
using Oceananigans
using Test

include("hydrostatic_moist_utils.jl")

using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.Fields: ZeroField
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: Periodic, Flat, Bounded
using Oceananigans.MultiRegion: YinYangGrid
using Oceananigans.Utils: launch!, getregion
using Breeze.HydrostaticEquations: VerletSubsteps, _compute_vertical_mass_flux!, _split_stage_mass_flux!,
                                   _implicit_stage_vertical_transport!, horizontal_mass_flux_divergence
using Breeze.PressureLevelDiscretization: PressureLevelVerticalDiscretization, previous_layer_mass

#####
##### Adaptive implicit vertical transport of the scalars of the convex stages on a PressureLevelGrid (D494):
##### a uniform mixing ratio stays uniform with moving mass, the column content is that of the explicit step,
##### and a bounded mixing ratio stays bounded at donor Courant numbers of 2 to 3
#####

# Hybrid coefficients with one thin layer in the middle, so that the vertical Courant number through it is large
# while the horizontal outflow of every layer stays below its mass
function thin_layer_coefficients(; p₀ = 1e5)
    Δη = [0.16, 0.16, 0.016, 0.16, 0.16, 0.144]
    η = [1.0; 1 .- cumsum(Δη)]
    B = [max(0, (e - 0.4) / 0.6) for e in η]
    A = [e * p₀ - b * p₀ for (e, b) in zip(η, B)]
    A[1] = 0; B[1] = 1; B[end] = 0
    return A, B
end

@testset "Implicit vertical transport of a convex stage on a PressureLevelGrid" begin
    A, B = thin_layer_coefficients()
    Nz = length(A) - 1
    grid = RectilinearGrid(CPU(), Float64; size = (4, Nz), x = (0, 4e5), halo = (2, 2),
                           z = PressureLevelVerticalDiscretization(A, B), topology = (Periodic, Flat, Bounded))
    model = AtmosphereModel(grid; dynamics = HydrostaticDynamics(), momentum_advection = VectorInvariant())
    dynamics = model.dynamics
    ρ = dynamics.density

    # Convergence into the lowest layer of columns 1 and 3 and out of their top layer (and the reverse in 2 and 4):
    # every column keeps its mass, and the flow crosses the thin layer
    u = XFaceField(grid); v = YFaceField(grid)
    U = [isodd(i) ? 1.0 : -1.0 for i in 1:4]
    for i in 1:4
        u[i, 1, 1] = U[i]
        u[i, 1, Nz] = -U[i]
    end
    fill_halo_regions!(u)
    ρw = ZFaceField(grid); w = ZFaceField(grid)
    launch!(CPU(), grid, :xy, _compute_vertical_mass_flux!, ρw, w, grid, dynamics.surface_pressure_coefficient, ρ, u, v)
    fill_halo_regions!(ρw); fill_halo_regions!(w)

    μ = [previous_layer_mass(i, 1, k, grid) for i in 1:4, k in 1:Nz]
    D = [horizontal_mass_flux_divergence(i, 1, k, grid, ρ, u, v) for i in 1:4, k in 1:Nz]
    donor_rate = maximum(abs(ρw[i, 1, k]) / μ[i, clamp(ρw[i, 1, k] > 0 ? k - 1 : k, 1, Nz)] for i in 1:4, k in 2:Nz)

    for target in (2.0, 3.0)
        Δtᵉ = target / donor_rate
        cfl = 0.5
        s = ZFaceField(grid); sρw = ZFaceField(grid); sw = ZFaceField(grid); γ = CenterField(grid)
        launch!(CPU(), grid, :xy, _split_stage_mass_flux!, sρw, sw, s, grid, ρw, w, Δtᵉ, cfl)

        μᴱ = [μ[i, k] - Δtᵉ * (D[i, k] + ρw[i, 1, k+1] - ρw[i, 1, k]) for i in 1:4, k in 1:Nz]
        μˣ = [μ[i, k] - Δtᵉ * (D[i, k] + sρw[i, 1, k+1] - sρw[i, 1, k]) for i in 1:4, k in 1:Nz]
        @test all(>(0), μᴱ) && all(>(0), μˣ)
        @test maximum(1 .- [s[i, 1, k] for i in 1:4, k in 2:Nz]) > 0.5      # most of the flux is implicit

        for (label, rˣ) in (("uniform", fill(0.007, 4, Nz)), ("alternating", [Float64(isodd(i + k)) for i in 1:4, k in 1:Nz]))
            M = CenterField(grid); M⁰ = CenterField(grid)
            for i in 1:4, k in 1:Nz
                M[i, 1, k] = μˣ[i, k] * rˣ[i, k]            # the explicit result R (weight 1: M⁰ does not enter)
            end
            R = [M[i, 1, k] for i in 1:4, k in 1:Nz]
            launch!(CPU(), grid, :xy, _implicit_stage_vertical_transport!, M, γ, grid, M⁰, ρ, u, v, ρw, s, 1.0, Δtᵉ)
            r = [M[i, 1, k] / μᴱ[i, k] for i in 1:4, k in 1:Nz]

            @info "Implicit vertical transport [$label, donor C = $target]" extrema(r) sum(M) - sum(R)
            # Column contents are those of the explicit step
            @test all(abs(sum(M[i, 1, k] for k in 1:Nz) - sum(R[i, :])) < 1e-13 * sum(R[i, :]) + 1e-300 for i in 1:4)
            if label == "uniform"
                @test maximum(abs, r .- 0.007) < 1e-15
            else
                @test minimum(r) ≥ -1e-14
                @test maximum(r) ≤ 1 + 1e-14
            end
        end
    end
end

@testset "Implicit vertical transport of the water of a Yin-Yang PressureLevelGrid model" begin
    Nz = 8
    η = range(1, 0.25, length = Nz + 1)
    B = [max(0, (e - 0.4) / 0.6) for e in η]; A = [e * 1e5 - b * 1e5 for (e, b) in zip(η, B)]; A[1] = 0; B[1] = 1; B[end] = 0
    grid = YinYangGrid(CPU(), Float64; size = (48, 16, Nz), halo = (4, 4, 3), z = PressureLevelVerticalDiscretization(A, B))
    microphysics = Breeze.Microphysics.InstantaneousPrecipitation(Float64; equilibrium = MixedPhaseEquilibrium(Float64))
    water = WENO(order = 5, bounds = (0, 1), time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5))
    model = AtmosphereModel(grid; dynamics = HydrostaticDynamics(), microphysics, formulation = moist_formulation(),
                            momentum_advection = WENO(order = 5), scalar_advection = (ρθ = WENO(order = 5), ρqᵛ = water),
                            timestepper = :GravityWaveMultirateInfinitesimalStep,
                            timestepper_kwargs = (; tableau = :ConvexLinearMIS3, inner_solver = VerletSubsteps(),
                                                    moist_gravity_wave_operator = true))
    set!(model; pˢ = (λ, φ) -> 1e5 + 500 * cosd(φ) * sind(λ), θ = (λ, φ, z) -> 290 + 4e-3 * z,
                u = (λ, φ, z) -> 20 * cosd(φ), v = (λ, φ, z) -> 5 * sind(2λ) * cosd(φ))
    set!(model; qᵗ = (λ, φ, z) -> 1e-4)          # uniform and subsaturated: no rain-out

    for _ in 1:3
        time_step!(model, 1200)
    end

    for r in 1:2
        regional = getregion(model, r)
        q = Array(interior(regional.moisture_density)) ./ Array(interior(regional.dynamics.density))
        @test all(isfinite, q)
        @test maximum(abs, q .- 1e-4) < 1e-15   # uniform to rounding
    end
end
