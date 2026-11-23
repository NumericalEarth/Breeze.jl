using Breeze
using Oceananigans
using Test

include("hydrostatic_moist_utils.jl")

using Oceananigans: Center
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.MultiRegion: YinYangGrid
using Breeze.HydrostaticEquations: VerletSubsteps, LayerMassFraction, HydrostaticConservationFixer,
                                   ConvexStageSeamBudgets, layer_mass_density, signed_scalar_content
using Breeze.AtmosphereModels: owned_integral, owned_integral_value
using Breeze.PressureLevelDiscretization: PressureLevelVerticalDiscretization

#####
##### The conservation fixer of the convex scalar stages of GravityWaveMultirateInfinitesimalStep on a Yin-Yang
##### PressureLevelGrid: the owned mass and the owned water are conserved to rounding, a uniform mixing ratio stays
##### uniform, and without the fixer the seam changes both
#####

function convex_fixer_coefficients(Nz; top_pressure = 2.5e4, p₀ = 1e5)
    η = range(1, top_pressure / p₀, length = Nz + 1)
    B = [max(0, (e - 0.4) / 0.6) for e in η]
    A = [e * p₀ - b * p₀ for (e, b) in zip(η, B)]
    A[1] = 0; B[1] = 1; B[end] = 0; A[end] = top_pressure
    return A, B
end

function convex_fixer_model()
    Nz = 8
    A, B = convex_fixer_coefficients(Nz)
    grid = YinYangGrid(CPU(), Float64; size = (48, 16, Nz), halo = (4, 4, 3), z = PressureLevelVerticalDiscretization(A, B))
    microphysics = Breeze.Microphysics.InstantaneousPrecipitation(Float64; equilibrium = MixedPhaseEquilibrium(Float64))
    model = AtmosphereModel(grid; dynamics = HydrostaticDynamics(), microphysics, formulation = moist_formulation(),
                            momentum_advection = WENO(order = 5),
                            scalar_advection = (ρθ = WENO(order = 5), ρqᵛ = WENO(order = 5, bounds = (0, 1))),
                            timestepper = :GravityWaveMultirateInfinitesimalStep,
                            timestepper_kwargs = (; tableau = :ConvexLinearMIS3, inner_solver = VerletSubsteps(),
                                                    moist_gravity_wave_operator = true))
    # Subsaturated everywhere (no rain-out), uniform mixing ratio, flow across the seam
    set!(model; pˢ = (λ, φ) -> 1e5 + 500 * cosd(φ) * sind(λ), θ = (λ, φ, z) -> 290 + 4e-3 * z,
                u = (λ, φ, z) -> 20 * cosd(φ), v = (λ, φ, z) -> 5 * sind(2λ) * cosd(φ))
    set!(model; qᵗ = (λ, φ, z) -> 1e-4)
    return model
end

function owned_contents(model)
    grid = model.grid
    op(f, args...) = KernelFunctionOperation{Center, Center, Center}(f, grid, args...)
    r = LayerMassFraction(model.moisture_density, model.dynamics.density)
    M = owned_integral(op(layer_mass_density)); compute!(M)
    W = owned_integral(op(signed_scalar_content, r, 1)); compute!(W)
    return owned_integral_value(M), owned_integral_value(W)
end

function qᵛ_range(model)
    ranges = map(1:2) do n
        regional = Oceananigans.Utils.getregion(model, n)
        q = Array(interior(regional.moisture_density)) ./ Array(interior(regional.dynamics.density))
        extrema(q)
    end
    return minimum(first, ranges), maximum(last, ranges)
end

@testset "Conservation fixer of the convex scalar stages on a Yin-Yang PressureLevelGrid" begin
    for fixer in (false, true)
        model = convex_fixer_model()
        simulation = Simulation(model; Δt = 600, stop_iteration = 3, verbose = false)
        fixer && add_conservation_fixer!(simulation)
        fixer && @test simulation.callbacks[:conservation_fixer].func.stage_budgets isa ConvexStageSeamBudgets

        M⁰, W⁰ = owned_contents(model)
        run!(simulation)
        M¹, W¹ = owned_contents(model)
        q⁻, q⁺ = qᵛ_range(model)

        @info "Convex-stage seam [fixer = $fixer]" (M¹ - M⁰) / M⁰ (W¹ - W⁰) / W⁰ q⁻ q⁺
        if fixer
            @test abs(M¹ - M⁰) / M⁰ < 1e-12
            @test abs(W¹ - W⁰) / W⁰ < 1e-11
            @test abs(q⁺ - q⁻) < 1e-12        # the uniform mixing ratio stays uniform
            correction = simulation.callbacks[:conservation_fixer].func.stage_budgets.last_correction
            @test correction.mass != 0
        else
            @test abs(M¹ - M⁰) / M⁰ > 1e-11   # the seam is not conservative without the fixer
        end
    end
end
