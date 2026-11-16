include(joinpath(@__DIR__, "setup.jl"))

#####
##### A uniform mass fraction under post-loop scalar tendencies
#####
##### With `post_loop_scalar_tendencies = true` every stage must transport the moisture and the
##### tracers with the mass flux its own acoustic loop applied to the dry air, so that a uniform
##### mass fraction stays uniform to round-off. Two things used to break that:
#####
#####   - a multirate infinitesimal step recombines the advective tendencies of earlier stages,
#####     which carry those stages' mass fluxes, not the current loop's
#####     (`ScalarMassFluxCorrection`);
#####   - the implicit remainder of adaptive implicit vertical advection formed the mass fraction
#####     it transports with the stage-entry density of a cell the loop had since compressed
#####     (`StageEndDensity`).
#####

using Breeze
using Breeze.AtmosphereModels: specific_prognostic_moisture, total_density
using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Test

# A warm bubble in a sheared wind, with an updraft strong enough that the vertical Courant number
# of a full step, 0.8, exceeds that of the adaptive implicit split, 0.5, from the first step.
function uniform_mass_fraction_errors(timestepper, coefficients; qᵛ₀ = 0.01, c₀ = 1e-3, Δt = 10, Nt = 20)
    FT = Float64
    grid = RectilinearGrid(default_arch, FT; size = (16, 40), halo = (5, 5), x = (-4e3, 4e3), z = (0, 4e3),
                           topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5,
                                    reference_potential_temperature = 300)
    adaptive() = AdaptiveVerticallyImplicitDiscretization(FT; cfl = 0.5)
    momentum_advection = WENO(FT; order = 5, time_discretization = adaptive())
    scalar_advection = WENO(FT; order = 5, bounds = (0, 1), time_discretization = adaptive())
    timestepper_kwargs = (; post_loop_scalar_tendencies = true)
    isnothing(coefficients) || (timestepper_kwargs = merge(timestepper_kwargs, (; coefficients)))
    model = AtmosphereModel(grid; dynamics, momentum_advection, scalar_advection, tracers = :ρc,
                            timestepper, timestepper_kwargs)

    bubble(x, z) = cos(π * min(1, sqrt(x^2 + (z - 1500)^2) / 1e3) / 2)^2
    set!(model; θ = (x, z) -> 300 + 2 * bubble(x, z), qᵛ = qᵛ₀, ρ = HydrostaticallyBalancedDensity(),
         u = (x, z) -> 10 * z / 4e3, w = (x, z) -> 8 * bubble(x, z))
    parent(model.tracers.ρc) .= c₀ .* parent(total_density(model.dynamics))

    maximum_w = maximum(abs, interior(model.velocities.w))
    for _ in 1:Nt
        time_step!(model, Δt)
        maximum_w = max(maximum_w, maximum(abs, interior(model.velocities.w)))
    end

    qᵛ = Array(interior(specific_prognostic_moisture(model)))
    c = Array(interior(model.tracers.ρc)) ./ Array(interior(total_density(model.dynamics)))
    Δz = 4e3 / 40
    return (; moisture = maximum(abs, qᵛ .- qᵛ₀) / qᵛ₀, tracer = maximum(abs, c .- c₀) / c₀,
            vertical_courant_number = maximum_w * Δt / Δz)
end

const uniform_mass_fraction_steppers = ((:AcousticRungeKutta3, nothing),
                                        (:MultirateInfinitesimalStep, :SSPMIS43),
                                        (:MultirateInfinitesimalStep, :TVDMISB),
                                        (:MultirateInfinitesimalStep, :DiagonalMIS42),
                                        (:MultirateInfinitesimalStep, :DiagonalMIS52),
                                        (:MultirateInfinitesimalStep, :LowStorageMIS52),
                                        (:MultirateInfinitesimalStep, :ConvexLinearMIS3),
                                        (:MultirateInfinitesimalStep, :ConvexLinearMIS4),
                                        (:MultirateInfinitesimalStep, :ConvexLinearMIS5),
                                        (:MultirateInfinitesimalStep, :ConvexLinearMIS6))

@testset "A uniform mass fraction stays uniform to round-off [$timestepper, $coefficients]" for
        (timestepper, coefficients) in uniform_mass_fraction_steppers
    errors = uniform_mass_fraction_errors(timestepper, coefficients)
    @test errors.vertical_courant_number > 0.5 # the implicit remainder is active
    @test errors.moisture < 1e-12
    @test errors.tracer < 1e-12
end
