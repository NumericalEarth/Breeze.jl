include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.HydrostaticEquations: VerletSubsteps, column_physics_in_time_step!, warm_rain_column_physics!
using Oceananigans
using Oceananigans.Advection: BoundsPreservation
using Test

include("hydrostatic_moist_utils.jl")

#####
##### The limited face flux of flux-limited WENO must not drain an empty cell by a cancellation residue
#####
#
# In Float32, a raining moist hydrostatic slice (convex MIS6 + Verlet, WENO5 + FCT water) left
# ρqʳ = -2.3347237e-17 at cell (2, 1, 7) after stage 5 of step 3: at its upper face the upwind flux
# Jᵘ = -7.2686754e-9 and the antidiffusive flux A = 7.268676e-9 (face factor 1) summed to one ulp of
# the operands, 4.440892e-16, instead of the bounded high-order flux, and that outflow exceeded the
# content of the cell (5.7e-24) by seven orders of magnitude. The limited flux is now the convex
# combination ϑ Jᴴ + (1 - ϑ) Jᵘ of two bounded fluxes with the face value clamped to the bounds
# (see `Breeze.Advection`). The test runs the production path (the convex MIS stages, the flux-limited
# divergence and the column physics) on the raining case in which rain went negative at (2, 1, 7)
# (step 3, stage 5) and, without contraction, at (9, 1, 8) (step 2, stage 5).

function raining_fct_slice(FT)
    stepper = (timestepper = :GravityWaveMultirateInfinitesimalStep,
               timestepper_kwargs = (; tableau = :ConvexLinearMIS6, inner_solver = VerletSubsteps(), moist_gravity_wave_operator = true))
    ridge(x) = FT(9.80665 * 1500) * exp(-((x - FT(8e5)) / FT(2e5))^2)
    fct() = WENO(FT; order=5, bounds=BoundsPreservation(0, 1; flux_limiter=true))
    grid = RectilinearGrid(CPU(), FT; size=(32, 12), x=(0, 1.6e6), z=(0, 1), halo=(5, 5), topology=(Periodic, Flat, Bounded))
    model = AtmosphereModel(grid; dynamics=HydrostaticDynamics(; surface_geopotential=ridge, top_pressure=2.5e4),
                            microphysics=warm_rain_microphysics(FT), formulation=moist_formulation(),
                            scalar_advection=(ρθ = WENO(FT; order=5), ρqᵉ = fct(), ρqʳ = fct()),
                            momentum_advection=VectorInvariant(), stepper...)
    box(x, a, b) = a < x < b
    set!(model; pˢ=1e5, θ = (x, z) -> 288 + 45z, u=20)
    set!(model; qᵗ = (x, z) -> 0.030 * exp(-4z) + box(x, 5e5, 7e5) * box(z, 0.2, 0.5) * 2e-3,
                qʳ = (x, z) -> box(x, 5e5, 7e5) * box(z, 0.2, 0.5) * 2e-3)
    Oceananigans.TimeSteppers.update_state!(model)
    return model
end

@testset "Raining Float32 slice: rain stays non-negative through the FCT stages (cells (2, 1, 7), (9, 1, 8))" begin
    model = raining_fct_slice(Float32)
    column_physics_in_time_step!(model, false)
    lowest = Inf32
    for n in 1:3
        time_step!(model, 600)
        lowest = min(lowest, minimum(interior(model.microphysical_fields.ρqʳ)))
        warm_rain_column_physics!(model.microphysics, model; Δt = 600)
        lowest = min(lowest, minimum(interior(model.microphysical_fields.ρqʳ)))
    end
    @info "Raining Float32 FCT slice, 3 steps: lowest ρqʳ $lowest; ρqʳ(2, 1, 7) = $(model.microphysical_fields.ρqʳ[2, 1, 7])"
    @test lowest ≥ 0
    @test model.microphysical_fields.ρqʳ[2, 1, 7] ≥ 0
    @test model.microphysical_fields.ρqʳ[9, 1, 8] ≥ 0
end
