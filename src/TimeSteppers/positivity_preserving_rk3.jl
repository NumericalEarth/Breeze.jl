using Oceananigans: AbstractModel, prognostic_fields
using Oceananigans.Fields: CenterField
using Oceananigans.Utils: time_difference_seconds

using Oceananigans.TimeSteppers:
    AbstractTimeStepper,
    tick!,
    stage_Δt,
    rk3_substep!,
    cache_previous_tendencies!,
    update_state!,
    compute_tendencies!,
    compute_flux_bc_tendencies!,
    compute_pressure_correction!,
    make_pressure_correction!,
    step_lagrangian_particles!

import Oceananigans.TimeSteppers: time_step!

using Oceananigans.Advection:
    _advective_tracer_flux_x,
    _advective_tracer_flux_y,
    _advective_tracer_flux_z,
    BoundsPreservingWENO,
    bounded_tracer_flux_divergence_x,
    bounded_tracer_flux_divergence_y,
    bounded_tracer_flux_divergence_z

using Oceananigans.Grids: Flat
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ

"""
    PositivityPreservingRK3TimeStepper{FT, TG, CS, TI} <: AbstractTimeStepper

A third-order Runge-Kutta time stepper with directionally-split advection
for positivity-preserving tracer transport.

This time stepper uses the algorithm described by the MITgcm documentation
(https://mitgcm.org/sealion/online_documents/node80.html) for multi-dimensional
positivity-preserving advection. The key insight is that applying advection
dimension-by-dimension (rather than all at once) maintains positivity when
each 1D step uses a bounds-preserving flux limiter.

The algorithm for tracer advection at each RK3 substep is:

```math
c^{n+1/3} = c^n - Δt \\, ∇ ⋅ (\\mathbf{u}_x c^n)
c^{n+2/3} = c^{n+1/3} - Δt \\, ∇ ⋅ (\\mathbf{u}_y c^{n+1/3})
c^{n+3/3} = c^{n+2/3} - Δt \\, ∇ ⋅ (\\mathbf{u}_z c^{n+2/3})
```

The effective advection tendency is then computed as:
```math
G_{adv} = (c^{n+3/3} - c^n) / Δt
```

This is combined with diffusion and other tendencies computed at the original
time level ``n``.

Fields
======

- `γ¹, γ², γ³`: RK3 coefficients for current stage
- `ζ², ζ³`: RK3 coefficients for previous stage
- `Gⁿ`: Tendency fields at current stage
- `G⁻`: Tendency fields at previous stage
- `cˢ`: Scratch storage for intermediate tracer values (one field per positivity-preserving tracer)
- `implicit_solver`: Optional implicit solver for diffusion
"""
struct PositivityPreservingRK3TimeStepper{FT, TG, CS, TI} <: AbstractTimeStepper
    γ¹ :: FT
    γ² :: FT
    γ³ :: FT
    ζ² :: FT
    ζ³ :: FT
    Gⁿ :: TG
    G⁻ :: TG
    cˢ :: CS  # Scratch storage for intermediate tracer state during directional splitting
    implicit_solver :: TI
end

"""
    PositivityPreservingRK3TimeStepper(grid, prognostic_fields;
                                        implicit_solver = nothing,
                                        Gⁿ = map(similar, prognostic_fields),
                                        G⁻ = map(similar, prognostic_fields),
                                        positivity_preserving_tracers = Tuple())

Construct a `PositivityPreservingRK3TimeStepper` on `grid` with `prognostic_fields`.

The scheme uses the same RK3 coefficients as `RungeKutta3TimeStepper` from Le and Moin (1991),
but applies advection dimension-by-dimension for tracers to maintain positivity.

Keyword Arguments
=================

- `implicit_solver`: Optional implicit solver for diffusion. Default: `nothing`
- `Gⁿ`: Tendency fields at current stage. Default: similar to `prognostic_fields`
- `G⁻`: Tendency fields at previous stage. Default: similar to `prognostic_fields`  
- `positivity_preserving_tracers`: Names of tracers to treat with positivity-preserving
  advection. Default: empty tuple (no tracers use positivity-preserving advection).

References
==========

Le, H. and Moin, P. (1991). An improvement of fractional step methods for the incompressible
    Navier–Stokes equations. Journal of Computational Physics, 92, 369–379.

MITgcm documentation on multi-dimensional advection:
    https://mitgcm.org/sealion/online_documents/node80.html
"""
function PositivityPreservingRK3TimeStepper(grid, prognostic_fields;
                                             implicit_solver::TI = nothing,
                                             Gⁿ::TG = map(similar, prognostic_fields),
                                             G⁻ = map(similar, prognostic_fields),
                                             positivity_preserving_tracers = Tuple()) where {TI, TG}

    FT = eltype(grid)
    
    # RK3 coefficients (same as standard RungeKutta3TimeStepper)
    γ¹ = 8 // 15
    γ² = 5 // 12
    γ³ = 3 // 4
    ζ² = -17 // 60
    ζ³ = -5 // 12

    # Create scratch storage for each tracer that needs positivity-preserving treatment.
    cˢ = NamedTuple(name => CenterField(grid) for name in positivity_preserving_tracers)
    CS = typeof(cˢ)

    return PositivityPreservingRK3TimeStepper{FT, TG, CS, TI}(
        γ¹, γ², γ³, ζ², ζ³, Gⁿ, G⁻, cˢ, implicit_solver
    )
end

#####
##### Directionally-split advection
#####

"""
    compute_split_advection_tendency!(Gc, c, cˢ, grid, advection, ρ, velocities, Δt)

Compute the advection tendency for tracer `c` using directional splitting.

The algorithm applies advection dimension-by-dimension to maintain positivity:

1. Copy `c` to scratch storage `cˢ`
2. Apply x-advection: `cˢ ← cˢ - Δt * ∇ ⋅ (u * cˢ)`
3. Apply y-advection: `cˢ ← cˢ - Δt * ∇ ⋅ (v * cˢ)`  
4. Apply z-advection: `cˢ ← cˢ - Δt * ∇ ⋅ (w * cˢ)`
5. Compute tendency: `Gc += (cˢ - c) / Δt`

This maintains positivity when each 1D step uses a bounds-preserving flux limiter.
"""
function compute_split_advection_tendency!(Gc, c, cˢ, grid, advection, ρ, velocities, Δt)
    arch = architecture(grid)
    
    # Step 1: Copy current tracer state to scratch
    launch!(arch, grid, :xyz, _copy_field!, cˢ, c)

    # Step 2-4: Apply advection in each direction sequentially
    launch!(arch, grid, :xyz, _apply_x_advection!, cˢ, grid, advection, ρ, velocities.u, Δt)
    launch!(arch, grid, :xyz, _apply_y_advection!, cˢ, grid, advection, ρ, velocities.v, Δt)
    launch!(arch, grid, :xyz, _apply_z_advection!, cˢ, grid, advection, ρ, velocities.w, Δt)
    
    # Step 5: Add the effective advection tendency to existing tendency
    launch!(arch, grid, :xyz, _add_advection_tendency!, Gc, c, cˢ, Δt)

    return nothing
end

@kernel function _copy_field!(dst, src)
    i, j, k = @index(Global, NTuple)
    @inbounds dst[i, j, k] = src[i, j, k]
end

@kernel function _add_advection_tendency!(Gc, c, cˢ, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] += (cˢ[i, j, k] - c[i, j, k]) / Δt
end

#####
##### Directional advection kernels for BoundsPreservingWENO
#####
##### These use the existing bounded operators from Oceananigans which are
##### designed for single-direction flux divergence computation.
#####

@kernel function _apply_x_advection!(c, grid, advection::BoundsPreservingWENO, ρ, u, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        div_x = V⁻¹ᶜᶜᶜ(i, j, k, grid) * bounded_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, u, c)
        c[i, j, k] -= Δt * div_x
    end
end

@kernel function _apply_y_advection!(c, grid, advection::BoundsPreservingWENO, ρ, v, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        div_y = V⁻¹ᶜᶜᶜ(i, j, k, grid) * bounded_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, v, c)
        c[i, j, k] -= Δt * div_y
    end
end

@kernel function _apply_z_advection!(c, grid, advection::BoundsPreservingWENO, ρ, w, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        div_z = V⁻¹ᶜᶜᶜ(i, j, k, grid) * bounded_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, w, c)
        c[i, j, k] -= Δt * div_z
    end
end

#####
##### Time stepping
#####

"""
    time_step!(model::AbstractModel{<:PositivityPreservingRK3TimeStepper}, Δt; callbacks=[])

Step forward `model` one time step `Δt` with a 3rd-order Runge-Kutta method
using directionally-split advection for positivity-preserving tracer transport.

This is similar to the standard RK3 time stepper, but after computing tendencies,
the split advection tendencies are computed for tracers in `timestepper.cˢ`.
"""
function time_step!(model::AbstractModel{<:PositivityPreservingRK3TimeStepper}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies = true)

    γ¹ = model.timestepper.γ¹
    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ¹ = nothing
    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    first_stage_Δt  = stage_Δt(Δt, γ¹, ζ¹)
    second_stage_Δt = stage_Δt(Δt, γ², ζ²)
    third_stage_Δt  = stage_Δt(Δt, γ³, ζ³)

    # Compute the next time step a priori to reduce floating point error accumulation
    tⁿ⁺¹ = model.clock.time + Δt

    #
    # First stage
    #

    compute_flux_bc_tendencies!(model)
    compute_split_advection_tendencies!(model, first_stage_Δt)
    rk3_substep!(model, Δt, γ¹, nothing)

    tick!(model.clock, first_stage_Δt; stage=true)

    compute_pressure_correction!(model, first_stage_Δt)
    make_pressure_correction!(model, first_stage_Δt)

    cache_previous_tendencies!(model)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, first_stage_Δt)

    #
    # Second stage
    #

    compute_flux_bc_tendencies!(model)
    compute_split_advection_tendencies!(model, second_stage_Δt)
    rk3_substep!(model, Δt, γ², ζ²)

    tick!(model.clock, second_stage_Δt; stage=true)

    compute_pressure_correction!(model, second_stage_Δt)
    make_pressure_correction!(model, second_stage_Δt)

    cache_previous_tendencies!(model)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, second_stage_Δt)

    #
    # Third stage
    #

    compute_flux_bc_tendencies!(model)
    compute_split_advection_tendencies!(model, third_stage_Δt)
    rk3_substep!(model, Δt, γ³, ζ³)

    # Adjust final time-step to reduce floating point error accumulation
    corrected_third_stage_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick!(model.clock, corrected_third_stage_Δt)
    model.clock.last_stage_Δt = corrected_third_stage_Δt
    model.clock.last_Δt = Δt

    compute_pressure_correction!(model, third_stage_Δt)
    make_pressure_correction!(model, third_stage_Δt)

    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, third_stage_Δt)

    return nothing
end

#####
##### Split advection tendency computation
#####

"""
    compute_split_advection_tendencies!(model, Δt)

Compute advection tendencies via directional splitting for tracers
that use positivity-preserving advection.
"""
compute_split_advection_tendencies!(model, Δt) = nothing

# TODO: Implement for AtmosphereModel. This requires:
# 1. Access to model.tracers, model.advection, model.velocities, model.formulation.reference_state.density
# 2. The implementation should look something like:
#
# function compute_split_advection_tendencies!(model::AtmosphereModel{<:PositivityPreservingRK3TimeStepper}, Δt)
#     for name in keys(model.timestepper.cˢ)
#         c = model.tracers[name]
#         cˢ = model.timestepper.cˢ[name]
#         Gc = model.timestepper.Gⁿ[name]
#         advection = model.advection[name]
#         ρ = model.formulation.reference_state.density
#         compute_split_advection_tendency!(Gc, c, cˢ, model.grid, advection, ρ, model.velocities, Δt)
#     end
#     return nothing
# end
