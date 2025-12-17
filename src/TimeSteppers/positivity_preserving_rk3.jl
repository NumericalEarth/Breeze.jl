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
    FluxFormAdvection,
    bounded_tracer_flux_divergence_x,
    bounded_tracer_flux_divergence_y,
    bounded_tracer_flux_divergence_z

using Oceananigans.BoundaryConditions: fill_halo_regions!

using Oceananigans.Grids: Flat
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, ∂xᶜᶜᶜ, ∂yᶜᶜᶜ, ∂zᶜᶜᶜ

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
- `split_advection`: Advection schemes for positivity-preserving tracers (stored here, not in model)
- `implicit_solver`: Optional implicit solver for diffusion
"""
struct PositivityPreservingRK3TimeStepper{FT, TG, CS, AD, TI} <: AbstractTimeStepper
    γ¹ :: FT
    γ² :: FT
    γ³ :: FT
    ζ² :: FT
    ζ³ :: FT
    Gⁿ :: TG
    G⁻ :: TG
    cˢ :: CS  # Scratch storage for intermediate tracer state during directional splitting
    split_advection :: AD  # Advection schemes for positivity-preserving tracers
    implicit_solver :: TI
end

"""
    PositivityPreservingRK3TimeStepper(grid, prognostic_fields;
                                        implicit_solver = nothing,
                                        Gⁿ = map(similar, prognostic_fields),
                                        G⁻ = map(similar, prognostic_fields),
                                        split_advection = NamedTuple())

Construct a `PositivityPreservingRK3TimeStepper` on `grid` with `prognostic_fields`.

The scheme uses the same RK3 coefficients as `RungeKutta3TimeStepper` from Le and Moin (1991),
but applies advection dimension-by-dimension for tracers to maintain positivity.

Keyword Arguments
=================

- `implicit_solver`: Optional implicit solver for diffusion. Default: `nothing`
- `Gⁿ`: Tendency fields at current stage. Default: similar to `prognostic_fields`
- `G⁻`: Tendency fields at previous stage. Default: similar to `prognostic_fields`  
- `split_advection`: NamedTuple of advection schemes for positivity-preserving tracers.
  These advection schemes are stored in the time stepper and used for directionally-split
  advection. **Important**: The corresponding tracers in the model should have 
  `advection = nothing` to avoid double-counting the advection tendency.

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
                                             split_advection::AD = NamedTuple()) where {TI, TG, AD}

    FT = eltype(grid)
    
    # RK3 coefficients (same as standard RungeKutta3TimeStepper)
    γ¹ = 8 // 15
    γ² = 5 // 12
    γ³ = 3 // 4
    ζ² = -17 // 60
    ζ³ = -5 // 12

    # Create scratch storage for each tracer that needs positivity-preserving treatment.
    # The tracer names come from the keys of split_advection.
    cˢ = NamedTuple(name => CenterField(grid) for name in keys(split_advection))
    CS = typeof(cˢ)

    return PositivityPreservingRK3TimeStepper{FT, TG, CS, AD, TI}(
        γ¹, γ², γ³, ζ², ζ³, Gⁿ, G⁻, cˢ, split_advection, implicit_solver
    )
end

#####
##### Directionally-split advection
#####

"""
    apply_split_advection!(c, cˢ, grid, advection, ρ, velocities, Δt)

Apply advection to tracer `c` using directional splitting, updating `c` in-place.

The algorithm applies advection dimension-by-dimension to maintain positivity,
following the MITgcm algorithm (equations 2.201-2.203):

1. Copy `c` to scratch storage `cˢ` (cˢ will be updated, c is preserved as cⁿ)
2. Apply x-advection: `cˢ ← cˢ - Δt * (∇ ⋅ (u * cˢ) + cⁿ * ∂u/∂x)`
3. Apply y-advection: `cˢ ← cˢ - Δt * (∇ ⋅ (v * cˢ) + cⁿ * ∂v/∂y)`
4. Apply z-advection: `cˢ ← cˢ - Δt * (∇ ⋅ (w * cˢ) + cⁿ * ∂w/∂z)`
5. Copy result back: `c ← cˢ`

Key: The flux divergence uses the UPDATED cˢ (for positivity preservation),
but the velocity divergence term uses the ORIGINAL c (= cⁿ) for all steps.

This maintains positivity when each 1D step uses a bounds-preserving flux limiter.

Note: This function directly updates the tracer field `c`, bypassing the normal
tendency-based time stepping. This is necessary because split advection cannot
be properly integrated via the RK3 tendency blending mechanism.
"""
function apply_split_advection!(c, cˢ, grid, advection, ρ, velocities, Δt)
    # Step 1: Copy current tracer state to scratch (including halos)
    # c is preserved as the "original" value (cⁿ) for the velocity divergence term
    parent(cˢ) .= parent(c)

    # Step 2-4: Apply advection in each direction sequentially
    # - First argument (cˢ): field being updated, also used for flux divergence
    # - Second argument (c): original field, used for velocity divergence correction
    # This matches MITgcm equations where velocity divergence always uses τ^n
    apply_x_advection!(cˢ, c, advection, grid, ρ, velocities.u, Δt)
    apply_y_advection!(cˢ, c, advection, grid, ρ, velocities.v, Δt)
    apply_z_advection!(cˢ, c, advection, grid, ρ, velocities.w, Δt)
    
    # Step 5: Copy result back to tracer field
    parent(c) .= parent(cˢ)

    return nothing
end

# Fallbacks for unsupported advection schemes
apply_x_advection!(cˢ, cⁿ, advection, grid, args...) = nothing
apply_y_advection!(cˢ, cⁿ, advection, grid, args...) = nothing
apply_z_advection!(cˢ, cⁿ, advection, grid, args...) = nothing

# FluxFormAdvection: dispatch to component schemes
apply_x_advection!(cˢ, cⁿ, advection::FluxFormAdvection, grid, args...) = apply_x_advection!(cˢ, cⁿ, advection.x, grid, args...)
apply_y_advection!(cˢ, cⁿ, advection::FluxFormAdvection, grid, args...) = apply_y_advection!(cˢ, cⁿ, advection.y, grid, args...)
apply_z_advection!(cˢ, cⁿ, advection::FluxFormAdvection, grid, args...) = apply_z_advection!(cˢ, cⁿ, advection.z, grid, args...)

# BoundsPreservingWENO: apply kernel and fill halos
function apply_x_advection!(cˢ, cⁿ, advection::BoundsPreservingWENO, grid, ρ, u, Δt)
    launch!(grid.architecture, grid, :xyz, _apply_x_advection!, cˢ, cⁿ, advection, grid, ρ, u, Δt)
    fill_halo_regions!(cˢ)
    return nothing
end

function apply_y_advection!(cˢ, cⁿ, advection::BoundsPreservingWENO, grid, ρ, v, Δt)
    launch!(grid.architecture, grid, :xyz, _apply_y_advection!, cˢ, cⁿ, advection, grid, ρ, v, Δt)
    fill_halo_regions!(cˢ)
    return nothing
end

function apply_z_advection!(cˢ, cⁿ, advection::BoundsPreservingWENO, grid, ρ, w, Δt)
    launch!(grid.architecture, grid, :xyz, _apply_z_advection!, cˢ, cⁿ, advection, grid, ρ, w, Δt)
    fill_halo_regions!(cˢ)
    return nothing
end


#####
##### Directional advection kernels for BoundsPreservingWENO
#####
##### These implement the MITgcm operator splitting algorithm (equations 2.201-2.203):
##### - The flux divergence uses the UPDATED tracer cˢ (for positivity preservation)
##### - The velocity divergence uses the ORIGINAL tracer cⁿ (for correct splitting)
#####
##### Each directional step computes:
#####   cˢ -= Δt * (1/V * δ(Flux(cˢ)) + cⁿ * δ(velocity)/Δ)
#####

@kernel function _apply_x_advection!(cˢ, cⁿ, advection, grid, ρ, u, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # Flux divergence term: uses updated cˢ
        flux_div = V⁻¹ᶜᶜᶜ(i, j, k, grid) * bounded_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, u, cˢ)
        # Velocity divergence term: uses original cⁿ (MITgcm eq 2.201)
        vel_div = cⁿ[i, j, k] * ∂xᶜᶜᶜ(i, j, k, grid, u)
        cˢ[i, j, k] -= Δt * (flux_div + vel_div)
    end
end

@kernel function _apply_y_advection!(cˢ, cⁿ, advection, grid, ρ, v, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # Flux divergence term: uses updated cˢ
        flux_div = V⁻¹ᶜᶜᶜ(i, j, k, grid) * bounded_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, v, cˢ)
        # Velocity divergence term: uses original cⁿ (MITgcm eq 2.202)
        vel_div = cⁿ[i, j, k] * ∂yᶜᶜᶜ(i, j, k, grid, v)
        cˢ[i, j, k] -= Δt * (flux_div + vel_div)
    end
end

@kernel function _apply_z_advection!(cˢ, cⁿ, advection, grid, ρ, w, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # Flux divergence term: uses updated cˢ
        flux_div = V⁻¹ᶜᶜᶜ(i, j, k, grid) * bounded_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, w, cˢ)
        # Velocity divergence term: uses original cⁿ (MITgcm eq 2.203)
        vel_div = cⁿ[i, j, k] * ∂zᶜᶜᶜ(i, j, k, grid, w)
        cˢ[i, j, k] -= Δt * (flux_div + vel_div)
    end
end

#####
##### Time stepping
#####

"""
    time_step!(model::AbstractModel{<:PositivityPreservingRK3TimeStepper}, Δt; callbacks=[])

Step forward `model` one time step `Δt` with a 3rd-order Runge-Kutta method
using directionally-split advection for positivity-preserving tracer transport.

For tracers with split advection:
1. Standard RK3 substep advances non-advection tendencies (diffusion, forcing)
2. Split advection is applied directly AFTER rk3_substep! as a direct field update

This approach is necessary because the split advection algorithm requires using
the result of each directional step as input to the next, which is incompatible
with the RK3 multi-stage tendency blending mechanism.
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
    apply_split_advection_updates!(model, first_stage_Δt)
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
    apply_split_advection_updates!(model, second_stage_Δt)
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
    apply_split_advection_updates!(model, third_stage_Δt)
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
##### Split advection updates
#####

"""
    apply_split_advection_updates!(model, Δt)

Apply directionally-split advection directly to tracer fields.

This is called AFTER rk3_substep! to apply the split advection as a direct
update, bypassing the tendency-based RK3 blending. This is necessary because
the split advection algorithm requires each directional step to use the
result of the previous step, which is incompatible with the RK3 multi-stage
tendency blending.
"""
apply_split_advection_updates!(model, Δt) = nothing

using Breeze.AtmosphereModels: AtmosphereModel

function apply_split_advection_updates!(model::AtmosphereModel{<:Any, <:Any, <:PositivityPreservingRK3TimeStepper}, Δt)
    timestepper = model.timestepper
    for name in keys(timestepper.cˢ)
        c = model.tracers[name]
        cˢ = timestepper.cˢ[name]
        # Use advection from the time stepper, NOT from the model.
        # The model should have advection = nothing for these tracers to avoid double-counting.
        advection = timestepper.split_advection[name]
        ρ = model.formulation.reference_state.density
        apply_split_advection!(c, cˢ, model.grid, advection, ρ, model.velocities, Δt)
    end
    return nothing
end
