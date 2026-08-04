using ..Thermodynamics: ReferenceState, ExnerReferenceState, compute_hydrostatic_reference!,
                        _compute_exner_reference!, _compute_exner_reference_3d!,
                        bottom_face_height, constant_moist_hydrostatic_pressure,
                        is_column_reference, moist_hydrostatic_pressure, dry_air_gas_constant,
                        set_surface_state!, surface_reference_density, vapor_gas_constant
using Oceananigans: CenterField
using Oceananigans: Oceananigans, prognostic_fields
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, set!, ZeroField, Field
using Oceananigans.Grids: Center, Face, znode
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ
using GPUArraysCore: @allowscalar
using Statistics: mean!

"""
    rescale_density_weighted_fields!(model, ρ⁻)

Rescale all density-weighted prognostic fields so that specific quantities
(velocity, potential temperature, moisture, etc.) are preserved after a change
in the reference density `ρᵣ`. Each field is multiplied by `ρᵣ_new / ρᵣ_old`.

Momentum fields (ρu, ρv, ρw) live at staggered face locations and require
interpolation of the cell-centered density; a dedicated kernel handles this.
All other prognostic fields are cell-centered and rescaled with broadcasting.
"""
function rescale_density_weighted_fields!(model, ρ⁻)
    grid = model.grid
    arch = grid.architecture
    ρ = dynamics_density(model.dynamics)

    # Momentum: kernel with interpolation to face locations
    launch!(arch, grid, :xyz, _rescale_momentum!, grid, model.momentum, ρ, ρ⁻)

    # Cell-centered prognostic fields: broadcasting
    formulation_fields = prognostic_fields(model.formulation)
    for field in formulation_fields
        parent(field) .*= parent(ρ) ./ parent(ρ⁻)
    end

    parent(model.moisture_density) .*= parent(ρ) ./ parent(ρ⁻)

    μ_names = prognostic_field_names(model.microphysics)
    for name in μ_names
        field = model.microphysical_fields[name]
        parent(field) .*= parent(ρ) ./ parent(ρ⁻)
    end

    for field in model.tracers
        parent(field) .*= parent(ρ) ./ parent(ρ⁻)
    end

    return nothing
end

function rescale_dry_density_weighted_fields!(model, ρᵈ⁻)
    grid = model.grid
    arch = grid.architecture
    ρᵈ = dynamics_density(model.dynamics)

    launch!(arch, grid, :xyz, _rescale_momentum!, grid, model.momentum, ρᵈ, ρᵈ⁻)

    formulation_fields = prognostic_fields(model.formulation)
    for field in formulation_fields
        parent(field) .*= parent(ρᵈ) ./ parent(ρᵈ⁻)
    end

    return nothing
end

function scale_total_density_weighted_fields!(model, ρ, ρᵈ⁻)
    parent(model.moisture_density) .*= parent(ρ) ./ parent(ρᵈ⁻)

    μ_names = prognostic_field_names(model.microphysics)
    for name in μ_names
        field = model.microphysical_fields[name]
        parent(field) .*= parent(ρ) ./ parent(ρᵈ⁻)
    end

    for field in model.tracers
        parent(field) .*= parent(ρ) ./ parent(ρᵈ⁻)
    end

    return nothing
end

@kernel function _rescale_momentum!(grid, momentum, ρ, ρ⁻)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᶠᶜᶜ  = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        ρ⁻ᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ⁻)
        momentum.ρu[i, j, k] *= ρᶠᶜᶜ / ρ⁻ᶠᶜᶜ

        ρᶜᶠᶜ  = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        ρ⁻ᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ⁻)
        momentum.ρv[i, j, k] *= ρᶜᶠᶜ / ρ⁻ᶜᶠᶜ

        ρᶜᶜᶠ  = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρ⁻ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ⁻)
        momentum.ρw[i, j, k] *= ρᶜᶜᶠ / ρ⁻ᶜᶜᶠ
    end
end

@kernel function _set_dry_density_from_total_density!(ρᵈ, ρ, microphysics, moisture_density, microphysical_fields)
    i, j, k = @index(Global, NTuple)
    ρqᵗ = total_condensate_density(i, j, k, microphysics, moisture_density, microphysical_fields)
    @inbounds ρᵈ[i, j, k] = ρ[i, j, k] - ρqᵗ
end

@kernel function _dry_weighted_specific_moisture!(qᵛ, ρqᵛ, ρᵈ)
    i, j, k = @index(Global, NTuple)
    @inbounds qᵛ[i, j, k] = ρqᵛ[i, j, k] / ρᵈ[i, j, k]
end

"""
    set_to_mean!(reference_state, model; rescale_densities=false)

Recompute the reference pressure and density profiles from horizontally-averaged
temperature and moisture mass fractions of the current model state.

When `rescale_densities=true`, density-weighted prognostic fields (ρe, ρqᵗ, ρu,
etc.) are rescaled by `ρᵣ_new / ρᵣ_old` so that the specific quantities
(e, qᵗ, u, etc.) are unchanged. When `false` (default), the density-weighted
fields are left as-is and only diagnostics are recomputed.
"""
function set_to_mean!(ref::ReferenceState, model; rescale_densities=false)
    constants = model.thermodynamic_constants

    if rescale_densities
        ρᵣ_old = similar(dynamics_density(model.dynamics))
        parent(ρᵣ_old) .= parent(dynamics_density(model.dynamics))
    end

    # Update reference temperature and moisture from horizontal means
    mean!(ref.temperature, model.temperature)
    fill_halo_regions!(ref.temperature)

    mean_mass_fraction!(ref.vapor_mass_fraction, specific_humidity(model))
    mean_mass_fraction!(ref.liquid_mass_fraction, liquid_mass_fraction(model))
    mean_mass_fraction!(ref.ice_mass_fraction, ice_mass_fraction(model))

    # Recompute hydrostatic pressure and density
    compute_hydrostatic_reference!(ref, constants)

    if rescale_densities
        rescale_density_weighted_fields!(model, ρᵣ_old)
    end

    # Recompute all diagnostic variables (T, qᵗ, u, v, w, diffusivities, etc.)
    TimeSteppers.update_state!(model; compute_tendencies=false)

    return nothing
end

@kernel function _compute_surface_pressure_from_base!(pˢ, grid, θ, qᵛ, p₀, pˢᵗ,
                                                      Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)
    i, j = @index(Global, NTuple)
    zˢ = znode(i, j, 1, grid, Center(), Center(), Face())
    @inbounds pˢ[i, j, 1] = constant_moist_hydrostatic_pressure(zˢ, p₀, θ[i, j, 1], qᵛ[i, j, 1],
                                                                pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)
end

"""
$(TYPEDSIGNATURES)

Rewrite the bottom-face pressure and density of an `ExnerReferenceState` from the horizontal-mean
near-surface state `(θˢ, qᵛˢ)`, in place.

The datum is reduced with [`moist_hydrostatic_pressure`](@ref) — the same function the constructor
anchors on — so a reset lands on the profile the constructor would have produced from this mean
state. `set_to_mean!` is only reached on a height-coordinate grid, whose bottom face is a single
level, so a horizontally uniform reduction is exact; terrain-following resets go through
`reset_reference_state!`, which reduces the datum per column along the terrain.
"""
function update_exner_surface_state!(ref::ExnerReferenceState, θ, qᵛ, grid, constants)
    FT  = eltype(ref)
    Rᵈ  = dry_air_gas_constant(constants)
    Rᵛ  = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity

    θˢ, qᵛˢ = @allowscalar (θ[1, 1, 1], qᵛ[1, 1, 1])
    zˢ = bottom_face_height(grid)
    pˢ = convert(FT, moist_hydrostatic_pressure(zˢ, ref.base_pressure, θˢ, qᵛˢ,
                                                ref.standard_pressure, constants))

    set_surface_state!(ref.surface_pressure, pˢ)
    update_exner_surface_density!(ref, pˢ, θˢ, qᵛˢ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    return nothing
end

function update_exner_surface_density!(ref::ExnerReferenceState, pˢ, θˢ, qᵛˢ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    # The 3D and terrain-following forms carry no bottom boundary value on `density`, so there is
    # nothing to keep in sync.
    isnothing(ref.surface_density) && return nothing
    ρˢ = surface_reference_density(pˢ, θˢ, qᵛˢ, ref.standard_pressure, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    set_surface_state!(ref.surface_density, convert(eltype(ref), ρˢ))
    return nothing
end

"""
    set_to_mean!(ref::ExnerReferenceState, model)

Exner analogue of the `ReferenceState` method, for split-explicit `CompressibleDynamics`. Recompute
the base `exner_function`/`pressure`/`density` by re-running the same discrete Exner column
integration the constructor uses, with the horizontal-mean liquid-ice potential temperature and vapor
mass fraction of the current model state. On height-coordinate grids the recomputed reference is
horizontally uniform. Terrain-following model resets use their specialized constant-height mean and
per-column integration path.

Unlike the anelastic `ReferenceState` method there is no `rescale_densities` option: the Exner
reference is only the perturbation-form base state, not the prognostic density (`ρᵈ`), so changing it
does not require rescaling the density-weighted prognostics.
"""
function set_to_mean!(ref::ExnerReferenceState, model)
    constants = model.thermodynamic_constants
    grid = ref.pressure.grid
    arch = architecture(grid)
    Nz   = size(grid, 3)

    # Horizontal-mean θˡⁱ and qᵛ as single-column reference profiles.
    θ̄ = Field{Nothing, Nothing, Center}(grid)
    mean!(θ̄, liquid_ice_potential_temperature(model))
    fill_halo_regions!(θ̄)

    q̄ᵛ = Field{Nothing, Nothing, Center}(grid)
    mean_mass_fraction!(q̄ᵛ, specific_humidity(model))

    Rᵈ  = dry_air_gas_constant(constants)
    Rᵛ  = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    g   = constants.gravitational_acceleration

    update_exner_surface_state!(ref, θ̄, q̄ᵛ, grid, constants)

    # A 3D reference on a height-coordinate grid has a horizontally uniform mean profile but
    # per-column fields, so each column is integrated from the (uniform) bottom-face anchor.
    kernel!, worksize = is_column_reference(ref) ? (_compute_exner_reference!, tuple(1)) :
                                                  (_compute_exner_reference_3d!, :xy)

    launch!(arch, grid, worksize, kernel!,
            ref.exner_function, ref.pressure, ref.density, θ̄, q̄ᵛ, grid, Nz,
            ref.surface_pressure, ref.standard_pressure, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)

    fill_halo_regions!(ref.exner_function)
    fill_halo_regions!(ref.pressure)
    fill_halo_regions!(ref.density)

    # Recompute all diagnostics (T, qᵗ, u, v, w, …) consistent with the new reference.
    TimeSteppers.update_state!(model; compute_tendencies=false)
    return nothing
end

"""
    reset_reference_state!(model)

Recompute the dynamics' reference state from the horizontal means of the model's current state via
[`set_to_mean!`](@ref) — works for both the anelastic `ReferenceState` and the split-explicit
`ExnerReferenceState` — if the dynamics carries one; a no-op otherwise. Invoked by
`set!(model; compute_reference_state=true)`.
"""
function reset_reference_state!(model)
    ref = dynamics_reference_state(model.dynamics)
    if ref isa ReferenceState
        set_to_mean!(ref, model; rescale_densities=true)
    elseif !isnothing(ref)
        set_to_mean!(ref, model)
    end

    return nothing
end

function mean_mass_fraction!(ref_field, field)
    mean!(ref_field, field)
    fill_halo_regions!(ref_field)
    return nothing
end

function mean_mass_fraction!(ref_field, ::Nothing)
    interior(ref_field) .= 0
    fill_halo_regions!(ref_field)
    return nothing
end

"""
    HydrostaticallyBalancedDensity(; surface_pressure = nothing)

Marker passed as the `ρ` value to [`set!`](@ref) to set the density in discrete moist hydrostatic
balance with the just-set `θˡⁱ`/`qᵛ`, by per-column integration of the hydrostatic equation upward
from the pressure at the bottom face of each column. For `CompressibleDynamics`.

When the dynamics carries an `ExnerReferenceState`, the default anchor is taken from it, so the
balanced state and the reference it will be differenced against use the same per-column pressure.
Without a reference, the anchor is obtained by reducing the dynamics' ``z = 0`` datum to each
column's bottom face along its current near-surface thermodynamic state. This matters on a
terrain-following grid, where anchoring every column at one scalar instead would leave the cold
start with no surface pressure gradient across the terrain.

`surface_pressure` overrides that anchor with a scalar applied to every column. On a
terrain-following grid, prefer the default: a scalar cannot represent the terrain-following
surface pressure, and one that disagrees with the reference reintroduces the inconsistency.

Unlike supplying a density field, this guarantees the initial column satisfies the discrete
hydrostatic balance `(pᵏ − pᵏ⁻¹)/Δz + g(ρᵏ + ρᵏ⁻¹)/2 = 0`, so the cold start carries no spurious
vertical pressure-gradient force. Combine with `compute_reference_state = true` (perturbation-form
base state) and `balancer` (nonhydrostatic `ρw` spin-up) for a full one-call initialization.
"""
struct HydrostaticallyBalancedDensity{P}
    surface_pressure :: P
end

HydrostaticallyBalancedDensity(; surface_pressure = nothing) = HydrostaticallyBalancedDensity(surface_pressure)

default_hydrostatic_surface_pressure(model, θ, qᵛ, ref::ExnerReferenceState) = ref.surface_pressure

# Without a reference state there is no callable profile to integrate below the domain bottom, so
# the datum is reduced along the constant-θ layer implied by each column's own lowest cell. That is
# the only reduction the model state supports here, and it agrees with the profile-integrating
# `moist_hydrostatic_pressure` used elsewhere whenever θ really is constant.
function default_hydrostatic_surface_pressure(model, θ, qᵛ, reference_state)
    grid = model.grid
    arch = architecture(grid)
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    Rᵈ  = dry_air_gas_constant(constants)
    Rᵛ  = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    g   = constants.gravitational_acceleration
    pˢ = Field{Center, Center, Nothing}(grid)

    launch!(arch, grid, :xy, _compute_surface_pressure_from_base!,
            pˢ, grid, θ, qᵛ, base_pressure(dynamics), standard_pressure(dynamics),
            Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)
    fill_halo_regions!(pˢ)
    return pˢ
end

"""
$(TYPEDSIGNATURES)

Set the prognostic density of a `CompressibleDynamics` model into discrete hydrostatic balance with
the current `θˡⁱ`/`qᵛ`, per [`HydrostaticallyBalancedDensity`](@ref). Runs the same per-column
Exner integration the reference-state constructor uses, then scales the dry density (and rescales
the density-weighted prognostics, preserving `θ`, `qˣ`, and velocities) so the total density matches
the balanced column.
"""
function set_hydrostatically_balanced_density!(model, spec::HydrostaticallyBalancedDensity)
    dynamics  = model.dynamics
    grid      = model.grid
    arch      = architecture(grid)
    Nz        = size(grid, 3)
    constants = model.thermodynamic_constants

    pˢᵗ = standard_pressure(dynamics)
    Rᵈ  = dry_air_gas_constant(constants)
    Rᵛ  = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    g   = constants.gravitational_acceleration

    θ  = model.formulation.potential_temperature   # specific θˡⁱ, filled by the preceding update_state!
    qᵛ = CenterField(grid)
    ρᵈ = dynamics_density(dynamics)
    ρᵈ_old = CenterField(grid)
    copyto!(parent(ρᵈ_old), parent(ρᵈ))
    launch!(arch, grid, :xyz, _dry_weighted_specific_moisture!, qᵛ, model.moisture_density, ρᵈ_old)

    reference_state = dynamics_reference_state(dynamics)
    pˢ = isnothing(spec.surface_pressure) ?
         default_hydrostatic_surface_pressure(model, θ, qᵛ, reference_state) :
         spec.surface_pressure

    # Per-column hydrostatic integration → balanced total density.
    π = CenterField(grid)
    pressure = CenterField(grid)
    ρ = CenterField(grid)
    launch!(arch, grid, :xy, _compute_exner_reference_3d!,
            π, pressure, ρ, θ, qᵛ, grid, Nz, pˢ, pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)

    # Scale total-density-weighted constituents by ρ / ρᵈ_old, set dry density as the residual,
    # then scale dry-density-weighted prognostics by ρᵈ_new / ρᵈ_old.
    scale_total_density_weighted_fields!(model, ρ, ρᵈ_old)
    launch!(arch, grid, :xyz, _set_dry_density_from_total_density!,
            ρᵈ, ρ, model.microphysics, model.moisture_density, model.microphysical_fields)
    fill_halo_regions!(ρᵈ)
    rescale_dry_density_weighted_fields!(model, ρᵈ_old)

    update_state!(model; compute_tendencies=false)
    return nothing
end

# ZeroField reference moisture: nothing to update
mean_mass_fraction!(::ZeroField, field) = nothing
mean_mass_fraction!(::ZeroField, ::Nothing) = nothing
