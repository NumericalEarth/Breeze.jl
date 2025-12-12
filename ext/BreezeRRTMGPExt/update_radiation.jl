using Oceananigans.Operators: ℑzᵃᵃᶠ

#####
##### Update radiation fluxes from model state
#####
#
# Type ownership:
#   RRTMGP types (external, cannot modify):
#     - GrayAtmosphericState: atmospheric state arrays (t_lay, p_lay, t_lev, p_lev, z_lev, t_sfc)
#     - NoScatLWRTE, NoScatSWRTE: longwave/shortwave RTE solvers
#     - FluxLW, FluxSW: flux storage (flux_up, flux_dn, flux_net, flux_dn_dir)
#
#   Breeze types (internal, can modify):
#     - GrayRadiativeTransferModel: wrapper containing RRTMGP solvers and Oceananigans flux fields
#     - SingleColumnGrid type alias
#

using KernelAbstractions: @kernel, @index
using Oceananigans.Utils: launch!

using Breeze.Thermodynamics: adiabatic_hydrostatic_pressure
using Breeze.AtmosphereModels: AtmosphereModels

"""
    $(TYPEDSIGNATURES)

Update the radiative fluxes from the current model state.

This function:
1. Updates the RRTMGP atmospheric state from model fields (T, p)
2. Computes the solar zenith angle from the model clock and grid location
3. Solves the longwave and shortwave RTE
4. Copies the fluxes to Oceananigans fields for output

Sign convention: positive flux = upward, negative flux = downward.
"""
function AtmosphereModels.update_radiation!(radiation::GrayRadiativeTransferModel, model)
    grid = model.grid
    clock = model.clock

    rrtmgp_state = radiation.atmospheric_state
    surface_temperature = radiation.surface_temperature

    # Update RRTMGP atmospheric state from model fields
    update_rrtmgp_state!(rrtmgp_state, model, surface_temperature)

    # Update solar zenith angle from clock
    update_solar_zenith_angle!(radiation.shortwave_solver, grid, clock)

    # Solve longwave RTE (RRTMGP external call)
    solve_lw!(radiation.longwave_solver, rrtmgp_state)

    # Solve shortwave RTE (only if sun is above horizon)
    cos_θz = radiation.shortwave_solver.bcs.cos_zenith[1]
    if cos_θz > 0
        solve_sw!(radiation.shortwave_solver, rrtmgp_state)
    else
        # Sun below horizon - zero shortwave fluxes
        radiation.shortwave_solver.flux.flux_up .= 0
        radiation.shortwave_solver.flux.flux_dn .= 0
        radiation.shortwave_solver.flux.flux_net .= 0
        radiation.shortwave_solver.flux.flux_dn_dir .= 0
    end

    # Copy RRTMGP flux arrays to Oceananigans fields with sign convention
    copy_fluxes_to_fields!(radiation, grid)

    return nothing
end

#####
##### Update RRTMGP atmospheric state from model fields
#####

"""
    $(TYPEDSIGNATURES)

Update the RRTMGP `GrayAtmosphericState` arrays from model fields.

# Grid staggering: layers vs levels

RRTMGP requires atmospheric state at both "layers" (cell centers) and "levels" (cell faces).
This matches the finite-volume staggering used in Oceananigans:

```
                        ┌─────────────────────────────────────────────────┐
    z_lev[Nz+1] ━━━━━━━ │  level Nz+1 (TOA):  p_lev, t_lev, z_lev         │ ← extrapolated
                        └─────────────────────────────────────────────────┘
                        ┌─────────────────────────────────────────────────┐
                        │  layer Nz:  T[Nz], p_lay[Nz] = pᵣ[Nz]           │ ← from model
                        └─────────────────────────────────────────────────┘
    z_lev[Nz]   ━━━━━━━   level Nz:   p_lev, t_lev, z_lev                   ← interpolated
                        ┌─────────────────────────────────────────────────┐
                        │  layer Nz-1                                     │
                        └─────────────────────────────────────────────────┘
                                            ⋮
                        ┌─────────────────────────────────────────────────┐
                        │  layer 2                                        │
                        └─────────────────────────────────────────────────┘
    z_lev[2]    ━━━━━━━   level 2:    p_lev, t_lev, z_lev                   ← interpolated
                        ┌─────────────────────────────────────────────────┐
                        │  layer 1:   T[1], p_lay[1] = pᵣ[1]              │ ← from model
                        └─────────────────────────────────────────────────┘
    z_lev[1]    ━━━━━━━   level 1 (surface, z=0):  p_lev = p₀, t_lev      │ ← from reference state
                        ══════════════════════════════════════════════════
                                        GROUND (t_sfc)
```

# Why the model must provide level values

RRTMGP is a general-purpose radiative transfer solver that operates on columns of
atmospheric data. It does not interpolate from layers to levels internally because:

1. **Boundary conditions**: The surface (level 1) and TOA (level Nz+1) require
   boundary values that only the atmospheric model knows. For pressure, we use
   the reference state's `surface_pressure` at z=0. For the top, we extrapolate
   using the adiabatic hydrostatic formula.

2. **Physics-appropriate interpolation**: Different quantities need different
   interpolation methods. Pressure uses geometric mean (log-linear interpolation)
   because it varies exponentially with height. Temperature uses arithmetic mean.

3. **Model consistency**: The pressure profile must be consistent with the
   atmospheric model's reference state. RRTMGP has no knowledge of the anelastic
   approximation or the reference potential temperature θ₀.

# Physics notes

**Temperature**: We use the actual temperature field `T` from the model state.
This is the temperature that matters for thermal emission and absorption.

**Pressure**: In the anelastic approximation, pressure perturbations are negligible
compared to the hydrostatic reference pressure. We use `reference_state.pressure`
at cell centers, computed via `adiabatic_hydrostatic_pressure(z, p₀, θ₀)`.

# RRTMGP array layout
- Layer arrays `(nlay, ncol)`: values at cell centers, layer 1 at bottom
- Level arrays `(nlev, ncol)`: values at cell faces, level 1 at surface (z=0)
- `nlev = nlay + 1`
"""
function update_rrtmgp_state!(rrtmgp_state::GrayAtmosphericState, model, surface_temperature)
    grid = model.grid
    arch = architecture(grid)
    
    # Temperature field (actual temperature from model state)
    # Reference state provides the hydrostatic pressure profile
    # In the anelastic approximation, pressure ≈ reference pressure
    p = model.formulation.reference_state.pressure
    T = model.temperature
    T₀ = surface_temperature

    launch!(arch, grid, :xyz, _update_rrtmgp_state!, rrtmgp_state, grid, p, T, T₀)

    return nothing
end

@inline rrtmgp_column_index(i, j, Nx) = i + (j - 1) * Nx

@kernel function _update_rrtmgp_state!(rrtmgp_state, grid, p, T, surface_temperature)
    i, j, k = @index(Global, NTuple)

    Nz = size(grid, 3)

    # Unpack RRTMGP arrays with Oceananigans naming conventions:
    #   ᶜ = cell center (RRTMGP "layer")
    #   ᶠ = cell face (RRTMGP "level")
    # Note: latitude (lat) and altitude (z_lev) are fixed at construction time
    Tᶜ = rrtmgp_state.t_lay  # Temperature at cell centers
    pᶜ = rrtmgp_state.p_lay  # Pressure at cell centers
    Tᶠ = rrtmgp_state.t_lev  # Temperature at cell faces
    pᶠ = rrtmgp_state.p_lev  # Pressure at cell faces
    T₀ = rrtmgp_state.t_sfc  # Surface temperature

    col = rrtmgp_column_index(i, j, grid.Nx)

    @inbounds begin
        # Surface temperature (scalar in this implementation)
        if k == 1
            T₀[col] = surface_temperature[i, j, 1]
        end

        Tᶜ[k, col] = T[i, j, k]
        pᶜ[k, col] = p[i, j, k]

        pᶠ[1, col] = ℑzᵃᵃᶠ(i, j, k, grid, p)
        Tᶠ[1, col] = ℑzᵃᵃᶠ(i, j, k, grid, T)
    end
end

#####
##### Update solar zenith angle
#####

"""
    $(TYPEDSIGNATURES)

Update the solar zenith angle in the shortwave solver from the model clock.

Uses the datetime from `clock.time` and the grid's location (latitude/longitude)
to compute the cosine of the solar zenith angle via celestial mechanics.
"""
function update_solar_zenith_angle!(sw_solver, grid::SingleColumnGrid, clock::DateTimeClock)
    cos_θz = cos_solar_zenith_angle(1, 1, grid, clock.time)
    sw_solver.bcs.cos_zenith[1] = max(cos_θz, 0)  # Clamp to positive (sun above horizon)
    return nothing
end

#####
##### Copy RRTMGP fluxes to Oceananigans fields
#####

"""
    $(TYPEDSIGNATURES)

Copy RRTMGP flux arrays to Oceananigans ZFaceFields.

Applies sign convention: positive = upward, negative = downward.
For the non-scattering shortwave solver, only the direct beam flux is computed.
"""
function copy_fluxes_to_fields!(radiation::GrayRadiativeTransferModel, grid)
    arch = architecture(grid)
    Nz = size(grid, 3)
    
    # Unpack flux arrays from RRTMGP solvers
    lw_flux_up = radiation.longwave_solver.flux.flux_up
    lw_flux_dn = radiation.longwave_solver.flux.flux_dn
    sw_flux_dn_dir = radiation.shortwave_solver.flux.flux_dn_dir

    # Unpack Oceananigans output fields
    ℐ_lw_up = radiation.upwelling_longwave_flux
    ℐ_lw_dn = radiation.downwelling_longwave_flux
    ℐ_sw_dn = radiation.downwelling_shortwave_flux

    Nx, Ny, Nz = size(grid)
    launch!(arch, grid, (Nx, Ny, Nz+1), _copy_rrtmgp_fluxes!,
            ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn, lw_flux_up, lw_flux_dn, sw_flux_dn_dir, grid)

    return nothing
end

@kernel function _copy_rrtmgp_fluxes!(ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn, 
                                      lw_flux_up, lw_flux_dn, sw_flux_dn_dir, grid)
    i, j, k = @index(Global, NTuple)

    # RRTMGP uses (nlev, ncol), we use (i, j, k) for ZFaceField
    # Sign convention: upwelling positive, downwelling negative
    col = rrtmgp_column_index(i, j, grid.Nx)

    @inbounds begin
        ℐ_lw_up[i, j, k] = lw_flux_up[k, col]
        ℐ_lw_dn[i, j, k] = -lw_flux_dn[k, col]  # Negate for downward
        ℐ_sw_dn[i, j, k] = -sw_flux_dn_dir[k, col]  # Negate for downward
    end
end

# Default no-op for models without radiation
update_radiation!(::Nothing, model) = nothing
