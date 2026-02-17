#####
##### GrayRadiativeTransferModel: Gray atmosphere radiative transfer model
#####
##### Uses the O'Gorman and Schneider (2008) optical thickness parameterization
##### with RRTMGP's two-stream or no-scattering solvers.
#####

using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.Grids: xnode, ynode, λnode, φnode, znodes
using Oceananigans.Grids: AbstractGrid, RectilinearGrid, Center, Face, Flat, Bounded
using Oceananigans.Fields: ConstantField
using Breeze.AtmosphereModels: AtmosphereModels, SurfaceRadiativeProperties, RadiativeTransferModel

using RRTMGP.AtmosphericStates: GrayAtmosphericState, GrayOpticalThicknessOGorman2008
using KernelAbstractions: @kernel, @index
using Dates: AbstractDateTime, Millisecond

# Dispatch on background_atmosphere = Nothing for gray radiation
const GrayRadiativeTransferModel = RadiativeTransferModel{<:Any, <:Any, <:Any, <:Any, Nothing, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}

materialize_surface_property(x::Number, grid) = convert(eltype(grid), x)
materialize_surface_property(x::Field, grid) = x

function maybe_infer_coordinate(::Nothing, grid::SingleColumnGrid)
    λ = xnode(1, 1, 1, grid, Center(), Center(), Center())
    φ = ynode(1, 1, 1, grid, Center(), Center(), Center())
    # TODO: Validate this?
    return (λ, φ)
end

# Leave as-is: coordinate likely inferred from grid in kernels
maybe_infer_coordinate(coordinate, grid) = coordinate

# For fixed zenith angle (RCEMIP-style perpetual insolation), store the cos(zenith)
maybe_infer_coordinate(cos_θ::Number, grid) = cos_θ

# TODO: blacklist invalid coordinate/grid combinations

"""
$(TYPEDSIGNATURES)

Construct a gray atmosphere radiative transfer model for the given grid.

# Keyword Arguments
- `optical_thickness`: Optical thickness parameterization (default: `GrayOpticalThicknessOGorman2008(FT)`).
- `surface_temperature`: Surface temperature in Kelvin (required).
- `coordinate`: Solar geometry specification. Can be:
  - `nothing` (default): extracts location from grid coordinates for time-varying zenith angle
  - `(longitude, latitude)` tuple in degrees: uses DateTime clock for time-varying zenith angle
  - A `Number` representing fixed `cos(zenith_angle)`: perpetual insolation for RCE experiments
    (e.g., `cosd(42.04) ≈ 0.743` for RCEMIP protocol)
- `epoch`: Optional epoch for computing time with floating-point clocks.
- `surface_emissivity`: Surface emissivity, 0-1 (default: 0.98). Scalar.
- `surface_albedo`: Surface albedo, 0-1. Can be scalar or 2D field.
                    Alternatively, provide both `direct_surface_albedo` and `diffuse_surface_albedo`.
- `direct_surface_albedo`: Direct surface albedo, 0-1. Can be scalar or 2D field.
- `diffuse_surface_albedo`: Diffuse surface albedo, 0-1. Can be scalar or 2D field.
- `solar_constant`: Top-of-atmosphere solar flux in W/m² (default: 1361)
"""
function AtmosphereModels.RadiativeTransferModel(grid::AbstractGrid,
                                                 ::GrayOptics,
                                                 constants::ThermodynamicConstants;
                                                 optical_thickness = GrayOpticalThicknessOGorman2008(eltype(grid)),
                                                 surface_temperature,
                                                 coordinate = nothing,
                                                 epoch = nothing,
                                                 surface_emissivity = 0.98,
                                                 direct_surface_albedo = nothing,
                                                 diffuse_surface_albedo = nothing,
                                                 surface_albedo = nothing,
                                                 solar_constant = 1361,
                                                 schedule = IterationInterval(1))

    FT = eltype(grid)
    parameters = RRTMGPParameters(constants)

    error_msg = "Must either provide surface_albedo or *both* of
                 direct_surface_albedo and diffuse_surface_albedo"

    coordinate = maybe_infer_coordinate(coordinate, grid)

    if !isnothing(surface_albedo)
        if !isnothing(direct_surface_albedo) || !isnothing(diffuse_surface_albedo)
            throw(ArgumentError(error_msg))
        end

        surface_albedo = materialize_surface_property(surface_albedo, grid)
        diffuse_surface_albedo = surface_albedo
        direct_surface_albedo = surface_albedo

    elseif !isnothing(diffuse_surface_albedo) && !isnothing(direct_surface_albedo)
        direct_surface_albedo = materialize_surface_property(direct_surface_albedo, grid)
        diffuse_surface_albedo = materialize_surface_property(diffuse_surface_albedo, grid)
    else
        throw(ArgumentError(error_msg))
    end

    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    Nc = Nx * Ny

    # Set up RRTMGP grid parameters
    context = rrtmgp_context(arch)
    ArrayType = ClimaComms.array_type(context.device)

    # Allocate RRTMGP arrays with dimensions (Nz, Nc) for layers or (Nz+1, Nc) for levels
    # Note: RRTMGP uses "lat" internally for its GrayAtmosphericState struct
    rrtmgp_φ  = ArrayType{FT}(undef, Nc)
    rrtmgp_T₀ = ArrayType{FT}(undef, Nc)
    rrtmgp_pᶜ = ArrayType{FT}(undef, Nz, Nc)
    rrtmgp_Tᶜ = ArrayType{FT}(undef, Nz, Nc)
    rrtmgp_Tᶠ = ArrayType{FT}(undef, Nz+1, Nc)
    rrtmgp_pᶠ = ArrayType{FT}(undef, Nz+1, Nc)
    rrtmgp_zᶠ = ArrayType{FT}(undef, Nz+1, Nc)

    set_latitude!(rrtmgp_φ, coordinate, grid)

    # Set z_lev (altitude at cell faces) - this is fixed and doesn't change during simulation
    zf = znodes(grid, Face())
    rrtmgp_zᶠ .= zf

    atmospheric_state = GrayAtmosphericState(rrtmgp_φ,
                                             rrtmgp_pᶜ,
                                             rrtmgp_pᶠ,
                                             rrtmgp_Tᶜ,
                                             rrtmgp_Tᶠ,
                                             rrtmgp_zᶠ,
                                             rrtmgp_T₀,
                                             optical_thickness)

    # Surface emissivity for the longwave solver
    rrtmgp_ε₀ = ArrayType{FT}(undef, Nc)

    # Shortwave solver objects
    cos_zenith = ArrayType{FT}(undef, Nc) # Cosine of the solar zenith angle
    rrtmgp_αb₀ = ArrayType{FT}(undef, Nc)  # Direct surface albedo
    rrtmgp_αw₀ = ArrayType{FT}(undef, Nc)  # Diffuse surface albedo
    rrtmgp_ℐ₀ = ArrayType{FT}(undef, Nc)  # Top-of-atmosphere solar flux
    rrtmgp_ε₀ = ArrayType{FT}(undef, Nc)  # Surface emissivity

    rrtmgp_ℐ₀ .= convert(FT, solar_constant)  # Top-of-atmosphere solar flux

    if surface_emissivity isa Number
        surface_emissivity = ConstantField(convert(FT, surface_emissivity))
        rrtmgp_ε₀ .= surface_emissivity.constant
    end

    if surface_temperature isa Number
        surface_temperature = ConstantField(convert(FT, surface_temperature))
        rrtmgp_T₀ .= surface_temperature.constant
    end

    if direct_surface_albedo isa Number
        direct_surface_albedo = ConstantField(convert(FT, direct_surface_albedo))
        rrtmgp_αb₀ .= direct_surface_albedo.constant
    end

    if diffuse_surface_albedo isa Number
        diffuse_surface_albedo = ConstantField(convert(FT, diffuse_surface_albedo))
        rrtmgp_αw₀ .= diffuse_surface_albedo.constant
    end

    grid_parameters = RRTMGPGridParams(FT; context, nlay=Nz, ncol=Nc)

    longwave_solver = NoScatLWRTE(grid_parameters;
                                  params = parameters,
                                  sfc_emis = rrtmgp_ε₀,
                                  inc_flux = nothing)

    shortwave_solver = NoScatSWRTE(grid_parameters;
                                   cos_zenith = cos_zenith,
                                   toa_flux = rrtmgp_ℐ₀,
                                   sfc_alb_direct = rrtmgp_αb₀,
                                   sfc_alb_diffuse = rrtmgp_αw₀,
                                   inc_flux_diffuse = nothing)

    # Create Oceananigans fields to store fluxes for output/plotting
    upwelling_longwave_flux = ZFaceField(grid)
    downwelling_longwave_flux = ZFaceField(grid)
    downwelling_shortwave_flux = ZFaceField(grid)  # Direct beam only
    heating_tendency = CenterField(grid)

    surface_properties = SurfaceRadiativeProperties(surface_temperature,
                                                    surface_emissivity,
                                                    direct_surface_albedo,
                                                    diffuse_surface_albedo)

    return RadiativeTransferModel(convert(FT, solar_constant),
                                  coordinate,
                                  epoch,
                                  surface_properties,
                                  nothing,  # background_atmosphere = nothing for gray
                                  atmospheric_state,
                                  longwave_solver,
                                  shortwave_solver,
                                  upwelling_longwave_flux,
                                  downwelling_longwave_flux,
                                  downwelling_shortwave_flux,
                                  heating_tendency,
                                  nothing,  # liquid_effective_radius = nothing for gray
                                  nothing,  # ice_effective_radius = nothing for gray
                                  schedule)
end

@inline rrtmgp_column_index(i, j, Nx) = i + (j - 1) * Nx

function set_latitude!(rrtmgp_latitude, coordinate, grid)
    # TODO: could launch kernel over xy to support ensembles of columns
    φ = coordinate[2]
    rrtmgp_latitude .= φ
    return nothing
end

function set_latitude!(rrtmgp_latitude, ::Nothing, grid)
    arch = grid.architecture
    launch!(arch, grid, :xy, _set_latitude_from_grid!, rrtmgp_latitude, grid)
    return nothing
end

@kernel function _set_latitude_from_grid!(rrtmgp_latitude, grid)
    i, j = @index(Global, NTuple)
    φ = ynode(i, j, 1, grid, Center(), Center(), Center())
    c = rrtmgp_column_index(i, j, grid.Nx)
    @inbounds rrtmgp_latitude[c] = φ
end

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
#     - RadiativeTransferModel: wrapper containing RRTMGP solvers and Oceananigans flux fields
#     - SingleColumnGrid type alias
#

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
function AtmosphereModels._update_radiation!(rtm::GrayRadiativeTransferModel, model)
    grid = model.grid
    clock = model.clock

    rrtmgp_state = rtm.atmospheric_state
    surface_temperature = rtm.surface_properties.surface_temperature

    # Update RRTMGP atmospheric state from model fields
    update_rrtmgp_state!(rrtmgp_state, model, surface_temperature)

    rrtmgp_surface_properties = (;
        rrtmgp_ε₀ = rtm.longwave_solver.bcs.sfc_emis,
        rrtmgp_αb₀ = rtm.shortwave_solver.bcs.sfc_alb_direct,
        rrtmgp_αw₀ = rtm.shortwave_solver.bcs.sfc_alb_diffuse,
    )

    update_rrtmgp_surface_properties!(rrtmgp_surface_properties, rtm.surface_properties)

    # Update solar zenith angle from clock
    datetime = compute_datetime(clock.time, rtm.epoch)
    update_solar_zenith_angle!(rtm.shortwave_solver, rtm.coordinate, grid, datetime)

    # Solve longwave RTE (RRTMGP external call)
    solve_lw!(rtm.longwave_solver, rrtmgp_state)

    # Solve shortwave RTE
    # Note: RRTMGP handles the case when sun is below horizon (cos_zenith <= 0)
    # by producing zero fluxes internally
    solve_sw!(rtm.shortwave_solver, rrtmgp_state)

    # Copy RRTMGP flux arrays to Oceananigans fields with sign convention
    copy_fluxes_to_fields!(rtm, grid)

    # Compute heating tendency from flux divergence
    compute_radiation_heating!(rtm, grid)

    return nothing
end

# TODO: This function will launch a kernel that will update the boundary conditions of RRTMGP.
function update_rrtmgp_surface_properties!(rrtmgp_surface_properties, surface_properties)
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
- Layer arrays `(Nz, Nc)`: values at cell centers, layer 1 at bottom
- Level arrays `(Nz+1, Nc)`: values at cell faces, level 1 at surface (z=0)
"""
function update_rrtmgp_state!(rrtmgp_state::GrayAtmosphericState, model, surface_temperature)
    grid = model.grid
    arch = architecture(grid)

    # Temperature field (actual temperature from model state)
    # Reference state provides the hydrostatic pressure profile
    # In the anelastic approximation, pressure ≈ reference pressure
    p = model.dynamics.reference_state.pressure
    T = model.temperature
    T₀ = surface_temperature

    launch!(arch, grid, :xyz, _update_rrtmgp_state!, rrtmgp_state, grid, p, T, T₀)

    return nothing
end

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

    c = rrtmgp_column_index(i, j, grid.Nx)

    @inbounds begin
        # Layer values (cell centers) - k runs from 1 to Nz
        Tᶜ[k, c] = T[i, j, k]
        pᶜ[k, c] = p[i, j, k]
        pᶠ[k, c] = ℑzᵃᵃᶠ(i, j, k, grid, p)
        Tᶠ[k, c] = ℑzᵃᵃᶠ(i, j, k, grid, T)

        # Special case setting the topmost level + surface temperature
        # Because kernel spans (Nx, Ny, Nz)
        if k == 1
            T₀[c] = surface_temperature[i, j, 1]
            pᶠ[Nz+1, c] = ℑzᵃᵃᶠ(i, j, Nz+1, grid, p)
            Tᶠ[Nz+1, c] = ℑzᵃᵃᶠ(i, j, Nz+1, grid, T)
        end
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

Does not support anything but single-column grids for now.
"""
function update_solar_zenith_angle!(sw_solver, coordinate::Tuple, grid, datetime)
    cos_θz = cos_solar_zenith_angle(datetime, coordinate...)
    sw_solver.bcs.cos_zenith .= max.(cos_θz, 0)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Update the solar zenith angle for perpetual/fixed insolation (RCEMIP-style).

When `cos_zenith_angle` is a `Number`, it represents a constant cosine of the
solar zenith angle that does not vary with time. This is used for radiative-convective
equilibrium experiments with perpetual insolation.
"""
function update_solar_zenith_angle!(sw_solver, cos_zenith_angle::Number, grid, datetime)
    sw_solver.bcs.cos_zenith .= max(cos_zenith_angle, 0)
    return nothing
end

function update_solar_zenith_angle!(sw_solver, ::Nothing, grid, datetime)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _update_solar_zenith_angle_from_grid!, sw_solver.bcs.cos_zenith, grid, datetime)
    return nothing
end

@kernel function _update_solar_zenith_angle_from_grid!(rrtmgp_cos_θz, grid, datetime)
    i, j = @index(Global, NTuple)
    λ = λnode(i, j, 1, grid, Center(), Center(), Center())
    φ = φnode(i, j, 1, grid, Center(), Center(), Center())
    cos_θz = cos_solar_zenith_angle(datetime, λ, φ)
    c = rrtmgp_column_index(i, j, grid.Nx)
    rrtmgp_cos_θz[c] = max(cos_θz, 0)  # Clamp to positive (sun above horizon)
end

#####
##### Copy RRTMGP fluxes to Oceananigans fields
#####

"""
$(TYPEDSIGNATURES)

Copy RRTMGP flux arrays to Oceananigans ZFaceFields.

Applies sign convention:
* positive = upward
* negative = downward.

For the non-scattering shortwave solver, only the direct beam flux is computed.
"""
function copy_fluxes_to_fields!(rtm::GrayRadiativeTransferModel, grid)
    arch = architecture(grid)
    Nz = size(grid, 3)

    # Unpack flux arrays from RRTMGP solvers
    lw_flux_up = rtm.longwave_solver.flux.flux_up
    lw_flux_dn = rtm.longwave_solver.flux.flux_dn
    sw_flux_dn_dir = rtm.shortwave_solver.flux.flux_dn_dir

    # Unpack Oceananigans output fields
    ℐ_lw_up = rtm.upwelling_longwave_flux
    ℐ_lw_dn = rtm.downwelling_longwave_flux
    ℐ_sw_dn = rtm.downwelling_shortwave_flux

    Nx, Ny, Nz = size(grid)
    launch!(arch, grid, (Nx, Ny, Nz+1), _copy_gray_fluxes!,
            ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn, lw_flux_up, lw_flux_dn, sw_flux_dn_dir, grid)

    return nothing
end

@kernel function _copy_gray_fluxes!(ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn,
                                    lw_flux_up, lw_flux_dn, sw_flux_dn_dir, grid)
    i, j, k = @index(Global, NTuple)

    # RRTMGP uses (Nz+1, Nc), we use (i, j, k) for ZFaceField
    # Sign convention: upwelling positive, downwelling negative
    c = rrtmgp_column_index(i, j, grid.Nx)

    @inbounds begin
        ℐ_lw_up[i, j, k] = lw_flux_up[k, c]
        ℐ_lw_dn[i, j, k] = -lw_flux_dn[k, c]  # Negate for downward
        ℐ_sw_dn[i, j, k] = -sw_flux_dn_dir[k, c]  # Negate for downward
    end
end

# Default no-op for models without radiation
update_radiation!(::Nothing, model) = nothing
