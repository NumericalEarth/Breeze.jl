#####
##### GrayRadiativeTransferModel: Gray atmosphere radiative transfer model
#####
##### Uses the O'Gorman and Schneider (2008) optical thickness parameterization
##### with RRTMGP's two-stream or no-scattering solvers.
#####

using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.Grids: ynode, znodes
using Oceananigans.Grids: AbstractGrid, Center, Face, Flat
using Breeze.AtmosphereModels: AtmosphereModels, RadiativeTransferModel,
                               AbstractSolarPosition, ApparentSolarPosition,
                               DiurnalSolarPosition, FixedCosineZenith

using RRTMGP.AtmosphericStates: GrayAtmosphericState, GrayOpticalThicknessOGorman2008
using KernelAbstractions: @kernel, @index

# Dispatch on background_atmosphere = Nothing for gray radiation
const GrayRadiativeTransferModel = RadiativeTransferModel{<:Any, <:Any, <:Any, Nothing}

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a gray atmosphere radiative transfer model for the given grid.

# Keyword Arguments
- `optical_thickness`: Optical thickness parameterization (default: `GrayOpticalThicknessOGorman2008(FT)`).
- `surface_temperature`: Surface temperature in Kelvin, a `Number` or 2D `Field`. Default: `nothing` —
  bind one before the first radiation update (a coupled model wires its interface surface
  temperature into the radiation automatically).
- `solar_position`: Specification of the solar zenith angle. See [`AbstractSolarPosition`](@ref) and its subtypes:
  - [`ApparentSolarPosition`](@ref) (default) — time-varying, computed from the model clock and grid (or explicit) longitude/latitude.
  - [`FixedCosineZenith`](@ref) — constant cos(θ_z), independent of the clock.
- `surface_emissivity`: Surface emissivity, 0-1 (default: 0.98). Can be scalar or 2D field.
- `surface_albedo`: Surface albedo, 0-1. Can be scalar or 2D field.
                    Alternatively, provide both `direct_surface_albedo` and `diffuse_surface_albedo`.
- `direct_surface_albedo`: Direct surface albedo, 0-1. Can be scalar or 2D field.
- `diffuse_surface_albedo`: Diffuse surface albedo, 0-1. Can be scalar or 2D field.
- `solar_constant`: Top-of-atmosphere solar flux in W/m² (default: 1361), a scalar or an `(Nx, Ny)` array of per-column values
"""
function AtmosphereModels.RadiativeTransferModel(grid::AbstractGrid,
                                                 ::GrayOptics,
                                                 constants::ThermodynamicConstants;
                                                 optical_thickness = GrayOpticalThicknessOGorman2008(eltype(grid)),
                                                 surface_temperature = nothing,
                                                 solar_position::AbstractSolarPosition = ApparentSolarPosition(),
                                                 surface_emissivity = 0.98,
                                                 direct_surface_albedo = nothing,
                                                 diffuse_surface_albedo = nothing,
                                                 surface_albedo = nothing,
                                                 solar_constant = 1361,
                                                 schedule = IterationInterval(1))

    FT = eltype(grid)
    parameters = RRTMGPParameters(constants)

    solar_position = maybe_infer_solar_position(solar_position, grid)

    surface_radiation = materialize_surface_radiation(grid, solar_position;
                                                      surface_temperature, surface_emissivity, surface_albedo,
                                                      direct_surface_albedo, diffuse_surface_albedo)

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

    set_latitude!(rrtmgp_φ, solar_position, grid)

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

    # Boundary conditions: RRTMGP expects (nbnd, ncol) for surface properties.
    # Gray optics has nbnd = 1.
    cos_zenith = ArrayType{FT}(undef, Nc)
    initialize_cos_zenith!(cos_zenith, solar_position)
    rrtmgp_ℐ₀ = ArrayType{FT}(undef, Nc)
    rrtmgp_ε₀ = ArrayType{FT}(undef, 1, Nc)
    rrtmgp_αb₀ = ArrayType{FT}(undef, 1, Nc)
    rrtmgp_αw₀ = ArrayType{FT}(undef, 1, Nc)

    fill_columns!(rrtmgp_ℐ₀, solar_constant)  # Top-of-atmosphere solar flux, a scalar or per column

    initialize_surface_temperature!(rrtmgp_T₀, surface_radiation.surface_temperature)

    rrtmgp_grid = RRTMGPGridParams(FT; context, domain_nlay=Nz, ncol=Nc)

    longwave_solver = NoScatLWRTE(rrtmgp_grid;
                                  params = parameters,
                                  sfc_emis = rrtmgp_ε₀,
                                  inc_flux = nothing)

    shortwave_solver = NoScatSWRTE(rrtmgp_grid;
                                   cos_zenith = cos_zenith,
                                   toa_flux = rrtmgp_ℐ₀,
                                   sfc_alb_direct = rrtmgp_αb₀,
                                   sfc_alb_diffuse = rrtmgp_αw₀,
                                   inc_flux_diffuse = nothing)

    # The non-scattering shortwave solver only ever writes the direct beam
    # (`flux_dn_dir`), leaving `flux_up` and `flux_dn` as allocated-but-unwritten
    # memory. Zero them once so that every shortwave array in the solver agrees
    # with the physics: no diffuse shortwave, and no upwelling shortwave within the solver.
    # The optical depth is written only for sunlit columns; zeroing it keeps the
    # (vanishing) reflected beam of a column that has never been sunlit finite.
    shortwave_flux = shortwave_solver.flux
    for array in (shortwave_flux.flux_up, shortwave_flux.flux_dn,
                  shortwave_flux.flux_net, shortwave_flux.flux_dn_dir, shortwave_solver.op.τ)
        fill!(array, 0)
    end

    # Create Oceananigans fields to store fluxes for output/plotting
    upwelling_longwave_flux = ZFaceField(grid)
    downwelling_longwave_flux = ZFaceField(grid)
    upwelling_shortwave_flux = ZFaceField(grid)    # Direct beam reflected by the surface
    downwelling_shortwave_flux = ZFaceField(grid)  # Direct beam only
    flux_divergence = CenterField(grid)

    update_rrtmgp_surface_boundary_conditions!(longwave_solver.bcs.sfc_emis,
                                               shortwave_solver.bcs.sfc_alb_direct,
                                               shortwave_solver.bcs.sfc_alb_diffuse,
                                               surface_radiation, grid)

    return RadiativeTransferModel(materialize_solar_constant(solar_constant, FT),
                                  solar_position,
                                  surface_radiation,
                                  nothing,  # background_atmosphere = nothing for gray
                                  atmospheric_state,
                                  longwave_solver,
                                  shortwave_solver,
                                  upwelling_longwave_flux,
                                  downwelling_longwave_flux,
                                  upwelling_shortwave_flux,
                                  downwelling_shortwave_flux,
                                  flux_divergence,
                                  nothing,  # liquid_effective_radius = nothing for gray
                                  nothing,  # ice_effective_radius = nothing for gray
                                  nothing,  # column_batches = nothing: gray is never batched
                                  schedule,
                                  approximate_radiation_updates(grid, schedule, solar_position),
                                  radiation_cache_validity(grid))
end

#####
##### Latitude initialization for the gray-optics latitude-dependent τ
#####

# Apparent sun: use the explicit coordinate if given, otherwise read φ from the grid
set_latitude!(rrtmgp_latitude, sp::ApparentSolarPosition{<:Tuple}, grid) =
    _set_latitude_from_coordinate!(rrtmgp_latitude, sp.coordinate, grid)

set_latitude!(rrtmgp_latitude, sp::ApparentSolarPosition{Nothing}, grid) =
    _set_latitude_from_grid!(rrtmgp_latitude, grid)

# Diurnal cycle: latitude is part of the solar-position spec; broadcast it
# directly to every column so gray τ uses the same φ as cos(θ_z).
function set_latitude!(rrtmgp_latitude, sp::DiurnalSolarPosition, grid)
    rrtmgp_latitude .= convert(eltype(rrtmgp_latitude), sp.latitude)
    return nothing
end

# Fixed zenith: gray τ_e/τ_p still depend on latitude — fall back to grid coords.
set_latitude!(rrtmgp_latitude, ::FixedCosineZenith, grid) =
    _set_latitude_from_grid!(rrtmgp_latitude, grid)

function _set_latitude_from_coordinate!(rrtmgp_latitude, coordinate::Tuple, grid)
    φ = coordinate[2]
    rrtmgp_latitude .= φ
    return nothing
end

function _set_latitude_from_grid!(rrtmgp_latitude, grid)
    arch = grid.architecture
    launch!(arch, grid, :xy, _set_latitude_from_grid_kernel!, rrtmgp_latitude, grid)
    return nothing
end

# A grid that is Flat in y has no latitude (see `_set_longitude_from_grid!`)
_set_latitude_from_grid!(rrtmgp_latitude, ::AbstractGrid{<:Any, <:Any, <:Flat}) = (fill!(rrtmgp_latitude, 0); nothing)

@kernel function _set_latitude_from_grid_kernel!(rrtmgp_latitude, grid)
    i, j = @index(Global, NTuple)
    φ = ynode(i, j, 1, grid, Center(), Center(), Center())
    c = column_index(i, j, grid.Nx)
    @inbounds rrtmgp_latitude[c] = φ
end

#####
##### Solar constant materialization
#####

# The solar constant as stored: a grid-eltype scalar, or the user's (Nx, Ny) array as is
materialize_solar_constant(solar_constant::Number, FT) = convert(FT, solar_constant)
materialize_solar_constant(solar_constant::AbstractArray, FT) = solar_constant

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
#

AtmosphereModels.solver_cos_zenith(rtm::GrayRadiativeTransferModel) = rtm.shortwave_solver.bcs.cos_zenith

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Solve the gray radiative fluxes for the current model state, at the solar zenith angle that
`_update_radiation!` has set:

1. Updates the RRTMGP atmospheric state from model fields (T, p)
2. Solves the longwave and shortwave RTE
3. Copies the fluxes to Oceananigans fields for output
4. Solves the longwave RTE again at the perturbed surface temperature, when the fluxes are updated
   between solves for a changing surface temperature

Sign convention: positive flux = upward, negative flux = downward.
"""
function AtmosphereModels.solve_radiative_transfer!(rtm::GrayRadiativeTransferModel, model)
    grid = model.grid

    rrtmgp_state = rtm.atmospheric_state
    surface_temperature = rtm.surface_radiation.surface_temperature

    # Update RRTMGP atmospheric state from model fields
    update_rrtmgp_state!(rrtmgp_state, model, surface_temperature)

    # Gray optics keeps the two RTE solvers side by side rather than inside one `RRTMGPSolver`.
    update_rrtmgp_surface_boundary_conditions!(rtm.longwave_solver.bcs.sfc_emis,
                                               rtm.shortwave_solver.bcs.sfc_alb_direct,
                                               rtm.shortwave_solver.bcs.sfc_alb_diffuse,
                                               rtm.surface_radiation, grid)

    # Solve longwave RTE (RRTMGP external call)
    solve_lw!(rtm.longwave_solver, rrtmgp_state)

    # Solve shortwave RTE
    # Note: RRTMGP handles the case when sun is below horizon (cos_zenith ≤ 0)
    # by producing zero fluxes internally
    solve_sw!(rtm.shortwave_solver, rrtmgp_state)

    # Copy RRTMGP flux arrays to Oceananigans fields with sign convention
    copy_fluxes_to_fields!(rtm, grid)

    solves_longwave_response(rtm) && solve_perturbed_surface_longwave!(rtm, grid)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Solve the longwave fluxes again at the perturbed surface temperature, into the longwave responses
of `rtm.approximate_updates`. Gray optics is one batch of all the columns.
"""
function solve_perturbed_surface_longwave!(rtm::GrayRadiativeTransferModel, grid)
    rrtmgp_state = rtm.atmospheric_state
    updates = rtm.approximate_updates
    Ny = size(grid, 2)

    set_batch_surface_temperature!(rrtmgp_state.t_sfc, rtm, grid, Ny, 0, perturbed_surface_temperature)
    solve_lw!(rtm.longwave_solver, rrtmgp_state)
    set_batch_surface_temperature!(rrtmgp_state.t_sfc, rtm, grid, Ny, 0, solved_surface_temperature)

    copy_longwave_fluxes!(updates.upwelling_longwave_response, updates.downwelling_longwave_response, rtm, grid)

    return nothing
end

#####
##### Update RRTMGP atmospheric state from model fields
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Update the RRTMGP `GrayAtmosphericState` arrays from model fields.

# Grid staggering: layers vs levels

RRTMGP requires atmospheric state at both "layers" (cell centers) and "levels" (cell faces).
This matches the finite-volume staggering used in Oceananigans:

```
                        ┌─────────────────────────────────────────────────┐
    z_lev[Nz+1] ━━━━━━━ │  level Nz+1 (TOA):  p_lev, t_lev, z_lev         │ ← from halo
                        └─────────────────────────────────────────────────┘
                        ┌─────────────────────────────────────────────────┐
                        │  layer Nz:  T[Nz], p_lay[Nz] = p[Nz]            │ ← from model
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
                        │  layer 1:   T[1], p_lay[1] = p[1]               │ ← from model
                        └─────────────────────────────────────────────────┘
    z_lev[1]    ━━━━━━━   level 1 (surface, bottom face):  p_lev, t_lev   │ ← from halo
                        ══════════════════════════════════════════════════
                                        GROUND (t_sfc)
```

# Why the model must provide level values

RRTMGP is a general-purpose radiative transfer solver that operates on columns of
atmospheric data. It does not interpolate from layers to levels internally because:

1. **Boundary conditions**: The surface (level 1) and TOA (level Nz+1) require boundary values
   that only the atmospheric model knows. Breeze applies the same interior average there, so each
   boundary level takes whatever that field's halo carries. A `Value` bottom boundary condition
   (carried by the anelastic and 1D-column Exner reference pressures) reproduces the prescribed
   surface pressure exactly; a default `NoFlux` halo mirrors the interior (`p[0] = p[1]`), which
   collapses the level value onto the adjacent layer value.

   That prescribed value is `surface_pressure`, the pressure at the column's bottom face, and not
   the `base_pressure` datum at z = 0. The two coincide only on a domain whose bottom is the
   ground; on a raised or terrain-following domain they differ by the hydrostatic reduction from
   the datum to that face, which is the level 1 pressure radiation must see.

   TODO: extrapolate to the boundary faces explicitly rather than inheriting the halo. Wherever
   the halo mirrors, `p_lev` equals `p_lay` at that end, so the layer spans only half a cell in
   pressure while `_compute_radiation_flux_divergence!` still divides its flux difference by the
   full `Δzᶜᶜᶜ`, roughly halving that cell's heating rate. This affects the top cell always, and
   the bottom cell whenever `dynamics_pressure` carries no `Value` bottom boundary condition
   (compressible dynamics, and every terrain reference state).

2. **Interpolation to levels**: Interior levels use `ℑzᵃᵃᶠ`, the arithmetic average of the two
   adjacent cell centers, for both pressure and temperature. That is second-order on a uniform
   column and first-order wherever `Δz` varies (stretched or terrain-following grids), since the
   face is then not the midpoint of its neighbouring centers.

3. **Model consistency**: The pressure profile must be consistent with the atmospheric model's
   thermodynamic state. RRTMGP has no knowledge of the anelastic approximation or of which
   dynamical pressure contributions affect thermodynamics.

# Physics notes

**Temperature**: We use the actual temperature field `T` from the model state.
This is the temperature that matters for thermal emission and absorption.

**Pressure**: We use `dynamics_pressure(model.dynamics)` at cell centers — the anelastic
hydrostatic reference pressure (in that approximation pressure perturbations are negligible), or
the compressible diagnosed pressure. Never the dynamics' pressure-gradient reference state.

# RRTMGP array layout
- Layer arrays `(Nz, Nc)`: values at cell centers, layer 1 at bottom
- Level arrays `(Nz+1, Nc)`: values at cell faces, level 1 at the surface (the bottom face)
"""
function update_rrtmgp_state!(rrtmgp_state::GrayAtmosphericState, model, surface_temperature)
    grid = model.grid
    arch = architecture(grid)

    p = dynamics_pressure(model.dynamics)
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

    c = column_index(i, j, grid.Nx)

    @inbounds begin
        # Face values at k and k+1
        pᶠ[k, c] = ℑzᵃᵃᶠ(i, j, k, grid, p)
        Tᶠ[k, c] = ℑzᵃᵃᶠ(i, j, k, grid, T)

        # Layer values: use face-averaged temperature for consistency with level sources,
        # preventing 2Δz oscillations in the radiative heating rate.
        pᶜ[k, c] = p[i, j, k]
        Tᶜ[k, c] = (ℑzᵃᵃᶠ(i, j, k, grid, T) + ℑzᵃᵃᶠ(i, j, k+1, grid, T)) / 2

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
##### Copy RRTMGP fluxes to Oceananigans fields
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Copy RRTMGP flux arrays to Oceananigans ZFaceFields.

Applies sign convention:
* positive = upward
* negative = downward.

The non-scattering shortwave solver computes the direct beam only. The surface reflects the
fraction `α` (the direct surface albedo) of it diffusely, and the reflected beam climbs the column
attenuated by `exp(-D τ)` across each layer of shortwave optical depth `τ`, with `D` the
`reflected_beam_diffusivity`.
"""
function copy_fluxes_to_fields!(rtm::GrayRadiativeTransferModel, grid)
    copy_longwave_fluxes!(rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux, rtm, grid)

    shortwave_solver = rtm.shortwave_solver
    launch!(architecture(grid), grid, :xy, _copy_gray_shortwave_fluxes!,
            rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux,
            shortwave_solver.flux.flux_dn_dir,
            shortwave_solver.bcs.sfc_alb_direct,
            shortwave_solver.op.τ,
            convert(eltype(grid), reflected_beam_diffusivity), grid)

    return nothing
end

# The longwave fluxes of the last longwave solve, into `ℐ_lw_up` and `ℐ_lw_dn`
function copy_longwave_fluxes!(ℐ_lw_up, ℐ_lw_dn, rtm::GrayRadiativeTransferModel, grid)
    flux = rtm.longwave_solver.flux
    Nx, Ny, Nz = size(grid)
    launch!(architecture(grid), grid, (Nx, Ny, Nz+1), _copy_gray_longwave_fluxes!,
            ℐ_lw_up, ℐ_lw_dn, flux.flux_up, flux.flux_dn, grid)
    return nothing
end

"""
The diffusivity `D ≈ 1.64` with which the direct beam reflected by the surface climbs a gray
atmosphere: the secant of the single angle of the Gauss–Jacobi-5 quadrature with which RRTMGP's gray
longwave solver integrates its diffuse fluxes by default. Fixed here, so that it remains a
diffusivity when the longwave solver is given more angles, whose secants are not.
"""
const reflected_beam_diffusivity = 1 / 0.6096748751

# RRTMGP compute buffers are indexed (Nc, Nz+1), we use (i, j, k) for ZFaceField
# Sign convention: upwelling positive, downwelling negative
@kernel function _copy_gray_longwave_fluxes!(ℐ_lw_up, ℐ_lw_dn, lw_flux_up, lw_flux_dn, grid)
    i, j, k = @index(Global, NTuple)
    c = column_index(i, j, grid.Nx)

    @inbounds begin
        ℐ_lw_up[i, j, k] = lw_flux_up[c, k]
        ℐ_lw_dn[i, j, k] = -lw_flux_dn[c, k]  # Negate for downward
    end
end

# The direct beam, and its reflection by the surface climbing the column
@kernel function _copy_gray_shortwave_fluxes!(ℐ_sw_up, ℐ_sw_dn, sw_flux_dn_dir, α, τ, D, grid)
    i, j = @index(Global, NTuple)
    c = column_index(i, j, grid.Nx)
    Nz = size(grid, 3)

    @inbounds begin
        for k in 1:Nz+1
            ℐ_sw_dn[i, j, k] = -sw_flux_dn_dir[c, k]  # Negate for downward
        end

        ℐ = - α[1, c] * ℐ_sw_dn[i, j, 1]
        ℐ_sw_up[i, j, 1] = ℐ

        for k in 1:Nz
            ℐ = ℐ * exp(- D * τ[c, k])
            ℐ_sw_up[i, j, k+1] = ℐ
        end
    end
end

# Default no-op for models without radiation
update_radiation!(::Nothing, model) = nothing
