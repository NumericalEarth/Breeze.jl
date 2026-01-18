#####
##### All-sky (gas + cloud optics) RadiativeTransferModel: full-spectrum RRTMGP radiative transfer model
#####

using Oceananigans.Utils: launch!
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.Grids: xnode, ynode, λnode, φnode, znodes
using Oceananigans.Grids: AbstractGrid, Center, Face
using Oceananigans.Fields: ConstantField

using Breeze.AtmosphereModels:
    AtmosphereModels,
    SurfaceRadiativeProperties,
    specific_humidity,
    BackgroundAtmosphere,
    AllSkyOptics,
    ConstantRadiusParticles,
    cloud_liquid_effective_radius,
    cloud_ice_effective_radius,
    compute_moisture_fractions,
    RadiativeTransferModel

using Breeze.Thermodynamics: ThermodynamicConstants

using Dates: AbstractDateTime, Millisecond
using KernelAbstractions: @kernel, @index

using RRTMGP: AllSkyRadiation, RRTMGPSolver, lookup_tables, update_lw_fluxes!, update_sw_fluxes!
using RRTMGP.AtmosphericStates: AtmosphericState, CloudState, MaxRandomOverlap
using RRTMGP.BCs: LwBCs, SwBCs
using RRTMGP.Fluxes: set_flux_to_zero!
using RRTMGP.Vmrs: init_vmr

# Dispatch on AtmosphericState having CloudState (not Nothing) for all-sky radiation
# AtmosphericState{FTA1D, FTA1DN, FTA2D, D, VMR, CLD, AER} where CLD is the 6th type parameter
const AllSkyAtmosphericState = AtmosphericState{<:Any, <:Any, <:Any, <:Any, <:Any, <:CloudState}
const AllSkyRadiativeTransferModel = RadiativeTransferModel{<:Any, <:Any, <:Any, <:Any, <:BackgroundAtmosphere, <:AllSkyAtmosphericState}

#####
##### Constructor
#####

"""
$(TYPEDSIGNATURES)

Construct an all-sky (gas + cloud) full-spectrum `RadiativeTransferModel` for the given grid.

This constructor requires that `NCDatasets` is loadable in the user environment because
RRTMGP loads lookup tables from netCDF via an extension.

# Keyword Arguments
- `background_atmosphere`: Background atmospheric gas composition (default: `BackgroundAtmosphere(FT)`).
  Each gas can be a constant (wrapped in `ConstantField`) or a `Field` for spatially varying profiles.
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
- `liquid_effective_radius`: Model for cloud liquid effective radius in μm (default: `ConstantRadiusParticles(10.0)`)
- `ice_effective_radius`: Model for cloud ice effective radius in μm (default: `ConstantRadiusParticles(30.0)`)
- `ice_roughness`: Ice crystal roughness for cloud optics (1=smooth, 2=medium, 3=rough; default: 2)
"""
function AtmosphereModels.RadiativeTransferModel(grid::AbstractGrid,
                                                 ::AllSkyOptics,
                                                 constants::ThermodynamicConstants;
                                                 background_atmosphere = BackgroundAtmosphere(eltype(grid)),
                                                 surface_temperature,
                                                 coordinate = nothing,
                                                 epoch = nothing,
                                                 surface_emissivity = 0.98,
                                                 direct_surface_albedo = nothing,
                                                 diffuse_surface_albedo = nothing,
                                                 surface_albedo = nothing,
                                                 solar_constant = 1361,
                                                 schedule = IterationInterval(1),
                                                 liquid_effective_radius = ConstantRadiusParticles(10.0),
                                                 ice_effective_radius = ConstantRadiusParticles(30.0),
                                                 ice_roughness = 2)

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

    # RRTMGP grid + context
    context = rrtmgp_context(arch)
    DA = ClimaComms.array_type(context.device)
    grid_params = RRTMGPGridParams(FT; context, nlay=Nz, ncol=Nc)

    # Lookup tables (requires NCDatasets extension for RRTMGP)
    # AllSkyRadiation(aerosol_radiation, reset_rng_seed)
    radiation_method = AllSkyRadiation(false, false)

    luts = try
        lookup_tables(grid_params, radiation_method)
    catch err
        if err isa MethodError
            msg = "Full-spectrum RRTMGP all-sky radiation requires NCDatasets to be loaded so that\n" *
                  "RRTMGP can read netCDF lookup tables.\n\n" *
                  "Try:\n\n    using NCDatasets\n\n" *
                  "and then construct RadiativeTransferModel again."
            throw(ArgumentError(msg))
        else
            rethrow()
        end
    end

    nbnd_lw = luts.lu_kwargs.nbnd_lw
    nbnd_sw = luts.lu_kwargs.nbnd_sw
    ngas = luts.lu_kwargs.ngas_sw

    # Atmospheric state arrays
    rrtmgp_λ = DA{FT}(undef, Nc)
    rrtmgp_φ = DA{FT}(undef, Nc)
    rrtmgp_layerdata = DA{FT}(undef, 4, Nz, Nc)
    rrtmgp_pᶠ = DA{FT}(undef, Nz+1, Nc)
    rrtmgp_Tᶠ = DA{FT}(undef, Nz+1, Nc)
    rrtmgp_T₀ = DA{FT}(undef, Nc)

    set_longitude!(rrtmgp_λ, coordinate, grid)
    set_latitude!(rrtmgp_φ, coordinate, grid)

    vmr = init_vmr(ngas, Nz, Nc, FT, DA; gm=true)
    set_global_mean_gases!(vmr, luts.lookups.idx_gases_sw, background_atmosphere)

    # Cloud state arrays
    cloud_liquid_radius = DA{FT}(undef, Nz, Nc)
    cloud_ice_radius = DA{FT}(undef, Nz, Nc)
    cloud_liquid_water_path = DA{FT}(undef, Nz, Nc)
    cloud_ice_water_path = DA{FT}(undef, Nz, Nc)
    cloud_fraction = DA{FT}(undef, Nz, Nc)
    cloud_mask_longwave = DA{Bool}(undef, Nz, Nc)
    cloud_mask_shortwave = DA{Bool}(undef, Nz, Nc)

    # Initialize cloud arrays to zero
    fill!(cloud_liquid_radius, zero(FT))
    fill!(cloud_ice_radius, zero(FT))
    fill!(cloud_liquid_water_path, zero(FT))
    fill!(cloud_ice_water_path, zero(FT))
    fill!(cloud_fraction, zero(FT))
    fill!(cloud_mask_longwave, false)
    fill!(cloud_mask_shortwave, false)

    cloud_state = CloudState(cloud_liquid_radius,
                             cloud_ice_radius,
                             cloud_liquid_water_path,
                             cloud_ice_water_path,
                             cloud_fraction,
                             cloud_mask_longwave,
                             cloud_mask_shortwave,
                             MaxRandomOverlap(),
                             ice_roughness)

    atmospheric_state = AtmosphericState(rrtmgp_λ, rrtmgp_φ, rrtmgp_layerdata, rrtmgp_pᶠ, rrtmgp_Tᶠ, rrtmgp_T₀, vmr, cloud_state, nothing)

    # Boundary conditions (bandwise emissivity/albedo; incident fluxes are unused here)
    cos_zenith = DA{FT}(undef, Nc)
    rrtmgp_ℐ₀ = DA{FT}(undef, Nc)
    rrtmgp_ℐ₀ .= convert(FT, solar_constant)

    rrtmgp_ε₀ = DA{FT}(undef, nbnd_lw, Nc)
    rrtmgp_αb₀ = DA{FT}(undef, nbnd_sw, Nc)
    rrtmgp_αw₀ = DA{FT}(undef, nbnd_sw, Nc)

    if surface_emissivity isa Number
        surface_emissivity = ConstantField(convert(FT, surface_emissivity))
        rrtmgp_ε₀ .= surface_emissivity.constant
    end

    if direct_surface_albedo isa Number
        direct_surface_albedo = ConstantField(convert(FT, direct_surface_albedo))
        rrtmgp_αb₀ .= direct_surface_albedo.constant
    end

    if diffuse_surface_albedo isa Number
        diffuse_surface_albedo = ConstantField(convert(FT, diffuse_surface_albedo))
        rrtmgp_αw₀ .= diffuse_surface_albedo.constant
    end

    if surface_temperature isa Number
        surface_temperature = ConstantField(convert(FT, surface_temperature))
        rrtmgp_T₀ .= surface_temperature.constant
    end

    lw_bcs = LwBCs(rrtmgp_ε₀, nothing)
    sw_bcs = SwBCs(cos_zenith, rrtmgp_ℐ₀, rrtmgp_αb₀, nothing, rrtmgp_αw₀)

    solver = RRTMGPSolver(grid_params, radiation_method, parameters, lw_bcs, sw_bcs, atmospheric_state)

    # Oceananigans output fields
    upwelling_longwave_flux = ZFaceField(grid)
    downwelling_longwave_flux = ZFaceField(grid)
    downwelling_shortwave_flux = ZFaceField(grid)

    surface_properties = SurfaceRadiativeProperties(surface_temperature,
                                                    surface_emissivity,
                                                    direct_surface_albedo,
                                                    diffuse_surface_albedo)

    # Convert effective radius models to proper float type if they are ConstantRadiusParticles
    liquid_eff_radius = liquid_effective_radius isa ConstantRadiusParticles ?
                        ConstantRadiusParticles(convert(FT, liquid_effective_radius.radius)) :
                        liquid_effective_radius

    ice_eff_radius = ice_effective_radius isa ConstantRadiusParticles ?
                     ConstantRadiusParticles(convert(FT, ice_effective_radius.radius)) :
                     ice_effective_radius

    return RadiativeTransferModel(convert(FT, solar_constant),
                                  coordinate,
                                  epoch,
                                  surface_properties,
                                  background_atmosphere,
                                  atmospheric_state,
                                  solver,
                                  nothing,
                                  upwelling_longwave_flux,
                                  downwelling_longwave_flux,
                                  downwelling_shortwave_flux,
                                  liquid_eff_radius,
                                  ice_eff_radius,
                                  schedule)
end

#####
##### Update radiation (gas + cloud state)
#####

"""
$(TYPEDSIGNATURES)

Update the all-sky (gas + cloud) full-spectrum radiative fluxes from the current model state.
"""
function AtmosphereModels._update_radiation!(rtm::AllSkyRadiativeTransferModel, model)
    grid = model.grid
    clock = model.clock
    solver = rtm.longwave_solver

    # Update gas state (shared with clear-sky)
    update_rrtmgp_gas_state!(solver.as, model, rtm.surface_properties.surface_temperature,
                             rtm.background_atmosphere, solver.params)

    # Update cloud state
    update_rrtmgp_cloud_state!(solver.as.cloud_state, model,
                               rtm.liquid_effective_radius,
                               rtm.ice_effective_radius)

    # Update solar zenith angle
    datetime = compute_datetime(clock.time, rtm.epoch)
    update_solar_zenith_angle!(solver.sws, rtm.coordinate, grid, datetime)

    # Longwave
    update_lw_fluxes!(solver)

    # Shortwave: we always call the solver; when `cos_zenith ≤ 0` the imposed
    # boundary condition should yield (near-)zero fluxes.
    set_flux_to_zero!(solver.sws.flux)
    update_sw_fluxes!(solver)

    copy_rrtmgp_fluxes_to_fields!(rtm, solver, grid)

    return nothing
end

#####
##### Update cloud state
#####

function update_rrtmgp_cloud_state!(cloud_state, model, liquid_effective_radius, ice_effective_radius)
    grid = model.grid
    arch = architecture(grid)

    ρᵣ = model.dynamics.reference_state.density
    microphysics = model.microphysics
    microphysical_fields = model.microphysical_fields
    qᵗ = model.specific_moisture

    launch!(arch, grid, :xyz, _update_rrtmgp_cloud_state!,
            cloud_state, grid, ρᵣ, microphysics, microphysical_fields, qᵗ,
            liquid_effective_radius, ice_effective_radius)

    return nothing
end

@kernel function _update_rrtmgp_cloud_state!(cloud_state, grid, ρᵣ, microphysics, microphysical_fields, qᵗ,
                                             liquid_effective_radius, ice_effective_radius)
    i, j, k = @index(Global, NTuple)

    col = rrtmgp_column_index(i, j, grid.Nx)

    FT = eltype(ρᵣ)
    kg_to_g = convert(FT, 1000)

    @inbounds begin
        ρ = ρᵣ[i, j, k]
        Δz = Δzᶜᶜᶜ(i, j, k, grid)
        qᵗ_ijk = qᵗ[i, j, k]

        # Get moisture fractions from microphysics
        q = compute_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ_ijk, microphysical_fields)

        # Extract liquid and ice mass fractions
        qˡ = q.liquid
        qⁱ = q.ice

        # Cloud water path in g/m² (RRTMGP convention)
        # Note: cld_path_liq/ice, cld_frac, cld_r_eff_liq/ice are RRTMGP's CloudState field names
        cloud_liquid_water_path = kg_to_g * ρ * qˡ * Δz
        cloud_ice_water_path = kg_to_g * ρ * qⁱ * Δz
        cloud_state.cld_path_liq[k, col] = cloud_liquid_water_path
        cloud_state.cld_path_ice[k, col] = cloud_ice_water_path

        # Binary cloud fraction (1 if any condensate, 0 otherwise)
        has_cloud = (qˡ + qⁱ) > zero(FT)
        cloud_state.cld_frac[k, col] = ifelse(has_cloud, one(FT), zero(FT))

        # Effective radii in microns
        rˡ = cloud_liquid_effective_radius(i, j, k, grid, liquid_effective_radius)
        rⁱ = cloud_ice_effective_radius(i, j, k, grid, ice_effective_radius)
        cloud_state.cld_r_eff_liq[k, col] = rˡ
        cloud_state.cld_r_eff_ice[k, col] = rⁱ
    end
end
