#####
##### All-sky (gas + cloud optics) RadiativeTransferModel: full-spectrum RRTMGP radiative transfer model
#####

using Oceananigans.Utils: launch!, prettysummary
using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶜ
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
    compute_moisture_fractions

using Breeze.Thermodynamics: ThermodynamicConstants
import Breeze.AtmosphereModels: RadiativeTransferModel

using Dates: AbstractDateTime, Millisecond
using KernelAbstractions: @kernel, @index

using RRTMGP: AllSkyRadiation, RRTMGPSolver, lookup_tables, update_lw_fluxes!, update_sw_fluxes!
using RRTMGP.AtmosphericStates: AtmosphericState, CloudState, MaxRandomOverlap
using RRTMGP.BCs: LwBCs, SwBCs
using RRTMGP.Fluxes: set_flux_to_zero!
using RRTMGP.Vmrs: init_vmr

#####
##### All-sky RadiativeTransferModel wrapper (stores cloud effective radius models)
#####

"""
    AllSkyRadiativeTransferModelWrapper

Wrapper that holds the standard `RadiativeTransferModel` plus additional fields
for all-sky cloud optics (effective radius models).
"""
struct AllSkyRadiativeTransferModelWrapper{RTM, LER, IER}
    radiative_transfer_model :: RTM
    liquid_effective_radius :: LER
    ice_effective_radius :: IER
end

# Forward property access to the underlying RadiativeTransferModel
function Base.getproperty(wrapper::AllSkyRadiativeTransferModelWrapper, s::Symbol)
    if s in (:radiative_transfer_model, :liquid_effective_radius, :ice_effective_radius)
        return getfield(wrapper, s)
    else
        return getproperty(getfield(wrapper, :radiative_transfer_model), s)
    end
end

Base.propertynames(wrapper::AllSkyRadiativeTransferModelWrapper) =
    (propertynames(getfield(wrapper, :radiative_transfer_model))..., :liquid_effective_radius, :ice_effective_radius)

Base.summary(::AllSkyRadiativeTransferModelWrapper) = "AllSkyRadiativeTransferModel"

function Base.show(io::IO, wrapper::AllSkyRadiativeTransferModelWrapper)
    rtm = wrapper.radiative_transfer_model
    print(io, summary(wrapper), "\n",
          "├── solar_constant: ", prettysummary(rtm.solar_constant), " W m⁻²\n",
          "├── surface_temperature: ", rtm.surface_properties.surface_temperature, " K\n",
          "├── surface_emissivity: ", rtm.surface_properties.surface_emissivity, "\n",
          "├── direct_surface_albedo: ", rtm.surface_properties.direct_surface_albedo, "\n",
          "├── diffuse_surface_albedo: ", rtm.surface_properties.diffuse_surface_albedo, "\n",
          "├── liquid_effective_radius: ", wrapper.liquid_effective_radius, "\n",
          "└── ice_effective_radius: ", wrapper.ice_effective_radius)
end

#####
##### Constructor
#####

"""
$(TYPEDSIGNATURES)

Construct an all-sky (gas + cloud) full-spectrum `RadiativeTransferModel` for the given grid.

This constructor requires that `NCDatasets` is loadable in the user environment because
RRTMGP loads lookup tables from netCDF via an extension.

# Keyword Arguments
- `background_atmosphere`: Background atmospheric gas composition (default: `BackgroundAtmosphere{FT}()`).
- `surface_temperature`: Surface temperature in Kelvin (required).
- `coordinate`: Tuple of (longitude, latitude) in degrees. If `nothing` (default), 
                extracted from grid coordinates.
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
function RadiativeTransferModel(grid::AbstractGrid,
                                ::AllSkyOptics,
                                constants::ThermodynamicConstants;
                                background_atmosphere = BackgroundAtmosphere{eltype(grid)}(),
                                surface_temperature,
                                coordinate = nothing,
                                epoch = nothing,
                                surface_emissivity = 0.98,
                                direct_surface_albedo = nothing,
                                diffuse_surface_albedo = nothing,
                                surface_albedo = nothing,
                                solar_constant = 1361,
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
    cld_r_eff_liq = DA{FT}(undef, Nz, Nc)
    cld_r_eff_ice = DA{FT}(undef, Nz, Nc)
    cld_path_liq = DA{FT}(undef, Nz, Nc)
    cld_path_ice = DA{FT}(undef, Nz, Nc)
    cld_frac = DA{FT}(undef, Nz, Nc)
    cld_mask_lw = DA{Bool}(undef, Nz, Nc)
    cld_mask_sw = DA{Bool}(undef, Nz, Nc)

    # Initialize cloud arrays to zero
    fill!(cld_r_eff_liq, zero(FT))
    fill!(cld_r_eff_ice, zero(FT))
    fill!(cld_path_liq, zero(FT))
    fill!(cld_path_ice, zero(FT))
    fill!(cld_frac, zero(FT))
    fill!(cld_mask_lw, false)
    fill!(cld_mask_sw, false)

    cloud_state = CloudState(cld_r_eff_liq,
                             cld_r_eff_ice,
                             cld_path_liq,
                             cld_path_ice,
                             cld_frac,
                             cld_mask_lw,
                             cld_mask_sw,
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

    rtm = RadiativeTransferModel(convert(FT, solar_constant),
                                 coordinate,
                                 epoch,
                                 surface_properties,
                                 background_atmosphere,
                                 atmospheric_state,
                                 solver,
                                 nothing,
                                 upwelling_longwave_flux,
                                 downwelling_longwave_flux,
                                 downwelling_shortwave_flux)

    # Convert effective radius models to proper float type if they are ConstantRadiusParticles
    liquid_eff_radius = liquid_effective_radius isa ConstantRadiusParticles ?
                        ConstantRadiusParticles(convert(FT, liquid_effective_radius.radius)) :
                        liquid_effective_radius

    ice_eff_radius = ice_effective_radius isa ConstantRadiusParticles ?
                     ConstantRadiusParticles(convert(FT, ice_effective_radius.radius)) :
                     ice_effective_radius

    return AllSkyRadiativeTransferModelWrapper(rtm, liquid_eff_radius, ice_eff_radius)
end

#####
##### Update radiation (gas + cloud state)
#####

"""
$(TYPEDSIGNATURES)

Update the all-sky (gas + cloud) full-spectrum radiative fluxes from the current model state.
"""
function AtmosphereModels.update_radiation!(wrapper::AllSkyRadiativeTransferModelWrapper, model)
    rtm = wrapper.radiative_transfer_model
    grid = model.grid
    clock = model.clock
    solver = rtm.longwave_solver

    # Update gas state (same as clear-sky)
    update_rrtmgp_all_sky_gas_state!(solver.as, model, rtm.surface_properties.surface_temperature,
                                     rtm.background_atmosphere, solver.params)

    # Update cloud state
    update_rrtmgp_cloud_state!(solver.as.cloud_state, model,
                               wrapper.liquid_effective_radius,
                               wrapper.ice_effective_radius)

    # Update solar zenith angle
    datetime = compute_datetime(clock.time, rtm.epoch)
    update_solar_zenith_angle!(solver.sws, rtm.coordinate, grid, datetime)

    # Longwave
    update_lw_fluxes!(solver)

    # Shortwave: we always call the solver; when `cos_zenith ≤ 0` the imposed
    # boundary condition should yield (near-)zero fluxes.
    set_flux_to_zero!(solver.sws.flux)
    update_sw_fluxes!(solver)

    copy_all_sky_fluxes_to_fields!(rtm, solver, grid)
    return nothing
end

#####
##### Update gas state (reuses clear-sky pattern)
#####

function update_rrtmgp_all_sky_gas_state!(as::AtmosphericState, model, surface_temperature,
                                          background_atmosphere::BackgroundAtmosphere, params)
    grid = model.grid
    arch = architecture(grid)

    pᵣ = model.dynamics.reference_state.pressure
    T = model.temperature
    qᵛ = specific_humidity(model)

    g = params.grav
    mᵈ = params.molmass_dryair
    mᵛ = params.molmass_water
    ℕᴬ = params.avogad
    O₃ = background_atmosphere.O₃

    launch!(arch, grid, :xyz, _update_rrtmgp_all_sky_gas_state!, as, grid, pᵣ, T, qᵛ, surface_temperature, g, mᵈ, mᵛ, ℕᴬ, O₃)
    return nothing
end

@kernel function _update_rrtmgp_all_sky_gas_state!(as, grid, pᵣ, T, qᵛ, surface_temperature, g, mᵈ, mᵛ, ℕᴬ, O₃)
    i, j, k = @index(Global, NTuple)

    Nz = size(grid, 3)
    col = rrtmgp_column_index(i, j, grid.Nx)

    layerdata = as.layerdata
    pᶠ = as.p_lev
    Tᶠ = as.t_lev
    T₀ = as.t_sfc

    vmr_h2o = as.vmr.vmr_h2o
    vmr_o3 = as.vmr.vmr_o3

    @inbounds begin
        # Layer (cell-centered) values
        pᶜ = pᵣ[i, j, k]
        Tᶜ = T[i, j, k]
        qᵛₖ = max(qᵛ[i, j, k], zero(eltype(qᵛ)))

        # Face values at k and k+1 (needed for column dry air mass)
        pᶠₖ = ℑzᵃᵃᶠ(i, j, k, grid, pᵣ)
        Tᶠₖ = ℑzᵃᵃᶠ(i, j, k, grid, T)
        pᶠₖ₊₁ = ℑzᵃᵃᶠ(i, j, k+1, grid, pᵣ)

        # RRTMGP Planck/source lookup tables are defined over a finite temperature range.
        Tmin = 160
        Tmax = 355
        Tᶜ = clamp(Tᶜ, Tmin, Tmax)
        Tᶠₖ = clamp(Tᶠₖ, Tmin, Tmax)

        # Store level values
        pᶠ[k, col] = pᶠₖ
        Tᶠ[k, col] = Tᶠₖ

        # Topmost level (once)
        if k == 1
            pᶠ[Nz+1, col] = ℑzᵃᵃᶠ(i, j, Nz+1, grid, pᵣ)
            Tᴺ⁺¹ = ℑzᵃᵃᶠ(i, j, Nz+1, grid, T)
            Tᶠ[Nz+1, col] = clamp(Tᴺ⁺¹, Tmin, Tmax)
            T₀[col] = clamp(surface_temperature[i, j, 1], Tmin, Tmax)
        end

        # Column dry air mass: molecules / cm² of dry air
        Δp = max(pᶠₖ - pᶠₖ₊₁, zero(pᶠₖ))
        dry_mass_fraction = 1 - qᵛₖ
        dry_mass_per_area = (Δp / g) * dry_mass_fraction
        m⁻²_to_cm⁻² = convert(eltype(pᶜ), 1e4)
        col_dry = dry_mass_per_area / mᵈ * ℕᴬ / m⁻²_to_cm⁻²

        # Populate layerdata: (col_dry, pᶜ, Tᶜ, relative_humidity)
        layerdata[1, k, col] = col_dry
        layerdata[2, k, col] = pᶜ
        layerdata[3, k, col] = Tᶜ
        layerdata[4, k, col] = zero(eltype(Tᶜ))

        # H₂O volume mixing ratio from specific humidity
        r = qᵛₖ / dry_mass_fraction
        vmr_h2o[k, col] = r * (mᵈ / mᵛ)
        vmr_o3[k, col] = O₃
    end
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
        cloud_state.cld_path_liq[k, col] = kg_to_g * ρ * qˡ * Δz
        cloud_state.cld_path_ice[k, col] = kg_to_g * ρ * qⁱ * Δz

        # Binary cloud fraction (1 if any condensate, 0 otherwise)
        has_cloud = (qˡ + qⁱ) > zero(FT)
        cloud_state.cld_frac[k, col] = ifelse(has_cloud, one(FT), zero(FT))

        # Effective radii in microns
        cloud_state.cld_r_eff_liq[k, col] = cloud_liquid_effective_radius(i, j, k, grid, liquid_effective_radius)
        cloud_state.cld_r_eff_ice[k, col] = cloud_ice_effective_radius(i, j, k, grid, ice_effective_radius)
    end
end

#####
##### Copy fluxes to Oceananigans fields
#####

function copy_all_sky_fluxes_to_fields!(rtm, solver, grid)
    arch = architecture(grid)
    Nz = size(grid, 3)

    lw_flux_up = solver.lws.flux.flux_up
    lw_flux_dn = solver.lws.flux.flux_dn
    sw_flux_dn = solver.sws.flux.flux_dn  # Total SW (direct + diffuse)

    ℐ_lw_up = rtm.upwelling_longwave_flux
    ℐ_lw_dn = rtm.downwelling_longwave_flux
    ℐ_sw_dn = rtm.downwelling_shortwave_flux

    Nx, Ny, Nz = size(grid)
    launch!(arch, grid, (Nx, Ny, Nz+1), _copy_all_sky_fluxes!,
            ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn, lw_flux_up, lw_flux_dn, sw_flux_dn, grid)

    return nothing
end

@kernel function _copy_all_sky_fluxes!(ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn,
                                       lw_flux_up, lw_flux_dn, sw_flux_dn, grid)
    i, j, k = @index(Global, NTuple)

    col = rrtmgp_column_index(i, j, grid.Nx)

    @inbounds begin
        ℐ_lw_up[i, j, k] = lw_flux_up[k, col]
        ℐ_lw_dn[i, j, k] = -lw_flux_dn[k, col]
        ℐ_sw_dn[i, j, k] = -sw_flux_dn[k, col]
    end
end

