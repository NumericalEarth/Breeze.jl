#####
##### Clear-sky (gas optics) RadiativeTransferModel: full-spectrum RRTMGP radiative transfer model
#####

using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.Grids: xnode, ynode, λnode, φnode, znodes
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: ConstantField

using Breeze.AtmosphereModels: AtmosphereModels, SurfaceRadiativeProperties, specific_humidity, BackgroundAtmosphere
using Breeze.Thermodynamics: ThermodynamicConstants
import Breeze.AtmosphereModels: RadiativeTransferModel

using Dates: AbstractDateTime, Millisecond
using KernelAbstractions: @kernel, @index

using RRTMGP: ClearSkyRadiation, RRTMGPSolver, lookup_tables, update_lw_fluxes!, update_sw_fluxes!
using RRTMGP.AtmosphericStates: AtmosphericState
using RRTMGP.BCs: LwBCs, SwBCs
using RRTMGP.Fluxes: set_flux_to_zero!
using RRTMGP.Vmrs: init_vmr

# Dispatch on background_atmosphere = BackgroundAtmosphere for clear-sky radiation
const ClearSkyRadiativeTransferModel = RadiativeTransferModel{<:Any, <:Any, <:Any, <:Any, <:BackgroundAtmosphere}

"""
$(TYPEDSIGNATURES)

Construct a clear-sky (gas-only) full-spectrum `RadiativeTransferModel` for the given grid.

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
"""
function RadiativeTransferModel(grid,
                                ::Val{:clear_sky},
                                constants::ThermodynamicConstants;
                                background_atmosphere = BackgroundAtmosphere{eltype(grid)}(),
                                surface_temperature,
                                coordinate = nothing,
                                epoch = nothing,
                                surface_emissivity = 0.98,
                                direct_surface_albedo = nothing,
                                diffuse_surface_albedo = nothing,
                                surface_albedo = nothing,
                                solar_constant = 1361)

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
    radiation_method = ClearSkyRadiation(false)

    luts = try
        lookup_tables(grid_params, radiation_method)
    catch err
        if err isa MethodError
            msg = "Full-spectrum RRTMGP clear-sky radiation requires NCDatasets to be loaded so that\n" *
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

    atmospheric_state = AtmosphericState(rrtmgp_λ, rrtmgp_φ, rrtmgp_layerdata, rrtmgp_pᶠ, rrtmgp_Tᶠ, rrtmgp_T₀, vmr, nothing, nothing)

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
                                  downwelling_shortwave_flux)
end

# Mapping from RRTMGP's internal gas names to BackgroundAtmosphere field names
const RRTMGP_GAS_NAME_MAP = Dict{String, Symbol}(
    "n2"      => :N₂,
    "o2"      => :O₂,
    "co2"     => :CO₂,
    "ch4"     => :CH₄,
    "n2o"     => :N₂O,
    "co"      => :CO,
    "no2"     => :NO₂,
    "o3"      => :O₃,
    "cfc11"   => :CFC₁₁,
    "cfc12"   => :CFC₁₂,
    "cfc22"   => :CFC₂₂,
    "ccl4"    => :CCl₄,
    "cf4"     => :CF₄,
    "hfc125"  => :HFC₁₂₅,
    "hfc134a" => :HFC₁₃₄ₐ,
    "hfc143a" => :HFC₁₄₃ₐ,
    "hfc23"   => :HFC₂₃,
    "hfc32"   => :HFC₃₂,
)

@inline function set_global_mean_gases!(vmr, idx_gases_sw, atm::BackgroundAtmosphere)
    FT = eltype(vmr.vmr)
    ngas = length(vmr.vmr)
    host = zeros(FT, ngas)

    for (name, ig) in idx_gases_sw
        sym = get(RRTMGP_GAS_NAME_MAP, name, nothing)
        if !isnothing(sym) && hasproperty(atm, sym)
            host[ig] = getproperty(atm, sym)
        end
    end

    vmr.vmr .= host
    return nothing
end

@inline function set_longitude!(rrtmgp_λ, coordinate::Tuple, grid)
    λ = coordinate[1]
    rrtmgp_λ .= λ
    return nothing
end

function set_longitude!(rrtmgp_λ, ::Nothing, grid)
    arch = grid.architecture
    launch!(arch, grid, :xy, _set_longitude_from_grid!, rrtmgp_λ, grid)
    return nothing
end

@kernel function _set_longitude_from_grid!(rrtmgp_λ, grid)
    i, j = @index(Global, NTuple)
    λ = xnode(i, j, 1, grid, Center(), Center(), Center())
    col = rrtmgp_column_index(i, j, grid.Nx)
    @inbounds rrtmgp_λ[col] = λ
end

"""
$(TYPEDSIGNATURES)

Update the clear-sky full-spectrum radiative fluxes from the current model state.
"""
function AtmosphereModels.update_radiation!(rtm::ClearSkyRadiativeTransferModel, model)
    grid = model.grid
    clock = model.clock
    solver = rtm.longwave_solver

    # Update atmospheric state
    update_rrtmgp_clear_sky_state!(solver.as, model, rtm.surface_properties.surface_temperature, rtm.background_atmosphere, solver.params)

    # Update solar zenith angle
    datetime = compute_datetime(clock.time, rtm.epoch)
    update_solar_zenith_angle!(solver.sws, rtm.coordinate, grid, datetime)

    # Longwave
    update_lw_fluxes!(solver)

    # Shortwave: we always call the solver; when `cos_zenith ≤ 0` the imposed
    # boundary condition should yield (near-)zero fluxes.
    set_flux_to_zero!(solver.sws.flux)
    update_sw_fluxes!(solver)

    copy_clear_sky_fluxes_to_fields!(rtm, solver, grid)
    return nothing
end

function update_rrtmgp_clear_sky_state!(as::AtmosphericState, model, surface_temperature, background_atmosphere::BackgroundAtmosphere, params)
    grid = model.grid
    arch = architecture(grid)

    pᵣ = model.formulation.reference_state.pressure
    T = model.temperature
    qᵛ = specific_humidity(model)

    g = params.grav
    mᵈ = params.molmass_dryair
    mᵛ = params.molmass_water
    ℕᴬ = params.avogad
    O₃ = background_atmosphere.O₃

    launch!(arch, grid, :xyz, _update_rrtmgp_clear_sky_state!, as, grid, pᵣ, T, qᵛ, surface_temperature, g, mᵈ, mᵛ, ℕᴬ, O₃)
    return nothing
end

@kernel function _update_rrtmgp_clear_sky_state!(as, grid, pᵣ, T, qᵛ, surface_temperature, g, mᵈ, mᵛ, ℕᴬ, O₃)
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
        # Clamp temperatures to avoid extrapolation that can yield tiny negative source values
        # and trigger DomainErrors in geometric means.
        # TODO: This clamping should ideally be done internally in RRTMGP.jl.
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
        m⁻²_to_cm⁻² = convert(FT, 1e4)
        col_dry = dry_mass_per_area / mᵈ * ℕᴬ / m⁻²_to_cm⁻² # (molecules / m²) -> (molecules / cm²)

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

function copy_clear_sky_fluxes_to_fields!(rtm::ClearSkyRadiativeTransferModel, solver, grid)
    arch = architecture(grid)
    Nz = size(grid, 3)

    lw_flux_up = solver.lws.flux.flux_up
    lw_flux_dn = solver.lws.flux.flux_dn
    sw_flux_dn_dir = solver.sws.flux.flux_dn_dir

    ℐ_lw_up = rtm.upwelling_longwave_flux
    ℐ_lw_dn = rtm.downwelling_longwave_flux
    ℐ_sw_dn = rtm.downwelling_shortwave_flux

    Nx, Ny, Nz = size(grid)
    launch!(arch, grid, (Nx, Ny, Nz+1), _copy_clear_sky_fluxes!,
            ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn, lw_flux_up, lw_flux_dn, sw_flux_dn_dir, grid)

    return nothing
end

@kernel function _copy_clear_sky_fluxes!(ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn,
                                        lw_flux_up, lw_flux_dn, sw_flux_dn_dir, grid)
    i, j, k = @index(Global, NTuple)

    col = rrtmgp_column_index(i, j, grid.Nx)

    @inbounds begin
        ℐ_lw_up[i, j, k] = lw_flux_up[k, col]
        ℐ_lw_dn[i, j, k] = -lw_flux_dn[k, col]
        ℐ_sw_dn[i, j, k] = -sw_flux_dn_dir[k, col]
    end
end
