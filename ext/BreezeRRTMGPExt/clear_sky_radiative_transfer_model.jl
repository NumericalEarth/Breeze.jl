#####
##### Clear-sky (gas optics) RadiativeTransferModel: full-spectrum RRTMGP radiative transfer model
#####

using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.Grids: xnode, ynode, λnode, φnode, znodes
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: ConstantField

using Breeze.AtmosphereModels: AtmosphereModels, SurfaceRadiativeProperties
import Breeze.AtmosphereModels: RadiativeTransferModel, RRTMGPGasOptics

using Dates: AbstractDateTime, Millisecond
using KernelAbstractions: @kernel, @index

using RRTMGP: ClearSkyRadiation, RRTMGPSolver, lookup_tables, update_lw_fluxes!, update_sw_fluxes!
using RRTMGP.AtmosphericStates: AtmosphericState
using RRTMGP.BCs: LwBCs, SwBCs
using RRTMGP.Fluxes: set_flux_to_zero!
using RRTMGP.Vmrs: init_vmr

"""
$(TYPEDSIGNATURES)

Construct a clear-sky (gas-only) full-spectrum `RadiativeTransferModel` for the given grid.

This constructor requires that `NCDatasets` is loadable in the user environment because
RRTMGP loads lookup tables from netCDF via an extension.
"""
function RadiativeTransferModel(grid, constants,
                                optics::RRTMGPGasOptics;
                                surface_temperature,
                                coordinate = nothing,
                                epoch = nothing,
                                stefan_boltzmann_constant = 5.670374419e-8,
                                avogadro_number = 6.02214076e23,
                                surface_emissivity = 0.98,
                                direct_surface_albedo = nothing,
                                diffuse_surface_albedo = nothing,
                                surface_albedo = nothing,
                                solar_constant = 1361)

    FT = eltype(grid)

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

    # RRTMGP parameters
    kappa_d = constants.dry_air.heat_capacity / constants.dry_air.molar_mass
    radiative_transfer_parameters = RRTMGPParameters(;
        grav = FT(constants.gravitational_acceleration),
        molmass_dryair = FT(constants.dry_air.molar_mass),
        molmass_water = FT(constants.vapor.molar_mass),
        gas_constant = FT(constants.molar_gas_constant),
        kappa_d = FT(kappa_d),
        Stefan = FT(stefan_boltzmann_constant),
        avogad = FT(avogadro_number),
    )

    # Atmospheric state arrays
    lon = DA{FT}(undef, Nc)
    lat = DA{FT}(undef, Nc)
    layerdata = DA{FT}(undef, 4, Nz, Nc)
    p_lev = DA{FT}(undef, Nz+1, Nc)
    t_lev = DA{FT}(undef, Nz+1, Nc)
    t_sfc = DA{FT}(undef, Nc)

    set_longitude!(lon, coordinate, grid)
    set_latitude!(lat, coordinate, grid)

    vmr = init_vmr(ngas, Nz, Nc, FT, DA; gm=true)
    set_global_mean_gases!(vmr, luts.lookups.idx_gases_sw, optics)

    atmospheric_state = AtmosphericState(lon, lat, layerdata, p_lev, t_lev, t_sfc, vmr, nothing, nothing)

    # Boundary conditions (bandwise emissivity/albedo; incident fluxes are unused here)
    cos_zenith = DA{FT}(undef, Nc)
    toa_flux = DA{FT}(undef, Nc)
    toa_flux .= convert(FT, solar_constant)

    sfc_emis = DA{FT}(undef, nbnd_lw, Nc)
    sfc_alb_direct = DA{FT}(undef, nbnd_sw, Nc)
    sfc_alb_diffuse = DA{FT}(undef, nbnd_sw, Nc)

    if surface_emissivity isa Number
        surface_emissivity = ConstantField(convert(FT, surface_emissivity))
        sfc_emis .= surface_emissivity.constant
    end

    if direct_surface_albedo isa Number
        direct_surface_albedo = ConstantField(convert(FT, direct_surface_albedo))
        sfc_alb_direct .= direct_surface_albedo.constant
    end

    if diffuse_surface_albedo isa Number
        diffuse_surface_albedo = ConstantField(convert(FT, diffuse_surface_albedo))
        sfc_alb_diffuse .= diffuse_surface_albedo.constant
    end

    if surface_temperature isa Number
        surface_temperature = ConstantField(convert(FT, surface_temperature))
        t_sfc .= surface_temperature.constant
    end

    lw_bcs = LwBCs(sfc_emis, nothing)
    sw_bcs = SwBCs(cos_zenith, toa_flux, sfc_alb_direct, nothing, sfc_alb_diffuse)

    solver = RRTMGPSolver(grid_params, radiation_method, radiative_transfer_parameters, lw_bcs, sw_bcs, atmospheric_state)

    # Oceananigans output fields
    upwelling_longwave_flux = ZFaceField(grid)
    downwelling_longwave_flux = ZFaceField(grid)
    downwelling_shortwave_flux = ZFaceField(grid)

    surface_properties = SurfaceRadiativeProperties(surface_temperature,
                                                    surface_emissivity,
                                                    direct_surface_albedo,
                                                    diffuse_surface_albedo)

    return RadiativeTransferModel(optics,
                                  convert(FT, solar_constant),
                                  coordinate,
                                  epoch,
                                  surface_properties,
                                  atmospheric_state,
                                  solver,
                                  nothing,
                                  upwelling_longwave_flux,
                                  downwelling_longwave_flux,
                                  downwelling_shortwave_flux)
end

@inline function set_global_mean_gases!(vmr, idx_gases_sw, optics::RRTMGPGasOptics)
    FT = eltype(vmr.vmr)
    ngas = length(vmr.vmr)
    host = zeros(FT, ngas)

    # Fill from the optics NamedTuple when available; otherwise default to zero.
    for (name, ig) in idx_gases_sw
        sym = Symbol(name)
        if hasproperty(optics.gas_vmr, sym)
            host[ig] = getproperty(optics.gas_vmr, sym)
        end
    end

    vmr.vmr .= host
    return nothing
end

@inline function set_longitude!(rrtmgp_longitude, coordinate::Tuple, grid)
    λ = coordinate[1]
    rrtmgp_longitude .= λ
    return nothing
end

function set_longitude!(rrtmgp_longitude, ::Nothing, grid)
    # Mirror latitude behavior; for now only single-column grids are supported.
    arch = grid.architecture
    launch!(arch, grid, :xy, _set_longitude_from_grid!, rrtmgp_longitude, grid)
    return nothing
end

@kernel function _set_longitude_from_grid!(rrtmgp_longitude, grid)
    i, j = @index(Global, NTuple)
    λ = xnode(i, j, 1, grid, Center(), Center(), Center())
    col = rrtmgp_column_index(i, j, grid.Nx)
    @inbounds rrtmgp_longitude[col] = λ
end

"""
$(TYPEDSIGNATURES)

Update the clear-sky full-spectrum radiative fluxes from the current model state.
"""
function AtmosphereModels.update_radiation!(rtm::RadiativeTransferModel{<:RRTMGPGasOptics}, model)
    grid = model.grid
    clock = model.clock
    solver = rtm.longwave_solver

    # Update atmospheric state
    update_rrtmgp_clear_sky_state!(solver.as, model, rtm.surface_properties.surface_temperature, rtm.optics, solver.params)

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

function update_rrtmgp_clear_sky_state!(as::AtmosphericState, model, surface_temperature, optics::RRTMGPGasOptics, params)
    grid = model.grid
    arch = architecture(grid)

    p = model.formulation.reference_state.pressure
    T = model.temperature
    μ = model.microphysical_fields
    q = (μ === nothing || !hasproperty(μ, :qᵛ)) ? model.specific_moisture : getproperty(μ, :qᵛ)

    g = params.grav
    M_dry = params.molmass_dryair
    M_w = params.molmass_water
    N_A = params.avogad
    o3 = optics.gas_vmr.o3

    launch!(arch, grid, :xyz, _update_rrtmgp_clear_sky_state!, as, grid, p, T, q, surface_temperature, g, M_dry, M_w, N_A, o3)
    return nothing
end

@kernel function _update_rrtmgp_clear_sky_state!(as, grid, p, T, q, surface_temperature, g, M_dry, M_w, N_A, o3)
    i, j, k = @index(Global, NTuple)

    Nz = size(grid, 3)
    col = rrtmgp_column_index(i, j, grid.Nx)

    layerdata = as.layerdata
    p_lev = as.p_lev
    t_lev = as.t_lev
    t_sfc = as.t_sfc

    vmr_h2o = as.vmr.vmr_h2o
    vmr_o3 = as.vmr.vmr_o3

    @inbounds begin
        # Layer values
        p_lay = p[i, j, k]
        t_lay = T[i, j, k]
        q_lay = max(q[i, j, k], zero(eltype(q)))

        # Face values at k and k+1 (needed for col_dry)
        p_face_k = ℑzᵃᵃᶠ(i, j, k, grid, p)
        t_face_k = ℑzᵃᵃᶠ(i, j, k, grid, T)
        p_face_kp1 = ℑzᵃᵃᶠ(i, j, k+1, grid, p)

        # RRTMGP Planck/source lookup tables are defined over a finite temperature range.
        # Clamp temperatures to avoid extrapolation that can yield tiny negative source values
        # and trigger DomainErrors in geometric means.
        # TODO: This clamping should ideally be done internally in RRTMGP.jl.
        # See https://github.com/CliMA/RRTMGP.jl/issues/XXX
        t_min = one(t_lay) * 160
        t_max = one(t_lay) * 355
        t_lay = clamp(t_lay, t_min, t_max)
        t_face_k = clamp(t_face_k, t_min, t_max)

        # Store level values
        p_lev[k, col] = p_face_k
        t_lev[k, col] = t_face_k

        # Topmost level (once)
        if k == 1
            p_lev[Nz+1, col] = ℑzᵃᵃᶠ(i, j, Nz+1, grid, p)
            t_top = ℑzᵃᵃᶠ(i, j, Nz+1, grid, T)
            t_lev[Nz+1, col] = clamp(t_top, t_min, t_max)
            t_sfc[col] = clamp(surface_temperature[i, j, 1], t_min, t_max)
        end

        # col_dry: molecules / cm^2 of dry air
        Δp = max(p_face_k - p_face_kp1, zero(p_face_k))
        one_minus_q = one(q_lay) - q_lay
        dry_mass_per_area = (Δp / g) * one_minus_q
        col_dry = dry_mass_per_area / M_dry * N_A / (one(dry_mass_per_area) * 1e4)  # (molecules / m^2) -> (molecules / cm^2)

        # Populate layerdata: (col_dry, p_lay, t_lay, rel_hum)
        layerdata[1, k, col] = col_dry
        layerdata[2, k, col] = p_lay
        layerdata[3, k, col] = t_lay
        layerdata[4, k, col] = zero(eltype(t_lay))

        # H2O vmr from specific humidity
        denom = one(q_lay) - q_lay
        r = q_lay / denom
        vmr_h2o[k, col] = r * (M_dry / M_w)
        vmr_o3[k, col] = o3
    end
end

function copy_clear_sky_fluxes_to_fields!(rtm::RadiativeTransferModel{<:RRTMGPGasOptics}, solver, grid)
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


