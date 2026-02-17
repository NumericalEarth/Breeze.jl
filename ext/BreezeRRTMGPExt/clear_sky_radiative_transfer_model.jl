#####
##### Clear-sky (gas optics) RadiativeTransferModel: full-spectrum RRTMGP radiative transfer model
#####

using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.Grids: xnode, ynode, λnode, φnode, znodes
using Oceananigans.Grids: AbstractGrid, Center, Face
using Oceananigans.Fields: ConstantField

using Breeze.AtmosphereModels: AtmosphereModels, SurfaceRadiativeProperties, specific_humidity,
                               BackgroundAtmosphere, materialize_background_atmosphere,
                               ClearSkyOptics, RadiativeTransferModel
using Breeze.Thermodynamics: ThermodynamicConstants

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
- `background_atmosphere`: Background atmospheric gas composition (default: `BackgroundAtmosphere()`).
  O₃ can be a Number or Function of `z`; other gases are global mean constants.
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
                                                 ::ClearSkyOptics,
                                                 constants::ThermodynamicConstants;
                                                 background_atmosphere = BackgroundAtmosphere(),
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

    # Materialize background atmosphere (converts O₃ functions to fields)
    background_atmosphere = materialize_background_atmosphere(background_atmosphere, grid)

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
    ArrayType = ClimaComms.array_type(context.device)
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

    Nband_lw = luts.lu_kwargs.nbnd_lw
    Nband_sw = luts.lu_kwargs.nbnd_sw
    Ngas = luts.lu_kwargs.ngas_sw

    # Atmospheric state arrays
    rrtmgp_λ = ArrayType{FT}(undef, Nc)
    rrtmgp_φ = ArrayType{FT}(undef, Nc)
    rrtmgp_layerdata = ArrayType{FT}(undef, 4, Nz, Nc)
    rrtmgp_pᶠ = ArrayType{FT}(undef, Nz+1, Nc)
    rrtmgp_Tᶠ = ArrayType{FT}(undef, Nz+1, Nc)
    rrtmgp_T₀ = ArrayType{FT}(undef, Nc)

    set_longitude!(rrtmgp_λ, coordinate, grid)
    set_latitude!(rrtmgp_φ, coordinate, grid)

    vmr = init_vmr(Ngas, Nz, Nc, FT, ArrayType; gm=true)
    set_global_mean_gases!(vmr, luts.lookups.idx_gases_sw, background_atmosphere)

    atmospheric_state = AtmosphericState(rrtmgp_λ, rrtmgp_φ, rrtmgp_layerdata, rrtmgp_pᶠ, rrtmgp_Tᶠ, rrtmgp_T₀, vmr, nothing, nothing)

    # Boundary conditions (bandwise emissivity/albedo; incident fluxes are unused here)
    cos_zenith = ArrayType{FT}(undef, Nc)
    rrtmgp_ℐ₀ = ArrayType{FT}(undef, Nc)
    rrtmgp_ℐ₀ .= convert(FT, solar_constant)

    rrtmgp_ε₀ = ArrayType{FT}(undef, Nband_lw, Nc)
    rrtmgp_αb₀ = ArrayType{FT}(undef, Nband_sw, Nc)
    rrtmgp_αw₀ = ArrayType{FT}(undef, Nband_sw, Nc)

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
    flux_divergence = CenterField(grid)

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
                                  downwelling_shortwave_flux,
                                  flux_divergence,
                                  nothing,  # liquid_effective_radius = nothing for clear-sky
                                  nothing,  # ice_effective_radius = nothing for clear-sky
                                  schedule)
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

@inline function set_global_mean_gases!(vmr, gas_indices, atm::BackgroundAtmosphere)
    FT = eltype(vmr.vmr)
    Ngas = length(vmr.vmr)
    host = zeros(FT, Ngas)

    # All gases except O₃ are stored as numbers in BackgroundAtmosphere
    # O₃ is handled per-layer in the kernel via vmr_o3
    for (name, ig) in gas_indices
        name == "o3" && continue  # O₃ handled per-layer in kernel
        sym = get(RRTMGP_GAS_NAME_MAP, name, nothing)
        if !isnothing(sym) && hasproperty(atm, sym)
            host[ig] = getproperty(atm, sym)
        end
    end

    # Use copyto! for proper CPU→GPU transfer
    copyto!(vmr.vmr, host)
    return nothing
end

@inline function set_longitude!(rrtmgp_λ, coordinate::Tuple, grid)
    λ = coordinate[1]
    rrtmgp_λ .= λ
    return nothing
end

# When coordinate is a Number (fixed cos zenith), we don't need real lon/lat
# Fill with zeros since RRTMGP still needs valid arrays
@inline function set_longitude!(rrtmgp_λ, ::Number, grid)
    rrtmgp_λ .= 0
    return nothing
end

@inline function set_latitude!(rrtmgp_φ, ::Number, grid)
    rrtmgp_φ .= 0
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
    c = rrtmgp_column_index(i, j, grid.Nx)
    @inbounds rrtmgp_λ[c] = λ
end

"""
$(TYPEDSIGNATURES)

Update the clear-sky full-spectrum radiative fluxes from the current model state.
"""
function AtmosphereModels._update_radiation!(rtm::ClearSkyRadiativeTransferModel, model)
    grid = model.grid
    clock = model.clock
    solver = rtm.longwave_solver

    # Update atmospheric state
    update_rrtmgp_gas_state!(solver.as, model, rtm.surface_properties.surface_temperature, rtm.background_atmosphere, solver.params)

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

    # Compute radiation flux divergence
    compute_radiation_flux_divergence!(rtm, grid)

    return nothing
end
