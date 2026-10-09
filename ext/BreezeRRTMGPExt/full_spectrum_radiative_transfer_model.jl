#####
##### Full-spectrum RadiativeTransferModel: the RRTMGP model shared by clear-sky and all-sky optics
#####
#
# Clear-sky and all-sky radiation differ only in the RRTMGP radiation method and in whether the
# atmospheric state carries a `CloudState`. Everything else (surface properties, gas state, solver,
# output fields and the radiation update) is built here once.

using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.Grids: xnode, ynode, λnode, φnode, znodes
using Oceananigans.Grids: AbstractGrid, Center, Face
using Oceananigans.Fields: ConstantField

using Breeze.AtmosphereModels: AtmosphereModels, SurfaceRadiation, specific_humidity,
                               BackgroundAtmosphere, materialize_background_atmosphere,
                               RadiativeTransferModel,
                               AbstractSolarPosition, ApparentSolarPosition,
                               DiurnalSolarPosition, FixedCosineZenith
using Breeze.Thermodynamics: ThermodynamicConstants

using Dates: AbstractDateTime, Millisecond
using KernelAbstractions: @kernel, @index

using RRTMGP: RRTMGPSolver, lookup_tables
using RRTMGP.AtmosphericStates: AtmosphericState
using RRTMGP.BCs: LwBCs, SwBCs

# Dispatch on background_atmosphere = BackgroundAtmosphere for full-spectrum (clear-sky or all-sky) radiation
const FullSpectrumRadiativeTransferModel = RadiativeTransferModel{<:Any, <:Any, <:Any, <:BackgroundAtmosphere}

# Shared by the clear-sky and all-sky `RadiativeTransferModel` docstrings.
const full_spectrum_keywords_docstring = """
- `background_atmosphere`: Background atmospheric gas composition (default: `BackgroundAtmosphere()`).
  O₃ can be a Number, Function, or Field; other gases are global mean constants.
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
- `solar_constant`: Top-of-atmosphere solar flux in W/m² (default: 1361)
$(column_batches_docstring)"""

"""
$(TYPEDSIGNATURES)

Build a full-spectrum `RadiativeTransferModel` that solves with `radiation_method`. Clear-sky
passes `nothing` for `cloud_state` and the effective radii; all-sky passes a `CloudState` and the
effective radius models. The keyword arguments are those documented for the clear-sky and all-sky
`RadiativeTransferModel` constructors.
"""
function full_spectrum_radiative_transfer_model(grid, radiation_method, constants,
                                                cloud_state, liquid_effective_radius, ice_effective_radius;
                                                background_atmosphere = BackgroundAtmosphere(),
                                                surface_temperature = nothing,
                                                solar_position::AbstractSolarPosition = ApparentSolarPosition(),
                                                surface_emissivity = 0.98,
                                                direct_surface_albedo = nothing,
                                                diffuse_surface_albedo = nothing,
                                                surface_albedo = nothing,
                                                solar_constant = 1361,
                                                schedule = IterationInterval(1),
                                                column_batches = nothing)

    FT = eltype(grid)
    parameters = RRTMGPParameters(constants)

    solar_position = maybe_infer_solar_position(solar_position, grid)

    surface_radiation = materialize_surface_radiation(grid, solar_position;
                                                      surface_temperature, surface_emissivity, surface_albedo,
                                                      direct_surface_albedo, diffuse_surface_albedo)

    # Materialize background atmosphere (converts O₃ functions to fields)
    background_atmosphere = materialize_background_atmosphere(background_atmosphere, grid)

    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    Nc = Nx * Ny

    # RRTMGP grid + context. `grid_params` sizes the solver workspace (one batch), not the state.
    context = rrtmgp_context(arch)
    ArrayType = ClimaComms.array_type(context.device)
    grid_params = rrtmgp_grid_params(FT, context, grid, column_batches)

    # Lookup tables (requires NCDatasets extension for RRTMGP)
    luts = try
        lookup_tables(grid_params, radiation_method)
    catch err
        if err isa MethodError
            msg = "Full-spectrum RRTMGP radiation requires NCDatasets to be loaded so that\n" *
                  "RRTMGP can read netCDF lookup tables.\n\n" *
                  "Try:\n\n    using NCDatasets\n\n" *
                  "and then construct RadiativeTransferModel again."
            throw(ArgumentError(msg))
        else
            rethrow()
        end
    end

    Nband_lw = luts.nbnd_lw
    Nband_sw = luts.nbnd_sw
    Ngas = luts.ngas_sw

    # Atmospheric state arrays, sized for the whole domain: each batch solves a `view` of them.
    rrtmgp_λ = ArrayType{FT}(undef, Nc)
    rrtmgp_φ = ArrayType{FT}(undef, Nc)
    rrtmgp_layerdata = ArrayType{FT}(undef, 4, Nz, Nc)
    rrtmgp_pᶠ = ArrayType{FT}(undef, Nz+1, Nc)
    rrtmgp_Tᶠ = ArrayType{FT}(undef, Nz+1, Nc)
    rrtmgp_T₀ = ArrayType{FT}(undef, Nc)

    set_longitude!(rrtmgp_λ, solar_position, grid)
    set_latitude!(rrtmgp_φ, solar_position, grid)
    initialize_surface_temperature!(rrtmgp_T₀, surface_radiation.surface_temperature)

    vmr = initialize_global_mean_vmr(Ngas, Nz, Nc, FT, ArrayType)
    set_global_mean_gases!(vmr, luts.idx_gases_sw, background_atmosphere)

    atmospheric_state = AtmosphericState(rrtmgp_λ, rrtmgp_φ, rrtmgp_layerdata, rrtmgp_pᶠ, rrtmgp_Tᶠ, rrtmgp_T₀, vmr, cloud_state, nothing)

    # Boundary conditions (bandwise emissivity/albedo; incident fluxes are unused here)
    cos_zenith = ArrayType{FT}(undef, Nc)
    initialize_cos_zenith!(cos_zenith, solar_position)
    rrtmgp_ℐ₀ = ArrayType{FT}(undef, Nc)
    rrtmgp_ℐ₀ .= convert(FT, solar_constant)

    rrtmgp_ε₀ = ArrayType{FT}(undef, Nband_lw, Nc)
    rrtmgp_αb₀ = ArrayType{FT}(undef, Nband_sw, Nc)
    rrtmgp_αw₀ = ArrayType{FT}(undef, Nband_sw, Nc)

    lw_bcs = LwBCs(rrtmgp_ε₀, nothing)
    sw_bcs = SwBCs(cos_zenith, rrtmgp_ℐ₀, rrtmgp_αb₀, nothing, rrtmgp_αw₀)

    solver = RRTMGPSolver(grid_params, radiation_method, parameters, lw_bcs, sw_bcs, atmospheric_state)

    # Oceananigans output fields
    upwelling_longwave_flux = ZFaceField(grid)
    downwelling_longwave_flux = ZFaceField(grid)
    upwelling_shortwave_flux = ZFaceField(grid)
    downwelling_shortwave_flux = ZFaceField(grid)
    flux_divergence = CenterField(grid)

    update_rrtmgp_surface_boundary_conditions!(solver, surface_radiation, grid)

    return RadiativeTransferModel(convert(FT, solar_constant),
                                  solar_position,
                                  surface_radiation,
                                  background_atmosphere,
                                  atmospheric_state,
                                  solver,
                                  nothing,
                                  upwelling_longwave_flux,
                                  downwelling_longwave_flux,
                                  upwelling_shortwave_flux,
                                  downwelling_shortwave_flux,
                                  flux_divergence,
                                  liquid_effective_radius,
                                  ice_effective_radius,
                                  column_batches,
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

#####
##### Longitude initialization (dispatched on AbstractSolarPosition)
#####

# Apparent sun with explicit (λ, φ): broadcast the λ to every column
set_longitude!(rrtmgp_λ, sp::ApparentSolarPosition{<:Tuple}, grid) =
    _set_longitude_from_coordinate!(rrtmgp_λ, sp.coordinate, grid)

# Apparent sun without explicit coordinate: per-column λ from the grid
set_longitude!(rrtmgp_λ, ::ApparentSolarPosition{Nothing}, grid) =
    _set_longitude_from_grid!(rrtmgp_λ, grid)

# Diurnal cycle and fixed zenith: longitude is irrelevant for cos(θ_z), but
# RRTMGP still wants column λ values for the gas-state setup. Use grid coords.
set_longitude!(rrtmgp_λ, ::DiurnalSolarPosition, grid) =
    _set_longitude_from_grid!(rrtmgp_λ, grid)
set_longitude!(rrtmgp_λ, ::FixedCosineZenith, grid) =
    _set_longitude_from_grid!(rrtmgp_λ, grid)

@inline function _set_longitude_from_coordinate!(rrtmgp_λ, coordinate::Tuple, grid)
    λ = coordinate[1]
    rrtmgp_λ .= λ
    return nothing
end

function _set_longitude_from_grid!(rrtmgp_λ, grid)
    arch = grid.architecture
    launch!(arch, grid, :xy, _set_longitude_from_grid_kernel!, rrtmgp_λ, grid)
    return nothing
end

@kernel function _set_longitude_from_grid_kernel!(rrtmgp_λ, grid)
    i, j = @index(Global, NTuple)
    λ = xnode(i, j, 1, grid, Center(), Center(), Center())
    c = rrtmgp_column_index(i, j, grid.Nx)
    @inbounds rrtmgp_λ[c] = λ
end

#####
##### Update radiation
#####

"""
$(TYPEDSIGNATURES)

Update the full-spectrum radiative fluxes from the current model state, including the cloud
state for all-sky radiation.
"""
function AtmosphereModels._update_radiation!(rtm::FullSpectrumRadiativeTransferModel, model)
    assert_bound_surface_temperature(rtm)
    grid = model.grid
    clock = model.clock
    solver = rtm.longwave_solver

    # Surface emissivity and albedos, re-read in case they evolve
    update_rrtmgp_surface_boundary_conditions!(solver, rtm.surface_radiation, grid)

    # Update gas state
    update_rrtmgp_gas_state!(solver.as, model, rtm.surface_radiation.surface_temperature,
                             rtm.background_atmosphere, solver.params)

    # Update cloud state (a no-op for clear-sky, whose `cloud_state` is `nothing`)
    update_rrtmgp_cloud_state!(solver.as.cloud_state, model,
                               rtm.liquid_effective_radius,
                               rtm.ice_effective_radius)

    # Update solar zenith angle from the solar_position specification
    update_solar_zenith_angle!(solver.sws, rtm.solar_position, grid, clock)

    # Longwave and shortwave, one batch of columns at a time; night columns (`cos_zenith ≤ 0`)
    # get zero fluxes, which RRTMGP handles internally.
    solve_radiation_batches!(rtm, solver, grid)

    # Compute radiation flux divergence
    compute_radiation_flux_divergence!(rtm, grid)

    return nothing
end
