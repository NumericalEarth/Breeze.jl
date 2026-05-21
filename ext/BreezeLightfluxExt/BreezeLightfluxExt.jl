module BreezeLightfluxExt

using Lightflux
using Breeze

using Adapt: adapt
using Breeze.AtmosphereModels: AtmosphereModels, RadiativeHeatingOptics
using Breeze.AtmosphereModels: RadiativeTransferModel, SurfaceRadiativeProperties
using Breeze.AtmosphereModels: specific_humidity
using Breeze.Thermodynamics: ThermodynamicConstants
using KernelAbstractions: @index, @kernel
using Oceananigans.Architectures: CPU, architecture, on_architecture
using Oceananigans.Fields: CenterField, ConstantField, ZFaceField
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: Δzᶜᶜᶜ, ℑzᵃᵃᶠ
using Oceananigans.Utils: IterationInterval, launch!

struct RadiativeHeatingAtmosphericState{G, C, A, W, DW, V}
    gas_optics::G
    cloud_optics::C
    aerosol_optics::A
    workspace::W
    device_workspace::DW
    gas_values::V
end

struct RadiativeHeatingWorkspace{A, G, LW, SW, C, AO, F, H, Atm}
    pressure_layers::A
    pressure_interfaces::A
    temperature_layers::A
    temperature_interfaces::A
    gases::G
    longwave_optics::LW
    shortwave_optics::SW
    cloud_optical_properties::C
    aerosol_optical_properties::AO
    fluxes::F
    heating_rates::H
    atmosphere::Atm
end

const SupportedEcCKDModel =
    Union{Lightflux.EcCKDGasOpticsModel,
          Lightflux.EcCKDTabulatedGasOpticsModel}
const GRAVITY = 9.80665
const MOLAR_MASS_DRY_AIR = 0.0289647
const MOLAR_MASS_WATER = 0.01801528
const TABULATED_ECCKD_H100_MISSING_KERNEL_REQUIREMENTS = [
    "CPU/GPU parity smoke test for validated ecCKD tabulated gas optics",
]

has_tabulated_ecckd_interpolation_grids(gas_optics::Lightflux.EcCKDTabulatedGasOpticsModel) =
    !isempty(gas_optics.pressure_grid) &&
    !isempty(gas_optics.temperature_grid) &&
    !isempty(gas_optics.h2o_mole_fraction_grid)

materialize_surface_property(x::Number, grid) = ConstantField(convert(eltype(grid), x))
materialize_surface_property(x, grid) = x

function _json_string_field(text, key)
    match = Base.match(Regex("\"$key\"\\s*:\\s*\"([^\"]*)\""), text)
    return match === nothing ? "" : match.captures[1]
end

function _json_bool_field(text, key)
    match = Base.match(Regex("\"$key\"\\s*:\\s*(true|false)"), text)
    return match !== nothing && match.captures[1] == "true"
end

function tabulated_ecckd_parity_evidence(gas_optics)
    path = get(ENV, "RADIATIVE_HEATING_TABULATED_ECCKD_PARITY_JSON", "")
    path == "" && return (
        passed = false,
        reason = "no CPU/GPU parity artifact configured",
        path = path,
    )
    isfile(path) || return (
        passed = false,
        reason = "configured CPU/GPU parity artifact does not exist: $path",
        path = path,
    )

    text = read(path, String)
    status = _json_string_field(text, "status")
    case = _json_string_field(text, "case")
    kind = _json_string_field(text, "gas_model_kind")
    source = _json_string_field(text, "gas_model_source")
    expected_source = get(ENV, "RADIATIVE_HEATING_GAS_MODEL_SOURCE", "validated_ecCKD")
    expected_kind = expected_source == "validated_ecCKD_reduced_preflight_table_refined" ?
        "official_ecCKD_reduced_preflight_table_refined_$(size(gas_optics.longwave_absorption, 1))_lw_$(size(gas_optics.shortwave_absorption, 1))_sw" :
        "official_ecCKD_$(size(gas_optics.longwave_absorption, 1))_lw_$(size(gas_optics.shortwave_absorption, 1))_sw"
    passed = _json_bool_field(text, "passed")
    valid = case == "radiative_heating_tabulated_ecckd_cpu_gpu_parity" &&
            status == "passed" &&
            passed &&
            source == expected_source &&
            kind == expected_kind
    return (
        passed = valid,
        reason = valid ? "CPU/GPU parity artifact passed: $path" :
            "CPU/GPU parity artifact is not passing evidence for $expected_kind",
        path = path,
    )
end

specified_gas_value(gas_values::AbstractDict, name::Symbol, FT) =
    FT(get(gas_values, name, get(gas_values, String(name), zero(FT))))

specified_gas_value(gas_values, name::Symbol, FT) =
    FT(hasproperty(gas_values, name) ? getproperty(gas_values, name) : zero(FT))

struct TabulatedEcCKDColumnAmountState{A}
    composite::A
    h2o::A
    co2::A
    o3::A
    ch4::A
    n2o::A
    cfc11::A
    cfc12::A
end

function TabulatedEcCKDColumnAmountState(::Type{FT}, dims::NTuple{3, Int}) where FT
    arrays = ntuple(_ -> zeros(FT, dims), 8)
    return TabulatedEcCKDColumnAmountState(arrays...)
end

function tabulated_ecckd_column_amounts!(state::TabulatedEcCKDColumnAmountState,
                                         grid,
                                         pressure_interfaces,
                                         qᵛ;
                                         co2,
                                         o3,
                                         ch4,
                                         n2o,
                                         cfc11,
                                         cfc12)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _tabulated_ecckd_column_amounts!,
            state.composite,
            state.h2o,
            state.co2,
            state.o3,
            state.ch4,
            state.n2o,
            state.cfc11,
            state.cfc12,
            pressure_interfaces,
            qᵛ,
            co2,
            o3,
            ch4,
            n2o,
            cfc11,
            cfc12)
    return state
end

function tabulated_ecckd_device_column_state!(workspace,
                                              grid,
                                              pressure,
                                              temperature,
                                              qᵛ;
                                              co2,
                                              o3,
                                              ch4,
                                              n2o,
                                              cfc11,
                                              cfc12)
    arch = architecture(grid)
    amounts = workspace.column_amounts
    launch!(arch, grid, :xyz, _tabulated_ecckd_device_column_state!,
            workspace.pressure_layers,
            workspace.pressure_interfaces,
            workspace.temperature_layers,
            amounts.composite,
            amounts.h2o,
            amounts.co2,
            amounts.o3,
            amounts.ch4,
            amounts.n2o,
            amounts.cfc11,
            amounts.cfc12,
            grid,
            pressure,
            temperature,
            qᵛ,
            co2,
            o3,
            ch4,
            n2o,
            cfc11,
            cfc12)
    return workspace
end

@kernel function _tabulated_ecckd_column_amounts!(composite,
                                                  h2o,
                                                  co2_amount,
                                                  o3_amount,
                                                  ch4_amount,
                                                  n2o_amount,
                                                  cfc11_amount,
                                                  cfc12_amount,
                                                  pressure_interfaces,
                                                  qᵛ,
                                                  co2,
                                                  o3,
                                                  ch4,
                                                  n2o,
                                                  cfc11,
                                                  cfc12)
    i, j, kt = @index(Global, NTuple)
    FT = eltype(composite)
    Δp = abs(pressure_interfaces[i, j, kt + 1] - pressure_interfaces[i, j, kt])
    dry_air_moles = Δp / FT(GRAVITY) / FT(MOLAR_MASS_DRY_AIR)
    h2o_moles = max(qᵛ[i, j, kt], zero(FT)) * Δp / FT(GRAVITY) / FT(MOLAR_MASS_WATER)

    composite[i, j, kt] = dry_air_moles
    h2o[i, j, kt] = h2o_moles
    co2_amount[i, j, kt] = FT(co2) * dry_air_moles
    o3_amount[i, j, kt] = FT(o3) * dry_air_moles
    ch4_amount[i, j, kt] = FT(ch4) * dry_air_moles
    n2o_amount[i, j, kt] = FT(n2o) * dry_air_moles
    cfc11_amount[i, j, kt] = FT(cfc11) * dry_air_moles
    cfc12_amount[i, j, kt] = FT(cfc12) * dry_air_moles
end

@kernel function _tabulated_ecckd_device_column_state!(pressure_layers,
                                                       pressure_interfaces,
                                                       temperature_layers,
                                                       composite,
                                                       h2o,
                                                       co2_amount,
                                                       o3_amount,
                                                       ch4_amount,
                                                       n2o_amount,
                                                       cfc11_amount,
                                                       cfc12_amount,
                                                       grid,
                                                       pressure,
                                                       temperature,
                                                       qᵛ,
                                                       co2,
                                                       o3,
                                                       ch4,
                                                       n2o,
                                                       cfc11,
                                                       cfc12)
    i, j, k = @index(Global, NTuple)
    FT = eltype(composite)

    pressure_layers[i, j, k] = pressure[i, j, k]
    temperature_layers[i, j, k] = temperature[i, j, k]
    pressure_bottom = ℑzᵃᵃᶠ(i, j, k, grid, pressure)
    pressure_top = ℑzᵃᵃᶠ(i, j, k + 1, grid, pressure)
    if k == 1
        pressure_interfaces[i, j, 1] = pressure_bottom
    end
    pressure_interfaces[i, j, k + 1] = pressure_top

    Δp = abs(pressure_top - pressure_bottom)
    dry_air_moles = Δp / FT(GRAVITY) / FT(MOLAR_MASS_DRY_AIR)
    h2o_moles = max(qᵛ[i, j, k], zero(FT)) * Δp / FT(GRAVITY) / FT(MOLAR_MASS_WATER)

    composite[i, j, k] = dry_air_moles
    h2o[i, j, k] = h2o_moles
    co2_amount[i, j, k] = FT(co2) * dry_air_moles
    o3_amount[i, j, k] = FT(o3) * dry_air_moles
    ch4_amount[i, j, k] = FT(ch4) * dry_air_moles
    n2o_amount[i, j, k] = FT(n2o) * dry_air_moles
    cfc11_amount[i, j, k] = FT(cfc11) * dry_air_moles
    cfc12_amount[i, j, k] = FT(cfc12) * dry_air_moles
end

struct TabulatedEcCKDInterpolationState{I, A}
    pressure_lo::I
    pressure_hi::I
    temperature_lo::I
    temperature_hi::I
    h2o_lo::I
    h2o_hi::I
    pressure_weight::A
    temperature_weight::A
    h2o_weight::A
end

function TabulatedEcCKDInterpolationState(::Type{FT}, dims::NTuple{3, Int}) where FT
    integer_arrays = ntuple(_ -> zeros(Int, dims), 6)
    weights = ntuple(_ -> zeros(FT, dims), 3)
    return TabulatedEcCKDInterpolationState(integer_arrays..., weights...)
end

_arch_array(arch, ::Type{T}, dims::Tuple) where T = on_architecture(arch, zeros(T, dims))

function TabulatedEcCKDColumnAmountState(arch, ::Type{FT}, dims::NTuple{3, Int}) where FT
    arrays = ntuple(_ -> _arch_array(arch, FT, dims), 8)
    return TabulatedEcCKDColumnAmountState(arrays...)
end

function TabulatedEcCKDInterpolationState(arch, ::Type{FT}, dims::NTuple{3, Int}) where FT
    integer_arrays = ntuple(_ -> _arch_array(arch, Int, dims), 6)
    weights = ntuple(_ -> _arch_array(arch, FT, dims), 3)
    return TabulatedEcCKDInterpolationState(integer_arrays..., weights...)
end

"""
Persistent device workspace for the tabulated ecCKD device path.

The streaming radiation kernel computes per-g-point optical depths inline at
each column step, so the workspace only owns three-dimensional state arrays.
There are no `(ngpt, Nx, Ny, Nz)` four-dimensional buffers — those would scale
as `Nx * Ny * Nz * Nk` and dominate memory at high resolution. See
`_tabulated_ecckd_streaming_radiation!` for the fused optics+transport kernel
that consumes this workspace.
"""
struct TabulatedEcCKDDeviceWorkspace{C, I, P, F}
    column_amounts::C
    interpolation::I
    pressure_layers::P
    pressure_interfaces::F
    temperature_layers::P
end

function TabulatedEcCKDDeviceWorkspace(arch,
                                       ::Type{FT},
                                       grid,
                                       gas_optics::Lightflux.EcCKDTabulatedGasOpticsModel) where FT
    Nx, Ny, Nz = size(grid)
    dims = (Nx, Ny, Nz)
    face_dims = (Nx, Ny, Nz + 1)

    return TabulatedEcCKDDeviceWorkspace(
        TabulatedEcCKDColumnAmountState(arch, FT, dims),
        TabulatedEcCKDInterpolationState(arch, FT, dims),
        _arch_array(arch, FT, dims),
        _arch_array(arch, FT, face_dims),
        _arch_array(arch, FT, dims),
    )
end

@inline function _linear_bracket(grid, x)
    FT = eltype(grid)
    x_clamped = clamp(x, grid[begin], grid[end])
    lo = firstindex(grid)
    hi = lastindex(grid)
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        if x_clamped < grid[mid]
            hi = mid
        else
            lo = mid
        end
    end
    weight = (x_clamped - grid[lo]) / (grid[hi] - grid[lo])
    return lo, hi, weight
end

@inline function _log_bracket(grid, x)
    FT = eltype(grid)
    x_positive = max(x, grid[begin])
    index = one(FT) +
        clamp((log(x_positive) - log(grid[begin])) / (log(grid[begin + 1]) - log(grid[begin])),
              zero(FT),
              FT(length(grid)) - FT(1.0001))
    lo = Int(floor(index))
    return lo, lo + 1, index - lo
end

@inline function _temperature_bracket(temperature_grid::AbstractVector,
                                      pressure_grid,
                                      pressure,
                                      temperature,
                                      pressure_lo,
                                      pressure_hi,
                                      pressure_weight)
    return _linear_bracket(temperature_grid, temperature)
end

@inline function _temperature_bracket(temperature_grid::AbstractMatrix,
                                      pressure_grid,
                                      pressure,
                                      temperature,
                                      pressure_lo,
                                      pressure_hi,
                                      pressure_weight)
    FT = eltype(temperature_grid)
    origin = (one(FT) - pressure_weight) * temperature_grid[pressure_lo, 1] +
             pressure_weight * temperature_grid[pressure_hi, 1]
    step = temperature_grid[1, 2] - temperature_grid[1, 1]
    index = one(FT) + clamp((temperature - origin) / step,
                            zero(FT),
                            FT(size(temperature_grid, 2)) - FT(1.0001))
    lo = Int(floor(index))
    return lo, lo + 1, index - lo
end

function tabulated_ecckd_interpolation_state!(state::TabulatedEcCKDInterpolationState,
                                              grid,
                                              pressure_layers,
                                              temperature_layers,
                                              column_amounts::TabulatedEcCKDColumnAmountState,
                                              pressure_grid,
                                              temperature_grid,
                                              h2o_grid)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _tabulated_ecckd_interpolation_state!,
            state.pressure_lo,
            state.pressure_hi,
            state.temperature_lo,
            state.temperature_hi,
            state.h2o_lo,
            state.h2o_hi,
            state.pressure_weight,
            state.temperature_weight,
            state.h2o_weight,
            pressure_layers,
            temperature_layers,
            column_amounts.composite,
            column_amounts.h2o,
            pressure_grid,
            temperature_grid,
            h2o_grid)
    return state
end

@kernel function _tabulated_ecckd_interpolation_state!(pressure_lo,
                                                       pressure_hi,
                                                       temperature_lo,
                                                       temperature_hi,
                                                       h2o_lo,
                                                       h2o_hi,
                                                       pressure_weight,
                                                       temperature_weight,
                                                       h2o_weight,
                                                       pressure_layers,
                                                       temperature_layers,
                                                       composite_amount,
                                                       h2o_amount,
                                                       pressure_grid,
                                                       temperature_grid,
                                                       h2o_grid)
    i, j, k = @index(Global, NTuple)
    ip0, ip1, wp = _log_bracket(pressure_grid, pressure_layers[i, j, k])
    it0, it1, wt = _temperature_bracket(temperature_grid,
                                        pressure_grid,
                                        pressure_layers[i, j, k],
                                        temperature_layers[i, j, k],
                                        ip0,
                                        ip1,
                                        wp)
    h2o_fraction = h2o_amount[i, j, k] / max(composite_amount[i, j, k], eps(eltype(h2o_amount)))
    ih0, ih1, wh = _log_bracket(h2o_grid, h2o_fraction)

    pressure_lo[i, j, k] = ip0
    pressure_hi[i, j, k] = ip1
    temperature_lo[i, j, k] = it0
    temperature_hi[i, j, k] = it1
    h2o_lo[i, j, k] = ih0
    h2o_hi[i, j, k] = ih1
    pressure_weight[i, j, k] = wp
    temperature_weight[i, j, k] = wt
    h2o_weight[i, j, k] = wh
end

@inline function _prepared_interp_table(table,
                                        ig,
                                        gas_index,
                                        ip0,
                                        ip1,
                                        wp,
                                        it0,
                                        it1,
                                        wt)
    c00 = table[ig, gas_index, ip0, it0]
    c10 = table[ig, gas_index, ip1, it0]
    c01 = table[ig, gas_index, ip0, it1]
    c11 = table[ig, gas_index, ip1, it1]
    cp0 = c00 + wp * (c10 - c00)
    cp1 = c01 + wp * (c11 - c01)
    return cp0 + wt * (cp1 - cp0)
end

@inline function _prepared_interp_h2o_table(table,
                                            ig,
                                            ip0,
                                            ip1,
                                            wp,
                                            it0,
                                            it1,
                                            wt,
                                            ih0,
                                            ih1,
                                            wh)
    c000 = table[ig, ip0, it0, ih0]
    c100 = table[ig, ip1, it0, ih0]
    c010 = table[ig, ip0, it1, ih0]
    c110 = table[ig, ip1, it1, ih0]
    c001 = table[ig, ip0, it0, ih1]
    c101 = table[ig, ip1, it0, ih1]
    c011 = table[ig, ip0, it1, ih1]
    c111 = table[ig, ip1, it1, ih1]

    c00 = c000 + wp * (c100 - c000)
    c10 = c010 + wp * (c110 - c010)
    c01 = c001 + wp * (c101 - c001)
    c11 = c011 + wp * (c111 - c011)
    ct0 = c00 + wt * (c10 - c00)
    ct1 = c01 + wt * (c11 - c01)
    return ct0 + wh * (ct1 - ct0)
end

@inline function _tabulated_amount_by_index(amounts::TabulatedEcCKDColumnAmountState, gas_index, i, j, k)
    gas_index == 1 && return amounts.composite[i, j, k]
    gas_index == 2 && return amounts.h2o[i, j, k]
    gas_index == 3 && return amounts.o3[i, j, k]
    gas_index == 4 && return amounts.co2[i, j, k]
    gas_index == 5 && return amounts.ch4[i, j, k]
    gas_index == 6 && return amounts.n2o[i, j, k]
    gas_index == 7 && return amounts.cfc11[i, j, k]
    return amounts.cfc12[i, j, k]
end

@inline function _tabulated_amount_by_index(composite, h2o, o3, co2, ch4, n2o,
                                            cfc11, cfc12, gas_index, i, j, k)
    gas_index == 1 && return composite[i, j, k]
    gas_index == 2 && return h2o[i, j, k]
    gas_index == 3 && return o3[i, j, k]
    gas_index == 4 && return co2[i, j, k]
    gas_index == 5 && return ch4[i, j, k]
    gas_index == 6 && return n2o[i, j, k]
    gas_index == 7 && return cfc11[i, j, k]
    return cfc12[i, j, k]
end

function tabulated_ecckd_absorption_optical_depths!(longwave_tau,
                                                    shortwave_tau,
                                                    grid,
                                                    column_amounts::TabulatedEcCKDColumnAmountState,
                                                    interpolation::TabulatedEcCKDInterpolationState,
                                                    gas_reference_mole_fractions,
                                                    longwave_absorption,
                                                    shortwave_absorption,
                                                    longwave_h2o_absorption,
                                                    shortwave_h2o_absorption)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _tabulated_ecckd_absorption_optical_depths!,
            longwave_tau,
            shortwave_tau,
            column_amounts,
            interpolation,
            gas_reference_mole_fractions,
            longwave_absorption,
            shortwave_absorption,
            longwave_h2o_absorption,
            shortwave_h2o_absorption)
    return longwave_tau, shortwave_tau
end

@kernel function _tabulated_ecckd_absorption_optical_depths!(longwave_tau,
                                                             shortwave_tau,
                                                             amounts,
                                                             interpolation,
                                                             gas_reference_mole_fractions,
                                                             longwave_absorption,
                                                             shortwave_absorption,
                                                             longwave_h2o_absorption,
                                                             shortwave_h2o_absorption)
    i, j, k = @index(Global, NTuple)
    FT = eltype(longwave_tau)
    ip0 = interpolation.pressure_lo[i, j, k]
    ip1 = interpolation.pressure_hi[i, j, k]
    it0 = interpolation.temperature_lo[i, j, k]
    it1 = interpolation.temperature_hi[i, j, k]
    ih0 = interpolation.h2o_lo[i, j, k]
    ih1 = interpolation.h2o_hi[i, j, k]
    wp = interpolation.pressure_weight[i, j, k]
    wt = interpolation.temperature_weight[i, j, k]
    wh = interpolation.h2o_weight[i, j, k]
    composite = amounts.composite[i, j, k]
    h2o = amounts.h2o[i, j, k]

    for ig in axes(longwave_absorption, 1)
        tau = zero(FT)
        for gas_index in axes(longwave_absorption, 2)
            amount = FT(_tabulated_amount_by_index(amounts, gas_index, i, j, k)) -
                     FT(gas_reference_mole_fractions[gas_index]) * FT(composite)
            coefficient = _prepared_interp_table(longwave_absorption, ig, gas_index,
                                                 ip0, ip1, wp, it0, it1, wt)
            tau += coefficient * amount
        end
        if length(longwave_h2o_absorption) != 0
            tau += _prepared_interp_h2o_table(longwave_h2o_absorption, ig,
                                              ip0, ip1, wp, it0, it1, wt,
                                              ih0, ih1, wh) * FT(h2o)
        end
        longwave_tau[ig, i, j, k] = tau
    end

    for ig in axes(shortwave_absorption, 1)
        tau = zero(FT)
        for gas_index in axes(shortwave_absorption, 2)
            amount = FT(_tabulated_amount_by_index(amounts, gas_index, i, j, k)) -
                     FT(gas_reference_mole_fractions[gas_index]) * FT(composite)
            coefficient = _prepared_interp_table(shortwave_absorption, ig, gas_index,
                                                 ip0, ip1, wp, it0, it1, wt)
            tau += coefficient * amount
        end
        if length(shortwave_h2o_absorption) != 0
            tau += _prepared_interp_h2o_table(shortwave_h2o_absorption, ig,
                                              ip0, ip1, wp, it0, it1, wt,
                                              ih0, ih1, wh) * FT(h2o)
        end
        shortwave_tau[ig, i, j, k] = tau
    end
end

function tabulated_ecckd_rayleigh_optical_depths!(rayleigh_tau,
                                                  grid,
                                                  column_amounts::TabulatedEcCKDColumnAmountState,
                                                  shortwave_rayleigh_molar_scattering)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _tabulated_ecckd_rayleigh_optical_depths!,
            rayleigh_tau,
            column_amounts.composite,
            shortwave_rayleigh_molar_scattering)
    return rayleigh_tau
end

@kernel function _tabulated_ecckd_rayleigh_optical_depths!(rayleigh_tau,
                                                           composite_amount,
                                                           shortwave_rayleigh_molar_scattering)
    i, j, k = @index(Global, NTuple)
    FT = eltype(rayleigh_tau)
    for ig in axes(rayleigh_tau, 1)
        rayleigh_tau[ig, i, j, k] =
            FT(shortwave_rayleigh_molar_scattering[ig]) * FT(composite_amount[i, j, k])
    end
end

@inline function _prepared_interp_source_table(source_table,
                                               source_temperature_grid,
                                               ig,
                                               temperature)
    it0, it1, wt = _linear_bracket(source_temperature_grid, temperature)
    return source_table[ig, it0] + wt * (source_table[ig, it1] - source_table[ig, it0])
end

function tabulated_ecckd_longwave_sources!(source,
                                           grid,
                                           temperature_layers,
                                           source_temperature_grid,
                                           source_table)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _tabulated_ecckd_longwave_sources!,
            source,
            temperature_layers,
            source_temperature_grid,
            source_table)
    return source
end

@kernel function _tabulated_ecckd_longwave_sources!(source,
                                                    temperature_layers,
                                                    source_temperature_grid,
                                                    source_table)
    i, j, k = @index(Global, NTuple)
    for ig in axes(source, 1)
        source[ig, i, j, k] =
            _prepared_interp_source_table(source_table,
                                          source_temperature_grid,
                                          ig,
                                          temperature_layers[i, j, k])
    end
end

function tabulated_ecckd_optical_properties!(longwave_tau,
                                             shortwave_tau,
                                             rayleigh_tau,
                                             longwave_source,
                                             grid,
                                             temperature_layers,
                                             column_amounts::TabulatedEcCKDColumnAmountState,
                                             interpolation::TabulatedEcCKDInterpolationState,
                                             gas_reference_mole_fractions,
                                             longwave_absorption,
                                             shortwave_absorption,
                                             longwave_h2o_absorption,
                                             shortwave_h2o_absorption,
                                             shortwave_rayleigh_molar_scattering,
                                             source_temperature_grid,
                                             source_table)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _tabulated_ecckd_optical_properties!,
            longwave_tau,
            shortwave_tau,
            rayleigh_tau,
            longwave_source,
            temperature_layers,
            column_amounts.composite,
            column_amounts.h2o,
            column_amounts.o3,
            column_amounts.co2,
            column_amounts.ch4,
            column_amounts.n2o,
            column_amounts.cfc11,
            column_amounts.cfc12,
            interpolation.pressure_lo,
            interpolation.pressure_hi,
            interpolation.temperature_lo,
            interpolation.temperature_hi,
            interpolation.h2o_lo,
            interpolation.h2o_hi,
            interpolation.pressure_weight,
            interpolation.temperature_weight,
            interpolation.h2o_weight,
            gas_reference_mole_fractions,
            longwave_absorption,
            shortwave_absorption,
            longwave_h2o_absorption,
            shortwave_h2o_absorption,
            shortwave_rayleigh_molar_scattering,
            source_temperature_grid,
            source_table)
    return longwave_tau, shortwave_tau, rayleigh_tau, longwave_source
end

@kernel function _tabulated_ecckd_optical_properties!(longwave_tau,
                                                      shortwave_tau,
                                                      rayleigh_tau,
                                                      longwave_source,
                                                      temperature_layers,
                                                      composite_amount,
                                                      h2o_amount,
                                                      o3_amount,
                                                      co2_amount,
                                                      ch4_amount,
                                                      n2o_amount,
                                                      cfc11_amount,
                                                      cfc12_amount,
                                                      pressure_lo,
                                                      pressure_hi,
                                                      temperature_lo,
                                                      temperature_hi,
                                                      h2o_lo,
                                                      h2o_hi,
                                                      pressure_weight,
                                                      temperature_weight,
                                                      h2o_weight,
                                                      gas_reference_mole_fractions,
                                                      longwave_absorption,
                                                      shortwave_absorption,
                                                      longwave_h2o_absorption,
                                                      shortwave_h2o_absorption,
                                                      shortwave_rayleigh_molar_scattering,
                                                      source_temperature_grid,
                                                      source_table)
    i, j, k = @index(Global, NTuple)
    FT = eltype(longwave_tau)
    ip0 = pressure_lo[i, j, k]
    ip1 = pressure_hi[i, j, k]
    it0 = temperature_lo[i, j, k]
    it1 = temperature_hi[i, j, k]
    ih0 = h2o_lo[i, j, k]
    ih1 = h2o_hi[i, j, k]
    wp = pressure_weight[i, j, k]
    wt = temperature_weight[i, j, k]
    wh = h2o_weight[i, j, k]
    composite = composite_amount[i, j, k]
    h2o = h2o_amount[i, j, k]

    for ig in axes(longwave_absorption, 1)
        tau = zero(FT)
        for gas_index in axes(longwave_absorption, 2)
            amount = FT(_tabulated_amount_by_index(composite_amount, h2o_amount,
                                                   o3_amount, co2_amount, ch4_amount,
                                                   n2o_amount, cfc11_amount,
                                                   cfc12_amount, gas_index, i, j, k)) -
                     FT(gas_reference_mole_fractions[gas_index]) * FT(composite)
            coefficient = _prepared_interp_table(longwave_absorption, ig, gas_index,
                                                 ip0, ip1, wp, it0, it1, wt)
            tau += coefficient * amount
        end
        if length(longwave_h2o_absorption) != 0
            tau += _prepared_interp_h2o_table(longwave_h2o_absorption, ig,
                                              ip0, ip1, wp, it0, it1, wt,
                                              ih0, ih1, wh) * FT(h2o)
        end
        longwave_tau[ig, i, j, k] = tau
        longwave_source[ig, i, j, k] =
            _prepared_interp_source_table(source_table,
                                          source_temperature_grid,
                                          ig,
                                          temperature_layers[i, j, k])
    end

    for ig in axes(shortwave_absorption, 1)
        tau = zero(FT)
        for gas_index in axes(shortwave_absorption, 2)
            amount = FT(_tabulated_amount_by_index(composite_amount, h2o_amount,
                                                   o3_amount, co2_amount, ch4_amount,
                                                   n2o_amount, cfc11_amount,
                                                   cfc12_amount, gas_index, i, j, k)) -
                     FT(gas_reference_mole_fractions[gas_index]) * FT(composite)
            coefficient = _prepared_interp_table(shortwave_absorption, ig, gas_index,
                                                 ip0, ip1, wp, it0, it1, wt)
            tau += coefficient * amount
        end
        if length(shortwave_h2o_absorption) != 0
            tau += _prepared_interp_h2o_table(shortwave_h2o_absorption, ig,
                                              ip0, ip1, wp, it0, it1, wt,
                                              ih0, ih1, wh) * FT(h2o)
        end
        shortwave_tau[ig, i, j, k] = tau
        rayleigh_tau[ig, i, j, k] =
            FT(shortwave_rayleigh_molar_scattering[ig]) * FT(composite)
    end
end

function tabulated_ecckd_flux_divergence!(lw_up_field,
                                          lw_down_field,
                                          sw_down_field,
                                          flux_divergence,
                                          grid,
                                          longwave_tau,
                                          shortwave_tau,
                                          rayleigh_tau,
                                          longwave_source,
                                          longwave_weights,
                                          shortwave_weights;
                                          solar_constant,
                                          surface_temperature,
                                          surface_albedo,
                                          surface_emissivity)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _tabulated_ecckd_flux_divergence!,
            lw_up_field,
            lw_down_field,
            sw_down_field,
            flux_divergence,
            grid,
            longwave_tau,
            shortwave_tau,
            rayleigh_tau,
            longwave_source,
            longwave_weights,
            shortwave_weights,
            solar_constant,
            surface_temperature,
            surface_albedo,
            surface_emissivity)
    return lw_up_field, lw_down_field, sw_down_field, flux_divergence
end

@kernel function _tabulated_ecckd_flux_divergence!(lw_up_field,
                                                   lw_down_field,
                                                   sw_down_field,
                                                   flux_divergence,
                                                   grid,
                                                   longwave_tau,
                                                   shortwave_tau,
                                                   rayleigh_tau,
                                                   longwave_source,
                                                   longwave_weights,
                                                   shortwave_weights,
                                                   solar_constant,
                                                   surface_temperature,
                                                   surface_albedo,
                                                   surface_emissivity)
    i, j = @index(Global, NTuple)

    Nz = size(grid, 3)
    FT = eltype(grid)
    σ = FT(5.670374419e-8)
    T_surface = _field_or_number(surface_temperature, i, j)
    α_surface = _field_or_number(surface_albedo, i, j)
    ε_surface = FT(surface_emissivity)
    surface_longwave_up = ε_surface * σ * T_surface^4

    for k in 1:(Nz + 1)
        lw_up_field[i, j, k] = zero(FT)
        lw_down_field[i, j, k] = zero(FT)
        sw_down_field[i, j, k] = zero(FT)
    end

    for ig in axes(longwave_tau, 1)
        weight = FT(longwave_weights[ig])
        up = surface_longwave_up
        lw_up_field[i, j, 1] += weight * up
        for k in 1:Nz
            τ = max(FT(longwave_tau[ig, i, j, k]), zero(FT))
            tr = exp(-τ)
            source = FT(longwave_source[ig, i, j, k])
            up = up * tr + source * (one(FT) - tr)
            lw_up_field[i, j, k + 1] += weight * up
        end

        down = zero(FT)
        for k in Nz:-1:1
            τ = max(FT(longwave_tau[ig, i, j, k]), zero(FT))
            tr = exp(-τ)
            source = FT(longwave_source[ig, i, j, k])
            down = down * tr + source * (one(FT) - tr)
            lw_down_field[i, j, k] -= weight * down
        end
    end

    for ig in axes(shortwave_tau, 1)
        weight = FT(shortwave_weights[ig])
        down = FT(solar_constant)
        sw_down_field[i, j, Nz + 1] -= weight * down
        for k in Nz:-1:1
            absorption = max(FT(shortwave_tau[ig, i, j, k]), zero(FT))
            down *= exp(-absorption)
            sw_down_field[i, j, k] -= weight * down
        end

        up = α_surface * down
        for k in 1:Nz
            absorption = max(FT(shortwave_tau[ig, i, j, k]), zero(FT))
            up *= exp(-absorption)
        end
    end

    for k in 1:Nz
        net_bottom = lw_up_field[i, j, k] +
                     lw_down_field[i, j, k] +
                     sw_down_field[i, j, k]
        net_top = lw_up_field[i, j, k + 1] +
                  lw_down_field[i, j, k + 1] +
                  sw_down_field[i, j, k + 1]
        flux_divergence[i, j, k] = -(net_top - net_bottom) / Δzᶜᶜᶜ(i, j, k, grid)
    end
end

"""
    tabulated_ecckd_streaming_radiation!(lw_up_field, lw_down_field, sw_down_field,
                                         flux_divergence, grid, workspace, gas_optics;
                                         solar_constant, surface_temperature,
                                         surface_albedo, surface_emissivity)

Fused tabulated-ecCKD optics + radiative transfer that consumes only the 3-D
state on `workspace` and streams g-points one at a time inside the column
kernel. Avoids the `(ngpt, Nx, Ny, Nz)` four-dimensional optical-property
buffers the older `tabulated_ecckd_optical_properties!` + `tabulated_ecckd_flux_divergence!`
pipeline required, so memory scales as `Nx * Ny * Nz` rather than
`Nx * Ny * Nz * ngpt`.

Cost: per (i, j) thread, longwave optical depth is computed twice per
(g-point × layer) — once for the upward sweep, once for the downward — and
shortwave is computed once. The eliminated 4-D allocations are typically
the dominant device memory at production grids (a 512×512×128 grid with
96-g shortwave saves 25.8 GiB per 4-D array).
"""
function tabulated_ecckd_streaming_radiation!(lw_up_field,
                                              lw_down_field,
                                              sw_down_field,
                                              flux_divergence,
                                              grid,
                                              workspace::TabulatedEcCKDDeviceWorkspace,
                                              gas_optics::Lightflux.EcCKDTabulatedGasOpticsModel;
                                              solar_constant,
                                              surface_temperature,
                                              surface_albedo,
                                              surface_emissivity)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _tabulated_ecckd_streaming_radiation!,
            lw_up_field, lw_down_field, sw_down_field, flux_divergence,
            grid,
            workspace.temperature_layers,
            workspace.column_amounts.composite,
            workspace.column_amounts.h2o,
            workspace.column_amounts.o3,
            workspace.column_amounts.co2,
            workspace.column_amounts.ch4,
            workspace.column_amounts.n2o,
            workspace.column_amounts.cfc11,
            workspace.column_amounts.cfc12,
            workspace.interpolation.pressure_lo,
            workspace.interpolation.pressure_hi,
            workspace.interpolation.temperature_lo,
            workspace.interpolation.temperature_hi,
            workspace.interpolation.h2o_lo,
            workspace.interpolation.h2o_hi,
            workspace.interpolation.pressure_weight,
            workspace.interpolation.temperature_weight,
            workspace.interpolation.h2o_weight,
            gas_optics.gas_reference_mole_fractions,
            gas_optics.longwave_absorption,
            gas_optics.shortwave_absorption,
            gas_optics.longwave_h2o_absorption,
            gas_optics.shortwave_h2o_absorption,
            gas_optics.longwave_source_temperature_grid,
            gas_optics.longwave_source_table,
            gas_optics.longwave_weights,
            gas_optics.shortwave_weights,
            solar_constant,
            surface_temperature,
            surface_albedo,
            surface_emissivity)
    return nothing
end

@kernel function _tabulated_ecckd_streaming_radiation!(lw_up_field,
                                                       lw_down_field,
                                                       sw_down_field,
                                                       flux_divergence,
                                                       grid,
                                                       temperature_layers,
                                                       composite_amount,
                                                       h2o_amount,
                                                       o3_amount,
                                                       co2_amount,
                                                       ch4_amount,
                                                       n2o_amount,
                                                       cfc11_amount,
                                                       cfc12_amount,
                                                       pressure_lo,
                                                       pressure_hi,
                                                       temperature_lo,
                                                       temperature_hi,
                                                       h2o_lo,
                                                       h2o_hi,
                                                       pressure_weight,
                                                       temperature_weight,
                                                       h2o_weight,
                                                       gas_reference_mole_fractions,
                                                       longwave_absorption,
                                                       shortwave_absorption,
                                                       longwave_h2o_absorption,
                                                       shortwave_h2o_absorption,
                                                       source_temperature_grid,
                                                       source_table,
                                                       longwave_weights,
                                                       shortwave_weights,
                                                       solar_constant,
                                                       surface_temperature,
                                                       surface_albedo,
                                                       surface_emissivity)
    i, j = @index(Global, NTuple)

    Nz = size(grid, 3)
    FT = eltype(grid)
    σ = FT(5.670374419e-8)
    T_surface = _field_or_number(surface_temperature, i, j)
    α_surface = _field_or_number(surface_albedo, i, j)
    ε_surface = FT(surface_emissivity)
    surface_longwave_up = ε_surface * σ * T_surface^4

    @inbounds for k in 1:(Nz + 1)
        lw_up_field[i, j, k] = zero(FT)
        lw_down_field[i, j, k] = zero(FT)
        sw_down_field[i, j, k] = zero(FT)
    end

    have_lw_h2o = length(longwave_h2o_absorption) != 0
    have_sw_h2o = length(shortwave_h2o_absorption) != 0
    n_gas_lw = size(longwave_absorption, 2)
    n_gas_sw = size(shortwave_absorption, 2)

    # Longwave streaming: for each g-point, do upward then downward sweep,
    # computing optical depth and Planck source inline. Never materialize a
    # 4-D (ngpt, Nx, Ny, Nz) array.
    @inbounds for ig in axes(longwave_absorption, 1)
        weight = FT(longwave_weights[ig])

        up = surface_longwave_up
        lw_up_field[i, j, 1] += weight * up
        for k in 1:Nz
            ip0 = pressure_lo[i, j, k]
            ip1 = pressure_hi[i, j, k]
            it0 = temperature_lo[i, j, k]
            it1 = temperature_hi[i, j, k]
            wp = pressure_weight[i, j, k]
            wt = temperature_weight[i, j, k]
            composite = composite_amount[i, j, k]

            tau = zero(FT)
            for gas_index in 1:n_gas_lw
                amount = FT(_tabulated_amount_by_index(composite_amount, h2o_amount,
                                                       o3_amount, co2_amount, ch4_amount,
                                                       n2o_amount, cfc11_amount, cfc12_amount,
                                                       gas_index, i, j, k)) -
                         FT(gas_reference_mole_fractions[gas_index]) * FT(composite)
                coefficient = _prepared_interp_table(longwave_absorption, ig, gas_index,
                                                     ip0, ip1, wp, it0, it1, wt)
                tau += coefficient * amount
            end
            if have_lw_h2o
                ih0 = h2o_lo[i, j, k]
                ih1 = h2o_hi[i, j, k]
                wh = h2o_weight[i, j, k]
                h2o = h2o_amount[i, j, k]
                tau += _prepared_interp_h2o_table(longwave_h2o_absorption, ig,
                                                  ip0, ip1, wp, it0, it1, wt,
                                                  ih0, ih1, wh) * FT(h2o)
            end
            tau = max(tau, zero(FT))
            source = _prepared_interp_source_table(source_table, source_temperature_grid,
                                                   ig, temperature_layers[i, j, k])
            tr = exp(-tau)
            up = up * tr + source * (one(FT) - tr)
            lw_up_field[i, j, k + 1] += weight * up
        end

        down = zero(FT)
        for k in Nz:-1:1
            ip0 = pressure_lo[i, j, k]
            ip1 = pressure_hi[i, j, k]
            it0 = temperature_lo[i, j, k]
            it1 = temperature_hi[i, j, k]
            wp = pressure_weight[i, j, k]
            wt = temperature_weight[i, j, k]
            composite = composite_amount[i, j, k]

            tau = zero(FT)
            for gas_index in 1:n_gas_lw
                amount = FT(_tabulated_amount_by_index(composite_amount, h2o_amount,
                                                       o3_amount, co2_amount, ch4_amount,
                                                       n2o_amount, cfc11_amount, cfc12_amount,
                                                       gas_index, i, j, k)) -
                         FT(gas_reference_mole_fractions[gas_index]) * FT(composite)
                coefficient = _prepared_interp_table(longwave_absorption, ig, gas_index,
                                                     ip0, ip1, wp, it0, it1, wt)
                tau += coefficient * amount
            end
            if have_lw_h2o
                ih0 = h2o_lo[i, j, k]
                ih1 = h2o_hi[i, j, k]
                wh = h2o_weight[i, j, k]
                h2o = h2o_amount[i, j, k]
                tau += _prepared_interp_h2o_table(longwave_h2o_absorption, ig,
                                                  ip0, ip1, wp, it0, it1, wt,
                                                  ih0, ih1, wh) * FT(h2o)
            end
            tau = max(tau, zero(FT))
            source = _prepared_interp_source_table(source_table, source_temperature_grid,
                                                   ig, temperature_layers[i, j, k])
            tr = exp(-tau)
            down = down * tr + source * (one(FT) - tr)
            lw_down_field[i, j, k] -= weight * down
        end
    end

    # Shortwave streaming: downward sweep only is integrated into the
    # downwelling shortwave flux field, matching the existing semantics in
    # `_tabulated_ecckd_flux_divergence!` (no shortwave-up flux field is
    # currently allocated on the radiative transfer model).
    @inbounds for ig in axes(shortwave_absorption, 1)
        weight = FT(shortwave_weights[ig])
        down = FT(solar_constant)
        sw_down_field[i, j, Nz + 1] -= weight * down
        for k in Nz:-1:1
            ip0 = pressure_lo[i, j, k]
            ip1 = pressure_hi[i, j, k]
            it0 = temperature_lo[i, j, k]
            it1 = temperature_hi[i, j, k]
            wp = pressure_weight[i, j, k]
            wt = temperature_weight[i, j, k]
            composite = composite_amount[i, j, k]

            tau = zero(FT)
            for gas_index in 1:n_gas_sw
                amount = FT(_tabulated_amount_by_index(composite_amount, h2o_amount,
                                                       o3_amount, co2_amount, ch4_amount,
                                                       n2o_amount, cfc11_amount, cfc12_amount,
                                                       gas_index, i, j, k)) -
                         FT(gas_reference_mole_fractions[gas_index]) * FT(composite)
                coefficient = _prepared_interp_table(shortwave_absorption, ig, gas_index,
                                                     ip0, ip1, wp, it0, it1, wt)
                tau += coefficient * amount
            end
            if have_sw_h2o
                ih0 = h2o_lo[i, j, k]
                ih1 = h2o_hi[i, j, k]
                wh = h2o_weight[i, j, k]
                h2o = h2o_amount[i, j, k]
                tau += _prepared_interp_h2o_table(shortwave_h2o_absorption, ig,
                                                  ip0, ip1, wp, it0, it1, wt,
                                                  ih0, ih1, wh) * FT(h2o)
            end
            tau = max(tau, zero(FT))
            down *= exp(-tau)
            sw_down_field[i, j, k] -= weight * down
        end
    end

    @inbounds for k in 1:Nz
        net_bottom = lw_up_field[i, j, k] +
                     lw_down_field[i, j, k] +
                     sw_down_field[i, j, k]
        net_top = lw_up_field[i, j, k + 1] +
                  lw_down_field[i, j, k + 1] +
                  sw_down_field[i, j, k + 1]
        flux_divergence[i, j, k] = -(net_top - net_bottom) / Δzᶜᶜᶜ(i, j, k, grid)
    end
end

function radiative_heating_device_support(backend, gas_optics)
    backend == "H100" || return (
        status = "supported",
        reason = "CPU benchmark path supports package-native gas optics",
        missing_kernel_requirements = String[],
        source = "BreezeLightfluxExt",
    )
    if gas_optics isa Lightflux.EcCKDGasOpticsModel &&
       Lightflux.gas_names(gas_optics) == (:h2o, :co2)
        return (
            status = "supported",
            reason = "H100 extension path supports fixed two-gas EcCKDGasOpticsModel",
            missing_kernel_requirements = String[],
            source = "BreezeLightfluxExt",
        )
    elseif gas_optics isa Lightflux.EcCKDTabulatedGasOpticsModel
        parity = tabulated_ecckd_parity_evidence(gas_optics)
        parity.passed && return (
            status = "supported",
            reason = "H100 extension path supports tabulated multi-gas ecCKD with recorded CPU/GPU parity evidence",
            missing_kernel_requirements = String[],
            source = "BreezeLightfluxExt",
        )
        return (
            status = "unsupported",
            reason = "H100 extension path has tabulated multi-gas ecCKD kernels wired but still requires CPU/GPU parity evidence ($(parity.reason))",
            missing_kernel_requirements = copy(TABULATED_ECCKD_H100_MISSING_KERNEL_REQUIREMENTS),
            source = "BreezeLightfluxExt",
        )
    end
    return (
        status = "unsupported",
        reason = "H100 extension path only supports fixed two-gas EcCKDGasOpticsModel",
        missing_kernel_requirements = ["device kernel for this gas optics model type"],
        source = "BreezeLightfluxExt",
    )
end

function materialize_gases(::Val{GasNames}, FT, nlayers, gas_values) where GasNames
    values = ntuple(Val(length(GasNames))) do n
        fill(specified_gas_value(gas_values, GasNames[n], FT), nlayers)
    end
    return NamedTuple{GasNames}(values)
end

function RadiativeHeatingWorkspace(grid::AbstractGrid,
                                   gas_optics::SupportedEcCKDModel;
                                   gas_values = (; co2 = 400e-6),
                                   cloud_optics = nothing,
                                   aerosol_optics = nothing)
    FT = eltype(grid)
    nlayers = size(grid, 3)
    longwave_ng = size(gas_optics.longwave_absorption, 1)
    shortwave_ng = size(gas_optics.shortwave_absorption, 1)

    pressure_layers = zeros(FT, nlayers)
    pressure_interfaces = zeros(FT, nlayers + 1)
    temperature_layers = zeros(FT, nlayers)
    temperature_interfaces = zeros(FT, nlayers + 1)
    gases = materialize_gases(Val(Lightflux.gas_names(gas_optics)), FT, nlayers, gas_values)

    longwave_optics = Lightflux.LongwaveOpticalProperties(
        zeros(FT, longwave_ng, nlayers),
        zeros(FT, longwave_ng, nlayers),
        weights = copy(gas_optics.longwave_weights),
    )
    shortwave_optics = Lightflux.ShortwaveOpticalProperties(
        zeros(FT, shortwave_ng, nlayers),
        weights = copy(gas_optics.shortwave_weights),
    )
    cloud_properties = isnothing(cloud_optics) ? nothing :
        Lightflux.CloudOpticalProperties(zeros(FT, nlayers), zeros(FT, nlayers))
    aerosol_properties = isnothing(aerosol_optics) ? nothing :
        Lightflux.AerosolOpticalProperties(zeros(FT, nlayers), zeros(FT, nlayers))
    fluxes = Lightflux.RadiativeFluxes(
        longwave_up = zeros(FT, nlayers + 1),
        longwave_down = zeros(FT, nlayers + 1),
        shortwave_up = zeros(FT, nlayers + 1),
        shortwave_down = zeros(FT, nlayers + 1),
    )
    heating_rates = zeros(FT, nlayers)
    atmosphere = Lightflux.ColumnAtmosphere(
        pressure_layers = pressure_layers,
        pressure_interfaces = pressure_interfaces,
        temperature_layers = temperature_layers,
        temperature_interfaces = temperature_interfaces,
        gases = gases,
        surface = (;),
        geometry = (;),
    )

    return RadiativeHeatingWorkspace(pressure_layers,
                                     pressure_interfaces,
                                     temperature_layers,
                                     temperature_interfaces,
                                     gases,
                                     longwave_optics,
                                     shortwave_optics,
                                     cloud_properties,
                                     aerosol_properties,
                                     fluxes,
                                     heating_rates,
                                     atmosphere)
end

function RadiativeHeatingWorkspace(grid::AbstractGrid,
                                   gas_optics::Lightflux.AbstractGasOpticsModel; kw...)
    throw(ArgumentError("BreezeLightfluxExt currently supports EcCKDGasOpticsModel and EcCKDTabulatedGasOpticsModel"))
end

"""
    AtmosphereModels.RadiativeTransferModel(grid, RadiativeHeatingOptics(), constants; gas_optics, ...)

Construct a Breeze radiation model backed by `Lightflux.jl`.

The extension keeps the component work arrays in `RadiativeHeatingWorkspace`.
Callers that own their own dynamics can reuse `fill_column_state!`,
`radiative_heating_column_fluxes!`, and `column_heating_rates!` independently
of Breeze's `_update_radiation!` path.
"""
function AtmosphereModels.RadiativeTransferModel(grid::AbstractGrid,
                                                 ::RadiativeHeatingOptics,
                                                 constants::ThermodynamicConstants;
                                                 gas_optics,
                                                 longwave_solver = Lightflux.CloudlessLongwave(),
                                                 shortwave_solver = Lightflux.CloudlessShortwave(),
                                                 workspace = nothing,
                                                 gas_values = (; co2 = 400e-6),
                                                 cloud_optics = nothing,
                                                 aerosol_optics = nothing,
                                                 solar_constant = 1361,
                                                 coordinate = nothing,
                                                 epoch = nothing,
                                                 surface_temperature = 300,
                                                 surface_albedo = 0.1,
                                                 surface_emissivity = 1,
                                                 schedule = IterationInterval(1))
    gas_optics isa Lightflux.AbstractGasOpticsModel ||
        throw(ArgumentError("gas_optics must be an Lightflux.AbstractGasOpticsModel"))
    isnothing(cloud_optics) || cloud_optics isa Lightflux.AbstractCloudOpticsModel ||
        throw(ArgumentError("cloud_optics must be nothing or an Lightflux.AbstractCloudOpticsModel"))
    isnothing(aerosol_optics) || aerosol_optics isa Lightflux.AbstractAerosolOpticsModel ||
        throw(ArgumentError("aerosol_optics must be nothing or an Lightflux.AbstractAerosolOpticsModel"))

    FT = eltype(grid)
    arch = architecture(grid)
    runtime_gas_optics = arch isa CPU ? gas_optics : adapt(arch.device, gas_optics)
    workspace = isnothing(workspace) ?
        RadiativeHeatingWorkspace(grid, gas_optics; gas_values, cloud_optics, aerosol_optics) :
        workspace
    device_workspace = gas_optics isa Lightflux.EcCKDTabulatedGasOpticsModel &&
        has_tabulated_ecckd_interpolation_grids(gas_optics) ?
        TabulatedEcCKDDeviceWorkspace(arch, FT, grid, gas_optics) : nothing
    atmospheric_state = RadiativeHeatingAtmosphericState(runtime_gas_optics, cloud_optics, aerosol_optics, workspace, device_workspace, gas_values)
    surface_properties = SurfaceRadiativeProperties(materialize_surface_property(surface_temperature, grid),
                                                    convert(FT, surface_emissivity),
                                                    materialize_surface_property(surface_albedo, grid),
                                                    materialize_surface_property(surface_albedo, grid))

    return RadiativeTransferModel(convert(FT, solar_constant),
                                  coordinate,
                                  epoch,
                                  surface_properties,
                                  nothing,
                                  atmospheric_state,
                                  longwave_solver,
                                  shortwave_solver,
                                  ZFaceField(grid),
                                  ZFaceField(grid),
                                  ZFaceField(grid),
                                  CenterField(grid),
                                  nothing,
                                  nothing,
                                  schedule)
end

@inline surface_value(x::Number, i, j) = x
@inline surface_value(x, i, j) = x[i, j, 1]

function fill_column_gases!(workspace,
                            ::Lightflux.EcCKDGasOpticsModel,
                            qᵛ,
                            gas_values,
                            grid,
                            i,
                            j,
                            kt,
                            kz)
    if haskey(workspace.gases, :h2o)
        workspace.gases.h2o[kt] = max(qᵛ[i, j, kz], zero(eltype(grid)))
    end
    return nothing
end

function fill_column_gases!(workspace,
                            ::Lightflux.EcCKDTabulatedGasOpticsModel,
                            qᵛ,
                            gas_values,
                            grid,
                            i,
                            j,
                            kt,
                            kz)
    FT = eltype(grid)
    Δp = abs(workspace.pressure_interfaces[kt + 1] - workspace.pressure_interfaces[kt])
    dry_air_moles = Δp / FT(GRAVITY) / FT(MOLAR_MASS_DRY_AIR)
    h2o_moles = max(qᵛ[i, j, kz], zero(FT)) * Δp / FT(GRAVITY) / FT(MOLAR_MASS_WATER)

    haskey(workspace.gases, :composite) && (workspace.gases.composite[kt] = dry_air_moles)
    haskey(workspace.gases, :h2o) && (workspace.gases.h2o[kt] = h2o_moles)
    for name in (:co2, :o3, :ch4, :n2o, :cfc11, :cfc12)
        if haskey(workspace.gases, name)
            workspace.gases[name][kt] = specified_gas_value(gas_values, name, FT) * dry_air_moles
        end
    end
    return nothing
end

function fill_column_state!(workspace::RadiativeHeatingWorkspace, rtm, model, i, j)
    grid = model.grid
    nlayers = size(grid, 3)
    pressure = model.dynamics.reference_state.pressure
    temperature = model.temperature
    qᵛ = specific_humidity(model)

    for kt in 1:nlayers
        kz = nlayers - kt + 1
        workspace.pressure_layers[kt] = pressure[i, j, kz]
        workspace.temperature_layers[kt] = temperature[i, j, kz]
        workspace.temperature_interfaces[kt] = ℑzᵃᵃᶠ(i, j, kz + 1, grid, temperature)
        workspace.pressure_interfaces[kt] = ℑzᵃᵃᶠ(i, j, kz + 1, grid, pressure)
    end

    workspace.pressure_interfaces[nlayers + 1] = ℑzᵃᵃᶠ(i, j, 1, grid, pressure)
    workspace.temperature_interfaces[nlayers + 1] = surface_value(rtm.surface_properties.surface_temperature, i, j)

    for kt in 1:nlayers
        kz = nlayers - kt + 1
        fill_column_gases!(workspace,
                           rtm.atmospheric_state.gas_optics,
                           qᵛ,
                           rtm.atmospheric_state.gas_values,
                           grid,
                           i,
                           j,
                           kt,
                           kz)
    end
    return workspace.atmosphere
end

function radiative_heating_column_fluxes!(workspace::RadiativeHeatingWorkspace, rtm, i = 1, j = 1)
    Lightflux.optical_properties!(workspace.longwave_optics,
                                              workspace.shortwave_optics,
                                              rtm.atmospheric_state.gas_optics,
                                              workspace.atmosphere)
    surface_temperature = workspace.temperature_interfaces[end]
    surface_longwave_up = rtm.surface_properties.surface_emissivity *
                          eltype(workspace.fluxes)(5.670374419e-8) * surface_temperature^4
    longwave_boundary = Lightflux.LongwaveBoundaryConditions(
        surface_longwave_up = surface_longwave_up,
        toa_longwave_down = zero(surface_longwave_up),
    )
    shortwave_boundary = Lightflux.ShortwaveBoundaryConditions(
        toa_shortwave_down = rtm.solar_constant,
        surface_albedo = surface_value(rtm.surface_properties.direct_surface_albedo, i, j),
    )
    Lightflux.radiative_fluxes!(workspace.fluxes,
                                            rtm.longwave_solver,
                                            workspace.longwave_optics,
                                            workspace.atmosphere,
                                            longwave_boundary)
    Lightflux.radiative_fluxes!(workspace.fluxes,
                                            rtm.shortwave_solver,
                                            workspace.shortwave_optics,
                                            workspace.atmosphere,
                                            shortwave_boundary)
    return workspace.fluxes
end

function column_heating_rates!(workspace::RadiativeHeatingWorkspace, constants::ThermodynamicConstants)
    Lightflux.heating_rates!(workspace.heating_rates,
                                         workspace.fluxes,
                                         workspace.atmosphere;
                                         gravity = constants.gravitational_acceleration,
                                         heat_capacity = constants.dry_air.heat_capacity)
    return workspace.heating_rates
end

function write_column_fluxes!(rtm, workspace::RadiativeHeatingWorkspace, grid, i, j)
    nlayers = size(grid, 3)
    for kt in 1:(nlayers + 1)
        kz = nlayers - kt + 2
        rtm.upwelling_longwave_flux[i, j, kz] = workspace.fluxes.longwave_up[kt]
        rtm.downwelling_longwave_flux[i, j, kz] = -workspace.fluxes.longwave_down[kt]
        rtm.downwelling_shortwave_flux[i, j, kz] = -workspace.fluxes.shortwave_down[kt]
    end
    return nothing
end

function write_flux_divergence!(rtm, grid, i, j)
    for k in 1:size(grid, 3)
        net_bottom = rtm.upwelling_longwave_flux[i, j, k] +
                     rtm.downwelling_longwave_flux[i, j, k] +
                     rtm.downwelling_shortwave_flux[i, j, k]
        net_top = rtm.upwelling_longwave_flux[i, j, k + 1] +
                  rtm.downwelling_longwave_flux[i, j, k + 1] +
                  rtm.downwelling_shortwave_flux[i, j, k + 1]
        rtm.flux_divergence[i, j, k] = -(net_top - net_bottom) / Δzᶜᶜᶜ(i, j, k, grid)
    end
    return nothing
end

function AtmosphereModels._update_radiation!(rtm::RadiativeTransferModel{<:Any, <:Any, <:Any, <:Any, Nothing, <:RadiativeHeatingAtmosphericState}, model)
    grid = model.grid
    if !(architecture(grid) isa CPU)
        return update_radiation_device!(rtm, model)
    end

    if rtm.atmospheric_state.gas_optics isa Lightflux.EcCKDTabulatedGasOpticsModel &&
       rtm.atmospheric_state.device_workspace !== nothing
        return update_tabulated_ecckd_radiation_device!(
            rtm,
            model,
            rtm.atmospheric_state.gas_optics,
        )
    end

    workspace = rtm.atmospheric_state.workspace
    Nx, Ny, _ = size(grid)
    for j in 1:Ny, i in 1:Nx
        fill_column_state!(workspace, rtm, model, i, j)
        radiative_heating_column_fluxes!(workspace, rtm, i, j)
        write_column_fluxes!(rtm, workspace, grid, i, j)
        write_flux_divergence!(rtm, grid, i, j)
    end
    return nothing
end

function update_radiation_device!(rtm, model)
    gas_optics = rtm.atmospheric_state.gas_optics
    if gas_optics isa Lightflux.EcCKDTabulatedGasOpticsModel
        return update_tabulated_ecckd_radiation_device!(rtm, model, gas_optics)
    end
    gas_optics isa Lightflux.EcCKDGasOpticsModel ||
        throw(ArgumentError("BreezeLightfluxExt device path currently supports EcCKDGasOpticsModel and EcCKDTabulatedGasOpticsModel"))
    Lightflux.gas_names(gas_optics) == (:h2o, :co2) ||
        throw(ArgumentError("BreezeLightfluxExt fixed-coefficient device path currently supports gas_names = (:h2o, :co2)"))

    grid = model.grid
    arch = architecture(grid)
    pressure = model.dynamics.reference_state.pressure
    temperature = model.temperature
    qᵛ = specific_humidity(model)
    co2 = specified_gas_value(rtm.atmospheric_state.gas_values, :co2, eltype(grid))
    surface_temperature = rtm.surface_properties.surface_temperature
    surface_albedo = rtm.surface_properties.direct_surface_albedo
    surface_emissivity = rtm.surface_properties.surface_emissivity
    launch!(arch, grid, :xy, _radiative_heating_fixed_ecckd_column!,
            rtm.upwelling_longwave_flux,
            rtm.downwelling_longwave_flux,
            rtm.downwelling_shortwave_flux,
            rtm.flux_divergence,
            grid,
            pressure,
            temperature,
            qᵛ,
            gas_optics.longwave_absorption,
            gas_optics.shortwave_absorption,
            gas_optics.longwave_weights,
            gas_optics.shortwave_weights,
            gas_optics.longwave_source_scale,
            rtm.solar_constant,
            surface_temperature,
            surface_albedo,
            surface_emissivity,
            co2)
    return nothing
end

function update_tabulated_ecckd_radiation_device!(rtm, model, gas_optics)
    workspace = rtm.atmospheric_state.device_workspace
    workspace === nothing &&
        throw(ArgumentError("BreezeLightfluxExt tabulated ecCKD device path requires a preallocated device workspace"))

    grid = model.grid
    pressure = model.dynamics.reference_state.pressure
    temperature = model.temperature
    qᵛ = specific_humidity(model)
    gas_values = rtm.atmospheric_state.gas_values
    FT = eltype(grid)

    tabulated_ecckd_device_column_state!(
        workspace,
        grid,
        pressure,
        temperature,
        qᵛ;
        co2 = specified_gas_value(gas_values, :co2, FT),
        o3 = specified_gas_value(gas_values, :o3, FT),
        ch4 = specified_gas_value(gas_values, :ch4, FT),
        n2o = specified_gas_value(gas_values, :n2o, FT),
        cfc11 = specified_gas_value(gas_values, :cfc11, FT),
        cfc12 = specified_gas_value(gas_values, :cfc12, FT),
    )
    tabulated_ecckd_interpolation_state!(
        workspace.interpolation,
        grid,
        workspace.pressure_layers,
        workspace.temperature_layers,
        workspace.column_amounts,
        gas_optics.pressure_grid,
        gas_optics.temperature_grid,
        gas_optics.h2o_mole_fraction_grid,
    )
    tabulated_ecckd_streaming_radiation!(
        rtm.upwelling_longwave_flux,
        rtm.downwelling_longwave_flux,
        rtm.downwelling_shortwave_flux,
        rtm.flux_divergence,
        grid,
        workspace,
        gas_optics;
        solar_constant = rtm.solar_constant,
        surface_temperature = rtm.surface_properties.surface_temperature,
        surface_albedo = rtm.surface_properties.direct_surface_albedo,
        surface_emissivity = rtm.surface_properties.surface_emissivity,
    )
    return nothing
end

@inline _field_or_number(x::Number, i, j) = x
@inline _field_or_number(x, i, j) = x[i, j, 1]

@inline function _fixed_tau(absorption, ig, h2o, co2)
    return absorption[ig, 1] * h2o + absorption[ig, 2] * co2
end

@inline function _layer_h2o(qᵛ, i, j, k)
    q = qᵛ[i, j, k]
    return max(q, zero(q))
end

@kernel function _radiative_heating_fixed_ecckd_column!(lw_up_field,
                                                        lw_down_field,
                                                        sw_down_field,
                                                        flux_divergence,
                                                        grid,
                                                        pressure,
                                                        temperature,
                                                        qᵛ,
                                                        lw_absorption,
                                                        sw_absorption,
                                                        lw_weights,
                                                        sw_weights,
                                                        lw_source_scale,
                                                        solar_constant,
                                                        surface_temperature,
                                                        surface_albedo,
                                                        surface_emissivity,
                                                        co2)
    i, j = @index(Global, NTuple)

    Nz = size(grid, 3)
    FT = eltype(grid)
    σ = FT(5.670374419e-8)
    T_surface = _field_or_number(surface_temperature, i, j)
    α_surface = _field_or_number(surface_albedo, i, j)
    ε_surface = FT(surface_emissivity)
    surface_longwave_up = ε_surface * σ * T_surface^4

    for kz in 1:(Nz + 1)
        lw_up_field[i, j, kz] = zero(FT)
        lw_down_field[i, j, kz] = zero(FT)
        sw_down_field[i, j, kz] = zero(FT)
    end

    for ig in axes(lw_absorption, 1)
        up = surface_longwave_up
        lw_up_field[i, j, 1] += lw_weights[ig] * up
        for k in 1:Nz
            h2o = _layer_h2o(qᵛ, i, j, k)
            τ = _fixed_tau(lw_absorption, ig, h2o, co2)
            tr = exp(-τ)
            T_layer = temperature[i, j, k]
            source = lw_source_scale[ig] * σ * T_layer^4
            up = up * tr + source * (one(FT) - tr)
            lw_up_field[i, j, k + 1] += lw_weights[ig] * up
        end

        down = zero(FT)
        lw_down_field[i, j, Nz + 1] += zero(FT)
        for k in Nz:-1:1
            h2o = _layer_h2o(qᵛ, i, j, k)
            τ = _fixed_tau(lw_absorption, ig, h2o, co2)
            tr = exp(-τ)
            T_layer = temperature[i, j, k]
            source = lw_source_scale[ig] * σ * T_layer^4
            down = down * tr + source * (one(FT) - tr)
            lw_down_field[i, j, k] -= lw_weights[ig] * down
        end
    end

    for ig in axes(sw_absorption, 1)
        down = FT(solar_constant)
        sw_down_field[i, j, Nz + 1] -= sw_weights[ig] * down
        for k in Nz:-1:1
            h2o = _layer_h2o(qᵛ, i, j, k)
            τ = _fixed_tau(sw_absorption, ig, h2o, co2)
            down *= exp(-τ)
            sw_down_field[i, j, k] -= sw_weights[ig] * down
        end

        up = α_surface * down
        for k in 1:Nz
            h2o = _layer_h2o(qᵛ, i, j, k)
            τ = _fixed_tau(sw_absorption, ig, h2o, co2)
            up *= exp(-τ)
        end
    end

    for k in 1:Nz
        net_bottom = lw_up_field[i, j, k] +
                     lw_down_field[i, j, k] +
                     sw_down_field[i, j, k]
        net_top = lw_up_field[i, j, k + 1] +
                  lw_down_field[i, j, k + 1] +
                  sw_down_field[i, j, k + 1]
        flux_divergence[i, j, k] = -(net_top - net_bottom) / Δzᶜᶜᶜ(i, j, k, grid)
    end
end

end
