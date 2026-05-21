using Lightflux
using Breeze
using CUDA
using ClimaComms
using Dates
using JSON
using NCDatasets
using Oceananigans
using Printf
using RRTMGP
using Statistics

using Breeze.AtmosphereModels: update_radiation!
using Oceananigans.Architectures: CPU, GPU

const DAY = 24 * 60 * 60
const DEFAULT_ABR_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", "Lightflux.jl"))
const OFFICIAL_ECCKD_GASES = (:composite, :h2o, :o3, :co2, :ch4, :n2o, :cfc11, :cfc12)
const REDUCED_PREFLIGHT_SW_16_INDICES = [1, 4, 9, 10, 12, 13, 14, 16, 21, 22, 25, 27, 28, 30, 31, 32]

envint(name, default) = parse(Int, get(ENV, name, string(default)))
envfloat(name, default) = parse(Float64, get(ENV, name, string(default)))
benchmark_status() = get(ENV, "RADIATIVE_HEATING_BENCHMARK_STATUS", "scaffold_not_final_4x_evidence")

function backend_name()
    requested = get(ENV, "RADIATIVE_HEATING_BACKEND", "CPU")
    if uppercase(requested) == "H100"
        CUDA.functional() || error("RADIATIVE_HEATING_BACKEND=H100 was requested but CUDA is not functional")
        return "H100", GPU()
    end
    return "CPU", CPU()
end

function rcemip_grid(arch, FT, Nx, Ny, Nz)
    RectilinearGrid(arch, FT;
                    size = (Nx, Ny, Nz),
                    x = (0, 96e3),
                    y = (-12, 12),
                    z = (0, 30e3),
                    topology = (Periodic, Periodic, Bounded))
end

function make_gas_optics(FT; ng_lw = 32, ng_sw = 16)
    lw = zeros(FT, ng_lw, 2)
    sw = zeros(FT, ng_sw, 2)
    for ig in 1:ng_lw
        lw[ig, 1] = FT(16 + 0.25ig)
        lw[ig, 2] = FT(18 + 0.1ig)
    end
    for ig in 1:ng_sw
        sw[ig, 1] = FT(220 + 0.8ig)
        sw[ig, 2] = FT(6 + 0.2ig)
    end
    return EcCKDGasOpticsModel(
        gas_names = (:h2o, :co2),
        longwave_absorption = lw,
        shortwave_absorption = sw,
        longwave_weights = fill(inv(FT(ng_lw)), ng_lw),
        shortwave_weights = fill(inv(FT(ng_sw)), ng_sw),
        longwave_source_scale = fill(one(FT), ng_lw),
    )
end

function abr_root()
    return normpath(get(ENV, "RADIATIVE_HEATING_ABR_ROOT", DEFAULT_ABR_ROOT))
end

function abr_path(parts...)
    return joinpath(abr_root(), parts...)
end

function accuracy_gate_status()
    path = abr_path("validation", "results", "ecrad_accuracy_gate.json")
    isfile(path) || return "missing_accuracy_gate"
    result = JSON.parsefile(path)
    return get(result, "status", "missing_status")
end

function official_ecckd_gas_optics()
    lw_path = get(ENV, "RADIATIVE_HEATING_ECCKD_LW_PATH",
                  abr_path("validation", "external", "ecrad", "data",
                           "ecckd-1.0_lw_climate_fsck-32b_ckd-definition.nc"))
    sw_path = get(ENV, "RADIATIVE_HEATING_ECCKD_SW_PATH",
                  abr_path("validation", "external", "ecrad", "data",
                           "ecckd-1.4_sw_climate_rgb-32b_ckd-definition.nc"))
    return read_ecckd_tabulated_gas_optics(lw_path, sw_path;
                                           gas_names = OFFICIAL_ECCKD_GASES,
                                           h2o_mole_fraction = envfloat("RADIATIVE_HEATING_ECCKD_H2O_MOLE_FRACTION", 0.005))
end

function reduced_preflight_json()
    return abr_path("validation", "results", "reduced_ecckd_optimization_preflight.json")
end

function reduced_targeted_entry_json()
    return abr_path("validation", "results", "reduced_ecckd_targeted_entry_refinement.json")
end

function reduced_global_entry_json()
    return abr_path("validation", "results", "reduced_ecckd_global_entry_refinement.json")
end

function reduced_global_block_json()
    return abr_path("validation", "results", "reduced_ecckd_global_block_refinement.json")
end

function reduced_global_block_linearized_json()
    return abr_path("validation", "results", "reduced_ecckd_global_block_linearized_refit.json")
end

function reduced_exact_weight_refit_json()
    return abr_path("validation", "results", "reduced_ecckd_exact_weight_refit.json")
end

function reduced_accuracy_json()
    return abr_path("validation", "results", "reduced_ecckd_accuracy.json")
end

function reduced_preflight_parameters(result)
    for key in ("post_coefficient_weight_refinement",
                "post_joint_coordinate_refinement",
                "coefficient_joint_direction_scan",
                "greedy_coordinate_descent")
        section = get(result, key, nothing)
        section isa AbstractDict || continue
        parameters = get(section, "final_parameters", nothing)
        parameters isa Vector || continue
        length(parameters) == 3length(REDUCED_PREFLIGHT_SW_16_INDICES) || continue
        return Float64.(parameters)
    end
    error("could not find reduced preflight final_parameters in $(reduced_preflight_json())")
end

function reduced_preflight_table_moves(result)
    refinement = get(result, "pressure_band_table_refinement", Dict{String, Any}())
    refinement isa AbstractDict || return Any[]
    moves = get(refinement, "accepted_moves", Any[])
    return moves isa Vector ? moves : Any[]
end

function reduced_preflight_active_table_entry_moves(result)
    refinement = get(result, "active_table_entry_refinement", Dict{String, Any}())
    refinement isa AbstractDict || return Any[]
    moves = get(refinement, "accepted_moves", Any[])
    return moves isa Vector ? moves : Any[]
end

function reduced_targeted_active_table_entry_moves()
    path = reduced_targeted_entry_json()
    isfile(path) || return Any[], Inf
    result = JSON.parsefile(path)
    refinement = get(result, "refinement", Dict{String, Any}())
    refinement isa AbstractDict || return Any[], Inf
    moves = get(refinement, "accepted_moves", Any[])
    objective = Float64(get(refinement, "final_objective", Inf))
    return moves isa Vector ? moves : Any[], objective
end

function reduced_global_active_table_entry_moves()
    path = reduced_global_entry_json()
    isfile(path) || return Any[], Inf
    result = JSON.parsefile(path)
    refinement = get(result, "refinement", Dict{String, Any}())
    refinement isa AbstractDict || return Any[], Inf
    moves = get(refinement, "all_active_moves", Any[])
    objective = Float64(get(refinement, "final_objective", Inf))
    return moves isa Vector ? moves : Any[], objective
end

function reduced_global_block_active_table_entry_moves()
    path = reduced_global_block_json()
    isfile(path) || return Any[], Inf
    result = JSON.parsefile(path)
    refinement = get(result, "refinement", Dict{String, Any}())
    refinement isa AbstractDict || return Any[], Inf
    moves = get(refinement, "all_active_moves", Any[])
    objective = Float64(get(refinement, "final_objective", Inf))
    return moves isa Vector ? moves : Any[], objective
end

function reduced_global_block_linearized_active_table_entry_moves()
    path = reduced_global_block_linearized_json()
    isfile(path) || return Any[], Inf
    result = JSON.parsefile(path)
    moves = get(result, "all_active_moves", Any[])
    objective = Float64(get(result, "final_objective", Inf))
    return moves isa Vector ? moves : Any[], objective
end

function reduced_exact_weight_refit_weights()
    path = reduced_exact_weight_refit_json()
    isfile(path) || return nothing
    result = JSON.parsefile(path)
    weights = get(result, "final_weights", nothing)
    weights isa Vector || return nothing
    length(weights) == length(REDUCED_PREFLIGHT_SW_16_INDICES) || return nothing
    return Float64.(weights)
end

function reduced_best_active_table_entry_moves(preflight)
    refinement = get(preflight, "active_table_entry_refinement", Dict{String, Any}())
    preflight_objective = refinement isa AbstractDict ?
        Float64(get(refinement, "final_objective", Inf)) : Inf
    best_moves = reduced_preflight_active_table_entry_moves(preflight)
    best_objective = preflight_objective
    targeted_moves, targeted_objective = reduced_targeted_active_table_entry_moves()
    if !isempty(targeted_moves) && targeted_objective < best_objective
        best_moves = targeted_moves
        best_objective = targeted_objective
    end
    global_moves, global_objective = reduced_global_active_table_entry_moves()
    if !isempty(global_moves) && global_objective < best_objective
        best_moves = global_moves
        best_objective = global_objective
    end
    global_block_moves, global_block_objective =
        reduced_global_block_active_table_entry_moves()
    if !isempty(global_block_moves) && global_block_objective < best_objective
        best_moves = global_block_moves
        best_objective = global_block_objective
    end
    global_block_linearized_moves, global_block_linearized_objective =
        reduced_global_block_linearized_active_table_entry_moves()
    if !isempty(global_block_linearized_moves) &&
       global_block_linearized_objective < best_objective
        best_moves = global_block_linearized_moves
    end
    return best_moves
end

function softmax(values)
    shifted = values .- maximum(values)
    exps = exp.(shifted)
    return exps ./ sum(exps)
end

function apply_reduced_preflight_parameters!(model, parameters)
    ng = length(REDUCED_PREFLIGHT_SW_16_INDICES)
    model.shortwave_weights .= softmax(parameters[1:ng])
    absorption_scales = exp.(clamp.(parameters[(ng + 1):(2ng)], -5.0, 5.0))
    rayleigh_scales = exp.(clamp.(parameters[(2ng + 1):(3ng)], -5.0, 5.0))
    for ig in 1:ng
        model.shortwave_absorption[ig, :, :, :] .*= absorption_scales[ig]
        if length(model.shortwave_h2o_absorption) != 0
            model.shortwave_h2o_absorption[ig, :, :, :] .*= absorption_scales[ig]
        end
        model.shortwave_rayleigh_molar_scattering[ig] *= rayleigh_scales[ig]
    end
    return model
end

function apply_reduced_preflight_table_moves!(model, moves)
    for move in moves
        component = move["component"]
        local_gpoint_index = Int(move["local_gpoint_index"])
        pressure_range = Int(move["pressure_index_start"]):Int(move["pressure_index_end"])
        scale = exp(clamp(Float64(move["log_scale"]), -5.0, 5.0))
        if component == "static_absorption"
            model.shortwave_absorption[local_gpoint_index, :, pressure_range, :] .*= scale
        elseif component == "dynamic_h2o"
            if length(model.shortwave_h2o_absorption) != 0
                model.shortwave_h2o_absorption[local_gpoint_index, pressure_range, :, :] .*= scale
            end
        else
            error("unsupported reduced preflight table move component $component")
        end
    end
    return model
end

function apply_reduced_preflight_active_table_entry_moves!(model, moves)
    for move in moves
        component = move["component"]
        local_gpoint_index = Int(move["local_gpoint_index"])
        pressure_index = Int(move["pressure_index"])
        temperature_index = Int(move["temperature_index"])
        scale = exp(clamp(Float64(move["log_scale"]), -5.0, 5.0))
        if component == "static_absorption"
            gas_index = Int(move["gas_index"])
            model.shortwave_absorption[
                local_gpoint_index,
                gas_index,
                pressure_index,
                temperature_index,
            ] *= scale
        elseif component == "dynamic_h2o"
            if length(model.shortwave_h2o_absorption) != 0
                h2o_index = Int(move["h2o_index"])
                model.shortwave_h2o_absorption[
                    local_gpoint_index,
                    pressure_index,
                    temperature_index,
                    h2o_index,
                ] *= scale
            end
        else
            error("unsupported reduced preflight active table-entry move component $component")
        end
    end
    return model
end

function table_refined_reduced_accuracy_status()
    path = reduced_accuracy_json()
    isfile(path) || return "missing_reduced_accuracy"
    result = JSON.parsefile(path)
    models = get(result, "models", Any[])
    target = "weighted greedy 16 shortwave g-point subset with latest preflight-optimized weights, coefficient scales, and pressure-band table moves"
    for model in models
        model isa AbstractDict || continue
        get(model, "reduction_method", "") == target || continue
        return get(model, "passed_hard_thresholds", false) ? "passed" : "failed_threshold"
    end
    return "missing_table_refined_reduced_accuracy"
end

function reduced_preflight_table_refined_gas_optics()
    full = official_ecckd_gas_optics()
    preflight = JSON.parsefile(reduced_preflight_json())
    lw_indices = collect(1:size(full.longwave_absorption, 1))
    sw_indices = REDUCED_PREFLIGHT_SW_16_INDICES
    source_table = full.longwave_source_table === nothing ||
        length(full.longwave_source_table) == 0 ?
        full.longwave_source_table :
        full.longwave_source_table[lw_indices, :]
    reduced = EcCKDTabulatedGasOpticsModel(
        gas_names = Lightflux.gas_names(full),
        pressure_grid = full.pressure_grid,
        temperature_grid = full.temperature_grid,
        h2o_mole_fraction_grid = full.h2o_mole_fraction_grid,
        gas_reference_mole_fractions = full.gas_reference_mole_fractions,
        longwave_absorption = copy(full.longwave_absorption[lw_indices, :, :, :]),
        shortwave_absorption = copy(full.shortwave_absorption[sw_indices, :, :, :]),
        longwave_h2o_absorption = length(full.longwave_h2o_absorption) == 0 ?
            full.longwave_h2o_absorption : copy(full.longwave_h2o_absorption[lw_indices, :, :, :]),
        shortwave_h2o_absorption = length(full.shortwave_h2o_absorption) == 0 ?
            full.shortwave_h2o_absorption : copy(full.shortwave_h2o_absorption[sw_indices, :, :, :]),
        shortwave_rayleigh_molar_scattering =
            copy(full.shortwave_rayleigh_molar_scattering[sw_indices]),
        longwave_source_scale = copy(full.longwave_source_scale[lw_indices]),
        longwave_source_temperature_grid = full.longwave_source_temperature_grid,
        longwave_source_table = source_table,
        longwave_weights = full.longwave_weights[lw_indices] ./ sum(full.longwave_weights[lw_indices]),
        shortwave_weights = full.shortwave_weights[sw_indices] ./ sum(full.shortwave_weights[sw_indices]),
    )
    apply_reduced_preflight_parameters!(reduced, reduced_preflight_parameters(preflight))
    apply_reduced_preflight_table_moves!(reduced, reduced_preflight_table_moves(preflight))
    apply_reduced_preflight_active_table_entry_moves!(
        reduced,
        reduced_best_active_table_entry_moves(preflight),
    )
    refit_weights = reduced_exact_weight_refit_weights()
    refit_weights === nothing || (reduced.shortwave_weights .= refit_weights)
    return reduced
end

function benchmark_gas_model(FT)
    source = get(ENV, "RADIATIVE_HEATING_GAS_MODEL_SOURCE", "synthetic_fixed_coefficients")
    if source == "validated_ecCKD"
        gas_optics = official_ecckd_gas_optics()
        return (
            gas_optics = gas_optics,
            kind = "official_ecCKD_$(size(gas_optics.longwave_absorption, 1))_lw_$(size(gas_optics.shortwave_absorption, 1))_sw",
            source = "validated_ecCKD",
            accuracy_status = accuracy_gate_status(),
        )
    elseif source == "validated_ecCKD_reduced_preflight_table_refined"
        preflight = JSON.parsefile(reduced_preflight_json())
        gas_optics = reduced_preflight_table_refined_gas_optics()
        return (
            gas_optics = gas_optics,
            kind = "official_ecCKD_reduced_preflight_table_refined_$(size(gas_optics.longwave_absorption, 1))_lw_$(size(gas_optics.shortwave_absorption, 1))_sw",
            source = "validated_ecCKD_reduced_preflight_table_refined",
            accuracy_status = table_refined_reduced_accuracy_status(),
            preflight_pressure_move_count =
                length(reduced_preflight_table_moves(preflight)),
            preflight_active_table_entry_move_count =
                length(reduced_best_active_table_entry_moves(preflight)),
        )
    elseif source == "synthetic_fixed_coefficients"
        ng_lw = envint("RADIATIVE_HEATING_NG_LW", 32)
        ng_sw = envint("RADIATIVE_HEATING_NG_SW", 16)
        return (
            gas_optics = make_gas_optics(FT; ng_lw, ng_sw),
            kind = "fixed_ecCKD_$(ng_lw)_lw_$(ng_sw)_sw",
            source = "synthetic_fixed_coefficients",
            accuracy_status = "not_checked_scaffold",
        )
    end
    error("unsupported RADIATIVE_HEATING_GAS_MODEL_SOURCE=$source")
end

function benchmark_gas_values(gas_model)
    FT = eltype(gas_model.gas_optics)
    if gas_model.source in ("validated_ecCKD",
                            "validated_ecCKD_reduced_preflight_table_refined")
        return Dict{Symbol, FT}(
            :co2 => FT(envfloat("RADIATIVE_HEATING_CO2_VMR", 420.0e-6)),
            :o3 => FT(envfloat("RADIATIVE_HEATING_O3_VMR", 0.0)),
            :ch4 => FT(envfloat("RADIATIVE_HEATING_CH4_VMR", 1.8e-6)),
            :n2o => FT(envfloat("RADIATIVE_HEATING_N2O_VMR", 330.0e-9)),
            :cfc11 => FT(envfloat("RADIATIVE_HEATING_CFC11_VMR", 230.0e-12)),
            :cfc12 => FT(envfloat("RADIATIVE_HEATING_CFC12_VMR", 520.0e-12)),
        )
    end
    return Dict{Symbol, FT}(
        :co2 => FT(envfloat("RADIATIVE_HEATING_CO2_VMR", 400.0e-6)),
    )
end

function json_gas_values(gas_values)
    return Dict(String(key) => value for (key, value) in gas_values)
end

function gas_model_device_support(backend, gas_optics)
    ext = Base.get_extension(Breeze, :BreezeLightfluxExt)
    ext !== nothing ||
        error("BreezeLightfluxExt is not loaded; load Lightflux with Breeze before benchmarking")
    return Base.invokelatest(ext.radiative_heating_device_support, backend, gas_optics)
end

function build_model(radiation, grid, constants)
    reference_state = ReferenceState(grid, constants;
                                     surface_pressure = 101325,
                                     potential_temperature = 300)
    dynamics = AnelasticDynamics(reference_state)
    clock = Clock(time = DateTime(2024, 7, 15, 12, 0, 0))
    model = AtmosphereModel(grid; clock, dynamics,
                            formulation = :LiquidIcePotentialTemperature,
                            radiation)
    θ(x, y, z) = 300 + 18 * (z / 30e3)^1.25
    qᵗ(x, y, z) = 0.018 * exp(-z / 2400)
    set!(model; θ, qᵗ)
    return model
end

function median_seconds!(f, samples)
    times = Float64[]
    for _ in 1:samples
        GC.gc(false)
        push!(times, @elapsed f())
    end
    return median(times)
end

sync_backend(backend) = backend == "H100" ? CUDA.synchronize() : nothing

function final_acceptance(result)
    result["status"] == "final_4x_evidence" ||
        return false, "benchmark status is $(result["status"]), not final_4x_evidence"
    result["backend"] == "H100" || return false, "backend is $(result["backend"]), not H100"
    result["radiative_heating_runtime_supported"] == true || return false, "RadiativeHeating runtime did not run on H100"
    result["rrtmgp_runtime_supported"] == true || return false, "RRTMGP baseline did not run on H100"
    result["radiation_update_speedup"] >= 4 || return false, "radiation update speedup is below 4x"
    result["nsys_report"] != "" || return false, "missing Nsight Systems report"
    result["ncu_report"] != "" || return false, "missing Nsight Compute report"
    result["gas_model_source"] != "synthetic_fixed_coefficients" ||
        return false, "gas model source is synthetic fixed coefficients"
    result["gas_model_accuracy_status"] == "passed" ||
        return false, "gas model accuracy status is $(result["gas_model_accuracy_status"]), not passed"
    get(result, "gas_model_device_support_status", "missing") == "supported" ||
        return false, "gas model device support is $(get(result, "gas_model_device_support_status", "missing")): $(get(result, "gas_model_device_support_reason", ""))"
    return true, "none"
end

function main()
    FT = Float64
    Nx = envint("RADIATIVE_HEATING_NX", 16)
    Ny = envint("RADIATIVE_HEATING_NY", 16)
    Nz = envint("RADIATIVE_HEATING_NZ", 64)
    samples = envint("RADIATIVE_HEATING_SAMPLES", 3)
    output_dir = get(ENV, "RADIATIVE_HEATING_RCEMIP_DIR",
                     joinpath(@__DIR__, "results"))
    mkpath(output_dir)

    backend, arch = backend_name()
    grid = rcemip_grid(arch, FT, Nx, Ny, Nz)
    constants = Breeze.Thermodynamics.ThermodynamicConstants()

    rh_model = nothing
    rh_seconds = Inf
    rh_allocations = nothing
    rh_supported = true
    rh_error = ""
    gas_model_kind = ""
    gas_model_source = get(ENV, "RADIATIVE_HEATING_GAS_MODEL_SOURCE", "synthetic_fixed_coefficients")
    gas_model_accuracy_status = "not_loaded"
    gas_values = Dict{Symbol, FT}()
    preflight_pressure_move_count = nothing
    preflight_active_table_entry_move_count = nothing
    gas_model_device_support_status = "not_checked"
    gas_model_device_support_reason = "gas model was not loaded"
    gas_model_device_support_source = "not_checked"
    try
        gas_model = benchmark_gas_model(FT)
        gas_optics = gas_model.gas_optics
        gas_values = benchmark_gas_values(gas_model)
        gas_model_kind = gas_model.kind
        gas_model_source = gas_model.source
        gas_model_accuracy_status = gas_model.accuracy_status
        preflight_pressure_move_count =
            hasproperty(gas_model, :preflight_pressure_move_count) ?
            gas_model.preflight_pressure_move_count : nothing
        preflight_active_table_entry_move_count =
            hasproperty(gas_model, :preflight_active_table_entry_move_count) ?
            gas_model.preflight_active_table_entry_move_count : nothing
        device_support = gas_model_device_support(backend, gas_optics)
        gas_model_device_support_status = device_support.status
        gas_model_device_support_reason = device_support.reason
        gas_model_device_support_source = getproperty(device_support, :source)
        rh_radiation = Base.invokelatest(() ->
            RadiativeTransferModel(grid, RadiativeHeatingOptics(), constants;
                                   gas_optics,
                                   gas_values,
                                   surface_temperature = 300,
                                   surface_albedo = 0.07,
                                   surface_emissivity = 0.98,
                                   solar_constant = 551))
        rh_model = build_model(rh_radiation, grid, constants)
        Base.invokelatest(update_radiation!, rh_model.radiation, rh_model)
        sync_backend(backend)
        rh_allocations = backend == "CPU" ? @allocated(Base.invokelatest(update_radiation!, rh_model.radiation, rh_model)) : nothing
        rh_seconds = median_seconds!(samples) do
            Base.invokelatest(update_radiation!, rh_model.radiation, rh_model)
            sync_backend(backend)
        end
    catch err
        rh_supported = false
        rh_error = sprint(showerror, err)
    end

    rrtmgp_seconds = Inf
    rrtmgp_allocations = nothing
    rrtmgp_supported = true
    rrtmgp_error = ""
    try
        rrtmgp_radiation = Base.invokelatest(() ->
            RadiativeTransferModel(grid, ClearSkyOptics(), constants;
                                   surface_temperature = 300,
                                   surface_emissivity = 0.98,
                                   surface_albedo = 0.07,
                                   solar_constant = 551,
                                   coordinate = (0.0, 0.0)))
        rrtmgp_model = build_model(rrtmgp_radiation, grid, constants)
        Base.invokelatest(update_radiation!, rrtmgp_model.radiation, rrtmgp_model)
        sync_backend(backend)
        rrtmgp_allocations = backend == "CPU" ? @allocated(Base.invokelatest(update_radiation!, rrtmgp_model.radiation, rrtmgp_model)) : nothing
        rrtmgp_seconds = median_seconds!(samples) do
            Base.invokelatest(update_radiation!, rrtmgp_model.radiation, rrtmgp_model)
            sync_backend(backend)
        end
    catch err
        rrtmgp_supported = false
        rrtmgp_error = sprint(showerror, err)
    end

    speedup = isfinite(rh_seconds) && isfinite(rrtmgp_seconds) ? rrtmgp_seconds / rh_seconds : 0.0
    result = Dict(
        "case" => "radiative_heating_rcemip_benchmark",
        "status" => benchmark_status(),
        "timestamp_utc" => string(now(UTC)),
        "backend" => backend,
        "grid" => Dict("nx" => Nx, "ny" => Ny, "nz" => Nz, "columns" => Nx * Ny),
        "samples" => samples,
        "workload" => "RCEMIP-style nontrivial column ensemble without expensive spinup",
        "gas_model_kind" => gas_model_kind,
        "gas_model_source" => gas_model_source,
        "gas_model_accuracy_status" => gas_model_accuracy_status,
        "preflight_pressure_move_count" => preflight_pressure_move_count,
        "preflight_active_table_entry_move_count" =>
            preflight_active_table_entry_move_count,
        "gas_values" => json_gas_values(gas_values),
        "gas_model_device_support_status" => gas_model_device_support_status,
        "gas_model_device_support_reason" => gas_model_device_support_reason,
        "gas_model_device_support_source" => gas_model_device_support_source,
        "radiative_heating_runtime_supported" => rh_supported,
        "radiative_heating_error" => rh_error,
        "radiation_update_allocations" => rh_allocations,
        "rrtmgp_runtime_supported" => rrtmgp_supported,
        "rrtmgp_error" => rrtmgp_error,
        "rrtmgp_radiation_update_allocations" => rrtmgp_allocations,
        "radiative_heating_update_median_ms" => isfinite(rh_seconds) ? 1000rh_seconds : nothing,
        "rrtmgp_update_median_ms" => isfinite(rrtmgp_seconds) ? 1000rrtmgp_seconds : nothing,
        "radiation_update_speedup" => speedup,
        "nsys_report" => get(ENV, "RADIATIVE_HEATING_NSYS_REPORT", ""),
        "ncu_report" => get(ENV, "RADIATIVE_HEATING_NCU_REPORT", ""),
    )
    accepted, reason = final_acceptance(result)
    result["final_4x_claim_supported"] = accepted
    result["final_4x_blocking_reason"] = reason

    json_path = joinpath(output_dir, "radiative_heating_rcemip_latest.json")
    open(json_path, "w") do io
        JSON.print(io, result, 2)
        println(io)
    end

    md_path = joinpath(output_dir, "radiative_heating_rcemip_latest.md")
    open(md_path, "w") do io
        println(io, "# Radiative Heating RCEMIP-Style Benchmark")
        println(io)
        println(io, "- status: ", result["status"])
        println(io, "- backend: ", result["backend"])
        println(io, "- grid: ", Nx, " x ", Ny, " x ", Nz)
        println(io, "- gas model source: ", result["gas_model_source"])
        println(io, "- gas model kind: ", result["gas_model_kind"])
        println(io, "- gas model accuracy status: ", result["gas_model_accuracy_status"])
        println(io, "- gas values: ", result["gas_values"])
        println(io, "- gas model device support: ", result["gas_model_device_support_status"])
        println(io, "- gas model device support reason: ", result["gas_model_device_support_reason"])
        println(io, "- gas model device support source: ", result["gas_model_device_support_source"])
        println(io, "- RadiativeHeating supported: ", rh_supported)
        println(io, "- RRTMGP supported: ", rrtmgp_supported)
        println(io, "- speedup: ", @sprintf("%.3f", speedup), "x")
        println(io, "- final 4x claim supported: ", accepted)
        println(io, "- blocking reason: ", reason)
    end

    println(json_path)
    return accepted ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
