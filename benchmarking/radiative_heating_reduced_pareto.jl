using Dates
using JSON
using Printf

const DEFAULT_REDUCED_PARETO_DIR =
    joinpath(@__DIR__, "results", "reduced_pareto")
const DEFAULT_REDUCED_ACCURACY_JSON =
    joinpath(@__DIR__, "results", "reduced_accuracy",
             "radiative_heating_reduced_accuracy_latest.json")
const TABLE_REFINED_METHOD =
    "weighted greedy 16 shortwave g-point subset with latest preflight-optimized weights, coefficient scales, and pressure-band table moves"

function read_json_if_exists(path)
    isfile(path) || return nothing
    return JSON.parsefile(path)
end

function model_accuracy_by_method(path)
    result = read_json_if_exists(path)
    result === nothing && return Dict{String, Any}()
    rows = Dict{String, Any}()
    for model in get(result, "models", Any[])
        model isa AbstractDict || continue
        method = get(model, "reduction_method", "")
        isempty(method) && continue
        cases = get(model, "cases", Any[])
        worst_toa = isempty(cases) ? nothing :
            maximum(case -> get(case, "toa_forcing_max_abs", 0.0), cases)
        worst_surface = isempty(cases) ? nothing :
            maximum(case -> get(case, "surface_forcing_max_abs", 0.0), cases)
        rows[method] = Dict(
            "passed_hard_thresholds" => get(model, "passed_hard_thresholds", false),
            "worst_toa_forcing_error_w_m2" => worst_toa,
            "worst_surface_forcing_error_w_m2" => worst_surface,
        )
    end
    return rows
end

function benchmark_row(path; label, reduction_method = nothing)
    result = read_json_if_exists(path)
    result === nothing && return nothing
    grid = get(result, "grid", Dict{String, Any}())
    return Dict(
        "label" => label,
        "source" => relpath(path, @__DIR__),
        "backend" => get(result, "backend", "missing"),
        "columns" => get(grid, "columns", nothing),
        "nx" => get(grid, "nx", nothing),
        "ny" => get(grid, "ny", nothing),
        "nz" => get(grid, "nz", nothing),
        "ng_lw" => occursin("32_lw", get(result, "gas_model_kind", "")) ? 32 :
            occursin("16_lw", get(result, "gas_model_kind", "")) ? 16 : nothing,
        "ng_sw" => occursin("16_sw", get(result, "gas_model_kind", "")) ? 16 :
            occursin("32_sw", get(result, "gas_model_kind", "")) ? 32 : nothing,
        "gas_model_source" => get(result, "gas_model_source", "missing"),
        "gas_model_kind" => get(result, "gas_model_kind", "missing"),
        "gas_model_accuracy_status" => get(result, "gas_model_accuracy_status", "missing"),
        "gas_model_device_support_status" =>
            get(result, "gas_model_device_support_status", "missing"),
        "radiative_heating_runtime_supported" =>
            get(result, "radiative_heating_runtime_supported", false),
        "rrtmgp_runtime_supported" => get(result, "rrtmgp_runtime_supported", false),
        "radiative_heating_update_median_ms" =>
            get(result, "radiative_heating_update_median_ms", nothing),
        "rrtmgp_update_median_ms" => get(result, "rrtmgp_update_median_ms", nothing),
        "speedup_vs_rrtmgp" => get(result, "radiation_update_speedup", nothing),
        "hot_path_allocations" => get(result, "radiation_update_allocations", nothing),
        "rrtmgp_hot_path_allocations" =>
            get(result, "rrtmgp_radiation_update_allocations", nothing),
        "nsys_report" => get(result, "nsys_report", ""),
        "ncu_report" => get(result, "ncu_report", ""),
        "benchmark_status" => get(result, "status", "missing"),
        "final_4x_claim_supported" => get(result, "final_4x_claim_supported", false),
        "reduction_method" => reduction_method,
    )
end

function enriched_rows(rows, accuracy)
    for row in rows
        method = get(row, "reduction_method", nothing)
        method === nothing && continue
        metrics = get(accuracy, method, nothing)
        metrics === nothing && continue
        row["passed_hard_thresholds"] = metrics["passed_hard_thresholds"]
        row["worst_toa_forcing_error_w_m2"] = metrics["worst_toa_forcing_error_w_m2"]
        row["worst_surface_forcing_error_w_m2"] =
            metrics["worst_surface_forcing_error_w_m2"]
    end
    return rows
end

function reduced_pareto_result()
    output_dir = get(ENV, "RADIATIVE_HEATING_REDUCED_PARETO_DIR",
                     DEFAULT_REDUCED_PARETO_DIR)
    accuracy_json = get(ENV, "RADIATIVE_HEATING_REDUCED_ACCURACY_JSON",
                        DEFAULT_REDUCED_ACCURACY_JSON)
    accuracy = model_accuracy_by_method(accuracy_json)
    benchmark_specs = [
        (
            label = "synthetic 16x16 H100 term-count scaffold",
            path = joinpath(output_dir, "ng_lw_16_ng_sw_16",
                            "radiative_heating_rcemip_latest.json"),
            method = nothing,
        ),
        (
            label = "synthetic 32x16 H100 term-count scaffold",
            path = joinpath(output_dir, "ng_lw_32_ng_sw_16",
                            "radiative_heating_rcemip_latest.json"),
            method = nothing,
        ),
        (
            label = "official 32x16 preflight table-refined H100 smoke",
            path = joinpath(@__DIR__, "results",
                            "reduced_preflight_table_refined_h100_smoke",
                            "radiative_heating_rcemip_latest.json"),
            method = TABLE_REFINED_METHOD,
        ),
    ]
    rows = Dict{String, Any}[]
    for spec in benchmark_specs
        row = benchmark_row(spec.path; label = spec.label, reduction_method = spec.method)
        row === nothing || push!(rows, row)
    end
    enriched_rows(rows, accuracy)
    accuracy_status = any(row -> get(row, "passed_hard_thresholds", false), rows) ?
        "contains_passing_reduced_model" : "failed_threshold"
    runtime_status = all(row -> get(row, "radiative_heating_runtime_supported", false),
                         rows) ? "runtime_available" : "runtime_incomplete"
    return Dict(
        "case" => "radiative_heating_reduced_pareto",
        "timestamp_utc" => string(now(UTC)),
        "status" => runtime_status == "runtime_available" &&
            accuracy_status == "failed_threshold" ?
            "h100_runtime_accuracy_failed_threshold" :
            "$(runtime_status)_$(accuracy_status)",
        "accuracy_status" => accuracy_status,
        "workload" => "RCEMIP-style Breeze radiation update without expensive spinup",
        "accuracy_source" => relpath(accuracy_json, @__DIR__),
        "models" => rows,
    )
end

function write_reduced_pareto(result)
    output_dir = get(ENV, "RADIATIVE_HEATING_REDUCED_PARETO_DIR",
                     DEFAULT_REDUCED_PARETO_DIR)
    mkpath(output_dir)
    json_path = joinpath(output_dir, "radiative_heating_reduced_pareto_latest.json")
    open(json_path, "w") do io
        JSON.print(io, result, 2)
        println(io)
    end

    md_path = joinpath(output_dir, "radiative_heating_reduced_pareto_latest.md")
    open(md_path, "w") do io
        println(io, "# Reduced ecCKD Pareto Evidence")
        println(io)
        println(io, "Status: ", result["status"])
        println(io)
        println(io, "| Label | backend | grid | RH ms | RRTMGP ms | speedup | accuracy | TOA W m^-2 | surface W m^-2 | support |")
        println(io, "|---|---|---:|---:|---:|---:|---|---:|---:|---|")
        for row in result["models"]
            grid = "$(row["nx"])x$(row["ny"])x$(row["nz"])"
            rh = row["radiative_heating_update_median_ms"]
            rrtmgp = row["rrtmgp_update_median_ms"]
            speedup = row["speedup_vs_rrtmgp"]
            toa = get(row, "worst_toa_forcing_error_w_m2", nothing)
            surface = get(row, "worst_surface_forcing_error_w_m2", nothing)
            println(io, "| ", row["label"], " | ", row["backend"], " | ", grid, " | ",
                    rh === nothing ? "n/a" : @sprintf("%.6g", rh), " | ",
                    rrtmgp === nothing ? "n/a" : @sprintf("%.6g", rrtmgp), " | ",
                    speedup === nothing ? "n/a" : @sprintf("%.6g", speedup), " | ",
                    get(row, "gas_model_accuracy_status", "missing"), " | ",
                    toa === nothing ? "n/a" : @sprintf("%.12g", toa), " | ",
                    surface === nothing ? "n/a" : @sprintf("%.12g", surface), " | ",
                    row["gas_model_device_support_status"], " |")
        end
        println(io)
        println(io, "Accuracy source: `", result["accuracy_source"], "`.")
        println(io, "The official table-refined row is smoke/runtime-path evidence until the reduced hard thresholds pass.")
    end
    println(json_path)
    println(md_path)
    return json_path, md_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_reduced_pareto(reduced_pareto_result())
end
