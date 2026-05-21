# Machine-readable gate for the terrain-following production validation plan.
#
# Usage:
#   julia --project=. validation_output/substepper/terrain_following_production_validation_gate.jl
#
# Outputs:
#   validation_output/substepper/terrain_following_production_validation_gate_report.csv
#   validation_output/substepper/terrain_following_production_validation_gate_report.md
#
# This gate is intentionally strict. Diagnostic and smoke artifacts are reported
# as evidence, but they cannot satisfy production-validation requirements.

using Printf

const ROOT = "validation_output/substepper"
const OUTPUT_PREFIX = joinpath(ROOT, "terrain_following_production_validation_gate_report")
const ONE_PERCENT = 0.01

struct Check
    case_name::String
    requirement::String
    status::String
    evidence::String
end

csv_escape(value) = "\"" * replace(string(value), "\"" => "\"\"") * "\""

function read_metrics_csv(path)
    isfile(path) || return nothing

    open(path) do io
        header = split(readline(io), ",")
        rows = Dict{String, String}[]

        for line in eachline(io)
            isempty(strip(line)) && continue
            columns = split(line, ",")
            row = Dict{String, String}()
            for index in eachindex(header)
                row[header[index]] = index <= length(columns) ? columns[index] : ""
            end
            push!(rows, row)
        end

        return (; header, rows)
    end
end

function parse_bool(value)
    lowercase(strip(value)) == "true"
end

function parse_float(row, key)
    haskey(row, key) || return NaN
    value = strip(row[key])
    isempty(value) && return NaN
    return parse(Float64, value)
end

function first_row(path)
    table = read_metrics_csv(path)
    isnothing(table) && return nothing
    isempty(table.rows) && return nothing
    return first(table.rows)
end

function first_existing(paths)
    for path in paths
        isfile(path) && return path
    end

    return first(paths)
end

function all_field_pass(row)
    isnothing(row) && return false
    fields = ("u", "w", "θ", "p")
    return all(field -> haskey(row, "$(field)_one_percent_pass") &&
                        parse_bool(row["$(field)_one_percent_pass"]),
               fields)
end

function schar_wave_pass(row)
    isnothing(row) && return false

    phase_error = parse_float(row, "w_phase_error_wavelengths")
    drag_key = haskey(row, "mountain_drag_best_convention_relative_error") ?
               "mountain_drag_best_convention_relative_error" :
               "mountain_drag_relative_error"
    drag_error = parse_float(row, drag_key)
    field_pass = all_field_pass(row)
    phase_pass = isfinite(phase_error) && phase_error <= ONE_PERCENT
    drag_pass = isfinite(drag_error) && drag_error <= ONE_PERCENT

    return field_pass && phase_pass && drag_pass
end

function schar_metric_evidence(row)
    isnothing(row) && return "metrics file missing or empty"

    entries = [
        "u_l2=$(row["u_relative_l2_error"])",
        "u_linf=$(row["u_relative_linf_error"])",
        "u_rmse=$(row["u_normalized_rmse"])",
        "u_amp=$(row["u_amplitude_error"])",
        "u_corr=$(row["u_pattern_correlation"])",
        "w_l2=$(row["w_relative_l2_error"])",
        "w_linf=$(row["w_relative_linf_error"])",
        "w_rmse=$(row["w_normalized_rmse"])",
        "w_amp=$(row["w_amplitude_error"])",
        "w_corr=$(row["w_pattern_correlation"])",
        "phase=$(row["w_phase_error_wavelengths"])",
        "drag=$(get(row, "mountain_drag_relative_error", "missing"))",
        "drag_best=$(get(row, "mountain_drag_best_convention_relative_error", "missing"))",
        "u_pass=$(row["u_one_percent_pass"])",
        "w_pass=$(row["w_one_percent_pass"])",
        "theta_pass=$(row["θ_one_percent_pass"])",
        "p_pass=$(row["p_one_percent_pass"])",
    ]

    return join(entries, "; ")
end

function projection_metrics_pass(path)
    table = read_metrics_csv(path)
    isnothing(table) && return false
    below_sponge_rows = filter(row -> get(row, "region", "") == "below_sponge",
                               table.rows)
    isempty(below_sponge_rows) && return false
    return all(row -> haskey(row, "one_percent_pass") &&
                      parse_bool(row["one_percent_pass"]),
               below_sponge_rows)
end

function projection_metric_evidence(path)
    table = read_metrics_csv(path)
    isnothing(table) && return "$path: metrics file missing or empty"

    entries = String[]
    for row in table.rows
        best_shift = get(row, "best_shift_cells", "not_computed")
        best_shift_projection_error =
            get(row, "best_shift_projection_amplitude_error",
                row["projection_amplitude_error"])
        push!(entries,
              "$(row["region"]): projection_error=$(row["projection_amplitude_error"]); " *
              "best_shift=$best_shift; " *
              "best_shift_projection_error=$best_shift_projection_error; " *
              "pass=$(row["one_percent_pass"])")
    end

    return "$path: " * join(entries, "; ")
end

function timeseries_comparison_evidence(path)
    table = read_metrics_csv(path)
    isnothing(table) && return "$path: metrics file missing or empty"
    isempty(table.rows) && return "$path: metrics file empty"

    fail_count = count(row -> haskey(row, "one_percent_pass") &&
                              !parse_bool(row["one_percent_pass"]),
                       table.rows)
    worst_index = argmax(parse_float(row, "relative_error") for row in table.rows)
    worst_row = table.rows[worst_index]

    return "$path: rows=$(length(table.rows)); fail_rows=$fail_count; " *
           "worst_time=$(worst_row["time"]); worst_metric=$(worst_row["metric"]); " *
           "worst_relative_error=$(worst_row["relative_error"])"
end

function saved_time_metrics_pass(path)
    table = read_metrics_csv(path)
    isnothing(table) && return false
    isempty(table.rows) && return false
    return all(row -> haskey(row, "one_percent_pass") &&
                      parse_bool(row["one_percent_pass"]),
               table.rows)
end

function saved_time_field_metrics_evidence(path)
    table = read_metrics_csv(path)
    isnothing(table) && return "$path: metrics file missing or empty"
    isempty(table.rows) && return "$path: metrics file empty"

    fail_count = count(row -> haskey(row, "one_percent_pass") &&
                              !parse_bool(row["one_percent_pass"]),
                       table.rows)
    error_key = "relative_l2_error" in table.header ? "relative_l2_error" :
                "relative_error" in table.header ? "relative_error" :
                "maximum_absolute_error"
    worst_index = argmax(parse_float(row, error_key) for row in table.rows)
    worst_row = table.rows[worst_index]
    time_value = get(worst_row, "cm1_time_seconds",
                     get(worst_row, "reference_time_seconds",
                         get(worst_row, "time", "missing")))
    field_value = get(worst_row, "field", get(worst_row, "metric", "missing"))

    return "$path: rows=$(length(table.rows)); fail_rows=$fail_count; " *
           "worst_time=$time_value; worst_field_or_metric=$field_value; " *
           "worst_$error_key=$(worst_row[error_key])"
end

function tier1_discriminator_rows(path)
    table = read_metrics_csv(path)
    isnothing(table) && return nothing
    rows_by_key = Dict{Tuple{String, String}, Dict{String, String}}()

    for row in table.rows
        rows_by_key[(get(row, "region", ""), get(row, "field", ""))] = row
    end

    required_keys = [
        ("below_sponge", "w_center"),
        ("below_sponge", "pressure_perturbation"),
        ("below_sponge", "u"),
        ("below_sponge", "theta_perturbation"),
        ("full_domain", "mountain_drag"),
        ("full_domain", "coordinate_x"),
        ("full_domain", "coordinate_z"),
    ]

    all(key -> haskey(rows_by_key, key), required_keys) || return nothing
    return rows_by_key
end

function tier1_discriminator_pass(path)
    rows = tier1_discriminator_rows(path)
    isnothing(rows) && return false

    required_keys = [
        ("below_sponge", "w_center"),
        ("below_sponge", "pressure_perturbation"),
        ("below_sponge", "u"),
        ("below_sponge", "theta_perturbation"),
        ("full_domain", "mountain_drag"),
        ("full_domain", "coordinate_x"),
        ("full_domain", "coordinate_z"),
    ]

    return all(key -> parse_bool(rows[key]["one_percent_pass"]), required_keys)
end

function tier1_discriminator_evidence(path)
    rows = tier1_discriminator_rows(path)
    isnothing(rows) && return "$path: metrics file missing, empty, or missing required rows"

    w = rows[("below_sponge", "w_center")]
    pressure = rows[("below_sponge", "pressure_perturbation")]
    drag = rows[("full_domain", "mountain_drag")]
    coordinate_x = rows[("full_domain", "coordinate_x")]
    coordinate_z = rows[("full_domain", "coordinate_z")]

    entries = [
        "w_l2=$(w["relative_l2_error"])",
        "w_linf=$(w["relative_linf_error"])",
        "w_rmse=$(w["normalized_rmse"])",
        "w_corr=$(w["pattern_correlation"])",
        "w_projection=$(w["projection_amplitude_error"])",
        "pressure_l2=$(pressure["relative_l2_error"])",
        "pressure_linf=$(pressure["relative_linf_error"])",
        "drag=$(drag["relative_l2_error"])",
        "max_dx=$(coordinate_x["relative_l2_error"])",
        "max_dz=$(coordinate_z["relative_l2_error"])",
        "w_pass=$(w["one_percent_pass"])",
        "pressure_pass=$(pressure["one_percent_pass"])",
        "drag_pass=$(drag["one_percent_pass"])",
        "coordinate_x_pass=$(coordinate_x["one_percent_pass"])",
        "coordinate_z_pass=$(coordinate_z["one_percent_pass"])",
    ]

    return "$path: " * join(entries, "; ")
end

function linear_wave_rows(path)
    table = read_metrics_csv(path)
    isnothing(table) && return nothing
    rows_by_key = Dict{Tuple{String, String}, Dict{String, String}}()

    for row in table.rows
        rows_by_key[(get(row, "region", ""), get(row, "field", ""))] = row
    end

    haskey(rows_by_key, ("below_sponge", "w")) || return nothing
    haskey(rows_by_key, ("full_run", "robustness")) || return nothing
    return rows_by_key
end

function linear_wave_pass(path)
    rows = linear_wave_rows(path)
    isnothing(rows) && return false
    w = rows[("below_sponge", "w")]
    robustness = rows[("full_run", "robustness")]
    return parse_bool(w["one_percent_pass"]) &&
           parse_bool(robustness["one_percent_pass"])
end

function linear_wave_field_pass(path, field)
    rows = linear_wave_rows(path)
    isnothing(rows) && return false
    key = ("below_sponge", field)
    haskey(rows, key) || return false
    row = rows[key]
    robustness = rows[("full_run", "robustness")]
    return parse_bool(row["one_percent_pass"]) &&
           parse_bool(robustness["one_percent_pass"])
end

function linear_wave_evidence(path)
    rows = linear_wave_rows(path)
    isnothing(rows) && return "$path: metrics file missing, empty, or missing required rows"

    return linear_wave_field_evidence(path, "w")
end

function linear_wave_field_evidence(path, field)
    rows = linear_wave_rows(path)
    isnothing(rows) && return "$path: metrics file missing, empty, or missing required rows"

    key = ("below_sponge", field)
    haskey(rows, key) || return "$path: missing below_sponge/$field row"
    w = rows[key]
    robustness = rows[("full_run", "robustness")]
    entries = [
        "$(field)_l2=$(w["relative_l2_error"])",
        "$(field)_linf=$(w["relative_linf_error"])",
        "$(field)_rmse=$(w["normalized_rmse"])",
        "$(field)_corr=$(w["pattern_correlation"])",
        "$(field)_amp=$(w["maximum_amplitude_error"])",
        "$(field)_projection=$(w["projection_amplitude_error"])",
        "$(field)_best_shift_projection=$(w["best_shift_projection_amplitude_error"])",
        "$(field)_best_shift=$(w["best_shift_cells"])",
        "$(field)_phase=$(w["phase_error_wavelengths"])",
        "$(field)_pass=$(w["one_percent_pass"])",
        "nan_count=$(robustness["projection_amplitude_error"])",
        "inf_count=$(robustness["best_shift_projection_amplitude_error"])",
        "stable_dt=$(robustness["best_shift_cells"])",
        "robustness_pass=$(robustness["one_percent_pass"])",
    ]

    return "$path: " * join(entries, "; ")
end

const REQUIRED_SCHAR_OPERATOR_BUDGET_TERMS =
    ("ub_pgrad", "wb_pgrad", "wb_buoy", "ub_vadv", "ub_vadv_velocity_form")

function schar_operator_budget_baseline_present(path)
    table = read_metrics_csv(path)
    isnothing(table) && return false
    terms = Set(get(row, "term", "") for row in table.rows)
    return all(term -> term in terms, REQUIRED_SCHAR_OPERATOR_BUDGET_TERMS)
end

function schar_operator_budget_baseline_evidence(path)
    table = read_metrics_csv(path)
    isnothing(table) && return "$path: missing or unreadable"

    rows_by_term = Dict{String, Dict{String, String}}()
    for row in table.rows
        term = get(row, "term", "")
        haskey(rows_by_term, term) || (rows_by_term[term] = row)
    end
    missing_terms = [term for term in REQUIRED_SCHAR_OPERATOR_BUDGET_TERMS
                     if !haskey(rows_by_term, term)]
    !isempty(missing_terms) &&
        return "$path: missing terms=$(join(missing_terms, "|"))"

    entries = String[]
    for term in REQUIRED_SCHAR_OPERATOR_BUDGET_TERMS
        row = rows_by_term[term]
        push!(entries, "$(term)_l2=$(row["relative_l2_error"])")
    end

    acoustic_path =
        joinpath(dirname(path),
                 "schar_2s_breeze_acoustic_increment_vs_cm1_ub_pgrad_summary.csv")
    acoustic_table = read_metrics_csv(acoustic_path)
    if !isnothing(acoustic_table)
        acoustic_rows =
            Dict(get(row, "term", "") => row for row in acoustic_table.rows)
        term = "ub_acoustic_increment_velocity"
        if haskey(acoustic_rows, term)
            row = acoustic_rows[term]
            push!(entries, "acoustic_increment_l2=$(row["relative_l2_error"])")
            push!(entries, "acoustic_increment_corr=$(row["pattern_correlation"])")
        end
    end

    closure_path = joinpath(dirname(path), "schar_2s_cm1_budget_closure_summary.csv")
    closure_table = read_metrics_csv(closure_path)
    if !isnothing(closure_table)
        closure_rows =
            Dict(get(row, "comparison", "") => row for row in closure_table.rows)
        for comparison in ("sum_ubudgets_vs_u_increment",
                           "inferred_pgrad_vs_emitted_ub_pgrad")
            if haskey(closure_rows, comparison)
                row = closure_rows[comparison]
                push!(entries, "$(comparison)_l2=$(row["relative_l2_error"])")
            end
        end
    end

    increment_path =
        joinpath(dirname(path), "schar_2s_breeze_dt0p1_cm1_u_increment_closure_summary.csv")
    increment_table = read_metrics_csv(increment_path)
    if !isnothing(increment_table)
        increment_rows =
            Dict(get(row, "comparison", "") => row for row in increment_table.rows)
        comparison = "breeze_u_increment_vs_cm1_u_increment"
        if haskey(increment_rows, comparison)
            row = increment_rows[comparison]
            push!(entries, "breeze_dt0p1_u_increment_vs_cm1_l2=$(row["relative_l2_error"])")
            push!(entries, "breeze_dt0p1_u_increment_vs_cm1_corr=$(row["pattern_correlation"])")
        end
    end

    aligned_increment_path =
        joinpath(dirname(path),
                 "schar_2s_breeze_dt0p1_cm1terrain_cm1constants_cm1_u_increment_closure_summary.csv")
    aligned_increment_table = read_metrics_csv(aligned_increment_path)
    if !isnothing(aligned_increment_table)
        aligned_increment_rows =
            Dict(get(row, "comparison", "") => row for row in aligned_increment_table.rows)
        comparison = "breeze_u_increment_vs_cm1_u_increment"
        if haskey(aligned_increment_rows, comparison)
            row = aligned_increment_rows[comparison]
            push!(entries, "breeze_dt0p1_cm1terrain_cm1constants_u_increment_vs_cm1_l2=$(row["relative_l2_error"])")
            push!(entries, "breeze_dt0p1_cm1terrain_cm1constants_u_increment_vs_cm1_corr=$(row["pattern_correlation"])")
        end
    end

    rk_split_path =
        first_existing((joinpath(dirname(path), "schar_2s_breeze_cm1_rk_split_conservative_increment_budget_summary.csv"),
                        joinpath(dirname(path), "schar_2s_breeze_cm1_rk_split_increment_budget_summary.csv")))
    rk_split_table = read_metrics_csv(rk_split_path)
    if !isnothing(rk_split_table)
        rk_split_rows =
            Dict(get(row, "comparison", "") => row for row in rk_split_table.rows)
        for comparison in ("breeze_rk_pressure_vs_cm1_ub_pgrad",
                           "breeze_rk_total_vs_breeze_u_increment")
            if haskey(rk_split_rows, comparison)
                row = rk_split_rows[comparison]
                push!(entries, "$(comparison)_l2=$(row["relative_l2_error"])")
                push!(entries, "$(comparison)_corr=$(row["pattern_correlation"])")
            end
        end
    end

    outside_rk_split_path =
        joinpath(dirname(path),
                 "schar_2s_breeze_cm1_rk_split_conservative_outside_pgf_increment_budget_summary.csv")
    outside_rk_split_table = read_metrics_csv(outside_rk_split_path)
    if !isnothing(outside_rk_split_table)
        outside_rk_split_rows =
            Dict(get(row, "comparison", "") => row for row in outside_rk_split_table.rows)
        comparison = "breeze_rk_pressure_vs_cm1_ub_pgrad"
        if haskey(outside_rk_split_rows, comparison)
            row = outside_rk_split_rows[comparison]
            push!(entries, "outside_pgf_$(comparison)_l2=$(row["relative_l2_error"])")
            push!(entries, "outside_pgf_$(comparison)_corr=$(row["pattern_correlation"])")
        end
    end

    acoustic_pgrad_path =
        joinpath(dirname(path), "cm1_schar_acoustic_pgrad_increment_validation.csv")
    acoustic_pgrad_table = read_metrics_csv(acoustic_pgrad_path)
    if !isnothing(acoustic_pgrad_table)
        acoustic_pgrad_rows =
            Dict(get(row, "comparison", "") => row for row in acoustic_pgrad_table.rows)
        comparison = "accumulated_total_vs_instrumented_emitted"
        if haskey(acoustic_pgrad_rows, comparison)
            row = acoustic_pgrad_rows[comparison]
            push!(entries, "cm1_acoustic_ppd_reconstruction_l2=$(row["relative_l2_error"])")
            push!(entries, "cm1_acoustic_ppd_reconstruction_corr=$(row["pattern_correlation"])")
        end
    end

    acoustic_component_path =
        joinpath(dirname(path),
                 "schar_2s_breeze_rk_pressure_vs_cm1_acoustic_components_summary.csv")
    acoustic_component_table = read_metrics_csv(acoustic_component_path)
    if !isnothing(acoustic_component_table)
        acoustic_component_rows =
            Dict(get(row, "comparison", "") => row for row in acoustic_component_table.rows)
        for comparison in ("breeze_rk_pressure_vs_cm1_acoustic_ppd",
                           "breeze_rk_pressure_vs_cm1_accumulated_total",
                           "breeze_rk_pressure_vs_cm1_emitted_ub_pgrad")
            if haskey(acoustic_component_rows, comparison)
                row = acoustic_component_rows[comparison]
                push!(entries, "$(comparison)_l2=$(row["relative_l2_error"])")
                push!(entries, "$(comparison)_corr=$(row["pattern_correlation"])")
            end
        end
    end

    substep_pressure_path =
        joinpath(dirname(path),
                 "schar_2s_breeze_substep_pressure_vs_cm1_acoustic_components_summary.csv")
    substep_pressure_table = read_metrics_csv(substep_pressure_path)
    if !isnothing(substep_pressure_table)
        substep_pressure_rows =
            Dict(get(row, "comparison", "") => row for row in substep_pressure_table.rows)
        for comparison in ("breeze_substep_pressure_plus_slow_vs_final_minus_initial",
                           "breeze_substep_frozen_plus_perturbation_vs_pressure",
                           "breeze_substep_pressure_vs_cm1_acoustic_ppd",
                           "breeze_substep_frozen_pressure_vs_cm1_acoustic_ppd",
                           "breeze_substep_perturbation_pressure_vs_cm1_acoustic_ppd",
                           "breeze_substep_cm1_exner_vs_cm1_acoustic_ppd",
                           "breeze_substep_frozen_cm1_exner_vs_cm1_acoustic_ppd",
                           "breeze_substep_perturbation_cm1_exner_vs_cm1_acoustic_ppd")
            if haskey(substep_pressure_rows, comparison)
                row = substep_pressure_rows[comparison]
                push!(entries, "$(comparison)_l2=$(row["relative_l2_error"])")
                push!(entries, "$(comparison)_corr=$(row["pattern_correlation"])")
            end
        end
    end

    substep_pressure_split_path =
        joinpath(dirname(path),
                 "schar_2s_breeze_substep_pressure_vs_cm1_acoustic_split_components_summary.csv")
    substep_pressure_split_table = read_metrics_csv(substep_pressure_split_path)
    if !isnothing(substep_pressure_split_table)
        substep_pressure_split_rows =
            Dict(get(row, "comparison", "") => row for row in substep_pressure_split_table.rows)
        for comparison in ("breeze_substep_pressure_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_substep_frozen_pressure_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_substep_cm1_exner_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_substep_frozen_cm1_exner_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_substep_pressure_vs_cm1_acoustic_terrain_ppd",
                           "breeze_substep_perturbation_pressure_vs_cm1_acoustic_terrain_ppd",
                           "breeze_substep_perturbation_cm1_exner_vs_cm1_acoustic_terrain_ppd",
                           "breeze_frozen_pressure_plus_cm1_acoustic_terrain_vs_cm1_acoustic_ppd",
                           "cm1_acoustic_horizontal_plus_breeze_perturbation_pressure_vs_cm1_acoustic_ppd",
                           "breeze_frozen_cm1_exner_plus_cm1_acoustic_terrain_vs_cm1_acoustic_ppd",
                           "cm1_acoustic_horizontal_plus_breeze_perturbation_cm1_exner_vs_cm1_acoustic_ppd",
                           "breeze_post_recovery_pressure_vs_cm1_acoustic_ppd",
                           "breeze_post_recovery_horizontal_pressure_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_post_recovery_terrain_pressure_vs_cm1_acoustic_terrain_ppd",
                           "breeze_nonlinear_pressure_vs_cm1_acoustic_ppd",
                           "breeze_nonlinear_horizontal_pressure_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_nonlinear_terrain_pressure_vs_cm1_acoustic_terrain_ppd",
                           "breeze_horizontal_pressure_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_horizontal_pressure_shift_m1_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_horizontal_pressure_shift_p1_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_horizontal_pressure_signflip_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_ungated_horizontal_pressure_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_frozen_horizontal_pressure_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_perturbation_horizontal_pressure_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_terrain_pressure_vs_cm1_acoustic_terrain_ppd",
                           "breeze_frozen_terrain_pressure_vs_cm1_acoustic_terrain_ppd",
                           "breeze_perturbation_terrain_pressure_vs_cm1_acoustic_terrain_ppd",
                           "breeze_ungated_terrain_pressure_vs_cm1_acoustic_terrain_ppd",
                           "breeze_horizontal_cm1_exner_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_horizontal_cm1_exner_shift_m1_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_horizontal_cm1_exner_shift_p1_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_horizontal_cm1_exner_signflip_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_ungated_horizontal_cm1_exner_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_frozen_horizontal_cm1_exner_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_perturbation_horizontal_cm1_exner_vs_cm1_acoustic_horizontal_ppd",
                           "breeze_terrain_cm1_exner_vs_cm1_acoustic_terrain_ppd",
                           "breeze_frozen_terrain_cm1_exner_vs_cm1_acoustic_terrain_ppd",
                           "breeze_perturbation_terrain_cm1_exner_vs_cm1_acoustic_terrain_ppd",
                           "breeze_ungated_terrain_cm1_exner_vs_cm1_acoustic_terrain_ppd",
                           "breeze_horizontal_pressure_vs_breeze_horizontal_cm1_exner",
                           "breeze_terrain_pressure_vs_breeze_terrain_cm1_exner",
                           "breeze_frozen_horizontal_pressure_vs_breeze_frozen_horizontal_cm1_exner",
                           "breeze_frozen_terrain_pressure_vs_breeze_frozen_terrain_cm1_exner",
                           "breeze_perturbation_horizontal_pressure_vs_breeze_perturbation_horizontal_cm1_exner",
                           "breeze_perturbation_terrain_pressure_vs_breeze_perturbation_terrain_cm1_exner",
                           "breeze_horizontal_plus_terrain_pressure_vs_pressure",
                           "breeze_horizontal_plus_terrain_cm1_exner_vs_cm1_exner",
                           "cm1_acoustic_horizontal_plus_terrain_vs_cm1_acoustic_ppd")
            if haskey(substep_pressure_split_rows, comparison)
                row = substep_pressure_split_rows[comparison]
                push!(entries, "$(comparison)_l2=$(row["relative_l2_error"])")
                push!(entries, "$(comparison)_corr=$(row["pattern_correlation"])")
            end
        end
    end

    output_exner_history_path =
        joinpath(dirname(path),
                 "schar_2s_cm1_output_exner_vs_acoustic_history_summary.csv")
    output_exner_history_table = read_metrics_csv(output_exner_history_path)
    if !isnothing(output_exner_history_table)
        output_exner_history_rows =
            Dict(get(row, "comparison", "") => row for row in output_exner_history_table.rows)
        for comparison in ("cm1_output_0s_exner_vs_cm1_acoustic_ppd",
                           "cm1_output_2s_exner_vs_cm1_acoustic_ppd",
                           "cm1_output_avg_exner_vs_cm1_acoustic_ppd",
                           "breeze_substep_cm1_exner_vs_cm1_output_2s_exner",
                           "breeze_substep_cm1_exner_vs_cm1_output_avg_exner")
            if haskey(output_exner_history_rows, comparison)
                row = output_exner_history_rows[comparison]
                push!(entries, "$(comparison)_l2=$(row["relative_l2_error"])")
                push!(entries, "$(comparison)_corr=$(row["pattern_correlation"])")
            end
        end
    end

    return "$path: " * join(entries, "; ")
end

function frame_pair_evidence(path)
    table = read_metrics_csv(path)
    isnothing(table) && return "$path: frame-pair file missing or empty"
    isempty(table.rows) && return "$path: frame-pair file empty"

    worst_index = argmax(parse_float(row, "maximum_abs_error") for row in table.rows)
    worst_row = table.rows[worst_index]
    maximum_time_offset =
        maximum(parse_float(row, "time_difference_seconds") for row in table.rows)

    return "$path: frames=$(length(table.rows)); " *
           "maximum_time_offset=$maximum_time_offset; " *
           "worst_frame=$(worst_row["frame"]); " *
           "worst_cm1_time=$(worst_row["cm1_time_seconds"]); " *
           "worst_maximum_abs_error=$(worst_row["maximum_abs_error"])"
end

function complex_frame_pair_evidence(path)
    table = read_metrics_csv(path)
    isnothing(table) && return "$path: frame-pair file missing or empty"
    isempty(table.rows) && return "$path: frame-pair file empty"

    worst_w_index = argmax(parse_float(row, "maximum_abs_w_error") for row in table.rows)
    worst_pressure_index =
        argmax(parse_float(row, "maximum_abs_pressure_error") for row in table.rows)
    worst_w_l2_index = argmax(parse_float(row, "w_relative_l2_error") for row in table.rows)
    worst_pressure_l2_index =
        argmax(parse_float(row, "pressure_relative_l2_error") for row in table.rows)
    worst_w_row = table.rows[worst_w_index]
    worst_pressure_row = table.rows[worst_pressure_index]
    worst_w_l2_row = table.rows[worst_w_l2_index]
    worst_pressure_l2_row = table.rows[worst_pressure_l2_index]
    maximum_time_offset =
        maximum(parse_float(row, "time_difference_seconds") for row in table.rows)

    return "$path: frames=$(length(table.rows)); " *
           "maximum_time_offset=$maximum_time_offset; " *
           "worst_w_frame=$(worst_w_row["frame"]); " *
           "worst_w_time=$(worst_w_row["cm1_time_seconds"]); " *
           "worst_maximum_abs_w_error=$(worst_w_row["maximum_abs_w_error"]); " *
           "worst_w_relative_l2_frame=$(worst_w_l2_row["frame"]); " *
           "worst_w_relative_l2=$(worst_w_l2_row["w_relative_l2_error"]); " *
           "worst_pressure_frame=$(worst_pressure_row["frame"]); " *
           "worst_pressure_time=$(worst_pressure_row["cm1_time_seconds"]); " *
           "worst_maximum_abs_pressure_error=$(worst_pressure_row["maximum_abs_pressure_error"]); " *
           "worst_pressure_relative_l2_frame=$(worst_pressure_l2_row["frame"]); " *
           "worst_pressure_relative_l2=$(worst_pressure_l2_row["pressure_relative_l2_error"])"
end

function file_evidence(paths)
    present = String[]
    missing = String[]

    for path in paths
        if isfile(path)
            push!(present, path)
        else
            push!(missing, path)
        end
    end

    parts = String[]
    !isempty(present) && push!(parts, "present: " * join(present, ", "))
    !isempty(missing) && push!(parts, "missing: " * join(missing, ", "))
    return join(parts, "; ")
end

function status_from_files(paths)
    all(isfile, paths) ? "pass" : "missing"
end

function metric_value_file_contains_all(path, metric_names)
    table = read_metrics_csv(path)
    isnothing(table) && return false
    names = Set(get(row, "metric", "") for row in table.rows)
    return all(name -> name in names, metric_names)
end

function csv_header_contains_all(path, field_names)
    table = read_metrics_csv(path)
    isnothing(table) && return false
    header_names = Set(table.header)
    return all(name -> name in header_names, field_names)
end

function csv_column_contains_all(path, column_name, expected_values)
    table = read_metrics_csv(path)
    isnothing(table) && return false
    values = Set(get(row, column_name, "") for row in table.rows)
    return all(value -> value in values, expected_values)
end

function timeseries_cadence_status_and_evidence()
    runs = [
        ("Schar explicit",
         joinpath(ROOT, "terrain_schar_6h_400x200_production_explicit",
                  "terrain_schar_mountain_wave_energy_timeseries.csv"),
         21600.0, 600.0, 5.0),
        ("Schar substepper",
         joinpath(ROOT, "terrain_schar_6h_400x200_production_substepper",
                  "terrain_schar_mountain_wave_energy_timeseries.csv"),
         21600.0, 600.0, 5.0),
        ("Schar linear explicit",
         joinpath(ROOT, "linear_mountain_wave_explicit_production_400x200_6h_gpu",
                  "terrain_schar_mountain_wave_energy_timeseries.csv"),
         21600.0, 600.0, 5.0),
        ("Schar linear substepper",
         joinpath(ROOT, "linear_mountain_wave_production_400x200_6h_gpu",
                  "terrain_schar_mountain_wave_energy_timeseries.csv"),
         21600.0, 600.0, 5.0),
        ("Complex explicit",
         joinpath(ROOT, "complex_mountain_production_explicit",
                  "complex_mountain_timeseries.csv"),
         21600.0, 600.0, 1.0),
        ("Complex substepper",
         joinpath(ROOT, "complex_mountain_production_substepper",
                  "complex_mountain_timeseries.csv"),
         21600.0, 600.0, 1.0),
    ]

    evidence = String[]
    all_ok = true
    any_missing = false

    for (label, path, required_final_time, maximum_allowed_gap, tolerance) in runs
        table = read_metrics_csv(path)
        if isnothing(table) || isempty(table.rows)
            any_missing = true
            all_ok = false
            push!(evidence, "$label: missing_or_empty ($path)")
            continue
        end

        times = sort([parse_float(row, "time") for row in table.rows])
        final_time = last(times)
        gaps = diff(times)
        maximum_gap = isempty(gaps) ? 0.0 : maximum(gaps)
        cadence_ok = maximum_gap <= maximum_allowed_gap + tolerance
        runtime_ok = final_time >= required_final_time - tolerance
        all_ok &= cadence_ok && runtime_ok
        push!(evidence,
              "$label: rows=$(length(times)); final_time=$final_time; " *
              "max_gap=$maximum_gap; cadence_ok=$cadence_ok; " *
              "runtime_ok=$runtime_ok; path=$path")
    end

    status = all_ok ? "pass" : any_missing ? "missing" : "blocked"
    return status, join(evidence, "; ")
end

function production_timeseries_metric_contract_status_and_evidence()
    required_common_fields = [
        "time",
        "maximum_w",
        "maximum_u",
        "maximum_pressure_perturbation",
        "domain_kinetic_energy",
        "mass_relative_drift",
        "maximum_cfl",
        "maximum_acoustic_cfl",
        "high_k_energy_fraction_near_terrain",
        "reflection_energy_fraction_above_sponge_start",
    ]

    runs = [
        ("Schar explicit",
         joinpath(ROOT, "terrain_schar_6h_400x200_production_explicit",
                  "terrain_schar_mountain_wave_energy_timeseries.csv"),
         ["mountain_drag"]),
        ("Schar substepper",
         joinpath(ROOT, "terrain_schar_6h_400x200_production_substepper",
                  "terrain_schar_mountain_wave_energy_timeseries.csv"),
         ["mountain_drag"]),
        ("Schar linear explicit",
         joinpath(ROOT, "linear_mountain_wave_explicit_production_400x200_6h_gpu",
                  "terrain_schar_mountain_wave_energy_timeseries.csv"),
         ["mountain_drag"]),
        ("Schar linear substepper",
         joinpath(ROOT, "linear_mountain_wave_production_400x200_6h_gpu",
                  "terrain_schar_mountain_wave_energy_timeseries.csv"),
         ["mountain_drag"]),
        ("Complex explicit",
         joinpath(ROOT, "complex_mountain_production_explicit",
                  "complex_mountain_timeseries.csv"),
         ["pressure_drag_x", "pressure_drag_y"]),
        ("Complex substepper",
         joinpath(ROOT, "complex_mountain_production_substepper",
                  "complex_mountain_timeseries.csv"),
         ["pressure_drag_x", "pressure_drag_y"]),
    ]

    evidence = String[]
    all_ok = true
    any_missing_file = false

    for (label, path, drag_fields) in runs
        table = read_metrics_csv(path)
        if isnothing(table)
            any_missing_file = true
            all_ok = false
            push!(evidence, "$label: missing_or_empty ($path)")
            continue
        end

        header = Set(table.header)
        missing_fields =
            filter(name -> !(name in header), vcat(required_common_fields,
                                                   drag_fields))
        ok = isempty(missing_fields)
        all_ok &= ok
        push!(evidence,
              "$label: required_timeseries_fields_present=$ok; " *
              "missing=" * join(missing_fields, ", ") * "; path=$path")
    end

    if all_ok
        return "pass", join(evidence, "; ")
    elseif any_missing_file
        return "missing", join(evidence, "; ")
    else
        return "blocked", join(evidence, "; ")
    end
end

function completed_case_region_coverage_status_and_evidence()
    schar_required_files = [
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_full_domain_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_near_terrain_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_400x200_substepper_vs_explicit_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_400x200_substepper_vs_explicit_full_domain_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_400x200_substepper_vs_explicit_near_terrain_production_1pct_metrics_state_metrics.csv"),
    ]

    complex_required_regions = [
        "below_sponge",
        "full_domain",
        "near_terrain",
        "centerline_slice",
        "lee_side_box",
        "hilltop_box",
    ]

    complex_files = [
        joinpath(ROOT, "complex_mountain_explicit_vs_cm1_production_1pct_metrics.csv"),
        joinpath(ROOT, "complex_mountain_substepper_vs_explicit_production_1pct_metrics.csv"),
    ]

    schar_ok = all(isfile, schar_required_files)
    complex_ok = all(path -> csv_column_contains_all(path, "region",
                                                     complex_required_regions),
                     complex_files)

    evidence = "Schar region files present=$schar_ok: " *
               file_evidence(schar_required_files) *
               "; complex 3D region coverage present=$complex_ok: " *
               join(complex_files, ", ")

    if schar_ok && complex_ok
        return "pass", evidence
    elseif all(isfile, schar_required_files) && all(isfile, complex_files)
        return "blocked", evidence
    else
        return "missing", evidence
    end
end

function direct_robustness_schema_status_and_evidence()
    final_metric_names = [
        "maximum_cfl",
        "maximum_acoustic_cfl",
        "nan_count",
        "inf_count",
        "mass_relative_drift",
        "bottom_normal_velocity_max_abs",
        "high_k_energy_fraction_near_terrain",
        "walltime_per_step",
    ]

    timeseries_field_names = [
        "time",
        "maximum_w",
        "maximum_u",
        "maximum_pressure_perturbation",
        "domain_kinetic_energy",
        "mass_relative_drift",
        "maximum_cfl",
        "maximum_acoustic_cfl",
        "high_k_energy_fraction_near_terrain",
        "reflection_energy_fraction_above_sponge_start",
        "walltime_per_step",
    ]

    runs = [
        ("Schar explicit",
         joinpath(ROOT, "terrain_schar_6h_400x200_production_explicit",
                  "terrain_schar_mountain_wave_metrics.csv"),
         joinpath(ROOT, "terrain_schar_6h_400x200_production_explicit",
                  "terrain_schar_mountain_wave_energy_timeseries.csv")),
        ("Schar substepper",
         joinpath(ROOT, "terrain_schar_6h_400x200_production_substepper",
                  "terrain_schar_mountain_wave_metrics.csv"),
         joinpath(ROOT, "terrain_schar_6h_400x200_production_substepper",
                  "terrain_schar_mountain_wave_energy_timeseries.csv")),
        ("Schar linear explicit",
         joinpath(ROOT, "linear_mountain_wave_explicit_production_400x200_6h_gpu",
                  "terrain_schar_mountain_wave_metrics.csv"),
         joinpath(ROOT, "linear_mountain_wave_explicit_production_400x200_6h_gpu",
                  "terrain_schar_mountain_wave_energy_timeseries.csv")),
        ("Schar linear substepper",
         joinpath(ROOT, "linear_mountain_wave_production_400x200_6h_gpu",
                  "terrain_schar_mountain_wave_metrics.csv"),
         joinpath(ROOT, "linear_mountain_wave_production_400x200_6h_gpu",
                  "terrain_schar_mountain_wave_energy_timeseries.csv")),
        ("Complex explicit",
         joinpath(ROOT, "complex_mountain_production_explicit",
                  "complex_mountain_metrics.csv"),
         joinpath(ROOT, "complex_mountain_production_explicit",
                  "complex_mountain_timeseries.csv")),
        ("Complex substepper",
         joinpath(ROOT, "complex_mountain_production_substepper",
                  "complex_mountain_metrics.csv"),
         joinpath(ROOT, "complex_mountain_production_substepper",
                  "complex_mountain_timeseries.csv")),
    ]

    evidence = String[]
    all_direct = true
    any_missing_file = false

    for (label, metrics_path, timeseries_path) in runs
        metrics_present = isfile(metrics_path)
        timeseries_present = isfile(timeseries_path)
        metrics_ok = metrics_present &&
                     metric_value_file_contains_all(metrics_path, final_metric_names)
        timeseries_ok = timeseries_present &&
                        csv_header_contains_all(timeseries_path,
                                                timeseries_field_names)

        all_direct &= metrics_ok && timeseries_ok
        any_missing_file |= !metrics_present || !timeseries_present
        push!(evidence,
              "$label: metrics=$(metrics_ok ? "direct" : metrics_present ? "incomplete" : "missing") " *
              "($metrics_path), timeseries=$(timeseries_ok ? "direct" : timeseries_present ? "incomplete" : "missing") " *
              "($timeseries_path)")
    end

    if all_direct
        return "pass", join(evidence, "; ")
    elseif any_missing_file
        return "missing", join(evidence, "; ")
    else
        return "blocked", join(evidence, "; ")
    end
end

function finite_bottom_robustness_status_and_evidence()
    runs = [
        ("Schar explicit",
         joinpath(ROOT, "terrain_schar_6h_400x200_production_explicit",
                  "terrain_schar_mountain_wave_energy_timeseries.csv")),
        ("Schar substepper",
         joinpath(ROOT, "terrain_schar_6h_400x200_production_substepper",
                  "terrain_schar_mountain_wave_energy_timeseries.csv")),
        ("Schar linear explicit",
         joinpath(ROOT, "linear_mountain_wave_explicit_production_400x200_6h_gpu",
                  "terrain_schar_mountain_wave_energy_timeseries.csv")),
        ("Schar linear substepper",
         joinpath(ROOT, "linear_mountain_wave_production_400x200_6h_gpu",
                  "terrain_schar_mountain_wave_energy_timeseries.csv")),
        ("Complex explicit",
         joinpath(ROOT, "complex_mountain_production_explicit",
                  "complex_mountain_timeseries.csv")),
        ("Complex substepper",
         joinpath(ROOT, "complex_mountain_production_substepper",
                  "complex_mountain_timeseries.csv")),
    ]

    evidence = String[]
    all_pass = true
    any_missing = false
    tolerance = 1e-12

    for (label, path) in runs
        table = read_metrics_csv(path)
        if isnothing(table) || isempty(table.rows)
            any_missing = true
            all_pass = false
            push!(evidence, "$label: missing or empty ($path)")
            continue
        end

        maximum_nan_count = maximum(parse_float(row, "nan_count") for row in table.rows)
        maximum_inf_count = maximum(parse_float(row, "inf_count") for row in table.rows)
        maximum_bottom =
            maximum(parse_float(row, "bottom_normal_velocity_max_abs") for row in table.rows)
        run_pass = maximum_nan_count == 0 &&
                   maximum_inf_count == 0 &&
                   maximum_bottom <= tolerance
        all_pass &= run_pass
        push!(evidence,
              "$label: max_nan=$maximum_nan_count; max_inf=$maximum_inf_count; " *
              "max_bottom_normal_velocity=$maximum_bottom; pass=$run_pass")
    end

    status = all_pass ? "pass" : any_missing ? "missing" : "fail"
    return status, join(evidence, "; ")
end

function askervein_output_contract_status_and_evidence()
    audit = joinpath(ROOT, "terrain_following_production_validation_metric_schema_audit.md")
    blocker = joinpath(ROOT, "askervein_production_blocker_summary.md")
    manifest = joinpath(ROOT, "askervein_coordinate_faithful_production_manifest.md")
    files = [audit, blocker, manifest]
    all_present = all(isfile, files)
    all_present || return "missing", file_evidence(files)

    required_audit_terms = [
        "production spin-up and averaging-window metrics",
        "production LES time series",
        "accepted observational/reference comparison over all required transects",
        "coordinate-faithful WEMEP transect/profile comparison files from a",
        "Schema status: `blocked/incomplete`.",
    ]

    audit_documents_gap = file_contains_all(audit, required_audit_terms)
    evidence = file_evidence(files) *
               "; audit_documents_required_output_gap=$audit_documents_gap"

    if audit_documents_gap
        return "blocked", evidence *
               "; missing production LES metrics/time series, transect/profile reference comparison, spin-up/averaging metrics, and accepted explicit-window output"
    else
        return "present", evidence
    end
end

function file_contains_all(path, needles)
    isfile(path) || return false
    text = read(path, String)
    return all(needle -> occursin(needle, text), needles)
end

function artifact_manifest_contract_status_and_evidence()
    manifest = joinpath(ROOT, "terrain_following_production_validation_artifact_manifest.md")
    isfile(manifest) || return "missing", "$manifest is missing"
    manifest_text = replace(lowercase(read(manifest, String)), r"\s+" => " ")

    required_terms = [
        "run command",
        "git commit",
        "machine/backend",
        "grid and timestep",
        "physical runtime",
        "artifact class",
        "metrics path",
        "summary path",
        "movie or plot path",
        "reference dataset path",
        "pass/fail status for each 1% metric",
        "Smoke and diagnostic artifacts are explicitly excluded from completion.",
    ]

    complete_current_sections =
        all(needle -> occursin(replace(lowercase(needle), r"\s+" => " "),
                               manifest_text), [
        "Schar: manifest coverage is complete",
        "Complex mountain: manifest coverage is complete",
        "Askervein: manifest coverage is intentionally incomplete",
        "coordinate-faithful Breeze production run command",
        "accepted production runtime",
        "spin-up and averaging window",
        "production movie or plot sequence from coordinate-faithful output",
        "accepted reference comparison path",
        "pass/fail status for a declared explicit-feasible production window",
    ])

    schema_terms_present =
        all(needle -> occursin(replace(lowercase(needle), r"\s+" => " "),
                               manifest_text),
            required_terms)
    evidence = "$manifest: schema_terms_present=$schema_terms_present; " *
               "current_case_coverage_documented=$complete_current_sections; " *
               "Askervein final-manifest fields remain intentionally incomplete"

    if !schema_terms_present
        return "missing", evidence
    elseif complete_current_sections
        return "blocked", evidence
    else
        return "present", evidence
    end
end

function completed_production_manifest_status_and_evidence()
    manifest = joinpath(ROOT, "terrain_following_production_validation_artifact_manifest.md")
    isfile(manifest) || return "missing", "$manifest is missing"
    text = replace(lowercase(read(manifest, String)), r"\s+" => " ")

    completed_run_terms = [
        "cm1 explicit reference",
        "breeze explicit",
        "breeze explicit field-snapshot rerun",
        "breeze acoustic substepper",
        "breeze acoustic substepper field-snapshot rerun",
        "complex mountain: cm1 doernbrack itern=3",
        "production declaration",
        "complex mountain comparisons",
        "command source:",
        "backend/machine:",
        "grid/runtime:",
        "metrics:",
        "summary:",
        "time series:",
        "plot:",
        "movie",
        "status: `pass` for artifact presence",
        "status: `fail`.",
    ]

    completed_coverage =
        all(term -> occursin(term, text), completed_run_terms)
    evidence = "$manifest: completed_production_manifest_terms_present=$completed_coverage"
    return completed_coverage ? ("pass", evidence) : ("missing", evidence)
end

function one_percent_summary_status_and_evidence()
    summary = joinpath(ROOT, "terrain_following_production_validation_1pct_summary.md")
    isfile(summary) || return "missing", "$summary is missing"
    text = replace(lowercase(read(summary, String)), r"\s+" => " ")

    required_terms = [
        "schar 6 h 400x200",
        "complex mountain 6 h 120x120x150",
        "askervein hill",
        "explicit-vs-cm1",
        "substepper-vs-explicit",
        "saved-time explicit-vs-cm1",
        "saved-time substepper-vs-explicit",
        "1% status: fail",
        "1% status: blocked",
        "smoke and diagnostic artifacts to be excluded",
    ]

    complete = all(needle -> occursin(needle, text), required_terms)
    evidence = "$summary: required_terms_present=$complete"
    return complete ? ("pass", evidence) : ("missing", evidence)
end

function time_resolved_comparison_contract_status_and_evidence()
    audit = joinpath(ROOT, "terrain_following_time_resolved_comparison_audit.md")
    partial_files = [
        joinpath(ROOT, "schar_6h_400x200_substepper_vs_explicit_timeseries_metrics.csv"),
        joinpath(ROOT, "schar_cm1_periodic_theta300_vs_breeze_substepper_raw_error_movie",
                 "frame_pairs.csv"),
        joinpath(ROOT, "schar_cm1_periodic_theta300_vs_breeze_substepper_field_error_movie",
                 "frame_pairs.csv"),
        joinpath(ROOT, "complex_mountain_substepper_vs_explicit_timeseries_metrics.csv"),
        joinpath(ROOT, "complex_mountain_cm1_vs_breeze_substepper_movie",
                 "frame_pairs.csv"),
    ]

    required_files = [
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_timeseries_metrics.csv"),
        joinpath(ROOT, "complex_mountain_explicit_vs_cm1_timeseries_metrics.csv"),
    ]

    partial_present = all(isfile, partial_files)
    missing_required = filter(!isfile, required_files)
    required_present = isempty(missing_required)
    audit_present = isfile(audit)

    evidence = "audit=$audit; audit_present=$audit_present; " *
               "partial_saved_time_metrics_present=$partial_present; " *
               "missing_required=" * join(missing_required, ", ")

    if !audit_present
        return "missing", evidence
    elseif required_present
        return "pass", evidence
    else
        return "blocked", evidence
    end
end

function resolution_adequacy_status_and_evidence()
    artifact_manifest = joinpath(ROOT, "terrain_following_production_validation_artifact_manifest.md")
    complex_manifest = joinpath(ROOT, "complex_mountain_doernbrack_production_manifest.md")
    askervein_manifest = joinpath(ROOT, "askervein_coordinate_faithful_production_manifest.md")

    isfile(artifact_manifest) || return "missing", "$artifact_manifest is missing"
    isfile(complex_manifest) || return "missing", "$complex_manifest is missing"
    isfile(askervein_manifest) || return "missing", "$askervein_manifest is missing"

    artifact_text = replace(lowercase(read(artifact_manifest, String)), r"\s+" => " ")
    complex_text = replace(lowercase(read(complex_manifest, String)), r"\s+" => " ")
    askervein_text = replace(lowercase(read(askervein_manifest, String)), r"\s+" => " ")

    schar_documented = all(needle -> occursin(needle, artifact_text), [
        "resolution adequacy",
        "400x200",
        "dx = 500 m",
        "dz = 150 m",
        "lambda = 4 km",
        "8 points per lambda",
        "10 points per a",
        "sponge base is `20 km`",
        "133 active vertical cells",
        "three-resolution convergence campaign",
    ])

    complex_documented = all(needle -> occursin(needle, complex_text), [
        "horizontal points per hill half-width",
        "aa / dx = 20",
        "100` active vertical cells below the sponge",
        "matched `nx = 120`",
        "dx = 1000 m",
        "dy = 1000 m",
        "dz = 200 m",
    ])

    askervein_gap_documented = all(needle -> occursin(needle, askervein_text), [
        "production definition partially specified",
        "20 m`, matching the erf `300 x 300` grid",
        "production targets",
        "explicit-window target",
        "spin-up and averaging",
        "declared wemep/erf production",
        "coordinate-faithful production-grid explicit window",
    ])

    evidence = "Schar documented=$schar_documented in $artifact_manifest; " *
               "complex documented=$complex_documented in $complex_manifest; " *
               "Askervein gap documented=$askervein_gap_documented in $askervein_manifest"

    if schar_documented && complex_documented && askervein_gap_documented
        return "blocked", evidence *
               "; completed-case resolution notes are present, but Askervein still lacks an accepted production-resolution validation artifact"
    elseif schar_documented && complex_documented
        return "blocked", evidence *
               "; completed-case resolution notes are present, but Askervein resolution acceptance is not documented"
    else
        return "missing", evidence
    end
end

function schar_400x200_6h_summary(path)
    return file_contains_all(path, [
        "Nx, Nz = 400, 200",
        "stop time = 2.160000e+04 s",
    ])
end

function parse_breeze_nx_nz_summary(path)
    isfile(path) || return nothing
    text = read(path, String)
    match_result = match(r"Nx, Nz =\s*(\d+),\s*(\d+)", text)
    isnothing(match_result) && return nothing
    return (parse(Int, match_result.captures[1]),
            parse(Int, match_result.captures[2]))
end

function parse_cm1_nx_nz_config(path)
    isfile(path) || return nothing
    text = read(path, String)
    nx_match = match(r"(?m)^\s*nx\s*=\s*(\d+)", text)
    nz_match = match(r"(?m)^\s*nz\s*=\s*(\d+)", text)
    (isnothing(nx_match) || isnothing(nz_match)) && return nothing
    return (parse(Int, nx_match.captures[1]),
            parse(Int, nz_match.captures[1]))
end

function resolution_match_status_and_evidence(cm1_config, breeze_summary)
    cm1_resolution = parse_cm1_nx_nz_config(cm1_config)
    breeze_resolution = parse_breeze_nx_nz_summary(breeze_summary)

    if isnothing(cm1_resolution) || isnothing(breeze_resolution)
        return ("missing",
                "could not parse CM1 or Breeze resolution from $cm1_config and $breeze_summary")
    end

    status = cm1_resolution == breeze_resolution ? "pass" : "fail"
    evidence = "CM1 nx,nz=$(cm1_resolution); Breeze Nx,Nz=$(breeze_resolution); " *
               "CM1 config=$cm1_config; Breeze summary=$breeze_summary"
    return status, evidence
end

function has_production_askervein_summary(path)
    isfile(path) || return false
    text = read(path, String)
    return occursin("artifact_class = production_validation", text) &&
           occursin("production_validation = true", text)
end

function askervein_summary_has_diagnostic_warning(path)
    isfile(path) || return false
    text = read(path, String)
    return occursin("diagnostic unless explicitly", text) ||
           occursin("still requires accepted production reference data", text)
end

function askervein_explicit_window_status_and_evidence(summary_path, metrics_path)
    files_present = isfile(summary_path) && isfile(metrics_path)
    files_present || return "missing", file_evidence([summary_path, metrics_path])

    flagged_production = has_production_askervein_summary(summary_path)
    has_diagnostic_warning = askervein_summary_has_diagnostic_warning(summary_path)
    metrics_pass = askervein_metrics_pass(metrics_path)

    evidence = file_evidence([summary_path, metrics_path]) *
               "; artifact_flags_production=$flagged_production" *
               "; summary_warns_diagnostic=$has_diagnostic_warning" *
               "; one_percent_metrics_pass=$metrics_pass"

    if has_diagnostic_warning
        return "blocked", evidence *
               "; plan requires an accepted explicit-feasible production window before this artifact can satisfy validation"
    end

    status = !flagged_production ? "blocked" : metrics_pass ? "pass" : "fail"
    return status, evidence
end

function askervein_metrics_pass(path)
    table = read_metrics_csv(path)
    isnothing(table) && return false
    isempty(table.rows) && return false
    return all(row -> haskey(row, "one_percent_pass") &&
                      parse_bool(row["one_percent_pass"]),
               table.rows)
end

function complex_comparison_pass(metrics_path, summary_path)
    table = read_metrics_csv(metrics_path)
    isnothing(table) && return false

    below_sponge_rows = filter(row -> get(row, "region", "") == "below_sponge",
                               table.rows)
    isempty(below_sponge_rows) && return false
    fields_pass = all(row -> haskey(row, "one_percent_pass") &&
                             parse_bool(row["one_percent_pass"]),
                      below_sponge_rows)

    isfile(summary_path) || return false
    summary = read(summary_path, String)
    drag_pass = occursin("pressure_drag_x_one_percent_pass = true", summary) &&
                occursin("pressure_drag_y_one_percent_pass = true", summary)

    return fields_pass && drag_pass
end

function complex_comparison_evidence(metrics_path, summary_path)
    table = read_metrics_csv(metrics_path)
    isnothing(table) && return "$metrics_path: metrics file missing or empty"

    entries = String[]
    for field in ("u", "v", "w", "theta", "pressure")
        row_index = findfirst(row -> get(row, "region", "") == "below_sponge" &&
                                     get(row, "field", "") == field,
                              table.rows)
        if isnothing(row_index)
            push!(entries, "$(field)=missing")
        else
            row = table.rows[row_index]
            push!(entries,
                  "$(field)_l2=$(row["relative_l2_error"]); " *
                  "$(field)_rmse=$(row["normalized_rmse"]); " *
                  "$(field)_corr=$(row["pattern_correlation"]); " *
                  "$(field)_pass=$(row["one_percent_pass"])")
        end
    end

    if isfile(summary_path)
        summary = read(summary_path, String)
        drag_x = match(r"pressure_drag_x_relative_error = ([^\n]+)", summary)
        drag_y = match(r"pressure_drag_y_relative_error = ([^\n]+)", summary)
        drag_x_sign = match(r"pressure_drag_x_sign_matches_reference = ([^\n]+)", summary)
        drag_y_sign = match(r"pressure_drag_y_sign_matches_reference = ([^\n]+)", summary)
        push!(entries,
              "drag_x=$(isnothing(drag_x) ? "missing" : strip(drag_x.captures[1])); " *
              "drag_x_sign=$(isnothing(drag_x_sign) ? "missing" : strip(drag_x_sign.captures[1])); " *
              "drag_y=$(isnothing(drag_y) ? "missing" : strip(drag_y.captures[1])); " *
              "drag_y_sign=$(isnothing(drag_y_sign) ? "missing" : strip(drag_y_sign.captures[1]))")
    else
        push!(entries, "drag_summary=missing")
    end

    return "$metrics_path: " * join(entries, "; ")
end

function build_checks()
    checks = Check[]

    schar_explicit_metrics = first_existing([
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_400x200_periodic_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_400x200_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_explicit_vs_cm1_periodic_production_1pct_metrics_state_metrics.csv"),
    ])
    schar_substepper_metrics = first_existing([
        joinpath(ROOT, "schar_6h_400x200_substepper_vs_explicit_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_substepper_vs_explicit_production_1pct_metrics_state_metrics.csv"),
    ])
    schar_full_explicit_metrics = first_existing([
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_full_domain_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_400x200_periodic_full_domain_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_400x200_full_domain_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_full_domain_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_explicit_vs_cm1_periodic_full_domain_production_1pct_metrics_state_metrics.csv"),
    ])
    schar_full_substepper_metrics = first_existing([
        joinpath(ROOT, "schar_6h_400x200_substepper_vs_explicit_full_domain_production_1pct_metrics_state_metrics.csv"),
        joinpath(ROOT, "schar_6h_substepper_vs_explicit_full_domain_production_1pct_metrics_state_metrics.csv"),
    ])
    schar_near_terrain_explicit_metrics =
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_near_terrain_production_1pct_metrics_state_metrics.csv")
    schar_near_terrain_substepper_metrics =
        joinpath(ROOT, "schar_6h_400x200_substepper_vs_explicit_near_terrain_production_1pct_metrics_state_metrics.csv")

    explicit_row = first_row(schar_explicit_metrics)
    substepper_row = first_row(schar_substepper_metrics)
    full_explicit_row = first_row(schar_full_explicit_metrics)
    full_substepper_row = first_row(schar_full_substepper_metrics)
    near_terrain_explicit_row = first_row(schar_near_terrain_explicit_metrics)
    near_terrain_substepper_row = first_row(schar_near_terrain_substepper_metrics)

    schar_cm1_config =
        first_existing([
            joinpath(ROOT, "cm1_schar_400x200_periodic_theta300_reference", "cm1_config.txt"),
            joinpath(ROOT, "cm1_schar_400x200_periodic_reference", "cm1_config.txt"),
            joinpath(ROOT, "cm1_schar_400x200_reference", "cm1_config.txt"),
        ])
    schar_explicit_summary =
        joinpath(ROOT, "terrain_schar_6h_400x200_production_explicit",
                 "terrain_schar_mountain_wave_summary.txt")
    resolution_status, resolution_evidence =
        resolution_match_status_and_evidence(schar_cm1_config,
                                             schar_explicit_summary)

    push!(checks, Check("Schär",
                        "CM1 explicit reference resolution matches the Breeze production grid",
                        resolution_status,
                        resolution_evidence))

    push!(checks, Check("Schär",
                        "Breeze explicit vs CM1 explicit satisfies below-sponge 1% field, phase, and drag gates",
                        schar_wave_pass(explicit_row) ? "pass" : "fail",
                        "$(schar_explicit_metrics): $(schar_metric_evidence(explicit_row))"))

    push!(checks, Check("Schär",
                        "Breeze substepper vs Breeze explicit satisfies below-sponge 1% field, phase, and drag gates",
                        schar_wave_pass(substepper_row) ? "pass" : "fail",
                        "$(schar_substepper_metrics): $(schar_metric_evidence(substepper_row))"))

    schar_explicit_projection_metrics =
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_projection_metrics.csv")
    schar_substepper_projection_metrics =
        joinpath(ROOT, "schar_6h_400x200_substepper_vs_explicit_projection_metrics.csv")
    push!(checks, Check("Schär",
                        "Breeze explicit vs CM1 explicit satisfies below-sponge 1% projection-amplitude gates",
                        !isfile(schar_explicit_projection_metrics) ? "missing" :
                        projection_metrics_pass(schar_explicit_projection_metrics) ? "pass" : "fail",
                        projection_metric_evidence(schar_explicit_projection_metrics)))

    push!(checks, Check("Schär",
                        "Breeze substepper vs Breeze explicit satisfies below-sponge 1% projection-amplitude gates",
                        !isfile(schar_substepper_projection_metrics) ? "missing" :
                        projection_metrics_pass(schar_substepper_projection_metrics) ? "pass" : "fail",
                        projection_metric_evidence(schar_substepper_projection_metrics)))

    push!(checks, Check("Schär",
                        "Full-domain explicit-vs-CM1 diagnostic metrics are present",
                        isnothing(full_explicit_row) ? "missing" : "present",
                        "$(schar_full_explicit_metrics): $(schar_metric_evidence(full_explicit_row))"))

    push!(checks, Check("Schär",
                        "Full-domain substepper-vs-explicit diagnostic metrics are present",
                        isnothing(full_substepper_row) ? "missing" : "present",
                        "$(schar_full_substepper_metrics): $(schar_metric_evidence(full_substepper_row))"))

    push!(checks, Check("Schär",
                        "Near-terrain explicit-vs-CM1 diagnostic metrics are present",
                        isnothing(near_terrain_explicit_row) ? "missing" : "present",
                        "$(schar_near_terrain_explicit_metrics): $(schar_metric_evidence(near_terrain_explicit_row))"))

    push!(checks, Check("Schär",
                        "Near-terrain substepper-vs-explicit diagnostic metrics are present",
                        isnothing(near_terrain_substepper_row) ? "missing" : "present",
                        "$(schar_near_terrain_substepper_metrics): $(schar_metric_evidence(near_terrain_substepper_row))"))

    schar_no_damping_no_sponge_discriminator =
        joinpath(ROOT, "schar_substepper_vs_explicit_tier1_6h_no_damping_no_upper_sponge_grid_schema_refresh_coordcheck",
                 "schar_substepper_vs_explicit_state_metrics.csv")
    push!(checks, Check("Schär",
                        "TEST E/F no-damping/no-upper-sponge production discriminator satisfies coordinate-matched 1% Tier-1 gates",
                        !isfile(schar_no_damping_no_sponge_discriminator) ? "missing" :
                        tier1_discriminator_pass(schar_no_damping_no_sponge_discriminator) ? "pass" : "fail",
                        tier1_discriminator_evidence(schar_no_damping_no_sponge_discriminator)))

    schar_matched_dt_discriminator =
        joinpath(ROOT, "schar_substepper_vs_explicit_tier1_6h_dt0p35_no_damping_no_upper_sponge_grid",
                 "schar_substepper_vs_explicit_state_metrics.csv")
    push!(checks, Check("Schär",
                        "Matched outer-dt no-damping/no-upper-sponge production discriminator satisfies coordinate-matched 1% Tier-1 gates",
                        !isfile(schar_matched_dt_discriminator) ? "missing" :
                        tier1_discriminator_pass(schar_matched_dt_discriminator) ? "pass" : "fail",
                        tier1_discriminator_evidence(schar_matched_dt_discriminator)))

    linear_substepper_metrics =
        joinpath(ROOT, "linear_mountain_wave_production_400x200_6h_gpu",
                 "linear_mountain_wave_state_metrics.csv")
    linear_explicit_metrics =
        joinpath(ROOT, "linear_mountain_wave_explicit_production_400x200_6h_gpu",
                 "linear_mountain_wave_state_metrics.csv")
    linear_substepper_plot =
        joinpath(ROOT, "linear_mountain_wave_production_400x200_6h_gpu",
                 "linear_mountain_wave_w_comparison.ppm")
    linear_explicit_plot =
        joinpath(ROOT, "linear_mountain_wave_explicit_production_400x200_6h_gpu",
                 "linear_mountain_wave_w_comparison.ppm")
    linear_tier1_metrics =
        joinpath(ROOT, "linear_mountain_wave_substepper_vs_explicit_400x200_6h_gpu",
                 "schar_substepper_vs_explicit_state_metrics.csv")
    linear_substepper_wtilde_metrics =
        joinpath(ROOT, "linear_mountain_wave_production_400x200_6h_gpu_wtilde",
                 "linear_mountain_wave_state_metrics.csv")
    linear_substepper_wtilde_slice =
        joinpath(ROOT, "linear_mountain_wave_production_400x200_6h_gpu_wtilde",
                 "terrain_schar_mountain_wave_w_slice.csv")
    linear_substepper_wtilde_plot =
        joinpath(ROOT, "linear_mountain_wave_production_400x200_6h_gpu_wtilde",
                 "linear_mountain_wave_w_comparison.ppm")
    linear_explicit_wtilde_metrics =
        joinpath(ROOT, "linear_mountain_wave_explicit_production_400x200_6h_gpu_wtilde",
                 "linear_mountain_wave_state_metrics.csv")
    linear_explicit_wtilde_slice =
        joinpath(ROOT, "linear_mountain_wave_explicit_production_400x200_6h_gpu_wtilde",
                 "terrain_schar_mountain_wave_w_slice.csv")
    linear_explicit_wtilde_plot =
        joinpath(ROOT, "linear_mountain_wave_explicit_production_400x200_6h_gpu_wtilde",
                 "linear_mountain_wave_w_comparison.ppm")

    push!(checks, Check("Schär linear",
                        "Low-amplitude substepper-vs-linear-theory production artifact has metrics and plot",
                        status_from_files([linear_substepper_metrics,
                                           linear_substepper_plot]) == "pass" ?
                        "present" : "missing",
                        file_evidence([linear_substepper_metrics,
                                       linear_substepper_plot])))

    push!(checks, Check("Schär linear",
                        "Low-amplitude substepper satisfies analytical linear-wave 1% gates",
                        !isfile(linear_substepper_metrics) ? "missing" :
                        linear_wave_pass(linear_substepper_metrics) ? "pass" : "fail",
                        linear_wave_evidence(linear_substepper_metrics)))

    push!(checks, Check("Schär linear",
                        "Low-amplitude explicit-control production artifact has metrics and plot",
                        status_from_files([linear_explicit_metrics,
                                           linear_explicit_plot]) == "pass" ?
                        "present" : "missing",
                        file_evidence([linear_explicit_metrics,
                                       linear_explicit_plot])))

    push!(checks, Check("Schär linear",
                        "Low-amplitude explicit control satisfies analytical linear-wave 1% gates",
                        !isfile(linear_explicit_metrics) ? "missing" :
                        linear_wave_pass(linear_explicit_metrics) ? "pass" : "fail",
                        linear_wave_evidence(linear_explicit_metrics)))

    push!(checks, Check("Schär linear",
                        "Low-amplitude substepper vs explicit satisfies coordinate-matched 1% Tier-1 gates",
                        !isfile(linear_tier1_metrics) ? "missing" :
                        tier1_discriminator_pass(linear_tier1_metrics) ? "pass" : "fail",
                        tier1_discriminator_evidence(linear_tier1_metrics)))

    push!(checks, Check("Schär linear",
                        "Low-amplitude exact-wtilde substepper production artifact has metrics, slice, and plot",
                        status_from_files([linear_substepper_wtilde_metrics,
                                           linear_substepper_wtilde_slice,
                                           linear_substepper_wtilde_plot]) == "pass" ?
                        "present" : "missing",
                        file_evidence([linear_substepper_wtilde_metrics,
                                       linear_substepper_wtilde_slice,
                                       linear_substepper_wtilde_plot])))

    push!(checks, Check("Schär linear",
                        "Low-amplitude exact-wtilde substepper satisfies analytical w_tilde 1% gates",
                        !isfile(linear_substepper_wtilde_metrics) ? "missing" :
                        linear_wave_field_pass(linear_substepper_wtilde_metrics, "w_tilde") ? "pass" : "fail",
                        linear_wave_field_evidence(linear_substepper_wtilde_metrics, "w_tilde")))

    push!(checks, Check("Schär linear",
                        "Low-amplitude exact-wtilde explicit-control production artifact has metrics, slice, and plot",
                        status_from_files([linear_explicit_wtilde_metrics,
                                           linear_explicit_wtilde_slice,
                                           linear_explicit_wtilde_plot]) == "pass" ?
                        "present" : "missing",
                        file_evidence([linear_explicit_wtilde_metrics,
                                       linear_explicit_wtilde_slice,
                                       linear_explicit_wtilde_plot])))

    push!(checks, Check("Schär linear",
                        "Low-amplitude exact-wtilde explicit control satisfies analytical w_tilde 1% gates",
                        !isfile(linear_explicit_wtilde_metrics) ? "missing" :
                        linear_wave_field_pass(linear_explicit_wtilde_metrics, "w_tilde") ? "pass" : "fail",
                        linear_wave_field_evidence(linear_explicit_wtilde_metrics, "w_tilde")))

    schar_operator_budget_baseline =
        joinpath(ROOT, "schar_2s_operator_budget_blocker_summary.csv")
    push!(checks, Check("Schär",
                        "Early-time 2 s operator-budget blocker baseline is present",
                        schar_operator_budget_baseline_present(schar_operator_budget_baseline) ?
                        "present" : "missing",
                        schar_operator_budget_baseline_evidence(schar_operator_budget_baseline)))

    schar_timeseries_comparison =
        joinpath(ROOT, "schar_6h_400x200_substepper_vs_explicit_timeseries_metrics.csv")
    push!(checks, Check("Schär",
                        "Saved-time scalar substepper-vs-explicit comparison metrics are present",
                        isfile(schar_timeseries_comparison) ? "present" : "missing",
                        timeseries_comparison_evidence(schar_timeseries_comparison)))

    push!(checks, Check("Schär",
                        "Saved-time scalar substepper-vs-explicit comparison satisfies 1% gates",
                        !isfile(schar_timeseries_comparison) ? "missing" :
                        saved_time_metrics_pass(schar_timeseries_comparison) ? "pass" : "fail",
                        saved_time_field_metrics_evidence(schar_timeseries_comparison)))

    schar_substepper_explicit_field_timeseries =
        joinpath(ROOT, "schar_6h_400x200_substepper_vs_explicit_field_timeseries_metrics.csv")
    push!(checks, Check("Schär",
                        "Saved-time field substepper-vs-explicit comparison metrics are present",
                        isfile(schar_substepper_explicit_field_timeseries) ? "present" : "missing",
                        saved_time_field_metrics_evidence(schar_substepper_explicit_field_timeseries)))

    push!(checks, Check("Schär",
                        "Saved-time field substepper-vs-explicit comparison satisfies 1% gates",
                        !isfile(schar_substepper_explicit_field_timeseries) ? "missing" :
                        saved_time_metrics_pass(schar_substepper_explicit_field_timeseries) ? "pass" : "fail",
                        saved_time_field_metrics_evidence(schar_substepper_explicit_field_timeseries)))

    schar_explicit_cm1_timeseries =
        joinpath(ROOT, "schar_6h_400x200_explicit_vs_cm1_timeseries_metrics.csv")
    push!(checks, Check("Schär",
                        "Saved-time explicit-vs-CM1 field comparison satisfies 1% gates",
                        !isfile(schar_explicit_cm1_timeseries) ? "missing" :
                        saved_time_metrics_pass(schar_explicit_cm1_timeseries) ? "pass" : "fail",
                        saved_time_field_metrics_evidence(schar_explicit_cm1_timeseries)))

    schar_cm1_frame_pairs =
        joinpath(ROOT, "schar_cm1_periodic_theta300_vs_breeze_substepper_raw_error_movie",
                 "frame_pairs.csv")
    push!(checks, Check("Schär",
                        "Saved-time CM1-vs-Breeze raw-error frame metrics are present",
                        isfile(schar_cm1_frame_pairs) ? "present" : "missing",
                        frame_pair_evidence(schar_cm1_frame_pairs)))

    schar_cm1_field_frame_pairs =
        joinpath(ROOT, "schar_cm1_periodic_theta300_vs_breeze_substepper_field_error_movie",
                 "frame_pairs.csv")
    push!(checks, Check("Schär",
                        "Saved-time CM1-vs-Breeze w and pressure frame error metrics are present",
                        isfile(schar_cm1_field_frame_pairs) ? "present" : "missing",
                        complex_frame_pair_evidence(schar_cm1_field_frame_pairs)))

    schar_diagnosis_files = [
        joinpath(ROOT, "terrain_schar_production_1pct_failure_diagnosis.md"),
        joinpath(ROOT, "terrain_schar_400x200_substepper_vs_explicit_diagnosis.md"),
        joinpath(ROOT, "terrain_schar_400x200_substepper_vs_explicit_diagnostics.csv"),
        joinpath(ROOT, "schar_cm1_boundary_condition_diagnosis.md"),
        joinpath(ROOT, "schar_cm1_thermodynamic_reference_diagnosis.md"),
        joinpath(ROOT, "schar_best_matched_cm1_failure_summary.md"),
        joinpath(ROOT, "schar_shared_drag_diagnostic.csv"),
        joinpath(ROOT, "schar_400x200_explicit_dt0p5_failure_summary.md"),
    ]
    push!(checks, Check("Schär",
                        "Measured 1% failures have diagnostic notes and matched-grid Breeze-pair breakdown",
                        status_from_files(schar_diagnosis_files) == "pass" ? "present" : "missing",
                        file_evidence(schar_diagnosis_files)))

    schar_existing_summary =
        joinpath(ROOT, "terrain_schar_6h_400x200_production_substepper", "terrain_schar_mountain_wave_summary.txt")
    schar_production_files = [
        schar_existing_summary,
        joinpath(ROOT, "terrain_schar_6h_400x200_production_substepper", "terrain_schar_mountain_wave_state_slice.csv"),
        joinpath(ROOT, "terrain_schar_6h_400x200_production_substepper", "terrain_schar_mountain_wave_energy_timeseries.csv"),
    ]
    push!(checks, Check("Schär",
                        "Existing 400x200 6 h substepper artifact has summary, state slice, and time series",
                        status_from_files(schar_production_files) == "pass" &&
                        schar_400x200_6h_summary(schar_existing_summary) ? "pass" : "missing",
                        file_evidence(schar_production_files)))

    schar_missing_explicit_400 = [
        schar_explicit_summary,
        joinpath(ROOT, "terrain_schar_6h_400x200_production_explicit",
                 "terrain_schar_mountain_wave_state_slice.csv"),
        joinpath(ROOT, "terrain_schar_6h_400x200_production_explicit",
                 "terrain_schar_mountain_wave_energy_timeseries.csv"),
    ]
    push!(checks, Check("Schär",
                        "Matched 400x200 6 h Breeze explicit artifact exists",
                        status_from_files(schar_missing_explicit_400) == "pass" &&
                        schar_400x200_6h_summary(schar_explicit_summary) ? "pass" : "missing",
                        file_evidence(schar_missing_explicit_400)))

    schar_plot_files = [
        joinpath(ROOT, "terrain_schar_6h_400x200_production_explicit",
                 "terrain_schar_mountain_wave_w_comparison.ppm"),
        joinpath(ROOT, "terrain_schar_6h_400x200_production_substepper",
                 "terrain_schar_mountain_wave_w_comparison.ppm"),
        joinpath(ROOT, "schar_cm1_periodic_theta300_vs_breeze_explicit_field_error_movie",
                 "schar_cm1_vs_breeze_substepper_field_error.mp4"),
        joinpath(ROOT, "schar_cm1_periodic_theta300_vs_breeze_substepper_raw_error_movie",
                 "schar_cm1_vs_breeze_substepper_raw_error.mp4"),
        joinpath(ROOT, "schar_cm1_periodic_theta300_vs_breeze_substepper_field_error_movie",
                 "schar_cm1_vs_breeze_substepper_field_error.mp4"),
    ]
    push!(checks, Check("Schär",
                        "Production plots are present for matched Schär explicit and substepper outputs",
                        status_from_files(schar_plot_files),
                        file_evidence(schar_plot_files)))

    convergence_summary =
        joinpath(ROOT, "terrain_schar_6h_substepper_convergence_production",
                 "terrain_schar_grid_convergence_summary.txt")
    convergence_metrics =
        joinpath(ROOT, "terrain_schar_6h_substepper_convergence_production",
                 "terrain_schar_grid_convergence_metrics.csv")
    convergence_state_metrics =
        joinpath(ROOT, "terrain_schar_6h_substepper_convergence_production",
                 "terrain_schar_grid_convergence_state_metrics.csv")
    convergence_files = [convergence_summary, convergence_metrics,
                         convergence_state_metrics]
    convergence_has_production_time =
        file_contains_all(convergence_summary, [
            "cases = coarse:100:50:2,medium:200:100:2,fine:400:200:2",
            "stop seconds = 21600.0",
        ])
    push!(checks, Check("Schär",
                        "Three-resolution convergence campaign reaches the same production validation time",
                        status_from_files(convergence_files) == "pass" &&
                        convergence_has_production_time ? "pass" : "missing",
                        file_evidence(convergence_files)))

    complex_required = [
        joinpath(ROOT, "complex_mountain_production_cm1_reference",
                 "complex_mountain_state_slice.csv"),
        joinpath(ROOT, "complex_mountain_production_explicit",
                 "complex_mountain_state_slice.csv"),
        joinpath(ROOT, "complex_mountain_production_substepper",
                 "complex_mountain_state_slice.csv"),
        joinpath(ROOT, "complex_mountain_explicit_vs_cm1_production_1pct_metrics.csv"),
        joinpath(ROOT, "complex_mountain_substepper_vs_explicit_production_1pct_metrics.csv"),
    ]
    push!(checks, Check("Complex mountain",
                        "Production CM1, Breeze explicit, Breeze substepper, and comparison metrics exist",
                        status_from_files(complex_required),
                        file_evidence(complex_required)))

    complex_explicit_vs_cm1_metrics =
        joinpath(ROOT, "complex_mountain_explicit_vs_cm1_production_1pct_metrics.csv")
    complex_explicit_vs_cm1_summary =
        joinpath(ROOT, "complex_mountain_explicit_vs_cm1_production_1pct_metrics_summary.txt")
    complex_explicit_vs_cm1_status =
        !isfile(complex_explicit_vs_cm1_metrics) ? "missing" :
        complex_comparison_pass(complex_explicit_vs_cm1_metrics,
                                complex_explicit_vs_cm1_summary) ? "pass" : "fail"
    push!(checks, Check("Complex mountain",
                        "Breeze explicit vs CM1 explicit satisfies below-sponge 1% field and drag gates",
                        complex_explicit_vs_cm1_status,
                        complex_comparison_evidence(complex_explicit_vs_cm1_metrics,
                                                    complex_explicit_vs_cm1_summary)))

    complex_substepper_vs_explicit_metrics =
        joinpath(ROOT, "complex_mountain_substepper_vs_explicit_production_1pct_metrics.csv")
    complex_substepper_vs_explicit_summary =
        joinpath(ROOT, "complex_mountain_substepper_vs_explicit_production_1pct_metrics_summary.txt")
    complex_substepper_vs_explicit_status =
        !isfile(complex_substepper_vs_explicit_metrics) ? "missing" :
        complex_comparison_pass(complex_substepper_vs_explicit_metrics,
                                complex_substepper_vs_explicit_summary) ? "pass" : "fail"
    push!(checks, Check("Complex mountain",
                        "Breeze substepper vs Breeze explicit satisfies below-sponge 1% field and drag gates",
                        complex_substepper_vs_explicit_status,
                        complex_comparison_evidence(complex_substepper_vs_explicit_metrics,
                                                    complex_substepper_vs_explicit_summary)))

    complex_explicit_projection_metrics =
        joinpath(ROOT, "complex_mountain_explicit_vs_cm1_projection_metrics.csv")
    complex_substepper_projection_metrics =
        joinpath(ROOT, "complex_mountain_substepper_vs_explicit_projection_metrics.csv")
    push!(checks, Check("Complex mountain",
                        "Breeze explicit vs CM1 explicit satisfies below-sponge 1% projection-amplitude gates",
                        !isfile(complex_explicit_projection_metrics) ? "missing" :
                        projection_metrics_pass(complex_explicit_projection_metrics) ? "pass" : "fail",
                        projection_metric_evidence(complex_explicit_projection_metrics)))

    push!(checks, Check("Complex mountain",
                        "Breeze substepper vs Breeze explicit satisfies below-sponge 1% projection-amplitude gates",
                        !isfile(complex_substepper_projection_metrics) ? "missing" :
                        projection_metrics_pass(complex_substepper_projection_metrics) ? "pass" : "fail",
                        projection_metric_evidence(complex_substepper_projection_metrics)))

    complex_timeseries_comparison =
        joinpath(ROOT, "complex_mountain_substepper_vs_explicit_timeseries_metrics.csv")
    push!(checks, Check("Complex mountain",
                        "Saved-time scalar substepper-vs-explicit comparison metrics are present",
                        isfile(complex_timeseries_comparison) ? "present" : "missing",
                        timeseries_comparison_evidence(complex_timeseries_comparison)))

    push!(checks, Check("Complex mountain",
                        "Saved-time scalar substepper-vs-explicit comparison satisfies 1% gates",
                        !isfile(complex_timeseries_comparison) ? "missing" :
                        saved_time_metrics_pass(complex_timeseries_comparison) ? "pass" : "fail",
                        saved_time_field_metrics_evidence(complex_timeseries_comparison)))

    complex_substepper_explicit_field_timeseries =
        joinpath(ROOT, "complex_mountain_substepper_vs_explicit_field_timeseries_metrics.csv")
    push!(checks, Check("Complex mountain",
                        "Saved-time field substepper-vs-explicit comparison metrics are present",
                        isfile(complex_substepper_explicit_field_timeseries) ? "present" : "missing",
                        saved_time_field_metrics_evidence(complex_substepper_explicit_field_timeseries)))

    push!(checks, Check("Complex mountain",
                        "Saved-time field substepper-vs-explicit comparison satisfies 1% gates",
                        !isfile(complex_substepper_explicit_field_timeseries) ? "missing" :
                        saved_time_metrics_pass(complex_substepper_explicit_field_timeseries) ? "pass" : "fail",
                        saved_time_field_metrics_evidence(complex_substepper_explicit_field_timeseries)))

    complex_explicit_cm1_timeseries =
        joinpath(ROOT, "complex_mountain_explicit_vs_cm1_timeseries_metrics.csv")
    push!(checks, Check("Complex mountain",
                        "Saved-time explicit-vs-CM1 field comparison satisfies 1% gates",
                        !isfile(complex_explicit_cm1_timeseries) ? "missing" :
                        saved_time_metrics_pass(complex_explicit_cm1_timeseries) ? "pass" : "fail",
                        saved_time_field_metrics_evidence(complex_explicit_cm1_timeseries)))

    complex_cm1_frame_pairs =
        joinpath(ROOT, "complex_mountain_cm1_vs_breeze_substepper_movie",
                 "frame_pairs.csv")
    push!(checks, Check("Complex mountain",
                        "Saved-time CM1-vs-Breeze frame error metrics are present",
                        isfile(complex_cm1_frame_pairs) ? "present" : "missing",
                        complex_frame_pair_evidence(complex_cm1_frame_pairs)))

    complex_movie_files = [
        joinpath(ROOT, "complex_mountain_cm1_vs_breeze_explicit_movie",
                 "complex_mountain_cm1_vs_breeze_substepper.mp4"),
        joinpath(ROOT, "complex_mountain_cm1_vs_breeze_substepper_movie",
                 "complex_mountain_cm1_vs_breeze_substepper.mp4"),
    ]
    push!(checks, Check("Complex mountain",
                        "Production explicit and substepper CM1 comparison movies are present",
                        status_from_files(complex_movie_files),
                        file_evidence(complex_movie_files)))

    complex_manifest = joinpath(ROOT, "complex_mountain_doernbrack_production_manifest.md")
    complex_manifest_complete =
        file_contains_all(complex_manifest, [
            "Artifact class: `production_validation`.",
            "Selected terrain: CM1-native Doernbrack-style 3D hill, `itern = 3`.",
            "`Nx = 120`",
            "`Ny = 120`",
            "`Nz = 150`",
            "The 120x120x150 grid is the feasible production grid for the local serial CM1",
            "`6 h` minimum production validation time.",
            "/shared/home/kai/Aeolus/cm1r21.1/run/cm1.exe",
            "periodic lateral boundaries: `wbc = ebc = sbc = nbc = 1`",
        ])
    complex_manifest_status = complex_manifest_complete ? "pass" : "blocked"
    complex_manifest_evidence = if complex_manifest_complete
        "selected Doernbrack itern=3 production case in $complex_manifest; CM1 executable and matched grid/runtime declared; production artifacts still tracked separately"
    else
        "local ERF Altamont real-terrain setup is available as a candidate at /shared/home/greg/ERF/Exec/CanonicalTests/Real_Terrain/Altamont; complex_mountain_production_benchmark_spec.md proposes CM1 itern=3 Doernbrack hill as a cleaner CM1-native benchmark, but no complete production manifest exists yet"
    end
    push!(checks, Check("Complex mountain",
                        "Production terrain, resolution, runtime, and CM1 reference source are specified",
                        complex_manifest_status,
                        complex_manifest_evidence))

    askervein_summary =
        joinpath(ROOT, "askervein_explicit_substepper_compare_production",
                 "askervein_explicit_substepper_summary.txt")
    askervein_metrics =
        joinpath(ROOT, "askervein_explicit_substepper_compare_production",
                 "askervein_explicit_substepper_metrics.csv")
    askervein_status, askervein_evidence =
        askervein_explicit_window_status_and_evidence(askervein_summary,
                                                      askervein_metrics)
    push!(checks, Check("Askervein",
                        "Accepted production explicit-vs-substepper window is present and passes 1% metrics",
                        askervein_status,
                        askervein_evidence))

    askervein_diagnosis_files = [
        joinpath(ROOT, "askervein_explicit_substepper_1pct_failure_diagnosis.md"),
        joinpath(ROOT, "askervein_production_blocker_summary.md"),
        joinpath(ROOT, "askervein_coordinate_faithful_production_manifest.md"),
        joinpath(ROOT, "askervein_real_terrain_helpers.jl"),
        joinpath(ROOT, "askervein_erf_terrain_plumbing_smoke",
                 "askervein_les_summary.txt"),
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_plumbing_smoke",
                 "askervein_explicit_substepper_summary.txt"),
        joinpath(ROOT, "askervein_erf_terrain_96x72x32_1step_diagnostic",
                 "askervein_les_summary.txt"),
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_96x72x32_1step_diagnostic",
                 "askervein_explicit_substepper_summary.txt"),
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_96x72x32_10step_diagnostic",
                 "askervein_explicit_substepper_summary.txt"),
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_96x72x32_100step_diagnostic",
                 "askervein_explicit_substepper_summary.txt"),
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_96x72x32_1s_diagnostic",
                 "askervein_explicit_substepper_summary.txt"),
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_96x72x32_1s_gpu_diagnostic",
                 "askervein_explicit_substepper_summary.txt"),
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_96x72x32_1p2s_gpu_diagnostic",
                 "askervein_explicit_substepper_summary.txt"),
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_96x72x32_1p25s_gpu_diagnostic",
                 "askervein_explicit_substepper_summary.txt"),
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_96x72x32_1p5s_gpu_diagnostic",
                 "askervein_explicit_substepper_summary.txt"),
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_96x72x32_2s_gpu_diagnostic",
                 "askervein_explicit_substepper_summary.txt"),
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_96x72x32_5s_gpu_diagnostic",
                 "askervein_explicit_substepper_summary.txt"),
        joinpath(ROOT, "askervein_explicit_substepper_production-894.log"),
        joinpath(ROOT, "askervein_explicit_substepper_compare_production",
                 "askervein_vertical_velocity_error_profile.csv"),
        joinpath(ROOT, "askervein_explicit_substepper_compare_production",
                 "askervein_vertical_velocity_error_extrema.csv"),
    ]
    push!(checks, Check("Askervein",
                        "Explicit-window 1% failure and failed 60 s attempt are diagnosed",
                        status_from_files(askervein_diagnosis_files) == "pass" ? "present" : "missing",
                        file_evidence(askervein_diagnosis_files)))

    askervein_required_regions = [
        "full_domain",
        "near_terrain",
        "centerline_slice",
        "lee_side_box",
        "hilltop_box",
    ]
    askervein_1p2_gpu_metrics =
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_96x72x32_1p2s_gpu_diagnostic",
                 "askervein_explicit_substepper_metrics.csv")
    askervein_1p25_gpu_metrics =
        joinpath(ROOT, "askervein_erf_terrain_explicit_substepper_96x72x32_1p25s_gpu_diagnostic",
                 "askervein_explicit_substepper_metrics.csv")
    askervein_region_schema_status =
        csv_column_contains_all(askervein_1p2_gpu_metrics, "region",
                                askervein_required_regions) &&
        csv_column_contains_all(askervein_1p25_gpu_metrics, "region",
                                askervein_required_regions) ? "present" : "missing"
    push!(checks, Check("Askervein",
                        "Diagnostic ERF-terrain explicit-window brackets include required 3D regions",
                        askervein_region_schema_status,
                        "required regions=$(join(askervein_required_regions, ", ")); " *
                        "pass-side metrics=$askervein_1p2_gpu_metrics; " *
                        "fail-side metrics=$askervein_1p25_gpu_metrics"))

    askervein_production_les = [
        joinpath(ROOT, "askervein_les_production", "askervein_les_summary.txt"),
        joinpath(ROOT, "askervein_les_production", "askervein_les_metrics.csv"),
        joinpath(ROOT, "askervein_les_production", "askervein_les_speedup_w.ppm"),
    ]
    askervein_les_warning =
        askervein_summary_has_diagnostic_warning(first(askervein_production_les))
    push!(checks, Check("Askervein",
                        "Diagnostic substepper LES metrics and plot are present",
                        status_from_files(askervein_production_les) == "pass" ? "present" : "missing",
                        file_evidence(askervein_production_les) *
                        "; summary_warns_reference_incomplete=$askervein_les_warning"))

    askervein_wemep_reference = [
        joinpath(ROOT, "askervein_wemep_reference",
                 "askervein_elevation-roughness.map"),
        joinpath(ROOT, "askervein_wemep_reference",
                 "askervein_inlet1.txt"),
        joinpath(ROOT, "askervein_wemep_reference",
                 "askervein_sensor1.txt"),
        joinpath(ROOT, "askervein_wemep_reference",
                 "askervein_validation1.txt"),
        joinpath(ROOT, "askervein_wemep_reference_manifest.md"),
        joinpath(ROOT, "askervein_wemep_reference",
                 "askervein_les_production_askervein_wemep_mast_metrics.csv"),
        joinpath(ROOT, "askervein_wemep_reference",
                 "askervein_les_production_askervein_wemep_mast_summary.txt"),
    ]
    push!(checks, Check("Askervein",
                        "WEMEP reference files and named-mast diagnostic comparison are present",
                        status_from_files(askervein_wemep_reference) == "pass" ? "present" : "missing",
                        file_evidence(askervein_wemep_reference)))

    push!(checks, Check("Askervein",
                        "Production grid, explicit stable window, spin-up, averaging window, and accepted reference are specified",
                        "blocked",
                        "local ERF Askervein setup is available at /shared/home/greg/ERF/Exec/CanonicalTests/Real_Terrain/Askervein; WEMEP reference files, named-mast diagnostic comparison, askervein_coordinate_faithful_production_manifest.md, and diagnostic ERF-terrain/WEMEP-mast plumbing smokes including 96x72x32 LES and GPU explicit-vs-substepper brackets are present; strict 1% pass holds through 1.2 s and fails by 1.25 s due to w_tilde relative_linf; production boundary conditions, declared spin-up/averaging, runtime, and accepted explicit-feasible window remain unresolved"))

    askervein_contract_status, askervein_contract_evidence =
        askervein_output_contract_status_and_evidence()
    push!(checks, Check("Askervein",
                        "Required production LES, reference-comparison, and explicit-window output contract is satisfied",
                        askervein_contract_status,
                        askervein_contract_evidence))

    cadence_status, cadence_evidence =
        timeseries_cadence_status_and_evidence()
    push!(checks, Check("Time series",
                        "Completed Schar and complex-mountain production time series reach 6 h with 10 min cadence",
                        cadence_status,
                        cadence_evidence))

    timeseries_metric_status, timeseries_metric_evidence =
        production_timeseries_metric_contract_status_and_evidence()
    push!(checks, Check("Time series",
                        "Completed Schar and complex-mountain production time series include required robustness, reflection, high-k, and drag metrics",
                        timeseries_metric_status,
                        timeseries_metric_evidence))

    region_status, region_evidence =
        completed_case_region_coverage_status_and_evidence()
    push!(checks, Check("Metric regions",
                        "Completed Schar and complex-mountain comparison metrics cover required primary, full-domain, near-terrain, and 3D regions",
                        region_status,
                        region_evidence))

    schema_status, schema_evidence =
        direct_robustness_schema_status_and_evidence()
    push!(checks, Check("Metric schema",
                        "Required Schar and complex-mountain robustness metric schema is complete",
                        schema_status,
                        schema_evidence))

    robustness_status, robustness_evidence =
        finite_bottom_robustness_status_and_evidence()
    push!(checks, Check("Robustness",
                        "Completed Schar and complex-mountain production runs have finite values and no bottom-normal leakage",
                        robustness_status,
                        robustness_evidence))

    manifest_status, manifest_evidence =
        artifact_manifest_contract_status_and_evidence()
    push!(checks, Check("Artifact manifest",
                        "Final manifest contract is audited and incomplete fields are identified",
                        manifest_status,
                        manifest_evidence))

    completed_manifest_status, completed_manifest_evidence =
        completed_production_manifest_status_and_evidence()
    push!(checks, Check("Artifact manifest",
                        "Completed Schar and complex-mountain production artifacts have concrete manifest provenance",
                        completed_manifest_status,
                        completed_manifest_evidence))

    summary_status, summary_evidence =
        one_percent_summary_status_and_evidence()
    push!(checks, Check("Summary",
                        "Final markdown summary reports 1% pass/fail status for Schar, complex mountain, and Askervein",
                        summary_status,
                        summary_evidence))

    time_resolved_status, time_resolved_evidence =
        time_resolved_comparison_contract_status_and_evidence()
    push!(checks, Check("Time series",
                        "Plan-required saved-time reference comparison coverage is audited",
                        time_resolved_status,
                        time_resolved_evidence))

    resolution_status, resolution_evidence =
        resolution_adequacy_status_and_evidence()
    push!(checks, Check("Resolution",
                        "Production resolution adequacy is documented for completed cases and Askervein gaps are identified",
                        resolution_status,
                        resolution_evidence))

    return checks
end

function write_csv(checks)
    path = OUTPUT_PREFIX * ".csv"
    open(path, "w") do io
        println(io, "case,requirement,status,evidence")
        for check in checks
            println(io, join(csv_escape.((check.case_name, check.requirement,
                                          check.status, check.evidence)), ","))
        end
    end
    return path
end

function write_markdown(checks)
    path = OUTPUT_PREFIX * ".md"
    overall_status = all(check.status == "pass" || check.status == "present" for check in checks) ?
                     "pass" : "incomplete"

    open(path, "w") do io
        println(io, "# Terrain-Following Production Validation Gate")
        println(io)
        println(io, "Overall status: **$overall_status**")
        println(io)
        println(io, "Only `pass` production-validation checks can satisfy the validation goal. `present` checks are diagnostic coverage, not completion.")
        println(io)
        println(io, "| Case | Requirement | Status | Evidence |")
        println(io, "|---|---|---:|---|")
        for check in checks
            evidence = replace(check.evidence, "|" => "\\|")
            requirement = replace(check.requirement, "|" => "\\|")
            println(io, "| $(check.case_name) | $requirement | $(check.status) | $evidence |")
        end
    end
    return path
end

function main()
    checks = build_checks()
    csv_path = write_csv(checks)
    markdown_path = write_markdown(checks)
    pass_count = count(check -> check.status == "pass", checks)
    present_count = count(check -> check.status == "present", checks)
    fail_count = count(check -> check.status == "fail", checks)
    missing_count = count(check -> check.status == "missing", checks)
    blocked_count = count(check -> check.status == "blocked", checks)

    @info "wrote $csv_path"
    @info "wrote $markdown_path"
    @printf("production validation gate: pass=%d present=%d fail=%d missing=%d blocked=%d\n",
            pass_count, present_count, fail_count, missing_count, blocked_count)

    return fail_count == 0 && missing_count == 0 && blocked_count == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    success = main()
    exit(success ? 0 : 1)
end
