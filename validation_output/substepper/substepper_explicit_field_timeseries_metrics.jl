using Printf

const ROOT = @__DIR__
const ONE_PERCENT = 0.01

function parse_float(text)
    return parse(Float64, strip(text))
end

function read_numeric_csv(path)
    lines = readlines(path)
    isempty(lines) && error("empty CSV: $path")
    header = split(lines[1], ",")
    rows = Vector{Vector{Float64}}(undef, length(lines) - 1)
    for index in 2:length(lines)
        rows[index - 1] = parse_float.(split(lines[index], ","))
    end
    return header, rows
end

function column_index(header, name)
    index = findfirst(==(name), header)
    isnothing(index) && error("missing column $name in $(join(header, ", "))")
    return index
end

function read_snapshot_times(path)
    header, rows = read_numeric_csv(path)
    frame_index = column_index(header, "frame")
    time_index = if "time" in header
        column_index(header, "time")
    else
        column_index(header, "time_seconds")
    end
    return [(frame = round(Int, row[frame_index]), time = row[time_index])
            for row in rows]
end

function nearest_frame(target_time, frame_times; tolerance)
    best = first(frame_times)
    best_offset = abs(best.time - target_time)
    for frame_time in frame_times
        offset = abs(frame_time.time - target_time)
        if offset < best_offset
            best = frame_time
            best_offset = offset
        end
    end
    best_offset <= tolerance || return nothing
    return best, best_offset
end

function field_values(path, field)
    header, rows = read_numeric_csv(path)
    field_index = column_index(header, field)
    return [row[field_index] for row in rows]
end

function table_by_time(path)
    header, rows = read_numeric_csv(path)
    time_index = column_index(header, "time")
    by_time = Dict{Float64, Vector{Vector{Float64}}}()
    for row in rows
        push!(get!(by_time, row[time_index], Vector{Float64}[]), row)
    end
    return header, by_time
end

function values_from_rows(header, rows, field)
    field_index = column_index(header, field)
    return [row[field_index] for row in rows]
end

function metric_row(comparison, frame, reference_time, candidate_time,
                    region, field, reference_values, candidate_values)
    length(reference_values) == length(candidate_values) ||
        error("field length mismatch for $field at $reference_time")

    n = length(reference_values)
    diff = candidate_values .- reference_values
    reference_norm = sqrt(sum(abs2, reference_values))
    diff_norm = sqrt(sum(abs2, diff))
    reference_max = maximum(abs, reference_values)
    candidate_max = maximum(abs, candidate_values)
    diff_max = maximum(abs, diff)
    bias = sum(diff) / n
    rmse = sqrt(sum(abs2, diff) / n)

    relative_l2 = reference_norm == 0 ? (diff_norm == 0 ? 0.0 : Inf) :
                  diff_norm / reference_norm
    relative_linf = reference_max == 0 ? (diff_max == 0 ? 0.0 : Inf) :
                     diff_max / reference_max
    normalized_rmse = reference_max == 0 ? (rmse == 0 ? 0.0 : Inf) :
                      rmse / reference_max
    maximum_amplitude_error =
        reference_max == 0 ? (candidate_max == 0 ? 0.0 : Inf) :
        abs(candidate_max / reference_max - 1)

    reference_mean = sum(reference_values) / n
    candidate_mean = sum(candidate_values) / n
    reference_anomaly = reference_values .- reference_mean
    candidate_anomaly = candidate_values .- candidate_mean
    covariance = sum(reference_anomaly .* candidate_anomaly)
    reference_variance = sum(abs2, reference_anomaly)
    candidate_variance = sum(abs2, candidate_anomaly)
    pattern_correlation =
        reference_variance == 0 || candidate_variance == 0 ?
        (diff_norm == 0 ? 1.0 : 0.0) :
        covariance / sqrt(reference_variance * candidate_variance)

    one_percent_pass =
        relative_l2 <= ONE_PERCENT &&
        relative_linf <= ONE_PERCENT &&
        normalized_rmse <= ONE_PERCENT &&
        maximum_amplitude_error <= ONE_PERCENT &&
        pattern_correlation >= 0.99

    return (; comparison, frame,
              reference_time_seconds = reference_time,
              candidate_time_seconds = candidate_time,
              time_difference_seconds = candidate_time - reference_time,
              region, field, points = n, relative_l2, relative_linf, bias,
              normalized_rmse, maximum_amplitude_error,
              maximum_absolute_error = diff_max, pattern_correlation,
              one_percent_pass)
end

function write_metrics(path, rows)
    open(path, "w") do io
        println(io, "comparison,frame,reference_time_seconds,candidate_time_seconds,time_difference_seconds,region,field,points,relative_l2_error,relative_linf_error,bias,normalized_rmse,maximum_amplitude_error,maximum_absolute_error,pattern_correlation,one_percent_pass")
        for row in rows
            println(io, join((row.comparison, row.frame,
                              @sprintf("%.8g", row.reference_time_seconds),
                              @sprintf("%.8g", row.candidate_time_seconds),
                              @sprintf("%.8g", row.time_difference_seconds),
                              row.region, row.field, row.points,
                              @sprintf("%.16g", row.relative_l2),
                              @sprintf("%.16g", row.relative_linf),
                              @sprintf("%.16g", row.bias),
                              @sprintf("%.16g", row.normalized_rmse),
                              @sprintf("%.16g", row.maximum_amplitude_error),
                              @sprintf("%.16g", row.maximum_absolute_error),
                              @sprintf("%.16g", row.pattern_correlation),
                              row.one_percent_pass), ","))
        end
    end
end

function write_summary(path, rows)
    fail_rows = filter(row -> !row.one_percent_pass, rows)
    worst_l2 = maximum(row -> row.relative_l2, rows)
    worst_l2_row = rows[argmax([row.relative_l2 for row in rows])]
    worst_linf = maximum(row -> row.relative_linf, rows)
    worst_linf_row = rows[argmax([row.relative_linf for row in rows])]

    open(path, "w") do io
        println(io, "Saved-time field comparison summary")
        println(io)
        println(io, "rows = $(length(rows))")
        println(io, "fail_rows = $(length(fail_rows))")
        println(io, "one_percent_pass = $(isempty(fail_rows))")
        println(io, "worst_relative_l2_error = $worst_l2")
        println(io, "worst_relative_l2_field = $(worst_l2_row.field)")
        println(io, "worst_relative_l2_time = $(worst_l2_row.reference_time_seconds)")
        println(io, "worst_relative_linf_error = $worst_linf")
        println(io, "worst_relative_linf_field = $(worst_linf_row.field)")
        println(io, "worst_relative_linf_time = $(worst_linf_row.reference_time_seconds)")
    end
end

function schar_substepper_vs_explicit()
    explicit_dir = joinpath(ROOT, "terrain_schar_6h_400x200_production_explicit_field_snapshots",
                            "terrain_schar_mountain_wave_field_snapshot_csvs")
    substepper_dir = joinpath(ROOT, "terrain_schar_6h_400x200_production_substepper_field_snapshots",
                              "terrain_schar_mountain_wave_field_snapshot_csvs")

    explicit_times = read_snapshot_times(joinpath(explicit_dir, "snapshot_times.csv"))
    substepper_times = read_snapshot_times(joinpath(substepper_dir, "snapshot_times.csv"))
    fields = ["w_center", "pressure_perturbation"]
    rows = NamedTuple[]

    for substepper_time in substepper_times
        match = nearest_frame(substepper_time.time, explicit_times; tolerance = 5.0)
        isnothing(match) && continue
        explicit_time, _ = match
        explicit_path = joinpath(explicit_dir, @sprintf("field_snapshot_%04d.csv", explicit_time.frame))
        substepper_path = joinpath(substepper_dir, @sprintf("field_snapshot_%04d.csv", substepper_time.frame))

        for field in fields
            push!(rows,
                  metric_row("substepper_vs_explicit", substepper_time.frame,
                             explicit_time.time, substepper_time.time,
                             "centerline_slice", field,
                             field_values(explicit_path, field),
                             field_values(substepper_path, field)))
        end
    end

    metrics_path = joinpath(ROOT, "schar_6h_400x200_substepper_vs_explicit_field_timeseries_metrics.csv")
    summary_path = joinpath(ROOT, "schar_6h_400x200_substepper_vs_explicit_field_timeseries_metrics_summary.txt")
    write_metrics(metrics_path, rows)
    write_summary(summary_path, rows)
    return metrics_path, summary_path
end

function complex_substepper_vs_explicit()
    explicit_path = joinpath(ROOT, "complex_mountain_production_explicit",
                             "complex_mountain_centerline_snapshots.csv")
    substepper_path = joinpath(ROOT, "complex_mountain_production_substepper",
                               "complex_mountain_centerline_snapshots.csv")
    explicit_header, explicit_by_time = table_by_time(explicit_path)
    substepper_header, substepper_by_time = table_by_time(substepper_path)
    fields = ["u", "w_center", "theta_perturbation", "pressure_perturbation"]
    rows = NamedTuple[]

    times = sort(collect(intersect(keys(explicit_by_time), keys(substepper_by_time))))
    for (frame, time) in enumerate(times)
        for field in fields
            push!(rows,
                  metric_row("substepper_vs_explicit", frame, time, time,
                             "centerline_slice", field,
                             values_from_rows(explicit_header,
                                              explicit_by_time[time], field),
                             values_from_rows(substepper_header,
                                              substepper_by_time[time], field)))
        end
    end

    metrics_path = joinpath(ROOT, "complex_mountain_substepper_vs_explicit_field_timeseries_metrics.csv")
    summary_path = joinpath(ROOT, "complex_mountain_substepper_vs_explicit_field_timeseries_metrics_summary.txt")
    write_metrics(metrics_path, rows)
    write_summary(summary_path, rows)
    return metrics_path, summary_path
end

function main()
    schar_metrics, schar_summary = schar_substepper_vs_explicit()
    complex_metrics, complex_summary = complex_substepper_vs_explicit()
    @info "wrote $schar_metrics"
    @info "wrote $schar_summary"
    @info "wrote $complex_metrics"
    @info "wrote $complex_summary"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
