using Dates
using Printf

function main()
    project_dir = dirname(Base.active_project())
    script_path = abspath(joinpath(@__DIR__, "rising_thermal_bubble_forward.jl"))

    profile_run_stamp = get(ENV, "BREEZE_PROFILE_RUN_STAMP", Dates.format(now(Dates.UTC), "yyyy-mm-dd_HHMMSS"))
    profile_output_dir = joinpath(project_dir, "reactant_forward_profiles", "run_" * profile_run_stamp)
    mkpath(profile_output_dir)

    # Keep this list in sync with rising_thermal_bubble_forward.jl.
    grid_sizes = [(128, 128, 128), (256, 256, 128), (512, 512, 128), (1024, 1024, 128)]

    ncu_bin = Sys.which("ncu")
    if ncu_bin === nothing
        error("`ncu` not found on PATH. Install Nsight Compute or add `ncu` to PATH.")
    end

    for (grid_idx, (Nx, Ny, Nz)) in enumerate(grid_sizes)
        grid_label = @sprintf("%dx%dx%d", Nx, Ny, Nz)
        ncu_report_base = joinpath(profile_output_dir, "julia_ncu_profile_" * grid_label)
        ncu_log_path = ncu_report_base * ".log"
        @info "Launching ncu child run" grid_label ncu_log_path ncu_report_base

        ncu_cmd = `$(ncu_bin) --target-processes all --kernel-name-base demangled --set basic --replay-mode application --log-file $(ncu_log_path) --export $(ncu_report_base) $(Base.julia_cmd()) --project=$(project_dir) $(script_path)`

        child_env = copy(ENV)
        child_env["BREEZE_PROFILE_RUN_STAMP"] = profile_run_stamp
        child_env["BREEZE_NCU_GRID_IDX"] = string(grid_idx)
        child_env["BREEZE_DISABLE_REACTANT_PROFILE"] = "true"
        child_env["BREEZE_DISABLE_BENCHMARK"] = "true"
        child_env["BREEZE_DISABLE_VISUALIZATION"] = "true"

        run(setenv(ncu_cmd, child_env))
    end

    @info "Finished per-grid ncu profiling run" profile_output_dir
end

main()

