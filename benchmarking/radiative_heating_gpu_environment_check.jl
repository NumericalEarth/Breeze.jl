using Dates

function json_escape(text)
    return replace(String(text), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n")
end

function json_value(value)
    if value isa AbstractString
        return "\"" * json_escape(value) * "\""
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Integer || value isa AbstractFloat
        return string(value)
    elseif value isa AbstractDict
        return json_object(value)
    elseif value isa AbstractVector || value isa Tuple
        return "[" * join(json_value.(value), ", ") * "]"
    else
        return "\"" * json_escape(sprint(show, value)) * "\""
    end
end

function json_object(object)
    pairs = collect(object)
    lines = ["{"]
    for (i, pair) in enumerate(pairs)
        key, value = pair
        comma = i == length(pairs) ? "" : ","
        push!(lines, "  \"$(json_escape(key))\": $(json_value(value))$(comma)")
    end
    push!(lines, "}")
    return join(lines, "\n")
end

function command_status(cmd)
    try
        output = read(cmd, String)
        return true, strip(output)
    catch err
        return false, sprint(showerror, err)
    end
end

function cuda_status()
    try
        cuda = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
        functional = Base.invokelatest(getproperty(cuda, :functional))
        devices = functional ? Base.invokelatest(() -> collect(getproperty(cuda, :devices)())) : []
        names = String[]
        for device in devices
            push!(names, String(Base.invokelatest(getproperty(cuda, :name), device)))
        end
        return Dict(
            "loadable" => true,
            "functional" => functional,
            "device_count" => length(devices),
            "device_names" => names,
        )
    catch err
        return Dict(
            "loadable" => false,
            "functional" => false,
            "device_count" => 0,
            "device_names" => String[],
            "error" => sprint(showerror, err),
        )
    end
end

function main()
    output_dir = get(ENV, "RADIATIVE_HEATING_GPU_ENV_DIR",
                     joinpath(@__DIR__, "results", "gpu_environment"))
    mkpath(output_dir)

    nvidia_ok, nvidia_output = command_status(`nvidia-smi -L`)
    nsys_ok, nsys_output = command_status(`nsys --version`)
    ncu_ok, ncu_output = command_status(`ncu --version`)
    cuda = cuda_status()

    h100_detected = any(contains("H100"), get(cuda, "device_names", String[])) ||
                    contains(nvidia_output, "H100")
    ready = nvidia_ok &&
            get(cuda, "functional", false) === true &&
            h100_detected &&
            nsys_ok &&
            ncu_ok

    result = Dict(
        "status" => ready ? "ready_for_h100_nsight_gate" : "not_ready",
        "timestamp_utc" => string(now(UTC)),
        "host" => gethostname(),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
        "slurm_step_id" => get(ENV, "SLURM_STEP_ID", ""),
        "h100_detected" => h100_detected,
        "nvidia_smi" => Dict("ok" => nvidia_ok, "output" => nvidia_output),
        "cuda" => cuda,
        "nsys" => Dict("ok" => nsys_ok, "output" => nsys_output),
        "ncu" => Dict("ok" => ncu_ok, "output" => ncu_output),
    )

    json_path = joinpath(output_dir, "radiative_heating_gpu_environment_latest.json")
    open(json_path, "w") do io
        println(io, json_object(result))
    end

    md_path = joinpath(output_dir, "radiative_heating_gpu_environment_latest.md")
    open(md_path, "w") do io
        println(io, "# Radiative Heating GPU Environment")
        println(io)
        println(io, "- status: ", result["status"])
        println(io, "- host: ", result["host"])
        println(io, "- slurm_job_id: ", result["slurm_job_id"])
        println(io, "- H100 detected: ", result["h100_detected"])
        println(io, "- nvidia-smi: ", nvidia_ok)
        println(io, "- CUDA functional: ", get(cuda, "functional", false))
        println(io, "- Nsight Systems: ", nsys_ok)
        println(io, "- Nsight Compute: ", ncu_ok)
    end

    println(json_path)
    return ready ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
