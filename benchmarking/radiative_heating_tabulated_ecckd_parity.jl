using JSON

if !@isdefined(benchmark_gas_model)
    include(joinpath(@__DIR__, "radiative_heating_rcemip_benchmark.jl"))
end

const PARITY_FIELDS = (
    :upwelling_longwave_flux,
    :downwelling_longwave_flux,
    :downwelling_shortwave_flux,
    :flux_divergence,
)

function radiation_field_arrays(radiation)
    return Dict(
        String(name) => Array(interior(getproperty(radiation, name)))
        for name in PARITY_FIELDS
    )
end

function field_error(candidate, reference)
    difference = candidate .- reference
    return Dict(
        "max_abs" => maximum(abs, difference),
        "rmse" => sqrt(sum(abs2, difference) / length(difference)),
        "reference_max_abs" => maximum(abs, reference),
    )
end

function parity_errors(cpu_arrays, gpu_arrays)
    return Dict(
        name => field_error(gpu_arrays[name], cpu_arrays[name])
        for name in keys(cpu_arrays)
    )
end

function parity_passed(errors; atol, rtol)
    for (name, metrics) in errors
        threshold = atol + rtol * metrics["reference_max_abs"]
        metrics["max_abs"] <= threshold || return false
    end
    return true
end

function build_radiative_heating_model(arch, FT, Nx, Ny, Nz, gas_model, gas_values)
    grid = rcemip_grid(arch, FT, Nx, Ny, Nz)
    constants = Breeze.Thermodynamics.ThermodynamicConstants()
    radiation = Base.invokelatest(() ->
        RadiativeTransferModel(grid, RadiativeHeatingOptics(), constants;
                               gas_optics = gas_model.gas_optics,
                               gas_values,
                               surface_temperature = 300,
                               surface_albedo = 0.07,
                               surface_emissivity = 0.98,
                               solar_constant = 551))
    model = build_model(radiation, grid, constants)
    Base.invokelatest(update_radiation!, model.radiation, model)
    return model
end

function write_parity_artifact(result, output_dir)
    mkpath(output_dir)
    json_path = joinpath(output_dir, "radiative_heating_tabulated_ecckd_parity_latest.json")
    open(json_path, "w") do io
        JSON.print(io, result, 2)
        println(io)
    end

    md_path = joinpath(output_dir, "radiative_heating_tabulated_ecckd_parity_latest.md")
    open(md_path, "w") do io
        println(io, "# Radiative Heating Tabulated ecCKD CPU/GPU Parity")
        println(io)
        println(io, "- status: ", result["status"])
        println(io, "- backend: ", result["backend"])
        println(io, "- gas model source: ", result["gas_model_source"])
        println(io, "- gas model kind: ", result["gas_model_kind"])
        println(io, "- grid: ", result["grid"]["nx"], " x ", result["grid"]["ny"], " x ", result["grid"]["nz"])
        println(io, "- tolerance atol: ", result["atol"])
        println(io, "- tolerance rtol: ", result["rtol"])
        println(io, "- passed: ", result["passed"])
        println(io, "- reason: ", result["reason"])
        if haskey(result, "field_errors")
            println(io, "- field errors:")
            for name in sort(collect(keys(result["field_errors"])))
                metrics = result["field_errors"][name]
                println(io, "  - ", name, ": max_abs=", metrics["max_abs"],
                        ", rmse=", metrics["rmse"],
                        ", reference_max_abs=", metrics["reference_max_abs"])
            end
        end
    end
    return json_path
end

function parity_main()
    FT = Float64
    Nx = envint("RADIATIVE_HEATING_PARITY_NX", 2)
    Ny = envint("RADIATIVE_HEATING_PARITY_NY", 2)
    Nz = envint("RADIATIVE_HEATING_PARITY_NZ", 8)
    atol = envfloat("RADIATIVE_HEATING_PARITY_ATOL", 1.0e-8)
    rtol = envfloat("RADIATIVE_HEATING_PARITY_RTOL", 1.0e-8)
    output_dir = get(ENV, "RADIATIVE_HEATING_PARITY_DIR",
                     joinpath(@__DIR__, "results", "tabulated_ecckd_parity"))

    source = get(ENV, "RADIATIVE_HEATING_GAS_MODEL_SOURCE", "validated_ecCKD")
    result = Dict(
        "case" => "radiative_heating_tabulated_ecckd_cpu_gpu_parity",
        "timestamp_utc" => string(now(UTC)),
        "backend" => "H100",
        "grid" => Dict("nx" => Nx, "ny" => Ny, "nz" => Nz, "columns" => Nx * Ny),
        "atol" => atol,
        "rtol" => rtol,
        "gas_model_source" => source,
        "gas_model_kind" => "",
        "passed" => false,
        "status" => "blocked",
        "reason" => "",
    )

    if !CUDA.functional()
        result["reason"] = "CUDA.functional() is false; cannot run CPU/GPU parity smoke test"
        path = write_parity_artifact(result, output_dir)
        println(path)
        return 2
    end

    try
        gas_model = withenv("RADIATIVE_HEATING_GAS_MODEL_SOURCE" => source) do
            benchmark_gas_model(FT)
        end
        result["gas_model_source"] = gas_model.source
        result["gas_model_kind"] = gas_model.kind
        gas_values = benchmark_gas_values(gas_model)

        cpu_model = build_radiative_heating_model(CPU(), FT, Nx, Ny, Nz, gas_model, gas_values)
        gpu_model = build_radiative_heating_model(GPU(), FT, Nx, Ny, Nz, gas_model, gas_values)
        CUDA.synchronize()

        cpu_arrays = radiation_field_arrays(cpu_model.radiation)
        gpu_arrays = radiation_field_arrays(gpu_model.radiation)
        errors = parity_errors(cpu_arrays, gpu_arrays)
        passed = parity_passed(errors; atol, rtol)
        result["field_errors"] = errors
        result["passed"] = passed
        result["status"] = passed ? "passed" : "failed_threshold"
        result["reason"] = passed ? "CPU/GPU field parity passed" :
            "one or more radiation fields exceeded parity tolerance"
    catch err
        result["status"] = "failed"
        result["reason"] = sprint(showerror, err)
    end

    path = write_parity_artifact(result, output_dir)
    println(path)
    return result["status"] == "passed" ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(parity_main())
end
