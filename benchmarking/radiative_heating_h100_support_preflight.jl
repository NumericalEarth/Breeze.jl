using Dates
using JSON

include(joinpath(@__DIR__, "radiative_heating_rcemip_benchmark.jl"))

function main()
    output_dir = get(ENV, "RADIATIVE_HEATING_H100_PREFLIGHT_DIR",
                     joinpath(@__DIR__, "results", "h100_validated_ecckd_preflight"))
    mkpath(output_dir)

    source = get(ENV, "RADIATIVE_HEATING_GAS_MODEL_SOURCE", "validated_ecCKD")
    gas_model = withenv("RADIATIVE_HEATING_GAS_MODEL_SOURCE" => source) do
        benchmark_gas_model(Float64)
    end
    support = gas_model_device_support("H100", gas_model.gas_optics)
    result = Dict(
        "case" => "radiative_heating_h100_support_preflight",
        "timestamp_utc" => string(now(UTC)),
        "status" => support.status == "supported" ? "supported" : "blocked",
        "backend" => "H100",
        "gas_model_source" => gas_model.source,
        "gas_model_kind" => gas_model.kind,
        "gas_model_accuracy_status" => gas_model.accuracy_status,
        "gas_model_device_support_status" => support.status,
        "gas_model_device_support_reason" => support.reason,
        "gas_model_device_support_source" => support.source,
        "missing_kernel_requirements" => support.missing_kernel_requirements,
        "next_required_implementation" => support.status == "supported" ?
            "none" :
            "Run and record CPU/GPU parity evidence for the wired official multi-gas EcCKDTabulatedGasOpticsModel H100 path.",
        "acceptance_unblocked_when" => support.status == "supported" ?
            "H100 support preflight passes" :
            "gas_model_device_support(\"H100\", validated_ecCKD.gas_optics) returns supported",
    )

    json_path = joinpath(output_dir, "radiative_heating_h100_support_preflight_latest.json")
    open(json_path, "w") do io
        JSON.print(io, result, 2)
        println(io)
    end

    md_path = joinpath(output_dir, "radiative_heating_h100_support_preflight_latest.md")
    open(md_path, "w") do io
        println(io, "# Radiative Heating H100 Support Preflight")
        println(io)
        println(io, "- status: ", result["status"])
        println(io, "- backend: ", result["backend"])
        println(io, "- gas model source: ", result["gas_model_source"])
        println(io, "- gas model kind: ", result["gas_model_kind"])
        println(io, "- gas model accuracy status: ", result["gas_model_accuracy_status"])
        println(io, "- gas model device support: ", result["gas_model_device_support_status"])
        println(io, "- reason: ", result["gas_model_device_support_reason"])
        println(io, "- source: ", result["gas_model_device_support_source"])
        println(io, "- missing kernel requirements:")
        for requirement in result["missing_kernel_requirements"]
            println(io, "  - ", requirement)
        end
        println(io, "- next required implementation: ", result["next_required_implementation"])
        println(io, "- acceptance unblocked when: ", result["acceptance_unblocked_when"])
    end

    println(json_path)
    return support.status == "supported" ? 0 : 2
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
