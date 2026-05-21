using Test

include(joinpath(@__DIR__, "..", "benchmarking", "radiative_heating_rcemip_benchmark.jl"))
include(joinpath(@__DIR__, "..", "benchmarking", "radiative_heating_h100_support_preflight.jl"))
include(joinpath(@__DIR__, "..", "benchmarking", "radiative_heating_tabulated_ecckd_parity.jl"))

function accepted_result(; status = "final_4x_evidence",
                         backend = "H100",
                         speedup = 4.1,
                         gas_model_source = "validated_ecCKD",
                         gas_model_accuracy_status = "passed",
                         gas_model_device_support_status = "supported")
    return Dict(
        "status" => status,
        "backend" => backend,
        "radiative_heating_runtime_supported" => true,
        "rrtmgp_runtime_supported" => true,
        "radiation_update_speedup" => speedup,
        "nsys_report" => "profile.nsys-rep",
        "ncu_report" => "profile.ncu-rep",
        "gas_model_source" => gas_model_source,
        "gas_model_accuracy_status" => gas_model_accuracy_status,
        "gas_model_device_support_status" => gas_model_device_support_status,
        "gas_model_device_support_reason" => "test fixture",
    )
end

@testset "RadiativeHeating reduced active table-entry moves" begin
    model = EcCKDTabulatedGasOpticsModel(
        gas_names = (:h2o,),
        pressure_grid = [1.0e4, 1.0e5],
        temperature_grid = [220.0, 300.0],
        h2o_mole_fraction_grid = [1.0e-3, 1.0e-2],
        gas_reference_mole_fractions = [0.0],
        longwave_absorption = fill(1.0e-8, 2, 1, 2, 2),
        shortwave_absorption = fill(2.0e-8, 2, 1, 2, 2),
        longwave_h2o_absorption = fill(1.0e-8, 2, 2, 2, 2),
        shortwave_h2o_absorption = fill(3.0e-8, 2, 2, 2, 2),
        shortwave_rayleigh_molar_scattering = fill(1.0e-8, 2),
        longwave_weights = fill(0.5, 2),
        shortwave_weights = fill(0.5, 2),
        longwave_source_scale = fill(1.0, 2),
    )
    moves = Any[
        Dict(
            "component" => "static_absorption",
            "local_gpoint_index" => 1,
            "gpoint" => 1,
            "gas_index" => 1,
            "pressure_index" => 1,
            "temperature_index" => 2,
            "h2o_index" => 0,
            "log_scale" => log(5.0),
        ),
        Dict(
            "component" => "dynamic_h2o",
            "local_gpoint_index" => 2,
            "gpoint" => 2,
            "gas_index" => 0,
            "pressure_index" => 2,
            "temperature_index" => 1,
            "h2o_index" => 2,
            "log_scale" => log(7.0),
        ),
    ]

    apply_reduced_preflight_active_table_entry_moves!(model, moves)

    @test model.shortwave_absorption[1, 1, 1, 2] ≈ 1.0e-7
    @test model.shortwave_absorption[1, 1, 1, 1] ≈ 2.0e-8
    @test model.shortwave_h2o_absorption[2, 2, 1, 2] ≈ 2.1e-7
    @test model.shortwave_h2o_absorption[2, 1, 1, 2] ≈ 3.0e-8

    if isfile(reduced_preflight_json()) && isfile(reduced_global_entry_json())
        preflight = JSON.parsefile(reduced_preflight_json())
        selected_moves = reduced_best_active_table_entry_moves(preflight)
        @test length(selected_moves) >= 3
    end
end

@testset "RadiativeHeating tabulated ecCKD CPU/GPU parity artifact" begin
    output_dir = mktempdir()
    withenv("RADIATIVE_HEATING_GAS_MODEL_SOURCE" => "validated_ecCKD",
            "RADIATIVE_HEATING_PARITY_DIR" => output_dir,
            "RADIATIVE_HEATING_PARITY_NX" => "1",
            "RADIATIVE_HEATING_PARITY_NY" => "1",
            "RADIATIVE_HEATING_PARITY_NZ" => "4") do
        code = parity_main()
        @test code in (0, 1, 2)
    end
    json_path = joinpath(output_dir, "radiative_heating_tabulated_ecckd_parity_latest.json")
    md_path = joinpath(output_dir, "radiative_heating_tabulated_ecckd_parity_latest.md")
    @test isfile(json_path)
    @test isfile(md_path)
    json = read(json_path, String)
    @test occursin("\"case\": \"radiative_heating_tabulated_ecckd_cpu_gpu_parity\"", json)
    @test occursin("\"gas_model_source\": \"validated_ecCKD\"", json)
    @test occursin("\"passed\":", json)
    @test occursin("\"status\":", json)
    @test occursin("\"reason\":", json)
    if !CUDA.functional()
        @test occursin("\"status\": \"blocked\"", json)
        @test occursin("CUDA.functional() is false", json)
    end
end

@testset "RadiativeHeating benchmark final acceptance" begin
    accepted, reason = final_acceptance(accepted_result())
    @test accepted
    @test reason == "none"

    accepted, reason = final_acceptance(accepted_result(status = "scaffold_not_final_4x_evidence"))
    @test !accepted
    @test occursin("not final_4x_evidence", reason)

    accepted, reason = final_acceptance(accepted_result(gas_model_source = "synthetic_fixed_coefficients"))
    @test !accepted
    @test occursin("synthetic fixed coefficients", reason)

    accepted, reason = final_acceptance(accepted_result(gas_model_accuracy_status = "failed_threshold"))
    @test !accepted
    @test occursin("not passed", reason)

    accepted, reason = final_acceptance(accepted_result(gas_model_device_support_status = "unsupported"))
    @test !accepted
    @test occursin("device support", reason)
end

@testset "RadiativeHeating benchmark gas model metadata" begin
    @test benchmark_status() == "scaffold_not_final_4x_evidence"
    withenv("RADIATIVE_HEATING_BENCHMARK_STATUS" => "final_4x_evidence") do
        @test benchmark_status() == "final_4x_evidence"
    end

    withenv("RADIATIVE_HEATING_GAS_MODEL_SOURCE" => "synthetic_fixed_coefficients",
            "RADIATIVE_HEATING_NG_LW" => "32",
            "RADIATIVE_HEATING_NG_SW" => "16") do
        gas_model = benchmark_gas_model(Float64)
        @test gas_model.source == "synthetic_fixed_coefficients"
        @test gas_model.kind == "fixed_ecCKD_32_lw_16_sw"
        @test gas_model.accuracy_status == "not_checked_scaffold"
        gas_values = benchmark_gas_values(gas_model)
        @test gas_values[:co2] == 400.0e-6
        @test !haskey(gas_values, :ch4)
        support = gas_model_device_support("H100", gas_model.gas_optics)
        @test support.status == "supported"
    end

    withenv("RADIATIVE_HEATING_GAS_MODEL_SOURCE" => "validated_ecCKD",
            "RADIATIVE_HEATING_TABULATED_ECCKD_PARITY_JSON" => nothing) do
        gas_model = benchmark_gas_model(Float64)
        @test gas_model.source == "validated_ecCKD"
        @test startswith(gas_model.kind, "official_ecCKD_")
        @test gas_model.accuracy_status == "passed"
        @test gas_model.gas_optics isa Lightflux.EcCKDTabulatedGasOpticsModel
        gas_values = benchmark_gas_values(gas_model)
        @test gas_values[:co2] == 420.0e-6
        @test gas_values[:ch4] == 1.8e-6
        @test gas_values[:n2o] == 330.0e-9
        @test gas_values[:cfc11] > 0
        @test gas_values[:cfc12] > 0
        cpu_support = gas_model_device_support("CPU", gas_model.gas_optics)
        @test cpu_support.status == "supported"
        h100_support = gas_model_device_support("H100", gas_model.gas_optics)
        @test h100_support.status == "unsupported"
        @test occursin("tabulated", h100_support.reason)
        @test !any(contains("tabulated absorption lookup accumulation"),
                   h100_support.missing_kernel_requirements)
        @test !any(contains("Rayleigh"),
                   h100_support.missing_kernel_requirements)
        @test !any(contains("source-table"),
                   h100_support.missing_kernel_requirements)
        @test !any(contains("allocation-free"),
                   h100_support.missing_kernel_requirements)
        @test !any(contains("radiation update path"),
                   h100_support.missing_kernel_requirements)
        @test !any(contains("column-molar-amount"),
                   h100_support.missing_kernel_requirements)
        @test !any(contains("dynamic h2o lookup-table interpolation"),
                   h100_support.missing_kernel_requirements)
        @test h100_support.source == "BreezeLightfluxExt"

        parity_path = joinpath(mktempdir(), "passed_parity.json")
        write(parity_path, """
        {
          "case": "radiative_heating_tabulated_ecckd_cpu_gpu_parity",
          "status": "passed",
          "passed": true,
          "gas_model_kind": "$(gas_model.kind)",
          "gas_model_source": "validated_ecCKD"
        }
        """)
        withenv("RADIATIVE_HEATING_TABULATED_ECCKD_PARITY_JSON" => parity_path) do
            supported = gas_model_device_support("H100", gas_model.gas_optics)
            @test supported.status == "supported"
            @test isempty(supported.missing_kernel_requirements)
            @test occursin("parity", supported.reason)
        end
    end
end

@testset "RadiativeHeating H100 final-gate preflight" begin
    withenv("RADIATIVE_HEATING_GAS_MODEL_SOURCE" => "validated_ecCKD",
            "RADIATIVE_HEATING_TABULATED_ECCKD_PARITY_JSON" => nothing) do
        gas_model = benchmark_gas_model(Float64)
        support = gas_model_device_support("H100", gas_model.gas_optics)
        @test support.status == "unsupported"
        @test occursin("parity", support.reason)
    end

    output_dir = mktempdir()
    withenv("RADIATIVE_HEATING_GAS_MODEL_SOURCE" => "validated_ecCKD",
            "RADIATIVE_HEATING_H100_PREFLIGHT_DIR" => output_dir) do
        @test main() == 2
    end
    json_path = joinpath(output_dir, "radiative_heating_h100_support_preflight_latest.json")
    @test isfile(json_path)
    json = read(json_path, String)
    @test occursin("\"status\": \"blocked\"", json)
    @test occursin("\"gas_model_device_support_status\": \"unsupported\"", json)
    @test occursin("\"gas_model_device_support_source\": \"BreezeLightfluxExt\"", json)
    @test occursin("\"next_required_implementation\"", json)
    @test occursin("EcCKDTabulatedGasOpticsModel", json)
    @test occursin("\"missing_kernel_requirements\"", json)
    @test occursin("CPU/GPU parity smoke test", json)
end
