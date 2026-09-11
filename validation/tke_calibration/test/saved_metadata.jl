using JLD2, Test
include(joinpath(@__DIR__, "..", "scripts", "saved_metadata.jl"))

@testset "Metadata from packages absent in the analysis process" begin
    mktempdir() do directory
        path = joinpath(directory, "metadata.jld2")
        # Write actual foreign types in a separate process. Native NamedTuple-only fixtures do not
        # reproduce JLD2's wrapper, which broke analysis of otherwise compatible production files.
        code = """
        using JLD2
        module SavedCalibrationFixture
            struct MoistStability end
            struct DryStability end
        end
        using .SavedCalibrationFixture
        config = (; dt = 7.5, windows = [(0.0, 10.0)], stability = [SavedCalibrationFixture.MoistStability()])
        dry = merge(config, (; stability = [SavedCalibrationFixture.DryStability()]))
        jldsave(ARGS[1]; config, dry)
        """
        project = dirname(Base.active_project())
        run(`$(Base.julia_cmd()) --startup-file=no --compiled-modules=existing --project=$project -e $code $path`)
        first_load = load(path)
        second_load = load(path)
        a = saved_metadata_value(first_load["config"])
        b = saved_metadata_value(second_load["config"])
        @test a isa NamedTuple
        @test a == b
        @test a.dt == 7.5
        @test a.windows == [(0.0, 10.0)]
        @test occursin("MoistStability", only(a.stability).saved_type)
        @test a != saved_metadata_value(first_load["dry"])
        @test a != merge(b, (; dt = 15.0))
        @test saved_metadata_value(Dict("configuration" => first_load["config"]))["configuration"] == a
    end
end
