include(joinpath(@__DIR__, "..", "mpi", "run_mpi_worker.jl"))

@testset "Distributed tracer transport with CompressibleDynamics [MPI]" begin
    test_mpi_worker(joinpath(@__DIR__, "..", "mpi", "tracer_transport.jl"), "DISTRIBUTED_TRACER_TRANSPORT_OK";
                    configurations = ((2, "x"), (4, "xy")))
end
