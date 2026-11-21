include(joinpath(@__DIR__, "..", "mpi", "run_mpi_worker.jl"))

@testset "Distributed turbulence closure fields with CompressibleDynamics [MPI]" begin
    test_mpi_worker(joinpath(@__DIR__, "..", "mpi", "closure_fields.jl"), "DISTRIBUTED_CLOSURE_FIELDS_OK";
                    configurations = ((2, "x"), (4, "xy")))
end
