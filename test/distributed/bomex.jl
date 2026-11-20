include(joinpath(@__DIR__, "..", "mpi", "run_mpi_worker.jl"))

@testset "Distributed BOMEX with AnelasticDynamics [MPI]" begin
    test_mpi_worker(joinpath(@__DIR__, "..", "mpi", "bomex.jl"), "DISTRIBUTED_BOMEX_OK")
end
