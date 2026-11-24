include(joinpath(@__DIR__, "..", "mpi", "run_mpi_worker.jl"))

@testset "Distributed CompressibleDynamics with open lateral boundaries [MPI]" begin
    test_mpi_worker(joinpath(@__DIR__, "..", "mpi", "open_boundary_seams.jl"), "DISTRIBUTED_OPEN_BOUNDARY_SEAMS_OK")
end
