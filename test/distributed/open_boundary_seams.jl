include(joinpath(@__DIR__, "..", "mpi", "run_mpi_worker.jl"))

# The xy partition covers the seams of a y partition.
@testset "Distributed CompressibleDynamics with open lateral boundaries [MPI]" begin
    test_mpi_worker(joinpath(@__DIR__, "..", "mpi", "open_boundary_seams.jl"), "DISTRIBUTED_OPEN_BOUNDARY_SEAMS_OK";
                    configurations = ((2, "x"), (4, "xy")))
end
