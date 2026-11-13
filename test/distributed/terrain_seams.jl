include(joinpath(@__DIR__, "..", "mpi", "run_mpi_worker.jl"))

@testset "Distributed terrain-following CompressibleDynamics seams [MPI]" begin
    test_mpi_worker(joinpath(@__DIR__, "..", "mpi", "terrain_seams.jl"), "DISTRIBUTED_TERRAIN_SEAMS_OK")
end
