include(joinpath(@__DIR__, "..", "mpi", "run_mpi_worker.jl"))

@testset "Distributed acoustic wave with CompressibleDynamics [MPI]" begin
    test_mpi_worker(joinpath(@__DIR__, "..", "mpi", "acoustic_wave.jl"), "DISTRIBUTED_ACOUSTIC_WAVE_OK")
end
