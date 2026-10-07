using MPI: mpiexec
using Test: @test, @testset

const MPI_WORKER = joinpath(@__DIR__, "..", "mpi", "acoustic_wave.jl")

function run_mpi_worker(nranks, partition)
    cmd = `$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 --project=$(Base.active_project()) $MPI_WORKER $partition`
    output = IOBuffer()
    process = run(pipeline(ignorestatus(cmd); stdout=output, stderr=output))
    return success(process), String(take!(output))
end

@testset "Distributed acoustic wave with CompressibleDynamics [MPI]" begin
    for (nranks, partition) in ((2, "x"), (2, "y"), (4, "xy"))
        @testset "Partition($(partition)) on $(nranks) ranks" begin
            succeeded, output = run_mpi_worker(nranks, partition)
            succeeded || println(output)
            @test succeeded
            @test occursin("DISTRIBUTED_ACOUSTIC_WAVE_OK", output)
        end
    end
end
