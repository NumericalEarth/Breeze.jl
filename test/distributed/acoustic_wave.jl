using MPI: mpiexec
using Test: @test, @testset

const MPI_WORKER = joinpath(@__DIR__, "..", "mpi", "acoustic_wave.jl")

function run_mpi_worker(nranks, partition)
    oversubscribed = nranks > Sys.CPU_THREADS
    if oversubscribed
        return nothing
    end
    cmd = `$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 --project=$(Base.active_project()) $MPI_WORKER $partition`
    output = IOBuffer()
    process = run(pipeline(ignorestatus(cmd); stdout=output, stderr=output))
    return success(process), String(take!(output))
end

@testset "Distributed acoustic wave with CompressibleDynamics [MPI]" begin
    for (nranks, partition) in ((2, "x"), (2, "y"), (4, "xy"))
        @testset "Partition($(partition)) on $(nranks) ranks" begin
            result = run_mpi_worker(nranks, partition)
            if !isnothing(result)
                succeeded, output = result
                succeeded || println(output)
                @test succeeded
                @test occursin("DISTRIBUTED_ACOUSTIC_WAVE_OK", output)
            end
        end
    end
end
