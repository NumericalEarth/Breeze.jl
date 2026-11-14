#####
##### Shared launcher for the MPI worker scripts in this directory, included by the drivers in
##### `test/distributed/`. Not a test itself: `runtests.jl` skips everything under `mpi/`.
#####

using CUDA: CUDA
using MPI: mpiexec
using Test: @test, @testset

function run_mpi_worker(worker, nranks, partition)
    oversubscribed = nranks > Sys.CPU_THREADS
    if oversubscribed
        return nothing
    end
    cmd = `$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 --project=$(Base.active_project()) $worker $partition`
    output = IOBuffer()
    process = run(pipeline(ignorestatus(cmd); stdout=output, stderr=output))
    return success(process), String(take!(output))
end

# Launch `worker` once per `(nranks, partition)` and check that it succeeds and prints `marker`.
function test_mpi_worker(worker, marker; configurations = ((2, "x"), (2, "y"), (4, "xy")))
    for (nranks, partition) in configurations
        @testset "Partition($(partition)) on $(nranks) ranks" begin
            if CUDA.functional() && length(CUDA.devices()) < nranks
                continue
            end

            result = run_mpi_worker(worker, nranks, partition)
            if !isnothing(result)
                succeeded, output = result
                succeeded || println(output)
                @test succeeded
                @test occursin(marker, output)
            end
        end
    end
end
