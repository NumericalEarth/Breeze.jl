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

# Every configuration starts fresh processes that compile the whole model again, and the drivers run
# serially, before the parallel tests. So only the last configuration that fits this machine runs
# (the one on the most ranks, which partitions both directions where there are four threads),
# and all of them in the exhaustive tests (BREEZE_TEST_EXHAUSTIVE, set on one CI runner).
function swept_configurations(configurations)
    get(ENV, "BREEZE_TEST_EXHAUSTIVE", "false") == "true" && return configurations
    fitting = filter(((nranks, _),) -> nranks ≤ Sys.CPU_THREADS, configurations)
    return isempty(fitting) ? configurations[1:1] : (last(fitting),)
end

# Launch `worker` once per `(nranks, partition)` and check that it succeeds and prints `marker`.
function test_mpi_worker(worker, marker; configurations = ((2, "x"), (2, "y"), (4, "xy")))
    for (nranks, partition) in swept_configurations(configurations)
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
