import Breeze
using ParallelTestRunner: find_tests, parse_args, filter_tests!, runtests, available_memory

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# Parse arguments
args = parse_args(ARGS)

const REACTANT_COMPAT = VERSION < v"1.13-" && Base.JLOptions().check_bounds != 1

# These aren't test files, they are only used as setup for other tests.
delete!(testsuite, "setup")
delete!(testsuite, "reactant/weno_compilation_setup")

if filter_tests!(testsuite, args)
    # Reactant compilation tests require --check-bounds=auto (Reactant/Enzyme
    # limitation).
    if !REACTANT_COMPAT
        for key in keys(testsuite)
            if startswith(key, "reactant/") && !REACTANT_COMPAT
                delete!(testsuite, key)
            end
        end
    end
end

if Sys.isapple() && get(ENV, "GITHUB_ACTIONS", "false") == "true"
    GC.gc(true); GC.gc(false); GC.gc(true)
    # macOS runners have little memory compared to the other runners, let's set a more
    # conservative limit to memory usage before reciclying a worker, to avoid trashing
    # memory or out-of-memory errors.  We set the memory limit dynamically, based on the
    # currently available memory (with a ~20% margin), with a lower bound of 1700 MiB.
    max_rss_memory = max(1_700, round(Int, available_memory() / 2 ^ 20 / 2  * 0.8))
    ENV["JULIA_TEST_MAXRSS_MB"] = string(max_rss_memory)
end

runtests(Breeze, args; testsuite)
