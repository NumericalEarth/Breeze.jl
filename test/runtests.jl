import Breeze
using ParallelTestRunner: find_tests, parse_args, filter_tests!, runtests

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# Parse arguments
args = parse_args(ARGS)

if filter_tests!(testsuite, args)
    # Skip this one for the time being
    delete!(testsuite, "anelastic_pressure_solver")
end

runtests(Breeze, args; testsuite)
