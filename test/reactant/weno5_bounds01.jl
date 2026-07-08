include("weno_compilation_setup.jl")

run_weno_tests("WENO(order=5, bounds=(0,1))", WENO(order=5, bounds=(0, 1)))
