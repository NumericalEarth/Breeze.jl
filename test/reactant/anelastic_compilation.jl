include("compilation_setup.jl")

#####
##### Reactant compilation tests — AnelasticDynamics
#####
#
# The compressible compilation tests (centered_compilation.jl, weno5.jl) with the anelastic
# dynamics, whose time step includes the anelastic pressure solve. Not yet expected to pass:
# runtests.jl leaves this file out of the full suite, so run it by name,
#   julia --project=test test/runtests.jl reactant/anelastic_compilation

run_compilation_tests("Reactant AnelasticDynamics — Centered";
                      dynamics=AnelasticDynamics, topologies=centered_topologies)

run_weno_tests("WENO(order=5), AnelasticDynamics", WENO(order=5); dynamics=AnelasticDynamics)
