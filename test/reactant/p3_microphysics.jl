include("microphysics_compilation_setup.jl")

microphysics = P3Microphysics()

initial_state = (; qᵛ=0.01, qᶜˡ=1e-4, qʳ=1e-5, nʳ=1e6, qⁱ=1e-5, nⁱ=1e5)

run_microphysics_tests("P3Microphysics", microphysics, initial_state)
