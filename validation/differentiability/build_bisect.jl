# Locate where building a small Breeze model on ReactantState exhausts memory.
#   julia --project=~/droplet-tasks-env build_bisect.jl variant
# variants: boussinesq_ppp, boussinesq_ppb, compressible_ppp, anelastic_ppp
using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using CUDA
using Reactant
using Printf
Reactant.set_default_backend("cpu")
variant = ARGS[1]
stage(s) = (@printf("[%6.1f s, maxrss %.1f GB] %s\n", time() - t₀, Sys.maxrss() / 1e9, s); flush(stdout))
t₀ = time()
L = 0.02
topology = endswith(variant, "ppb") ? (Periodic, Periodic, Bounded) : (Periodic, Periodic, Periodic)
grid = RectilinearGrid(ReactantState(); size=(8, 8, 8), x=(0, L), y=(0, L), z=(0, L), topology)
stage("grid")
if startswith(variant, "boussinesq")
    dynamics = BoussinesqDynamics(grid; pressure=85000.0, temperature=282.0)
elseif startswith(variant, "anelastic")
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=85000.0, potential_temperature=282.0)
    dynamics = AnelasticDynamics(reference_state)
else
    dynamics = CompressibleDynamics()
end
stage("dynamics $(summary(dynamics))")
if get(ENV, "PROFILE_BUILD", "0") == "1"
    # Print the call stack every 20 s to see where construction spends its time
    Threads.@spawn while true
        sleep(20)
        stage("still building")
    end
end
model = AtmosphereModel(grid; dynamics)
stage("model built")
