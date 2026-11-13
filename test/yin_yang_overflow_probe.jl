# TEMPORARY (#1103 CI stack overflow): run with `--trace-compile=stderr`, so the last methods
# compiled before the overflow appear on stderr
using Breeze
using Oceananigans
using Oceananigans.MultiRegion: YinYangGrid
using Oceananigans.Units

stack_size = parse(Int, ARGS[1])

grid = YinYangGrid(CPU(); size=(48, 16, 16), z=(0, 3kilometers), halo=(3, 3, 3))
dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5,
                                reference_potential_temperature = 300)
boundary_conditions = (; ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(0.3)))
model = AtmosphereModel(grid; dynamics, boundary_conditions, closure = TKEBasedTurbulenceClosure(),
                        advection = WENO(order=5))
set!(model; θ = (λ, φ, z) -> 300 + 3e-3 * max(z - 1000, 0), u = (λ, φ, z) -> 10 * cosd(φ) * z / 3000,
     ρ = HydrostaticallyBalancedDensity())

println(stderr, "PROBE: stepping in a task with a ", stack_size, "-byte stack")
task = Task(() -> time_step!(model, 60), stack_size)
schedule(task)
try
    wait(task)
    println(stderr, "PROBE: time step completed")
catch e
    println(stderr, "PROBE: ", typeof(e.task.exception))
end
