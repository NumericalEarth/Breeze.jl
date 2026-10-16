# Experimental Reactant check of the native column kernel; no LES integration.
# Run with the same BreezeNumericalRadiationExt environment as the included check.
# Develop both local Breeze-radiation and NumericalRadiation.jl in an isolated
# environment to include their separate layer/interface storage fixes.
# Current result: Reactant fails to raise an llvm.getelementptr table lookup to
# StableHLO. This is a retained reproducer, not a passing adjoint validation.
include("ena_radiation_differentiability.jl")
using Reactant
using CUDA
Reactant.set_default_backend("cpu")

function radiation_kernel_loss!(radiation, grid, temperature)
    radiation.atmospheric_state.temperature_layers .= temperature
    ENARadiationExt.solve_spectral_columns!(radiation, grid)
    return sum(radiation.atmospheric_state.longwave_up .^ 2) / 1000
end

function radiation_kernel_gradient!(radiation, dradiation, grid, temperature, dtemperature)
    Enzyme.remake_zero!(dradiation)
    dtemperature .= 0
    _, value = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        radiation_kernel_loss!, Enzyme.Active, Enzyme.Duplicated(radiation, dradiation),
        Enzyme.Const(grid), Enzyme.Duplicated(temperature, dtemperature))
    return value, dtemperature
end

@info "Building native column on CPU"
grid, radiation = ena_radiation_column(CPU(), Float32)
ena_stage_radiation_column!(radiation.atmospheric_state, zeros(Float32, 5))
@info "Constructing Reactant grid and adapting radiation"
reactant_grid = RectilinearGrid(Oceananigans.Architectures.ReactantState(), Float32;
    size=16, z=(0,16000), topology=(Flat,Flat,Bounded))
reactant_radiation = Reactant.to_rarray(radiation)
temperature = Reactant.to_rarray(copy(radiation.atmospheric_state.temperature_layers))
dtemperature = zero(temperature)
dradiation = Enzyme.make_zero(reactant_radiation)
@info "Compiling native radiation kernel reverse derivative"
task = Task(() -> Reactant.@compile(raise=true, raise_first=true, sync=true,
    radiation_kernel_gradient!(reactant_radiation, dradiation, reactant_grid, temperature, dtemperature)), 64 * 1024^2)
schedule(task)
compiled = fetch(task)
@info "Running compiled reverse derivative"
loss, gradient = compiled(reactant_radiation, dradiation, reactant_grid, temperature, dtemperature)
@show loss Array(gradient)
@test all(isfinite, Array(gradient))
@test any(!iszero, Array(gradient))

# Compare the summed temperature gradient with an independent directional
# finite difference of the same native kernel on CPU.
baseline = copy(radiation.atmospheric_state.temperature_layers)
step = 0.1f0
plus = radiation_kernel_loss!(radiation, grid, baseline .+ step)
minus = radiation_kernel_loss!(radiation, grid, baseline .- step)
finite_difference = (plus - minus) / (2step)
automatic_derivative = sum(Array(gradient))
@test automatic_derivative ≈ finite_difference rtol=0.01 atol=1e-4
@show automatic_derivative finite_difference
directory = joinpath(@__DIR__, "ena_output", "optimization", "radiation")
ena_record_radiation_configuration(directory)
write(joinpath(directory, "reactant_gradient_check.csv"),
      "automatic_derivative,finite_difference\n$automatic_derivative,$finite_difference\n")
@info "Reactant native radiation derivative and finite difference passed"
