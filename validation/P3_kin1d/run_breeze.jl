# Run the kinematic example without plotting, exporting a common comparison format.
# Usage: julia --project=<environment> run_breeze.jl [case] [dt]
# The setup is read from the example so changes to that case remain visible here.
using Oceananigans
using Breeze
using DelimitedFiles
using Printf
using SHA
using Breeze.Thermodynamics: FlatauPolynomial
using Breeze.Microphysics.PredictedParticleProperties: ProcessRate

case = get(ARGS, 1, "fixed")
Δt = parse(Float64, get(ARGS, 2, "1"))
case in ("reservoir", "fixed", "upwind", "reference_options", "dry") || error("Unknown case: $case")
output_directory = get(ENV, "P3_COMPARISON_OUTPUT",
                       normpath(joinpath(@__DIR__, "../../validation_output/p3_kin1d")))
mkpath(output_directory)
example = normpath(joinpath(@__DIR__, "../../examples/kinematic_driver.jl"))
setup = first(split(read(example, String), "# ## Simulation and output"))
if case != "reservoir"
    setup = replace(setup, "prognostic=true" => "prognostic=false")
    # A fixed aerosol distribution has no transported aerosol field or inflow BC.
    setup = replace(setup, r"       ρnᵃ = FieldBoundaryConditions[^\n]*\n" => "")
end
if case in ("upwind", "reference_options", "dry")
    setup = replace(setup, "WENO(order=5)" => "UpwindBiased(order=1)")
end
if case == "dry"
    setup = replace(setup, "ℋ₀ = 0.8" => "ℋ₀ = 0")
end
if case == "reference_options"
    setup = replace(setup, "constants = ThermodynamicConstants()" =>
                           "constants = ThermodynamicConstants(; saturation_vapor_pressure=FlatauPolynomial())")
    setup = replace(setup, "microphysics = P3Microphysics(; aerosol, predict_supersaturation=true)" =>
        "process_rates = ProcessRate(; thermodynamic_constants=constants, predict_supersaturation=true, " *
        "sink_limiting_timescale=Δt, ice_nucleation_timescale=Δt, homogeneous_freezing_timescale=Δt)\n" *
        "microphysics = P3Microphysics(; aerosol, process_rates, thermodynamic_constants=constants)")
    # Preserve the original vapor inflow when changing the saturation formulation.
    inflow = vec(readdlm(joinpath(output_directory, "inflow.dat")))
    setup = replace(setup, "qᵗ₀ = ℋ₀ * qᵛ⁺ˡ₀" => "qᵗ₀ = $(inflow[2])")
end
include_string(Main, setup, example)
if case == "reference_options"
    original = reverse(readdlm(joinpath(output_directory, "initial_state.dat")); dims=1)
    set!(model; T=reshape(original[:, 3], 1, 1, grid.Nz),
                qᵗ=reshape(original[:, 5], 1, 1, grid.Nz))
    set!(model; ρsᵛ⁺ˡ=ρ * (μ.qᵛ - qᵛ⁺ˡ))
end

profile(field) = Array(field[1, 1, 1:grid.Nz])
z = collect(znodes(grid, Center()))
pressure = profile(reference_state.pressure)
density = profile(reference_state.density)
initial = hcat(z, pressure, profile(model.temperature), density,
               profile(μ.qᵛ), profile(μ.sᵛ⁺ˡ))
if case != "reference_options"
    writedlm(joinpath(output_directory, "initial_state.dat"), reverse(initial; dims=1))
    writedlm(joinpath(output_directory, "inflow.dat"), [θ₀ qᵗ₀ sᵛ⁺ˡ₀ W₀])
end
write(joinpath(output_directory, "breeze_$(case)_setup.jl"), setup)
write(joinpath(output_directory, "breeze_$(case)_provenance.txt"),
      "Example SHA256: $(bytes2hex(sha256(read(example))))\nJulia: $(VERSION)\ndt: $(Δt)\ncase: $(case)\n")

filename = joinpath(output_directory, "breeze_$(case)_dt$(Δt).dat")
diagnosed_supersaturation = Field(μ.qᵛ - qᵛ⁺ˡ)
open(filename, "w") do io
    println(io, "# time z T p rho qv qc qr qi qli nc nr ni ssat diagnosed_ssat")
end
last_output_time = Ref(-Inf)
function record_profiles(sim)
    t = time(sim)
    t <= last_output_time[] && return nothing
    compute!(diagnosed_supersaturation)
    data = hcat(fill(t, grid.Nz), z, profile(model.temperature), pressure, density,
                profile(μ.qᵛ), profile(μ.qᶜˡ), profile(μ.qʳ), profile(μ.qⁱ),
                profile(μ.qʷⁱ), profile(μ.nᶜˡ), profile(μ.nʳ), profile(μ.nⁱ),
                profile(μ.sᵛ⁺ˡ), profile(diagnosed_supersaturation))
    all(isfinite, data) || error("Nonfinite output at t=$t")
    open(filename, "a") do io
        writedlm(io, data)
    end
    last_output_time[] = t
    return nothing
end

simulation = Simulation(model; Δt, stop_time=3600)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
record_profiles(simulation)
add_callback!(simulation, record_profiles, TimeInterval(30))
run!(simulation)
@info "Breeze comparison complete" case Δt filename
