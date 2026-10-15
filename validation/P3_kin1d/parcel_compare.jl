# Production ParcelModel: initialization + Simulation, with read-only output.
# No reference_conventions, custom thermodynamic methods, or prognostic callbacks.
using Oceananigans
using Breeze
using Breeze.Thermodynamics: Thermodynamics as TD
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties as PPP
using DelimitedFiles
using Printf

function run_production_parcel(speed; dt=0.1, timescale=10., configuration="default")
    grid = RectilinearGrid(size=12001, z=(-0.5, 12000.5), topology=(Flat, Flat, Bounded))
    constants = if configuration == "reference_constants"
        ThermodynamicConstants(; dry_air_molar_mass=8.314462618/287.15,
            vapor_molar_mass=8.314462618/461.51, gravitational_acceleration=9.816,
            liquid=TD.CondensedPhase(reference_latent_heat=3.1484e6-2370*273.15, heat_capacity=4218, density=1000),
            saturation_vapor_pressure=TD.FlatauPolynomial(reference_temperature=273.15))
    else
        ThermodynamicConstants()
    end
    rates = PPP.ProcessRate(; thermodynamic_constants=constants, predict_supersaturation=true,
        sink_limiting_timescale=timescale, ice_nucleation_timescale=timescale,
        homogeneous_freezing_timescale=timescale, refreezing_timescale=timescale,
        rime_densification_timescale=timescale, rain_evaporation_timescale=timescale)
    activation_timescale = configuration == "native_defaults" ? 1. : timescale
    aerosol = if configuration == "reference_constants"
        PPP.AerosolActivation(PPP.AerosolMode(number_mixing_ratio=3e8*0.985, molecular_weight_water=0.018);
                              prognostic=false, activation_timescale, molecular_weight_water=0.018, universal_gas_constant=8.314)
    else
        PPP.AerosolActivation(PPP.AerosolMode(number_mixing_ratio=3e8*0.985);
                              prognostic=false, activation_timescale)
    end
    microphysics = P3Microphysics(; thermodynamic_constants=constants, process_rates=rates, aerosol)
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics,
                            formulation=:LiquidIcePotentialTemperature, thermodynamic_constants=constants)
    pressure(z) = 101325 * (1 - 9.816*z/(1005*300))^(1005/287.15)
    environmental_temperature(z) = 300 - 9.816*z/1005
    environmental_density(z) = pressure(z) / (287.15 * environmental_temperature(z))
    set!(model; T=300, p=pressure, ρ=environmental_density, qᵗ=0.015, z=0, w=speed)
    state = model.dynamics.state
    saturation = saturation_specific_humidity(temperature(state.𝒰, constants), TD.density(state.𝒰, constants), constants, PlanarLiquidSurface())
    state.μ = merge(state.μ, (; ρsᵛ⁺ˡ=state.ρ*(state.𝒰.moisture_mass_fractions.vapor-saturation)))
    rows = Vector{Float64}[]
    function record(sim)
        state = sim.model.dynamics.state
        q = state.𝒰.moisture_mass_fractions
        μ = state.μ
        rho = state.ρ
        push!(rows, [12+speed, sim.model.clock.time, state.𝒰.reference_pressure,
                     temperature(state.𝒰, constants), q.vapor, μ.ρqᶜˡ/rho, μ.ρnᶜˡ/rho,
                     μ.ρqʳ/rho, μ.ρnʳ/rho, μ.ρqⁱ/rho, μ.ρnⁱ/rho, μ.ρqᶠ/rho,
                     μ.ρbᶠ/rho, μ.ρqʷⁱ/rho, μ.ρsᵛ⁺ˡ/rho,
                     state.ℰ, state.qᵗ, rho, TD.density(state.𝒰, constants),
                     q.vapor - TD.saturation_specific_humidity(state.𝒰, constants, PlanarLiquidSurface())])
    end
    simulation = Simulation(model; Δt=dt, stop_time=5400, verbose=false)
    add_callback!(simulation, record, TimeInterval(10))
    if get(ENV, "P3_DEBUG_NONFINITE", "false") == "true"
        previous = Ref((; z=state.z, ρ=state.ρ, μ=state.μ, 𝒰=state.𝒰))
        function check_state(sim)
            if !isfinite(temperature(state.𝒰, constants))
                @show previous[]
                error("First nonfinite parcel temperature at $(sim.model.clock.time)")
            end
            previous[] = (; z=state.z, ρ=state.ρ, μ=state.μ, 𝒰=state.𝒰)
        end
        add_callback!(simulation, check_state, IterationInterval(1))
    end
    run!(simulation)
    return permutedims(hcat(rows...))
end

root = normpath(joinpath(@__DIR__, "../../validation_output"))
dt = parse(Float64, get(ARGS, 1, "0.0025"))
timescale = parse(Float64, get(ARGS, 2, "0.025"))
configuration = get(ARGS, 3, "reference_constants")
tag = get(ARGS, 4, "verified")
directory = joinpath(root, "p3_parcel_direct", "$(tag)_$(configuration)_dt$(dt)_tau$(timescale)")
mkpath(directory)
@assert configuration in ("default", "native_defaults", "reference_constants")
@assert dt > 0 && timescale > 0
write(joinpath(directory, "configuration.txt"),
      "Julia: $VERSION\nParcelModel SSP RK3\nΔt=$dt s\nτ=$timescale s\nConfiguration: $configuration\n" *
      "Fixed aerosol, no sedimentation or entrainment; θ_li conserved.\n" *
      "Native defaults: activation τ=1 s; other process τ=10 s.\n" *
      "Other configurations: activation and listed process timescales all equal τ.\n")
open(joinpath(directory, "metrics.txt"), "w") do io
    for speed in (1, 2)
        data = run_production_parcel(speed; dt, timescale, configuration)
        writedlm(joinpath(directory, "parcel_$(speed).dat"), data)
        @assert all(isfinite, data) "Nonfinite parcel state; raw trajectory retained"
        @assert maximum(abs.(data[:,16] .- data[1,16])) < 1e-8
        @assert maximum(abs.(data[:,17] .- 0.015)) < 1e-12
        @printf(io, "Maximum supersaturation identity error: %.6g kg/kg\n", maximum(abs.(data[:,15]-data[:,20])))
        @printf(io, "%d m/s: final T %.8g K; max θ_li drift %.6g K; max total-water drift %.6g kg/kg\n",
                speed, data[end,4], maximum(abs.(data[:,16] .- data[1,16])), maximum(abs.(data[:,17] .- .015)))
        flush(io)
    end
end
print(read(joinpath(directory, "metrics.txt"), String))
