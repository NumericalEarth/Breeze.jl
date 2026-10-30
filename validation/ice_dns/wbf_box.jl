# Prototype DNS of ice depositional growth among supercooled droplets (the Wegener–Bergeron–Findeisen
# process) in forced homogeneous turbulence, in a periodic box at -15 °C, compared to a well-mixed parcel
# (one cell) with the same particles.
#
#   julia --project=~/ice-prototype-env validation/ice_dns/wbf_box.jl CASE [M] [STOP_TIME]
#
# CASE:
# - wbf:       uniform cloud saturated over liquid (supersaturated over ice by 16%), droplets and ice crystals
#              everywhere. Ice grows at the expense of the droplets.
# - fragments: the lower half of the box is cloud saturated over liquid (100 cm⁻³ droplets of 10 μm), the upper
#              half clear air at 70% relative humidity over ice, with an interface of a few cells. Small ice fragments
#              (secondary ice, 0.3 to 5 μm, log-uniform) are spread through the box. The mixed box is supersaturated
#              over ice once the droplets have evaporated. In the parcel the box is mixed instantly; in the DNS fragments in clear air
#              sublimate unless turbulence mixes them into cloud first.
#
# Environment: OUTPUT_DIR (default validation/ice_dns/output), ARCH (GPU or CPU), SELF (1: correct particle growth
# for the particle's own sink in its cell, default; 0: do not), LABEL (suffix of the output names).

using Breeze
using Breeze.Thermodynamics: saturation_vapor_pressure, vapor_gas_constant, dry_air_gas_constant, mixture_gas_constant,
                             mixture_heat_capacity, liquid_latent_heat, ice_latent_heat, reference_value,
                             MoistureMassFractions, PlanarLiquidSurface, PlanarIceSurface
using Breeze.ParticleMicrophysics: droplet_water_mass, ice_particle_mass, ventilation_factor
using Oceananigans
using CUDA
using DelimitedFiles
using Printf
using Random
using Statistics

const T₀ = 258.15  # K (-15 °C)
const p₀ = 70000.0 # Pa
const L = 0.064    # m
const ν = 1.6e-5
const κᵀ = 2.22e-5
const κᵛ = 2.55e-5
const ε_forcing = 0.05 # W kg⁻¹, as in the NCAR L3 runs

vapor_pressure(qᵛ, p, constants) = qᵛ * vapor_gas_constant(constants) * p / mixture_gas_constant(MoistureMassFractions(qᵛ), constants)

function vapor_mass_fraction(saturation_ratio, T, p, constants, surface)
    pᵛ = saturation_ratio * saturation_vapor_pressure(T, constants, surface)
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    return ε * pᵛ / (p - (1 - ε) * pᵛ)
end

supersaturations(T, qᵛ, constants) =
    (vapor_pressure(qᵛ, p₀, constants) / saturation_vapor_pressure(T, constants, PlanarLiquidSurface()) - 1,
     vapor_pressure(qᵛ, p₀, constants) / saturation_vapor_pressure(T, constants, PlanarIceSurface()) - 1)

#####
##### Initial particles
#####

function initial_particles(case; rng = Xoshiro(1))
    V = L^3
    if case == :wbf
        Nd = round(Int, 50e6 * V)    # 50 cm⁻³
        Ni = round(Int, 2e6 * V)     # 2 cm⁻³
        droplets = (x = L .* rand(rng, Nd), y = L .* rand(rng, Nd), z = L .* rand(rng, Nd),
                    r = 8e-6 .+ 1e-6 .* randn(rng, Nd), rᵈ = fill(5e-8, Nd), κ = fill(0.6, Nd))
        ice = (x = L .* rand(rng, Ni), y = L .* rand(rng, Ni), z = L .* rand(rng, Ni), r = fill(10e-6, Ni))
    else # fragments
        Nd = round(Int, 100e6 * V / 2) # 100 cm⁻³ in the cloudy half
        Ni = round(Int, 10e6 * V)      # 10 cm⁻³ fragments
        droplets = (x = L .* rand(rng, Nd), y = L .* rand(rng, Nd), z = L / 2 .* rand(rng, Nd),
                    r = 10e-6 .+ 1e-6 .* randn(rng, Nd), rᵈ = fill(5e-8, Nd), κ = fill(0.6, Nd))
        ice = (x = L .* rand(rng, Ni), y = L .* rand(rng, Ni), z = L .* rand(rng, Ni),
               r = 0.3e-6 .* exp.(log(5 / 0.3) .* rand(rng, Ni)))
    end
    return droplets, ice
end

# The relative humidity over ice of the initial state
function initial_saturation_ratio(case, z, constants)
    case == :wbf && return saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface()) /
                           saturation_vapor_pressure(T₀, constants, PlanarIceSurface())
    cloudy = saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface()) / saturation_vapor_pressure(T₀, constants, PlanarIceSurface())
    clear = 0.7
    δ = 2 * L / 128 # interfaces at z = L/2 and z = 0 (≡ L)
    upper = (1 + tanh((z - L / 2) / δ)) / 2 * (1 - tanh((z - L) / δ)) / 2 + (1 - tanh(z / δ)) / 2
    return cloudy + (clear - cloudy) * clamp(upper, 0, 1)
end

#####
##### Well-mixed parcel with the same particles
#####

struct Parcel{D, I}
    T :: Base.RefValue{Float64}
    qᵛ :: Base.RefValue{Float64}
    droplets :: D
    ice :: I
    ρ :: Float64
    V :: Float64
end

function parcel_step!(parcel, Δt, constants, droplet_physics, ice_physics)
    T, qᵛ = parcel.T[], parcel.qᵛ[]
    𝒮, 𝒮ⁱ = supersaturations(T, qᵛ, constants)
    d, i = parcel.droplets, parcel.ice
    C = 0.0
    for n in eachindex(d.r)
        fᵛ = ventilation_factor(d.r[n], d.𝕎[n], T, p₀, constants, droplet_physics)
        step = condensational_growth_step(d.r[n], d.rᵈ[n], d.κ[n], 𝒮, fᵛ, T, p₀, Δt, constants, droplet_physics)
        d.r[n] = step.radius
        C += step.condensed_mass
    end
    D = 0.0
    for n in eachindex(i.r)
        fᵛ = ventilation_factor(ice_physics.capacitance_ratio * i.r[n], i.𝕎[n], T, p₀, constants, ice_physics)
        step = depositional_growth_step(i.r[n], 𝒮ⁱ, fᵛ, T, p₀, Δt, constants, ice_physics)
        i.r[n] = step.radius
        D += step.deposited_mass
    end
    M = parcel.ρ * parcel.V
    qᵛ⁺ = qᵛ - (C + D) / M
    cᵖᵐ = mixture_heat_capacity(MoistureMassFractions(qᵛ⁺), constants)
    parcel.T[] = T + (liquid_latent_heat(T, constants) * C + ice_latent_heat(T, constants) * D) / (M * cᵖᵐ)
    parcel.qᵛ[] = qᵛ⁺
    return nothing
end

#####
##### Run
#####

function run_case(case, M; stop_time = 10.0, arch = GPU(), output_dir, self_interaction = DiffusiveSelfInteraction(), suffix = "")
    label = @sprintf("%s_M%d%s", case, M, suffix)
    grid = RectilinearGrid(arch; size=(M, M, M), halo=(3, 3, 3), x=(0, L), y=(0, L), z=(0, L),
                           topology=(Periodic, Periodic, Periodic))

    droplets, ice = initial_particles(case)
    droplet_physics = DropletPhysicsParameters(Float64; terminal_velocity=StokesTerminalVelocity())
    ice_physics = IcePhysicsParameters(Float64)
    microphysics = LagrangianMicrophysics(; physics=droplet_physics, ice_physics, droplets, ice,
                                          self_interaction)
    dynamics = BoussinesqDynamics(grid; pressure=p₀, temperature=T₀)
    closure = ScalarDiffusivity(; ν, κ=(ρθ=κᵀ, ρqᵛ=κᵛ))
    forcing = ConstantPowerBandForcing(grid; energy_injection_rate=ε_forcing)
    model = AtmosphereModel(grid; dynamics, microphysics, closure, advection=Centered(order=4),
                            forcing=(u=forcing, v=forcing, w=forcing))
    constants = model.thermodynamic_constants

    rng = Xoshiro(2)
    u₀, v₀, w₀ = (0.05 .* randn(rng, M, M, M) for _ in 1:3)
    qᵛ₀(x, y, z) = vapor_mass_fraction(initial_saturation_ratio(case, z, constants), T₀, p₀, constants, PlanarIceSurface())
    set!(model; T=T₀, qᵗ=qᵛ₀, u=u₀, v=v₀, w=w₀)

    # The parcel: the box mean of the initial state, with copies of the particles
    ρ = reference_value(model.dynamics.reference_state.density)
    Nd, Ni = length(droplets.r), length(ice.r)
    parcel = Parcel(Ref(mean(model.temperature)), Ref(mean(model.microphysical_fields.qᵛ)),
                    (r = copy(droplets.r), rᵈ = copy(droplets.rᵈ), κ = copy(droplets.κ), 𝕎 = zeros(Nd)),
                    (r = copy(ice.r), 𝕎 = [ice_terminal_velocity(r, T₀, p₀, constants, ice_physics) for r in ice.r]),
                    ρ, L^3)
    for n in 1:Nd
        parcel.droplets.𝕎[n] = droplet_terminal_velocity(parcel.droplets.r[n], T₀, p₀, constants, droplet_physics)
    end
    clear_start = Array(ice.z) .> L / 2 # fragments that start in the clear half (fragments case)
    rⁱ₀ = copy(ice.r)

    rows = Dict{String, Vector{Float64}}()
    record(key, value) = push!(get!(rows, key, Float64[]), value)
    snapshots = Dict{Float64, Any}()
    snapshot_times = case == :wbf ? [0.0, 2.0, stop_time] : [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, stop_time]

    cell(ξ) = clamp(floor(Int, ξ / (L / M)) + 1, 1, M)

    function diagnose!(sim)
        t = sim.model.clock.time
        T = Array(interior(model.temperature))
        qᵛ = Array(interior(model.microphysical_fields.qᵛ))
        S = supersaturations.(T, qᵛ, Ref(constants))
        𝒮 = first.(S)
        𝒮ⁱ = last.(S)
        d = model.microphysics.droplets.properties
        i = model.microphysics.ice.properties
        rᵈ = Array(d.r)
        rⁱ = Array(i.r)
        liquid = sum(droplet_water_mass.(rᵈ, Array(d.rᵈ), Ref(constants))) / (ρ * L^3)
        frozen = sum(ice_particle_mass.(rⁱ, Ref(ice_physics))) / (ρ * L^3)
        𝒮ᵖ, 𝒮ⁱᵖ = supersaturations(parcel.T[], parcel.qᵛ[], constants)
        liquidᵖ = sum(droplet_water_mass.(parcel.droplets.r, parcel.droplets.rᵈ, Ref(constants))) / (ρ * L^3)
        frozenᵖ = sum(ice_particle_mass.(parcel.ice.r, Ref(ice_physics))) / (ρ * L^3)
        record("time", t)
        record("liquid", liquid); record("ice", frozen)
        record("vapor", mean(qᵛ)); record("temperature", mean(T))
        record("S_mean", mean(𝒮)); record("S_std", std(𝒮))
        record("Si_mean", mean(𝒮ⁱ)); record("Si_std", std(𝒮ⁱ))
        record("ice_radius_mean", mean(rⁱ[rⁱ .> 0])); record("ice_radius_std", std(rⁱ[rⁱ .> 0]))
        record("ice_surviving", count(>(0), rⁱ) / Ni)
        record("ice_surviving_clear", count(>(0), rⁱ[clear_start]) / max(1, count(clear_start)))
        record("droplets_surviving", count(r -> r > 2e-7, rᵈ) / Nd)
        record("parcel_liquid", liquidᵖ); record("parcel_ice", frozenᵖ)
        record("parcel_S", 𝒮ᵖ); record("parcel_Si", 𝒮ⁱᵖ); record("parcel_temperature", parcel.T[])
        record("parcel_ice_radius_mean", mean(parcel.ice.r[parcel.ice.r .> 0]))
        record("parcel_ice_radius_std", std(parcel.ice.r[parcel.ice.r .> 0]))
        record("parcel_ice_surviving", count(>(0), parcel.ice.r) / Ni)
        record("kinetic_energy", sum(mean(Array(interior(u)) .^ 2) for u in model.velocities) / 2)

        for τ in snapshot_times
            if !haskey(snapshots, τ) && t ≥ τ - 1e-9
                j = M ÷ 2
                y = Array(i.y); Δ = L / M
                slab = abs.(y .- (j - 0.5) * Δ) .< 2Δ
                yd = Array(d.y)
                slabd = abs.(yd .- (j - 0.5) * Δ) .< 2Δ
                snapshots[τ] = (S = 𝒮[:, j, :], Si = 𝒮ⁱ[:, j, :],
                                ice = hcat(Array(i.x)[slab], Array(i.z)[slab], rⁱ[slab]),
                                droplets = hcat(Array(d.x)[slabd], Array(d.z)[slabd], rᵈ[slabd]),
                                ice_radius = rⁱ, parcel_ice_radius = copy(parcel.ice.r),
                                droplet_radius = rᵈ, parcel_droplet_radius = copy(parcel.droplets.r),
                                Si_ice = 𝒮ⁱ[CartesianIndex.(cell.(Array(i.x)), cell.(Array(i.y)), cell.(Array(i.z)))],
                                S_cells = vec(𝒮), Si_cells = vec(𝒮ⁱ))
            end
        end
        return nothing
    end
    Δ = L / M
    cfl, max_Δt = 0.3, 2e-3
    simulation = Simulation(model; Δt=1e-4, stop_time, verbose=false)
    wizard = TimeStepWizard(; cfl, max_change=1.1, diffusive_cfl=0.2, max_Δt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

    # The parcel advances with implicit steps of 0.01 s (each a small fraction of the phase relaxation times)
    last_time = Ref(0.0)
    simulation.callbacks[:parcel] = Callback(TimeInterval(0.01)) do sim
        Δtᵖ = sim.model.clock.time - last_time[]
        Δtᵖ > 0 && parcel_step!(parcel, Δtᵖ, constants, droplet_physics, ice_physics)
        last_time[] = sim.model.clock.time
    end
    simulation.callbacks[:diagnose] = Callback(diagnose!, TimeInterval(0.05))
    simulation.callbacks[:progress] = Callback(IterationInterval(500)) do sim
        @info @sprintf("%s: t = %.3f s, Δt = %.2e, 𝒮 = %.3f%% ± %.3f%%, 𝒮ⁱ = %.3f%%, qⁱ = %.3e (parcel %.3e), qˡ = %.3e (parcel %.3e), survivors %.3f (parcel %.3f)",
                       label, sim.model.clock.time, sim.Δt, 100rows["S_mean"][end], 100rows["S_std"][end], 100rows["Si_mean"][end],
                       rows["ice"][end], rows["parcel_ice"][end], rows["liquid"][end], rows["parcel_liquid"][end],
                       rows["ice_surviving"][end], rows["parcel_ice_surviving"][end])
    end

    @info "Ice DNS" label Nd Ni Δ
    wall = time_ns()
    run!(simulation)
    elapsed = (time_ns() - wall) / 1e9
    @info @sprintf("%s: %d iterations in %.1f s", label, model.clock.iteration, elapsed)

    mkpath(output_dir)
    keys_sorted = sort(collect(keys(rows)))
    open(joinpath(output_dir, "ice_dns_$(label)_timeseries.csv"), "w") do io
        println(io, join(keys_sorted, ","))
        writedlm(io, hcat((rows[k] for k in keys_sorted)...), ',')
    end
    for (τ, s) in snapshots
        prefix = joinpath(output_dir, @sprintf("ice_dns_%s_t%.2f", label, τ))
        writedlm(prefix * "_S.txt", s.S)
        writedlm(prefix * "_Si.txt", s.Si)
        writedlm(prefix * "_ice_slab.txt", s.ice)
        writedlm(prefix * "_droplet_slab.txt", s.droplets)
        writedlm(prefix * "_ice_radius.txt", hcat(s.ice_radius, s.parcel_ice_radius, s.Si_ice, rⁱ₀, clear_start))
        writedlm(prefix * "_droplet_radius.txt", hcat(s.droplet_radius, s.parcel_droplet_radius))
        writedlm(prefix * "_cells.txt", hcat(s.S_cells, s.Si_cells))
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    case = Symbol(ARGS[1])
    M = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 64
    stop_time = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : 10.0
    arch = get(ENV, "ARCH", "GPU") == "GPU" ? GPU() : CPU()
    output_dir = get(ENV, "OUTPUT_DIR", joinpath(@__DIR__, "output"))
    self_interaction = get(ENV, "SELF", "1") == "1" ? DiffusiveSelfInteraction() : nothing
    run_case(case, M; stop_time, arch, output_dir, self_interaction, suffix = get(ENV, "LABEL", ""))
end
