# L3: condensational growth of cloud droplets in forced turbulence in a rising parcel, in Breeze,
# started from the initial state of the NCAR parcel-DNS L3 runs (~/ncar-dns/runs/L3/cond_s<seed>_a).
#
#   julia --project=validation/ncar_dns validation/ncar_dns/condensation.jl NCAR_L3_DIR RUNS [STOP_TIME]
#
# - RUNS: comma-separated seed:M pairs, e.g. "1:32,1:64"
# - STOP_TIME: duration (s) of the run after the NCAR initial time (default 10)
#
# Environment: ARCH, CFL (default 0.3), MAX_DT (default 1e-3), OUTPUT_DIR, LABEL, OUTPUT_INTERVAL (default 0.02),
# LAPSE (none, isothermal or moist), QUIESCENT (1: no forcing, no initial flow or fluctuations),
# UNIFORM_VELOCITY (with QUIESCENT: "u,v,w" in m/s), SELF (1: correct droplet growth for self-depletion, default;
# 2: the same, for droplets at rest relative to the grid, as before the grid-translation fix; 0: do not).
#
# Matched to NCAR (param.inc and microhydroall.F90 of the L3 build):
# - thermodynamics: Rᵈ = 287, Rᵛ = 467, cᵖ = 1004 for all components (so that cᵖᵐ = cᵖ and ℒ is constant),
#   ℒ = 2.477e6, pᵛ⁺ = 2.53e11 exp(-5420 / T), g = 9.8 for the rising frame, and no buoyancy (expsiv = 0);
# - droplets: σ = 7.61e-2, Dᵛ(T) and Kᵀ(T) of NCAR, no gas-kinetic correction, dilute-solution activity
#   with κ = 2 (mʷ/mˢ)(ρˢ/ρʷ) for ammonium sulfate, Stokes terminal velocity, Beard-Pruppacher ventilation;
# - flow: ν = 1.6e-5, κᵀ = 2.22e-5, κᵛ = 2.55e-5, band forcing at constant power ε = 0.05;
# - adiabatic displacement within the frame as a background-gradient forcing (adiabatic_background.jl), LAPSE =
#   none (default; NCAR's `gammand * wk` term is effectively absent because `wk` holds (w T′)^ at that point),
#   isothermal (the intended NCAR term ∂T′/∂t = - (g / cᵖ) w′) or moist (the full A₁ w′ supersaturation source).
# Differences: Breeze's supersaturation is pᵛ / pᵛ⁺ - 1; NCAR's is the ratio of mixing ratios rᵛ / rᵛ⁺ - 1,
# which is larger by about 𝒮 pᵛ⁺ / p (0.02 percentage points at 𝒮 = 1.6%). NCAR adds the condensation
# increments of T′ and qᵛ′ to the current leapfrog level only, so that each leapfrog chain receives
# half of the condensation fluctuations (the parcel mean receives all of it).
# Droplets are not inertial (NCAR integrates Stokes drag; St ≲ 0.1 here); the droplet
# condensation sink is applied to the Breeze cell containing the droplet; NCAR's ventilation uses the
# droplet velocity relative to the frame, Breeze uses the terminal velocity.

include(joinpath(@__DIR__, "NCARDNS.jl"))
using .NCARDNS
using Breeze
using Breeze.Thermodynamics: Thermodynamics, PlanarLiquidSurface, CondensedPhase, ThermodynamicConstants,
                             MoistureMassFractions, mixture_gas_constant
using Breeze.ParticleMicrophysics: ParticleMicrophysics
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Printf
using Statistics
using CUDA

include(joinpath(@__DIR__, "adiabatic_background.jl"))

#####
##### NCAR physics
#####

struct NCARSaturationVaporPressure end
const NCARConstants = ThermodynamicConstants{FT, C, I, NCARSaturationVaporPressure} where {FT, C, I}
Thermodynamics.saturation_vapor_pressure(T, ::NCARConstants, ::PlanarLiquidSurface) = oftype(T, 2.53e11) * exp(oftype(T, -5.42e3) / T)

struct NCARVaporDiffusivity end
struct NCARThermalConductivity end
ParticleMicrophysics.vapor_diffusivity(::NCARVaporDiffusivity, T, p) = oftype(T, 1e-5) * (oftype(T, 0.015) * T - oftype(T, 1.9))
ParticleMicrophysics.thermal_conductivity(::NCARThermalConductivity, T, p) =
    oftype(T, 1.5e-11) * T^3 - oftype(T, 4.8e-8) * T^2 + oftype(T, 1e-4) * T - oftype(T, 3.9e-4)

const ℛ = 8.314462618
const Rᵈ_ncar = 287.0
const Rᵛ_ncar = 467.0
const cᵖ_ncar = 1004.0
const ℒ_ncar = 2.477e6
const g_ncar = 9.8
const ρʷ = 1000.0
const κ_ncar = 2 * (18e-3 / 132.14e-3) * (1726 / ρʷ)
const ascent_velocity = 2.0

ncar_constants(g, FT=Float64; saturation_vapor_pressure = NCARSaturationVaporPressure()) =
    ThermodynamicConstants(FT;
                           dry_air_molar_mass = ℛ / Rᵈ_ncar,
                           vapor_molar_mass = ℛ / Rᵛ_ncar,
                           dry_air_heat_capacity = cᵖ_ncar,
                           vapor_heat_capacity = cᵖ_ncar,
                           liquid = CondensedPhase(FT; reference_latent_heat = ℒ_ncar,
                                                   heat_capacity = cᵖ_ncar, density = ρʷ),
                           gravitational_acceleration = g,
                           saturation_vapor_pressure)

ncar_droplet_physics(FT=Float64) = DropletPhysicsParameters(FT;
                                                  surface_tension = 7.61e-2,
                                                  vapor_diffusivity = NCARVaporDiffusivity(),
                                                  thermal_conductivity = NCARThermalConductivity(),
                                                  kinetic_correction = NoKineticCorrection(),
                                                  solution_activity = DiluteSolutionActivity(),
                                                  terminal_velocity = StokesTerminalVelocity())

# NCAR's parcel supersaturation from mixing ratios, sp = rᵛ / rᵛ⁺ - 1 with rᵛ⁺ = ε pᵛ⁺ / (p - pᵛ⁺)
ncar_saturation_mixing_ratio(T, p) = (Rᵈ_ncar / Rᵛ_ncar) * 2.53e11 * exp(-5.42e3 / T) / (p - 2.53e11 * exp(-5.42e3 / T))

#####
##### Diagnostics
#####

const dsd_bins = 100 # NCAR .dsd: counts in bins nint(r / 1 μm), r = 0, …, 99 μm

dsd_counts(r) = [count(x -> round(Int, x / 1e-6) == b, r) for b in 0:dsd_bins-1]

function cell_supersaturation(model, constants)
    p = model.dynamics.reference_state.frame.pressure
    T = Array{Float64}(interior(model.temperature))
    qᵛ = Array{Float64}(interior(model.microphysical_fields.qᵛ))
    return map(T, qᵛ) do T, qᵛ
        q = MoistureMassFractions(qᵛ)
        pᵛ = qᵛ * Thermodynamics.vapor_gas_constant(constants) * p / mixture_gas_constant(q, constants)
        pᵛ / Thermodynamics.saturation_vapor_pressure(T, constants, PlanarLiquidSurface()) - 1
    end
end

function droplet_cells(x, y, z, Δ, M)
    cell(ξ) = clamp(floor(Int, ξ / Δ) + 1, 1, M)
    return CartesianIndex.(cell.(x), cell.(y), cell.(z))
end

function record!(state, model)
    (; rows, dsd, pdfs, constants, L, M, ρᵃ) = state
    frame = model.dynamics.reference_state.frame
    droplets = model.microphysics.droplets.properties
    # Statistics in Float64: a Float32 mean of 10⁵ values of T ≈ 281 K is in error by ~10⁻⁴ K, comparable to std T′
    r = Array{Float64}(droplets.r)
    x, y, z = Array{Float64}(droplets.x), Array{Float64}(droplets.y), Array{Float64}(droplets.z)
    Tᶜ = Array{Float64}(interior(model.temperature))
    qᶜ = Array{Float64}(interior(model.microphysical_fields.qᵛ))

    T̄ = mean(Tᶜ)
    q̄ᵛ = mean(qᶜ)
    p = frame.pressure
    rᵛ = q̄ᵛ / (1 - q̄ᵛ)
    S_parcel = rᵛ / ncar_saturation_mixing_ratio(T̄, p) - 1

    S = cell_supersaturation(model, constants)
    Sᵈ = S[droplet_cells(x, y, z, L / M, M)]
    lwc = 4π / 3 * ρʷ * sum(r .^ 3) / (ρᵃ * L^3) # as in NCAR: wet volume, density of the reference state

    push!(rows.time, model.clock.time)
    push!(rows.pressure, p)
    push!(rows.temperature, T̄)
    push!(rows.vapor_mixing_ratio, rᵛ)
    push!(rows.parcel_supersaturation, S_parcel)
    push!(rows.mean_supersaturation, mean(S))
    push!(rows.std_supersaturation, std(S))
    push!(rows.droplet_mean_supersaturation, mean(Sᵈ))
    push!(rows.droplet_std_supersaturation, std(Sᵈ))
    push!(rows.liquid_water_content, lwc)
    push!(rows.mean_radius, mean(r))
    push!(rows.std_radius, std(r))
    push!(rows.mean_radius_cubed, mean(r .^ 3))
    push!(rows.std_temperature, std(Tᶜ))
    push!(rows.std_vapor, std(qᶜ))
    push!(rows.kinetic_energy, sum(mean(Array{Float64}(interior(u)) .^ 2) for u in model.velocities) / 2)
    push!(rows.droplets, length(r))

    # Per-step condensation increments of θ and qᵛ, relative to the spacing of the stored values
    C = Array{Float64}(interior(model.microphysical_fields.C))
    Π = T̄ / mean(Array{Float64}(interior(model.formulation.potential_temperature)))
    Δθ = ℒ_ncar .* abs.(C) ./ (ρᵃ * cᵖ_ncar * Π)
    Δq = abs.(C) ./ ρᵃ
    active = C .!= 0
    θ̄ = T̄ / Π
    ρθ_type = eltype(model.formulation.potential_temperature_density)
    ρq_type = eltype(model.moisture_density)
    push!(rows.time_step, model.clock.last_Δt)
    push!(rows.condensing_cells, count(active) / length(C))
    push!(rows.theta_increment, any(active) ? mean(Δθ[active]) : 0.0)
    push!(rows.vapor_increment, any(active) ? mean(Δq[active]) : 0.0)
    push!(rows.theta_spacing, Float64(eps(ρθ_type(ρᵃ * θ̄))) / ρᵃ)
    push!(rows.vapor_spacing, Float64(eps(ρq_type(ρᵃ * q̄ᵛ))) / ρᵃ)
    # Fractions of the condensing cells whose increment is below half the single-precision spacing of the stored
    # value, so that it would be rounded away entirely
    push!(rows.theta_subhalf_fraction, any(active) ? count(Δθ[active] .< eps(Float32(ρᵃ * θ̄)) / 2ρᵃ) / count(active) : 0.0)
    push!(rows.vapor_subhalf_fraction, any(active) ? count(Δq[active] .< eps(Float32(ρᵃ * q̄ᵛ)) / 2ρᵃ) / count(active) : 0.0)

    push!(dsd.time, model.clock.time)
    push!(dsd.counts, dsd_counts(r))

    if state.pdf_index[] ≤ length(state.pdf_times) && model.clock.time ≥ state.pdf_times[state.pdf_index[]] - 1e-9
        push!(pdfs, (time = model.clock.time, cell = vec(S), droplet = Sᵈ, radius = r))
        state.pdf_index[] += 1
    end
    return nothing
end

# Round x to one of the two neighboring Float32 values with probabilities proportional to proximity (u uniform in [0, 1))
@inline function stochastic_float32(x, u)
    y = Float32(x, RoundDown)
    ŷ = nextfloat(y)
    return Float64(ifelse(u < (x - y) / (ŷ - y), ŷ, y))
end

#####
##### Cost: time per step without diagnostics, and device memory in use
#####

function write_timing(output_dir, label, model, Δt; steps=200)
    arch = Oceananigans.architecture(model.grid)
    GC.gc(true)
    arch isa GPU && CUDA.reclaim()
    memory = arch isa GPU ? CUDA.used_memory() : 0
    for _ in 1:5
        time_step!(model, Δt)
    end
    arch isa GPU && CUDA.synchronize()
    wall = time_ns()
    for _ in 1:steps
        time_step!(model, Δt)
    end
    arch isa GPU && CUDA.synchronize()
    per_step = (time_ns() - wall) / 1e9 / steps
    @info @sprintf("%s: %.3f ms per step (no diagnostics), %.1f MiB device memory in use", label, 1e3 * per_step,
                   memory / 2^20)
    if get(ENV, "PROFILE", "0") == "1" && arch isa GPU
        profile = CUDA.@profile trace=false for _ in 1:20
            time_step!(model, Δt)
        end
        show(IOContext(stdout, :limit => false), profile)
        println()
    end
    open(joinpath(output_dir, "timing.csv"), "a") do io
        println(io, label, ",", eltype(model.grid), ",", per_step, ",", memory)
    end
    return nothing
end

#####
##### Run
#####

function run_l3(ncar_dir, seed, M; arch, stop_time=10.0, cfl=0.3, max_Δt=1e-3, output_interval=0.02,
                output_dir, suffix="", lapse=:none, self_interaction=DiffusiveSelfInteraction(), quiescent=false,
                uniform_velocity=(0.0, 0.0, 0.0), FT=Float64, thermodynamic_float_type=nothing)
    label = @sprintf("s%d_M%d%s", seed, M, suffix)
    case = joinpath(ncar_dir, "cond_s$(seed)_a")
    zk = read_zk(joinpath(case, "Zk.in.ncf"))
    d = read_drop(joinpath(case, "drop.in.ncf"); npe=4)
    @info "L3 condensation" label arch zk d.time length(d.radius) d.supersaturation

    p = zk.parameters
    L = p.domain_length
    ν = p.viscosity

    grid = RectilinearGrid(arch, FT; size=(M, M, M), halo=(3, 3, 3), x=(0, L), y=(0, L), z=(0, L),
                           topology=(Periodic, Periodic, Periodic))

    # Mean parcel state (NCAR: mixing ratio qvpp, potential temperature thetapp with exner (p / 10⁵)^(Rᵈ/cᵖ))
    p₀ = d.pressure
    T₀ = d.potential_temperature * (p₀ / 1e5)^(Rᵈ_ncar / cᵖ_ncar)
    q₀ = d.vapor_mixing_ratio / (1 + d.vapor_mixing_ratio)

    # The frame rises with g = 9.8; buoyancy is off (g = 0 in the model constants), as in NCAR
    frame_constants = ncar_constants(g_ncar, FT)
    constants = ncar_constants(0.0, FT)
    dynamics = BoussinesqDynamics(grid, frame_constants; pressure=p₀, temperature=T₀, vapor_mass_fraction=q₀,
                                  frame_vertical_velocity=ascent_velocity, adiabatic_expansion=nothing)
    ρᵃ = Thermodynamics.reference_value(dynamics.reference_state.density)

    droplets = (x = d.x, y = d.y, z = d.z, r = d.radius, rᵈ = d.dry_radius, κ = fill(κ_ncar, length(d.radius)))
    if get(ENV, "NO_DROPLETS", "0") == "1" # the flow and scalar fields alone, to isolate rounding drift of θ and qᵛ
        droplets = map(a -> a[1:0], droplets)
    end
    microphysics = LagrangianMicrophysics(FT; physics=ncar_droplet_physics(FT), self_interaction, droplets)

    forcing = ConstantPowerBandForcing(grid; energy_injection_rate=0.05)
    closure = ScalarDiffusivity(; ν, κ=(ρθ=2.22e-5, ρqᵛ=2.55e-5))

    background = AdiabaticBackground(lapse)
    update_background!(background, p₀, T₀, q₀, g_ncar, 1e5, constants)
    @info "Adiabatic background" background
    forcings = quiescent ? NamedTuple() : (u=forcing, v=forcing, w=forcing)
    forcings = lapse === :none ? forcings : merge(forcings, background_forcings(background, ρᵃ))

    precision = thermodynamic_float_type === nothing ? (;) : (; thermodynamic_float_type)
    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants=constants, closure,
                            advection=Centered(order=4), forcing=forcings, precision...)

    # Initial flow, temperature and vapor perturbations from the NCAR spectral fields
    u₀, v₀, w₀ = staggered_velocities(zk, M)
    T′ = zk.T === nothing ? zeros(M, M, M) : centered_scalar(zk, :T, M)
    r′ = zk.qv === nothing ? zeros(M, M, M) : centered_scalar(zk, :qv, M)
    rᵛ = d.vapor_mixing_ratio .+ r′
    if quiescent # no flow and no initial fluctuations: droplets only settle through still air
        # optionally translated uniformly through the grid (Galilean invariant, so any change is numerical)
        u₀, v₀, w₀ = (fill(U, size(u)) for (U, u) in zip(uniform_velocity, (u₀, v₀, w₀)))
        T′, r′ = zero(T′), zero(r′)
        rᵛ = d.vapor_mixing_ratio .+ r′
    end
    model.clock.time = d.time # before set!, which starts the clock of the rising frame
    set!(model; T = T₀ .+ T′, qᵗ = rᵛ ./ (1 .+ rᵛ), u=u₀, v=v₀, w=w₀)

    state = (rows = (time=Float64[], pressure=Float64[], temperature=Float64[], vapor_mixing_ratio=Float64[],
                     parcel_supersaturation=Float64[], mean_supersaturation=Float64[], std_supersaturation=Float64[],
                     droplet_mean_supersaturation=Float64[], droplet_std_supersaturation=Float64[],
                     liquid_water_content=Float64[], mean_radius=Float64[], std_radius=Float64[],
                     mean_radius_cubed=Float64[], std_temperature=Float64[], std_vapor=Float64[],
                     kinetic_energy=Float64[], droplets=Int[], time_step=Float64[], condensing_cells=Float64[],
                     theta_increment=Float64[], vapor_increment=Float64[], theta_spacing=Float64[],
                     vapor_spacing=Float64[], theta_subhalf_fraction=Float64[], vapor_subhalf_fraction=Float64[]),
             dsd = (time=Float64[], counts=Vector{Int}[]),
             pdfs = NamedTuple[], pdf_times = d.time .+ [0.0, 1.0, 2.0, 5.0, 10.0], pdf_index = Ref(1),
             constants, L, M, ρᵃ)

    Δ = L / M
    umax₀ = max(maximum(abs, u₀), maximum(abs, v₀), maximum(abs, w₀), 0.1)
    simulation = Simulation(model; Δt=min(max_Δt, cfl * Δ / umax₀), stop_time=d.time + stop_time, verbose=false)
    wizard = TimeStepWizard(; cfl, max_change=1.1, diffusive_cfl=0.2, max_Δt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))
    simulation.callbacks[:record] = Callback(sim -> record!(state, sim.model), TimeInterval(output_interval))
    simulation.callbacks[:background] = Callback(sim -> update_background!(background, sim.model, g_ncar, constants),
                                                 IterationInterval(10))
    emulation = get(ENV, "STORAGE_EMULATION", get(ENV, "FLUCTUATION_STORAGE", "0") == "1" ? "fluct" : "none")
    if emulation != "none"
        # Emulate single-precision storage of θ and qᵛ in a model that keeps them in Float64 (THERMO_FLOAT=Float64):
        # after every step, round ρθ and ρqᵛ to Float32 (rn: to nearest, sr: stochastically) or round their
        # fluctuations about the initial means to Float32 (fluct)
        ρθ, ρqᵛ = model.formulation.potential_temperature_density, model.moisture_density
        ρθ₀, ρqᵛ₀ = mean(ρθ), mean(ρqᵛ)
        round_fluctuation!(φ, φ₀) = parent(φ) .= φ₀ .+ Float64.(Float32.(parent(φ) .- φ₀))
        round_nearest!(φ) = parent(φ) .= Float64.(Float32.(parent(φ)))
        function round_stochastically!(φ)
            u = CUDA.rand(Float64, size(parent(φ)))
            parent(φ) .= stochastic_float32.(parent(φ), u)
        end
        simulation.callbacks[:storage_emulation] = Callback(IterationInterval(1)) do sim
            if emulation == "fluct"
                round_fluctuation!(ρθ, ρθ₀)
                round_fluctuation!(ρqᵛ, ρqᵛ₀)
            elseif emulation == "rn"
                round_nearest!(ρθ)
                round_nearest!(ρqᵛ)
            elseif emulation == "sr"
                round_stochastically!(ρθ)
                round_stochastically!(ρqᵛ)
            end
            update_state!(sim.model)
        end
    end
    simulation.callbacks[:progress] = Callback(IterationInterval(500)) do sim
        rows = state.rows
        @info @sprintf("%s: t = %.3f s, Δt = %.2e, S = %.4f%%, r̄ = %.3f μm, σʳ = %.4f μm", label, sim.model.clock.time,
                       sim.Δt, 100 * rows.parcel_supersaturation[end], 1e6 * rows.mean_radius[end], 1e6 * rows.std_radius[end])
    end

    wall = time_ns()
    run!(simulation)
    elapsed = (time_ns() - wall) / 1e9
    @info @sprintf("%s: %d iterations in %.1f s (%.3f ms per step including diagnostics)", label,
                   model.clock.iteration, elapsed, 1e3 * elapsed / max(1, model.clock.iteration))

    write_timing(output_dir, label, model, simulation.Δt)

    write_csv(joinpath(output_dir, "breeze_l3_$(label)_timeseries.csv"), state.rows)
    open(joinpath(output_dir, "breeze_l3_$(label)_dsd.csv"), "w") do io
        for (t, c) in zip(state.dsd.time, state.dsd.counts)
            println(io, join([@sprintf("%.6f", t); string.(c)], ","))
        end
    end
    for s in state.pdfs
        name = @sprintf("breeze_l3_%s_t%.1f", label, s.time)
        write_csv(joinpath(output_dir, name * "_cell_supersaturation.csv"), (supersaturation = s.cell,))
        write_csv(joinpath(output_dir, name * "_droplets.csv"), (supersaturation = s.droplet, radius = s.radius))
    end
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    ncar_dir = ARGS[1]
    runs = [(parse(Int, a), parse(Int, b)) for (a, b) in split.(split(ARGS[2], ','), ':')]
    stop_time = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : 10.0

    arch = get(ENV, "ARCH", CUDA.functional() ? "GPU" : "CPU") == "GPU" ? GPU() : CPU()
    cfl = parse(Float64, get(ENV, "CFL", "0.3"))
    max_Δt = parse(Float64, get(ENV, "MAX_DT", "1e-3"))
    output_interval = parse(Float64, get(ENV, "OUTPUT_INTERVAL", "0.02"))
    output_dir = get(ENV, "OUTPUT_DIR", joinpath(@__DIR__, "output", "l3"))
    FT = get(ENV, "FLOAT", "Float64") == "Float32" ? Float32 : Float64
    thermodynamic_float_type = Dict("" => nothing, "Float64" => Float64, "Float32" => Float32)[get(ENV, "THERMO_FLOAT", "")]
    suffix = get(ENV, "LABEL", "")
    lapse = background_kind(get(ENV, "LAPSE", "none"))
    self = get(ENV, "SELF", "1")
    self_interaction = self == "1" ? DiffusiveSelfInteraction() :
                       self == "2" ? DiffusiveSelfInteraction(grid_translation=false) : nothing
    quiescent = get(ENV, "QUIESCENT", "0") == "1"
    uniform_velocity = Tuple(parse.(Float64, split(get(ENV, "UNIFORM_VELOCITY", "0,0,0"), ',')))
    mkpath(output_dir)

    for (seed, M) in runs
        run_l3(ncar_dir, seed, M; arch, stop_time, cfl, max_Δt, output_interval, output_dir, suffix, lapse, self_interaction, quiescent,
               uniform_velocity, FT, thermodynamic_float_type)
    end
end
