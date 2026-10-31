# Moist Bryan & Fritsch (2002) benchmark on the compressible and anelastic cores. See CRITERIA.md.
#
#     FORMULATIONS=C-theta,C-E273 MICRO=equilibrium RESOLUTIONS=100 NULL=0 TAU=2 \
#         julia --project=examples validation/moist_bubble/bubble.jl
#
# Saturated neutral environment: rᵗ = 0.020, reversible θₑ = 320 K, p(0) = 1000 hPa, in discrete hydrostatic balance on
# the model's cell centers (compressible); the anelastic core uses its own reference pressure. Bubble: θρ′ / θ̄ρ = Δθ / 300,
# saturated at the unperturbed pressure, as in Bryan's code (CM1 isnd = 4, iinit = 4). No rain, no sedimentation.
#
# MICRO = equilibrium:    SaturationAdjustment(WarmPhaseEquilibrium()).
# MICRO = nonequilibrium: OneMomentCloudMicrophysics with no rain autoconversion, cloud liquid relaxing to saturation
#                         with τ = TAU seconds, and the Stokes fall speed of cloud droplets turned off.
#
# Writes $RESULTS_DIR/<micro>[-null]_<formulation>_<Δ>m[_tau<τ>].{csv,bin,txt}: the time series, the fields
# (T, p, ρ, qᵛ, qˡ, u, w at centers, Nx × Nz Float64, at OUTPUT_TIMES), and the initial-state checks.

include(joinpath(@__DIR__, "common.jl"))
using Breeze.Thermodynamics: CondensedPhase, saturation_vapor_pressure, PlanarLiquidSurface, MoistureMassFractions, moist_entropy
using CloudMicrophysics
using CloudMicrophysics.Parameters: Microphysics1MParams, StokesRegimeVelType, TerminalVelocityParams
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, one_moment_cloud_microphysics_categories

micro = get(ENV, "MICRO", "equilibrium")
null_run = get(ENV, "NULL", "0") == "1"
τ_relax = parse(Float64, get(ENV, "TAU", "2"))
formulations = parse_list("FORMULATIONS", "C-theta,C-E273,A-theta")
resolutions = parse.(Float64, parse_list("RESOLUTIONS", "100"))
stop_time = parse(Float64, get(ENV, "STOP_TIME", "1000"))
Δt_override = haskey(ENV, "DT") ? parse(Float64, ENV["DT"]) : nothing
outdir = get(ENV, "RESULTS_DIR", joinpath(@__DIR__, "results"))
mkpath(outdir)

const R̄ = 8.314462618
const bf02 = (Rd = 287.04, Rv = 461.5, cpd = 1005.7, cpv = 1870.0, cl = 4190.0, L = 2501000.0, Tr = 273.15, g = 9.81)
const constants = ThermodynamicConstants(Float64; molar_gas_constant = R̄, gravitational_acceleration = bf02.g,
                                         energy_reference_temperature = bf02.Tr,
                                         dry_air_molar_mass = R̄ / bf02.Rd, dry_air_heat_capacity = bf02.cpd,
                                         vapor_molar_mass = R̄ / bf02.Rv, vapor_heat_capacity = bf02.cpv,
                                         liquid = CondensedPhase(Float64; reference_latent_heat = bf02.L,
                                                                 heat_capacity = bf02.cl, density = 1000))
const c = Constants64(constants)
const p₀ = 1e5
const rᵗ = 0.020
const qᵗ = rᵗ / (1 + rᵗ)
const ϵ = bf02.Rd / bf02.Rv
const surface = PlanarLiquidSurface()

eₛ(T) = saturation_vapor_pressure(T, constants, surface)

# Reversible θₑ (BF02 eq. for wet θₑ) with mixing ratios rᵛ, total water r, relative humidity H = e / eₛ(T)
function reversible_θₑ(T, p, rᵛ, r)
    e = p * rᵛ / (ϵ + rᵛ)
    cpt = bf02.cpd + bf02.cl * r
    Lᵥ = bf02.L + (bf02.cpv - bf02.cl) * (T - bf02.Tr)
    H = e / eₛ(T)
    return T * (p₀ / (p - e))^(bf02.Rd / cpt) * H^(-rᵛ * bf02.Rv / cpt) * exp(Lᵥ * rᵛ / (cpt * T))
end

function θₑ_from_state(T, p, qᵛ, qˡ)
    qᵈ = 1 - qᵛ - qˡ
    return reversible_θₑ(T, p, qᵛ / qᵈ, (qᵛ + qˡ) / qᵈ)
end

# Saturated partition at (T, p) with rᵗ total water
function saturated(T, p)
    e = eₛ(T)
    rᵛ = min(rᵗ, ϵ * e / (p - e))
    qᵛ = rᵛ / (1 + rᵗ)
    qˡ = qᵗ - qᵛ
    Tᵨ = T * (1 - qᵗ + qᵛ * bf02.Rv / bf02.Rd)
    ρ = p / (bf02.Rd * Tᵨ)
    return (; T, p, qᵛ, qˡ, Tᵨ, ρ, rᵛ)
end

function bisect(f, a, b; iterations = 100)
    fa = f(a)
    for _ in 1:iterations
        m = (a + b) / 2
        fm = f(m)
        (fm > 0) == (fa > 0) ? (a = m; fa = fm) : (b = m)
    end
    return (a + b) / 2
end

neutral_temperature(p, θₑ₀ = 320.0) = bisect(T -> (s = saturated(T, p); reversible_θₑ(T, p, s.rᵛ, rᵗ) - θₑ₀), 150.0, 330.0)
environment(p) = saturated(neutral_temperature(p), p)

# Continuous column integrated from the surface to z with midpoint steps
function column_pressure(z; dz = 2.0)
    n = max(1, ceil(Int, z / dz))
    h = z / n
    p = p₀
    for _ in 1:n
        pᵐ = p - bf02.g * environment(p).ρ * h / 2
        p -= bf02.g * environment(pᵐ).ρ * h
    end
    return p
end

# p at the centers z, in discrete balance (pₖ − pₖ₋₁) / Δz + g (ρₖ + ρₖ₋₁) / 2 = 0
function discrete_column(z)
    p = zeros(length(z))
    p[1] = column_pressure(z[1])
    for k in 2:length(z)
        Δz = z[k] - z[k-1]
        ρ⁻ = environment(p[k-1]).ρ
        pᵏ = p[k-1] - bf02.g * ρ⁻ * Δz
        for _ in 1:30
            pᵏ = p[k-1] - bf02.g * Δz * (ρ⁻ + environment(pᵏ).ρ) / 2
        end
        p[k] = pᵏ
    end
    return p
end

function interpolator(z, a)
    function (zᵢ)
        k = clamp(searchsortedlast(z, zᵢ), 1, length(z) - 1)
        f = (zᵢ - z[k]) / (z[k+1] - z[k])
        return (1 - f) * a[k] + f * a[k+1]
    end
end

function nonequilibrium_microphysics()
    FT = Float64
    parameters = Microphysics1MParams(FT; rain_autoconversion = nothing)
    velocities = TerminalVelocityParams(FT)
    stokes = velocities.stokes
    still = StokesRegimeVelType(; ρw = stokes.ρw, ν_air = stokes.ν_air, grav = zero(FT))
    hydrometeor_velocities = TerminalVelocityParams(; stokes = still, velocities.chen2022, velocities.blk1m)
    categories = one_moment_cloud_microphysics_categories(FT; parameters, hydrometeor_velocities)
    cloud_formation = NonEquilibriumCloudFormation(ConstantRateCondensateFormation(1 / τ_relax), nothing)
    return OneMomentCloudMicrophysics(FT; cloud_formation, categories)
end

microphysics_for(name) = micro == "equilibrium" ? SaturationAdjustment(Float64; equilibrium = WarmPhaseEquilibrium()) :
                                                   nonequilibrium_microphysics()

function write_fields(io, s)
    Nx, Nz = size(s.T)
    uᶜ = (s.u[1:Nx, :] .+ s.u[2:Nx+1, :]) ./ 2
    for a in (s.T, s.p, s.ρ, s.qv, s.ql, uᶜ, s.wc)
        write(io, Float64.(a))
    end
end

function run_bubble(name, Δ)
    t₀ = time_ns()
    microphysics = microphysics_for(name)
    Lx, Lz = 20e3, 10e3
    Nx, Nz = round(Int, Lx / Δ), round(Int, Lz / Δ)
    grid = RectilinearGrid(architecture(); size = (Nx, Nz), halo = (5, 5), x = (-Lx / 2, Lx / 2), z = (0, Lz),
                           topology = (Bounded, Flat, Bounded))
    x = collect(xnodes(grid, Center()))
    z = collect(znodes(grid, Center()))

    # The environment on the centers, for the reference states and the initial state
    p_center = discrete_column(z)
    states = environment.(p_center)
    θ_ref = interpolator(z, [s.T * (c.pst / s.p)^(c.Rd / c.cpd) for s in states])
    qᵛ_ref = interpolator(z, [s.qᵛ for s in states])
    qˡ_ref = interpolator(z, [s.qˡ for s in states])

    if is_compressible(name)
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = p₀,
                                        reference_potential_temperature = θ_ref, reference_vapor_mass_fraction = qᵛ_ref)
    else
        reference_state = ReferenceState(grid, constants; base_pressure = p₀, potential_temperature = θ_ref,
                                         vapor_mass_fraction = qᵛ_ref, liquid_mass_fraction = qˡ_ref)
        a = reference_column_values(reference_state, grid)
        @info @sprintf("%s default reference vs environment: max|Δp|/p %.2e, max|Δρ|/ρ %.2e", name,
                       maximum(abs, (a.p .- p_center) ./ p_center), maximum(abs, (a.ρ .- getfield.(states, :ρ)) ./ getfield.(states, :ρ)))
        overwrite_reference_with_environment!(reference_state, grid, p_center, states)
        dynamics = AnelasticDynamics(reference_state)
    end

    model = AtmosphereModel(grid; dynamics, formulation = thermodynamic_formulation(name),
                            thermodynamic_constants = constants, advection = WENO(order = 5), microphysics)

    # Environmental pressure of each core at the centers: the anelastic reference pressure on the anelastic core
    pₑ = is_compressible(name) ? p_center : reference_column(model, :pressure)

    β(x, z) = sqrt((x / 2000)^2 + ((z - 2000) / 2000)^2)
    Δθ(x, z) = null_run ? 0.0 : (β(x, z) < 1 ? 2 * cos(π * β(x, z) / 2)^2 : 0.0)

    T = zeros(Nx, Nz); ρ = zeros(Nx, Nz); qv = zeros(Nx, Nz); ql = zeros(Nx, Nz)
    for k in 1:Nz
        sₑ = environment(pₑ[k])
        for i in 1:Nx
            Tᵨ = sₑ.Tᵨ * (1 + Δθ(x[i], z[k]) / 300)
            s = Δθ(x[i], z[k]) == 0 ? sₑ : saturated(bisect(T -> saturated(T, pₑ[k]).Tᵨ - Tᵨ, sₑ.T - 5, sₑ.T + 10), pₑ[k])
            T[i, k], ρ[i, k], qv[i, k], ql[i, k] = s.T, s.ρ, s.qᵛ, s.qˡ
        end
    end

    Tf = center_field_from(grid, T)
    kw = microphysics isa SaturationAdjustment ? (; T = Tf, qᵗ = qᵗ) :
                                                 (; T = Tf, qᵛ = center_field_from(grid, qv), qᶜˡ = center_field_from(grid, ql))
    is_compressible(name) ? set!(model; kw..., ρ = center_field_from(grid, ρ)) : set!(model; kw...)

    label = @sprintf("%s%s_%s_%dm%s", micro, null_run ? "-null" : "", name, round(Int, Δ),
                     micro == "equilibrium" ? "" : @sprintf("_tau%g", τ_relax))
    label = isnothing(Δt_override) ? label : label * @sprintf("_dt%g", Δt_override)

    θₑ_field(s) = θₑ_from_state.(s.T, s.p, s.qv, s.ql)

    s₀ = snapshot(model)
    θₑ₀ = θₑ_field(s₀)
    background = θₑ₀[1, :]' # the west wall column is undisturbed at t = 0

    checks = (T_error = maximum(abs, s₀.T .- T),
              ql_error = maximum(abs, s₀.ql .- ql),
              qv_error = maximum(abs, s₀.qv .- qv),
              p_error = maximum(abs, (s₀.p .- pₑ') ./ pₑ'),
              ρ_error = maximum(abs, (s₀.ρ .- ρ) ./ ρ),
              env_θₑ_min = minimum(background), env_θₑ_max = maximum(background),
              bubble_θₑ′_max = maximum(θₑ₀ .- background), bubble_T′_max = maximum(T .- T[1, :]'),
              anelastic_eos_density_error = maximum(abs, (s₀.ρ .- ρ) ./ ρ),
              wall_setup = wall_seconds(t₀))
    open(joinpath(outdir, label * ".txt"), "w") do io
        for (k, v) in pairs(checks)
            println(io, k, " ", v)
        end
        println(io, "T_env ", join(T[1, :], ","))
        println(io, "qv_env ", join(qv[1, :], ","))
        println(io, "p_env ", join(pₑ, ","))
    end
    @info "$label initial state" checks...
    # Environment θₑ from the model column must be 320 K; the intended column is too by construction

    Δt = isnothing(Δt_override) ? Δ / 100 : Δt_override
    simulation = Simulation(model; Δt, stop_time, verbose = false)
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

    Δx = fill(Δ, Nx); Δz = fill(Δ, Nz)
    csv = open(joinpath(outdir, label * ".csv"), "w")
    println(csv, "t,wmax,wmin,umax,θₑ′max,θₑ′min,E,S,M,W,KE,qlmax,ρᵈθₑ′")
    bin = open(joinpath(outdir, label * ".bin"), "w")
    function record(sim)
        s = snapshot(sim.model)
        θₑ′ = θₑ_field(s) .- background
        b = budgets(s, z, Δx, Δz, c, is_compressible(name))
        ρᵈθₑ′ = sum((1 .- s.qv .- s.ql) .* s.ρ .* θₑ′) * Δ * Δ / Lx
        println(csv, join((time(sim), maximum(s.w), minimum(s.w), maximum(abs, s.u), maximum(θₑ′), minimum(θₑ′),
                           b.E, b.S, b.M, b.W, b.KE, maximum(s.ql), ρᵈθₑ′), ","))
        flush(csv)
        return nothing
    end
    dump_fields(sim) = (write_fields(bin, snapshot(sim.model)); flush(bin); nothing)
    record(simulation); dump_fields(simulation)
    add_callback!(simulation, record, TimeInterval(10))
    add_callback!(simulation, dump_fields, TimeInterval(100))
    run!(simulation)
    close(csv); close(bin)
    s = snapshot(model)
    θₑ′ = θₑ_field(s) .- background
    @info @sprintf("%s: θₑ′ ∈ [%.3f, %.3f] K, w ∈ [%.2f, %.2f] m/s, wall %.0f s", label, minimum(θₑ′), maximum(θₑ′),
                   minimum(s.w), maximum(s.w), wall_seconds(t₀))
    return nothing
end

# `ReferenceState` integrates a dry column from θ(z), whatever moisture it is given, so it misses the
# saturated environment's density by the vapor and condensate loading. The anelastic reference should be
# the environment itself, so we write it in: p, ρ and T at the centers, and the surface values for the halos.
function overwrite_reference_with_environment!(ref, grid, p, states)
    ρ = getfield.(states, :ρ)
    T = getfield.(states, :T)
    Nz = length(p)
    for (field, column) in ((ref.pressure, p), (ref.density, ρ), (ref.temperature, T))
        interior(field) .= on_architecture(Oceananigans.Architectures.architecture(grid), reshape(column, 1, 1, Nz))
    end
    # Linear extrapolation to the surface for the bottom boundary values
    z = collect(znodes(grid, Center()))
    extrapolate(a) = a[1] - (a[2] - a[1]) / (z[2] - z[1]) * z[1]
    set!(ref.surface_pressure, extrapolate(p))
    set!(ref.surface_density, extrapolate(ρ))
    set!(ref.surface_temperature, extrapolate(T))
    Oceananigans.BoundaryConditions.fill_halo_regions!(ref.pressure)
    Oceananigans.BoundaryConditions.fill_halo_regions!(ref.density)
    Oceananigans.BoundaryConditions.fill_halo_regions!(ref.temperature)
    return nothing
end

function reference_column_values(reference_state, grid)
    Nz = size(grid, 3)
    p = vec(Float64.(Array(interior(reference_state.pressure))))[1:Nz]
    ρ = vec(Float64.(Array(interior(reference_state.density))))[1:Nz]
    return (; p, ρ)
end

for Δ in resolutions, name in formulations
    try
        run_bubble(name, Δ)
    catch err
        @error "$name Δ = $Δ failed" exception = (err, catch_backtrace())
    end
end
