#####
##### Verify Klemp damping doesn't break hydrostatic balance, IGW, etc.
#####

include("common.jl")

using Breeze, Oceananigans, Oceananigans.Units, CUDA, Statistics, Printf

const arch = CUDA.functional() ? GPU() : CPU()

# === HYDROSTATIC BALANCE TEST ===
@info "=== Hydrostatic balance with KlempDivergenceDamping (default coef = 0.1) ==="
let
    grid = RectilinearGrid(arch; size = (64, 64), halo = (5, 5),
                           x = (-10e3, 10e3), z = (0, 10e3),
                           topology = (Periodic, Flat, Bounded))
    θ̄(z) = 300.0 * exp(1e-4 * z / 9.80665)
    for damping in (NoDivergenceDamping(), KlempDivergenceDamping(coefficient = 0.1))
        td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55, damping = damping)
        dyn = CompressibleDynamics(td; reference_potential_temperature = θ̄)
        model = AtmosphereModel(grid; dynamics = dyn,
                               advection = WENO(order = 9),
                               thermodynamic_constants = ThermodynamicConstants(eltype(grid)),
                               timestepper = :AcousticRungeKutta3)
        ref = model.dynamics.reference_state
        set!(model; θ = (x,z) -> θ̄(z), ρ = ref.density)
        sim = Simulation(model; Δt = 1.0, stop_iteration = 600, verbose = false)
        run!(sim)
        wmax = Float64(maximum(abs, interior(model.velocities.w)))
        umax = Float64(maximum(abs, interior(model.velocities.u)))
        nan = any(isnan, parent(model.velocities.w))
        damping_name = damping isa NoDivergenceDamping ? "NoDamp" : "Klemp(0.1)"
        mark = nan ? "NaN" : "✓"
        @info @sprintf("  %s %-12s wmax=%.3e umax=%.3e", mark, damping_name, wmax, umax)
    end
end

# === IGW (test iv) ===
@info "=== IGW with KlempDivergenceDamping ==="
let
    Lx, Lz, Nx, Nz = 300e3, 10e3, 300, 50
    θ̄(z) = 300.0 * exp(1e-4 * z / 9.80665)
    σ_pulse_x = 5e3
    Δθ_pulse = 1e-2
    grid = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                           x = (-Lx/2, Lx/2), z = (0, Lz),
                           topology = (Periodic, Flat, Bounded))
    θᵢ(x, z) = θ̄(z) + Δθ_pulse * sin(π * z / Lz) / (1 + (x / σ_pulse_x)^2)
    for damping in (NoDivergenceDamping(), KlempDivergenceDamping(coefficient = 0.1))
        td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55, damping = damping)
        dyn = CompressibleDynamics(td; reference_potential_temperature = θ̄)
        model = AtmosphereModel(grid; dynamics = dyn, advection = WENO(order = 5),
                               timestepper = :AcousticRungeKutta3)
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ, ρ = ref.density, u = (x,z) -> 20.0)
        sim = Simulation(model; Δt = 1.0, stop_time = 600.0, verbose = false)
        wmax_ov = Ref(0.0)
        function _track_igw(sim)
            wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
            wmax_ov[] = max(wmax_ov[], wmax)
            return nothing
        end
        add_callback!(sim, _track_igw, IterationInterval(2))
        try; run!(sim); catch; end
        damping_name = damping isa NoDivergenceDamping ? "NoDamp" : "Klemp(0.1)"
        mark = any(isnan, parent(model.velocities.w)) ? "NaN" : "✓"
        @info @sprintf("  %s %-12s wmax_overall=%.4e", mark, damping_name, wmax_ov[])
    end
end

# === BUBBLE at multiple Δt ===
@info "=== Bubble Δt sweep with Klemp damping ==="
let
    Lx, Lz, Nx, Nz = 20e3, 10e3, 64, 64
    Δθ = 0.001
    r₀ = 2e3
    θ̄(z) = 300.0 * exp(1e-4 * z / 9.80665)
    grid_builder() = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                                     x = (-Lx/2, Lx/2), z = (0, Lz),
                                     topology = (Periodic, Flat, Bounded))
    θᵢ_builder(grid) = let
        x₀ = mean(xnodes(grid, Center())); z₀ = 0.3 * grid.Lz
        (x, z) -> θ̄(z) + Δθ * exp(-((x-x₀)^2 + (z-z₀)^2) / r₀^2)
    end

    # Explicit ground truth
    grid = grid_builder()
    expl = AtmosphereModel(grid; dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = θ̄),
                           advection = WENO(order = 9))
    ref = expl.dynamics.reference_state
    set!(expl; θ = θᵢ_builder(grid), ρ = ref.density)
    sim = Simulation(expl; Δt = 0.05, stop_time = 300.0, verbose = false)
    wmax_expl_ref = Ref(0.0)
    function _track_expl(sim)
        wmax_expl_ref[] = max(wmax_expl_ref[], Float64(maximum(abs, interior(sim.model.velocities.w))))
        return nothing
    end
    add_callback!(sim, _track_expl, IterationInterval(60))
    run!(sim)
    wmax_expl = wmax_expl_ref[]
    @info @sprintf("  explicit wmax_overall=%.4e", wmax_expl)

    # Substepper at various Δt with Klemp(0.1)
    for Δt in (0.5, 1.0, 1.25, 1.5, 1.75, 2.0)
        grid = grid_builder()
        td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55,
                                             damping = KlempDivergenceDamping(coefficient = 0.1))
        dyn = CompressibleDynamics(td; reference_potential_temperature = θ̄)
        m = AtmosphereModel(grid; dynamics = dyn, advection = WENO(order = 9),
                            timestepper = :AcousticRungeKutta3)
        set!(m; θ = θᵢ_builder(grid), ρ = m.dynamics.reference_state.density)
        sim = Simulation(m; Δt, stop_time = 300.0, verbose = false)
        wmax_ov = Ref(0.0)
        function _track_subs(sim)
            wmax_ov[] = max(wmax_ov[], Float64(maximum(abs, interior(sim.model.velocities.w))))
            return nothing
        end
        add_callback!(sim, _track_subs, IterationInterval(max(1, round(Int, 30.0/Δt))))
        try; run!(sim); catch; end
        nan = any(isnan, parent(m.velocities.w))
        ratio = wmax_ov[] / wmax_expl
        @info @sprintf("  %s Δt=%.1f Klemp(0.1)  wmax_overall=%.4e  ratio=%.3f",
                       nan ? "NaN" : "✓", Δt, wmax_ov[], ratio)
    end
end
