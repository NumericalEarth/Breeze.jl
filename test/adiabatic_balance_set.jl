#####
##### Tests for `balance_adiabatically!(model, balancer)` / `set!(model; balancer = …)` — in-place
##### adiabatic (FV3 na_init) initialization on a memory-sharing, stripped twin
##### (`adiabatic_balance_twin`).
#####
##### - Memory sharing: the twin aliases every heavy field + the stepper's Gⁿ/U⁰ tendency storage,
#####   and is correctly stripped: a reversible LEAPFROG stepper, ExplicitTimeStepping dynamics,
#####   Centered(order=2) advection, no microphysics/closure/radiation, own clock. The excursion is a
#####   Lynch–Huang digital filter (`span` forward + `span` backward leapfrog steps, low-pass average);
#####   these three — leapfrog, fully-explicit acoustic, and centered advection — are exactly the
#####   pieces that make the ±Δt trajectory reversible so the filter can remove the fast acoustic modes.
##### - Moisture-key remap: a SaturationAdjustment production model (:ρqᵉ) maps to the twin's :ρqᵛ.
##### - resolve_balance_Δt: auto acoustic-CFL step, explicit override round-trips.
##### - Fixed point: a discrete-balanced rest state is preserved; the production clock is untouched.
##### - with_moisture: false preserves ρqᵉ bit-for-bit; true lets it relax.
##### - Equivalence + non-degeneracy: the in-place twin reproduces an independent leapfrog+Centered
#####   graft, and an out-of-balance analysis develops nonzero ρw (catches a silent no-op / NaN blow-up).
##### - Anelastic: the balance also works on AnelasticDynamics (dynamics reused as-is).
#####

using Breeze
using Breeze: dynamics_density, AdiabaticBalancer
using Breeze.AtmosphereModels: adiabatic_balance_twin, compute_pressure_correction!,
                               make_pressure_correction!, resolve_balance_Δt
using Breeze.TimeSteppers: Leapfrog
using Oceananigans
using Oceananigans: Centered
using Oceananigans.Grids: minimum_zspacing
using Oceananigans.TimeSteppers: update_state!
using Test

const T₀_BS  = 250.0
const G_BS   = 9.80665
const CPD_BS = 1005.0
θ_iso_bs(z) = T₀_BS * exp(G_BS * z / (CPD_BS * T₀_BS))

ρθ_of(model) = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)

# Production-style compressible model: split-explicit WITH an upper sponge and (default) divergence
# damping — exactly the ingredients the twin must strip for a reversible adiabatic excursion.
function _build_production(arch; microphysics = nothing, Nx = 8, Ny = 8, Nz = 16, Lz = 10e3, Lh = 100e3)
    grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (0, Lh), y = (0, Lh), z = (0, Lz),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(eltype(grid))
    td  = SplitExplicitTimeDiscretization(sponge = UpperSponge(damping_rate = 1/5, depth = 2000.0))
    dyn = CompressibleDynamics(td; reference_potential_temperature = θ_iso_bs,
                               surface_pressure = 1e5, standard_pressure = 1e5)
    return AtmosphereModel(grid; dynamics = dyn, thermodynamic_constants = constants,
                           microphysics, coriolis = nothing, timestepper = :AcousticRungeKutta3)
end

function _set_discrete_rest!(model)
    ref = model.dynamics.reference_state
    Rᵈ  = Breeze.dry_air_gas_constant(model.thermodynamic_constants)
    parent(model.dynamics.dry_density) .= parent(ref.density)
    parent(ρθ_of(model)) .= parent(ref.pressure) ./ (Rᵈ .* parent(ref.exner_function))
    fill!(parent(model.velocities.u), 0)
    fill!(parent(model.velocities.v), 0)
    fill!(parent(model.velocities.w), 0)
    update_state!(model)
    return nothing
end

@testset "AdiabaticBalancer / balance_adiabatically!(model, balancer)" begin

    @testset "twin shares memory and is correctly stripped" begin
        model = _build_production(default_arch; Nx = 32, Ny = 32, Nz = 48)
        twin  = adiabatic_balance_twin(model)

        # Every heavy field is the SAME object, not a copy.
        @test twin.momentum.ρu      === model.momentum.ρu
        @test twin.momentum.ρv      === model.momentum.ρv
        @test twin.momentum.ρw      === model.momentum.ρw
        @test twin.velocities       === model.velocities
        @test twin.moisture_density === model.moisture_density
        @test twin.temperature      === model.temperature
        @test twin.pressure_solver  === model.pressure_solver
        @test dynamics_density(twin.dynamics) === dynamics_density(model.dynamics)
        # ρθ is rebuilt over the SAME data with microphysics-stripped surface-flux BCs
        # (`adiabatic_twin_formulation`), so it is a distinct Field wrapper but shares storage — and it
        # is that shared storage that lets the twin balance ρθ in place.
        @test ρθ_of(twin).data === ρθ_of(model).data
        # Stepper tendency storage is aliased, not reallocated.
        @test twin.timestepper.Gⁿ.ρu === model.timestepper.Gⁿ.ρu
        @test twin.timestepper.U⁰.ρθ === model.timestepper.U⁰.ρθ

        # Stripped for a reversible adiabatic excursion: no microphysics, no closure/diffusion
        # (the implicit vertical-diffusion solve is irreversible), no radiation.
        @test twin.microphysics === nothing
        @test twin.closure      === nothing
        @test twin.timestepper.implicit_solver === nothing
        @test twin.radiation    === nothing
        # Reversible leapfrog stepper (not the production RK3) + fully-explicit acoustic dynamics: the
        # digital-filter excursion needs an exactly time-reversible trajectory.
        @test twin.timestepper isa Leapfrog
        @test twin.dynamics.time_discretization isa ExplicitTimeStepping
        @test model.dynamics.time_discretization isa SplitExplicitTimeDiscretization
        # Centered (non-dissipative ⇒ reversible) advection replaces the production WENO.
        @test twin.advection.momentum isa Centered
        @test twin.clock !== model.clock

        # The leapfrog carries one EXTRA time level uⁿ⁻¹ (`Uᵐ`, one field per prognostic) that cannot
        # alias anything — that is the twin's only heavy allocation. On top of it the builder's
        # NamedTuple/forcing/stepper scratch is grid-INDEPENDENT (~270 KB), well under one field on this
        # large grid. So the twin allocates ≈ (nprognostic + <1) fields; reallocating either aliased
        # tendency set (Gⁿ or U⁰, nprognostic fields each) instead of sharing it would roughly double it.
        one_field_bytes = sizeof(parent(model.momentum.ρu))
        nprog = length(Oceananigans.prognostic_fields(twin))
        adiabatic_balance_twin(model)  # compile
        @test (@allocated adiabatic_balance_twin(model)) < (nprog + 2) * one_field_bytes
    end

    @testset "compressible twin ignores time_stepping (always fully-explicit leapfrog)" begin
        model = _build_production(default_arch)

        # The compressible twin's stepper is the reversible leapfrog, which does no acoustic
        # substepping — so a split-explicit discretization would be incoherent. The `time_stepping`
        # request is therefore ignored for CompressibleDynamics: the twin is ALWAYS
        # ExplicitTimeStepping + Leapfrog, whatever is passed.
        for ts in (nothing, SplitExplicitTimeDiscretization(), ExplicitTimeStepping())
            twin = adiabatic_balance_twin(model, AdiabaticBalancer(time_stepping = ts))
            @test twin.dynamics.time_discretization isa ExplicitTimeStepping
            @test twin.timestepper isa Leapfrog
            # Tendency storage is still aliased regardless.
            @test twin.timestepper.Gⁿ.ρu === model.timestepper.Gⁿ.ρu
            @test twin.momentum.ρw === model.momentum.ρw
        end

        # Fixed point still holds with an explicit time_stepping override.
        _set_discrete_rest!(model)
        ρ₀ = deepcopy(dynamics_density(model.dynamics))
        set!(model; balancer = AdiabaticBalancer(Δt = 0.5, span = 4, time_stepping = nothing))
        @test dynamics_density(model.dynamics) ≈ ρ₀ rtol=1e-8
        @test isapprox(maximum(abs, model.momentum.ρw), 0; atol=1e-7)
        @test model.clock.iteration == 0
    end

    @testset "moisture-key remap (SaturationAdjustment :ρqᵉ → twin :ρqᵛ)" begin
        model = _build_production(default_arch; microphysics = SaturationAdjustment())
        @test Breeze.AtmosphereModels.moisture_prognostic_name(model.microphysics) == :ρqᵉ
        twin = adiabatic_balance_twin(model)
        @test twin.moisture_density === model.moisture_density
        @test twin.timestepper.Gⁿ.ρqᵛ === model.timestepper.Gⁿ.ρqᵉ
        @test :ρqᵛ ∈ keys(Oceananigans.prognostic_fields(twin))
    end

    @testset "resolve_balance_Δt" begin
        model = _build_production(default_arch)
        _set_discrete_rest!(model)
        @test resolve_balance_Δt(AdiabaticBalancer(Δt = 0.123), model) ≈ 0.123  # explicit round-trips
        Δt_auto = resolve_balance_Δt(AdiabaticBalancer(), model)               # auto from acoustic CFL
        @test 0 < Δt_auto < minimum_zspacing(model.grid) / 250                # ≈ Δz/c, c ≳ 300 m/s
    end

    @testset "fixed point: discrete rest state preserved; production clock untouched" begin
        model = _build_production(default_arch)
        _set_discrete_rest!(model)
        ρ₀  = deepcopy(dynamics_density(model.dynamics))
        ρθ₀ = deepcopy(ρθ_of(model))

        set!(model; balancer = true)   # exercises the Bool dispatch + auto-Δt path (rest is a fixed point for any Δt)

        @test dynamics_density(model.dynamics) ≈ ρ₀  rtol=1e-8
        @test ρθ_of(model)                     ≈ ρθ₀ rtol=1e-8
        @test isapprox(maximum(abs, model.momentum.ρw), 0; atol=1e-7)
        @test model.clock.iteration == 0
        @test model.clock.time == 0
    end

    @testset "with_moisture: false preserves ρqᵉ exactly, true lets it relax" begin
        # Seed a horizontally-uniform vapor perturbation so the nudge would move ρqᵉ.
        model = _build_production(default_arch; microphysics = SaturationAdjustment())
        _set_discrete_rest!(model)
        set!(model; qᵗ = (x, y, z) -> 1e-3 * exp(-z / 2e3))
        set!(model; w = (x, y, z) -> 0.01 * sin(2π * z / 2e3))

        ρqᵉ₀ = deepcopy(model.moisture_density)
        set!(model; balancer = AdiabaticBalancer(Δt = 0.1, with_moisture = false))
        @test maximum(abs, model.moisture_density - ρqᵉ₀) == 0   # preserved bit-for-bit (snapshot/restore)

        set!(model; balancer = AdiabaticBalancer(Δt = 0.1, with_moisture = true))
        @test isfinite(maximum(abs, model.moisture_density))
    end

    @testset "in-place twin reproduces a hand-stripped leapfrog graft (non-degenerate)" begin
        # Shared balancer: Path A uses it directly; Path B mirrors its span and robert_asselin so both
        # paths run the identical DFI. Sourcing them from the balancer keeps the test tracking the
        # defaults rather than hard-coding them.
        balancer = AdiabaticBalancer(Δt = 0.5, span = 4, with_moisture = false)
        Δt = balancer.Δt

        # Near-balanced moist analysis shared by both paths: discrete rest state + a small smooth
        # thermal bump (mild hydrostatic imbalance → ρw spins up but stays finite) + subsaturated qᵗ.
        function analysis!(m)
            _set_discrete_rest!(m)
            set!(m; qᵗ = (x, y, z) -> 4e-3 * exp(-z / 3e3))
            δ = CenterField(m.grid)
            set!(δ, (x, y, z) -> 5.0 * exp(-((z - 5e3) / 1.5e3)^2))
            parent(ρθ_of(m)) .+= parent(δ)
            update_state!(m)
            return m
        end

        # Path A — in-place twin on a moist production model. The compressible twin balances
        # (ρ, ρu, ρv, ρw, ρθ) on a leapfrog; with_moisture = false preserves ρqᵉ.
        model = _build_production(default_arch; microphysics = SaturationAdjustment())
        analysis!(model)
        ρqᵉ_analysis = deepcopy(model.moisture_density)
        balance_adiabatically!(model, balancer)

        # Path B — an independent, hand-stripped moistureless model built to mirror the twin EXACTLY:
        # ExplicitTimeStepping dynamics, a Leapfrog stepper, and Centered(order=2) advection on
        # momentum and scalars. Balanced via the low-level routine with the balancer's span and a
        # matched Robert–Asselin coefficient, so the two DFI trajectories are bit-for-bit identical.
        grid = model.grid
        dyn  = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = θ_iso_bs,
                                    surface_pressure = 1e5, standard_pressure = 1e5)
        ref  = AtmosphereModel(grid; dynamics = dyn,
                               thermodynamic_constants = ThermodynamicConstants(eltype(grid)),
                               microphysics = nothing, coriolis = nothing,
                               momentum_advection = Centered(order = 2),
                               scalar_advection = Centered(order = 2),
                               timestepper = :Leapfrog)
        ref.timestepper.robert_asselin = convert(eltype(grid), balancer.robert_asselin)
        analysis!(ref)
        balance_adiabatically!(ref; Δt = Δt, span = balancer.span)

        # The in-place twin reproduces the graft (identical leapfrog scheme + identical IC).
        for (a, b) in zip((dynamics_density(model.dynamics), model.momentum.ρu, model.momentum.ρv,
                           model.momentum.ρw, ρθ_of(model)),
                          (dynamics_density(ref.dynamics), ref.momentum.ρu, ref.momentum.ρv,
                           ref.momentum.ρw, ρθ_of(ref)))
            @test a ≈ b rtol=1e-8
        end

        # Non-degenerate: the imbalance develops nonzero ρw (else the match above is vacuous; this
        # also catches a NaN blow-up, since `NaN > 0` is false).
        @test maximum(abs, model.momentum.ρw) > 0
        @test maximum(abs, ref.momentum.ρw)   > 0

        # ρqᵉ preserved bit-for-bit — the graft never touches moisture.
        @test maximum(abs, model.moisture_density - ρqᵉ_analysis) == 0
    end

    @testset "edge case: Bool false is a no-op" begin
        model = _build_production(default_arch)
        @test balance_adiabatically!(model, false) === model   # Bool false → no-op
    end

    @testset "AnelasticDynamics: twin reuses the projection scheme; rest state preserved" begin
        grid = RectilinearGrid(default_arch; size = (8, 8, 16), halo = (3, 3, 3),
                               x = (0, 1e4), y = (0, 1e4), z = (0, 4e3),
                               topology = (Periodic, Periodic, Bounded))
        constants = ThermodynamicConstants(eltype(grid))
        reference_state = ReferenceState(grid, constants; surface_pressure = 1e5,
                                         potential_temperature = 300, vapor_mass_fraction = 0)
        model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), microphysics = nothing)

        # The anelastic twin shares memory and reuses the dynamics as-is (no time-discretization to
        # swap); time_stepping is ignored. Tendency storage is still aliased. Crucially it KEEPS its
        # native projection stepper — anelastic has no acoustic modes (so no reversibility problem)
        # and needs the pressure projection every step, which the compressible leapfrog omits.
        twin = adiabatic_balance_twin(model)
        @test twin.dynamics === model.dynamics
        @test twin.momentum.ρw === model.momentum.ρw
        @test twin.timestepper.Gⁿ.ρu === model.timestepper.Gⁿ.ρu
        @test twin.microphysics === nothing
        @test !(twin.timestepper isa Leapfrog)
        @test typeof(twin.timestepper) == typeof(model.timestepper)

        # θ = θᵣ at rest is an anelastic fixed point. Use the positional `(model, balancer)` interface.
        set!(model; θ = (x, y, z) -> 300.0)
        ρθ₀ = deepcopy(ρθ_of(model))
        balance_adiabatically!(model, AdiabaticBalancer(Δt = 1.0, span = 4))
        @test ρθ_of(model) ≈ ρθ₀ rtol=1e-6
        @test isapprox(maximum(abs, model.momentum.ρw), 0; atol=1e-6)
        @test model.clock.iteration == 0
    end

    @testset "AnelasticDynamics: set! with balancer preserves mass projection" begin
        grid = RectilinearGrid(default_arch; size = (8, 8, 16), halo = (3, 3, 3),
                               x = (0, 1e4), y = (0, 1e4), z = (0, 4e3),
                               topology = (Periodic, Periodic, Bounded))
        constants = ThermodynamicConstants(eltype(grid))
        reference_state = ReferenceState(grid, constants; surface_pressure = 1e5,
                                         potential_temperature = 300, vapor_mass_fraction = 0)
        model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), microphysics = nothing)

        set!(model;
             θ = (x, y, z) -> 300 + sin(2π * x / 1e4) * cos(2π * z / 4e3),
             u = (x, y, z) -> 50 * sin(2π * x / 1e4) * cos(2π * z / 4e3),
             v = (x, y, z) -> 50 * cos(2π * y / 1e4) * sin(2π * z / 4e3),
             w = 0,
             enforce_mass_conservation = true,
             balancer = AdiabaticBalancer(Δt = 2.0, span = 4))

        ρu = Array(interior(model.momentum.ρu))
        ρv = Array(interior(model.momentum.ρv))
        ρw = Array(interior(model.momentum.ρw))

        compute_pressure_correction!(model, 1)
        make_pressure_correction!(model, 1)
        update_state!(model, compute_tendencies = false)

        projection_change = maximum((
            maximum(abs, Array(interior(model.momentum.ρu)) .- ρu),
            maximum(abs, Array(interior(model.momentum.ρv)) .- ρv),
            maximum(abs, Array(interior(model.momentum.ρw)) .- ρw)))

        momentum_scale = maximum((1, maximum(abs, ρu), maximum(abs, ρv), maximum(abs, ρw)))
        @test projection_change <= 1e-8 * momentum_scale
    end

end
