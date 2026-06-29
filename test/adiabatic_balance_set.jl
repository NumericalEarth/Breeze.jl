#####
##### Tests for `set!(model; balance = AdiabaticBalance(...))` — in-place adiabatic (FV3 na_init)
##### initialization on a memory-sharing, stripped twin (`adiabatic_balance_twin`).
#####
##### - Memory sharing: the twin aliases every heavy field + the stepper's Gⁿ/U⁰ tendency storage,
#####   and is correctly stripped (ExplicitTimeStepping, no microphysics/radiation, own clock).
##### - Moisture-key remap: a SaturationAdjustment production model (:ρqᵉ) maps to the twin's :ρqᵛ.
##### - resolve_balance_Δt: auto acoustic-CFL step, explicit override round-trips.
##### - Fixed point: a discrete-balanced rest state is preserved; the production clock is untouched.
##### - with_moisture: false preserves ρqᵉ bit-for-bit; true lets it relax.
##### - Edge: non-compressible dynamics throws a clear error.
#####

using Breeze
using Breeze: dynamics_density, adiabatic_balance_twin, resolve_balance_Δt, AdiabaticBalance
using Oceananigans
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

@testset "set!(; balance) adiabatic initialization" begin

    @testset "twin shares memory and is correctly stripped" begin
        model = _build_production(default_arch)
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
        @test ρθ_of(twin)           === ρθ_of(model)
        # Stepper tendency storage is aliased, not reallocated.
        @test twin.timestepper.Gⁿ.ρu === model.timestepper.Gⁿ.ρu
        @test twin.timestepper.U⁰.ρθ === model.timestepper.U⁰.ρθ

        # Stripped for a reversible adiabatic excursion.
        @test twin.microphysics === nothing
        @test twin.radiation    === nothing
        @test twin.dynamics.time_discretization isa ExplicitTimeStepping
        @test model.dynamics.time_discretization isa SplitExplicitTimeDiscretization
        @test twin.clock !== model.clock

        # No field SETS allocated: the two aliased tendency sets (Gⁿ+U⁰ ≈ 12 fields) dwarf the
        # one-time NamedTuple/struct scratch the builder does allocate.
        one_field_bytes = sizeof(parent(model.momentum.ρu))
        adiabatic_balance_twin(model)  # compile
        @test (@allocated adiabatic_balance_twin(model)) < 12 * one_field_bytes
    end

    @testset "native time stepping (time_stepping = nothing) reuses the split-explicit scheme" begin
        model = _build_production(default_arch)
        twin  = adiabatic_balance_twin(model, nothing)

        # Native scheme reused, but the (irreversible) sponge is stripped.
        @test twin.dynamics.time_discretization isa SplitExplicitTimeDiscretization
        @test twin.dynamics.time_discretization.sponge === nothing
        @test model.dynamics.time_discretization.sponge !== nothing
        # Tendency storage is still aliased (the rebuilt acoustic substepper is the only new memory).
        @test twin.timestepper.Gⁿ.ρu === model.timestepper.Gⁿ.ρu
        @test twin.momentum.ρw === model.momentum.ρw

        # Fixed point holds with the native scheme too.
        _set_discrete_rest!(model)
        ρ₀ = Array(interior(dynamics_density(model.dynamics)))
        set!(model; balance = AdiabaticBalance(Δt = 0.5, cycles = 1, time_stepping = nothing))
        @test maximum(abs, Array(interior(dynamics_density(model.dynamics))) .- ρ₀) <= 1e-8 * maximum(abs, ρ₀)
        @test maximum(abs, Array(interior(model.momentum.ρw))) <= 1e-7
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
        @test resolve_balance_Δt(AdiabaticBalance(Δt = 0.123), model) == 0.123  # explicit round-trips
        Δt_auto = resolve_balance_Δt(AdiabaticBalance(), model)               # auto from acoustic CFL
        @test 0 < Δt_auto < minimum_zspacing(model.grid) / 250                # ≈ Δz/c, c ≳ 300 m/s
    end

    @testset "fixed point: discrete rest state preserved; production clock untouched" begin
        model = _build_production(default_arch)
        _set_discrete_rest!(model)
        ρ  = dynamics_density(model.dynamics)
        ρθ = ρθ_of(model)
        ρ₀  = Array(interior(ρ))
        ρθ₀ = Array(interior(ρθ))

        set!(model; balance = AdiabaticBalance(Δt = 0.1, cycles = 1))

        @test maximum(abs, Array(interior(ρ))  .- ρ₀)  <= 1e-8 * maximum(abs, ρ₀)
        @test maximum(abs, Array(interior(ρθ)) .- ρθ₀) <= 1e-8 * maximum(abs, ρθ₀)
        @test maximum(abs, Array(interior(model.momentum.ρw))) <= 1e-7
        @test model.clock.iteration == 0
        @test model.clock.time == 0
    end

    @testset "with_moisture: false preserves ρqᵉ exactly, true lets it relax" begin
        # Seed a horizontally-uniform vapor perturbation so the nudge would move ρqᵉ.
        model = _build_production(default_arch; microphysics = SaturationAdjustment())
        _set_discrete_rest!(model)
        set!(model; qᵗ = (x, y, z) -> 1e-3 * exp(-z / 2e3))
        set!(model; w = (x, y, z) -> 0.01 * sin(2π * z / 2e3))

        ρqᵉ₀ = Array(interior(model.moisture_density))
        set!(model; balance = AdiabaticBalance(Δt = 0.1, with_moisture = false))
        @test Array(interior(model.moisture_density)) == ρqᵉ₀   # bit-identical

        set!(model; balance = AdiabaticBalance(Δt = 0.1, with_moisture = true))
        @test all(isfinite, Array(interior(model.moisture_density)))
    end

    @testset "non-compressible dynamics throws a clear error" begin
        grid = RectilinearGrid(default_arch; size = (8, 8, 8), halo = (3, 3, 3),
                               x = (0, 1e4), y = (0, 1e4), z = (0, 4e3),
                               topology = (Periodic, Periodic, Bounded))
        constants = ThermodynamicConstants(eltype(grid))
        reference_state = ReferenceState(grid, constants; surface_pressure = 1e5,
                                         potential_temperature = 300, vapor_mass_fraction = 0)
        model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state),
                                microphysics = nothing)
        @test_throws ArgumentError set!(model; θ = (x, y, z) -> 300.0, balance = true)
    end

end
