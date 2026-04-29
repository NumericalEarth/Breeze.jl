using Breeze
using Oceananigans
using Oceananigans.Utils: IterationInterval
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics
using Breeze.Thermodynamics: TetensFormula, dry_air_gas_constant
using GPUArraysCore: @allowscalar
using Test

@testset "Scheduled microphysics: construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), extent=(1000, 1000, 1000))

    @testset "default (no schedule)" begin
        model = AtmosphereModel(grid)
        @test model.microphysics_schedule === nothing
        @test model.microphysics_tendencies === nothing
        @test model.microphysics_state === nothing
    end

    @testset "with IterationInterval(5)" begin
        model = AtmosphereModel(grid; microphysics_schedule = IterationInterval(5))
        @test model.microphysics_schedule isa IterationInterval
        @test model.microphysics_tendencies isa NamedTuple
        @test :ρθ in keys(model.microphysics_tendencies)
        @test :ρqᵛ in keys(model.microphysics_tendencies)
        @test model.microphysics_state isa Breeze.AtmosphereModels.MicrophysicsScheduleState{FT}
        # Construction calls set!(model, θ=θ₀) → update_state! → update_microphysics!, which
        # fires at iteration 0 (IterationInterval(5) returns true at 0). So last_fire_iteration
        # is 0 after construction, not -1.
        @test model.microphysics_state.last_fire_iteration == 0
    end
end

@testset "microphysics_model_update! Δt_eff plumbing" begin
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), extent=(100, 100, 100))
    model = AtmosphereModel(grid)  # microphysics = nothing → no-op
    # 3-arg form must work and be a no-op
    @test Breeze.AtmosphereModels.microphysics_model_update!(model.microphysics, model, 1.0) === nothing
    # 2-arg shim must forward to the 3-arg method
    @test Breeze.AtmosphereModels.microphysics_model_update!(model.microphysics, model) === nothing
end

@testset "DCMIP2016KM consumes Δt_eff" begin
    FT = Float64
    Nz = 8

    # DCMIP2016 Kessler uses TetensFormula with liquid_temperature_offset=36
    tetens = TetensFormula(liquid_temperature_offset=36)
    constants = ThermodynamicConstants(FT; saturation_vapor_pressure=tetens)

    grid = RectilinearGrid(CPU(); size=(1, 1, Nz), x=(0, 100), y=(0, 100),
                           z=(0, 4000), topology=(Periodic, Periodic, Bounded))

    microphysics = DCMIP2016KesslerMicrophysics(FT)

    p₀ = FT(100000)
    ref_state = ReferenceState(grid, constants; surface_pressure=p₀)
    dynamics = AnelasticDynamics(ref_state)
    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants=constants)

    # Constant reference profile
    ρ_prof = fill(FT(1.0), Nz)
    p_prof = fill(FT(90000), Nz)
    set!(model.dynamics.reference_state.density, reshape(ρ_prof, 1, 1, Nz))
    set!(model.dynamics.reference_state.pressure, reshape(p_prof, 1, 1, Nz))

    # ρqᵛ = 0.02 is initialized well above saturation at the parcel's actual
    # temperature, and ρqᶜˡ = 0.01 >> autoconversion threshold 0.001, with no rain.
    # In this regime saturation adjustment drives condensation (Δrˢᵃᵗ > 0) so
    # evaporation is suppressed, and the net cloud→rain rate is dominated by
    # autoconversion which is linear in Δt. Rain production therefore scales as 2×
    # when Δt doubles.
    ρqᶜˡ_init = fill(FT(0.01), Nz)
    ρqᵛ_init  = fill(FT(0.02), Nz)
    set!(model.microphysical_fields.ρqᶜˡ, reshape(ρqᶜˡ_init, 1, 1, Nz))
    set!(model.microphysical_fields.ρqʳ,  reshape(zeros(FT, Nz), 1, 1, Nz))
    set!(model.moisture_density, reshape(ρqᵛ_init, 1, 1, Nz))

    # Set θˡⁱ consistent with T = 288 K at the reference pressure level.
    # Π = (p/p₀)^(Rᵈ/cᵖᵈ), θ = T/Π. Use project constants so this tracks
    # if they change.
    T_init = FT(288)
    Rᵈ = FT(dry_air_gas_constant(constants))
    cᵖᵈ = FT(constants.dry_air.heat_capacity)
    Π = (p_prof[1] / p₀)^(Rᵈ / cᵖᵈ)
    θ_init = T_init / Π
    ρθ_init = fill(FT(1.0) * θ_init, Nz)
    set!(model.formulation.potential_temperature_density, reshape(ρθ_init, 1, 1, Nz))
    model.clock.last_Δt = FT(1.0)

    snapshot(field) = copy(Array(interior(field, 1, 1, :)))
    saved_ρqᶜˡ = snapshot(model.microphysical_fields.ρqᶜˡ)
    saved_ρqʳ  = snapshot(model.microphysical_fields.ρqʳ)

    # Integrate for Δt_eff = 1 s
    Breeze.AtmosphereModels.microphysics_model_update!(model.microphysics, model, FT(1.0))
    state_a_ρqʳ = snapshot(model.microphysical_fields.ρqʳ)

    # Restore and integrate for Δt_eff = 2 s
    set!(model.microphysical_fields.ρqᶜˡ, reshape(saved_ρqᶜˡ, 1, 1, Nz))
    set!(model.microphysical_fields.ρqʳ,  reshape(saved_ρqʳ, 1, 1, Nz))
    Breeze.AtmosphereModels.microphysics_model_update!(model.microphysics, model, FT(2.0))
    state_b_ρqʳ = snapshot(model.microphysical_fields.ρqʳ)

    # Rain produced over Δt=1 and Δt=2 — expect 2× scaling to first order
    Δa = state_a_ρqʳ .- saved_ρqʳ      # rain produced in Δt=1 s
    Δb = state_b_ρqʳ .- saved_ρqʳ      # rain produced in Δt=2 s
    nonzero = findall(>(1e-9), abs.(Δa))
    @test !isempty(nonzero)
    @test all(@. abs(Δb[nonzero] / Δa[nonzero] - 2) < 0.5)
end

@testset "grid_microphysical_tendency cache overload" begin
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), extent=(100, 100, 100))
    cache = (ρqᶜˡ = CenterField(grid), ρqʳ = CenterField(grid))
    @allowscalar fill!(parent(cache.ρqᶜˡ), 0.25)
    @allowscalar fill!(parent(cache.ρqʳ), -0.5)

    # Stand-in args (microphysics = nothing, ρ = 1, fields, 𝒰, constants, velocities)
    constants = ThermodynamicConstants()
    fields = (;)
    𝒰 = nothing
    velocities = nothing

    FT = eltype(grid)
    val_qcl  = @allowscalar Breeze.AtmosphereModels.grid_microphysical_tendency(2, 2, 2, grid, nothing, Val(:ρqᶜˡ), cache, one(FT), fields, 𝒰, constants, velocities)
    val_qr   = @allowscalar Breeze.AtmosphereModels.grid_microphysical_tendency(2, 2, 2, grid, nothing, Val(:ρqʳ),  cache, one(FT), fields, 𝒰, constants, velocities)
    val_miss = @allowscalar Breeze.AtmosphereModels.grid_microphysical_tendency(2, 2, 2, grid, nothing, Val(:ρθ),   cache, one(FT), fields, 𝒰, constants, velocities)

    @test val_qcl  ≈ 0.25
    @test val_qr   ≈ -0.5
    @test val_miss == 0
end

@testset "compute_microphysics_tendencies! fills the cache" begin
    using Breeze.Microphysics: SaturationAdjustment

    grid = RectilinearGrid(default_arch; size=(4, 4, 8), extent=(100, 100, 1000))
    μ = SaturationAdjustment()
    model = AtmosphereModel(grid; microphysics = μ, microphysics_schedule = IterationInterval(1))

    # Initialize state so 𝒰 / ℳ build cleanly
    set!(model, ρθ = 300, ρqᵛ = 1e-3)
    Breeze.AtmosphereModels.update_state!(model; compute_tendencies = false)

    # Sanity: cache exists and has the expected names
    @test model.microphysics_tendencies isa NamedTuple
    @test :ρθ in keys(model.microphysics_tendencies) || :ρe in keys(model.microphysics_tendencies)

    # Manually invoke the cache-filling kernel and ensure no errors
    Breeze.AtmosphereModels.compute_microphysics_tendencies!(model.microphysics_tendencies, model.microphysics, model, 1.0)

    # Cache fields should be finite
    for (_, f) in pairs(model.microphysics_tendencies)
        @test all(isfinite, parent(f))
    end
end

@testset "update_microphysics! honors the schedule" begin
    using Breeze.Microphysics: SaturationAdjustment

    grid = RectilinearGrid(default_arch; size=(4, 4, 4), extent=(100, 100, 100))
    μ = SaturationAdjustment()
    model = AtmosphereModel(grid; microphysics = μ, microphysics_schedule = IterationInterval(3))
    set!(model, ρθ = 300, ρqᵛ = 1e-3)

    # First iteration always fires
    @allowscalar model.clock.iteration = 0
    @allowscalar model.clock.time = 0
    @allowscalar model.clock.last_Δt = 1.0
    Breeze.AtmosphereModels.update_microphysics!(model)
    @test model.microphysics_state.last_fire_iteration == 0

    # Iterations 1, 2 should NOT fire
    @allowscalar model.clock.iteration = 1
    @allowscalar model.clock.time = 1
    Breeze.AtmosphereModels.update_microphysics!(model)
    @test model.microphysics_state.last_fire_iteration == 0

    @allowscalar model.clock.iteration = 2
    @allowscalar model.clock.time = 2
    Breeze.AtmosphereModels.update_microphysics!(model)
    @test model.microphysics_state.last_fire_iteration == 0

    # Iteration 3 should fire (IterationInterval(3))
    @allowscalar model.clock.iteration = 3
    @allowscalar model.clock.time = 3
    Breeze.AtmosphereModels.update_microphysics!(model)
    @test model.microphysics_state.last_fire_iteration == 3
end

@testset "Cache path matches inline path at IterationInterval(1)" begin
    using Breeze.Microphysics: SaturationAdjustment
    using Breeze.AtmosphereModels: moisture_prognostic_name

    grid = RectilinearGrid(default_arch; size=(4, 4, 8), extent=(1000, 1000, 1000))
    μ = SaturationAdjustment()

    function build_and_step(; schedule)
        model = AtmosphereModel(grid; microphysics = μ, microphysics_schedule = schedule)
        set!(model, ρθ = 300, ρqᵉ = 1e-3)
        sim = Simulation(model, Δt = 1.0, stop_iteration = 1, verbose = false)
        run!(sim)
        moist_name = moisture_prognostic_name(model.microphysics)
        return (ρθ   = copy(parent(model.timestepper.Gⁿ.ρθ)),
                ρqᵉ  = copy(parent(model.timestepper.Gⁿ[moist_name])))
    end

    inline_G = build_and_step(schedule = nothing)
    cached_G = build_and_step(schedule = IterationInterval(1))

    @test inline_G.ρθ  ≈ cached_G.ρθ  atol = 1e-12 rtol = 1e-12
    @test inline_G.ρqᵉ ≈ cached_G.ρqᵉ atol = 1e-12 rtol = 1e-12
end
