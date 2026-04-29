using Breeze
using Oceananigans
using Oceananigans.Utils: IterationInterval
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics
using Breeze.Thermodynamics: TetensFormula
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
        @test model.microphysics_state.last_fire_iteration == -1
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

    # Supersaturated vapor (ρqᵛ = 0.02 >> saturation ≈ 0.009 at T=288K, p=90kPa),
    # large cloud water (ρqᶜˡ = 0.01 >> autoconversion threshold 0.001), no rain.
    # In this regime saturation adjustment drives condensation (Δrˢᵃᵗ > 0) so
    # evaporation is suppressed, and the net cloud→rain rate is dominated by
    # autoconversion which is linear in Δt. Rain production therefore scales as 2×
    # when Δt doubles.
    ρqᶜˡ_init = fill(FT(0.01), Nz)
    ρqᵛ_init  = fill(FT(0.02), Nz)
    set!(model.microphysical_fields.ρqᶜˡ, reshape(ρqᶜˡ_init, 1, 1, Nz))
    set!(model.microphysical_fields.ρqʳ,  reshape(zeros(FT, Nz), 1, 1, Nz))
    set!(model.moisture_density, reshape(ρqᵛ_init, 1, 1, Nz))

    # Set θˡⁱ consistent with T ≈ 288 K
    T_init = FT(288)
    Π = (p_prof[1] / p₀)^(FT(287) / FT(1003))
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
