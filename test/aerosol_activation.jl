include(joinpath(@__DIR__, "setup.jl"))

using Test
import Breeze
using Breeze.AtmosphereModels: aerosol_field_names, prognostic_field_names,
                               settable_specific_microphysical_names
using Breeze.Microphysics.PredictedParticleProperties:
    AerosolMode,
    AerosolActivation,
    P3MicrophysicalState,
    activated_number,
    compute_cloud_droplet_activation,
    has_prognostic_aerosol,
    total_activated_number,
    sum_aerosol_number,
    aerosol_activation_rate

using Oceananigans: Flat, Bounded, RectilinearGrid, CenterField, time_step!
using Oceananigans.Fields: interior, set!
using Oceananigans.TimeSteppers: update_state!

@testset "Aerosol Activation" begin
    FT = Float64

    @testset "AerosolMode construction" begin
        thermodynamic_constants = Breeze.ThermodynamicConstants(FT)
        mode = AerosolMode(FT)
        # Default ammonium sulfate: βact = vi * osm * epsm * mw * rhoa / (map * rhow)
        expected_beta = 3 * 0.9 * thermodynamic_constants.vapor.molar_mass * 1777 /
                        (0.132 * thermodynamic_constants.liquid.density)
        @test mode.solute_activity ≈ expected_beta rtol=1e-10
        @test mode.number_mixing_ratio == 3e8
        @test mode.mean_radius == 5e-8

        aerosol = AerosolActivation(mode)
        @test aerosol.molecular_weight_water == thermodynamic_constants.vapor.molar_mass
        @test aerosol.universal_gas_constant == thermodynamic_constants.molar_gas_constant
        @test aerosol.liquid_water_density == thermodynamic_constants.liquid.density
        @test aerosol.surface_tension_reference_temperature ==
              thermodynamic_constants.energy_reference_temperature

        custom_liquid = Breeze.CondensedPhase(FT;
            reference_latent_heat = 2500800,
            heat_capacity = 4181,
            density = 950)
        custom_constants = Breeze.ThermodynamicConstants(FT;
            molar_gas_constant = 8,
            energy_reference_temperature = 275,
            vapor_molar_mass = 0.02,
            liquid = custom_liquid)
        custom_mode = AerosolMode(FT; thermodynamic_constants = custom_constants)
        custom_aerosol = AerosolActivation(custom_mode;
                                           thermodynamic_constants = custom_constants)
        @test custom_mode.solute_activity ≈ 3 * 0.9 * 0.02 * 1777 / (0.132 * 950)
        @test custom_aerosol.molecular_weight_water == 0.02
        @test custom_aerosol.universal_gas_constant == 8
        @test custom_aerosol.liquid_water_density == 950
        @test custom_aerosol.surface_tension_reference_temperature == 275
    end

    @testset "Single-mode activated number" begin
        mode = AerosolMode(FT)
        aerosol = AerosolActivation(mode)
        T = FT(280.0)
        S = FT(0.003)  # 0.3% supersaturation

        N_act = activated_number(mode, aerosol, T, S)
        # Must be positive and less than total aerosol
        @test N_act > 0
        @test N_act <= mode.number_mixing_ratio
        # At high supersaturation, nearly all aerosol activates
        N_high = activated_number(mode, aerosol, T, FT(0.05))
        @test N_high > 0.9 * mode.number_mixing_ratio
        # At zero supersaturation, none activates
        N_zero = activated_number(mode, aerosol, T, FT(0.0))
        @test N_zero ≈ 0 atol=1e-3
    end

    @testset "Multi-mode activation" begin
        mode1 = AerosolMode(FT; number_mixing_ratio=300e6, mean_radius=0.05e-6, geometric_std=2.0)
        mode2 = AerosolMode(FT; number_mixing_ratio=100e6, mean_radius=1.3e-6,  geometric_std=2.5)
        aerosol = AerosolActivation(mode1, mode2)

        T = FT(280.0)
        S = FT(0.003)

        N_total = total_activated_number(aerosol, T, S)
        @test N_total > 0
        @test N_total <= sum_aerosol_number(aerosol)
        @test sum_aerosol_number(aerosol) == 400e6
    end

    @testset "Aerosol activation rate" begin
        mode = AerosolMode(FT)
        aerosol = AerosolActivation(mode)

        nᶜˡ = FT(100e6)    # current cloud number [kg⁻¹]
        qᵛ = FT(0.015)      # vapor mixing ratio [kg/kg]
        qᵛ⁺ˡ = FT(0.0145)   # saturation mixing ratio (supersaturated)
        T = FT(280.0)

        result = aerosol_activation_rate(aerosol, nᶜˡ, qᵛ, qᵛ⁺ˡ, T)

        # Supersaturated: should produce positive rates
        @test result.ncnuc > 0
        @test result.qcnuc > 0
        # Mass = number * seed mass
        seed_mass = 4π/3 * 1000 * (1e-6)^3
        @test result.qcnuc ≈ result.ncnuc * seed_mass rtol=1e-10

        # Subsaturated: should produce zero rates
        qᵛ_sub = FT(0.014)
        result_sub = aerosol_activation_rate(aerosol, nᶜˡ, qᵛ_sub, qᵛ⁺ˡ, T)
        @test result_sub.ncnuc == 0
        @test result_sub.qcnuc == 0
    end

    @testset "Float32 support" begin
        mode = AerosolMode(Float32)
        aerosol = AerosolActivation(mode)
        T = Float32(280.0)
        S = Float32(0.003)
        N_act = activated_number(mode, aerosol, T, S)
        @test N_act isa Float32
        @test N_act > 0
    end
end

@testset "Aerosol activation integration with P3" begin
    using Breeze.Microphysics.PredictedParticleProperties:
        PredictedParticlePropertiesMicrophysics

    FT = Float64

    # Prognostic droplet number and aerosol reservoir
    p3 = PredictedParticlePropertiesMicrophysics(FT;
        aerosol = AerosolActivation(AerosolMode(FT); prognostic = true))

    @test !isnothing(p3.aerosol)
    @test length(p3.aerosol.modes) == 1

    # Prescribed droplet number (default)
    p3_prescribed = PredictedParticlePropertiesMicrophysics(FT)
    @test isnothing(p3_prescribed.aerosol)
    @test aerosol_field_names(p3) == (:ρnᵃ,)
    @test isempty(aerosol_field_names(p3_prescribed))
    @test isempty(aerosol_field_names(nothing))

    @testset "P3MicrophysicalState stores aerosol number" begin
        state = P3MicrophysicalState(ntuple(_ -> zero(FT), 12)...)
        @test state.nᵃ == 0
    end

    # The scheme reports its population per unit mass [kg⁻¹]; `ρnᵃ` is the ρ-weighted form.
    nᵃ₀ = FT(sum_aerosol_number(p3.aerosol))
    @test Breeze.initial_aerosol_number(p3) == nᵃ₀

    @testset "The reservoir is a ρ-weighted number density" begin
        # `ρnᵃ` holds ρ nᵃ, so the default is the number mixing ratio times the air density,
        # and the round trip the rate functions perform recovers [kg⁻¹].
        ρ = FT(0.8)
        ρnᵃ = Breeze.initial_aerosol_number_density(p3, ρ)
        @test ρnᵃ ≈ ρ * nᵃ₀
        @test ρnᵃ / ρ ≈ nᵃ₀
    end

    @testset "Anelastic construction seeds ρnᵃ from the reference density" begin
        grid = RectilinearGrid(default_arch, size=(1, 1, 4), x=(0, 1), y=(0, 1), z=(0, 2000))
        # `initialize_model_thermodynamics!` runs `set!(model, θ=θ₀)` from the constructor, and
        # the anelastic reference density is already physical there, so the reservoir is seeded.
        model = Breeze.AtmosphereModel(grid; microphysics = p3)

        ρ̄ = Breeze.AtmosphereModels.dynamics_density(model.dynamics)
        ρnᵃ = Array(interior(model.microphysical_fields.ρnᵃ))
        ρ̄_interior = Array(interior(ρ̄))
        @test ρnᵃ ≈ ρ̄_interior .* nᵃ₀
        # The reference density decreases with height, so the seeded reservoir must too.
        @test ρnᵃ[1, 1, 4] < ρnᵃ[1, 1, 1]
    end

    @testset "A prescribed density seeds ρnᵃ at construction" begin
        # The kinematic driver never runs `set!` from the constructor, so this covers the
        # constructor's own call rather than the one the anelastic path inherits.
        grid = RectilinearGrid(default_arch, size=(1, 1, 4), x=(0, 1), y=(0, 1), z=(0, 2000))
        reference_state = Breeze.ReferenceState(grid, Breeze.ThermodynamicConstants())
        model = Breeze.AtmosphereModel(grid;
                                       dynamics = Breeze.PrescribedDynamics(reference_state),
                                       microphysics = p3)

        ρ = Array(interior(Breeze.AtmosphereModels.total_density(model.dynamics)))
        ρnᵃ = Array(interior(model.microphysical_fields.ρnᵃ))
        @test ρnᵃ ./ ρ ≈ fill(nᵃ₀, size(ρ)) rtol=1e-12
    end

    @testset "Compressible ρnᵃ stays zero until set! supplies a density" begin
        grid = RectilinearGrid(default_arch, size=(1, 1, 4), x=(0, 1), y=(0, 1), z=(0, 2000))
        # Compressible density fields are zero at construction, so there is nothing to weight
        # a ρ-weighted reservoir by: the constructor writes zero and `set!` fills it in.
        model = Breeze.AtmosphereModel(grid;
                                       dynamics = Breeze.CompressibleDynamics(Breeze.ExplicitTimeStepping()),
                                       microphysics = p3)

        @test all(iszero, Array(interior(model.microphysical_fields.ρnᵃ)))
    end

    @testset "The default sums the number mixing ratio over all modes" begin
        multimode = AerosolActivation(
            AerosolMode(FT; number_mixing_ratio = 300e6),
            AerosolMode(FT; number_mixing_ratio = 100e6, mean_radius = 1.0e-6, geometric_std = 2.5),
            AerosolMode(FT; number_mixing_ratio = 25e6,  mean_radius = 2.0e-6);
            prognostic = true)
        p3_multimode = PredictedParticlePropertiesMicrophysics(FT; aerosol = multimode)

        nᵃ_summed = FT(425e6)
        @test sum_aerosol_number(multimode) == nᵃ_summed

        grid = RectilinearGrid(default_arch, size=(1, 1, 4), x=(0, 1), y=(0, 1), z=(0, 2000))
        model = Breeze.AtmosphereModel(grid; microphysics = p3_multimode)

        ρ̄ = Array(interior(Breeze.AtmosphereModels.dynamics_density(model.dynamics)))
        ρnᵃ = Array(interior(model.microphysical_fields.ρnᵃ))
        @test ρnᵃ ./ ρ̄ ≈ fill(nᵃ_summed, size(ρ̄)) rtol=1e-12
    end

    @testset "Compressible set! seeds ρnᵃ from the density it establishes" begin
        grid = RectilinearGrid(default_arch, size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
        model = Breeze.AtmosphereModel(grid;
                                       dynamics = Breeze.CompressibleDynamics(Breeze.ExplicitTimeStepping()),
                                       microphysics = p3)

        # Compressible dynamics builds its density field at zero, so `set!(ρ)` is the first
        # point at which a ρ-weighted reservoir can be written.
        set!(model; ρ = FT(0.8), θ = FT(300), qᵛ = FT(0), enforce_mass_conservation = false)

        @test only(Array(interior(model.microphysical_fields.ρnᵃ))) ≈ FT(0.8) * nᵃ₀
    end

    @testset "A hydrostatically balanced density seeds the balanced reservoir" begin
        grid = RectilinearGrid(default_arch, size=(1, 1, 4), x=(0, 1), y=(0, 1), z=(0, 2000))
        model = Breeze.AtmosphereModel(grid;
                                       dynamics = Breeze.CompressibleDynamics(Breeze.ExplicitTimeStepping()),
                                       microphysics = p3)

        # The balanced density is not known until after the column solve, so this checks that
        # the solve's rescaling of density-weighted fields carries the seeded reservoir.
        set!(model; ρ = Breeze.HydrostaticallyBalancedDensity(), θ = FT(300), qᵛ = FT(0),
             enforce_mass_conservation = false)

        ρ = Array(interior(model.dynamics.total_density))
        ρnᵃ = Array(interior(model.microphysical_fields.ρnᵃ))
        @test ρnᵃ ./ ρ ≈ fill(nᵃ₀, size(ρ)) rtol=1e-12
    end

    @testset "Dry-density initialization uses reconciled total density" begin
        grid = RectilinearGrid(default_arch, size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
        model = Breeze.AtmosphereModel(grid;
                                       dynamics = Breeze.CompressibleDynamics(Breeze.ExplicitTimeStepping()),
                                       microphysics = p3)

        set!(model; ρᵈ = FT(0.8), θ = FT(300), qᵛ = FT(0.2),
             enforce_mass_conservation = false)

        ρ = only(Array(interior(model.dynamics.total_density)))
        ρnᵃ = only(Array(interior(model.microphysical_fields.ρnᵃ)))
        @test ρ ≈ FT(1)
        @test ρnᵃ ≈ ρ * nᵃ₀
    end

    @testset "A later set! resets the reservoir to the distribution default" begin
        grid = RectilinearGrid(default_arch, size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
        model = Breeze.AtmosphereModel(grid;
                                       dynamics = Breeze.CompressibleDynamics(Breeze.ExplicitTimeStepping()),
                                       microphysics = p3)

        set!(model; ρ = FT(0.8), θ = FT(300), qᵛ = FT(0),
             enforce_mass_conservation = false)
        set!(model.microphysical_fields.ρnᵃ, FT(0.25) * nᵃ₀)

        # `set!` re-initializes the state, aerosol included, so a depletion written between
        # calls is not preserved unless it is passed back in (see the next testset).
        set!(model; ρ = FT(0.7), θ = FT(300), qᵛ = FT(0),
             enforce_mass_conservation = false)

        @test only(Array(interior(model.microphysical_fields.ρnᵃ))) ≈ FT(0.7) * nᵃ₀
    end

    @testset "Explicit nᵃ and ρnᵃ own the reservoir" begin
        grid = RectilinearGrid(default_arch, size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
        model = Breeze.AtmosphereModel(grid;
                                       dynamics = Breeze.CompressibleDynamics(Breeze.ExplicitTimeStepping()),
                                       microphysics = p3)

        set!(model; ρ = FT(0.8), θ = FT(300), qᵛ = FT(0), ρnᵃ = FT(5e7),
             enforce_mass_conservation = false)
        @test only(Array(interior(model.microphysical_fields.ρnᵃ))) ≈ FT(5e7)

        # `nᵃ` is per unit mass, so it is weighted by the density established in the same call.
        set!(model; ρ = FT(0.8), θ = FT(300), qᵛ = FT(0), nᵃ = FT(0.25) * nᵃ₀,
             enforce_mass_conservation = false)
        @test only(Array(interior(model.microphysical_fields.ρnᵃ))) ≈ FT(0.8) * FT(0.25) * nᵃ₀
    end

    @testset "Adiabatic balancing preserves aerosol number per unit mass" begin
        grid = RectilinearGrid(default_arch, size=(2, 2, 4),
                               x=(0, 1), y=(0, 1), z=(0, 1000))
        model = Breeze.AtmosphereModel(grid;
                                       dynamics = Breeze.CompressibleDynamics(Breeze.ExplicitTimeStepping()),
                                       microphysics = p3)

        set!(model; ρ = FT(0.8), θ = FT(300), qᵛ = FT(0),
             enforce_mass_conservation = false,
             balancer = Breeze.AdiabaticBalancer(Δt=FT(1), cycles=2))

        ρ = Array(interior(model.dynamics.total_density))
        ρnᵃ = Array(interior(model.microphysical_fields.ρnᵃ))
        @test ρnᵃ ./ ρ ≈ fill(nᵃ₀, size(ρ)) rtol=1e-12
    end

    @testset "Parcel aerosol follows the parcel density" begin
        # Parcel models interpolate the environmental profiles on the host, so like every
        # other parcel test they run on the CPU regardless of `default_arch`.
        grid = RectilinearGrid(size=4, z=(0, 1), topology=(Flat, Flat, Bounded))
        model = Breeze.AtmosphereModel(grid;
                                       dynamics = Breeze.ParcelDynamics(),
                                       microphysics = p3)

        set!(model; T = FT(288), ρ = FT(0.8), p = FT(1e5), z = FT(0.1))
        @test model.dynamics.state.μ.ρnᵃ ≈ FT(0.8) * nᵃ₀

        set!(model; T = FT(288), ρ = FT(0.4), p = FT(1e5), z = FT(0.2))
        @test model.dynamics.state.μ.ρnᵃ ≈ FT(0.4) * nᵃ₀

        set!(model; T = FT(288), ρ = FT(0.4), p = FT(1e5), z = FT(0.2), nᵃ = FT(5e7))
        @test model.dynamics.state.μ.ρnᵃ ≈ FT(0.4) * FT(5e7)
    end
end

# Droplet number remains prognostic with a fixed aerosol population.
@testset "Fixed aerosol reservoir" begin
    using Breeze.Microphysics.PredictedParticleProperties:
        PredictedParticlePropertiesMicrophysics

    FT = Float64
    prognostic = AerosolActivation(AerosolMode(FT); prognostic = true)
    fixed = AerosolActivation(AerosolMode(FT))

    @testset "The switch is carried in the type, not a field" begin
        # A fixed population is the default: the reservoir is opt-in.
        @test !has_prognostic_aerosol(AerosolActivation(AerosolMode(FT)))
        @test has_prognostic_aerosol(prognostic)
        @test !has_prognostic_aerosol(fixed)
        # A runtime branch here would leak a Union into `prognostic_field_names` and force
        # the GPU prognostic-extraction recursion to allocate.
        p3 = PredictedParticlePropertiesMicrophysics(FT; aerosol = fixed)
        @test @inferred(aerosol_field_names(p3)) == ()
        @test @inferred(prognostic_field_names(p3)) isa Tuple{Vararg{Symbol}}
        @test summary(fixed) == "AerosolActivation(1 mode, fixed reservoir)"
        @test summary(prognostic) == "AerosolActivation(1 mode, prognostic reservoir)"
        # Both settings share every activation parameter; only the reservoir differs.
        @test fixed.modes == prognostic.modes
        @test fixed.activation_timescale == prognostic.activation_timescale
    end

    @testset "Droplet number stays prognostic while the reservoir does not" begin
        p3_fixed = PredictedParticlePropertiesMicrophysics(FT; aerosol = fixed)
        p3_prognostic = PredictedParticlePropertiesMicrophysics(FT; aerosol = prognostic)

        @test :ρnᶜˡ ∈ prognostic_field_names(p3_fixed)
        @test :ρnᵃ ∉ prognostic_field_names(p3_fixed)
        @test :ρnᶜˡ ∈ prognostic_field_names(p3_prognostic)
        @test :ρnᵃ ∈ prognostic_field_names(p3_prognostic)

        @test aerosol_field_names(p3_fixed) == ()
        @test aerosol_field_names(p3_prognostic) == (:ρnᵃ,)

        # `nᵃ` is only settable where it is state.
        @test :nᶜˡ ∈ settable_specific_microphysical_names(p3_fixed)
        @test :nᵃ ∉ settable_specific_microphysical_names(p3_fixed)
        @test :nᵃ ∈ settable_specific_microphysical_names(p3_prognostic)
    end

    @testset "Activation draws on the whole distribution" begin
        # With no reservoir the `min(N_act, nᶜˡ + nᵃ)` cap cannot bind, leaving a plain
        # relaxation of `nᶜˡ` toward the equilibrium activated count.
        p3_fixed = PredictedParticlePropertiesMicrophysics(FT; aerosol = fixed)
        p3_prognostic = PredictedParticlePropertiesMicrophysics(FT; aerosol = prognostic)
        constants = Breeze.ThermodynamicConstants(FT)

        nᶜˡ, qᶜˡ = FT(1e6), FT(1e-5)
        qᵛ, qᵛ⁺ˡ, T, ρ = FT(0.015), FT(0.0145), FT(280), FT(1)

        whole_distribution = aerosol_activation_rate(fixed, nᶜˡ, qᵛ, qᵛ⁺ˡ, T)
        @test whole_distribution.ncnuc > 0

        # `ℳ.nᵃ` is zero on the fixed path, so the dispatch must ignore it and activate
        # against the whole distribution anyway.
        fixed_rate = compute_cloud_droplet_activation(fixed, p3_fixed, qᶜˡ, nᶜˡ, zero(FT),
                                                      qᵛ, qᵛ⁺ˡ, T, ρ, constants)
        @test fixed_rate.number == whole_distribution.ncnuc

        # The prognostic path reads that same argument as the remaining reservoir, so an
        # exhausted one shuts activation off.
        drained = compute_cloud_droplet_activation(prognostic, p3_prognostic, qᶜˡ, nᶜˡ, zero(FT),
                                                   qᵛ, qᵛ⁺ˡ, T, ρ, constants)
        @test drained.number == 0
    end

    @testset "Nothing is allocated or advected for the reservoir" begin
        grid = RectilinearGrid(default_arch, FT; size = (2, 2, 2), extent = (100, 100, 100))
        constants = Breeze.ThermodynamicConstants(FT)
        reference_state = Breeze.ReferenceState(grid, constants;
                                                base_pressure = FT(101325),
                                                potential_temperature = FT(285))

        build(aerosol) = Breeze.AtmosphereModel(grid;
            dynamics = Breeze.AnelasticDynamics(reference_state),
            thermodynamic_constants = constants,
            microphysics = PredictedParticlePropertiesMicrophysics(FT; aerosol))

        fixed_model = build(fixed)
        prognostic_model = build(prognostic)

        for name in (:ρnᵃ, :nᵃ)
            @test !haskey(fixed_model.microphysical_fields, name)
            @test haskey(prognostic_model.microphysical_fields, name)
        end
        @test !hasproperty(fixed_model.timestepper.Gⁿ, :ρnᵃ)
        @test hasproperty(fixed_model.timestepper.Gⁿ, :ρnᶜˡ)

        # There is no reservoir to own, so `set!` rejects the key rather than dropping it.
        @test_throws ArgumentError set!(fixed_model; θ = FT(285), qᵛ = FT(0.011),
                                        nᵃ = FT(1e8), enforce_mass_conservation = false)

        # A prognostic reservoir stops activating once drained; a fixed one keeps relaxing
        # toward the equilibrium count.
        small_pool = FT(2e7)
        nᶜˡ₀ = FT(1e6)
        set!(fixed_model; θ = FT(285), qᵛ = FT(0.011), qᶜˡ = FT(1e-5), nᶜˡ = nᶜˡ₀,
             enforce_mass_conservation = false)
        set!(prognostic_model; θ = FT(285), qᵛ = FT(0.011), qᶜˡ = FT(1e-5), nᶜˡ = nᶜˡ₀,
             nᵃ = small_pool, enforce_mass_conservation = false)
        for model in (fixed_model, prognostic_model)
            update_state!(model)
            for _ in 1:20
                time_step!(model, FT(1))
            end
        end

        fixed_nᶜˡ = Array(interior(fixed_model.microphysical_fields.nᶜˡ))
        prognostic_nᶜˡ = Array(interior(prognostic_model.microphysical_fields.nᶜˡ))
        prognostic_nᵃ = Array(interior(prognostic_model.microphysical_fields.nᵃ))

        # Both activate, but the prognostic path cannot pass `nᶜˡ₀ + nᵃ₀`. The tolerance
        # covers drift of the *specific* number as condensation and sedimentation move the
        # density it is divided by; the cap itself is enforced on ρ-weighted counts.
        reservoir_ceiling = nᶜˡ₀ + small_pool
        @test all(prognostic_nᶜˡ .> nᶜˡ₀)
        @test all(prognostic_nᶜˡ .<= FT(1.01) * reservoir_ceiling)
        @test all(prognostic_nᵃ .< FT(1e-3) * small_pool)

        @test all(fixed_nᶜˡ .> reservoir_ceiling)
        @test all(fixed_nᶜˡ .<= FT(1.01) * sum_aerosol_number(fixed))
    end

    @testset "Parcels carry no reservoir either" begin
        # Parcel models interpolate on the host, so they run on the CPU regardless of
        # `default_arch`.
        grid = RectilinearGrid(size = 4, z = (0, 1), topology = (Flat, Flat, Bounded))
        model = Breeze.AtmosphereModel(grid;
            dynamics = Breeze.ParcelDynamics(),
            microphysics = PredictedParticlePropertiesMicrophysics(FT; aerosol = fixed))

        set!(model; T = FT(288), ρ = FT(0.8), p = FT(1e5), z = FT(0.1))
        @test !haskey(model.dynamics.state.μ, :ρnᵃ)
        @test haskey(model.dynamics.state.μ, :ρnᶜˡ)
        @test_throws ArgumentError set!(model; T = FT(288), ρ = FT(0.8), p = FT(1e5),
                                        z = FT(0.1), nᵃ = FT(5e7))
    end
end
