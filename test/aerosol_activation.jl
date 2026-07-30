using Test
import Breeze
using Breeze.Microphysics.PredictedParticleProperties:
    AerosolMode,
    AerosolActivation,
    P3MicrophysicalState,
    activated_number,
    total_activated_number,
    sum_aerosol_number,
    prognostic_ccn_activation_rate

using Oceananigans: Flat, Bounded, RectilinearGrid, CenterField
using Oceananigans.Fields: interior, set!

@testset "Aerosol Activation" begin
    FT = Float64

    @testset "AerosolMode construction" begin
        mode = AerosolMode(FT)
        # Default ammonium sulfate (Fortran P3): βact = vi * osm * epsm * mw * rhoa / (map * rhow)
        expected_beta = 3.0 * 1.0 * 0.9 * 0.018 * 1777.0 / (0.132 * 1000.0)
        @test mode.solute_activity ≈ expected_beta rtol=1e-10
        @test mode.number_mixing_ratio == 300e6
        @test mode.mean_radius == 0.05e-6
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

    @testset "Prognostic CCN activation rate" begin
        mode = AerosolMode(FT)
        aerosol = AerosolActivation(mode)

        nᶜˡ = FT(100e6)    # current cloud number [kg⁻¹]
        qᵛ = FT(0.015)      # vapor mixing ratio [kg/kg]
        qᵛ⁺ˡ = FT(0.0145)   # saturation mixing ratio (supersaturated)
        T = FT(280.0)

        result = prognostic_ccn_activation_rate(aerosol, nᶜˡ, qᵛ, qᵛ⁺ˡ, T)

        # Supersaturated: should produce positive rates
        @test result.ncnuc > 0
        @test result.qcnuc > 0
        # Mass = number * seed mass
        seed_mass = 4π/3 * 1000 * (1e-6)^3
        @test result.qcnuc ≈ result.ncnuc * seed_mass rtol=1e-10

        # Subsaturated: should produce zero rates
        qᵛ_sub = FT(0.014)
        result_sub = prognostic_ccn_activation_rate(aerosol, nᶜˡ, qᵛ_sub, qᵛ⁺ˡ, T)
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

@testset "Prognostic CCN integration with P3" begin
    using Breeze.Microphysics.PredictedParticleProperties:
        PredictedParticlePropertiesMicrophysics

    FT = Float64

    # Construct P3 with prognostic CCN
    p3 = PredictedParticlePropertiesMicrophysics(FT;
        aerosol = AerosolActivation(AerosolMode(FT)))

    @test !isnothing(p3.aerosol)
    @test length(p3.aerosol.modes) == 1

    # Construct P3 with prescribed CCN (default)
    p3_prescribed = PredictedParticlePropertiesMicrophysics(FT)
    @test isnothing(p3_prescribed.aerosol)

    @testset "P3MicrophysicalState defaults missing aerosol to zero" begin
        state = P3MicrophysicalState(ntuple(_ -> zero(FT), 11)...)
        @test state.nᵃ == 0
    end

    # P3's aerosol distribution is per unit mass [kg⁻¹], and the prognostic `ρnᵃ` holds
    # ρ nᵃ [m⁻³], so the reservoir must be seeded ρ-weighted. Without the weighting the
    # `min(N_activated, nᶜˡ + nᵃ)` cap in `prognostic_ccn_activation_rate` acquires a
    # spurious inverse-density dependence, because that comparison is entirely per unit mass.
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
            AerosolMode(FT; number_mixing_ratio = 25e6,  mean_radius = 2.0e-6))
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
