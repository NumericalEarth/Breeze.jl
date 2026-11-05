include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using CloudMicrophysics
using Oceananigans
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

using Breeze.Thermodynamics: adjustment_saturation_specific_humidity, saturation_specific_humidity
using Breeze.Microphysics: equilibrated_moisture_mass_fractions

# `set!(model; T, qᵗ)` under saturation adjustment must convert T with the moisture partitioned at
# equilibrium at T. Otherwise the adjustment in `update_state!` condenses the difference, and the
# latent heat moves T away from the value that was set (by 18–30 K in a saturated column).

host(field) = Array(interior(field))
ice_or_zero(model) = haskey(model.microphysical_fields, :qⁱ) ? host(model.microphysical_fields.qⁱ) : 0

# A saturated and an unsaturated column of T = 290 K − 6 K/km over 5 km; the 260 K top holds ice
# under mixed-phase equilibrium. Running them in the order below on one model also checks that the
# conversion does not depend on the partition left behind by the previous state.
const columns = (saturated = 0.020 / 1.020, unsaturated = 0.001, saturated_again = 0.020 / 1.020)
T_profile(z) = 290 - 6e-3 * z

# Equilibrium condensate at T under each dynamics' saturation constraint:
# the reference pressure (anelastic) or the density (compressible).
function expected_condensate(model, T, qᵗ, equilibrium)
    constants = model.thermodynamic_constants
    qᵛ⁺ = if model.dynamics isa CompressibleDynamics
        saturation_specific_humidity.(T, host(model.dynamics.total_density), Ref(constants), Ref(equilibrium))
    else
        pᵣ = host(model.dynamics.reference_state.pressure)
        adjustment_saturation_specific_humidity.(T, pᵣ, qᵗ, Ref(constants), Ref(equilibrium))
    end
    q = equilibrated_moisture_mass_fractions.(T, qᵗ, qᵛ⁺, Ref(equilibrium))
    return map(x -> x.liquid, q), map(x -> x.ice, q)
end

function check_recovered_state(model, column, qᵗ, equilibrium, FT)
    T = FT.(T_profile.(reshape(znodes(model.grid, Center()), 1, 1, :))) .+ zero(host(model.temperature))
    qˡ, qⁱ = expected_condensate(model, T, qᵗ, equilibrium)
    q_atol = FT == Float64 ? 1e-7 : 1e-5

    @test maximum(abs, host(model.temperature) .- T) < (FT == Float64 ? 1e-3 : 2e-2)
    @test maximum(abs, host(model.microphysical_fields.qˡ) .- qˡ) < q_atol
    @test maximum(abs, ice_or_zero(model) .- qⁱ) < q_atol
    column == :unsaturated ? (@test maximum(qˡ .+ qⁱ) == 0) : (@test maximum(qˡ .+ qⁱ) > 1e-3)
end

anelastic_dynamics(grid, constants) =
    AnelasticDynamics(ReferenceState(grid, constants; base_pressure = 1e5, potential_temperature = 290))

@testset "set!(model; T, qᵗ) under saturation adjustment [$FT, $dynamics_name, $formulation]" for FT in test_float_types(),
        (dynamics_name, formulation) in ((:anelastic, :LiquidIcePotentialTemperature),
                                         (:anelastic, :StaticEnergy),
                                         (:compressible, :LiquidIcePotentialTemperature))

    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (2, 2, 20), x = (0, 1e3), y = (0, 1e3), z = (0, 5e3))
    constants = ThermodynamicConstants(FT)
    density = dynamics_name == :compressible ? (; ρ = 1) : (;)

    for equilibrium in (WarmPhaseEquilibrium(), MixedPhaseEquilibrium())
        dynamics = dynamics_name == :compressible ? CompressibleDynamics(; reference_potential_temperature = 290) :
                                                    anelastic_dynamics(grid, constants)
        microphysics = SaturationAdjustment(FT; equilibrium)
        model = AtmosphereModel(grid; dynamics, formulation, microphysics, thermodynamic_constants = constants)

        for (column, qᵗ) in pairs(columns)
            @testset "$(summary(equilibrium)), $column" begin
                set!(model; T = (x, y, z) -> T_profile(z), qᵗ = FT(qᵗ), density...)
                check_recovered_state(model, column, FT(qᵗ), equilibrium, FT)

                # update_state! is idempotent on the recovered state
                Oceananigans.TimeSteppers.update_state!(model)
                check_recovered_state(model, column, FT(qᵗ), equilibrium, FT)
            end
        end
    end
end

# One-moment microphysics with saturation adjustment holds the rain fixed during the adjustment
@testset "set!(model; T, qᵗ, qʳ) under one-moment saturation adjustment [$(summary(equilibrium))]" for equilibrium in (WarmPhaseEquilibrium(), MixedPhaseEquilibrium())
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(default_arch; size = (2, 2, 20), x = (0, 1e3), y = (0, 1e3), z = (0, 5e3))
    constants = ThermodynamicConstants()
    dynamics = anelastic_dynamics(grid, constants)
    microphysics = OneMomentCloudMicrophysics(cloud_formation = SaturationAdjustment(; equilibrium))
    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants = constants)

    qᵗ, qʳ = columns.saturated, 1e-3
    set!(model; T = (x, y, z) -> T_profile(z), qᵗ, qʳ)

    T = T_profile.(reshape(znodes(grid, Center()), 1, 1, :)) .+ zeros(2, 2, 1)
    @test maximum(abs, host(model.temperature) .- T) < 1e-3

    # Cloud condensate is the equilibrium excess of the non-precipitating water, qᵗ - qʳ
    qᵛ⁺ = adjustment_saturation_specific_humidity.(T, host(dynamics.reference_state.pressure), qᵗ, Ref(constants), Ref(equilibrium))
    qᶜ = max.(0, qᵗ - qʳ .- qᵛ⁺)
    @test maximum(qᶜ) > 1e-3
    @test maximum(abs, host(model.microphysical_fields.qᵛ) .- (qᵗ - qʳ .- qᶜ)) < 1e-7
end
