include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using CloudMicrophysics
using Oceananigans
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

using Breeze.Thermodynamics: adjustment_saturation_specific_humidity, saturation_specific_humidity,
                             dry_air_gas_constant, vapor_gas_constant
using Breeze.Microphysics: equilibrated_moisture_mass_fractions

#####
##### set!(model; T, qᵗ) under saturation adjustment recovers T
#####
#
# `set!` converts T into the prognostic thermodynamic variable (ρθˡⁱ or ρs). Under saturation
# adjustment the conversion must partition the moisture at equilibrium at T. Otherwise the adjustment
# in `update_state!` condenses or evaporates the difference, and the latent heat moves T away from
# the value that was set (by 18–30 K in a saturated column).

host(field) = Array(interior(field))

ice_mass_fraction_or_zero(model) =
    haskey(model.microphysical_fields, :qⁱ) ? host(model.microphysical_fields.qⁱ) : zero(host(model.temperature))

# Equilibrium partition at T under the saturation constraint of each dynamics:
# the reference pressure (anelastic) or the total density (compressible).
function expected_condensate(model, T, qᵗ, equilibrium)
    constants = model.thermodynamic_constants
    if model.dynamics isa CompressibleDynamics
        ρ = host(model.dynamics.total_density)
        qᵛ⁺ = saturation_specific_humidity.(T, ρ, Ref(constants), Ref(equilibrium))
    else
        pᵣ = host(model.dynamics.reference_state.pressure)
        qᵛ⁺ = adjustment_saturation_specific_humidity.(T, pᵣ, qᵗ, Ref(constants), Ref(equilibrium))
    end
    q = equilibrated_moisture_mass_fractions.(T, qᵗ, qᵛ⁺, Ref(equilibrium))
    return map(x -> x.liquid, q), map(x -> x.ice, q)
end

# A saturated (qᵗ ≈ 0.0196) and an unsaturated (qᵗ = 0.001) column of T = 290 K − 6 K/km over 5 km.
# The top of the column is at 260 K, so the mixed-phase equilibrium holds both liquid and ice.
const set_T_columns = (saturated = 0.020 / 1.020, unsaturated = 0.001)
set_T_profile(z) = 290 - 6e-3 * z

function check_recovered_temperature(model, qᵗ, equilibrium, FT; T_atol, q_atol)
    T = FT.(set_T_profile.(reshape(znodes(model.grid, Center()), 1, 1, :))) .+ zero(host(model.temperature))
    @test maximum(abs, host(model.temperature) .- T) < T_atol

    qˡ, qⁱ = expected_condensate(model, T, qᵗ, equilibrium)
    @test maximum(abs, host(model.microphysical_fields.qˡ) .- qˡ) < q_atol
    @test maximum(abs, ice_mass_fraction_or_zero(model) .- qⁱ) < q_atol
    return maximum(qˡ .+ qⁱ)
end

@testset "set!(model; T, qᵗ) under saturation adjustment, anelastic [$FT, $formulation]" for FT in test_float_types(),
                                                                                              formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (2, 2, 20), x = (0, 1e3), y = (0, 1e3), z = (0, 5e3))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure = 1e5, potential_temperature = 290)
    dynamics = AnelasticDynamics(reference_state)

    T_atol = FT == Float64 ? 1e-3 : 2e-2
    q_atol = FT == Float64 ? 1e-7 : 1e-5

    for equilibrium in (WarmPhaseEquilibrium(), MixedPhaseEquilibrium()), (column, qᵗ) in pairs(set_T_columns)
        @testset "$(summary(equilibrium)), $column" begin
            microphysics = SaturationAdjustment(FT; equilibrium)
            model = AtmosphereModel(grid; dynamics, formulation, microphysics, thermodynamic_constants = constants)
            set!(model; T = (x, y, z) -> set_T_profile(z), qᵗ = FT(qᵗ))

            max_condensate = check_recovered_temperature(model, FT(qᵗ), equilibrium, FT; T_atol, q_atol)
            column == :saturated ? (@test max_condensate > 1e-3) : (@test max_condensate == 0)

            # update_state! is idempotent on the recovered state
            Oceananigans.TimeSteppers.update_state!(model)
            check_recovered_temperature(model, FT(qᵗ), equilibrium, FT; T_atol, q_atol)
        end
    end
end

# The conversion must not depend on the partition left in the diagnostic fields by an earlier state
@testset "set!(model; T, qᵗ) does not depend on the previous state" begin
    FT = Float64
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (2, 2, 20), x = (0, 1e3), y = (0, 1e3), z = (0, 5e3))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure = 1e5, potential_temperature = 290)
    dynamics = AnelasticDynamics(reference_state)
    equilibrium = WarmPhaseEquilibrium()
    microphysics = SaturationAdjustment(FT; equilibrium)
    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants = constants)

    # Saturated, then unsaturated, then saturated again
    for qᵗ in (set_T_columns.saturated, set_T_columns.unsaturated, set_T_columns.saturated)
        set!(model; T = (x, y, z) -> set_T_profile(z), qᵗ)
        check_recovered_temperature(model, qᵗ, equilibrium, FT; T_atol = 1e-3, q_atol = 1e-7)
    end
end

#####
##### Compressible dynamics: the saturation constraint is at fixed density
#####

# Density and pressure of a hydrostatic column with temperature profile `set_T_profile`, integrated
# with the trapezoidal rule, independently of the model. Each level is the equilibrium state at (T, p).
function hydrostatic_equilibrium_column(z, qᵗ, constants, equilibrium)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    g = constants.gravitational_acceleration

    function equilibrium_density(T, pressure)
        density = pressure / (Rᵈ * T)
        for _ in 1:50
            qᵛ = min(qᵗ, saturation_specific_humidity(T, density, constants, equilibrium))
            density = pressure / (((1 - qᵗ) * Rᵈ + qᵛ * Rᵛ) * T)
        end
        return density
    end

    Nz = length(z)
    p = zeros(Nz)
    ρ = zeros(Nz)
    p[1] = 1e5 - g * 1.2 * z[1]
    ρ[1] = equilibrium_density(set_T_profile(z[1]), p[1])
    for k in 2:Nz
        p[k] = p[k-1]
        for _ in 1:20
            ρk = equilibrium_density(set_T_profile(z[k]), p[k])
            p[k] = p[k-1] - g * (ρ[k-1] + ρk) / 2 * (z[k] - z[k-1])
        end
        ρ[k] = equilibrium_density(set_T_profile(z[k]), p[k])
    end
    return ρ, p
end

function compressible_saturation_adjustment_model(grid, constants, equilibrium)
    θ₀ = 290.0
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5,
                                    reference_potential_temperature = z -> θ₀)
    microphysics = SaturationAdjustment(eltype(grid); equilibrium)
    return AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants = constants,
                           advection = WENO(order = 5))
end

@testset "set!(model; T, qᵗ, ρ) under saturation adjustment, compressible [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (8, 20), halo = (5, 5), x = (0, 800), z = (0, 5e3),
                           topology = (Periodic, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    z = znodes(grid, Center())

    T_atol = FT == Float64 ? 1e-3 : 2e-2
    q_atol = FT == Float64 ? 1e-7 : 1e-5

    for equilibrium in (WarmPhaseEquilibrium(), MixedPhaseEquilibrium()), (column, qᵗ) in pairs(set_T_columns)
        @testset "$(summary(equilibrium)), $column" begin
            model = compressible_saturation_adjustment_model(grid, constants, equilibrium)
            ρ, p = hydrostatic_equilibrium_column(z, qᵗ, constants, equilibrium)
            set!(model; T = (x, z) -> set_T_profile(z), qᵗ = FT(qᵗ), ρ = reshape(FT.(ρ), 1, 1, :) .+ zeros(FT, 8, 1, 1))

            max_condensate = check_recovered_temperature(model, FT(qᵗ), equilibrium, FT; T_atol, q_atol)
            column == :saturated ? (@test max_condensate > 1e-3) : (@test max_condensate == 0)

            # The model pressure p = ρ Rᵐ T matches the independently integrated column
            p_column = FT.(reshape(p, 1, 1, :)) .+ zeros(FT, 8, 1, 1)
            @test maximum(abs, (host(model.dynamics.pressure) .- p_column) ./ p_column) < (FT == Float64 ? 1e-5 : 1e-3)
        end
    end
end

@testset "A saturated compressible rest state set from T stays near rest" begin
    FT = Float64
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (8, 50), halo = (5, 5), x = (0, 800), z = (0, 5e3),
                           topology = (Periodic, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    equilibrium = WarmPhaseEquilibrium()
    qᵗ = set_T_columns.saturated

    model = compressible_saturation_adjustment_model(grid, constants, equilibrium)
    ρ, _ = hydrostatic_equilibrium_column(znodes(grid, Center()), qᵗ, constants, equilibrium)
    set!(model; T = (x, z) -> set_T_profile(z), qᵗ, ρ = reshape(ρ, 1, 1, :) .+ zeros(8, 1, 1))

    simulation = Simulation(model; Δt = 1, stop_iteration = 60, verbose = false)
    run!(simulation)

    # Before the fix, the latent heat released by the adjustment warmed the column by 18–30 K
    # and drove w to 2–8 m/s within 60 s.
    @test maximum(abs, host(model.velocities.w)) < 1e-3
    T = set_T_profile.(reshape(znodes(grid, Center()), 1, 1, :)) .+ zeros(8, 1, 1)
    @test maximum(abs, host(model.temperature) .- T) < 1e-3
end

#####
##### One-moment microphysics with saturation adjustment: the adjustment holds the rain fixed
#####

@testset "set!(model; T, qᵗ, qʳ) under one-moment saturation adjustment [$(summary(equilibrium))]" for equilibrium in (WarmPhaseEquilibrium(), MixedPhaseEquilibrium())
    FT = Float64
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (2, 2, 20), x = (0, 1e3), y = (0, 1e3), z = (0, 5e3))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure = 1e5, potential_temperature = 290)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics(cloud_formation = SaturationAdjustment(FT; equilibrium))
    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants = constants)

    qᵗ = set_T_columns.saturated
    qʳ = 1e-3
    set!(model; T = (x, y, z) -> set_T_profile(z), qᵗ, qʳ)

    T = set_T_profile.(reshape(znodes(grid, Center()), 1, 1, :)) .+ zeros(2, 2, 1)
    @test maximum(abs, host(model.temperature) .- T) < 1e-3

    # Cloud condensate is the equilibrium excess of the non-precipitating water, qᵗ - qʳ
    pᵣ = host(reference_state.pressure)
    qᵛ⁺ = adjustment_saturation_specific_humidity.(T, pᵣ, qᵗ, Ref(constants), Ref(equilibrium))
    qᶜ = max.(0, qᵗ - qʳ .- qᵛ⁺)
    @test maximum(qᶜ) > 1e-3
    @test maximum(abs, host(model.microphysical_fields.qᵛ) .- (qᵗ - qʳ .- qᶜ)) < 1e-7
end
