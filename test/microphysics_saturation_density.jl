include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Breeze.AtmosphereModels: total_density, specific_prognostic_moisture, microphysics_model_update!
using Breeze.Thermodynamics: PlanarLiquidSurface, saturation_vapor_pressure, vapor_gas_constant,
                             dry_air_gas_constant
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, TwoMomentCloudMicrophysics

# Microphysics saturation is referenced to the density that the mass fractions are referenced to
# (`gas_phase_density`, the local moist density pᵣ / (Rᵐ T) on the anelastic core), not to the
# dry reference density ρᵣ that weights the anelastic tendencies. A column of air held exactly at
# saturation, with no condensate, must therefore neither condense nor evaporate.
#
# The reference state is 10 K colder than the air, so ρᵣ exceeds the local density by ~3%: a scheme
# that saturates at ρᵣ sees the column ~3% supersaturated and condenses.

# Saturation specific humidity of air whose moisture is all vapor, qᵗ = qᵛ⁺, in closed form at the
# pressure p (equivalently at the density p / (Rᵐ T) with Rᵐ the mixture gas constant of that vapor).
function all_vapor_saturation_specific_humidity(T, p, constants)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
    ϵ = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    return ϵ * pᵛ⁺ / (p + (ϵ - 1) * pᵛ⁺)
end

function saturated_column_model(FT, microphysics, dynamics_kind; constants = ThermodynamicConstants(FT))
    grid = RectilinearGrid(default_arch, FT; size = 8, z = (0, 2000), topology = (Flat, Flat, Bounded))
    if dynamics_kind === :anelastic
        reference_state = ReferenceState(grid, constants; base_pressure = 101325, potential_temperature = 290)
        dynamics = AnelasticDynamics(reference_state)
    else
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 290)
    end
    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants = constants, advection = nothing)

    # Air 10 K warmer than the reference, saturated at its own pressure and temperature.
    T = z -> FT(300) - FT(0.0065) * z
    p = Breeze.AtmosphereModels.dynamics_pressure(model.dynamics)
    Tᵢ = CenterField(grid)
    set!(Tᵢ, T)
    qᵛ⁺ = CenterField(grid)
    set!(qᵛ⁺, all_vapor_saturation_specific_humidity.(interior(Tᵢ), interior(p), Ref(constants)))
    kw = dynamics_kind === :compressible ? (; ρ = model.dynamics.reference_state.density) : (;)
    set!(model; kw..., T, qᵗ = qᵛ⁺)

    # Compressible pressure is diagnosed from the state just set, p = ρ Rᵐ T, and Rᵐ depends
    # on the vapor: iterate to the saturated fixed point.
    if dynamics_kind === :compressible
        for _ in 1:8
            set!(qᵛ⁺, all_vapor_saturation_specific_humidity.(interior(Tᵢ), interior(p), Ref(constants)))
            set!(model; kw..., T, qᵗ = qᵛ⁺)
        end
    end

    update_state!(model; compute_tendencies = true)
    return model
end

column(f) = Array(interior(f, 1, 1, :))

@testset "Non-equilibrium cloud formation saturates at the local density [$FT, $dynamics_kind]" for FT in test_float_types(),
                                                                                                    dynamics_kind in (:anelastic, :compressible)
    schemes = (warm_one_moment = OneMomentCloudMicrophysics(FT; cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), nothing)),
               mixed_one_moment = OneMomentCloudMicrophysics(FT; cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))),
               two_moment = TwoMomentCloudMicrophysics(FT))

    for (name, microphysics) in pairs(schemes)
        model = saturated_column_model(FT, microphysics, dynamics_kind)
        ρ = column(total_density(model.dynamics))
        qᵛ = column(specific_prognostic_moisture(model))

        # The column is saturated by Breeze's own diagnostic.
        qᵛ⁺ = column(SaturationSpecificHumidityField(model))
        @test all(isapprox.(qᵛ, qᵛ⁺; rtol = 100eps(FT)))

        # The anelastic reference density is far enough from the local density to matter.
        if dynamics_kind === :anelastic
            pᵣ = column(model.dynamics.reference_state.pressure)
            ρᵐ = pᵣ ./ (column(model.temperature) .* (dry_air_gas_constant(model.thermodynamic_constants) .* (1 .- qᵛ) .+
                                                       vapor_gas_constant(model.thermodynamic_constants) .* qᵛ))
            @test minimum(ρ ./ ρᵐ) > FT(1.02)
        end

        # Neither condensation nor evaporation: the condensation rate is measured against the
        # rate a 1% supersaturation would drive, ρ qᵛ⁺ / (100 τ) with τ ≈ 10 s.
        Gρqᶜˡ = column(model.timestepper.Gⁿ.ρqᶜˡ)
        Gρqᵛ = column(model.timestepper.Gⁿ.ρqᵛ)
        scale = maximum(ρ .* qᵛ⁺) / 1000
        tolerance = FT == Float64 ? 1e-6 * scale : 1e-2 * scale
        @test maximum(abs, Gρqᶜˡ) < tolerance
        @test maximum(abs, Gρqᵛ) < tolerance
    end
end

@testset "Kessler saturates at the local density [$FT, $dynamics_kind]" for FT in test_float_types(),
                                                                             dynamics_kind in (:anelastic, :compressible)
    constants = ThermodynamicConstants(FT; saturation_vapor_pressure = TetensFormula(FT))
    model = saturated_column_model(FT, DCMIP2016KesslerMicrophysics(FT), dynamics_kind; constants)

    qᵛ = column(specific_prognostic_moisture(model))
    qᵛ⁺ = column(SaturationSpecificHumidityField(model))
    @test all(isapprox.(qᵛ, qᵛ⁺; rtol = 100eps(FT)))

    # Kessler adjusts to saturation within its update: a saturated column without condensate
    # must not form any cloud.
    model.clock.last_Δt = FT(10)
    microphysics_model_update!(model.microphysics, model)
    qᶜˡ = column(model.microphysical_fields.ρqᶜˡ) ./ column(total_density(model.dynamics))
    tolerance = FT == Float64 ? 1e-6 * maximum(qᵛ⁺) : 1e-2 * maximum(qᵛ⁺)
    @test maximum(abs, qᶜˡ) < tolerance
end
