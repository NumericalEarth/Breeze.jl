include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: set_to_mean!, liquid_mass_fraction, ice_mass_fraction
using Breeze.Microphysics: NonEquilibriumCloudFormation
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using Oceananigans
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, TwoMomentCloudMicrophysics

# set_to_mean! averages the liquid and ice mass fractions of the model into the reference state,
# so every microphysics scheme must provide them, including the precipitating ones.

function horizontal_mean_profile(mass_fraction)
    field = Field(mass_fraction)
    compute!(field)
    return dropdims(sum(Array(interior(field)), dims=(1, 2)), dims=(1, 2)) ./ (size(field, 1) * size(field, 2))
end

reference_profile(field) = Array(interior(field))[1, 1, :]

@testset "set_to_mean! with precipitating microphysics [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1000), y=(0, 1000), z=(0, 1000))
    constants = ThermodynamicConstants(FT)

    # Rain varies horizontally so that the reference must hold its horizontal mean
    qʳ(x, y, z) = 1e-3 * x / 1000

    cases = (
        ("one-moment, warm phase, non-equilibrium", OneMomentCloudMicrophysics(FT),
         (; qᶜˡ=1e-3, qʳ)),
        ("one-moment, warm phase, saturation adjustment",
         OneMomentCloudMicrophysics(FT; cloud_formation=SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())),
         (; qʳ)),
        ("one-moment, mixed phase, non-equilibrium",
         OneMomentCloudMicrophysics(FT; cloud_formation=NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))),
         (; qᶜˡ=1e-3, qʳ, qᶜⁱ=2e-4, qˢⁿ=3e-4)),
        ("one-moment, mixed phase, saturation adjustment",
         OneMomentCloudMicrophysics(FT; cloud_formation=SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))),
         (; qʳ, qˢⁿ=3e-4)),
        ("two-moment", TwoMomentCloudMicrophysics(FT),
         (; qᶜˡ=1e-3, nᶜˡ=1e8, qʳ, nʳ=1e5)),
        ("P3", PredictedParticlePropertiesMicrophysics(FT),
         (; qᶜˡ=1e-3, qʳ, qⁱ=2e-4, qʷⁱ=5e-5)),
    )

    for (name, microphysics, condensate) in cases
        @testset "$name" begin
            reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=300,
                                             liquid_mass_fraction=0, ice_mass_fraction=0)
            model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), microphysics,
                                    thermodynamic_constants=constants)
            set!(model; θ=300, qᵗ=0.015, condensate...)

            # Without rescaling, set_to_mean! changes ρᵣ under fixed ρq, so the means are taken before
            q̄ˡ = horizontal_mean_profile(liquid_mass_fraction(model))
            qⁱ = ice_mass_fraction(model)
            q̄ⁱ = isnothing(qⁱ) ? zero(q̄ˡ) : horizontal_mean_profile(qⁱ)

            set_to_mean!(reference_state, model)

            @test all(q̄ˡ .> 0)
            @test reference_profile(reference_state.liquid_mass_fraction) ≈ q̄ˡ
            @test isnothing(qⁱ) || all(q̄ⁱ .> 0)
            @test reference_profile(reference_state.ice_mass_fraction) ≈ q̄ⁱ
        end
    end
end
