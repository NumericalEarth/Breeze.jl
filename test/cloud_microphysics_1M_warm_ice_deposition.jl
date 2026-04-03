using Breeze
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudIce, CloudLiquid
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.ThermodynamicsInterface as TDI
using Test

using Breeze.Thermodynamics:
    MoistureMassFractions,
    LiquidIcePotentialTemperatureState,
    PlanarIceSurface,
    density,
    saturation_specific_humidity,
    with_temperature

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

@testset "MPNE1M suppresses warm cloud-ice growth [$(FT)]" for FT in test_float_types()
    constants = ThermodynamicConstants()
    microphysics = OneMomentCloudMicrophysics(FT;
                                              cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT)))

    T = FT(276)
    qᵛ = FT(0.007)
    qᶜˡ = FT(0)
    qᶜⁱ = FT(0)
    qʳ = FT(0)
    qˢ = FT(0)

    q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ, qᶜⁱ + qˢ)
    𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, FT(1e5), FT(101325)), T, constants)
    ρ = density(𝒰, constants)

    qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
    @test qᵛ > qᵛ⁺ⁱ

    ℳ = BreezeCloudMicrophysicsExt.MixedPhaseOneMomentState(qᶜˡ, qᶜⁱ, qʳ, qˢ)
    G = BreezeCloudMicrophysicsExt.mpne1m_tendencies(microphysics, ρ, ℳ, 𝒰, constants)

    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics1MParams(FT)
    reference = BMT.bulk_microphysics_tendencies(
        BMT.Microphysics1Moment(),
        mp,
        tps,
        ρ,
        T,
        qᵛ + qᶜˡ + qᶜⁱ + qʳ + qˢ,
        qᶜˡ,
        qᶜⁱ,
        qʳ,
        qˢ,
    )

    @test reference.dq_icl_dt == zero(FT)
    @test G.ρqᶜⁱ / ρ == zero(FT)
end
