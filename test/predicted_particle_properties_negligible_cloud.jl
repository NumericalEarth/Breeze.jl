include(joinpath(@__DIR__, "setup.jl"))

using Test
using Breeze
using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.Thermodynamics: ThermodynamicConstants, StaticEnergyState, MoistureMassFractions, mixture_heat_capacity
using Breeze.Microphysics.PredictedParticleProperties:
    PredictedParticlePropertiesMicrophysics, AerosolActivation, AerosolMode,
    p3_state_tendencies, cloud_number_per_cloud_mass, cloud_riming_number_rate

# Advection and sedimentation leave positive but negligible cloud mass (down to subnormal
# values) in cloud-free cells while the DSD diagnosis floors the droplet number above zero.
# The number-per-mass quotient must then be zero rather than Inf, otherwise every
# companion number rate of zero becomes Inf × 0 = NaN and the prognostic ρnᶜˡ is poisoned.
@testset "P3 cloud number rates with negligible cloud mass [$FT]" for FT in (Float64, Float32)
    constants = ThermodynamicConstants(FT)
    p3 = PredictedParticlePropertiesMicrophysics(FT; aerosol = AerosolActivation(AerosolMode(FT)))
    qmin = FT(p3.minimum_mass_mixing_ratio)
    ρ = FT(0.9)

    for qᶜˡ in (nextfloat(zero(FT)), FT(1e-30), qmin / 2), Nᶜˡ in (zero(FT), FT(1e8))
        @test cloud_number_per_cloud_mass(Nᶜˡ, ρ, qᶜˡ, qmin) == 0
        @test cloud_riming_number_rate(p3, qᶜˡ, Nᶜˡ, ρ, zero(FT)) == 0
    end
    @test cloud_number_per_cloud_mass(FT(1e8), ρ, FT(1e-6), qmin) ≈ FT(1e8) / (ρ * FT(1e-6))

    # Full tendency bundle in a subsaturated free-tropospheric cell (no activation)
    T = FT(275)
    z = FT(2800)
    p = FT(71000)
    qᵛ = FT(3e-3)
    g = constants.gravitational_acceleration
    ℒ = constants.liquid.reference_latent_heat
    for qᶜˡ in (nextfloat(zero(FT)), FT(1e-30)), nᶜˡ in (zero(FT), FT(1e-12), FT(1e8))
        q = MoistureMassFractions(qᵛ, qᶜˡ, zero(FT))
        s = mixture_heat_capacity(q, constants) * T + g * z - ℒ * qᶜˡ
        𝒰 = StaticEnergyState{FT}(s, q, z, p)
        μ = (; ρqᶜˡ = ρ * qᶜˡ, ρnᶜˡ = ρ * nᶜˡ, ρqʳ = zero(FT), ρnʳ = zero(FT), ρqⁱ = zero(FT), ρnⁱ = zero(FT),
               ρqᶠ = zero(FT), ρbᶠ = zero(FT), ρqʷⁱ = zero(FT), ρnᵃ = ρ * FT(3e8))
        ℳ = AM.microphysical_state(p3, ρ, μ, 𝒰, (; w = zero(FT)))
        result = p3_state_tendencies(p3, ρ, ℳ, 𝒰, constants)
        for name in propertynames(result)
            @test isfinite(getproperty(result, name))
        end
        @test result.tendency_ρnᶜˡ == 0
    end
end
