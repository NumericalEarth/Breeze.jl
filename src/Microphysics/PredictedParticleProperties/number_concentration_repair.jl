#####
##### Number concentration repair for mass without number
#####
##### Selected by the `number_repair` field of the top-level
##### [`PredictedParticlePropertiesMicrophysics`](@ref) struct; applied in
##### `update_microphysical_fields!` (p3_microphysical_state.jl).
#####

"""
$(TYPEDEF)

Re-diagnosis of the rain and ice number where a species carries mass but (almost) no number,
after the cleanup step of the two-moment scheme of [Milbrandt and Yau (2005)](@cite MilbrandtYau2005)
(`module_mp_milbrandt2mom.F` in WRF).

Where the mass is at least `minimum_mass_mixing_ratio` but the number density `ρn` is below
`minimum_number_density`, the prognostic number is replaced by a one-moment estimate:

- rain: an exponential distribution with fixed intercept `N₀ʳ = rain_intercept`, so that
  `N = (N₀ʳ)^¾ (ρ qʳ / π ρʷ)^¼`;
- ice: the temperature-dependent number of [Cooper (1986)](@cite Cooper1986),
  `N = max(2 Nᵐⁱⁿ, ℂᶜᵒᵒᵖ₁ exp(ℂᶜᵒᵒᵖ₂ (T₀ - max(ℂᶜᵒᵒᵖ₃, T))))`, where `Nᵐⁱⁿ` is
  `minimum_number_density`, `ℂᶜᵒᵒᵖ₁`, `ℂᶜᵒᵒᵖ₂`, `ℂᶜᵒᵒᵖ₃` are the three Cooper parameters below
  and `T₀` is the freezing temperature.

The estimate is then bounded by P3's own size limits for that mass (the rain slope bounds, and
the global ice number cap and λ limiter), so a trace of mass cannot carry more number than
particles of the smallest admissible size would. Only number densities change; no water moves.

Without this repair the mean-size limiters give mass with zero number the largest admissible
particles (2 mm drops, 2 cm rimed ice). Aloft, where the fall-speed density correction is
large, those trace cells fall at tens of m/s and set the sedimentation CFL number.

# Fields
$(TYPEDFIELDS)
"""
struct NumberConcentrationRepair{FT}
    "Number density below which the number is re-diagnosed [m⁻³]"
    minimum_number_density :: FT
    "Intercept of the exponential rain size distribution [m⁻⁴]"
    rain_intercept :: FT
    "Cooper (1986) ice number coefficient ℂᶜᵒᵒᵖ₁ [m⁻³]"
    ice_number_coefficient :: FT
    "Cooper (1986) ice number temperature sensitivity ℂᶜᵒᵒᵖ₂ [K⁻¹]"
    ice_number_temperature_sensitivity :: FT
    "Temperature ℂᶜᵒᵒᵖ₃ below which the Cooper (1986) ice number stops increasing [K]"
    ice_number_minimum_temperature :: FT
end

"""
$(TYPEDSIGNATURES)

Construct a [`NumberConcentrationRepair`](@ref). The defaults are those of the
[Milbrandt and Yau (2005)](@cite MilbrandtYau2005) scheme: `epsN = 10⁻³ m⁻³`, the rain intercept
`10⁷ m⁻⁴`, and [Cooper (1986)](@cite Cooper1986) ice, `5 exp(0.304 (T₀ - max(233, T))) m⁻³`.

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: NumberConcentrationRepair

NumberConcentrationRepair(Float64)

# output
NumberConcentrationRepair(Nᵐⁱⁿ=0.001 m⁻³, N₀ʳ=1.0e7 m⁻⁴, ice: 5.0 exp(0.304 (T₀ - max(233.0, T))) m⁻³)
```
"""
function NumberConcentrationRepair(FT::DataType = Oceananigans.defaults.FloatType;
                                   minimum_number_density = 1e-3,
                                   rain_intercept = 1e7,
                                   ice_number_coefficient = 5,
                                   ice_number_temperature_sensitivity = 0.304,
                                   ice_number_minimum_temperature = 233)
    return NumberConcentrationRepair(FT(minimum_number_density),
                                     FT(rain_intercept),
                                     FT(ice_number_coefficient),
                                     FT(ice_number_temperature_sensitivity),
                                     FT(ice_number_minimum_temperature))
end

Base.summary(::NumberConcentrationRepair) = "NumberConcentrationRepair"

function Base.show(io::IO, repair::NumberConcentrationRepair)
    print(io, "NumberConcentrationRepair(",
          "Nᵐⁱⁿ=", repair.minimum_number_density, " m⁻³, ",
          "N₀ʳ=", repair.rain_intercept, " m⁻⁴, ",
          "ice: ", repair.ice_number_coefficient,
          " exp(", repair.ice_number_temperature_sensitivity,
          " (T₀ - max(", repair.ice_number_minimum_temperature, ", T))) m⁻³)")
end
