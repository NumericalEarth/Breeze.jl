#####
##### Process Rate Parameters
#####
##### Container for all P3 microphysical process rate parameters.
##### These parameters control timescales, efficiencies, and thresholds.
#####

export ProcessRateParameters, NumericalFloors

"""
    NumericalFloors

The smallest value each quantity is allowed to take before it enters a division
or a logarithm, so that a process rate stays finite where the physics may
legitimately reach zero. `divisor` is the last resort — the smallest strictly
positive number substituted for anything that would otherwise send a quotient or
a logarithm to infinity. `mean_particle_mass_fallback` is the one entry that
replaces a quantity outright rather than bounding it, standing in for a mean
particle mass where the number concentration is exactly zero.

Each value is a field rather than a literal so that a configuration can move all
of them together. The defaults sit far below any atmospheric value at double
precision and remain normal numbers in `Float32`, whose smallest normal is
``1.2 × 10^{-38}``. They do *not* survive `Float16`, whose smallest normal is
``6.1 × 10^{-5}`` — half precision needs floors raised into that range, which is
the reason they are settable rather than baked in.

See [`NumericalFloors()`](@ref) for the defaults.
"""
struct NumericalFloors{FT}
    saturation_mass_fraction :: FT  # floors qᵛ⁺ˡ, qᵛ⁺ⁱ in supersaturation denominators [kg/kg]
    transport_coefficient :: FT     # floors Dᵛ and ν in Schmidt-number ratios [m²/s]
    mass_scale :: FT                # floors mass and volume denominators and log₁₀
                                    # arguments [kg/kg, kg, or m³/kg]
    number_scale :: FT              # floors number denominators [1/kg]; the scheme-less
                                    # counterpart of `minimum_number_mixing_ratio`
    rate_scale :: FT                # rate below which a process is treated as inactive
    divisor :: FT                   # smallest positive value admitted beneath a division or a `log`
    mean_particle_mass_fallback :: FT   # mean particle mass substituted where the number
                                        # concentration is exactly zero [kg]; only has to
                                        # land inside the ice tables' mass axis
end

"""
$(TYPEDSIGNATURES)

Construct the numerical floors shared by the P3 process rates.

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: NumericalFloors
NumericalFloors(Float64)

# output
NumericalFloors(mass_scale=1.0e-20)
```
"""
function NumericalFloors(FT::Type{<:AbstractFloat} = Float64;
                         saturation_mass_fraction = 1e-10,
                         transport_coefficient = 1e-10,
                         mass_scale = 1e-20,
                         number_scale = 1e-16,
                         rate_scale = 1e-20,
                         divisor = 1e-30,
                         mean_particle_mass_fallback = 1e-12)
    return NumericalFloors{FT}(FT(saturation_mass_fraction),
                               FT(transport_coefficient),
                               FT(mass_scale),
                               FT(number_scale),
                               FT(rate_scale),
                               FT(divisor),
                               FT(mean_particle_mass_fallback))
end

# Re-typing conversion, used when `ProcessRateParameters(FT)` is handed floors
# that were built at a different precision.
function NumericalFloors{FT}(floors::NumericalFloors) where FT
    return NumericalFloors{FT}(floors.saturation_mass_fraction,
                               floors.transport_coefficient,
                               floors.mass_scale,
                               floors.number_scale,
                               floors.rate_scale,
                               floors.divisor,
                               floors.mean_particle_mass_fallback)
end

# The four pure PSD helpers (`liquid_fraction_on_ice`, `mean_total_ice_mass`,
# `bounded_ice_number`, `ventilation_sc_correction`) are reached without a scheme
# argument, so they read their floors from here. Everything else reads the
# settable `floors` field of the scheme's `ProcessRateParameters`.
const DEFAULT_FLOORS = NumericalFloors(Float64)

Base.summary(::NumericalFloors) = "NumericalFloors"

function Base.show(io::IO, floors::NumericalFloors)
    print(io, summary(floors), "(mass_scale=", floors.mass_scale, ")")
end

"""
    ProcessRateParameters

Parameters for P3 microphysical process rates.
See [`ProcessRateParameters()`](@ref) constructor for usage.
"""
struct ProcessRateParameters{FT, PS}
    # Physical constants
    liquid_water_density :: FT       # ρʷ [kg/m³]
    pure_ice_density :: FT           # ρⁱ [kg/m³]
    reference_air_density :: FT      # ρ₀ [kg/m³] for rain fall speed correction (Fortran rhosur)
    nucleated_ice_mass :: FT         # mᵢ₀ [kg], mass of newly nucleated ice crystal
    activated_droplet_radius :: FT   # r₀ [m], radius of a newly activated cloud droplet
    activation_supersaturation_threshold :: FT  # S above which CCN activation proceeds [-]
    freezing_temperature :: FT       # T₀ [K]

    # Rain autoconversion (Khairoutdinov-Kogan 2000)
    autoconversion_coefficient :: FT         # k₁ = 1350 × (Nc_ref_cm)^β, see KK2000 Eq. 29
    autoconversion_exponent_cloud :: FT      # α [-]
    autoconversion_exponent_droplet :: FT    # β [-]
    autoconversion_threshold :: FT           # qᶜˡ threshold [kg/kg]
    autoconversion_reference_concentration :: FT  # Nᶜˡ reference [1/m³]

    # Rain accretion (Khairoutdinov-Kogan 2000)
    accretion_coefficient :: FT              # k₂ [s⁻¹]
    accretion_exponent :: FT                 # α [-]

    # Rain self-collection and breakup (KK2000 self-collection rate combined with
    # Verlinde-Cotton 1993 breakup multiplier; matches Fortran P3 v5.5.0 autoAccr_param=2)
    rain_self_collection_coefficient :: FT        # k_rr [-]
    rain_breakup_diameter_threshold :: FT    # D_th threshold for breakup [m] (1/λ_r convention)
    rain_breakup_coefficient :: FT           # κ_br [1/m]

    # Evaporation/sublimation timescales
    rain_evaporation_timescale :: FT         # τ_evap [s]
    ice_deposition_timescale :: FT           # τ_dep [s]

    # Ice aggregation. The sticking efficiency ramps linearly from its minimum to maximum
    # between the two temperatures, and is then shut off between the two rime
    # fractions, above which heavily rimed particles no longer aggregate.
    maximum_aggregation_efficiency :: FT     # Eⁱⁱ_max [-]
    minimum_aggregation_efficiency :: FT     # Eⁱⁱ_min [-], the cold-ice value
    aggregation_timescale :: FT              # τ_agg [s]
    aggregation_efficiency_ramp_start_temperature :: FT # T where Eⁱⁱ starts increasing [K]
    aggregation_efficiency_ramp_end_temperature :: FT   # T where Eⁱⁱ reaches Eⁱⁱ_max [K]
    minimum_aggregation_rime_fraction :: FT  # Fᶠ below which aggregation is unreduced [-]
    maximum_aggregation_rime_fraction :: FT  # Fᶠ above which aggregation is off [-]

    # Cloud riming
    cloud_ice_collection_efficiency :: FT    # Eᶜⁱ [-]

    # Rain riming
    rain_ice_collection_efficiency :: FT     # Eʳⁱ [-]

    # Rime density bounds
    minimum_rime_density :: FT               # ρ_rim_min [kg/m³]
    maximum_rime_density :: FT               # ρ_rim_max [kg/m³]

    # Riming impact parameter Ri, which sets the density of freshly accreted rime.
    # Ri = c Dᶜ |vⁱ - vᶜ| / (T₀ - T) (Fortran P3 v5.5.0 p3_main cloud-riming branch).
    rime_impact_coefficient :: FT            # c [K s / m²], Fortran 0.5e6
    minimum_rime_impact :: FT                # Ri floor [-]
    maximum_rime_impact :: FT                # Ri ceiling [-]
    minimum_riming_supercooling :: FT        # smallest T₀ - T admitted in Ri [K]
    unrimed_rime_density :: FT               # ρᶠ assigned when cloud riming is inactive [kg/m³]

    # Shedding
    shed_drop_mass :: FT                     # m_shed [kg] (cloud/wet-growth shedding)
    shed_drop_mass_liqfrac :: FT             # m_shed [kg] (liquid-fraction shedding, Fortran 1.928e6)

    # Wet growth is off where there is too little cloud plus rain to collect, or where
    # collection barely outpaces the freezing capacity.
    wet_growth_hydrometeor_threshold :: FT   # qᶜˡ + qʳ below which wet growth is off [kg/kg]
    wet_growth_excess_threshold :: FT        # collection in excess of the freezing capacity
                                             # needed to fire wet growth [kg/kg/s]

    # Refreezing
    refreezing_timescale :: FT               # τ_frz [s]

    # Deposition nucleation (Cooper 1986)
    ice_nucleation_temperature_threshold :: FT   # T below which nucleation occurs [K]
    ice_nucleation_supersaturation_threshold :: FT  # Sⁱ threshold [-]
    maximum_ice_nucleation_concentration :: FT   # Nⁱ_max [1/m³]
    ice_nucleation_timescale :: FT               # τ_nuc [s]
    ice_nucleation_coefficient :: FT             # Cooper (1986) prefactor [1/m³] (default 5.0)
    ice_nucleation_temperature_coefficient :: FT # Cooper (1986) supercooling rate [1/K] (default 0.304)

    # Immersion freezing (Barklie-Gokhale 1959)
    maximum_immersion_freezing_temperature :: FT # T_max [K]
    immersion_freezing_coefficient :: FT     # aimm [-]
    immersion_freezing_nucleation_coefficient :: FT  # bimm [m⁻³s⁻¹]

    # Rime splintering (Hallett-Mossop)
    minimum_splintering_temperature :: FT    # lower temperature bound [K]
    maximum_splintering_temperature :: FT    # upper temperature bound [K]
    splintering_temperature_peak :: FT       # T_peak [K]
    splintering_rate :: FT                   # splinters per kg rime
    splintering_crystal_mass :: FT           # mass per HM splinter [kg] (Fortran Dinit_HM = 10 μm)
    splintering_diameter_threshold :: FT     # minimum diameter [m] for HM splintering
    splintering_cloud_riming_scale :: FT     # 1.0 for nCat=1 (include), 0.0 for nCat>1 (exclude)
    maximum_splintering_liquid_fraction :: FT # Fˡ max for HM splintering
    maximum_splintering_surface_temperature :: FT # warm-surface shutoff [K] (Inf disables)

    # Initial rain drop mass (for autoconversion number tendency)
    initial_rain_drop_mass :: FT             # m_rain_init [kg]

    # Homogeneous freezing (Koop et al. 2000)
    homogeneous_freezing_temperature :: FT   # T < threshold: all cloud/rain freezes [K]
    homogeneous_freezing_timescale :: FT     # τ_hom [s], effective instantaneous

    # Rime densification
    rime_densification_timescale :: FT       # τ_densif [s]

    # Rain size distribution bounds (Fortran P3 v5.5.0: lamr_min, lamr_max)
    minimum_rain_slope :: FT                # λʳ minimum [1/m]
    maximum_rain_slope :: FT                # λʳ maximum [1/m]

    # Sink-limiting safety timescale [s]
    # If total sinks for any species × dt_safety exceed available mass,
    # all sink rates for that species are rescaled proportionally.
    sink_limiting_timescale :: FT            # dt_safety [s]

    # Coupled donor-budget limiter
    # Fixed re-projection passes for the coupled dry-ice, rain, total-ice,
    # and coating-water sink budgets. Must be positive.
    coupled_sink_limiting_iterations :: Int

    # Global ice number limiter (Fortran P3 v5.5.0 impose_max_Ni)
    # Applied as a relaxation sink whenever nⁱ × ρ exceeds the maximum.
    maximum_ice_number_density :: FT         # Nⁱ_max [1/m³]

    # Liquid fraction clipping threshold (Milbrandt et al. 2025)
    # Fl < this: instantly freeze all qwi to rime; Fl > (1 - this): fully melt to rain.
    # Implemented as a relaxation drain over refreezing_timescale.
    liquid_fraction_clipping_threshold :: FT              # Fortran liqfracsmall [-]

    # Fortran's separate "complete melting" diagnostic: a particle this liquid is
    # transferred whole to rain regardless of the clipping threshold above.
    complete_melting_liquid_fraction :: FT                # [-]

    # M12(c): Tiny-ice threshold for warm pre-processing (Fortran qsmall_dry).
    # Ice with qi ∈ [qsmall, qsmall_dry) at T ≥ T₀ is converted to rain.
    tiny_ice_to_rain_threshold :: FT                         # [kg/kg]

    # Tiny-mass instant-evaporation clauses of the saturation adjustment
    # (Fortran microphy_p3.f90 3684-3685, 3715-3719, 3753-3756): a species
    # holding less than `tiny_mass_evaporation_threshold` in air more than
    # `subsaturation_evaporation_threshold` below saturation is drained outright.
    tiny_mass_evaporation_threshold :: FT                    # [kg/kg]
    subsaturation_evaporation_threshold :: FT                # [-]

    # Liquid fraction mode (Fortran log_LiquidFrac).
    # When true: wet growth rime densification is suppressed (liquid tracked
    # explicitly in qʷⁱ), and melt-densification is skipped.
    liquid_fraction_active :: Bool

    # Predicted supersaturation mode (Fortran log_predictSsat).
    # When true, carry supersaturation as a prognostic variable and use
    # bounded Grabowski-Morrison (2008) adjustment for condensation.
    # When false (default), use relaxation-to-saturation.
    #
    # The public field remains a `Bool`. The same value is carried by the `PS` type
    # parameter so allocation decisions and kernel gates can dispatch at compile time.
    # `prognostic_field_names` must fold to a constant tuple (a `Union` return type makes
    # the host-side prognostic loop allocate), and `materialize_microphysical_fields`
# needs the type value to decide whether `ρsᵛ⁺ˡ` exists at all.
    predict_supersaturation :: Bool

    # Deposition/sublimation calibration factors (Fortran P3 v5.5.0 clbfact_dep, clbfact_sub).
    # Ad hoc multipliers to increase or decrease deposition and/or sublimation rates.
    # The representation of ice capacitances is highly simplified and the appropriate
    # values in the diffusional growth equation are uncertain (Fortran comment, line 3721).
    calibration_factor_deposition :: FT
    calibration_factor_sublimation :: FT

    # Numerical floors shared by every process rate
    floors :: NumericalFloors{FT}
end

"""
$(TYPEDSIGNATURES)

Construct process rate parameters with default values from P3 literature.

The liquid-water density and the dry-air gas constant used by the reference-density
calculation come from `thermodynamic_constants`.

These parameters control the rates of all microphysical processes:
autoconversion, accretion, aggregation, riming, melting, evaporation,
deposition, nucleation, and freezing.

Ice terminal-velocity, projected-area, collection, and ventilation integrals are
read from the Fortran lookup tables by [`read_lookup_tables`](@ref).
Rain velocity and evaporation integrals are generated with Julia quadrature.
Cloud PSD shape is diagnosed from droplet number, while the active rain-process
path uses ``μ_r = 0``. None are duplicated in this rate-parameter container.

# Default Sources

- Autoconversion/accretion: Khairoutdinov and Kogan (2000)
- Self-collection/breakup: Seifert and Beheng (2001, 2006)
- Aggregation: Morrison and Milbrandt (2015)
- Nucleation: Cooper (1986)
- Freezing: Barklie and Gokhale (1959)
- Splintering: Hallett and Mossop (1974)

# Example

The second type parameter carries the value of the Boolean
`predict_supersaturation` field, so the default `false` drops `ρsᵛ⁺ˡ` from the
prognostic set entirely while `params.predict_supersaturation` remains usable in
ordinary Boolean expressions.

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: ProcessRateParameters
params = ProcessRateParameters(Float64)
typeof(params)

# output
ProcessRateParameters{Float64, false}
```

All parameters are keyword arguments with physically-based defaults. The coupled
donor-budget limiter uses four re-projection passes by default; set
`coupled_sink_limiting_iterations` to tune that count.
"""
function ProcessRateParameters(FT::Type{<:AbstractFloat} = Float64;
        # Physical constants
        thermodynamic_constants = ThermodynamicConstants(FT),
        pure_ice_density = thermodynamic_constants.ice.density,
        # Reference density for rain fall speed correction (P=1000 hPa, T=0°C)
        reference_air_density = 100000 / (dry_air_gas_constant(thermodynamic_constants) * 273.15),
        nucleated_ice_mass = 4 * FT(π) / 3 * 900 * (1e-6)^3,  # Fortran mi0: sphere of radius 1 μm, ρ=900 kg/m³ [kg]
        activated_droplet_radius = 1e-6,               # Fortran cons7 uses a 1 μm droplet
        activation_supersaturation_threshold = 1e-6,   # Fortran sup_cld threshold
        freezing_temperature = 273.15,

        # Rain autoconversion
    # KK2000 Eq. 29: dqʳ/dt = 1350 qᶜˡ^2.47 Nᶜˡ^(-1.79) with Nᶜˡ in cm⁻³.
    # Rescaled for (Nᶜˡ/Nᶜˡ_ref)^β with Nᶜˡ_ref = 1e8 m⁻³ = 100 cm⁻³:
        # k₁ = 1350 × (100)^(-1.79) ≈ 0.355
        autoconversion_coefficient = 1350 * 100.0^(-1.79),
        autoconversion_exponent_cloud = 2.47,
        autoconversion_exponent_droplet = -1.79,
        autoconversion_threshold = 1e-8,  # Fortran P3 v5.5.0 qsmall_dry1
        autoconversion_reference_concentration = 1e8,

        # Rain accretion
        accretion_coefficient = 67.0,
        accretion_exponent = 1.15,

        # Rain self-collection and breakup
        rain_self_collection_coefficient = 5.78,
        rain_breakup_diameter_threshold = 280e-6,  # 280 μm: Fortran P3 breakup threshold (1/λ_r convention)
        rain_breakup_coefficient = 2300.0,

        # Timescales
        rain_evaporation_timescale = 10.0,
        ice_deposition_timescale = 10.0,

        # Ice aggregation (Morrison & Milbrandt 2015a; Fortran Eii / Eii_fact)
        maximum_aggregation_efficiency = 0.3,
        minimum_aggregation_efficiency = 0.001,
        aggregation_timescale = 600.0,
        aggregation_efficiency_ramp_start_temperature = 253.15,
        aggregation_efficiency_ramp_end_temperature = 273.15,
        minimum_aggregation_rime_fraction = 0.6,
        maximum_aggregation_rime_fraction = 0.9,

        # Cloud riming
        cloud_ice_collection_efficiency = 0.5,

        # Rain riming
        rain_ice_collection_efficiency = 1.0,

        # Rime density
        minimum_rime_density = 50.0,
        maximum_rime_density = 900.0,

        # Riming impact parameter. The Ri range and the supercooling floor are the
        # Fortran P3 v5.5.0 bounds; outside 1 ≤ Ri ≤ 12 the underlying rime-density
        # fit is extrapolating beyond the data it was fit to.
        rime_impact_coefficient = 0.5e6,
        minimum_rime_impact = 1.0,
        maximum_rime_impact = 12.0,
        minimum_riming_supercooling = 0.001,
        unrimed_rime_density = 400.0,

        # Shedding
        shed_drop_mass = 1 / 1.923e6,  # m19: Fortran 1 mm drop mass (microphy_p3.f90 1.923e6 drops/kg)
        # Fortran uses 1.928e6 for liquid-fraction shedding (nlshd, line 3350)
        shed_drop_mass_liqfrac = 1 / 1.928e6,

        # Wet growth
        wet_growth_hydrometeor_threshold = 1e-6,
        wet_growth_excess_threshold = 1e-10,

        # Refreezing
        refreezing_timescale = 10.0,

        # Deposition nucleation
        ice_nucleation_temperature_threshold = 258.15,
        ice_nucleation_supersaturation_threshold = 0.05,
        maximum_ice_nucleation_concentration = 100e3,
        ice_nucleation_timescale = 10.0,
        ice_nucleation_coefficient = 5.0,
        # Cooper (1986): N = c exp(b (T₀ - T)) with b = 0.304 K⁻¹
        ice_nucleation_temperature_coefficient = 0.304,

        # Immersion freezing
        maximum_immersion_freezing_temperature = 269.15,
        immersion_freezing_coefficient = 0.65,
        # Barklie-Gokhale nucleation coefficient
        immersion_freezing_nucleation_coefficient = 2.0,

        # Rime splintering
        minimum_splintering_temperature = 265.15,
        maximum_splintering_temperature = 270.15,
        splintering_temperature_peak = 268.15,
        # Hallett-Mossop: 3.5e5 splinters/g × 1000 g/kg = 3.5e8 splinters/kg
        # (Fortran: 35.e+4 * 1000. — the ×1000 kg→g conversion is baked in)
        splintering_rate = 3.5e8,
        # Fortran Dinit_HM = 10e-6 m; mass = π/6 × 900 × (10e-6)³ = 4.712e-13 kg
        splintering_crystal_mass = FT(π) / 6 * 900 * (10e-6)^3,
        # Fortran P3 v5.5.0: Dmin_HM = 250e-6 (nCat=1) or 1000e-6 (nCat>1)
        splintering_diameter_threshold = 250e-6,
        # Cloud-riming splintering scale: 1.0 includes (nCat=1), 0.0 excludes (nCat>1).
        # Fortran only includes cloud riming HM for nCat == 1.
        splintering_cloud_riming_scale = 1.0,
        maximum_splintering_liquid_fraction = 0.1,
        # Warm-surface shutoff: nCat=1 uses 282 K, nCat>1 sets Inf (no shutoff).
        maximum_splintering_surface_temperature = 282.0,

        # Initial rain drop
        # Fortran P3 v5.5.0 uses a 25 μm radius; mass follows the configured liquid density.
        initial_rain_drop_mass = 4 * FT(π) / 3 * thermodynamic_constants.liquid.density * (25e-6)^3,

        # Homogeneous freezing
        homogeneous_freezing_temperature = 233.15,
        homogeneous_freezing_timescale = 10.0,

        # Rime densification
        rime_densification_timescale = 10.0,

        # Rain DSD bounds (Fortran P3 v5.5.0 get_rain_dsd2: lammin = (mu_r+1)*inv_Drmax)
        # inv_Drmax = 1/0.002 = 500 [1/m]. Note: table generation uses 200, runtime uses 500.
        minimum_rain_slope = 500.0,    # lamr_min [1/m] ≈ D_max ~2mm (Fortran runtime parity)
        maximum_rain_slope = 100000.0, # lamr_max [1/m] ≈ minimum diameter ~10μm

        # Sink-limiting safety timescale
        sink_limiting_timescale = 10.0, # dt_safety [s]

        # Coupled donor-budget limiter
        coupled_sink_limiting_iterations::Integer = 4,

        # Global ice number limiter (Fortran P3 v5.5.0 impose_max_Ni)
        # Relaxation sink drains nⁱ toward the configured maximum divided by ρ.
        maximum_ice_number_density = 2e6,  # [1/m³], Fortran impose_max_Ni cap

        # Liquid fraction clipping (Milbrandt et al. 2025)
        liquid_fraction_clipping_threshold = 0.01,  # Fortran liqfracsmall
        complete_melting_liquid_fraction = 0.99,

        # M12(c): Tiny-ice threshold for warm pre-processing (Fortran qsmall_dry).
        # Ice with qi ∈ [qsmall, qsmall_dry) at T ≥ T₀ is converted to rain.
        tiny_ice_to_rain_threshold = 1e-12,

        # Tiny-mass instant-evaporation clauses of the saturation adjustment
        tiny_mass_evaporation_threshold = 1e-12,
        subsaturation_evaporation_threshold = 0.001,

        # Liquid fraction mode (Fortran log_LiquidFrac)
        liquid_fraction_active = true,

        # Predicted supersaturation (Fortran log_predictSsat, default .false.)
        predict_supersaturation = false,

        # Deposition/sublimation calibration factors (Fortran clbfact_dep, clbfact_sub)
        calibration_factor_deposition = 1.0,
        calibration_factor_sublimation = 1.0,

        # Numerical floors (see `NumericalFloors`)
        floors = NumericalFloors(FT))

    coupled_sink_limiting_iterations > 0 ||
        throw(ArgumentError("coupled_sink_limiting_iterations must be positive"))

    predict_supersaturation = Bool(predict_supersaturation)

    return ProcessRateParameters{FT, predict_supersaturation}(
        FT(thermodynamic_constants.liquid.density),
        FT(pure_ice_density),
        FT(reference_air_density),
        FT(nucleated_ice_mass),
        FT(activated_droplet_radius),
        FT(activation_supersaturation_threshold),
        FT(freezing_temperature),
        FT(autoconversion_coefficient),
        FT(autoconversion_exponent_cloud),
        FT(autoconversion_exponent_droplet),
        FT(autoconversion_threshold),
        FT(autoconversion_reference_concentration),
        FT(accretion_coefficient),
        FT(accretion_exponent),
        FT(rain_self_collection_coefficient),
        FT(rain_breakup_diameter_threshold),
        FT(rain_breakup_coefficient),
        FT(rain_evaporation_timescale),
        FT(ice_deposition_timescale),
        FT(maximum_aggregation_efficiency),
        FT(minimum_aggregation_efficiency),
        FT(aggregation_timescale),
        FT(aggregation_efficiency_ramp_start_temperature),
        FT(aggregation_efficiency_ramp_end_temperature),
        FT(minimum_aggregation_rime_fraction),
        FT(maximum_aggregation_rime_fraction),
        FT(cloud_ice_collection_efficiency),
        FT(rain_ice_collection_efficiency),
        FT(minimum_rime_density),
        FT(maximum_rime_density),
        FT(rime_impact_coefficient),
        FT(minimum_rime_impact),
        FT(maximum_rime_impact),
        FT(minimum_riming_supercooling),
        FT(unrimed_rime_density),
        FT(shed_drop_mass),
        FT(shed_drop_mass_liqfrac),
        FT(wet_growth_hydrometeor_threshold),
        FT(wet_growth_excess_threshold),
        FT(refreezing_timescale),
        FT(ice_nucleation_temperature_threshold),
        FT(ice_nucleation_supersaturation_threshold),
        FT(maximum_ice_nucleation_concentration),
        FT(ice_nucleation_timescale),
        FT(ice_nucleation_coefficient),
        FT(ice_nucleation_temperature_coefficient),
        FT(maximum_immersion_freezing_temperature),
        FT(immersion_freezing_coefficient),
        FT(immersion_freezing_nucleation_coefficient),
        FT(minimum_splintering_temperature),
        FT(maximum_splintering_temperature),
        FT(splintering_temperature_peak),
        FT(splintering_rate),
        FT(splintering_crystal_mass),
        FT(splintering_diameter_threshold),
        FT(splintering_cloud_riming_scale),
        FT(maximum_splintering_liquid_fraction),
        FT(maximum_splintering_surface_temperature),
        FT(initial_rain_drop_mass),
        FT(homogeneous_freezing_temperature),
        FT(homogeneous_freezing_timescale),
        FT(rime_densification_timescale),
        FT(minimum_rain_slope),
        FT(maximum_rain_slope),
        FT(sink_limiting_timescale),
        Int(coupled_sink_limiting_iterations),
        FT(maximum_ice_number_density),
        FT(liquid_fraction_clipping_threshold),
        FT(complete_melting_liquid_fraction),
        FT(tiny_ice_to_rain_threshold),
        FT(tiny_mass_evaporation_threshold),
        FT(subsaturation_evaporation_threshold),
        Bool(liquid_fraction_active),
        predict_supersaturation,
        FT(calibration_factor_deposition),
        FT(calibration_factor_sublimation),
        NumericalFloors{FT}(floors)
    )
end

# Gate a rate on the predicted-supersaturation switch. Dispatching on the type value
# folds the branch at compile time while preserving a user-facing `Bool` field.
@inline gate_predicted_supersaturation(::ProcessRateParameters{FT, false}, x) where FT = 0 * x
@inline gate_predicted_supersaturation(::ProcessRateParameters{FT, true}, x) where FT = x

@inline predicts_supersaturation(::ProcessRateParameters{FT, PS}) where {FT, PS} = PS

Base.summary(::ProcessRateParameters) = "ProcessRateParameters"

function Base.show(io::IO, p::ProcessRateParameters)
    print(io, summary(p), "(")
    print(io, "T₀=", p.freezing_temperature, "K, ")
    print(io, "ρʷ=", p.liquid_water_density, "kg/m³, ")
    print(io, "Eᶜⁱ=", p.cloud_ice_collection_efficiency, ")")
end
