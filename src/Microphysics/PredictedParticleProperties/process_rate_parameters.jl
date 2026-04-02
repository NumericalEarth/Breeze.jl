#####
##### Process Rate Parameters
#####
##### Container for all P3 microphysical process rate parameters.
##### These parameters control timescales, efficiencies, and thresholds.
#####

export ProcessRateParameters

"""
    ProcessRateParameters

Parameters for P3 microphysical process rates.
See [`ProcessRateParameters()`](@ref) constructor for usage.
"""
struct ProcessRateParameters{FT}
    # Physical constants
    liquid_water_density :: FT       # ρʷ [kg/m³]
    pure_ice_density :: FT           # ρⁱ [kg/m³]
    reference_air_density :: FT      # ρ₀ [kg/m³] for rain fall speed correction (Fortran rhosur)
    nucleated_ice_mass :: FT         # mᵢ₀ [kg], mass of newly nucleated ice crystal
    freezing_temperature :: FT       # T₀ [K]

    # Rain autoconversion (Khairoutdinov-Kogan 2000)
    autoconversion_coefficient :: FT         # k₁ = 1350 × (Nc_ref_cm)^β, see KK2000 Eq. 29
    autoconversion_exponent_cloud :: FT      # α [-]
    autoconversion_exponent_droplet :: FT    # β [-]
    autoconversion_threshold :: FT           # qᶜˡ threshold [kg/kg]
    autoconversion_reference_concentration :: FT  # Nc reference [1/m³]

    # Rain accretion (Khairoutdinov-Kogan 2000)
    accretion_coefficient :: FT              # k₂ [s⁻¹]
    accretion_exponent :: FT                 # α [-]

    # Rain self-collection and breakup (Seifert-Beheng 2001; Fortran P3 v5.5.0 breakup)
    self_collection_coefficient :: FT        # k_rr [-]
    rain_breakup_diameter_threshold :: FT    # D_th threshold for breakup [m] (1/λ_r convention)
    rain_breakup_coefficient :: FT           # κ_br [1/m]

    # Evaporation/sublimation timescales
    rain_evaporation_timescale :: FT         # τ_evap [s]
    ice_deposition_timescale :: FT           # τ_dep [s]

    # Ice aggregation
    aggregation_efficiency_max :: FT         # Eᵢᵢ_max [-]
    aggregation_timescale :: FT              # τ_agg [s]
    aggregation_efficiency_temperature_low :: FT   # T below which E=0.001 [K]
    aggregation_efficiency_temperature_high :: FT  # T above which E=max [K]
    aggregation_reference_concentration :: FT # n_ref [1/kg]

    # Cloud riming
    cloud_ice_collection_efficiency :: FT    # Eᶜⁱ [-]

    # Rain riming
    rain_ice_collection_efficiency :: FT     # Eʳⁱ [-]

    # Rime density bounds
    minimum_rime_density :: FT               # ρ_rim_min [kg/m³]
    maximum_rime_density :: FT               # ρ_rim_max [kg/m³]

    # Shedding
    shedding_timescale :: FT                 # τ_shed [s]
    maximum_liquid_fraction :: FT            # qʷⁱ_max_frac [-]
    shed_drop_mass :: FT                     # m_shed [kg]

    # Refreezing
    refreezing_timescale :: FT               # τ_frz [s]

    # Deposition nucleation (Cooper 1986)
    nucleation_temperature_threshold :: FT   # T below which nucleation occurs [K]
    nucleation_supersaturation_threshold :: FT  # Sⁱ threshold [-]
    nucleation_maximum_concentration :: FT   # N_max [1/m³]
    nucleation_timescale :: FT               # τ_nuc [s]
    nucleation_coefficient :: FT             # Cooper (1986) prefactor [1/m³] (default 5.0)

    # Immersion freezing (Barklie-Gokhale 1959)
    immersion_freezing_temperature_max :: FT # T_max [K]
    immersion_freezing_coefficient :: FT     # aimm [-]

    # Rime splintering (Hallett-Mossop)
    splintering_temperature_low :: FT        # T_low [K]
    splintering_temperature_high :: FT       # T_high [K]
    splintering_temperature_peak :: FT       # T_peak [K]
    splintering_rate :: FT                   # splinters per kg rime
    splintering_diameter_threshold :: FT     # D_min [m] for HM splintering
    splintering_liquid_fraction_max :: FT    # Fˡ max for HM splintering
    splintering_surface_temperature_max :: FT # warm-surface shutoff [K]

    # Rain terminal velocity (power law v = a D^b)
    rain_fall_speed_coefficient :: FT        # a [m^(1-b)/s]
    rain_fall_speed_exponent :: FT           # b [-]
    rain_diameter_min :: FT                  # D_min [m]
    rain_diameter_max :: FT                  # D_max [m]
    rain_velocity_min :: FT                  # v_min [m/s]
    rain_velocity_max :: FT                  # v_max [m/s]

    # Ice terminal velocity
    ice_fall_speed_coefficient_unrimed :: FT # a for aggregates [Mitchell 1996]
    ice_fall_speed_exponent_unrimed :: FT    # b for aggregates
    ice_fall_speed_coefficient_rimed :: FT   # a for graupel
    ice_fall_speed_exponent_rimed :: FT      # b for graupel
    ice_small_particle_coefficient :: FT     # Stokes regime coefficient
    ice_diameter_threshold :: FT             # D_threshold [m]
    ice_diameter_min :: FT                   # D_min [m]
    ice_diameter_max :: FT                   # D_max [m]
    ice_velocity_min :: FT                   # v_min [m/s]
    ice_velocity_max :: FT                   # v_max [m/s]
    ice_effective_density_unrimed :: FT      # ρ_eff for aggregates [kg/m³]

    # Ice projected area (Mitchell 1996): A = γ D^σ for aggregates
    ice_projected_area_coefficient :: FT     # γ [m^(2-σ)]
    ice_projected_area_exponent :: FT        # σ [-]

    # Ratio factors for weighted velocities
    velocity_ratio_number_to_mass :: FT      # Vₙ/Vₘ
    velocity_ratio_reflectivity_to_mass :: FT  # Vᵤ/Vₘ

    # Initial rain drop mass (for autoconversion number tendency)
    initial_rain_drop_mass :: FT             # m_rain_init [kg]

    # Immersion freezing nucleation coefficient (Barklie-Gokhale 1959)
    immersion_freezing_nucleation_coefficient :: FT  # bimm [m⁻³s⁻¹]

    # PSD correction factors: account for the PSD-integrated rate being
    # larger than the mean-mass value due to the nonlinear dependence on
    # particle size. These factors bridge the gap between the mean-mass
    # approximation and full PSD integration (lookup tables).
    riming_psd_correction :: FT              # For cloud and rain riming [-]
    freezing_cloud_psd_correction :: FT      # For cloud immersion freezing [-]
    freezing_rain_psd_correction :: FT       # For rain immersion freezing [-]

    # Homogeneous freezing (Koop et al. 2000)
    homogeneous_freezing_temperature :: FT   # T < threshold: all cloud/rain freezes [K]
    homogeneous_freezing_timescale :: FT     # τ_hom [s], effective instantaneous
    minimum_cloud_drop_mass :: FT           # mass-number consistency cap for N_hom [kg]

    # Rain size distribution bounds (Fortran P3 v5.5.0: lamr_min, lamr_max)
    rain_lambda_min :: FT                   # λ_r minimum [1/m]
    rain_lambda_max :: FT                   # λ_r maximum [1/m]

    # Sink-limiting safety timescale [s]
    # If total sinks for any species × dt_safety exceed available mass,
    # all sink rates for that species are rescaled proportionally.
    sink_limiting_timescale :: FT            # dt_safety [s]

    # Global ice number limiter (Fortran P3 v5.5.0 impose_max_Ni)
    # Applied as a relaxation sink whenever nⁱ × ρ > N_max.
    maximum_ice_number_density :: FT         # Nᵢ_max [1/m³]

    # Liquid fraction clipping threshold (Milbrandt et al. 2025)
    # Fl < this: instantly freeze all qwi to rime; Fl > (1 - this): fully melt to rain.
    # Implemented as a relaxation drain over refreezing_timescale.
    liquid_fraction_small :: FT              # Fortran liqfracsmall [-]

    # Liquid fraction mode (Fortran log_LiquidFrac).
    # When true: wet growth rime densification is suppressed (liquid tracked
    # explicitly in qʷⁱ), and melt-densification is skipped.
    liquid_fraction_active :: Bool

    # Deposition/sublimation calibration factors (Fortran P3 v5.5.0 clbfact_dep, clbfact_sub).
    # Ad hoc multipliers to increase or decrease deposition and/or sublimation rates.
    # The representation of ice capacitances is highly simplified and the appropriate
    # values in the diffusional growth equation are uncertain (Fortran comment, line 3721).
    calibration_factor_deposition :: FT
    calibration_factor_sublimation :: FT

    # Fortran PSD-based partitioning always sends some meltwater to rain
    # (from small particles that fully melt). This floor approximates that
    # effect without requiring size-threshold table integrals (f1pr24-f1pr27).
    minimum_complete_melting_fraction :: FT
end

"""
$(TYPEDSIGNATURES)

Construct process rate parameters with default values from P3 literature.

These parameters control the rates of all microphysical processes:
autoconversion, accretion, aggregation, riming, melting, evaporation,
deposition, nucleation, and sedimentation.

# Default Sources

- Autoconversion/accretion: Khairoutdinov and Kogan (2000)
- Self-collection/breakup: Seifert and Beheng (2001, 2006)
- Aggregation: Morrison and Milbrandt (2015)
- Nucleation: Cooper (1986)
- Freezing: Barklie and Gokhale (1959)
- Splintering: Hallett and Mossop (1974)
- Fall speeds: Mitchell (1996), Seifert and Beheng (2006)

# Example

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: ProcessRateParameters
params = ProcessRateParameters(Float64)
typeof(params)

# output
ProcessRateParameters{Float64}
```

All parameters are keyword arguments with physically-based defaults.
"""
function ProcessRateParameters(FT::Type{<:AbstractFloat} = Float64;
        # Physical constants
        liquid_water_density = 1000,
        pure_ice_density = 917,
        # Reference density for rain fall speed correction (P=1000 hPa, T=0°C)
        # computed from Breeze's internal dry-air gas constant.
        reference_air_density = 100000 / (dry_air_gas_constant(ThermodynamicConstants()) * 273.15),
        nucleated_ice_mass = π/6 * 900 * (10e-6)^3,  # Fortran Dinit_HM = 10e-6 m, ρ=900 kg/m³ [kg]
        freezing_temperature = 273.15,

        # Rain autoconversion
        # KK2000 Eq. 29: dqʳ/dt = 1350 qᶜˡ^2.47 Nᶜ^(-1.79) with Nᶜ in cm⁻³.
        # Rescaled for (Nᶜ/Nᶜ_ref)^β with Nᶜ_ref = 1e8 m⁻³ = 100 cm⁻³:
        # k₁ = 1350 × (100)^(-1.79) ≈ 0.355
        autoconversion_coefficient = 1350 * 100.0^(-1.79),
        autoconversion_exponent_cloud = 2.47,
        autoconversion_exponent_droplet = -1.79,
        autoconversion_threshold = 1e-8,  # Fortran P3 v5.5.0 qsmall
        autoconversion_reference_concentration = 1e8,

        # Rain accretion
        accretion_coefficient = 67.0,
        accretion_exponent = 1.15,

        # Rain self-collection and breakup
        self_collection_coefficient = 5.78,
        rain_breakup_diameter_threshold = 280e-6,  # 280 μm: Fortran P3 breakup threshold (1/λ_r convention)
        rain_breakup_coefficient = 2300.0,

        # Timescales
        rain_evaporation_timescale = 10.0,
        ice_deposition_timescale = 10.0,

        # Ice aggregation
        aggregation_efficiency_max = 0.3,
        aggregation_timescale = 600.0,
        aggregation_efficiency_temperature_low = 253.15,
        aggregation_efficiency_temperature_high = 273.15,
        aggregation_reference_concentration = 1e4,

        # Cloud riming
        cloud_ice_collection_efficiency = 0.5,

        # Rain riming
        rain_ice_collection_efficiency = 1.0,

        # Rime density
        minimum_rime_density = 50.0,
        maximum_rime_density = 900.0,

        # Shedding
        shedding_timescale = 60.0,
        maximum_liquid_fraction = 0.3,
        shed_drop_mass = 1 / 1.928e6,

        # Refreezing
        refreezing_timescale = 30.0,

        # Deposition nucleation
        nucleation_temperature_threshold = 258.15,
        nucleation_supersaturation_threshold = 0.05,
        nucleation_maximum_concentration = 100e3,
        nucleation_timescale = 60.0,
        nucleation_coefficient = 5.0,

        # Immersion freezing
        immersion_freezing_temperature_max = 269.15,
        immersion_freezing_coefficient = 0.65,

        # Rime splintering
        splintering_temperature_low = 265.15,
        splintering_temperature_high = 270.15,
        splintering_temperature_peak = 268.15,
        splintering_rate = 3.5e8,
        splintering_diameter_threshold = 250e-6,
        splintering_liquid_fraction_max = 0.1,
        splintering_surface_temperature_max = 282.0,

        # Rain terminal velocity
        rain_fall_speed_coefficient = 841.99667,
        rain_fall_speed_exponent = 0.8,
        rain_diameter_min = 1e-4,
        rain_diameter_max = 5e-3,
        rain_velocity_min = 0.1,
        rain_velocity_max = 15.0,

        # Ice terminal velocity
        ice_fall_speed_coefficient_unrimed = 11.72,
        ice_fall_speed_exponent_unrimed = 0.41,
        ice_fall_speed_coefficient_rimed = 19.3,
        ice_fall_speed_exponent_rimed = 0.37,
        ice_small_particle_coefficient = 700.0,
        ice_diameter_threshold = 100e-6,
        ice_diameter_min = 1e-5,
        ice_diameter_max = 0.02,
        ice_velocity_min = 0.01,
        ice_velocity_max = 8.0,
        ice_effective_density_unrimed = 100.0,

        # Ice projected area (Mitchell 1996): A = γ D^σ
        # CGS value 0.2285 converted to MKS: 0.2285 × 100^(1.88 - 2) ≈ 0.1315
        ice_projected_area_coefficient = 0.2285 * 100.0^(1.88 - 2.0),
        ice_projected_area_exponent = 1.88,

        # Velocity ratios
        velocity_ratio_number_to_mass = 0.6,
        velocity_ratio_reflectivity_to_mass = 1.2,

        # Initial rain drop
        initial_rain_drop_mass = 4π/3 * 1000 * (25e-6)^3,  # Fortran P3 v5.5.0: 25 μm radius drop [kg]

        # Barklie-Gokhale nucleation coefficient
        immersion_freezing_nucleation_coefficient = 2.0,

        # PSD correction factors: account for the PSD-integrated rate being
        # larger than the mean-mass value due to the nonlinear (volumetric)
        # dependence on drop size. For spherical drops with a gamma PSD N'(D) = N₀ D^μ exp(-λD)
        # the analytical correction is C(μ) = Γ(μ+7)Γ(μ+1) / Γ(μ+4)²
        # (see psd_correction_spherical_volume).
        # Cloud drops: μ ≈ 2.3 → C ≈ 5.08; rain drops: μ = 0 → C = 20.0 (exact)
        riming_psd_correction = 2.0,  # Tunable: keep as empirical parameter
        freezing_cloud_psd_correction = psd_correction_spherical_volume(2.3),
        freezing_rain_psd_correction = psd_correction_spherical_volume(0.0),

        # Homogeneous freezing
        homogeneous_freezing_temperature = 233.15,
        homogeneous_freezing_timescale = 1.0,
        # Mass-number consistency cap: at most one particle per minimum-size droplet
        # (≈ 6 μm radius cloud droplet → m ≈ 4/3 π ρ_w r³ ≈ 9e-13 kg; use 1e-12 kg)
        minimum_cloud_drop_mass = 1e-12,

        # Rain DSD bounds (Fortran P3 v5.5.0: 1/(dlamr) intervals)
        rain_lambda_min = 500.0,    # lamr_min [1/m] ≈ D_max ~2mm
        rain_lambda_max = 100000.0, # lamr_max [1/m] ≈ D_min ~10μm

        # Sink-limiting safety timescale
        sink_limiting_timescale = 1.0, # dt_safety [s]

        # Global ice number limiter (Fortran P3 v5.5.0 impose_max_Ni)
        # Relaxation sink drains nⁱ toward N_max/ρ when nⁱ × ρ > N_max.
        maximum_ice_number_density = 2e6,  # [1/m³], Fortran impose_max_Ni cap

        # Liquid fraction clipping (Milbrandt et al. 2025)
        liquid_fraction_small = 0.01,  # Fortran liqfracsmall

        # Liquid fraction mode (Fortran log_LiquidFrac)
        liquid_fraction_active = true,

        # Deposition/sublimation calibration factors (Fortran clbfact_dep, clbfact_sub)
        calibration_factor_deposition = 1.0,
        calibration_factor_sublimation = 1.0,

        # Fortran PSD-based partitioning always sends some meltwater to rain
        # (from small particles that fully melt). This floor approximates that
        # effect without requiring size-threshold table integrals (f1pr24-f1pr27).
        minimum_complete_melting_fraction = 0.2)

    return ProcessRateParameters(
        FT(liquid_water_density),
        FT(pure_ice_density),
        FT(reference_air_density),
        FT(nucleated_ice_mass),
        FT(freezing_temperature),
        FT(autoconversion_coefficient),
        FT(autoconversion_exponent_cloud),
        FT(autoconversion_exponent_droplet),
        FT(autoconversion_threshold),
        FT(autoconversion_reference_concentration),
        FT(accretion_coefficient),
        FT(accretion_exponent),
        FT(self_collection_coefficient),
        FT(rain_breakup_diameter_threshold),
        FT(rain_breakup_coefficient),
        FT(rain_evaporation_timescale),
        FT(ice_deposition_timescale),
        FT(aggregation_efficiency_max),
        FT(aggregation_timescale),
        FT(aggregation_efficiency_temperature_low),
        FT(aggregation_efficiency_temperature_high),
        FT(aggregation_reference_concentration),
        FT(cloud_ice_collection_efficiency),
        FT(rain_ice_collection_efficiency),
        FT(minimum_rime_density),
        FT(maximum_rime_density),
        FT(shedding_timescale),
        FT(maximum_liquid_fraction),
        FT(shed_drop_mass),
        FT(refreezing_timescale),
        FT(nucleation_temperature_threshold),
        FT(nucleation_supersaturation_threshold),
        FT(nucleation_maximum_concentration),
        FT(nucleation_timescale),
        FT(nucleation_coefficient),
        FT(immersion_freezing_temperature_max),
        FT(immersion_freezing_coefficient),
        FT(splintering_temperature_low),
        FT(splintering_temperature_high),
        FT(splintering_temperature_peak),
        FT(splintering_rate),
        FT(splintering_diameter_threshold),
        FT(splintering_liquid_fraction_max),
        FT(splintering_surface_temperature_max),
        FT(rain_fall_speed_coefficient),
        FT(rain_fall_speed_exponent),
        FT(rain_diameter_min),
        FT(rain_diameter_max),
        FT(rain_velocity_min),
        FT(rain_velocity_max),
        FT(ice_fall_speed_coefficient_unrimed),
        FT(ice_fall_speed_exponent_unrimed),
        FT(ice_fall_speed_coefficient_rimed),
        FT(ice_fall_speed_exponent_rimed),
        FT(ice_small_particle_coefficient),
        FT(ice_diameter_threshold),
        FT(ice_diameter_min),
        FT(ice_diameter_max),
        FT(ice_velocity_min),
        FT(ice_velocity_max),
        FT(ice_effective_density_unrimed),
        FT(ice_projected_area_coefficient),
        FT(ice_projected_area_exponent),
        FT(velocity_ratio_number_to_mass),
        FT(velocity_ratio_reflectivity_to_mass),
        FT(initial_rain_drop_mass),
        FT(immersion_freezing_nucleation_coefficient),
        FT(riming_psd_correction),
        FT(freezing_cloud_psd_correction),
        FT(freezing_rain_psd_correction),
        FT(homogeneous_freezing_temperature),
        FT(homogeneous_freezing_timescale),
        FT(minimum_cloud_drop_mass),
        FT(rain_lambda_min),
        FT(rain_lambda_max),
        FT(sink_limiting_timescale),
        FT(maximum_ice_number_density),
        FT(liquid_fraction_small),
        Bool(liquid_fraction_active),
        FT(calibration_factor_deposition),
        FT(calibration_factor_sublimation),
        FT(minimum_complete_melting_fraction)
    )
end

Base.summary(::ProcessRateParameters) = "ProcessRateParameters"

function Base.show(io::IO, p::ProcessRateParameters)
    print(io, summary(p), "(")
    print(io, "T₀=", p.freezing_temperature, "K, ")
    print(io, "ρʷ=", p.liquid_water_density, "kg/m³, ")
    print(io, "Eᶜⁱ=", p.cloud_ice_collection_efficiency, ")")
end
