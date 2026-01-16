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
    reference_air_density :: FT      # ρ₀ [kg/m³] for fall speed correction
    nucleated_ice_mass :: FT         # mᵢ₀ [kg], mass of newly nucleated ice crystal
    freezing_temperature :: FT       # T₀ [K]

    # Rain autoconversion (Khairoutdinov-Kogan 2000)
    autoconversion_coefficient :: FT         # k₁ [s⁻¹]
    autoconversion_exponent_cloud :: FT      # α [-]
    autoconversion_exponent_droplet :: FT    # β [-]
    autoconversion_threshold :: FT           # qᶜˡ threshold [kg/kg]
    autoconversion_reference_concentration :: FT  # Nc reference [1/m³]

    # Rain accretion (Khairoutdinov-Kogan 2000)
    accretion_coefficient :: FT              # k₂ [s⁻¹]
    accretion_exponent :: FT                 # α [-]

    # Rain self-collection (Seifert-Beheng 2001)
    self_collection_coefficient :: FT        # k_rr [-]

    # Evaporation/sublimation timescales
    rain_evaporation_timescale :: FT         # τ_evap [s]
    ice_deposition_timescale :: FT           # τ_dep [s]

    # Melting
    ice_melting_timescale :: FT              # τ_melt [s]

    # Ice aggregation
    aggregation_efficiency_max :: FT         # Eᵢᵢ_max [-]
    aggregation_timescale :: FT              # τ_agg [s]
    aggregation_efficiency_temperature_low :: FT   # T below which E=0.1 [K]
    aggregation_efficiency_temperature_high :: FT  # T above which E=max [K]
    aggregation_reference_concentration :: FT # n_ref [1/kg]

    # Cloud riming
    cloud_ice_collection_efficiency :: FT    # Eᶜⁱ [-]
    cloud_riming_timescale :: FT             # τ_rim [s]

    # Rain riming
    rain_ice_collection_efficiency :: FT     # Eʳⁱ [-]
    rain_riming_timescale :: FT              # τ_rim [s]

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

    # Immersion freezing (Bigg 1953)
    immersion_freezing_temperature_max :: FT # T_max [K]
    immersion_freezing_coefficient :: FT     # aimm [-]
    immersion_freezing_timescale_cloud :: FT # base timescale for cloud [s]
    immersion_freezing_timescale_rain :: FT  # base timescale for rain [s]

    # Rime splintering (Hallett-Mossop)
    splintering_temperature_low :: FT        # T_low [K]
    splintering_temperature_high :: FT       # T_high [K]
    splintering_temperature_peak :: FT       # T_peak [K]
    splintering_temperature_width :: FT      # width [K]
    splintering_rate :: FT                   # splinters per kg rime

    # Rain terminal velocity (power law v = a D^b)
    rain_fall_speed_coefficient :: FT        # a [m^(1-b)/s]
    rain_fall_speed_exponent :: FT           # b [-]
    rain_diameter_min :: FT                  # D_min [m]
    rain_diameter_max :: FT                  # D_max [m]
    rain_velocity_min :: FT                  # v_min [m/s]
    rain_velocity_max :: FT                  # v_max [m/s]

    # Ice terminal velocity
    ice_fall_speed_coefficient_unrimed :: FT # a for aggregates
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

    # Ratio factors for weighted velocities
    velocity_ratio_number_to_mass :: FT      # Vₙ/Vₘ
    velocity_ratio_reflectivity_to_mass :: FT  # Vᵤ/Vₘ

    # Initial rain drop mass (for autoconversion number tendency)
    initial_rain_drop_mass :: FT             # m_rain_init [kg]
end

"""
$(TYPEDSIGNATURES)

Construct process rate parameters with default values from P3 literature.

These parameters control the rates of all microphysical processes:
autoconversion, accretion, aggregation, riming, melting, evaporation,
deposition, nucleation, and sedimentation.

# Default Sources

- Autoconversion/accretion: Khairoutdinov and Kogan (2000)
- Self-collection: Seifert and Beheng (2001)
- Aggregation: Morrison and Milbrandt (2015)
- Nucleation: Cooper (1986)
- Freezing: Bigg (1953)
- Splintering: Hallett and Mossop (1974)
- Fall speeds: Mitchell (1996), Seifert and Beheng (2006)

# Example

```julia
params = ProcessRateParameters(Float64)
```

All parameters are keyword arguments with physically-based defaults.
"""
function ProcessRateParameters(FT::Type{<:AbstractFloat} = Float64;
        # Physical constants
        liquid_water_density = 1000,
        pure_ice_density = 917,
        reference_air_density = 1.225,
        nucleated_ice_mass = 1e-12,
        freezing_temperature = 273.15,

        # Rain autoconversion
        autoconversion_coefficient = 2.47e-2,
        autoconversion_exponent_cloud = 2.47,
        autoconversion_exponent_droplet = -1.79,
        autoconversion_threshold = 1e-4,
        autoconversion_reference_concentration = 1e8,

        # Rain accretion
        accretion_coefficient = 67.0,
        accretion_exponent = 1.15,

        # Rain self-collection
        self_collection_coefficient = 4.33,

        # Timescales
        rain_evaporation_timescale = 10.0,
        ice_deposition_timescale = 10.0,
        ice_melting_timescale = 60.0,

        # Ice aggregation
        aggregation_efficiency_max = 1.0,
        aggregation_timescale = 600.0,
        aggregation_efficiency_temperature_low = 253.15,
        aggregation_efficiency_temperature_high = 268.15,
        aggregation_reference_concentration = 1e4,

        # Cloud riming
        cloud_ice_collection_efficiency = 1.0,
        cloud_riming_timescale = 300.0,

        # Rain riming
        rain_ice_collection_efficiency = 1.0,
        rain_riming_timescale = 200.0,

        # Rime density
        minimum_rime_density = 50.0,
        maximum_rime_density = 900.0,

        # Shedding
        shedding_timescale = 60.0,
        maximum_liquid_fraction = 0.3,
        shed_drop_mass = 5.2e-7,

        # Refreezing
        refreezing_timescale = 30.0,

        # Deposition nucleation
        nucleation_temperature_threshold = 258.15,
        nucleation_supersaturation_threshold = 0.05,
        nucleation_maximum_concentration = 100e3,
        nucleation_timescale = 60.0,

        # Immersion freezing
        immersion_freezing_temperature_max = 269.15,
        immersion_freezing_coefficient = 0.66,
        immersion_freezing_timescale_cloud = 1000.0,
        immersion_freezing_timescale_rain = 300.0,

        # Rime splintering
        splintering_temperature_low = 265.15,
        splintering_temperature_high = 270.15,
        splintering_temperature_peak = 268.15,
        splintering_temperature_width = 2.5,
        splintering_rate = 3.5e8,

        # Rain terminal velocity
        rain_fall_speed_coefficient = 842.0,
        rain_fall_speed_exponent = 0.8,
        rain_diameter_min = 1e-4,
        rain_diameter_max = 5e-3,
        rain_velocity_min = 0.1,
        rain_velocity_max = 15.0,

        # Ice terminal velocity
        ice_fall_speed_coefficient_unrimed = 11.7,
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

        # Velocity ratios
        velocity_ratio_number_to_mass = 0.6,
        velocity_ratio_reflectivity_to_mass = 1.2,

        # Initial rain drop
        initial_rain_drop_mass = 5e-10)

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
        FT(rain_evaporation_timescale),
        FT(ice_deposition_timescale),
        FT(ice_melting_timescale),
        FT(aggregation_efficiency_max),
        FT(aggregation_timescale),
        FT(aggregation_efficiency_temperature_low),
        FT(aggregation_efficiency_temperature_high),
        FT(aggregation_reference_concentration),
        FT(cloud_ice_collection_efficiency),
        FT(cloud_riming_timescale),
        FT(rain_ice_collection_efficiency),
        FT(rain_riming_timescale),
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
        FT(immersion_freezing_temperature_max),
        FT(immersion_freezing_coefficient),
        FT(immersion_freezing_timescale_cloud),
        FT(immersion_freezing_timescale_rain),
        FT(splintering_temperature_low),
        FT(splintering_temperature_high),
        FT(splintering_temperature_peak),
        FT(splintering_temperature_width),
        FT(splintering_rate),
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
        FT(velocity_ratio_number_to_mass),
        FT(velocity_ratio_reflectivity_to_mass),
        FT(initial_rain_drop_mass)
    )
end

Base.summary(::ProcessRateParameters) = "ProcessRateParameters"

function Base.show(io::IO, p::ProcessRateParameters)
    print(io, summary(p), "(")
    print(io, "T₀=", p.freezing_temperature, "K, ")
    print(io, "ρʷ=", p.liquid_water_density, "kg/m³, ")
    print(io, "τ_melt=", p.ice_melting_timescale, "s)")
end
