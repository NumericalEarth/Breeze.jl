#####
##### Microphysics helpers. Case constructors take a Breeze microphysics object directly
##### (e.g. `P3Microphysics(; cloud, aerosol)`); the helpers here build the aerosol parts the
##### ENA protocols prescribe and return Breeze types. Precision follows
##### `Oceananigans.defaults.FloatType`.
#####

"""
    lasso_aerosol(; setting=:aer2, reference_density, maximum_supersaturation=nothing,
                    aerosol_density=1790, molecular_weight_aerosol=0.115, vant_hoff_factor=3,
                    mass_fraction_soluble=1)

The two lognormal aerosol modes of the LASSO-ENA spectral-bin configuration as a Breeze
`AerosolActivation` with a prognostic (depleting) reservoir, as the SBM's. LASSO quotes
number *concentrations* per cm³:

| setting | mode 1 (r = 0.018 μm, σ = 1.53) | mode 2 (r = 0.066 μm, σ = 1.78) |
|---------|---------------------------------|---------------------------------|
| aer1    | 138 cm⁻³                        | 140.5 cm⁻³                      |
| aer2    | 276 cm⁻³                        | 281 cm⁻³                        |
| aer3    | 552 cm⁻³                        | 562 cm⁻³                        |

P3 takes the number *per unit mass* of air, `nᵃ = 10⁶ N / ρ` [kg⁻¹]. The LASSO HUJI-SBM
initializes its CCN with a constant *mixing ratio* with height
(`MICRO_HUJISBM/microphysics.f90`: `FCCN0 = FCCNR_mp * rhocgs(k)/rhocgs(1)`), so the
conversion uses the first-level reference density (see
[`first_level_reference_density`](@ref)): `number_mixing_ratio = 1e6 N / ρ₁`, uniform in z.
The chemistry is the HUJI-SBM's (aerosol density 1790 kg m⁻³, molecular weight
0.115 kg mol⁻¹, van 't Hoff factor 3, fully soluble: `micro_prm.f90` has no insoluble
fraction, so Breeze's default `mass_fraction_soluble = 0.9` is overridden).

`maximum_supersaturation` emulates the SBM's activation cap (`ss_max = 0.003` in
`MICRO_HUJISBM/microphysics.f90`: no particle activates above 0.3 %): each mode's number is
reduced to the fraction of its particles with critical supersaturation below the cap
([`activated_fraction`](@ref)). Breeze's Morrison–Grabowski activation itself has no cap
(see `docs/cases/ena_aerosol_audit.md`). The cap is an approximation that keeps the
lognormal shape and scales the number; it is not the SBM.
"""
function lasso_aerosol(; setting = :aer2, reference_density, maximum_supersaturation = nothing,
                         aerosol_density = 1790, molecular_weight_aerosol = 0.115, vant_hoff_factor = 3,
                         mass_fraction_soluble = 1)
    modes = lasso_aerosol_modes(; setting, reference_density, maximum_supersaturation,
                                aerosol_density, molecular_weight_aerosol, vant_hoff_factor, mass_fraction_soluble)
    return AerosolActivation(modes...; prognostic = true)
end

function lasso_aerosol_modes(; setting, reference_density, maximum_supersaturation, chemistry...)
    N₁, N₂ = setting === :aer1 ? (138.0, 140.5) :
             setting === :aer2 ? (276.0, 281.0) :
             setting === :aer3 ? (552.0, 562.0) :
             throw(ArgumentError("unknown LASSO aerosol setting $setting (aer1, aer2, aer3)"))
    geometry = ((; mean_radius = 0.018e-6, geometric_std = 1.53), (; mean_radius = 0.066e-6, geometric_std = 1.78))
    fractions = isnothing(maximum_supersaturation) ? (1.0, 1.0) :
        Tuple(activated_fraction(g.mean_radius, g.geometric_std, maximum_supersaturation; chemistry...) for g in geometry)
    n₁ = fractions[1] * 1e6 * N₁ / reference_density
    n₂ = fractions[2] * 1e6 * N₂ / reference_density
    return (AerosolMode(; number_mixing_ratio = n₁, geometry[1]..., chemistry...),
            AerosolMode(; number_mixing_ratio = n₂, geometry[2]..., chemistry...))
end

function sounding_reference_state(grid, profiles::SoundingProfiles, constants; extension = nothing)
    ζ(z) = isnothing(extension) || !(extension[1] ≤ z < extension[2]) ? z : extension[1]
    return ReferenceState(grid, constants;
                          base_pressure = profiles.surface_pressure,
                          potential_temperature = z -> profiles(:θ, ζ(z)),
                          vapor_mass_fraction = z -> profiles(:qᵗ, ζ(z)))
end

is_p3(microphysics) = microphysics isa PredictedParticlePropertiesMicrophysics

"""
    has_aerosol_reservoir(microphysics)

`true` for P3 with a prognostic (depleting) aerosol reservoir `ρnᵃ`.
"""
has_aerosol_reservoir(microphysics) =
    is_p3(microphysics) && !isnothing(microphysics.aerosol) && has_prognostic_aerosol(microphysics.aerosol)


function activated_fraction(mean_radius, geometric_std, S; T=285.0, aerosol_density=1790, molecular_weight_aerosol=0.115,
                            vant_hoff_factor=3, mass_fraction_soluble=1, osmotic_potential=1)
    mode = AerosolMode(Float64; number_mixing_ratio=1.0, mean_radius, geometric_std, aerosol_density,
                       molecular_weight_aerosol, vant_hoff_factor, mass_fraction_soluble, osmotic_potential)
    return activated_number(mode, AerosolActivation(mode), Float64(T), Float64(S))
end


specific_prognostic_names(microphysics) =
    Tuple(Symbol(string(ρname)[nextind(string(ρname), 1):end])
          for ρname in Breeze.AtmosphereModels.prognostic_field_names(microphysics))
