"""
The Single-Moment Microphysics model.

Tracers
=======

* Nitrates: NO₃ (mmol N/m³)
* Ammonia: NH₄ (mmol N/m³)
* Phytoplankton: P (mmol N/m³)
* Zooplankton: Z (mmol N/m³)
* Small (slow sinking) particulate organic matter: sPOM (mmol N/m³)
* Large (fast sinking) particulate organic matter: bPOM (mmol N/m³)
* Dissolved organic matter: DOM (mmol N/m³)

Optional tracers
================

Carbonate chemistry
* Dissolved inorganic carbon: DIC (mmol C/m³)
* Alkalinity: Alk (meq/m³)

Oxygen chemistry
* Oxygen: O₂ (mmol O₂/m³)

Variable redfield

* Small (slow sinking) particulate organic matter carbon content: sPOC (mmol C/m³)
* Large (fast sinking) particulate organic matter carbon content: bPOC (mmol C/m³)
* Dissolved organic matter carbon content: DOC (mmol C/m³)
* When this option is enabled then the usual sPOM and bPOM change to sPON and bPON as they explicitly represent the nitrogen contained in the particulate matter

Required submodels
==================

* Photosynthetically available radiation: PAR (W/m²)

For optional tracers:
* Temperature: T (ᵒC)
* Salinity: S (‰)
"""
module Microphysics1M

export Microphysics1M

using Oceananigans.Units
using Oceananigans.Fields: Field, TracerFields, CenterField, ZeroField
using Oceananigans.Grids: AbstractGrid

#using OceanBioME.Light: TwoBandPhotosyntheticallyActiveRadiation, default_surface_PAR
#using OceanBioME: setup_velocity_fields, show_sinking_velocities, Biogeochemistry, ScaleNegativeTracers
#using OceanBioME.BoxModels: BoxModel

import Breeze.Microphysics: AbstractMicrophysics, 
                            required_microphysics_tracers,
                            required_microphysics_auxiliary_fields,
                            microphysics_drift_velocity

#import OceanBioME: redfield, conserved_tracers, maximum_sinking_velocity, chlorophyll

import Adapt: adapt_structure, adapt
import Base: show, summary

struct Microphysics1M{FT, B, W} <: AbstractMicrophysics
    # phytoplankton_preference :: FT
    # maximum_grazing_rate :: FT
    # grazing_half_saturation :: FT
    # light_half_saturation :: FT
    # nitrate_ammonia_inhibition :: FT
    # nitrate_half_saturation :: FT
    # ammonia_half_saturation :: FT
    # maximum_phytoplankton_growthrate :: FT
    # zooplankton_assimilation_fraction :: FT
    # zooplankton_mortality :: FT
    # zooplankton_excretion_rate :: FT
    # phytoplankton_mortality :: FT
    # small_detritus_remineralisation_rate :: FT
    # large_detritus_remineralisation_rate :: FT
    # phytoplankton_exudation_fraction :: FT
    # nitrification_rate :: FT
    # ammonia_fraction_of_exudate :: FT
    # ammonia_fraction_of_excriment :: FT
    # ammonia_fraction_of_detritus :: FT
    # phytoplankton_redfield :: FT
    # organic_redfield :: FT
    # phytoplankton_chlorophyll_ratio :: FT
    # organic_carbon_calcate_ratio :: FT
    # respiration_oxygen_nitrogen_ratio :: FT
    # nitrification_oxygen_nitrogen_ratio :: FT
    # slow_sinking_mortality_fraction :: FT
    # fast_sinking_mortality_fraction :: FT
    # dissolved_organic_breakdown_rate :: FT
    # zooplankton_calcite_dissolution :: FT

    # optionals :: B

    # sinking_velocities :: W

    function LOBSTER(phytoplankton_preference::FT,
                     maximum_grazing_rate::FT,
                     grazing_half_saturation::FT,
                     light_half_saturation::FT,
                     nitrate_ammonia_inhibition::FT,
                     nitrate_half_saturation::FT,
                     ammonia_half_saturation::FT,
                     maximum_phytoplankton_growthrate::FT,
                     zooplankton_assimilation_fraction::FT,
                     zooplankton_mortality::FT,
                     zooplankton_excretion_rate::FT,
                     phytoplankton_mortality::FT,
                     small_detritus_remineralisation_rate::FT,
                     large_detritus_remineralisation_rate::FT,
                     phytoplankton_exudation_fraction::FT,
                     nitrification_rate::FT,
                     ammonia_fraction_of_exudate::FT,
                     ammonia_fraction_of_excriment::FT,
                     ammonia_fraction_of_detritus::FT,
                     phytoplankton_redfield::FT,
                     organic_redfield::FT,
                     phytoplankton_chlorophyll_ratio::FT,
                     organic_carbon_calcate_ratio::FT,
                     respiration_oxygen_nitrogen_ratio::FT,
                     nitrification_oxygen_nitrogen_ratio::FT,
                     slow_sinking_mortality_fraction::FT,
                     fast_sinking_mortality_fraction::FT,
                     dissolved_organic_breakdown_rate::FT,
                     zooplankton_calcite_dissolution::FT,

                     optionals::B,
                
                     sinking_velocities::W) where {FT, B, W}

        return new{FT, B, W}(phytoplankton_preference,
                             maximum_grazing_rate,
                             grazing_half_saturation,
                             light_half_saturation,
                             nitrate_ammonia_inhibition,
                             nitrate_half_saturation,
                             ammonia_half_saturation,
                             maximum_phytoplankton_growthrate,
                             zooplankton_assimilation_fraction,
                             zooplankton_mortality,
                             zooplankton_excretion_rate,
                             phytoplankton_mortality,
                             small_detritus_remineralisation_rate,
                             large_detritus_remineralisation_rate,
                             phytoplankton_exudation_fraction,
                             nitrification_rate,
                             ammonia_fraction_of_exudate,
                             ammonia_fraction_of_excriment,
                             ammonia_fraction_of_detritus,
                             phytoplankton_redfield,
                             organic_redfield,
                             phytoplankton_chlorophyll_ratio,
                             organic_carbon_calcate_ratio,
                             respiration_oxygen_nitrogen_ratio,
                             nitrification_oxygen_nitrogen_ratio,
                             slow_sinking_mortality_fraction,
                             fast_sinking_mortality_fraction,
                             dissolved_organic_breakdown_rate,
                             zooplankton_calcite_dissolution,

                             optionals,

                             sinking_velocities)
    end
end

"""
    LOBSTER(; grid::AbstractGrid{FT},
              phytoplankton_preference::FT = 0.5,
              maximum_grazing_rate::FT = 9.26e-6, # 1/s
              grazing_half_saturation::FT = 1.0, # mmol N/m³
              light_half_saturation::FT = 33.0, # W/m² (?)
              nitrate_ammonia_inhibition::FT = 3.0,
              nitrate_half_saturation::FT = 0.7, # mmol N/m³
              ammonia_half_saturation::FT = 0.001, # mmol N/m³
              maximum_phytoplankton_growthrate::FT = 1.21e-5, # 1/s
              zooplankton_assimilation_fraction::FT = 0.7,
              zooplankton_mortality::FT = 2.31e-6, # 1/s/mmol N/m³
              zooplankton_excretion_rate::FT = 5.8e-7, # 1/s
              phytoplankton_mortality::FT = 5.8e-7, # 1/s
              small_detritus_remineralisation_rate::FT = 5.88e-7, # 1/s
              large_detritus_remineralisation_rate::FT = 5.88e-7, # 1/s
              phytoplankton_exudation_fraction::FT = 0.05,
              nitrification_rate::FT = 5.8e-7, # 1/s
              ammonia_fraction_of_exudate::FT = 0.75, 
              ammonia_fraction_of_excriment::FT = 0.5,
              ammonia_fraction_of_detritus::FT = 0.0,
              phytoplankton_redfield::FT = 6.56, # mol C/mol N
              organic_redfield::FT = 6.56, # mol C/mol N
              phytoplankton_chlorophyll_ratio::FT = 1.31, # g Chl/mol N
              organic_carbon_calcate_ratio::FT = 0.1, # mol CaCO₃/mol C
              respiration_oxygen_nitrogen_ratio::FT = 10.75, # mol O/molN
              nitrification_oxygen_nitrogen_ratio::FT = 2.0, # mol O/molN
              slow_sinking_mortality_fraction::FT = 0.5, 
              fast_sinking_mortality_fraction::FT = 0.5,
              dissolved_organic_breakdown_rate::FT = 3.86e-7, # 1/s
              zooplankton_calcite_dissolution::FT = 0.3,

              surface_photosynthetically_active_radiation = default_surface_PAR,

              light_attenuation_model::LA =
                  TwoBandPhotosyntheticallyActiveRadiation(; grid, 
                                                             surface_PAR = surface_photosynthetically_active_radiation),
              sediment_model::S = nothing,

              carbonates::Bool = false,
              oxygen::Bool = false,
              variable_redfield::Bool = false,

              sinking_speeds = (sPOM = 3.47e-5, bPOM = 200/day),
              open_bottom::Bool = true,

              scale_negatives = false,

              particles::P = nothing,
              modifiers::M = nothing)

Construct an instance of the [LOBSTER](@ref LOBSTER) biogeochemical model.

Keyword Arguments
=================

- `grid`: (required) the geometry to build the model on, required to calculate sinking
- `phytoplankton_preference`, ..., `dissolved_organic_breakdown_rate`: LOBSTER parameter values
- `surface_photosynthetically_active_radiation`: funciton (or array in the future) for the photosynthetically available radiation at the surface, should be shape `f(x, y, t)`
- `light_attenuation_model`: light attenuation model which integrated the attenuation of available light
- `sediment_model`: slot for `BiogeochemicalSediment`
- `carbonates`, `oxygen`, and `variable_redfield`: include models for carbonate chemistry and/or oxygen chemistry and/or variable redfield ratio dissolved and particulate organic matter
- `sinking_speed`: named tuple of constant sinking, of fields (i.e. `ZFaceField(...)`) for any tracers which sink (convention is that a sinking speed is positive, but a field will need to follow the usual down being negative)
- `open_bottom`: should the sinking velocity be smoothly brought to zero at the bottom to prevent the tracers leaving the domain
- `scale_negatives`: scale negative tracers?
- `particles`: slot for `BiogeochemicalParticles`
- `modifiers`: slot for components which modify the biogeochemistry when the tendencies have been calculated or when the state is updated

Example
=======

```jldoctest
julia> using OceanBioME

julia> using Oceananigans

julia> grid = RectilinearGrid(size=(3, 3, 30), extent=(10, 10, 200));

julia> model = LOBSTER(; grid)
LOBSTER{Float64} with carbonates ❌, oxygen ❌, variable Redfield ratio ❌ and (:sPOM, :bPOM) sinking 
 Light attenuation: Two-band light attenuation model (Float64)
 Sediment: Nothing
 Particles: Nothing
 Modifiers: Nothing
```
"""
function LOBSTER(; grid::AbstractGrid{FT},
                   phytoplankton_preference::FT = 0.5,
                   maximum_grazing_rate::FT = 9.26e-6, # 1/s
                   grazing_half_saturation::FT = 1.0, # mmol N/m³
                   light_half_saturation::FT = 33.0, # W/m² (?)
                   nitrate_ammonia_inhibition::FT = 3.0,
                   nitrate_half_saturation::FT = 0.7, # mmol N/m³
                   ammonia_half_saturation::FT = 0.001, # mmol N/m³
                   maximum_phytoplankton_growthrate::FT = 1.21e-5, # 1/s
                   zooplankton_assimilation_fraction::FT = 0.7,
                   zooplankton_mortality::FT = 2.31e-6, # 1/s/mmol N/m³
                   zooplankton_excretion_rate::FT = 5.8e-7, # 1/s
                   phytoplankton_mortality::FT = 5.8e-7, # 1/s
                   small_detritus_remineralisation_rate::FT = 5.88e-7, # 1/s
                   large_detritus_remineralisation_rate::FT = 5.88e-7, # 1/s
                   phytoplankton_exudation_fraction::FT = 0.05,
                   nitrification_rate::FT = 5.8e-7, # 1/s
                   ammonia_fraction_of_exudate::FT = 0.75, 
                   ammonia_fraction_of_excriment::FT = 0.5,
                   ammonia_fraction_of_detritus::FT = 0.0,
                   phytoplankton_redfield::FT = 6.56, # mol C/mol N
                   organic_redfield::FT = 6.56, # mol C/mol N
                   phytoplankton_chlorophyll_ratio::FT = 1.31, # g Chl/mol N
                   organic_carbon_calcate_ratio::FT = 0.1, # mol CaCO₃/mol C
                   respiration_oxygen_nitrogen_ratio::FT = 10.75, # mol O/molN
                   nitrification_oxygen_nitrogen_ratio::FT = 2.0, # mol O/molN
                   slow_sinking_mortality_fraction::FT = 0.5, 
                   fast_sinking_mortality_fraction::FT = 0.5,
                   dissolved_organic_breakdown_rate::FT = 3.86e-7, # 1/s
                   zooplankton_calcite_dissolution::FT = 0.3,

                   surface_photosynthetically_active_radiation = default_surface_PAR,

                   light_attenuation_model::LA =
                       TwoBandPhotosyntheticallyActiveRadiation(; grid, 
                                                                  surface_PAR = surface_photosynthetically_active_radiation),
                   sediment_model::S = nothing,

                   carbonates::Bool = false,
                   oxygen::Bool = false,
                   variable_redfield::Bool = false,

                   sinking_speeds = (sPOM = 3.47e-5, bPOM = 200/day),
                   open_bottom::Bool = true,

                   scale_negatives = false,
                   invalid_fill_value = NaN,

                   particles::P = nothing,
                   modifiers::M = nothing) where {FT, LA, S, P, M}

    if !isnothing(sediment_model) && !open_bottom
        @warn "You have specified a sediment model but not `open_bottom` which will not work as the tracer will settle in the bottom cell"
    end

    sinking_velocities = setup_velocity_fields(sinking_speeds, grid, open_bottom)

    optionals = Val((carbonates, oxygen, variable_redfield))

    underlying_biogeochemistry = LOBSTER(phytoplankton_preference,
                                         maximum_grazing_rate,
                                         grazing_half_saturation,
                                         light_half_saturation,
                                         nitrate_ammonia_inhibition,
                                         nitrate_half_saturation,
                                         ammonia_half_saturation,
                                         maximum_phytoplankton_growthrate,
                                         zooplankton_assimilation_fraction,
                                         zooplankton_mortality,
                                         zooplankton_excretion_rate,
                                         phytoplankton_mortality,
                                         small_detritus_remineralisation_rate,
                                         large_detritus_remineralisation_rate,
                                         phytoplankton_exudation_fraction,
                                         nitrification_rate,
                                         ammonia_fraction_of_exudate,
                                         ammonia_fraction_of_excriment,
                                         ammonia_fraction_of_detritus,
                                         phytoplankton_redfield,
                                         organic_redfield,
                                         phytoplankton_chlorophyll_ratio,
                                         organic_carbon_calcate_ratio,
                                         respiration_oxygen_nitrogen_ratio,
                                         nitrification_oxygen_nitrogen_ratio,
                                         slow_sinking_mortality_fraction,
                                         fast_sinking_mortality_fraction,
                                         dissolved_organic_breakdown_rate,
                                         zooplankton_calcite_dissolution,

                                         optionals,

                                         sinking_velocities)

    if scale_negatives
        scaler = ScaleNegativeTracers(underlying_biogeochemistry, grid; invalid_fill_value)
        if isnothing(modifiers)
            modifiers = scaler
        elseif modifiers isa Tuple
            modifiers = (modifiers..., scaler)
        else
            modifiers = (modifiers, scaler)
        end
    end

    return Biogeochemistry(underlying_biogeochemistry;
                           light_attenuation = light_attenuation_model, 
                           sediment = sediment_model, 
                           particles,
                           modifiers)
end

# wrote this functionally and it took 2.5x longer so even though this is long going to use this way instead
@inline required_biogeochemical_tracers(::LOBSTER{<:Any, <:Val{(false, false, false)}, <:Any}) = (:NO₃, :NH₄, :P, :Z, :sPOM, :bPOM, :DOM)
# @inline required_biogeochemical_tracers(::LOBSTER{<:Any, <:Val{(true, false, false)}, <:Any}) = (:NO₃, :NH₄, :P, :Z, :sPOM, :bPOM, :DOM, :DIC, :Alk)
# @inline required_biogeochemical_tracers(::LOBSTER{<:Any, <:Val{(false, true, false)}, <:Any}) = (:NO₃, :NH₄, :P, :Z, :sPOM, :bPOM, :DOM, :O₂)
# @inline required_biogeochemical_tracers(::LOBSTER{<:Any, <:Val{(false, false, true)}, <:Any}) = (:NO₃, :NH₄, :P, :Z, :sPON, :bPON, :DON, :sPOC, :bPOC, :DOC)
# @inline required_biogeochemical_tracers(::LOBSTER{<:Any, <:Val{(true, true, false)}, <:Any}) = (:NO₃, :NH₄, :P, :Z, :sPOM, :bPOM, :DOM, :DIC, :Alk, :O₂)
# @inline required_biogeochemical_tracers(::LOBSTER{<:Any, <:Val{(true, false, true)}, <:Any}) = (:NO₃, :NH₄, :P, :Z, :sPON, :bPON, :DON, :DIC, :Alk, :sPOC, :bPOC, :DOC)
# @inline required_biogeochemical_tracers(::LOBSTER{<:Any, <:Val{(false, true, true)}, <:Any}) = (:NO₃, :NH₄, :P, :Z, :sPON, :bPON, :DON, :O₂, :sPOC, :bPOC, :DOC)
# @inline required_biogeochemical_tracers(::LOBSTER{<:Any, <:Val{(true, true, true)}, <:Any}) = (:NO₃, :NH₄, :P, :Z, :sPON, :bPON, :DON, :DIC, :Alk, :O₂, :sPOC, :bPOC, :DOC)

required_biogeochemical_auxiliary_fields(::LOBSTER) = (:PAR, )

# const small_detritus = Union{Val{:sPON}, Val{:sPOC}}
# const large_detritus = Union{Val{:bPON}, Val{:bPOC}}
# const dissolved_organic_matter = Union{Val{:DON}, Val{:DOC}}

# const sPOM = Union{Val{:sPOM}, Val{:sPON}}
# const bPOM = Union{Val{:bPOM}, Val{:bPON}}
# const DOM = Union{Val{:DOM}, Val{:DON}}

@inline biogeochemical_drift_velocity(bgc::LOBSTER, ::small_detritus) = biogeochemical_drift_velocity(bgc, Val(:sPOM))
@inline biogeochemical_drift_velocity(bgc::LOBSTER, ::large_detritus) = biogeochemical_drift_velocity(bgc, Val(:bPOM))
# @inline biogeochemical_drift_velocity(bgc::LOBSTER, ::dissolved_organic_matter) = biogeochemical_drift_velocity(bgc, Val(:DOM))

@inline function biogeochemical_drift_velocity(bgc::LOBSTER, ::Val{tracer_name}) where tracer_name
    if tracer_name in keys(bgc.sinking_velocities)
        return (u = ZeroField(), v = ZeroField(), w = bgc.sinking_velocities[tracer_name])
    else
        return (u = ZeroField(), v = ZeroField(), w = ZeroField())
    end
end

adapt_structure(to, lobster::LOBSTER) =
    LOBSTER(adapt(to, lobster.phytoplankton_preference),
            adapt(to, lobster.maximum_grazing_rate),
            adapt(to, lobster.grazing_half_saturation),
            adapt(to, lobster.light_half_saturation),
            adapt(to, lobster.nitrate_ammonia_inhibition),
            adapt(to, lobster.nitrate_half_saturation),
            adapt(to, lobster.ammonia_half_saturation),
            adapt(to, lobster.maximum_phytoplankton_growthrate),
            adapt(to, lobster.zooplankton_assimilation_fraction),
            adapt(to, lobster.zooplankton_mortality),
            adapt(to, lobster.zooplankton_excretion_rate),
            adapt(to, lobster.phytoplankton_mortality),
            adapt(to, lobster.small_detritus_remineralisation_rate),
            adapt(to, lobster.large_detritus_remineralisation_rate),
            adapt(to, lobster.phytoplankton_exudation_fraction),
            adapt(to, lobster.nitrification_rate),
            adapt(to, lobster.ammonia_fraction_of_exudate),
            adapt(to, lobster.ammonia_fraction_of_excriment),
            adapt(to, lobster.ammonia_fraction_of_detritus),
            adapt(to, lobster.phytoplankton_redfield),
            adapt(to, lobster.organic_redfield),
            adapt(to, lobster.phytoplankton_chlorophyll_ratio),
            adapt(to, lobster.organic_carbon_calcate_ratio),
            adapt(to, lobster.respiration_oxygen_nitrogen_ratio),
            adapt(to, lobster.nitrification_oxygen_nitrogen_ratio),
            adapt(to, lobster.slow_sinking_mortality_fraction),
            adapt(to, lobster.fast_sinking_mortality_fraction),
            adapt(to, lobster.dissolved_organic_breakdown_rate),
            adapt(to, lobster.zooplankton_calcite_dissolution),
            adapt(to, lobster.optionals),
            adapt(to, lobster.sinking_velocities))

summary(::LOBSTER{FT, Val{B}, NamedTuple{K, V}}) where {FT, B, K, V} = string("LOBSTER{$FT} with carbonates $(B[1] ? :✅ : :❌), oxygen $(B[2] ? :✅ : :❌), variable Redfield ratio $(B[3] ? :✅ : :❌) and $K sinking")

show(io::IO, model::LOBSTER{FT, Val{B}, W}) where {FT, B, W}  = print(io, string("Lodyc-DAMTP Ocean Biogeochemical Simulation Tools for Ecosystem and Resources (LOBSTER) model \n",
                                                                                 "├── Optional components:", "\n",
                                                                                 "│   ├── Carbonates $(B[1] ? :✅ : :❌) \n",
                                                                                 "│   ├── Oxygen $(B[2] ? :✅ : :❌) \n",
                                                                                 "│   └── Variable Redfield Ratio $(B[3] ? :✅ : :❌)", "\n",
                                                                                 "└── Sinking Velocities:", "\n", show_sinking_velocities(model.sinking_velocities)))

@inline maximum_sinking_velocity(bgc::LOBSTER) = maximum(abs, bgc.sinking_velocities.bPOM.w)

include("fallbacks.jl")
include("core.jl")
include("carbonate_chemistry.jl")
include("oxygen_chemistry.jl")
include("variable_redfield.jl")

# const VariableRedfieldLobster = Union{LOBSTER{<:Any, <:Val{(false, false, true)}, <:Any},
#                                       LOBSTER{<:Any, <:Val{(true, false, true)}, <:Any},
#                                       LOBSTER{<:Any, <:Val{(false, true, true)}, <:Any},
#                                       LOBSTER{<:Any, <:Val{(true, true, true)}, <:Any}}

# @inline redfield(i, j, k, val_tracer_name, bgc::LOBSTER, tracers) = redfield(val_tracer_name, bgc)

# @inline redfield(::Val{:P}, bgc::LOBSTER) = (1 + bgc.organic_carbon_calcate_ratio) * bgc.phytoplankton_redfield
# @inline redfield(::Val{:Z}, bgc::LOBSTER) = bgc.phytoplankton_redfield
# @inline redfield(::Union{Val{:NO₃}, Val{:NH₄}, Val{:Alk}, Val{:O₂}}, bgc::LOBSTER{FT}) where FT = convert(FT, 0)
# @inline redfield(::Union{Val{:sPOM}, Val{:bPOM}, Val{:DOM}}, bgc::LOBSTER) = bgc.organic_redfield
# @inline redfield(::Union{Val{:sPOC}, Val{:bPOC}, Val{:DOC}, Val{:DIC}}, bgc::LOBSTER{FT}) where FT = convert(FT, 1)

# @inline redfield(i, j, k, ::Val{:sPON}, bgc::VariableRedfieldLobster, tracers) = @inbounds tracers.sPOC[i, j, k] / tracers.sPON[i, j, k]
# @inline redfield(i, j, k, ::Val{:bPON}, bgc::VariableRedfieldLobster, tracers) = @inbounds tracers.bPOC[i, j, k] / tracers.bPON[i, j, k]
# @inline redfield(i, j, k, ::Val{:DON}, bgc::VariableRedfieldLobster, tracers) = @inbounds tracers.DOC[i, j, k] / tracers.DON[i, j, k]

# @inline redfield(::Val{:sPON}, bgc::VariableRedfieldLobster, tracers) = tracers.sPOC / tracers.sPON
# @inline redfield(::Val{:bPON}, bgc::VariableRedfieldLobster, tracers) = tracers.bPOC / tracers.bPON
# @inline redfield(::Val{:DON}, bgc::VariableRedfieldLobster, tracers) = tracers.DOC / tracers.DON

# @inline conserved_tracers(::LOBSTER) = (:NO₃, :NH₄, :P, :Z, :sPOM, :bPOM, :DOM)
# @inline conserved_tracers(::VariableRedfieldLobster) = (:NO₃, :NH₄, :P, :Z, :sPON, :bPON, :DON)

@inline sinking_tracers(::LOBSTER) = (:sPOM, :bPOM)
# @inline sinking_tracers(::VariableRedfieldLobster) = (:sPON, :bPON, :sPOC, :bPOC)

# @inline chlorophyll(bgc::LOBSTER, model) = bgc.phytoplankton_chlorophyll_ratio * model.tracers.P

end # module
