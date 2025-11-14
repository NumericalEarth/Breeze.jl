"""
Extension module for integrating CloudMicrophysics.jl schemes with Breeze.jl.

This extension provides integration between CloudMicrophysics.jl microphysics schemes
and Breeze.jl's microphysics interface, allowing CloudMicrophysics schemes to be used
with AtmosphereModel.

The extension is automatically loaded when CloudMicrophysics is available in the environment.
"""
module BreezeCloudMicrophysicsExt

using CloudMicrophysics
using CloudMicrophysics.Parameters: Parameters0M, Parameters1M
using CloudMicrophysics.Microphysics0M: remove_precipitation
using CloudMicrophysics.Microphysics1M:
    conv_q_lcl_to_q_rai,
    conv_q_icl_to_q_sno_no_supersat,
    accretion,
    evaporation_sublimation,
    snow_melt

# Import Breeze modules needed for integration
using ..Breeze
using ..Breeze.AtmosphereModels
using ..Breeze.Thermodynamics
using ..Breeze.Microphysics

import ..Breeze.AtmosphereModels:
    compute_thermodynamic_state,
    prognostic_field_names,
    materialize_microphysical_fields,
    update_microphysical_fields!,
    moisture_mass_fractions

import ..Breeze.Thermodynamics:
    total_moisture_mass_fraction,
    with_moisture,
    MoistureMassFractions

using Oceananigans: Oceananigans
using DocStringExtensions: TYPEDSIGNATURES

#####
##### Zero-moment bulk microphysics (CloudMicrophysics 0M)
#####

"""
    ZeroMomentBulkMicrophysics

Type alias for `BulkMicrophysics` with CloudMicrophysics 0M precipitation scheme.

The 0M scheme instantly removes precipitable condensate above a threshold.
Interface is identical to non-precipitating microphysics except that
`compute_thermodynamic_state` calls CloudMicrophysics `remove_precipitation` first.
"""
const ZeroMomentBulkMicrophysics = BulkMicrophysics{<:Any, <:Parameters0M}
const ZMBM = ZeroMomentBulkMicrophysics

"""
$(TYPEDSIGNATURES)

Return `tuple()` - zero-moment scheme has no prognostic variables.
"""
prognostic_field_names(::ZMBM) = tuple()

"""
$(TYPEDSIGNATURES)

Delegate to clouds scheme (saturation adjustment) for microphysical fields.
"""
materialize_microphysical_fields(bμp::ZMBM, grid, bcs) = 
    materialize_microphysical_fields(bμp.clouds, grid, bcs)

"""
$(TYPEDSIGNATURES)

Delegate to clouds scheme for updating microphysical fields.
"""
@inline update_microphysical_fields!(μ, bμp::ZMBM, i, j, k, grid, 𝒰, thermo) =
    update_microphysical_fields!(μ, bμp.clouds, i, j, k, grid, 𝒰, thermo)

"""
$(TYPEDSIGNATURES)

Delegate to clouds scheme for extracting moisture mass fractions.
"""
@inline moisture_mass_fractions(i, j, k, grid, bμp::ZMBM, μ, qᵗ) =
    moisture_mass_fractions(i, j, k, grid, bμp.clouds, μ, qᵗ)

"""
$(TYPEDSIGNATURES)

Compute thermodynamic state for zero-moment bulk microphysics.

First removes excess condensate using CloudMicrophysics `remove_precipitation`,
then delegates to clouds scheme (saturation adjustment) for vapor↔cloud conversion.

For instantaneous removal, we compute the equilibrium state where excess condensate
above the threshold is removed. The removed condensate is converted to vapor
(to conserve total moisture), then saturation adjustment handles vapor↔cloud conversion.
"""
@inline function compute_thermodynamic_state(𝒰₀::AbstractThermodynamicState, bμp::ZMBM, thermo)
    # Extract current condensate amounts
    q = 𝒰₀.moisture_mass_fractions
    q_lcl = q.liquid
    q_icl = q.ice
    
    # Compute removal tendency from CloudMicrophysics 0M scheme
    # This returns dq_tot/dt = -max(0, q_lcl + q_icl - qc_0) / τ_precip
    # For instantaneous removal (τ_precip → dt), we need to compute the equilibrium
    # state directly by removing excess condensate above threshold
    dq_tot_dt = remove_precipitation(bμp.precipitation, q_lcl, q_icl)
    
    # Extract threshold from parameters
    params = bμp.precipitation
    qc_0 = hasfield(typeof(params), :qc_0) ? params.qc_0 : zero(eltype(𝒰₀))
    
    # Compute excess condensate above threshold
    q_cond = q_lcl + q_icl
    excess = max(0, q_cond - qc_0)
    
    # Remove excess condensate proportionally from liquid and ice
    if excess > 0 && q_cond > 0
        removal_fraction = excess / q_cond
        q_lcl_adjusted = max(0, q_lcl * (1 - removal_fraction))
        q_icl_adjusted = max(0, q_icl * (1 - removal_fraction))
    else
        q_lcl_adjusted = q_lcl
        q_icl_adjusted = q_icl
    end
    
    # Create adjusted state with removed condensate
    # The removed condensate becomes vapor (conserves total moisture)
    q_vap_adjusted = q.vapor + (q_lcl - q_lcl_adjusted) + (q_icl - q_icl_adjusted)
    q_adjusted = MoistureMassFractions(q_vap_adjusted, q_lcl_adjusted, q_icl_adjusted)
    𝒰_adjusted = with_moisture(𝒰₀, q_adjusted)
    
    # Delegate to clouds scheme (saturation adjustment) for vapor↔cloud conversion
    return compute_thermodynamic_state(𝒰_adjusted, bμp.clouds, thermo)
end

#####
##### One-moment bulk microphysics (CloudMicrophysics 1M)
#####

"""
    OneMomentBulkMicrophysics

Type alias for `BulkMicrophysics` with CloudMicrophysics 1M precipitation scheme.

The 1M scheme handles cloud-to-precipitation processes (autoconversion, accretion,
evaporation/sublimation, melting). It is designed to work WITH saturation adjustment
for vapor↔cloud conversion.

Prognostic variables:
- `:qˡ` - cloud liquid mass fraction (q_lcl)
- `:qⁱ` - cloud ice mass fraction (q_icl)
- `:qʳ` - rain mass fraction (q_rai)
- `:qˢ` - snow mass fraction (q_sno)
"""
const OneMomentBulkMicrophysics = BulkMicrophysics{<:Any, <:Parameters1M}
const OMBM = OneMomentBulkMicrophysics

"""
$(TYPEDSIGNATURES)

Return prognostic field names for 1M scheme: `(:qˡ, :qⁱ, :qʳ, :qˢ)`.
"""
prognostic_field_names(::OMBM) = (:qˡ, :qⁱ, :qʳ, :qˢ)

"""
$(TYPEDSIGNATURES)

Create microphysical fields for 1M scheme: cloud liquid, cloud ice, rain, and snow.
"""
function materialize_microphysical_fields(bμp::OMBM, grid, bcs)
    clouds_fields = materialize_microphysical_fields(bμp.clouds, grid, bcs)
    precip_fields = center_field_tuple(grid, :qʳ, :qˢ)
    return merge(clouds_fields, precip_fields)
end

# Helper function to create center fields
center_field_tuple(grid, names...) = 
    NamedTuple{names}(Oceananigans.CenterField(grid) for name in names)

"""
$(TYPEDSIGNATURES)

Update microphysical fields for 1M scheme.

Computes tendencies from CloudMicrophysics 1M scheme and updates prognostic variables.
Saturation adjustment (via clouds scheme) handles vapor↔cloud conversion,
so this function only handles cloud↔precipitation processes.

Note: The actual tendency computation from CloudMicrophysics 1M functions needs to be
completed. This is a placeholder implementation that updates cloud fields from
saturation adjustment. The precipitation fields (qʳ, qˢ) will need to be updated
by computing tendencies from CloudMicrophysics functions and integrating them
in the time-stepper.
"""
@inline function update_microphysical_fields!(μ, bμp::OMBM, i, j, k, grid, 𝒰, thermo)
    # Update cloud fields from saturation adjustment
    update_microphysical_fields!(μ, bμp.clouds, i, j, k, grid, 𝒰, thermo)
    
    # TODO: Compute tendencies from CloudMicrophysics 1M scheme:
    # - Autoconversion: conv_q_lcl_to_q_rai, conv_q_icl_to_q_sno_no_supersat
    # - Accretion: accretion (cloud liquid + rain, cloud ice + snow)
    # - Accretion: accretion_snow_rain (rain + snow)
    # - Evaporation/sublimation: evaporation_sublimation (rain, snow)
    # - Snow melt: snow_melt
    # Then integrate tendencies to update qʳ and qˢ fields
    
    # For now, precipitation fields are not updated here
    # They will be updated by the time-stepper once tendency computation is implemented
    return nothing
end

"""
$(TYPEDSIGNATURES)

Extract moisture mass fractions from microphysical fields for 1M scheme.
"""
@inline function moisture_mass_fractions(i, j, k, grid, bμp::OMBM, μ, qᵗ)
    @inbounds begin
        qᵛ = μ.qᵛ[i, j, k]
        qˡ = μ.qˡ[i, j, k]
        qⁱ = μ.qⁱ[i, j, k]
    end
    # Note: qʳ and qˢ are precipitation, not part of MoistureMassFractions
    # Total moisture qᵗ includes vapor + cloud liquid + cloud ice
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

"""
$(TYPEDSIGNATURES)

Compute thermodynamic state for one-moment bulk microphysics.

Delegates to clouds scheme (saturation adjustment) for vapor↔cloud conversion.
CloudMicrophysics 1M handles cloud↔precipitation processes via tendencies
computed in `update_microphysical_fields!`.
"""
@inline compute_thermodynamic_state(𝒰₀::AbstractThermodynamicState, bμp::OMBM, thermo) =
    compute_thermodynamic_state(𝒰₀, bμp.clouds, thermo)

end # module BreezeCloudMicrophysicsExt

