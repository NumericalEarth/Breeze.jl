#####
##### Optional energy-closing sources of a thermodynamic formulation
#####
#
# A prognostic thermodynamic variable other than the total energy closes the energy budget only
# up to the sources it omits. The types below select three of them; a formulation stores
# `nothing` for an option it does not apply, so the omitted source costs nothing and changes
# nothing. `LiquidIcePotentialTemperatureFormulation` turns them on through its keyword
# arguments; every other formulation applies none of them.

"""
$(TYPEDEF)

Heat the air by the potential energy that falling condensate loses, ``g Σₓ ρqˣ |wˣ|`` per unit
volume. Each unit of condensate mass that sediments through a cell delivers ``g`` times the height
it falls within the cell, ``g (zᶠ - zᶜ)`` across each face, so that the heating summed over a
column equals the potential energy lost by the falling mass.
"""
struct PrecipitationDragHeating end

"""
$(TYPEDEF)

Correct the microphysical phase changes for the fixed volume at which they act on the compressible
core: microphysics converts water at fixed density, while holding ``θˡⁱ`` fixed through the
conversion conserves the enthalpy at fixed pressure. The source
``ρθ̇ = - Σ_q 𝒥_q Ṡ_q / 𝒥_Θ`` restores the internal energy, with ``𝒥_ψ = ∂(ρ𝓊)/∂ψ`` at fixed
volume and ``Ṡ_q`` the microphysical tendencies of the water densities. Not applied on dynamics
whose thermodynamic pressure is prescribed (the anelastic core), where the fixed-pressure form is
exact.
"""
struct FixedVolumePhaseChange end

"""
$(TYPEDEF)

Heat the air by the resolved kinetic energy that the turbulence closure dissipates,
``ρ ε = - 𝒯ᵢⱼ ∂ⱼ uᵢ`` with ``𝒯`` the closure's own viscous stress. See
[`subgrid_dissipation`](@ref) for the closures that supply it.
"""
struct SubgridDissipativeHeating end

Base.summary(::PrecipitationDragHeating) = "PrecipitationDragHeating()"
Base.summary(::FixedVolumePhaseChange) = "FixedVolumePhaseChange()"
Base.summary(::SubgridDissipativeHeating) = "SubgridDissipativeHeating()"

"""
$(TYPEDSIGNATURES)

Return the [`PrecipitationDragHeating`](@ref) option of `formulation`, or `nothing` (the default)
when the formulation does not heat the air by the drag of falling condensate. The sedimentation
tendency of the thermodynamic variable reads it.
"""
precipitation_heating(formulation) = nothing

"""
$(TYPEDSIGNATURES)

Return the fixed-volume phase-change `option` that applies on `dynamics`: `nothing` by default,
for dynamics whose thermodynamic pressure is prescribed and on which phase change at fixed
``θˡⁱ`` is already exact. Dynamics that close the thermodynamics on the density
(`CompressibleDynamics`) extend this to return `option`.
"""
materialize_phase_change_source(option, dynamics) = nothing

"""
$(TYPEDSIGNATURES)

Hooks around `compute_microphysical_tendencies!` through which a formulation forms a source from
the microphysical tendencies of the water densities: `compute_tendencies!` calls
`begin_phase_change_source!` before the microphysics adds its tendencies to `Gⁿ`, and
[`finish_phase_change_source!`](@ref) after, so the difference of the two is the microphysical
part alone. No-ops by default.
"""
begin_phase_change_source!(formulation, model) = nothing

"""
$(TYPEDSIGNATURES)

See [`begin_phase_change_source!`](@ref).
"""
finish_phase_change_source!(formulation, model) = nothing

"""
$(TYPEDSIGNATURES)

Return the rate ``ρ ε`` [W m⁻³] at which the turbulence `closure` dissipates resolved kinetic
energy at cell `(i, j, k)`, ``- 𝒯ᵢⱼ ∂ⱼ uᵢ`` with ``𝒯 = ρ τ`` its viscous stress weighted by the
density `ρ` that weights the momentum stresses. Zero by default and for `closure = nothing`.
`Breeze.TurbulenceClosures` implements it for every closure whose momentum stresses are
Oceananigans viscous fluxes (`ScalarDiffusivity`, `SmagorinskyLilly`, `DynamicSmagorinsky`,
`AnisotropicMinimumDissipation`, and tuples of them), with the stresses of a vertically implicit
closure evaluated explicitly. For [`TKEBasedTurbulenceClosure`](@ref Breeze.TurbulenceClosures.TKEBasedTurbulenceClosure)
it is zero: the kinetic energy it removes feeds the subgrid TKE, whose own dissipation is the
heating (not yet applied).
"""
@inline subgrid_dissipation(i, j, k, grid, ρ, closure, closure_fields, clock, model_fields) = zero(grid)

"""
$(TYPEDSIGNATURES)

Return `false` for microphysics that updates the temperature itself (see
`OperatorSplitSaturationAdjustment`), whose water tendencies therefore carry no phase-change source.
"""
phase_change_source_applies(microphysics) = true

"""
$(TYPEDSIGNATURES)

Return `true` if `microphysics` can drive a potential temperature that is not phase invariant:
the latent heating comes from its tendencies of the water species, or from an update of the
thermodynamic variable it makes itself. Schemes with a prognostic vapor density qualify by default;
diagnostic saturation adjustment, whose moisture prognostic is the equilibrium moisture `ρqᵉ`, does
not, and operator-split schemes that assume ``θˡⁱ`` extend this function to return `false`.
"""
dry_potential_temperature_compatible(microphysics) = moisture_prognostic_name(microphysics) === :ρqᵛ

"""
$(TYPEDSIGNATURES)

Return `true` if the potential temperature of a wall at temperature `Tˢ` that enters a bulk
sensible heat flux counts the condensate of the near-wall air, as ``θˡⁱ`` does, and `false`
if it is simply `Tˢ / Π`.
"""
wall_potential_temperature_counts_condensate(formulation) = true
wall_potential_temperature_counts_condensate(name::Symbol) = wall_potential_temperature_counts_condensate(Val(name))
