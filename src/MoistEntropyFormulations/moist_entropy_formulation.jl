#####
##### MoistEntropyFormulation
#####

"""
$(TYPEDEF)

`MoistEntropyFormulation` uses the moist entropy density `ρη` as the prognostic thermodynamic
variable, where ``η`` is the specific entropy of the moist air, condensate included,

```math
η = qᵈ ηᵈ + qᵛ ηᵛ + qˡ ηˡ + qⁱ ηⁱ ,
```

with the entropy of liquid water zero at the triple point (see
[`moist_entropy`](@ref Breeze.Thermodynamics.moist_entropy)). The prognostic `ρη` is the
physical entropy density: ``η`` is per unit *total* mass and is advected and diffused with the
total density, the same carrier as the water mass fractions. It is supported with
[`CompressibleDynamics`](@ref Breeze.CompressibleEquations.CompressibleDynamics) only, where the
temperature follows from ``η``, the total density and the water mass fractions in closed form,
``T = Tᵗʳ \\exp[(η + f(ρ, q)) / cᵛᵐ]``.

The entropy changes only through its sources: heating ``Q`` enters as ``Q / T``; non-equilibrium
phase changes at fixed internal energy and density as ``-Σₓ 𝒢ˣ Ṡˣ / T`` from the microphysical
tendencies ``Ṡˣ`` and the specific Gibbs functions ``𝒢ˣ`` (zero at saturation); falling condensate
carries its own entropy ``ηˣ(T)`` and heats the air it falls through by drag, ``g wˣ ρqˣ / T``; and an
evaporating surface supplies ``𝒬ᵀ / T + Jᵛ ηᵛ``. Expansion needs no source.

Saturation-adjustment (equilibrium) microphysics is not supported.

# Example

```jldoctest
using Breeze

MoistEntropyFormulation()

# output
MoistEntropyFormulation
└── water_entropy_offset: -7000
```
"""
struct MoistEntropyFormulation{D, S, FT}
    entropy_density :: D         # ρη (prognostic, total-density weighted)
    specific_entropy :: S        # η = ρη / ρ (diagnostic)
    water_entropy_offset :: FT   # b: the entropy of every water species is shifted by b
end

"""
$(TYPEDSIGNATURES)

Return a `MoistEntropyFormulation` whose fields are materialized in the model constructor.

The prognostic specific entropy is ``η = η₀ + b qᵗ``, the moist entropy ``η₀`` of
[`moist_entropy`](@ref Breeze.Thermodynamics.moist_entropy) (liquid water zero at the triple
point) plus `water_entropy_offset` ``b`` times the total water mass fraction. Entropy is defined
up to one constant per conserved substance, so ``b`` changes no physics: every water flux carries it
with the water, and phase changes conserve ``qᵗ``. It sets how the pressure responds to water at
fixed ``ρη``, ``∂p/∂ρᵛ = (Rᵐ T / cᵛᵐ) (Rᵛ cᵖᵐ / Rᵐ - ηᵛ - b)`` with ``ηᵛ ≈ 8600`` J kg⁻¹ K⁻¹ the
partial specific entropy of boundary-layer vapor, which matters because the acoustic substeps hold
the water densities fixed: with ``b = 0`` a parcel displaced in the substep loop carries its vapor's
large entropy but not its vapor, and moist air with vapor decreasing upward appears strongly
unstable to the loop. The default ``b = -7000`` J kg⁻¹ K⁻¹ ≈ ``Rᵛ cᵖᵈ / Rᵈ - ηᵛ`` at 300 K and a vapor pressure of 3 kPa
nearly removes that response for boundary-layer vapor.
"""
MoistEntropyFormulation(; water_entropy_offset = default_water_entropy_offset) =
    MoistEntropyFormulation(nothing, nothing, water_entropy_offset)

const default_water_entropy_offset = -7000

"""
$(TYPEDSIGNATURES)

Return the entropy offset ``b`` per unit water mass of `formulation` (see [`MoistEntropyFormulation`](@ref)).
"""
water_entropy_offset(formulation::MoistEntropyFormulation) = formulation.water_entropy_offset
water_entropy_offset(formulation) = default_water_entropy_offset

# The moist entropy η₀ of `moist_entropy` from the prognostic specific entropy η = η₀ + b qᵗ
@inline offset_free_entropy(formulation, η, q) = η - formulation.water_entropy_offset * total_specific_moisture(q)

# The prognostic specific entropy from the moist entropy η₀
@inline offset_entropy(formulation, η₀, q) = η₀ + formulation.water_entropy_offset * total_specific_moisture(q)

Adapt.adapt_structure(to, formulation::MoistEntropyFormulation) =
    MoistEntropyFormulation(adapt(to, formulation.entropy_density),
                            adapt(to, formulation.specific_entropy),
                            formulation.water_entropy_offset)

function OceananigansBC.fill_halo_regions!(formulation::MoistEntropyFormulation)
    fill_halo_regions!(formulation.specific_entropy)
    return nothing
end

#####
##### Field naming interface
#####

const MoistEntropyNames = Union{Val{:MoistEntropy}, Val{:η}, Val{:ρη}}

AtmosphereModels.prognostic_thermodynamic_field_names(::MoistEntropyFormulation) = tuple(:ρη)
AtmosphereModels.additional_thermodynamic_field_names(::MoistEntropyFormulation) = tuple(:η)
AtmosphereModels.thermodynamic_density_name(::MoistEntropyFormulation) = :ρη
AtmosphereModels.thermodynamic_density(formulation::MoistEntropyFormulation) = formulation.entropy_density
AtmosphereModels.specific_thermodynamic_field(formulation::MoistEntropyFormulation) = formulation.specific_entropy
AtmosphereModels.with_thermodynamic_density(formulation::MoistEntropyFormulation, ρη) =
    MoistEntropyFormulation(ρη, formulation.specific_entropy, formulation.water_entropy_offset)

AtmosphereModels.prognostic_thermodynamic_field_names(::MoistEntropyNames) = tuple(:ρη)
AtmosphereModels.additional_thermodynamic_field_names(::MoistEntropyNames) = tuple(:η)
AtmosphereModels.thermodynamic_density_name(::MoistEntropyNames) = :ρη

Oceananigans.fields(formulation::MoistEntropyFormulation) = (; η=formulation.specific_entropy)
Oceananigans.prognostic_fields(formulation::MoistEntropyFormulation) = (; ρη=formulation.entropy_density)

#####
##### Materialization
#####

AtmosphereModels.materialize_formulation(::MoistEntropyNames, dynamics, grid, boundary_conditions) =
    materialize_formulation(MoistEntropyFormulation(), dynamics, grid, boundary_conditions)

# `CompressibleDynamics` extends this with the method that allocates the fields.
AtmosphereModels.materialize_formulation(::MoistEntropyFormulation, dynamics, grid, boundary_conditions) =
    throw(ArgumentError("MoistEntropyFormulation is supported with CompressibleDynamics only, " *
                        "not $(nameof(typeof(dynamics))). Its temperature inversion closes on the " *
                        "prognostic density; use formulation = :LiquidIcePotentialTemperature or " *
                        ":StaticEnergy with $(nameof(typeof(dynamics)))."))

function materialize_moist_entropy_formulation(formulation, grid, boundary_conditions)
    entropy_density = CenterField(grid, boundary_conditions=boundary_conditions.ρη)
    specific_entropy = CenterField(grid)
    b = convert(eltype(grid), formulation.water_entropy_offset)
    return MoistEntropyFormulation(entropy_density, specific_entropy, b)
end

#####
##### Supported microphysics
#####

"""
$(TYPEDSIGNATURES)

Return `true` if `microphysics` couples to the thermodynamics only through tendencies of its
water species and the temperature of the thermodynamic state, which is what
`MoistEntropyFormulation` needs to evaluate the entropy of each phase change. Schemes with a
prognostic vapor density qualify by default; saturation-adjustment schemes (whose moisture
prognostic is the equilibrium moisture `ρqᵉ`) do not, and schemes that write the thermodynamic
variable directly in an operator-split update extend this function to return `false`.
"""
moist_entropy_compatible(microphysics) = moisture_prognostic_name(microphysics) === :ρqᵛ

function validate_moist_entropy_microphysics(microphysics)
    moist_entropy_compatible(microphysics) && return nothing
    throw(ArgumentError("MoistEntropyFormulation does not support $(summary(microphysics)). " *
                        "It needs microphysics that changes the water species through tendencies " *
                        "(non-equilibrium cloud formation), so that the entropy of each phase change " *
                        "can be evaluated; saturation adjustment and operator-split schemes that " *
                        "rewrite the thermodynamic variable are not supported."))
end

#####
##### Auxiliary variable computation
#####

# η = ρη / ρ with the total density
function AtmosphereModels.compute_auxiliary_thermodynamic_variables!(formulation::MoistEntropyFormulation, dynamics, i, j, k, grid)
    @inbounds begin
        ρ = total_density(dynamics)[i, j, k]
        ρη = formulation.entropy_density[i, j, k]
        formulation.specific_entropy[i, j, k] = ρη / ρ
    end
    return nothing
end

#####
##### Thermodynamic state diagnosis
#####

"""
$(TYPEDSIGNATURES)

Build a `MoistEntropyState` at grid point `(i, j, k)` from the specific entropy (less its water
offset), the total density and the moisture mass fractions `q`.
"""
@inline function AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid,
                                                               formulation::MoistEntropyFormulation,
                                                               dynamics, q)
    η = @inbounds formulation.specific_entropy[i, j, k]
    ρ = @inbounds total_density(dynamics)[i, j, k]
    return MoistEntropyState(offset_free_entropy(formulation, η, q), q, ρ)
end

#####
##### Prognostic field collection
#####

function AtmosphereModels.collect_prognostic_fields(formulation::MoistEntropyFormulation,
                                                    dynamics,
                                                    momentum,
                                                    moisture_density,
                                                    moisture_name,
                                                    microphysical_fields,
                                                    tracers)
    ρη = formulation.entropy_density
    thermodynamic_variables = merge((ρη=ρη,), NamedTuple{(moisture_name,)}((moisture_density,)))
    dynamics_fields = dynamics_prognostic_fields(dynamics)
    return merge(dynamics_fields, momentum, thermodynamic_variables, microphysical_fields, tracers)
end

#####
##### Show methods
#####

Base.summary(::MoistEntropyFormulation) = "MoistEntropyFormulation"

function Base.show(io::IO, formulation::MoistEntropyFormulation)
    print(io, summary(formulation), '\n')
    if formulation.entropy_density !== nothing
        print(io, "├── entropy_density: ", prettysummary(formulation.entropy_density), '\n')
        print(io, "├── specific_entropy: ", prettysummary(formulation.specific_entropy), '\n')
    end
    print(io, "└── water_entropy_offset: ", prettysummary(formulation.water_entropy_offset))
end
