#####
##### Potential temperature flavors
#####

"""
$(TYPEDEF)

Which potential temperature a [`PotentialTemperatureFormulation`](@ref) evolves.
"""
abstract type AbstractPotentialTemperatureFlavor end

"""
$(TYPEDEF)

The liquid-ice potential temperature ``θˡⁱ = H / (ĉᵖ Π)``, which is invariant under phase change at
fixed pressure. See [`LiquidIcePotentialTemperatureFormulation`](@ref).
"""
struct LiquidIce <: AbstractPotentialTemperatureFlavor end

"""
$(TYPEDEF)

Close the thermodynamics of a [`Dry`](@ref) potential temperature as CM1's `eqtset = 2` does: heating
and phase change act at fixed volume, with the coefficient ``(cᵛᵈ / cᵖᵈ) / (cᵛᵐ Π)``, and the
expansion source is written with the velocity divergence, ``- θ (Rᵐ - κᵈ cᵖᵐ) / cᵛᵐ ∇ ⋅ 𝐮``.
"""
struct IsochoricPressureWork end

"""
$(TYPEDEF)

Close the thermodynamics of a [`Dry`](@ref) potential temperature at fixed pressure: heating and phase
change enter with ``1 / (cᵖᵐ Π)`` and the expansion source is ``θ (κᵐ - κᵈ) D \\ln p / Dt`` with
the hydrostatic estimate ``D \\ln p / Dt ≈ - g ρ w / p``. In the continuous equations this is the
same as [`IsochoricPressureWork`](@ref) to the extent that the pressure is hydrostatic.
"""
struct IsobaricPressureWork end

"""
$(TYPEDEF)

The potential temperature ``θ = T / Π`` with ``Π = (p / pˢᵗ)^{κᵈ}``, ``κᵈ = Rᵈ / cᵖᵈ``, that CM1 and
most cloud models evolve. It is not invariant under phase change, so the latent heating is an
explicit source. `pressure_work` selects the closure, [`IsochoricPressureWork`](@ref) or
[`IsobaricPressureWork`](@ref). `condensate_enthalpy_flux` selects whether falling condensate carries
its enthalpy (CM1's `efall = 1`).
"""
struct Dry{W, C} <: AbstractPotentialTemperatureFlavor
    pressure_work :: W
    condensate_enthalpy_flux :: C # Val{true} or Val{false}
end

Dry(; pressure_work = IsochoricPressureWork(), condensate_enthalpy_flux = true) =
    Dry(pressure_work, Val(condensate_enthalpy_flux))

Base.summary(::LiquidIce) = "LiquidIce()"
Base.summary(::IsochoricPressureWork) = "IsochoricPressureWork()"
Base.summary(::IsobaricPressureWork) = "IsobaricPressureWork()"
Base.summary(f::Dry{<:Any, Val{C}}) where C =
    string("Dry(pressure_work = ", summary(f.pressure_work), ", condensate_enthalpy_flux = ", C, ")")

#####
##### PotentialTemperatureFormulation
#####

"""
$(TYPEDEF)

`PotentialTemperatureFormulation` uses a potential temperature density `ρθ` as the prognostic
thermodynamic variable; its `flavor` selects which potential temperature. The rest of this docstring
describes [`LiquidIce`](@ref), the default flavor, also known as
`LiquidIcePotentialTemperatureFormulation`; see [`DryPotentialTemperatureFormulation`](@ref)
for the [`Dry`](@ref) flavor.

`LiquidIcePotentialTemperatureFormulation` uses liquid-ice potential temperature density `ρθ`
as the prognostic thermodynamic variable.

The liquid-ice potential temperature is defined exactly from the enthalpy,

```math
θˡⁱ = \\frac{H}{ĉᵖ Π} , \\qquad H = cᵖᵐ T - ℒˡ₀ qˡ - ℒⁱ₀ qⁱ , \\qquad Π = \\left( \\frac{p}{pˢᵗ} \\right)^κ ,
```

where ``ĉᵖ = qᵈ cᵖᵈ + qᵗ cᵖᵛ`` is the vapor-equivalent heat capacity (the heat capacity the
parcel would have if all of its water were vapor) and ``ℒˡ₀``, ``ℒⁱ₀`` are the latent heats
extrapolated to absolute zero. With Kirchhoff's law ``ℒ(T) = ℒ₀ + (cᵖᵛ - cˣ) T`` this is
``θˡⁱ = [T - (ℒˡ(T) qˡ + ℒⁱ(T) qⁱ) / ĉᵖ] / Π``, and ``θˡⁱ = T / Π`` in unsaturated air.
The exponent ``κ`` is chosen by `exner_exponent`:

* [`VaporEquivalentExponent`](@ref Breeze.Thermodynamics.VaporEquivalentExponent) (default):
  ``κ̂ = R̂ / ĉᵖ`` with ``R̂ = qᵈ Rᵈ + qᵗ Rᵛ``.
* [`DryExponent`](@ref Breeze.Thermodynamics.DryExponent): ``κᵈ = Rᵈ / cᵖᵈ``, so that ``Π``
  depends on pressure alone.

Because ``ĉᵖ Π`` depends on the composition only through ``qᵗ``, any phase change at fixed
``θˡⁱ``, pressure, and total water conserves the enthalpy exactly and releases the latent heat
``ℒ(T)``. Expansion at fixed composition is made consistent with the first law by the source

```math
ρ \\frac{Rᵐ T - κ ĉᵖ Π θˡⁱ}{ĉᵖ Π} \\frac{D \\ln p}{Dt} ,
```

which the dynamics add to the ``ρθ`` equation with ``D \\ln p / Dt ≈ - g ρ w / p``; it vanishes
in unsaturated air with the default exponent. Energy sources ``Q`` enter as ``Q / (ĉᵖ Π)``.

Recovering temperature from θˡⁱ may require an iterative inversion, depending on the
dynamics: with prognostic-density (compressible) dynamics, temperature solves the implicit
relation `cᵖᵐ T = ĉᵖ (ρRᵐT/pˢᵗ)^κ θ + ℒˡ₀ qˡ + ℒⁱ₀ qⁱ`. The inversion is controlled by `temperature_solver`:

* `DefaultTemperatureSolver()` (default): resolved at materialization to
  [`default_temperature_solver(dynamics)`](@ref Breeze.AtmosphereModels.default_temperature_solver) —
  `nothing` for anelastic dynamics (closed-form inversion) and `NewtonSolver()` for
  compressible dynamics.
* [`NewtonSolver`](@ref Breeze.Solvers.NewtonSolver): tolerance-based Newton iteration.
* [`FixedIterations`](@ref Breeze.Solvers.FixedIterations): a fixed number of Newton steps with no convergence test,
  which unrolls to straight-line code (required for Reactant tracing and cheap
  reverse-mode differentiation).
* `nothing`: the non-iterated closed-form inversion.

Three optional sources close the energy budget further (see
[Optional energy-closing sources](@ref theta-li-energy-sources)). Each is off by default:

* `precipitation_heating` ([`PrecipitationDragHeating`](@ref Breeze.AtmosphereModels.PrecipitationDragHeating)):
  the potential energy that falling condensate loses, ``g Σₓ ρqˣ |wˣ|``, heats the air it falls
  through, on either dynamical core.
* `fixed_volume_phase_change` ([`FixedVolumePhaseChange`](@ref Breeze.AtmosphereModels.FixedVolumePhaseChange)):
  on the compressible core, the source ``ρθ̇ = - Σ_q 𝒥_q Ṡ_q / 𝒥_Θ`` that makes the microphysical
  phase changes conserve the internal energy at fixed volume. Ignored on the anelastic core, where
  phase change at fixed ``θˡⁱ`` and reference pressure is exact.
* `dissipative_heating` ([`SubgridDissipativeHeating`](@ref Breeze.AtmosphereModels.SubgridDissipativeHeating)):
  the kinetic energy that the turbulence closure dissipates, ``ρ ε = - 𝒯ᵢⱼ ∂ⱼ uᵢ``, heats the air.

Each enters the ``ρθ`` tendency as a heating ``Q`` does, ``(ρᵈ / ρ) Q / (ĉᵖ Π)``, except the
phase-change source, which is formed from the fixed-volume Jacobians directly.
"""
struct PotentialTemperatureFormulation{L, F, T, S, E, P, V, D}
    flavor :: L                         # LiquidIce or Dry: which potential temperature is prognostic
    potential_temperature_density :: F  # ρθ (prognostic)
    potential_temperature :: T          # θ = ρθ / ρ (diagnostic)
    temperature_solver :: S             # solver for the θˡⁱ→T inversion (or Nothing when closed-form)
    exner_exponent :: E                       # Exner exponent (VaporEquivalentExponent or DryExponent)
    precipitation_heating :: P          # PrecipitationDragHeating or Nothing
    fixed_volume_phase_change :: V      # FixedVolumePhaseChange or Nothing
    dissipative_heating :: D            # SubgridDissipativeHeating or Nothing
end

const LiquidIcePotentialTemperatureFormulation = PotentialTemperatureFormulation{LiquidIce}
const DryPotentialTemperatureFormulation = PotentialTemperatureFormulation{<:Dry}

LiquidIcePotentialTemperatureFormulation(ρθ, θ, temperature_solver, exner_exponent, precipitation_heating,
                                         fixed_volume_phase_change, dissipative_heating) =
    PotentialTemperatureFormulation(LiquidIce(), ρθ, θ, temperature_solver, exner_exponent, precipitation_heating,
                                    fixed_volume_phase_change, dissipative_heating)

"""
$(TYPEDSIGNATURES)

Return a `LiquidIcePotentialTemperatureFormulation` with the given `temperature_solver` and
Exner exponent `exner_exponent`, either `VaporEquivalentExponent()` (the default) or `DryExponent()`.
`precipitation_heating`, `fixed_volume_phase_change` and `dissipative_heating` turn on the
optional energy-closing sources (`true`, or the option's instance) or leave them off (`false`,
the default, or `nothing`).
The prognostic and diagnostic fields are materialized later in the model constructor.

```jldoctest
using Breeze

LiquidIcePotentialTemperatureFormulation(temperature_solver = FixedIterations(2))

# output
LiquidIcePotentialTemperatureFormulation
├── exner_exponent: VaporEquivalentExponent()
├── temperature_solver: FixedIterations(2)
├── precipitation_heating: false
├── fixed_volume_phase_change: false
└── dissipative_heating: false
```

```jldoctest
using Breeze

LiquidIcePotentialTemperatureFormulation(exner_exponent = DryExponent(),
                                         precipitation_heating = true,
                                         fixed_volume_phase_change = true,
                                         dissipative_heating = true)

# output
LiquidIcePotentialTemperatureFormulation
├── exner_exponent: DryExponent()
├── temperature_solver: DefaultTemperatureSolver
├── precipitation_heating: PrecipitationDragHeating()
├── fixed_volume_phase_change: FixedVolumePhaseChange()
└── dissipative_heating: SubgridDissipativeHeating()
```
"""
LiquidIcePotentialTemperatureFormulation(; temperature_solver = DefaultTemperatureSolver(),
                                           exner_exponent = VaporEquivalentExponent(),
                                           precipitation_heating = false,
                                           fixed_volume_phase_change = false,
                                           dissipative_heating = false) =
    LiquidIcePotentialTemperatureFormulation(nothing, nothing, temperature_solver, validate_exponent(exner_exponent),
                                             energy_source_option(precipitation_heating, PrecipitationDragHeating),
                                             energy_source_option(fixed_volume_phase_change, FixedVolumePhaseChange),
                                             energy_source_option(dissipative_heating, SubgridDissipativeHeating))

"""
$(TYPEDSIGNATURES)

Return a `PotentialTemperatureFormulation` of the given `flavor`, [`LiquidIce`](@ref) (the default,
see `LiquidIcePotentialTemperatureFormulation`) or [`Dry`](@ref) (see
[`DryPotentialTemperatureFormulation`](@ref)). The keyword arguments are those of the flavor's constructor.

```jldoctest
using Breeze
using Breeze.PotentialTemperatureFormulations: Dry

PotentialTemperatureFormulation(flavor = Dry())

# output
DryPotentialTemperatureFormulation
├── flavor: Dry(pressure_work = IsochoricPressureWork(), condensate_enthalpy_flux = true)
├── exner_exponent: DryExponent()
├── precipitation_heating: false
└── dissipative_heating: false
```
"""
PotentialTemperatureFormulation(; flavor = LiquidIce(), kw...) = potential_temperature_formulation(flavor; kw...)

potential_temperature_formulation(::LiquidIce; kw...) = LiquidIcePotentialTemperatureFormulation(; kw...)
potential_temperature_formulation(flavor::Dry; kw...) = dry_potential_temperature_formulation(flavor; kw...)

function dry_potential_temperature_formulation(flavor; precipitation_heating = false, dissipative_heating = false)
    return PotentialTemperatureFormulation(flavor, nothing, nothing, nothing, DryExponent(),
                                           energy_source_option(precipitation_heating, PrecipitationDragHeating),
                                           FixedVolumePhaseChange(),
                                           energy_source_option(dissipative_heating, SubgridDissipativeHeating))
end

"""
$(TYPEDSIGNATURES)

Return a `PotentialTemperatureFormulation` that evolves the density `ρθ = ρᵈ θ` of the potential
temperature ``θ = T / Π``, ``Π = (p / pˢᵗ)^{κᵈ}``, of CM1 (`eqtset = 2`, [Bryan and Fritsch (2002)](@cite BryanFritsch2002)).
Available on `CompressibleDynamics` only. The temperature is the closed form of the equation of state,
``T = θ^{γᵈ} (ρ Rᵐ / pˢᵗ)^{γᵈ - 1}``, so no temperature solver is needed.

Unlike ``θˡⁱ``, ``θ`` changes under phase change, and this source is always on (the
`fixed_volume_phase_change` flag of the liquid-ice flavor has no counterpart here).

- `pressure_work`: [`IsochoricPressureWork`](@ref) (default, CM1's form) or
  [`IsobaricPressureWork`](@ref) (the hydrostatic form used by the liquid-ice flavor).
- `condensate_enthalpy_flux`: whether falling condensate carries its enthalpy (default `true`, CM1's
  `efall = 1`, which is what the other formulations do; CM1's namelists default to `efall = 0`).
- `precipitation_heating`, `dissipative_heating`: as for `LiquidIcePotentialTemperatureFormulation`.

```jldoctest
using Breeze

DryPotentialTemperatureFormulation()

# output
DryPotentialTemperatureFormulation
├── flavor: Dry(pressure_work = IsochoricPressureWork(), condensate_enthalpy_flux = true)
├── exner_exponent: DryExponent()
├── precipitation_heating: false
└── dissipative_heating: false
```
"""
DryPotentialTemperatureFormulation(; pressure_work = IsochoricPressureWork(), condensate_enthalpy_flux = true, kw...) =
    dry_potential_temperature_formulation(Dry(; pressure_work, condensate_enthalpy_flux); kw...)

validate_exponent(κ̂::AbstractExnerExponent) = κ̂

validate_exponent(κ̂) =
    throw(ArgumentError("exner_exponent must be VaporEquivalentExponent() or DryExponent(), got $κ̂"))

# An energy-closing source is selected by a Bool, its option instance, or `nothing`, and stored as
# the instance or `nothing` so kernels dispatch on it rather than branch.
energy_source_option(flag::Bool, Option) = flag ? Option() : nothing
energy_source_option(::Nothing, Option) = nothing
energy_source_option(option, Option) = option isa Option ? option :
    throw(ArgumentError("expected a Bool, nothing or $(nameof(Option))(), got $option"))

AtmosphereModels.liquid_ice_exner_exponent(formulation::PotentialTemperatureFormulation) = formulation.exner_exponent

AtmosphereModels.precipitation_heating(formulation::PotentialTemperatureFormulation) = formulation.precipitation_heating

Adapt.adapt_structure(to, formulation::PotentialTemperatureFormulation) =
    PotentialTemperatureFormulation(formulation.flavor,
                                    adapt(to, formulation.potential_temperature_density),
                                             adapt(to, formulation.potential_temperature),
                                             formulation.temperature_solver,
                                             formulation.exner_exponent,
                                             formulation.precipitation_heating,
                                             formulation.fixed_volume_phase_change,
                                             formulation.dissipative_heating)

function BoundaryConditions.fill_halo_regions!(formulation::PotentialTemperatureFormulation)
    fill_halo_regions!(formulation.potential_temperature)
    return nothing
end

#####
##### Field naming interface
#####

AtmosphereModels.prognostic_thermodynamic_field_names(::PotentialTemperatureFormulation) = tuple(:ρθ)
AtmosphereModels.additional_thermodynamic_field_names(::PotentialTemperatureFormulation) = tuple(:θ)
AtmosphereModels.thermodynamic_density_name(::PotentialTemperatureFormulation) = :ρθ
AtmosphereModels.thermodynamic_density(formulation::PotentialTemperatureFormulation) = formulation.potential_temperature_density
AtmosphereModels.with_thermodynamic_density(f::PotentialTemperatureFormulation, ρθ) =
    PotentialTemperatureFormulation(f.flavor, ρθ, f.potential_temperature, f.temperature_solver, f.exner_exponent,
                                             f.precipitation_heating, f.fixed_volume_phase_change,
                                             f.dissipative_heating)

AtmosphereModels.set_hydrostatically_balanced_density!(
    formulation::PotentialTemperatureFormulation,
    model,
    spec::AtmosphereModels.HydrostaticallyBalancedDensity) =
    AtmosphereModels.set_potential_temperature_hydrostatically_balanced_density!(
        model, spec, formulation.potential_temperature, formulation.exner_exponent)

# Val-based versions for pre-materialization (called via Symbol fallback in interface)
AtmosphereModels.prognostic_thermodynamic_field_names(::Val{:LiquidIcePotentialTemperature}) = tuple(:ρθ)
AtmosphereModels.additional_thermodynamic_field_names(::Val{:LiquidIcePotentialTemperature}) = tuple(:θ)
AtmosphereModels.thermodynamic_density_name(::Val{:LiquidIcePotentialTemperature}) = :ρθ
AtmosphereModels.prognostic_thermodynamic_field_names(::Val{:DryPotentialTemperature}) = tuple(:ρθ)
AtmosphereModels.additional_thermodynamic_field_names(::Val{:DryPotentialTemperature}) = tuple(:θ)
AtmosphereModels.thermodynamic_density_name(::Val{:DryPotentialTemperature}) = :ρθ

Oceananigans.fields(formulation::PotentialTemperatureFormulation) = (; θ=formulation.potential_temperature)
Oceananigans.prognostic_fields(formulation::PotentialTemperatureFormulation) = (; ρθ=formulation.potential_temperature_density)

#####
##### Materialization
#####

AtmosphereModels.materialize_formulation(::Val{:LiquidIcePotentialTemperature}, dynamics, grid, boundary_conditions) =
    materialize_formulation(LiquidIcePotentialTemperatureFormulation(), dynamics, grid, boundary_conditions)

function AtmosphereModels.materialize_formulation(formulation::LiquidIcePotentialTemperatureFormulation,
                                                  dynamics, grid, boundary_conditions)
    potential_temperature_density = CenterField(grid, boundary_conditions=boundary_conditions.ρθ)
    potential_temperature = CenterField(grid)  # θ = ρθ / ρ (diagnostic)
    temperature_solver = materialize_temperature_solver(formulation.temperature_solver, dynamics, grid)
    fixed_volume_phase_change = materialize_phase_change_source(formulation.fixed_volume_phase_change, dynamics)
    return LiquidIcePotentialTemperatureFormulation(potential_temperature_density,
                                                    potential_temperature,
                                                    temperature_solver,
                                                    formulation.exner_exponent,
                                                    formulation.precipitation_heating,
                                                    fixed_volume_phase_change,
                                                    formulation.dissipative_heating)
end

AtmosphereModels.materialize_formulation(::Val{:DryPotentialTemperature}, dynamics, grid, boundary_conditions) =
    materialize_formulation(DryPotentialTemperatureFormulation(), dynamics, grid, boundary_conditions)

function AtmosphereModels.materialize_formulation(formulation::DryPotentialTemperatureFormulation,
                                                  dynamics, grid, boundary_conditions)
    # The phase-change source is the only latent heating of ρθ, so it cannot be dropped: dynamics on
    # which it does not apply (a prescribed thermodynamic pressure) cannot carry this flavor.
    materialize_phase_change_source(formulation.fixed_volume_phase_change, dynamics) === nothing &&
        throw(ArgumentError("DryPotentialTemperatureFormulation needs CompressibleDynamics, got $(summary(dynamics))"))

    potential_temperature_density = CenterField(grid, boundary_conditions=boundary_conditions.ρθ)
    potential_temperature = CenterField(grid)  # θ = ρθ / ρᵈ (diagnostic)
    return PotentialTemperatureFormulation(formulation.flavor,
                                           potential_temperature_density,
                                           potential_temperature,
                                           nothing,
                                           formulation.exner_exponent,
                                           formulation.precipitation_heating,
                                           formulation.fixed_volume_phase_change,
                                           formulation.dissipative_heating)
end

# `DefaultTemperatureSolver` defers the choice of solver to the dynamics: the need for an
# iterative θˡⁱ→T inversion is dictated by the intersection of the formulation and the
# dynamics (anelastic: closed-form, no solver; compressible: implicit, Newton by default).
materialize_temperature_solver(::DefaultTemperatureSolver, dynamics, grid) =
    materialize_solver(default_temperature_solver(dynamics), eltype(grid))

materialize_temperature_solver(temperature_solver, dynamics, grid) =
    materialize_solver(temperature_solver, eltype(grid))

#####
##### Auxiliary variable computation
#####

function AtmosphereModels.compute_auxiliary_thermodynamic_variables!(formulation::PotentialTemperatureFormulation, dynamics, i, j, k, grid)
    ρ = dynamics_density(dynamics)
    @inbounds begin
        ρᵢ = ρ[i, j, k]
        ρθ = formulation.potential_temperature_density[i, j, k]
        formulation.potential_temperature[i, j, k] = ρθ / ρᵢ
    end
    return nothing
end

#####
##### Thermodynamic state diagnosis
#####

"""
$(TYPEDSIGNATURES)

Build a `LiquidIcePotentialTemperatureState` at grid point `(i, j, k)` from the
given `formulation`, `dynamics`, and pre-computed moisture mass fractions `q`.
"""
function AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid,
                                                       formulation::LiquidIcePotentialTemperatureFormulation,
                                                       dynamics,
                                                       q)

    θ = @inbounds formulation.potential_temperature[i, j, k]
    pᵣ = @inbounds dynamics_pressure(dynamics)[i, j, k]
    pˢᵗ = standard_pressure(dynamics)

    return LiquidIcePotentialTemperatureState(θ, q, pˢᵗ, pᵣ, formulation.exner_exponent)
end

#####
##### Prognostic field collection
#####

function AtmosphereModels.collect_prognostic_fields(formulation::PotentialTemperatureFormulation,
                                                    dynamics,
                                                    momentum,
                                                    moisture_density,
                                                    moisture_name,
                                                    microphysical_fields,
                                                    tracers)

    ρθ = formulation.potential_temperature_density
    thermodynamic_variables = merge((ρθ=ρθ,), NamedTuple{(moisture_name,)}((moisture_density,)))
    dynamics_fields = dynamics_prognostic_fields(dynamics)
    return merge(dynamics_fields, momentum, thermodynamic_variables, microphysical_fields, tracers)
end

#####
##### Show methods
#####

function Base.summary(::LiquidIcePotentialTemperatureFormulation)
    return "LiquidIcePotentialTemperatureFormulation"
end

function Base.show(io::IO, formulation::LiquidIcePotentialTemperatureFormulation)
    print(io, summary(formulation))
    if formulation.potential_temperature_density !== nothing
        print(io, '\n')
        print(io, "├── potential_temperature_density: ", prettysummary(formulation.potential_temperature_density), '\n')
        print(io, "├── potential_temperature: ", prettysummary(formulation.potential_temperature), '\n')
    else
        print(io, '\n')
    end
    print(io, "├── exner_exponent: ", summary(formulation.exner_exponent), '\n')
    print(io, "├── temperature_solver: ", summary(formulation.temperature_solver), '\n')
    print(io, "├── precipitation_heating: ", energy_source_summary(formulation.precipitation_heating), '\n')
    print(io, "├── fixed_volume_phase_change: ", energy_source_summary(formulation.fixed_volume_phase_change), '\n')
    print(io, "└── dissipative_heating: ", energy_source_summary(formulation.dissipative_heating))
end

energy_source_summary(::Nothing) = "false"
energy_source_summary(option) = summary(option)

Base.summary(::DryPotentialTemperatureFormulation) = "DryPotentialTemperatureFormulation"

function Base.show(io::IO, formulation::DryPotentialTemperatureFormulation)
    print(io, summary(formulation))
    if formulation.potential_temperature_density !== nothing
        print(io, '\n')
        print(io, "├── potential_temperature_density: ", prettysummary(formulation.potential_temperature_density), '\n')
        print(io, "├── potential_temperature: ", prettysummary(formulation.potential_temperature), '\n')
    else
        print(io, '\n')
    end
    print(io, "├── flavor: ", summary(formulation.flavor), '\n')
    print(io, "├── exner_exponent: ", summary(formulation.exner_exponent), '\n')
    print(io, "├── precipitation_heating: ", energy_source_summary(formulation.precipitation_heating), '\n')
    print(io, "└── dissipative_heating: ", energy_source_summary(formulation.dissipative_heating))
end
