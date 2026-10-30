#####
##### LiquidIcePotentialTemperatureFormulation
#####

"""
$(TYPEDEF)

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
"""
struct LiquidIcePotentialTemperatureFormulation{F, T, S, E}
    potential_temperature_density :: F  # ρθ (prognostic)
    potential_temperature :: T          # θ = ρθ / ρ (diagnostic)
    temperature_solver :: S             # solver for the θˡⁱ→T inversion (or Nothing when closed-form)
    exner_exponent :: E                       # Exner exponent (VaporEquivalentExponent or DryExponent)
end

"""
$(TYPEDSIGNATURES)

Return a `LiquidIcePotentialTemperatureFormulation` with the given `temperature_solver` and
Exner exponent `exner_exponent`, either `VaporEquivalentExponent()` (the default) or `DryExponent()`.
The prognostic and diagnostic fields are materialized later in the model constructor.

```jldoctest
using Breeze

LiquidIcePotentialTemperatureFormulation(temperature_solver = FixedIterations(2))

# output
LiquidIcePotentialTemperatureFormulation
├── exner_exponent: VaporEquivalentExponent()
└── temperature_solver: FixedIterations(2)
```

```jldoctest
using Breeze

LiquidIcePotentialTemperatureFormulation(exner_exponent=DryExponent())

# output
LiquidIcePotentialTemperatureFormulation
├── exner_exponent: DryExponent()
└── temperature_solver: DefaultTemperatureSolver
```
"""
LiquidIcePotentialTemperatureFormulation(; temperature_solver = DefaultTemperatureSolver(),
                                           exner_exponent = VaporEquivalentExponent()) =
    LiquidIcePotentialTemperatureFormulation(nothing, nothing, temperature_solver, validate_exponent(exner_exponent))

validate_exponent(κ̂::AbstractExnerExponent) = κ̂

validate_exponent(κ̂) =
    throw(ArgumentError("exner_exponent must be VaporEquivalentExponent() or DryExponent(), got $κ̂"))

AtmosphereModels.liquid_ice_exner_exponent(formulation::LiquidIcePotentialTemperatureFormulation) = formulation.exner_exponent

Adapt.adapt_structure(to, formulation::LiquidIcePotentialTemperatureFormulation) =
    LiquidIcePotentialTemperatureFormulation(adapt(to, formulation.potential_temperature_density),
                                             adapt(to, formulation.potential_temperature),
                                             formulation.temperature_solver,
                                             formulation.exner_exponent)

@regional LiquidIcePotentialTemperatureFormulation

function BoundaryConditions.fill_halo_regions!(formulation::LiquidIcePotentialTemperatureFormulation)
    fill_halo_regions!(formulation.potential_temperature)
    return nothing
end

#####
##### Field naming interface
#####

AtmosphereModels.prognostic_thermodynamic_field_names(::LiquidIcePotentialTemperatureFormulation) = tuple(:ρθ)
AtmosphereModels.additional_thermodynamic_field_names(::LiquidIcePotentialTemperatureFormulation) = tuple(:θ)
AtmosphereModels.thermodynamic_density_name(::LiquidIcePotentialTemperatureFormulation) = :ρθ
AtmosphereModels.thermodynamic_density(formulation::LiquidIcePotentialTemperatureFormulation) = formulation.potential_temperature_density
AtmosphereModels.with_thermodynamic_density(f::LiquidIcePotentialTemperatureFormulation, ρθ) =
    LiquidIcePotentialTemperatureFormulation(ρθ, f.potential_temperature, f.temperature_solver, f.exner_exponent)

AtmosphereModels.set_hydrostatically_balanced_density!(
    formulation::LiquidIcePotentialTemperatureFormulation,
    model,
    spec::AtmosphereModels.HydrostaticallyBalancedDensity) =
    AtmosphereModels.set_potential_temperature_hydrostatically_balanced_density!(
        model, spec, formulation.potential_temperature, formulation.exner_exponent)

# Val-based versions for pre-materialization (called via Symbol fallback in interface)
AtmosphereModels.prognostic_thermodynamic_field_names(::Val{:LiquidIcePotentialTemperature}) = tuple(:ρθ)
AtmosphereModels.additional_thermodynamic_field_names(::Val{:LiquidIcePotentialTemperature}) = tuple(:θ)
AtmosphereModels.thermodynamic_density_name(::Val{:LiquidIcePotentialTemperature}) = :ρθ

Oceananigans.fields(formulation::LiquidIcePotentialTemperatureFormulation) = (; θ=formulation.potential_temperature)
Oceananigans.prognostic_fields(formulation::LiquidIcePotentialTemperatureFormulation) = (; ρθ=formulation.potential_temperature_density)

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
    return LiquidIcePotentialTemperatureFormulation(potential_temperature_density,
                                                    potential_temperature,
                                                    temperature_solver,
                                                    formulation.exner_exponent)
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

function AtmosphereModels.compute_auxiliary_thermodynamic_variables!(formulation::LiquidIcePotentialTemperatureFormulation, dynamics, i, j, k, grid)
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

function AtmosphereModels.collect_prognostic_fields(formulation::LiquidIcePotentialTemperatureFormulation,
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
    print(io, "└── temperature_solver: ", summary(formulation.temperature_solver))
end
