#####
##### Gustiness: the unresolved wind added to the resolved wind in the bulk formulas
#####
##### The bulk fluxes and their transfer coefficients both see the effective wind speed
##### Ũ = √(u² + v² + Uᵍ²), with Uᵍ the gustiness. Gustiness is either a number (a constant
##### speed) or a `TKEGustiness`, which estimates Uᵍ from the subgrid turbulent kinetic energy.
##### The bulk-flux constructors default to the placeholder `DefaultGustiness()`, which the
##### model replaces through `materialize_gustiness(gustiness, closure)` when it materializes its
##### boundary conditions.
#####

# The prognostic TKE density that `TKEBasedTurbulenceClosure` adds to the model as a tracer
# (`TurbulenceClosures.TKE_NAME`; that module is loaded after this one)
const TKE_density_name = :ρe

"""
    DefaultGustiness()

The placeholder gustiness of a bulk-flux boundary condition constructed without one. When the
`AtmosphereModel` materializes its boundary conditions it is replaced by
[`materialize_gustiness`](@ref): with [`TKEGustiness`](@ref) when the closure is a
`TKEBasedTurbulenceClosure`, and with a constant gustiness of zero otherwise.

```jldoctest
using Breeze.BoundaryConditions: DefaultGustiness, materialize_gustiness

materialize_gustiness(DefaultGustiness(), nothing)

# output
0
```
"""
struct DefaultGustiness end

struct TKEGustiness{FT}
    coefficient :: FT
    minimum :: FT
end

"""
$(TYPEDSIGNATURES)

A gustiness ``Uᵍ`` estimated from the subgrid turbulent kinetic energy ``e`` of a
`TKEBasedTurbulenceClosure` in the cell next to the wall,

```math
Uᵍ = \\max(c \\sqrt{e}, Uᵍ_{min}),
```

where ``c`` is the `coefficient` and ``Uᵍ_{min}`` the `minimum`. The default
``c = \\sqrt{2/3}`` is the isotropic estimate: for isotropic turbulence
``e = (u'^2 + v'^2 + w'^2) / 2`` and the variance of each velocity component is ``(2/3) e``,
so ``c \\sqrt{e}`` is the root-mean-square of one horizontal component of the subgrid wind.
The default `minimum` of 0.5 m/s is the floor [Large and Yeager (2009)](@cite LargeYeager2009)
place on the wind speed in their bulk formulas.

The gustiness enters the bulk fluxes through the effective wind ``Ũ = \\sqrt{u² + v² + Uᵍ²}``,
which both the flux and its transfer coefficient see. It requires the prognostic TKE density
`ρe` of a `TKEBasedTurbulenceClosure`, and is the gustiness a bulk-flux boundary condition
takes by default under one (see [`DefaultGustiness`](@ref)).

# Keyword Arguments

- `coefficient`: The coefficient ``c`` in ``Uᵍ = c \\sqrt{e}`` (default: ``\\sqrt{2/3}``).
- `minimum`: The floor ``Uᵍ_{min}`` on the gustiness, in m/s (default: 0.5).

# Example

```jldoctest
using Breeze

TKEGustiness(minimum=0.2)

# output
TKEGustiness{Float64}(coefficient=0.816497, minimum=0.2)
```
"""
function TKEGustiness(FT = Oceananigans.defaults.FloatType; coefficient = sqrt(2/3), minimum = 0.5)
    coefficient ≥ 0 || throw(ArgumentError("TKEGustiness coefficient must be non-negative, got $coefficient"))
    minimum ≥ 0 || throw(ArgumentError("TKEGustiness minimum must be non-negative, got $minimum"))
    return TKEGustiness(convert(FT, coefficient), convert(FT, minimum))
end

Base.summary(::DefaultGustiness) = "DefaultGustiness"
Base.show(io::IO, ::DefaultGustiness) = print(io, "DefaultGustiness()")

Base.summary(g::TKEGustiness{FT}) where FT =
    string("TKEGustiness{", FT, "}(coefficient=", prettysummary(g.coefficient),
           ", minimum=", prettysummary(g.minimum), ")")

Base.show(io::IO, g::TKEGustiness) = print(io, summary(g))

"""
$(TYPEDSIGNATURES)

Return the gustiness a bulk-flux boundary condition uses in a model with turbulence `closure`.
A [`DefaultGustiness`](@ref) placeholder becomes a [`TKEGustiness`](@ref) under a
`TKEBasedTurbulenceClosure` (a method the `TurbulenceClosures` module adds) and a constant
gustiness of zero under any other closure, or none; a gustiness that was given, a number or a
`TKEGustiness`, is returned unchanged.

```jldoctest
using Breeze
using Breeze.BoundaryConditions: DefaultGustiness, materialize_gustiness

materialize_gustiness(DefaultGustiness(), TKEBasedTurbulenceClosure())

# output
TKEGustiness{Float64}(coefficient=0.816497, minimum=0.5)
```
"""
materialize_gustiness(gustiness, closure) = gustiness

# The constant default for closures without a prognostic TKE
materialize_gustiness(::DefaultGustiness, closure) = 0

# Several closures: TKE gustiness if any of them carries a prognostic TKE
function materialize_gustiness(gustiness::DefaultGustiness, closures::Tuple)
    candidates = map(closure -> materialize_gustiness(gustiness, closure), closures)
    tke = filter(g -> g isa TKEGustiness, candidates)
    return isempty(tke) ? materialize_gustiness(gustiness, nothing) : first(tke)
end

function materialize_gustiness(gustiness::TKEGustiness, closure)
    TKE_density_name ∈ closure_required_tracers(closure) ||
        throw(ArgumentError("TKEGustiness reads the subgrid turbulent kinetic energy of a " *
                            "TKEBasedTurbulenceClosure, but the model closure is $(summary(closure))."))
    return gustiness
end

# TODO: a convective-velocity-scale gustiness, Uᵍ = β w★ with β ≈ 1.2 and the Deardorff velocity
# w★ = (B h)^(1/3) from the surface buoyancy flux B and the boundary-layer depth h (Beljaars 1995;
# the COARE algorithm of Fairall et al. 1996). It needs the buoyancy flux the bulk formulas are
# computing, so an iteration or a lagged flux stored in a field, and a diagnosed boundary-layer
# depth that Breeze does not yet provide.

#####
##### Gustiness in the kernels: the squared gustiness Uᵍ² next to the wall
#####

@inline gustiness²(i, j, k, grid, Uᵍ::Number, fields) = Uᵍ^2

# The TKE of the near-wall cell, e = ρe / ρ, with `fields.ρ` the total density from
# `dynamics_thermodynamic_fields`. The bulk fluxes are evaluated outside `update_state!`, while the
# tracers hold their densities.
@inline function gustiness²(i, j, k, grid, gustiness::TKEGustiness, fields)
    ρe = @inbounds getproperty(fields, TKE_density_name)[i, j, k]
    ρ = @inbounds fields.ρ[i, j, k]
    e = max(0, ρe / ρ)
    c = gustiness.coefficient
    Uᵍ_min = gustiness.minimum
    return max(c^2 * e, Uᵍ_min^2)
end
