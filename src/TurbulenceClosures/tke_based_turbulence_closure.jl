#####
##### `TKEBasedTurbulenceClosure`: a vertical eddy-diffusivity closure with prognostic turbulent
##### kinetic energy, in the spirit of CATKE (Wagner et al. 2025)
#####
#####   Kᵘ = Sᵘ ℓ √e,   Kᶜ = Sᶜ ℓ √e,   Kᵉ = Sᵉ ℓ √e,   ε = Sᴰ e^{3/2} / ℓ
#####   ℓ  = min(z, Cᴺ √e / N)
#####   ∂ₜ(ρe) + ∇·(ρ u e) = ∂z(ρ Kᵉ ∂z e) + ρ (P + B − ε),   P = Kᵘ S²,  B = −Kᶜ N²
#####
##### The stability functions Sᵘ, Sᶜ, Sᵉ, Sᴰ are constants for now (`ConstantStabilityFunctions`);
##### a Richardson-number-dependent variant is a new type plus four methods.
#####
##### The tracer `ρe` is advected and vertically diffused (with Kᵉ) by the dynamical core like every
##### other scalar. Following CATKE, the sinks — dissipation, the negative part of the buoyancy flux
##### and the damping of negative TKE — enter the same vertically implicit tridiagonal solve as the
##### diffusion, through the linear coefficient `Lᵉ` and `implicit_linear_coefficient`, and the
##### sources — shear production and the positive part of the buoyancy flux — enter the stage
##### tendency through `compute_closure_tendencies!`. Under an explicit time discretization the
##### sinks are added to the tendency as well.
#####

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    shearᶜᶜᶠ, ℑbzᵃᵃᶜ, mask_diffusivity, turbulent_velocityᶜᶜᶜ

using Oceananigans.TurbulenceClosures: clip, height_above_bottomᶜᶜᶠ, height_above_bottomᶜᶜᶜ
using Oceananigans.Units: minute

#####
##### Stability functions
#####

"""
$(TYPEDEF)

Constant stability functions for [`TKEBasedTurbulenceClosure`](@ref): the mixing lengths for
momentum, tracers and turbulent kinetic energy are constant multiples of the primary length ``ℓ``,

```math
ℓᵘ = Cᵘ ℓ, \\qquad ℓᶜ = Cᶜ ℓ, \\qquad ℓᵉ = Cᵉ ℓ,
```

and the dissipation length is ``ℓᴰ = ℓ / Cᴰ``, so that ``ε = Cᴰ e^{3/2} / ℓ``.

The turbulent Prandtl number is ``Pr = Cᵘ / Cᶜ`` and the TKE Schmidt number ``Cᵘ / Cᵉ``. In a
neutral constant-stress layer, where ``ℓ = z``, production balances dissipation at
``e / u_\\star² = 1 / \\sqrt{Cᵘ Cᴰ}`` with a logarithmic wind profile of von Kármán constant
``κ = (Cᵘ³ / Cᴰ)^{1/4}``; in stratified steady state the gradient Richardson number is
``Ri^\\dagger = Cᵘ Cᴺ² / (Cᶜ Cᴺ² + Cᴰ)``, with ``Cᴺ`` the coefficient of the stratification
length.

The defaults are the Mellor–Yamada coefficients of [Nakanishi and Niino (2009)](@cite NakanishiNiino2009)
re-expressed with the von Kármán constant absorbed (``κ = 0.4``, ``e/u_\\star² = 4.2``,
``Pr = 0.74``); they are placeholders for calibration.

Fields
======

$(TYPEDFIELDS)
"""
Base.@kwdef struct ConstantStabilityFunctions{FT}
    "momentum stability function, ``ℓᵘ = Cᵘ ℓ``"
    Cᵘ :: FT = 0.196
    "tracer stability function, ``ℓᶜ = Cᶜ ℓ``"
    Cᶜ :: FT = 0.265
    "turbulent kinetic energy stability function, ``ℓᵉ = Cᵉ ℓ``"
    Cᵉ :: FT = 0.392
    "dissipation stability function, ``ℓᴰ = ℓ / Cᴰ``"
    Cᴰ :: FT = 0.295
end

## `Base.@kwdef` on a `{FT}` struct requires every field to share one type, so mixed
## integer/float keyword arguments — `ConstantStabilityFunctions(Cᵉ = 1)` — are promoted here.
ConstantStabilityFunctions(Cᵘ, Cᶜ, Cᵉ, Cᴰ) = ConstantStabilityFunctions(promote(Cᵘ, Cᶜ, Cᵉ, Cᴰ)...)

Base.summary(sf::ConstantStabilityFunctions{FT}) where FT = "ConstantStabilityFunctions{$FT}"

Base.show(io::IO, sf::ConstantStabilityFunctions) =
    print(io, summary(sf), '\n',
              "├── Cᵘ: ", prettysummary(sf.Cᵘ), '\n',
              "├── Cᶜ: ", prettysummary(sf.Cᶜ), '\n',
              "├── Cᵉ: ", prettysummary(sf.Cᵉ), '\n',
              "└── Cᴰ: ", prettysummary(sf.Cᴰ))

#####
##### Mixing length
#####

"""
$(TYPEDEF)

The primary mixing length of [`TKEBasedTurbulenceClosure`](@ref),

```math
ℓ = \\min(z, \\, Cᴺ \\sqrt{e} / N),
```

the smaller of the height above the surface ``z`` and the stratification length
``ℓᴺ = Cᴺ \\sqrt{e} / N``: the distance a parcel with kinetic energy ``e`` travels against a
stable stratification of buoyancy frequency ``N``. In neutral or unstable air ``ℓᴺ`` is infinite
and ``ℓ = z``. The height above the surface carries no coefficient; the stability functions set the
scale of each diffusivity. The default ``Cᴺ = 0.76`` is Deardorff's
([Deardorff 1980](@cite Deardorff1980)).
"""
Base.@kwdef struct TKEMixingLength{FT}
    Cᴺ :: FT = 0.76 # coefficient of the stratification length ℓᴺ = Cᴺ √e / N
end

Base.summary(ml::TKEMixingLength{FT}) where FT = "TKEMixingLength{$FT}"
Base.show(io::IO, ml::TKEMixingLength) = print(io, summary(ml), " (Cᴺ = ", prettysummary(ml.Cᴺ), ")")

#####
##### The closure
#####

"""
$(TYPEDEF)

A vertical eddy-diffusivity closure carrying one prognostic equation for the subgrid turbulent
kinetic energy ``e``, in the spirit of CATKE ([Wagner et al. 2025](@cite Wagner25catke)):

```math
Kᵘ = Sᵘ ℓ \\sqrt{e}, \\qquad Kᶜ = Sᶜ ℓ \\sqrt{e}, \\qquad Kᵉ = Sᵉ ℓ \\sqrt{e}, \\qquad
ε = Sᴰ e^{3/2} / ℓ,
```

```math
∂_t (ρ e) + ∇ ⋅ (ρ 𝐮 e) = ∂_z (ρ Kᵉ ∂_z e) + ρ (P + B - ε), \\qquad P = Kᵘ S², \\qquad B = -Kᶜ N²,
```

where ``Kᵘ``, ``Kᶜ`` and ``Kᵉ`` are the eddy diffusivities of momentum, scalars and turbulent
kinetic energy, ``S²`` the squared vertical shear, ``N²`` the squared buoyancy frequency, ``ℓ`` the
primary mixing length ([`TKEMixingLength`](@ref)), and ``Sᵘ, Sᶜ, Sᵉ, Sᴰ`` stability functions
([`ConstantStabilityFunctions`](@ref)). The prognostic TKE density is the tracer `ρe`, which the
closure adds to the model; it is advected and vertically diffused like every other scalar, and the
closure applies the local production, buoyancy flux and dissipation.

The square root of ``e`` is floored at `minimum_tke` wherever it enters a diffusivity or a length
scale, and negative ``e`` — which advection can produce — is damped on
`negative_tke_damping_time_scale` rather than clipped. The three `maximum_*` diffusivities clip
the diffusivities, `Inf` by default.
"""
struct TKEBasedTurbulenceClosure{TD, ML, SF, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    mixing_length :: ML                   # the primary mixing length ℓ
    stability_functions :: SF             # Sᵘ, Sᶜ, Sᵉ, Sᴰ
    maximum_viscosity :: FT               # upper bound on Kᵘ, m² s⁻¹
    maximum_tracer_diffusivity :: FT      # upper bound on Kᶜ, m² s⁻¹
    maximum_tke_diffusivity :: FT         # upper bound on Kᵉ, m² s⁻¹
    minimum_tke :: FT                     # floor on e inside √e, m² s⁻²
    negative_tke_damping_time_scale :: FT # time scale on which negative e is damped, s
end

const TKEClosureArray{TD} = AbstractArray{<:TKEBasedTurbulenceClosure{TD}} where TD

"""Either a single `TKEBasedTurbulenceClosure` or an ensemble array of them."""
const FlavorOfTKEClosure{TD} = Union{TKEBasedTurbulenceClosure{TD}, TKEClosureArray{TD}} where TD

function TKEBasedTurbulenceClosure{TD}(mixing_length::ML, stability_functions::SF,
                                       maximum_viscosity::FT, maximum_tracer_diffusivity::FT,
                                       maximum_tke_diffusivity::FT, minimum_tke::FT,
                                       negative_tke_damping_time_scale::FT) where {TD, ML, SF, FT}
    return TKEBasedTurbulenceClosure{TD, ML, SF, FT}(mixing_length, stability_functions,
                                                     maximum_viscosity, maximum_tracer_diffusivity,
                                                     maximum_tke_diffusivity, minimum_tke,
                                                     negative_tke_damping_time_scale)
end

"""
$(TYPEDSIGNATURES)

Construct a [`TKEBasedTurbulenceClosure`](@ref) with the given time discretization (default
`VerticallyImplicitTimeDiscretization()`), float type, mixing length, stability functions and
numerical parameters.
"""
function TKEBasedTurbulenceClosure(time_discretization::TD = VerticallyImplicitTimeDiscretization(),
                                   FT = Oceananigans.defaults.FloatType;
                                   mixing_length = TKEMixingLength(),
                                   stability_functions = ConstantStabilityFunctions(),
                                   maximum_viscosity = Inf,
                                   maximum_tracer_diffusivity = Inf,
                                   maximum_tke_diffusivity = Inf,
                                   minimum_tke = 1e-6,
                                   # CATKE's value; atmospheric turbulence evolves faster than the
                                   # ocean's, so a shorter time scale may be more appropriate here.
                                   negative_tke_damping_time_scale = 1minute) where TD

    mixing_length = convert_eltype(FT, mixing_length)
    stability_functions = convert_eltype(FT, stability_functions)

    return TKEBasedTurbulenceClosure{TD}(mixing_length,
                                         stability_functions,
                                         convert(FT, maximum_viscosity),
                                         convert(FT, maximum_tracer_diffusivity),
                                         convert(FT, maximum_tke_diffusivity),
                                         convert(FT, minimum_tke),
                                         convert(FT, negative_tke_damping_time_scale))
end

TKEBasedTurbulenceClosure(FT::DataType; kw...) =
    TKEBasedTurbulenceClosure(VerticallyImplicitTimeDiscretization(), FT; kw...)

@inline convert_eltype(::Type{FT}, ml::TKEMixingLength) where FT = TKEMixingLength{FT}(convert(FT, ml.Cᴺ))
@inline convert_eltype(::Type{FT}, sf::ConstantStabilityFunctions) where FT =
    ConstantStabilityFunctions{FT}(convert(FT, sf.Cᵘ), convert(FT, sf.Cᶜ), convert(FT, sf.Cᵉ), convert(FT, sf.Cᴰ))

#####
##### Tracer wiring
#####

"""The name of the prognostic TKE tracer, which holds the density ``ρ e``."""
const TKE_NAME = :ρe

Oceananigans.TurbulenceClosures.closure_required_tracers(::FlavorOfTKEClosure) = tuple(TKE_NAME)

function Utils.with_tracers(tracer_names, closure::FlavorOfTKEClosure)
    TKE_NAME ∈ tracer_names ||
        throw(ArgumentError("Tracers must contain :$(TKE_NAME) to represent turbulent kinetic " *
                            "energy for `TKEBasedTurbulenceClosure`."))
    return closure
end

#####
##### Closure fields
#####

"""
$(TYPEDEF)

Precomputed fields for [`TKEBasedTurbulenceClosure`](@ref). The mixing length is not stored;
like CATKE, the closure computes it on the fly wherever it is needed; evaluating
`mixing_lengthᶜᶜᶠ` in a `KernelFunctionOperation` diagnoses it from the model state.
"""
struct TKEClosureFields{K, L, KC, LC}
    Kᵘ :: K # eddy diffusivity for momentum, at (Center, Center, Face)
    Kᶜ :: K # eddy diffusivity for scalars, at (Center, Center, Face)
    Kᵉ :: K # eddy diffusivity for turbulent kinetic energy, at (Center, Center, Face)
    # The linear implicit coefficient of the TKE equation, ∂ₜe = Lᵉ e + ⋯, at (Center, Center, Center):
    # the dissipation rate, the negative part of the buoyancy flux and the damping of negative TKE.
    Lᵉ :: L
    tupled_tracer_diffusivities :: KC         # per-tracer diffusivity lookup, by closure-scalar position
    tupled_implicit_linear_coefficients :: LC # `Lᵉ` for the TKE tracer, zero for every other
end

Adapt.adapt_structure(to, fields::TKEClosureFields) =
    TKEClosureFields(adapt(to, fields.Kᵘ),
                     adapt(to, fields.Kᶜ),
                     adapt(to, fields.Kᵉ),
                     adapt(to, fields.Lᵉ),
                     adapt(to, fields.tupled_tracer_diffusivities),
                     adapt(to, fields.tupled_implicit_linear_coefficients))

BoundaryConditions.fill_halo_regions!(fields::TKEClosureFields, args...; kw...) =
    fill_halo_regions!((fields.Kᵘ, fields.Kᶜ, fields.Kᵉ), args...; kw...)

function Oceananigans.TurbulenceClosures.build_closure_fields(grid, clock, tracer_names, bcs, closure::FlavorOfTKEClosure)
    face_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()))
    default_bcs = (Kᵘ = face_bcs, Kᶜ = face_bcs, Kᵉ = face_bcs)
    bcs = merge(default_bcs, bcs)

    Kᵘ = ZFaceField(grid, boundary_conditions=bcs.Kᵘ)
    Kᶜ = ZFaceField(grid, boundary_conditions=bcs.Kᶜ)
    Kᵉ = ZFaceField(grid, boundary_conditions=bcs.Kᵉ)
    Lᵉ = CenterField(grid)

    # Indexed by the `Val(id)` the model hands to `diffusivity` and `implicit_linear_coefficient`:
    # TKE is transported with `Kᵉ` and damped with `Lᵉ`, every other scalar is transported with
    # `Kᶜ` and has no linear coefficient.
    tracer_diffusivities = NamedTuple(name => name === TKE_NAME ? Kᵉ : Kᶜ for name in tracer_names)
    implicit_linear_coefficients = NamedTuple(name => name === TKE_NAME ? Lᵉ : ZeroField() for name in tracer_names)

    return TKEClosureFields(Kᵘ, Kᶜ, Kᵉ, Lᵉ, tracer_diffusivities, implicit_linear_coefficients)
end

@inline Oceananigans.TurbulenceClosures.viscosity_location(::FlavorOfTKEClosure) = (Center(), Center(), Face())
@inline Oceananigans.TurbulenceClosures.diffusivity_location(::FlavorOfTKEClosure) = (Center(), Center(), Face())

@inline Oceananigans.TurbulenceClosures.viscosity(::FlavorOfTKEClosure, fields) = fields.Kᵘ

@inline Oceananigans.TurbulenceClosures.diffusivity(::FlavorOfTKEClosure, fields, ::Val{id}) where id =
    fields.tupled_tracer_diffusivities[id]

# The vertically implicit tridiagonal solve of the host model carries a linear term, so that
# `(1 - Δt Lᵉ - Δt ∂z Kᵉ ∂z) eⁿ⁺¹ = e⋆` — the sinks of the TKE equation are solved together with
# its vertical diffusion, every stage. Momentum (`id = nothing`) and the other scalars fall back
# to zero.
@inline Oceananigans.TurbulenceClosures.implicit_linear_coefficient(i, j, k, grid,
                                                                    closure::FlavorOfTKEClosure{<:VerticallyImplicitTimeDiscretization},
                                                                    fields, ::Val{id}, args...) where id =
    @inbounds fields.tupled_implicit_linear_coefficients[id][i, j, k]

#####
##### Stability functions: dispatch on the stability-function type. The arguments beyond the
##### closure are what a Richardson-number-dependent variant needs.
#####

const ConstantStabilityClosure = TKEBasedTurbulenceClosure{<:Any, <:Any, <:ConstantStabilityFunctions}

@inline momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.Cᵘ
@inline tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.Cᶜ
@inline tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.Cᵉ
@inline dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.Cᴰ

#####
##### Mixing length: ℓ = min(z, Cᴺ √e / N), evaluated where the caller supplies √e
#####

"""
$(TYPEDSIGNATURES)

The stratification length ``ℓᴺ = Cᴺ \\sqrt{e} / N`` at (Center, Center, Face), given the
turbulent velocity ``w★ = \\sqrt{e}`` there; infinite where ``N² ≤ 0``.
"""
@inline function stratification_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, w★, tracers, buoyancy)
    FT = eltype(grid)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺ = clip(N²)
    Cᴺ = closure.mixing_length.Cᴺ
    return ifelse(N²⁺ == 0, FT(Inf), Cᴺ * w★ / sqrt(N²⁺))
end

@inline function stratification_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, w★, tracers, buoyancy)
    FT = eltype(grid)
    N² = ℑbzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    N²⁺ = clip(N²)
    Cᴺ = closure.mixing_length.Cᴺ
    return ifelse(N²⁺ == 0, FT(Inf), Cᴺ * w★ / sqrt(N²⁺))
end

"""
$(TYPEDSIGNATURES)

The primary mixing length ``ℓ = \\min(z, ℓᴺ)`` at (Center, Center, Face), given the specific
turbulent kinetic energy field `e` at the centers, whose square root — floored at `minimum_tke` —
is reconstructed at the face. The same function computes the closure's diffusivities and, evaluated
in a `KernelFunctionOperation` at (Center, Center, Face), diagnoses ``ℓ`` from the model state.
"""
@inline function mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, e, tracers, buoyancy)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
    d = height_above_bottomᶜᶜᶠ(i, j, k, grid)
    ℓᴺ = stratification_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, w★, tracers, buoyancy)
    ℓ = min(d, ℓᴺ)
    return ifelse(isnan(ℓ), d, ℓ)
end

"""
$(TYPEDSIGNATURES)

`mixing_lengthᶜᶜᶠ` at cell centers, where the dissipation lives with ``e``.
"""
@inline function mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    w★ = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
    d = height_above_bottomᶜᶜᶜ(i, j, k, grid)
    ℓᴺ = stratification_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, w★, tracers, buoyancy)
    ℓ = min(d, ℓᴺ)
    return ifelse(isnan(ℓ), d, ℓ)
end

#####
##### Diffusivities
#####

@kernel function _compute_tke_closure_fields!(closure_fields, grid, closure, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)
    e = tracers[TKE_NAME]

    # √e, floored at the minimum TKE, reconstructed from the centers to the face
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure_ij, e)
    ℓ = mixing_lengthᶜᶜᶠ(i, j, k, grid, closure_ij, e, tracers, buoyancy)

    Sᵘ = momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)
    Sᶜ = tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)
    Sᵉ = tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)

    Kᵘ = min(Sᵘ * ℓ * w★, closure_ij.maximum_viscosity)
    Kᶜ = min(Sᶜ * ℓ * w★, closure_ij.maximum_tracer_diffusivity)
    Kᵉ = min(Sᵉ * ℓ * w★, closure_ij.maximum_tke_diffusivity)

    FT = eltype(grid)
    @inbounds begin
        closure_fields.Kᵘ[i, j, k] = mask_diffusivity(i, j, k, grid, FT(Kᵘ))
        closure_fields.Kᶜ[i, j, k] = mask_diffusivity(i, j, k, grid, FT(Kᶜ))
        closure_fields.Kᵉ[i, j, k] = mask_diffusivity(i, j, k, grid, FT(Kᵉ))
    end
end

"""
$(TYPEDSIGNATURES)

Shear production ``Kᵘ S²`` at (Center, Center, Face).
"""
@inline shear_productionᶜᶜᶠ(i, j, k, grid, Kᵘ, u, v) = @inbounds Kᵘ[i, j, k] * shearᶜᶜᶠ(i, j, k, grid, u, v)

"""
$(TYPEDSIGNATURES)

Buoyancy production ``-Kᶜ N²`` at (Center, Center, Face); negative in stable stratification.
"""
@inline buoyancy_productionᶜᶜᶠ(i, j, k, grid, Kᶜ, buoyancy, tracers) = @inbounds -Kᶜ[i, j, k] * ∂z_b(i, j, k, grid, buoyancy, tracers)

"""
$(TYPEDSIGNATURES)

The rate at which the sinks of the TKE equation remove turbulent kinetic energy, ``-Lᵉ ≥ 0``:
the dissipation rate ``ω = Sᴰ \\sqrt{e} / ℓ`` — or, where ``e`` is negative, the damping rate
``1/τ`` — plus the negative part of the buoyancy flux divided by ``e``, where there is TKE to
remove. Following CATKE, these are the terms treated implicitly in ``e``, so that ``e`` stays
positive for any time step.
"""
@inline function tke_sink_rate(i, j, k, grid, closure, e, B, velocities, tracers, buoyancy)
    eᵐⁱⁿ = closure.minimum_tke
    eᵢ = @inbounds e[i, j, k]
    ℓ = mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    Sᴰ = dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy)

    # `minimum_tke` floors only the turbulent velocity of the mixing length above; the dissipation
    # rate follows √e all the way down, so that ε ∝ e^{3/2} below the floor too (as in CATKE).
    # The `abs` keeps the unselected branch of the `ifelse` from taking √ of a negative number.
    τ = closure.negative_tke_damping_time_scale
    ω = ifelse(eᵢ < 0, 1 / τ, Sᴰ * sqrt(abs(eᵢ)) / ℓ)

    B⁻ = min(0, B)
    ωᴮ = -B⁻ / max(eᵢ, eᵐⁱⁿ) * (eᵢ > eᵐⁱⁿ)

    return ω + ωᴮ
end

# The linear implicit coefficient Lᵉ of `∂t e = Lᵉ e + ⋯`, at cell centers, from the stored `Kᶜ`
# and the raw specific TKE. Launched after the diffusivity kernel, since it reads `Kᶜ` at the faces
# above and below the cell.
@kernel function _compute_tke_implicit_linear_coefficient!(Lᵉ, grid, closure, closure_fields, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)
    e = tracers[TKE_NAME]

    B = ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_productionᶜᶜᶠ, closure_fields.Kᶜ, buoyancy, tracers)
    ω = tke_sink_rate(i, j, k, grid, closure_ij, e, B, velocities, tracers, buoyancy)
    active = !inactive_cell(i, j, k, grid)

    @inbounds Lᵉ[i, j, k] = - ω * active
end

# Called from `update_state!`, where every tracer — `ρe` included — momentarily holds its specific
# value, so the kernels read `e` directly from the tracer.
function Oceananigans.TurbulenceClosures.compute_closure_fields!(closure_fields,
                                                         closure::FlavorOfTKEClosure,
                                                         model; parameters = :xyz)
    grid = model.grid
    arch = grid.architecture
    tracers = Oceananigans.TurbulenceClosures.buoyancy_tracers(model)
    buoyancy = Oceananigans.TurbulenceClosures.buoyancy_force(model)

    launch!(arch, grid, parameters, _compute_tke_closure_fields!,
            closure_fields, grid, closure, model.velocities, tracers, buoyancy)

    launch!(arch, grid, parameters, _compute_tke_implicit_linear_coefficient!,
            closure_fields.Lᵉ, grid, closure, closure_fields, model.velocities, tracers, buoyancy)

    return nothing
end

#####
##### The TKE equation: sources in the stage tendency
#####

# Under a vertically implicit time discretization the sinks live in the tridiagonal solve; under
# an explicit one they are added to the tendency here as `Lᵉ e`, from the stored rate `Lᵉ` that the
# last `update_state!` computed from the same stage state as `Kᵘ` and `Kᶜ`.
@inline explicit_tke_sinks(i, j, k, grid, ::TKEBasedTurbulenceClosure{<:VerticallyImplicitTimeDiscretization},
                           closure_fields, e) = zero(grid)

@inline explicit_tke_sinks(i, j, k, grid, ::TKEBasedTurbulenceClosure{<:ExplicitTimeDiscretization},
                           closure_fields, e) = @inbounds closure_fields.Lᵉ[i, j, k] * e

"""
$(TYPEDSIGNATURES)

Add the local sources of the TKE equation, ``ρ (P + B⁺)`` — shear production and the positive part
of the buoyancy flux, formed at faces where ``Kᵘ``, ``Kᶜ``, ``S²`` and ``N²`` live and reconstructed
to centers — to the tendency of the `ρe` tracer. Under an explicit time discretization the sinks
``ρ Lᵉ e`` are added too.
"""
@kernel function _add_tke_tendencies!(Gρe, grid, closure, closure_fields, velocities, tracers, ρe, ρ, buoyancy)
    i, j, k = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)

    P = ℑbzᵃᵃᶜ(i, j, k, grid, shear_productionᶜᶜᶠ, closure_fields.Kᵘ, velocities.u, velocities.v)
    B = ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_productionᶜᶜᶠ, closure_fields.Kᶜ, buoyancy, tracers)
    B⁺ = max(0, B)

    ρᵢ = @inbounds ρ[i, j, k]
    e = @inbounds ρe[i, j, k] / ρᵢ
    sinks = explicit_tke_sinks(i, j, k, grid, closure_ij, closure_fields, e)

    @inbounds Gρe[i, j, k] += ρᵢ * (P + B⁺ + sinks)
end

# Called by the time steppers at the start of every stage, after the flux boundary conditions have
# been added to the tendencies; the tracers hold their density-weighted values here.
function AtmosphereModels.compute_closure_tendencies!(Gⁿ, closure_fields, closure::FlavorOfTKEClosure, model)
    grid = model.grid
    arch = grid.architecture
    tracers = Oceananigans.TurbulenceClosures.buoyancy_tracers(model)
    buoyancy = Oceananigans.TurbulenceClosures.buoyancy_force(model)
    ρ = AtmosphereModels.total_density(model.dynamics)

    launch!(arch, grid, :xyz, _add_tke_tendencies!,
            Gⁿ[TKE_NAME], grid, closure, closure_fields,
            model.velocities, tracers, model.tracers[TKE_NAME], ρ, buoyancy)

    return nothing
end

#####
##### Show
#####

function Base.summary(closure::TKEBasedTurbulenceClosure)
    TD = nameof(typeof(time_discretization(closure)))
    return string("TKEBasedTurbulenceClosure{$TD}")
end

function Base.show(io::IO, closure::TKEBasedTurbulenceClosure)
    print(io, summary(closure), '\n',
              "├── mixing_length: ", summary(closure.mixing_length), " (Cᴺ = ", prettysummary(closure.mixing_length.Cᴺ), ")", '\n',
              "├── stability_functions: ", summary(closure.stability_functions), '\n',
              "│   ├── Cᵘ: ", prettysummary(closure.stability_functions.Cᵘ), '\n',
              "│   ├── Cᶜ: ", prettysummary(closure.stability_functions.Cᶜ), '\n',
              "│   ├── Cᵉ: ", prettysummary(closure.stability_functions.Cᵉ), '\n',
              "│   └── Cᴰ: ", prettysummary(closure.stability_functions.Cᴰ), '\n',
              "├── maximum_viscosity: ", prettysummary(closure.maximum_viscosity), '\n',
              "├── maximum_tracer_diffusivity: ", prettysummary(closure.maximum_tracer_diffusivity), '\n',
              "├── maximum_tke_diffusivity: ", prettysummary(closure.maximum_tke_diffusivity), '\n',
              "├── minimum_tke: ", prettysummary(closure.minimum_tke), '\n',
              "└── negative_tke_damping_time_scale: ", prettysummary(closure.negative_tke_damping_time_scale))
end
