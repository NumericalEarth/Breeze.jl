#####
##### `TKEBasedTurbulenceClosure`: a vertical eddy-diffusivity closure with prognostic turbulent
##### kinetic energy, in the spirit of CATKE (Wagner et al. 2025)
#####
#####   Kᵘ = Sᵘ ℓ √e,   Kᶜ = Sᶜ ℓ √e,   Kᵉ = Sᵉ ℓ √e,   ε = Sᴰ e^{3/2} / ℓ
#####   ℓ  = a mixing-length formulation built on the buoyancy penetration depth ℓᵇ = √e / N and the wall length Cˢ z
#####   ∂ₜ(ρe) + ∇·(ρ u e) = ∂z(ρ Kᵉ ∂z e) + ρ (P + B − ε),   P = Kᵘ S²,  B = −Kᶜ N²
#####
##### The static stability N² is diagnosed once per stage at the cell interfaces and stored with the
##### closure fields, so that the mixing length, the buoyancy flux and the stability functions all
##### see one value. `MoistStaticStability` (the default) is the gradient of the buoyancy of the
##### dynamics, ∂z_b, where the air is subsaturated, and the buoyancy frequency of a saturated
##### displacement where it is saturated; `DryStaticStability` is ∂z_b everywhere (static_stability.jl).
#####
##### Three formulations of the mixing length share the wall coefficient Cˢ and are computed column by
##### column, once per stage, into `closure_fields.ℓ`: `LocalMinimumMixingLength`, ℓ = min(Cˢ z, ℓᵇ),
##### which knows only the ground and the stratification at the eddy's own level; `IntegralMixingLength`,
##### the parcel lengths of Bougeault & Lacarrère (1989), a walk per face; and the default
##### `GradientLimitedMixingLength`, the local minimum limited so that |∂z ℓ| ≤ Cˢ — the lower envelope of
##### every level's penetration depth, two sweeps per column, the construction of NEMO's TKE scheme —
##### which bounds ℓ by the stratified air above and below an elevated neutral or unstable layer, where
##### the local minimum lets it grow to the distance to the ground.
#####
##### The stability functions Sᵘ, Sᶜ, Sᵉ, Sᴰ are either constants (`ConstantStabilityFunctions`)
##### or piecewise-linear functions of the Richardson number in the form of CATKE
##### (`RiDependentStabilityFunctions`, richardson_number_stability_functions.jl).
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
neutral constant-stress layer, where ``ℓ = Cˢ z`` with ``Cˢ`` the wall coefficient of
[`AbstractMixingLength`](@ref), production balances dissipation at ``e / u_\\star² = 1 / \\sqrt{Cᵘ Cᴰ}``
with a logarithmic wind profile of von Kármán constant ``κ = Cˢ (Cᵘ³ / Cᴰ)^{1/4}``; in stratified
steady state, where ``ℓ = \\sqrt{e} / N``, the gradient Richardson number is
``Ri^\\dagger = Cᵘ / (Cᶜ + Cᴰ)``.

The defaults are the Mellor–Yamada coefficients of [Nakanishi and Niino (2009)](@cite NakanishiNiino2009)
re-expressed for this normalization of the mixing length (``κ = 0.4``, ``e/u_\\star² = 4.2``,
``Pr = 0.74``, ``Ri^\\dagger = 0.25``); they are placeholders for calibration. The conversion from
the normalization ``ℓ = \\min(z, Cᴺ \\sqrt{e} / N)`` is [`absorb_stratification_coefficient`](@ref).

Fields
======

$(TYPEDFIELDS)
"""
Base.@kwdef struct ConstantStabilityFunctions{FT}
    "momentum stability function, ``ℓᵘ = Cᵘ ℓ``"
    Cᵘ :: FT = 0.149
    "tracer stability function, ``ℓᶜ = Cᶜ ℓ``"
    Cᶜ :: FT = 0.201
    "turbulent kinetic energy stability function, ``ℓᵉ = Cᵉ ℓ``"
    Cᵉ :: FT = 0.298
    "dissipation stability function, ``ℓᴰ = ℓ / Cᴰ``"
    Cᴰ :: FT = 0.388
end

## `Base.@kwdef` on a `{FT}` struct requires every field to share one type, so mixed
## integer/float keyword arguments — `ConstantStabilityFunctions(Cᵉ = 1)` — are promoted here.
ConstantStabilityFunctions(Cᵘ, Cᶜ, Cᵉ, Cᴰ) = ConstantStabilityFunctions(promote(Cᵘ, Cᶜ, Cᵉ, Cᴰ)...)

Base.summary(sf::ConstantStabilityFunctions{FT}) where FT = "ConstantStabilityFunctions{$FT}"

# The lines under the stability functions' summary, with `prefix` for nesting in the closure's show
function show_stability_function_lines(io::IO, sf::ConstantStabilityFunctions, prefix)
    print(io, prefix, "├── Cᵘ: ", prettysummary(sf.Cᵘ), '\n',
              prefix, "├── Cᶜ: ", prettysummary(sf.Cᶜ), '\n',
              prefix, "├── Cᵉ: ", prettysummary(sf.Cᵉ), '\n',
              prefix, "└── Cᴰ: ", prettysummary(sf.Cᴰ))
    return nothing
end

function Base.show(io::IO, sf::ConstantStabilityFunctions)
    print(io, summary(sf), '\n')
    show_stability_function_lines(io, sf, "")
end

#####
##### Mixing length
#####

"""
$(TYPEDEF)

Supertype of the formulations of the primary mixing length ``ℓ`` of [`TKEBasedTurbulenceClosure`](@ref).
All are built from two lengths: the wall length ``Cˢ z``, ``Cˢ`` times the height above the surface,
and the buoyancy penetration depth

```math
ℓᵇ = \\frac{\\sqrt{e}}{N},
```

how far an eddy with kinetic energy ``e`` penetrates air of buoyancy frequency ``N`` before its kinetic
energy is spent against buoyancy — infinite where ``N² ≤ 0``. Each formulation carries the one coefficient
``Cˢ``; the penetration depth carries none, the stability functions set the scale of every diffusivity,
so ``Cˢ`` alone fixes the ratio of the two lengths. The default ``Cˢ = 1.316`` is the reciprocal of
Deardorff's coefficient ``0.76`` of the stratification length ([Deardorff 1980](@cite Deardorff1980)),
which the equivalent normalization ``ℓ = \\min(z, 0.76 \\sqrt{e} / N)`` carries on ``ℓᵇ`` instead; see
[`absorb_stratification_coefficient`](@ref). In a neutral surface layer every formulation gives
``ℓ = Cˢ z`` and in uniformly stratified air every formulation gives ``ℓ = ℓᵇ``; they differ in what
else bounds ``ℓ``. The formulations are [`LocalMinimumMixingLength`](@ref),
[`IntegralMixingLength`](@ref) and the default [`GradientLimitedMixingLength`](@ref).
"""
abstract type AbstractMixingLength end

"""
$(TYPEDEF)

The mixing length as the smaller of the wall length and the buoyancy penetration depth at the eddy's
own level,

```math
ℓ = \\min(Cˢ z, \\, ℓᵇ),
```

the form of [Deardorff (1980)](@cite Deardorff1980) and of Nakanishi and Niino's closure without its
turbulent-layer depth scale. It is local: it knows two obstacles, the ground and the stratification
at the level itself. In an elevated layer whose static stability is not positive — a saturated cloud
layer under [`MoistStaticStability`](@ref), a neutral layer — ``ℓᵇ`` is infinite and nothing bounds
``ℓ`` but the distance to the ground, so ``ℓ = Cˢ z`` reaches kilometers and the diffusivity
``Cᵘ ℓ \\sqrt{e}`` jumps by orders of magnitude when such a layer forms, mixes it away within a time
step and collapses. [`GradientLimitedMixingLength`](@ref), the default, adds the missing bound.
"""
Base.@kwdef struct LocalMinimumMixingLength{FT} <: AbstractMixingLength
    Cˢ :: FT = 1.316 # coefficient of the wall length ℓ = Cˢ z
end

"""
$(TYPEDEF)

The mixing length from the parcel displacements of [Bougeault and Lacarrère (1989)](@cite BougeaultLacarrere1989):
a parcel released at height ``z`` with kinetic energy ``e★`` rises until the buoyancy deficit it
accumulates has consumed that energy, and likewise sinks,

```math
∫_0^{ℓ↑} [b(z) - b(z + s)] \\, ds = e★, \\qquad ∫_0^{ℓ↓} [b(z - s) - b(z)] \\, ds = e★, \\qquad ℓ↓ ≤ z,
```

and the mixing length is ``ℓ = Cˢ \\min(ℓ↑, ℓ↓)``, the combination of the original paper. The parcel's
energy is ``e★ = e / (2 Cˢ²)`` so that the formulation coincides with the others in the two limits every
formulation shares: against the ground ``ℓ↓ = z`` and ``ℓ = Cˢ z``, and in uniform stratification the
deficit is ``N² s² / 2`` and ``ℓ = ℓᵇ = \\sqrt{e} / N``. Unlike [`GradientLimitedMixingLength`](@ref) it
carries the parcel's own energy into the air it penetrates and credits the energy a parcel gains
crossing unstable air, so it penetrates inversions and mixes convective layers more. Its cost is a
walk per face whose length depends on the state — one to three cells in stratified air, the depth of
a mixed layer inside one — so its kernel has data-dependent loop lengths; the top of the domain is
not an obstacle. The buoyancy differences along the path are integrated from the stored ``N²``.
"""
Base.@kwdef struct IntegralMixingLength{FT} <: AbstractMixingLength
    Cˢ :: FT = 1.316 # coefficient of the wall length ℓ = Cˢ z
end

"""
$(TYPEDEF)

The default mixing length: the local minimum ``\\min(Cˢ z, ℓᵇ)`` limited so that ``ℓ`` changes by no more
than ``Cˢ`` per unit height, ``|∂_z ℓ| ≤ Cˢ``. Equivalently, the lower envelope of the buoyancy
penetration depths of every level in the column and of the ground,

```math
ℓ(z) = \\min_{z′} \\left[ ℓᵇ(z′) + Cˢ |z - z′| \\right],
```

an eddy centered at ``z`` reaches past a level ``z′`` only by the penetration depth there. Where only the
ground and the level itself bind it is [`LocalMinimumMixingLength`](@ref) exactly — the wall length
``Cˢ z`` in neutral air attached to the surface, ``\\sqrt{e} / N`` in stably stratified air — and it
differs where that formula fails: in a mixed layer it falls off toward the capping inversion instead
of growing as ``Cˢ z``, and in an elevated neutral or unstable layer it is bounded by ``Cˢ`` times the
distance to the stratified air above and below rather than by the distance to the ground. Because the
bound propagates from a level only to its neighbors, two sweeps per column compute the envelope exactly,
upward from the ground then downward, with fixed loop lengths. It is the local-penetration form of
[`IntegralMixingLength`](@ref): the two agree where the stratification beyond an obstacle is uniform,
but the envelope takes the eddy energy at the obstacle rather than at the eddy's origin and does not
credit energy gained in unstable air, so it penetrates inversions less. The same construction, with
slope one, is the gradient-limited mixing length of NEMO's TKE scheme
([Gaspar et al. 1990](@cite Gaspar1990)).
"""
Base.@kwdef struct GradientLimitedMixingLength{FT} <: AbstractMixingLength
    Cˢ :: FT = 1.316 # slope of ℓ away from an obstacle; the coefficient of the wall length ℓ = Cˢ z
end

Base.summary(ml::LocalMinimumMixingLength{FT}) where FT = "LocalMinimumMixingLength{$FT}"
Base.summary(ml::IntegralMixingLength{FT}) where FT = "IntegralMixingLength{$FT}"
Base.summary(ml::GradientLimitedMixingLength{FT}) where FT = "GradientLimitedMixingLength{$FT}"
Base.show(io::IO, ml::AbstractMixingLength) = print(io, summary(ml), " (Cˢ = ", prettysummary(ml.Cˢ), ")")

@inline convert_eltype(::Type{FT}, ml::LocalMinimumMixingLength) where FT = LocalMinimumMixingLength{FT}(convert(FT, ml.Cˢ))
@inline convert_eltype(::Type{FT}, ml::IntegralMixingLength) where FT = IntegralMixingLength{FT}(convert(FT, ml.Cˢ))
@inline convert_eltype(::Type{FT}, ml::GradientLimitedMixingLength) where FT = GradientLimitedMixingLength{FT}(convert(FT, ml.Cˢ))

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
primary mixing length ([`AbstractMixingLength`](@ref): [`GradientLimitedMixingLength`](@ref) by default,
[`LocalMinimumMixingLength`](@ref) or [`IntegralMixingLength`](@ref)), and ``Sᵘ, Sᶜ, Sᵉ, Sᴰ`` stability functions
([`ConstantStabilityFunctions`](@ref) or [`RiDependentStabilityFunctions`](@ref)). ``N²`` is
diagnosed once per time-step stage at the cell
interfaces by the `static_stability` component ([`MoistStaticStability`](@ref) by default, or
[`DryStaticStability`](@ref)) and stored with the closure fields. The prognostic TKE density is
the tracer `ρe`, which the
closure adds to the model; it is advected and vertically diffused like every other scalar, and the
closure applies the local production, buoyancy flux and dissipation.

The square root of ``e`` is floored at `minimum_tke` wherever it enters a diffusivity or a length
scale, and negative ``e`` — which advection can produce — is damped on
`negative_tke_damping_time_scale` rather than clipped. The three `maximum_*` diffusivities clip
the diffusivities, `Inf` by default.
"""
struct TKEBasedTurbulenceClosure{TD, ML, SF, SS, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    mixing_length :: ML                   # the primary mixing length ℓ
    stability_functions :: SF             # Sᵘ, Sᶜ, Sᵉ, Sᴰ
    static_stability :: SS                # how N² is diagnosed
    maximum_viscosity :: FT               # upper bound on Kᵘ, m² s⁻¹
    maximum_tracer_diffusivity :: FT      # upper bound on Kᶜ, m² s⁻¹
    maximum_tke_diffusivity :: FT         # upper bound on Kᵉ, m² s⁻¹
    minimum_tke :: FT                     # floor on e inside √e, m² s⁻²
    negative_tke_damping_time_scale :: FT # time scale on which negative e is damped, s
end

const TKEClosureArray{TD} = AbstractArray{<:TKEBasedTurbulenceClosure{TD}} where TD

"""Either a single `TKEBasedTurbulenceClosure` or an ensemble array of them."""
const FlavorOfTKEClosure{TD} = Union{TKEBasedTurbulenceClosure{TD}, TKEClosureArray{TD}} where TD

function TKEBasedTurbulenceClosure{TD}(mixing_length::ML, stability_functions::SF, static_stability::SS,
                                       maximum_viscosity::FT, maximum_tracer_diffusivity::FT,
                                       maximum_tke_diffusivity::FT, minimum_tke::FT,
                                       negative_tke_damping_time_scale::FT) where {TD, ML, SF, SS, FT}
    return TKEBasedTurbulenceClosure{TD, ML, SF, SS, FT}(mixing_length, stability_functions, static_stability,
                                                         maximum_viscosity, maximum_tracer_diffusivity,
                                                         maximum_tke_diffusivity, minimum_tke,
                                                         negative_tke_damping_time_scale)
end

"""
$(TYPEDSIGNATURES)

Construct a [`TKEBasedTurbulenceClosure`](@ref) with the given time discretization (default
`VerticallyImplicitTimeDiscretization()`), float type, mixing length, stability functions, static
stability and numerical parameters.
"""
function TKEBasedTurbulenceClosure(time_discretization::TD = VerticallyImplicitTimeDiscretization(),
                                   FT = Oceananigans.defaults.FloatType;
                                   mixing_length = GradientLimitedMixingLength(),
                                   stability_functions = ConstantStabilityFunctions(),
                                   static_stability = MoistStaticStability(),
                                   maximum_viscosity = Inf,
                                   maximum_tracer_diffusivity = Inf,
                                   maximum_tke_diffusivity = Inf,
                                   minimum_tke = 1e-6,
                                   # CATKE's value; atmospheric turbulence evolves faster than the
                                   # ocean's, so a shorter time scale may be more appropriate here.
                                   negative_tke_damping_time_scale = 1minute) where TD

    mixing_length = convert_eltype(FT, mixing_length)
    stability_functions = convert_eltype(FT, stability_functions)
    static_stability = convert_eltype(FT, static_stability)

    return TKEBasedTurbulenceClosure{TD}(mixing_length,
                                         stability_functions,
                                         static_stability,
                                         convert(FT, maximum_viscosity),
                                         convert(FT, maximum_tracer_diffusivity),
                                         convert(FT, maximum_tke_diffusivity),
                                         convert(FT, minimum_tke),
                                         convert(FT, negative_tke_damping_time_scale))
end

TKEBasedTurbulenceClosure(FT::DataType; kw...) =
    TKEBasedTurbulenceClosure(VerticallyImplicitTimeDiscretization(), FT; kw...)

@inline convert_eltype(::Type{FT}, sf::ConstantStabilityFunctions) where FT =
    ConstantStabilityFunctions{FT}(convert(FT, sf.Cᵘ), convert(FT, sf.Cᶜ), convert(FT, sf.Cᵉ), convert(FT, sf.Cᴰ))
@inline convert_eltype(::Type{FT}, ss::DryStaticStability) where FT = ss
@inline convert_eltype(::Type{FT}, ss::MoistStaticStability) where FT = ss

"""
$(TYPEDSIGNATURES)

Convert the parameters of the normalization ``ℓ = \\min(z, Cᴺ \\sqrt{e} / N)`` — a coefficient
``Cᴺ`` on the stratification length and none on the wall distance — to the present
``ℓ = \\min(Cˢ z, \\sqrt{e} / N)``, returning the equivalent [`LocalMinimumMixingLength`](@ref) and
[`ConstantStabilityFunctions`](@ref). The two lengths differ by the factor ``Cᴺ`` everywhere, so
the diffusivities and the dissipation rate are unchanged when ``Cˢ = 1 / Cᴺ``, the diffusivity
coefficients are multiplied by ``Cᴺ`` and the dissipation coefficient is divided by it.

```jldoctest
using Breeze
using Breeze.TurbulenceClosures: absorb_stratification_coefficient

mixing_length, stability_functions =
    absorb_stratification_coefficient(0.76, ConstantStabilityFunctions(Cᵘ=0.196, Cᶜ=0.265, Cᵉ=0.392, Cᴰ=0.295))

mixing_length

# output
LocalMinimumMixingLength{Float64} (Cˢ = 1.31579)
```
"""
function absorb_stratification_coefficient(Cᴺ, sf::ConstantStabilityFunctions)
    mixing_length = LocalMinimumMixingLength(Cˢ = 1 / Cᴺ)
    stability_functions = ConstantStabilityFunctions(Cᵘ = Cᴺ * sf.Cᵘ, Cᶜ = Cᴺ * sf.Cᶜ, Cᵉ = Cᴺ * sf.Cᵉ, Cᴰ = sf.Cᴰ / Cᴺ)
    return mixing_length, stability_functions
end

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

Precomputed fields for [`TKEBasedTurbulenceClosure`](@ref): the three diffusivities, the static
stability ``N²``, the mixing length ``ℓ`` and the implicit linear coefficient. Unlike CATKE's, the
mixing length is stored, because its nonlocal formulations are computed column by column rather than
pointwise; `closure_fields.ℓ` is the diagnostic of ``ℓ`` from the model state.
"""
struct TKEClosureFields{K, N, L, KC, LC}
    Kᵘ :: K # eddy diffusivity for momentum, at (Center, Center, Face)
    Kᶜ :: K # eddy diffusivity for scalars, at (Center, Center, Face)
    Kᵉ :: K # eddy diffusivity for turbulent kinetic energy, at (Center, Center, Face)
    N² :: N # static stability, at (Center, Center, Face)
    ℓ  :: N # the primary mixing length, at (Center, Center, Face)
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
                     adapt(to, fields.N²),
                     adapt(to, fields.ℓ),
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
    N² = ZFaceField(grid)
    ℓ = ZFaceField(grid)
    Lᵉ = CenterField(grid)

    # Indexed by the `Val(id)` the model hands to `diffusivity` and `implicit_linear_coefficient`:
    # TKE is transported with `Kᵉ` and damped with `Lᵉ`, every other scalar is transported with
    # `Kᶜ` and has no linear coefficient.
    tracer_diffusivities = NamedTuple(name => name === TKE_NAME ? Kᵉ : Kᶜ for name in tracer_names)
    implicit_linear_coefficients = NamedTuple(name => name === TKE_NAME ? Lᵉ : ZeroField() for name in tracer_names)

    return TKEClosureFields(Kᵘ, Kᶜ, Kᵉ, N², ℓ, Lᵉ, tracer_diffusivities, implicit_linear_coefficients)
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
##### closure — the velocities and the stored N² — are what a Richardson-number-dependent variant
##### needs.
#####

const ConstantStabilityClosure = TKEBasedTurbulenceClosure{<:Any, <:Any, <:ConstantStabilityFunctions}

@inline momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.Cᵘ
@inline tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.Cᶜ
@inline tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.Cᵉ
@inline dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.Cᴰ

#####
##### Computing the mixing length, column by column, into `closure_fields.ℓ`
#####

# The stored face field as a kernel function, for the boundary-aware reconstruction `ℑbzᵃᵃᶜ`
@inline face_valueᶜᶜᶠ(i, j, k, grid, field) = @inbounds field[i, j, k]

"""
$(TYPEDSIGNATURES)

The buoyancy penetration depth ``ℓᵇ = \\sqrt{e} / N`` at (Center, Center, Face): how far an eddy
with kinetic energy ``e`` — floored at `minimum_tke` and reconstructed at the face from the cell
centers — penetrates air of static stability ``N²`` before its kinetic energy is spent against
buoyancy. Infinite where ``N² ≤ 0``, so that neutral and unstable air is no obstacle.
"""
@inline function buoyancy_penetration_depthᶜᶜᶠ(i, j, k, grid, closure, e, N²)
    FT = eltype(grid)
    N²⁺ = clip(@inbounds N²[i, j, k])
    ℓᵇ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e) / sqrt(N²⁺)
    return ifelse(N²⁺ == 0, FT(Inf), ℓᵇ)
end

# The local minimum at a face: the ground — or an immersed bottom — through the wall length, and
# the stratification there through the penetration depth
@inline function local_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, e, N²)
    d = closure.mixing_length.Cˢ * height_above_bottomᶜᶜᶠ(i, j, k, grid)
    ℓᵇ = buoyancy_penetration_depthᶜᶜᶠ(i, j, k, grid, closure, e, N²)
    ℓ = min(d, ℓᵇ)
    return ifelse(isnan(ℓ), d, ℓ)
end

# The local minimum at a cell center, with N² reconstructed from the two adjacent faces
@inline function local_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, N²)
    FT = eltype(grid)
    d = closure.mixing_length.Cˢ * height_above_bottomᶜᶜᶜ(i, j, k, grid)
    N²⁺ = clip(ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, N²))
    ℓᵇ = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e) / sqrt(N²⁺)
    ℓ = min(d, ifelse(N²⁺ == 0, FT(Inf), ℓᵇ))
    return ifelse(isnan(ℓ), d, ℓ)
end

# `LocalMinimumMixingLength`: the local minimum at every face
@inline function fill_mixing_length!(ℓ, i, j, grid, ::LocalMinimumMixingLength, closure, e, N²)
    for k in 1:grid.Nz+1
        @inbounds ℓ[i, j, k] = local_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, e, N²)
    end
    return nothing
end

# `GradientLimitedMixingLength`: the local minimum, then the two sweeps that bound its slope by Cˢ —
# upward from the ground, where ℓ = 0, then downward. The top of the domain is not an obstacle.
@inline function fill_mixing_length!(ℓ, i, j, grid, ml::GradientLimitedMixingLength, closure, e, N²)
    Cˢ = ml.Cˢ
    Nz = grid.Nz
    @inbounds begin
        ℓ[i, j, 1] = 0 # the ground; the diffusivities at the bottom face are masked regardless
        for k in 2:Nz+1
            ℓₖ = local_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, e, N²)
            ℓ[i, j, k] = min(ℓₖ, ℓ[i, j, k-1] + Cˢ * Δzᶜᶜᶜ(i, j, k-1, grid))
        end
        for k in Nz:-1:1
            ℓ[i, j, k] = min(ℓ[i, j, k], ℓ[i, j, k+1] + Cˢ * Δzᶜᶜᶜ(i, j, k, grid))
        end
    end
    return nothing
end

# `IntegralMixingLength`: Bougeault & Lacarrère's parcel walks from every face. The environment's
# buoyancy relative to the parcel, Δb, grows by N² Δz across each cell — N² at the cell from the two
# faces — and crossing a cell costs the energy Δb Δz + N² Δz² / 2. The parcel stops inside the cell
# where the accumulated cost reaches its energy e★, at the distance that solves the quadratic.
@inline function crossing_distance(Δb, N², Δz, r)
    # Δb s + N² s² / 2 = r, in the form that is stable as N² → 0 and valid for either sign of N²
    s = 2r / (Δb + sqrt(max(Δb^2 + 2 * N² * r, 0)))
    return clamp(s, 0, Δz)
end

@inline function parcel_ascent(i, j, k, grid, N², e★)
    FT = eltype(grid)
    Nz = grid.Nz
    s = zero(FT); D = zero(FT); Δb = zero(FT)
    m = k
    @inbounds while m ≤ Nz && D < e★
        Δz = Δzᶜᶜᶜ(i, j, m, grid)
        N²ₘ = (N²[i, j, m] + N²[i, j, m+1]) / 2
        cost = Δb * Δz + N²ₘ * Δz^2 / 2
        stops = D + cost ≥ e★
        s += ifelse(stops, crossing_distance(Δb, N²ₘ, Δz, e★ - D), Δz)
        D = ifelse(stops, e★, D + cost)
        Δb += N²ₘ * Δz
        m += 1
    end
    return ifelse(D < e★, FT(Inf), s) # out the top with energy to spare: no obstacle above
end

@inline function parcel_descent(i, j, k, grid, N², e★)
    FT = eltype(grid)
    s = zero(FT); D = zero(FT); Δb = zero(FT)
    m = k - 1
    @inbounds while m ≥ 1 && D < e★
        Δz = Δzᶜᶜᶜ(i, j, m, grid)
        N²ₘ = (N²[i, j, m] + N²[i, j, m+1]) / 2
        cost = Δb * Δz + N²ₘ * Δz^2 / 2
        stops = D + cost ≥ e★
        s += ifelse(stops, crossing_distance(Δb, N²ₘ, Δz, e★ - D), Δz)
        D = ifelse(stops, e★, D + cost)
        Δb += N²ₘ * Δz
        m -= 1
    end
    return s # the ground stops what the stratification does not
end

@inline function fill_mixing_length!(ℓ, i, j, grid, ml::IntegralMixingLength, closure, e, N²)
    Cˢ = ml.Cˢ
    for k in 1:grid.Nz+1
        w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
        e★ = w★^2 / (2 * Cˢ^2)
        ascent = parcel_ascent(i, j, k, grid, N², e★)
        descent = parcel_descent(i, j, k, grid, N², e★)
        @inbounds ℓ[i, j, k] = Cˢ * min(ascent, descent)
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute the primary mixing length at every face of every column, by the formulation of
`closure.mixing_length` ([`LocalMinimumMixingLength`](@ref), [`IntegralMixingLength`](@ref) or
[`GradientLimitedMixingLength`](@ref)), from the specific turbulent kinetic energy `e` at the cell
centers and the stored static stability `N²` at the faces. One thread per column.
"""
@kernel function _compute_mixing_length!(ℓ, grid, closure, e, N²)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    fill_mixing_length!(ℓ, i, j, grid, closure_ij.mixing_length, closure_ij, e, N²)
end

"""
$(TYPEDSIGNATURES)

The mixing length at cell center `(i, j, k)`, where the dissipation lives with ``e``. For the local
minimum it is the local minimum at the center, with ``N²`` reconstructed from the two adjacent faces.
For the nonlocal formulations the stored face values reach the center through the slope ``Cˢ`` over
half a cell, ``\\min(ℓₖ, ℓₖ₊₁) + Cˢ Δz / 2``, and the center's own local minimum bounds the result as
well, so that stably stratified air and the neutral surface layer are unchanged from the local
formulation.
"""
@inline mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, closure_fields) =
    mixing_lengthᶜᶜᶜ(i, j, k, grid, closure.mixing_length, closure, e, closure_fields)

@inline mixing_lengthᶜᶜᶜ(i, j, k, grid, ::LocalMinimumMixingLength, closure, e, closure_fields) =
    local_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, closure_fields.N²)

@inline function mixing_lengthᶜᶜᶜ(i, j, k, grid, ml::AbstractMixingLength, closure, e, closure_fields)
    ℓ = closure_fields.ℓ
    ℓᶠ = @inbounds min(ℓ[i, j, k], ℓ[i, j, k+1]) + ml.Cˢ * Δzᶜᶜᶜ(i, j, k, grid) / 2
    return min(ℓᶠ, local_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, closure_fields.N²))
end

#####
##### Diffusivities
#####

# The static stability at the faces, diagnosed once per stage and read by everything downstream
@kernel function _compute_tke_static_stability!(N², grid, closure, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    @inbounds N²[i, j, k] = static_stabilityᶜᶜᶠ(i, j, k, grid, closure_ij.static_stability, buoyancy, tracers)
end

@kernel function _compute_tke_closure_fields!(closure_fields, grid, closure, velocities, tracers)
    i, j, k = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)
    e = tracers[TKE_NAME]
    N² = closure_fields.N²

    # The one diffusivity the closure forms, ℓ √e — with the stored mixing length and √e floored
    # at the minimum TKE and reconstructed from the centers to the face — which the stability
    # functions scale into Kᵘ, Kᶜ and Kᵉ.
    ℓ = @inbounds closure_fields.ℓ[i, j, k]
    K = ℓ * ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure_ij, e)

    Sᵘ = momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, N²)
    Sᶜ = tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, N²)
    Sᵉ = tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, N²)

    Kᵘ = min(Sᵘ * K, closure_ij.maximum_viscosity)
    Kᶜ = min(Sᶜ * K, closure_ij.maximum_tracer_diffusivity)
    Kᵉ = min(Sᵉ * K, closure_ij.maximum_tke_diffusivity)

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

Buoyancy production ``-Kᶜ N²`` at (Center, Center, Face) from the stored diffusivity and static
stability; negative in stable stratification.
"""
@inline buoyancy_productionᶜᶜᶠ(i, j, k, grid, Kᶜ, N²) = @inbounds -Kᶜ[i, j, k] * N²[i, j, k]

"""
$(TYPEDSIGNATURES)

The dissipation rate ``ε = Sᴰ e^{3/2} / ℓ`` at cell centers, from the specific turbulent kinetic
energy field `e`, the velocities and the closure fields, whose stored ``ℓ`` and ``N²`` it reads;
a diagnostic, evaluated in a `KernelFunctionOperation` at (Center, Center, Center) to compare the
closure's TKE budget with a large-eddy simulation's. Negative ``e`` dissipates nothing.
"""
@inline function dissipationᶜᶜᶜ(i, j, k, grid, closure, e, velocities, closure_fields)
    eᵢ = max(0, @inbounds e[i, j, k])
    ℓ = mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, closure_fields)
    Sᴰ = dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure, velocities, closure_fields.N²)
    return Sᴰ * eᵢ * sqrt(eᵢ) / ℓ
end

"""
$(TYPEDSIGNATURES)

The rate at which the sinks of the TKE equation remove turbulent kinetic energy, ``-Lᵉ ≥ 0``:
the dissipation rate ``ω = Sᴰ \\sqrt{e} / ℓ`` — or, where ``e`` is negative, the damping rate
``1/τ`` — plus the negative part of the buoyancy flux divided by ``e``, where there is TKE to
remove. Following CATKE, these are the terms treated implicitly in ``e``, so that ``e`` stays
positive for any time step.
"""
@inline function tke_sink_rate(i, j, k, grid, closure, e, B, velocities, closure_fields)
    eᵐⁱⁿ = closure.minimum_tke
    eᵢ = @inbounds e[i, j, k]
    ℓ = mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, closure_fields)
    Sᴰ = dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure, velocities, closure_fields.N²)

    # `minimum_tke` floors only the turbulent velocity of the mixing length above; the dissipation
    # rate follows √e all the way down, so that ε ∝ e^{3/2} below the floor too (as in CATKE).
    # The `abs` keeps the unselected branch of the `ifelse` from taking √ of a negative number.
    τ = closure.negative_tke_damping_time_scale
    ω = ifelse(eᵢ < 0, 1 / τ, Sᴰ * sqrt(abs(eᵢ)) / ℓ)

    B⁻ = min(0, B)
    ωᴮ = -B⁻ / max(eᵢ, eᵐⁱⁿ) * (eᵢ > eᵐⁱⁿ)

    return ω + ωᴮ
end

# The linear implicit coefficient Lᵉ of `∂t e = Lᵉ e + ⋯`, at cell centers, from the stored `Kᶜ`,
# `N²` and the raw specific TKE. Launched after the diffusivity kernel, since it reads `Kᶜ` at the
# faces above and below the cell.
@kernel function _compute_tke_implicit_linear_coefficient!(Lᵉ, grid, closure, closure_fields, velocities, tracers)
    i, j, k = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)
    e = tracers[TKE_NAME]
    N² = closure_fields.N²

    B = ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_productionᶜᶜᶠ, closure_fields.Kᶜ, N²)
    ω = tke_sink_rate(i, j, k, grid, closure_ij, e, B, velocities, closure_fields)
    active = !inactive_cell(i, j, k, grid)

    @inbounds Lᵉ[i, j, k] = - ω * active
end

# Called from `update_state!`, where every tracer — `ρe` included — momentarily holds its specific
# value, so the kernels read `e` directly from the tracer. The static stability is diagnosed
# first, then the mixing length column by column from it; the diffusivities and the implicit
# coefficient then read both stored fields.
function Oceananigans.TurbulenceClosures.compute_closure_fields!(closure_fields,
                                                         closure::FlavorOfTKEClosure,
                                                         model; parameters = :xyz)
    grid = model.grid
    arch = grid.architecture
    tracers = Oceananigans.TurbulenceClosures.buoyancy_tracers(model)
    buoyancy = Oceananigans.TurbulenceClosures.buoyancy_force(model)

    launch!(arch, grid, parameters, _compute_tke_static_stability!,
            closure_fields.N², grid, closure, tracers, buoyancy)

    launch!(arch, grid, :xy, _compute_mixing_length!,
            closure_fields.ℓ, grid, closure, tracers[TKE_NAME], closure_fields.N²)

    launch!(arch, grid, parameters, _compute_tke_closure_fields!,
            closure_fields, grid, closure, model.velocities, tracers)

    launch!(arch, grid, parameters, _compute_tke_implicit_linear_coefficient!,
            closure_fields.Lᵉ, grid, closure, closure_fields, model.velocities, tracers)

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
@kernel function _add_tke_tendencies!(Gρe, grid, closure, closure_fields, velocities, ρe, ρ)
    i, j, k = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)

    P = ℑbzᵃᵃᶜ(i, j, k, grid, shear_productionᶜᶜᶠ, closure_fields.Kᵘ, velocities.u, velocities.v)
    B = ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_productionᶜᶜᶠ, closure_fields.Kᶜ, closure_fields.N²)
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
    ρ = AtmosphereModels.total_density(model.dynamics)

    launch!(arch, grid, :xyz, _add_tke_tendencies!,
            Gⁿ[TKE_NAME], grid, closure, closure_fields,
            model.velocities, model.tracers[TKE_NAME], ρ)

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
              "├── mixing_length: ", summary(closure.mixing_length), " (Cˢ = ", prettysummary(closure.mixing_length.Cˢ), ")", '\n',
              "├── stability_functions: ", summary(closure.stability_functions), '\n')
    show_stability_function_lines(io, closure.stability_functions, "│   ")
    print(io, '\n',
              "├── static_stability: ", summary(closure.static_stability), '\n',
              "├── maximum_viscosity: ", prettysummary(closure.maximum_viscosity), '\n',
              "├── maximum_tracer_diffusivity: ", prettysummary(closure.maximum_tracer_diffusivity), '\n',
              "├── maximum_tke_diffusivity: ", prettysummary(closure.maximum_tke_diffusivity), '\n',
              "├── minimum_tke: ", prettysummary(closure.minimum_tke), '\n',
              "└── negative_tke_damping_time_scale: ", prettysummary(closure.negative_tke_damping_time_scale))
end

# Kᵘ, Kᶜ, Kᵉ and Lᵉ are recomputed exactly by update_state! from the prognostic state alone
# but this may change in the future if the closure becomes dependent on the previous state.
Oceananigans.prognostic_state(::TKEClosureFields) = nothing
Oceananigans.restore_prognostic_state!(closure_fields::TKEClosureFields, ::Nothing) = closure_fields
