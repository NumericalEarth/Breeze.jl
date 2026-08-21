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
##### other scalar; the local sources and sinks are applied by the closure in `step_closure_prognostics!`.
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
scale of each diffusivity.

Fields
======

$(TYPEDFIELDS)
"""
Base.@kwdef struct TKEMixingLength{FT}
    "coefficient of the stratification length ``ℓᴺ = Cᴺ \\sqrt{e} / N``; the default is
     Deardorff's ``0.76 \\sqrt{e} / N`` ([Deardorff 1980](@cite Deardorff1980))"
    Cᴺ :: FT = 0.76
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

Fields
======

$(TYPEDFIELDS)
"""
struct TKEBasedTurbulenceClosure{TD, ML, SF, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    "the primary mixing length ``ℓ``"
    mixing_length :: ML
    "the stability functions ``Sᵘ, Sᶜ, Sᵉ, Sᴰ``"
    stability_functions :: SF
    "upper bound on ``Kᵘ``, m² s⁻¹"
    maximum_viscosity :: FT
    "upper bound on ``Kᶜ``, m² s⁻¹"
    maximum_tracer_diffusivity :: FT
    "upper bound on ``Kᵉ``, m² s⁻¹"
    maximum_tke_diffusivity :: FT
    "floor on ``e`` inside ``\\sqrt{e}``, m² s⁻²"
    minimum_tke :: FT
    "time scale on which negative ``e`` is damped, s"
    negative_tke_damping_time_scale :: FT
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

Precomputed fields for [`TKEBasedTurbulenceClosure`](@ref).

Fields
======

$(TYPEDFIELDS)
"""
struct TKEClosureFields{K, KC}
    "eddy diffusivity for momentum ``Kᵘ``, at (Center, Center, Face)"
    Kᵘ :: K
    "eddy diffusivity for scalars ``Kᶜ``, at (Center, Center, Face)"
    Kᶜ :: K
    "eddy diffusivity for turbulent kinetic energy ``Kᵉ``, at (Center, Center, Face)"
    Kᵉ :: K
    "the primary mixing length ``ℓ``, at (Center, Center, Face); a diagnostic"
    ℓ :: K
    "per-tracer diffusivity lookup, indexed by the tracer's position among the closure scalars"
    tupled_tracer_diffusivities :: KC
end

Adapt.adapt_structure(to, fields::TKEClosureFields) =
    TKEClosureFields(adapt(to, fields.Kᵘ),
                     adapt(to, fields.Kᶜ),
                     adapt(to, fields.Kᵉ),
                     adapt(to, fields.ℓ),
                     adapt(to, fields.tupled_tracer_diffusivities))

BoundaryConditions.fill_halo_regions!(fields::TKEClosureFields, args...; kw...) =
    fill_halo_regions!((fields.Kᵘ, fields.Kᶜ, fields.Kᵉ, fields.ℓ), args...; kw...)

function Oceananigans.TurbulenceClosures.build_closure_fields(grid, clock, tracer_names, bcs,
                                                      closure::FlavorOfTKEClosure)
    face_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()))
    default_bcs = (Kᵘ = face_bcs, Kᶜ = face_bcs, Kᵉ = face_bcs, ℓ = face_bcs)
    bcs = merge(default_bcs, bcs)

    Kᵘ = ZFaceField(grid, boundary_conditions=bcs.Kᵘ)
    Kᶜ = ZFaceField(grid, boundary_conditions=bcs.Kᶜ)
    Kᵉ = ZFaceField(grid, boundary_conditions=bcs.Kᵉ)
    ℓ  = ZFaceField(grid, boundary_conditions=bcs.ℓ)

    # Indexed by the `Val(id)` the model hands to `diffusivity`: TKE is transported with `Kᵉ`
    # and every other scalar with `Kᶜ`.
    tracer_diffusivities = NamedTuple(name => name === TKE_NAME ? Kᵉ : Kᶜ for name in tracer_names)

    return TKEClosureFields(Kᵘ, Kᶜ, Kᵉ, ℓ, tracer_diffusivities)
end

@inline Oceananigans.TurbulenceClosures.viscosity_location(::FlavorOfTKEClosure) = (Center(), Center(), Face())
@inline Oceananigans.TurbulenceClosures.diffusivity_location(::FlavorOfTKEClosure) = (Center(), Center(), Face())

@inline Oceananigans.TurbulenceClosures.viscosity(::FlavorOfTKEClosure, fields) = fields.Kᵘ

@inline Oceananigans.TurbulenceClosures.diffusivity(::FlavorOfTKEClosure, fields, ::Val{id}) where id =
    fields.tupled_tracer_diffusivities[id]

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

The primary mixing length ``ℓ = \\min(z, ℓᴺ)`` at (Center, Center, Face), given the turbulent
velocity ``w★ = \\sqrt{e}`` there.
"""
@inline function mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, w★, tracers, buoyancy)
    d = height_above_bottomᶜᶜᶠ(i, j, k, grid)
    ℓᴺ = stratification_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, w★, tracers, buoyancy)
    ℓ = min(d, ℓᴺ)
    return ifelse(isnan(ℓ), d, ℓ)
end

"""
$(TYPEDSIGNATURES)

`mixing_lengthᶜᶜᶠ` at cell centers, where the dissipation lives with ``e``.
"""
@inline function mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, w★, tracers, buoyancy)
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
    ℓ = mixing_lengthᶜᶜᶠ(i, j, k, grid, closure_ij, w★, tracers, buoyancy)

    Sᵘ = momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)
    Sᶜ = tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)
    Sᵉ = tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)

    Kᵘ = min(Sᵘ * ℓ * w★, closure_ij.maximum_viscosity)
    Kᶜ = min(Sᶜ * ℓ * w★, closure_ij.maximum_tracer_diffusivity)
    Kᵉ = min(Sᵉ * ℓ * w★, closure_ij.maximum_tke_diffusivity)

    FT = eltype(grid)
    @inbounds begin
        closure_fields.ℓ[i, j, k]  = mask_diffusivity(i, j, k, grid, FT(ℓ))
        closure_fields.Kᵘ[i, j, k] = mask_diffusivity(i, j, k, grid, FT(Kᵘ))
        closure_fields.Kᶜ[i, j, k] = mask_diffusivity(i, j, k, grid, FT(Kᶜ))
        closure_fields.Kᵉ[i, j, k] = mask_diffusivity(i, j, k, grid, FT(Kᵉ))
    end
end

# Called from `update_state!`, where every tracer — `ρe` included — momentarily holds its specific
# value, so the kernel reads `e` directly from the tracer.
function Oceananigans.TurbulenceClosures.compute_closure_fields!(closure_fields,
                                                         closure::FlavorOfTKEClosure,
                                                         model; parameters = :xyz)
    grid = model.grid
    arch = grid.architecture
    tracers = Oceananigans.TurbulenceClosures.buoyancy_tracers(model)
    buoyancy = Oceananigans.TurbulenceClosures.buoyancy_force(model)

    launch!(arch, grid, parameters, _compute_tke_closure_fields!,
            closure_fields, grid, closure, model.velocities, tracers, buoyancy)

    return nothing
end

#####
##### The TKE equation
#####

"""
$(TYPEDSIGNATURES)

Shear production ``Kᵘ S²`` at (Center, Center, Face).
"""
@inline shear_productionᶜᶜᶠ(i, j, k, grid, Kᵘ, u, v) =
    @inbounds Kᵘ[i, j, k] * shearᶜᶜᶠ(i, j, k, grid, u, v)

"""
$(TYPEDSIGNATURES)

Buoyancy production ``-Kᶜ N²`` at (Center, Center, Face); negative in stable stratification.
"""
@inline buoyancy_productionᶜᶜᶠ(i, j, k, grid, Kᶜ, buoyancy, tracers) =
    @inbounds -Kᶜ[i, j, k] * ∂z_b(i, j, k, grid, buoyancy, tracers)

"""
$(TYPEDSIGNATURES)

Advance the local part of the TKE budget, ``∂_t e = P + B - ε``, over `Δt`. Transport is left to
the ordinary scalar machinery acting on the `ρe` tracer, so this kernel is purely columnwise.

Production and buoyancy flux are formed at faces, where ``Kᵘ``, ``Kᶜ``, ``S²`` and ``N²`` live,
and reconstructed to centers. Following CATKE, the sources — shear production and a positive
buoyancy flux — are explicit, while the sinks — dissipation and a negative buoyancy flux — are
implicit in the rate, so that the update

```math
e⁺ = \\frac{e + Δt (P + B⁺)}{1 + Δt (ω + |B⁻| / e)}, \\qquad ω = Sᴰ \\sqrt{e} / ℓ,
```

is positive for any `Δt`. Negative ``e`` is damped on `negative_tke_damping_time_scale` instead.
"""
@kernel function _step_tke!(ρe, grid, closure, closure_fields, velocities, tracers, buoyancy, ρ, Δt)
    i, j, k = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)

    ρᵢ = @inbounds ρ[i, j, k]
    e = @inbounds ρe[i, j, k] / ρᵢ

    P = ℑbzᵃᵃᶜ(i, j, k, grid, shear_productionᶜᶜᶠ, closure_fields.Kᵘ, velocities.u, velocities.v)
    B = ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_productionᶜᶜᶠ, closure_fields.Kᶜ, buoyancy, tracers)
    B⁺ = max(0, B)
    B⁻ = min(0, B)

    eᵐⁱⁿ = closure_ij.minimum_tke
    w★ = sqrt(max(eᵐⁱⁿ, e))
    ℓ = mixing_lengthᶜᶜᶜ(i, j, k, grid, closure_ij, w★, tracers, buoyancy)
    Sᴰ = dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)

    # The dissipation rate ε/e, or the damping rate of negative TKE
    τ = closure_ij.negative_tke_damping_time_scale
    ω = ifelse(e < 0, 1 / τ, Sᴰ * w★ / ℓ)

    # The negative buoyancy flux as a rate, where there is TKE to remove
    ωᴮ = -B⁻ / max(e, eᵐⁱⁿ) * (e > eᵐⁱⁿ)

    e⁺ = (e + Δt * (P + B⁺)) / (1 + Δt * (ω + ωᴮ))

    @inbounds ρe[i, j, k] = ρᵢ * e⁺
end

function Oceananigans.TurbulenceClosures.step_closure_prognostics!(closure_fields,
                                                           closure::FlavorOfTKEClosure,
                                                           model, Δt)
    grid = model.grid
    arch = grid.architecture
    tracers = Oceananigans.TurbulenceClosures.buoyancy_tracers(model)
    buoyancy = Oceananigans.TurbulenceClosures.buoyancy_force(model)
    ρ = AtmosphereModels.total_density(model.dynamics)

    launch!(arch, grid, :xyz, _step_tke!,
            model.tracers[TKE_NAME], grid, closure, closure_fields,
            model.velocities, tracers, buoyancy, ρ, convert(eltype(grid), Δt))

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
