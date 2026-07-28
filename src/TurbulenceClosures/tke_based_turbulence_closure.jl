#####
##### `TKEBasedTurbulenceClosure`: an eddy-diffusivity-only, prognostic-TKE closure
#####
##### ν = Cᴷ ℓ √e ,  K = ν / Pr ,  ε = Cᵋ e^{3/2} / ℓ
##### ∂e/∂t = P + B - ε + transport ,  P = ν S² ,  B = -K N²
#####
##### Transport is the ordinary scalar machinery acting on the `:ρtke` tracer; the local source and
##### sink are applied once per outer step in `step_closure_prognostics!`.
#####
##### Like every Breeze scalar the tracer holds the density-weighted `ρ e`. The closure works in
##### specific `e`: `compute_closure_fields!` divides once into `closure_fields.e`, and
##### `step_closure_prognostics!` brackets its update with `/ρ … *ρ`.
#####

"""
$(TYPEDEF)

An eddy-diffusivity closure carrying one prognostic equation for the subgrid turbulent kinetic
energy ``e``,

```math
ν = Cᴷ ℓ \\sqrt{e}, \\qquad K = ν / \\mathrm{Pr}, \\qquad ε = Cᵋ e^{3/2} / ℓ,
```

```math
∂e/∂t = P + B - ε + \\text{transport}, \\qquad P = ν S², \\qquad B = -K N²,
```

with the mixing length ``ℓ`` supplied by a dispatched component (default `MesoscaleLengthScale`)
and the turbulent Prandtl number closed on the gradient Richardson number,

```math
\\mathrm{Pr} = \\mathrm{Pr₀} \\left(1 + C^{Ri} \\frac{Ri⁺}{1 + Ri⁺}\\right).
```

# The two coefficients

Only ``Cᴷ`` and ``Cμ`` are stored; the dissipation coefficient, the surface turbulence level and
the stress coefficient all derive from them (`dissipation_coefficient`, `surface_tke_coefficient`,
`stress_coefficient`). This pair is chosen because it *separates*:

  - **``Cμ`` alone sets the turbulence level**, ``e/u_\\star² = (Cμ)^{-1/2}``, independently of
    ``Cᴷ``. It is the ``k``–``ε`` coefficient in ``ν = Cμ e²/ε``, so it is directly comparable
    across closure families: 0.058 here (MYNN), 0.094 (MY82), 0.090 (standard ``k``–``ε``),
    0.148 (MYJ), 0.200 (SHOC).

  - **``Cᴷ`` carries log-layer consistency.** In a neutral constant-flux layer with ``ℓᵍ = a z``,
    the logarithmic wind profile constrains only the combination ``Cˢ a = κ``, where
    ``Cˢ = Cᴷ / (Cμ)^{1/4}``. With ``a = κ`` — the plain geometric branch this closure uses — that
    reads ``Cˢ = 1``, i.e. the locus **``Cμ = Cᴷ⁴``**. Off that locus a column fitted for a von
    Kármán constant returns ``κ_\\mathrm{eff} = Cˢ κ`` instead of ``κ``.

The defaults are MYNN's, ``Cᴷ = 0.4903`` and ``Cμ = 0.0578``, which satisfy ``Cμ = Cᴷ⁴`` exactly:
initialized consistent for mesoscale modelling in which no turbulent motion is resolved. Both
remain independently settable, and `stress_coefficient` is exposed so that a departure from the
locus is visible rather than silent. `MY82Coefficients` and `MYJCoefficients` return the other two
published sets; all three sit on the locus, because ``Cμ = Cᴷ⁴`` is equivalent to
``S_M(\\text{neutral}) = B₁^{-1/3}``, which every member of the Mellor–Yamada line satisfies.

Constants carried over from a large-eddy subgrid model do *not* belong here: with ``ℓ = Δ`` the
relation is broken on purpose, because the filter width is not the equilibrium mixing length.
Anchoring this closure on Deardorff's ``Cᴷ = 0.1`` would imply ``e/u_\\star² = 100``, i.e.
``q = 14 u_\\star`` against an observed ``≈ 2.5``.

Fields
======

$(TYPEDFIELDS)
"""
struct TKEBasedTurbulenceClosure{TD, ML, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    "mixing-length formulation; a dispatched component"
    mixing_length :: ML
    "``Cᴷ``, the diffusivity coefficient in ``ν = Cᴷ ℓ \\sqrt{e}``"
    Cᴷ :: FT
    "``Cμ``, the ``k``–``ε`` coefficient; sets the turbulence level ``e/u_\\star² = (Cμ)^{-1/2}``"
    Cμ :: FT
    "``\\mathrm{Pr₀}``, the turbulent Prandtl number in neutral stratification"
    Pr₀ :: FT
    "``C^{Ri}``, the growth of ``\\mathrm{Pr}`` with the gradient Richardson number"
    CRi :: FT
    "floor on ``e``, m² s⁻²; also the value subtracted from the ``ℓᵗ`` integrand"
    eᵐⁱⁿ :: FT
end

const TKEClosureArray{TD} = AbstractArray{<:TKEBasedTurbulenceClosure{TD}} where TD

"""Either a single `TKEBasedTurbulenceClosure` or an ensemble array of them."""
const FlavorOfTKEClosure{TD} = Union{TKEBasedTurbulenceClosure{TD}, TKEClosureArray{TD}} where TD

function TKEBasedTurbulenceClosure{TD}(mixing_length::ML, Cᴷ::FT, Cμ::FT, Pr₀::FT,
                                       CRi::FT, eᵐⁱⁿ::FT) where {TD, ML, FT}
    return TKEBasedTurbulenceClosure{TD, ML, FT}(mixing_length, Cᴷ, Cμ, Pr₀, CRi, eᵐⁱⁿ)
end

"""
$(TYPEDSIGNATURES)

Construct a `TKEBasedTurbulenceClosure`. The defaults are MYNN's coefficients, which lie on the
neutral log-layer locus ``Cμ = Cᴷ⁴``, so `stress_coefficient` is one.

```jldoctest
using Breeze

closure = TKEBasedTurbulenceClosure()
round(stress_coefficient(closure), digits=4)

# output
1.0
```
"""
function TKEBasedTurbulenceClosure(time_discretization::TD = VerticallyImplicitTimeDiscretization(),
                                   FT = Oceananigans.defaults.FloatType;
                                   mixing_length = MesoscaleLengthScale(),
                                   Cᴷ = 0.4903,
                                   Cμ = 0.0578,
                                   Pr₀ = 0.74,
                                   CRi = 3,
                                   eᵐⁱⁿ = 1e-6) where TD

    mixing_length = convert_eltype(FT, mixing_length)

    return TKEBasedTurbulenceClosure{TD}(mixing_length,
                                         convert(FT, Cᴷ),
                                         convert(FT, Cμ),
                                         convert(FT, Pr₀),
                                         convert(FT, CRi),
                                         convert(FT, eᵐⁱⁿ))
end

TKEBasedTurbulenceClosure(FT::DataType; kw...) =
    TKEBasedTurbulenceClosure(VerticallyImplicitTimeDiscretization(), FT; kw...)

@inline convert_eltype(::Type{FT}, m::MesoscaleLengthScale) where FT =
    MesoscaleLengthScale{FT}(; Dict(p => getproperty(m, p) for p in propertynames(m))...)
@inline convert_eltype(::Type{FT}, m::MesoscaleLengthScale{FT}) where FT = m

"""
$(TYPEDSIGNATURES)

`TKEBasedTurbulenceClosure` with the Mellor–Yamada (1982) constants, ``B₁ = 16.6``. The 1982 set
was fit to neutral laboratory and tower data; its critical gradient Richardson number of 0.19 makes
stable turbulence decay too readily.
"""
MY82Coefficients(args...; kw...) = TKEBasedTurbulenceClosure(args...; Cᴷ = 0.5544, Cμ = 0.0945, kw...)

"""
$(TYPEDSIGNATURES)

`TKEBasedTurbulenceClosure` with the Janjić (2001) MYJ constants, ``B₁ = 11.88``, which support a
nonzero equilibrium up to a gradient Richardson number of 0.505.
"""
MYJCoefficients(args...; kw...) = TKEBasedTurbulenceClosure(args...; Cᴷ = 0.6198, Cμ = 0.1476, kw...)

#####
##### Derived coefficients
#####

"""
$(TYPEDSIGNATURES)

``Cᴷ``, the coefficient in ``ν = Cᴷ ℓ \\sqrt{e}``.
"""
@inline diffusivity_coefficient(closure::TKEBasedTurbulenceClosure) = closure.Cᴷ

"""
$(TYPEDSIGNATURES)

``Cᵋ = Cμ / Cᴷ``, the coefficient in ``ε = Cᵋ e^{3/2} / ℓ``.
"""
@inline dissipation_coefficient(closure::TKEBasedTurbulenceClosure) = closure.Cμ / closure.Cᴷ

"""
$(TYPEDSIGNATURES)

``C^{\\mathrm{sfc}} = (Cμ)^{-1/2}``, the neutral surface-layer turbulence level ``e/u_\\star²``.

This is simultaneously the surface floor and the log-layer equilibrium: with ``ℓᵍ = a z``, local
equilibrium gives ``e/u_\\star² = (Cˢ Cᵋ)^{-2/3} = (Cμ)^{-1/2}`` identically, on *or* off the
log-law locus. The floor is therefore never an independent constraint fighting the interior.
"""
@inline surface_tke_coefficient(closure::TKEBasedTurbulenceClosure) = inv(sqrt(closure.Cμ))

"""
$(TYPEDSIGNATURES)

``Cˢ = Cᴷ / (Cμ)^{1/4}``, the stress coefficient. In a neutral constant-flux layer the closure
collapses onto Prandtl's mixing-length model ``ν = (Cˢ ℓᵍ)² S``, so ``Cˢ = 1`` is exactly the
log-law locus ``Cμ = Cᴷ⁴``, and a column off the locus reports ``κ_\\mathrm{eff} = Cˢ κ``.

A diagnostic: nothing in the closure enforces it, which is the point — the constraint is defaulted
to and then checked.
"""
@inline stress_coefficient(closure::TKEBasedTurbulenceClosure) = closure.Cᴷ / sqrt(sqrt(closure.Cμ))

#####
##### Tracer wiring
#####

"""The name of the prognostic TKE tracer. It holds ``ρ e``, matching every other Breeze scalar."""
const TKE_NAME = :ρtke

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

Precomputed fields for `TKEBasedTurbulenceClosure`.

Fields
======

$(TYPEDFIELDS)
"""
struct TKEClosureFields{V, C, T, KC}
    "eddy viscosity ``ν``, at (Center, Center, Face)"
    νₑ :: V
    "eddy diffusivity ``K = ν/\\mathrm{Pr}``, at (Center, Center, Face)"
    κₑ :: V
    "master mixing length ``ℓ``, at (Center, Center, Face), where it closes ``ν``"
    ℓ :: V
    "master mixing length at (Center, Center, Center), where it closes ``ε``"
    ℓᶜ :: C
    "specific turbulent kinetic energy ``e = ρe/ρ``, at (Center, Center, Center)"
    e :: C
    "turbulence length scale ``ℓᵗ``, one value per column"
    ℓᵗ :: T
    "squared friction velocity ``u_\\star²``, one value per column"
    u★² :: T
    "per-tracer diffusivity lookup, indexed by tracer name"
    tupled_tracer_diffusivities :: KC
end

Adapt.adapt_structure(to, fields::TKEClosureFields) =
    TKEClosureFields(adapt(to, fields.νₑ),
                     adapt(to, fields.κₑ),
                     adapt(to, fields.ℓ),
                     adapt(to, fields.ℓᶜ),
                     adapt(to, fields.e),
                     adapt(to, fields.ℓᵗ),
                     adapt(to, fields.u★²),
                     adapt(to, fields.tupled_tracer_diffusivities))

BoundaryConditions.fill_halo_regions!(fields::TKEClosureFields, args...; kw...) =
    fill_halo_regions!((fields.νₑ, fields.κₑ, fields.ℓ, fields.ℓᶜ, fields.e), args...; kw...)

function Oceananigans.TurbulenceClosures.build_closure_fields(grid, clock, tracer_names, bcs,
                                                      closure::FlavorOfTKEClosure)
    face_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()))
    default_bcs = (νₑ = face_bcs, κₑ = face_bcs, ℓ = face_bcs)
    bcs = merge(default_bcs, bcs)

    νₑ = ZFaceField(grid, boundary_conditions=bcs.νₑ)
    κₑ = ZFaceField(grid, boundary_conditions=bcs.κₑ)
    ℓ  = ZFaceField(grid, boundary_conditions=bcs.ℓ)
    ℓᶜ = CenterField(grid)
    e  = CenterField(grid)
    ℓᵗ = Field{Center, Center, Nothing}(grid)
    u★² = Field{Center, Center, Nothing}(grid)

    # Indexed by the `Val(id)` the model hands to `diffusivity`. TKE is transported with the eddy
    # viscosity, i.e. a turbulent Schmidt number of one.
    tracer_diffusivities = NamedTuple(name => name === TKE_NAME ? νₑ : κₑ for name in tracer_names)

    return TKEClosureFields(νₑ, κₑ, ℓ, ℓᶜ, e, ℓᵗ, u★², tracer_diffusivities)
end

@inline Oceananigans.TurbulenceClosures.viscosity_location(::FlavorOfTKEClosure) = (Center(), Center(), Face())
@inline Oceananigans.TurbulenceClosures.diffusivity_location(::FlavorOfTKEClosure) = (Center(), Center(), Face())

@inline Oceananigans.TurbulenceClosures.viscosity(::FlavorOfTKEClosure, fields) = fields.νₑ

@inline Oceananigans.TurbulenceClosures.diffusivity(::FlavorOfTKEClosure, fields, ::Val{id}) where id =
    fields.tupled_tracer_diffusivities[id]

#####
##### Kernel helpers
#####

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

"""
$(TYPEDSIGNATURES)

Squared vertical shear ``S² = (∂_z u)² + (∂_z v)²`` at (Center, Center, Face).
"""
@inline function shearᶜᶜᶠ(i, j, k, grid, u, v)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, v)
    return ∂z_u² + ∂z_v²
end

"""
$(TYPEDSIGNATURES)

Zero the diffusivity on the periphery and poison inactive nodes with `NaN`, so that mixing across
an immersed boundary shows up as a `NaN` rather than as a quiet wrong answer.

This mirrors Oceananigans' `mask_diffusivity`
(`TKEBasedVerticalDiffusivities.jl`) exactly, including the `NaN`, which is a deliberate debugging
canary upstream. On the `RectilinearGrid`s this closure targets `inactive_node` implies
`peripheral_node`, so the canary never fires; keeping it means it is already in place the day
immersed boundaries come into scope.
"""
@inline function mask_diffusivity(i, j, k, grid, κ)
    on_periphery = peripheral_node(i, j, k, grid, Center(), Center(), Face())
    within_inactive = inactive_node(i, j, k, grid, Center(), Center(), Face())
    nan = convert(eltype(grid), NaN)
    return ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κ))
end

"""
$(TYPEDSIGNATURES)

Reconstruct a face-located quantity at cell centers, ignoring peripheral faces.

The plain average would drag in the masked boundary value at ``k = 1`` and halve the result there —
which is exactly the cell whose budget decides the near-surface wind profile. Where a face is
peripheral its partner's value is used instead.
"""
@inline function ℑbzᵃᵃᶜ(i, j, k, grid, fᵃᵃᶠ, args...)
    f⁺ = fᵃᵃᶠ(i, j, k+1, grid, args...)
    f⁻ = fᵃᵃᶠ(i, j, k,   grid, args...)

    p⁺ = peripheral_node(i, j, k+1, grid, Center(), Center(), Face())
    p⁻ = peripheral_node(i, j, k,   grid, Center(), Center(), Face())

    f⁺ = ifelse(p⁺, f⁻, f⁺)
    f⁻ = ifelse(p⁻, f⁺, f⁻)

    return (f⁺ + f⁻) / 2
end

"""
$(TYPEDSIGNATURES)

Turbulent velocity ``\\sqrt{e}`` at (Center, Center, Center), with ``e`` floored at `eᵐⁱⁿ`.
"""
@inline function turbulent_velocityᶜᶜᶜ(i, j, k, grid, eᵐⁱⁿ, e)
    eᵢ = @inbounds e[i, j, k]
    return sqrt(max(eᵐⁱⁿ, eᵢ))
end

"""
$(TYPEDSIGNATURES)

Turbulent Prandtl number ``\\mathrm{Pr} = \\mathrm{Pr₀}(1 + C^{Ri} Ri⁺/(1 + Ri⁺))``. It saturates
at ``\\mathrm{Pr₀}(1 + C^{Ri})`` in strongly stable stratification and reduces to
``\\mathrm{Pr₀}`` in neutral and unstable air.

This stands in for the full algebraic stability functions ``S_M(G_M, G_H)``, ``S_H(G_M, G_H)`` of
the Mellor–Yamada hierarchy. The neutral log-layer argument that pins ``Cᴷ`` and ``Cμ`` is
momentum-only and says nothing about ``\\mathrm{Pr₀}``, so it is an independent parameter.
"""
@inline function turbulent_prandtl_number(Pr₀, CRi, Ri)
    Ri⁺ = max(0, Ri)
    return Pr₀ * (1 + CRi * Ri⁺ / (1 + Ri⁺))
end

#####
##### Precomputed fields
#####

@kernel function _floor_tke!(e, tke, eᵐⁱⁿ)
    i, j, k = @index(Global, NTuple)
    @inbounds e[i, j, k] = max(eᵐⁱⁿ, tke[i, j, k])
end

@kernel function _compute_tke_closure_fields!(closure_fields, grid, closure, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)
    e = closure_fields.e

    q★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure_ij.eᵐⁱⁿ, e)   # √e at the face
    q = sqrt(convert(eltype(grid), 2)) * q★                                # q = √(2e)

    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    S² = shearᶜᶜᶠ(i, j, k, grid, velocities.u, velocities.v)

    ℓ = mixing_lengthᶜᶜᶠ(i, j, k, grid, closure_ij.mixing_length, q, N², closure_fields.ℓᵗ)

    ν = closure_ij.Cᴷ * ℓ * q★

    # Ri = N²/S² is unbounded where the shear vanishes. Pr saturates in that limit, so clamping Ri
    # to a large value rather than dividing by zero gives the right answer and no NaN.
    Riᵐᵃˣ = convert(eltype(grid), 1e8)
    Ri = ifelse(S² > 0, N² / S², ifelse(N² > 0, Riᵐᵃˣ, zero(grid)))
    Ri = min(Ri, Riᵐᵃˣ)
    Pr = turbulent_prandtl_number(closure_ij.Pr₀, closure_ij.CRi, Ri)

    # ℓ is needed at centers too, for the dissipation. Evaluate it there rather than interpolating:
    # the surface face is masked, so an interpolated ℓ would be wrong in the first cell.
    qᶜ = sqrt(convert(eltype(grid), 2)) * turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure_ij.eᵐⁱⁿ, e)
    N²ᶜ = ℑbzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    ℓᶜ = mixing_lengthᶜᶜᶜ(i, j, k, grid, closure_ij.mixing_length, qᶜ, N²ᶜ, closure_fields.ℓᵗ)

    FT = eltype(grid)
    @inbounds begin
        closure_fields.ℓ[i, j, k]  = mask_diffusivity(i, j, k, grid, FT(ℓ))
        closure_fields.νₑ[i, j, k] = mask_diffusivity(i, j, k, grid, FT(ν))
        # `K` follows from the capped `ν`; it is not capped independently, so that `Pr = ν/K` is
        # exactly the value `turbulent_prandtl_number` returned even where the cap binds.
        closure_fields.κₑ[i, j, k] = mask_diffusivity(i, j, k, grid, FT(ν / Pr))
        closure_fields.ℓᶜ[i, j, k] = FT(ℓᶜ)
    end
end

function Oceananigans.TurbulenceClosures.compute_closure_fields!(closure_fields,
                                                         closure::FlavorOfTKEClosure,
                                                         model; parameters = :xyz)
    grid = model.grid
    arch = grid.architecture
    tracers = Oceananigans.TurbulenceClosures.buoyancy_tracers(model)
    buoyancy = Oceananigans.TurbulenceClosures.buoyancy_force(model)

    # `update_state!` calls `tracer_density_to_specific!` before it reaches here and
    # `tracer_specific_to_density!` on the way out, so *inside* this function the tracer already
    # holds specific `e` — dividing by ρ again would be a second division. (Outside `update_state!`
    # it holds ρe, which is why `step_closure_prognostics!` brackets its update with /ρ … *ρ.)
    # Only the floor and the halo are applied here: the face-located √e at k = 1 reaches one cell
    # below the surface.
    launch!(arch, grid, :xyz, _floor_tke!,
            closure_fields.e, model.tracers[TKE_NAME], minimum_tke(closure))
    fill_halo_regions!(closure_fields.e)

    # ℓᵗ is a column integral and must be current before the pointwise pass reads it.
    launch!(arch, grid, :xy, _compute_turbulence_length_scale!,
            closure_fields.ℓᵗ, grid, closure, closure_fields.e)

    launch!(arch, grid, parameters, _compute_tke_closure_fields!,
            closure_fields, grid, closure, model.velocities, tracers, buoyancy)

    return nothing
end

@inline minimum_tke(closure::TKEBasedTurbulenceClosure) = closure.eᵐⁱⁿ
@inline minimum_tke(closure::TKEClosureArray) = @allowscalar closure[1].eᵐⁱⁿ

#####
##### The TKE equation
#####

"""
$(TYPEDSIGNATURES)

Shear production ``ν S²`` at (Center, Center, Face).
"""
@inline shear_productionᶜᶜᶠ(i, j, k, grid, νₑ, u, v) =
    @inbounds νₑ[i, j, k] * shearᶜᶜᶠ(i, j, k, grid, u, v)

"""
$(TYPEDSIGNATURES)

Buoyancy production ``-K N²`` at (Center, Center, Face); negative in stable stratification.
"""
@inline buoyancy_productionᶜᶜᶠ(i, j, k, grid, κₑ, buoyancy, tracers) =
    @inbounds -κₑ[i, j, k] * ∂z_b(i, j, k, grid, buoyancy, tracers)

"""
$(TYPEDSIGNATURES)

Advance the local part of the TKE budget, ``∂e/∂t = P + B - ε``, over `Δt`. Transport is left to
the ordinary scalar tendency acting on the `:ρtke` tracer, so this kernel is purely columnwise.

Production and buoyancy are formed at faces, where ``ν``, ``K``, ``S²`` and ``N²`` all live, and
reconstructed to centers. Dissipation is treated implicitly in the rate ``ω = ε/e``,

```math
eⁿ⁺¹ = \\frac{e^\\star}{1 + Δt \\, Cᵋ \\sqrt{e^\\star} / ℓ},
```

which is unconditionally positive for any `Δt`. An explicit ``-ε Δt`` is not, and TKE that goes
negative takes ``ν`` with it.
"""
@kernel function _step_tke!(ρe, grid, closure, closure_fields, velocities, tracers, buoyancy, ρ, Δt)
    i, j, k = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)

    ρᵢ = @inbounds ρ[i, j, k]
    e = @inbounds ρe[i, j, k] / ρᵢ

    P = ℑbzᵃᵃᶜ(i, j, k, grid, shear_productionᶜᶜᶠ, closure_fields.νₑ, velocities.u, velocities.v)
    B = ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_productionᶜᶜᶠ, closure_fields.κₑ, buoyancy, tracers)

    e★ = max(closure_ij.eᵐⁱⁿ, e + Δt * (P + B))

    ℓ = @inbounds closure_fields.ℓᶜ[i, j, k]
    ω = dissipation_coefficient(closure_ij) * sqrt(e★) / ℓ
    e⁺ = e★ / (1 + Δt * ω)

    # The surface floor. On the log-law locus this is the same number the interior equilibrium
    # relaxes to (see `surface_tke_coefficient`), so it is inert in a spun-up constant-flux layer;
    # it matters during spin-up and wherever a column has run down.
    u★² = @inbounds closure_fields.u★²[i, j, 1]
    eᶠˡᵒᵒʳ = surface_tke_coefficient(closure_ij) * u★²
    e⁺ = ifelse((k == 1) & (eᶠˡᵒᵒʳ > e⁺), eᶠˡᵒᵒʳ, e⁺)

    @inbounds ρe[i, j, k] = ρᵢ * max(closure_ij.eᵐⁱⁿ, e⁺)
end

"""
$(TYPEDSIGNATURES)

Squared friction velocity ``u_\\star² = \\sqrt{(τˣ)² + (τʸ)²}/ρ`` per column, read from the bottom
momentum-flux boundary conditions the model is already applying.

Breeze does not call `add_closure_specific_boundary_conditions`, so rather than owning a TKE
boundary condition of its own, the closure diagnoses the surface stress from the momentum
boundary conditions. The two flux components are evaluated at their own staggered locations, which
is exact for a horizontally uniform surface stress and offset by half a cell otherwise.
"""
@kernel function _compute_friction_velocity!(u★², grid, τˣ_bc, τʸ_bc, ρ, clock, model_fields)
    i, j = @index(Global, NTuple)
    τˣ = getbc(τˣ_bc, i, j, grid, clock, model_fields)
    τʸ = getbc(τʸ_bc, i, j, grid, clock, model_fields)
    ρ₁ = @inbounds ρ[i, j, 1]
    @inbounds u★²[i, j, 1] = sqrt(τˣ^2 + τʸ^2) / ρ₁
end

function Oceananigans.TurbulenceClosures.step_closure_prognostics!(closure_fields,
                                                           closure::FlavorOfTKEClosure,
                                                           model, Δt)
    grid = model.grid
    arch = grid.architecture
    tracers = Oceananigans.TurbulenceClosures.buoyancy_tracers(model)
    buoyancy = Oceananigans.TurbulenceClosures.buoyancy_force(model)
    ρ = AtmosphereModels.total_density(model.dynamics)

    launch!(arch, grid, :xy, _compute_friction_velocity!,
            closure_fields.u★², grid,
            model.momentum.ρu.boundary_conditions.bottom,
            model.momentum.ρv.boundary_conditions.bottom,
            ρ, model.clock, fields(model))

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
              "├── Cᴷ:   ", prettysummary(closure.Cᴷ), " (diffusivity coefficient)", '\n',
              "├── Cμ:   ", prettysummary(closure.Cμ), " (turbulence level; e/u★² = ",
                             prettysummary(surface_tke_coefficient(closure)), ")", '\n',
              "├── Cᵋ:   ", prettysummary(dissipation_coefficient(closure)), " (derived)", '\n',
              "├── Cˢ:   ", prettysummary(stress_coefficient(closure)),
                             " (derived; 1 on the log-law locus Cμ = Cᴷ⁴)", '\n',
              "├── Pr₀:  ", prettysummary(closure.Pr₀), '\n',
              "├── CRi:  ", prettysummary(closure.CRi), '\n',
              "├── eᵐⁱⁿ: ", prettysummary(closure.eᵐⁱⁿ), " m² s⁻²", '\n',
              "└── mixing_length: ", summary(closure.mixing_length))
end
