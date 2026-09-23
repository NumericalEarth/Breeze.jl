#####
##### Richardson-number-dependent stability functions
#####
##### Two families of Sᵘ, Sᶜ, Sᵉ, Sᴰ as functions of the gradient Richardson number Ri = N² / S², each
##### carrying twelve endpoints — an unstable value C⁻, a neutral value C⁰ and a stable asymptote C⁺
##### for every one of the four functions — and differing in how they travel between them:
#####
#####   `PiecewiseStabilityFunction`, the form of CATKE (Wagner et al. 2025),
#####
#####     S(Ri) = C⁻                                             for Ri < 0
#####     S(Ri) = C⁰ + (C⁺ − C⁰) clamp((Ri − Ri⁰) / Riᵟ, 0, 1)   for Ri ≥ 0,
#####
#####   constant in unstable air, with a neutral plateau out to Ri⁰ and a linear ramp of width Riᵟ; and
#####
#####   `RationalStabilityFunction`,
#####
#####     S(Ri) = C⁰ + (C^± − C⁰) xᵖ / (1 + xᵖ),   x = |Ri| / Ri^±,
#####
#####   with the branch ± taken from the sign of Ri, which leaves C⁰ continuously in both directions
#####   over the scales Ri⁻ and Ri⁺ and has neither a plateau nor a threshold.
#####
##### Ri is formed from the stored static stability and the vertical shear, at the faces for the
##### diffusivities and from N² and S² reconstructed to the centers for the dissipation. Both families
##### reduce to constants when each function's three endpoints coincide.
#####

#####
##### The Richardson number
#####

# Ri = N² / S², zero where N² = 0 (so that N² = S² = 0 is neutral rather than NaN) and ±∞ where
# only the shear vanishes, which the stability functions map to their stable or unstable value.
@inline richardson_number(N², S²) = ifelse(N² == 0, zero(N²), N² / S²)

"""
$(TYPEDSIGNATURES)

The gradient Richardson number ``Ri = N² / S²`` at (Center, Center, Face), from the stored static
stability `N²` and the squared vertical shear of the velocities; zero where ``N² = 0``, and ``±∞``
where the shear alone vanishes.
"""
@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, N²)
    S² = shearᶜᶜᶠ(i, j, k, grid, velocities.u, velocities.v)
    N²ᵢ = @inbounds N²[i, j, k]
    return richardson_number(N²ᵢ, S²)
end

"""
$(TYPEDSIGNATURES)

The gradient Richardson number at cell centers, from ``N²`` and ``S²`` each reconstructed from the
two adjacent interfaces (ignoring peripheral ones), for the dissipation. CATKE reconstructs ``Ri``
itself; reconstructing its numerator and denominator instead keeps a center between a stable and
an unstable interface with no shear at a finite or infinite ``Ri`` rather than at the `NaN` of
``(+∞ - ∞) / 2``.
"""
@inline function Riᶜᶜᶜ(i, j, k, grid, velocities, N²)
    S² = ℑbzᵃᵃᶜ(i, j, k, grid, shearᶜᶜᶠ, velocities.u, velocities.v)
    N²ᵢ = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, N²)
    return richardson_number(N²ᵢ, S²)
end

#####
##### The piecewise-linear family, in the form of CATKE
#####

"""
$(TYPEDEF)

Stability functions of [`TKEBasedTurbulenceClosure`](@ref) that depend on the gradient Richardson
number ``Ri = N² / S²``, in the form of CATKE ([Wagner et al. 2025](@cite Wagner25catke)). Each of
``Sᵘ, Sᶜ, Sᵉ, Sᴰ`` is the piecewise-linear function

```math
S(Ri) = \\begin{cases}
C⁻ & Ri < 0 \\\\
C⁰ + (C⁺ - C⁰) \\, \\mathrm{clamp}\\left( \\frac{Ri - Ri⁰}{Riᵟ}, 0, 1 \\right) & Ri ≥ 0,
\\end{cases}
```

a constant ``C⁻`` in unstable stratification, the neutral value ``C⁰`` from ``Ri = 0`` to the
onset ``Ri⁰`` of the stable transition, and a linear ramp over the width ``Riᵟ`` to the stable
asymptote ``C⁺``. The dissipation function multiplies the dissipation rate, ``ε = Sᴰ e^{3/2} / ℓ``,
so its stable asymptote is smaller than its neutral value: stratification lengthens the dissipation
length ``ℓ / Sᴰ`` relative to the diffusivities' ``Sᵘ ℓ``.

The Richardson number is formed at each interface from the closure's stored ``N²`` and the squared
vertical shear ([`Riᶜᶜᶠ`](@ref)), and at the cell centers, for the dissipation, from ``N²`` and
``S²`` reconstructed there ([`Riᶜᶜᶜ`](@ref)). Vanishing shear gives ``Ri = ±∞``, which the
piecewise-linear functions take in stride.

The defaults are the twelve endpoints, the onset and the width of CATKE's stability functions,
calibrated against ocean large-eddy simulations by [Wagner et al. (2025)](@cite Wagner25catke),
frozen here rather than inferred. In CATKE's notation ``C⁻, C⁰, C⁺`` are ``Cᵘⁿ, Cˡᵒ, Cʰⁱ`` and
``Ri⁰, Riᵟ`` are ``CRi⁰, CRiᵟ``; CATKE's wall coefficient ``Cˢ = 1.131`` goes with them, see
[`catke_parameters`](@ref). Two things CATKE's calibration relied on are not part of this closure:
the convective and entrainment length scales driven by the surface buoyancy flux, which dominate
CATKE's mixing and lengthen its dissipation length in convecting layers, and the surface flux of
turbulent kinetic energy. Weaker mixing and stronger dissipation in convective boundary layers
than in CATKE are therefore to be expected. In a neutral constant-stress layer these values give
``κ = Cˢ (Cᵘ⁰³ / Cᴰ⁰)^{1/4} = 0.47``, ``e / u_\\star² = 1 / \\sqrt{Cᵘ⁰ Cᴰ⁰} = 1.3`` and
``Pr = 0.98``, against the atmospheric ``0.40``, ``≈ 4`` and ``≈ 0.74`` of
[`ConstantStabilityFunctions`](@ref), which is why the latter remain the default.

[`RationalStabilityFunction`](@ref) is the other Richardson-number-dependent family: the same twelve
endpoints, reached smoothly, with neither a neutral plateau nor a threshold.

Fields
======

$(TYPEDFIELDS)
"""
Base.@kwdef struct PiecewiseStabilityFunction{FT}
    "momentum, unstable (``Ri < 0``)"
    Cᵘ⁻ :: FT = 0.370
    "momentum, neutral"
    Cᵘ⁰ :: FT = 0.361
    "momentum, stable asymptote"
    Cᵘ⁺ :: FT = 0.242
    "tracers, unstable"
    Cᶜ⁻ :: FT = 0.572
    "tracers, neutral"
    Cᶜ⁰ :: FT = 0.369
    "tracers, stable asymptote"
    Cᶜ⁺ :: FT = 0.098
    "turbulent kinetic energy, unstable"
    Cᵉ⁻ :: FT = 1.447
    "turbulent kinetic energy, neutral"
    Cᵉ⁰ :: FT = 7.863
    "turbulent kinetic energy, stable asymptote"
    Cᵉ⁺ :: FT = 0.548
    "dissipation, unstable"
    Cᴰ⁻ :: FT = 0.923
    "dissipation, neutral"
    Cᴰ⁰ :: FT = 1.604
    "dissipation, stable asymptote"
    Cᴰ⁺ :: FT = 0.579
    "Richardson number at the onset of the stable transition"
    Ri⁰ :: FT = 0.254
    "width of the stable transition in Richardson number"
    Riᵟ :: FT = 1.02
end

"""
Deprecated name of [`PiecewiseStabilityFunction`](@ref), kept because scripts, calibration drivers
and stored analyses use it. New code should use `PiecewiseStabilityFunction`, which says which family
it is rather than only that it depends on ``Ri`` — [`RationalStabilityFunction`](@ref) does too.
"""
const RiDependentStabilityFunctions = PiecewiseStabilityFunction

## Mixed integer/float keyword arguments are promoted, as for `ConstantStabilityFunctions`; the
## untyped signature is less specific than the default constructor, which takes over once the
## fourteen coefficients share a type.
PiecewiseStabilityFunction(coefficients::Vararg{Any, 14}) =
    PiecewiseStabilityFunction(promote(coefficients...)...)

@inline convert_eltype(::Type{FT}, sf::PiecewiseStabilityFunction) where FT =
    PiecewiseStabilityFunction{FT}((convert(FT, getfield(sf, name)) for name in fieldnames(PiecewiseStabilityFunction))...)

Base.summary(sf::PiecewiseStabilityFunction{FT}) where FT = "PiecewiseStabilityFunction{$FT}"

# One row of four endpoints in the display, e.g. the four unstable values
stability_endpoint_summary(Cᵘ, Cᶜ, Cᵉ, Cᴰ) = join(prettysummary.((Cᵘ, Cᶜ, Cᵉ, Cᴰ)), ", ")

function show_stability_function_lines(io::IO, sf::PiecewiseStabilityFunction, prefix)
    print(io, prefix, "├── Ri < 0 (Cᵘ⁻, Cᶜ⁻, Cᵉ⁻, Cᴰ⁻): ", stability_endpoint_summary(sf.Cᵘ⁻, sf.Cᶜ⁻, sf.Cᵉ⁻, sf.Cᴰ⁻), '\n',
              prefix, "├── Ri = 0 (Cᵘ⁰, Cᶜ⁰, Cᵉ⁰, Cᴰ⁰): ", stability_endpoint_summary(sf.Cᵘ⁰, sf.Cᶜ⁰, sf.Cᵉ⁰, sf.Cᴰ⁰), '\n',
              prefix, "├── Ri → ∞ (Cᵘ⁺, Cᶜ⁺, Cᵉ⁺, Cᴰ⁺): ", stability_endpoint_summary(sf.Cᵘ⁺, sf.Cᶜ⁺, sf.Cᵉ⁺, sf.Cᴰ⁺), '\n',
              prefix, "└── stable transition: Ri⁰ = ", prettysummary(sf.Ri⁰), ", Riᵟ = ", prettysummary(sf.Riᵟ))
    return nothing
end

function Base.show(io::IO, sf::PiecewiseStabilityFunction)
    print(io, summary(sf), '\n')
    show_stability_function_lines(io, sf, "")
end

"""
$(TYPEDSIGNATURES)

The mixing length and stability functions of CATKE ([Wagner et al. 2025](@cite Wagner25catke)) as
keyword arguments for [`TKEBasedTurbulenceClosure`](@ref): CATKE's wall coefficient ``Cˢ = 1.131`` in
the mixing-length formulation `MixingLength` ([`GradientLimitedMixingLength`](@ref) by default), and
[`PiecewiseStabilityFunction`](@ref) with CATKE's values.

```jldoctest
using Breeze

closure = TKEBasedTurbulenceClosure(; catke_parameters()...)
closure.mixing_length

# output
GradientLimitedMixingLength{Float64} (Cˢ = 1.131)
```

CATKE's convective length scales and surface flux of turbulent kinetic energy are not part of the
closure; see [`PiecewiseStabilityFunction`](@ref).
"""
catke_parameters(MixingLength = GradientLimitedMixingLength) = (mixing_length = MixingLength(Cˢ = 1.131),
                                                                  stability_functions = PiecewiseStabilityFunction())

"""
$(TYPEDSIGNATURES)

The piecewise-linear stability function of [`PiecewiseStabilityFunction`](@ref): `C⁻` for
`Ri < 0`, and otherwise `C⁰` ramping linearly to `C⁺` between `Ri⁰` and `Ri⁰ + Riᵟ`.
"""
@inline function stability_ramp(Ri, C⁻, C⁰, C⁺, Ri⁰, Riᵟ)
    ramp = clamp((Ri - Ri⁰) / Riᵟ, 0, 1)
    C⁺ˢ = C⁰ + (C⁺ - C⁰) * ramp
    return ifelse(Ri < 0, C⁻, C⁺ˢ)
end

@inline momentum_stability_function(sf::PiecewiseStabilityFunction, Ri) =
    stability_ramp(Ri, sf.Cᵘ⁻, sf.Cᵘ⁰, sf.Cᵘ⁺, sf.Ri⁰, sf.Riᵟ)

@inline tracer_stability_function(sf::PiecewiseStabilityFunction, Ri) =
    stability_ramp(Ri, sf.Cᶜ⁻, sf.Cᶜ⁰, sf.Cᶜ⁺, sf.Ri⁰, sf.Riᵟ)

@inline tke_stability_function(sf::PiecewiseStabilityFunction, Ri) =
    stability_ramp(Ri, sf.Cᵉ⁻, sf.Cᵉ⁰, sf.Cᵉ⁺, sf.Ri⁰, sf.Riᵟ)

@inline dissipation_stability_function(sf::PiecewiseStabilityFunction, Ri) =
    stability_ramp(Ri, sf.Cᴰ⁻, sf.Cᴰ⁰, sf.Cᴰ⁺, sf.Ri⁰, sf.Riᵟ)

#####
##### The rational family
#####

"""
$(TYPEDEF)

Stability functions of [`TKEBasedTurbulenceClosure`](@ref) that leave their neutral values smoothly
in both directions, with neither a plateau around ``Ri = 0`` nor a threshold at which the stable
transition begins. Each of ``Sᵘ, Sᶜ, Sᵉ, Sᴰ`` is

```math
S(Ri) = C⁰ + (C^\\pm - C⁰) \\frac{x^p}{1 + x^p}, \\qquad x = \\frac{|Ri|}{Ri^\\pm},
```

where the branch ``\\pm`` is ``-`` for ``Ri < 0`` and ``+`` for ``Ri ≥ 0``. Each function keeps the
three endpoints of [`PiecewiseStabilityFunction`](@ref) — the unstable limit ``C⁻``, now approached as
``Ri → -∞`` rather than held at every ``Ri < 0``, the neutral value ``C⁰`` at ``Ri = 0``, and the
stable asymptote ``C⁺`` as ``Ri → ∞`` — and the four functions share the two transition scales ``Ri⁻``
and ``Ri⁺``, the Richardson numbers at which every function stands halfway between ``C⁰`` and its
unstable or its stable limit. The unstable and the stable side therefore keep independent shapes, as
in the piecewise family, and the twelve endpoints with the two scales are the only free coefficients:
fourteen in all, rising to sixteen only if the exponents are freed as well. The dissipation function
keeps the closure's convention — it multiplies the dissipation rate, ``ε = Sᴰ e^{3/2} / ℓ`` — so a
``Cᴰ⁺`` below ``Cᴰ⁰`` lengthens the dissipation length in stable air, exactly as in the piecewise
family.

With ``p = 1``, the default, the function is the Michaelis–Menten form

```math
S(Ri) = C⁰ + (C^\\pm - C⁰) \\frac{|Ri|}{|Ri| + Ri^\\pm},
```

which covers half its range by ``|Ri| = Ri^\\pm`` and nine tenths by ``|Ri| = 9 Ri^\\pm``. Raising ``p``
sharpens the transition around ``Ri^\\pm`` towards a step; lowering it flattens the approach to the
asymptotes while steepening the departure from neutral. The exponents `p⁻` and `p⁺` are held fixed at
one for calibration, two more free parameters buying little that the scales do not.

``S`` is continuous everywhere, including at ``Ri = 0``, where the two branches meet at ``C⁰``. It is
differentiable there **only for** ``p > 1``, where the one-sided derivatives both vanish. At ``p = 1``
they are ``+(C⁺ - C⁰) / Ri⁺`` from the stable side and ``-(C⁻ - C⁰) / Ri⁻`` from the unstable one, and
generally disagree: the function has a kink at neutral. For ``p < 1`` they are infinite — ``|Ri|^p``
has a cusp of unbounded slope at the origin, so the derivative there is singular rather than merely
two-valued. A closure that is smooth at neutral therefore needs ``p > 1``, which the default,
``p = 1``, does not provide.

The twelve endpoints must be positive, as must the two scales and the two exponents; the constructor
checks all sixteen. Since ``x^p / (1 + x^p)`` lies in ``[0, 1]``, every ``S(Ri)`` lies between ``C⁰``
and the branch endpoint and is therefore positive everywhere. When a function's three endpoints
coincide it is exactly that constant at every ``Ri``, so the constant submodel of
[`ConstantStabilityFunctions`](@ref) is nested in this family with no limit to take.

The evaluation survives ``Ri = 0``, ``Ri = ±∞`` and every finite ratio in between, in `Float32` as in
`Float64`; see [`rational_stability`](@ref).

The defaults carry the twelve endpoints of [`PiecewiseStabilityFunction`](@ref) over unchanged —
CATKE's ocean values ([Wagner et al. 2025](@cite Wagner25catke)), fitted to the piecewise shape and
not to this one — with ``Ri⁻ = Ri⁺ = 0.764``, the midpoint ``Ri⁰ + Riᵟ/2`` of the piecewise ramp,
where that ramp too stands halfway between ``C⁰`` and ``C⁺``. They are a starting point for
calibration, not a calibration: no campaign has fitted this shape.

Fields
======

$(TYPEDFIELDS)
"""
struct RationalStabilityFunction{FT}
    "momentum, unstable limit (``Ri → -∞``)"
    Cᵘ⁻ :: FT
    "momentum, neutral (``Ri = 0``)"
    Cᵘ⁰ :: FT
    "momentum, stable limit (``Ri → ∞``)"
    Cᵘ⁺ :: FT
    "tracers, unstable limit"
    Cᶜ⁻ :: FT
    "tracers, neutral"
    Cᶜ⁰ :: FT
    "tracers, stable limit"
    Cᶜ⁺ :: FT
    "turbulent kinetic energy, unstable limit"
    Cᵉ⁻ :: FT
    "turbulent kinetic energy, neutral"
    Cᵉ⁰ :: FT
    "turbulent kinetic energy, stable limit"
    Cᵉ⁺ :: FT
    "dissipation, unstable limit"
    Cᴰ⁻ :: FT
    "dissipation, neutral"
    Cᴰ⁰ :: FT
    "dissipation, stable limit"
    Cᴰ⁺ :: FT
    "unstable transition scale: the ``|Ri|`` at which every function is halfway from ``C⁰`` to ``C⁻``"
    Ri⁻ :: FT
    "stable transition scale: the ``Ri`` at which every function is halfway from ``C⁰`` to ``C⁺``"
    Ri⁺ :: FT
    "unstable exponent, fixed at one for calibration"
    p⁻ :: FT
    "stable exponent, fixed at one for calibration"
    p⁺ :: FT
end

"""
$(TYPEDSIGNATURES)

Construct [`RationalStabilityFunction`](@ref) from the twelve endpoints, the two transition scales
and the two exponents, every one of which must be positive. Mixed integer and float arguments are
promoted to a single float type.

```jldoctest
using Breeze

RationalStabilityFunction(Cᵘ⁻ = 0.4, Cᵘ⁰ = 0.3, Cᵘ⁺ = 0.2, Ri⁺ = 1)

# output
RationalStabilityFunction{Float64}
├── Ri → -∞ (Cᵘ⁻, Cᶜ⁻, Cᵉ⁻, Cᴰ⁻): 0.4, 0.572, 1.447, 0.923
├── Ri = 0  (Cᵘ⁰, Cᶜ⁰, Cᵉ⁰, Cᴰ⁰): 0.3, 0.369, 7.863, 1.604
├── Ri → ∞  (Cᵘ⁺, Cᶜ⁺, Cᵉ⁺, Cᴰ⁺): 0.2, 0.098, 0.548, 0.579
├── transition scales: Ri⁻ = 0.764, Ri⁺ = 1.0
└── exponents: p⁻ = 1.0, p⁺ = 1.0
```
"""
function RationalStabilityFunction(; Cᵘ⁻ = 0.370, Cᵘ⁰ = 0.361, Cᵘ⁺ = 0.242,
                                     Cᶜ⁻ = 0.572, Cᶜ⁰ = 0.369, Cᶜ⁺ = 0.098,
                                     Cᵉ⁻ = 1.447, Cᵉ⁰ = 7.863, Cᵉ⁺ = 0.548,
                                     Cᴰ⁻ = 0.923, Cᴰ⁰ = 1.604, Cᴰ⁺ = 0.579,
                                     Ri⁻ = 0.764, Ri⁺ = 0.764,
                                     p⁻ = 1, p⁺ = 1)

    endpoints = (Cᵘ⁻, Cᵘ⁰, Cᵘ⁺, Cᶜ⁻, Cᶜ⁰, Cᶜ⁺, Cᵉ⁻, Cᵉ⁰, Cᵉ⁺, Cᴰ⁻, Cᴰ⁰, Cᴰ⁺)

    all(C -> C > 0, endpoints) ||
        throw(ArgumentError("The twelve endpoints of RationalStabilityFunction must be positive, but " *
                            "(Cᵘ⁻, Cᵘ⁰, Cᵘ⁺, Cᶜ⁻, Cᶜ⁰, Cᶜ⁺, Cᵉ⁻, Cᵉ⁰, Cᵉ⁺, Cᴰ⁻, Cᴰ⁰, Cᴰ⁺) = $endpoints"))

    Ri⁻ > 0 && Ri⁺ > 0 ||
        throw(ArgumentError("The transition scales of RationalStabilityFunction must be positive, but " *
                            "Ri⁻ = $Ri⁻ and Ri⁺ = $Ri⁺"))

    p⁻ > 0 && p⁺ > 0 ||
        throw(ArgumentError("The exponents of RationalStabilityFunction must be positive, but " *
                            "p⁻ = $p⁻ and p⁺ = $p⁺"))

    return RationalStabilityFunction(promote(endpoints..., Ri⁻, Ri⁺, p⁻, p⁺)...)
end

@inline convert_eltype(::Type{FT}, sf::RationalStabilityFunction) where FT =
    RationalStabilityFunction{FT}((convert(FT, getfield(sf, name)) for name in fieldnames(RationalStabilityFunction))...)

Base.summary(sf::RationalStabilityFunction{FT}) where FT = "RationalStabilityFunction{$FT}"

function show_stability_function_lines(io::IO, sf::RationalStabilityFunction, prefix)
    print(io, prefix, "├── Ri → -∞ (Cᵘ⁻, Cᶜ⁻, Cᵉ⁻, Cᴰ⁻): ", stability_endpoint_summary(sf.Cᵘ⁻, sf.Cᶜ⁻, sf.Cᵉ⁻, sf.Cᴰ⁻), '\n',
              prefix, "├── Ri = 0  (Cᵘ⁰, Cᶜ⁰, Cᵉ⁰, Cᴰ⁰): ", stability_endpoint_summary(sf.Cᵘ⁰, sf.Cᶜ⁰, sf.Cᵉ⁰, sf.Cᴰ⁰), '\n',
              prefix, "├── Ri → ∞  (Cᵘ⁺, Cᶜ⁺, Cᵉ⁺, Cᴰ⁺): ", stability_endpoint_summary(sf.Cᵘ⁺, sf.Cᶜ⁺, sf.Cᵉ⁺, sf.Cᴰ⁺), '\n',
              prefix, "├── transition scales: Ri⁻ = ", prettysummary(sf.Ri⁻), ", Ri⁺ = ", prettysummary(sf.Ri⁺), '\n',
              prefix, "└── exponents: p⁻ = ", prettysummary(sf.p⁻), ", p⁺ = ", prettysummary(sf.p⁺))
    return nothing
end

function Base.show(io::IO, sf::RationalStabilityFunction)
    print(io, summary(sf), '\n')
    show_stability_function_lines(io, sf, "")
end

"""
$(TYPEDSIGNATURES)

The rational stability function of [`RationalStabilityFunction`](@ref): `C⁰` at `Ri = 0`, travelling
to `C⁻` as `Ri → -∞` over the scale `Ri⁻` with exponent `p⁻`, and to `C⁺` as `Ri → ∞` over `Ri⁺`
with `p⁺`.

The branch is chosen first, from the sign of `Ri`, and only the chosen branch's coefficients enter
the arithmetic, so a scale or an exponent belonging to the inactive branch cannot pollute the result.
The shape factor is then evaluated as

```math
\\frac{x^p}{1 + x^p} = \\frac{1}{1 + (Ri^\\pm / |Ri|)^p}
```

rather than in the first form. The two agree at every `Ri`, but the reciprocal form has no overflow
to trip over: `Ri = 0` sends the inner ratio to `Inf` and the factor to exactly `0`, `Ri = ±Inf` sends
it to `0` and the factor to exactly `1`, and a huge finite `|Ri|` underflows the ratio harmlessly
towards zero, where the first form would form `Inf / (1 + Inf)` and return `NaN`. This holds in
`Float32` as in `Float64`, and needs neither a clamp nor a guard on `Ri`.
"""
@inline function rational_stability(Ri, C⁻, C⁰, C⁺, Ri⁻, Ri⁺, p⁻, p⁺)
    stable = Ri ≥ 0
    Cᵇ = ifelse(stable, C⁺, C⁻)
    Riᵇ = ifelse(stable, Ri⁺, Ri⁻)
    p = ifelse(stable, p⁺, p⁻)
    shape = 1 / (1 + (Riᵇ / abs(Ri))^p)
    return C⁰ + (Cᵇ - C⁰) * shape
end

@inline momentum_stability_function(sf::RationalStabilityFunction, Ri) =
    rational_stability(Ri, sf.Cᵘ⁻, sf.Cᵘ⁰, sf.Cᵘ⁺, sf.Ri⁻, sf.Ri⁺, sf.p⁻, sf.p⁺)

@inline tracer_stability_function(sf::RationalStabilityFunction, Ri) =
    rational_stability(Ri, sf.Cᶜ⁻, sf.Cᶜ⁰, sf.Cᶜ⁺, sf.Ri⁻, sf.Ri⁺, sf.p⁻, sf.p⁺)

@inline tke_stability_function(sf::RationalStabilityFunction, Ri) =
    rational_stability(Ri, sf.Cᵉ⁻, sf.Cᵉ⁰, sf.Cᵉ⁺, sf.Ri⁻, sf.Ri⁺, sf.p⁻, sf.p⁺)

@inline dissipation_stability_function(sf::RationalStabilityFunction, Ri) =
    rational_stability(Ri, sf.Cᴰ⁻, sf.Cᴰ⁰, sf.Cᴰ⁺, sf.Ri⁻, sf.Ri⁺, sf.p⁻, sf.p⁺)

#####
##### Evaluation on the grid, shared by both families
#####

"""Either family of stability functions that depends on the gradient Richardson number."""
const RichardsonStabilityFunction = Union{PiecewiseStabilityFunction, RationalStabilityFunction}

"""A [`TKEBasedTurbulenceClosure`](@ref) whose stability functions depend on ``Ri``."""
const RichardsonStabilityClosure = TKEBasedTurbulenceClosure{<:Any, <:Any, <:RichardsonStabilityFunction}

"""Deprecated name of [`RichardsonStabilityClosure`](@ref)."""
const RiDependentStabilityClosure = RichardsonStabilityClosure

@inline function momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure::RichardsonStabilityClosure, velocities, N²)
    Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, N²)
    return momentum_stability_function(closure.stability_functions, Ri)
end

@inline function tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure::RichardsonStabilityClosure, velocities, N²)
    Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, N²)
    return tracer_stability_function(closure.stability_functions, Ri)
end

@inline function tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure::RichardsonStabilityClosure, velocities, N²)
    Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, N²)
    return tke_stability_function(closure.stability_functions, Ri)
end

@inline function dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure::RichardsonStabilityClosure, velocities, N²)
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, N²)
    return dissipation_stability_function(closure.stability_functions, Ri)
end
