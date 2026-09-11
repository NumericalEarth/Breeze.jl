#####
##### Richardson-number-dependent stability functions, in the form of CATKE (Wagner et al. 2025)
#####
#####   S(Ri) = C⁻                                             for Ri < 0
#####   S(Ri) = C⁰ + (C⁺ − C⁰) clamp((Ri − Ri⁰) / Riᵟ, 0, 1)   for Ri ≥ 0
#####
##### with Ri = N² / S² from the stored static stability and the vertical shear, at the faces for
##### the diffusivities and from N² and S² reconstructed to the centers for the dissipation.
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

Fields
======

$(TYPEDFIELDS)
"""
Base.@kwdef struct RiDependentStabilityFunctions{FT}
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

## Mixed integer/float keyword arguments are promoted, as for `ConstantStabilityFunctions`; the
## untyped signature is less specific than the default constructor, which takes over once the
## fourteen coefficients share a type.
RiDependentStabilityFunctions(coefficients::Vararg{Any, 14}) =
    RiDependentStabilityFunctions(promote(coefficients...)...)

@inline convert_eltype(::Type{FT}, sf::RiDependentStabilityFunctions) where FT =
    RiDependentStabilityFunctions{FT}((convert(FT, getfield(sf, name)) for name in fieldnames(RiDependentStabilityFunctions))...)

Base.summary(sf::RiDependentStabilityFunctions{FT}) where FT = "RiDependentStabilityFunctions{$FT}"

function show_stability_function_lines(io::IO, sf::RiDependentStabilityFunctions, prefix)
    endpoints(Cᵘ, Cᶜ, Cᵉ, Cᴰ) = join(prettysummary.((Cᵘ, Cᶜ, Cᵉ, Cᴰ)), ", ")
    print(io, prefix, "├── Ri < 0 (Cᵘ⁻, Cᶜ⁻, Cᵉ⁻, Cᴰ⁻): ", endpoints(sf.Cᵘ⁻, sf.Cᶜ⁻, sf.Cᵉ⁻, sf.Cᴰ⁻), '\n',
              prefix, "├── Ri = 0 (Cᵘ⁰, Cᶜ⁰, Cᵉ⁰, Cᴰ⁰): ", endpoints(sf.Cᵘ⁰, sf.Cᶜ⁰, sf.Cᵉ⁰, sf.Cᴰ⁰), '\n',
              prefix, "├── Ri → ∞ (Cᵘ⁺, Cᶜ⁺, Cᵉ⁺, Cᴰ⁺): ", endpoints(sf.Cᵘ⁺, sf.Cᶜ⁺, sf.Cᵉ⁺, sf.Cᴰ⁺), '\n',
              prefix, "└── stable transition: Ri⁰ = ", prettysummary(sf.Ri⁰), ", Riᵟ = ", prettysummary(sf.Riᵟ))
    return nothing
end

function Base.show(io::IO, sf::RiDependentStabilityFunctions)
    print(io, summary(sf), '\n')
    show_stability_function_lines(io, sf, "")
end

"""
$(TYPEDSIGNATURES)

The mixing length and stability functions of CATKE ([Wagner et al. 2025](@cite Wagner25catke)) as
keyword arguments for [`TKEBasedTurbulenceClosure`](@ref): CATKE's wall coefficient ``Cˢ = 1.131`` in
the mixing-length formulation `MixingLength` ([`GradientLimitedMixingLength`](@ref) by default), and
[`RiDependentStabilityFunctions`](@ref) with CATKE's values.

```jldoctest
using Breeze

closure = TKEBasedTurbulenceClosure(; catke_parameters()...)
closure.mixing_length

# output
GradientLimitedMixingLength{Float64} (Cˢ = 1.131)
```

CATKE's convective length scales and surface flux of turbulent kinetic energy are not part of the
closure; see [`RiDependentStabilityFunctions`](@ref).
"""
catke_parameters(MixingLength = GradientLimitedMixingLength) = (mixing_length = MixingLength(Cˢ = 1.131),
                                                                  stability_functions = RiDependentStabilityFunctions())

#####
##### Evaluation
#####

"""
$(TYPEDSIGNATURES)

The piecewise-linear stability function of [`RiDependentStabilityFunctions`](@ref): `C⁻` for
`Ri < 0`, and otherwise `C⁰` ramping linearly to `C⁺` between `Ri⁰` and `Ri⁰ + Riᵟ`.
"""
@inline function stability_ramp(Ri, C⁻, C⁰, C⁺, Ri⁰, Riᵟ)
    ramp = clamp((Ri - Ri⁰) / Riᵟ, 0, 1)
    C⁺ˢ = C⁰ + (C⁺ - C⁰) * ramp
    return ifelse(Ri < 0, C⁻, C⁺ˢ)
end

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

const RiDependentStabilityClosure = TKEBasedTurbulenceClosure{<:Any, <:Any, <:RiDependentStabilityFunctions}

@inline function momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure::RiDependentStabilityClosure, velocities, N²)
    sf = closure.stability_functions
    Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, N²)
    return stability_ramp(Ri, sf.Cᵘ⁻, sf.Cᵘ⁰, sf.Cᵘ⁺, sf.Ri⁰, sf.Riᵟ)
end

@inline function tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure::RiDependentStabilityClosure, velocities, N²)
    sf = closure.stability_functions
    Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, N²)
    return stability_ramp(Ri, sf.Cᶜ⁻, sf.Cᶜ⁰, sf.Cᶜ⁺, sf.Ri⁰, sf.Riᵟ)
end

@inline function tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure::RiDependentStabilityClosure, velocities, N²)
    sf = closure.stability_functions
    Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, N²)
    return stability_ramp(Ri, sf.Cᵉ⁻, sf.Cᵉ⁰, sf.Cᵉ⁺, sf.Ri⁰, sf.Riᵟ)
end

@inline function dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure::RiDependentStabilityClosure, velocities, N²)
    sf = closure.stability_functions
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, N²)
    return stability_ramp(Ri, sf.Cᴰ⁻, sf.Cᴰ⁰, sf.Cᴰ⁺, sf.Ri⁰, sf.Riᵟ)
end
