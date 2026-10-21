#####
##### Richardson-number-dependent rational stability functions
#####

# Ri = N² / S², zero where N² = 0 (so that N² = S² = 0 is neutral rather than NaN) and ±∞ where
# only the shear vanishes, which the stability functions map to their stable or unstable value.
@inline richardson_number(N², S²) = ifelse(N² == 0, zero(N²), N² / S²)

# A 1-2-1 average over the face and its two neighbors, for faces whose neighbors are both interior:
# the bottom and top faces carry no shear or stratification of their own, so they and the faces
# next to them keep their own values.
# The stability functions are steep in Ri, so a grid-scale (2Δz) mode in N² and S² — which a
# saturated layer under the dry static stability sustains — would otherwise be amplified into
# alternating diffusivities. The filter removes that mode before it reaches the stability functions.
@inline function smoothed_faceᶜᶜᶠ(i, j, k, grid, f, args...)
    interior = (k > 2) & (k < grid.Nz)
    f⁻ = f(i, j, ifelse(interior, k - 1, k), grid, args...)
    f⁰ = f(i, j, k, grid, args...)
    f⁺ = f(i, j, ifelse(interior, k + 1, k), grid, args...)
    return ifelse(interior, (f⁻ + 2f⁰ + f⁺) / 4, f⁰)
end

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, N²)
    S² = smoothed_faceᶜᶜᶠ(i, j, k, grid, shearᶜᶜᶠ, velocities.u, velocities.v)
    N²ᵢ = smoothed_faceᶜᶜᶠ(i, j, k, grid, face_valueᶜᶜᶠ, N²)
    return richardson_number(N²ᵢ, S²)
end

# Reconstruct N² and S² separately so opposite infinite face Richardson numbers cannot yield NaN.
@inline function Riᶜᶜᶜ(i, j, k, grid, velocities, N²)
    S² = ℑbzᵃᵃᶜ(i, j, k, grid, shearᶜᶜᶠ, velocities.u, velocities.v)
    N²ᵢ = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, N²)
    return richardson_number(N²ᵢ, S²)
end

struct RationalStabilityFunction{FT}
    Cᵘ⁻ :: FT # momentum, unstable limit (Ri → -∞)
    Cᵘ⁰ :: FT # momentum, neutral (Ri = 0)
    Cᵘ⁺ :: FT # momentum, stable limit (Ri → ∞)
    Cᶜ⁻ :: FT # tracers, unstable limit
    Cᶜ⁰ :: FT # tracers, neutral
    Cᶜ⁺ :: FT # tracers, stable limit
    Cᵉ⁻ :: FT # turbulent kinetic energy, unstable limit
    Cᵉ⁰ :: FT # turbulent kinetic energy, neutral
    Cᵉ⁺ :: FT # turbulent kinetic energy, stable limit
    Cᴰ⁻ :: FT # dissipation, unstable limit
    Cᴰ⁰ :: FT # dissipation, neutral
    Cᴰ⁺ :: FT # dissipation, stable limit
    Ri⁻ :: FT # unstable transition scale: the |Ri| at which every function is halfway from C⁰ to C⁻
    Ri⁺ :: FT # stable transition scale: the Ri at which every function is halfway from C⁰ to C⁺
    p⁻ :: FT # unstable exponent, fixed at one for calibration
    p⁺ :: FT # stable exponent, fixed at one for calibration
end

"""
$(TYPEDSIGNATURES)

Construct rational stability functions for momentum, tracers, TKE, and dissipation:
`S(Ri) = C⁰ + (C± - C⁰) xᵖ / (1 + xᵖ)`, where `x = |Ri| / Ri±`.
Each function has unstable, neutral, and stable endpoints; the four functions share
transition scales `Ri⁻`, `Ri⁺` and exponents `p⁻`, `p⁺`. All coefficients must be positive.
The default exponents are one, giving continuous functions with a kink at neutral.
Mixed integer and float arguments are promoted to a single float type.

Defaults are the provisional atmospheric LES fit described in the turbulence-closure
manual; use `TKEBasedTurbulenceClosure()` for its paired mixing-length coefficients.

```jldoctest
using Breeze

RationalStabilityFunction().Ri⁺

# output
0.0501147186437052
```
"""
function RationalStabilityFunction(; Cᵘ⁻ = 0.36113018152281917, Cᵘ⁰ = 0.15900569445804447, Cᵘ⁺ = 0.010956324667755535,
                                     Cᶜ⁻ = 0.08170851244564359, Cᶜ⁰ = 0.19872163735761372, Cᶜ⁺ = 0.532550451554028,
                                     Cᵉ⁻ = 1.9176239082347815, Cᵉ⁰ = 0.00731292096489522, Cᵉ⁺ = 0.02299192764300173,
                                     Cᴰ⁻ = 0.005407068485856925, Cᴰ⁰ = 0.029182354585815418, Cᴰ⁺ = 0.8022301180336915,
                                     Ri⁻ = 12.669701017862883, Ri⁺ = 0.0501147186437052,
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

stability_endpoint_summary(Cᵘ, Cᶜ, Cᵉ, Cᴰ) = join(prettysummary.((Cᵘ, Cᶜ, Cᵉ, Cᴰ)), ", ")

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

# The reciprocal shape avoids Inf/Inf at infinite Ri and gives zero at neutral.
@inline function rational_stability(Ri, C⁻, C⁰, C⁺, Ri⁻, Ri⁺, p⁻, p⁺)
    stable = Ri ≥ 0
    Cᵇ = ifelse(stable, C⁺, C⁻)
    Riᵇ = ifelse(stable, Ri⁺, Ri⁻)
    p = ifelse(stable, p⁺, p⁻)
    shape = 1 / (1 + (Riᵇ / abs(Ri))^p)
    return C⁰ + (Cᵇ - C⁰) * shape
end

@inline momentum_stability_function(sf::RationalStabilityFunction, Ri) = rational_stability(Ri, sf.Cᵘ⁻, sf.Cᵘ⁰, sf.Cᵘ⁺, sf.Ri⁻, sf.Ri⁺, sf.p⁻, sf.p⁺)

@inline tracer_stability_function(sf::RationalStabilityFunction, Ri) = rational_stability(Ri, sf.Cᶜ⁻, sf.Cᶜ⁰, sf.Cᶜ⁺, sf.Ri⁻, sf.Ri⁺, sf.p⁻, sf.p⁺)

@inline tke_stability_function(sf::RationalStabilityFunction, Ri) = rational_stability(Ri, sf.Cᵉ⁻, sf.Cᵉ⁰, sf.Cᵉ⁺, sf.Ri⁻, sf.Ri⁺, sf.p⁻, sf.p⁺)

@inline dissipation_stability_function(sf::RationalStabilityFunction, Ri) = rational_stability(Ri, sf.Cᴰ⁻, sf.Cᴰ⁰, sf.Cᴰ⁺, sf.Ri⁻, sf.Ri⁺, sf.p⁻, sf.p⁺)

#####
##### Evaluation on the grid
#####

const RichardsonStabilityClosure = TKEBasedTurbulenceClosure{<:Any, <:Any, <:RationalStabilityFunction}

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
