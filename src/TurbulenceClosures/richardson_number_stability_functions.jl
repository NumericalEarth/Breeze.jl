#####
##### Richardson-number-dependent rational stability functions
#####

# Ri = N² / (S² + S²ᵐⁱⁿ), with the closure's `minimum_shear_squared` S²ᵐⁱⁿ > 0 added to the
# shear. The floor makes Ri a smooth function of N² and S² everywhere: zero at N² = 0 for any shear
# (N² = S² = 0 is neutral rather than NaN), finite where the shear vanishes, and passing through
# zero continuously as N² changes sign. Without it, Ri where S² = 0 jumps between the unstable
# limit, neutral and the stable limit with the sign of a rounding-level N². Where S² ≫ S²ᵐⁱⁿ the
# relative change from N² / S² is at most S²ᵐⁱⁿ / S².
@inline richardson_number(N², S², S²ᵐⁱⁿ) = N² / (S² + S²ᵐⁱⁿ)

@inline function Riᶜᶜᶠ(i, j, k, grid, closure, velocities, N²)
    S² = shearᶜᶜᶠ(i, j, k, grid, velocities.u, velocities.v)
    N²ᵢ = @inbounds N²[i, j, k]
    return richardson_number(N²ᵢ, S², closure.minimum_shear_squared)
end

# Reconstruct N² and S² separately rather than averaging the face Richardson numbers, so that the
# center sees the stratification and shear of both faces even when Ri changes sign between them.
@inline function Riᶜᶜᶜ(i, j, k, grid, closure, velocities, N²)
    S² = ℑbzᵃᵃᶜ(i, j, k, grid, shearᶜᶜᶠ, velocities.u, velocities.v)
    N²ᵢ = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, N²)
    return richardson_number(N²ᵢ, S², closure.minimum_shear_squared)
end

struct RationalStabilityFunction{FT}
    ℂᵘ⁻ :: FT # momentum, unstable limit (Ri → -∞)
    ℂᵘ⁰ :: FT # momentum, neutral (Ri = 0)
    ℂᵘ⁺ :: FT # momentum, stable limit (Ri → ∞)
    ℂᶜ⁻ :: FT # tracers, unstable limit
    ℂᶜ⁰ :: FT # tracers, neutral
    ℂᶜ⁺ :: FT # tracers, stable limit
    ℂᵉ⁻ :: FT # turbulent kinetic energy, unstable limit
    ℂᵉ⁰ :: FT # turbulent kinetic energy, neutral
    ℂᵉ⁺ :: FT # turbulent kinetic energy, stable limit
    ℂᴰ⁻ :: FT # dissipation, unstable limit
    ℂᴰ⁰ :: FT # dissipation, neutral
    ℂᴰ⁺ :: FT # dissipation, stable limit
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
function RationalStabilityFunction(; ℂᵘ⁻ = 0.36113018152281917, ℂᵘ⁰ = 0.15900569445804447, ℂᵘ⁺ = 0.010956324667755535,
                                     ℂᶜ⁻ = 0.08170851244564359, ℂᶜ⁰ = 0.19872163735761372, ℂᶜ⁺ = 0.532550451554028,
                                     ℂᵉ⁻ = 1.9176239082347815, ℂᵉ⁰ = 0.00731292096489522, ℂᵉ⁺ = 0.02299192764300173,
                                     ℂᴰ⁻ = 0.005407068485856925, ℂᴰ⁰ = 0.029182354585815418, ℂᴰ⁺ = 0.8022301180336915,
                                     Ri⁻ = 12.669701017862883, Ri⁺ = 0.0501147186437052,
                                     p⁻ = 1, p⁺ = 1)

    endpoints = (ℂᵘ⁻, ℂᵘ⁰, ℂᵘ⁺, ℂᶜ⁻, ℂᶜ⁰, ℂᶜ⁺, ℂᵉ⁻, ℂᵉ⁰, ℂᵉ⁺, ℂᴰ⁻, ℂᴰ⁰, ℂᴰ⁺)

    all(C -> C > 0, endpoints) ||
        throw(ArgumentError("The twelve endpoints of RationalStabilityFunction must be positive, but " *
                            "(ℂᵘ⁻, ℂᵘ⁰, ℂᵘ⁺, ℂᶜ⁻, ℂᶜ⁰, ℂᶜ⁺, ℂᵉ⁻, ℂᵉ⁰, ℂᵉ⁺, ℂᴰ⁻, ℂᴰ⁰, ℂᴰ⁺) = $endpoints"))

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

stability_endpoint_summary(ℂᵘ, ℂᶜ, ℂᵉ, ℂᴰ) = join(prettysummary.((ℂᵘ, ℂᶜ, ℂᵉ, ℂᴰ)), ", ")

function show_stability_function_lines(io::IO, sf::RationalStabilityFunction, prefix)
    print(io, prefix, "├── Ri → -∞ (ℂᵘ⁻, ℂᶜ⁻, ℂᵉ⁻, ℂᴰ⁻): ", stability_endpoint_summary(sf.ℂᵘ⁻, sf.ℂᶜ⁻, sf.ℂᵉ⁻, sf.ℂᴰ⁻), '\n',
              prefix, "├── Ri = 0  (ℂᵘ⁰, ℂᶜ⁰, ℂᵉ⁰, ℂᴰ⁰): ", stability_endpoint_summary(sf.ℂᵘ⁰, sf.ℂᶜ⁰, sf.ℂᵉ⁰, sf.ℂᴰ⁰), '\n',
              prefix, "├── Ri → ∞  (ℂᵘ⁺, ℂᶜ⁺, ℂᵉ⁺, ℂᴰ⁺): ", stability_endpoint_summary(sf.ℂᵘ⁺, sf.ℂᶜ⁺, sf.ℂᵉ⁺, sf.ℂᴰ⁺), '\n',
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

@inline momentum_stability_function(sf::RationalStabilityFunction, Ri) = rational_stability(Ri, sf.ℂᵘ⁻, sf.ℂᵘ⁰, sf.ℂᵘ⁺, sf.Ri⁻, sf.Ri⁺, sf.p⁻, sf.p⁺)

@inline tracer_stability_function(sf::RationalStabilityFunction, Ri) = rational_stability(Ri, sf.ℂᶜ⁻, sf.ℂᶜ⁰, sf.ℂᶜ⁺, sf.Ri⁻, sf.Ri⁺, sf.p⁻, sf.p⁺)

@inline tke_stability_function(sf::RationalStabilityFunction, Ri) = rational_stability(Ri, sf.ℂᵉ⁻, sf.ℂᵉ⁰, sf.ℂᵉ⁺, sf.Ri⁻, sf.Ri⁺, sf.p⁻, sf.p⁺)

@inline dissipation_stability_function(sf::RationalStabilityFunction, Ri) = rational_stability(Ri, sf.ℂᴰ⁻, sf.ℂᴰ⁰, sf.ℂᴰ⁺, sf.Ri⁻, sf.Ri⁺, sf.p⁻, sf.p⁺)

#####
##### Evaluation on the grid
#####

const RichardsonStabilityClosure = TKEBasedTurbulenceClosure{<:Any, <:Any, <:RationalStabilityFunction}

@inline function momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure::RichardsonStabilityClosure, velocities, N²)
    Ri = Riᶜᶜᶠ(i, j, k, grid, closure, velocities, N²)
    return momentum_stability_function(closure.stability_functions, Ri)
end

@inline function tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure::RichardsonStabilityClosure, velocities, N²)
    Ri = Riᶜᶜᶠ(i, j, k, grid, closure, velocities, N²)
    return tracer_stability_function(closure.stability_functions, Ri)
end

@inline function tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure::RichardsonStabilityClosure, velocities, N²)
    Ri = Riᶜᶜᶠ(i, j, k, grid, closure, velocities, N²)
    return tke_stability_function(closure.stability_functions, Ri)
end

@inline function dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure::RichardsonStabilityClosure, velocities, N²)
    Ri = Riᶜᶜᶜ(i, j, k, grid, closure, velocities, N²)
    return dissipation_stability_function(closure.stability_functions, Ri)
end
