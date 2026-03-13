#####
##### Rain PSD Quadrature Evaluators
#####
##### Numerically integrate rain size-distribution (DSD) integrals over the
##### exponential DSD N'(D) = N_0 * exp(-λ_r * D) using Chebyshev-Gauss quadrature
##### on the transformed domain D ∈ [0, ∞).
#####
##### Three integrals are tabulated as functions of log10(λ_r):
#####
#####  1. Mass-weighted terminal velocity:
#####       V_mass = ∫ V(D) m(D) exp(-λ_r D) dD / ∫ m(D) exp(-λ_r D) dD  [m/s]
#####
#####  2. Number-weighted terminal velocity:
#####       V_num = ∫ V(D) exp(-λ_r D) dD / ∫ exp(-λ_r D) dD              [m/s]
#####
#####  3. Evaporation ventilation integral:
#####       I_evap = ∫ D f_v(D) exp(-λ_r D) dD                             [m²]
#####       where f_v(D) = f1r + f2r * sqrt(Re),  Re = ar * D^(1+br) / ν  [= V(D)*D/ν]
#####
##### The integration uses the same domain transformation as ice quadrature:
#####   D = (scale/λ) * (1+x) / (1-x+ε),  x ∈ [-1, 1]
##### with a scale of 10 (10 exponential decay lengths covers >99.99% of the integral).
#####
##### References:
##### - P3 Fortran v5.5.0: ar=842, br=0.8, f1r=0.78, f2r=0.308 (Sc^(1/3) baked in), ν=1.5e-5 m²/s
#####

export RainMassWeightedVelocityEvaluator,
       RainNumberWeightedVelocityEvaluator,
       RainEvaporationVentilationEvaluator

# Rain ventilation constants (Fortran P3 v5.5.0)
const RAIN_F1R = 0.78       # constant term in ventilation factor [-]
const RAIN_F2R = 0.308      # Reynolds-dependent coefficient [-] (Fortran P3: Sc^(1/3) baked in)
const RAIN_NU  = 1.5e-5     # kinematic viscosity [m²/s]

#####
##### RainMassWeightedVelocityEvaluator
#####

"""
    RainMassWeightedVelocityEvaluator{N, W}

Callable evaluator for the mass-weighted rain terminal velocity:

```math
V_{\\mathrm{mass}}(\\lambda_r) =
    \\frac{\\int_0^\\infty V(D)\\, m(D)\\, e^{-\\lambda_r D}\\, dD}
         {\\int_0^\\infty m(D)\\, e^{-\\lambda_r D}\\, dD}
```

where `m(D) = (π/6) ρ_w D³` (liquid sphere, ρ_w = 997 kg/m³) and `V(D)` is the
piecewise Gunn-Kinzer/Beard rain fall speed from [`rain_fall_speed`](@ref) at
reference density (no density correction applied here; apply at call site).

Quadrature uses the same exponential-tail transformation as the ice integrals,
via [`chebyshev_gauss_nodes_weights`](@ref).

# Fields
$(TYPEDFIELDS)
"""
struct RainMassWeightedVelocityEvaluator{N, W}
    "Pre-computed Chebyshev-Gauss nodes on [-1, 1]"
    nodes :: N
    "Pre-computed Chebyshev-Gauss weights"
    weights :: W
end

"""
$(TYPEDSIGNATURES)

Construct a `RainMassWeightedVelocityEvaluator` with `n_points` quadrature points.
"""
function RainMassWeightedVelocityEvaluator(FT::Type{<:AbstractFloat} = Float64;
                                            n_points::Int = 128)
    nodes, weights = chebyshev_gauss_nodes_weights(FT, n_points)
    return RainMassWeightedVelocityEvaluator(nodes, weights)
end

"""
    (e::RainMassWeightedVelocityEvaluator)(log10_lambda_r)

Evaluate the mass-weighted rain terminal velocity at the given `log10(λ_r)`.

Returns the velocity in [m/s] at reference air density (no density correction).
Apply `(ρ₀/ρ)^0.54` at the call site if needed.
"""
@inline function (e::RainMassWeightedVelocityEvaluator)(log10_lambda_r)
    FT = eltype(e.nodes)
    λ_r = FT(10)^log10_lambda_r

    # Density correction is 1 at reference conditions (applied at call site)
    ρ_correction = one(FT)

    mass_vel_integral = zero(FT)
    mass_integral     = zero(FT)
    n = length(e.nodes)

    for i in 1:n
        x = @inbounds e.nodes[i]
        w = @inbounds e.weights[i]
        D = transform_to_diameter(x, λ_r)
        J = jacobian_diameter_transform(x, λ_r)

        m = (FT(π) / 6) * FT(997) * D^3
        V = rain_fall_speed(D, ρ_correction)
        psd = exp(-λ_r * D)

        mass_vel_integral += w * V * m * psd * J
        mass_integral     += w * m * psd * J
    end

    denom  = max(mass_integral, eps(FT))
    result = mass_vel_integral / denom
    return ifelse(isfinite(result), result, zero(FT))
end

#####
##### RainNumberWeightedVelocityEvaluator
#####

"""
    RainNumberWeightedVelocityEvaluator{N, W}

Callable evaluator for the number-weighted rain terminal velocity:

```math
V_{\\mathrm{num}}(\\lambda_r) =
    \\frac{\\int_0^\\infty V(D)\\, e^{-\\lambda_r D}\\, dD}
         {\\int_0^\\infty e^{-\\lambda_r D}\\, dD}
```

Quadrature uses the same exponential-tail transformation as ice integrals.

# Fields
$(TYPEDFIELDS)
"""
struct RainNumberWeightedVelocityEvaluator{N, W}
    "Pre-computed Chebyshev-Gauss nodes on [-1, 1]"
    nodes :: N
    "Pre-computed Chebyshev-Gauss weights"
    weights :: W
end

"""
$(TYPEDSIGNATURES)

Construct a `RainNumberWeightedVelocityEvaluator` with `n_points` quadrature points.
"""
function RainNumberWeightedVelocityEvaluator(FT::Type{<:AbstractFloat} = Float64;
                                              n_points::Int = 128)
    nodes, weights = chebyshev_gauss_nodes_weights(FT, n_points)
    return RainNumberWeightedVelocityEvaluator(nodes, weights)
end

"""
    (e::RainNumberWeightedVelocityEvaluator)(log10_lambda_r)

Evaluate the number-weighted rain terminal velocity at the given `log10(λ_r)`.

Returns the velocity in [m/s] at reference air density.
"""
@inline function (e::RainNumberWeightedVelocityEvaluator)(log10_lambda_r)
    FT = eltype(e.nodes)
    λ_r = FT(10)^log10_lambda_r
    ρ_correction = one(FT)

    vel_integral    = zero(FT)
    number_integral = zero(FT)
    n = length(e.nodes)

    for i in 1:n
        x = @inbounds e.nodes[i]
        w = @inbounds e.weights[i]
        D = transform_to_diameter(x, λ_r)
        J = jacobian_diameter_transform(x, λ_r)

        V   = rain_fall_speed(D, ρ_correction)
        psd = exp(-λ_r * D)

        vel_integral    += w * V * psd * J
        number_integral += w * psd * J
    end

    denom  = max(number_integral, eps(FT))
    result = vel_integral / denom
    return ifelse(isfinite(result), result, zero(FT))
end

#####
##### RainEvaporationVentilationEvaluator
#####

"""
    RainEvaporationVentilationEvaluator{N, W}

Callable evaluator for the rain evaporation ventilation integral:

```math
I_{\\mathrm{evap}}(\\lambda_r) =
    \\int_0^\\infty D\\, f_v(D)\\, e^{-\\lambda_r D}\\, dD
```

where the ventilation factor is `f_v(D) = f1r + f2r × √Re(D)` with
`Re(D) = V(D) × D / ν = ar × D^(1+br) / ν` (Reynolds number based on drop diameter),
`f1r = 0.78`, `f2r = 0.308` (Sc^(1/3) baked in), `ar = 842`, `br = 0.8`, `ν = 1.5e-5 m²/s`
(Fortran P3 v5.5.0 constants).

This integral appears in the PSD-integrated rain evaporation rate (Mason 1971,
capacitance `C = D/2` for a sphere, so `4πC = 2πD`):

```math
\\frac{dq^r}{dt}  \\approx  \\frac{2 \\pi N_0}{A + B}\\,(S - 1)\\, I_{\\mathrm{evap}}
```

where A+B is the thermodynamic resistance factor.

**Analytical limit**: At λ_r → ∞ (tiny drops), `f_v ≈ f1r = 0.78`, giving
`I_evap ≈ 0.78 * 1/λ_r²` (from `∫₀^∞ D exp(-λD) dD = 1/λ²`).

# Fields
$(TYPEDFIELDS)
"""
struct RainEvaporationVentilationEvaluator{N, W}
    "Pre-computed Chebyshev-Gauss nodes on [-1, 1]"
    nodes :: N
    "Pre-computed Chebyshev-Gauss weights"
    weights :: W
end

"""
$(TYPEDSIGNATURES)

Construct a `RainEvaporationVentilationEvaluator` with `n_points` quadrature points.
"""
function RainEvaporationVentilationEvaluator(FT::Type{<:AbstractFloat} = Float64;
                                              n_points::Int = 128)
    nodes, weights = chebyshev_gauss_nodes_weights(FT, n_points)
    return RainEvaporationVentilationEvaluator(nodes, weights)
end

"""
    (e::RainEvaporationVentilationEvaluator)(log10_lambda_r)

Evaluate `I_evap(λ_r)` = ∫ D f_v(D) exp(-λ_r D) dD at the given `log10(λ_r)`.

Returns the integral value in [m²].
"""
@inline function (e::RainEvaporationVentilationEvaluator)(log10_lambda_r)
    FT = eltype(e.nodes)
    λ_r  = FT(10)^log10_lambda_r
    ar   = FT(842)
    br   = FT(0.8)
    f1r  = FT(RAIN_F1R)
    f2r  = FT(RAIN_F2R)
    ν    = FT(RAIN_NU)

    result = zero(FT)
    n = length(e.nodes)

    for i in 1:n
        x = @inbounds e.nodes[i]
        w = @inbounds e.weights[i]
        D = transform_to_diameter(x, λ_r)
        J = jacobian_diameter_transform(x, λ_r)

        # Reynolds number: Re = V(D) × D / ν = ar × D^br × D / ν = ar × D^(1+br) / ν
        Re_sqrt = sqrt(max(ar * D^(br + 1) / ν, zero(FT)))
        f_v = f1r + f2r * Re_sqrt
        psd = exp(-λ_r * D)

        result += w * D * f_v * psd * J
    end

    return ifelse(isfinite(result), result, zero(FT))
end
