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
#####  3. Evaporation Reynolds integral (M3):
#####       I_Re = ∫ D √Re(D) exp(-λ_r D) dD                              [m²]
#####       where Re(D) = ar * D^(1+br) / ν  [= V(D)*D/ν]
#####       Full evaporation integral assembled at runtime:
#####       I_evap = f1r/λ² + f2r × Sc^(1/3) × I_Re
#####
##### The integration uses the same domain transformation as ice quadrature:
#####   D = (scale/λ) * (1+x) / (1-x+ε),  x ∈ [-1, 1]
##### with a scale of 10 (10 exponential decay lengths covers >99.99% of the integral).
#####
##### References:
##### - P3 Fortran v5.5.0: ar=842, br=0.8, f1r=0.78, f2r=0.32, ν=1.5e-5 m²/s
#####

##### NOTE (M13): Both the tabulated and analytical rain paths now use the same
##### 4-regime Gunn-Kinzer/Beard piecewise V(D) formula (rain_fall_speed in quadrature.jl).
##### The piecewise law captures the terminal velocity plateau above D ~5mm and Stokes
##### drag below D ~100μm. The previous analytical path used a single power law
##### V = ar × D^br (842 × D^0.8), which has been replaced for consistency.

export RainMassWeightedVelocityEvaluator,
       RainNumberWeightedVelocityEvaluator,
       RainEvaporationVentilationEvaluator

# Rain ventilation constants (Fortran P3 v5.5.0)
const RAIN_F1R = 0.78       # constant term in ventilation factor [-]
const RAIN_F2R = 0.32       # Reynolds-dependent ventilation coefficient [-] (Fortran P3 v5.5.0)
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

where `m(D) = (π/6) ρ_w D³` (liquid sphere, ρ_w = 1000 kg/m³) and `V(D)` is the
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

        m = (FT(π) / 6) * FT(1000) * D^3
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

Callable evaluator for the Reynolds part of the rain evaporation ventilation integral:

```math
I_{\\mathrm{Re}}(\\lambda_r) =
    \\int_0^\\infty D\\, \\sqrt{\\mathrm{Re}(D)}\\, e^{-\\lambda_r D}\\, dD
```

where `Re(D) = V(D) × D / ν` (Reynolds number based on drop diameter),
`ν = 1.5e-5 m²/s`, and `V(D)` is given by the same piecewise Gunn-Kinzer/Beard
law used in the Fortran rain lookup-table generation.

The full evaporation ventilation integral is assembled at runtime (M3):

```math
I_{\\mathrm{evap}} = \\frac{f_{1r}}{\\lambda_r^2}
    + f_{2r}\\, \\mathrm{Sc}^{1/3}\\, I_{\\mathrm{Re}}
```

where `f1r = 0.78`, `f2r = 0.32`, and `Sc = ν / D_v` is the Schmidt number
computed from T,P-dependent transport properties. The constant term
`f1r / λ_r²` is the analytical result of `f1r × ∫ D exp(-λD) dD`.

This integral appears in the PSD-integrated rain evaporation rate (Mason 1971,
capacitance `C = D/2` for a sphere, so `4πC = 2πD`):

```math
\\frac{dq^r}{dt}  \\approx  \\frac{2 \\pi N_0}{A + B}\\,(S - 1)\\, I_{\\mathrm{evap}}
```

where A+B is the thermodynamic resistance factor.

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

Evaluate `I_Re(λ_r)` = ∫ D √Re(D) exp(-λ_r D) dD at the given `log10(λ_r)`.

Returns the Reynolds integral in [m²]. The constant (f1r) and Schmidt number
(Sc^(1/3)) contributions are applied at runtime (M3).
"""
@inline function (e::RainEvaporationVentilationEvaluator)(log10_lambda_r)
    FT = eltype(e.nodes)
    λ_r  = FT(10)^log10_lambda_r
    ar   = FT(842)
    br   = FT(0.8)
    ν    = FT(RAIN_NU)

    result = zero(FT)
    n = length(e.nodes)

    for i in 1:n
        x = @inbounds e.nodes[i]
        w = @inbounds e.weights[i]
        D = transform_to_diameter(x, λ_r)
        J = jacobian_diameter_transform(x, λ_r)

        # √(Re) = √(V(D) × D / ν)
        Re_sqrt = sqrt(max(ar * D^(br + 1) / ν, zero(FT)))
        psd = exp(-λ_r * D)

        # M3: Reynolds integral only (f1r and Sc^(1/3) applied at runtime)
        result += w * D * Re_sqrt * psd * J
    end

    return ifelse(isfinite(result), result, zero(FT))
end
