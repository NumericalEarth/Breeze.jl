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
#####  3. Evaporation velocity-diameter integral:
#####       I_VD = ∫ D √(V(D)×D) exp(-λ_r D) dD                           [m^(5/2)]
#####       where V(D) is the piecewise Gunn-Kinzer/Beard fall speed.
#####       ν is NOT baked in; 1/√ν is applied at runtime.
#####       Full evaporation integral assembled at runtime:
#####       I_evap = f1r/λ² + f2r × Sc^(1/3) / √ν × I_VD
#####
##### The integration uses the same domain transformation as ice quadrature:
#####   D = (scale/λ) * (1+x) / (1-x+ε),  x ∈ [-1, 1]
##### with a scale of 10 (10 exponential decay lengths covers >99.99% of the integral).
#####
##### Both the tabulated and analytical rain paths use the same 4-regime
##### Gunn-Kinzer/Beard piecewise V(D) formula (`rain_fall_speed` below).
##### The piecewise law captures the terminal-velocity plateau above D ~5 mm and
##### Stokes drag below D ~100 μm.
#####

export RainMassWeightedVelocityEvaluator,
       RainNumberWeightedVelocityEvaluator,
       RainEvaporationVentilationEvaluator

# Rain ventilation factor fᵛᵉ = f₁ + f₂ Sc^(1/3) Re^(1/2), the same form the ice side
# uses (see the `*_ventilation_constant` / `*_ventilation_reynolds` fields of
# `IceDeposition`). The pair below is P3's `f1r`/`f2r`; the ventilation factor itself is
# the classical one of [Pruppacher & Klett (2010)](@cite pruppacher2010microphysics).
# `RainEvaporationVentilationEvaluator` documents where each term enters.
const RAIN_VENTILATION_CONSTANT = 0.78  # f₁, the still-air term [-]
const RAIN_VENTILATION_REYNOLDS = 0.32  # f₂, multiplying Sc^(1/3) Re^(1/2) [-]

#####
##### Rain fall speed (Gunn-Kinzer / Beard piecewise power law)
#####

# The fit is stated in terms of a drop mass in grams computed at the water density the
# fit itself was derived with, so this density belongs to the formula rather than to the
# model: substituting the configurable ρʷ would rescale the published coefficients.
const GUNN_KINZER_WATER_DENSITY = 997     # [kg/m³], mass basis of the fit
const GUNN_KINZER_PLATEAU_SPEED = 917     # [cm/s], terminal speed above the largest branch edge

"""
$(TYPEDSIGNATURES)

Piecewise Gunn-Kinzer / Beard rain terminal velocity [m/s]. Captures the Stokes-drag
regime below D ≈ 100 μm and the terminal-velocity plateau above D ≈ 5 mm. Used by the
rain quadrature tabulation path.
"""
@inline function rain_fall_speed(D, ρ_correction)
    FT = typeof(D)

    m_kg = (FT(π)/6) * FT(GUNN_KINZER_WATER_DENSITY) * D^3
    m_g = m_kg * 1000

    # Piecewise power law: branch edges in m, coefficients in cm/s per g^exponent, and a
    # terminal plateau above the largest edge. A published fit, so the numbers stay with
    # the formula they belong to.
    V_cm = ifelse(D <= FT(134.43e-6),  FT(4.5795e5) * cbrt(m_g)^2,
           ifelse(D <  FT(1511.64e-6), FT(4.962e3)  * cbrt(m_g),
           ifelse(D <  FT(3477.84e-6), FT(1.732e3)  * sqrt(cbrt(m_g)),
                                       FT(GUNN_KINZER_PLATEAU_SPEED))))

    # cm/s → m/s
    return V_cm / 100 * ρ_correction
end

#####
##### RainMassWeightedVelocityEvaluator
#####

"""
    RainMassWeightedVelocityEvaluator{N, W, F}

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
struct RainMassWeightedVelocityEvaluator{N, W, F}
    "Pre-computed Chebyshev-Gauss nodes on [-1, 1]"
    nodes :: N
    "Pre-computed Chebyshev-Gauss weights"
    weights :: W
    "Numerical floors, carried because tabulation runs before a scheme exists"
    floors :: F
end

"""
$(TYPEDSIGNATURES)

Construct a `RainMassWeightedVelocityEvaluator` with `n_points` quadrature points.
"""
function RainMassWeightedVelocityEvaluator(FT::DataType = Oceananigans.defaults.FloatType;
                                            n_points::Int = 128,
                                            floors = NumericalFloors(FT))
    nodes, weights = chebyshev_gauss_nodes_weights(FT, n_points)
    return RainMassWeightedVelocityEvaluator(nodes, weights, floors)
end

"""
$(TYPEDSIGNATURES)

Velocity moment ratio of an exponential rain PSD on the Chebyshev-Gauss nodes,

```math
\\frac{\\int_0^\\infty V(D) \\, g(D) \\, e^{-λ^r D} \\, dD}
      {\\int_0^\\infty g(D) \\, e^{-λ^r D} \\, dD}
```

where `diameter_weight` supplies ``g(D)``: `identity_weight` for the
number-weighted velocity, `cubed_weight` for the mass-weighted one (the constant
spherical-water mass factor cancels between numerator and denominator).

`V` is the piecewise Gunn-Kinzer/Beard fall speed at reference density; apply
`(ρ₀/ρ)^0.54` at the call site. The floor on the denominator is a divide-by-zero
guard on that same integral; a machine-epsilon floor would instead suppress valid
`Float32` velocities.
"""
@inline function rain_velocity_moment_ratio(nodes, weights, λʳ, diameter_weight::G, floors) where G
    FT = eltype(nodes)

    # Density correction is 1 at reference conditions (applied at call site)
    ρ_correction = one(FT)

    weighted_velocity_integral = zero(FT)
    weighted_integral          = zero(FT)

    for i in eachindex(nodes)
        x = @inbounds nodes[i]
        w = @inbounds weights[i]
        D = transform_to_diameter(x, λʳ)
        J = jacobian_diameter_transform(x, λʳ)

        V = rain_fall_speed(D, ρ_correction)
        g = diameter_weight(D)
        psd = exp(-λʳ * D)

        weighted_velocity_integral += w * V * g * psd * J
        weighted_integral          += w * g * psd * J
    end

    denominator = max(weighted_integral, FT(floors.divisor))
    result = weighted_velocity_integral / denominator
    return ifelse(isfinite(result), result, zero(FT))
end

@inline identity_weight(D) = one(D)
@inline cubed_weight(D) = D^3

"""
    (e::RainMassWeightedVelocityEvaluator)(log10_slope)

Evaluate the mass-weighted rain terminal velocity at the given `log10(λ_r)`.

Returns the velocity in [m/s] at reference air density (no density correction).
Apply `(ρ₀/ρ)^0.54` at the call site if needed.
"""
@inline (e::RainMassWeightedVelocityEvaluator)(log10_slope) =
    rain_velocity_moment_ratio(e.nodes, e.weights,
                               exp10(eltype(e.nodes)(log10_slope)), cubed_weight,
                               e.floors)

#####
##### RainNumberWeightedVelocityEvaluator
#####

"""
    RainNumberWeightedVelocityEvaluator{N, W, F}

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
struct RainNumberWeightedVelocityEvaluator{N, W, F}
    "Pre-computed Chebyshev-Gauss nodes on [-1, 1]"
    nodes :: N
    "Pre-computed Chebyshev-Gauss weights"
    weights :: W
    "Numerical floors, carried because tabulation runs before a scheme exists"
    floors :: F
end

"""
$(TYPEDSIGNATURES)

Construct a `RainNumberWeightedVelocityEvaluator` with `n_points` quadrature points.
"""
function RainNumberWeightedVelocityEvaluator(FT::DataType = Oceananigans.defaults.FloatType;
                                              n_points::Int = 128,
                                              floors = NumericalFloors(FT))
    nodes, weights = chebyshev_gauss_nodes_weights(FT, n_points)
    return RainNumberWeightedVelocityEvaluator(nodes, weights, floors)
end

"""
    (e::RainNumberWeightedVelocityEvaluator)(log10_slope)

Evaluate the number-weighted rain terminal velocity at the given `log10(λ_r)`.

Returns the velocity in [m/s] at reference air density.
"""
@inline (e::RainNumberWeightedVelocityEvaluator)(log10_slope) =
    rain_velocity_moment_ratio(e.nodes, e.weights,
                               exp10(eltype(e.nodes)(log10_slope)), identity_weight,
                               e.floors)

#####
##### RainEvaporationVentilationEvaluator
#####

"""
    RainEvaporationVentilationEvaluator{N, W}

Callable evaluator for the velocity-diameter part of the rain evaporation
ventilation integral:

```math
I_{\\mathrm{VD}}(\\lambda_r) =
    \\int_0^\\infty D\\, \\sqrt{V(D) \\times D}\\, e^{-\\lambda_r D}\\, dD
```

where `V(D)` is the piecewise Gunn-Kinzer/Beard rain fall speed at reference
density. The kinematic viscosity `ν` is **not** baked into the table; `1/√ν`
is applied at runtime from T,P-dependent transport properties.

The full evaporation ventilation integral is assembled at runtime:

```math
I_{\\mathrm{evap}} = \\frac{f_{1r}}{\\lambda_r^2}
    + f_{2r}\\, \\frac{\\mathrm{Sc}^{1/3}}{\\sqrt{\\nu}}\\, I_{\\mathrm{VD}}
```

where `Sc = ν / Dᵛ` is the Schmidt number and `ν` is the T,P-dependent kinematic
viscosity. The ventilation coefficients `f1r = 0.78` and `f2r = 0.32` are the
standard values for falling drops tabulated by
[Pruppacher and Klett (2010)](@cite pruppacher2010microphysics). The constant term
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
function RainEvaporationVentilationEvaluator(FT::DataType = Oceananigans.defaults.FloatType;
                                              n_points::Int = 128)
    nodes, weights = chebyshev_gauss_nodes_weights(FT, n_points)
    return RainEvaporationVentilationEvaluator(nodes, weights)
end

"""
    (e::RainEvaporationVentilationEvaluator)(log10_slope)

Evaluate `I_VD(λ_r)` = ∫ D √(V(D)×D) exp(-λ_r D) dD at the given `log10(λ_r)`.

Returns the velocity-diameter integral in [m^(5/2)]. The `1/√ν`, constant (f1r),
and Schmidt number (Sc^(1/3)) contributions are applied at runtime.
"""
@inline function (e::RainEvaporationVentilationEvaluator)(log10_slope)
    FT = eltype(e.nodes)
    λʳ = exp10(FT(log10_slope))

    result = zero(FT)
    n = length(e.nodes)

    for i in 1:n
        x = @inbounds e.nodes[i]
        w = @inbounds e.weights[i]
        D = transform_to_diameter(x, λʳ)
        J = jacobian_diameter_transform(x, λʳ)

        # Use the piecewise Gunn-Kinzer/Beard fall speed.
        # ν is NOT baked in; 1/√ν applied at runtime from T,P-dependent transport.
        V = rain_fall_speed(D, one(FT))
        VD_sqrt = sqrt(max(V * D, zero(FT)))
        psd = exp(-λʳ * D)

        result += w * D * VD_sqrt * psd * J
    end

    return ifelse(isfinite(result), result, zero(FT))
end
