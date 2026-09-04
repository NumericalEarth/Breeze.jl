#####
##### κ-Köhler droplet physics
#####
##### A wet aerosol particle of diameter D, dry diameter Dᵈ, and hygroscopicity κ is in
##### equilibrium with its environment at the supersaturation
#####
#####     𝒮ᵉ(D) = exp(A / D) (D³ − Dᵈ³) / (D³ − Dᵈ³ (1 − κ)) − 1,
#####
##### where A = 4 σ Mʷ / (R T ρʷ) is the Kelvin length ([Petters & Kreidenweis (2007)](@cite PettersKreidenweis2007)).
##### The Köhler curve has a maximum at the critical diameter Dᶜ, and a particle is counted
##### as an activated droplet when D ≥ Dᶜ. Away from equilibrium the particle grows or
##### evaporates by diffusion of vapor and heat (Maxwell–Mason),
#####
#####     d(D²)/dt = 8 G(T, p, D) [𝒮 − 𝒮ᵉ(D)],
#####
##### with the growth coefficient G including the kinetic (non-continuum) corrections of
##### the vapor diffusivity and thermal conductivity near a small particle
##### ([Pruppacher & Klett (2010)](@cite pruppacher2010microphysics), chapter 13). These are the
##### formulas of the `pyrcel` parcel model used by [Anderson et al. (2023)](@cite Anderson2023).
#####
##### All functions are scalar, allocation-free, and usable inside kernels. Every
##### temperature-dependent coefficient is written for a generic float type `FT`.
#####

"""
$(TYPEDSIGNATURES)

Surface tension of liquid water against air at temperature `T`, in N m⁻¹,
`σ = 0.0761 − 1.55 × 10⁻⁴ (T − 273.15)` (`pyrcel`).
"""
@inline function surface_tension(T::FT) where FT
    return FT(0.0761) - FT(1.55e-4) * (T - FT(273.15))
end

"""
$(TYPEDSIGNATURES)

The Kelvin length `A = 4 σ Mʷ / (R T ρʷ)`, so that the curvature (Kelvin) factor of a droplet
of diameter `D` is `exp(A / D)`.
"""
@inline function kelvin_length(T, constants)
    R = constants.molar_gas_constant
    Mʷ = constants.vapor.molar_mass
    ρʷ = constants.liquid.density
    return 4 * surface_tension(T) * Mʷ / (R * T * ρʷ)
end

"""
$(TYPEDSIGNATURES)

The κ-Köhler equilibrium supersaturation of a wet particle of diameter `D` with dry diameter
`Dᵈ` and hygroscopicity `κ` at temperature `T`,

```math
𝒮ᵉ(D) = \\exp(A / D) \\frac{D³ - Dᵈ³}{D³ - Dᵈ³ (1 - κ)} - 1 .
```
"""
@inline function equilibrium_supersaturation(D, Dᵈ, κ, T, constants)
    A = kelvin_length(T, constants)
    D³ = D^3
    Dᵈ³ = Dᵈ^3
    B = (D³ - Dᵈ³) / (D³ - Dᵈ³ * (1 - κ))
    return exp(A / D) * B - 1
end

"""
$(TYPEDSIGNATURES)

The derivative `d𝒮ᵉ/dD` of the κ-Köhler equilibrium supersaturation with respect to the wet diameter.
"""
@inline function equilibrium_supersaturation_derivative(D, Dᵈ, κ, T, constants)
    A = kelvin_length(T, constants)
    D³ = D^3
    Dᵈ³ = Dᵈ^3
    denominator = D³ - Dᵈ³ * (1 - κ)
    B = (D³ - Dᵈ³) / denominator
    ∂B = 3 * D^2 * Dᵈ³ * κ / denominator^2
    kelvin = exp(A / D)
    return kelvin * (∂B - A / D^2 * B)
end

"""
$(TYPEDSIGNATURES)

The critical diameter `Dᶜ` of the κ-Köhler curve, at which `𝒮ᵉ(D)` is maximal, found by a
golden-section search on `[Dᵈ, D⁺]`. The bracket `D⁺` is a hundred times the dilute-limit
estimate `√(3 κ Dᵈ³ / A)`. The search runs a fixed number of `iterations` so that it can be
used inside kernels; the default locates the maximum to better than one part in 10⁻¹².
"""
@inline function critical_diameter(Dᵈ, κ, T, constants; iterations=64)
    FT = typeof(Dᵈ)
    A = kelvin_length(T, constants)
    D⁺ = 100 * sqrt(3 * κ * Dᵈ^3 / A)
    φ = (sqrt(FT(5)) - 1) / 2

    # Search in the logarithm of the diameter, which makes the Köhler maximum well conditioned
    a = log(Dᵈ * (1 + FT(1e-6)))
    b = log(D⁺)
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    𝒮c = equilibrium_supersaturation(exp(c), Dᵈ, κ, T, constants)
    𝒮d = equilibrium_supersaturation(exp(d), Dᵈ, κ, T, constants)

    for _ in 1:iterations
        # Keep the part of the bracket that contains the larger value: [a, d] or [c, b]
        left = 𝒮c > 𝒮d
        a = ifelse(left, a, c)
        b = ifelse(left, d, b)
        c = b - φ * (b - a)
        d = a + φ * (b - a)
        𝒮c = equilibrium_supersaturation(exp(c), Dᵈ, κ, T, constants)
        𝒮d = equilibrium_supersaturation(exp(d), Dᵈ, κ, T, constants)
    end

    return exp((a + b) / 2)
end

"""
$(TYPEDSIGNATURES)

The critical supersaturation `𝒮ᶜ = 𝒮ᵉ(Dᶜ)`, the maximum of the κ-Köhler curve.
"""
@inline function critical_supersaturation(Dᵈ, κ, T, constants; iterations=64)
    Dᶜ = critical_diameter(Dᵈ, κ, T, constants; iterations)
    return equilibrium_supersaturation(Dᶜ, Dᵈ, κ, T, constants)
end

"""
$(TYPEDSIGNATURES)

The wet diameter at which a haze particle is in equilibrium with the supersaturation `𝒮`,
that is, the solution of `𝒮ᵉ(D) = 𝒮` on the stable branch `Dᵈ < D < Dᶜ` of the Köhler curve,
found by bisection in a fixed number of `iterations`. The supersaturation must lie below
the critical supersaturation; above it no equilibrium exists and the critical diameter
is returned.
"""
@inline function equilibrium_diameter(𝒮, Dᵈ, κ, T, constants; iterations=64)
    FT = typeof(Dᵈ)
    Dᶜ = critical_diameter(Dᵈ, κ, T, constants; iterations)
    a = Dᵈ * (1 + FT(1e-9))
    b = Dᶜ
    for _ in 1:iterations
        m = (a + b) / 2
        below = equilibrium_supersaturation(m, Dᵈ, κ, T, constants) < 𝒮
        a = ifelse(below, m, a)
        b = ifelse(below, b, m)
    end
    return (a + b) / 2
end

#####
##### Diffusional growth
#####

"""
$(TYPEDSIGNATURES)

The diffusivity of water vapor in air near a droplet of diameter `D`, in m² s⁻¹: the
continuum value `Dᵛ = 2.11 × 10⁻⁵ (T / 273 K)^1.94 (1 atm / p)` divided by the kinetic
correction `1 + (2 Dᵛ / (αᶜ D)) √(2π Mʷ / (R T))`, with `αᶜ` the condensation (mass
accommodation) coefficient.
"""
@inline function vapor_diffusivity(T, p, D, accommodation, constants)
    FT = typeof(T)
    R = constants.molar_gas_constant
    Mʷ = constants.vapor.molar_mass
    Dᵛ = FT(2.11e-5) * (T / FT(273))^FT(1.94) * (FT(101325) / p)
    mean_speed_factor = sqrt(2 * FT(π) * Mʷ / (R * T))
    return Dᵛ / (1 + 2 * Dᵛ / (accommodation * D) * mean_speed_factor)
end

"""
$(TYPEDSIGNATURES)

The thermal conductivity of air near a droplet of diameter `D`, in W m⁻¹ K⁻¹: the continuum
value `kᵃ = 10⁻³ (4.39 + 0.071 T)` divided by the kinetic correction
`1 + (2 kᵃ / (αᵀ D ρᵃ cᵖ)) √(2π Mᵃ / (R T))`, with `αᵀ` the thermal accommodation coefficient
and `ρᵃ` the air density.
"""
@inline function thermal_conductivity(T, D, ρᵃ, thermal_accommodation, constants)
    FT = typeof(T)
    R = constants.molar_gas_constant
    Mᵃ = constants.dry_air.molar_mass
    cᵖ = constants.dry_air.heat_capacity
    kᵃ = FT(1e-3) * (FT(4.39) + FT(0.071) * T)
    mean_speed_factor = sqrt(2 * FT(π) * Mᵃ / (R * T))
    return kᵃ / (1 + 2 * kᵃ / (thermal_accommodation * D * ρᵃ * cᵖ) * mean_speed_factor)
end

"""
$(TYPEDSIGNATURES)

The Maxwell–Mason growth coefficient `G`, in m² s⁻¹, of a droplet of diameter `D` at
temperature `T` and pressure `p`, defined by `d(D²)/dt = 8 G (𝒮 − 𝒮ᵉ)`:

```math
G = \\left[ \\frac{ρʷ R T}{pᵛ⁺ Dᵛ Mʷ} + \\frac{ℒ ρʷ}{kᵃ T} \\left( \\frac{ℒ Mʷ}{R T} - 1 \\right) \\right]^{-1},
```

with the kinetically corrected vapor diffusivity `Dᵛ` and thermal conductivity `kᵃ` from
[`vapor_diffusivity`](@ref) and [`thermal_conductivity`](@ref). The `accommodation`
coefficients are taken from `parameters`.
"""
@inline function growth_coefficient(T, p, D, parameters, constants)
    R = constants.molar_gas_constant
    Mʷ = constants.vapor.molar_mass
    ρʷ = constants.liquid.density
    Rᵈ = dry_air_gas_constant(constants)
    ℒ = liquid_latent_heat(T, constants)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
    ρᵃ = p / (Rᵈ * T)
    Dᵛ = vapor_diffusivity(T, p, D, parameters.accommodation, constants)
    kᵃ = thermal_conductivity(T, D, ρᵃ, parameters.thermal_accommodation, constants)
    Gᵃ = ρʷ * R * T / (pᵛ⁺ * Dᵛ * Mʷ)
    Gᵇ = ℒ * ρʷ * (ℒ * Mʷ / (R * T) - 1) / (kᵃ * T)
    return 1 / (Gᵃ + Gᵇ)
end

"""
$(TYPEDSIGNATURES)

Advance the squared wet diameter `D²` of a droplet with dry diameter `Dᵈ` and hygroscopicity
`κ` over a time `Δt` at the ambient supersaturation `𝒮`, temperature `T`, and pressure `p`,
by one backward-Euler step of `d(D²)/dt = 8 G (𝒮 − 𝒮ᵉ(D))`. The implicit equation is solved
by a fixed number of Newton iterations in `D²`, with the wet diameter floored at the dry
diameter, so that the stiff equilibration of sub-micron haze is stable at the time step of
the flow. The number of iterations and the accommodation coefficients are taken from
`parameters`.
"""
@inline function implicit_growth_step(D², 𝒮, T, p, Dᵈ, κ, Δt, parameters, constants)
    D²ₘᵢₙ = Dᵈ^2
    D²⁺ = max(D², D²ₘᵢₙ)
    for _ in 1:parameters.newton_iterations
        D = sqrt(D²⁺)
        G = growth_coefficient(T, p, D, parameters, constants)
        𝒮ᵉ = equilibrium_supersaturation(D, Dᵈ, κ, T, constants)
        ∂𝒮ᵉ = equilibrium_supersaturation_derivative(D, Dᵈ, κ, T, constants)
        residual = D²⁺ - D² - 8 * Δt * G * (𝒮 - 𝒮ᵉ)
        # d(residual)/d(D²) with G frozen; ∂𝒮ᵉ/∂(D²) = ∂𝒮ᵉ/∂D / (2D)
        slope = 1 + 8 * Δt * G * ∂𝒮ᵉ / (2 * D)
        D²⁺ = max(D²⁺ - residual / slope, D²ₘᵢₙ)
    end
    return D²⁺
end

"""
$(TYPEDSIGNATURES)

The supersaturation with respect to liquid water of moist air at temperature `T`, vapor
mass fraction `qᵛ`, and pressure `p`: `𝒮 = pᵛ / pᵛ⁺(T) − 1` with `pᵛ = ρ qᵛ Rᵛ T` and the
gas-phase density `ρ = p / (Rᵐ T)` diagnosed at the pressure `p`, exactly as Breeze's
anelastic model computes it for its own microphysics (`gas_phase_density`).
"""
@inline function ambient_supersaturation(T, qᵛ, p, constants)
    q = MoistureMassFractions(qᵛ)
    ρ = density(T, p, q, constants)
    return supersaturation(T, ρ, q, constants, PlanarLiquidSurface())
end
