#####
##### Manufactured subgrid states: horizontally resolved columns whose joint distribution of
##### (w, θˡⁱ, qᵗ) at a level is exactly the double Gaussian that the turbulence closure assumes.
#####
##### Not a test file (runtests.jl skips it). Included by `manufactured_subgrid_statistics.jl` and
##### by the convergence script of the harness.
#####
##### The construction runs forward from the moments that the closure uses,
#####
#####     w̄, θ̄ˡⁱ, q̄ᵗ, w′², Sk_w, θˡⁱ′², qᵗ′², r_wθ, r_wq, r_θq,
#####
##### to component weights, component means and within-component (co)variances (section 2 of the design):
#####
#####     a = (1 - Sk_w / √(4 (1 - ℂᵍ₁)³ + Sk_w²)) / 2,
#####     ŵ₁ = √((1 - a)/a) √(1 - ℂᵍ₁),  ŵ₂ = -√(a/(1 - a)) √(1 - ℂᵍ₁),  within-component w variance ℂᵍ₁ w′²,
#####     φ̂₁ = -r_wφ / ŵ₂,  φ̂₂ = -r_wφ / ŵ₁,
#####     within-component variances of φ from the variance and the skewness Sk_φ of φ
#####     (Sk_θ = 0; Sk_q = ℂᵍ₂ Sk_w times the share that rises with |φ̂₂ - φ̂₁| about (ℂᵍ₃ + ℂᵍ₄) / 2),
#####     one within-component θ-q correlation, from θ′q′ minus the part carried by the component means.
#####
##### This forward construction is written here independently of the closure code, which inverts it.
##### w is uncorrelated with the scalars within each component. Each fine column is an independent draw,
##### so any block of n × n fine columns is a sample of the same distribution: averaging by a factor
##### of 2 in each horizontal direction keeps the distribution and quadruples the sample.
#####
##### "Truth" in a coarse cell is the average over its fine columns of quantities computed column by
##### column with the model's thermodynamics: saturation adjustment of (θˡⁱ, qᵗ, p) on the warm
##### (liquid) surface, the anelastic specific volume α = Rᵐ T / p, and P3's vapor excess over liquid.
#####

using Breeze
using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions, LiquidIcePotentialTemperatureState,
                             VaporEquivalentExponent, WarmPhaseEquilibrium, PlanarLiquidSurface, temperature,
                             mixture_gas_constant, mixture_heat_capacity, vapor_equivalent_heat_capacity,
                             exner_exponent, saturation_specific_humidity
using Breeze.Microphysics: SaturationAdjustment, adjust_thermodynamic_state
using Breeze.Solvers: SecantSolver
using Breeze.AtmosphereModels: PrescribedMixtureStatistics, supersaturation_statistics
using Breeze.Microphysics.PredictedParticleProperties: subgrid_vapor_excess_correlation
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: assumed_pdf, scalar_component_parameters, component_saturation,
                                                               liquid_water_temperature, component_condensate,
                                                               buoyancy_flux_coefficients, skewness_share, DoubleGaussianPDF
using SpecialFunctions: erf

#####
##### A deterministic generator (SplitMix64 and Box-Muller), so the samples do not depend on the
##### Julia version's default generator
#####

mutable struct SplitMix64
    state :: UInt64
end

function next_uint!(generator::SplitMix64)
    generator.state += 0x9e3779b97f4a7c15
    z = generator.state
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

uniform!(generator::SplitMix64) = ((next_uint!(generator) >> 11) + 0.5) * 2.0^-53   # in (0, 1)

function normal_pair!(generator::SplitMix64)
    u₁, u₂ = uniform!(generator), uniform!(generator)
    radius = sqrt(-2log(u₁))
    return radius * cospi(2u₂), radius * sinpi(2u₂)
end

#####
##### Thermodynamics of the manufactured states: the model's, as the closure sees it
#####

manufactured_thermodynamics(constants = ThermodynamicConstants(Float64)) =
    (constants, standard_pressure = 1e5, exner_exponent = VaporEquivalentExponent(), phase_equilibrium = WarmPhaseEquilibrium())

const manufactured_pdf = DoubleGaussianPDF(Float64)
const manufactured_clipping = manufactured_pdf.clipping
const warm_adjustment = SaturationAdjustment(Float64; equilibrium = WarmPhaseEquilibrium(),
                                             solver = SecantSolver(Float64; abstol = 1e-10, maxiter = 60))

#####
##### The prescribed moments of a level, and the double Gaussian they define
#####

"""
The moments of a level from which the double Gaussian is constructed. `δq` sets the mean total
water relative to the closure's saturation at the mean: `q̄ᵗ = qᵛ⁺(Π θ̄) + δq`.
"""
Base.@kwdef struct LevelMoments
    name :: String
    p :: Float64 = 9e4
    w̄ :: Float64 = 0.0
    θ̄ :: Float64 = 297.0
    δq :: Float64 = 0.0
    w′² :: Float64 = 0.5
    Sk_w :: Float64 = 0.0
    σθ :: Float64 = 0.3
    σq :: Float64 = 5e-4
    r_wθ :: Float64 = 0.0
    r_wq :: Float64 = 0.0
    r_θq :: Float64 = 0.0
end

function saturation_at_mean(level::LevelMoments, thermodynamics)
    Tˡ = liquid_water_temperature(thermodynamics, level.θ̄, 0.0, level.p)
    qᵛ⁺ = first(component_saturation(thermodynamics, Tˡ, 0.0, level.p))
    # one fixed-point pass for the qᵗ dependence of Π and of qᵛ⁺
    q̄ = qᵛ⁺ + level.δq
    Tˡ = liquid_water_temperature(thermodynamics, level.θ̄, q̄, level.p)
    qᵛ⁺ = first(component_saturation(thermodynamics, Tˡ, q̄, level.p))
    return qᵛ⁺
end

mean_total_water(level::LevelMoments, thermodynamics) = saturation_at_mean(level, thermodynamics) + level.δq

# Within-component variances (over the variance) φ̂′²₁, φ̂′²₂ of a standardized scalar with component means φ̂₁, φ̂₂,
# from  a(φ̂₁² + φ̂′²₁) + (1-a)(φ̂₂² + φ̂′²₂) = 1  and  a(φ̂₁³ + 3φ̂₁φ̂′²₁) + (1-a)(φ̂₂³ + 3φ̂₂φ̂′²₂) = Sk.
function within_component_variances(a, φ̂₁, φ̂₂, Sk)
    unexplained = 1 - a * φ̂₁^2 - (1 - a) * φ̂₂^2
    third = Sk - a * φ̂₁^3 - (1 - a) * φ̂₂^3
    # [a  1-a; 3aφ̂₁  3(1-a)φ̂₂] [φ̂′²₁; φ̂′²₂] = [unexplained; third]
    determinant = a * 3(1 - a) * φ̂₂ - (1 - a) * 3a * φ̂₁
    φ̂′²₁ = (3(1 - a) * φ̂₂ * unexplained - (1 - a) * third) / determinant
    φ̂′²₂ = (a * third - 3a * φ̂₁ * unexplained) / determinant
    return φ̂′²₁, φ̂′²₂
end

"""
    DoubleGaussianLevel(level, thermodynamics)

The double Gaussian of `level`: component weights `a`, component means and within-component standard
deviations of w, θˡⁱ and qᵗ, and the within-component θ-q correlation `r`.
"""
struct DoubleGaussianLevel
    moments :: LevelMoments
    p :: Float64
    a :: Float64
    w :: NTuple{2, Float64}
    θ :: NTuple{2, Float64}
    q :: NTuple{2, Float64}
    σw :: Float64
    σθ :: NTuple{2, Float64}
    σq :: NTuple{2, Float64}
    r :: Float64
    q̄ :: Float64
end

function DoubleGaussianLevel(m::LevelMoments, thermodynamics; pdf = manufactured_pdf)
    ℂᵍ₁ = pdf.ℂᵍ₁
    Sk = m.Sk_w
    a = (1 - Sk / sqrt(4 * (1 - ℂᵍ₁)^3 + Sk^2)) / 2
    ŵ₁ = sqrt((1 - a) / a) * sqrt(1 - ℂᵍ₁)
    ŵ₂ = -sqrt(a / (1 - a)) * sqrt(1 - ℂᵍ₁)
    σw = sqrt(m.w′²)

    θ̂ = (-m.r_wθ / ŵ₂, -m.r_wθ / ŵ₁)
    q̂ = (-m.r_wq / ŵ₂, -m.r_wq / ŵ₁)
    θ̂′² = within_component_variances(a, θ̂..., 0.0)

    Sk_q = pdf.ℂᵍ₂ * Sk * skewness_share(pdf, abs(q̂[2] - q̂[1]))
    q̂′² = within_component_variances(a, q̂..., Sk_q)
    all(>(0), (θ̂′²..., q̂′²...)) || error("level $(m.name): a within-component variance is not positive: $θ̂′², $q̂′²")

    q̄ = mean_total_water(m, thermodynamics)
    σθ = m.σθ .* sqrt.(θ̂′²)
    σq = m.σq .* sqrt.(q̂′²)
    between = a * θ̂[1] * q̂[1] + (1 - a) * θ̂[2] * q̂[2]
    r = (m.r_θq - between) / (a * sqrt(θ̂′²[1] * q̂′²[1]) + (1 - a) * sqrt(θ̂′²[2] * q̂′²[2]))
    abs(r) < 1 || error("level $(m.name): within-component correlation $r is not realizable")

    return DoubleGaussianLevel(m, m.p, a, m.w̄ .+ σw .* (ŵ₁, ŵ₂), m.θ̄ .+ m.σθ .* θ̂, q̄ .+ m.σq .* q̂,
                               sqrt(ℂᵍ₁) * σw, σθ, σq, r, q̄)
end

#####
##### Fine columns
#####

"""
    sample_columns(mixture, L; seed)

`L × L` fine columns of the level `mixture`: matrices of w, θˡⁱ, qᵗ and the component label (1 or 2),
each column an independent draw.
"""
function sample_columns(mixture::DoubleGaussianLevel, L; seed = 1)
    generator = SplitMix64(UInt64(seed))
    w = zeros(L, L); θ = zeros(L, L); q = zeros(L, L); component = zeros(Int8, L, L)
    for j in 1:L, i in 1:L
        n = uniform!(generator) < mixture.a ? 1 : 2
        z₁, z₂ = normal_pair!(generator)
        z₃, _ = normal_pair!(generator)
        component[i, j] = n
        w[i, j] = mixture.w[n] + mixture.σw * z₁
        θ[i, j] = mixture.θ[n] + mixture.σθ[n] * z₂
        q[i, j] = mixture.q[n] + mixture.σq[n] * (mixture.r * z₂ + sqrt(1 - mixture.r^2) * z₃)
    end
    return (; w, θ, q, component)
end

"""
    column_thermodynamics(θ, q, p, thermodynamics)

The thermodynamics of one fine column, with the model's functions: the equilibrium liquid `qˡ`
and temperature `T` of saturation adjustment, the specific volume `α = Rᵐ T / p`, the vapor
excess over liquid `δqᵛ` that P3 sees when the column holds no condensate
(`qᵗ - qᵛ⁺ˡ(T = Π θˡⁱ, ρ)`), and the signed saturation excess `s`: `qˡ` where saturated and
`(qᵗ - qᵛ⁺(Π θˡⁱ)) / (1 + β qᵛ⁺)` where not, the quantity whose flux the closure's aʷ divides by.
"""
function column_thermodynamics(θ, q, p, thermodynamics)
    constants = thermodynamics.constants
    pˢᵗ = thermodynamics.standard_pressure
    𝒰₀ = LiquidIcePotentialTemperatureState(θ, MoistureMassFractions(q), pˢᵗ, p)
    𝒰 = adjust_thermodynamic_state(𝒰₀, warm_adjustment, constants)
    T = temperature(𝒰, constants)
    qˡ = 𝒰.moisture_mass_fractions.liquid
    α = mixture_gas_constant(𝒰.moisture_mass_fractions, constants) * T / p

    Tˡ = temperature(𝒰₀, constants)
    ρˡ = p / (mixture_gas_constant(MoistureMassFractions(q), constants) * Tˡ)
    δqᵛ = q - saturation_specific_humidity(Tˡ, ρˡ, constants, PlanarLiquidSurface())

    qᵛ⁺, β = component_saturation(thermodynamics, Tˡ, q, p)
    s = qˡ > 0 ? qˡ : (q - qᵛ⁺) / (1 + β * qᵛ⁺)
    return (; qˡ, T, α, δqᵛ, s)
end

function column_fields(columns, p, thermodynamics)
    L = size(columns.θ, 1)
    out = [column_thermodynamics(columns.θ[i, j], columns.q[i, j], p, thermodynamics) for i in 1:L, j in 1:L]
    return (; columns..., qˡ = getfield.(out, :qˡ), α = getfield.(out, :α), δqᵛ = getfield.(out, :δqᵛ),
              s = getfield.(out, :s))
end

#####
##### Coarse cells: averages over n × n blocks of fine columns
#####

block(x, n, I, J) = view(x, (I-1)*n+1:I*n, (J-1)*n+1:J*n)

mean_of(x) = sum(x) / length(x)
covariance(x, y, x̄, ȳ) = sum((xᵢ - x̄) * (yᵢ - ȳ) for (xᵢ, yᵢ) in zip(x, y)) / length(x)

"""
    cell_statistics(fields, n, I, J, p, g)

The sampled moments and the "truth" of coarse cell `(I, J)` of size `n × n` fine columns.
`g` is the gravitational acceleration.
"""
function cell_statistics(f, n, I, J, p, g)
    w, θ, q = block(f.w, n, I, J), block(f.θ, n, I, J), block(f.q, n, I, J)
    qˡ, α, δqᵛ, s, component = block(f.qˡ, n, I, J), block(f.α, n, I, J), block(f.δqᵛ, n, I, J), block(f.s, n, I, J),
                               block(f.component, n, I, J)
    w̄, θ̄, q̄ = mean_of(w), mean_of(θ), mean_of(q)
    w′² = covariance(w, w, w̄, w̄)
    w′³ = sum((wᵢ - w̄)^3 for wᵢ in w) / length(w)
    moments = (; w̄, θ̄, q̄, w′², w′³,
                 θ′² = covariance(θ, θ, θ̄, θ̄), q′² = covariance(q, q, q̄, q̄),
                 w′θ′ = covariance(w, θ, w̄, θ̄), w′q′ = covariance(w, q, w̄, q̄), θ′q′ = covariance(θ, q, θ̄, q̄))

    q̄ˡ = mean_of(qˡ)
    ᾱ = mean_of(α)
    # b = g (α / ᾱ - 1), the anelastic buoyancy about the cell-mean specific volume
    w′b′ = g * covariance(w, α, w̄, ᾱ) / ᾱ
    w′qˡ′ = covariance(w, qˡ, w̄, q̄ˡ)
    w′s′ = covariance(w, s, w̄, mean_of(s))
    in₁ = component .== 1
    n₁ = count(in₁)
    component_liquid = (n₁ > 0 ? sum(qˡ[in₁]) / n₁ : NaN, n₁ < length(qˡ) ? sum(qˡ[.!in₁]) / (length(qˡ) - n₁) : NaN)
    truth = (; aᶜ = count(>(0), qˡ) / length(qˡ), qˡ = q̄ˡ, w′qˡ′, w′b′, w′s′, aʷ = w′qˡ′ / w′s′,
               excess = mean_of(max.(0, δqᵛ)), f⁺ = count(>(0), δqᵛ) / length(δqᵛ),
               updraft_fraction = n₁ / length(qˡ), component_liquid)
    return (; moments, truth)
end

#####
##### The analytic diagnostics, from the moments of a cell
#####

"""
    closure_components(moments, p, thermodynamics)

The double Gaussian that the closure reconstructs from the moments of a cell, per component, with the
closure's functions: the weight of the updraft component from Sk_w (as `assumed_pdf` computes it), the scalar
component parameters (`scalar_component_parameters`), the within-component correlation, and each component's
saturation (`component_saturation`) and equilibrium condensate (`component_condensate`).
"""
function closure_components(m, p, thermodynamics; pdf = manufactured_pdf)
    clipping = manufactured_clipping
    σw = sqrt(m.w′²)
    Sk = m.w′³ / m.w′²^(3/2)
    ℂᵍ₁ = pdf.ℂᵍ₁
    a = clamp((1 - Sk / sqrt(4 * (1 - ℂᵍ₁)^3 + Sk^2)) / 2, pdf.minimum_updraft_fraction, pdf.maximum_updraft_fraction)
    ŵ₁ = sqrt((1 - a) / a) * sqrt(1 - ℂᵍ₁)
    ŵ₂ = -sqrt(a / (1 - a)) * sqrt(1 - ℂᵍ₁)
    σθ₊ = max(pdf.potential_temperature_tolerance, sqrt(m.θ′²))
    σq₊ = max(pdf.total_water_tolerance, sqrt(m.q′²))
    θ₁, θ₂, θ′²₁, θ′²₂, σθ₁, σθ₂ = scalar_component_parameters(clipping, m.w′θ′, σw, σθ₊, m.θ′², m.θ̄, ŵ₁, ŵ₂, Sk, a, false, pdf)
    q₁, q₂, q′²₁, q′²₂, σq₁, σq₂ = scalar_component_parameters(clipping, m.w′q′, σw, σq₊, m.q′², m.q̄, ŵ₁, ŵ₂, Sk, a, true, pdf)
    spread = a * σq₁ * σθ₁ + (1 - a) * σq₂ * σθ₂
    r = clamp((m.θ′q′ - a * (q₁ - m.q̄) * (θ₁ - m.θ̄) - (1 - a) * (q₂ - m.q̄) * (θ₂ - m.θ̄)) / spread, -1, 1)

    component(θᵢ, qᵢ, θ′²ᵢ, q′²ᵢ, σθᵢ, σqᵢ) = begin
        Tˡ = liquid_water_temperature(thermodynamics, θᵢ, qᵢ, p)
        qᵛ⁺, β, cᵖᵐℒ⁻¹, ∂Tˡ∂θ = component_saturation(thermodynamics, Tˡ, qᵢ, p)
        s, σₛ, qᶜ, aᶜ = component_condensate(qᵢ, qᵛ⁺, β, cᵖᵐℒ⁻¹, ∂Tˡ∂θ, θ′²ᵢ, q′²ᵢ, σθᵢ, σqᵢ, r)
        ∂qᵛ⁺∂θ = cᵖᵐℒ⁻¹ * β * qᵛ⁺ * ∂Tˡ∂θ
        # The vapor excess over liquid without condensate, as P3 defines its saturation, and its spread
        ρ = p / (mixture_gas_constant(MoistureMassFractions(qᵢ), thermodynamics.constants) * Tˡ)
        μ = qᵢ - saturation_specific_humidity(Tˡ, ρ, thermodynamics.constants, PlanarLiquidSurface())
        σ = sqrt(max(0, q′²ᵢ + ∂qᵛ⁺∂θ^2 * θ′²ᵢ - 2 * ∂qᵛ⁺∂θ * r * σθᵢ * σqᵢ))
        (; Tˡ, qᵛ⁺, β, s, σₛ, qᶜ, aᶜ, μ, σ)
    end
    components = (component(θ₁, q₁, θ′²₁, q′²₁, σθ₁, σq₁), component(θ₂, q₂, θ′²₂, q′²₂, σθ₂, σq₂))
    w = (m.w̄ + ŵ₁ * σw, m.w̄ + ŵ₂ * σw)
    return (; a, w, θ = (θ₁, θ₂), q = (q₁, q₂), σθ = (σθ₁, σθ₂), σq = (σq₁, σq₂), r, components)
end

"""
    vapor_excess_statistics(mixture)

The statistics of the vapor excess over liquid that P3's `SubgridCondensation` reads. Until the
closure-side `TurbulenceClosureStatistics` lands, they are formed here from the closure's components
and passed through `PrescribedMixtureStatistics`; switch this one function when it does.
"""
vapor_excess_statistics(mixture) =
    PrescribedMixtureStatistics((mixture.a, 1 - mixture.a), (mixture.components[1].μ, mixture.components[2].μ),
                                (mixture.components[1].σ, mixture.components[2].σ))

standard_normal_cdf(x) = (1 + erf(x / sqrt(2))) / 2

"""
    analytic_diagnostics(moments, p, thermodynamics)

The diagnostics of the closure and of P3 from the moments of a cell:

* `aᶜ`, `qˡ`, `w′qˡ′`, `w′θᵛ′`: `assumed_pdf` with the model's saturation;
* `w′b′ = (∂b/∂θˡⁱ) w′θᵛ′`, the closure's buoyancy flux, and `w′b′ˡⁱⁿ`, the same linear coefficients
  applied to a given `w′qˡ′` (the sampled one), which isolates the linearization of b. `∂b/∂θˡⁱ = g ĉᵖ Π / (cᵖᵐ T)`
  is that of the model's buoyancy `b = g (α / ᾱ - 1)`, `α = Rᵐ T / p`, at the cell mean;
* `aʷ = w′qˡ′ / w′s′`, with `w′s′` from the linearized saturation excess at the cell mean;
* `Γ`, `excess = μ̄ + Γ` and `f⁺` of P3's `SubgridCondensation` about the grid-mean vapor excess `μ̄`;
* `qᶜ`, the components' equilibrium condensate, for the split rule.
"""
function analytic_diagnostics(m, p, thermodynamics; w′qˡ′_given = nothing, pdf = manufactured_pdf)
    constants = thermodynamics.constants
    g = constants.gravitational_acceleration
    moments = (m.w′³, m.θ′², m.w′θ′, m.θ′q′, m.w′q′, m.q′²)
    aᶜ, qˡ, _, w′qˡ′, w′θᵛ′ = assumed_pdf(pdf, thermodynamics, 1.0, 0.0, 0.5, p, m.θ̄, m.w̄, m.q̄, m.w′², moments, moments)

    A_q, A_c = buoyancy_flux_coefficients(thermodynamics, p, m.θ̄, m.q̄, qˡ)
    # ∂b/∂θˡⁱ = g ĉᵖ Π / (cᵖᵐ T) at the cell mean with its mean liquid, from the model's functions
    q = MoistureMassFractions(m.q̄ - qˡ, qˡ, zero(qˡ))
    𝒰 = LiquidIcePotentialTemperatureState(m.θ̄, q, thermodynamics.standard_pressure, p)
    Π = (p / thermodynamics.standard_pressure)^exner_exponent(thermodynamics.exner_exponent, q, constants)
    ∂b∂θ = g * vapor_equivalent_heat_capacity(q, constants) * Π / (mixture_heat_capacity(q, constants) * temperature(𝒰, constants))
    w′b′ = ∂b∂θ * w′θᵛ′
    w′b′ˡⁱⁿ = ∂b∂θ * (m.w′θ′ + A_q * m.w′q′ + A_c * something(w′qˡ′_given, w′qˡ′))

    # The linearized saturation excess at the cell mean: s = c_q qᵗ′ - c_θ θˡⁱ′
    Tˡ = liquid_water_temperature(thermodynamics, m.θ̄, m.q̄, p)
    qᵛ⁺, β, cᵖᵐℒ⁻¹, ∂Tˡ∂θ = component_saturation(thermodynamics, Tˡ, m.q̄, p)
    c_θ = ((1 + β * m.q̄) / (1 + β * qᵛ⁺)^2) * cᵖᵐℒ⁻¹ * β * qᵛ⁺ * ∂Tˡ∂θ
    c_q = 1 / (1 + β * qᵛ⁺)
    w′s′ = c_q * m.w′q′ - c_θ * m.w′θ′

    mixture = closure_components(m, p, thermodynamics; pdf)
    statistics = supersaturation_statistics(vapor_excess_statistics(mixture), 1, 1, 1, nothing, nothing, nothing, nothing)
    ρ̄ = p / (mixture_gas_constant(MoistureMassFractions(m.q̄), constants) * Tˡ)
    μ̄ = m.q̄ - saturation_specific_humidity(Tˡ, ρ̄, constants, PlanarLiquidSurface())
    Γ = subgrid_vapor_excess_correlation(statistics, μ̄)
    shift = μ̄ - (statistics.a[1] * statistics.μ[1] + statistics.a[2] * statistics.μ[2])
    f⁺ = sum(statistics.a[i] * standard_normal_cdf((statistics.μ[i] + shift) / statistics.σ[i]) for i in 1:2)

    return (; aᶜ, qˡ, w′qˡ′, w′θᵛ′, w′b′, w′b′ˡⁱⁿ, w′s′, aʷ = w′qˡ′ / w′s′, Γ, μ̄, excess = μ̄ + Γ, f⁺,
              qᶜ = (mixture.components[1].qᶜ, mixture.components[2].qᶜ), mixture)
end

#####
##### The component condensate split (notes/les-closure-tests.md §D)
#####

"""
    component_condensate_split(q̄ᶜˡ, a, weights, qᵗ)

Split the grid-mean prognostic condensate `q̄ᶜˡ` between the components in proportion to `weights`
(the components' equilibrium condensate `qᶜᵢ`, or their supersaturated fractions `fᵢ`):
`qᶜˡᵢ = q̄ᶜˡ wᵢ / Σⱼ aⱼ wⱼ`. Where a component would hold more condensate than its total water
`qᵗᵢ`, the excess goes to the other component, which keeps the grid mean. Where `Σ a w = 0` the
condensate goes to the component of larger `qᵗᵢ`. Not in src: this is the reference implementation
the tests check.
"""
function component_condensate_split(q̄ᶜˡ, a, weights, qᵗ)
    a = (a, 1 - a)
    total = a[1] * weights[1] + a[2] * weights[2]
    weights = total > 0 ? weights : (qᵗ[1] ≥ qᵗ[2] ? (1.0, 0.0) : (0.0, 1.0))
    total = a[1] * weights[1] + a[2] * weights[2]
    qᶜˡ = (q̄ᶜˡ * weights[1] / total, q̄ᶜˡ * weights[2] / total)
    for (i, j) in ((1, 2), (2, 1))
        if qᶜˡ[i] > qᵗ[i]
            excess = a[i] * (qᶜˡ[i] - qᵗ[i])
            qᶜˡ = i == 1 ? (qᵗ[1], qᶜˡ[2] + excess / a[2]) : (qᶜˡ[1] + excess / a[1], qᵗ[2])
        end
    end
    return qᶜˡ
end

#####
##### The levels of the tests and the script
#####

const manufactured_levels = (
    # Shallow cumulus: subsaturated mean, a narrow moist updraft component that reaches saturation
    LevelMoments(name = "cumulus", δq = -6e-4, w′² = 0.6, Sk_w = 0.8, σθ = 0.25, σq = 6e-4,
                 r_wθ = -0.2, r_wq = 0.5, r_θq = -0.5),
    # Stratocumulus: mean near saturation, negatively skewed (narrow downdrafts)
    LevelMoments(name = "stratocumulus", δq = 1e-4, w′² = 0.4, Sk_w = -0.5, σθ = 0.2, σq = 4e-4,
                 r_wθ = -0.3, r_wq = 0.4, r_θq = -0.6),
    # Overcast: saturated mean, the dry tail of the downdraft component clears
    LevelMoments(name = "overcast", δq = 8e-4, w′² = 0.3, Sk_w = 0.2, σθ = 0.2, σq = 5e-4,
                 r_wθ = -0.1, r_wq = 0.3, r_θq = -0.3),
    # Nearly clear: only the far tail of the updraft component saturates
    LevelMoments(name = "clear", δq = -1.4e-3, w′² = 0.5, Sk_w = 0.6, σθ = 0.3, σq = 5e-4,
                 r_wθ = -0.1, r_wq = 0.5, r_θq = -0.4),
)
