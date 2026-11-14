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
##### to plume weights, plume means and within-plume (co)variances (section 2 of the design):
#####
#####     a = (1 - Sk_w / √(4 (1 - ℂᵍ₁)³ + Sk_w²)) / 2,
#####     ŵ₁ = √((1 - a)/a) √(1 - ℂᵍ₁),  ŵ₂ = -√(a/(1 - a)) √(1 - ℂᵍ₁),  within-plume w variance ℂᵍ₁ w′²,
#####     φ̂₁ = -r_wφ / ŵ₂,  φ̂₂ = -r_wφ / ŵ₁,
#####     within-plume variances of φ from the variance and the skewness Sk_φ of φ
#####     (Sk_θ = 0; Sk_q = ℂᵍ₂ Sk_w ramped in |φ̂₂ - φ̂₁| between ℂᵍ₃ and ℂᵍ₄),
#####     one within-plume θ-q correlation, from θ′q′ minus the part carried by the plume means.
#####
##### This forward construction is written here independently of the closure code, which inverts it.
##### w is uncorrelated with the scalars within each plume. Each fine column is an independent draw,
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
                             mixture_gas_constant, saturation_specific_humidity
using Breeze.Microphysics: SaturationAdjustment, adjust_thermodynamic_state
using Breeze.Solvers: SecantSolver
using Breeze.AtmosphereModels: PrescribedPlumeStatistics, supersaturation_statistics
using Breeze.Microphysics.PredictedParticleProperties: subgrid_vapor_excess_correlation
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: assumed_pdf, scalar_plume_parameters, plume_saturation,
                                                               plume_liquid_water_temperature, plume_condensate,
                                                               buoyancy_flux_coefficients, DoubleGaussianPDF,
                                                               ModelThermodynamicsSaturation, HardClipping, SHOCConstants
using SpecialFunctions: erf

#####
##### A deterministic generator (SplitMix64 and Box-Muller), so the samples do not depend on the
##### Julia version's default generator
#####

mutable struct SplitMix64
    state :: UInt64
end

function next_uint!(g::SplitMix64)
    g.state += 0x9e3779b97f4a7c15
    z = g.state
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

uniform!(g::SplitMix64) = ((next_uint!(g) >> 11) + 0.5) * 2.0^-53   # in (0, 1)

function normal_pair!(g::SplitMix64)
    u₁, u₂ = uniform!(g), uniform!(g)
    ρ = sqrt(-2log(u₁))
    return ρ * cospi(2u₂), ρ * sinpi(2u₂)
end

#####
##### Thermodynamics of the manufactured states: the model's, as the closure sees it
#####

manufactured_thermodynamics(constants = ThermodynamicConstants(Float64)) =
    (constants, standard_pressure = 1e5, exner_exponent = VaporEquivalentExponent())

const manufactured_pdf = DoubleGaussianPDF(Float64)
const manufactured_saturation = ModelThermodynamicsSaturation()
const manufactured_clipping = HardClipping()
const manufactured_shoc_constants = SHOCConstants(Float64)
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

function saturation_at_mean(level::LevelMoments, thermo)
    c = manufactured_shoc_constants
    Tˡ = plume_liquid_water_temperature(manufactured_saturation, thermo, c, level.θ̄, 0.0, level.p)
    qᵛ⁺ = first(plume_saturation(manufactured_saturation, thermo, c, Tˡ, 0.0, level.p))
    # one fixed-point pass for the qᵗ dependence of Π and of qᵛ⁺
    q̄ = qᵛ⁺ + level.δq
    Tˡ = plume_liquid_water_temperature(manufactured_saturation, thermo, c, level.θ̄, q̄, level.p)
    qᵛ⁺ = first(plume_saturation(manufactured_saturation, thermo, c, Tˡ, q̄, level.p))
    return qᵛ⁺
end

mean_total_water(level::LevelMoments, thermo) = saturation_at_mean(level, thermo) + level.δq

# Within-plume variances (over the variance) of a standardized scalar with plume means φ̂₁, φ̂₂,
# from  a(φ̂₁² + v₁) + (1-a)(φ̂₂² + v₂) = 1  and  a(φ̂₁³ + 3φ̂₁v₁) + (1-a)(φ̂₂³ + 3φ̂₂v₂) = Sk.
function within_plume_variances(a, φ̂₁, φ̂₂, Sk)
    b₁ = 1 - a * φ̂₁^2 - (1 - a) * φ̂₂^2
    b₂ = Sk - a * φ̂₁^3 - (1 - a) * φ̂₂^3
    # [a  1-a; 3aφ̂₁  3(1-a)φ̂₂] [v₁; v₂] = [b₁; b₂]
    det = a * 3(1 - a) * φ̂₂ - (1 - a) * 3a * φ̂₁
    v₁ = (3(1 - a) * φ̂₂ * b₁ - (1 - a) * b₂) / det
    v₂ = (a * b₂ - 3a * φ̂₁ * b₁) / det
    return v₁, v₂
end

"""
    DoubleGaussianLevel(level, thermo)

The double Gaussian of `level`: plume weights `a`, plume means and within-plume standard
deviations of w, θˡⁱ and qᵗ, and the within-plume θ-q correlation `r`.
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

function DoubleGaussianLevel(m::LevelMoments, thermo; pdf = manufactured_pdf)
    ℂᵍ₁ = pdf.ℂᵍ₁
    Sk = m.Sk_w
    a = (1 - Sk / sqrt(4 * (1 - ℂᵍ₁)^3 + Sk^2)) / 2
    ŵ₁ = sqrt((1 - a) / a) * sqrt(1 - ℂᵍ₁)
    ŵ₂ = -sqrt(a / (1 - a)) * sqrt(1 - ℂᵍ₁)
    σw = sqrt(m.w′²)

    θ̂ = (-m.r_wθ / ŵ₂, -m.r_wθ / ŵ₁)
    q̂ = (-m.r_wq / ŵ₂, -m.r_wq / ŵ₁)
    vθ = within_plume_variances(a, θ̂..., 0.0)

    separation = abs(q̂[2] - q̂[1])
    ramp = clamp((separation - pdf.ℂᵍ₃) / (pdf.ℂᵍ₄ - pdf.ℂᵍ₃), 0, 1)
    Sk_q = pdf.ℂᵍ₂ * Sk * ramp
    vq = within_plume_variances(a, q̂..., Sk_q)
    all(v -> v > 0, (vθ..., vq...)) || error("level $(m.name): a within-plume variance is not positive: $vθ, $vq")

    q̄ = mean_total_water(m, thermo)
    σθ = m.σθ .* sqrt.(vθ)
    σq = m.σq .* sqrt.(vq)
    between = a * θ̂[1] * q̂[1] + (1 - a) * θ̂[2] * q̂[2]
    r = (m.r_θq - between) / (a * sqrt(vθ[1] * vq[1]) + (1 - a) * sqrt(vθ[2] * vq[2]))
    abs(r) < 1 || error("level $(m.name): within-plume correlation $r is not realizable")

    return DoubleGaussianLevel(m, m.p, a, m.w̄ .+ σw .* (ŵ₁, ŵ₂), m.θ̄ .+ m.σθ .* θ̂, q̄ .+ m.σq .* q̂,
                               sqrt(ℂᵍ₁) * σw, σθ, σq, r, q̄)
end

#####
##### Fine columns
#####

"""
    sample_columns(dg, L; seed)

`L × L` fine columns of the level `dg`: matrices of w, θˡⁱ, qᵗ and the plume label (1 or 2),
each column an independent draw.
"""
function sample_columns(dg::DoubleGaussianLevel, L; seed = 1)
    g = SplitMix64(UInt64(seed))
    w = zeros(L, L); θ = zeros(L, L); q = zeros(L, L); plume = zeros(Int8, L, L)
    for j in 1:L, i in 1:L
        n = uniform!(g) < dg.a ? 1 : 2
        z₁, z₂ = normal_pair!(g)
        z₃, _ = normal_pair!(g)
        plume[i, j] = n
        w[i, j] = dg.w[n] + dg.σw * z₁
        θ[i, j] = dg.θ[n] + dg.σθ[n] * z₂
        q[i, j] = dg.q[n] + dg.σq[n] * (dg.r * z₂ + sqrt(1 - dg.r^2) * z₃)
    end
    return (; w, θ, q, plume)
end

"""
    column_thermodynamics(θ, q, p, thermo)

The thermodynamics of one fine column, with the model's functions: the equilibrium liquid `qˡ`
and temperature `T` of saturation adjustment, the specific volume `α = Rᵐ T / p`, the vapor
excess over liquid `δqᵛ` that P3 sees when the column holds no condensate
(`qᵗ - qᵛ⁺ˡ(T = Π θˡⁱ, ρ)`), and the signed saturation excess `s`: `qˡ` where saturated and
`(qᵗ - qᵛ⁺(Π θˡⁱ)) / (1 + β qᵛ⁺)` where not, the quantity whose flux the closure's aʷ divides by.
"""
function column_thermodynamics(θ, q, p, thermo)
    constants = thermo.constants
    pˢᵗ = thermo.standard_pressure
    𝒰₀ = LiquidIcePotentialTemperatureState(θ, MoistureMassFractions(q), pˢᵗ, p)
    𝒰 = adjust_thermodynamic_state(𝒰₀, warm_adjustment, constants)
    T = temperature(𝒰, constants)
    qm = 𝒰.moisture_mass_fractions
    qˡ = qm.liquid
    α = mixture_gas_constant(qm, constants) * T / p

    Tˡ = temperature(𝒰₀, constants)
    ρˡ = p / (mixture_gas_constant(MoistureMassFractions(q), constants) * Tˡ)
    δqᵛ = q - saturation_specific_humidity(Tˡ, ρˡ, constants, PlanarLiquidSurface())

    qᵛ⁺, β = plume_saturation(manufactured_saturation, thermo, manufactured_shoc_constants, Tˡ, q, p)
    s = qˡ > 0 ? qˡ : (q - qᵛ⁺) / (1 + β * qᵛ⁺)
    return (; qˡ, T, α, δqᵛ, s)
end

function column_fields(columns, p, thermo)
    L = size(columns.θ, 1)
    out = [column_thermodynamics(columns.θ[i, j], columns.q[i, j], p, thermo) for i in 1:L, j in 1:L]
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
    qˡ, α, δqᵛ, s, plume = block(f.qˡ, n, I, J), block(f.α, n, I, J), block(f.δqᵛ, n, I, J), block(f.s, n, I, J),
                           block(f.plume, n, I, J)
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
    in₁ = plume .== 1
    n₁ = count(in₁)
    plume_liquid = (n₁ > 0 ? sum(qˡ[in₁]) / n₁ : NaN, n₁ < length(qˡ) ? sum(qˡ[.!in₁]) / (length(qˡ) - n₁) : NaN)
    truth = (; C = count(>(0), qˡ) / length(qˡ), qˡ = q̄ˡ, w′qˡ′, w′b′, w′s′, aʷ = w′qˡ′ / w′s′,
               excess = mean_of(max.(0, δqᵛ)), f⁺ = count(>(0), δqᵛ) / length(δqᵛ),
               plume_fraction = n₁ / length(qˡ), plume_liquid)
    return (; moments, truth)
end

#####
##### The analytic diagnostics, from the moments of a cell
#####

"""
    closure_plumes(moments, p, thermo)

The double Gaussian that the closure reconstructs from the moments of a cell, per plume, with the
closure's functions: the plume weight from Sk_w (as `assumed_pdf` computes it), the scalar plume
parameters (`scalar_plume_parameters`), the within-plume correlation, and each plume's saturation
(`plume_saturation`) and equilibrium condensate (`plume_condensate`).
"""
function closure_plumes(m, p, thermo; pdf = manufactured_pdf)
    cp, c, sat = manufactured_clipping, manufactured_shoc_constants, manufactured_saturation
    sqrtw2 = sqrt(m.w′²)
    Sk = m.w′³ / m.w′²^(3/2)
    ℂᵍ₁ = pdf.ℂᵍ₁
    a = clamp((1 - Sk / sqrt(4 * (1 - ℂᵍ₁)^3 + Sk^2)) / 2, pdf.plume_fraction_min, pdf.plume_fraction_max)
    ŵ₁ = sqrt((1 - a) / a) * sqrt(1 - ℂᵍ₁)
    ŵ₂ = -sqrt(a / (1 - a)) * sqrt(1 - ℂᵍ₁)
    sθ, sq = max(pdf.thl_tol, sqrt(m.θ′²)), max(pdf.rt_tol, sqrt(m.q′²))
    θ₁, θ₂, vθ₁, vθ₂, σθ₁, σθ₂ = scalar_plume_parameters(cp, m.w′θ′, sqrtw2, sθ, m.θ′², pdf.thl_tol, m.θ̄, ŵ₁, ŵ₂, Sk, a, false, pdf)
    q₁, q₂, vq₁, vq₂, σq₁, σq₂ = scalar_plume_parameters(cp, m.w′q′, sqrtw2, sq, m.q′², pdf.rt_tol, m.q̄, ŵ₁, ŵ₂, Sk, a, true, pdf)
    denominator = a * σq₁ * σθ₁ + (1 - a) * σq₂ * σθ₂
    r = clamp((m.θ′q′ - a * (q₁ - m.q̄) * (θ₁ - m.θ̄) - (1 - a) * (q₂ - m.q̄) * (θ₂ - m.θ̄)) / denominator, -1, 1)

    plume(θᵢ, qᵢ, vθ, vq, σθ, σq) = begin
        Tˡ = plume_liquid_water_temperature(sat, thermo, c, θᵢ, qᵢ, p)
        qᵛ⁺, β, cp_over_L, Π = plume_saturation(sat, thermo, c, Tˡ, qᵢ, p)
        s, σs, Q, C = plume_condensate(qᵢ, qᵛ⁺, β, cp_over_L, Π, vθ, vq, σθ, σq, r)
        dqᵛ⁺dθ = cp_over_L * β * qᵛ⁺ * Π
        # The vapor excess over liquid without condensate, as P3 defines its saturation, and its spread
        ρ = p / (mixture_gas_constant(MoistureMassFractions(qᵢ), thermo.constants) * Tˡ)
        μ = qᵢ - saturation_specific_humidity(Tˡ, ρ, thermo.constants, PlanarLiquidSurface())
        σ = sqrt(max(0, vq + dqᵛ⁺dθ^2 * vθ - 2 * dqᵛ⁺dθ * r * σθ * σq))
        (; Tˡ, qᵛ⁺, β, s, σs, Q, C, μ, σ)
    end
    P₁ = plume(θ₁, q₁, vθ₁, vq₁, σθ₁, σq₁)
    P₂ = plume(θ₂, q₂, vθ₂, vq₂, σθ₂, σq₂)
    w = (m.w̄ + ŵ₁ * sqrtw2, m.w̄ + ŵ₂ * sqrtw2)
    return (; a, w, θ = (θ₁, θ₂), q = (q₁, q₂), σθ = (σθ₁, σθ₂), σq = (σq₁, σq₂), r, plumes = (P₁, P₂))
end

"""
    vapor_excess_statistics(plumes)

The statistics of the vapor excess over liquid that P3's `SubgridCondensation` reads. Until the
closure-side `TurbulenceClosureStatistics` lands, they are formed here from the closure's plumes
and passed through `PrescribedPlumeStatistics`; switch this one function when it does.
"""
vapor_excess_statistics(cp) =
    PrescribedPlumeStatistics((cp.a, 1 - cp.a), (cp.plumes[1].μ, cp.plumes[2].μ), (cp.plumes[1].σ, cp.plumes[2].σ))

standard_normal_cdf(x) = (1 + erf(x / sqrt(2))) / 2

"""
    analytic_diagnostics(moments, p, thermo)

The diagnostics of the closure and of P3 from the moments of a cell:

* `C`, `qˡ`, `w′qˡ′`, `w′θᵥ′`: `assumed_pdf` with the model's saturation;
* `w′b′ = (g / θ̄) w′θᵥ′`, the closure's buoyancy flux, and `w′b′ˡⁱⁿ`, the same linear coefficients
  applied to a given `w′qˡ′` (the sampled one), which isolates the linearization of b;
* `aʷ = w′qˡ′ / w′s′`, with `w′s′` from the linearized saturation excess at the cell mean;
* `Γ`, `excess = μ̄ + Γ` and `f⁺` of P3's `SubgridCondensation` about the grid-mean vapor excess `μ̄`;
* `Q`, the plumes' equilibrium condensate, for the split rule.
"""
function analytic_diagnostics(m, p, thermo; w′qˡ′_given = nothing, pdf = manufactured_pdf)
    cp, c, sat = manufactured_clipping, manufactured_shoc_constants, manufactured_saturation
    constants = thermo.constants
    g = constants.gravitational_acceleration
    moments = (m.w′³, m.θ′², m.w′θ′, m.θ′q′, m.w′q′, m.q′²)
    C, qˡ, _, w′qˡ′, w′θᵥ′ = assumed_pdf(pdf, sat, thermo, cp, c, 1.0, 0.0, 0.5, p, m.θ̄, m.w̄, m.q̄, m.w′², moments, moments)

    A_q, A_c = buoyancy_flux_coefficients(sat, thermo, c, p, m.θ̄, m.q̄, qˡ)
    w′qˡ′ˢ = something(w′qˡ′_given, w′qˡ′)
    w′b′ = g / m.θ̄ * w′θᵥ′
    w′b′ˡⁱⁿ = g / m.θ̄ * (m.w′θ′ + A_q * m.w′q′ + A_c * w′qˡ′ˢ)

    # The linearized saturation excess at the cell mean: s = c_q qᵗ′ - c_θ θˡⁱ′
    Tˡ = plume_liquid_water_temperature(sat, thermo, c, m.θ̄, m.q̄, p)
    qᵛ⁺, β, cp_over_L, Π = plume_saturation(sat, thermo, c, Tˡ, m.q̄, p)
    c_θ = ((1 + β * m.q̄) / (1 + β * qᵛ⁺)^2) * cp_over_L * β * qᵛ⁺ * Π
    c_q = 1 / (1 + β * qᵛ⁺)
    w′s′ = c_q * m.w′q′ - c_θ * m.w′θ′

    plumes = closure_plumes(m, p, thermo; pdf)
    statistics = supersaturation_statistics(vapor_excess_statistics(plumes), 1, 1, 1, nothing, nothing, nothing, nothing)
    ρ̄ = p / (mixture_gas_constant(MoistureMassFractions(m.q̄), constants) * Tˡ)
    μ̄ = m.q̄ - saturation_specific_humidity(Tˡ, ρ̄, constants, PlanarLiquidSurface())
    Γ = subgrid_vapor_excess_correlation(statistics, μ̄)
    shift = μ̄ - (statistics.a[1] * statistics.μ[1] + statistics.a[2] * statistics.μ[2])
    f⁺ = sum(statistics.a[i] * standard_normal_cdf((statistics.μ[i] + shift) / statistics.σ[i]) for i in 1:2)

    return (; C, qˡ, w′qˡ′, w′θᵥ′, w′b′, w′b′ˡⁱⁿ, w′s′, aʷ = w′qˡ′ / w′s′, Γ, μ̄, excess = μ̄ + Γ, f⁺,
              Q = (plumes.plumes[1].Q, plumes.plumes[2].Q), plumes)
end

#####
##### The plume condensate split (notes/les-closure-tests.md §D)
#####

"""
    plume_condensate_split(q̄ᶜˡ, a, weights, qᵗ)

Split the grid-mean prognostic condensate `q̄ᶜˡ` between the plumes in proportion to `weights`
(the plumes' equilibrium condensate `Qᵢ`, or their supersaturated fractions `fᵢ`):
`qᶜˡᵢ = q̄ᶜˡ wᵢ / Σⱼ aⱼ wⱼ`. Where a plume would hold more condensate than its total water
`qᵗᵢ`, the excess goes to the other plume, which keeps the grid mean. Where `Σ a w = 0` the
condensate goes to the plume of larger `qᵗᵢ`. Not in src: this is the reference implementation
the tests check.
"""
function plume_condensate_split(q̄ᶜˡ, a, weights, qᵗ)
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
    # Shallow cumulus: subsaturated mean, a narrow moist updraft plume that reaches saturation
    LevelMoments(name = "cumulus", δq = -6e-4, w′² = 0.6, Sk_w = 0.8, σθ = 0.25, σq = 6e-4,
                 r_wθ = -0.2, r_wq = 0.5, r_θq = -0.5),
    # Stratocumulus: mean near saturation, negatively skewed (narrow downdrafts)
    LevelMoments(name = "stratocumulus", δq = 1e-4, w′² = 0.4, Sk_w = -0.5, σθ = 0.2, σq = 4e-4,
                 r_wθ = -0.3, r_wq = 0.4, r_θq = -0.6),
    # Overcast: saturated mean, the dry tail of the downdraft plume clears
    LevelMoments(name = "overcast", δq = 8e-4, w′² = 0.3, Sk_w = 0.2, σθ = 0.2, σq = 5e-4,
                 r_wθ = -0.1, r_wq = 0.3, r_θq = -0.3),
    # Nearly clear: only the far tail of the updraft plume saturates
    LevelMoments(name = "clear", δq = -1.4e-3, w′² = 0.5, Sk_w = 0.6, σθ = 0.3, σq = 5e-4,
                 r_wθ = -0.1, r_wq = 0.5, r_θq = -0.4),
)
