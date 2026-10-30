# Methods to beat the spectral-width (σ) inflation caused by stochastic rounding (SR) of low-precision droplet
# radius storage. Same closed adiabatic parcel as `parcel_low_precision.jl` (Jensen and Nugent 2017 Sc case,
# modified polluted aerosol, w = 0.4 m/s), Float64 compute with Breeze's implicit `condensational_growth_step`,
# only the per-particle stored state is rounded. Methods:
#
#   plain        s = T(r) stored in one low-precision word (RN or SR); T ∈ r, r², r³, ln r, r^(2/3)
#   compensated  s ≈ hi + lo: hi = RN(s) in the value format, lo = SR(s - hi) in a residual format
#                (BF16+BF16 "double-BF16", BF16+uint8, FP16+uint8, FP16+FP32)
#   deviation    store δ = r² - r_ref²(t), with r_ref a single shared Float64 "ghost" droplet (largest dry
#                radius) grown alongside; particles with r² < r_ref²/4 store r² itself (1 flag bit)
#   window       Float32 storage until the supersaturation has peaked and decayed to 𝒮max/2, then low precision
#   interval k   growth substepped k times in Float64 "registers" (frozen 𝒮) and stored once every k steps;
#                compared with the Float64 run using the same k
#
# For every SR store we also accumulate the exact conditional SR variance ulp² f (1 - f) (f = fractional
# position between grid points) and propagate it linearly to r at the output time (see COMPENSATED_STORAGE.md):
# a perturbation δs made at time t when the radius is rₜ becomes δr = δs rₜ / (r s'(rₜ)) at a later time when the
# radius is r, because r δr is conserved by r dr/dt = G 𝒮. Predicted σ² = σ_ref² + ⟨Var r⟩.
#
#   julia --project=<env> validation/low_precision/compensated_storage.jl <experiment> [output.csv]
#   experiment ∈ (methods, interval, quick)

include(joinpath(@__DIR__, "..", "parcel_benchmarks", "parcel_benchmarks.jl"))
include(joinpath(@__DIR__, "quantize.jl"))
using Random
using Printf
using Statistics

@inline ulp_of(x, f::StorageFormat) = exp2(max(Int((reinterpret(UInt64, Float64(x)) >> 52) & 0x7ff) - 1023, f.emin) - (f.p - 1))

# Stochastic or nearest rounding that also returns the conditional SR variance ulp² f(1-f)
@inline function round_with_variance(x, f::StorageFormat, stochastic, u)
    ulp = ulp_of(x, f)
    y = x / ulp
    fraction = y - floor(y)
    q = stochastic ? floor(y + u) : round(y)
    return q * ulp, ifelse(stochastic, ulp^2 * fraction * (1 - fraction), 0.0)
end

# Unsigned 8-bit fixed-point residual in units of ulp(hi)/256, with hi = x rounded toward -∞ so that
# 0 ≤ x - hi < ulp. A rounded-up residual of 256 is a carry into hi (hi + ulp is representable).
# (A signed residual of an RN-rounded hi saturates at ±128 units and stalls growth when x sits just below the
# midpoint between two hi values; an earlier version did this and stalled at Δt = 0.01 s.)
@inline function round_uint8_residual(x, ulp_hi, u)
    unit = ulp_hi / 256
    y = x / unit
    fraction = y - floor(y)
    q = floor(y + u)
    return q * unit, unit^2 * fraction * (1 - fraction)
end

# Transforms: forward s = T(r [μm]), inverse, derivative s'(r)
const transforms = (
    r = (r -> r, s -> s, r -> 1.0),
    r² = (r -> r^2, s -> sqrt(max(s, 0)), r -> 2r),
    r³ = (r -> r^3, s -> cbrt(max(s, 0)), r -> 3r^2),
    lnr = (r -> log(r), s -> exp(s), r -> 1 / r),
    r⅔ = (r -> r^(2/3), s -> max(s, 0)^(3/2), r -> (2/3) * r^(-1/3)),
    r²_100 = (r -> r^2 / 100, s -> sqrt(max(100s, 0)), r -> r / 50), # r² in units of (10 μm)²
)

const Int8Residual = StorageFormat(0, 0, 0.0) # sentinel

struct Method
    label :: String
    kind :: Symbol             # :float64, :plain, :compensated, :deviation, :window
    format :: StorageFormat    # value format
    residual :: StorageFormat  # residual format (compensated only)
    stochastic :: Bool
    transform :: Symbol
    bytes :: Float64
end

Method(label, kind, format, stochastic, transform, bytes) = Method(label, kind, format, Float32Format, stochastic, transform, bytes)

const reference_method = Method("Float64", :float64, Float32Format, false, :r, 8)

const methods = [
    reference_method,
    Method("Float32 RN r", :plain, Float32Format, false, :r, 4),
    Method("Float32 RN r²", :plain, Float32Format, false, :r², 4),
    Method("Float32 SR r", :plain, Float32Format, true, :r, 4),
    Method("BF16 SR r", :plain, BFloat16Format, true, :r, 2),
    Method("BF16 SR r²", :plain, BFloat16Format, true, :r², 2),
    Method("BF16 SR r³", :plain, BFloat16Format, true, :r³, 2),
    Method("BF16 SR ln r", :plain, BFloat16Format, true, :lnr, 2),
    Method("BF16 SR r^(2/3)", :plain, BFloat16Format, true, :r⅔, 2),
    Method("FP16 SR r", :plain, Float16Format, true, :r, 2),
    Method("FP16 SR r²", :plain, Float16Format, true, :r², 2),
    Method("FP16 SR r² [(10μm)²]", :plain, Float16Format, true, :r²_100, 2),
    Method("FP16 SR r³", :plain, Float16Format, true, :r³, 2),
    Method("FP16 SR ln r", :plain, Float16Format, true, :lnr, 2),
    Method("BF16+BF16 r²", :compensated, BFloat16Format, BFloat16Format, true, :r², 4),
    Method("BF16+uint8 r²", :compensated, BFloat16Format, Int8Residual, true, :r², 3),
    Method("FP16+uint8 r²", :compensated, Float16Format, Int8Residual, true, :r², 3),
    Method("BF16+BF16 r", :compensated, BFloat16Format, BFloat16Format, true, :r, 4),
    Method("FP16+FP32 r", :compensated, Float16Format, Float32Format, false, :r, 6),
    Method("BF16 SR δr² (ghost ref)", :deviation, BFloat16Format, true, :r², 2.125),
    Method("FP16 SR δr² (ghost ref)", :deviation, Float16Format, true, :r², 2.125),
    Method("BF16 SR δr² (mean ref)", :deviation_mean, BFloat16Format, true, :r², 2.125),
    Method("FP16 SR δr² (mean ref)", :deviation_mean, Float16Format, true, :r², 2.125),
    Method("BF16 SR r² + FP32 window", :window, BFloat16Format, true, :r², 2),
    Method("FP16 SR r² + FP32 window", :window, Float16Format, true, :r², 2),
]

# Store radius r [μm]; returns (stored radius [μm], SR variance in s, s'(r_stored)); `reference` = r_ref² [μm²]
@inline function store(r, m::Method, rng, reference, use_float32)
    m.kind === :float64 && return r, 0.0
    m.kind === :deviation_mean && isinf(reference) && return store(r, Method(m.label, :plain, m.format, m.stochastic, :r², 2), rng, reference, false)
    forward, inverse, _ = transforms[m.transform]
    if m.kind === :plain || (m.kind === :window && use_float32)
        format = use_float32 ? Float32Format : m.format
        stochastic = use_float32 ? false : m.stochastic
        s, v = round_with_variance(forward(r), format, stochastic, rand(rng))
        return inverse(s), v
    elseif m.kind === :window
        s, v = round_with_variance(forward(r), m.format, m.stochastic, rand(rng))
        return inverse(s), v
    elseif m.kind === :compensated
        x = forward(r)
        if m.residual === Int8Residual
            ulp = ulp_of(x, m.format)
            hi = floor(x / ulp) * ulp
            lo, v = round_uint8_residual(x - hi, ulp, rand(rng))
        else
            hi, _ = round_with_variance(x, m.format, false, 0.0)
            lo, v = round_with_variance(x - hi, m.residual, m.stochastic, rand(rng))
        end
        return inverse(hi + lo), v
    else # deviation from the ghost-droplet reference, in r²
        s = r^2
        if s < reference / 4
            q, v = round_with_variance(s, m.format, m.stochastic, rand(rng))
            return sqrt(max(q, 0)), v
        else
            δ, v = round_with_variance(s - reference, m.format, m.stochastic, rand(rng))
            return sqrt(max(reference + δ, 0)), v
        end
    end
end

function run_parcel_method(m::Method; Nsd, Δt, seed = 1, k = 1, t_end = 1500.0)
    rng = Xoshiro(seed)
    constants = ThermodynamicConstants(Float64)
    parameters = droplet_parameters(; αᶜ = 0.036, αᵀ = 0.7)
    p₀, T₀, ℋ₀ = 93850.0, 284.3, 0.8561
    qᵛ₀ = specific_humidity_from_relative_humidity(ℋ₀, T₀, p₀, constants)
    pᵛ = ℋ₀ * saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface())
    ρᵈ₀ = (p₀ - pᵛ) / (dry_air_gas_constant(constants) * T₀)
    rᵈ, counts = lognormal_superdroplets(((48e6, 0.029e-6, 1.36), (114e6, 0.071e-6, 1.57)), Nsd)
    n = counts ./ ρᵈ₀ .* (1 - qᵛ₀)
    κ = 1.28
    w = 0.4
    g = constants.gravitational_acceleration
    μm = 1e-6

    parcel = Parcel(T₀, p₀, qᵛ₀, 0.0, 600.0)
    𝒮₀ = parcel_supersaturation(parcel, constants)
    rᵍ = equilibrium_radius(maximum(rᵈ), κ, 𝒮₀, T₀, constants, parameters) # ghost reference droplet [m]
    use_float32 = m.kind === :window
    r = zeros(Nsd)
    rnew = zeros(Nsd)
    reference = m.kind === :deviation_mean ? Inf : (rᵍ / μm)^2
    A = zeros(Nsd) # Σ Var(s) (r/s'(r))², propagated variance numerator
    for i in 1:Nsd
        rμ, v = store(equilibrium_radius(rᵈ[i], κ, 𝒮₀, T₀, constants, parameters) / μm, m, rng, reference, use_float32)
        r[i] = rμ * μm
    end
    parcel.qˡ = sum(n[i] * droplet_water_mass(r[i], rᵈ[i], constants) for i in 1:Nsd)

    heights = 50:50:300
    out = Dict{Int, NamedTuple}()
    𝒮max = -Inf
    passed_peak = false
    Nt = round(Int, t_end / Δt)
    stores = 0
    for step in 1:Nt
        q = MoistureMassFractions(parcel.qᵛ, parcel.qˡ)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        ρ = density(parcel, constants)
        parcel.p -= ρ * g * w * Δt
        parcel.T -= g * w * Δt / cᵖᵐ
        parcel.z += w * Δt
        𝒮 = parcel_supersaturation(parcel, constants)
        𝒮 < 𝒮max && (passed_peak = true)
        𝒮max = max(𝒮max, 𝒮)
        use_float32 = m.kind === :window && !(passed_peak && 𝒮 < 𝒮max / 2)
        if step % k == 0 # growth over k substeps, stored once
            for _ in 1:k
                rᵍ = condensational_growth_step(rᵍ, maximum(rᵈ), κ, 𝒮, 1.0, parcel.T, parcel.p, Δt, constants, parameters).radius
            end
            Δqˡ = 0.0
            for i in 1:Nsd
                rⁿ = r[i]
                for _ in 1:k
                    rⁿ = condensational_growth_step(rⁿ, rᵈ[i], κ, 𝒮, 1.0, parcel.T, parcel.p, Δt, constants, parameters).radius
                end
                rnew[i] = rⁿ
            end
            if m.kind === :deviation_mean # shared reference: n-weighted mean r² of the (previously) large droplets
                large = (rnew ./ μm) .^ 2 .≥ reference / 4
                reference = any(large) ? sum(n[large] .* (rnew[large] ./ μm) .^ 2) / sum(n[large]) : 0.0
            else
                reference = (rᵍ / μm)^2
            end
            for i in 1:Nsd
                rⁿ = rnew[i]
                rμ, v = store(rⁿ / μm, m, rng, reference, use_float32)
                s′ = m.kind in (:deviation, :deviation_mean) ? 2rμ : transforms[m.transform][3](rμ)
                A[i] += v * (rμ / s′)^2
                rⁿ = max(rμ * μm, rᵈ[i])
                Δqˡ += n[i] * (droplet_water_mass(rⁿ, rᵈ[i], constants) - droplet_water_mass(r[i], rᵈ[i], constants))
                r[i] = rⁿ
            end
            stores += 1
            q = MoistureMassFractions(parcel.qᵛ, parcel.qˡ)
            cᵖᵐ = mixture_heat_capacity(q, constants)
            parcel.qᵛ -= Δqˡ
            parcel.qˡ += Δqˡ
            parcel.T += liquid_latent_heat(parcel.T, constants) * Δqˡ / cᵖᵐ
        end

        h = round(Int, parcel.z - 900)
        if h in heights && abs(parcel.z - 900 - h) < w * Δt / 2 && !haskey(out, h)
            rᶜ = [critical_radius(rᵈ[i], κ, parcel.T, constants, parameters) for i in 1:Nsd]
            active = r .> rᶜ
            N = sum(n[active])
            r̄ = sum(n[active] .* r[active]) / N
            σ = sqrt(sum(n[active] .* (r[active] .- r̄) .^ 2) / N)
            noise = sum(n[active] .* A[active] ./ (r[active] ./ μm) .^ 2) / N * μm^2 # ⟨Var r⟩ [m²]
            out[h] = (qˡ = parcel.qˡ, N = N, r̄ = r̄, σ = σ, 𝒮 = 𝒮, noise = noise, stores = stores)
        end
    end
    return (; out, 𝒮max)
end

relerr(a, b) = 100 * (a - b) / b

const header = "experiment,method,kind,bytes,Nsd,dt,k,seed,stores,dLWC,dN,drbar,dsigma,dsigma50,dSmax,sigma_ref,sigma,sigma_pred,rbar_ref"

function emit(io, experiment, m, Nsd, Δt, k, seed, res, ref)
    a, b = res.out[300], ref.out[300]
    σpred = sqrt(b.σ^2 + a.noise)
    @printf(io, "%s,%s,%s,%g,%d,%g,%d,%d,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n",
            experiment, m.label, m.kind, m.bytes, Nsd, Δt, k, seed, a.stores,
            relerr(a.qˡ, b.qˡ), relerr(a.N, b.N), relerr(a.r̄, b.r̄), relerr(a.σ, b.σ),
            relerr(res.out[50].σ, ref.out[50].σ), relerr(res.𝒮max, ref.𝒮max), b.σ, a.σ, σpred, b.r̄)
    flush(io)
end

function run_cases(io, experiment, ms, cases; seeds = 1:3)
    for (Nsd, Δt, k) in cases
        ref = run_parcel_method(reference_method; Nsd, Δt, k)
        jobs = [(m, s) for m in ms for s in (m.stochastic ? seeds : (1:1))]
        results = Vector{Any}(undef, length(jobs))
        Threads.@threads for j in eachindex(jobs)
            results[j] = run_parcel_method(jobs[j][1]; Nsd, Δt, k, seed = jobs[j][2])
        end
        for j in eachindex(jobs)
            emit(io, experiment, jobs[j][1], Nsd, Δt, k, jobs[j][2], results[j], ref)
        end
        @info "done" experiment Nsd Δt k
    end
end

function main_compensated(experiment, path)
    io = open(path, "w")
    println(io, header)
    lowprec = methods[2:end]
    if experiment == "quick"
        run_cases(io, "methods", lowprec, ((100, 1.0, 1),); seeds = 1:1)
    elseif experiment == "methods"
        run_cases(io, "methods", lowprec, ((1000, 1.0, 1), (1000, 0.1, 1), (1000, 0.01, 1), (100, 0.1, 1), (10000, 0.1, 1)))
    elseif experiment == "uint8"
        ms = [m for m in methods if occursin("uint8", m.label)]
        run_cases(io, "methods", ms, ((1000, 1.0, 1), (1000, 0.1, 1), (1000, 0.01, 1), (100, 0.1, 1), (10000, 0.1, 1)))
    elseif experiment == "interval"
        ms = [m for m in methods if m.label in ("BF16 SR r", "BF16 SR r²", "FP16 SR r²", "BF16 SR r³", "FP16 SR r")]
        run_cases(io, "interval", ms, Tuple((1000, 0.01, k) for k in (1, 3, 10, 30, 100, 300)))
        run_cases(io, "interval_dt", ms, Tuple((1000, Δt, 1) for Δt in (0.003, 0.03, 0.3, 3.0)))
    end
    close(io)
    @info "wrote $path"
end

main_compensated(get(ARGS, 1, "quick"), get(ARGS, 2, joinpath(@__DIR__, "output", "compensated_$(get(ARGS, 1, "quick")).csv")))
