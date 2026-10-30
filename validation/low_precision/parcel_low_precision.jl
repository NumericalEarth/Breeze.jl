# Low-precision storage of droplet radius in a closed adiabatic parcel (Jensen and Nugent 2017 Sc case,
# modified polluted aerosol, w = 0.4 m/s ascent). Growth is computed in Float64 with Breeze's implicit
# `condensational_growth_step`; only the stored radius is rounded after each step. The vapor/heat source uses
# the difference of *stored* droplet masses, so total water is conserved exactly and any stagnation of the
# stored radius feeds back on the supersaturation.
#
#   julia --project=<env with Breeze> validation/low_precision/parcel_low_precision.jl

include(joinpath(@__DIR__, "..", "parcel_benchmarks", "parcel_benchmarks.jl"))
include(joinpath(@__DIR__, "quantize.jl"))
using Random

# Reference BFloat16 round-to-nearest-even via Float32 bit manipulation (independent of the emulator)
function bfloat16_rn(x::Float32)
    b = reinterpret(UInt32, x)
    b += 0x00007fff + ((b >> 16) & 0x00000001)
    return Float64(reinterpret(Float32, b & 0xffff0000))
end

# Storage transforms (forward: radius [m] → stored quantity; inverse)
const transforms = (
    r_μm = (r -> 1e6r, s -> 1e-6s),
    r²_μm² = (r -> 1e12r^2, s -> 1e-6sqrt(max(s, 0))),
    lnr = (r -> log(1e6r), s -> 1e-6exp(s)),
)

struct Storage
    label :: String
    format :: Union{Nothing, StorageFormat}
    stochastic :: Bool
    transform :: Symbol
end

store(r, s::Storage, rng) = s.format === nothing ? r :
    (transforms[s.transform][2])(quantize((transforms[s.transform][1])(r), s.format, s.stochastic, rand(rng)))

const storages = [
    Storage("Float64 (reference)", nothing, false, :r_μm),
    Storage("Float32 RN, r", Float32Format, false, :r_μm),
    Storage("BFloat16 RN, r", BFloat16Format, false, :r_μm),
    Storage("BFloat16 SR, r", BFloat16Format, true, :r_μm),
    Storage("BFloat16 SR, r²", BFloat16Format, true, :r²_μm²),
    Storage("BFloat16 RN, ln r", BFloat16Format, false, :lnr),
    Storage("BFloat16 SR, ln r", BFloat16Format, true, :lnr),
    Storage("Float16 RN, r [μm]", Float16Format, false, :r_μm),
    Storage("Float16 SR, r [μm]", Float16Format, true, :r_μm),
    Storage("Float16 SR, r² [μm²]", Float16Format, true, :r²_μm²),
    Storage("FP8-E4M3 SR, r [μm]", E4M3Format, true, :r_μm),
    Storage("FP8-E5M2 SR, r [μm]", E5M2Format, true, :r_μm),
]

function run_parcel_stored(storage; Nsd, Δt, seed = 1, t_end = 1500.0)
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

    parcel = Parcel(T₀, p₀, qᵛ₀, 0.0, 600.0)
    𝒮₀ = parcel_supersaturation(parcel, constants)
    r = [store(equilibrium_radius(rᵈ[i], κ, 𝒮₀, T₀, constants, parameters), storage, rng) for i in 1:Nsd]
    parcel.qˡ = sum(n[i] * droplet_water_mass(r[i], rᵈ[i], constants) for i in 1:Nsd)

    heights = 50:50:300
    out = Dict{Int, NamedTuple}()
    𝒮max = -Inf
    Nt = round(Int, t_end / Δt)
    for step in 1:Nt
        q = MoistureMassFractions(parcel.qᵛ, parcel.qˡ)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        ρ = density(parcel, constants)
        parcel.p -= ρ * g * w * Δt
        parcel.T -= g * w * Δt / cᵖᵐ
        parcel.z += w * Δt
        𝒮 = parcel_supersaturation(parcel, constants)
        𝒮max = max(𝒮max, 𝒮)
        Δqˡ = 0.0
        for i in 1:Nsd
            growth = condensational_growth_step(r[i], rᵈ[i], κ, 𝒮, 1.0, parcel.T, parcel.p, Δt, constants, parameters)
            rⁿ = max(store(growth.radius, storage, rng), rᵈ[i])
            Δqˡ += n[i] * (droplet_water_mass(rⁿ, rᵈ[i], constants) - droplet_water_mass(r[i], rᵈ[i], constants))
            r[i] = rⁿ
        end
        q = MoistureMassFractions(parcel.qᵛ, parcel.qˡ)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        parcel.qᵛ -= Δqˡ
        parcel.qˡ += Δqˡ
        parcel.T += liquid_latent_heat(parcel.T, constants) * Δqˡ / cᵖᵐ

        h = round(Int, parcel.z - 900)
        if h in heights && abs(parcel.z - 900 - h) < w * Δt / 2 && !haskey(out, h)
            rᶜ = [critical_radius(rᵈ[i], κ, parcel.T, constants, parameters) for i in 1:Nsd]
            active = r .> rᶜ
            N = sum(n[active])
            r̄ = sum(n[active] .* r[active]) / N
            σ = sqrt(sum(n[active] .* (r[active] .- r̄) .^ 2) / N)
            out[h] = (qˡ = parcel.qˡ, N = N, r̄ = r̄, σ = σ, 𝒮 = 𝒮)
        end
    end
    return (; out, 𝒮max)
end

relerr(a, b) = 100 * (a - b) / b

function main_low_precision(cases = ((100, 1.0), (1000, 1.0), (10000, 1.0), (1000, 0.1), (100, 0.01)))
    # sanity check of the emulator against real BFloat16 / Float16 round-to-nearest
    xs = Float64.(rand(Xoshiro(0), Float32, 10^5) .* 100)
    @info "emulator mismatch vs BFloat16: $(count(x -> quantize(x, BFloat16Format, false, 0.0) != bfloat16_rn(Float32(x)), xs))/$(length(xs))"
    @info "emulator mismatch vs Float16: $(count(x -> quantize(x, Float16Format, false, 0.0) != Float64(Float16(x)), xs))/$(length(xs))"
    # SR is unbiased
    x = 1 + 1e-4
    m = mean(quantize(x, BFloat16Format, true, rand()) for _ in 1:10^6)
    @info "SR mean of BFloat16(1+1e-4) = $(m) (RN gives $(quantize(x, BFloat16Format, false, 0.0)))"

    io = stdout
    println(io, "| Nsd | Δt [s] | storage | ΔLWC@300m [%] | ΔN_act@300m [%] | Δr̄@300m [%] | Δσ@300m [%] | Δσ@50m [%] | Δ𝒮max [%] | σ/r̄@300m |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    for (Nsd, Δt) in cases
        ref = nothing
        for s in storages
            seeds = s.stochastic ? (1:3) : (1:1)
            for seed in seeds
                t₀ = time_ns()
                res = run_parcel_stored(s; Nsd, Δt, seed)
                ref === nothing && (ref = res)
                a, b = res.out[300], ref.out[300]
                @printf(io, "| %d | %g | %s%s | %+.3g | %+.3g | %+.3g | %+.3g | %+.3g | %+.3g | %.4f |\n", Nsd, Δt, s.label,
                        length(seeds) > 1 ? " (seed $seed)" : "",
                        relerr(a.qˡ, b.qˡ), relerr(a.N, b.N), relerr(a.r̄, b.r̄), relerr(a.σ, b.σ),
                        relerr(res.out[50].σ, ref.out[50].σ), relerr(res.𝒮max, ref.𝒮max), a.σ / a.r̄)
                flush(io)
            end
        end
    end
end

# Usage: julia parcel_low_precision.jl [Nsd Δt ...]
const cli_cases = length(ARGS) ≥ 2 ? Tuple((parse(Int, ARGS[i]), parse(Float64, ARGS[i+1])) for i in 1:2:length(ARGS)) : nothing
cli_cases === nothing ? main_low_precision() : main_low_precision(cli_cases)
