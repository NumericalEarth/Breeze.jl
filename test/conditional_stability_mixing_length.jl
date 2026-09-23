include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.TurbulenceClosures: TKE_NAME, saturated_fraction, normal_cdf, saturation_excessᶜᶜᶜ,
                                saturation_excess_lapse_rateᶜᶜᶠ, conditional_static_stabilityᶜᶜᶠ,
                                saturated_static_stabilityᶜᶜᶠ, mixing_length_summary
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units
using Test

column(field) = Array(interior(field, 1, 1, :))

function set_tke!(model, e₀)
    ρ = model.dynamics.reference_state.density
    set!(model.tracers[TKE_NAME], e₀)
    parent(model.tracers[TKE_NAME]) .*= parent(ρ)
    update_state!(model)
    return nothing
end

# The envelope written out in full, as the independent reference: at every face the minimum over
# every face z′ of the bound there — the wall length or the penetration depth, whichever is smaller —
# plus Cˢ times the distance. The ground, the bottom face, is a zero.
function brute_force_envelope(zf, ℓᵇ, Cˢ)
    obstacles = min.(Cˢ .* zf, ℓᵇ)
    obstacles[1] = 0
    return [minimum(obstacles[m] + Cˢ * abs(zf[k] - zf[m]) for m in eachindex(zf)) for k in eachindex(zf)]
end

#####
##### Construction and display
#####

@testset "ConditionalStabilityMixingLength construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    ml = ConditionalStabilityMixingLength()
    @test ml isa ConditionalStabilityMixingLength
    @test isbits(ml) # a GPU kernel can only carry an isbits closure
    @test ml.mixing_length isa GradientLimitedMixingLength

    # Every coefficient is off by default: the wrapper starts as the model it wraps
    @test ml.C𝒟ᵍ == 0
    @test ml.C𝒟⁰ == 0
    @test ml.Cʰ == 0
    @test ml.Cᶜᵒⁿᵈ == 0

    # Four new coefficients, and no length coefficient of its own — those are the wrapped model's
    @test length(fieldnames(ConditionalStabilityMixingLength)) == 5
    @test :Cˢ ∉ fieldnames(ConditionalStabilityMixingLength)
    @test hasproperty(ml.mixing_length, :Cˢ)

    wrapped = GradientLimitedMixingLength(Cˢ = 1.2)
    ml = ConditionalStabilityMixingLength(wrapped; C𝒟ᵍ = 1, C𝒟⁰ = 1e-4, Cʰ = 1, Cᶜᵒⁿᵈ = 0.5)
    @test ml.mixing_length.Cˢ == 1.2
    @test ml.Cᶜᵒⁿᵈ == 0.5

    # Mixed integer/float arguments are promoted, as elsewhere in the closure
    @test ml.C𝒟ᵍ isa Float64 && ml.Cʰ isa Float64

    closure = TKEBasedTurbulenceClosure(; mixing_length = ml)
    @test closure.mixing_length isa ConditionalStabilityMixingLength{<:Any, FT}
    @test closure.mixing_length.mixing_length isa GradientLimitedMixingLength{FT}
    @test isbits(closure)

    closure32 = TKEBasedTurbulenceClosure(Float32; mixing_length = ml)
    @test closure32.mixing_length isa ConditionalStabilityMixingLength{<:Any, Float32}
    @test closure32.mixing_length.mixing_length isa GradientLimitedMixingLength{Float32}

    # The wrapper carries no Cˢ, so it names the model that does rather than reading one off itself
    @test occursin("wrapping", mixing_length_summary(ml))
    @test occursin("GradientLimitedMixingLength", mixing_length_summary(ml))
    str = sprint(show, closure)
    @test occursin("ConditionalStabilityMixingLength", str)
    @test occursin("GradientLimitedMixingLength", str)
    @test occursin("Cᶜᵒⁿᵈ", sprint(show, ml))

    @test_throws ArgumentError ConditionalStabilityMixingLength(wrapped; C𝒟ᵍ = -1)
    @test_throws ArgumentError ConditionalStabilityMixingLength(wrapped; C𝒟⁰ = -1e-6)
    @test_throws ArgumentError ConditionalStabilityMixingLength(wrapped; Cʰ = -0.5)
    @test_throws ArgumentError ConditionalStabilityMixingLength(wrapped; Cᶜᵒⁿᵈ = -0.1)
    @test_throws ArgumentError ConditionalStabilityMixingLength(wrapped; Cᶜᵒⁿᵈ = 1.5)
end

#####
##### The saturated fraction of a trial excursion
#####

@testset "the saturated fraction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    f(𝒟, σ, A, h) = saturated_fraction(FT(𝒟), FT(σ), FT(A), FT(h))

    @test normal_cdf(zero(FT)) == FT(0.5)
    @test normal_cdf(FT(-Inf)) == 0
    @test normal_cdf(FT(Inf)) == 1

    # The sharp limit σ𝒟 = 0 is the saturation indicator, with qʷ ≥ qˢ counting as saturated,
    # and never a 0/0
    @test f(-1, 0, 0, 0) == 0
    @test f(1, 0, 0, 0) == 1
    @test f(0, 0, 0, 0) == 1
    @test isfinite(f(0, 0, 0, 0))
    @test f(0, 0, 0, 0) isa FT

    # h = 0 is the local saturation probability, and A cannot matter there
    @test f(0, 1, 0, 0) ≈ 0.5
    @test f(0, 1, 5, 0) ≈ 0.5
    @test f(0, 1, -5, 0) ≈ 0.5
    @test f(-1e9, 1, 0, 0) ≈ 0 atol=1e-9
    @test f(1e9, 1, 0, 0) ≈ 1

    # Bounded in [0, 1], and rising with the mean excess and with the excursion
    𝒟s = range(FT(-5e-3), FT(5e-3), length = 21)
    fs = [f(𝒟, 1e-3, 3e-6, 500) for 𝒟 in 𝒟s]
    @test all(0 .≤ fs .≤ 1)
    @test issorted(fs)
    @test issorted([f(-1e-3, 1e-3, 3e-6, h) for h in (0, 100, 300, 1000, 3000)])

    # A wider distribution moves a subsaturated point towards even odds
    @test f(-2e-3, 1e-4, 0, 0) < f(-2e-3, 1e-2, 0, 0) < FT(0.5)

    # Finite for extreme arguments, in both float types
    for (𝒟, σ, A, h) in ((0, 0, 0, 0), (1e30, 1e-30, 0, 0), (-1e30, 1e-30, 0, 0),
                         (0, floatmin(FT), 1, 1e30), (0, 1, 0, 0))
        v = f(𝒟, σ, A, h)
        @test isfinite(v) && 0 ≤ v ≤ 1
    end
end

#####
##### In a model
#####

@testset "ConditionalStabilityMixingLength in a model [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz, Lz = 40, 2500
    grid = RectilinearGrid(default_arch; size = Nz, z = (0, Lz), topology = (Flat, Flat, Bounded))
    zf = Array(znodes(grid, Face()))
    Cˢ = FT(1.316)
    e₀ = FT(0.4)

    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    # A conditionally unstable column: moist below, a sharp inversion, drier above
    θᵢ(z) = 295 + 0.004z + 6 * (z > 1200)
    qᵢ(z) = ifelse(z < 1200, 0.014 - 2e-6z, 0.004)

    function run_column(mixing_length; steps = 0, dry = false)
        closure = TKEBasedTurbulenceClosure(; mixing_length)
        model = dry ? AtmosphereModel(grid; closure, advection = nothing) :
                      AtmosphereModel(grid; closure, microphysics, advection = nothing)
        if dry
            set!(model; θ = z -> 295 + 0.004z, qᵗ = 1e-6, u = z -> 0.004z)
        else
            set!(model; θ = θᵢ, qᵗ = qᵢ, u = z -> 0.004z)
        end
        set_tke!(model, e₀)
        for _ in 1:steps
            time_step!(model, 5)
        end
        return model
    end

    base = GradientLimitedMixingLength(; Cˢ)
    trial = (C𝒟ᵍ = 1, C𝒟⁰ = 1e-3, Cʰ = 1, Cᶜᵒⁿᵈ = FT(0.7))

    reference = run_column(base)
    ℓ₀ = column(reference.closure_fields.ℓ)

    @testset "zero strength reproduces the wrapped model exactly" begin
        # Every coefficient zero, and the correction's widths on with Cᶜᵒⁿᵈ = 0: both must be the
        # wrapped model bitwise, not merely close, since the correction term is identically zero
        for ml in (ConditionalStabilityMixingLength(base),
                   ConditionalStabilityMixingLength(base; C𝒟ᵍ = 1, C𝒟⁰ = 1e-3, Cʰ = 1, Cᶜᵒⁿᵈ = 0))
            model = run_column(ml)
            @test column(model.closure_fields.ℓ) == ℓ₀
            @test column(model.closure_fields.Kᵘ) == column(reference.closure_fields.Kᵘ)
            @test column(model.closure_fields.Kᶜ) == column(reference.closure_fields.Kᶜ)
            @test column(model.closure_fields.Lᵉ) == column(reference.closure_fields.Lᵉ)
        end
    end

    @testset "the correction only lengthens, and stays finite" begin
        for Cᶜᵒⁿᵈ in FT.((0.25, 0.5, 1.0))
            ml = ConditionalStabilityMixingLength(base; C𝒟ᵍ = 1, C𝒟⁰ = 1e-3, Cʰ = 1, Cᶜᵒⁿᵈ)
            ℓ₁ = column(run_column(ml).closure_fields.ℓ)
            @test all(isfinite, ℓ₁)
            @test all(ℓ₁ .≥ ℓ₀ .- sqrt(eps(FT)) * Lz)
        end

        # A zero-width, zero-excursion correction still only weakens, and a very dry column has
        # nothing to condense, so the correction is inert there
        inert = ConditionalStabilityMixingLength(base; C𝒟ᵍ = 0, C𝒟⁰ = 0, Cʰ = 0, Cᶜᵒⁿᵈ = 1)
        @test all(column(run_column(inert).closure_fields.ℓ) .≥ ℓ₀ .- sqrt(eps(FT)) * Lz)

        # A Gaussian has infinite support, so for σ𝒟 > 0 the saturated fraction is never *exactly*
        # zero and the correction is never identically zero — only negligible. In a column at
        # 1 mg/kg the excess sits eight or so widths below saturation, Φ(-8) ~ 1e-16, and ℓ moves by
        # a last ulp or two at the top. Exact equality holds only for Cᶜᵒⁿᵈ = 0 or σ𝒟 = 0.
        ℓdry₀ = column(run_column(base; dry = true).closure_fields.ℓ)
        ℓdry₁ = column(run_column(ConditionalStabilityMixingLength(base; trial...); dry = true).closure_fields.ℓ)
        @test all(ℓdry₁ .≥ ℓdry₀ .- sqrt(eps(FT)) * Lz)
        @test maximum(abs, ℓdry₁ .- ℓdry₀) < 1e-6 * maximum(ℓdry₀)
    end

    @testset "both envelopes match a brute-force reference" begin
        ml = ConditionalStabilityMixingLength(base; trial...)
        model = run_column(ml)
        N₀² = column(model.closure_fields.N²)
        ℓ₁ = column(model.closure_fields.ℓ)

        # The wrapped model's own envelope, from its penetration depths
        penetration(N²) = [N²[k] > 0 ? sqrt(e₀) / sqrt(N²[k]) : FT(Inf) for k in eachindex(zf)]
        @test all(isapprox.(ℓ₀, brute_force_envelope(zf, penetration(N₀²), Cˢ), rtol = 1e-5))

        # and the second envelope, from the corrected stability reconstructed independently
        buoyancy = Oceananigans.TurbulenceClosures.buoyancy_force(model)
        tracers = Oceananigans.TurbulenceClosures.buoyancy_tracers(model)
        N₁² = [conditional_static_stabilityᶜᶜᶠ(1, 1, k, grid, ml, ℓ₀[k], N₀²[k], buoyancy, tracers)
               for k in eachindex(zf)]
        @test all(N₁² .≤ N₀² .+ sqrt(eps(FT)))
        @test all(isapprox.(ℓ₁, brute_force_envelope(zf, penetration(N₁²), Cˢ), rtol = 1e-5))

        # Where the air is already saturated the stored N² *is* the saturated response, so there is
        # nothing to weaken and the correction vanishes identically
        Nₘ² = [saturated_static_stabilityᶜᶜᶠ(1, 1, k, grid, buoyancy, tracers.T, tracers.qᵛ)
               for k in eachindex(zf)]
        already = findall(k -> N₀²[k] == Nₘ²[k], eachindex(zf))
        @test all(N₁²[already] .== N₀²[already])

        # A rising unsaturated parcel approaches saturation: A > 0 through the troposphere
        A = [saturation_excess_lapse_rateᶜᶜᶠ(1, 1, k, grid, buoyancy, tracers.T, tracers.qᵛ)
             for k in 2:Nz]
        @test all(A .> 0)
    end

    @testset "stepping stays finite" begin
        ml = ConditionalStabilityMixingLength(base; C𝒟ᵍ = 1, C𝒟⁰ = 1e-3, Cʰ = 1, Cᶜᵒⁿᵈ = FT(0.8))
        model = run_column(ml; steps = 10)
        ρe = column(model.tracers.ρe)
        @test all(isfinite, ρe)
        @test all(ρe .≥ 0)
        @test all(isfinite, column(model.closure_fields.ℓ))
        @test all(isfinite, column(model.closure_fields.Kᶜ))
    end
end
