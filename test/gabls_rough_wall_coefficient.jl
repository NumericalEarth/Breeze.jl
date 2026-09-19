include(joinpath(@__DIR__, "setup.jl"))

#####
##### Tests for `GABLSRoughWallCoefficient`, the rough-wall Monin-Obukhov transfer
##### coefficients with the linear stable functions specified by GABLS1
##### (Beare et al. 2006).
#####
##### The point of the type is that linear stability functions close the
##### Monin-Obukhov balance in closed form, so most of what follows checks the
##### *balance itself* rather than memorized constants:
#####
##### - T1 — neutral limit: Riᴮ = 0 gives ζ = 0 and the log-law coefficient κ²/α².
##### - T2 — the returned ζ satisfies the Monin-Obukhov definition ζ = κ g z θ★/(u★² θᵣ)
#####        when u★ and θ★ are rebuilt from it. This is the real test: it would catch
#####        an algebra slip in deriving the quadratic, which a pinned number would not.
##### - T3 — Cᴰ decreases monotonically with stability, and Cᵀ ≤ Cᴰ once stable.
##### - T4 — the coefficient is continuous across the critical Richardson number
#####        Riᴮ = βᵀ/(βᴰ)², where the quadratic's leading coefficient vanishes. The
#####        same bound applies on both sides precisely so that it does not jump.
##### - T5 — the unstable branch returns the neutral coefficient rather than
#####        extrapolating a stable formula outside its range.
##### - T6 — ζ never exceeds `maximum_stability`, for any Riᴮ including absurd ones.
##### - T7 — the struct is concretely typed and `isbits` at every supported float
#####        type, which is what makes it usable inside a GPU kernel.
#####
##### Behaviour inside a model -- materialization stamping Cᴰ on momentum walls and
##### Cᵀ on scalar walls, and the Exner conversion of the wall temperature -- is
##### covered by the GABLS1 validation runs, which need a model to exercise.
#####

using Breeze
using Breeze.BoundaryConditions: gabls_stability_parameter, gabls_transfer_coefficient
using Test

# The GABLS1 32³ grid has Δz = 12.5 m, so the first cell center is at 6.25 m.
const wall_distance_m = 6.25

"""Rebuild `α`, `αʰ` and the stable slopes from a coefficient, at height `z`."""
function similarity_parameters(c, z)
    α = log(z / c.roughness_length)
    αʰ = log(z / c.scalar_roughness_length)
    return α, αʰ, c.momentum_stability_parameter, c.temperature_stability_parameter
end

@testset "GABLSRoughWallCoefficient" begin
    c = GABLSRoughWallCoefficient()
    z = wall_distance_m
    α, αʰ, βᴰ, βᵀ = similarity_parameters(c, z)
    κ = c.von_karman_constant
    ζmax = c.maximum_stability

    ζ(Riᴮ) = gabls_stability_parameter(Riᴮ, α, αʰ, βᴰ, βᵀ, ζmax)
    Cᴰ(Riᴮ) = gabls_transfer_coefficient(c, α, αʰ, ζ(Riᴮ), Val(:momentum))
    Cᵀ(Riᴮ) = gabls_transfer_coefficient(c, α, αʰ, ζ(Riᴮ), Val(:scalar))

    @testset "T1: neutral limit is the log law" begin
        @test ζ(0) == 0
        @test Cᴰ(0) ≈ κ^2 / α^2
        @test Cᵀ(0) ≈ κ^2 / (α * αʰ)
        ## Equal roughness lengths by default, so momentum and heat coincide at neutral
        @test Cᴰ(0) ≈ Cᵀ(0)
        ## An unmaterialized coefficient falls back to momentum
        @test gabls_transfer_coefficient(c, α, αʰ, 0.0, nothing) == Cᴰ(0)
    end

    @testset "T2: ζ satisfies the Monin-Obukhov definition" begin
        g = c.gravitational_acceleration
        θᵣ = c.reference_temperature

        ## Stay strictly below the critical Richardson number, where a root exists and
        ## the cap is not active -- the cap is T4 and T6's business, not T2's.
        for Riᴮ in (1e-4, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25)
            ζᵢ = ζ(Riᴮ)
            @test 0 < ζᵢ < ζmax

            ## Any (U, Δθ) pair with this bulk Richardson number will do; fix U and solve
            ## for Δθ, then rebuild the similarity scales the stability functions imply.
            U = 5.0
            Δθ = Riᴮ * θᵣ * U^2 / (g * z)
            u★ = κ * U / (α + βᴰ * ζᵢ)
            θ★ = κ * Δθ / (αʰ + βᵀ * ζᵢ)

            @test κ * g * z * θ★ / (u★^2 * θᵣ) ≈ ζᵢ rtol=1e-12

            ## and the coefficients are consistent with those scales
            @test Cᴰ(Riᴮ) ≈ u★^2 / U^2
            @test Cᵀ(Riᴮ) ≈ u★ * θ★ / (U * Δθ)
        end
    end

    @testset "T3: stability suppresses exchange, heat more than momentum" begin
        stabilities = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
        drags = Cᴰ.(stabilities)

        @test issorted(ζ.(stabilities))
        @test issorted(drags, rev=true)
        @test all(>(0), drags)

        ## βᵀ > βᴰ, so the scalar denominator grows faster and Cᵀ/Cᴰ falls below one
        @test all(Riᴮ -> Cᵀ(Riᴮ) ≤ Cᴰ(Riᴮ), stabilities)
        @test Cᵀ(0.3) / Cᴰ(0.3) < 0.7
    end

    @testset "T4: continuous across the critical Richardson number" begin
        ## The leading coefficient βᵀ - Riᴮ (βᴰ)² vanishes here; above it the quadratic
        ## has no root and the bound takes over. Applying the *same* bound on both sides
        ## is what keeps the coefficient from jumping.
        Riᴮc = βᵀ / βᴰ^2
        @test Riᴮc ≈ 0.3385416666666667

        below = Cᴰ(prevfloat(Riᴮc))
        above = Cᴰ(nextfloat(Riᴮc))
        @test below ≈ above
        @test ζ(prevfloat(Riᴮc)) ≈ ζ(nextfloat(Riᴮc))

        ## Well past the critical point the coefficient is flat, not divergent
        @test Cᴰ(10.0) == Cᴰ(1.0e6)
        @test isfinite(Cᴰ(1.0e6))
        @test Cᴰ(1.0e6) > 0
    end

    @testset "T5: unstable side returns the neutral coefficient" begin
        for Riᴮ in (-1.0e-6, -0.01, -1.0, -100.0)
            @test ζ(Riᴮ) == 0
            @test Cᴰ(Riᴮ) == Cᴰ(0)
            @test Cᵀ(Riᴮ) == Cᵀ(0)
        end
    end

    @testset "T6: ζ respects the bound everywhere" begin
        for Riᴮ in (-1e3, -1.0, 0.0, 0.1, 0.3, 0.34, 1.0, 1e3, 1e9)
            @test 0 ≤ ζ(Riᴮ) ≤ ζmax
            @test isfinite(ζ(Riᴮ))
        end

        ## A tighter bound must bite, and a looser one must not loosen the neutral value
        tight = GABLSRoughWallCoefficient(maximum_stability=1)
        @test gabls_stability_parameter(1.0, α, αʰ, βᴰ, βᵀ, tight.maximum_stability) == 1
        @test gabls_stability_parameter(0.0, α, αʰ, βᴰ, βᵀ, tight.maximum_stability) == 0
    end

    @testset "T7: concretely typed and GPU-ready" begin
        for FT in all_float_types()
            cᶠ = GABLSRoughWallCoefficient(FT)
            @test cᶠ.von_karman_constant isa FT
            @test cᶠ.roughness_length isa FT
            @test cᶠ.scalar_roughness_length isa FT
            @test cᶠ.reference_temperature isa FT
            @test isbits(cᶠ)

            ## The stability solve must stay in FT rather than promoting to Float64
            αᶠ, αʰᶠ, βᴰᶠ, βᵀᶠ = similarity_parameters(cᶠ, FT(wall_distance_m))
            ζᶠ = gabls_stability_parameter(FT(0.1), αᶠ, αʰᶠ, βᴰᶠ, βᵀᶠ, cᶠ.maximum_stability)
            @test ζᶠ isa FT
            @test gabls_transfer_coefficient(cᶠ, αᶠ, αʰᶠ, ζᶠ, Val(:momentum)) isa FT
            @test gabls_transfer_coefficient(cᶠ, αᶠ, αʰᶠ, ζᶠ, Val(:scalar)) isa FT
        end

        ## A materialized coefficient carries `Val`s and stays `isbits`
        cᵐ = GABLSRoughWallCoefficient(transfer_type=Val(:scalar))
        @test isbits(cᵐ)
        @test gabls_transfer_coefficient(cᵐ, α, αʰ, 1.0, cᵐ.transfer_type) ==
              gabls_transfer_coefficient(c, α, αʰ, 1.0, Val(:scalar))
    end
end
