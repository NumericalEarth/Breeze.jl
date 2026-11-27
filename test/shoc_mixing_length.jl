include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans.Grids: znodes
using Breeze.TurbulenceClosures: mixing_lengthᶜᶜᶜ
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: blended_length, asymptotic_length
using Test

#####
##### The mixing length of SHOC, `BlendedMixingLength`, a mixing length of `TKEBasedTurbulenceClosure`, compared
##### with the expressions of shoc.F90 written out with its literals. The closure bounds the length from above by
##### sums of inverse squares (maxlen, the horizontal grid spacing) where shoc.F90 takes a `min`, does not impose
##### `minlen`, and takes the stable part of N² regular through N² = 0 where shoc.F90 takes `max(N², 0)`: in the
##### regime where `minlen` is inactive the two differ by at most the fraction
##### (ℓ / maxlen)² + (ℓ / Δʰ)² + ℂˡ₃ (ℂˡ₁ / ℂˡ₂)² / 4.
#####

# `compute_shoc_mix_shoc_length` followed by `check_length_scale_shoc_length` of E3SM `shoc.F90`, for turbulent
# kinetic energy `e`, static stability `N²`, height `z`, asymptotic length `ℓ∞` and horizontal grid spacing `Δʰ`
function shoc_mixing_length(e, N², z, ℓ∞, Δʰ)
    FT = typeof(e)
    ℂˡ₁, ℂˡ₅, κ, maximum_length, minimum_length = FT(0.5), FT(400), FT(0.4), FT(20000), FT(20)
    u = sqrt(e)
    N²₊ = N² >= 0 ? N² : zero(FT)
    ℓ = min(maximum_length, (FT(2.8284) * sqrt(1 / ((1 / (ℂˡ₅ * u * κ * z))
            + (1 / (ℂˡ₅ * u * ℓ∞)) + FT(0.01) * (N²₊ / e)))) / ℂˡ₁)
    ℓ = min(maximum_length, ℓ)
    ℓ = max(minimum_length, ℓ)
    ℓ = min(Δʰ, ℓ)
    return ℓ
end

# `compute_l_inf_shoc_length`: 0.1 ∑ √e z Δz / ∑ √e Δz
shoc_asymptotic_length(e, z, Δz) = 0.1 * sum(sqrt.(e) .* z .* Δz) / sum(sqrt.(e) .* Δz)

# The largest fraction by which the closure's length is shorter than that of shoc.F90: the bounds as sums of
# inverse squares, and the regular stable part of N², which adds at most e / (2 ℓ₀²) to max(N², 0)
regular_fraction(ml) = ml.ℂˡ₃ * (ml.ℂˡ₁ / ml.ℂˡ₂)^2 / 4
bound_fraction(ml, ℓ, Δʰ) = (ℓ / 20000)^2 + (ℓ / Δʰ)^2 + regular_fraction(ml)

@testset "BlendedMixingLength is the length of shoc.F90, regular through N² = 0 [$FT]" for FT in test_float_types()
    ml = BlendedMixingLength(FT)
    @test ml isa AbstractMixingLength
    @test summary(ml) == "BlendedMixingLength{$FT}"
    @test ml.ℂˡ₁ === FT(0.5) && ml.ℂˡ₂ === FT(2.8284) && ml.ℂˡ₅ === FT(400) && ml.maximum_length === FT(20000)
    @test_throws ArgumentError BlendedMixingLength(FT; minlen = 20)
    @test regular_fraction(ml) < 1e-4

    roundoff = FT == Float32 ? 1e-5 : 1e-12
    for e in FT[0.01, 0.1, 0.5, 2], N² in FT[-1e-4, 0, 1e-5, 3e-4], z in FT[20, 100, 500, 2000],
        ℓ∞ in FT[100, 300], Δʰ in FT[1e5, 3000, Inf]
        ℓˢ = shoc_mixing_length(e, N², z, ℓ∞, Δʰ)
        ℓˢ > 20 || continue # minlen is not imposed by the closure
        ℓ = blended_length(ml, z, sqrt(e), ℓ∞, N², Δʰ^2)
        @test abs(ℓ - ℓˢ) / ℓˢ ≤ bound_fraction(ml, ℓˢ, Δʰ) + roundoff
        @test ℓ ≤ ℓˢ * (1 + roundoff)
    end

    # The length vanishes with the height above the surface and with √e, without a floor
    @test blended_length(ml, zero(FT), FT(0.3), FT(200), zero(FT), FT(Inf)) == 0
    @test blended_length(ml, FT(100), FT(1e-6), FT(200), FT(1e-4), FT(Inf)) < FT(1)

    # The stable part of N² is regular through N² = 0 on the scale e / ℓ₀² of the turbulence of neutral air: the
    # length decreases with N² everywhere, its one-sided slopes in N² agree at N² = 0, and far from N² = 0 on that
    # scale it is the length of the `max`: that of neutral air in unstable air, the buoyancy length in stable air
    e, z, ℓ∞ = FT(0.5), FT(500), FT(300)
    ℓ₀ = shoc_mixing_length(e, zero(FT), z, ℓ∞, FT(Inf))
    scale = e / ℓ₀^2
    lengths = [blended_length(ml, z, sqrt(e), ℓ∞, x * scale, FT(Inf)) for x in range(FT(-50), FT(50); length = 201)]
    @test issorted(lengths; rev = true)
    @test all(ℓ -> 0 < ℓ ≤ ℓ₀ * (1 + roundoff), lengths)
    for x in FT[-1e4, -1e2, 1e2, 1e4, 1e6]
        ℓˢ = shoc_mixing_length(e, x * scale, z, ℓ∞, FT(Inf))
        @test isapprox(blended_length(ml, z, sqrt(e), ℓ∞, x * scale, FT(Inf)), ℓˢ; rtol = regular_fraction(ml) / abs(x) + (ℓˢ / 20000)^2 + roundoff)
    end
    if FT == Float64
        h = 1e-3 * scale
        length_of(N²) = blended_length(ml, z, sqrt(e), ℓ∞, N², Inf)
        below = (length_of(0.0) - length_of(-h)) / h
        above = (length_of(h) - length_of(0.0)) / h
        # the one-sided slopes of the `max` are 0 and -ℂˡ₃ (ℂˡ₁ / ℂˡ₂)² ℓ₀³ / (2 e)
        kink = ml.ℂˡ₃ * (ml.ℂˡ₁ / ml.ℂˡ₂)^2 * ℓ₀^3 / (2 * e)
        @test abs(above - below) < 1e-2 * kink
        @test isapprox((above + below) / 2, -kink / 2; rtol = 1e-2)
    end
end

function bomex_like_model(FT; closure = SimplifiedHigherOrderClosure(FT), Nz = 40)
    grid = RectilinearGrid(default_arch, FT; size = Nz, z = (0, 2000), topology = (Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    θᵢ(z) = 298.7 + 4e-3 * max(z - 500, 0)
    qᵢ(z) = 0.017 - 3e-6 * z
    reference_state = ReferenceState(grid, constants; base_pressure = 101500, potential_temperature = θᵢ,
                                     vapor_mass_fraction = qᵢ)
    ρ₀ = FT(1.16)
    boundary_conditions = (ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * FT(8e-3))),
                           ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * FT(5.2e-5))))
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), closure, boundary_conditions,
                            microphysics = SaturationAdjustment(FT; equilibrium = WarmPhaseEquilibrium()))
    set!(model, θ = θᵢ, qᵗ = qᵢ, u = -5)
    e₀ = CenterField(grid)
    set!(e₀, z -> max(0.5 * (1 - z / 1000), 0.01))
    set!(model.tracers.ρe, reference_state.density * e₀)
    return model
end

column(f) = Array{Float64}(interior(f))[1, 1, :]

@testset "BlendedMixingLength in a column [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    model = bomex_like_model(FT)
    for _ in 1:10
        time_step!(model, 30)
    end
    closure = model.closure
    fields = model.closure_fields
    grid = model.grid
    ml = closure.mixing_length
    ρ = column(model.dynamics.reference_state.density)
    e = column(model.tracers.ρe) ./ ρ
    eᵐⁱⁿ = closure.minimum_tke
    zᶜ = collect(znodes(grid, Center()))
    zᶠ = collect(znodes(grid, Face()))
    Δz = diff(zᶠ)
    N² = column(fields.N²)
    ℓ = column(fields.ℓᵐ)

    # ℓ∞ of shoc.F90 from the TKE floored at `minimum_tke`, kept in ℓˢ
    ℓ∞ = shoc_asymptotic_length(max.(e, eᵐⁱⁿ), zᶜ, Δz)
    @test all(isapprox.(column(fields.ℓˢ), ℓ∞; rtol = FT == Float32 ? 1e-4 : 1e-10))

    # The faces: √e reconstructed from the centers, the stored N²
    Nz = size(grid, 3)
    roundoff = FT == Float32 ? 1e-4 : 1e-10
    for k in 2:Nz
        u = (sqrt(max(e[k-1], eᵐⁱⁿ)) + sqrt(max(e[k], eᵐⁱⁿ))) / 2
        ℓˢ = shoc_mixing_length(u^2, N²[k], zᶠ[k], ℓ∞, Inf)
        ℓˢ > 20 || continue
        @test abs(ℓ[k] - ℓˢ) / ℓˢ ≤ bound_fraction(ml, ℓˢ, Inf) + roundoff
    end

    # The centers, where the dissipation lives, with N² from the two faces
    e_field = Field(model.tracers.ρe / model.dynamics.reference_state.density)
    compute!(e_field)
    for k in 2:Nz-1
        ℓᶜ = @allowscalar mixing_lengthᶜᶜᶜ(1, 1, k, grid, closure, e_field, fields)
        ℓˢ = shoc_mixing_length(max(e[k], eᵐⁱⁿ), (N²[k] + N²[k+1]) / 2, zᶜ[k], ℓ∞, Inf)
        ℓˢ > 20 || continue
        @test abs(ℓᶜ - ℓˢ) / ℓˢ ≤ bound_fraction(ml, ℓˢ, Inf) + roundoff
    end
    @test all(isfinite, ℓ)
    @test all(≥(0), ℓ)
end
