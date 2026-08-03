using Breeze
using Oceananigans
using Test

using Breeze.TurbulenceClosures: TKE_NAME,
                                 length_scaleᶜᶜᶠ,
                                 length_scaleᶜᶜᶜ,
                                 mixing_lengthᶜᶜᶠ,
                                 mixing_lengthᶜᶜᶜ,
                                 smooth_positive,
                                 turbulence_length_coefficient,
                                 von_karman_constant,
                                 turbulent_prandtl_number,
                                 _compute_turbulence_length_scale!
using Oceananigans: prognostic_fields
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!

#####
##### Coefficients
#####
##### The neutral constant-flux-layer relations are derived in the closure coefficient note.
##### With ℓᵍ = a z the log law constrains only Cˢ a = κ, where Cˢ = Cᴷ/(Cμ)^(1/4); choosing a = κ
##### makes that Cˢ = 1, i.e. the locus Cμ = Cᴷ⁴. All three published Mellor–Yamada sets satisfy it,
##### because Cμ = Cᴷ⁴ is equivalent to S_M(neutral) = B₁^(-1/3).
#####

@testset "TKEBasedTurbulenceClosure coefficients [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    constant_sets = (MYNN = TKEBasedTurbulenceClosure(),
                     MY82 = MY82Coefficients(),
                     MYJ  = MYJCoefficients())

    # The published sets are quoted to four digits, so the locus holds to that, not to eps
    rtol = FT === Float32 ? 1e-3 : 1e-3

    @testset "$name lies on the log-law locus" for (name, closure) in pairs(constant_sets)
        Cᴷ = diffusivity_coefficient(closure)
        Cμ = closure.Cμ

        @test isapprox(Cμ, Cᴷ^4; rtol)
        @test isapprox(stress_coefficient(closure), 1; rtol)

        # Cᵋ = Cμ/Cᴷ is the stored relation; on the locus it is also Cᴷ³
        @test isapprox(dissipation_coefficient(closure), Cμ / Cᴷ; rtol = 1e-6)
        @test isapprox(dissipation_coefficient(closure), Cᴷ^3; rtol)

        # The surface floor equals the log-layer equilibrium TKE, e/u★² = (Cˢ Cᵋ)^(-2/3),
        # identically — this is what makes the floor not an independent constraint.
        Cˢ = stress_coefficient(closure)
        Cᵋ = dissipation_coefficient(closure)
        @test isapprox(surface_tke_coefficient(closure), (Cˢ * Cᵋ)^(-2//3); rtol = 1e-5)
        @test isapprox(surface_tke_coefficient(closure), 1 / sqrt(Cμ); rtol = 1e-6)

        # ... and on the locus it is also Cᴷ⁻²
        @test isapprox(surface_tke_coefficient(closure), Cᴷ^-2; rtol)
    end

    @testset "the equilibrium TKE tracks Cμ off the locus too" begin
        # e/u★² = (Cμ)^(-1/2) holds for any (Cᴷ, Cμ), which is the reason this pair is stored
        for Cᴷ in (0.2, 0.4903, 0.8), Cμ in (0.02, 0.0578, 0.2)
            closure = TKEBasedTurbulenceClosure(; Cᴷ, Cμ)
            Cˢ = stress_coefficient(closure)
            Cᵋ = dissipation_coefficient(closure)
            @test isapprox(surface_tke_coefficient(closure), (Cˢ * Cᵋ)^(-2//3); rtol = 1e-5)
        end
    end

    @testset "MYNN is the default" begin
        closure = TKEBasedTurbulenceClosure()
        @test closure.Cᴷ ≈ FT(0.4903)
        @test closure.Cμ ≈ FT(0.0578)
        @test isapprox(surface_tke_coefficient(closure), 4.16; rtol = 1e-3)
    end

    @testset "isbits" begin
        @test isbits(TKEBasedTurbulenceClosure())
        @test isbits(BlendedMixingLength(GeometricLengthScale(), BuoyancyLengthScale()))
        @test isbits(BlendedMixingLength(GeometricLengthScale(); blend = PowerBlend(p = 2.0)))
    end

    @testset "show" begin
        closure = TKEBasedTurbulenceClosure()
        @test occursin("TKEBasedTurbulenceClosure", summary(closure))
        @test occursin("Cˢ", sprint(show, closure))
        @test occursin("BlendedMixingLength", sprint(show, closure.mixing_length))
        @test occursin("GeometricLengthScale", sprint(show, closure.mixing_length))
    end
end

#####
##### Mixing length, branch by branch
#####

@testset "Mixing-length branches [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nz = 32
    Lz = FT(1000)
    grid = RectilinearGrid(default_arch; size=Nz, z=(0, Lz), topology=(Flat, Flat, Bounded))

    ℓᵗ_field = Field{Center, Center, Nothing}(grid)
    set!(ℓᵗ_field, 300)
    u★²_field = Field{Center, Center, Nothing}(grid)
    set!(u★²_field, 0.09)
    Jᵇ_field = Field{Center, Center, Nothing}(grid)      # neutral surface
    state = (; ℓᵗ = ℓᵗ_field, u★² = u★²_field, Jᵇ = Jᵇ_field)

    geometric = GeometricLengthScale{FT}()
    buoyancy = BuoyancyLengthScale{FT}()
    turbulence = TurbulenceLengthScale{FT}()

    @testset "smooth_positive" begin
        δ = FT(1e-9)
        @test smooth_positive(FT(1), δ) ≈ 1
        @test smooth_positive(FT(-1), δ) < 1e-9
        @test smooth_positive(FT(0), δ) ≈ δ / 2
        # monotone and never negative
        @test all(smooth_positive(x, δ) ≥ 0 for x in FT(-1):FT(0.1):FT(1))
    end

    @testset "ℓᵍ = κ(z + ℓʳ)" begin
        κ, ℓʳ = geometric.κ, geometric.ℓʳ
        Δz = Lz / Nz
        # Face k sits at z = (k - 1) Δz
        for k in (1, 2, 8, Nz)
            z = (k - 1) * Δz
            @test length_scaleᶜᶜᶠ(1, 1, k, grid, geometric, FT(1), FT(0), state) ≈ κ * (z + ℓʳ)
        end
        # Finite at the surface — this is what ℓʳ is for
        @test length_scaleᶜᶜᶠ(1, 1, 1, grid, geometric, FT(1), FT(0), state) ≈ κ * ℓʳ
        @test length_scaleᶜᶜᶠ(1, 1, 1, grid, geometric, FT(1), FT(0), state) > 0
        # Centers sit half a cell above the face below them
        @test length_scaleᶜᶜᶜ(1, 1, 1, grid, geometric, FT(1), FT(0), state) ≈ κ * (Δz / 2 + ℓʳ)
    end

    @testset "ℓᵇ = Cᵇ q / N" begin
        Cᵇ = buoyancy.Cᵇ
        q = FT(1)
        ℓᵇ(q, N²) = length_scaleᶜᶜᶠ(1, 1, 1, grid, buoyancy, q, N², state)

        N² = FT(1e-4)
        @test ℓᵇ(q, N²) ≈ Cᵇ * q / sqrt(N²) rtol=1e-5

        # Scales like q and like 1/N
        @test ℓᵇ(2q, N²) ≈ 2 * ℓᵇ(q, N²) rtol=1e-5
        @test ℓᵇ(q, 4N²) ≈ ℓᵇ(q, N²) / 2 rtol=1e-5

        # Inactive in unstable and neutral air: the branch returns a length so large that it
        # drops out of any blend that selects the smallest scale
        @test ℓᵇ(q, FT(-1e-4)) > 1e6
        @test ℓᵇ(q, FT(0)) > 1e3
        @test ℓᵇ(q, FT(-1e-4)) > ℓᵇ(q, FT(0))   # smooth, monotone through N² = 0

        # Deardorff's constant is the default; MYNN's realizability bound is the looser Cᵇ = 1
        @test buoyancy.Cᵇ ≈ FT(0.53)

        # Setting one coefficient to an integer must work: @kwdef alone would demand that every
        # field share a type, so `Cᵇ = 1` beside a float default would find no method.
        @test BuoyancyLengthScale(Cᵇ = 1).Cᵇ == 1
        @test GeometricLengthScale(ℓʳ = 1).ℓʳ == 1
        @test length_scaleᶜᶜᶠ(1, 1, 1, grid, BuoyancyLengthScale{FT}(Cᵇ = 1), q, N², state) ≈
              q / sqrt(N²) rtol=1e-5
    end

    @testset "ℓᵗ is read from the column field, at both locations" begin
        for k in (1, 8, Nz)
            @test length_scaleᶜᶜᶠ(1, 1, k, grid, turbulence, FT(1), FT(0), state) == 300
            @test length_scaleᶜᶜᶜ(1, 1, k, grid, turbulence, FT(1), FT(0), state) == 300
        end
    end
end

@testset "SurfaceLayerLengthScale [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nz = 32
    Lz = FT(1000)
    grid = RectilinearGrid(default_arch; size=Nz, z=(0, Lz), topology=(Flat, Flat, Bounded))
    surface = SurfaceLayerLengthScale{FT}()
    geometric = GeometricLengthScale{FT}(κ = surface.κ, ℓʳ = surface.ℓʳ)

    ## ζ = z/L with L = -u★³/(κ Jᵇ): Jᵇ < 0 is stable, Jᵇ > 0 unstable, Jᵇ = 0 neutral.
    function state(u★², Jᵇ)
        u = Field{Center, Center, Nothing}(grid); set!(u, u★²)
        J = Field{Center, Center, Nothing}(grid); set!(J, Jᵇ)
        ℓᵗ = Field{Center, Center, Nothing}(grid); set!(ℓᵗ, 300)
        return (; ℓᵗ, u★² = u, Jᵇ = J)
    end

    ℓˢ(k, u★², Jᵇ) = length_scaleᶜᶜᶠ(1, 1, k, grid, surface, FT(1), FT(0), state(u★², Jᵇ))
    ℓᵍ(k) = length_scaleᶜᶜᶠ(1, 1, k, grid, geometric, FT(1), FT(0), state(FT(0.1), FT(0)))

    @testset "neutral reproduces the plain geometric branch exactly" begin
        # This is what makes the correction safe to adopt: it cannot disturb a neutral column.
        for k in (2, 8, Nz)
            @test ℓˢ(k, FT(0.09), FT(0)) ≈ ℓᵍ(k)
        end
    end

    @testset "stable shrinks the branch, unstable grows it" begin
        for k in (4, 8, Nz)
            @test ℓˢ(k, FT(0.09), FT(-1e-3)) < ℓᵍ(k)   # stable
            @test ℓˢ(k, FT(0.09), FT(+1e-3)) > ℓᵍ(k)   # unstable
        end
    end

    @testset "the strongly stable limit is ℓᵍ/Cⁿ" begin
        # ζ ≫ 1 saturates at the first branch of MYNN Eq. 53
        k = 8
        @test ℓˢ(k, FT(1e-4), FT(-1)) ≈ ℓᵍ(k) / surface.Cⁿ
    end

    @testset "the branches join continuously" begin
        k = 8
        z = (k - 1) * Lz / Nz
        # ζ = -κ z Jᵇ / u★³; pick Jᵇ that straddles ζ = 1 and ζ = 0
        u★² = FT(0.09)
        u★³ = u★² * sqrt(u★²)
        Jᵇ_at(ζ) = -ζ * u★³ / (surface.κ * z)
        for ζ in (FT(0), FT(1))
            below = ℓˢ(k, u★², Jᵇ_at(ζ - FT(1e-6)))
            above = ℓˢ(k, u★², Jᵇ_at(ζ + FT(1e-6)))
            @test isapprox(below, above; rtol = 1e-3)
        end
    end

    @testset "free convection is finite, not NaN" begin
        # u★ = 0 sends ζ → -∞. The shear scale is irrelevant there, so the branch must grow rather
        # than divide by zero.
        for Jᵇ in (FT(0), FT(1e-2))
            ℓ = ℓˢ(8, FT(0), Jᵇ)
            @test !isnan(ℓ)
            @test ℓ > 0
        end
        @test ℓˢ(8, FT(0), FT(1e-2)) > ℓᵍ(8)
    end

    @testset "the unstable branch is bounded by ζᵐⁱⁿ" begin
        # Without the floor, zero mean wind sends ζ → -10¹⁴ and the branch to ~2000 κz, removing
        # the wall constraint entirely. The ceiling is the branch evaluated at ζᵐⁱⁿ.
        ceiling(k) = ℓᵍ(k) * (1 - surface.Cᶜ * surface.ζᵐⁱⁿ)^surface.nᶜ

        for k in (2, 8, Nz)
            # u★ = 0 is the degenerate limit and must land exactly on the ceiling
            @test ℓˢ(k, FT(0), FT(1e-2)) ≈ ceiling(k) rtol=1e-5
            # and nothing may exceed it, however extreme the forcing
            for (u★², Jᵇ) in ((FT(1e-8), FT(1)), (FT(1e-4), FT(1e-1)), (FT(0), FT(1e3)))
                @test ℓˢ(k, u★², Jᵇ) ≤ ceiling(k) + sqrt(eps(FT))
            end
        end

        # The default floor sits just beyond Nakanishi (2001)'s data, ζ ∈ [-3.13, 0.44]
        @test surface.ζᵐⁱⁿ ≈ FT(-4)
        @test ceiling(8) / ℓᵍ(8) ≈ FT(401)^FT(0.2) rtol=1e-5
    end

    @testset "the floor leaves the fitted range untouched" begin
        # ζ inside the data must be unaffected: compare against a branch with a far deeper floor.
        deep = SurfaceLayerLengthScale{FT}(ζᵐⁱⁿ = -1000)
        k, z = 8, 7 * Lz / Nz
        u★² = FT(0.09)
        u★³ = u★² * sqrt(u★²)
        for ζ in (FT(-0.5), FT(-2), FT(-3))          # inside [-3.13, 0]
            Jᵇ = -ζ * u★³ / (surface.κ * z)
            @test ℓˢ(k, u★², Jᵇ) ≈
                  length_scaleᶜᶜᶠ(1, 1, k, grid, deep, FT(1), FT(0), state(u★², Jᵇ)) rtol=1e-5
        end
        # and beyond it, the floor binds while the deeper one keeps growing
        ζ = FT(-40)
        Jᵇ = -ζ * u★³ / (surface.κ * z)
        @test ℓˢ(k, u★², Jᵇ) <
              length_scaleᶜᶜᶠ(1, 1, k, grid, deep, FT(1), FT(0), state(u★², Jᵇ))
    end
end

#####
##### Convective enhancement of ℓᵇ (MYNN Eq. 55)
#####

@testset "BuoyancyLengthScale convective enhancement [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nz = 8
    grid = RectilinearGrid(default_arch; size=Nz, z=(0, FT(1000)), topology=(Flat, Flat, Bounded))

    function state(Jᵇ, ℓᵗ_value)
        J = Field{Center, Center, Nothing}(grid); set!(J, Jᵇ)
        ℓᵗ = Field{Center, Center, Nothing}(grid); set!(ℓᵗ, ℓᵗ_value)
        u = Field{Center, Center, Nothing}(grid); set!(u, FT(0.09))
        return (; ℓᵗ, u★² = u, Jᵇ = J)
    end

    q = FT(1)
    N² = FT(1e-4)
    plain = BuoyancyLengthScale{FT}()                  # Cᶜᵇ = 0 by default
    enhanced = BuoyancyLengthScale{FT}(Cᶜᵇ = 5)

    ℓᵇ(branch, Jᵇ, ℓᵗ) = length_scaleᶜᶜᶠ(1, 1, 2, grid, branch, q, N², state(Jᵇ, ℓᵗ))

    @testset "off by default" begin
        @test plain.Cᶜᵇ == 0
        @test ℓᵇ(plain, FT(1e-2), FT(300)) ≈ ℓᵇ(plain, FT(0), FT(300))
    end

    @testset "lengthens ℓᵇ only under an unstable surface" begin
        @test ℓᵇ(enhanced, FT(1e-2), FT(300)) > ℓᵇ(plain, FT(1e-2), FT(300))
        # A stable or neutral surface leaves the branch as plain Cᵇ q / N
        @test ℓᵇ(enhanced, FT(0), FT(300)) ≈ ℓᵇ(plain, FT(0), FT(300))
        @test ℓᵇ(enhanced, FT(-1e-2), FT(300)) ≈ ℓᵇ(plain, FT(-1e-2), FT(300))
    end

    @testset "grows with the surface flux and is finite for an unbounded ℓᵗ" begin
        @test ℓᵇ(enhanced, FT(4e-2), FT(300)) > ℓᵇ(enhanced, FT(1e-2), FT(300))
        # A quiescent column has ℓᵗ = Inf; written naively the enhancement would be Inf/Inf
        ℓ = ℓᵇ(enhanced, FT(1e-2), FT(Inf))
        @test !isnan(ℓ)
        @test ℓ ≈ ℓᵇ(plain, FT(1e-2), FT(Inf))
    end
end

#####
##### Blending rules
#####

@testset "Length-scale blends [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    ℓs = (FT(3), FT(7), FT(50))
    ℓᵐⁱⁿ = minimum(ℓs)

    @testset "each rule computes what it claims" begin
        @test MinimumBlend()(ℓs) == ℓᵐⁱⁿ
        @test HarmonicBlend()(ℓs) ≈ inv(sum(inv, ℓs))
        @test PowerBlend{FT}(p = 2)(ℓs) ≈ inv(sqrt(sum(ℓ -> ℓ^-2, ℓs)))
    end

    @testset "PowerBlend interpolates between harmonic and min" begin
        # p = 1 is the harmonic blend exactly
        @test PowerBlend{FT}(p = 1)(ℓs) ≈ HarmonicBlend()(ℓs) rtol=1e-5
        # and large p approaches the minimum from below
        @test PowerBlend{FT}(p = 40)(ℓs) ≈ ℓᵐⁱⁿ rtol=1e-2
        # monotone in p
        ps = FT[1, 2, 4, 8, 16]
        blended = [PowerBlend{FT}(p = p)(ℓs) for p in ps]
        @test issorted(blended)
    end

    @testset "every rule is bounded by the smallest branch" begin
        for blend in (MinimumBlend(), HarmonicBlend(), PowerBlend{FT}(p = 2), PowerBlend{FT}(p = 5))
            @test 0 < blend(ℓs) ≤ ℓᵐⁱⁿ + sqrt(eps(FT))
        end
        # An inactive branch (Inf) must not change the result
        @test HarmonicBlend()((ℓs..., FT(Inf))) ≈ HarmonicBlend()(ℓs)
        @test MinimumBlend()((ℓs..., FT(Inf))) == MinimumBlend()(ℓs)
        @test PowerBlend{FT}(p = 2)((ℓs..., FT(Inf))) ≈ PowerBlend{FT}(p = 2)(ℓs)
    end
end

#####
##### Composition
#####

@testset "BlendedMixingLength [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nz = 32
    Lz = FT(1000)
    grid = RectilinearGrid(default_arch; size=Nz, z=(0, Lz), topology=(Flat, Flat, Bounded))
    ℓᵗ_field = Field{Center, Center, Nothing}(grid)
    set!(ℓᵗ_field, 300)
    u★²_field = Field{Center, Center, Nothing}(grid)
    set!(u★²_field, 0.09)
    Jᵇ_field = Field{Center, Center, Nothing}(grid)      # neutral surface
    state = (; ℓᵗ = ℓᵗ_field, u★² = u★²_field, Jᵇ = Jᵇ_field)

    q = FT(0.5)
    N² = FT(1e-4)

    @testset "min is the default blend, and the kwarg overrides it" begin
        @test BlendedMixingLength(GeometricLengthScale()).blend isa MinimumBlend
        @test BlendedMixingLength(GeometricLengthScale();
                                  blend = HarmonicBlend()).blend isa HarmonicBlend
        @test length(BlendedMixingLength(GeometricLengthScale(), BuoyancyLengthScale()).branches) == 2
    end

    @testset "the master length is the blend of its branches" begin
        branches = (GeometricLengthScale{FT}(), TurbulenceLengthScale{FT}(),
                    BuoyancyLengthScale{FT}())
        for blend in (MinimumBlend(), HarmonicBlend(), PowerBlend{FT}(p = 2))
            ml = BlendedMixingLength(branches...; blend)
            for k in (1, 2, 8, Nz)
                ℓs = map(b -> length_scaleᶜᶜᶠ(1, 1, k, grid, b, q, N², state), branches)
                @test mixing_lengthᶜᶜᶠ(1, 1, k, grid, ml, q, N², state) ≈ blend(ℓs)
                @test mixing_lengthᶜᶜᶠ(1, 1, k, grid, ml, q, N², state) ≤
                      minimum(ℓs) + sqrt(eps(FT))
            end
        end
    end

    @testset "a single branch blends to itself" begin
        ml = BlendedMixingLength(GeometricLengthScale{FT}())
        for k in (1, 8, Nz)
            @test mixing_lengthᶜᶜᶠ(1, 1, k, grid, ml, q, N², state) ≈
                  length_scaleᶜᶜᶠ(1, 1, k, grid, GeometricLengthScale{FT}(), q, N², state)
        end
    end

    @testset "Deardorff is expressible" begin
        # ℓ = min(Δ, 0.76 √e / N); with q = √(2e) the stable branch is 0.53 q / N
        ml = BlendedMixingLength(BuoyancyLengthScale{FT}(Cᵇ = 0.53))
        @test mixing_lengthᶜᶜᶠ(1, 1, 4, grid, ml, q, N², state) ≈ FT(0.53) * q / sqrt(N²) rtol=1e-5
    end
end

@testset "NakanishiNiinoLengthScale [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    ml = NakanishiNiinoLengthScale()

    @testset "it is MYNN's three branches under their harmonic blend" begin
        @test ml.blend isa HarmonicBlend
        @test length(ml.branches) == 3
        @test ml.branches[1] isa SurfaceLayerLengthScale
        @test ml.branches[2] isa TurbulenceLengthScale
        @test ml.branches[3] isa BuoyancyLengthScale
    end

    @testset "it carries MYNN's coefficients, not this package's defaults" begin
        # Cᵇ and Cᶜᵇ are exactly where MYNN differs from the branch defaults, which are Deardorff's
        # 0.53 with no convective enhancement.
        @test ml.branches[3].Cᵇ == 1
        @test ml.branches[3].Cᶜᵇ == 5
        @test BuoyancyLengthScale().Cᵇ ≈ 0.53
        @test BuoyancyLengthScale().Cᶜᵇ == 0
        @test ml.branches[2].Cᵗ ≈ 0.23           # MYNN Eq. 54
    end

    @testset "the roughness length propagates" begin
        @test NakanishiNiinoLengthScale(ℓʳ = 0.03).branches[1].ℓʳ ≈ 0.03
        @test ml.branches[1].ℓʳ ≈ 0.1
    end

    @testset "it drops into the closure, with Cq left to the caller" begin
        closure = TKEBasedTurbulenceClosure(FT; mixing_length = NakanishiNiinoLengthScale(), Cq = 3)
        @test isbits(closure)
        @test closure.Cq == 3
        @test closure.mixing_length.blend isa HarmonicBlend
        @test all(b -> eltype(typeof(b).parameters[1]) == FT || typeof(b).parameters[1] == FT,
                  closure.mixing_length.branches)   # convert_eltype reached every branch

        # Cq is not folded into the length scale: the default closure is untouched by it
        @test TKEBasedTurbulenceClosure(FT; mixing_length = NakanishiNiinoLengthScale()).Cq == 1
    end
end

#####
##### The column integral behind ℓᵗ
#####

@testset "ℓᵗ column integral [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    @testset "ℓᵗ is a q-weighted centroid, not a domain height" begin
        # The same boundary layer on a shallow and a deep column must give the same ℓᵗ. With `q`
        # rather than `q - qᵐⁱⁿ` in the integrand, the floored free atmosphere would contribute,
        # and the z weighting would make the deep column's ℓᵗ much larger.
        closure = TKEBasedTurbulenceClosure()
        eᵐⁱⁿ = closure.eᵐⁱⁿ

        turbulent(z) = z < 500 ? 1.0 : eᵐⁱⁿ

        function turbulence_length_scale(Lz, Nz; closure = closure)
            g = RectilinearGrid(default_arch; size=Nz, z=(0, Lz), topology=(Flat, Flat, Bounded))
            e = CenterField(g)
            set!(e, z -> turbulent(z))
            ℓᵗ = Field{Center, Center, Nothing}(g)
            launch!(architecture(g), g, :xy, _compute_turbulence_length_scale!, ℓᵗ, g, closure, e)
            return Array(interior(ℓᵗ))[1]
        end

        ℓᵗ_shallow = turbulence_length_scale(FT(2000), 100)
        ℓᵗ_deep    = turbulence_length_scale(FT(20000), 1000)

        @test isapprox(ℓᵗ_shallow, ℓᵗ_deep; rtol = 1e-2)

        # And it is the centroid of the turbulent layer times Cᵗ: uniform q over 0 ≤ z < 500 has
        # centroid 250
        @test isapprox(ℓᵗ_shallow, TurbulenceLengthScale{FT}().Cᵗ * 250; rtol = 5e-2)

        # A mixing length carrying no TurbulenceLengthScale branch gets ℓᵗ = Inf, which drops out
        # of every blend rather than collapsing ℓ to zero.
        without = TKEBasedTurbulenceClosure(mixing_length =
            BlendedMixingLength(GeometricLengthScale(), BuoyancyLengthScale()))
        @test isnothing(turbulence_length_coefficient(without.mixing_length))

        # κ must be findable from either surface branch, since the example builds its drag law
        # from it; a mixing length with neither reports `nothing` rather than a wrong number.
        @test von_karman_constant(BlendedMixingLength(GeometricLengthScale())) == 0.4
        @test von_karman_constant(BlendedMixingLength(SurfaceLayerLengthScale())) == 0.4
        @test isnothing(von_karman_constant(BlendedMixingLength(BuoyancyLengthScale())))
        @test isinf(turbulence_length_scale(FT(2000), 100; closure = without))
    end
end


#####
##### Prandtl number
#####

@testset "Turbulent Prandtl number [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    closure = TKEBasedTurbulenceClosure()
    Pr₀ = closure.Pr₀
    CRi = closure.CRi

    @test turbulent_prandtl_number(Pr₀, CRi, FT(0)) ≈ Pr₀
    @test turbulent_prandtl_number(Pr₀, CRi, FT(-1)) ≈ Pr₀       # unstable: no enhancement
    @test turbulent_prandtl_number(Pr₀, CRi, FT(1)) ≈ Pr₀ * (1 + CRi / 2)
    @test turbulent_prandtl_number(Pr₀, CRi, FT(1e8)) ≈ Pr₀ * (1 + CRi) rtol=1e-6

    # Monotone increasing in Ri, and bounded
    Pr = [turbulent_prandtl_number(Pr₀, CRi, FT(Ri)) for Ri in 0:0.5:20]
    @test issorted(Pr)
    @test maximum(Pr) ≤ Pr₀ * (1 + CRi)
end

#####
##### Model integration
#####

@testset "TKEBasedTurbulenceClosure in an AtmosphereModel [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    grid = RectilinearGrid(default_arch; size=32, z=(0, 2000), topology=(Flat, Flat, Bounded))

    @testset "tracer and closure-field wiring" begin
        model = AtmosphereModel(grid; closure = TKEBasedTurbulenceClosure())

        # `closure_required_tracers` is threaded through `AtmosphereModel`, so the user does not
        # have to remember to ask for the TKE tracer
        @test TKE_NAME ∈ keys(model.tracers)
        @test TKE_NAME ∈ keys(prognostic_fields(model))

        for name in (:νₑ, :κₑ, :ℓ, :e, :ℓᵗ, :u★²)
            @test name ∈ propertynames(model.closure_fields)
        end

        # A user tracer coexists with the closure's
        model = AtmosphereModel(grid; closure = TKEBasedTurbulenceClosure(), tracers = :ρc)
        @test TKE_NAME ∈ keys(model.tracers)
        @test :ρc ∈ keys(model.tracers)
    end

    @testset "positivity and finiteness over 20 steps" begin
        model = AtmosphereModel(grid; closure = TKEBasedTurbulenceClosure(),
                                coriolis = FPlane(latitude = 45))

        ## A sheared profile, so shear production is actually nonzero — a uniform wind gives
        ## S² = 0 and the column would sit at the TKE floor no matter what the closure did.
        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ = z -> θ₀ + max(0, z - 1000) * FT(0.008), ρu = z -> FT(10) * z / 2000)

        for _ in 1:20
            time_step!(model, 10)
        end

        e = Array(interior(model.closure_fields.e))
        ν = Array(interior(model.closure_fields.νₑ))
        κ = Array(interior(model.closure_fields.κₑ))
        ℓ = Array(interior(model.closure_fields.ℓ))
        ℓᶜ = Array(interior(model.closure_fields.ℓᶜ))
        ρe = Array(interior(model.tracers[TKE_NAME]))

        @test all(isfinite, e) && all(isfinite, ν) && all(isfinite, κ) && all(isfinite, ℓ)
        @test all(≥(0), e)
        @test all(≥(0), ρe)
        @test all(≥(0), ν)
        @test all(≥(0), κ)

        # `ℓ` is masked to zero on the bottom boundary face and left at zero on the top one, where
        # `ν` is zero anyway; the dissipation reads `ℓᶜ`, which must be strictly positive in every
        # cell or `ε` blows up.
        @test all(>(0), ℓ[2:end-1])
        @test all(>(0), ℓᶜ)
        @test all(isfinite, ℓᶜ)

        # Shear drives turbulence: something must have grown past the floor
        @test maximum(e) > 10 * TKEBasedTurbulenceClosure().eᵐⁱⁿ

        # Pr ≥ Pr₀ everywhere, so K never exceeds ν/Pr₀
        @test all(κ .≤ ν ./ TKEBasedTurbulenceClosure().Pr₀ .+ sqrt(eps(FT)))
    end

    @testset "surface TKE floor tracks u★²" begin
        # A momentum flux at the bottom sets u★², and the floor holds e(z₁) ≥ Csfc u★²
        τˣ = FT(-0.1)   # kg m⁻¹ s⁻², downward momentum flux
        ρu_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(τˣ))
        closure = TKEBasedTurbulenceClosure()
        model = AtmosphereModel(grid; closure, boundary_conditions = (; ρu = ρu_bcs))

        time_step!(model, 1)

        ρ₁ = Array(interior(Breeze.AtmosphereModels.total_density(model.dynamics)))[1]
        u★² = abs(τˣ) / ρ₁
        @test Array(interior(model.closure_fields.u★²))[1] ≈ u★² rtol=1e-5

        e₁ = Array(interior(model.closure_fields.e))[1]
        @test e₁ ≥ surface_tke_coefficient(closure) * u★² * (1 - sqrt(eps(FT)))
    end

    @testset "K = ν/Pr holds pointwise" begin
        # Nothing clips ν or K, so the Prandtl number the closure advertises is exactly the one
        # the fields carry — in a neutral column that is Pr₀ everywhere.
        closure = TKEBasedTurbulenceClosure()
        model = AtmosphereModel(grid; closure, coriolis = FPlane(latitude = 45))
        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ = z -> θ₀, ρu = z -> FT(20) * z / 2000)

        for _ in 1:5
            time_step!(model, 10)
        end

        ν = Array(interior(model.closure_fields.νₑ))
        κ = Array(interior(model.closure_fields.κₑ))

        @test maximum(ν) > 0                            # the column is actually mixing
        @test all(isapprox.(κ, ν ./ closure.Pr₀; rtol = sqrt(eps(FT))))
    end

    @testset "Cq scales the TKE diffusivity only" begin
        # MYNN transport TKE at S_q = 3S_M (their Eq. 67). We default to Cq = 1, and the TKE tracer
        # carries its own field so that momentum and heat are untouched by the choice.
        for Cq in (FT(1), FT(3))
            closure = TKEBasedTurbulenceClosure(FT; Cq)
            model = AtmosphereModel(grid; closure, coriolis = FPlane(latitude = 45))
            θ₀ = model.dynamics.reference_state.potential_temperature
            set!(model; θ = z -> θ₀, ρu = z -> FT(20) * z / 2000)

            for _ in 1:5
                time_step!(model, 10)
            end

            ν = Array(interior(model.closure_fields.νₑ))
            νᵗ = Array(interior(model.closure_fields.νₑᵗ))
            κ = Array(interior(model.closure_fields.κₑ))

            @test maximum(ν) > 0                        # the column is actually mixing
            @test all(isapprox.(νᵗ, Cq .* ν; rtol = sqrt(eps(FT))))
            @test all(isapprox.(κ, ν ./ closure.Pr₀; rtol = sqrt(eps(FT))))   # heat unaffected

            # and the TKE tracer is the field that reads it
            @test model.closure_fields.tupled_tracer_diffusivities[TKE_NAME] ===
                  model.closure_fields.νₑᵗ
        end

        @test TKEBasedTurbulenceClosure().Cq == 1        # shipped default leaves TKE as momentum
    end

    @testset "alternative constant sets run" begin
        for closure in (MY82Coefficients(), MYJCoefficients())
            model = AtmosphereModel(grid; closure)
            time_step!(model, 10)
            @test all(isfinite, Array(interior(model.closure_fields.νₑ)))
        end
    end
end
