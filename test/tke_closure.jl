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
            @test length_scaleᶜᶜᶠ(1, 1, k, grid, geometric, FT(1), FT(0), ℓᵗ_field) ≈ κ * (z + ℓʳ)
        end
        # Finite at the surface — this is what ℓʳ is for
        @test length_scaleᶜᶜᶠ(1, 1, 1, grid, geometric, FT(1), FT(0), ℓᵗ_field) ≈ κ * ℓʳ
        @test length_scaleᶜᶜᶠ(1, 1, 1, grid, geometric, FT(1), FT(0), ℓᵗ_field) > 0
        # Centers sit half a cell above the face below them
        @test length_scaleᶜᶜᶜ(1, 1, 1, grid, geometric, FT(1), FT(0), ℓᵗ_field) ≈ κ * (Δz / 2 + ℓʳ)
    end

    @testset "ℓᵇ = Cᵇ q / N" begin
        Cᵇ = buoyancy.Cᵇ
        q = FT(1)
        ℓᵇ(q, N²) = length_scaleᶜᶜᶠ(1, 1, 1, grid, buoyancy, q, N², ℓᵗ_field)

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
        @test length_scaleᶜᶜᶠ(1, 1, 1, grid, BuoyancyLengthScale{FT}(Cᵇ = 1), q, N², ℓᵗ_field) ≈
              q / sqrt(N²) rtol=1e-5
    end

    @testset "ℓᵗ is read from the column field, at both locations" begin
        for k in (1, 8, Nz)
            @test length_scaleᶜᶜᶠ(1, 1, k, grid, turbulence, FT(1), FT(0), ℓᵗ_field) == 300
            @test length_scaleᶜᶜᶜ(1, 1, k, grid, turbulence, FT(1), FT(0), ℓᵗ_field) == 300
        end
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
                ℓs = map(b -> length_scaleᶜᶜᶠ(1, 1, k, grid, b, q, N², ℓᵗ_field), branches)
                @test mixing_lengthᶜᶜᶠ(1, 1, k, grid, ml, q, N², ℓᵗ_field) ≈ blend(ℓs)
                @test mixing_lengthᶜᶜᶠ(1, 1, k, grid, ml, q, N², ℓᵗ_field) ≤
                      minimum(ℓs) + sqrt(eps(FT))
            end
        end
    end

    @testset "a single branch blends to itself" begin
        ml = BlendedMixingLength(GeometricLengthScale{FT}())
        for k in (1, 8, Nz)
            @test mixing_lengthᶜᶜᶠ(1, 1, k, grid, ml, q, N², ℓᵗ_field) ≈
                  length_scaleᶜᶜᶠ(1, 1, k, grid, GeometricLengthScale{FT}(), q, N², ℓᵗ_field)
        end
    end

    @testset "Deardorff is expressible" begin
        # ℓ = min(Δ, 0.76 √e / N); with q = √(2e) the stable branch is 0.53 q / N
        ml = BlendedMixingLength(BuoyancyLengthScale{FT}(Cᵇ = 0.53))
        @test mixing_lengthᶜᶜᶠ(1, 1, 4, grid, ml, q, N², ℓᵗ_field) ≈ FT(0.53) * q / sqrt(N²) rtol=1e-5
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

    @testset "alternative constant sets run" begin
        for closure in (MY82Coefficients(), MYJCoefficients())
            model = AtmosphereModel(grid; closure)
            time_step!(model, 10)
            @test all(isfinite, Array(interior(model.closure_fields.νₑ)))
        end
    end
end
