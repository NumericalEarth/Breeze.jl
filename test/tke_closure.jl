using Breeze
using Oceananigans
using Test

using Breeze.TurbulenceClosures: TKE_NAME,
                                 buoyancy_length_scaleᶜᶜᶠ,
                                 geometric_length_scaleᶜᶜᶠ,
                                 mixing_lengthᶜᶜᶠ,
                                 smooth_positive,
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
        @test isbits(MesoscaleLengthScale())
    end

    @testset "show" begin
        closure = TKEBasedTurbulenceClosure()
        @test occursin("TKEBasedTurbulenceClosure", summary(closure))
        @test occursin("Cˢ", sprint(show, closure))
        @test occursin("MesoscaleLengthScale", sprint(show, closure.mixing_length))
    end
end

#####
##### Mixing length, branch by branch
#####

@testset "MesoscaleLengthScale branches [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nz = 32
    Lz = FT(1000)
    grid = RectilinearGrid(default_arch; size=Nz, z=(0, Lz), topology=(Flat, Flat, Bounded))
    mixing_length = MesoscaleLengthScale{FT}()

    @testset "smooth_positive" begin
        δ = FT(1e-9)
        @test smooth_positive(FT(1), δ) ≈ 1
        @test smooth_positive(FT(-1), δ) < 1e-9
        @test smooth_positive(FT(0), δ) ≈ δ / 2
        # monotone and never negative
        @test all(smooth_positive(x, δ) ≥ 0 for x in FT(-1):FT(0.1):FT(1))
    end

    @testset "ℓᵍ = κ(z + ℓʳ)" begin
        κ = mixing_length.κ
        ℓʳ = mixing_length.ℓʳ
        Δz = Lz / Nz
        # Face k sits at z = (k - 1) Δz
        for k in (1, 2, 8, Nz)
            z = (k - 1) * Δz
            @test geometric_length_scaleᶜᶜᶠ(1, 1, k, grid, mixing_length) ≈ κ * (z + ℓʳ)
        end
        # Finite at the surface — this is what ℓʳ is for
        @test geometric_length_scaleᶜᶜᶠ(1, 1, 1, grid, mixing_length) ≈ κ * ℓʳ
        @test geometric_length_scaleᶜᶜᶠ(1, 1, 1, grid, mixing_length) > 0
    end

    @testset "ℓᵇ = Cᵇ q / N" begin
        Cᵇ = mixing_length.Cᵇ
        q = FT(1)

        N² = FT(1e-4)
        @test buoyancy_length_scaleᶜᶜᶠ(1, 1, 1, grid, mixing_length, q, N²) ≈ Cᵇ * q / sqrt(N²) rtol=1e-5

        # Scales like q and like 1/N
        @test buoyancy_length_scaleᶜᶜᶠ(1, 1, 1, grid, mixing_length, 2q, N²) ≈
              2 * buoyancy_length_scaleᶜᶜᶠ(1, 1, 1, grid, mixing_length, q, N²) rtol=1e-5
        @test buoyancy_length_scaleᶜᶜᶠ(1, 1, 1, grid, mixing_length, q, 4N²) ≈
              buoyancy_length_scaleᶜᶜᶠ(1, 1, 1, grid, mixing_length, q, N²) / 2 rtol=1e-5

        # Inactive in unstable and neutral air: the branch returns a length so large that it
        # drops out of the harmonic blend
        ℓᵇ_unstable = buoyancy_length_scaleᶜᶜᶠ(1, 1, 1, grid, mixing_length, q, FT(-1e-4))
        ℓᵇ_neutral  = buoyancy_length_scaleᶜᶜᶠ(1, 1, 1, grid, mixing_length, q, FT(0))
        @test ℓᵇ_unstable > 1e6
        @test ℓᵇ_neutral > 1e3
        @test ℓᵇ_unstable > ℓᵇ_neutral   # smooth, monotone through N² = 0
    end

    @testset "harmonic blend is bounded by every branch" begin
        ℓᵗ_field = Field{Center, Center, Nothing}(grid)
        set!(ℓᵗ_field, 300)
        q = FT(0.5)
        N² = FT(1e-4)

        for k in (2, 8, 20, Nz)
            ℓ = mixing_lengthᶜᶜᶠ(1, 1, k, grid, mixing_length, q, N², ℓᵗ_field)
            ℓᵍ = geometric_length_scaleᶜᶜᶠ(1, 1, k, grid, mixing_length)
            ℓᵇ = buoyancy_length_scaleᶜᶜᶠ(1, 1, k, grid, mixing_length, q, N²)
            @test ℓ > 0
            @test ℓ ≤ min(ℓᵍ, ℓᵇ, FT(300)) + sqrt(eps(FT))
        end
    end

    @testset "ℓᵗ is a q-weighted centroid, not a domain height" begin
        # The same boundary layer on a shallow and a deep column must give the same ℓᵗ. With `q`
        # rather than `q - qᵐⁱⁿ` in the integrand, the floored free atmosphere would contribute,
        # and the z weighting would make the deep column's ℓᵗ much larger.
        closure = TKEBasedTurbulenceClosure()
        eᵐⁱⁿ = closure.eᵐⁱⁿ

        turbulent(z) = z < 500 ? 1.0 : eᵐⁱⁿ

        function turbulence_length_scale(Lz, Nz)
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
        @test isapprox(ℓᵗ_shallow, mixing_length.Cᵗ * 250; rtol = 5e-2)
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
        @test maximum(ν) ≤ TKEBasedTurbulenceClosure().νᵐᵃˣ

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

    @testset "νᵐᵃˣ caps the diffusivity as well as the viscosity" begin
        # `K = ν/Pr` with Pr ≥ Pr₀ < 1 would otherwise exceed the stated ceiling by 1/Pr₀
        closure = TKEBasedTurbulenceClosure(νᵐᵃˣ = 1e-3)
        model = AtmosphereModel(grid; closure, coriolis = FPlane(latitude = 45))
        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ = z -> θ₀, ρu = z -> FT(20) * z / 2000)

        for _ in 1:5
            time_step!(model, 10)
        end

        @test maximum(Array(interior(model.closure_fields.νₑ))) ≤ closure.νᵐᵃˣ
        @test maximum(Array(interior(model.closure_fields.κₑ))) ≤ closure.νᵐᵃˣ
    end

    @testset "alternative constant sets run" begin
        for closure in (MY82Coefficients(), MYJCoefficients())
            model = AtmosphereModel(grid; closure)
            time_step!(model, 10)
            @test all(isfinite, Array(interior(model.closure_fields.νₑ)))
        end
    end
end
