include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.TurbulenceClosures: TKE_NAME, TKEClosureFields
using Oceananigans
using Oceananigans.TimeSteppers: update_state!, time_discretization
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, ExplicitTimeDiscretization
using Oceananigans.Units
using Test

#####
##### Construction
#####

@testset "TKEBasedTurbulenceClosure construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    closure = TKEBasedTurbulenceClosure()
    @test closure isa TKEBasedTurbulenceClosure
    @test time_discretization(closure) isa VerticallyImplicitTimeDiscretization
    @test closure.mixing_length isa TKEMixingLength{FT}
    @test closure.stability_functions isa ConstantStabilityFunctions{FT}

    @testset "defaults and the constants they imply" begin
        sf = closure.stability_functions
        @test closure.mixing_length.Cᴺ ≈ 0.76
        @test (sf.Cᵘ, sf.Cᶜ, sf.Cᵉ, sf.Cᴰ) == FT.((0.196, 0.265, 0.392, 0.295))
        @test closure.minimum_tke == FT(1e-6)
        @test closure.negative_tke_damping_time_scale == FT(60)
        @test isinf(closure.maximum_viscosity)
        @test isinf(closure.maximum_tracer_diffusivity)
        @test isinf(closure.maximum_tke_diffusivity)

        # The neutral log layer: von Kármán constant, surface TKE, Prandtl number
        @test (sf.Cᵘ^3 / sf.Cᴰ)^(1/4) ≈ 0.40 atol=0.005
        @test 1 / sqrt(sf.Cᵘ * sf.Cᴰ) ≈ 4.2 atol=0.05
        @test sf.Cᵘ / sf.Cᶜ ≈ 0.74 atol=0.005
    end

    @testset "keyword arguments, promotion and float type" begin
        closure = TKEBasedTurbulenceClosure(; mixing_length = TKEMixingLength(Cᴺ = 1),
                                              stability_functions = ConstantStabilityFunctions(Cᵘ = 0.3, Cᶜ = 0.3, Cᵉ = 1, Cᴰ = 1),
                                              maximum_viscosity = 100,
                                              minimum_tke = 1e-8,
                                              negative_tke_damping_time_scale = 10minutes)
        @test closure.mixing_length.Cᴺ === FT(1)
        @test closure.stability_functions.Cᵘ === FT(0.3)
        @test closure.stability_functions.Cᵉ === FT(1)
        @test closure.maximum_viscosity === FT(100)
        @test closure.minimum_tke === FT(1e-8)
        @test closure.negative_tke_damping_time_scale === FT(600)

        explicit = TKEBasedTurbulenceClosure(ExplicitTimeDiscretization())
        @test time_discretization(explicit) isa ExplicitTimeDiscretization

        closure32 = TKEBasedTurbulenceClosure(Float32)
        @test closure32.mixing_length isa TKEMixingLength{Float32}
        @test closure32.stability_functions isa ConstantStabilityFunctions{Float32}
        @test closure32.minimum_tke isa Float32
    end

    @testset "isbits and show" begin
        @test isbits(closure)
        @test isbits(TKEMixingLength())
        @test isbits(ConstantStabilityFunctions())
        @test summary(closure) == "TKEBasedTurbulenceClosure{VerticallyImplicitTimeDiscretization}"
        str = sprint(show, closure)
        @test occursin("Cᴺ", str)
        @test occursin("ConstantStabilityFunctions", str)
        @test occursin("minimum_tke", str)
        @test occursin("Cᴰ", sprint(show, ConstantStabilityFunctions()))
        @test occursin("Cᴺ", sprint(show, TKEMixingLength()))
    end
end

#####
##### In an AtmosphereModel
#####

# Set a uniform specific TKE `e₀` and refresh the closure fields
function set_tke!(model, e₀)
    ρ = model.dynamics.reference_state.density
    set!(model.tracers[TKE_NAME], e₀)
    parent(model.tracers[TKE_NAME]) .*= parent(ρ)
    update_state!(model)
    return nothing
end

column(field) = Array(interior(field, 1, 1, :))

@testset "TKEBasedTurbulenceClosure in an AtmosphereModel [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 32
    Lz = 1000
    grid = RectilinearGrid(default_arch; size = Nz, z = (0, Lz), topology = (Flat, Flat, Bounded))
    zf = znodes(grid, Face())
    closure = TKEBasedTurbulenceClosure()

    @testset "the TKE tracer and the closure fields" begin
        model = AtmosphereModel(grid; closure)
        @test TKE_NAME === :ρe
        @test :ρe ∈ keys(model.tracers)
        @test :ρe ∈ keys(Oceananigans.prognostic_fields(model))
        @test model.closure_fields isa TKEClosureFields
        @test keys(model.closure_fields.tupled_tracer_diffusivities) == (:ρθ, :ρqᵛ, :ρe)
        @test model.closure_fields.tupled_tracer_diffusivities.ρe === model.closure_fields.Kᵉ
        @test model.closure_fields.tupled_tracer_diffusivities.ρθ === model.closure_fields.Kᶜ

        # A user tracer coexists with the closure's, and naming the closure's tracer is harmless
        model = AtmosphereModel(grid; closure, tracers = :ρc)
        @test :ρc ∈ keys(model.tracers)
        @test :ρe ∈ keys(model.tracers)
        model = AtmosphereModel(grid; closure, tracers = :ρe)
        @test count(==(:ρe), keys(Oceananigans.prognostic_fields(model))) == 1

        # Prognostic names must be unique: a tracer cannot take the name of another prognostic
        @test_throws ArgumentError AtmosphereModel(grid; closure, tracers = :ρqᵛ)
    end

    @testset "mixing length and diffusivities in a neutral column" begin
        model = AtmosphereModel(grid; closure, advection = nothing)
        e₀ = FT(0.5)
        set!(model; θ = 300)
        set_tke!(model, e₀)

        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        Kᵉ = column(model.closure_fields.Kᵉ)
        sf = closure.stability_functions
        interior_faces = 2:Nz

        # Neutral air: the mixing length is the height above the surface. It is not stored
        # (as in CATKE), so it is tested through Kᵘ = Cᵘ z √e, masked on the boundary faces.
        @test Kᵘ[1] == 0
        @test Kᵘ[Nz+1] == 0
        @test all(Kᵘ[interior_faces] .≈ sf.Cᵘ .* zf[interior_faces] .* sqrt(e₀))

        # The ratios Kᶜ/Kᵘ, Kᵉ/Kᵘ are the stability-function ratios
        @test all(Kᶜ[interior_faces] ./ Kᵘ[interior_faces] .≈ sf.Cᶜ / sf.Cᵘ)
        @test all(Kᵉ[interior_faces] ./ Kᵘ[interior_faces] .≈ sf.Cᵉ / sf.Cᵘ)
    end

    @testset "the stratification length" begin
        model = AtmosphereModel(grid; closure, advection = nothing)
        e₀ = FT(0.5)
        Γ = FT(0.005)
        set!(model; θ = z -> 300 + Γ * z)
        set_tke!(model, e₀)

        # The mixing length is not stored; diagnose it from Kᵘ = Cᵘ ℓ √e with the uniform √e₀
        Kᵘ = column(model.closure_fields.Kᵘ)
        ℓ = Kᵘ ./ (closure.stability_functions.Cᵘ * sqrt(e₀))
        θ = column(model.formulation.potential_temperature)
        g = model.thermodynamic_constants.gravitational_acceleration
        Δz = Lz / Nz
        Cᴺ = closure.mixing_length.Cᴺ

        for k in 2:Nz
            N² = g * (log(θ[k]) - log(θ[k-1])) / Δz
            ℓᴺ = Cᴺ * sqrt(e₀) / sqrt(N²)
            @test ℓ[k] ≈ min(zf[k], ℓᴺ) rtol=1e-5
        end

        # Stratification limits the length well above the surface
        @test ℓ[Nz] < zf[Nz] / 2
    end

    @testset "diffusivity caps" begin
        capped = TKEBasedTurbulenceClosure(maximum_viscosity = 1e-3, maximum_tracer_diffusivity = 2e-3,
                                           maximum_tke_diffusivity = 3e-3)
        model = AtmosphereModel(grid; closure = capped, advection = nothing)
        set!(model; θ = 300)
        set_tke!(model, FT(1))
        @test maximum(column(model.closure_fields.Kᵘ)) ≈ 1e-3
        @test maximum(column(model.closure_fields.Kᶜ)) ≈ 2e-3
        @test maximum(column(model.closure_fields.Kᵉ)) ≈ 3e-3
    end

    @testset "a sheared, capped boundary layer stays finite and positive" begin
        model = AtmosphereModel(grid; closure, advection = nothing)
        θᵢ(z) = 300 + 0.01 * max(0, z - 500)
        uᵢ(z) = 5 * min(1, z / 300)
        set!(model; θ = θᵢ, u = uᵢ)
        set_tke!(model, FT(0.1))

        for _ in 1:20
            time_step!(model, 10)
        end

        ρe = column(model.tracers.ρe)
        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        @test all(isfinite, ρe)
        @test all(ρe .≥ 0)
        @test all(isfinite, Kᵘ)
        @test maximum(Kᵘ) > 0
        sf = closure.stability_functions
        @test all(Kᶜ[2:Nz] .≈ Kᵘ[2:Nz] .* (sf.Cᶜ / sf.Cᵘ))
    end

    @testset "the implicit linear coefficient is the dissipation rate" begin
        model = AtmosphereModel(grid; closure, advection = nothing)
        e₀ = FT(0.5)
        set!(model; θ = 300)
        set_tke!(model, e₀)

        @test model.closure_fields.tupled_implicit_linear_coefficients.ρe === model.closure_fields.Le
        @test model.closure_fields.tupled_implicit_linear_coefficients.ρθ isa Oceananigans.Fields.ZeroField

        # Neutral air, no buoyancy flux: Le = - Sᴰ √e / ℓ with ℓ = z at the cell centers
        Le = column(model.closure_fields.Le)
        zc = znodes(grid, Center())
        Cᴰ = closure.stability_functions.Cᴰ
        @test all(Le .≈ -Cᴰ * sqrt(e₀) ./ zc)
        @test all(Le .< 0)

        # Below the minimum TKE the dissipation rate keeps following √e — the floor applies to
        # the diffusivities, not to the dissipation (as in CATKE)
        e₋ = FT(1e-8)
        @test e₋ < closure.minimum_tke
        set_tke!(model, e₋)
        Le = column(model.closure_fields.Le)
        @test all(Le .≈ -Cᴰ * sqrt(e₋) ./ zc)
    end

    @testset "the sources enter the stage tendency" begin
        model = AtmosphereModel(grid; closure, advection = nothing)
        e₀ = FT(0.5)
        S = FT(0.01)
        set!(model; θ = 300, u = z -> S * z)
        set_tke!(model, e₀)

        Gρe = model.timestepper.Gⁿ.ρe
        fill!(Gρe, 0)
        Breeze.AtmosphereModels.compute_closure_tendencies!(model)

        # Neutral: only shear production, P = Kᵘ S² at the faces averaged to the centers; in the
        # first and last cells the masked boundary face is replaced by its interior neighbour
        ρ = column(model.dynamics.reference_state.density)
        Kᵘ = column(model.closure_fields.Kᵘ)
        P = [S^2 * (Kᵘ[k] + Kᵘ[k+1]) / 2 for k in 1:Nz]
        P[1] = S^2 * Kᵘ[2]
        P[Nz] = S^2 * Kᵘ[Nz]
        @test all(column(Gρe) .≈ ρ .* P)
        @test all(column(Gρe) .> 0)
    end

    @testset "dissipation decays TKE at the analytic rate" begin
        # No TKE diffusion, no wind, no stratification: each cell decays as ∂ₜe = -Cᴰ e^{3/2}/ℓ
        # with ℓ = z, i.e. e(t) = (e₀^{-1/2} + Cᴰ t / 2ℓ)^{-2}
        undiffused = TKEBasedTurbulenceClosure(maximum_tke_diffusivity = 0)
        model = AtmosphereModel(grid; closure = undiffused, advection = nothing)
        e₀ = FT(1)
        set!(model; θ = 300)
        set_tke!(model, e₀)

        Δt = FT(1)
        Nt = 100
        for _ in 1:Nt
            time_step!(model, Δt)
        end

        e = column(model.tracers.ρe) ./ column(model.dynamics.reference_state.density)
        zc = znodes(grid, Center())
        Cᴰ = undiffused.stability_functions.Cᴰ
        t = Nt * Δt
        e_analytic = @. (e₀^(-1/2) + Cᴰ * t / (2 * zc))^(-2)
        # The sinks are implicit per stage, so the decay is first-order accurate in Δt ω ≈ 0.02
        @test all(e .≥ 0)
        @test all(isapprox.(e, e_analytic; rtol = 0.05))
    end

    @testset "negative TKE is damped on the damping time scale" begin
        τ = FT(60)
        model = AtmosphereModel(grid; closure, advection = nothing)
        set!(model; θ = 300)
        set_tke!(model, FT(-0.1))

        # No shear, no stratification and no TKE gradient, so the damping rate 1/τ is the only
        # term. It is applied implicitly in each of the three SSP-RK3 stages with αΔt = Δt, Δt/4,
        # 2Δt/3, and the stage combinations make one step of length τ multiply a uniform e by
        # [1/3 + 2/3 (3/4 + 1/4 (1/2)) / (5/4)] / (5/3) = 0.48
        time_step!(model, τ)
        e = column(model.tracers.ρe) ./ column(model.dynamics.reference_state.density)
        @test all(e .≈ -0.048)
    end

    @testset "explicit time discretization" begin
        explicit = TKEBasedTurbulenceClosure(ExplicitTimeDiscretization())
        model = AtmosphereModel(grid; closure = explicit, advection = nothing)
        set!(model; θ = z -> 300 + 0.003z, u = z -> 5 * min(1, z / 300))
        set_tke!(model, FT(0.1))
        for _ in 1:5
            time_step!(model, 1)
        end
        @test all(isfinite, column(model.tracers.ρe))
        @test all(column(model.tracers.ρe) .≥ 0)
    end

    @testset "compressible dynamics" begin
        dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = 300)
        model = AtmosphereModel(grid; dynamics, closure)
        @test keys(model.closure_fields.tupled_tracer_diffusivities) == (:ρθ, :ρqᵛ, :ρe)
        set!(model; θ = 300, ρ = model.dynamics.reference_state.density)
        time_step!(model, 1)
        @test model.clock.iteration == 1
        @test all(isfinite, Array(interior(model.dynamics.dry_density)))
        @test all(isfinite, Array(interior(model.tracers.ρe)))
    end
end
