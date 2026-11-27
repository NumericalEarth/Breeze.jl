include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: materialize_closure, closure_scalar_names
using Breeze.TurbulenceClosures: shoc_distribution
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: trainable_parameter_names
using Oceananigans
using Oceananigans.Grids: ColumnEnsembleSize
using Test

#####
##### SimplifiedHigherOrderClosure: SHOC as a configuration of TKEBasedTurbulenceClosure
#####

const Nz = 24

# A moist convective boundary layer with wind, whose surface fluxes are the model's bottom boundary
# conditions, with saturation adjustment
function shoc_boundary_layer_model(grid; closure = SimplifiedHigherOrderClosure(eltype(grid)))
    FT = eltype(grid)
    θᵢ(z) = 298.7 + 4e-3 * max(z - 600, 0)
    qᵢ(z) = 0.017 - 4e-6 * z
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure = 101500, potential_temperature = θᵢ,
                                     vapor_mass_fraction = qᵢ)
    ρ₀ = FT(1.16)
    flux(J) = FieldBoundaryConditions(bottom = FluxBoundaryCondition(FT(J)))
    boundary_conditions = (ρθ = flux(ρ₀ * 8e-3), ρqᵗ = flux(ρ₀ * 5.2e-5), ρu = flux(ρ₀ * 0.28^2), ρv = flux(0))
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), closure, boundary_conditions,
                            microphysics = SaturationAdjustment(FT; equilibrium = WarmPhaseEquilibrium()))
    set!(model, θ = (args...) -> θᵢ(last(args)), qᵗ = (args...) -> qᵢ(last(args)), u = -8)
    e₀ = CenterField(grid)
    set!(e₀, (args...) -> max(0.5 * (1 - last(args) / 1200), 0))
    set!(model.tracers.ρe, reference_state.density * e₀)
    return model
end

column_data(field) = Array(interior(field))

@testset "SimplifiedHigherOrderClosure [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    @testset "A configuration of TKEBasedTurbulenceClosure" begin
        closure = SimplifiedHigherOrderClosure(FT)
        @test closure isa SimplifiedHigherOrderClosure
        @test closure isa TKEBasedTurbulenceClosure
        @test SHOC === SimplifiedHigherOrderClosure
        @test !(TKEBasedTurbulenceClosure(FT) isa SimplifiedHigherOrderClosure)
        @test !(TKEBasedTurbulenceClosure(FT; subgrid_distribution = DoubleGaussian(FT),
                                          static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()))
                isa SimplifiedHigherOrderClosure)
        @test isbits(closure)
        @test closure.mixing_length isa BlendedMixingLength{FT}
        @test closure.stability_functions isa ColumnStabilityDamping{FT}
        @test closure.subgrid_distribution isa DoubleGaussian
        d = closure.subgrid_distribution
        @test d.time_scale isa ReturnToIsotropyTimeScale
        @test d.second_moments isa DowngradientSecondMoments{FT}
        @test d.third_moment isa AlgebraicThirdMoment{FT}
        @test d.skewness isa PolynomialSaturatedSkewness{FT}
        @test d.stratification_number_bound isa RationalSaturatedStratificationNumber{FT}
        @test d.assumed_pdf isa DoubleGaussianPDF{FT}
        @test closure.static_stability isa SaturationWeightedStaticStability

        # The published coefficients of shoc.F90
        p = closure_parameters(closure)
        @test p.ℂᶜ == FT(0.1) && p.ℂᵘ == FT(0.1)
        @test p.ℂˡ₁ == FT(0.5) && p.ℂˡ₅ == FT(400)
        @test p.ℂᵝ₁ == FT(0.001) && p.ℂᵝ₂ == FT(0.08) && p.ℂᵝ₃ == FT(2.65) && p.ℂᵝ₄ == FT(0.02)
        @test p.ℂʷ₂ == FT(7) && p.ℂᵍ₁ == FT(0.4)
        @test :ℂˡ₅ ∉ trainable_parameter_names(closure)

        # show
        shown = sprint(show, closure)
        @test startswith(shown, "SimplifiedHigherOrderClosure{VerticallyImplicitTimeDiscretization}")
        @test occursin("mixing_length: BlendedMixingLength{$FT}", shown)
        @test occursin("stability_functions: ColumnStabilityDamping{$FT}", shown)
        @test occursin("third_moment: AlgebraicThirdMoment{$FT}", shown)
        @test summary(closure) == "SimplifiedHigherOrderClosure{VerticallyImplicitTimeDiscretization}"
        @test summary(TKEBasedTurbulenceClosure(FT)) == "TKEBasedTurbulenceClosure{VerticallyImplicitTimeDiscretization}"

        # Every component is a keyword argument
        swapped = SimplifiedHigherOrderClosure(FT; mixing_length = BlendedMixingLength(FT; ℂˡ₁ = 0.4),
                                               stability_functions = ColumnStabilityDamping(FT; ℂᶜ = 0.2),
                                               subgrid_distribution = shoc_distribution(FT; skewness = TanhSaturatedSkewness(FT)))
        @test swapped isa SimplifiedHigherOrderClosure
        @test swapped.mixing_length.ℂˡ₁ == FT(0.4)
        @test swapped.stability_functions.ℂᶜ == FT(0.2)
        @test swapped.subgrid_distribution.skewness isa TanhSaturatedSkewness
        # Another mixing length or other stability functions give a TKE closure with SHOC's other components
        other = SimplifiedHigherOrderClosure(FT; mixing_length = GradientLimitedMixingLength(FT(1.3)))
        @test other isa TKEBasedTurbulenceClosure && !(other isa SimplifiedHigherOrderClosure)
        # The moment time scale of the stability functions needs stability functions that have one
        @test_throws ArgumentError SimplifiedHigherOrderClosure(FT; stability_functions = TurbulentRichardsonStabilityFunction())

        # with_parameters keeps the configuration
        rebuilt = with_parameters(closure; ℂᶜ = 0.2, ℂʷ₂ = 6.0)
        @test rebuilt isa SimplifiedHigherOrderClosure
        @test rebuilt.stability_functions.ℂᶜ == FT(0.2)
        @test rebuilt.subgrid_distribution.third_moment.ℂʷ₂ == FT(6)
        @test_throws ArgumentError with_parameters(closure; ℂ_not_a_coefficient = 1)

        # The float type
        @test SimplifiedHigherOrderClosure(Float32).mixing_length.ℂˡ₅ === 400f0
        @test SimplifiedHigherOrderClosure(Float32).stability_functions.ℂᴰ₂ === 0.15f0
    end

    @testset "A column" begin
        grid = RectilinearGrid(default_arch, FT; size = Nz, z = (0, 2400), topology = (Flat, Flat, Bounded))
        model = shoc_boundary_layer_model(grid)
        for _ in 1:40
            time_step!(model, 30)
        end
        ρ = column_data(model.dynamics.reference_state.density)
        e = column_data(model.tracers.ρe) ./ ρ
        cloud = model.closure_fields.subgrid_cloud.cloud
        λ = model.closure_fields.column_stability
        @test all(isfinite, e)
        @test all(≥(0), e)
        @test maximum(e) > 0.05
        @test all(x -> 0 ≤ x ≤ 1, column_data(cloud.aᶜ))
        @test all(≥(0), column_data(cloud.qˡ))
        @test all(isfinite, column_data(cloud.w′³))
        @test all(isfinite, column_data(model.formulation.potential_temperature_density))
        @test FT(0.001) ≤ λ[1, 1, 1] ≤ FT(0.08)
        @test all(isfinite, column_data(model.closure_fields.Kᶜ))
        @test all(≥(0), column_data(model.closure_fields.Kᶜ))
    end

    # The dissipation ratio ℂᴰ₂ / ℂᴰ₁ sets 𝕊ᴰ ∝ (ℂᴰ₂ / ℂᴰ₁)⁻⁴ / ℂᴰ₁ and so the time scale τ₀ ∝ 1 / 𝕊ᴰ: with the
    # stratification damping τ on both sides the column stays finite and its TKE bounded from 0.5 to 3 times ℂᴰ₁
    @testset "A column across the dissipation ratio ℂᴰ₂ / ℂᴰ₁" begin
        grid = RectilinearGrid(default_arch, FT; size = Nz, z = (0, 2400), topology = (Flat, Flat, Bounded))
        for ratio in (0.5, 1.5, 2.25, 3)
            stability_functions = ColumnStabilityDamping(FT; ℂᴰ₂ = ratio * 0.1)
            model = shoc_boundary_layer_model(grid; closure = SimplifiedHigherOrderClosure(FT; stability_functions))
            for _ in 1:60
                time_step!(model, 100)
            end
            ρ = column_data(model.dynamics.reference_state.density)
            e = column_data(model.tracers.ρe) ./ ρ
            @test all(isfinite, e)
            @test maximum(e) < 10
            @test all(isfinite, column_data(model.closure_fields.subgrid_cloud.cloud.w′³))
            @test all(isfinite, column_data(model.formulation.potential_temperature_density))
        end
    end

    # The buoyancy sink rate ωᴮ = -B⁻ / e of a cell without TKE between turbulent cells: B at the center is formed from
    # the conversions at its faces, whose diffusivities the turbulent neighbours give, so dividing by e at the center
    # made ωᴮ ~ 10¹⁵⁰ s⁻¹; divided by the TKE of those faces it is the diffusivity per unit TKE there
    @testset "The buoyancy sink rate next to a cell without TKE" begin
        grid = RectilinearGrid(default_arch, FT; size = Nz, z = (0, 2400), topology = (Flat, Flat, Bounded))
        model = shoc_boundary_layer_model(grid)
        ρ = column_data(model.dynamics.reference_state.density)[1, 1, :]
        interior(model.tracers.ρe)[1, 1, :] .= ρ .* [isodd(k) ? FT(0.5) : zero(FT) for k in 1:Nz]
        Oceananigans.TimeSteppers.update_state!(model)
        cloud = model.closure_fields.subgrid_cloud.cloud
        @test minimum(column_data(cloud.B)) < 0
        @test all(isfinite, column_data(cloud.ωᴮ))
        @test maximum(column_data(cloud.ωᴮ)) < 1
        @test all(isfinite, column_data(model.closure_fields.Lᵉ))
        @test minimum(column_data(model.closure_fields.Lᵉ)) > -1
    end

    @testset "Rebuilding and materializing the closure" begin
        grid = RectilinearGrid(default_arch, FT; size = Nz, z = (0, 2400), topology = (Flat, Flat, Bounded))
        model = shoc_boundary_layer_model(grid)
        closure = with_parameters(model.closure; ℂᶜ = FT(0.05), ℂᵘ = FT(0.05))
        materialized = materialize_closure(closure, closure_scalar_names(model), grid.architecture)
        @test materialized.stability_functions.ℂᶜ == FT(0.05)
        model.closure = materialized
        time_step!(model, 30)
        @test all(isfinite, column_data(model.tracers.ρe))
    end

    @testset "Column forest and three-dimensional grid" begin
        column_grid = RectilinearGrid(default_arch, FT; size = Nz, z = (0, 2400), topology = (Flat, Flat, Bounded))
        forest_grid = RectilinearGrid(default_arch, FT; size = ColumnEnsembleSize(Nz = Nz, ensemble = (3, 2), Hz = 3),
                                      z = (0, 2400), topology = (Flat, Flat, Bounded))
        box_grid = RectilinearGrid(default_arch, FT; size = (4, 4, Nz), x = (0, 4e5), y = (0, 4e5), z = (0, 2400),
                                   topology = (Periodic, Periodic, Bounded))

        models = map(shoc_boundary_layer_model, (column_grid, forest_grid, box_grid))
        for model in models, _ in 1:4
            time_step!(model, 30)
        end

        column, forest, box = models
        θ(model) = column_data(model.formulation.potential_temperature_density)
        e(model) = column_data(model.tracers.ρe)

        # Every column of the forest is the single column
        @test all(θ(forest) .== θ(column))
        @test all(e(forest) .== e(column))
        @test all(column_data(forest.closure_fields.ℓᵐ) .== column_data(column.closure_fields.ℓᵐ))

        # A horizontally uniform three-dimensional model stays uniform and follows the column; its mixing length is
        # also bounded by the horizontal grid spacing (a change of (ℓ / Δʰ)² ≈ 10⁻⁵)
        θ_box = θ(box)
        @test all(isfinite, θ_box)
        @test maximum(abs, θ_box .- θ_box[1:1, 1:1, :]) ≤ 100eps(FT) * maximum(abs, θ_box)
        @test isapprox(θ_box[1:1, 1:1, :], θ(column); rtol = FT == Float64 ? 1e-7 : 1e-5)
        @test isapprox(e(box)[1:1, 1:1, :], e(column); rtol = FT == Float64 ? 1e-3 : 1e-2)
    end

    @testset "Formulation check" begin
        grid = RectilinearGrid(default_arch, FT; size = Nz, z = (0, 3000), topology = (Flat, Flat, Bounded))
        @test_throws ArgumentError AtmosphereModel(grid; closure = SimplifiedHigherOrderClosure(FT), formulation = :StaticEnergy)
    end
end

# The static stability of SHOC: the grid-mean state with the distribution's mean liquid as its condensate
@testset "MeanLiquidStratification" begin
    grid = RectilinearGrid(CPU(), Float64; size = Nz, z = (0, 2400), topology = (Flat, Flat, Bounded))
    closure = SimplifiedHigherOrderClosure()
    @test closure.static_stability.fraction.stratification isa MeanLiquidStratification
    @test SimplifiedHigherOrderClosure(static_stability = SaturationWeightedStaticStability(CondensateFluxFraction())) isa SimplifiedHigherOrderClosure
    @test_throws ArgumentError SimplifiedHigherOrderClosure(solver = BracketedFractionSolve())
    @test occursin("MeanLiquidStratification()", sprint(show, closure))

    # On one state: against the stratification N²ᵇ of the state with the microphysics' condensate (the latent form plus
    # its latent part, N²ᵇ = N² + ΔN²ᴸ), the difference at the interior faces is (b_qˡ - b_qᵛ) ∂z (q̄ˡ - qᶜ) with the stored
    # derivatives, the distribution's liquid and the microphysics' condensate
    stratification(s) = SimplifiedHigherOrderClosure(static_stability = SaturationWeightedStaticStability(CondensateFluxFraction(stratification = s)))
    mean_liquid = shoc_boundary_layer_model(grid; closure = stratification(MeanLiquidStratification()))
    initial = shoc_boundary_layer_model(grid; closure = stratification(InitialStratification()))
    latent = shoc_boundary_layer_model(grid; closure = stratification(LatentStratification()))
    for model in (mean_liquid, initial, latent)
        for _ in 1:20
            time_step!(model, 30)
        end
    end
    # the same state in all three before the closure fields are formed
    for model in (initial, latent)
        set!(model, θ = Field(mean_liquid.formulation.potential_temperature_density / mean_liquid.dynamics.reference_state.density),
             qᵗ = Field(mean_liquid.moisture_density / mean_liquid.dynamics.reference_state.density))
        parent(model.tracers.ρe) .= parent(mean_liquid.tracers.ρe)
        parent(model.momentum.ρu) .= parent(mean_liquid.momentum.ρu)
        Oceananigans.TimeSteppers.update_state!(model)
    end
    Oceananigans.TimeSteppers.update_state!(mean_liquid)
    cl = mean_liquid.closure_fields.subgrid_cloud.cloud
    @test maximum(column_data(cl.qˡ)) > 0
    col(f) = vec(column_data(f))
    bˡᵛ = col(cl.b_qˡ) .- col(cl.b_qᵛ)
    qᶜ = col(mean_liquid.microphysical_fields.qˡ)
    qˡ = col(cl.qˡ)
    zc = collect(znodes(grid, Center()))
    expected = [(bˡᵛ[k] + bˡᵛ[k-1]) / 2 * ((qˡ[k] - qᶜ[k]) - (qˡ[k-1] - qᶜ[k-1])) / (zc[k] - zc[k-1]) for k in 2:Nz]
    N²ᵇ = col(latent.closure_fields.N²) .+ col(latent.closure_fields.subgrid_cloud.cloud.ΔN²ᴸ)
    ΔN² = col(mean_liquid.closure_fields.N²)[2:Nz] .- N²ᵇ[2:Nz]
    @test maximum(abs, expected) > 0
    @test ΔN² ≈ expected rtol = 1e-8 atol = 1e-12
    # the initial form keeps the static stability of the dynamics
    @test !(col(latent.closure_fields.N²)[2:Nz] ≈ col(mean_liquid.closure_fields.N²)[2:Nz])
    @test !(col(initial.closure_fields.N²)[2:Nz] ≈ col(mean_liquid.closure_fields.N²)[2:Nz])
    # the buoyancy production carries the same latent part whatever the static stability reads
    @test col(mean_liquid.closure_fields.subgrid_cloud.cloud.ΔN²ᴸ) ≈ col(initial.closure_fields.subgrid_cloud.cloud.ΔN²ᴸ)
end
