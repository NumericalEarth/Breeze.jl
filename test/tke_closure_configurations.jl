include(joinpath(@__DIR__, "setup.jl"))

#####
##### The calibrated configurations of TKEBasedTurbulenceClosure: the defaults (the updated TKE closure) and
##### SimplifiedHigherOrderClosure, with P3 condensing from the closure's distribution
#####

using Breeze
using Breeze.Microphysics.PredictedParticleProperties: SubgridCondensation, AllProcesses
using Oceananigans
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Test

column_values(f) = Array(interior(f))[1, 1, :]

# A trade-cumulus column: a mixed layer, a conditionally unstable cloud layer and an inversion, with surface fluxes
function trade_cumulus_column(closure; FT = Float64, processes = AllProcesses())
    grid = RectilinearGrid(default_arch, FT; size = 32, z = (0, 2400), topology = (Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure = 101500, potential_temperature = 299)
    condensation = SubgridCondensation(TurbulenceClosureStatistics(); processes)
    microphysics = PredictedParticlePropertiesMicrophysics(FT; thermodynamic_constants = constants, condensation,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    flux(J) = FieldBoundaryConditions(bottom = FluxBoundaryCondition(FT(J)))
    boundary_conditions = (ρθ = flux(1.17 * 8e-3), ρqᵗ = flux(1.17 * 5.2e-5), ρu = flux(1.17 * 0.08), ρv = flux(0))
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), thermodynamic_constants = constants,
                            microphysics, closure, boundary_conditions)
    θ₀(z) = z < 520 ? 298.7 : z < 1480 ? 298.7 + (z - 520) * 3.7 / 960 : 302.4 + (z - 1480) * 5.8 / 520
    qᵗ₀(z) = 1e-3 * (z < 520 ? 17.0 - z * 0.7 / 520 : z < 1480 ? 16.3 - (z - 520) * 5.6 / 960 : max(4.2, 10.7 - (z - 1480) * 6.5 / 520))
    set!(model; θ = θ₀, qᵗ = qᵗ₀, u = z -> -8.75 + 1.8e-3 * max(z - 700, 0), ρe = z -> 1.17 * 0.6 * exp(-z / 500))
    return model
end

@testset "The constructors' defaults are the calibrated configurations" begin
    closure = TKEBasedTurbulenceClosure()
    d = closure.subgrid_distribution
    @test closure.mixing_length isa GradientLimitedMixingLength
    @test closure.stability_functions isa TurbulentRichardsonStabilityFunction
    @test closure.static_stability isa SaturationWeightedStaticStability
    @test closure.static_stability.fraction.stratification isa LatentStratification
    @test closure.heating_budget_flux isa HeatingBudgetFlux
    @test d isa DoubleGaussian
    @test d.time_scale isa DissipationTimeScale
    @test d.third_moment isa TKEBudgetThirdMoment
    @test d.assumed_pdf isa EqualWidthDoubleGaussianPDF
    @test d.variance_transport isa ScalarVarianceTransport
    @test d.variance_stability_functions isa VarianceStabilityFunctions
    @test !(closure isa SimplifiedHigherOrderClosure)
    @test isbits(closure)

    shoc = SimplifiedHigherOrderClosure()
    @test shoc isa SimplifiedHigherOrderClosure && SHOC === SimplifiedHigherOrderClosure
    @test shoc.mixing_length isa BlendedMixingLength
    @test shoc.stability_functions isa ColumnStabilityDamping
    @test shoc.static_stability.fraction.stratification isa MeanLiquidStratification
    @test isnothing(shoc.heating_budget_flux)
    s = shoc.subgrid_distribution
    @test s.time_scale isa ReturnToIsotropyTimeScale
    @test s.third_moment isa AlgebraicThirdMoment
    @test s.assumed_pdf isa DoubleGaussianPDF
    @test isnothing(s.variance_transport) && isnothing(s.variance_stability_functions)
    @test isbits(shoc)

    # The published coefficients of shoc.F90
    p = closure_parameters(shoc)
    @test p.ℂᶜ == 0.1 && p.ℂᵘ == 0.1 && p.ℂᴾ == 0.1
    @test p.ℂˡ₁ == 0.5 && p.ℂˡ₅ == 400
    @test p.ℂʷ₂ == 7 && p.ℂᵍ₁ == 0.4

    # with_parameters replaces the coefficients and keeps the configuration and the float type
    for c in (closure, shoc)
        names = keys(closure_parameters(c))
        rebuilt = with_parameters(c; NamedTuple{(names[1],)}((0.3,))...)
        @test typeof(rebuilt) == typeof(c)
        @test closure_parameters(rebuilt)[names[1]] == 0.3
        @test with_parameters(c) === c
        @test_throws ArgumentError with_parameters(c; ℂ_not_a_coefficient = 1)
    end
    @test closure_parameters(TKEBasedTurbulenceClosure(Float32)) == map(Float32, closure_parameters(closure))
    @test summary(shoc) == "SimplifiedHigherOrderClosure{VerticallyImplicitTimeDiscretization}"
    @test startswith(sprint(show, closure), "TKEBasedTurbulenceClosure{VerticallyImplicitTimeDiscretization}")
end

# One short column of each calibrated configuration: finite, realizable, and the same as when it was calibrated
# Σ e (m² s⁻²), the largest cloud fraction and Σ ρqᶜˡ (kg m⁻³) after 20 minutes, at the calibrated commit
const REGRESSION = Dict("TKEBasedTurbulenceClosure" => (2.825995495882156, 0.0041570081060038, 1.5924532691005748e-7),
                        "SimplifiedHigherOrderClosure" => (0.9032918412584622, 9.494176956646707e-9, 5.3137606541913e-8),
                        "SimplifiedHigherOrderClosure with ℂᴾ = 1" => (2.6726511998460794, 2.198283854539088e-6, 5.3802022565686416e-8))

configurations = (("TKEBasedTurbulenceClosure", TKEBasedTurbulenceClosure(), AllProcesses()),
                  ("SimplifiedHigherOrderClosure", SimplifiedHigherOrderClosure(), Breeze.Microphysics.PredictedParticleProperties.CloudCondensationOnly()),
                  ("SimplifiedHigherOrderClosure with ℂᴾ = 1", with_parameters(SimplifiedHigherOrderClosure(); ℂᴾ = 1.0),
                   Breeze.Microphysics.PredictedParticleProperties.CloudCondensationOnly()))

@testset "A trade-cumulus column [$label]" for (label, closure, processes) in configurations
    model = trade_cumulus_column(closure; processes)
    for _ in 1:40
        time_step!(model, 30)
    end
    ρ = column_values(model.dynamics.reference_state.density)
    e = column_values(model.tracers.ρe) ./ ρ
    cloud = model.closure_fields.subgrid_cloud.cloud
    fields = Oceananigans.prognostic_fields(model)
    @test all(name -> all(isfinite, interior(fields[name])), keys(fields))
    @test minimum(e) > -1e-6 && maximum(e) > 0.05
    @test all(a -> 0 ≤ a ≤ 1, column_values(cloud.aᶜ))
    @test all(≥(0), column_values(model.microphysical_fields.ρqᶜˡ))
    @test all(≥(0), column_values(model.closure_fields.Kᶜ))
    regression = (sum(e), maximum(column_values(cloud.aᶜ)), sum(column_values(model.microphysical_fields.ρqᶜˡ)))
    @test all(isapprox.(regression, REGRESSION[label]; rtol = 1e-6))
end
