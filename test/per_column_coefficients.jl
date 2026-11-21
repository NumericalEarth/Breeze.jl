include(joinpath(@__DIR__, "setup.jl"))

#####
##### Closures whose coefficients vary from column to column: one model integrating an ensemble of
##### closures must reproduce, column by column, separate single-column models of each closure.
#####

using Breeze
using Breeze.TurbulenceClosures: column_closure
using Oceananigans
using Oceananigans.Grids: ColumnEnsembleSize
using Test

const Nz = 12

function column_grid(FT, N₁)
    size = N₁ == 1 ? Nz : ColumnEnsembleSize(Nz=Nz, ensemble=(N₁, 1), Hz=3)
    return RectilinearGrid(default_arch, FT; size, z=(0, 2000), topology=(Flat, Flat, Bounded))
end

# `closure` is a closure, or a function of the grid that returns one
function integrate_columns(FT, closure, N₁; steps = 10, Δt = 10)
    grid = column_grid(FT, N₁)
    reference_state = ReferenceState(grid, ThermodynamicConstants(FT); base_pressure=101325, potential_temperature=290)
    closure = closure isa Function ? closure(grid) : closure
    boundary_conditions = (; ρθ = FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(0.1))),
                             ρqᵗ = FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(5e-5))))
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), closure, boundary_conditions,
                            microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()))
    set!(model; θ = z -> 290 + 3e-3 * max(z - 800, 0), qᵗ = z -> 0.014 - 2e-6 * z, u = 5)
    set!(model.tracers.ρe, reference_state.density * FT(0.1))
    for _ in 1:steps
        time_step!(model, Δt)
    end
    fields = Oceananigans.prognostic_fields(model)
    return map(name -> Array(interior(fields[name])), (:ρθ, :ρqᵉ, :ρu, :ρe))
end

function test_per_column_equals_single_columns(FT, closures)
    N₁ = length(closures)
    matrix = reshape(collect(closures), N₁, 1)
    ensemble = integrate_columns(FT, grid -> per_column_closure(grid, matrix), N₁)
    tolerance = 10 * eps(FT)
    for (n, closure) in enumerate(closures)
        single = integrate_columns(FT, closure, 1)
        for (e, s) in zip(ensemble, single)
            @test isapprox(e[n, 1, :], s[1, 1, :]; rtol = tolerance)
        end
    end
    # The members differ: the test is not one of identical columns
    @test ensemble[4][1, 1, :] != ensemble[4][N₁, 1, :]
end

tke_closure(FT, ℂⁱ, ℂᴰ⁰; kw...) =
    TKEBasedTurbulenceClosure(FT; mixing_length = GradientLimitedMixingLength(ℂˢ=1.3, ℂⁱ=ℂⁱ),
                              stability_functions = TurbulentRichardsonStabilityFunction(ℂᴰ⁰=ℂᴰ⁰), kw...)

@testset "Per-column closure coefficients [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    @testset "construction, selection and the in-place update" begin
        grid = column_grid(FT, 3)
        closures = reshape([tke_closure(FT, ℂⁱ, 2) for ℂⁱ in (1.2, 1.6, 2.0)], 3, 1)
        closure = per_column_closure(grid, closures)
        @test closure.mixing_length.ℂⁱ isa PerColumnCoefficient
        @test Array(closure.mixing_length.ℂⁱ.values) == FT[1.2; 1.6; 2.0;;]
        @test closure.mixing_length.ℂˢ isa PerColumnCoefficient # shared scalars are per column too
        @test all(column_closure(i, 1, closure) === closures[i, 1] for i in 1:3)

        # A closure without per-column coefficients is its own column closure
        @test column_closure(2, 1, closures[1]) === closures[1]

        updated = reshape([tke_closure(FT, ℂⁱ, 3) for ℂⁱ in (2.2, 1.1, 0.9)], 3, 1)
        set_per_column_coefficients!(closure, updated)
        @test all(column_closure(i, 1, closure) === updated[i, 1] for i in 1:3)

        @test_throws ArgumentError per_column_closure(grid, closures[1:2, :])
        @test_throws ArgumentError per_column_closure(grid, reshape([closures[1], closures[2], SimplifiedHigherOrderClosure(FT)], 3, 1))
    end

    @testset "TKEBasedTurbulenceClosure: per column equals single columns" begin
        test_per_column_equals_single_columns(FT, [tke_closure(FT, ℂⁱ, ℂᴰ⁰) for (ℂⁱ, ℂᴰ⁰) in ((1.2, 2.0), (1.6, 2.5), (2.0, 3.0))])
    end

    @testset "TKEBasedTurbulenceClosure with a DoubleGaussian: per column equals single columns" begin
        closure(ℂⁱ, ℂᴰ⁰) = tke_closure(FT, ℂⁱ, ℂᴰ⁰; subgrid_distribution = DoubleGaussian(FT),
                                       static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()))
        test_per_column_equals_single_columns(FT, [closure(1.2, 2.0), closure(1.8, 3.0)])
    end

    @testset "SimplifiedHigherOrderClosure: per column equals single columns" begin
        closures = [with_parameters(SimplifiedHigherOrderClosure(FT; surface_temperature_flux=0.05); ℂᶜ=ℂᶜ, ℂʷ₂=ℂʷ₂)
                    for (ℂᶜ, ℂʷ₂) in ((0.1, 7.0), (0.2, 5.0))]
        test_per_column_equals_single_columns(FT, closures)
    end
end
