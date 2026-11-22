include(joinpath(@__DIR__, "setup.jl"))

#####
##### Closures whose coefficients vary from column to column: one model integrating an ensemble of
##### closures must reproduce, column by column, separate single-column models of each closure.
#####

using Breeze
using Breeze.TurbulenceClosures: column_closure
using Breeze.Microphysics.PredictedParticleProperties: SubgridCondensation, AllProcesses
using Oceananigans
using Oceananigans.Grids: ColumnEnsembleSize
using Adapt: adapt
using Test

const Nz = 12

function column_grid(FT, N₁)
    size = N₁ == 1 ? Nz : ColumnEnsembleSize(Nz=Nz, ensemble=(N₁, 1), Hz=3)
    return RectilinearGrid(default_arch, FT; size, z=(0, 2000), topology=(Flat, Flat, Bounded))
end

# `closure` is a closure, or a function of the grid that returns one
default_microphysics(FT) = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())

function integrate_columns(FT, closure, N₁; steps = 10, Δt = 10, microphysics = default_microphysics(FT))
    grid = column_grid(FT, N₁)
    reference_state = ReferenceState(grid, ThermodynamicConstants(FT); base_pressure=101325, potential_temperature=290)
    closure = closure isa Function ? closure(grid) : closure
    boundary_conditions = (; ρθ = FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(0.1))),
                             ρqᵗ = FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(5e-5))))
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), closure, boundary_conditions, microphysics)
    set!(model; θ = z -> 290 + 3e-3 * max(z - 800, 0), qᵗ = z -> 0.014 - 2e-6 * z, u = 5)
    set!(model.tracers.ρe, reference_state.density * FT(0.1))
    for _ in 1:steps
        time_step!(model, Δt)
    end
    fields = Oceananigans.prognostic_fields(model)
    names = (:ρe, :ρu, :ρθ, filter(name -> name ∉ (:ρe, :ρu, :ρv, :ρw, :ρθ), keys(fields))...)
    return map(name -> Array(interior(fields[name])), names)
end

function test_per_column_equals_single_columns(FT, closures; kw...)
    N₁ = length(closures)
    matrix = reshape(collect(closures), N₁, 1)
    ensemble = integrate_columns(FT, grid -> per_column_closure(grid, matrix), N₁; kw...)
    tolerance = 10 * eps(FT)
    for (n, closure) in enumerate(closures)
        single = integrate_columns(FT, closure, 1; kw...)
        for (e, s) in zip(ensemble, single)
            @test isapprox(e[n, 1, :], s[1, 1, :]; rtol = tolerance)
        end
    end
    # The members differ: the test is not one of identical columns
    @test ensemble[1][1, 1, :] != ensemble[1][N₁, 1, :]
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

    @testset "DoubleGaussian with VarianceStabilityFunctions and ℂᵐ, P3 with AllProcesses: per column equals single columns" begin
        # The coefficients of the variance stability functions and the mixing time ℂᵐ, which the
        # microphysics reads through ωᵐ, vary from column to column
        stability(ℂᵛ₁⁰, ℂᵛ₃ᵐ) = VarianceStabilityFunctions(FT; ℂᵛ₁⁰, ℂᵛ₂⁰ = ℂᵛ₁⁰, ℂᵛ₃ᵐ)
        closure(ℂᵛ₁⁰, ℂᵛ₃ᵐ, ℂᵐ) =
            tke_closure(FT, 1.4, 2.0; static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()),
                        subgrid_distribution = DoubleGaussian(FT; variance_stability_functions = stability(ℂᵛ₁⁰, ℂᵛ₃ᵐ), ℂᵐ))
        closures = [closure(0.5, 0.9, 0.5), closure(0.9, 0.6, 2.0)]

        grid = column_grid(FT, 2)
        ensemble = per_column_closure(grid, reshape(closures, 2, 1))
        vs = ensemble.subgrid_distribution.variance_stability_functions
        @test vs.ℂᵛ₁⁰ isa PerColumnCoefficient
        @test Array(vs.ℂᵛ₃ᵐ.values) == FT[0.9; 0.6;;]
        @test Array(ensemble.subgrid_distribution.ℂᵐ.values) == FT[0.5; 2.0;;]
        updated = reshape([closure(0.7, 0.3, 1.0), closure(1.1, -0.2, 3.0)], 2, 1)
        set_per_column_coefficients!(ensemble, updated)
        @test Array(vs.ℂᵛ₃ᵐ.values) == FT[0.3; -0.2;;]
        @test Array(ensemble.subgrid_distribution.ℂᵐ.values) == FT[1.0; 3.0;;]
        @test all(column_closure(i, 1, ensemble) === updated[i, 1] for i in 1:2)

        condensation = SubgridCondensation(TurbulenceClosureStatistics(); processes = AllProcesses())
        microphysics = PredictedParticlePropertiesMicrophysics(FT; condensation)
        test_per_column_equals_single_columns(FT, closures; microphysics)
    end

    @testset "SimplifiedHigherOrderClosure: per column equals single columns" begin
        closures = [with_parameters(SimplifiedHigherOrderClosure(FT); ℂᶜ=ℂᶜ, ℂʷ₂=ℂʷ₂, ℂˡ₁=ℂˡ₁)
                    for (ℂᶜ, ℂʷ₂, ℂˡ₁) in ((0.1, 7.0, 0.5), (0.2, 5.0, 0.4))]
        @test all(closure -> closure isa SimplifiedHigherOrderClosure, closures)
        ensemble = per_column_closure(column_grid(FT, 2), reshape(closures, 2, 1))
        @test ensemble isa SimplifiedHigherOrderClosure
        @test ensemble.stability_functions.ℂᶜ isa PerColumnCoefficient
        test_per_column_equals_single_columns(FT, closures)
    end
    @testset "SimplifiedHigherOrderClosure with SmoothClipping: per column equals single columns" begin
        # Rebuilding the closure of a column (`getclosure`, also in GPU kernels) must not check the width
        @test adapt(nothing, SmoothClipping{FT}(-one(FT))).width == -one(FT)
        distribution = Breeze.TurbulenceClosures.shoc_distribution(FT; clipping = SmoothClipping(FT))
        closures = [with_parameters(SimplifiedHigherOrderClosure(FT; subgrid_distribution = distribution); ℂᶜ=ℂᶜ)
                    for ℂᶜ in (0.1, 0.2)]
        test_per_column_equals_single_columns(FT, closures)
    end

    @testset "A profile shared by the columns forces every column alike" begin
        function forced_increment(N₁)
            grid = column_grid(FT, N₁)
            reference_state = ReferenceState(grid, ThermodynamicConstants(FT); base_pressure=101325, potential_temperature=290)
            cooling = Field{Nothing, Nothing, Center}(grid)
            set!(cooling, z -> -1e-5 * (1 + z / 1000))
            drying = Field{Nothing, Nothing, Center}(grid)
            interior(drying, 1, 1, :) .= [-1e-8 * k for k in 1:Nz]
            model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state),
                                    microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()),
                                    forcing = (; θ = Forcing(cooling), qᵉ = Forcing(drying),
                                                 u = SubsidenceForcing(z -> -1e-3 * z / 2000)))
            set!(model; θ = z -> 290 + 3e-3 * z, qᵗ = 0.01, u = z -> 5 + z / 1000)
            fields = Oceananigans.prognostic_fields(model)
            before = map(name -> Array(interior(fields[name])), (:ρθ, :ρqᵉ, :ρu))
            time_step!(model, 10)
            return map((name, b) -> Array(interior(fields[name])) .- b, (:ρθ, :ρqᵉ, :ρu), before)
        end
        # Oceananigans' `set!` of a reduced field from an array copies it linearly into the interior,
        # which on an ensemble has a row per column (see the TODO in `src/single_column_mode.jl`)
        grid = column_grid(FT, 2)
        profile = Field{Nothing, Nothing, Center}(grid)
        set!(profile, collect(FT, 1:Nz))
        @test_broken [profile[1, 1, k] for k in 1:Nz] == 1:Nz

        single = forced_increment(1)
        ensemble = forced_increment(3)
        for (e, s) in zip(ensemble, single)
            @test any(!iszero, s)
            @test all(e[i, 1, :] == s[1, 1, :] for i in 1:3)
        end
    end
end

# The closure of a column infers for either location of the third moment (a closure that does not infer cannot be
# adapted inside a GPU kernel, where `getclosure` selects each column's)
@testset "Per-column closures with the third moment at faces and at centers infer" begin
    grid = RectilinearGrid(size=ColumnEnsembleSize(Nz=4, ensemble=(2, 1)), z=(0, 1000), topology=(Flat, Flat, Bounded))
    for location in (Face(), Center())
        closures = (TKEBasedTurbulenceClosure(subgrid_distribution = DoubleGaussian(third_moment = TKEBudgetThirdMoment(; location))),
                    SimplifiedHigherOrderClosure(subgrid_distribution = Breeze.TurbulenceClosures.shoc_distribution(third_moment = AlgebraicThirdMoment(; location))))
        for closure in closures
            @test closure.subgrid_distribution.third_moment.location === location
            @inferred Breeze.TurbulenceClosures.column_closure(1, 1, closure)
            @inferred adapt(Array, closure)
            per_column = per_column_closure(grid, [closure for _ in 1:2, _ in 1:1])
            column = @inferred Breeze.TurbulenceClosures.column_closure(2, 1, per_column)
            @test column.subgrid_distribution.third_moment.location === location
        end
    end
end
