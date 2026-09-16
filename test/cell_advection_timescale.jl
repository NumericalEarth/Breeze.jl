include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using CloudMicrophysics
using Oceananigans
using Oceananigans.Advection: cell_advection_timescale, AdaptiveVerticallyImplicitDiscretization
using Oceananigans.Simulations: TimeStepWizard
using Oceananigans.TurbulenceClosures: HorizontalFormulation, ThreeDimensionalFormulation
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, TwoMomentCloudMicrophysics

@testset "Sedimentation constrains the advective timescale [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=(4, 4), x=(0, 100), z=(0, 200),
                          topology=(Periodic, Flat, Bounded))
    Δx, Δz = FT(25), FT(50)
    cloud_formation = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; microphysics)

    # Rain moves even in still air. The former air-only timescale was infinite.
    set!(model.microphysical_fields.wʳ, -10)
    @test cell_advection_timescale(model) ≈ Δz / 10

    # Use the signed sum of air and fall velocities, including downdrafts and
    # partial cancellation in updrafts, while retaining the air's own CFL limit.
    set!(model.velocities.u, 2)
    for w in FT.((-4, 4, 20))
        set!(model.velocities.w, w)
        expected = inv(2 / Δx + max(abs(w), abs(w - 10)) / Δz)
        @test cell_advection_timescale(model) ≈ expected
        @test CellAdvectionTimescale(ThreeDimensionalFormulation())(model) ≈ expected
        @test CellAdvectionTimescale(HorizontalFormulation())(model) ≈ Δx / 2
    end

    # Every precipitating species participates, not just rain.
    set!(model.velocities.w, 0)
    set!(model.microphysical_fields.wˢⁿ, -15)
    expected = inv(2 / Δx + 15 / Δz)
    @test cell_advection_timescale(model) ≈ expected
    wizard = TimeStepWizard(cfl=FT(0.7))
    @test wizard.cfl * wizard.cell_advection_timescale(model) ≈ FT(0.7) * expected

    # Number-weighted fall speeds can exceed mass-weighted fall speeds.
    two_moment_model = AtmosphereModel(grid; microphysics=TwoMomentCloudMicrophysics(FT))
    set!(two_moment_model.microphysical_fields.wʳ, -10)
    set!(two_moment_model.microphysical_fields.wⁿʳ, -20)
    @test cell_advection_timescale(two_moment_model) ≈ Δz / 20

    # A model without sedimentation keeps the original air-advection timescale.
    dry_model = AtmosphereModel(grid)
    set!(dry_model.velocities.u, 2)
    set!(dry_model.velocities.w, -4)
    @test cell_advection_timescale(dry_model) ≈ inv(2 / Δx + 4 / Δz)

    # Fully implicit vertical advection still floats on the horizontal CFL.
    implicit_advection = WENO(FT; time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.5))
    implicit_model = AtmosphereModel(grid; microphysics, advection=implicit_advection)
    set!(implicit_model.velocities.u, 2)
    set!(implicit_model.microphysical_fields.wʳ, -10)
    @test cell_advection_timescale(implicit_model) ≈ Δx / 2
    @test CellAdvectionTimescale(ThreeDimensionalFormulation())(implicit_model) ≈ inv(2 / Δx + 10 / Δz)
end
