include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization
using Test

@testset "TKE shear production uses the current stage [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    # Only the viscosity: no scalar or TKE diffusivity and no dissipation, so the shear production is the only source
    coefficients = ConstantStabilityFunctions(ℂᵘ=1, ℂᶜ=0, ℂᵉ=0, ℂᴰ=0)

    @testset "Uniform shear stays uniform across periodic boundaries" for time_discretization in
        (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())

        grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 4), y=(0, 4), z=(0, 8))
        closure = TKEBasedTurbulenceClosure(time_discretization; stability_functions = coefficients)
        model = AtmosphereModel(grid; closure, advection=nothing)
        density = Array(interior(model.dynamics.reference_state.density))

        # Reversing the shear must still produce positive Kᵘ S². Check every column,
        # including those that interpolate the periodic velocity halos. The viscosity is that of the same
        # e and N² at every stage, so the production over S² = 5 shear² is the same positive profile at each,
        # in every column.
        viscosity = map(enumerate((FT(1), FT(-0.5), FT(2)))) do (stage, shear)
            model.clock.stage = stage
            set!(model; θ=300, u=(x, y, z) -> shear * z,
                              v=(x, y, z) -> -2shear * z, ρe=1)
            fill!(model.timestepper.Gⁿ.ρe, 0)
            Breeze.AtmosphereModels.compute_closure_tendencies!(model)
            production = Array(interior(model.timestepper.Gⁿ.ρe)) ./ density
            production ./ (5 * shear^2)
        end
        @test all(>(0), viscosity[1])
        @test all(isapprox.(viscosity[1], viscosity[1][1:1, 1:1, :]; rtol=1e-5))
        @test all(isapprox.(viscosity[2], viscosity[1]; rtol=1e-5))
        @test all(isapprox.(viscosity[3], viscosity[1]; rtol=1e-5))
    end

    @testset "Runge–Kutta integrates prescribed evolving shear" begin
        grid = RectilinearGrid(default_arch; size=8, z=(0, 8), topology=(Flat, Flat, Bounded))
        reference_state = ReferenceState(grid, ThermodynamicConstants())
        # With `minimum_tke` above e the turbulent velocity, and so the viscosity ν(z) = ℓ √eᵐⁱⁿ, does not change as
        # e grows: the only TKE source is ν (1 + t)². SSP-RK3 integrates this quadratic
        # exactly; averaging with whole-step-old velocities introduces a first-order bias.
        closure = TKEBasedTurbulenceClosure(ExplicitTimeDiscretization(); stability_functions = coefficients, minimum_tke = 1e4)
        model = AtmosphereModel(grid; dynamics=PrescribedDynamics(reference_state),
                                closure, advection=nothing)
        set!(model; θ=300, u=z -> z, ρe=1)
        function prescribe_shear!(model)
            time = model.clock.time
            set!(model.velocities.u, z -> (1 + time) * z)
            fill_halo_regions!(model.velocities)
            return nothing
        end
        callbacks = [Callback(prescribe_shear!, IterationInterval(1); callsite=UpdateStateCallsite())]
        density = Array(interior(reference_state.density))
        # ν at the centers: the production of the initial state, whose squared shear is one
        fill!(model.timestepper.Gⁿ.ρe, 0)
        Breeze.AtmosphereModels.compute_closure_tendencies!(model)
        ν = Array(interior(model.timestepper.Gⁿ.ρe)) ./ density
        @test all(>(0), ν)
        Oceananigans.TimeSteppers.update_state!(model)
        initial_tke = Array(interior(model.tracers.ρe)) ./ density
        for _ in 1:4
            time_step!(model, FT(0.25); callbacks)
        end
        final_tke = Array(interior(model.tracers.ρe)) ./ density
        @test maximum(final_tke) < 1e4
        @test final_tke .- initial_tke ≈ 7ν / 3 rtol=1e-4
    end
end
