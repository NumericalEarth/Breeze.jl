using Test
using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, OpenBoundaryCondition,
                                       fill_halo_regions!
using GPUArraysCore: @allowscalar
using Breeze
using Breeze.AtmosphereModels: compute_velocities!
using Breeze.AnelasticEquations: AnelasticDynamics
using Breeze.CompressibleEquations: CompressibleDynamics, ExplicitTimeStepping
using Breeze.Thermodynamics: ReferenceState

# Regression tests for the OBC-on-momentum velocity-clobber bug.
#
# Prior to the fix, `model.velocities` (u, v, w) were constructed with
# DEFAULT impenetrable BCs regardless of what the user set on momentum
# (ρu, ρv, ρw). The call to `fill_halo_regions!(model.velocities)` after
# `compute_velocities!` then overwrote the boundary face with zero, even
# when the user prescribed a non-zero open boundary on momentum. The
# inconsistency between ρu (=user value at wall) and u (=0 at wall)
# drove a wall-mode instability that NaN'd the run within ~100 iters.
#
# After the fix:
#   • velocity has `nothing` on the normal-component face of each Bounded direction,
#     so `fill_halo_regions!` cannot clobber the boundary face
#   • compute_velocities! covers the full Face range incl. boundary faces, so the
#     kernel writes `u = ρu/ρ` at the wall each step

@testset "OBC on momentum propagates to derived velocities [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nx, Ny, Nz = 8, 8, 4
    grid = RectilinearGrid(default_arch, FT;
                           size = (Nx, Ny, Nz),
                           x = (0, 1000),
                           y = (0, 1000),
                           z = (0, 100),
                           topology = (Bounded, Periodic, Bounded))

    @testset "Anelastic — Open ρu propagates to u at boundary face" begin
        reference_state = ReferenceState(grid; surface_pressure=FT(101325),
                                                potential_temperature=FT(300))
        dynamics = AnelasticDynamics(reference_state)

        ρ_b = FT(1.225)
        U_bg = FT(5.0)
        ρu_value = ρ_b * U_bg

        ρu_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(ρu_value),
                                          east = OpenBoundaryCondition(ρu_value))
        model = AtmosphereModel(grid; dynamics=dynamics,
                                       formulation=:LiquidIcePotentialTemperature,
                                       boundary_conditions=(; ρu=ρu_bcs))

        # Velocity BCs on west/east should be `nothing` (so fill_halo doesn't clobber)
        @test isnothing(model.velocities.u.boundary_conditions.west)
        @test isnothing(model.velocities.u.boundary_conditions.east)

        set!(model; θ=FT(300), ρu=(x,y,z) -> ρ_b * U_bg, ρv=0, ρw=0)
        fill_halo_regions!(model.momentum, model.clock, fields(model))
        compute_velocities!(model)

        # u at the west and east boundary faces should NOT be zero
        u_west = @allowscalar interior(model.velocities.u, 1, :, :)
        u_east = @allowscalar interior(model.velocities.u, Nx+1, :, :)
        @test all(>(FT(0.5) * U_bg), u_west)
        @test all(>(FT(0.5) * U_bg), u_east)
    end

    @testset "Compressible — Open ρu propagates to u at boundary face" begin
        td = ExplicitTimeStepping()
        dynamics = CompressibleDynamics(td; reference_potential_temperature=FT(300))

        ρ_b = FT(1.225)
        U_bg = FT(5.0)
        ρu_value = ρ_b * U_bg

        ρu_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(ρu_value),
                                          east = OpenBoundaryCondition(ρu_value))
        model = AtmosphereModel(grid; dynamics=dynamics,
                                       formulation=:LiquidIcePotentialTemperature,
                                       boundary_conditions=(; ρu=ρu_bcs))

        @test isnothing(model.velocities.u.boundary_conditions.west)
        @test isnothing(model.velocities.u.boundary_conditions.east)

        ref = model.dynamics.reference_state
        set!(model; θ=FT(300), ρ=ref.density,
                     ρu=(x,y,z) -> ρ_b * U_bg, ρv=0, ρw=0)
        fill_halo_regions!(model.momentum, model.clock, fields(model))
        compute_velocities!(model)

        u_west = @allowscalar interior(model.velocities.u, 1, :, :)
        u_east = @allowscalar interior(model.velocities.u, Nx+1, :, :)
        @test all(>(FT(0.5) * U_bg), u_west)
        @test all(>(FT(0.5) * U_bg), u_east)
    end

    @testset "Anelastic — uniform OBC inflow stays uniform under SSP-RK3" begin
        # Pre-fix: this NaN'd within ~100 iters.
        # Post-fix: max|u| should remain ≈ U_bg over a few iters.
        reference_state = ReferenceState(grid; surface_pressure=FT(101325),
                                                potential_temperature=FT(300))
        dynamics = AnelasticDynamics(reference_state)

        ρ_b = FT(1.225)
        U_bg = FT(5.0)
        ρu_value = ρ_b * U_bg

        ρu_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(ρu_value),
                                          east = OpenBoundaryCondition(ρu_value))
        model = AtmosphereModel(grid; dynamics=dynamics,
                                       formulation=:LiquidIcePotentialTemperature,
                                       boundary_conditions=(; ρu=ρu_bcs))
        set!(model; θ=FT(300), ρu=(x,y,z) -> ρ_b * U_bg, ρv=0, ρw=0)

        simulation = Simulation(model; Δt=FT(0.5), stop_iteration=20, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 20
        u_max = @allowscalar maximum(abs, interior(model.velocities.u))
        @test isfinite(u_max)
        # Should stay close to U_bg (cell-centered ρ slightly different from ρ_b makes
        # the exact u value slightly off; 50% is a comfortable bound to detect blow-up).
        @test u_max < FT(2) * U_bg
    end

    @testset "Default impenetrable wall (no user momentum BC) gives u = 0 at wall" begin
        # `nothing` on velocity is the catch-all override, but the default impenetrable
        # behavior must still hold: when momentum has the default OpenBoundaryCondition(0),
        # the kernel writes u = ρu/ρ = 0 at the wall and the wall stays impenetrable.
        reference_state = ReferenceState(grid; surface_pressure=FT(101325),
                                                potential_temperature=FT(300))
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid; dynamics=dynamics,
                                       formulation=:LiquidIcePotentialTemperature)

        # No user BC on ρu → default impenetrable (OpenBoundaryCondition(0))
        # → velocity west/east set to `nothing`
        @test isnothing(model.velocities.u.boundary_conditions.west)
        @test isnothing(model.velocities.u.boundary_conditions.east)

        set!(model; θ=FT(300))
        fill_halo_regions!(model.momentum, model.clock, fields(model))
        compute_velocities!(model)

        u_west = @allowscalar interior(model.velocities.u, 1, :, :)
        u_east = @allowscalar interior(model.velocities.u, Nx+1, :, :)
        @test all(==(zero(FT)), u_west)
        @test all(==(zero(FT)), u_east)
    end
end
