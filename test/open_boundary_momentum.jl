using Test
using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, NormalFlowBoundaryCondition,
                                       PeriodicBoundaryCondition, fill_halo_regions!
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
#   • velocities are constructed via `XFaceField(grid)` etc. with no `boundary_conditions=`
#     kwarg, picking up the *auxiliary* defaults (`nothing` on Bounded-Face sides,
#     Periodic on Periodic sides). `fill_halo_regions!` cannot clobber the boundary face.
#   • compute_velocities! launches a fused kernel over (1:Nx+1, 1:Ny+1, 1:Nz+1), so the
#     kernel writes `u = ρu/ρ` at the wall each step.

# Wrap `interior(...)` views in `Array(...)` so boundary-face slice reductions like
# `all(f, ...)` run on CPU. On GPU, `all(f, ::CuArray)` runs as a `mapreduce` that can't
# infer the boolean element type from a closure predicate; pulling to CPU avoids it.
boundary_slice(field, args...) = Array(interior(field, args...))

@testset "OBC on momentum propagates to derived velocities [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nx, Ny, Nz = 8, 8, 4
    Lx, Ly, Lz = FT(1000), FT(1000), FT(100)

    # Mixed-topology grid: Bounded x and z, Periodic y. Tests west/east on u
    # and bottom/top on w against Bounded, while south/north on v should stay
    # PeriodicBoundaryCondition.
    grid_bpb = RectilinearGrid(default_arch, FT;
                               size = (Nx, Ny, Nz),
                               x = (0, Lx), y = (0, Ly), z = (0, Lz),
                               topology = (Bounded, Periodic, Bounded))

    # All-Bounded grid: enables testing south/north (v) and bottom/top (w) as
    # normal-component faces under Bounded.
    grid_bbb = RectilinearGrid(default_arch, FT;
                               size = (Nx, Ny, Nz),
                               x = (0, Lx), y = (0, Ly), z = (0, Lz),
                               topology = (Bounded, Bounded, Bounded))

    @testset "Anelastic — Open ρu propagates to u at west/east" begin
        reference_state = ReferenceState(grid_bpb; surface_pressure=FT(101325),
                                                    potential_temperature=FT(300))
        dynamics = AnelasticDynamics(reference_state)

        ρ_b = FT(1.225)
        U_bg = FT(5.0)
        ρu_value = ρ_b * U_bg

        ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(ρu_value),
                                          east = NormalFlowBoundaryCondition(ρu_value))
        model = AtmosphereModel(grid_bpb; dynamics=dynamics,
                                           formulation=:LiquidIcePotentialTemperature,
                                           boundary_conditions=(; ρu=ρu_bcs))

        @test isnothing(model.velocities.u.boundary_conditions.west)
        @test isnothing(model.velocities.u.boundary_conditions.east)

        set!(model; θ=FT(300), ρu=(x,y,z) -> ρ_b * U_bg, ρv=0, ρw=0)
        fill_halo_regions!(model.momentum, model.clock, fields(model))
        compute_velocities!(model)

        # u at west/east boundary face should equal ρu_value / ρ_face ≈ U_bg
        # (column ρ varies <1% over Lz=100m, so 10% rtol is comfortable).
        u_west = boundary_slice(model.velocities.u, 1,    :, :)
        u_east = boundary_slice(model.velocities.u, Nx+1, :, :)
        @test all(u -> isapprox(u, U_bg; rtol=FT(0.1)), u_west)
        @test all(u -> isapprox(u, U_bg; rtol=FT(0.1)), u_east)
    end

    @testset "Anelastic — Open ρv propagates to v at south/north" begin
        reference_state = ReferenceState(grid_bbb; surface_pressure=FT(101325),
                                                    potential_temperature=FT(300))
        dynamics = AnelasticDynamics(reference_state)

        ρ_b = FT(1.225)
        V_bg = FT(5.0)
        ρv_value = ρ_b * V_bg

        ρv_bcs = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(ρv_value),
                                          north = NormalFlowBoundaryCondition(ρv_value))
        model = AtmosphereModel(grid_bbb; dynamics=dynamics,
                                           formulation=:LiquidIcePotentialTemperature,
                                           boundary_conditions=(; ρv=ρv_bcs))

        @test isnothing(model.velocities.v.boundary_conditions.south)
        @test isnothing(model.velocities.v.boundary_conditions.north)

        set!(model; θ=FT(300), ρu=0, ρv=(x,y,z) -> ρ_b * V_bg, ρw=0)
        fill_halo_regions!(model.momentum, model.clock, fields(model))
        compute_velocities!(model)

        v_south = boundary_slice(model.velocities.v, :, 1,    :)
        v_north = boundary_slice(model.velocities.v, :, Ny+1, :)
        @test all(v -> isapprox(v, V_bg; rtol=FT(0.1)), v_south)
        @test all(v -> isapprox(v, V_bg; rtol=FT(0.1)), v_north)
    end

    @testset "Anelastic — Open ρw propagates to w at bottom/top" begin
        reference_state = ReferenceState(grid_bbb; surface_pressure=FT(101325),
                                                    potential_temperature=FT(300))
        dynamics = AnelasticDynamics(reference_state)

        ρ_b = FT(1.225)
        W_bg = FT(0.1)  # Small w to stay in hydrostatic-noise regime
        ρw_value = ρ_b * W_bg

        ρw_bcs = FieldBoundaryConditions(bottom = NormalFlowBoundaryCondition(ρw_value),
                                          top    = NormalFlowBoundaryCondition(ρw_value))
        model = AtmosphereModel(grid_bbb; dynamics=dynamics,
                                           formulation=:LiquidIcePotentialTemperature,
                                           boundary_conditions=(; ρw=ρw_bcs))

        @test isnothing(model.velocities.w.boundary_conditions.bottom)
        @test isnothing(model.velocities.w.boundary_conditions.top)

        set!(model; θ=FT(300), ρu=0, ρv=0, ρw=(x,y,z) -> ρ_b * W_bg)
        fill_halo_regions!(model.momentum, model.clock, fields(model))
        compute_velocities!(model)

        w_bottom = boundary_slice(model.velocities.w, :, :, 1)
        w_top    = boundary_slice(model.velocities.w, :, :, Nz+1)
        @test all(w -> isapprox(w, W_bg; rtol=FT(0.1)), w_bottom)
        @test all(w -> isapprox(w, W_bg; rtol=FT(0.1)), w_top)
    end

    @testset "Compressible — Open ρu propagates to u at west/east" begin
        td = ExplicitTimeStepping()
        dynamics = CompressibleDynamics(td; reference_potential_temperature=FT(300))

        ρ_b = FT(1.225)
        U_bg = FT(5.0)
        ρu_value = ρ_b * U_bg

        ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(ρu_value),
                                          east = NormalFlowBoundaryCondition(ρu_value))
        model = AtmosphereModel(grid_bpb; dynamics=dynamics,
                                           formulation=:LiquidIcePotentialTemperature,
                                           boundary_conditions=(; ρu=ρu_bcs))

        @test isnothing(model.velocities.u.boundary_conditions.west)
        @test isnothing(model.velocities.u.boundary_conditions.east)

        ref = model.dynamics.reference_state
        set!(model; θ=FT(300), ρ=ref.density,
                     ρu=(x,y,z) -> ρ_b * U_bg, ρv=0, ρw=0)
        fill_halo_regions!(model.momentum, model.clock, fields(model))
        compute_velocities!(model)

        u_west = boundary_slice(model.velocities.u, 1,    :, :)
        u_east = boundary_slice(model.velocities.u, Nx+1, :, :)
        @test all(u -> isapprox(u, U_bg; rtol=FT(0.1)), u_west)
        @test all(u -> isapprox(u, U_bg; rtol=FT(0.1)), u_east)
    end

    @testset "Periodic direction → velocity gets PeriodicBoundaryCondition (not nothing)" begin
        # Confirms the auxiliary defaults still install PeriodicBoundaryCondition on
        # Periodic sides (vs `nothing` on Bounded-Face sides). Periodic halo filling
        # must continue to work on tangential and normal components alike.
        reference_state = ReferenceState(grid_bpb; surface_pressure=FT(101325),
                                                    potential_temperature=FT(300))
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid_bpb; dynamics=dynamics,
                                           formulation=:LiquidIcePotentialTemperature)

        # y is Periodic on grid_bpb. Tangential u south/north and normal v south/north
        # should both be Periodic (not nothing).
        pbc_type = typeof(PeriodicBoundaryCondition())
        @test typeof(model.velocities.u.boundary_conditions.south) === pbc_type
        @test typeof(model.velocities.u.boundary_conditions.north) === pbc_type
        @test typeof(model.velocities.v.boundary_conditions.south) === pbc_type
        @test typeof(model.velocities.v.boundary_conditions.north) === pbc_type
        # x is Bounded → u west/east should be the override (nothing).
        @test isnothing(model.velocities.u.boundary_conditions.west)
        @test isnothing(model.velocities.u.boundary_conditions.east)
    end

    @testset "fill_halo_regions! on velocities does not clobber boundary face" begin
        # Direct regression for the original bug mechanism: snapshot u at the boundary
        # face after compute_velocities!, run fill_halo_regions!(velocities), and confirm
        # the boundary face is unchanged. Pre-fix, this re-set u_west to zero.
        reference_state = ReferenceState(grid_bpb; surface_pressure=FT(101325),
                                                    potential_temperature=FT(300))
        dynamics = AnelasticDynamics(reference_state)

        ρ_b = FT(1.225)
        U_bg = FT(5.0)
        ρu_value = ρ_b * U_bg

        ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(ρu_value),
                                          east = NormalFlowBoundaryCondition(ρu_value))
        model = AtmosphereModel(grid_bpb; dynamics=dynamics,
                                           formulation=:LiquidIcePotentialTemperature,
                                           boundary_conditions=(; ρu=ρu_bcs))
        set!(model; θ=FT(300), ρu=(x,y,z) -> ρ_b * U_bg, ρv=0, ρw=0)
        fill_halo_regions!(model.momentum, model.clock, fields(model))
        compute_velocities!(model)

        u_west_before = boundary_slice(model.velocities.u, 1,    :, :)
        u_east_before = boundary_slice(model.velocities.u, Nx+1, :, :)

        fill_halo_regions!(model.velocities)

        u_west_after = boundary_slice(model.velocities.u, 1,    :, :)
        u_east_after = boundary_slice(model.velocities.u, Nx+1, :, :)
        @test u_west_before == u_west_after
        @test u_east_before == u_east_after
        # And both are still close to U_bg (sanity, in case both got clobbered identically).
        @test all(u -> isapprox(u, U_bg; rtol=FT(0.1)), u_west_after)
    end

    @testset "Default impenetrable wall (no user momentum BC) gives u = 0 at wall" begin
        # `nothing` on velocity is the catch-all override, but the default impenetrable
        # behavior must still hold: when momentum has the default NormalFlowBoundaryCondition(0),
        # the kernel writes u = ρu/ρ = 0 at the wall and the wall stays impenetrable.
        reference_state = ReferenceState(grid_bpb; surface_pressure=FT(101325),
                                                    potential_temperature=FT(300))
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid_bpb; dynamics=dynamics,
                                           formulation=:LiquidIcePotentialTemperature)

        @test isnothing(model.velocities.u.boundary_conditions.west)
        @test isnothing(model.velocities.u.boundary_conditions.east)

        set!(model; θ=FT(300))
        fill_halo_regions!(model.momentum, model.clock, fields(model))
        compute_velocities!(model)

        u_west = boundary_slice(model.velocities.u, 1,    :, :)
        u_east = boundary_slice(model.velocities.u, Nx+1, :, :)
        @test all(==(zero(FT)), u_west)
        @test all(==(zero(FT)), u_east)
    end

    @testset "Anelastic — uniform OBC inflow stays uniform under SSP-RK3" begin
        # Pre-fix: this NaN'd within ~100 iters.
        # Post-fix: max|u| should remain ≈ U_bg over a few iters.
        reference_state = ReferenceState(grid_bpb; surface_pressure=FT(101325),
                                                    potential_temperature=FT(300))
        dynamics = AnelasticDynamics(reference_state)

        ρ_b = FT(1.225)
        U_bg = FT(5.0)
        ρu_value = ρ_b * U_bg

        ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(ρu_value),
                                          east = NormalFlowBoundaryCondition(ρu_value))
        model = AtmosphereModel(grid_bpb; dynamics=dynamics,
                                           formulation=:LiquidIcePotentialTemperature,
                                           boundary_conditions=(; ρu=ρu_bcs))
        set!(model; θ=FT(300), ρu=(x,y,z) -> ρ_b * U_bg, ρv=0, ρw=0)

        simulation = Simulation(model; Δt=FT(0.5), stop_iteration=20, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 20
        u_max = @allowscalar maximum(abs, interior(model.velocities.u))
        @test isfinite(u_max)
        @test u_max < FT(2) * U_bg
    end

    @testset "Compressible — uniform OBC inflow stays finite under SSP-RK3 (CFL≈0.05)" begin
        # Companion to the anelastic stability test. Compressible Explicit needs a smaller
        # CFL than the anelastic case (acoustic vs advective). At CFL=0.3 the noise grows
        # to NaN within ~100 iters (Test 5 in the campaign); CFL=0.05 stays stable.
        td = ExplicitTimeStepping()
        dynamics = CompressibleDynamics(td; reference_potential_temperature=FT(300))

        ρ_b = FT(1.225)
        U_bg = FT(5.0)
        ρu_value = ρ_b * U_bg

        ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(ρu_value),
                                          east = NormalFlowBoundaryCondition(ρu_value))
        model = AtmosphereModel(grid_bpb; dynamics=dynamics,
                                           formulation=:LiquidIcePotentialTemperature,
                                           boundary_conditions=(; ρu=ρu_bcs))
        ref = model.dynamics.reference_state
        set!(model; θ=FT(300), ρ=ref.density,
                     ρu=(x,y,z) -> ρ_b * U_bg, ρv=0, ρw=0)

        Δx = Lx / Nx
        c_sound = sqrt(FT(1.4) * FT(287) * FT(300))
        Δt = FT(0.05) * Δx / c_sound
        simulation = Simulation(model; Δt=Δt, stop_iteration=20, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 20
        u_max = @allowscalar maximum(abs, interior(model.velocities.u))
        @test isfinite(u_max)
        @test u_max < FT(2) * U_bg
    end
end
