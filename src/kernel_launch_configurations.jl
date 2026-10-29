#####
##### Per-kernel launch configuration of Breeze's GPU kernels
#####
##### Port of the handoff launch table (2317a823 L10, 7db6c7e1 / 8d26f3e0 / 59f09b94 L10b, af440d17 L1, 32031b88 L2,
##### fa0b3a75 L1 static shapes, 575e003c L3, 6483995e static-shape groups; ed127fa7 L4 has no kernel here) onto e8.
##### Only entries whose kernel exists in this build are made (`(module, name)` pairs checked with `isdefined`).
#####
##### Where the Oceananigans build's cached launch takes a per-kernel configuration (`kernel_launch_workgroup`,
##### `kernel_launch_maxregs`, `kernel_launch_static_shapes`: methods on the type of a kernel's GPU function), these
##### entries change only how the same kernel body is launched: which thread computes which index, how many registers a
##### thread is allotted, and whether array dimensions are compile-time constants. Every kernel listed computes the
##### value at an index from that index alone (no shared memory, group size or barrier), so its results are the same
##### bits under any entry. The measurements quoted are the handoff's (387 × 252 × 50 and 1679 × 1092 × 50, H200).
#####
##### Defaults: the handoff keyed its workgroup entries on the configured default `Val{(16, 16)}`. Our Oceananigans
##### configures `(32, 4)` for three-dimensional grids, so the entries here match either configured default
##### (`configured_default_workgroup`) and leave any other explicitly chosen workgroup alone.
#####
##### Switches (read at load in `Utils.__init__`; the entries are defined in `Breeze.__init__`, so they apply or not
##### without recompiling): `BREEZE_KERNEL_LAUNCH_TABLE=false` defines none of them, `BREEZE_LAUNCH_MAXREGS=false` none
##### of the register caps, `BREEZE_STATIC_SHAPES_<GROUP>=false` / `BREEZE_STATIC_SHAPES_GROUPS=false` turn off groups
##### of the size-keyed static-shape entries. Oceananigans' `OCEANANIGANS_KERNEL_LAUNCH_WORKGROUPS`, `_MAXREGS`,
##### `OCEANANIGANS_STATIC_SHAPES` and `OCEANANIGANS_CACHED_KERNEL_LAUNCH` act on the launch side.
#####

import KernelAbstractions

const launch_table_available = isdefined(Oceananigans.Utils, :kernel_launch_workgroup)
const static_shapes_available = isdefined(Oceananigans.Utils, :kernel_launch_static_shapes)

"""
The configured workgroups (Oceananigans' three-dimensional default, before and after `(32, 4)`) that a workgroup
entry replaces.
"""
const configured_default_workgroups = ((16, 16), (32, 4))

@inline replaces_default(default) = default in configured_default_workgroups

present_kernels(entries) = Tuple(getfield(m, name) => value for ((m, name), value) in entries if isdefined(m, name))

#####
##### Workgroup (32, 8): warps of 32 consecutive i, for column solves and pointwise acoustic bookkeeping
#####

const x_contiguous_workgroup = (32, 8)

const x_contiguous_kernel_names = (
    (TimeSteppers, :_rk3_implicit_scalar_solve!),
    (TimeSteppers, :_implicit_diffusion_solve!),
    (CompressibleEquations, :_solve_acoustic_tridiagonal!),
    (CompressibleEquations, :_initialize_stage_perturbations!),
    (CompressibleEquations, :_initialize_scalar_and_vertical_perturbations!),
    # ours (T17 acpert): the one-launch stage initialization, as the handoff kernel it stands in for
    (CompressibleEquations, :_initialize_stage_accumulators_and_perturbations!),
    (CompressibleEquations, :_build_vertical_rhs_carried!),
    (CompressibleEquations, :_assemble_terrain_slow_vertical_momentum_tendency!),
    (CompressibleEquations, :_explicit_horizontal_step!),
    (CompressibleEquations, :_post_solve_recovery!),
    (CompressibleEquations, :_assemble_terrain_slow_vertical_momentum_tendency_stored!),
    (CompressibleEquations, :_finalize_time_averaged_velocity!),
    (CompressibleEquations, :_recover_full_state!),
    (CompressibleEquations, :_recover_full_state_after_solve!),
    (CompressibleEquations, :_compute_contravariant_velocity!),
    (TurbulenceClosures, :_add_tke_tendencies!),
    # ours (E-multirhs-solve): the multi-right-hand-side column solve, the scalar-solve choice (32 × 8, no cap)
    (AtmosphereModels, :_multiple_right_hand_side_solve!),
)

#####
##### Register caps (L10b, after the follow-ups removing caps on configuration-dependent kernels)
#####

const register_capped_kernel_names = (
    (CompressibleEquations, :_build_vertical_rhs!) => 64,
    (TurbulenceClosures, :_compute_tke_closure_fields!) => 32,
    (TimeSteppers, :_rk3_implicit_scalar_solve!) => 64,
    (TimeSteppers, :_implicit_diffusion_solve!) => 64,
)

#####
##### k-tiled workgroups on launches of at least `k_tile_min_level_cells` cells per level (L1, L2, L3)
#####

"""
    k_tile_min_level_cells

The number of cells per level from which the k-tiled entries apply: `2^18` by default (compile-time preference
`k_tile_min_level_cells` under `[Breeze]`). The 13 km grid (387 × 252) is below it; the 3 km grid is above.
"""
const k_tile_min_level_cells = let uuid = Base.UUID("660aa2fb-d4c8-4359-a52c-9c057bc511da")  # Breeze
    Base.record_compiletime_preference(uuid, "k_tile_min_level_cells")
    Int(get(Base.get_preferences(uuid), "k_tile_min_level_cells", 2^18))::Int
end

@inline k_tiled_launch(launch_size::Tuple) =
    length(launch_size) == 3 && launch_size[1] * launch_size[2] >= k_tile_min_level_cells

const k_tiled_kernel_names = (
    (CompressibleEquations, :_store_linearization_pressure_gradients!) => (32, 1, 4),
    (CompressibleEquations, :_assemble_terrain_slow_vertical_momentum_tendency!) => (32, 1, 8),
    (CompressibleEquations, :_initialize_terrain_vertical_momentum_perturbation!) => (64, 1, 4),
    (CompressibleEquations, :_initialize_terrain_vertical_momentum_perturbation_loaded!) => (64, 1, 4),
    (CompressibleEquations, :_assemble_terrain_slow_vertical_momentum_tendency_stored!) => (32, 1, 8),
    (CompressibleEquations, :_explicit_horizontal_step!) => (32, 1, 4),
    (CompressibleEquations, :_build_predictors!) => (32, 1, 4),
    (CompressibleEquations, :_build_vertical_rhs!) => (64, 1, 4),
    (CompressibleEquations, :_post_solve_recovery!) => (64, 1, 4),
    (CompressibleEquations, :_finalize_time_averaged_velocity!) => (128, 1, 2),
    (CompressibleEquations, :_recover_full_state!) => (64, 1, 4),
    (CompressibleEquations, :_recover_full_state_after_solve!) => (64, 1, 4),
    (AtmosphereModels, :_compute_velocities!) => (64, 1, 4),
    (Utils, :_compute_tendency!) => (32, 1, 4),
    (AtmosphereModels, :compute_z_momentum_tendency!) => (32, 1, 4),
    (AtmosphereModels, :compute_scalar_tendency!) => (32, 1, 4),
    (Advection, :_x_momentum_bounded_face_fluxes!) => (32, 1, 4),
    (Advection, :_y_momentum_bounded_face_fluxes!) => (32, 1, 4),
    (Advection, :_z_momentum_bounded_face_fluxes!) => (32, 1, 4),
    (TurbulenceClosures, :_compute_tke_closure_fields!) => (64, 1, 4),
    (TurbulenceClosures, :_add_tke_tendencies!) => (64, 1, 4),
)

#####
##### Register caps on large column launches
#####

const register_cap_min_columns = k_tile_min_level_cells

@inline large_column_launch(launch_size::Tuple) =
    length(launch_size) == 2 && launch_size[1] * launch_size[2] >= register_cap_min_columns

const register_capped_column_kernel_names = (
    (CompressibleEquations, :_solve_acoustic_tridiagonal!) => 40,
)

#####
##### Static shapes (fa0b3a75; groups on large launches 6483995e)
#####

const static_shape_kernel_names = (
    (AtmosphereModels, :compute_x_momentum_tendency!),
    (AtmosphereModels, :compute_y_momentum_tendency!),
    (AtmosphereModels, :compute_z_momentum_tendency!),
    (PotentialTemperatureFormulations, :compute_potential_temperature_tendency!),
    (CompressibleEquations, :_compute_terrain_density_tendency!),
    (CompressibleEquations, :_compute_adjusted_compressible_thermodynamics!),
    (CompressibleEquations, :_compute_terrain_fused_thermodynamics!),
    (CompressibleEquations, :_compute_forwarded_terrain_fused_thermodynamics!),
    (CompressibleEquations, :_build_vertical_rhs_carried!),
    (CompressibleEquations, :_assemble_terrain_slow_vertical_momentum_tendency_stored!),
    (CompressibleEquations, :_store_linearization_pressure_gradients!),
    (CompressibleEquations, :_assemble_terrain_slow_vertical_momentum_tendency!),
    (CompressibleEquations, :_explicit_horizontal_step!),
    (CompressibleEquations, :_build_vertical_rhs!),
    (Advection, :_compute_bounds_preserving_edges!),
    (Utils, :_add_tendency_sources!),
    (Advection, :_x_momentum_bounded_face_fluxes!),
    (Advection, :_y_momentum_bounded_face_fluxes!),
    (Advection, :_z_momentum_bounded_face_fluxes!),
    (Advection, :_kfirst_bounds_preserving_edges!),
)

const grouped_static_shape_kernel_names = (
    (AtmosphereModels, :compute_scalar_tendency!) => :scalar_transports,
    (CompressibleEquations, :_build_predictors!) => :ktiled_gains,
    (TurbulenceClosures, :_compute_tke_closure_fields!) => :ktiled_gains,
    (AtmosphereModels, :_compute_velocities!) => :ktiled_gains,
    (CompressibleEquations, :_finalize_time_averaged_velocity!) => :recovery,
    (CompressibleEquations, :_recover_full_state_after_solve!) => :recovery,
    (AtmosphereModels, :_compute_batched_z_bcs!) => :misc,
    (TurbulenceClosures, :_add_tke_tendencies!) => :misc,
)

static_ndrange_size(::Type{KernelAbstractions.NDIteration.StaticSize{S}}) where S = S
static_ndrange_size(::Type{Oceananigans.Utils.OffsetStaticSize{S}}) where S = map(r -> r isa Integer ? Int(r) : length(r), S)
static_ndrange_size(::Type) = ()

"Whether the launch whose argument tuple type is `TT` (the kernel's context first) is large: k-tiled (3-D) or a large column launch (2-D)."
function large_static_launch(::Type{TT}) where TT
    C = fieldtype(TT, 1)
    C <: KernelAbstractions.CompilerMetadata || return false
    S = static_ndrange_size(C.parameters[1])
    S isa Tuple{Vararg{Int}} || return false
    return length(S) == 3 ? k_tiled_launch(S) : length(S) == 2 ? large_column_launch(S) : false
end

#####
##### Definition of the entries (from `Breeze.__init__`, so the switches act at load)
#####

kernel_function_type(kernel!) = typeof(Oceananigans.Utils.gpu_kernel_function(kernel!))

"""
    define_kernel_launch_configurations!()

Define Breeze's per-kernel launch entries on Oceananigans' launch-configuration functions, as the switches
`Utils.KERNEL_LAUNCH_TABLE`, `Utils.LAUNCH_MAXREGS` say. Returns the number of entries defined.
"""
function define_kernel_launch_configurations!()
    launch_table_available && Utils.KERNEL_LAUNCH_TABLE[] || return 0
    n = 0
    UT = Oceananigans.Utils

    for (kernel!, _) in present_kernels(map(name -> name => nothing, x_contiguous_kernel_names))
        F = kernel_function_type(kernel!)
        @eval $UT.kernel_launch_workgroup(::$F, ::Val{D}) where D = replaces_default(D) ? $x_contiguous_workgroup : D
        n += 1
    end

    if hasmethod(UT.kernel_launch_workgroup, Tuple{Any, Val, Val})
        for (kernel!, workgroup) in present_kernels(k_tiled_kernel_names)
            F = kernel_function_type(kernel!)
            @eval $UT.kernel_launch_workgroup(f::$F, default::Val{D}, ::Val{L}) where {D, L} =
                replaces_default(D) && k_tiled_launch(L) ? $workgroup : $UT.kernel_launch_workgroup(f, default)
            n += 1
        end
    end

    if Utils.LAUNCH_MAXREGS[]
        for (kernel!, cap) in present_kernels(register_capped_kernel_names)
            F = kernel_function_type(kernel!)
            @eval $UT.kernel_launch_maxregs(::$F) = $cap
            n += 1
        end
        if hasmethod(UT.kernel_launch_maxregs, Tuple{Any, Val})
            for (kernel!, cap) in present_kernels(register_capped_column_kernel_names)
                F = kernel_function_type(kernel!)
                @eval $UT.kernel_launch_maxregs(f::$F, ::Val{L}) where L =
                    large_column_launch(L) ? $cap : $UT.kernel_launch_maxregs(f)
                n += 1
            end
        end
    end

    if static_shapes_available
        for (kernel!, _) in present_kernels(map(name -> name => nothing, static_shape_kernel_names))
            F = kernel_function_type(kernel!)
            @eval $UT.kernel_launch_static_shapes(::$F, ::Type) = true
            n += 1
        end
        for (kernel!, group) in present_kernels(grouped_static_shape_kernel_names)
            F = kernel_function_type(kernel!)
            @eval $UT.kernel_launch_static_shapes(::$F, ::Type{TT}) where TT =
                Utils.STATIC_SHAPES_GROUPS[$(QuoteNode(group))] && large_static_launch(TT)
            n += 1
        end
    end

    return n
end
