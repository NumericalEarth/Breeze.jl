#####
##### CompressibleDynamics definition
#####

using Breeze.TerrainFollowingDiscretization: TerrainMetrics,
                                              TerrainFollowingVerticalDiscretization,
                                              SlopeOutsideInterpolation,
                                              build_terrain_metrics

"""
$(TYPEDEF)

Fully compressible dynamics with prognostic density and diagnostic pressure.

Fields
======

- `dry_density`: Prognostic dry-air density field ρᵈ
- `total_density`: Diagnosed total air density ρ = ρᵈ + Σρˣ (used for thermodynamics, scalar advection, EOS, buoyancy)
- `pressure`: Diagnostic pressure field p = ρ Rᵐ T
- `standard_pressure`: Reference pressure pˢᵗ for potential temperature (default 10⁵ Pa)
- `surface_pressure`: Mean pressure at the bottom of the atmosphere p₀
- `time_discretization`: Time discretization scheme ([`SplitExplicitTimeDiscretization`](@ref) or [`ExplicitTimeStepping`](@ref))
- `reference_state`: Fixed hydrostatically-balanced reference state for base-state pressure correction (`nothing` or [`ExnerReferenceState`](@ref))
- `terrain_metrics`: [`TerrainMetrics`](@ref) for terrain-following coordinates (or `nothing`)
- `w̃`, `ρw̃`: contravariant vertical velocity / momentum diagnostic fields (or `nothing` when no terrain metrics)
- `terrain_reference_pressure`, `terrain_reference_density`: 3D reference pressure / density for the terrain pressure gradient force (or `nothing`)
- `reference_from_state`: `true` when the terrain reference was left to be deduced from the initial
  state (no explicit reference profile supplied); gates the auto-reset in `set!`. `false` otherwise.

The `time_discretization` determines how tendencies are computed and which
time-stepper is used:
- [`SplitExplicitTimeDiscretization`](@ref): Acoustic substepping with separate slow/fast tendencies
- [`ExplicitTimeStepping`](@ref): All tendencies computed together (small Δt required)

The moist equation-of-state θˡⁱ→T temperature inversion is controlled by the
thermodynamic formulation, not the dynamics: see `temperature_solver` on
`LiquidIcePotentialTemperatureFormulation`.
"""
struct CompressibleDynamics{TD, D, DT, P, FT, RS, TM, CV, CM, TRP, TRD}
    time_discretization :: TD                  # SplitExplicitTimeDiscretization or ExplicitTimeStepping
    dry_density :: D                           # ρᵈ (prognostic dry-air density)
    total_density :: DT                        # ρ = ρᵈ + Σρˣ (diagnosed; thermo/advection/EOS/buoyancy)
    pressure :: P                              # p = ρ R^m T (diagnostic)
    standard_pressure :: FT                    # pˢᵗ (reference pressure for potential temperature)
    surface_pressure :: FT                     # p₀ (mean pressure at the bottom of the atmosphere)
    reference_state :: RS                      # ExnerReferenceState for base-state pressure correction (or Nothing)
    terrain_metrics :: TM                      # TerrainMetrics for terrain-following coordinates (or Nothing)
    contravariant_vertical_velocity :: CV      # w̃ diagnostic field (or Nothing)
    contravariant_vertical_momentum :: CM      # ρw̃ diagnostic field (or Nothing)
    terrain_reference_pressure :: TRP          # 3D reference pressure for terrain PG (or Nothing)
    terrain_reference_density :: TRD           # 3D reference density for terrain buoyancy (or Nothing)
    reference_from_state :: Bool               # terrain reference deduced from the initial state (no explicit spec)
end

# Skeleton-only sentinel stored in the reference-spec slot when `terrain_reference = false`, to
# distinguish "reference explicitly disabled" from "no spec → deduce from state" (both otherwise
# read as `nothing`). Resolved in `materialize_dynamics`; never survives into a materialized object.
struct NoReferenceState end

"""
$(TYPEDSIGNATURES)

Construct `CompressibleDynamics`. The density and pressure fields are materialized
later in the model constructor.

Positional Arguments
====================

- `time_discretization`: Time discretization scheme. Default: [`ExplicitTimeStepping`](@ref).
  Use [`SplitExplicitTimeDiscretization`](@ref) for acoustic substepping.

Keyword Arguments
=================

- `standard_pressure`: Reference pressure for potential temperature (default: 10⁵ Pa)
- `surface_pressure`: Mean surface pressure (default: 101325.0 Pa)
- `reference_potential_temperature`: Potential temperature for building a fixed
  hydrostatically-balanced reference state used in base-state subtraction. Can be a constant `θ₀`
  or a function `θ(z)`. Default: `nothing` (no base-state correction).
  When provided, an [`ExnerReferenceState`](@ref) is built during materialization.
- `reference_vapor_mass_fraction`: Optional vapor mass fraction for building a moist
  compressible reference state. Can be a constant `qᵛ`, function `qᵛ(z)`, or field,
  and is used with `reference_potential_temperature`.
- `slope_stencil`: Pressure-gradient slope-interpolation stencil for terrain-following grids.
  Default: [`SlopeOutsideInterpolation`](@ref). Ignored on non-terrain-following grids.
- `terrain_metrics`: Escape hatch — pass a pre-built [`TerrainMetrics`](@ref) to bypass the
  automatic build. Default: `nothing` (auto-build from the grid using `slope_stencil`).
- `terrain_reference`: Whether to carry a reference state on terrain-following grids (default:
  `true`). By default the terrain reference is enabled and, absent an explicit profile, deduced
  from the initial state's horizontal mean by `set!` (perturbation-form PGF/buoyancy, which
  suppresses the terrain-following pressure-gradient error). Set `false` to disable it entirely —
  the PGF and buoyancy then difference the full pressure, reproducing the un-corrected behavior
  (useful for testing). Mutually exclusive with an explicit reference profile.
"""
function CompressibleDynamics(time_discretization::TD = ExplicitTimeStepping();
                              standard_pressure = 1e5,
                              surface_pressure = 101325.0,
                              reference_potential_temperature = nothing,
                              reference_temperature = nothing,
                              reference_vapor_mass_fraction = nothing,
                              slope_stencil = SlopeOutsideInterpolation(),
                              terrain_metrics = nothing,
                              terrain_reference = true,
                              temperature_tolerance = nothing,
                              temperature_maxiter = nothing) where TD

    if temperature_tolerance !== nothing || temperature_maxiter !== nothing
        throw(ArgumentError("The `temperature_tolerance` and `temperature_maxiter` keyword arguments \
                             have moved from `CompressibleDynamics` to the thermodynamic formulation. \
                             Use, for example, \
                             `formulation = LiquidIcePotentialTemperatureFormulation(temperature_solver = NewtonSolver(abstol=1e-4, maxiter=8))`, \
                             `temperature_solver = FixedIterations(2)` for Reactant / differentiable runs, \
                             or `temperature_solver = nothing` for the non-iterated closed-form inversion."))
    end

    FT = float(promote_type(typeof(standard_pressure), typeof(surface_pressure)))
    pˢᵗ = convert(FT, standard_pressure)
    p₀ = convert(FT, surface_pressure)
    # Store reference spec temporarily; ExnerReferenceState is built in materialize_dynamics.
    # If reference_temperature or reference_vapor_mass_fraction is given, wrap in a
    # NamedTuple to distinguish from a bare θ₀ spec.
    ref_spec = if reference_temperature !== nothing
        (; reference_temperature, reference_vapor_mass_fraction)
    elseif reference_vapor_mass_fraction !== nothing
        (; reference_potential_temperature, reference_vapor_mass_fraction)
    else
        reference_potential_temperature
    end
    # `terrain_reference = false` disables the reference state entirely (full-pressure PGF/buoyancy).
    # It is mutually exclusive with an explicit reference profile; the `NoReferenceState` sentinel
    # keeps this distinct from `nothing` ("deduce from state"), and is resolved in materialization.
    if !terrain_reference
        ref_spec === nothing || throw(ArgumentError(
            "`terrain_reference = false` disables the reference state and is mutually exclusive with an \
             explicit reference profile (`reference_potential_temperature`/`reference_temperature`/\
             `reference_vapor_mass_fraction`)."))
        ref_spec = NoReferenceState()
    end
    # Stash either a pre-built TerrainMetrics (escape hatch) or the stencil flavor in the
    # `terrain_metrics` slot. `materialize_dynamics` resolves it: a stencil instance triggers
    # `build_terrain_metrics(grid, stencil)` for TFVD grids; on non-TFVD grids the slot is
    # zeroed regardless. contravariant fields, terrain_reference_pressure, and
    # terrain_reference_density are built later in materialize_dynamics.
    terrain_metrics_spec = terrain_metrics === nothing ? slope_stencil : terrain_metrics
    return CompressibleDynamics(time_discretization, nothing, nothing, nothing, pˢᵗ, p₀, ref_spec,
                                terrain_metrics_spec,
                                nothing, nothing, nothing, nothing, false)
end

Adapt.adapt_structure(to, dynamics::CompressibleDynamics) =
    CompressibleDynamics(dynamics.time_discretization,
                         adapt(to, dynamics.dry_density),
                         adapt(to, dynamics.total_density),
                         adapt(to, dynamics.pressure),
                         dynamics.standard_pressure,
                         dynamics.surface_pressure,
                         adapt(to, dynamics.reference_state),
                         adapt(to, dynamics.terrain_metrics),
                         adapt(to, dynamics.contravariant_vertical_velocity),
                         adapt(to, dynamics.contravariant_vertical_momentum),
                         adapt(to, dynamics.terrain_reference_pressure),
                         adapt(to, dynamics.terrain_reference_density),
                         dynamics.reference_from_state)

# The compressible θˡⁱ→T inversion is implicit (T = (ρRᵐT/pˢᵗ)^κ θ + ΔL/cᵖᵐ), so the
# default temperature solver iterates to convergence. The anelastic inversion is
# closed-form and uses the generic `nothing` fallback.
AtmosphereModels.default_temperature_solver(::CompressibleDynamics) = NewtonSolver()

# Translate a stored reference spec — a bare θ₀, a (; reference_temperature, …)
# NamedTuple, or a (; reference_potential_temperature, …) NamedTuple — into the
# kwargs accepted by `ExnerReferenceState`. A `nothing` θ in the NamedTuple is
# elided so `ExnerReferenceState`'s own `potential_temperature = 288` default
# takes effect.
exner_kwargs(ref_spec) = (; potential_temperature = ref_spec)
function exner_kwargs(ref_spec::NamedTuple)
    if haskey(ref_spec, :reference_temperature)
        return (; reference_temperature = ref_spec.reference_temperature,
                  vapor_mass_fraction = ref_spec.reference_vapor_mass_fraction)
    elseif ref_spec.reference_potential_temperature === nothing
        return (; vapor_mass_fraction = ref_spec.reference_vapor_mass_fraction)
    else
        return (; potential_temperature = ref_spec.reference_potential_temperature,
                  vapor_mass_fraction = ref_spec.reference_vapor_mass_fraction)
    end
end

#####
##### Materialization
#####

"""
$(TYPEDSIGNATURES)

Materialize a stub `CompressibleDynamics` into a full dynamics object with density and pressure fields.
"""
function AtmosphereModels.materialize_dynamics(dynamics::CompressibleDynamics, grid, boundary_conditions, thermodynamic_constants)
    # Get density boundary conditions if provided
    if haskey(boundary_conditions, :ρᵈ)
        density = CenterField(grid, boundary_conditions=boundary_conditions.ρᵈ)
    else
        density = CenterField(grid)  # Use default for grid topology
    end

    pressure = CenterField(grid)  # Diagnostic pressure from equation of state
    total_density = CenterField(grid)  # Diagnosed total air density ρ = ρᵈ + Σρˣ

    FT = eltype(grid)
    standard_pressure = convert(FT, dynamics.standard_pressure)
    surface_pressure = convert(FT, dynamics.surface_pressure)

    # Build reference state from the stored spec (θ₀, T₀ NamedTuple, or nothing).
    # ExnerReferenceState builds the Exner function π₀ by discrete integration,
    # ensuring exact discrete Exner hydrostatic balance. This is used for both
    # split-explicit (acoustic substepping) and explicit time stepping.
    #
    # For terrain-following grids, the 1D column ExnerReferenceState is NOT used
    # because Δz varies per column. The column-1 reference creates a mismatch at
    # other columns that generates spurious vertical accelerations. Instead, terrain
    # grids use only the 3D terrain_reference_pressure for the horizontal PG.
    ref_spec = dynamics.reference_state

    # Resolve terrain metrics: nothing on non-TFVD grids; on TFVD grids, a pre-built
    # `TerrainMetrics` passes through, anything else (a stencil flavor like
    # `SlopeOutsideInterpolation()`) drives `build_terrain_metrics(grid, ·)`.
    terrain_metrics_spec = dynamics.terrain_metrics
    terrain_metrics = if grid.z isa TerrainFollowingVerticalDiscretization
        terrain_metrics_spec isa TerrainMetrics ?
            terrain_metrics_spec :
            build_terrain_metrics(grid, terrain_metrics_spec)
    else
        nothing
    end

    if ref_spec === nothing || ref_spec isa NoReferenceState || terrain_metrics !== nothing
        reference_state = nothing
    else
        reference_state = ExnerReferenceState(grid, thermodynamic_constants;
                                              surface_pressure, standard_pressure,
                                              exner_kwargs(ref_spec)...)
    end

    # Create contravariant velocity/momentum fields and terrain reference state
    # if terrain metrics are present.
    if terrain_metrics === nothing
        contravariant_vertical_velocity = nothing
        contravariant_vertical_momentum = nothing
        terrain_reference_pressure = nothing
        terrain_reference_density = nothing
    elseif ref_spec isa NoReferenceState
        # Reference explicitly disabled (`terrain_reference = false`): allocate the contravariant
        # transport fields (terrain kinematics still apply) but carry no reference — the PGF and
        # buoyancy fall back to their full-pressure `::Nothing` dispatch.
        contravariant_vertical_velocity = ZFaceField(grid)
        contravariant_vertical_momentum = ZFaceField(grid)
        terrain_reference_pressure = nothing
        terrain_reference_density = nothing
    else
        contravariant_vertical_velocity = ZFaceField(grid)
        contravariant_vertical_momentum = ZFaceField(grid)

        # Allocate the 3D reference pressure/density fields. On terrain-following grids the
        # reference is mandatory: without it the horizontal PG and vertical buoyancy difference
        # the *full* pressure, whose near-cancellation over steep terrain drives spurious
        # slope-correlated surface winds. The reference is horizontally uniform in θ̄(z) but its
        # per-column hydrostatic integration ensures δ(pᵣ)/Δz + g ℑ(ρᵣ) ≈ 0 at every face.
        #
        # An explicit `ref_spec` (user-supplied θ profile) is built here. With no spec, the
        # fields are left zero and the reference is deduced from the initial state's
        # height-resolved horizontal mean by `set!` (compute_reference_state, auto-enabled for
        # terrain *without* an explicit spec) — see `reset_reference_state!`. No fabricated
        # constant profile is used. An explicit spec is preserved: `set!` does not auto-reset it.
        terrain_reference_pressure = CenterField(grid)
        terrain_reference_density = CenterField(grid)
        if ref_spec !== nothing
            compute_terrain_reference_state!(terrain_reference_pressure,
                                             terrain_reference_density,
                                             grid, surface_pressure, ref_spec,
                                             standard_pressure, thermodynamic_constants)
        end
    end

    # Terrain grids with no explicit reference spec deduce their (mandatory) reference from the
    # initial state, so `set!` auto-resets by default; an explicit spec is honored as-is.
    reference_from_state = terrain_metrics !== nothing && ref_spec === nothing

    # Seed the diagnostic pressure so the first `update_state!` sat-adjust does not divide by an
    # uninitialized (zero) pressure — `(p/pˢᵗ)^κ` would collapse to zero and produce NaN
    # temperatures. `compute_auxiliary_dynamics_variables!` overwrites pressure on every
    # subsequent call. Seed from a built reference when one exists, else from surface pressure
    # (which also covers the deferred terrain case whose reference is filled later by `set!`).
    reference_built = terrain_reference_pressure !== nothing && ref_spec !== nothing
    if reference_state isa ExnerReferenceState
        seed_pressure!(pressure, grid, reference_state.pressure)
    elseif reference_built
        seed_pressure!(pressure, grid, terrain_reference_pressure)
    else
        seed_pressure!(pressure, grid, surface_pressure)
    end

    return CompressibleDynamics(dynamics.time_discretization, density, total_density, pressure,
                                standard_pressure, surface_pressure, reference_state,
                                terrain_metrics,
                                contravariant_vertical_velocity,
                                contravariant_vertical_momentum,
                                terrain_reference_pressure, terrain_reference_density,
                                reference_from_state)
end

function seed_pressure!(pressure, grid, pressure_reference)
    arch = grid.architecture
    launch!(arch, grid, :xyz, _seed_pressure_from_field!, pressure, pressure_reference)
    fill_halo_regions!(pressure)
    return nothing
end

function seed_pressure!(pressure, grid, pressure_value::Number)
    arch = grid.architecture
    launch!(arch, grid, :xyz, _seed_pressure_from_value!, pressure, pressure_value)
    fill_halo_regions!(pressure)
    return nothing
end

@kernel function _seed_pressure_from_field!(pressure, pressure_reference)
    i, j, k = @index(Global, NTuple)
    @inbounds pressure[i, j, k] = pressure_reference[i, j, k]
end

@kernel function _seed_pressure_from_value!(pressure, pressure_value)
    i, j, k = @index(Global, NTuple)
    @inbounds pressure[i, j, k] = pressure_value
end

#####
##### Pressure interface
#####

"""
$(TYPEDSIGNATURES)

Return the mean (reference) pressure for `CompressibleDynamics`.
For compressible dynamics, there is no separate mean pressure - returns the full pressure field.
"""
AtmosphereModels.mean_pressure(dynamics::CompressibleDynamics) = dynamics.pressure

"""
$(TYPEDSIGNATURES)

Return the pressure anomaly for `CompressibleDynamics`.
For compressible dynamics, there is no decomposition - returns zero.
"""
AtmosphereModels.pressure_anomaly(dynamics::CompressibleDynamics) = 0

"""
$(TYPEDSIGNATURES)

Return the total pressure for `CompressibleDynamics`, in Pa.
"""
AtmosphereModels.total_pressure(dynamics::CompressibleDynamics) = dynamics.pressure

#####
##### Density and pressure access interface
#####

"""
$(TYPEDSIGNATURES)

Return the prognostic density field for `CompressibleDynamics`.
"""
AtmosphereModels.dynamics_density(dynamics::CompressibleDynamics) = dynamics.dry_density

"""
$(TYPEDSIGNATURES)

Return a `CompressibleDynamics` identical to `dynamics` but with its `time_discretization`
replaced. Every field (densities, pressure, reference and terrain states) is shared by
reference — only the immutable scheme wrapper changes — so this allocates no field memory. Used
to build the adiabatic-balance twin (an `ExplicitTimeStepping` view of a production model).
"""
with_time_discretization(dynamics::CompressibleDynamics, time_discretization) =
    CompressibleDynamics(time_discretization,
                         dynamics.dry_density,
                         dynamics.total_density,
                         dynamics.pressure,
                         dynamics.standard_pressure,
                         dynamics.surface_pressure,
                         dynamics.reference_state,
                         dynamics.terrain_metrics,
                         dynamics.contravariant_vertical_velocity,
                         dynamics.contravariant_vertical_momentum,
                         dynamics.terrain_reference_pressure,
                         dynamics.terrain_reference_density,
                         dynamics.reference_from_state)

"""
$(TYPEDSIGNATURES)

Return a copy of `time_discretization` with its upper sponge removed. The adiabatic-balance
excursion must be reversible, and the sponge (like divergence damping) is an irreversible term;
`balance_adiabatically!` therefore requires a sponge-free model. No-op for discretizations that
carry no sponge (e.g. `ExplicitTimeStepping`).
"""
without_sponge(time_discretization) = time_discretization

without_sponge(td::SplitExplicitTimeDiscretization) =
    SplitExplicitTimeDiscretization(td.substeps,
                                    td.acoustic_cfl,
                                    td.forward_weight,
                                    td.thermodynamic_tendency_factor,
                                    td.vertical_momentum_tendency_factor,
                                    td.vertical_pressure_tendency_factor,
                                    td.final_stage_vertical_pressure_tendency_factor,
                                    td.apply_first_substep_pressure_gradient,
                                    td.damping,
                                    nothing,
                                    td.substep_distribution,
                                    td.open_boundary_relaxation)

# Adiabatic-balance twin dynamics (extends the solver-agnostic fallback in AtmosphereModels). The
# sponge is always stripped — it is irreversible. The default builds the fully-explicit twin
# (memory-minimal, cleanly reversible); `nothing` reuses the model's native scheme; any other value
# is taken as the twin's time discretization.
AtmosphereModels.adiabatic_twin_dynamics(dynamics::CompressibleDynamics, ::AtmosphereModels.DefaultTimeStepping) =
    with_time_discretization(dynamics, ExplicitTimeStepping())

AtmosphereModels.adiabatic_twin_dynamics(dynamics::CompressibleDynamics, ::Nothing) =
    with_time_discretization(dynamics, without_sponge(dynamics.time_discretization))

AtmosphereModels.adiabatic_twin_dynamics(dynamics::CompressibleDynamics, time_stepping) =
    with_time_discretization(dynamics, without_sponge(time_stepping))

# Total air density ρ = ρᵈ + Σρˣ (diagnosed once per update into `total_density`); this is the
# density used by the thermodynamics, scalar advection, equation of state, and buoyancy. The
# coupling density `dynamics_density` (ρᵈ) is used only by velocity/momentum/continuity/ρθ.
AtmosphereModels.total_density(dynamics::CompressibleDynamics) = dynamics.total_density

"""
$(TYPEDSIGNATURES)

Return the pressure field for `CompressibleDynamics`.
Pressure is computed diagnostically from the equation of state.
"""
AtmosphereModels.dynamics_pressure(dynamics::CompressibleDynamics) = dynamics.pressure

#####
##### Prognostic fields
#####

# Compressible dynamics has prognostic density
AtmosphereModels.prognostic_dynamics_field_names(::CompressibleDynamics) = (:ρᵈ,)
AtmosphereModels.additional_dynamics_field_names(::CompressibleDynamics) = ()

"""
$(TYPEDSIGNATURES)

Return prognostic fields specific to compressible dynamics.
Returns the density field as a prognostic variable.
"""
AtmosphereModels.dynamics_prognostic_fields(dynamics::CompressibleDynamics) = (; ρᵈ=dynamics.dry_density)

"""
$(TYPEDSIGNATURES)

Return a standard surface pressure for boundary condition regularization.
For compressible dynamics, uses the standard atmospheric pressure (101325 Pa).
"""
AtmosphereModels.surface_pressure(dynamics::CompressibleDynamics) = dynamics.surface_pressure

"""
$(TYPEDSIGNATURES)

Return the standard pressure for potential temperature calculations.
"""
AtmosphereModels.standard_pressure(dynamics::CompressibleDynamics) = dynamics.standard_pressure

AtmosphereModels.dynamics_reference_state(dynamics::CompressibleDynamics) = dynamics.reference_state

"""
$(TYPEDSIGNATURES)

Return a reference state suitable for boundary-condition diagnostics.

Boundary conditions are materialized before `materialize_dynamics` runs, so the
stub `CompressibleDynamics.reference_state` field still holds the user's raw
`reference_potential_temperature` spec (a constant, function, or NamedTuple)
rather than an `ExnerReferenceState`. When called on the stub, this method
builds the `ExnerReferenceState` on demand using the same logic as
`materialize_dynamics`. When the dynamics has already been materialized (or has
no reference state), the existing field is returned.
"""
function AtmosphereModels.boundary_conditions_reference_state(dynamics::CompressibleDynamics, grid, thermodynamic_constants)
    ref_spec = dynamics.reference_state
    ref_spec === nothing && return nothing
    ref_spec isa ExnerReferenceState && return ref_spec

    standard_pressure = dynamics.standard_pressure
    surface_pressure = dynamics.surface_pressure

    return ExnerReferenceState(grid, thermodynamic_constants;
                               surface_pressure, standard_pressure,
                               exner_kwargs(ref_spec)...)
end

"""
$(TYPEDSIGNATURES)

`BulkDrag` under `CompressibleDynamics` requires the user to supply
`surface_temperature` explicitly. Unlike `AnelasticDynamics`, compressible
dynamics does not carry a reference profile from which a surface temperature
can be unambiguously derived. A clean default would require either coupling
to a surface model or diagnosing the surface state from the prognostic fields
(which would make ρ₀ grid-dependent and break MO consistency at the surface);
both are out of scope for now.
"""
AtmosphereModels.default_drag_surface_temperature(::CompressibleDynamics, grid, constants) =
    throw(ArgumentError(
        "BulkDrag under CompressibleDynamics requires `surface_temperature` to be " *
        "provided explicitly. There is no default surface temperature for compressible " *
        "dynamics (no reference profile to draw from). Construct BulkDrag with a " *
        "`surface_temperature` keyword (a `Number`, `Function`, or `Field`)."))

#####
##### Pressure solver (none needed for compressible dynamics)
#####

"""
$(TYPEDSIGNATURES)

Return `nothing` for `CompressibleDynamics` - no pressure solver is needed.
Pressure is computed directly from the equation of state.
"""
AtmosphereModels.dynamics_pressure_solver(dynamics::CompressibleDynamics, grid) = nothing

"""
$(TYPEDSIGNATURES)

Return the default timestepper for `CompressibleDynamics` based on its `time_discretization`.

- [`SplitExplicitTimeDiscretization`](@ref): Returns `:AcousticRungeKutta3` for acoustic substepping
- [`ExplicitTimeStepping`](@ref): Returns `:SSPRungeKutta3` for standard explicit time-stepping
"""
AtmosphereModels.default_timestepper(dynamics::CompressibleDynamics) =
    default_timestepper(dynamics.time_discretization)

default_timestepper(::SplitExplicitTimeDiscretization) = :AcousticRungeKutta3
default_timestepper(::ExplicitTimeStepping) = :SSPRungeKutta3

#####
##### Show methods
#####

Base.summary(::SplitExplicitTimeDiscretization) = "SplitExplicitTimeDiscretization"
Base.summary(::ExplicitTimeStepping) = "ExplicitTimeStepping"

function Base.summary(dynamics::CompressibleDynamics)
    td = summary(dynamics.time_discretization)
    return "CompressibleDynamics{$td}"
end

function Base.show(io::IO, dynamics::CompressibleDynamics)
    print(io, summary(dynamics), '\n')
    if dynamics.dry_density === nothing
        print(io, "├── dry_density: not materialized\n")
        print(io, "├── pressure: not materialized\n")
    else
        print(io, "├── dry_density: ", prettysummary(dynamics.dry_density), '\n')
        print(io, "├── total_density: ", prettysummary(dynamics.total_density), '\n')
        print(io, "├── pressure: ", prettysummary(dynamics.pressure), '\n')
    end
    print(io, "├── terrain_metrics: ", summary(dynamics.terrain_metrics), '\n')
    print(io, "├── time_discretization: ", summary(dynamics.time_discretization), '\n')
    print(io, "└── reference_state: ", summary(dynamics.reference_state))
end

#####
##### Momentum and velocity materialization
#####

function AtmosphereModels.materialize_momentum_and_velocities(::CompressibleDynamics, grid, boundary_conditions)
    ρu = XFaceField(grid, boundary_conditions=boundary_conditions.ρu)
    ρv = YFaceField(grid, boundary_conditions=boundary_conditions.ρv)
    # On a terrain-following grid, `ρw` gets the kinematic terrain bottom BC
    # (ρw|₁ = slopeₓ·ρu + slopeᵧ·ρv ⟺ w̃|₁ = 0), dispatched by grid type; other
    # grids keep their given BCs. See `terrain_ρw_boundary_conditions`.
    ρw = ZFaceField(grid, boundary_conditions=terrain_ρw_boundary_conditions(grid, boundary_conditions.ρw))
    momentum = (; ρu, ρv, ρw)

    # Velocity is diagnostic (u = ρu/ρ via compute_velocities!). Use the auxiliary-field
    # default BCs (`nothing` on Bounded-Face sides, Periodic on Periodic sides), which
    # is what XFaceField gives us when constructed with no `boundary_conditions=` kwarg.
    # `nothing` on Bounded-Face prevents `fill_halo_regions!(velocities)` from clobbering
    # the kernel-computed boundary face — momentum carries the wall BC.
    u = XFaceField(grid)
    v = YFaceField(grid)
    w = ZFaceField(grid)
    velocities = (; u, v, w)

    return momentum, velocities
end

#####
##### Potential temperature diagnostics interface
#####

"""
$(TYPEDSIGNATURES)

Return the pressure field for potential temperature diagnostics.
For compressible dynamics, uses the actual pressure field.
"""
AtmosphereModels.Diagnostics.dynamics_pressure_for_potential_temperature(dynamics::CompressibleDynamics) = dynamics.pressure

"""
$(TYPEDSIGNATURES)

Return the density field for potential temperature diagnostics.
For compressible dynamics, uses the diagnosed total air density (the moisture fractions need total ρ).
"""
AtmosphereModels.Diagnostics.dynamics_density_for_potential_temperature(dynamics::CompressibleDynamics) = dynamics.total_density

"""
$(TYPEDSIGNATURES)

Return the standard pressure for potential temperature diagnostics.
"""
AtmosphereModels.Diagnostics.dynamics_standard_pressure(dynamics::CompressibleDynamics) = dynamics.standard_pressure
