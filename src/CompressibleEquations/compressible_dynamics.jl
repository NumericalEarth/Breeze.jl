#####
##### CompressibleDynamics definition
#####

using Breeze.TerrainFollowingDiscretization: TerrainMetrics,
                                              TerrainFollowingVerticalDiscretization,
                                              SlopeOutsideInterpolation,
                                              build_terrain_metrics

using Oceananigans.Grids: Bounded, topology

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
- `reference_state`: The single fixed hydrostatically-balanced reference state for base-state
  pressure/buoyancy correction (perturbation-form PGF), or `nothing` when disabled. An
  [`ExnerReferenceState`](@ref) whose fields are grid-polymorphic: a 1D column on
  height-coordinate grids, and horizontally-varying 3D fields on terrain-following grids (where
  a single column is not hydrostatically consistent per terrain column).
- `terrain_metrics`: [`TerrainMetrics`](@ref) for terrain-following coordinates (or `nothing`).
  This — not `reference_state` — is the sole "is this a terrain grid?" signal.
- `w̃`, `ρw̃`: contravariant vertical velocity / momentum diagnostic fields (or `nothing` when no terrain metrics)
- `reference_from_state`: `true` when a reference state was left to be deduced from the initial
  state (no explicit reference profile supplied) on a compatible bounded vertical grid; gates the
  auto-reset in `set!`. `false` otherwise.

The `time_discretization` determines how tendencies are computed and which
time-stepper is used:
- [`SplitExplicitTimeDiscretization`](@ref): Acoustic substepping with separate slow/fast tendencies
- [`ExplicitTimeStepping`](@ref): All tendencies computed together (small Δt required)

The moist equation-of-state θˡⁱ→T temperature inversion is controlled by the
thermodynamic formulation, not the dynamics: see `temperature_solver` on
`LiquidIcePotentialTemperatureFormulation`.
"""
struct CompressibleDynamics{TD, D, DT, P, FT, RS, TM, CV, CM}
    time_discretization :: TD                  # SplitExplicitTimeDiscretization or ExplicitTimeStepping
    dry_density :: D                           # ρᵈ (prognostic dry-air density)
    total_density :: DT                        # ρ = ρᵈ + Σρˣ (diagnosed; thermo/advection/EOS/buoyancy)
    pressure :: P                              # p = ρ R^m T (diagnostic)
    standard_pressure :: FT                    # pˢᵗ (reference pressure for potential temperature)
    surface_pressure :: FT                     # p₀ (mean pressure at the bottom of the atmosphere)
    reference_state :: RS                      # single grid-polymorphic ExnerReferenceState (1D flat / 3D terrain) or Nothing
    terrain_metrics :: TM                      # TerrainMetrics for terrain-following coordinates (or Nothing) — the "is terrain?" signal
    contravariant_vertical_velocity :: CV      # w̃ diagnostic field (or Nothing)
    contravariant_vertical_momentum :: CM      # ρw̃ diagnostic field (or Nothing)
    reference_from_state :: Bool               # reference deduced from the initial state (no explicit spec)
end

# Skeleton-only marker stored in the reference-spec slot for the default `reference_state = :auto`
# with no explicit reference profile: "build a reference state and deduce it from the initial state
# in `set!`". Distinct from `nothing`, which now means the reference is explicitly disabled
# (`reference_state = nothing`). Resolved in `materialize_dynamics`; never survives materialization.
struct AutoReference end

Base.summary(::AutoReference) = "auto (deduced from the initial state by set!)"

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
- `reference_state`: Whether to carry the single hydrostatic reference state used for the
  perturbation-form pressure-gradient force and buoyancy. Default: `:auto` — on a bounded vertical
  grid, build a valid provisional hydrostatic reference (a 1D column on height-coordinate grids,
  3D fields on terrain-following grids) and, absent an explicit profile, deduce it from the initial
  state's horizontal mean when `set!` supplies both density and a thermodynamic variable. Periodic
  and flat vertical topologies carry no automatic reference because a nontrivial hydrostatic
  atmosphere is incompatible with periodicity and unnecessary without a vertical dimension. Pass
  `reference_state = nothing` to disable it entirely — the PGF and buoyancy then difference the
  full pressure, reproducing the un-corrected behavior (useful for testing). Disabling is mutually
  exclusive with an explicit reference profile.

  !!! note "Deep near-isentropic initial states"
      The deduced reference integrates the hydrostatic equation up each column using the mean
      `θˡⁱ(z)`. A (nearly) constant-`θ` column is isentropic and its hydrostatic pressure reaches
      zero at a finite height (≈ `cᵖ θ / g`, about 30 km for `θ = 300` K); if the domain top exceeds
      that height the integration has no positive-pressure solution and `set!` will error (or, on
      GPU, fill the reference with `NaN`). Physical, stably-stratified atmospheres are unaffected.
      For such a deep, near-isentropic setup pass `reference_state = nothing` (full-pressure form).
"""
function CompressibleDynamics(time_discretization::TD = ExplicitTimeStepping();
                              standard_pressure = 1e5,
                              surface_pressure = 101325.0,
                              reference_potential_temperature = nothing,
                              reference_temperature = nothing,
                              reference_vapor_mass_fraction = nothing,
                              slope_stencil = SlopeOutsideInterpolation(),
                              terrain_metrics = nothing,
                              reference_state = :auto,
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
    # An explicit reference profile (θ₀, T₀, or a moist NamedTuple). `nothing` means none was given.
    # If reference_temperature or reference_vapor_mass_fraction is present, wrap in a NamedTuple to
    # distinguish from a bare θ₀ spec.
    profile_spec = if reference_temperature !== nothing
        (; reference_temperature, reference_vapor_mass_fraction)
    elseif reference_vapor_mass_fraction !== nothing
        (; reference_potential_temperature, reference_vapor_mass_fraction)
    else
        reference_potential_temperature
    end
    explicit_profile = reference_potential_temperature !== nothing ||
                       reference_temperature !== nothing ||
                       reference_vapor_mass_fraction !== nothing

    # Resolve the reference-spec slot stored in the skeleton (built in `materialize_dynamics`):
    #   `nothing`            → reference explicitly disabled (`reference_state = nothing`),
    #   an explicit profile  → build the reference from it and preserve it through `set!`,
    #   `AutoReference()`    → build a reference by default and deduce it from the initial state.
    # `reference_state = nothing` is mutually exclusive with an explicit reference profile.
    ref_spec = if reference_state === nothing
        explicit_profile && throw(ArgumentError(
            "`reference_state = nothing` disables the reference state and is mutually exclusive with an \
             explicit reference profile (`reference_potential_temperature`/`reference_temperature`/\
             `reference_vapor_mass_fraction`)."))
        nothing
    else
        explicit_profile ? profile_spec : AutoReference()
    end
    # Stash either a pre-built TerrainMetrics (escape hatch) or the stencil flavor in the
    # `terrain_metrics` slot. `materialize_dynamics` resolves it: a stencil instance triggers
    # `build_terrain_metrics(grid, stencil)` for TFVD grids; on non-TFVD grids the slot is
    # zeroed regardless. The contravariant fields and reference state are built there too.
    terrain_metrics_spec = terrain_metrics === nothing ? slope_stencil : terrain_metrics
    return CompressibleDynamics(time_discretization, nothing, nothing, nothing, pˢᵗ, p₀, ref_spec,
                                terrain_metrics_spec,
                                nothing, nothing, false)
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

    # The reference-spec slot, set by the constructor:
    #   `nothing`           → reference explicitly disabled,
    #   an explicit profile → build the reference from it (preserved by `set!`),
    #   `AutoReference()`   → build a reference by default and deduce it from the state in `set!`.
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

    # Build the single, grid-polymorphic reference state (or `nothing`). The reference is an
    # `ExnerReferenceState` in discrete hydrostatic balance, used identically by the flat and
    # terrain PGF/buoyancy — the only difference is the dimensionality of its fields: a 1D column
    # on height-coordinate grids, and horizontally-varying 3D fields on terrain-following grids
    # (where Δz varies per column, so a single column is not hydrostatically consistent). This
    # supports both split-explicit (acoustic substepping) and explicit time stepping.
    #
    # With `AutoReference()` (the default, no explicit profile), a valid hydrostatic reference is
    # built from the standard 288 K profile and later overwritten by `set!` from the initial state's
    # height-resolved horizontal mean (`reference_from_state`; see `reset_reference_state!`). This
    # ensures the public `reference_state` is usable immediately after model construction. Automatic
    # references are disabled for Periodic and Flat vertical topologies: gravity makes a nontrivial
    # hydrostatic column nonperiodic, while a Flat vertical dimension needs no hydrostatic split.
    # Explicit profiles retain their existing opt-in behavior on every topology.
    auto_reference = ref_spec isa AutoReference
    reference_from_state = auto_reference && topology(grid)[3] === Bounded
    reference_state = if ref_spec === nothing || (auto_reference && !reference_from_state)
        nothing
    elseif auto_reference
        build_reference_state(grid, terrain_metrics, FT(288),
                              surface_pressure, standard_pressure, thermodynamic_constants)
    else
        build_reference_state(grid, terrain_metrics, ref_spec,
                              surface_pressure, standard_pressure, thermodynamic_constants)
    end

    # Contravariant transport fields exist on terrain grids regardless of the reference state.
    if terrain_metrics === nothing
        contravariant_vertical_velocity = nothing
        contravariant_vertical_momentum = nothing
    else
        contravariant_vertical_velocity = ZFaceField(grid)
        contravariant_vertical_momentum = ZFaceField(grid)
    end

    # Seed the diagnostic pressure so the first `update_state!` sat-adjust does not divide by an
    # uninitialized (zero) pressure — `(p/pˢᵗ)^κ` would collapse to zero and produce NaN
    # temperatures. `compute_auxiliary_dynamics_variables!` overwrites pressure on every subsequent
    # call. Seed from any built reference, else from surface pressure.
    if reference_state isa ExnerReferenceState
        seed_pressure!(pressure, grid, reference_state.pressure)
    else
        seed_pressure!(pressure, grid, surface_pressure)
    end

    return CompressibleDynamics(dynamics.time_discretization, density, total_density, pressure,
                                standard_pressure, surface_pressure, reference_state,
                                terrain_metrics,
                                contravariant_vertical_velocity,
                                contravariant_vertical_momentum,
                                reference_from_state)
end

#####
##### Reference-state builders (dispatched on whether the grid is terrain-following)
#####

# Explicit-profile reference on a height-coordinate grid: a 1D-column `ExnerReferenceState`
# (or 3D when the profile depends on the horizontal coordinates). The terrain-following method is
# defined in `terrain_compressible_physics.jl`.
build_reference_state(grid, ::Nothing, ref_spec, surface_pressure, standard_pressure, constants) =
    ExnerReferenceState(grid, constants; surface_pressure, standard_pressure, exner_kwargs(ref_spec)...)

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

# Compressible dynamics whose reference was left to be deduced from the initial state (the default
# `reference_state = :auto` with no explicit profile on a bounded vertical grid) has `set!`
# recompute it from the horizontal mean by default when that call supplies both density and a
# thermodynamic variable. An explicit reference profile is authoritative, so this is `false` and
# `set!` leaves it untouched. The `compute_reference_state` keyword to `set!` overrides either way.
AtmosphereModels.auto_reset_reference_state(dynamics::CompressibleDynamics) = dynamics.reference_from_state

"""
$(TYPEDSIGNATURES)

Return a reference state suitable for boundary-condition diagnostics.

Boundary conditions are materialized before `materialize_dynamics` runs, so the
stub `CompressibleDynamics.reference_state` field still holds the reference *spec*
rather than an `ExnerReferenceState`. When that spec is an explicit
`reference_potential_temperature` profile (a constant, function, or NamedTuple),
this method builds the `ExnerReferenceState` on demand using the same logic as
`materialize_dynamics`. When the reference is disabled (`nothing`) or is to be
deduced from the initial state (`AutoReference`, unavailable at this point), or
when the dynamics has already been materialized, the existing field (or `nothing`)
is returned.
"""
function AtmosphereModels.boundary_conditions_reference_state(dynamics::CompressibleDynamics, grid, thermodynamic_constants)
    ref_spec = dynamics.reference_state
    ref_spec === nothing && return nothing         # reference disabled
    ref_spec isa AutoReference && return nothing    # deduced from the state later; unavailable here
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
