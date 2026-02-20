#####
##### CompressibleDynamics definition
#####

"""
$(TYPEDEF)

Fully compressible dynamics with prognostic density and diagnostic pressure.

Fields
======

- `density`: Prognostic density field ρ
- `pressure`: Diagnostic pressure field p = ρ Rᵐ T
- `standard_pressure`: Reference pressure pˢᵗ for potential temperature (default 10⁵ Pa)
- `surface_pressure`: Mean pressure at the bottom of the atmosphere p₀
- `time_discretization`: Time discretization scheme ([`SplitExplicitTimeDiscretization`](@ref) or [`ExplicitTimeStepping`](@ref))
- `reference_state`: Fixed hydrostatically-balanced reference state for base-state pressure correction (`nothing` or [`ExnerReferenceState`](@ref))

The `time_discretization` determines how tendencies are computed and which
time-stepper is used:
- [`SplitExplicitTimeDiscretization`](@ref): Acoustic substepping with separate slow/fast tendencies
- [`ExplicitTimeStepping`](@ref): All tendencies computed together (small Δt required)
"""
struct CompressibleDynamics{TD, D, P, FT, RS, TM, CV, CM, TRP, TRD}
    time_discretization :: TD # SplitExplicitTimeDiscretization or ExplicitTimeStepping
    density :: D              # ρ (prognostic)
    pressure :: P             # p = ρ R^m T (diagnostic)
    standard_pressure :: FT   # pˢᵗ (reference pressure for potential temperature)
    surface_pressure :: FT    # p₀ (mean pressure at the bottom of the atmosphere)
    reference_state :: RS     # ExnerReferenceState for base-state pressure correction (or Nothing)
    terrain_metrics :: TM     # TerrainMetrics for terrain-following coordinates (or Nothing)
    Ω̃ :: CV                   # Contravariant vertical velocity diagnostic field (or Nothing)
    ρΩ̃ :: CM                  # Contravariant vertical momentum diagnostic field (or Nothing)
    terrain_reference_pressure :: TRP # 3D reference pressure for terrain PG (or Nothing)
    terrain_reference_density :: TRD  # 3D reference density for terrain buoyancy (or Nothing)
end

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
"""
function CompressibleDynamics(time_discretization::TD = ExplicitTimeStepping();
                              standard_pressure = 1e5,
                              surface_pressure = 101325.0,
                              reference_potential_temperature = nothing,
                              terrain_metrics = nothing) where TD

    FT = promote_type(typeof(standard_pressure), typeof(surface_pressure))
    pˢᵗ = convert(FT, standard_pressure)
    p₀ = convert(FT, surface_pressure)
    # Store reference_potential_temperature temporarily; ExnerReferenceState is built in materialize_dynamics
    # terrain_metrics is passed through and stored; Ω̃ and ρΩ̃ are created during materialization
    return CompressibleDynamics(time_discretization, nothing, nothing, pˢᵗ, p₀,
                                reference_potential_temperature, terrain_metrics,
                                nothing, nothing, nothing, nothing)
end

Adapt.adapt_structure(to, dynamics::CompressibleDynamics) =
    CompressibleDynamics(dynamics.time_discretization,
                         adapt(to, dynamics.density),
                         adapt(to, dynamics.pressure),
                         dynamics.standard_pressure,
                         dynamics.surface_pressure,
                         adapt(to, dynamics.reference_state),
                         adapt(to, dynamics.terrain_metrics),
                         adapt(to, dynamics.Ω̃),
                         adapt(to, dynamics.ρΩ̃),
                         adapt(to, dynamics.terrain_reference_pressure),
                         adapt(to, dynamics.terrain_reference_density))

#####
##### Materialization
#####

"""
$(TYPEDSIGNATURES)

Materialize a stub `CompressibleDynamics` into a full dynamics object with density and pressure fields.
"""
function AtmosphereModels.materialize_dynamics(dynamics::CompressibleDynamics, grid, boundary_conditions, thermodynamic_constants)
    # Get density boundary conditions if provided
    if haskey(boundary_conditions, :ρ)
        density = CenterField(grid, boundary_conditions=boundary_conditions.ρ)
    else
        density = CenterField(grid)  # Use default for grid topology
    end

    pressure = CenterField(grid)  # Diagnostic pressure from equation of state

    FT = eltype(grid)
    standard_pressure = convert(FT, dynamics.standard_pressure)
    surface_pressure = convert(FT, dynamics.surface_pressure)

    # Build reference state if reference_potential_temperature was provided.
    # ExnerReferenceState builds the Exner function π₀ by discrete integration,
    # ensuring exact discrete Exner hydrostatic balance. This is used for both
    # split-explicit (acoustic substepping) and explicit time stepping.
    #
    # For terrain-following grids, the 1D column ExnerReferenceState is NOT used
    # because Δz varies per column. The column-1 reference creates a mismatch at
    # other columns that generates spurious vertical accelerations. Instead, terrain
    # grids use only the 3D terrain_reference_pressure for the horizontal PG.
    θ₀ = dynamics.reference_state  # temporarily stored θ₀ (or nothing)
    terrain_metrics = dynamics.terrain_metrics

    if θ₀ === nothing || terrain_metrics !== nothing
        reference_state = nothing
    else
        reference_state = ExnerReferenceState(grid, thermodynamic_constants;
                                              surface_pressure,
                                              potential_temperature = θ₀,
                                              standard_pressure)
    end

    # Create contravariant velocity/momentum fields if terrain metrics are present
    if terrain_metrics === nothing
        Ω̃ = nothing
        ρΩ̃ = nothing
        terrain_reference_pressure = nothing
        terrain_reference_density = nothing
    else
        Ω̃ = ZFaceField(grid)
        ρΩ̃ = ZFaceField(grid)

        # Build 3D reference pressure and density fields via per-column discrete
        # Exner integration. The discrete integration ensures that
        #   δ(p_ref)/Δz + g ℑ(ρ_ref) ≈ 0
        # to high accuracy at every grid face, which is essential for reducing
        # the truncation error from the near-cancellation of ∂p/∂z and -gρ in
        # the vertical momentum equation. The reference pressure is also used for
        # the perturbation horizontal PG to reduce terrain-following PGF errors.
        if θ₀ === nothing
            terrain_reference_pressure = nothing
            terrain_reference_density = nothing
        else
            terrain_reference_pressure = CenterField(grid)
            terrain_reference_density = CenterField(grid)
            compute_terrain_reference_state!(terrain_reference_pressure,
                                             terrain_reference_density,
                                             grid, surface_pressure, θ₀,
                                             standard_pressure, thermodynamic_constants)
        end
    end

    return CompressibleDynamics(dynamics.time_discretization, density, pressure,
                                standard_pressure, surface_pressure, reference_state,
                                terrain_metrics, Ω̃, ρΩ̃, terrain_reference_pressure,
                                terrain_reference_density)
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
AtmosphereModels.dynamics_density(dynamics::CompressibleDynamics) = dynamics.density

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
AtmosphereModels.prognostic_dynamics_field_names(::CompressibleDynamics) = (:ρ,)
AtmosphereModels.additional_dynamics_field_names(::CompressibleDynamics) = ()

"""
$(TYPEDSIGNATURES)

Return prognostic fields specific to compressible dynamics.
Returns the density field as a prognostic variable.
"""
AtmosphereModels.dynamics_prognostic_fields(dynamics::CompressibleDynamics) = (; ρ=dynamics.density)

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

- [`SplitExplicitTimeDiscretization`](@ref): Returns `:AcousticSSPRungeKutta3` for acoustic substepping
- [`ExplicitTimeStepping`](@ref): Returns `:SSPRungeKutta3` for standard explicit time-stepping
"""
AtmosphereModels.default_timestepper(dynamics::CompressibleDynamics) =
    default_timestepper(dynamics.time_discretization)

default_timestepper(::SplitExplicitTimeDiscretization) = :AcousticSSPRungeKutta3
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
    if dynamics.density === nothing
        print(io, "├── density: not materialized\n")
        print(io, "├── pressure: not materialized\n")
        print(io, "├── time_discretization: ", summary(dynamics.time_discretization), '\n')
        print(io, "└── reference_state: ", summary(dynamics.reference_state))
    else
        print(io, "├── density: ", prettysummary(dynamics.density), '\n')
        print(io, "├── pressure: ", prettysummary(dynamics.pressure), '\n')
        print(io, "├── time_discretization: ", summary(dynamics.time_discretization), '\n')
        print(io, "└── reference_state: ", summary(dynamics.reference_state))
    end
end

#####
##### Momentum and velocity materialization
#####

function AtmosphereModels.materialize_momentum_and_velocities(::CompressibleDynamics, grid, boundary_conditions)
    ρu = XFaceField(grid, boundary_conditions=boundary_conditions.ρu)
    ρv = YFaceField(grid, boundary_conditions=boundary_conditions.ρv)
    ρw = ZFaceField(grid, boundary_conditions=boundary_conditions.ρw)
    momentum = (; ρu, ρv, ρw)

    velocity_bcs = NamedTuple(name => FieldBoundaryConditions() for name in (:u, :v, :w))
    velocity_bcs = regularize_field_boundary_conditions(velocity_bcs, grid, (:u, :v, :w))
    u = XFaceField(grid, boundary_conditions=velocity_bcs.u)
    v = YFaceField(grid, boundary_conditions=velocity_bcs.v)
    w = ZFaceField(grid, boundary_conditions=velocity_bcs.w)
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
For compressible dynamics, uses the actual density field.
"""
AtmosphereModels.Diagnostics.dynamics_density_for_potential_temperature(dynamics::CompressibleDynamics) = dynamics.density

"""
$(TYPEDSIGNATURES)

Return the standard pressure for potential temperature diagnostics.
"""
AtmosphereModels.Diagnostics.dynamics_standard_pressure(dynamics::CompressibleDynamics) = dynamics.standard_pressure
