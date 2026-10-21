using ..Thermodynamics: Thermodynamics, mixture_gas_constant

using Oceananigans: Face, UpdateStateCallsite, TendencyCallsite, instantiated_location
using Oceananigans.Advection: update_advection!
using Oceananigans.Architectures: AbstractSerialArchitecture, CPU, GPU
using Oceananigans.BoundaryConditions: fill_halo_regions!, compute_x_bcs!, compute_y_bcs!, compute_z_bcs!,
                                       update_boundary_conditions!, BoundaryCondition, Flux, Gradient,
                                       NormalFlow, Value, PBC, _fill_bottom_halo!, _fill_top_halo!
using Oceananigans.Fields: Field, flattened_unique_values
using Oceananigans.Grids: AbstractUnderlyingGrid, topology, halo_size, Flat, Periodic, RectilinearGrid, StaticVerticalDiscretization
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models: boundary_condition_args
using Oceananigans.OutputReaders: update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: TimeSteppers, Clock
using Oceananigans.TurbulenceClosures: compute_closure_fields!
using Oceananigans.Units: Time
using Oceananigans.Utils: launch!, KernelParameters
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ

# Advance any `FieldTimeSeries` embedded in boundary conditions or forcings so that
# halo fill and tendency computation see the current time slice. Mirrors
# `Oceananigans.Models.update_model_field_time_series!(::OceananigansModels, ::Clock)`
# (`Models.jl:150`); `AtmosphereModel` is not in the `OceananigansModels` Union, so
# without this extension calls fall through to the no-op `::AbstractModel` fallback.
function Oceananigans.Models.update_model_field_time_series!(model::AtmosphereModel, clock::Clock)
    time = Time(clock.time)
    possible_fts = (fields(model), model.forcing)
    time_series_tuple = extract_field_time_series(possible_fts)
    time_series_tuple = flattened_unique_values(time_series_tuple)
    for fts in time_series_tuple
        update_field_time_series!(fts, time)
    end
    return nothing
end

# `update_state!` is idempotent: it can be called any number of times and always
# leaves the model in the same consistent state (refreshed halos, diagnostics, and
# tendencies) for the prognostic fields it is given. It deliberately does *not* advance
# the operator-split microphysics update (`microphysics_model_update!`), which mutates
# prognostic fields by a full `Δt` and so must run exactly once per step — the
# time-steppers invoke it directly. (The tendency-interface microphysics still runs
# every RK stage as part of `compute_tendencies!`.)
function TimeSteppers.update_state!(model::AtmosphereModel, callbacks=[]; compute_tendencies=true)
    fix_negative_moisture!(model)  # fix negative moisture from advection
    compute_total_density!(model)  # diagnose total air density ρ = ρᵈ + Σρˣ (no-op unless compressible)
    tracer_density_to_specific!(model) # convert tracer density to specific tracer distribution

    Oceananigans.Models.update_model_field_time_series!(model, model.clock)

    fill_model_halo_regions!(prognostic_fields(model), model; async=true)
    compute_auxiliary_variables!(model)
    update_boundary_conditions!(prognostic_fields(model), model)
    update_radiation!(model.radiation, model)
    compute_forcings!(model)

    for callback in callbacks
        callback.callsite isa UpdateStateCallsite && callback(model)
    end

    # Refresh per-scheme advection state — the adaptive-implicit split time step and any
    # bounds-preserving limiter — before the tendencies that consume it.
    update_advection!(model.advection, model)

    compute_tendencies && compute_tendencies!(model, callbacks)

    tracer_specific_to_density!(model) # convert specific tracer distribution to tracer density

    return nothing
end

#####
##### Compute forcing-specific quantities (e.g., horizontal averages for subsidence)
#####

"""
$(TYPEDSIGNATURES)

Compute forcing-specific quantities needed before tendency calculation.
For example, [`SubsidenceForcing`](@ref Breeze.Forcings.SubsidenceForcing)
requires horizontal averages of the fields being advected.
"""
function compute_forcings!(model)
    for forcing in model.forcing
        compute_forcing!(forcing)
    end
    return nothing
end

# Tracer↔specific conversion uses the TOTAL air density (mass fractions qˣ = ρˣ/ρ), keeping the
# thermodynamics in mass fractions. For anelastic, `total_density === dynamics_density`.
tracer_density_to_specific!(model) = tracer_density_to_specific!(model.tracers, total_density(model.dynamics))
tracer_specific_to_density!(model) = tracer_specific_to_density!(model.tracers, total_density(model.dynamics))

# Diagnose the total air density ρ = ρᵈ + Σρˣ. No-op unless the dynamics carries a distinct
# total-density field (CompressibleDynamics overrides this); anelastic aliases dynamics_density.
compute_total_density!(model) = nothing

# Breeze's advection container is keyed by prognostic name rather than Oceananigans'
# `(momentum, tracers...)`, so pair each scheme with the specific field it reconstructs.
function Oceananigans.Advection.update_advection!(advection::NamedTuple, model::AtmosphereModel)
    fields = reconstructed_fields(model, advection)
    return update_each_advection!(values(advection), values(fields), model)
end

@inline update_each_advection!(::Tuple{}, ::Tuple{}, model) = nothing

@inline function update_each_advection!(schemes::Tuple, fields::Tuple, model)
    Oceananigans.Advection.update_advection!(first(schemes), model, first(fields))
    return update_each_advection!(Base.tail(schemes), Base.tail(fields), model)
end

# The specific field each scheme reconstructs, keyed like `advection`. `momentum` limits
# nothing; the rest name a scalar Breeze advects as a mass fraction.
@inline function reconstructed_fields(model, advection::NamedTuple{names}) where names
    θ = NamedTuple{(thermodynamic_density_name(model.formulation),)}((specific_thermodynamic_field(model.formulation),))
    q = NamedTuple{(moisture_prognostic_name(model.microphysics),)}((specific_prognostic_moisture(model),))
    micro = NamedTuple{prognostic_field_names(model.microphysics)}(
        map(name -> model.microphysical_fields[specific_field_name(name)],
            prognostic_field_names(model.microphysics)))
    everything = merge((; momentum = nothing), θ, q, micro, model.tracers)
    return NamedTuple{names}(everything)
end

function tracer_density_to_specific!(tracers, density)
    # TODO: do all tracers a single kernel
    ρ = parent(density)
    for ρc in tracers
        parent(ρc) .= ifelse.(ρ .== 0, 0, parent(ρc) ./ ρ)
    end
    return nothing
end

function tracer_specific_to_density!(tracers, density)
    # TODO: do all tracers a single kernel
    for c in tracers
        parent(c) .*= parent(density)
    end
    return nothing
end

#####
##### Halo filling
#####

# Whether filling the halo of a boundary with boundary condition `bc` reads the clock or the model
# fields, `boundary_condition_args(model)`. Flux conditions fill by copying the interior whatever the
# flux; periodic, impenetrable, and `Number`-valued `Value` and `Gradient` conditions ignore the
# arguments. Everything else (functions, arrays, field time series, open and distributed
# boundaries) is assumed to need them.
@inline halo_fill_needs_model_args(bc) = true
@inline halo_fill_needs_model_args(::Union{Nothing, Missing}) = false
@inline halo_fill_needs_model_args(::BoundaryCondition{<:Flux}) = false

const ArgumentFreeBoundaryClassification =
    Union{Oceananigans.BoundaryConditions.Periodic, Gradient, Value{Nothing}, NormalFlow{Nothing}}

@inline halo_fill_needs_model_args(::BoundaryCondition{<:ArgumentFreeBoundaryClassification, <:Union{Nothing, Number}}) = false

@inline halo_fill_needs_model_args(bcs::FieldBoundaryConditions) =
    halo_fill_needs_model_args(bcs.west)   | halo_fill_needs_model_args(bcs.east)  |
    halo_fill_needs_model_args(bcs.south)  | halo_fill_needs_model_args(bcs.north) |
    halo_fill_needs_model_args(bcs.bottom) | halo_fill_needs_model_args(bcs.top)

@inline halo_fill_needs_model_args(field::Field) = halo_fill_needs_model_args(field.boundary_conditions)

"""
$(TYPEDSIGNATURES)

Fill the halo regions of `fields`, passing `boundary_condition_args(model)` only to the fields
whose boundary conditions read them (see `halo_fill_needs_model_args`). The result is the same
as `fill_halo_regions!(fields, boundary_condition_args(model)...; kwargs...)`, but cheaper:
converting the model-field tuple for every halo kernel dominates the cost of filling a field
with trivial boundary conditions.
"""
fill_model_halo_regions!(fields::NamedTuple, model; kwargs...) = fill_model_halo_regions!(values(fields), model; kwargs...)
fill_model_halo_regions!(::Tuple{}, model; kwargs...) = nothing

function fill_model_halo_regions!(fields::Tuple, model; kwargs...)
    fill_model_halo_regions!(first(fields), model; kwargs...)
    fill_model_halo_regions!(Base.tail(fields), model; kwargs...)
    return nothing
end

function fill_model_halo_regions!(field, model; kwargs...)
    if halo_fill_needs_model_args(field)
        fill_halo_regions!(field, boundary_condition_args(model)...; kwargs...)
    else
        fill_halo_regions!(field; kwargs...)
    end
    return nothing
end

#####
##### Diagnostics over the horizontal halos
#####

# On a serial rectilinear grid whose horizontal directions are all periodic (or flat), the
# prognostic halo fill in `update_state!` makes every x/y halo of a prognostic field a periodic
# image of the interior. A diagnostic computed pointwise from the prognostics over the
# halo-extended horizontal range (`i ∈ 1-Hx:Nx+Hx`, `j ∈ 1-Hy:Ny+Hy`) is then bitwise the periodic
# image that its x/y halo fill would copy, so only its bottom and top halos still need filling.
# The vertical range stays `1:Nz`: z halos are filled by boundary conditions, not computed.
#
# The column (anelastic) density is required because `u = ρu / ℑx(ρ)` reads `ρ` at `i - 1`,
# which lies outside the halo at `i = 1 - Hx` unless `ρ` is independent of `i` and `j`.
#
# TODO: distributed grids could do the same in their periodic, unpartitioned directions.
#
# Dispatch on `AbstractUnderlyingGrid`, whose leading parameters (FT, TX, TY, TZ, CZ, Arch) are
# stable across Oceananigans versions, rather than on `RectilinearGrid`'s own parameter list,
# which grows when grids store more metrics.
const HorizontallyPeriodicGrid = AbstractUnderlyingGrid{<:Any, <:Union{Periodic, Flat}, <:Union{Periodic, Flat}, <:Any,
                                                        <:StaticVerticalDiscretization, <:Union{CPU, GPU}}

const ColumnDensity = Field{Nothing, Nothing}

computes_horizontal_halos(grid, density) = false
computes_horizontal_halos(grid::HorizontallyPeriodicGrid, ::ColumnDensity) = grid isa RectilinearGrid

# Horizontal launch ranges covering the interior and the halos (`Hx = 0` for `Flat`).
function horizontal_halo_ranges(grid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    return 1-Hx:Nx+Hx, 1-Hy:Ny+Hy
end

# A field whose x/y halos are periodic images (or absent) and whose bottom and top halos fill
# without model arguments can have its z halos filled alone, once its horizontal halos hold the
# periodic images.
@inline horizontally_periodic_bc(bc) = false
@inline horizontally_periodic_bc(::Union{Nothing, PBC}) = true

@inline function fills_vertical_halos_alone(field::Field)
    bcs = field.boundary_conditions
    return field.indices isa Tuple{Colon, Colon, Colon} &&
           horizontally_periodic_bc(bcs.west)  && horizontally_periodic_bc(bcs.east) &&
           horizontally_periodic_bc(bcs.south) && horizontally_periodic_bc(bcs.north) &&
           !halo_fill_needs_model_args(bcs.bottom) && !halo_fill_needs_model_args(bcs.top)
end

@inline fills_vertical_halos_alone(field) = false

"""
$(TYPEDSIGNATURES)

Fill the halos of `fields` whose x/y halos already hold their periodic images, as left by a
kernel launched over the horizontal halos (see `computes_horizontal_halos`). Fields that admit it
(`fills_vertical_halos_alone`) have only their bottom and top halos filled, over the full
horizontal extent and in a single kernel launch for all of them; the others are filled whole.
"""
function fill_vertical_halo_regions!(fields::Tuple, model)
    vertical = filter(fills_vertical_halos_alone, fields)
    others = filter(!fills_vertical_halos_alone, fields)

    if !isempty(vertical)
        grid = model.grid
        ii, jj = horizontal_halo_ranges(grid)
        data = map(f -> f.data, vertical)
        bottom_bcs = map(f -> f.boundary_conditions.bottom, vertical)
        top_bcs = map(f -> f.boundary_conditions.top, vertical)
        locations = map(instantiated_location, vertical)
        launch!(grid.architecture, grid, KernelParameters(ii, jj),
                _fill_bottom_and_top_halos!, data, bottom_bcs, top_bcs, locations, grid)
    end

    fill_model_halo_regions!(others, model)

    return nothing
end

@kernel function _fill_bottom_and_top_halos!(data, bottom_bcs, top_bcs, locations, grid)
    i, j = @index(Global, NTuple)
    fill_bottom_and_top_halos!(i, j, grid, data, bottom_bcs, top_bcs, locations)
end

@inline fill_bottom_and_top_halos!(i, j, grid, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing

@inline function fill_bottom_and_top_halos!(i, j, grid, data, bottom_bcs, top_bcs, locations)
    c = first(data)
    loc = first(locations)
    _fill_bottom_halo!(i, j, grid, c, first(bottom_bcs), loc)
    _fill_top_halo!(i, j, grid, c, first(top_bcs), loc)
    fill_bottom_and_top_halos!(i, j, grid, Base.tail(data), Base.tail(bottom_bcs), Base.tail(top_bcs), Base.tail(locations))
    return nothing
end

#####
##### Velocity and momentum computation
#####

"""
$(TYPEDSIGNATURES)

Compute velocities from momentum: `u = ρu / ρ` for each velocity component.

With `fill_momentum_halos = false` the density and momentum halos are assumed to be
filled already, as they are within `update_state!` on a serial architecture.
"""
function compute_velocities!(model::AtmosphereModel; fill_momentum_halos=true)
    grid = model.grid
    arch = grid.architecture

    # Ensure halos are filled before velocity computation
    # (prognostic field halo fill in update_state! is async).
    # Thread clock and model fields so time-dependent boundary conditions
    # (e.g. continuous-callable or FieldTimeSeries Open BCs) on density/momentum
    # dispatch correctly; without them `getbc` falls through to a signature that
    # cannot evaluate the time argument. Note: velocities are still stale here
    # (recomputed just below), so momentum BCs with velocity `field_dependencies`
    # would see last-stage values.
    density = dynamics_density(model.dynamics)

    if fill_momentum_halos
        fill_model_halo_regions!(density, model)
        fill_model_halo_regions!(model.momentum, model)
    end

    # Per-dim launch size from `Base.length(::Face, ::AbstractTopology, N)`:
    # N+1 for Bounded (covers the boundary face), N for Periodic (halo refilled by
    # the trailing `fill_halo_regions!(model.velocities)`), N for Flat.
    # Where possible the launch also covers the horizontal halos, which then need no fill.
    Nx, Ny, Nz = size(grid)
    TX, TY, TZ = topology(grid)
    horizontal_halos = computes_horizontal_halos(grid, density)

    if horizontal_halos
        ii, jj = horizontal_halo_ranges(grid)
    else
        ii, jj = 1:length(Face(), TX(), Nx), 1:length(Face(), TY(), Ny)
    end

    launch!(arch, grid, KernelParameters(ii, jj, 1:length(Face(), TZ(), Nz)),
            _compute_velocities!,
            model.velocities.u, model.velocities.v, model.velocities.w,
            model.momentum.ρu,   model.momentum.ρv,   model.momentum.ρw,
            grid, model.dynamics)

    foreach(mask_immersed_field!, model.velocities)

    if horizontal_halos
        fill_vertical_halo_regions!(values(model.velocities), model)
    else
        fill_halo_regions!(model.velocities)
    end

    return nothing
end

function compute_momentum_tendencies!(model::AtmosphereModel, model_fields)
    grid = model.grid
    arch = grid.architecture
    Gρu = model.timestepper.Gⁿ.ρu
    Gρv = model.timestepper.Gⁿ.ρv
    Gρw = model.timestepper.Gⁿ.ρw

    momentum_args = (
        dynamics_density(model.dynamics),
        model.advection.momentum,
        model.velocities,
        model.closure,
        model.closure_fields,
        advecting_momentum(model),
        model.coriolis,
        model.clock,
        model_fields)

    dynamics = slow_dynamics(model.timestepper, model.dynamics)
    u_args = tuple(momentum_args..., model.forcing.ρu, dynamics)
    v_args = tuple(momentum_args..., model.forcing.ρv, dynamics)

    # Extra arguments for vertical velocity are required to compute buoyancy
    w_args = tuple(momentum_args..., model.forcing.ρw,
                   dynamics,
                   model.formulation,
                   model.temperature,
                   specific_prognostic_moisture(model),
                   model.microphysics,
                   model.microphysical_fields,
                   model.thermodynamic_constants)

    launch!(arch, grid, :xyz, compute_x_momentum_tendency!, Gρu, grid, u_args)
    launch!(arch, grid, :xyz, compute_y_momentum_tendency!, Gρv, grid, v_args)
    compute_vertical_momentum_tendency!(model, Gρw, w_args)

    return nothing
end

# Compute the vertical-momentum tendency `Gρw`. Extendable so that dynamics can omit vertical-velocity
# stepping — the anelastic single-column mode overrides this to hold `Gρw ≡ 0` (`w ≡ 0`), since the
# anelastic mass constraint with rigid boundaries forces no resolved vertical velocity in a column.
function compute_vertical_momentum_tendency!(model::AtmosphereModel, Gρw, w_args)
    grid = model.grid
    launch!(grid.architecture, grid, :xyz, compute_z_momentum_tendency!, Gρw, grid, w_args)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute auxiliary model variables:

- velocities from momentum and density (eg ``u = ρu / ρ``)

- thermodynamic variables from the prognostic thermodynamic state,
    * temperature ``T``, possibly involving saturation adjustment
    * specific thermodynamic variable (``s = ρs / ρ`` or ``θ = ρθ / ρ``)
    * moisture mass fraction ``qᵗ = ρqᵗ / ρ``
"""
function compute_auxiliary_variables!(model)
    # Compute velocities from momentum (skip for kinematic dynamics with prescribed velocities).
    # `update_state!` has just filled the momentum halos and the dynamics density is either
    # prognostic (filled with it) or a static reference profile, so the refill is only needed
    # to complete the asynchronous halo exchange of a distributed architecture.
    compute_velocities!(model; fill_momentum_halos = !(model.architecture isa AbstractSerialArchitecture))

    # Dispatch on thermodynamic formulation type
    compute_auxiliary_thermodynamic_variables!(model)

    # Dispatch on dynamics type (computes pressure for compressible dynamics)
    compute_auxiliary_dynamics_variables!(model)

    # Compute diffusivities. Oceananigans has no `compute_closure_fields!` for a bare array of
    # field-less closures (a per-column ensemble of e.g. `VerticalScalarDiffusivity`, for which
    # `closure_fields === nothing`); there is nothing to compute or fill, so skip it. Field-carrying
    # closure arrays (e.g. CATKE) have non-`nothing` closure fields and are handled as usual.
    if !(model.closure isa AbstractArray && isnothing(model.closure_fields))
        compute_closure_fields!(model.closure_fields, model.closure, model)
        fill_halo_regions!(model.closure_fields; only_local_halos=true)
    end

    # TODO: should we mask the auxiliary variables? They can also be masked in the kernel

    return nothing
end

function compute_auxiliary_thermodynamic_variables!(model::AtmosphereModel)
    grid = model.grid
    arch = grid.architecture

    # The kernel is pointwise in the horizontal for the formulations, and for microphysics schemes
    # that declare `horizontally_pointwise_auxiliaries`, so it can cover the horizontal halos.
    horizontal_halos = computes_horizontal_halos(grid, dynamics_density(model.dynamics)) &&
                       horizontally_pointwise_auxiliaries(model.microphysics)

    if horizontal_halos
        ii, jj = horizontal_halo_ranges(grid)
        workspec = KernelParameters(ii, jj, 1:size(grid, 3))
    else
        workspec = :xyz
    end

    launch!(arch, grid, workspec,
            _compute_auxiliary_thermodynamic_variables!,
            model.temperature,
            specific_prognostic_moisture(model),
            model.formulation,
            model.dynamics,
            grid,
            model.thermodynamic_constants,
            model.microphysics,
            model.microphysical_fields,
            model.moisture_density)

    μ = microphysical_halo_fields(model.microphysics, model.microphysical_fields)

    if horizontal_halos
        # The formulations' `fill_halo_regions!` fills exactly their specific thermodynamic field.
        θ = specific_thermodynamic_field(model.formulation)
        fill_vertical_halo_regions!((model.temperature, θ, values(μ)...), model)
    else
        fill_halo_regions!(model.temperature)
        fill_model_halo_regions!(μ, model)
        fill_halo_regions!(model.formulation)
    end

    return nothing
end

@kernel function _compute_velocities!(u, v, w, ρu, ρv, ρw, grid, dynamics)
    i, j, k = @index(Global, NTuple)
    ρ = dynamics_density(dynamics)
    @inbounds u[i, j, k] = ρu[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, ρ)
    @inbounds v[i, j, k] = ρv[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, ρ)
    @inbounds w[i, j, k] = ρw[i, j, k] / ℑzᵃᵃᶠ(i, j, k, grid, ρ)
end

@kernel function _compute_auxiliary_thermodynamic_variables!(temperature,
                                                             specific_prognostic_moisture,
                                                             formulation,
                                                             dynamics,
                                                             grid,
                                                             constants,
                                                             microphysics,
                                                             microphysical_fields,
                                                             moisture_density)
    i, j, k = @index(Global, NTuple)

    compute_auxiliary_thermodynamic_variables!(formulation, dynamics, i, j, k, grid)

    ρ_field = total_density(dynamics)  # total ρ → qᵛᵉ is a mass fraction (thermo unchanged)
    @inbounds begin
        ρ = ρ_field[i, j, k]
        ρqᵛᵉ = moisture_density[i, j, k]
        # qᵛᵉ: vapor specific humidity (non-equilibrium) or equilibrium moisture (saturation adjustment)
        qᵛᵉ = ρqᵛᵉ / ρ
        specific_prognostic_moisture[i, j, k] = qᵛᵉ
    end

    # Compute moisture fractions first (needed by diagnose_thermodynamic_state)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)

    𝒰₀ = diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)

    # Adjust the thermodynamic state if using a microphysics scheme
    # that invokes saturation adjustment
    μ = extract_microphysical_prognostics(i, j, k, microphysics, microphysical_fields)
    𝒰₁ = maybe_adjust_thermodynamic_state(𝒰₀, microphysics, qᵛᵉ, constants, μ, ρ)

    update_microphysical_fields!(microphysical_fields, i, j, k, grid,
                                 microphysics, ρ, 𝒰₁, constants)

    T = Thermodynamics.temperature(𝒰₁, constants)
    @inbounds temperature[i, j, k] = T
end

function compute_tendencies!(model::AtmosphereModel, callbacks=[])
    grid = model.grid
    arch = grid.architecture

    model_fields = fields(model)

    # The sedimentation velocities the scalar kernels below read are established by
    # `update_microphysical_auxiliaries!` in `compute_auxiliary_variables!`, so they are
    # already current for this stage — and, unlike a tendency-time refresh, they stay
    # current after an `update_state!` that skips tendency computation.

    #####
    ##### Momentum tendencies (skip for kinematic dynamics)
    #####

    compute_momentum_tendencies!(model, model_fields)

    # Arguments common to energy density, moisture density, and tracer density tendencies,
    # given the advecting velocities (contravariant for terrain-following grids):
    tendency_args(advecting_velocities) = (
        model.dynamics,
        model.formulation,
        model.thermodynamic_constants,
        specific_prognostic_moisture(model),
        advecting_velocities,
        model.microphysics,
        model.microphysical_fields,
        model.closure,
        model.closure_fields,
        model.clock,
        model_fields)

    advecting_velocities = transport_velocities(model)
    common_args = tendency_args(advecting_velocities)

    #####
    ##### Thermodynamic density tendency (dispatches on thermodynamic formulation type)
    #####

    # The thermodynamic variable may advect with its own velocities, but its condensate
    # sedimentation term pairs with the moisture and tracer tendencies below, so it reads theirs.
    compute_thermodynamic_tendency!(model, tendency_args(thermodynamic_transport_velocities(model)),
                                    advecting_velocities.w)

    #####
    ##### Moisture density tendency
    #####

    moist_name = moisture_prognostic_name(model.microphysics)
    ρq_args = (
        specific_prognostic_moisture(model),
        Val(2),
        Val(moist_name),
        model.forcing[moist_name],
        model.advection[moist_name],
        common_args...)

    Gρqᵛᵉ = getproperty(model.timestepper.Gⁿ, moist_name)
    launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρqᵛᵉ, grid, ρq_args)

    #####
    ##### Tracer density tendencies
    #####

    # Pass specific (per-mass) fields for scalar advection: div_ρUc computes ∇·(ρ₀uq),
    # so passing density-weighted ρ₀q would double-count ρ₀.
    prognostic_microphysical_fields = NamedTuple(name => model.microphysical_fields[specific_field_name(name)]
                                                 for name in prognostic_field_names(model.microphysics))

    scalars = merge(prognostic_microphysical_fields, model.tracers)
    for (i, name) in enumerate(keys(scalars))
        ρc = scalars[name]

        scalar_args = (
            ρc,
            Val(i + 2),
            Val(name),
            model.forcing[name],
            model.advection[name],
            common_args...)

        Gρc = getproperty(model.timestepper.Gⁿ, name)
        launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρc, grid, scalar_args)
    end

    #####
    ##### Fused microphysics tendencies (no-op by default; schemes opt in)
    #####

    compute_microphysical_tendencies!(model)

    #####
    ##### Dynamics-specific tendencies (e.g., density for compressible dynamics)
    #####

    compute_dynamics_tendency!(model)

    for callback in callbacks
        callback.callsite isa TendencyCallsite && callback(model)
    end

    return nothing
end

# See dynamics_kernel_functions.jl
@kernel function compute_scalar_tendency!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = scalar_tendency(i, j, k, grid, args...)
end

# Note: compute_static_energy_tendency! and compute_potential_temperature_tendency!
# are now defined in their respective formulation submodules

@kernel function compute_x_momentum_tendency!(Gρu, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρu[i, j, k] = x_momentum_tendency(i, j, k, grid, args...)
end

@kernel function compute_y_momentum_tendency!(Gρv, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρv[i, j, k] = y_momentum_tendency(i, j, k, grid, args...)
end

@kernel function compute_z_momentum_tendency!(Gρw, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρw[i, j, k] = z_momentum_tendency(i, j, k, grid, args...)
end

"""
$(TYPEDSIGNATURES)

Apply boundary conditions by adding flux divergences to the right-hand-side.
"""
function TimeSteppers.compute_flux_bc_tendencies!(model::AtmosphereModel)

    arch  = model.architecture

    # Compute boundary flux contributions
    prognostic_model_fields = prognostic_fields(model)
    args = (arch, boundary_condition_args(model)...)
    field_indices = 1:length(prognostic_model_fields)
    Gⁿ = model.timestepper.Gⁿ

    # TODO: should we call tracer_density_to_specific!(model) here?
    foreach(q -> compute_x_bcs!(Gⁿ[q], prognostic_model_fields[q], args...), field_indices)
    foreach(q -> compute_y_bcs!(Gⁿ[q], prognostic_model_fields[q], args...), field_indices)
    foreach(q -> compute_z_bcs!(Gⁿ[q], prognostic_model_fields[q], args...), field_indices)

    return nothing
end

#####
##### Closure tendencies
#####

"""
$(TYPEDSIGNATURES)

Add a turbulence closure's own tendencies — the local sources of a prognostic the closure carries,
such as the shear and buoyancy production of turbulent kinetic energy — to the model's tendencies
`Gⁿ`. The time steppers call this at the start of every stage, after `compute_flux_bc_tendencies!`.
A no-op for closures that carry no prognostic.
"""
compute_closure_tendencies!(model) =
    compute_closure_tendencies!(model.timestepper.Gⁿ, model.closure_fields, model.closure, model)

compute_closure_tendencies!(Gⁿ, closure_fields, closure, model) = nothing

compute_closure_tendencies!(Gⁿ, ::Tuple{}, ::Tuple{}, model) = nothing

function compute_closure_tendencies!(Gⁿ, closure_fields::Tuple{F, Vararg},
                                     closures::Tuple{C, Vararg}, model) where {F, C}
    compute_closure_tendencies!(Gⁿ, first(closure_fields), first(closures), model)
    compute_closure_tendencies!(Gⁿ, Base.tail(closure_fields), Base.tail(closures), model)
    return nothing
end

compute_closure_tendencies!(Gⁿ, closure_fields::Tuple, closures::Tuple, model) =
    throw(ArgumentError("The numbers of closures and closure fields must match."))
