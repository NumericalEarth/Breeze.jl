using ..Thermodynamics: Thermodynamics, mixture_gas_constant

using Oceananigans: Face, UpdateStateCallsite, TendencyCallsite
using Oceananigans.Advection: update_advection!
using Oceananigans.BoundaryConditions: fill_halo_regions!, compute_x_bcs!, compute_y_bcs!, compute_z_bcs!,
                                       update_boundary_conditions!
using Oceananigans.Grids: Flat, topology, TwoSided, Connected
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models: boundary_condition_args
using Oceananigans.OutputReaders: update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: TimeSteppers, Clock
using Oceananigans.TurbulenceClosures: compute_closure_fields!
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveragedDynamicSmagorinsky
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
TimeSteppers.update_state!(model::AtmosphereModel, callbacks=[]; compute_tendencies=true) =
    update_atmosphere_state!(update_radiation!, model, callbacks; compute_tendencies)

# `update_state!` after a step's first stage, which has already checked the radiation schedule
update_state_between_radiation_solves!(model, callbacks=[]; compute_tendencies=true) =
    update_atmosphere_state!(update_radiation_between_solves!, model, callbacks; compute_tendencies)

function update_atmosphere_state!(update_radiation_fluxes!, model, callbacks; compute_tendencies)
    fix_negative_moisture!(model)  # fix negative moisture from advection
    compute_total_density!(model)  # diagnose total air density ρ = ρᵈ + Σρˣ (no-op unless compressible)
    tracer_density_to_specific!(model) # convert tracer density to specific tracer distribution

    Oceananigans.Models.update_model_field_time_series!(model, model.clock)

    # Synchronous: nothing completes an asynchronous fill here, so on a `Distributed` grid the halos
    # of every prognostic that is not filled again below (the tracers) would never be received.
    fill_halo_regions!(prognostic_fields(model), boundary_condition_args(model)...)
    compute_auxiliary_variables!(model)
    update_boundary_conditions!(prognostic_fields(model), model)
    update_radiation_fluxes!(model.radiation, model)
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
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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
# thermodynamics in mass fractions. For anelastic, `total_density === continuity_density`.
tracer_density_to_specific!(model) = tracer_density_to_specific!(model.tracers, total_density(model.dynamics))
tracer_specific_to_density!(model) = tracer_specific_to_density!(model.tracers, total_density(model.dynamics))

# Diagnose the total air density ρ = ρᵈ + Σρˣ. No-op unless the dynamics carries a distinct
# total-density field (CompressibleDynamics overrides this); anelastic aliases continuity_density.
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
        parent(ρc) .= ifelse.(ρ .== 0, zero(eltype(ρ)), parent(ρc) ./ ρ)
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
##### Velocity and momentum computation
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Compute velocities from momentum: `u = ρu / ρ` for each velocity component.
"""
function compute_velocities!(model::AtmosphereModel)
    grid = model.grid
    arch = grid.architecture

    # Ensure halos are filled before velocity computation (not every caller is `update_state!`).
    # Thread clock and model fields so time-dependent boundary conditions
    # (e.g. continuous-callable or FieldTimeSeries Open BCs) on density/momentum
    # dispatch correctly; without them `getbc` falls through to a signature that
    # cannot evaluate the time argument. Note: velocities are still stale here
    # (recomputed just below), so momentum BCs with velocity `field_dependencies`
    # would see last-stage values.
    density = continuity_density(model.dynamics)
    fill_halo_regions!(density, boundary_condition_args(model)...)
    fill_halo_regions!(model.momentum, boundary_condition_args(model)...)

    # Per-dim launch size from `Base.length(::Face, ::AbstractTopology, N)`:
    # N+1 for Bounded (covers the boundary face), N for Periodic (halo refilled by
    # the trailing `fill_halo_regions!(model.velocities)`), N for Flat.
    Nx, Ny, Nz = size(grid)
    TX, TY, TZ = topology(grid)
    launch!(arch, grid, KernelParameters(1:length(Face(), TX(), Nx),
                                         1:length(Face(), TY(), Ny),
                                         1:length(Face(), TZ(), Nz)),
            _compute_velocities!,
            model.velocities.u, model.velocities.v, model.velocities.w,
            model.momentum.ρu,   model.momentum.ρv,   model.momentum.ρw,
            grid, model.dynamics)

    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities)

    return nothing
end

function compute_momentum_tendencies!(model::AtmosphereModel, model_fields)
    grid = model.grid
    arch = grid.architecture
    Gρu = model.timestepper.Gⁿ.ρu
    Gρv = model.timestepper.Gⁿ.ρv
    Gρw = model.timestepper.Gⁿ.ρw

    momentum_args = (
        continuity_density(model.dynamics),
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

    u_parameters, v_parameters = horizontal_momentum_tendency_kernel_parameters(model)
    launch!(arch, grid, u_parameters, compute_x_momentum_tendency!, Gρu, grid, u_args)
    launch!(arch, grid, v_parameters, compute_y_momentum_tendency!, Gρv, grid, v_args)
    compute_vertical_momentum_tendency!(model, Gρw, w_args)

    return nothing
end

# Kernel parameters for the `Gρu` and `Gρv` launches. Dynamics whose kernels read `Gρu` and `Gρv`
# one face past the interior extend this to return `tendency_kernel_parameters`.
horizontal_momentum_tendency_kernel_parameters(model) = (:xyz, :xyz)

# Dynamics whose tendency kernels reach further than the advection scheme and closure check for
# a large enough halo here, once the dynamics is materialized.
validate_dynamics_halo(dynamics, grid, momentum_advection, closure) = nothing

# Face-located tendencies are also computed at face N + 1 where the right side of a dimension is a
# seam with a neighboring rank (`Connected`). Computing that face locally avoids exchanging it.
# On a periodic dimension face N + 1 is the image of face 1 and is filled as a halo instead,
# which needs no extra halo cell.
extended_tendency_faces(::TwoSided{<:Any, Connected}) = 1
extended_tendency_faces(topo) = 0

tendency_range(::Face, topo, N) = 1:N+extended_tendency_faces(topo)
tendency_range(ℓ, topo, N) = 1:N

# `KernelParameters` for a tendency at `(ℓx, ℓy, Center())` that also cover face N + 1 in each
# horizontal dimension whose right side is a seam between ranks.
function tendency_kernel_parameters(grid, ℓx, ℓy)
    Nx, Ny, Nz = size(grid)
    TX, TY, _ = topology(grid)
    return KernelParameters(tendency_range(ℓx, TX(), Nx),
                            tendency_range(ℓy, TY(), Ny),
                            1:Nz)
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
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Compute auxiliary model variables:

- velocities from momentum and density (eg ``u = ρu / ρ``)

- thermodynamic variables from the prognostic thermodynamic state,
    * temperature ``T``, possibly involving saturation adjustment
    * specific thermodynamic variable (``s = ρs / ρ`` or ``θ = ρθ / ρ``)
    * moisture mass fraction ``qᵗ = ρqᵗ / ρ``
"""
function compute_auxiliary_variables!(model)
    # Compute velocities from momentum (skip for kinematic dynamics with prescribed velocities)
    compute_velocities!(model)

    # Dispatch on thermodynamic formulation type
    compute_auxiliary_thermodynamic_variables!(model)

    # Dispatch on dynamics type (computes pressure for compressible dynamics)
    compute_auxiliary_dynamics_variables!(model)

    # Compute diffusivities. Oceananigans has no `compute_closure_fields!` for a bare array of
    # field-less closures (a per-column ensemble of e.g. `VerticalScalarDiffusivity`, for which
    # `closure_fields === nothing`); there is nothing to compute or fill, so skip it. Field-carrying
    # closure arrays (e.g. CATKE) have non-`nothing` closure fields and are handled as usual.
    if !(model.closure isa AbstractArray && isnothing(model.closure_fields))
        compute_extended_closure_fields!(model.closure_fields, model.closure, model)
        # The local fill imposes the boundary conditions of the closure fields and the periodic
        # images within a rank. Nothing is exchanged between ranks: the halo cells the tendencies
        # read have just been computed.
        fill_halo_regions!(model.closure_fields; only_local_halos=true)
    end

    # TODO: should we mask the auxiliary variables? They can also be masked in the kernel

    return nothing
end

#####
##### Closure fields, computed one cell into the horizontal halos
#####

# The viscous and diffusive fluxes of an interior tendency read an eddy viscosity or diffusivity one
# cell into the halo, corners included (`νᶠᶠᶜ` at `(1, 1)` averages the centers `(0:1, 0:1)`). On a
# `Distributed` grid a closure field that is computed in the interior alone is stale there, because
# its halos are filled locally and never received from a neighbouring rank. So, as Oceananigans'
# `HydrostaticFreeSurfaceModel` does with `horizontally_extended_interior_kernel_parameters`, the
# closure fields are computed over `0:Nx+1` and `0:Ny+1`, from the halos of the velocities,
# thermodynamic variables and tracers, which are filled (and exchanged between ranks) before the
# closure fields are computed. A closure's own stencil reaches one further cell, to index `-1` or
# `N+2`, which is why these closures require a halo of at least two (`required_halo_size_x`).
#
# The ranges are written out, rather than taken from `horizontally_extended_interior_kernel_parameters`,
# because on a grid with an active cells map that function returns a map of the active cells only,
# whereas `:xyz` covers every cell and column kernels need the horizontal ranges.
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the kernel parameters over which the fields of `closure` are computed on `grid`: the
interior plus one cell into the halo in every horizontal direction that is not `Flat`.
The fallback `:xyz` (the interior alone) applies to closures whose fields cannot be computed
from the halos of the model state.
"""
function closure_fields_kernel_parameters(grid, closure)
    Nx, Ny, Nz = size(grid)
    TX, TY, _ = topology(grid)
    ii = TX === Flat ? (1:Nx) : (0:Nx+1)
    jj = TY === Flat ? (1:Ny) : (0:Ny+1)
    return KernelParameters(ii, jj, 1:Nz)
end

# An array of closures (a per-column ensemble) is indexed by column, and has no halo columns.
closure_fields_kernel_parameters(grid, ::AbstractArray) = :xyz

# The Lagrangian-averaged coefficient is advanced in the interior by `step_closure_prognostics!`,
# from its own previous values, so that its halos hold nothing a viscosity could be computed from.
# TODO: on a `Distributed` grid this closure is wrong next to a rank boundary, in Oceananigans too:
# `𝒥ᴸᴹ` and `𝒥ᴹᴹ` are time-stepped state whose halos must be exchanged after each update, as must
# those of `Σ` and `Σ̄`, which Oceananigans fills with `only_local_halos=true` before filtering them.
closure_fields_kernel_parameters(grid, ::LagrangianAveragedDynamicSmagorinsky) = :xyz

compute_extended_closure_fields!(closure_fields, closure, model) =
    compute_closure_fields!(closure_fields, closure, model;
                            parameters = closure_fields_kernel_parameters(model.grid, closure))

# Each closure of a tuple over its own region
compute_extended_closure_fields!(::Tuple{}, ::Tuple{}, model) = nothing

function compute_extended_closure_fields!(closure_fields::Tuple, closures::Tuple, model)
    compute_extended_closure_fields!(first(closure_fields), first(closures), model)
    compute_extended_closure_fields!(Base.tail(closure_fields), Base.tail(closures), model)
    return nothing
end

function compute_auxiliary_thermodynamic_variables!(model::AtmosphereModel)
    grid = model.grid
    arch = grid.architecture

    launch!(arch, grid, :xyz,
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

    fill_halo_regions!(model.temperature)
    fill_halo_regions!(model.microphysical_fields, boundary_condition_args(model)...)
    fill_halo_regions!(model.formulation)

    return nothing
end

@kernel function _compute_velocities!(u, v, w, ρu, ρv, ρw, grid, dynamics)
    i, j, k = @index(Global, NTuple)
    ρ = continuity_density(dynamics)
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

    advecting_velocities = transport_velocities(model)

    #####
    ##### Thermodynamic density tendency (dispatches on thermodynamic formulation type)
    #####

    # The thermodynamic variable may advect with its own velocities, but its condensate
    # sedimentation term pairs with the moisture and tracer tendencies below, so it reads theirs.
    compute_thermodynamic_tendency!(model,
                                    scalar_tendency_arguments(model, model_fields, thermodynamic_transport_velocities(model)),
                                    advecting_velocities.w)

    #####
    ##### Moisture, microphysical and tracer density tendencies, with the microphysical sources
    #####

    # A time stepper may rebuild these after its acoustic loop instead (see
    # `scalar_tendencies_in_update_state`), in which case the build here is skipped.
    scalar_tendencies_in_update_state(model.timestepper) &&
        compute_scalar_tendencies!(model, model_fields, advecting_velocities, compute_scalar_tendency!)

    #####
    ##### Dynamics-specific tendencies (e.g., density for compressible dynamics)
    #####

    compute_dynamics_tendency!(model)

    for callback in callbacks
        callback.callsite isa TendencyCallsite && callback(model)
    end

    return nothing
end

# Whether `compute_tendencies!` builds the moisture, microphysical and tracer tendencies. A time
# stepper that rebuilds them after its acoustic substep loop returns `false`.
scalar_tendencies_in_update_state(timestepper) = true

# Arguments common to the thermodynamic, moisture, microphysical and tracer density tendencies,
# given the advecting velocities (contravariant for terrain-following grids).
scalar_tendency_arguments(model, model_fields, advecting_velocities) = (
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

# Build the moisture, microphysical and tracer tendencies into `Gⁿ` with `kernel`, then add the
# microphysical sources. The tracers hold their specific values here, as in `update_state!`.
function compute_scalar_tendencies!(model, model_fields, advecting_velocities, kernel)
    grid = model.grid
    arch = grid.architecture
    common_args = scalar_tendency_arguments(model, model_fields, advecting_velocities)

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
    launch_scalar_tendency!(kernel, arch, grid, moist_name, Gρqᵛᵉ, ρq_args)

    #####
    ##### Tracer density tendencies
    #####

    # Pass specific (per-mass) fields for scalar advection: div_ρUc computes ∇·(ρ₀uq),
    # so passing density-weighted ρ₀q would double-count ρ₀.
    scalars = specific_scalar_fields(model)
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
        launch_scalar_tendency!(kernel, arch, grid, name, Gρc, scalar_args)
    end

    #####
    ##### Fused microphysics tendencies (no-op by default; schemes opt in)
    #####

    compute_microphysical_tendencies!(model)

    return nothing
end

# See dynamics_kernel_functions.jl
@kernel function compute_scalar_tendency!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = scalar_tendency(i, j, k, grid, args...)
end

# The same tendency, added to what `Gc` already holds (sources built before an acoustic loop).
@kernel function add_scalar_tendency!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] += scalar_tendency(i, j, k, grid, args...)
end

# The specific (per-mass) fields of the microphysical prognostics and the tracers, keyed by
# their prognostic (density) names. The tracers are specific only inside `update_state!` and
# the time steppers' post-loop scalar tendency build (`tracer_density_to_specific!`).
function specific_scalar_fields(model)
    density_names = prognostic_field_names(model.microphysics)
    specific_names = map(specific_field_name, density_names)
    specific_fields = map(name -> model.microphysical_fields[name], specific_names)
    microphysical_scalars = NamedTuple{density_names}(specific_fields)
    return merge(microphysical_scalars, model.tracers)
end

# Build one scalar's tendency `Gc` with `kernel`, a kernel function writing `Gc` alone or a
# `SplitScalarTendencies` (below).
launch_scalar_tendency!(kernel, arch, grid, name, Gc, args) = launch!(arch, grid, :xyz, kernel, Gc, grid, args)

# Adds the non-advective part of each scalar's tendency to `Gⁿ` and assigns the advective part to
# `advective_tendencies[name]`, each evaluated once. A scalar with no entry there gets the
# non-advective part alone: whoever builds with the split evaluates its advective part later.
struct SplitScalarTendencies{C}
    advective_tendencies :: C
end

advective_tendency_storage(split::SplitScalarTendencies, name) = get(split.advective_tendencies, name, nothing)

launch_scalar_tendency!(split::SplitScalarTendencies, arch, grid, name, Gc, args) =
    launch_split_scalar_tendency!(advective_tendency_storage(split, name), arch, grid, Gc, args)

launch_split_scalar_tendency!(C, arch, grid, Gc, args) =
    launch!(arch, grid, :xyz, add_split_scalar_tendency!, Gc, C, grid, args)

launch_split_scalar_tendency!(::Nothing, arch, grid, Gc, args) =
    launch!(arch, grid, :xyz, add_nonadvective_scalar_tendency!, Gc, grid, args)

@kernel function add_split_scalar_tendency!(Gc, C, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        C[i, j, k] = scalar_advective_tendency(i, j, k, grid, args...)
        Gc[i, j, k] += nonadvective_scalar_tendency(i, j, k, grid, args...)
    end
end

@kernel function add_nonadvective_scalar_tendency!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] += nonadvective_scalar_tendency(i, j, k, grid, args...)
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
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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
    names = keys(prognostic_model_fields)
    foreach(q -> compute_stage_z_bcs!(Gⁿ[q], prognostic_model_fields[q],
                                      closure_applies_bottom_flux(model.closure, model, names[q]), args...),
            field_indices)

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Whether `closure` itself applies the bottom boundary flux of the prognostic field `name`, in which
case the tendencies of the Runge–Kutta stages leave that flux out (the boundary condition is left
untouched: it stays the one place the flux is specified, and the closure reads it from there).
The default is `false`; an operator-split closure that applies the surface fluxes inside its own
implicit solve, once per step in `step_closure_prognostics!`, adds a method.
"""
closure_applies_bottom_flux(closure, model, name) = false

closure_applies_bottom_flux(closures::Tuple, model, name) =
    any(closure -> closure_applies_bottom_flux(closure, model, name), closures)

# Add the top and bottom boundary fluxes of `c` to `Gc`, or only the top one where a closure
# applies the bottom one.
function compute_stage_z_bcs!(Gc, c, closure_applies_bottom, args...)
    closure_applies_bottom || return compute_z_bcs!(Gc, c, args...)
    return compute_z_bcs!(Gc, Gc.grid, c, nothing, c.boundary_conditions.top, args...)
end

compute_stage_z_bcs!(::Nothing, c, closure_applies_bottom, args...) = nothing

#####
##### Closure tendencies
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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

#####
##### Closure work around the vertically implicit solves of a stage
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Hooks a closure may extend to act on the state around the vertically implicit solves of a
Runge–Kutta stage of length `Δt` (the stage's `α Δt`), so that what it adds is formed from the
state the solves produce:

* `closure_before_stage_combination!(closure_fields, closure, model, Δt, α)` — before the explicit
  update ``u = (1 - α) u⁰ + α (u + Δt G)`` of the stage, with the step's `Δt` and the stage's `α`.
  A multirate infinitesimal step passes the stage's `Δt` (its ``dᵢ Δt``) and a
  [`StageRecombination`](@ref) in place of `α`;
* `closure_before_implicit_solve!(closure_fields, closure, model, name)` — just before the solve of
  the prognostic `name` (after its explicit stage update);
* `closure_before_thermodynamic_solve!(closure_fields, closure, model, Δt)` — after the solves of
  momentum and every tracer, before the solve of the thermodynamic variable;
* `closure_after_implicit_solves!(closure_fields, closure, model, Δt)` — after every solve.

All are no-ops by default.
"""
closure_before_stage_combination!(closure_fields, closure, model, Δt, α) = nothing
closure_after_implicit_solves!(closure_fields, closure, model, Δt) = nothing
closure_before_thermodynamic_solve!(closure_fields, closure, model, Δt) = nothing
closure_before_implicit_solve!(closure_fields, closure, model, name) = nothing

"""
    solved_after_thermodynamic_variable(closure, name)

Whether the closure asks for the vertically implicit solve of its prognostic `name` to follow the solve of the
thermodynamic variable, so that it can form, before that solve, what the solves of momentum, the tracers and the
thermodynamic variable converted. Time steppers for which [`closure_prognostics_solved_last`](@ref) holds honor it.
"""
solved_after_thermodynamic_variable(closure, name) = false

"""
    closure_prognostics_solved_last(model)

Whether a stage of the time stepper of `model` solves momentum, then the tracers, then the thermodynamic variable,
and last the closure's prognostics for which [`solved_after_thermodynamic_variable`](@ref) holds, calling
`closure_before_implicit_solve!` just before each of these. `false` unless a time stepper says so.
"""
closure_prognostics_solved_last(model) = false

"""
$(TYPEDEF)

How stage ``i`` of a multirate infinitesimal step recombines the states and the slow tendencies of
the stages before it. Without the fast tendency, the state its vertically implicit solves start
from is

```math
Vᵢ = Σⱼ₌₀^{i-1} wᵢⱼ Yⱼ + dᵢ Δt Σⱼ₌₁^{i} bᵢⱼ Sⱼ , \\qquad Y₀ = Uⁿ , \\quad Sⱼ = S(Yⱼ₋₁) ,
```

with ``wᵢⱼ = αᵢⱼ + γᵢⱼ`` for ``j ≥ 1``, ``wᵢ₀ = 1 - Σⱼ wᵢⱼ`` and ``bᵢⱼ = βᵢⱼ / dᵢ`` (``Σⱼ bᵢⱼ = 1``).
The weights of the states sum to one but need not be positive.

Fields
======

- `stage`, `stages`: ``i`` and the number of stages ``s``.
- `state_weights`: ``(wᵢ₀, …, wᵢ,ᵢ₋₁)``.
- `tendency_weights`: ``(bᵢ₁, …, bᵢᵢ)``.
- `step_start_state`: the prognostic fields of ``Uⁿ``.
- `storage`: what the closure keeps from stage to stage (see [`closure_stage_storage`](@ref)).
"""
struct StageRecombination{W, B, U, S}
    stage :: Int
    stages :: Int
    state_weights :: W
    tendency_weights :: B
    step_start_state :: U
    storage :: S
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

What a closure keeps from stage to stage of `timestepper` (see [`StageRecombination`](@ref)), or
`nothing` for a time stepper whose stages recombine nothing the closure accounts for.
"""
closure_stage_storage(timestepper) = nothing

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The fields a closure with `closure_fields` keeps from stage to stage of a time stepper with
`stages` stages that recombines them (see [`StageRecombination`](@ref)). `nothing` by default.
"""
build_closure_stage_storage(closure_fields, closure, grid, stages) = nothing

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `timestepper` holding the stage storage of the closure (see
[`build_closure_stage_storage`](@ref)). The time stepper is built before the closure fields, so
`AtmosphereModel` calls this once both exist. The default returns `timestepper` unchanged.
"""
materialize_closure_stage_storage(timestepper, closure_fields, closure, grid) = timestepper

