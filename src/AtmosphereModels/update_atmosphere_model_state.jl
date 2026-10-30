using ..Thermodynamics: Thermodynamics, mixture_gas_constant

using Oceananigans: Face, UpdateStateCallsite, TendencyCallsite, instantiated_location
using Oceananigans.Advection: update_advection!
using Oceananigans.BoundaryConditions: fill_halo_regions!, compute_x_bcs!, compute_y_bcs!, compute_z_bcs!,
                                       update_boundary_conditions!, compute_z_bottom_bc!, compute_z_top_bc!,
                                       NotFluxBC
using Oceananigans.Fields: flattened_unique_values
using Oceananigans.Grids: topology
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
#
# `compute_closure_fields = false` skips the closure-field (e.g. eddy viscosity) computation and its
# halo fill. Only callers that overwrite the closure fields with a later `update_state!` before any
# consumer reads them may pass it (the pre-Kessler pass in the acoustic RK3 step).
function TimeSteppers.update_state!(model::AtmosphereModel, callbacks=[];
                                    compute_tendencies = true,
                                    compute_closure_fields = true)
    fix_negative_moisture!(model)  # fix negative moisture from advection
    # F6: a model whose auxiliary pass diagnoses ρ does both there (`diagnoses_total_density_in_auxiliary_pass`)
    if !diagnoses_total_density_in_auxiliary_pass(model)
        compute_total_density!(model)  # diagnose total air density ρ = ρᵈ + Σρˣ (no-op unless compressible)
        tracer_density_to_specific!(model) # convert tracer density to specific tracer distribution
    end

    Oceananigans.Models.update_model_field_time_series!(model, model.clock)
    end_kernel_argument_epoch!()

    fill_halo_regions!(prognostic_fields(model), kernel_boundary_condition_args(model)..., async=true)
    compute_auxiliary_variables!(model; compute_closure_fields)
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

"""
Whether `compute_auxiliary_variables!` diagnoses the total density (and then converts the tracers to specific form)
in its thermodynamics pass, so `update_state!` skips `compute_total_density!` and `tracer_density_to_specific!`
(F6, handoff 95077e10). False unless a dynamics says otherwise.
"""
diagnoses_total_density_in_auxiliary_pass(model) = false

# Breeze's advection container is keyed by prognostic name rather than Oceananigans'
# `(momentum, tracers...)`, so pair each scheme with the specific field it reconstructs.
function Oceananigans.Advection.update_advection!(advection::NamedTuple, model::AtmosphereModel)
    fields = reconstructed_fields(model, advection)
    update_each_advection!(values(advection), values(fields), model)
    return update_reconstruction_caches!(values(advection), values(fields), model.grid)
end

# Refresh, all in one pass, the reconstruction caches whose per-scheme refresh `update_advection!` deferred
# (Breeze.Advection's batched edges pass); nothing by default.
update_reconstruction_caches!(schemes, fields, grid) = nothing

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

# Linear-index kernels over the parent arrays (halos included): the same arithmetic as the
# broadcasts they replace, without the cartesian-index broadcast path.
@kernel function _tracer_density_to_specific!(c, ρ)
    n = @index(Global, Linear)
    @inbounds c[n] = ifelse(ρ[n] == 0, 0, c[n] / ρ[n])
end

@kernel function _tracer_specific_to_density!(c, ρ)
    n = @index(Global, Linear)
    @inbounds c[n] *= ρ[n]
end

function launch_parent_kernel!(kernel, c, ρ)
    a = parent(c)
    kernel(get_backend(a), 256)(a, parent(ρ); ndrange = length(a))
    return nothing
end

# The linear-index kernels are valid only when the density's parent array has the tracer's
# parent shape (e.g. CompressibleDynamics, whose density is a full 3D CenterField). Anelastic
# dynamics has a reference density with parent size (1, 1, Nz+2Hz) that must be broadcast;
# indexing it linearly reads out of bounds (NaN tracers, task 13).
same_parent_shape(c, ρ) = size(parent(c)) == size(parent(ρ))

function tracer_density_to_specific!(tracers, density)
    # TODO: do all tracers a single kernel
    for ρc in tracers
        if same_parent_shape(ρc, density)
            launch_parent_kernel!(_tracer_density_to_specific!, ρc, density)
        else
            ρ = parent(density)
            parent(ρc) .= ifelse.(ρ .== 0, 0, parent(ρc) ./ ρ)
        end
    end
    return nothing
end

function tracer_specific_to_density!(tracers, density)
    # TODO: do all tracers a single kernel
    for c in tracers
        if same_parent_shape(c, density)
            launch_parent_kernel!(_tracer_specific_to_density!, c, density)
        else
            parent(c) .*= parent(density)
        end
    end
    return nothing
end

#####
##### Velocity and momentum computation
#####

"""
$(TYPEDSIGNATURES)

Compute velocities from momentum: `u = ρu / ρ` for each velocity component.
"""
function compute_velocities!(model::AtmosphereModel, auxiliaries = nothing)
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
    fill_halo_regions!(density, kernel_boundary_condition_args(model)...)
    fill_halo_regions!(model.momentum, kernel_boundary_condition_args(model)...)

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
            grid, model.dynamics, auxiliaries)

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
function compute_auxiliary_variables!(model; compute_closure_fields=true)
    # Velocities, thermodynamic and dynamics auxiliaries (dynamics may fuse these into one launch)
    compute_auxiliary_state_variables!(model)

    # Compute diffusivities. Oceananigans has no `compute_closure_fields!` for a bare array of
    # field-less closures (a per-column ensemble of e.g. `VerticalScalarDiffusivity`, for which
    # `closure_fields === nothing`); there is nothing to compute or fill, so skip it. Field-carrying
    # closure arrays (e.g. CATKE) have non-`nothing` closure fields and are handled as usual.
    if compute_closure_fields && !(model.closure isa AbstractArray && isnothing(model.closure_fields))
        compute_closure_fields!(model.closure_fields, model.closure, model)
        fill_halo_regions!(model.closure_fields; only_local_halos=true)
    end

    # TODO: should we mask the auxiliary variables? They can also be masked in the kernel

    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute velocities, auxiliary thermodynamic variables and auxiliary dynamics variables
(in that order). Dynamics may extend this to fuse the three passes into fewer kernel launches.
"""
function compute_auxiliary_state_variables!(model)
    # Compute velocities from momentum (skip for kinematic dynamics with prescribed velocities)
    compute_velocities!(model)

    # Dispatch on thermodynamic formulation type
    compute_auxiliary_thermodynamic_variables!(model)

    # Dispatch on dynamics type (computes pressure for compressible dynamics)
    compute_auxiliary_dynamics_variables!(model)

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

    fill_thermodynamic_auxiliaries!(model, formulation_halo_fields(model.formulation))

    return nothing
end

# E19/P4a call site: T and the formulation's diagnostic field in one tupled fill, then the
# microphysical fields (the only ones taking the boundary-condition arguments).
function fill_thermodynamic_auxiliaries!(model, formulation_fields)
    if FILLS_TOGETHER[]
        fill_halo_regions!((model.temperature, formulation_fields...))
        fill_halo_regions!(model.microphysical_fields, kernel_boundary_condition_args(model)...)
    else
        fill_thermodynamic_auxiliaries!(model, nothing)
    end
    return nothing
end

function fill_thermodynamic_auxiliaries!(model, ::Nothing)
    fill_halo_regions!(model.temperature)
    fill_halo_regions!(model.microphysical_fields, kernel_boundary_condition_args(model)...)
    fill_halo_regions!(model.formulation)
    return nothing
end

@kernel function _compute_velocities!(u, v, w, ρu, ρv, ρw, grid, dynamics, auxiliaries)
    i, j, k = @index(Global, NTuple)
    ρ = dynamics_density(dynamics)
    @inbounds u[i, j, k] = ρu[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, ρ)
    @inbounds v[i, j, k] = ρv[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, ρ)
    @inbounds w[i, j, k] = ρw[i, j, k] / ℑzᵃᵃᶠ(i, j, k, grid, ρ)
    compute_velocity_pass_auxiliaries!(i, j, k, auxiliaries, grid, dynamics, ρu, ρv, ρw)
end

# N10 (handoff 08bcc1cf): auxiliary dynamics variables the velocity launch also stores from its own inputs
# (`nothing`: none; the terrain contravariant pair, see CompressibleEquations).
@inline compute_velocity_pass_auxiliaries!(i, j, k, ::Nothing, grid, dynamics, ρu, ρv, ρw) = nothing

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
    T = auxiliary_thermodynamics_at!(i, j, k, grid, specific_prognostic_moisture, formulation, dynamics,
                                     constants, microphysics, microphysical_fields, moisture_density)
    @inbounds temperature[i, j, k] = T
end

# Pointwise body of `_compute_auxiliary_thermodynamic_variables!`: writes θ (or s), qᵛᵉ and the
# microphysical diagnostics at (i, j, k) and returns the diagnosed temperature. Shared with fused
# auxiliary kernels (see CompressibleEquations).
@inline function auxiliary_thermodynamics_at!(i, j, k, grid, specific_prognostic_moisture, formulation, dynamics,
                                              constants, microphysics, microphysical_fields, moisture_density)

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

    return Thermodynamics.temperature(𝒰₁, constants)
end

function compute_tendencies!(model::AtmosphereModel, callbacks=[])
    grid = model.grid
    arch = grid.architecture

    model_fields = kernel_fields(model)

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

    #####
    ##### Tracer density tendencies
    #####

    # Pass specific (per-mass) fields for scalar advection: div_ρUc computes ∇·(ρ₀uq),
    # so passing density-weighted ρ₀q would double-count ρ₀.
    prognostic_microphysical_fields = NamedTuple(name => model.microphysical_fields[specific_field_name(name)]
                                                 for name in prognostic_field_names(model.microphysics))

    scalars = merge(prognostic_microphysical_fields, model.tracers)
    names = keys(scalars)
    scalar_argss = ntuple(length(names)) do i
        name = names[i]
        (scalars[name], Val(i + 2), Val(name), model.forcing[name], model.advection[name], common_args...)
    end
    Gρcs = map(name -> getproperty(model.timestepper.Gⁿ, name), names)

    # GPU only: on x86 CPUs the contract-flagged WENO arithmetic fuses differently inside the batched kernel (job 5996:
    # CPU arms differ, GPU arms bitwise), as with the uncached bounded flux (see Breeze.Advection.EDGE_CACHE_ON_CPU).
    if BATCHED_SCALAR_TENDENCIES[] && arch isa Oceananigans.Architectures.GPU
        # T22 scalars: one launch; every tendency is the expression of `compute_scalar_tendency!`, unchanged.
        launch!(arch, grid, :xyz, compute_scalar_tendencies!, (Gρqᵛᵉ, Gρcs...), grid, (ρq_args, scalar_argss...))
    else
        launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρqᵛᵉ, grid, ρq_args)
        for n in eachindex(scalar_argss)
            launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρcs[n], grid, scalar_argss[n])
        end
    end

    #####
    ##### Fused microphysics tendencies (no-op by default; schemes opt in)
    #####

    compute_microphysical_tendencies!(model)

    #####
    ##### Dynamics-specific tendencies (e.g., density for compressible dynamics)
    #####

    compute_dynamics_tendency!(model, callbacks)

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

# Several scalar tendencies per thread: all evaluated before any store, so their common loads may be shared.
@kernel function compute_scalar_tendencies!(Gs, grid, argss)
    i, j, k = @index(Global, NTuple)
    tendencies = map(args -> scalar_tendency(i, j, k, grid, args...), argss)
    store_scalar_tendencies!(Gs, tendencies, i, j, k)
end

@inline store_scalar_tendencies!(::Tuple{}, ::Tuple{}, i, j, k) = nothing

@inline function store_scalar_tendencies!(Gs, tendencies, i, j, k)
    @inbounds first(Gs)[i, j, k] = first(tendencies)
    return store_scalar_tendencies!(Base.tail(Gs), Base.tail(tendencies), i, j, k)
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
    args = (arch, kernel_boundary_condition_args(model)...)
    field_indices = 1:length(prognostic_model_fields)
    Gⁿ = model.timestepper.Gⁿ

    # TODO: should we call tracer_density_to_specific!(model) here?
    foreach(q -> compute_x_bcs!(Gⁿ[q], prognostic_model_fields[q], args...), field_indices)
    foreach(q -> compute_y_bcs!(Gⁿ[q], prognostic_model_fields[q], args...), field_indices)
    if BATCHED_Z_BCS[]
        # M2 (handoff e9b5747b): every prognostic's z flux-BC contribution in one launch. Each G still
        # receives x, y, then z, with Oceananigans' per-cell expressions and arguments; fields whose
        # bottom and top are both non-flux conditions are skipped by type, as `compute_z_bcs!` does.
        entries = z_flux_bc_entries(values(Gⁿ), values(prognostic_model_fields))
        isempty(entries) || launch!(arch, model.grid, :xy, _compute_batched_z_bcs!, entries, model.grid,
                                    Tuple(kernel_boundary_condition_args(model)))
    else
        foreach(q -> compute_z_bcs!(Gⁿ[q], prognostic_model_fields[q], args...), field_indices)
    end

    return nothing
end

@inline z_flux_bc_entries(::Tuple, ::Tuple{}) = ()

@inline function z_flux_bc_entries(Gⁿ::Tuple, prognostic::Tuple)
    rest = z_flux_bc_entries(Base.tail(Gⁿ), Base.tail(prognostic))
    G = first(Gⁿ)
    c = first(prognostic)
    bcs = c.boundary_conditions
    return z_flux_bc_entry(G, c, bcs.bottom, bcs.top, rest)
end

@inline z_flux_bc_entry(G, c, bottom, top, rest) = ((G, instantiated_location(G), c, bottom, top), rest...)
@inline z_flux_bc_entry(G, c, ::NotFluxBC, ::NotFluxBC, rest) = rest
@inline z_flux_bc_entry(::Nothing, c, bottom, top, rest) = rest
@inline z_flux_bc_entry(::Nothing, c, ::NotFluxBC, ::NotFluxBC, rest) = rest

# Oceananigans' `_compute_z_bcs!` body for each entry at the same (i, j): bottom, then top.
@kernel function _compute_batched_z_bcs!(entries, grid, args)
    i, j = @index(Global, NTuple)
    compute_batched_z_bcs!(entries, i, j, grid, args)
end

@inline compute_batched_z_bcs!(::Tuple{}, i, j, grid, args) = nothing

@inline function compute_batched_z_bcs!(entries::Tuple, i, j, grid, args)
    G, loc, c, bottom, top = first(entries)
    compute_z_bottom_bc!(G, loc, bottom, i, j, grid, c, args...)
    compute_z_top_bc!(G, loc, top, i, j, grid, c, args...)
    return compute_batched_z_bcs!(Base.tail(entries), i, j, grid, args)
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
