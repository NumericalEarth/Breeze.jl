####
#### SurfaceLayerDiffusivity
####
#### A shallow, vertically implicit diffusivity that supplies the neutral-similarity
#### momentum and scalar flux not carried by time-filtered resolved covariance.
####

using Oceananigans.TurbulenceClosures: buoyancy_tracers
using Oceananigans: fields, prognostic_state, restore_prognostic_state!
using Oceananigans.BoundaryConditions: getbc
using Oceananigans.Fields: Field, set!
using Oceananigans.Grids: Bounded, RectilinearGrid, topology
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶠ
using Oceananigans.Utils: KernelParameters, time_difference_seconds

using ..AtmosphereModels: dynamics_thermodynamic_fields
using ..PotentialTemperatureFormulations: LiquidIcePotentialTemperatureFormulation

"""
$(TYPEDEF)

A shallow vertical eddy diffusivity that complements resolved near-wall transport.

At each supported interior vertical face, exponentially filtered local covariances estimate
resolved vertical fluxes. Covariances use a centered online recurrence; raw filtered products are
retained for diagnostics but do not drive the closure. The momentum viscosity is

```math
ν_{SL} = W(z) κ u_⋆ z [1 - τ^r_∥ / u_⋆²]_+,
```

and each scalar diffusivity independently replaces its signed flux deficit. The default support
is the first interior vertical face. `support=2` also activates the second interior face with
weight `1/2`. Surface boundary fluxes are diagnosed but are not modified.

`minimum_scalar_fluxes` is a named tuple keyed by transported prognostic scalar name. Each value
has the kinematic flux units of that scalar and explicitly defines its near-zero guard. Scalars
omitted from the tuple are inactive.

The filters advance once per accepted time step, after `UpdateState` callbacks have refreshed
time-dependent wall operands. Consequently the wall fluxes and sampled resolved state share the
completed-step clock time, while the resulting coefficient is first consumed by the tendencies
for the next step (a one-completed-step coefficient lag).

```jldoctest surfacelayerdiffusivity
using Breeze

closure = SurfaceLayerDiffusivity(filter_timescale=300,
                                  minimum_scalar_fluxes=(ρθ=1e-8, ρqᵗ=1e-12))
summary(closure)

# output
"SurfaceLayerDiffusivity{VerticallyImplicitTimeDiscretization}"
```
"""
struct SurfaceLayerDiffusivity{TD, FT, G} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 1}
    filter_timescale :: FT
    von_karman_constant :: FT
    turbulent_prandtl_number :: FT
    minimum_friction_velocity :: FT
    minimum_scalar_fluxes :: G
    maximum_viscosity :: FT
    maximum_diffusivity :: FT
    support :: Int
end

function SurfaceLayerDiffusivity(time_discretization::TD = VerticallyImplicitTimeDiscretization(),
                                 FT = Oceananigans.defaults.FloatType;
                                 filter_timescale = 300,
                                 von_karman_constant = 0.4,
                                 turbulent_prandtl_number = 1,
                                 minimum_friction_velocity = 1e-4,
                                 minimum_scalar_fluxes = NamedTuple(),
                                 maximum_viscosity = Inf,
                                 maximum_diffusivity = Inf,
                                 support = 1) where TD
    isfinite(filter_timescale) && filter_timescale > 0 ||
        throw(ArgumentError("filter_timescale must be finite and positive"))
    isfinite(von_karman_constant) && von_karman_constant > 0 ||
        throw(ArgumentError("von_karman_constant must be finite and positive"))
    isfinite(turbulent_prandtl_number) && turbulent_prandtl_number > 0 ||
        throw(ArgumentError("turbulent_prandtl_number must be finite and positive"))
    isfinite(minimum_friction_velocity) && minimum_friction_velocity ≥ 0 ||
        throw(ArgumentError("minimum_friction_velocity must be finite and nonnegative"))
    maximum_viscosity ≥ 0 && !isnan(maximum_viscosity) ||
        throw(ArgumentError("maximum_viscosity must be nonnegative and not NaN"))
    maximum_diffusivity ≥ 0 && !isnan(maximum_diffusivity) ||
        throw(ArgumentError("maximum_diffusivity must be nonnegative and not NaN"))
    support in (1, 2) || throw(ArgumentError("support must be 1 or 2"))
    all(value -> isfinite(value) && value ≥ 0, values(minimum_scalar_fluxes)) ||
        throw(ArgumentError("minimum_scalar_fluxes must be finite and nonnegative"))

    guards = map(value -> convert(FT, value), minimum_scalar_fluxes)
    return SurfaceLayerDiffusivity{TD, FT, typeof(guards)}(
        convert(FT, filter_timescale),
        convert(FT, von_karman_constant),
        convert(FT, turbulent_prandtl_number),
        convert(FT, minimum_friction_velocity),
        guards,
        convert(FT, maximum_viscosity),
        convert(FT, maximum_diffusivity),
        support)
end

SurfaceLayerDiffusivity(FT::DataType; kw...) =
    SurfaceLayerDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

function Utils.with_tracers(tracer_names, closure::SurfaceLayerDiffusivity{TD, FT}) where {TD, FT}
    guards = NamedTuple(name => convert(FT, get(closure.minimum_scalar_fluxes, name, Inf))
                        for name in tracer_names)
    return SurfaceLayerDiffusivity{TD, FT, typeof(guards)}(
        closure.filter_timescale,
        closure.von_karman_constant,
        closure.turbulent_prandtl_number,
        closure.minimum_friction_velocity,
        guards,
        closure.maximum_viscosity,
        closure.maximum_diffusivity,
        closure.support)
end

Base.summary(::SurfaceLayerDiffusivity{TD}) where TD =
    "SurfaceLayerDiffusivity{$(nameof(TD))}"

function Base.show(io::IO, closure::SurfaceLayerDiffusivity)
    print(io, summary(closure), '\n',
          "├── filter_timescale: ", prettysummary(closure.filter_timescale), '\n',
          "├── support: ", closure.support, '\n',
          "├── von_karman_constant: ", prettysummary(closure.von_karman_constant), '\n',
          "├── turbulent_prandtl_number: ", prettysummary(closure.turbulent_prandtl_number), '\n',
          "├── minimum_friction_velocity: ", prettysummary(closure.minimum_friction_velocity), '\n',
          "├── minimum_scalar_fluxes: ", prettysummary(closure.minimum_scalar_fluxes), '\n',
          "├── maximum_viscosity: ", prettysummary(closure.maximum_viscosity), '\n',
          "└── maximum_diffusivity: ", prettysummary(closure.maximum_diffusivity))
end

@inline support_weight(face, support) =
    ifelse(face == 2, 1, ifelse((face == 3) & (support == 2), 0.5, 0))

@inline exponential_filter_weight(elapsed_time, filter_timescale) =
    -expm1(-elapsed_time / filter_timescale)

@inline function exponential_mean_and_covariance(mean_x, mean_y, covariance, x, y, α)
    δx = x - mean_x
    δy = y - mean_y
    new_mean_x = mean_x + α * δx
    new_mean_y = mean_y + α * δy
    new_covariance = (1 - α) * (covariance + α * δx * δy)
    return (; mean_x=new_mean_x, mean_y=new_mean_y, covariance=new_covariance)
end

@inline function momentum_surface_layer_properties(resolved_u_flux, resolved_v_flux,
                                                   surface_u_flux, surface_v_flux,
                                                   z, weight, closure)
    stress_u = -surface_u_flux
    stress_v = -surface_v_flux
    stress_magnitude = sqrt(stress_u^2 + stress_v^2)
    minimum_stress = closure.minimum_friction_velocity^2
    valid = isfinite(stress_magnitude) & (stress_magnitude > minimum_stress) & (weight > 0)
    safe_stress = ifelse(valid, stress_magnitude, one(stress_magnitude))
    resolved_stress_u = -resolved_u_flux
    resolved_stress_v = -resolved_v_flux
    parallel_resolved_stress = (resolved_stress_u * stress_u + resolved_stress_v * stress_v) /
                               safe_stress
    transverse_resolved_stress = (resolved_stress_v * stress_u - resolved_stress_u * stress_v) /
                                 safe_stress
    deficit = max(0, 1 - parallel_resolved_stress / safe_stress)
    friction_velocity = sqrt(stress_magnitude)
    raw_viscosity = weight * closure.von_karman_constant * friction_velocity * z * deficit
    cap_active = valid & isfinite(closure.maximum_viscosity) &
                 (raw_viscosity > closure.maximum_viscosity)
    viscosity_value = ifelse(valid, min(raw_viscosity, closure.maximum_viscosity),
                             oftype(raw_viscosity, 0))
    return (; viscosity=viscosity_value, friction_velocity, deficit,
            parallel_resolved_stress, transverse_resolved_stress, valid, cap_active)
end

@inline function scalar_surface_layer_properties(resolved_flux, surface_flux, friction_velocity,
                                                 z, weight, flux_guard, closure)
    valid = isfinite(surface_flux) & (abs(surface_flux) > flux_guard) &
            (friction_velocity > closure.minimum_friction_velocity) & (weight > 0)
    safe_surface_flux = ifelse(valid, surface_flux, one(surface_flux))
    deficit = max(0, 1 - resolved_flux / safe_surface_flux)
    raw_diffusivity = weight * closure.von_karman_constant * friction_velocity * z /
                      closure.turbulent_prandtl_number * deficit
    cap_active = valid & isfinite(closure.maximum_diffusivity) &
                 (raw_diffusivity > closure.maximum_diffusivity)
    diffusivity_value = ifelse(valid, min(raw_diffusivity, closure.maximum_diffusivity),
                               oftype(raw_diffusivity, 0))
    return (; diffusivity=diffusivity_value, deficit, valid, cap_active)
end

struct SurfaceLayerDiffusivityFields{K, TK, F, TF, SF, B, TB, R1, R2}
    Kᵘ :: K
    tupled_tracer_diffusivities :: TK
    u_mean :: Tuple{F, F}
    v_mean :: Tuple{F, F}
    w_mean :: Tuple{F, F}
    uw_product_mean :: Tuple{F, F}
    vw_product_mean :: Tuple{F, F}
    resolved_u_flux :: Tuple{F, F}
    resolved_v_flux :: Tuple{F, F}
    scalar_mean :: TF
    scalar_w_product_mean :: TF
    resolved_scalar_flux :: TF
    surface_u_flux :: F
    surface_v_flux :: F
    surface_scalar_flux :: SF
    momentum_deficit :: Tuple{F, F}
    transverse_stress :: Tuple{F, F}
    momentum_active :: Tuple{F, F}
    viscosity_cap_active :: Tuple{F, F}
    scalar_deficit :: TF
    scalar_active :: TF
    diffusivity_cap_active :: TF
    momentum_boundary_conditions :: B
    scalar_boundary_conditions :: TB
    previous_update_time :: R1
    previous_update_iteration :: R2
end

# Only diffusivities are read by the momentum, tracer, and implicit-solver kernels.
# The full fields object remains on the host: its wall boundary conditions and Ref clocks
# are needed to sample/update filters and to restore their state from checkpoints.
struct SurfaceLayerDiffusivityDeviceFields{K, TK}
    Kᵘ :: K
    tupled_tracer_diffusivities :: TK
end

Adapt.adapt_structure(to, fields::SurfaceLayerDiffusivityFields) =
    SurfaceLayerDiffusivityDeviceFields(
        adapt(to, fields.Kᵘ),
        adapt(to, fields.tupled_tracer_diffusivities))

two_surface_fields(grid) =
    (Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(grid))

tracer_surface_fields(grid, tracer_names) =
    NamedTuple(name => two_surface_fields(grid) for name in tracer_names)

function validate_surface_layer_configuration(grid, closure, model)
    grid isa RectilinearGrid ||
        throw(ArgumentError("SurfaceLayerDiffusivity currently supports RectilinearGrid only"))
    topology(grid)[3] === Bounded ||
        throw(ArgumentError("SurfaceLayerDiffusivity requires a bottom-bounded vertical grid"))
    grid.Nz ≥ closure.support + 1 ||
        throw(ArgumentError("SurfaceLayerDiffusivity support=$(closure.support) requires " *
                            "at least $(closure.support + 1) vertical cells"))
    model.formulation isa LiquidIcePotentialTemperatureFormulation ||
        throw(ArgumentError("SurfaceLayerDiffusivity currently supports " *
                            "LiquidIcePotentialTemperatureFormulation only"))
    return nothing
end

function Oceananigans.TurbulenceClosures.build_closure_fields(grid, clock, tracer_names, bcs,
                                                               closure::SurfaceLayerDiffusivity)
    grid isa RectilinearGrid ||
        throw(ArgumentError("SurfaceLayerDiffusivity currently supports RectilinearGrid only"))
    topology(grid)[3] === Bounded ||
        throw(ArgumentError("SurfaceLayerDiffusivity requires a bottom-bounded vertical grid"))
    grid.Nz ≥ closure.support + 1 ||
        throw(ArgumentError("SurfaceLayerDiffusivity support=$(closure.support) requires " *
                            "at least $(closure.support + 1) vertical cells"))

    Kᵘ = ZFaceField(grid)
    set!(Kᵘ, 0)
    tupled_tracer_diffusivities = NamedTuple(name => ZFaceField(grid) for name in tracer_names)
    foreach(field -> set!(field, 0), values(tupled_tracer_diffusivities))

    u_mean = two_surface_fields(grid)
    v_mean = two_surface_fields(grid)
    w_mean = two_surface_fields(grid)
    uw_product_mean = two_surface_fields(grid)
    vw_product_mean = two_surface_fields(grid)
    resolved_u_flux = two_surface_fields(grid)
    resolved_v_flux = two_surface_fields(grid)
    scalar_mean = tracer_surface_fields(grid, tracer_names)
    scalar_w_product_mean = tracer_surface_fields(grid, tracer_names)
    resolved_scalar_flux = tracer_surface_fields(grid, tracer_names)
    surface_u_flux = Field{Center, Center, Nothing}(grid)
    surface_v_flux = Field{Center, Center, Nothing}(grid)
    surface_scalar_flux = NamedTuple(name => Field{Center, Center, Nothing}(grid)
                                     for name in tracer_names)
    momentum_deficit = two_surface_fields(grid)
    transverse_stress = two_surface_fields(grid)
    momentum_active = two_surface_fields(grid)
    viscosity_cap_active = two_surface_fields(grid)
    scalar_deficit = tracer_surface_fields(grid, tracer_names)
    scalar_active = tracer_surface_fields(grid, tracer_names)
    diffusivity_cap_active = tracer_surface_fields(grid, tracer_names)
    momentum_boundary_conditions = (; u=bcs.ρu.bottom, v=bcs.ρv.bottom)
    scalar_boundary_conditions = NamedTuple(name => bcs[name].bottom for name in tracer_names)

    return SurfaceLayerDiffusivityFields(
        Kᵘ, tupled_tracer_diffusivities,
        u_mean, v_mean, w_mean, uw_product_mean, vw_product_mean,
        resolved_u_flux, resolved_v_flux,
        scalar_mean, scalar_w_product_mean, resolved_scalar_flux,
        surface_u_flux, surface_v_flux, surface_scalar_flux,
        momentum_deficit, transverse_stress, momentum_active, viscosity_cap_active,
        scalar_deficit, scalar_active, diffusivity_cap_active,
        momentum_boundary_conditions, scalar_boundary_conditions,
        Ref(clock.time), Ref(clock.iteration))
end

@inline Oceananigans.TurbulenceClosures.viscosity_location(::SurfaceLayerDiffusivity) =
    (Center(), Center(), Face())
@inline Oceananigans.TurbulenceClosures.diffusivity_location(::SurfaceLayerDiffusivity) =
    (Center(), Center(), Face())
@inline Oceananigans.TurbulenceClosures.viscosity(::SurfaceLayerDiffusivity, closure_fields) =
    closure_fields.Kᵘ
@inline Oceananigans.TurbulenceClosures.diffusivity(::SurfaceLayerDiffusivity,
                                                    closure_fields, ::Val{id}) where id =
    closure_fields.tupled_tracer_diffusivities[id]

BoundaryConditions.fill_halo_regions!(closure_fields::SurfaceLayerDiffusivityFields,
                                      args...; kw...) =
    fill_halo_regions!((closure_fields.Kᵘ, values(closure_fields.tupled_tracer_diffusivities)...),
                       args...; kw...)

@inline bottom_dynamic_u_flux(i, j, k, grid, boundary_condition, clock,
                              model_fields, dynamics_fields) =
    getbc(boundary_condition, i, j, grid, clock, model_fields, dynamics_fields)

@inline bottom_dynamic_v_flux(i, j, k, grid, boundary_condition, clock,
                              model_fields, dynamics_fields) =
    getbc(boundary_condition, i, j, grid, clock, model_fields, dynamics_fields)

@kernel function _initialize_surface_momentum_flux!(surface_u_flux, surface_v_flux, grid,
                                                    momentum_boundary_conditions, clock,
                                                    model_fields, dynamics_fields)
    i, j = @index(Global, NTuple)
    density = dynamics_fields.ρ
    surface_density = ℑzᵃᵃᶠ(i, j, 1, grid, density)
    dynamic_u_flux = ℑxᶜᵃᵃ(i, j, 1, grid, bottom_dynamic_u_flux,
                            momentum_boundary_conditions.u, clock, model_fields, dynamics_fields)
    dynamic_v_flux = ℑyᵃᶜᵃ(i, j, 1, grid, bottom_dynamic_v_flux,
                            momentum_boundary_conditions.v, clock, model_fields, dynamics_fields)
    @inbounds surface_u_flux[i, j, 1] = dynamic_u_flux / surface_density
    @inbounds surface_v_flux[i, j, 1] = dynamic_v_flux / surface_density
end

@kernel function _update_surface_momentum_flux!(surface_u_flux, surface_v_flux, grid,
                                                momentum_boundary_conditions, clock,
                                                model_fields, dynamics_fields, α)
    i, j = @index(Global, NTuple)
    density = dynamics_fields.ρ
    surface_density = ℑzᵃᵃᶠ(i, j, 1, grid, density)
    dynamic_u_flux = ℑxᶜᵃᵃ(i, j, 1, grid, bottom_dynamic_u_flux,
                            momentum_boundary_conditions.u, clock, model_fields, dynamics_fields)
    dynamic_v_flux = ℑyᵃᶜᵃ(i, j, 1, grid, bottom_dynamic_v_flux,
                            momentum_boundary_conditions.v, clock, model_fields, dynamics_fields)
    @inbounds surface_u_flux[i, j, 1] = (1 - α) * surface_u_flux[i, j, 1] +
                                        α * dynamic_u_flux / surface_density
    @inbounds surface_v_flux[i, j, 1] = (1 - α) * surface_v_flux[i, j, 1] +
                                        α * dynamic_v_flux / surface_density
end

@kernel function _initialize_surface_scalar_flux!(surface_flux, grid, boundary_condition,
                                                  clock, model_fields, dynamics_fields)
    i, j = @index(Global, NTuple)
    density = dynamics_fields.ρ
    surface_density = ℑzᵃᵃᶠ(i, j, 1, grid, density)
    dynamic_flux = getbc(boundary_condition, i, j, grid, clock, model_fields, dynamics_fields)
    @inbounds surface_flux[i, j, 1] = dynamic_flux / surface_density
end

@kernel function _update_surface_scalar_flux!(surface_flux, grid, boundary_condition,
                                              clock, model_fields, dynamics_fields, α)
    i, j = @index(Global, NTuple)
    density = dynamics_fields.ρ
    surface_density = ℑzᵃᵃᶠ(i, j, 1, grid, density)
    dynamic_flux = getbc(boundary_condition, i, j, grid, clock, model_fields, dynamics_fields)
    @inbounds surface_flux[i, j, 1] = (1 - α) * surface_flux[i, j, 1] +
                                      α * dynamic_flux / surface_density
end

@kernel function _initialize_momentum_moments!(u_mean, v_mean, w_mean,
                                               uw_product_mean, vw_product_mean,
                                               resolved_u_flux, resolved_v_flux,
                                               grid, velocities, face)
    i, j = @index(Global, NTuple)
    u = ℑxzᶜᵃᶠ(i, j, face, grid, velocities.u)
    v = ℑyzᵃᶜᶠ(i, j, face, grid, velocities.v)
    w = @inbounds velocities.w[i, j, face]
    @inbounds begin
        u_mean[i, j, 1] = u
        v_mean[i, j, 1] = v
        w_mean[i, j, 1] = w
        uw_product_mean[i, j, 1] = u * w
        vw_product_mean[i, j, 1] = v * w
        resolved_u_flux[i, j, 1] = 0
        resolved_v_flux[i, j, 1] = 0
    end
end

@kernel function _update_momentum_moments!(u_mean, v_mean, w_mean,
                                           uw_product_mean, vw_product_mean,
                                           resolved_u_flux, resolved_v_flux,
                                           grid, velocities, face, α)
    i, j = @index(Global, NTuple)
    u = ℑxzᶜᵃᶠ(i, j, face, grid, velocities.u)
    v = ℑyzᵃᶜᶠ(i, j, face, grid, velocities.v)
    w = @inbounds velocities.w[i, j, face]
    @inbounds begin
        old_u_mean = u_mean[i, j, 1]
        old_v_mean = v_mean[i, j, 1]
        old_w_mean = w_mean[i, j, 1]
        u_statistics = exponential_mean_and_covariance(
            old_u_mean, old_w_mean, resolved_u_flux[i, j, 1], u, w, α)
        v_statistics = exponential_mean_and_covariance(
            old_v_mean, old_w_mean, resolved_v_flux[i, j, 1], v, w, α)
        uw = (1 - α) * uw_product_mean[i, j, 1] + α * u * w
        vw = (1 - α) * vw_product_mean[i, j, 1] + α * v * w
        u_mean[i, j, 1] = u_statistics.mean_x
        v_mean[i, j, 1] = v_statistics.mean_x
        w_mean[i, j, 1] = u_statistics.mean_y
        uw_product_mean[i, j, 1] = uw
        vw_product_mean[i, j, 1] = vw
        resolved_u_flux[i, j, 1] = u_statistics.covariance
        resolved_v_flux[i, j, 1] = v_statistics.covariance
    end
end

@kernel function _initialize_scalar_moments!(scalar_mean, scalar_w_product_mean,
                                             resolved_scalar_flux, grid, scalar, w, face)
    i, j = @index(Global, NTuple)
    c = ℑzᵃᵃᶠ(i, j, face, grid, scalar)
    w_value = @inbounds w[i, j, face]
    @inbounds begin
        scalar_mean[i, j, 1] = c
        scalar_w_product_mean[i, j, 1] = c * w_value
        resolved_scalar_flux[i, j, 1] = 0
    end
end

@kernel function _update_scalar_moments!(scalar_mean, scalar_w_product_mean,
                                         resolved_scalar_flux, w_mean,
                                         grid, scalar, w, face, α)
    i, j = @index(Global, NTuple)
    c = ℑzᵃᵃᶠ(i, j, face, grid, scalar)
    w_value = @inbounds w[i, j, face]
    @inbounds begin
        statistics = exponential_mean_and_covariance(
            scalar_mean[i, j, 1], w_mean[i, j, 1], resolved_scalar_flux[i, j, 1],
            c, w_value, α)
        cw = (1 - α) * scalar_w_product_mean[i, j, 1] + α * c * w_value
        scalar_mean[i, j, 1] = statistics.mean_x
        scalar_w_product_mean[i, j, 1] = cw
        resolved_scalar_flux[i, j, 1] = statistics.covariance
    end
end

@kernel function _compute_momentum_diffusivity!(Kᵘ, deficit, transverse_stress, active,
                                                cap_active, resolved_u_flux, resolved_v_flux,
                                                surface_u_flux, surface_v_flux,
                                                grid, closure, face, weight)
    i, j = @index(Global, NTuple)
    z = height_above_bottomᶜᶜᶠ(i, j, face, grid)
    resolved_u = @inbounds resolved_u_flux[i, j, 1]
    resolved_v = @inbounds resolved_v_flux[i, j, 1]
    surface_u = @inbounds surface_u_flux[i, j, 1]
    surface_v = @inbounds surface_v_flux[i, j, 1]
    properties = momentum_surface_layer_properties(
        resolved_u, resolved_v, surface_u, surface_v, z, weight, closure)
    @inbounds begin
        Kᵘ[i, j, face] = properties.viscosity
        deficit[i, j, 1] = properties.deficit
        transverse_stress[i, j, 1] = properties.transverse_resolved_stress
        active[i, j, 1] = properties.valid
        cap_active[i, j, 1] = properties.cap_active
    end
end

@kernel function _compute_scalar_diffusivity!(Kᶜ, deficit, active, cap_active,
                                              resolved_flux, surface_flux,
                                              surface_u_flux, surface_v_flux,
                                              grid, closure, flux_guard, face, weight)
    i, j = @index(Global, NTuple)
    z = height_above_bottomᶜᶜᶠ(i, j, face, grid)
    surface_u = @inbounds surface_u_flux[i, j, 1]
    surface_v = @inbounds surface_v_flux[i, j, 1]
    stress_magnitude = sqrt(surface_u^2 + surface_v^2)
    friction_velocity = sqrt(stress_magnitude)
    resolved = @inbounds resolved_flux[i, j, 1]
    filtered_surface_flux = @inbounds surface_flux[i, j, 1]
    properties = scalar_surface_layer_properties(
        resolved, filtered_surface_flux, friction_velocity,
        z, weight, flux_guard, closure)
    @inbounds begin
        Kᶜ[i, j, face] = properties.diffusivity
        deficit[i, j, 1] = properties.deficit
        active[i, j, 1] = properties.valid
        cap_active[i, j, 1] = properties.cap_active
    end
end

function surface_kernel_parameters(grid)
    Nx, Ny, _ = size(grid)
    return KernelParameters(1:Nx, 1:Ny)
end

function initialize_surface_layer_filters!(closure_fields, closure, model)
    validate_surface_layer_configuration(model.grid, closure, model)
    grid = model.grid
    arch = grid.architecture
    parameters = surface_kernel_parameters(grid)
    model_fields = fields(model)
    dynamics_fields = dynamics_thermodynamic_fields(model.dynamics)
    tracers = buoyancy_tracers(model)

    launch!(arch, grid, parameters, _initialize_surface_momentum_flux!,
            closure_fields.surface_u_flux, closure_fields.surface_v_flux, grid,
            closure_fields.momentum_boundary_conditions, model.clock,
            model_fields, dynamics_fields)

    for name in keys(closure_fields.tupled_tracer_diffusivities)
        launch!(arch, grid, parameters, _initialize_surface_scalar_flux!,
                closure_fields.surface_scalar_flux[name], grid,
                closure_fields.scalar_boundary_conditions[name], model.clock,
                model_fields, dynamics_fields)
    end

    for (slot, face) in enumerate((2, 3))
        launch!(arch, grid, parameters, _initialize_momentum_moments!,
                closure_fields.u_mean[slot], closure_fields.v_mean[slot],
                closure_fields.w_mean[slot], closure_fields.uw_product_mean[slot],
                closure_fields.vw_product_mean[slot], closure_fields.resolved_u_flux[slot],
                closure_fields.resolved_v_flux[slot], grid, model.velocities, face)
        for name in keys(closure_fields.tupled_tracer_diffusivities)
            launch!(arch, grid, parameters, _initialize_scalar_moments!,
                    closure_fields.scalar_mean[name][slot],
                    closure_fields.scalar_w_product_mean[name][slot],
                    closure_fields.resolved_scalar_flux[name][slot],
                    grid, tracers[name], model.velocities.w, face)
        end
    end

    closure_fields.previous_update_time[] = model.clock.time
    closure_fields.previous_update_iteration[] = model.clock.iteration
    compute_surface_layer_diffusivities!(closure_fields, closure, model)
    return nothing
end

function update_surface_layer_filters!(closure_fields, closure, model, elapsed_time)
    grid = model.grid
    arch = grid.architecture
    parameters = surface_kernel_parameters(grid)
    model_fields = fields(model)
    dynamics_fields = dynamics_thermodynamic_fields(model.dynamics)
    tracers = buoyancy_tracers(model)
    α = exponential_filter_weight(elapsed_time, closure.filter_timescale)

    launch!(arch, grid, parameters, _update_surface_momentum_flux!,
            closure_fields.surface_u_flux, closure_fields.surface_v_flux, grid,
            closure_fields.momentum_boundary_conditions, model.clock,
            model_fields, dynamics_fields, α)

    for name in keys(closure_fields.tupled_tracer_diffusivities)
        launch!(arch, grid, parameters, _update_surface_scalar_flux!,
                closure_fields.surface_scalar_flux[name], grid,
                closure_fields.scalar_boundary_conditions[name], model.clock,
                model_fields, dynamics_fields, α)
    end

    for (slot, face) in enumerate((2, 3))
        # Every tracer covariance must use the same old w mean. Update all scalars before the
        # momentum kernel advances that shared mean.
        for name in keys(closure_fields.tupled_tracer_diffusivities)
            launch!(arch, grid, parameters, _update_scalar_moments!,
                    closure_fields.scalar_mean[name][slot],
                    closure_fields.scalar_w_product_mean[name][slot],
                    closure_fields.resolved_scalar_flux[name][slot],
                    closure_fields.w_mean[slot], grid, tracers[name],
                    model.velocities.w, face, α)
        end
        launch!(arch, grid, parameters, _update_momentum_moments!,
                closure_fields.u_mean[slot], closure_fields.v_mean[slot],
                closure_fields.w_mean[slot], closure_fields.uw_product_mean[slot],
                closure_fields.vw_product_mean[slot], closure_fields.resolved_u_flux[slot],
                closure_fields.resolved_v_flux[slot], grid, model.velocities, face, α)
    end
    return nothing
end

function compute_surface_layer_diffusivities!(closure_fields, closure, model)
    grid = model.grid
    arch = grid.architecture
    parameters = surface_kernel_parameters(grid)
    FT = eltype(grid)
    for (slot, face) in enumerate((2, 3))
        weight = FT(support_weight(face, closure.support))
        launch!(arch, grid, parameters, _compute_momentum_diffusivity!,
                closure_fields.Kᵘ, closure_fields.momentum_deficit[slot],
                closure_fields.transverse_stress[slot], closure_fields.momentum_active[slot],
                closure_fields.viscosity_cap_active[slot],
                closure_fields.resolved_u_flux[slot], closure_fields.resolved_v_flux[slot],
                closure_fields.surface_u_flux, closure_fields.surface_v_flux,
                grid, closure, face, weight)
        for name in keys(closure_fields.tupled_tracer_diffusivities)
            launch!(arch, grid, parameters, _compute_scalar_diffusivity!,
                    closure_fields.tupled_tracer_diffusivities[name],
                    closure_fields.scalar_deficit[name][slot],
                    closure_fields.scalar_active[name][slot],
                    closure_fields.diffusivity_cap_active[name][slot],
                    closure_fields.resolved_scalar_flux[name][slot],
                    closure_fields.surface_scalar_flux[name],
                    closure_fields.surface_u_flux, closure_fields.surface_v_flux,
                    grid, closure, closure.minimum_scalar_fluxes[name], face, weight)
        end
    end
    return nothing
end

function Oceananigans.TurbulenceClosures.initialize_closure_fields!(
        closure_fields::SurfaceLayerDiffusivityFields,
        closure::SurfaceLayerDiffusivity, model)
    initialize_surface_layer_filters!(closure_fields, closure, model)
    return nothing
end

function Oceananigans.TurbulenceClosures.compute_closure_fields!(
        closure_fields::SurfaceLayerDiffusivityFields,
        closure::SurfaceLayerDiffusivity, model; parameters=:xyz)
    compute_surface_layer_diffusivities!(closure_fields, closure, model)
    return nothing
end

function AtmosphereModels.update_completed_step_closure_state!(
        closure_fields::SurfaceLayerDiffusivityFields,
        closure::SurfaceLayerDiffusivity, model)
    # Breeze increments `clock.iteration` only on the final Runge--Kutta stage. Thus this
    # hook samples the accepted end-of-step state once, after UpdateState callbacks have
    # refreshed time-dependent wall operands such as GABLS3 surface humidity. The filtered
    # coefficient is then used by the tendencies prepared for the next step: a documented
    # one-completed-step coefficient lag, without shortening T by the number of RK stages.
    iteration = model.clock.iteration
    iteration == closure_fields.previous_update_iteration[] && return nothing
    elapsed_time = time_difference_seconds(model.clock.time,
                                           closure_fields.previous_update_time[])
    elapsed_time ≤ 0 && return nothing
    update_surface_layer_filters!(closure_fields, closure, model, elapsed_time)
    closure_fields.previous_update_time[] = model.clock.time
    closure_fields.previous_update_iteration[] = iteration
    compute_surface_layer_diffusivities!(closure_fields, closure, model)
    # The completed-step recomputation follows the ordinary auxiliary halo fill. Keep the
    # coefficient halos synchronized so a checkpoint pickup starts from the same state.
    fill_halo_regions!(closure_fields; only_local_halos=true)
    return nothing
end

surface_layer_prognostic_fields(closure_fields) = (;
    Kᵘ=closure_fields.Kᵘ,
    tupled_tracer_diffusivities=closure_fields.tupled_tracer_diffusivities,
    u_mean=closure_fields.u_mean,
    v_mean=closure_fields.v_mean,
    w_mean=closure_fields.w_mean,
    uw_product_mean=closure_fields.uw_product_mean,
    vw_product_mean=closure_fields.vw_product_mean,
    resolved_u_flux=closure_fields.resolved_u_flux,
    resolved_v_flux=closure_fields.resolved_v_flux,
    scalar_mean=closure_fields.scalar_mean,
    scalar_w_product_mean=closure_fields.scalar_w_product_mean,
    resolved_scalar_flux=closure_fields.resolved_scalar_flux,
    surface_u_flux=closure_fields.surface_u_flux,
    surface_v_flux=closure_fields.surface_v_flux,
    surface_scalar_flux=closure_fields.surface_scalar_flux,
    momentum_deficit=closure_fields.momentum_deficit,
    transverse_stress=closure_fields.transverse_stress,
    momentum_active=closure_fields.momentum_active,
    viscosity_cap_active=closure_fields.viscosity_cap_active,
    scalar_deficit=closure_fields.scalar_deficit,
    scalar_active=closure_fields.scalar_active,
    diffusivity_cap_active=closure_fields.diffusivity_cap_active)

function Oceananigans.prognostic_state(closure_fields::SurfaceLayerDiffusivityFields)
    field_state = prognostic_state(surface_layer_prognostic_fields(closure_fields))
    return merge(field_state, (;
        previous_update_time=closure_fields.previous_update_time[],
        previous_update_iteration=closure_fields.previous_update_iteration[]))
end

function Oceananigans.restore_prognostic_state!(restored::SurfaceLayerDiffusivityFields, from)
    restored_fields = surface_layer_prognostic_fields(restored)
    names = keys(restored_fields)
    from_fields = NamedTuple{names}(getproperty(from, name) for name in names)
    restore_prognostic_state!(restored_fields, from_fields)
    restored.previous_update_time[] = from.previous_update_time
    restored.previous_update_iteration[] = from.previous_update_iteration
    return restored
end

Oceananigans.restore_prognostic_state!(::SurfaceLayerDiffusivityFields, ::Nothing) = nothing
