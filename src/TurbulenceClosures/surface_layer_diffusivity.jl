####
#### SurfaceLayerDiffusivity
####
#### A shallow, vertically implicit diffusivity that supplies the neutral-similarity
#### momentum and scalar flux not carried by time-filtered resolved covariance.
####

using Oceananigans.TurbulenceClosures: buoyancy_tracers
using Oceananigans: fields, prognostic_state, restore_prognostic_state!
using Oceananigans.BoundaryConditions: getbc
using Oceananigans.Advection: AbstractCenteredAdvectionScheme,
                                AbstractUpwindBiasedAdvectionScheme,
                                BoundsPreservingWENO, FluxFormAdvection,
                                _advective_momentum_flux_Wu, _advective_momentum_flux_Wv,
                                _advective_tracer_flux_z, _biased_interpolate_zᵃᵃᶠ,
                                LeftBias, RightBias, rescaled_reconstruction,
                                upwind_biased_product
using Oceananigans.Fields: Field, set!
using Oceananigans.Grids: Bounded, RectilinearGrid, topology
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶠ,
                              Azᶜᶜᶠ
using Oceananigans.TimeSteppers: time_discretization
using Oceananigans.Utils: KernelParameters, time_difference_seconds

using ..AtmosphereModels: dynamics_thermodynamic_fields, dynamics_density, total_density
using ..AtmosphereModels: reconstructed_fields, tracer_density_to_specific!,
                          tracer_specific_to_density!
using ..AnelasticEquations: AnelasticDynamics
using ..PotentialTemperatureFormulations: LiquidIcePotentialTemperatureFormulation

"""
$(TYPEDEF)

A shallow vertical eddy diffusivity that complements resolved near-wall transport.

At each supported interior vertical face, exponentially filtered local covariances estimate
resolved vertical fluxes. Covariances use a centered online recurrence; raw filtered products are
retained for diagnostics but do not drive the closure. The momentum viscosity is

```math
ν_{SL} = W(z) κ u_⋆ z [1 - a τ^r_∥ / u_⋆²]_+,
```

and each scalar diffusivity independently replaces its signed flux deficit. The default support
is the first interior vertical face. `support=2` also activates the second interior face with
weight `1/2`. Surface boundary fluxes are diagnosed but are not modified.

`resolved_flux_factor=a` (default `1`) scales the signed resolved contribution only when
computing the momentum and scalar deficits. For example, `a=2` assumes additional transport
equal to the resolved flux, so an aligned resolved flux carrying half the wall flux shuts off
the corresponding coefficient. This is a sensitivity parameter, not a measurement of numerical
transport: numerical flux need not be proportional to, or have the sign of, resolved covariance.
Countergradient resolved transport increases the deficit. Stored resolved covariances, projected
stresses, and prescribed surface fluxes retain their physical, unscaled values.

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
struct SurfaceLayerDiffusivity{TD, FT, G, M, A} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 1}
    filter_timescale :: FT
    resolved_flux_factor :: FT
    resolved_transport :: M
    advection :: A
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
                                 resolved_flux_factor = 1,
                                 resolved_transport = :covariance,
                                 von_karman_constant = 0.4,
                                 turbulent_prandtl_number = 1,
                                 minimum_friction_velocity = 1e-4,
                                 minimum_scalar_fluxes = NamedTuple(),
                                 maximum_viscosity = Inf,
                                 maximum_diffusivity = Inf,
                                 support = 1) where TD
    isfinite(filter_timescale) && filter_timescale > 0 ||
        throw(ArgumentError("filter_timescale must be finite and positive"))
    isfinite(resolved_flux_factor) && resolved_flux_factor ≥ 0 ||
        throw(ArgumentError("resolved_flux_factor must be finite and nonnegative"))
    resolved_transport in (:covariance, :scheme_native) ||
        throw(ArgumentError("resolved_transport must be :covariance or :scheme_native"))
    resolved_transport === :scheme_native && resolved_flux_factor != 1 &&
        throw(ArgumentError("scheme-native transport requires resolved_flux_factor=1"))
    resolved_flux_factor = convert(FT, resolved_flux_factor)
    isfinite(resolved_flux_factor) ||
        throw(ArgumentError("resolved_flux_factor must be finite and nonnegative in the closure float type"))
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
    mode = Val(resolved_transport)
    return SurfaceLayerDiffusivity{TD, FT, typeof(guards), typeof(mode), Nothing}(
        convert(FT, filter_timescale),
        resolved_flux_factor,
        mode, nothing,
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
    return SurfaceLayerDiffusivity{TD, FT, typeof(guards), typeof(closure.resolved_transport), typeof(closure.advection)}(
        closure.filter_timescale,
        closure.resolved_flux_factor,
        closure.resolved_transport, closure.advection,
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
          "├── resolved_flux_factor: ", prettysummary(closure.resolved_flux_factor), '\n',
          "├── resolved_transport: ", closure.resolved_transport, '\n',
          "├── support: ", closure.support, '\n',
          "├── von_karman_constant: ", prettysummary(closure.von_karman_constant), '\n',
          "├── turbulent_prandtl_number: ", prettysummary(closure.turbulent_prandtl_number), '\n',
          "├── minimum_friction_velocity: ", prettysummary(closure.minimum_friction_velocity), '\n',
          "├── minimum_scalar_fluxes: ", prettysummary(closure.minimum_scalar_fluxes), '\n',
          "├── maximum_viscosity: ", prettysummary(closure.maximum_viscosity), '\n',
          "└── maximum_diffusivity: ", prettysummary(closure.maximum_diffusivity))
end

Adapt.adapt_structure(to, closure::SurfaceLayerDiffusivity{TD, FT}) where {TD, FT} =
    SurfaceLayerDiffusivity{TD, FT, typeof(adapt(to, closure.minimum_scalar_fluxes)),
                            typeof(closure.resolved_transport), typeof(adapt(to, closure.advection))}(
        closure.filter_timescale, closure.resolved_flux_factor,
        closure.resolved_transport, adapt(to, closure.advection),
        closure.von_karman_constant, closure.turbulent_prandtl_number,
        closure.minimum_friction_velocity, adapt(to, closure.minimum_scalar_fluxes),
        closure.maximum_viscosity, closure.maximum_diffusivity, closure.support)

validate_native_scheme(::AbstractCenteredAdvectionScheme) = nothing
validate_native_scheme(::AbstractUpwindBiasedAdvectionScheme) = nothing
validate_native_scheme(scheme::FluxFormAdvection) = validate_native_scheme(scheme.z)
validate_native_scheme(scheme::BoundsPreservingWENO) = nothing
validate_native_scheme(scheme) =
    throw(ArgumentError("unsupported advection $(typeof(scheme)) for scheme-native SLD"))

AtmosphereModels.bind_closure_advection(closure::SurfaceLayerDiffusivity{TD, FT, G, Val{:covariance}},
                                       advection) where {TD, FT, G} = closure

function AtmosphereModels.bind_closure_advection(closure::SurfaceLayerDiffusivity{TD, FT, G, Val{:scheme_native}},
                                                advection) where {TD, FT, G}
    for scheme in values(advection)
        validate_native_scheme(scheme)
        time_discretization(scheme) isa ExplicitTimeDiscretization ||
            throw(ArgumentError("scheme-native SLD currently requires explicit advection"))
    end
    return SurfaceLayerDiffusivity{TD, FT, G, Val{:scheme_native}, typeof(advection)}(
        closure.filter_timescale, closure.resolved_flux_factor,
        closure.resolved_transport, advection,
        closure.von_karman_constant, closure.turbulent_prandtl_number,
        closure.minimum_friction_velocity, closure.minimum_scalar_fluxes,
        closure.maximum_viscosity, closure.maximum_diffusivity, closure.support)
end

function AtmosphereModels.bind_closure_advection(closures::Tuple, advection)
    any(closure -> closure isa SurfaceLayerDiffusivity, closures) &&
        throw(ArgumentError("SurfaceLayerDiffusivity in a closure tuple is not supported by its stateful filter"))
    return closures
end

AtmosphereModels.bind_closure_advection(closures::AbstractArray{<:SurfaceLayerDiffusivity},
                                       advection) =
    throw(ArgumentError("SurfaceLayerDiffusivity closure arrays are not supported by its stateful filter"))

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
    deficit = max(0, 1 - closure.resolved_flux_factor * parallel_resolved_stress / safe_stress)
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
    deficit = max(0, 1 - closure.resolved_flux_factor * resolved_flux / safe_surface_flux)
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
    scheme_u_flux :: Tuple{F, F}
    scheme_v_flux :: Tuple{F, F}
    numerical_u_correction :: Tuple{F, F}
    numerical_v_correction :: Tuple{F, F}
    scalar_mean :: TF
    scalar_w_product_mean :: TF
    resolved_scalar_flux :: TF
    scheme_scalar_flux :: TF
    numerical_scalar_correction :: TF
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
    if closure.resolved_transport isa Val{:scheme_native}
        model.dynamics isa AnelasticDynamics ||
            throw(ArgumentError("scheme-native SLD currently supports AnelasticDynamics only"))
        isnothing(closure.advection) &&
            throw(ArgumentError("scheme-native SLD requires bound model advection"))
    end
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
    scheme_u_flux = two_surface_fields(grid)
    scheme_v_flux = two_surface_fields(grid)
    numerical_u_correction = two_surface_fields(grid)
    numerical_v_correction = two_surface_fields(grid)
    scalar_mean = tracer_surface_fields(grid, tracer_names)
    scalar_w_product_mean = tracer_surface_fields(grid, tracer_names)
    resolved_scalar_flux = tracer_surface_fields(grid, tracer_names)
    scheme_scalar_flux = tracer_surface_fields(grid, tracer_names)
    numerical_scalar_correction = tracer_surface_fields(grid, tracer_names)
    for fields in (scheme_u_flux, scheme_v_flux, numerical_u_correction,
                   numerical_v_correction)
        foreach(field -> set!(field, 0), fields)
    end
    for named_fields in (scheme_scalar_flux, numerical_scalar_correction)
        for fields in values(named_fields)
            foreach(field -> set!(field, 0), fields)
        end
    end
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
        scheme_u_flux, scheme_v_flux, numerical_u_correction, numerical_v_correction,
        scalar_mean, scalar_w_product_mean, resolved_scalar_flux,
        scheme_scalar_flux, numerical_scalar_correction,
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

# These functions call the same vertical flux operators as the prognostic tendencies.
# Oceananigans includes face area in each operator; divide by area and density after
# horizontal collocation to express the transport as a kinematic flux at (Center, Center, Face).
@inline function native_u_transport(i, j, face, grid, scheme, mass_flux_w, u, density)
    area_flux = ℑxᶜᵃᵃ(i, j, face, grid, _advective_momentum_flux_Wu,
                      scheme, mass_flux_w, u)
    return area_flux / (Azᶜᶜᶠ(i, j, face, grid) * ℑzᵃᵃᶠ(i, j, face, grid, density))
end

@inline function native_v_transport(i, j, face, grid, scheme, mass_flux_w, v, density)
    area_flux = ℑyᵃᶜᵃ(i, j, face, grid, _advective_momentum_flux_Wv,
                      scheme, mass_flux_w, v)
    return area_flux / (Azᶜᶜᶠ(i, j, face, grid) * ℑzᵃᵃᶠ(i, j, face, grid, density))
end

@inline function native_scalar_transport(i, j, face, grid, scheme, w, scalar)
    # `div_ρUc` multiplies this exact operator by ρ at the face. The same ρ
    # cancels when we convert its dynamic flux to a kinematic flux.
    return _advective_tracer_flux_z(i, j, face, grid, scheme, w, scalar) /
           Azᶜᶜᶠ(i, j, face, grid)
end

@inline native_scalar_transport(i, j, face, grid, scheme::FluxFormAdvection, w, scalar) =
    native_scalar_transport(i, j, face, grid, scheme.z, w, scalar)

# Oceananigans' bounds-preserving tendency forms a face flux inside its two-face
# divergence rather than exposing a standalone face operator. Use the identical
# reconstruction, cell-wise rescaling, and upwind product at this face.
@inline function native_scalar_transport(i, j, face, grid,
                                         scheme::BoundsPreservingWENO, w, scalar)
    limiter = scheme.bounds.limiter
    left = _biased_interpolate_zᵃᵃᶠ(i, j, face, grid, scheme, LeftBias, scalar)
    right = _biased_interpolate_zᵃᵃᶠ(i, j, face, grid, scheme, RightBias, scalar)
    left = rescaled_reconstruction(left, i, j, face-1, grid, limiter, scalar)
    right = rescaled_reconstruction(right, i, j, face, grid, limiter, scalar)
    vertical_velocity = @inbounds w[i, j, face]
    return upwind_biased_product(vertical_velocity, left, right)
end

@kernel function _update_native_momentum_flux!(scheme_u_flux, scheme_v_flux,
                                                numerical_u_correction,
                                                numerical_v_correction,
                                                grid, scheme, mass_flux_w, velocities,
                                                density, face, α)
    i, j = @index(Global, NTuple)
    u_flux = native_u_transport(i, j, face, grid, scheme, mass_flux_w,
                                velocities.u, density)
    v_flux = native_v_transport(i, j, face, grid, scheme, mass_flux_w,
                                velocities.v, density)
    u = ℑxzᶜᵃᶠ(i, j, face, grid, velocities.u)
    v = ℑyzᵃᶜᶠ(i, j, face, grid, velocities.v)
    w = @inbounds velocities.w[i, j, face]
    u_correction = u_flux - u * w
    v_correction = v_flux - v * w
    @inbounds begin
        filtered_u = (1 - α) * scheme_u_flux[i, j, 1] + α * u_flux
        filtered_v = (1 - α) * scheme_v_flux[i, j, 1] + α * v_flux
        scheme_u_flux[i, j, 1] = filtered_u
        scheme_v_flux[i, j, 1] = filtered_v
        numerical_u_correction[i, j, 1] = (1 - α) * numerical_u_correction[i, j, 1] +
                                          α * u_correction
        numerical_v_correction[i, j, 1] = (1 - α) * numerical_v_correction[i, j, 1] +
                                          α * v_correction
    end
end

@kernel function _update_native_scalar_flux!(scheme_scalar_flux,
                                              numerical_scalar_correction,
                                              grid, scheme, w, scalar, face, α)
    i, j = @index(Global, NTuple)
    flux = native_scalar_transport(i, j, face, grid, scheme, w, scalar)
    centered_scalar = ℑzᵃᵃᶠ(i, j, face, grid, scalar)
    vertical_velocity = @inbounds w[i, j, face]
    correction = flux - centered_scalar * vertical_velocity
    @inbounds begin
        filtered = (1 - α) * scheme_scalar_flux[i, j, 1] + α * flux
        scheme_scalar_flux[i, j, 1] = filtered
        numerical_scalar_correction[i, j, 1] =
            (1 - α) * numerical_scalar_correction[i, j, 1] + α * correction
    end
end

@kernel function _compute_momentum_diffusivity!(Kᵘ, deficit, transverse_stress, active,
                                                cap_active, resolved_u_flux, resolved_v_flux,
                                                numerical_u_correction,
                                                numerical_v_correction,
                                                surface_u_flux, surface_v_flux,
                                                grid, closure, face, weight)
    i, j = @index(Global, NTuple)
    z = height_above_bottomᶜᶜᶠ(i, j, face, grid)
    native = closure.resolved_transport isa Val{:scheme_native}
    @inbounds begin
        covariance_u = resolved_u_flux[i, j, 1]
        covariance_v = resolved_v_flux[i, j, 1]
        native_u = covariance_u + numerical_u_correction[i, j, 1]
        native_v = covariance_v + numerical_v_correction[i, j, 1]
    end
    resolved_u = ifelse(native, native_u, covariance_u)
    resolved_v = ifelse(native, native_v, covariance_v)
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
                                              resolved_flux, numerical_correction, surface_flux,
                                              surface_u_flux, surface_v_flux,
                                              grid, closure, flux_guard, face, weight)
    i, j = @index(Global, NTuple)
    z = height_above_bottomᶜᶜᶠ(i, j, face, grid)
    surface_u = @inbounds surface_u_flux[i, j, 1]
    surface_v = @inbounds surface_v_flux[i, j, 1]
    stress_magnitude = sqrt(surface_u^2 + surface_v^2)
    friction_velocity = sqrt(stress_magnitude)
    native = closure.resolved_transport isa Val{:scheme_native}
    @inbounds begin
        covariance = resolved_flux[i, j, 1]
        native_flux = covariance + numerical_correction[i, j, 1]
    end
    resolved = ifelse(native, native_flux, covariance)
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
    tracers = surface_layer_scalar_fields(model, closure)

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

    update_native_transport!(closure_fields, closure, model, one(eltype(grid)))

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
    tracers = surface_layer_scalar_fields(model, closure)
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
    update_native_transport!(closure_fields, closure, model, α)
    return nothing
end

update_native_transport!(closure_fields,
                         closure::SurfaceLayerDiffusivity{TD, FT, G, Val{:covariance}},
                         model, α) where {TD, FT, G} = nothing

function update_native_transport!(closure_fields,
                                  closure::SurfaceLayerDiffusivity{TD, FT, G, Val{:scheme_native}},
                                  model, α) where {TD, FT, G}
    grid = model.grid
    arch = grid.architecture
    parameters = surface_kernel_parameters(grid)
    tracers = surface_layer_scalar_fields(model, closure)
    for (slot, face) in enumerate((2, 3))
        launch!(arch, grid, parameters, _update_native_momentum_flux!,
                closure_fields.scheme_u_flux[slot], closure_fields.scheme_v_flux[slot],
                closure_fields.numerical_u_correction[slot],
                closure_fields.numerical_v_correction[slot],
                grid, closure.advection.momentum, model.momentum.ρw, model.velocities,
                dynamics_density(model.dynamics), face, α)
        for name in keys(closure_fields.tupled_tracer_diffusivities)
            launch!(arch, grid, parameters, _update_native_scalar_flux!,
                    closure_fields.scheme_scalar_flux[name][slot],
                    closure_fields.numerical_scalar_correction[name][slot],
                    grid, closure.advection[name], model.velocities.w,
                    tracers[name], face, α)
        end
    end
    return nothing
end

surface_layer_scalar_fields(model,
                            closure::SurfaceLayerDiffusivity{TD, FT, G, Val{:covariance}}) where {TD, FT, G} =
    buoyancy_tracers(model)

surface_layer_scalar_fields(model,
                            closure::SurfaceLayerDiffusivity{TD, FT, G, Val{:scheme_native}}) where {TD, FT, G} =
    reconstructed_fields(model, closure.advection)

function AtmosphereModels.initialize_closure_fields_with_specific_tracers!(
        closure_fields::SurfaceLayerDiffusivityFields,
        closure::SurfaceLayerDiffusivity{TD, FT, G, Val{:scheme_native}},
        model) where {TD, FT, G}
    # `set!` has restored user tracers to their density form. The advection
    # operator reconstructs specific tracers, so temporarily match that state.
    tracer_density_to_specific!(model)
    try
        Oceananigans.TurbulenceClosures.initialize_closure_fields!(closure_fields, closure, model)
    finally
        tracer_specific_to_density!(model)
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
                closure_fields.numerical_u_correction[slot],
                closure_fields.numerical_v_correction[slot],
                closure_fields.surface_u_flux, closure_fields.surface_v_flux,
                grid, closure, face, weight)
        for name in keys(closure_fields.tupled_tracer_diffusivities)
            launch!(arch, grid, parameters, _compute_scalar_diffusivity!,
                    closure_fields.tupled_tracer_diffusivities[name],
                    closure_fields.scalar_deficit[name][slot],
                    closure_fields.scalar_active[name][slot],
                    closure_fields.diffusivity_cap_active[name][slot],
                    closure_fields.resolved_scalar_flux[name][slot],
                    closure_fields.numerical_scalar_correction[name][slot],
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
    scheme_u_flux=closure_fields.scheme_u_flux,
    scheme_v_flux=closure_fields.scheme_v_flux,
    numerical_u_correction=closure_fields.numerical_u_correction,
    numerical_v_correction=closure_fields.numerical_v_correction,
    scalar_mean=closure_fields.scalar_mean,
    scalar_w_product_mean=closure_fields.scalar_w_product_mean,
    resolved_scalar_flux=closure_fields.resolved_scalar_flux,
    scheme_scalar_flux=closure_fields.scheme_scalar_flux,
    numerical_scalar_correction=closure_fields.numerical_scalar_correction,
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
    # Checkpoints written before the scheme-native option contain only the original
    # covariance state. The new diagnostic filter fields remain zero on such pickup;
    # covariance-mode evolution is therefore unchanged.
    names = Tuple(name for name in keys(restored_fields) if hasproperty(from, name))
    matching_fields = NamedTuple{names}(getproperty(restored_fields, name) for name in names)
    from_fields = NamedTuple{names}(getproperty(from, name) for name in names)
    restore_prognostic_state!(matching_fields, from_fields)
    restored.previous_update_time[] = from.previous_update_time
    restored.previous_update_iteration[] = from.previous_update_iteration
    return restored
end

Oceananigans.restore_prognostic_state!(::SurfaceLayerDiffusivityFields, ::Nothing) = nothing
