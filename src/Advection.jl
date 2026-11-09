module Advection

export div_ρUc,
       bottom_advective_tracer_flux,
       FrozenWeightsWENO

using Oceananigans.Advection:
    Advection as OceananigansAdvection,
    _advective_tracer_flux_x,
    _advective_tracer_flux_y,
    _advective_tracer_flux_z,
    AdaptiveImplicitVerticalAdvection,
    AbstractUpwindBiasedAdvectionScheme,
    _biased_interpolate_xᶠᵃᵃ,
    _biased_interpolate_yᵃᶠᵃ,
    _biased_interpolate_zᵃᵃᶠ,
    BoundsPreservation,
    BoundsPreservingWENO,
    FunctionStencil,
    LeftBias,
    RightBias,
    upwind_biased_product,
    rescaled_reconstruction,
    implicit_vertical_velocityᶜᶜᶠ,
    bounded_tracer_flux_divergence_x,
    bounded_tracer_flux_divergence_y,
    bounded_tracer_flux_divergence_z,
    explicit_velocity_scaleᶜᶜᶠ,
    materialize_advection,
    adapt_advection_order,
    update_adaptive_timestep!,
    vertical_scheme,
    WENO

using Adapt: Adapt, adapt
using Oceananigans: Oceananigans, CenterField
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: Architectures, on_architecture
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.Grids: Center
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, Azᶜᶜᶠ
using Oceananigans.Utils: SumOfArrays, AdaptiveVerticallyImplicitDiscretization
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, time_discretization
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF

using Breeze.TimeSteppers: tendencies_feed_first_stage

using ..AtmosphereModels:
    AtmosphereModels,
    div_ρUc,
    total_density,
    transport_velocities

const AIVA = AdaptiveImplicitVerticalAdvection

# Simple wrappers: interpolate ρ to face, multiply existing flux
@inline tracer_mass_flux_x(i, j, k, grid, ρ, args...) =
    ℑxᶠᵃᵃ(i, j, k, grid, ρ) * _advective_tracer_flux_x(i, j, k, grid, args...)

@inline tracer_mass_flux_y(i, j, k, grid, ρ, args...) =
    ℑyᵃᶠᵃ(i, j, k, grid, ρ) * _advective_tracer_flux_y(i, j, k, grid, args...)

@inline tracer_mass_flux_z(i, j, k, grid, ρ, args...) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ) * _advective_tracer_flux_z(i, j, k, grid, args...)

# Main operator
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection, ρ, U, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (
        δxᶜᵃᵃ(i, j, k, grid, tracer_mass_flux_x, ρ, advection, U.u, c) +
        δyᵃᶜᵃ(i, j, k, grid, tracer_mass_flux_y, ρ, advection, U.v, c) +
        δzᵃᵃᶜ(i, j, k, grid, tracer_mass_flux_z, ρ, advection, U.w, c))
end

# Fallback for nothing advection
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::Nothing, ρ, U, c) = zero(grid)
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::BoundsPreservingWENO, ρ, U, ::ZeroField) = zero(grid)

# Is this immersed-boundary safe without having to extend it in ImmersedBoundaries.jl? I think so... (velocity on immmersed boundaries is masked to 0)
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection::BoundsPreservingWENO, ρ, U, c)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, U.v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, U.w, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

# A bounds-preserving WENO whose vertical time discretization is adaptive-implicit.
const BoundsPreservingAVIDWENO = WENO{<:Any, <:Any, <:Any, <:AdaptiveVerticallyImplicitDiscretization, <:BoundsPreservation}

# Indexing yields wᵉ = s·w, the explicit fraction of the IMEX split, so the bounded flux
# functions consume it without duplicating their reconstruction (issue #913).
struct ExplicitVerticalVelocity{G, S, T, W}
    grid :: G
    advection_scheme :: S
    time_discretization :: T
    vertical_velocity :: W
end

Adapt.adapt_structure(to, v::ExplicitVerticalVelocity) =
    ExplicitVerticalVelocity(Adapt.adapt(to, v.grid), Adapt.adapt(to, v.advection_scheme),
                             Adapt.adapt(to, v.time_discretization), Adapt.adapt(to, v.vertical_velocity))

@inline Base.getindex(v::ExplicitVerticalVelocity, i, j, k) =
    @inbounds explicit_velocity_scaleᶜᶜᶠ(i, j, k, v.grid, v.advection_scheme, v.time_discretization,
                                         v.vertical_velocity) * v.vertical_velocity[i, j, k]

# Disambiguates against the `ZeroField` shortcut above.
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::BoundsPreservingAVIDWENO, ρ, U, ::ZeroField) = zero(grid)

# Without the s-scaled velocity the bounded path transported 1 + (1 - s) times: a full
# explicit flux plus the implicit remainder (issue #913). Horizontal fluxes stay explicit.
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection::BoundsPreservingAVIDWENO, ρ, U, c)
    wᵉ = ExplicitVerticalVelocity(grid, advection, time_discretization(advection), U.w)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, U.v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, wᵉ, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

# The vertical velocity the bounded flux functions below transport with: the velocity itself
# for an explicit scheme, its s-scaled explicit fraction under adaptive implicit vertical
# advection (the same `ExplicitVerticalVelocity` the bounded divergence uses, so the scaling is
# applied exactly once).
@inline explicit_vertical_velocity(advection, grid, w) = w
@inline explicit_vertical_velocity(advection::BoundsPreservingAVIDWENO, grid, w) =
    ExplicitVerticalVelocity(grid, advection, time_discretization(advection), w)

#####
##### Advective mass fluxes for the sedimentation of condensate content
#####
#
# General method of `AtmosphereModels.sedimentation_mass_fluxes`, consumed by
# `AtmosphereModels.sedimentation_tendency`: through the two faces
# of cell (i, j, k), the vertical advective flux of `q` at the combined velocity `wᵗ + wˢ` and at
# the transport velocity `wᵗ` alone, with the tracer's own advection scheme, so that their
# difference is by construction the sedimentation part of the mass flux the tracer tendency
# applies to the cell through `div_ρUc`. For `advection === nothing` every flux vanishes
# (Oceananigans returns zero), so no mass and no latent heat move, consistently.
#
# Under adaptive implicit vertical advection these are the CFL-scaled explicit fluxes only,
# the part of the transport the tendency applies; the first-order remainder the implicit solve
# applies is moved with its content after the solve, from the solved state
# (`AtmosphereModels.implicit_sedimentation_mass_fluxes`).

@inline function AtmosphereModels.sedimentation_mass_fluxes(i, j, k, grid, advection, wᵗ, wˢ, q)
    w = SumOfArrays{2}(wᵗ, wˢ)
    F⁻ = (_advective_tracer_flux_z(i, j, k,   grid, advection, w,  q),
          _advective_tracer_flux_z(i, j, k,   grid, advection, wᵗ, q))
    F⁺ = (_advective_tracer_flux_z(i, j, k+1, grid, advection, w,  q),
          _advective_tracer_flux_z(i, j, k+1, grid, advection, wᵗ, q))
    return F⁻, F⁺
end

# Bounds-preserving WENO rescales its face reconstructions by the cached limiter of the cell
# each one draws on, so the fluxes of cell k are rebuilt from the same limited reconstructions
# `bounded_tracer_flux_divergence_z` forms its mass fluxes from: the latent heat stays with the
# mass at cloud and precipitation edges, where the limiter acts and the unlimited WENO fluxes
# would move heat the tracer tendency does not move.
@inline function AtmosphereModels.sedimentation_mass_fluxes(i, j, k, grid, advection::BoundsPreservingWENO, wᵗ, wˢ, q)
    w = SumOfArrays{2}(wᵗ, wˢ)
    c₋ᴸ, c₋ᴿ, c₊ᴸ, c₊ᴿ = bounded_face_reconstructions(i, j, k, grid, advection, q)
    F⁻ = (bounded_sedimentation_mass_flux(i, j, k,   grid, advection, w,  q, c₋ᴸ, c₋ᴿ),
          bounded_sedimentation_mass_flux(i, j, k,   grid, advection, wᵗ, q, c₋ᴸ, c₋ᴿ))
    F⁺ = (bounded_sedimentation_mass_flux(i, j, k+1, grid, advection, w,  q, c₊ᴸ, c₊ᴿ),
          bounded_sedimentation_mass_flux(i, j, k+1, grid, advection, wᵗ, q, c₊ᴸ, c₊ᴿ))
    return F⁻, F⁺
end

# Face flux from the limited reconstructions `cᴸ`, `cᴿ`, with the explicit CFL scaling that
# `div_ρUc` applies under adaptive implicit vertical advection.
@inline function bounded_sedimentation_mass_flux(i, j, k, grid, advection, w, q, cᴸ, cᴿ)
    explicit_w = explicit_vertical_velocity(advection, grid, w)
    @inbounds wₑ = explicit_w[i, j, k]
    return Azᶜᶜᶠ(i, j, k, grid) * upwind_biased_product(wₑ, cᴸ, cᴿ)
end

# Reconstructions of `c` at the lower (`k`) and upper (`k + 1`) faces of cell (i, j, k), returned
# as `(c₋ᴸ, c₋ᴿ, c₊ᴸ, c₊ᴿ)`, each rescaled by the cached bounds-preserving limiter θ of the cell it
# draws on (k − 1, k, k and k + 1), exactly as `bounded_tracer_flux_divergence_z` forms its face
# states, so that the two cells sharing a face see the same limited flux through it and the
# latent heat moves with exactly the mass the tracer tendency moves. The limiter is refreshed by
# `update_advection!` before the tendencies that consume it (see `update_state!`).
@inline function bounded_face_reconstructions(i, j, k, grid, advection::BoundsPreservingWENO, c)
    θ = advection.bounds.limiter

    c₊ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, LeftBias,  c)
    c₊ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, RightBias, c)
    c₋ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, RightBias, c)

    c₊ᴸ = rescaled_reconstruction(c₊ᴸ, i, j, k,   grid, θ, c)
    c₊ᴿ = rescaled_reconstruction(c₊ᴿ, i, j, k+1, grid, θ, c)
    c₋ᴸ = rescaled_reconstruction(c₋ᴸ, i, j, k-1, grid, θ, c)
    c₋ᴿ = rescaled_reconstruction(c₋ᴿ, i, j, k,   grid, θ, c)

    return c₋ᴸ, c₋ᴿ, c₊ᴸ, c₊ᴿ
end

#####
##### Advection-consistent surface flux
#####

"""
$(TYPEDSIGNATURES)

Compute the downward advective mass flux per unit area at the bottom face (`k = 1`)
for tracer `c` advected by vertical velocity `w` through density field `ρ`, using the
given `advection` scheme.

For an explicit scheme this evaluates the same face flux that `div_ρUc` uses at
the bottom boundary. For adaptive implicit vertical advection it returns the
instantaneous split-operator flux at the supplied tracer state; the step-integrated
implicit outflow depends on the post-solve tracer and must be accumulated separately.

For adaptive implicit vertical advection, the result includes both the CFL-scaled
high-order explicit flux and its first-order implicit remainder, evaluated at `c`.

Returns a positive value for downward (out-of-domain) flux.
"""
@inline function bottom_advective_tracer_flux(i, j, grid, advection, ρ, w, c)
    flux_Az = _advective_tracer_flux_z(i, j, 1, grid, advection, w, c)
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    explicit_flux = -ρ_face * flux_Az / Azᶜᶜᶠ(i, j, 1, grid)
    return explicit_flux + implicit_bottom_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c)
end

@inline implicit_bottom_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c) = 0

@inline function implicit_bottom_advective_tracer_flux(i, j, grid, advection::AIVA, ρ_face, w, c)
    scheme = vertical_scheme(advection)
    td = time_discretization(scheme)
    wⁱ = implicit_vertical_velocityᶜᶜᶠ(i, j, 1, grid, scheme, td, w)
    @inbounds c_above = c[i, j, 1]
    return -ρ_face * min(wⁱ, 0) * c_above
end

# Bounds-preserving WENO: the bottom face flux of cell 1 as `bounded_tracer_flux_divergence_z`
# forms it, from the reconstructions rescaled by the cached limiter of each donor cell.
@inline function bottom_advective_tracer_flux(i, j, grid, advection::BoundsPreservingWENO, ρ, w, c)
    c₋ᴸ, c₋ᴿ, _, _ = bounded_face_reconstructions(i, j, 1, grid, advection, c)
    explicit_w = explicit_vertical_velocity(advection, grid, w)
    @inbounds w⁻ = explicit_w[i, j, 1]
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    explicit_flux = -ρ_face * upwind_biased_product(w⁻, c₋ᴸ, c₋ᴿ)
    return explicit_flux + implicit_bottom_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c)
end

#####
##### Precipitation flux through the bottom boundary
#####
#
# Scheme-independent implementation of `AtmosphereModels.bottom_precipitation_flux`: the
# bottom-face flux of every sedimenting condensate, summed inside one kernel function. It
# lives here rather than in `AtmosphereModels` because it builds on
# `bottom_advective_tracer_flux`, and `AtmosphereModels` is loaded before this module.
# Reusing the `SedimentingCondensate`s the model resolved once means the diagnostic can never
# disagree with the thermodynamic tendencies about which masses fall, with which humidity field
# and advection scheme. The tuple recursion keeps the kernel type-stable across condensates that
# carry different advection schemes.
@inline bottom_precipitation_flux_kernel(i, j, k, grid, condensates, ρ, wᵗ) =
    sedimenting_bottom_flux(i, j, grid, condensates, ρ, wᵗ)

@inline sedimenting_bottom_flux(i, j, grid, ::Tuple{}, ρ, wᵗ) = zero(grid)

@inline function sedimenting_bottom_flux(i, j, grid, condensates, ρ, wᵗ)
    condensate = first(condensates)
    w = SumOfArrays{2}(wᵗ, condensate.velocity)
    flux = bottom_advective_tracer_flux(i, j, grid, condensate.advection, ρ, w, condensate.specific_humidity)
    return flux + sedimenting_bottom_flux(i, j, grid, Base.tail(condensates), ρ, wᵗ)
end

# Any scheme that declares its sedimenting condensate through `sedimentation_velocity` and
# `condensate_liquid_fraction` gets the advection-consistent diagnostic for free; with nothing sedimenting
# (including `Nothing` microphysics) the sum is empty and the flux is zero. Schemes that move
# precipitation by internal means (such as `DCMIP2016KM`) override this method instead.
function AtmosphereModels.bottom_precipitation_flux(model, microphysics)
    operation = KernelFunctionOperation{Center, Center, Nothing}(bottom_precipitation_flux_kernel, model.grid,
                                                                 values(model.sedimentation),
                                                                 total_density(model.dynamics),
                                                                 transport_velocities(model).w)
    return Field(operation)
end

#####
##### WENO with nonlinear weights frozen at the outer-step start
#####

# Reads the stored step-start tracer. A `Function` subtype, so WENO's `FunctionStencil` path
# loads its smoothness stencil from this field in place of the reconstructed one.
struct StepStartTracer{F} <: Function
    field :: F
end

@inline (c::StepStartTracer)(i, j, k, grid) = @inbounds c.field[i, j, k]

Adapt.adapt_structure(to, c::StepStartTracer) = StepStartTracer(adapt(to, c.field))
Architectures.on_architecture(to, c::StepStartTracer) = StepStartTracer(on_architecture(to, c.field))

Adapt.adapt_structure(to, VI::FunctionStencil{<:StepStartTracer}) = FunctionStencil(adapt(to, VI.func))
Architectures.on_architecture(to, VI::FunctionStencil{<:StepStartTracer}) = FunctionStencil(on_architecture(to, VI.func))

"""
$(TYPEDEF)

A scalar advection scheme that reconstructs with the stencils and optimal weights of a `WENO`
scheme, but with the nonlinear smoothness weights computed from the tracer at the start of the
outer time step rather than from the stage state being reconstructed. The fluxes stay live: only
the weights are frozen, so the stage reconstruction is linear in the stage tracer and the
[`MultirateInfinitesimalStep`](@ref) stage recombination sees one consistent advection operator
for the whole step, as it does with `UpwindBiased`. Under any other time stepper the frozen
tracer is refreshed before every tendency evaluation, and the scheme reduces to its `WENO`.

The scheme carries one `CenterField` per tracer, the tracer copy at the step start, which
`update_advection!` refreshes before the tendency evaluations that feed the first stage of a
step (`Breeze.TimeSteppers.tendencies_feed_first_stage`). It is a scalar
scheme: momentum has no step-start copy, so it cannot be given as `momentum_advection`.
Bounds preservation is not supported.

```jldoctest
using Breeze

FrozenWeightsWENO(order=9)

# output
FrozenWeightsWENO(order=9)
└── weno: WENO{5, Float64, Nothing}(order=9)
```
"""
struct FrozenWeightsWENO{N, FT, TD, W, S} <: AbstractUpwindBiasedAdvectionScheme{N, FT, TD}
    weno :: W                # stencils, smoothness indicators and optimal weights
    smoothness_stencil :: S  # `nothing` until materialized; then reads the step-start tracer

    function FrozenWeightsWENO(weno::WENO{N, FT, WCT, TD}, smoothness_stencil::S) where {N, FT, WCT, TD, S}
        isnothing(weno.bounds) ||
            throw(ArgumentError("FrozenWeightsWENO does not support bounds-preserving WENO."))
        return new{N, FT, TD, typeof(weno), S}(weno, smoothness_stencil)
    end
end

"""
$(TYPEDSIGNATURES)

Construct a [`FrozenWeightsWENO`](@ref) from `weno`, or from the `WENO` built with `FT` and the
keyword arguments of `WENO`.
"""
FrozenWeightsWENO(weno::WENO) = FrozenWeightsWENO(weno, nothing)
FrozenWeightsWENO(FT::DataType=Oceananigans.defaults.FloatType; kwargs...) = FrozenWeightsWENO(WENO(FT; kwargs...))

step_start_tracer(scheme::FrozenWeightsWENO) = scheme.smoothness_stencil.func.field

Base.summary(::FrozenWeightsWENO{N}) where N = string("FrozenWeightsWENO(order=", 2N-1, ")")

function Base.show(io::IO, scheme::FrozenWeightsWENO)
    print(io, summary(scheme), '\n')
    print(io, "└── weno: ", summary(scheme.weno))
end

Adapt.adapt_structure(to, scheme::FrozenWeightsWENO) =
    FrozenWeightsWENO(adapt(to, scheme.weno), adapt(to, scheme.smoothness_stencil))

Architectures.on_architecture(to, scheme::FrozenWeightsWENO) =
    FrozenWeightsWENO(on_architecture(to, scheme.weno), on_architecture(to, scheme.smoothness_stencil))

@inline OceananigansTimeSteppers.time_discretization(scheme::FrozenWeightsWENO) = time_discretization(scheme.weno)

# The step-start copy is allocated here, once per tracer, like the bounds-preserving limiter.
OceananigansAdvection.materialize_advection(scheme::FrozenWeightsWENO, grid) =
    FrozenWeightsWENO(materialize_advection(scheme.weno, grid), FunctionStencil(StepStartTracer(CenterField(grid))))

# Order adaptation on small grids happens after materialization in `AtmosphereModel`, so the
# lowered-order WENO is materialized here and the step-start copy is kept.
function OceananigansAdvection.adapt_advection_order(scheme::FrozenWeightsWENO{B}, N::Int, grid) where B
    N >= B && return scheme
    return FrozenWeightsWENO(materialize_advection(adapt_advection_order(scheme.weno, N, grid), grid), scheme.smoothness_stencil)
end

AtmosphereModels.validate_momentum_advection(::FrozenWeightsWENO, grid) =
    throw(ArgumentError("FrozenWeightsWENO is a scalar advection scheme; pass it as `scalar_advection` " *
                        "and give `momentum_advection` a scheme of its own."))

# The reconstruction: WENO's `FunctionStencil` path loads the stencil of `ψ` for the anchor and
# differences, and the stencil of the step-start tracer for the smoothness indicators. Routing
# through the topologically conditional `_biased_interpolate` of the inner scheme keeps the
# buffer-scheme fallback near boundaries, which passes the stencil on to the lower-order WENOs.
@inline OceananigansAdvection._biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::FrozenWeightsWENO, bias, ψ) =
    _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.weno, bias, ψ, scheme.smoothness_stencil)

@inline OceananigansAdvection._biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::FrozenWeightsWENO, bias, ψ) =
    _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.weno, bias, ψ, scheme.smoothness_stencil)

@inline OceananigansAdvection._biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::FrozenWeightsWENO, bias, ψ) =
    _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.weno, bias, ψ, scheme.smoothness_stencil)

"""
$(TYPEDSIGNATURES)

Refresh the step-start copy of `tracer` that `scheme` computes its smoothness weights from, when the
tendencies about to be built feed the first stage of an outer step; between the stages of a
[`MultirateInfinitesimalStep`](@ref) the copy is left as it is. The halos are copied with the
interior, so no halo fill follows.
"""
function OceananigansAdvection.update_advection!(scheme::FrozenWeightsWENO, model, tracer)
    update_adaptive_timestep!(scheme, model)
    if tendencies_feed_first_stage(model.timestepper)
        copyto!(parent(step_start_tracer(scheme)), parent(tracer))
    end
    return nothing
end

end # module
