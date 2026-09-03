#####
##### Adaptive implicit vertical advection (AIVA) for the anelastic, mass-flux formulation
#####
##### Breeze advects momentum with the mass flux `(ρu, ρv, ρw)` as the advecting field, so the
##### upstream explicit-flux scaling — which builds the vertical CFL from the advecting field it is
##### handed — splits on `|ρw|` instead of `|w|`, inconsistently with the implicit solve. The AIVA
##### methods below scale the vertical momentum flux with the velocity CFL instead.
#####
##### The vertically-implicit *diffusion* half of the tridiagonal row is not density weighted
##### upstream: Breeze's prognostics are density weighted while the explicit flux divergence forms
##### `∂z(ρ κ ∂z c)` on the *specific* variable. The coefficients in
##### `density_weighted_implicit_diffusion.jl` fix that at both z-locations.

using Oceananigans.Advection:
    AdaptiveImplicitVerticalAdvection,
    vertical_scheme,
    explicit_velocity_scaleᶠᶜᶠ,
    explicit_velocity_scaleᶜᶠᶠ,
    explicit_velocity_scaleᶜᶜᶜ,
    advective_momentum_flux_Wu,
    advective_momentum_flux_Wv,
    advective_momentum_flux_Ww,
    _advective_momentum_flux_Uu,
    _advective_momentum_flux_Vu,
    _advective_momentum_flux_Uv,
    _advective_momentum_flux_Vv,
    _advective_momentum_flux_Uw,
    _advective_momentum_flux_Vw

using Oceananigans.Operators: V⁻¹ᶠᶜᶜ, V⁻¹ᶜᶠᶜ, V⁻¹ᶜᶜᶠ,
                              δxᶠᵃᵃ, δxᶜᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ
using Oceananigans.TimeSteppers: ExplicitTimeDiscretization, time_discretization

const AIVA = AdaptiveImplicitVerticalAdvection

#####
##### Per-field advection lookup for the implicit step
#####

# Momentum prognostics share the single `:momentum` scheme; scalars are keyed by name.
@inline function field_advection_scheme(advection, name::Symbol)
    (name === :ρu || name === :ρv || name === :ρw) && return advection.momentum
    return haskey(advection, name) ? advection[name] : nothing
end

# Assembles the scheme object that configures a prognostic's tridiagonal row. It travels in
# `implicit_step!`'s `advection` slot and adds the density-weighted diffusion coefficients, which
# dispatch on the field's z-location: `ρw` takes the z-Face row, everything else the z-Center one.
# Explicit schemes are wrapped too, since the diffusion half needs the weighting either way; the
# advection half passes `ℓz` through to upstream's coefficients unchanged.
#
# `density` weights the diffusion half when it must differ from the density the solve is
# called with, which happens only under the acoustic substepper (see `DensityWeightedImplicitOperator`).
implicit_step_scheme(advection, density=nothing) = DensityWeightedImplicitOperator(advection, density)

# Density weighting the advective flux of each prognostic. Momentum and the thermodynamic
# variable are carried by the coupling density (`ρu = ρᵈ u`, `ρθ = ρᵈ θ`; see `dynamics_density`),
# while water species and tracers advect as mass fractions of the total density ρ = ρᵈ + Σρˣ
# (see `scalar_tendency`). The implicit solve must weight its upwind coefficients with the same
# density the explicit flux divergence uses; on the anelastic core the two densities coincide.
function implicit_advection_density(dynamics, formulation, name::Symbol)
    coupling = name === :ρu || name === :ρv || name === :ρw ||
               name === thermodynamic_density_name(formulation)
    return coupling ? dynamics_density(dynamics) : total_density(dynamics)
end

# Velocities whose vertical component the implicit solve splits — these must match the velocity
# each prognostic's tendency advects with. Momentum advects with the (possibly contravariant)
# advecting vertical velocity; every other prognostic advects with `velocities` as given.
function implicit_advection_velocities(dynamics, velocities, name::Symbol)
    momentum = name === :ρu || name === :ρv || name === :ρw
    return momentum ? (; w = advecting_vertical_velocity(dynamics, velocities)) : velocities
end

#####
##### Explicit vertical momentum fluxes scaled by the velocity CFL
#####
##### These mirror Oceananigans' adaptive-implicit flux scaling but compute the scale from the
##### velocity `w` rather than from the advecting mass flux `ρw`, matching the implicit
##### velocities used by the tridiagonal solve.
#####

@inline function scaled_momentum_flux_Wu(i, j, k, grid, advection, W, u, w)
    scheme = vertical_scheme(advection)
    td = time_discretization(scheme)
    s = explicit_velocity_scaleᶠᶜᶠ(i, j, k, grid, scheme, td, w)
    return s * advective_momentum_flux_Wu(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, u)
end

@inline function scaled_momentum_flux_Wv(i, j, k, grid, advection, W, v, w)
    scheme = vertical_scheme(advection)
    td = time_discretization(scheme)
    s = explicit_velocity_scaleᶜᶠᶠ(i, j, k, grid, scheme, td, w)
    return s * advective_momentum_flux_Wv(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, v)
end

@inline function scaled_momentum_flux_Ww(i, j, k, grid, advection, W, w)
    scheme = vertical_scheme(advection)
    td = time_discretization(scheme)
    s = explicit_velocity_scaleᶜᶜᶜ(i, j, k, grid, scheme, td, w)
    return s * advective_momentum_flux_Ww(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, w)
end

# The AIVA methods reproduce `div_𝐯u/v/w` with the vertical flux routed through the
# velocity-CFL scaling above. Horizontal fluxes dispatch to the fully-explicit methods
# Oceananigans defines for the adaptive-implicit time discretization.
@inline function x_momentum_flux_divergence(i, j, k, grid, advection::AIVA, momentum, velocities, dynamics)
    w = advecting_vertical_velocity(dynamics, velocities)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uu, advection, momentum[1], velocities.u) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vu, advection, momentum[2], velocities.u) +
                                    δzᵃᵃᶜ(i, j, k, grid, scaled_momentum_flux_Wu, advection, momentum[3], velocities.u, w)) +
           U_dot_∇u_metric(i, j, k, grid, advection, momentum, velocities)
end

@inline function y_momentum_flux_divergence(i, j, k, grid, advection::AIVA, momentum, velocities, dynamics)
    w = advecting_vertical_velocity(dynamics, velocities)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uv, advection, momentum[1], velocities.v) +
                                    δyᵃᶠᵃ(i, j, k, grid, _advective_momentum_flux_Vv, advection, momentum[2], velocities.v) +
                                    δzᵃᵃᶜ(i, j, k, grid, scaled_momentum_flux_Wv, advection, momentum[3], velocities.v, w)) +
           U_dot_∇v_metric(i, j, k, grid, advection, momentum, velocities)
end

@inline function z_momentum_flux_divergence(i, j, k, grid, advection::AIVA, momentum, velocities, dynamics)
    w = advecting_vertical_velocity(dynamics, velocities)
    return V⁻¹ᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uw, advection, momentum[1], velocities.w) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vw, advection, momentum[2], velocities.w) +
                                    δzᵃᵃᶠ(i, j, k, grid, scaled_momentum_flux_Ww, advection, momentum[3], w)) +
           U_dot_∇w_metric(i, j, k, grid, advection, momentum, velocities)
end
