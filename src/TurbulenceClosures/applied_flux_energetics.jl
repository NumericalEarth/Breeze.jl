#####
##### The energy conversions of the fluxes `TKEBasedTurbulenceClosure` applies, for the TKE
##### equation of a closure with a `AppliedFluxEnergetics`
#####
##### The closure diffuses every scalar `φ` (θˡⁱ, vapor, every condensate species, ...) with `Kᶜ`
##### and momentum with `Kᵘ`, density weighted, through the flux `ρᶠ K ∂z φ` at the interior
##### faces. The turbulent kinetic energy it produces is the energy those fluxes take from the
##### mean state, face by face:
#####
#####   shear:     Pᶠ = Kᵘ S²                                  (S² of the stored velocities)
#####   buoyancy:  Bᶠ = Σ_χ b_χᶠ F_χ,   Fₛ = -Kᶜ (φₛ,ₖ - φₛ,ₖ₋₁) / Δzᶠ,   b_χ = ∂b/∂χ
#####
##### where `b` is the buoyancy of the dynamics (for anelastic dynamics b = g (1 - Rᵐᵣ Tᵣ / (Rᵐ T)),
##### condensate loading through Rᵐ) and `b_χ` its exact derivative with respect to each mixed
##### specific variable at fixed pressure, at the centers, averaged to the face (`b_χᶠ`). The
##### mixed variables that change `b` are θˡⁱ and the vapor, liquid and ice mass fractions of the
##### microphysics (`grid_moisture_fractions`, linear in its prognostic densities, so the flux of
##### all liquid is the sum of the liquid species' fluxes, and so on).
#####
##### Summation by parts makes these exact discrete conversions:
#####
#####   Σₖ uₖ Δzₖ ∂z(ρ Kᵘ ∂z u)ₖ = -Σᶠ ρᶠ Δzᶠ Pᶠ + boundary terms,
#####   d/dt Σₖ ρₖ Δzₖ h‡ₖ (closure part) = -Σᶠ ρᶠ Δzᶠ Bᶠ + boundary terms,
#####
##### with the dynamic enthalpy h‡ = -∫₀ᶻ b dz′ discretized by the trapezoid rule on the
##### centers (∂h‡/∂φₛ at center k is -Zₛ,ₖ, Zₛ,ₖ - Zₛ,ₖ₋₁ = b_χᶠ Δzᶠ). The face conversions are
##### moved to the centers, where the TKE lives, with the half-cell weights of the two faces of
##### each cell,
#####
#####   ρᶜ Δzᶜ Xᶜ = ρᶠₖ (zᶜₖ - zᶠₖ) Xᶠₖ + ρᶠₖ₊₁ (zᶠₖ₊₁ - zᶜₖ) Xᶠₖ₊₁,
#####
##### so that Σ ρᶜ Δzᶜ Xᶜ = Σ ρᶠ Δzᶠ Xᶠ exactly. The closure's diffusivities vanish at the
##### boundary faces; the surface fluxes of the boundary conditions are boundary terms of the
##### energy budget, not conversions.
#####
##### Time level: the conversions are formed from the stage state and the diffusivities the stage
##### solve uses. The solve is backward Euler in each stage, so the fluxes it applies are those of
##### the solved state; the difference is O(Δt) and is not closed here.
#####

using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶠ
using Oceananigans.Models: boundary_condition_args
using SpecialFunctions: erfc

using Oceananigans.BoundaryConditions: BoundaryCondition, Flux, Bottom, total_boundary_flux

# Whether the fields `cl` of the closure hold the field `name`, as a type: the optional components (the budget fluxes,
# the flux of liquid) add their fields to `cl`, and what reads them is chosen by dispatch on this
@inline has_field(::NamedTuple{names}, ::Val{name}) where {names, name} = Val(name ∈ names)

# The field `name` of `cl` at (i, j, k), or the zero of the type `FT` without it
@inline stored_or_zero(cl, name::Val, i, j, k, FT) = stored_or_zero(has_field(cl, name), cl, name, i, j, k, FT)
@inline stored_or_zero(::Val{true}, cl, ::Val{name}, i, j, k, FT) where name = @inbounds getproperty(cl, name)[i, j, k]
@inline stored_or_zero(::Val{false}, cl, name, i, j, k, FT) = zero(FT)

# The density-weighted flux through the bottom of column (i, j) that the boundary condition `bc` of the
# prognostic field `c` prescribes: positive upward, into the domain. Only `Flux` boundary conditions carry one
# (the default no-flux condition gives zero).
@inline bottom_boundary_flux(bc::BoundaryCondition{<:Flux}, i, j, grid, c, args...) =
    total_boundary_flux(bc, Bottom(), i, j, 1, grid, c, args...)

@inline bottom_boundary_flux(bc, i, j, grid, c, args...) = zero(grid)

using Breeze.Thermodynamics: mixture_heat_capacity, mixture_gas_constant, exner_exponent,
                             VaporEquivalentExponent, DryExponent

# ∂κ/∂qᵗ of the Exner exponent; κ̂ = R̂ / ĉᵖ with R̂, ĉᵖ linear in qᵗ
@inline exner_exponent_moisture_derivative(::DryExponent, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ) = zero(R̂)
@inline exner_exponent_moisture_derivative(::VaporEquivalentExponent, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ) =
    R̂ / ĉᵖ * ((Rᵛ - Rᵈ) / R̂ - (cᵖᵛ - cᵖᵈ) / ĉᵖ)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The derivatives `(b_θ, b_qᵛ, b_qˡ, b_qⁱ)` of the buoyancy `b` of the dynamics with respect to the
liquid-ice potential temperature and the vapor, liquid and ice mass fractions, at fixed pressure,
at cell center `(i, j, k)`, followed by the liquid-ice potential temperature `θ` and the moisture
mass fractions `q` there. With ``X = Rᵐ T``, ``T = (ĉᵖ Π θˡⁱ + ℒˡ₀ qˡ + ℒⁱ₀ qⁱ) / cᵖᵐ`` and
``b = g (1 - Xᵣ / X)``, ``∂b/∂φ = (g - b) ∂ ln X / ∂φ``; `g - b` is taken from the buoyancy force
of the dynamics, so the derivatives hold for any dynamics whose buoyancy is a function of
``Rᵐ T`` at fixed pressure.
"""
@inline function buoyancy_derivativesᶜᶜᶜ(i, j, k, grid, s)
    @inbounds begin
        ρ = s.ρ[i, j, k]
        θ = s.ρθ[i, j, k] / ρ
        qᵛᵉ = s.qᵛᵉ[i, j, k]
        p = s.p[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉ, s.microphysical_fields)

    # g - b = g ρ / ρᵣ from the buoyancy force density -g (ρ - ρᵣ) of the dynamics
    g = s.constants.gravitational_acceleration
    b = AtmosphereModels.buoyancy_forceᶜᶜᶜ(i, j, k, grid, s.dynamics, s.T, s.qᵛᵉ,
                                          s.microphysics, s.microphysical_fields, s.constants) / ρ

    b_θ, b_qᵛ, b_qˡ, b_qⁱ = buoyancy_derivatives(θ, p, q, g - b, s.constants, s.exner_exponent, s.standard_pressure)
    return b_θ, b_qᵛ, b_qˡ, b_qⁱ, θ, q
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The derivatives `(b_θ, b_qᵛ, b_qˡ, b_qⁱ)` of the buoyancy at fixed pressure `p` of air with liquid-ice potential
temperature `θ` and moisture mass fractions `q`, given `G = g - b` (see `buoyancy_derivativesᶜᶜᶜ`).
"""
@inline function buoyancy_derivatives(θ, p, q, G, constants, exner_exponent_formulation, pˢᵗ)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    cⁱ = constants.ice.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)

    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    R̂ = vapor_equivalent_gas_constant(q, constants)
    κ = exner_exponent(exner_exponent_formulation, q, constants)
    κ′ = exner_exponent_moisture_derivative(exner_exponent_formulation, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    ln_p = log(p / pˢᵗ)
    Π = (p / pˢᵗ)^κ

    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    H = ĉᵖ * Π * θ + ℒˡ₀ * q.liquid + ℒⁱ₀ * q.ice # = cᵖᵐ T
    ∂H∂qᵗ = θ * Π * ((cᵖᵛ - cᵖᵈ) + ĉᵖ * ln_p * κ′)

    b_θ = G * ĉᵖ * Π / H
    b_qᵛ = G * ((Rᵛ - Rᵈ) / Rᵐ + ∂H∂qᵗ / H - (cᵖᵛ - cᵖᵈ) / cᵖᵐ)
    b_qˡ = G * (-Rᵈ / Rᵐ + (∂H∂qᵗ + ℒˡ₀) / H - (cˡ - cᵖᵈ) / cᵖᵐ)
    b_qⁱ = G * (-Rᵈ / Rᵐ + (∂H∂qᵗ + ℒⁱ₀) / H - (cⁱ - cᵖᵈ) / cᵖᵐ)

    return b_θ, b_qᵛ, b_qˡ, b_qⁱ
end

# The derivatives of the buoyancy and the enthalpy coefficient d(cᵖᵐ T)/dθˡⁱ = ĉᵖ Π at the centers,
# stored once per update so that the conversions after the implicit solves use the same ones
@kernel function _compute_buoyancy_derivatives!(cl, grid, s)
    i, j, k = @index(Global, NTuple)
    b_θ, b_qᵛ, b_qˡ, b_qⁱ, θ, q = buoyancy_derivativesᶜᶜᶜ(i, j, k, grid, s)
    constants = s.constants
    κ = exner_exponent(s.exner_exponent, q, constants)
    Π = (@inbounds(s.p[i, j, k]) / s.standard_pressure)^κ
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    # The derivative with respect to the condensate of the distributed water: liquid and ice in the proportion of its
    # phase equilibrium at the temperature of the cell (liquid for a microphysics with prognostic cloud liquid)
    λ, _ = distributed_condensate_liquid_fraction(s.phase_equilibrium, @inbounds(s.T[i, j, k]))
    @inbounds begin
        cl.b_θ[i, j, k] = b_θ
        cl.b_qᵛ[i, j, k] = b_qᵛ
        cl.b_qˡ[i, j, k] = b_qˡ
        cl.b_qⁱ[i, j, k] = b_qⁱ
        cl.b_qᶜ[i, j, k] = λ * b_qˡ + (1 - λ) * b_qⁱ
        cl.ĉᵖΠ[i, j, k] = ĉᵖ * Π
    end
end

compute_buoyancy_derivatives!(fields, model) = nothing

function compute_buoyancy_derivatives!(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, model)
    grid = model.grid
    cl = fields.cloud
    launch!(grid.architecture, grid, :xyz, _compute_buoyancy_derivatives!, cl, grid, applied_flux_state(model))
    fill_halo_regions!((cl.b_θ, cl.b_qᵛ, cl.b_qˡ, cl.b_qⁱ, cl.b_qᶜ))
    return nothing
end

# Bᶠ at the interior face k from the stored derivatives and the current state
@inline applied_buoyancy_conversionᶜᶜᶠ(i, j, k, grid, s, cl, Kᶜ) =
    - @inbounds(Kᶜ[i, j, k]) * applied_stratificationᶜᶜᶠ(i, j, k, grid, s, cl)

# The moisture mass fractions of the current state at center k. The stage's partition with the vapor changed by the change
# of the prognostic moisture since the stage, qᵛᵉ - qᵛᵉₛ (qᵛᵉₛ the specific moisture the stage diagnosed): at the stage
# state the stage's own partition; in the state the implicit solves leave, the vapor and condensate that the solves move.
# With prognostic condensate (P3) the moisture density is the vapor and this is the solved vapor and the solved
# condensate. With saturation adjustment, whose moisture density is the total water and whose condensate the solves do
# not move, it is the stage's condensate (`distributed_condensateᶜᶜᶜ`) and the solved total water less it.
@inline function current_moisture_fractionsᶜᶜᶜ(i, j, k, grid, s)
    @inbounds begin
        ρ = s.ρ[i, j, k]
        qᵛᵉ = s.ρqᵛᵉ[i, j, k] / ρ
        qᵛᵉₛ = s.qᵛᵉ[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉₛ, s.microphysical_fields)
    return MoistureMassFractions(q.vapor + (qᵛᵉ - qᵛᵉₛ), q.liquid, q.ice)
end

# The stratification of the mixed variables, Σ b̄_χ δz χ: the buoyancy gradient a displacement
# without phase change feels (the stratification behind the applied buoyancy flux)
@inline function applied_stratificationᶜᶜᶠ(i, j, k, grid, s, cl)
    @inbounds begin
        ρ⁺ = s.ρ[i, j, k]
        ρ⁻ = s.ρ[i, j, k-1]
        θ⁺ = s.ρθ[i, j, k] / ρ⁺
        θ⁻ = s.ρθ[i, j, k-1] / ρ⁻
        q⁺ = current_moisture_fractionsᶜᶜᶜ(i, j, k, grid, s)
        q⁻ = current_moisture_fractionsᶜᶜᶜ(i, j, k-1, grid, s)
        Δb = (cl.b_θ[i, j, k] + cl.b_θ[i, j, k-1]) / 2 * (θ⁺ - θ⁻) +
             (cl.b_qᵛ[i, j, k] + cl.b_qᵛ[i, j, k-1]) / 2 * (q⁺.vapor - q⁻.vapor) +
             (cl.b_qˡ[i, j, k] + cl.b_qˡ[i, j, k-1]) / 2 * (q⁺.liquid - q⁻.liquid) +
             (cl.b_qⁱ[i, j, k] + cl.b_qⁱ[i, j, k-1]) / 2 * (q⁺.ice - q⁻.ice)
        return Δb / Δzᶜᶜᶠ(i, j, k, grid)
    end
end

# The weights that move a face quantity X to the center k so that Σ ρᶜ Δzᶜ Xᶜ = Σ ρᶠ Δzᶠ Xᶠ: the masses of the lower and
# upper half-cells with the density of their face, and the mass of the cell, Xᶜ = (wˡᵒʷ Xᶠₖ + wᵘᵖ Xᶠₖ₊₁) / ρΔz
@inline function half_cell_weights(i, j, k, grid, ρ)
    zᶜ = znode(i, j, k, grid, Center(), Center(), Center())
    wˡᵒʷ = ℑzᵃᵃᶠ(i, j, k, grid, ρ) * (zᶜ - znode(i, j, k, grid, Center(), Center(), Face()))
    wᵘᵖ = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ) * (znode(i, j, k + 1, grid, Center(), Center(), Face()) - zᶜ)
    ρΔz = @inbounds ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
    return wˡᵒʷ, wᵘᵖ, ρΔz
end

# The conversion at the center k of a budget flux from its conversions at the two faces of the cell, `lower` and `upper`
# (evaluated at the nearest interior face where the face is a boundary, which carries no flux), with the half-cell
# weights: the one form for the budget fluxes of the heating and of the total water
@inline function half_cell_conversionᶜᶜᶜ(i, j, k, grid, ρ, lower, upper)
    wˡᵒʷ, wᵘᵖ, ρΔz = half_cell_weights(i, j, k, grid, ρ)
    wˡᵒʷ = ifelse(k > 1, wˡᵒʷ, zero(wˡᵒʷ))
    wᵘᵖ = ifelse(k < size(grid, 3), wᵘᵖ, zero(wᵘᵖ))
    return (wˡᵒʷ * lower + wᵘᵖ * upper) / ρΔz
end

# The conversions of the interior fluxes at center k, from those at its two faces by half-cell
# weights (the boundary faces carry none: the closure's diffusivities vanish there)
@inline function interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    kˡᵒʷ = max(k, 2)
    kᵘᵖ = min(k + 1, Nz)
    Bˡᵒʷ = ifelse(k > 1, applied_buoyancy_conversionᶜᶜᶠ(i, j, kˡᵒʷ, grid, s, cl, Kᶜ), zero(FT))
    Bᵘᵖ = ifelse(k < Nz, applied_buoyancy_conversionᶜᶜᶠ(i, j, kᵘᵖ, grid, s, cl, Kᶜ), zero(FT))
    Lˡᵒʷ = ifelse(k > 1, latent_conversionᶜᶜᶠ(i, j, kˡᵒʷ, grid, cl, Kᶜ), zero(FT))
    Lᵘᵖ = ifelse(k < Nz, latent_conversionᶜᶜᶠ(i, j, kᵘᵖ, grid, cl, Kᶜ), zero(FT))
    Pˡᵒʷ = ifelse(k > 1, shear_productionᶜᶜᶠ(i, j, kˡᵒʷ, grid, Kᵘ, u, v), zero(FT))
    Pᵘᵖ = ifelse(k < Nz, shear_productionᶜᶜᶠ(i, j, kᵘᵖ, grid, Kᵘ, u, v), zero(FT))
    wˡᵒʷ, wᵘᵖ, ρΔz = half_cell_weights(i, j, k, grid, s.ρ)
    return (wˡᵒʷ * Bˡᵒʷ + wᵘᵖ * Bᵘᵖ) / ρΔz, (wˡᵒʷ * Pˡᵒʷ + wᵘᵖ * Pᵘᵖ) / ρΔz, (wˡᵒʷ * Lˡᵒʷ + wᵘᵖ * Lᵘᵖ) / ρΔz
end

@inline latent_conversionᶜᶜᶠ(i, j, k, grid, cl, Kᶜ) = @inbounds Kᶜ[i, j, k] * cl.ΔN²ᴸ[i, j, k]

# The conversions of the surface fluxes within the bottom half-cell: the work of the surface
# stress on the lowest level, -u₁ 𝒯ˣ - v₁ 𝒯ʸ (𝒯 the upward flux of ρu, ρv), and the dynamic
# enthalpy the surface fluxes of θˡⁱ and vapor remove, (z₁ - z_surface) Σ b_χ,₁ J_χ, per unit mass
# of the lowest cell. Condensate has no surface flux here.
@inline function surface_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, u, v, bottom_bcs, bc_args)
    FT = grid_float_type(grid)
    Jᶿ = bottom_boundary_flux(bottom_bcs.ρθ, i, j, grid, s.ρθ, bc_args...)
    Jᵛ = bottom_boundary_flux(bottom_bcs.ρqᵛᵉ, i, j, grid, s.ρqᵛᵉ, bc_args...)
    𝒯ˣ = bottom_boundary_flux(bottom_bcs.ρu, i, j, grid, s.ρu, bc_args...)
    𝒯ʸ = bottom_boundary_flux(bottom_bcs.ρv, i, j, grid, s.ρv, bc_args...)
    h = znode(i, j, 1, grid, Center(), Center(), Center()) - znode(i, j, 1, grid, Center(), Center(), Face())
    @inbounds begin
        ρΔz = s.ρ[i, j, 1] * Δzᶜᶜᶜ(i, j, 1, grid)
        Bˢ = h * (cl.b_θ[i, j, 1] * Jᶿ + cl.b_qᵛ[i, j, 1] * Jᵛ) / ρΔz
        Pˢ = -(u[i, j, 1] * 𝒯ˣ + v[i, j, 1] * 𝒯ʸ) / ρΔz
    end
    bottom = k == 1
    return ifelse(bottom, Bˢ, zero(FT)), ifelse(bottom, Pˢ, zero(FT))
end

@kernel function _compute_applied_flux_conversions!(cl, grid, s, Kᶜ, Kᵘ, u, v, e, bottom_bcs, bc_args)
    i, j, k = @index(Global, NTuple)
    B, P, Bᴸ = interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    Bˢ, Pˢ = surface_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, u, v, bottom_bcs, bc_args)
    # The conversions of the budget fluxes (zero without them); their latent parts are drawn from the internal energy with Bᴸ
    Gᑫ = store_heating_flux_conversions!(cl, i, j, k, grid, s.ρ)
    B = B + Bˢ + Bᴸ + Gᑫ
    @inbounds begin
        cl.B[i, j, k] = B
        cl.Bᴸ[i, j, k] = Bᴸ
        cl.P[i, j, k] = P + Pˢ
        cl.Bˢ[i, j, k] = Bˢ
        cl.Pˢ[i, j, k] = Pˢ
        cl.ωᴮ[i, j, k] = -min(0, B) / applied_flux_buoyancy_sink_tke(i, j, k, grid, e)
    end
end

# What the conversions read
function applied_flux_state(model)
    ρ = total_density(model.dynamics)
    return (ρ = ρ,
            ρθ = distribution_potential_temperature_density(model),
            ρqᵛᵉ = model.moisture_density,
            qᵛᵉ = AtmosphereModels.specific_prognostic_moisture(model),
            ρu = model.momentum.ρu,
            ρv = model.momentum.ρv,
            T = model.temperature,
            p = dynamics_pressure(model.dynamics),
            standard_pressure = standard_pressure(model.dynamics),
            exner_exponent = AtmosphereModels.liquid_ice_exner_exponent(model.formulation),
            dynamics = model.dynamics,
            formulation = model.formulation,
            microphysics = model.microphysics,
            microphysical_fields = model.microphysical_fields,
            constants = model.thermodynamic_constants,
            phase_equilibrium = condensate_phase_equilibrium(model.microphysics))
end

function applied_flux_bottom_bcs(model)
    ρθ = distribution_potential_temperature_density(model)
    return (ρθ = ρθ.boundary_conditions.bottom,
            ρqᵛᵉ = model.moisture_density.boundary_conditions.bottom,
            ρu = model.momentum.ρu.boundary_conditions.bottom,
            ρv = model.momentum.ρv.boundary_conditions.bottom)
end

compute_applied_flux_conversions!(fields, closure_fields, closure, model, tracers) = nothing

# Within `update_state!`, where the tracers hold their specific values
function compute_applied_flux_conversions!(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, closure, model, tracers)
    grid = model.grid
    arch = grid.architecture
    cl = fields.cloud
    s = applied_flux_state(model)
    u, v = model.velocities.u, model.velocities.v
    launch!(arch, grid, :xyz, _compute_applied_flux_conversions!, cl, grid, s, closure_fields.Kᶜ,
            closure_fields.Kᵘ, u, v, tracers[TKE_NAME], applied_flux_bottom_bcs(model), boundary_condition_args(model))
    return nothing
end

# The shear production of the TKE equation: CATKE's without the energetics of the applied
# fluxes, the conversion of the applied momentum fluxes with them
@inline tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields, closure_fields, u, v) =
    shear_production(i, j, k, grid, closure_fields.Kᵘ, u, u, v, v)

@inline tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, u, v) =
    @inbounds fields.cloud.P[i, j, k]

# The shear production in the stage tendency: all of it, or, where the TKE receives the shear conversion of the solved
# momentum before its own solve (`solved_conversions_before_tke_solve`), only the work of the surface stress Pˢ
@inline explicit_tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields, closure_fields, u, v, injected) =
    tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields, closure_fields, u, v)

@inline explicit_tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, u, v, injected) =
    explicit_shear_conversion(fields.cloud, i, j, k, injected)

@inline explicit_shear_conversion(cl, i, j, k, injected) = @inbounds ifelse(injected, cl.Pˢ[i, j, k], cl.P[i, j, k])

@inline explicit_tke_buoyancy_production(fields, i, j, k, B, injected) = max(zero(B), B)
@inline explicit_tke_buoyancy_production(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, i, j, k, B, injected) =
    explicit_buoyancy_conversion(fields.cloud, i, j, k, B, injected)

#####
##### The conversions of the solved state, given to the TKE before its solve
#####
##### Under the vertically implicit discretization with a time stepper that solves momentum, the tracers and the
##### thermodynamic variable before the TKE (`AtmosphereModels.closure_prognostics_solved_last`), the TKE receives,
##### just before its own solve, what the stage's other solves converted:
#####
#####   shear:     Pⁱ = Pᵖ + Dᵘ, the conversion of the solved momentum fluxes and the dissipation of backward Euler
#####              itself, the fraction ℂᴾ to the TKE (the rest heats the air after the solves);
#####   buoyancy:  Bⁱ = max(Bᵖ + Bᴸⁱ, 0), the conversion of the solved scalar fluxes, Bᵖ = Σ b_χ F_χ, and its latent
#####              part from the solved gradients, Bᴸⁱ = Kᶜ η bˡᵛ (∂z qᶜ - aʷ ∂z s̄) (drawn from the internal energy
#####              after the solves); its negative part stays with the implicit sink and the correction after the solves.
#####
##### Its solve then diffuses the energy the solves released. Formed instead from the stage state, Kᵘ S² and Kᶜ ΔN²ᴸ
##### overshoot where the solves mix the gradients away within the stage (K Δt / Δz² ≫ 1): the TKE solve diffused that
##### overshoot into the neighbouring cells before the correction after the solves took it back from the cell alone,
##### and the internal energy paid the rest, a pump of TKE at the scale of the grid. The heating of the dissipation,
##### of the share (1 - ℂᴾ) of Pⁱ, and the latent draw are applied to θˡⁱ after the solves, with the solved TKE.
#####

solved_conversions_before_tke_solve(fields, closure, model) = false
solved_conversions_before_tke_solve(::SubgridCloudFields{<:AppliedFluxEnergetics}, closure, model) =
    implicit_stage(closure, model) && AtmosphereModels.closure_prognostics_solved_last(model)

# The TKE is solved after the thermodynamic variable where the time stepper allows (`closure_prognostics_solved_last`)
AtmosphereModels.solved_after_thermodynamic_variable(::FlavorOfTKEClosure, name) = name === TKE_NAME

# The dissipation of backward Euler itself in the momentum solve of the stage, |u - u⁻|² / (2 Δt), at the cell k:
# the squared increments of the horizontal momentum interpolated from their faces to the center, as the kinetic
# energy of the stage combination is (`_stage_combination_kinetic_energy!`)
@inline function squared_increment(i, j, k, grid, ρu, ρu⁻)
    @inbounds δ = ρu[i, j, k] - ρu⁻[i, j, k]
    return δ * δ
end

@inline function implicit_momentum_dissipationᶜᶜᶜ(i, j, k, grid, s, cl, Δt)
    δ² = ℑxᶜᵃᵃ(i, j, k, grid, squared_increment, s.ρu, cl.ρu⁻) + ℑyᵃᶜᵃ(i, j, k, grid, squared_increment, s.ρv, cl.ρv⁻)
    ρᵢ = @inbounds s.ρ[i, j, k]
    return δ² / (ρᵢ * ρᵢ) / (2 * Δt)
end

# b̄ᶜ - b̄ᵛ, b̄ⁱ - b̄ᵛ and b̄ˡ - b̄ᵛ at the interior face k: the buoyancy of vapor that becomes distributed condensate, ice
# and liquid, from the derivatives stored at the two centers
@inline function latent_buoyancy_derivativesᶜᶜᶠ(i, j, k, cl)
    @inbounds begin
        bᶜᵛ = (cl.b_qᶜ[i, j, k] + cl.b_qᶜ[i, j, k-1] - cl.b_qᵛ[i, j, k] - cl.b_qᵛ[i, j, k-1]) / 2
        bⁱᵛ = (cl.b_qⁱ[i, j, k] + cl.b_qⁱ[i, j, k-1] - cl.b_qᵛ[i, j, k] - cl.b_qᵛ[i, j, k-1]) / 2
        bˡᵛ = (cl.b_qˡ[i, j, k] + cl.b_qˡ[i, j, k-1] - cl.b_qᵛ[i, j, k] - cl.b_qᵛ[i, j, k-1]) / 2
    end
    return bᶜᵛ, bⁱᵛ, bˡᵛ
end

# The latent part of the buoyancy conversion at the interior face k from the solved state: Kᶜ ΔN²ᴸ with the gradients
# of the solved θʷ, distributed water and condensate and the stage's shares η, aᴺ = η aʷ, ηⁱ and coefficients c_q, c_θ,
# c_qⁱ, c_θⁱ (see `_latent_corrected_stability!`)
@inline function solved_latent_conversionᶜᶜᶠ(i, j, k, grid, s, cl, Kᶜ, qʷ, θʷ)
    @inbounds begin
        θ⁺ = θʷ[i, j, k]
        θ⁻ = θʷ[i, j, k-1]
        qʷ⁺ = qʷ[i, j, k]
        qʷ⁻ = qʷ[i, j, k-1]
        qᶜ⁺ = distributed_condensateᶜᶜᶜ(i, j, k, grid, s)
        qᶜ⁻ = distributed_condensateᶜᶜᶜ(i, j, k-1, grid, s)
        Δz = Δzᶜᶜᶠ(i, j, k, grid)
        ∂z_s = (cl.c_q[i, j, k] * (qʷ⁺ - qʷ⁻) - cl.c_θ[i, j, k] * (θ⁺ - θ⁻)) / Δz
        ∂z_sⁱ = (cl.c_qⁱ[i, j, k] * (qʷ⁺ - qʷ⁻) - cl.c_θⁱ[i, j, k] * (θ⁺ - θ⁻)) / Δz
        ∂z_qᶜ = (qᶜ⁺ - qᶜ⁻) / Δz
        bᶜᵛ, bⁱᵛ, bˡᵛ = latent_buoyancy_derivativesᶜᶜᶠ(i, j, k, cl)
        condensate = bᶜᵛ * (cl.η[i, j, k] * ∂z_qᶜ - cl.aᴺ[i, j, k] * ∂z_s)
        outside = - bⁱᵛ * cl.ηⁱ[i, j, k] * ∂z_sⁱ - bˡᵛ * cl.ηʷⁱ[i, j, k] * ∂z_s
        return Kᶜ[i, j, k] * (condensate + outside)
    end
end

# At the center k, from its two faces with the half-cell weights of `interior_conversionsᶜᶜᶜ`
@inline function solved_latent_conversionᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, qʷ, θʷ)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    Lˡᵒʷ = ifelse(k > 1, solved_latent_conversionᶜᶜᶠ(i, j, max(k, 2), grid, s, cl, Kᶜ, qʷ, θʷ), zero(FT))
    Lᵘᵖ = ifelse(k < Nz, solved_latent_conversionᶜᶜᶠ(i, j, min(k + 1, Nz), grid, s, cl, Kᶜ, qʷ, θʷ), zero(FT))
    wˡᵒʷ, wᵘᵖ, ρΔz = half_cell_weights(i, j, k, grid, s.ρ)
    return (wˡᵒʷ * Lˡᵒʷ + wᵘᵖ * Lᵘᵖ) / ρΔz
end

@kernel function _give_solved_conversions!(ρe, cl, grid, closure, s, Kᶜ, Kᵘ, u, v)
    i, j, k = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    θʷ, qʷ, _ = distributed_water(s, grid)
    Bᵖ, Pᵖ, _ = interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    Dᵘ = implicit_momentum_dissipationᶜᶜᶜ(i, j, k, grid, s, cl, @inbounds(cl.stage_time_step[1, 1, 1]))
    Bᴸⁱ = solved_latent_conversionᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, qʷ, θʷ)
    Pⁱ = Pᵖ + Dᵘ
    Bⁱ = max(Bᵖ + Bᴸⁱ, zero(Bᵖ))
    ℂᴾ = shear_production_fraction(closure_ij.stability_functions)
    @inbounds begin
        cl.Pⁱ[i, j, k] = Pⁱ
        cl.Bⁱ[i, j, k] = Bⁱ
        cl.Bᴸⁱ[i, j, k] = Bᴸⁱ
        ρe[i, j, k] += cl.stage_time_step[1, 1, 1] * s.ρ[i, j, k] * (ℂᴾ * Pⁱ + Bⁱ) * !inactive_cell(i, j, k, grid)
    end
end

function give_solved_conversions!(closure_fields, closure, model)
    grid = model.grid
    cl = closure_fields.subgrid_cloud.cloud
    s = applied_flux_state(model)
    fill_halo_regions!((model.momentum.ρu, model.momentum.ρv))
    u = XFaceVelocity(model.momentum.ρu, s.ρ, grid)
    v = YFaceVelocity(model.momentum.ρv, s.ρ, grid)
    launch!(grid.architecture, grid, :xyz, _give_solved_conversions!, model.tracers[TKE_NAME], cl, grid, closure, s,
            closure_fields.Kᶜ, closure_fields.Kᵘ, u, v)
    return nothing
end

# The buoyancy production in the stage tendency: the positive part of all of it, or, where the conversions of the solved
# state are given before the TKE solve, of the stage-state conversions of the surface fluxes and of the heating budget flux
@inline explicit_buoyancy_conversion(cl, i, j, k, B, injected) =
    @inbounds ifelse(injected, max(cl.Bˢ[i, j, k] + heating_flux_conversion(cl, i, j, k), zero(B)), max(B, zero(B)))

# The stage-state conversions the correction after the solves counts as converted: with the conversions of the solved
# state given before the TKE solve, the latent part is that of the solved state, not of the stage
@inline stage_state_conversion(cl, i, j, k, injected) =
    @inbounds cl.Pˢ[i, j, k] + cl.Bˢ[i, j, k] + heating_flux_conversion(cl, i, j, k) + ifelse(injected, zero(cl.Bᴸ[i, j, k]), cl.Bᴸ[i, j, k])

#####
##### The floor of e in the buoyancy sink rate
#####

# The implicit buoyancy sink is ωᴮ e with ωᴮ = -B⁻ / max(e, floor). Above the floor it removes
# exactly -B⁻, the energy the applied fluxes give the mean state. The plain closure floors at
# `minimum_tke`, so below it the sink removes less than the mean state gains. With the cloud
# buoyancy flux the floor is a tiny positive number instead, kept only so that ωᴮ is finite at
# e = 0 (its benefit: the implicit sink cannot divide by zero, and e stays non-negative in the
# implicit solve); the conversion is exact wherever e exceeds it. The floor is taken in the float
# type `FT` of the grid, where ωᴮ is stored: a floor in the type of closure parameters wider than the
# grid's (Float64 parameters on a Float32 grid) would underflow there and make ωᴮ infinite.
@inline buoyancy_sink_floor(fields, closure, FT) = closure.minimum_tke
@inline buoyancy_sink_floor(::SubgridCloudFields{<:AppliedFluxEnergetics}, closure, FT) = buoyancy_sink_floor_value(FT)
@inline buoyancy_sink_floor_value(FT) = sqrt(floatmin(FT))

# The TKE that the buoyancy sink rate ωᴮ = -B⁻ / e divides by. The plain closure: e at the center, floored at
# `minimum_tke`. With the conversions of the applied fluxes, B at the center is the half-cell weighted sum of the
# conversions at its two faces, each proportional to the diffusivity there and so to the TKE there; the sink rate
# divides by the TKE interpolated to those faces with the same weights, so that it is a ratio of collocated
# quantities: bounded by the diffusivity per unit TKE of the faces, whatever e at the center. (Dividing by e at the
# center, a cell whose own e vanishes while a turbulent neighbour's face converts makes ωᴮ unbounded: its implicit
# sink then absorbs the TKE that diffuses in during the solve, and the correction after the solves, which replaces
# the implicit sink by the solved conversion, returns all of it to that cell.)
@inline buoyancy_sink_tke(i, j, k, grid, fields, closure, e) = @inbounds max(e[i, j, k], buoyancy_sink_floor(fields, closure, grid_float_type(grid)))

@inline buoyancy_sink_tke(i, j, k, grid, ::SubgridCloudFields{<:AppliedFluxEnergetics}, closure, e) =
    applied_flux_buoyancy_sink_tke(i, j, k, grid, e)

@inline positive_tkeᶜᶜᶜ(i, j, k, grid, e) = @inbounds max(e[i, j, k], zero(eltype(e)))

@inline function applied_flux_buoyancy_sink_tke(i, j, k, grid, e)
    FT = grid_float_type(grid)
    Nz = size(grid, 3)
    zᶜ = znode(i, j, k, grid, Center(), Center(), Center())
    wˡᵒʷ = ifelse(k > 1, zᶜ - znode(i, j, k, grid, Center(), Center(), Face()), zero(FT))
    wᵘᵖ = ifelse(k < Nz, znode(i, j, k + 1, grid, Center(), Center(), Face()) - zᶜ, zero(FT))
    # the boundary faces carry no conversion, and their halo values are not read
    eˡᵒʷ = ifelse(k > 1, ℑzᵃᵃᶠ(i, j, max(k, 2), grid, positive_tkeᶜᶜᶜ, e), zero(FT))
    eᵘᵖ = ifelse(k < Nz, ℑzᵃᵃᶠ(i, j, min(k + 1, Nz), grid, positive_tkeᶜᶜᶜ, e), zero(FT))
    ē = (wˡᵒʷ * eˡᵒʷ + wᵘᵖ * eᵘᵖ) / (wˡᵒʷ + wᵘᵖ)
    return max(ē, buoyancy_sink_floor_value(FT))
end

#####
##### Dissipative heating and the conversions of the solved state
#####
##### Under the vertically implicit time discretization each Runge–Kutta stage solves
##### (1 - Δt ∂z K ∂z) φ = φ⁻ for momentum and every scalar (Δt the stage's α Δt), and the TKE
##### equation solves its sinks ωᴰ e + ωᴮ e (dissipation, damping of negative e, buoyancy sink)
##### implicitly too. The energy the solves take from the mean state is that of the fluxes of the
##### SOLVED state:
#####
#####   momentum:  Σ ρ Δz (|u|² - |u⁻|²)/2 = -Δt Σ ρᶠ Δzᶠ Kᵘ |δz u|² / Δzᶠ² - Σ ρ Δz |u - u⁻|² / 2
#####              (the last term is the dissipation of backward Euler itself),
#####   scalars:   Σ ρ Δz Δh‡ = -Δt Σ ρᶠ Δzᶠ Bᶠ(φ),  h‡ linearized with the stage's b_χ,
#####
##### while the stage tendency gave the TKE the conversions of the stage state. After the solves the
##### closure adds the difference to ρe, so that over the stage the TKE gains exactly what the mean
##### state loses; and before the thermodynamic solve it heats the air by what the implicit sinks
##### removed from the solved e, ρ ĉᵖ Π Δθˡⁱ = Δt ρ ωᴰ e (ωᴰ = -Lᵉ - ωᴮ).
#####

const ImplicitAppliedFluxFields = SubgridCloudFields{<:AppliedFluxEnergetics}

implicit_stage(closure, model) =
    time_discretization(closure) isa VerticallyImplicitTimeDiscretization && !isnothing(model.timestepper.implicit_solver)

implicit_stage(closure::TKEClosureArray, model) = implicit_stage(first(closure), model)

# The kinetic energy the convex combination of a Runge–Kutta stage removes from the mean flow. The stage
# forms u = (1 - α) u⁰ + α V, V = u⁽ᵐ⁻¹⁾ + Δt G, and per unit volume
#
#     (1 - α) ρ |u⁰|² / 2 + α ρ |V|² / 2 - ρ |(1 - α) u⁰ + α V|² / 2 = α (1 - α) ρ |V - u⁰|² / 2 ≥ 0,
#
# which reaches no budget unless it is given to the TKE: with it the TKE receives over a whole step exactly
# the kinetic energy the mean flow loses. Of the horizontal momentum, its squares interpolated to the centers.
@inline function squared_combination_increment(i, j, k, grid, ρu, Gρu, ρu⁰, Δt)
    @inbounds δ = ρu[i, j, k] + Δt * Gρu[i, j, k] - ρu⁰[i, j, k]
    return δ * δ
end

@kernel function _stage_combination_kinetic_energy!(K, grid, ρu, ρv, Gρu, Gρv, ρu⁰, ρv⁰, ρ, Δt, α)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        δ² = ℑxᶜᵃᵃ(i, j, k, grid, squared_combination_increment, ρu, Gρu, ρu⁰, Δt) +
             ℑyᵃᶜᵃ(i, j, k, grid, squared_combination_increment, ρv, Gρv, ρv⁰, Δt)
        K[i, j, k] = α * (1 - α) * δ² / (2 * ρ[i, j, k]) * !inactive_cell(i, j, k, grid)
    end
end

function AtmosphereModels.closure_before_stage_combination!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitAppliedFluxFields},
                                                             closure::FlavorOfTKEClosure, model, Δt, α)
    implicit_stage(closure, model) || return nothing
    fill!(closure_fields.subgrid_cloud.cloud.stage_time_step, α * Δt)
    grid = model.grid
    U⁰ = model.timestepper.U⁰
    Gⁿ = model.timestepper.Gⁿ
    ρu, ρv = model.momentum.ρu, model.momentum.ρv
    fill_halo_regions!((ρu, ρv, Gⁿ.ρu, Gⁿ.ρv, U⁰.ρu, U⁰.ρv))
    launch!(grid.architecture, grid, :xyz, _stage_combination_kinetic_energy!, closure_fields.subgrid_cloud.cloud.stage_combination,
            grid, ρu, ρv, Gⁿ.ρu, Gⁿ.ρv, U⁰.ρu, U⁰.ρv, total_density(model.dynamics), Δt, α)
    return nothing
end

#####
##### The stages of a multirate infinitesimal step
#####
##### Stage i recombines the states and the slow tendencies of the stages before it (see
##### `AtmosphereModels.StageRecombination`): the state its solves start from is
#####
#####     Vᵢ = Σⱼ wᵢⱼ Yⱼ + dᵢ Δt Σⱼ bᵢⱼ Sⱼ,     Σⱼ wᵢⱼ = 1,   Σⱼ bᵢⱼ = 1,   Sⱼ = S(Yⱼ₋₁),
#####
##### and the solves over dᵢ Δt give Yᵢ. Two things differ from a Shu–Osher stage.
#####
##### The kinetic energy the recombination of the states removes from the mean flow, per unit volume,
#####
#####     Σⱼ wᵢⱼ ρ |uⱼ|² / 2 - ρ |Σⱼ wᵢⱼ uⱼ|² / 2 = Σ_{p<q} wᵢₚ wᵢq ρ |uₚ - u_q|² / 2,
#####
##### is given to the TKE (`stage_combination`). The identity holds for any weights that sum to
##### one. It is the Shu–Osher α (1 - α) ρ |V - u⁰|² / 2 for the weights (1 - α, α). The weights of
##### a multirate infinitesimal step need not be positive (stage 2 of `:SSPMIS43` extrapolates,
##### w = (-0.05, 1.05)), and then the recombination adds kinetic energy to the mean flow, which
##### the TKE pays (or, where it has none, the internal energy: the shortfall).
#####
##### The stage-state conversions are in the slow tendencies, so the stage reapplies those of the
##### stages before it: over the stage the TKE tendency gives Δt Σⱼ βᵢⱼ ρ (Pⱼ + B⁺ⱼ) and the
##### tendency of θˡⁱ draws Δt Σⱼ βᵢⱼ ρ Bᴸⱼ. The correction after the solves therefore uses
#####
#####     Xᵢ = Σⱼ bᵢⱼ xⱼ,     xⱼ = (Pˢ + Bˢ + Bᴸ + Gᑫ - P - B⁺)ⱼ    (Gᑫ of a `HeatingBudgetFlux`),
#####
##### the recombined difference between what the stage-state fluxes converted and what the TKE
##### tendency gave, in place of the stage's own xᵢ: ρe gains dᵢ Δt ρ (Pᵖ + Dᵘ + Bᵖ + ωᴮ e + Xᵢ)
##### and the kinetic energy of the recombination. Over the stage the TKE then receives exactly
##### what the solves converted, what the recombined tendencies of the surface fluxes and the
##### latent conversion carry, and what the recombination removed from the mean flow.
#####

function AtmosphereModels.build_closure_stage_storage(::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitAppliedFluxFields},
                                                      closure, grid, stages)
    exchange = ntuple(_ -> CenterField(grid), stages)
    momentum = ntuple(_ -> (ρu = XFaceField(grid), ρv = YFaceField(grid)), stages - 1)
    return (; exchange, recombined_exchange = CenterField(grid), momentum)
end

@kernel function _store_stage_exchange!(x, cl, injected)
    i, j, k = @index(Global, NTuple)
    # the conversion of the heating budget flux is a stage-state conversion too: its flux is in the slow tendency of ρθ
    @inbounds x[i, j, k] = stage_state_conversion(cl, i, j, k, injected) -
                           explicit_shear_conversion(cl, i, j, k, injected) -
                           explicit_buoyancy_conversion(cl, i, j, k, cl.B[i, j, k], injected)
end

@inline weighted_sum(i, j, k, x::Tuple{Any}, b::Tuple{Any}) = @inbounds b[1] * x[1][i, j, k]
@inline weighted_sum(i, j, k, x::Tuple, b::Tuple) =
    @inbounds b[1] * x[1][i, j, k] + weighted_sum(i, j, k, Base.tail(x), Base.tail(b))

@kernel function _recombine_stage_exchange!(X, x, b)
    i, j, k = @index(Global, NTuple)
    @inbounds X[i, j, k] = weighted_sum(i, j, k, x, b)
end

# Σ_{q} w_q |u - u_q|² over the states `U` and Σ_{p<q} w_p w_q |u_p - u_q|², at the location of the momentum
@inline weighted_squared_differences(i, j, k, u, ::Tuple{}, ::Tuple{}) = zero(eltype(u))
@inline function weighted_squared_differences(i, j, k, u, U::Tuple, w::Tuple)
    @inbounds δ = u[i, j, k] - U[1][i, j, k]
    return w[1] * δ * δ + weighted_squared_differences(i, j, k, u, Base.tail(U), Base.tail(w))
end

@inline pairwise_squared_differences(i, j, k, grid, U::Tuple{Any}, w::Tuple{Any}) = zero(grid)
@inline pairwise_squared_differences(i, j, k, grid, U::Tuple, w::Tuple) =
    w[1] * weighted_squared_differences(i, j, k, U[1], Base.tail(U), Base.tail(w)) +
    pairwise_squared_differences(i, j, k, grid, Base.tail(U), Base.tail(w))

@kernel function _stage_recombination_kinetic_energy!(K, grid, ρu, ρv, w, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        δ² = ℑxᶜᵃᵃ(i, j, k, grid, pairwise_squared_differences, ρu, w) +
             ℑyᵃᶜᵃ(i, j, k, grid, pairwise_squared_differences, ρv, w)
        K[i, j, k] = δ² / (2 * ρ[i, j, k]) * !inactive_cell(i, j, k, grid)
    end
end

function AtmosphereModels.closure_before_stage_combination!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitAppliedFluxFields},
                                                             closure::FlavorOfTKEClosure, model, Δt,
                                                             recombination::AtmosphereModels.StageRecombination)
    implicit_stage(closure, model) || return nothing
    fill!(closure_fields.subgrid_cloud.cloud.stage_time_step, Δt)
    grid = model.grid
    arch = grid.architecture
    cl = closure_fields.subgrid_cloud.cloud
    storage = recombination.storage
    i = recombination.stage

    # The conversions of the stage state Yᵢ₋₁ are in hand: keep their exchange, and recombine
    launch!(arch, grid, :xyz, _store_stage_exchange!, storage.exchange[i], cl, solved_conversions_before_tke_solve(closure_fields.subgrid_cloud, closure, model))
    launch!(arch, grid, :xyz, _recombine_stage_exchange!, storage.recombined_exchange,
            storage.exchange[1:i], recombination.tendency_weights)

    # The momentum is that of Yᵢ₋₁: keep it for the stages after this one
    ρu, ρv = model.momentum.ρu, model.momentum.ρv
    ρuⁿ, ρvⁿ = recombination.step_start_state.ρu, recombination.step_start_state.ρv
    fill_halo_regions!((ρu, ρv, ρuⁿ, ρvⁿ))
    if i > 1
        parent(storage.momentum[i-1].ρu) .= parent(ρu)
        parent(storage.momentum[i-1].ρv) .= parent(ρv)
    end
    ρus = (ρuⁿ, ntuple(j -> storage.momentum[j].ρu, i - 1)...)
    ρvs = (ρvⁿ, ntuple(j -> storage.momentum[j].ρv, i - 1)...)
    launch!(arch, grid, :xyz, _stage_recombination_kinetic_energy!, cl.stage_combination, grid, ρus, ρvs,
            recombination.state_weights, total_density(model.dynamics))
    return nothing
end

# What the correction after the solves adds to the stage tendency's conversions, per unit mass and
# time: with no recombination, what the solves and the stage-state fluxes converted less what the
# stage gave; with one, the solves' part and the recombined exchange of the stage states.
@inline net_stage_conversion(::Nothing, i, j, k, converted, given, solved) = converted - given
@inline net_stage_conversion(X, i, j, k, converted, given, solved) = @inbounds solved + X[i, j, k]

recombined_stage_exchange(::Nothing) = nothing
recombined_stage_exchange(storage) = storage.recombined_exchange

function AtmosphereModels.closure_before_implicit_solve!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitAppliedFluxFields},
                                                          closure::FlavorOfTKEClosure, model, name)
    implicit_stage(closure, model) || return nothing
    cl = closure_fields.subgrid_cloud.cloud
    # The halos too: the increment of the solve is interpolated to the centers (`implicit_momentum_dissipationᶜᶜᶜ`)
    name === :ρu && (interior(cl.ρu⁻) .= interior(model.momentum.ρu); fill_halo_regions!(cl.ρu⁻))
    name === :ρv && (interior(cl.ρv⁻) .= interior(model.momentum.ρv); fill_halo_regions!(cl.ρv⁻))
    pre = closure_fields.subgrid_cloud.presolve
    name ∈ keys(pre) && (parent(pre[name]) .= parent(Oceananigans.prognostic_fields(model)[name]))
    # Momentum is solved: the TKE receives what that solve converted before its own solve
    name === TKE_NAME && solved_conversions_before_tke_solve(closure_fields.subgrid_cloud, closure, model) &&
        give_solved_conversions!(closure_fields, closure, model)
    return nothing
end

@kernel function _heat_by_implicit_dissipation!(ρθ, cl, grid, Lᵉ, ρe, ρ, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵢ = ρ[i, j, k]
        e = ρe[i, j, k] / ρᵢ
        ε = (-Lᵉ[i, j, k] - cl.ωᴮ[i, j, k]) * e
        cl.ε[i, j, k] = ε
        ρθ[i, j, k] += Δt * ρᵢ * ε / cl.ĉᵖΠ[i, j, k] * !inactive_cell(i, j, k, grid)
    end
end

function AtmosphereModels.closure_before_thermodynamic_solve!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitAppliedFluxFields},
                                                               closure::FlavorOfTKEClosure, model, Δt)
    implicit_stage(closure, model) || return nothing
    fields = closure_fields.subgrid_cloud
    grid = model.grid
    ρθ = distribution_potential_temperature_density(model)
    # Without the conversions of the solved state given before the TKE solve, the TKE is solved before θˡⁱ: the
    # dissipation its implicit sinks removed heats the air before the solve of θˡⁱ (with them, after the solves)
    solved_conversions_before_tke_solve(fields, closure, model) ||
        launch!(grid.architecture, grid, :xyz, _heat_by_implicit_dissipation!, ρθ,
                fields.cloud, grid, closure_fields.Lᵉ, model.tracers[TKE_NAME], total_density(model.dynamics), Δt)
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    name ∈ keys(fields.presolve) && (parent(fields.presolve[name]) .= parent(ρθ))
    return nothing
end

@kernel function _correct_tke_to_solved_conversions!(ρe, cl, grid, closure, s, Kᶜ, Kᵘ, u, v, Lᵉ, Δt, X, injected)
    i, j, k = @index(Global, NTuple)
    Bᵖ, Pᵖ, _ = interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    @inbounds begin
        ρᵢ = s.ρ[i, j, k]
        e = ρe[i, j, k] / ρᵢ
        Dᵘ = implicit_momentum_dissipationᶜᶜᶜ(i, j, k, grid, s, cl, Δt)
        B = cl.B[i, j, k]
        # what the stage gave: P and B⁺ explicitly, -ωᴮ e implicitly; what the solves converted:
        # Pᵖ + Dᵘ + Pˢ and Bᵖ + Bˢ
        # with the solved shear conversion given before the solve, it is given (Pⁱ, equal to Pᵖ + Dᵘ here)
        # and with the solved conversions given before the TKE solve, they are given too (Pⁱ and Bⁱ)
        Pⁱ = ifelse(injected, cl.Pⁱ[i, j, k], zero(e))
        Bⁱ = ifelse(injected, cl.Bⁱ[i, j, k], zero(e))
        Bᴸⁱ = ifelse(injected, cl.Bᴸⁱ[i, j, k], zero(e))
        Pᵉ = explicit_shear_conversion(cl, i, j, k, injected)
        Bᵉ = explicit_buoyancy_conversion(cl, i, j, k, B, injected)
        given = Pᵉ + Pⁱ + Bᵉ + Bⁱ - cl.ωᴮ[i, j, k] * e
        # the latent part of the stage state was drawn from the internal energy in the stage tendency, that of the
        # solved state after the solves (as is the conversion of the heating budget flux, an explicit flux of the stage)
        stage_state = stage_state_conversion(cl, i, j, k, injected)
        converted = Pᵖ + Dᵘ + Bᵖ + Bᴸⁱ + stage_state
        # the dissipation the implicit sinks removed from the solved e, which heats the air
        ε = (-Lᵉ[i, j, k] - cl.ωᴮ[i, j, k]) * e
        cl.Pᵖ[i, j, k] = Pᵖ + Dᵘ
        cl.Bᵖ[i, j, k] = Bᵖ
        # the stage's values, kept for the energy budget after `update_state!` recomputes the others
        # what reached the TKE and the air other than through the TKE's solve and this correction: the stage tendency,
        # and the share (1 - ℂᴾ) of the shear conversion given before the solve, which heated the air
        ℂᴾ = shear_production_fraction(getclosure(i, j, closure).stability_functions)
        cl.given[i, j, k] = Pᵉ + Bᵉ + (1 - ℂᴾ) * Pⁱ
        cl.stage_b_θ[i, j, k] = cl.b_θ[i, j, k]
        cl.stage_b_qᵛ[i, j, k] = cl.b_qᵛ[i, j, k]
        cl.stage_b_qˡ[i, j, k] = cl.b_qˡ[i, j, k]
        cl.stage_b_qⁱ[i, j, k] = cl.b_qⁱ[i, j, k]
        cl.stage_surface[i, j, k] = stage_state
        cl.Bᴸⁱ[i, j, k] = Bᴸⁱ
        # Where the TKE cannot pay what the solved fluxes converted (e ≈ 0 while the diffusivities,
        # floored at `minimum_tke`, still mix), the shortfall is taken from the internal energy
        # instead of driving e negative: the mixing of that floor is powered by the heat of the air.
        # Benefit: e stays non-negative and the total energy still closes. The TKE also receives the
        # kinetic energy the stage combination removed from the mean flow (`stage_combination`, per volume).
        active = !inactive_cell(i, j, k, grid)
        net = net_stage_conversion(X, i, j, k, converted, given, Pᵖ + Dᵘ - Pⁱ + Bᵖ + Bᴸⁱ - Bⁱ + cl.ωᴮ[i, j, k] * e)
        ρe⁺ = ρe[i, j, k] + (Δt * ρᵢ * net + cl.stage_combination[i, j, k]) * active
        ρe⁺′ = ifelse(ρe⁺ < 0, min(ρe[i, j, k], zero(ρe⁺)), ρe⁺)
        shortfall = ρe⁺′ - ρe⁺ # ≥ 0: energy the TKE did not pay
        ρe[i, j, k] = ρe⁺′
        cl.shortfall[i, j, k] = shortfall
        # What θˡⁱ receives after the solves: the internal energy pays the shortfall, and, with the solved conversions
        # given before the TKE solve, receives the dissipation, the share (1 - ℂᴾ) of Pⁱ and gives the latent part
        heating = ifelse(injected, Δt * ρᵢ * (ε + (1 - ℂᴾ) * Pⁱ - Bᴸⁱ), zero(e))
        cl.ε[i, j, k] = ifelse(injected, ε, cl.ε[i, j, k])
        cl.after_solves_ρθ[i, j, k] = (heating - shortfall) / cl.ĉᵖΠ[i, j, k] * active
    end
end

# The cooling by the shortfall, after all the conversions of the solved state are formed (they read
# θ at the neighbouring cells, so it cannot be applied in the same kernel)
@kernel function _apply_after_solves_heating!(ρθ, cl)
    i, j, k = @index(Global, NTuple)
    @inbounds ρθ[i, j, k] += cl.after_solves_ρθ[i, j, k]
end

function AtmosphereModels.closure_after_implicit_solves!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitAppliedFluxFields},
                                                          closure::FlavorOfTKEClosure, model, Δt)
    implicit_stage(closure, model) || return nothing
    grid = model.grid
    cl = closure_fields.subgrid_cloud.cloud
    s = applied_flux_state(model)
    ρ = s.ρ
    fill_halo_regions!((model.momentum.ρu, model.momentum.ρv))
    u = XFaceVelocity(model.momentum.ρu, ρ, grid)
    v = YFaceVelocity(model.momentum.ρv, ρ, grid)
    launch!(grid.architecture, grid, :xyz, _correct_tke_to_solved_conversions!, model.tracers[TKE_NAME], cl, grid, closure, s,
            closure_fields.Kᶜ, closure_fields.Kᵘ, u, v, closure_fields.Lᵉ, Δt,
            recombined_stage_exchange(AtmosphereModels.closure_stage_storage(model.timestepper)),
            solved_conversions_before_tke_solve(closure_fields.subgrid_cloud, closure, model))
    launch!(grid.architecture, grid, :xyz, _apply_after_solves_heating!, s.ρθ, cl)
    return nothing
end

# Under the explicit time discretization the sinks are in the stage tendency, and so is the heating
@kernel function _add_dissipative_heating!(Gρθ, grid, closure_fields, ρe, ρ)
    i, j, k = @index(Global, NTuple)
    cl = closure_fields.subgrid_cloud.cloud
    @inbounds begin
        ρᵢ = ρ[i, j, k]
        e = ρe[i, j, k] / ρᵢ
        ε = (-closure_fields.Lᵉ[i, j, k] - cl.ωᴮ[i, j, k]) * e
        Gρθ[i, j, k] += ρᵢ * ε / cl.ĉᵖΠ[i, j, k] * !inactive_cell(i, j, k, grid)
    end
end

add_dissipative_heating!(Gⁿ, fields, closure_fields, closure, model) = nothing

function add_dissipative_heating!(Gⁿ, fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, closure, model)
    implicit_stage(closure, model) && return nothing
    grid = model.grid
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    launch!(grid.architecture, grid, :xyz, _add_dissipative_heating!, Gⁿ[name], grid, closure_fields,
            model.tracers[TKE_NAME], total_density(model.dynamics))
    return nothing
end

#####
##### The share of the shear production that the TKE does not receive
#####
##### With stability functions whose shear production coefficient ℂᴾ is below one (`shear_production_fraction`),
##### the TKE receives ℂᴾ of the shear conversion and the air is heated by the rest, ρ ĉᵖ Π ∂t θˡⁱ += ρ (1 - ℂᴾ) P: in
##### the stage tendency for the part it carries (all of P, or the work of the surface stress where the conversion of
##### the solved momentum is given before the TKE solve), and before the solves of the TKE and of θˡⁱ for that
##### conversion. Together they receive the shear conversion, so the corrections after the solves (which top up the
##### TKE to what the solves converted) are unchanged and the energy budget closes as with ℂᴾ = 1.
#####

# Stability functions without a coefficient of their own give the TKE the whole shear production
add_shear_production_heating!(Gⁿ, stability_functions, fields, closure_fields, closure, model) = nothing

@kernel function _add_shear_production_heating!(Gρθ, grid, closure, cl, ρ, injected)
    i, j, k = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    ℂᴾ = shear_production_fraction(closure_ij.stability_functions)
    P = explicit_shear_conversion(cl, i, j, k, injected)
    @inbounds Gρθ[i, j, k] += ρ[i, j, k] * (1 - ℂᴾ) * P / cl.ĉᵖΠ[i, j, k] * !inactive_cell(i, j, k, grid)
end

function add_shear_production_heating!(Gⁿ, ::ColumnStabilityDamping, fields::SubgridCloudFields{<:AppliedFluxEnergetics},
                                       closure_fields, closure, model)
    grid = model.grid
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    launch!(grid.architecture, grid, :xyz, _add_shear_production_heating!, Gⁿ[name], grid, closure, fields.cloud,
            total_density(model.dynamics), solved_conversions_before_tke_solve(fields, closure, model))
    return nothing
end

#####
##### The latent correction of the static stability and of the buoyancy production
#####
##### A displacement that condenses as it goes feels the saturated stratification N²⁺; one
##### without phase change feels N²ᵇ = Σ b̄_χ δz χ. With (θˡⁱ, qᵗ, qᶜ) for (θˡⁱ, vapor, cloud
##### condensate), a saturated displacement keeps s = 0 with its condensate following the
##### saturation excess s̄ at fixed pressure, so
#####
#####   N²⁺ - N²ᵇ = (b̄ˡ - b̄ᵛ) (∂z s̄ - ∂z qᶜ),   ∂z s̄ = c̄_q ∂z qʷ - c̄_θ ∂z θˡⁱ
#####
##### (c_q, c_θ: the linearized saturation excess of the level with the model's thermodynamics).
##### The turbulence carries the condensate flux
#####
#####   w′qᶜ′ = η aʷ w′s′ + (1 - η) (-Kᶜ ∂z qᶜ),    w′s′ = -Kᶜ ∂z s̄,    η(x) = 1 - (1 - e⁻ˣ) / x,
#####
##### aʷ the share of the flux of the saturation excess carried by the supersaturated part of the
##### distribution, the mean saturated fraction between the components of the double Gaussian, which carry its fluxes
##### (`center_condensate_flux_fraction`; Kᶜ cancels),
##### and η the fraction of the excess that condenses over an eddy time tₑ = e / ε when it relaxes at
##### the rate ωᶜ (η = 1 for instantaneous condensation): the share η of the condensate follows the
##### saturation excess of the air that carries it, and only the share 1 - η, which does not change
##### phase within an eddy time, is mixed as a passive scalar. The closure applies -Kᶜ ∂z qᶜ to the
##### prognostic condensate, so the static stability and the buoyancy production use
#####
#####   N² = N²ᵇ - ΔN²ᴸ,    ΔN²ᴸ = η (b̄ˡ - b̄ᵛ) (∂z qᶜ - aʷ ∂z s̄)
#####      = N²ᶜ + (b̄ˡ - b̄ᵛ) [η aʷ ∂z s̄ + (1 - η) ∂z qᶜ],    N²ᶜ = N²ᵇ - (b̄ˡ - b̄ᵛ) ∂z qᶜ,
#####
##### N²ᶜ the stratification of the conserved variables (θˡⁱ, qʷ). With instantaneous condensation this
##### is (1 - aʷ) N²ᶜ + aʷ N²⁺: the unsaturated share of the distribution holds no condensate and feels
##### the stratification of the conserved variables, not that of the mean condensate. (Weighting N²ᵇ
##### and N²⁺ instead, N²ᵇ - η aʷ (N²ᵇ - N²⁺), leaves (1 - aʷ)(b̄ˡ - b̄ᵛ) ∂z qᶜ > 0 wherever the mean
##### condensate rises with height in partly cloudy air: a stable stratification at the base of a
##### well-mixed cloud layer and a buoyancy flux of the wrong sign there.)
##### The buoyancy production gains Bᴸ = Kᶜ ΔN²ᴸ over the conversion of the applied fluxes; that energy
##### is the work of the latent heat of the subgrid condensation and is taken from the internal energy.
##### aᴺ = η aʷ is stored: the share of the flux of the saturation excess that condenses.
#####
##### (b̄ˡ - b̄ᵛ) is that of the distributed condensate, b̄ᶜ - b̄ᵛ with b_qᶜ = λ b_qˡ + (1 - λ) b_qⁱ for the liquid
##### fraction λ of its phase equilibrium (λ = 1 for cloud liquid), so that the latent heat of its buoyancy is the
##### one with which c_θ is formed.
#####
##### Ice outside the distributed water (P3's ice) fills the cell and relaxes the vapor excess over ice at its own
##### rate ωⁱ; it is not equilibrium condensate, so it has no target of its own and the closure applies -Kᶜ ∂z qⁱ to
##### it as to a passive scalar. A displaced parcel deposits on its ice the share ηⁱ of the change of its saturation
##### excess over ice sⁱ = c_qⁱ qʷ - c_θⁱ θʷ (linearized with qᵛ⁺ⁱ and ℒⁱ), so the ice flux the turbulence carries
##### exceeds the applied one by ηⁱ w′sⁱ′ = -Kᶜ ηⁱ ∂z s̄ⁱ, and the latent correction gains the ice term
#####
#####   ΔN²ᴸ = η (b̄ᶜ - b̄ᵛ) (∂z qᶜ - aʷ ∂z s̄) - ηⁱ (b̄ⁱ - b̄ᵛ) ∂z s̄ⁱ,
#####
##### with the shares of `saturated_response_sharesᶜᶜᶠ`: in the saturated part the distributed condensate and the
##### ice share the vapor excess in proportion to their rates, and in the subsaturated part the ice alone takes it.
##### Deposition is buoyant with ℒⁱ (through b_qⁱ), condensation with the latent heat of the distributed condensate.
##### Without ice ηⁱ = 0 and the correction is unchanged. The ice term enters Bᴸ, the conversion of the solved state
##### and the latent part of the heating budget flux like the condensate term, and its energy is drawn from the
##### internal energy with them.
#####

# The condensate of the distributed water at center k: the cloud liquid where it is prognostic (P3, Kessler,
# non-equilibrium cloud formation; the vapor and the cloud liquid are the distributed water), and otherwise the
# condensate of the adjusted state (saturation adjustment, whose moisture density is the total water), so that the
# stratification N²ᵇ of that state and the latent correction count the condensate gradient once. In the state the
# implicit solves leave, the condensate of saturation adjustment is that of the stage (the solves move its total water,
# not its partition, as the conversion of the solved fluxes reads it).
@inline function distributed_condensateᶜᶜᶜ(i, j, k, grid, s)
    @inbounds begin
        ρ = s.ρ[i, j, k]
        qᵛᵉ = s.ρqᵛᵉ[i, j, k] / ρ
    end
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉ, s.microphysical_fields)
    return distributed_condensate(cloud_liquid_density(s.microphysical_fields), i, j, k, ρ, q)
end

@inline distributed_condensate(::Nothing, i, j, k, ρ, q) = q.liquid + q.ice
@inline distributed_condensate(ρqᶜˡ, i, j, k, ρ, q) = @inbounds ρqᶜˡ[i, j, k] / ρ

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The coefficients `(c_q, c_θ)` of the change of the equilibrium condensate of the distributed water at center `k`,
``δqᶜ = c_q δqʷ - c_θ δθʷ`` at fixed pressure, linearized about the cell's own state: its temperature `T`, its
moisture fractions and the saturation over the surface of the phase equilibrium of `thermodynamics` (the
condensate of the distribution, or ice with `ice_saturation_thermodynamics`). From ``qᶜ = qʷ - qᵛ⁺``,
``qᵛ⁺ = pᵛ⁺(T) Rᵐ / (p Rᵛ)`` and ``cᵖᵐ T = ĉᵖ Π θʷ + ℒ₀ qᶜ`` (+ the held condensate), with ``ℒ = ℒ₀ + (cᵖᵛ - cᶜ) T``:

    D c_q = 1 - qᵛ⁺ (Rᵛ - Rᵈ) / Rᵐ - (∂qᵛ⁺/∂T) B / cᵖᵐ,    D c_θ = (∂qᵛ⁺/∂T) ĉᵖ Π / cᵖᵐ,
    D = 1 - qᵛ⁺ Rᵛ / Rᵐ + (∂qᵛ⁺/∂T) ℒ / cᵖᵐ,    B = θʷ Π ((cᵖᵛ - cᵖᵈ) + ĉᵖ ln(p / pˢᵗ) ∂κ/∂qᵗ) - (cᵖᵛ - cᵖᵈ) T,

with ``∂ ln qᵛ⁺/∂T`` that of the equilibrium (the change of its liquid fraction included). A saturated displacement
holds ``s = 0``, so these give the saturated stratification ``N²ᶜ + (b_qᶜ - b_qᵛ) ∂z s̄`` exactly to first order.
The linearization of the assumed PDF is about the temperature of the water all vapor instead, which in a
cloud is off by ``ℒ qᶜ / ĉᵖ`` (several kelvin in a deep cloud) and misses the saturated stratification by tens
of percent there.
"""
@inline function cell_saturation_linearization(thermodynamics, i, j, k, grid, s, θʷ)
    @inbounds begin
        ρ = s.ρ[i, j, k]
        T = s.T[i, j, k]
        p = s.p[i, j, k]
        qᵛᵉ = s.ρqᵛᵉ[i, j, k] / ρ
    end
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉ, s.microphysical_fields)
    return saturation_linearization(thermodynamics, T, p, q, θʷ, s.exner_exponent, s.standard_pressure)
end

# The linearization of `cell_saturation_linearization` at the temperature `T`, pressure `p` and moisture mass fractions
# `q` of a cell, whose distributed water has the liquid-ice potential temperature `θʷ`
@inline function saturation_linearization(thermodynamics, T, p, q, θʷ, exner_exponent_formulation, pˢᵗ)
    constants = thermodynamics.constants
    equilibrium = thermodynamics.phase_equilibrium
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    R̂ = vapor_equivalent_gas_constant(q, constants)
    κ = exner_exponent(exner_exponent_formulation, q, constants)
    κ′ = exner_exponent_moisture_derivative(exner_exponent_formulation, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    ln_p = log(p / pˢᵗ)
    Π = (p / pˢᵗ)^κ

    qᵛ⁺ = saturation_specific_humidity(T, p / (Rᵐ * T), constants, saturation_surface(equilibrium, T))
    ℒ, log_slope = condensate_latent_heat_and_log_slope(equilibrium, constants, T)
    ∂qᵛ⁺∂T = qᵛ⁺ * log_slope
    B = θʷ * Π * ((cᵖᵛ - cᵖᵈ) + ĉᵖ * ln_p * κ′) - (cᵖᵛ - cᵖᵈ) * T
    D = 1 - qᵛ⁺ * Rᵛ / Rᵐ + ∂qᵛ⁺∂T * ℒ / cᵖᵐ
    v_q = 1 - qᵛ⁺ * (Rᵛ - Rᵈ) / Rᵐ - ∂qᵛ⁺∂T * B / cᵖᵐ
    v_θ = ∂qᵛ⁺∂T * ĉᵖ * Π / cᵖᵐ
    return (; qᵛ⁺, v_q, v_θ, D)
end

@inline function saturated_condensate_coefficients(thermodynamics, i, j, k, grid, s, θʷ)
    lin = cell_saturation_linearization(thermodynamics, i, j, k, grid, s, θʷ)
    return lin.v_q / lin.D, lin.v_θ / lin.D
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The coefficients `(v_q, v_θ)` of the change of the vapor excess over the surface of `thermodynamics` at fixed
condensate, ``δ(qᵛ - qᵛ⁺) = v_q δqʷ - v_θ δθʷ`` at fixed pressure, linearized about the state of center `k`: the
numerators of [`saturated_condensate_coefficients`](@ref), before the latent feedback `D` of the condensate that
the excess would form.
"""
@inline function vapor_excess_coefficients(thermodynamics, i, j, k, grid, s, θʷ)
    lin = cell_saturation_linearization(thermodynamics, i, j, k, grid, s, θʷ)
    return lin.v_q, lin.v_θ
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The saturation excess ``s̄`` of the mean state of center `k` in the linearization of
[`saturated_condensate_coefficients`](@ref) about the cell's own state: its equilibrium condensate ``qᶜ`` of the
distributed water plus its vapor excess over the surface of `thermodynamics`, condensed with the latent feedback `D`,

    s̄ = qᶜ + (qʷ - qᶜ - qᵛ⁺(T)) / D,

so that ``s̄ = qᶜ`` in saturated air, ``s̄ < 0`` is the deficit in subsaturated air, and ``δs̄ = c_q δqʷ - c_θ δθʷ`` in
both.
"""
@inline function mean_saturation_excess(thermodynamics, i, j, k, grid, s, θʷ, qʷ)
    lin = cell_saturation_linearization(thermodynamics, i, j, k, grid, s, θʷ)
    qᶜ = distributed_condensateᶜᶜᶜ(i, j, k, grid, s)
    return qᶜ + (qʷ - qᶜ - lin.qᵛ⁺) / lin.D
end

# The rates (s⁻¹) at which the condensate of the cell absorbs vapor excess, from the microphysics
# (`AtmosphereModels.phase_relaxation_rates`; 0 without condensing microphysics): ωˡ of the distributed condensate
# and ωⁱ of the ice outside the distributed water. They include the psychrometric correction ξ of the latent
# heating (P3: ωˡ = 4π Dᵛ Nᶜˡ r̄ / ξˡ), so x = tₑ ω.
@inline function cell_phase_relaxation_rates(i, j, k, grid, s)
    @inbounds begin
        ρ = s.ρ[i, j, k]
        qᵛᵉ = s.qᵛᵉ[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉ, s.microphysical_fields)
    𝒰 = AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid, s.formulation, s.dynamics, q)
    return AtmosphereModels.phase_relaxation_rates(i, j, k, grid, s.microphysics, s.microphysical_fields, 𝒰, s.constants)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The share ``η(x) = 1 - (1 - e^{-x}) / x`` of the saturated response that a parcel displaced for an
eddy time ``tₑ`` feels when its condensate relaxes the vapor excess at the rate ``ωᶜ``, with
``x = tₑ ωᶜ``. Below ``x = 10⁻³`` it is the series ``x/2 - x²/6 + x³/24`` (``1 - (1 - e^{-x})/x`` loses
its digits to cancellation there), so that the value and its derivative are accurate and finite
down to ``x = 0``, where ``η = 0``.
"""
@inline function condensing_fraction(x)
    FT = typeof(x)
    small = x < convert(FT, 1e-3)
    xˢ = ifelse(small, one(FT), x)
    return ifelse(small, x * (1 / convert(FT, 2) - x * (1 / convert(FT, 6) - x / 24)), 1 + expm1(-xˢ) / xˢ)
end

# The shares (η, ηⁱ, ηʷⁱ) of the saturated response at the face k, with the rates of the two cells interpolated to it.
# The distributed condensate lies in the saturated part of the distribution; the ice and the liquid on ice outside the
# distributed water fill the cell. In the saturated part all three absorb the vapor excess, together at
# ω⁺ = ωˡ + ωⁱ + ωʷⁱ, in proportion to their rates; in the subsaturated part the ice and its liquid coating, at
# ω⁻ = ωⁱ + ωʷⁱ. The saturated part carries the share aʷ of the flux of the saturation excess s, and the same share of
# that of the excess over ice sⁱ, which a displacement changes in proportion to s (`center_condensate_flux_fraction`), so
#
#     η = (ωˡ / ω⁺) η(tₑ ω⁺),
#     ηⁱ = aʷ (ωⁱ / ω⁺) η(tₑ ω⁺) + (1 - aʷ) (ωⁱ / ω⁻) η(tₑ ω⁻),
#     ηʷⁱ = aʷ (ωʷⁱ / ω⁺) η(tₑ ω⁺) + (1 - aʷ) (ωʷⁱ / ω⁻) η(tₑ ω⁻).
#
# Without ice, η = η(tₑ ωˡ) and ηⁱ = ηʷⁱ = 0. A microphysics that removes the vapor excess instantly (saturation
# adjustment) has η = 1 and nothing outside its distributed water, without forming tₑ ω = ∞.
@inline saturated_response_sharesᶜᶜᶠ(::Val{true}, tₑ, aʷ, i, j, k, grid, s) = (one(tₑ), zero(tₑ), zero(tₑ))

@inline function saturated_response_sharesᶜᶜᶠ(::Val{false}, tₑ, aʷ, i, j, k, grid, s)
    above = cell_phase_relaxation_rates(i, j, k, grid, s)
    below = cell_phase_relaxation_rates(i, j, k-1, grid, s)
    ωˡ = (above.ωˡ + below.ωˡ) / 2
    ωⁱ = (above.ωᵈⁱ + below.ωᵈⁱ) / 2
    ωʷⁱ = (above.ωʷⁱ + below.ωʷⁱ) / 2
    ω⁺ = ωˡ + ωⁱ + ωʷⁱ
    ω⁻ = ωⁱ + ωʷⁱ
    FT = typeof(ω⁺)
    η⁺ = condensing_fraction(tₑ * ω⁺)
    η⁻ = condensing_fraction(tₑ * ω⁻)
    share(ω, total, fallback) = ifelse(total > 0, ω / ifelse(total > 0, total, one(FT)), fallback)
    η = share(ωˡ, ω⁺, one(FT)) * η⁺
    ηⁱ = aʷ * share(ωⁱ, ω⁺, zero(FT)) * η⁺ + (1 - aʷ) * share(ωⁱ, ω⁻, zero(FT)) * η⁻
    ηʷⁱ = aʷ * share(ωʷⁱ, ω⁺, zero(FT)) * η⁺ + (1 - aʷ) * share(ωʷⁱ, ω⁻, zero(FT)) * η⁻
    return η, ηⁱ, ηʷⁱ
end

# aʷ at the center k: the share of the flux of the saturation excess s that the saturated part of the double Gaussian
# carries, from its components in the vapor excess (`store_vapor_excess_components!`). Within a component w is
# uncorrelated with s, so the fluxes are carried by the difference of the component means, and the displacement from one
# component to the other changes s by Δμ = μ₁ - μ₂. The saturated part takes the share of that change accrued where
# s > 0: the mean of the saturated fraction Φ(s / σ) over the segment,
#
#     aʷ = [g(μ₁, σ) - g(μ₂, σ)] / (μ₁ - μ₂),    g(μ, σ) = E[max(s, 0)] = μ Φ(μ / σ) + σ φ(μ / σ),
#
# averaged over the spreads σ₁ and σ₂ of the two components. A mean of values in [0, 1], so in [0, 1] itself. With equal
# spreads it is the distribution's own w′qˡ′ / w′s′ = Δqˡ / Δμ; a difference of the spreads gives the components a
# contrast of condensate at no contrast of s, which is not a share of the flux of s, and the quotient of the two contrasts
# leaves [0, 1] (to -0.06 in SHOC's RICO column). The excess over ice sⁱ changes in proportion to s along the same
# displacement, so the saturated part carries the share aʷ of its flux too. Where rounding does not resolve the segment,
# its midpoint.
@inline function center_condensate_flux_fraction(i, j, k, cl)
    @inbounds begin
        μ₁ = cl.μ₁[i, j, k]
        μ₂ = cl.μ₂[i, j, k]
        σ₁ = cl.σ₁[i, j, k]
        σ₂ = cl.σ₂[i, j, k]
    end
    return (saturated_share_of_segment(μ₁, μ₂, σ₁) + saturated_share_of_segment(μ₁, μ₂, σ₂)) / 2
end

# The mean of Φ(s / σ) over the segment of s from μ₂ to μ₁
@inline function saturated_share_of_segment(μ₁, μ₂, σ)
    FT = typeof(μ₁)
    Δμ = μ₁ - μ₂
    resolved = abs(Δμ) > sqrt(eps(FT)) * (abs(μ₁) + abs(μ₂) + σ)
    secant = (component_saturated_excess(μ₁, σ) - component_saturated_excess(μ₂, σ)) / ifelse(resolved, Δμ, one(FT))
    return ifelse(resolved, secant, component_saturated_fraction((μ₁ + μ₂) / 2, σ))
end

# The mean E[max(s, 0)] and the saturated fraction P(s > 0) of a Gaussian of mean μ and spread σ, with the
# `standard_score` of μ, regular at σ = 0 (where they are the step of the point mass, max(μ, 0) and [μ > 0])
@inline function component_saturated_excess(μ, σ)
    FT = typeof(μ)
    z = standard_score(μ, zero(FT), σ)
    Φ = erfc(-z / sqrt(FT(2))) / 2
    φ = exp(-z * z / 2) / sqrt(2 * FT(π))
    return μ * Φ + σ * φ
end

@inline component_saturated_fraction(μ, σ) = erfc(-standard_score(μ, zero(μ), σ) / sqrt(2 * one(μ))) / 2

# aʷ of the distribution at the face k: the average of its centers
@inline face_condensate_flux_fraction(i, j, k, cl) =
    (center_condensate_flux_fraction(i, j, k, cl) + center_condensate_flux_fraction(i, j, k-1, cl)) / 2

# The eddy time tₑ = e / ε = ℓ / (𝕊ᴰ √e) at the face k
@inline function face_eddy_time(i, j, k, grid, closure_ij, closure_fields, velocities, e)
    ℓ = @inbounds closure_fields.ℓᵐ[i, j, k]
    u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure_ij, e)
    𝕊ᴰ = dissipation_stability_functionᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, closure_fields.N², e, closure_fields.ℓᵐ)
    return ℓ / (𝕊ᴰ * u)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The latent correction of the static stability at the interior face `k`, for the saturated share `aʷ` of the fluxes of
the saturation excess and of the excess over ice:

    ΔN²ᴸ = η (b̄ᶜ - b̄ᵛ) (∂z qᶜ - aʷ ∂z s̄) - ηⁱ (b̄ⁱ - b̄ᵛ) ∂z s̄ⁱ - ηʷⁱ (b̄ˡ - b̄ᵛ) ∂z s̄,

with the shares of `saturated_response_sharesᶜᶜᶠ` over the eddy time of the face, the saturation excesses
linearized about each cell (`saturated_condensate_coefficients`), and the condensate of the distributed water
(`distributed_condensateᶜᶜᶜ`). Returns the correction and its parts, which the conversion of the solved state reads.
"""
@inline function latent_correctionᶜᶜᶠ(i, j, k, grid, closure_ij, closure_fields, velocities, s, qʷ, θʷ, e, thermodynamics, cl,
                                      aʷ)
    @inbounds begin
        θ⁺ = θʷ[i, j, k]
        θ⁻ = θʷ[i, j, k-1]
        qʷ⁺ = qʷ[i, j, k]
        qʷ⁻ = qʷ[i, j, k-1]
        Δz = Δzᶜᶜᶠ(i, j, k, grid)
        # The saturation excess s = c_q qʷ - c_θ θʷ, and over ice sⁱ = c_qⁱ qʷ - c_θⁱ θʷ, linearized about each cell
        c_q⁺, c_θ⁺ = saturated_condensate_coefficients(thermodynamics, i, j, k, grid, s, θ⁺)
        c_q⁻, c_θ⁻ = saturated_condensate_coefficients(thermodynamics, i, j, k-1, grid, s, θ⁻)
        ice = ice_saturation_thermodynamics(thermodynamics)
        c_qⁱ⁺, c_θⁱ⁺ = saturated_condensate_coefficients(ice, i, j, k, grid, s, θ⁺)
        c_qⁱ⁻, c_θⁱ⁻ = saturated_condensate_coefficients(ice, i, j, k-1, grid, s, θ⁻)
        c_q = (c_q⁺ + c_q⁻) / 2
        c_θ = (c_θ⁺ + c_θ⁻) / 2
        c_qⁱ = (c_qⁱ⁺ + c_qⁱ⁻) / 2
        c_θⁱ = (c_θⁱ⁺ + c_θⁱ⁻) / 2
        ∂z_s = (c_q * (qʷ⁺ - qʷ⁻) - c_θ * (θ⁺ - θ⁻)) / Δz
        ∂z_sⁱ = (c_qⁱ * (qʷ⁺ - qʷ⁻) - c_θⁱ * (θ⁺ - θ⁻)) / Δz
        Δqᶜ = distributed_condensateᶜᶜᶜ(i, j, k, grid, s) - distributed_condensateᶜᶜᶜ(i, j, k-1, grid, s)
        ∂z_qᶜ = Δqᶜ / Δz
        bᵛ = cl.b_qᵛ[i, j, k] + cl.b_qᵛ[i, j, k-1]
        bᶜᵛ = (cl.b_qᶜ[i, j, k] + cl.b_qᶜ[i, j, k-1] - bᵛ) / 2
        bⁱᵛ = (cl.b_qⁱ[i, j, k] + cl.b_qⁱ[i, j, k-1] - bᵛ) / 2
        bˡᵛ = (cl.b_qˡ[i, j, k] + cl.b_qˡ[i, j, k-1] - bᵛ) / 2
    end
    tₑ = face_eddy_time(i, j, k, grid, closure_ij, closure_fields, velocities, e)
    instantaneous = Val(AtmosphereModels.instantaneous_phase_relaxation(s.microphysics))
    η, ηⁱ, ηʷⁱ = saturated_response_sharesᶜᶜᶠ(instantaneous, tₑ, aʷ, i, j, k, grid, s)
    ΔN²ᴸ = η * bᶜᵛ * (∂z_qᶜ - aʷ * ∂z_s) - ηⁱ * bⁱᵛ * ∂z_sⁱ - ηʷⁱ * bˡᵛ * ∂z_s
    return (; ΔN²ᴸ, η, ηⁱ, ηʷⁱ, aᴺ = η * aʷ, c_q, c_θ, c_qⁱ, c_θⁱ, bᶜᵛ, Δqᶜ, Δz,
              outside = -ηⁱ * bⁱᵛ * ∂z_sⁱ - ηʷⁱ * bˡᵛ * ∂z_s)
end

@inline function store_latent_correction!(cl, i, j, k, interior, aʷ, L)
    FT = typeof(L.ΔN²ᴸ)
    @inbounds begin
        cl.aʷ[i, j, k] = ifelse(interior, aʷ, zero(FT))
        cl.aᴺ[i, j, k] = ifelse(interior, L.aᴺ, zero(FT))
        cl.c_θ[i, j, k] = ifelse(interior, L.c_θ, zero(FT))
        cl.c_q[i, j, k] = ifelse(interior, L.c_q, zero(FT))
        cl.c_θⁱ[i, j, k] = ifelse(interior, L.c_θⁱ, zero(FT))
        cl.c_qⁱ[i, j, k] = ifelse(interior, L.c_qⁱ, zero(FT))
        cl.η[i, j, k] = ifelse(interior, L.η, zero(FT))
        cl.ηⁱ[i, j, k] = ifelse(interior, L.ηⁱ, zero(FT))
        cl.ηʷⁱ[i, j, k] = ifelse(interior, L.ηʷⁱ, zero(FT))
        cl.ΔN²ᴸ[i, j, k] = ifelse(interior, L.ΔN²ᴸ, zero(FT))
    end
    return nothing
end

@kernel function _latent_corrected_stability!(N², grid, closure, closure_fields, velocities, s, e)
    i, j, k = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    cl = closure_fields.subgrid_cloud.cloud
    θʷ, qʷ, thermodynamics = distributed_water(s, grid)
    kₛ = max(k, 2)
    aʷ = face_condensate_flux_fraction(i, j, kₛ, cl)
    L = latent_correctionᶜᶜᶠ(i, j, kₛ, grid, closure_ij, closure_fields, velocities, s, qʷ, θʷ, e, thermodynamics, cl, aʷ)
    interior = k > 1
    store_latent_correction!(cl, i, j, k, interior, aʷ, L)
    @inbounds begin
        N²ᵇ = applied_stratificationᶜᶜᶠ(i, j, kₛ, grid, s, cl)
        N²ᵉ = N²ᵇ - L.ΔN²ᴸ
        # The static stability the mixing length and the stability functions read: the latent one, that of the mean state
        # with the distribution's mean liquid, or the initial one
        N²⁰ = N²[i, j, k]
        N²⁺ = condensate_flux_stratification(closure_ij.static_stability.fraction.stratification, N²ᵉ, N²ᵇ, L.bᶜᵛ,
                                             cl.qˡ[i, j, kₛ] - cl.qˡ[i, j, kₛ-1], L.Δqᶜ, L.Δz, N²⁰)
        N²[i, j, k] = ifelse(interior, N²⁺, N²⁰)
    end
end

# The pass of the stability coupling with the energetics of the applied fluxes: the latent-corrected N², then the
# mixing length and the diffusivities from it
function applied_flux_stability_pass!(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, closure, model, tracers,
                                      parameters)
    grid = model.grid
    arch = grid.architecture
    launch!(arch, grid, parameters, _latent_corrected_stability!, closure_fields.N², grid, closure, closure_fields,
            model.velocities, applied_flux_state(model), tracers[TKE_NAME])
    compute_mixing_length_and_diffusivities!(closure_fields, closure, model, tracers, parameters)
    return nothing
end

# The latent part of the buoyancy production is drawn from the internal energy: ρ ĉᵖ Π ∂t θˡⁱ -= ρ Bᴸ
@kernel function _draw_latent_conversion!(Gρθ, grid, cl, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρθ[i, j, k] -= ρ[i, j, k] * cl.Bᴸ[i, j, k] / cl.ĉᵖΠ[i, j, k] * !inactive_cell(i, j, k, grid)
end

draw_latent_conversion!(Gⁿ, fields, closure, model) = nothing

function draw_latent_conversion!(Gⁿ, fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure, model)
    # with the conversions of the solved state given before the TKE solve, the latent part is drawn after the solves
    solved_conversions_before_tke_solve(fields, closure, model) && return nothing
    grid = model.grid
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    launch!(grid.architecture, grid, :xyz, _draw_latent_conversion!, Gⁿ[name], grid, fields.cloud, total_density(model.dynamics))
    return nothing
end
