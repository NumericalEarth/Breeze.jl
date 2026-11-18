#####
##### `CompressibleVectorInvariant` momentum advection for `HydrostaticDynamics`
#####
##### `CompressibleVectorInvariant` writes the advection of the momentum density ρᵘ u, with
##### ρᵘ = ℑx(ρ) the density of the mass flux through the x-faces, as
#####
#####     ∂ₜ(ρᵘ u) = Lᵘ - ρᵘ ∂xK - u 𝒟ᵘ - V⁻¹ δz(Wᵘ uᴿ) ,
#####
##### with Lᵘ the Lamb term, 𝒟ᵘ the divergence of the horizontal mass flux at the u point, which
##### the scheme may reconstruct, and Wᵘ uᴿ the vertical flux of u through the z-faces of its cell.
##### `HydrostaticDynamics` advances the velocity on layers whose mass changes: with continuity
##### averaged to the u point, ∂ₜρᵘ = - ℑx(D) - V⁻¹ δz(Wᵘ), D the horizontal divergence that
##### continuity carries,
#####
#####     ρᵘ ∂ₜu = ∂ₜ(ρᵘ u) - u ∂ₜρᵘ = Lᵘ - ρᵘ ∂xK - u [𝒟ᵘ - ℑx(D)] - V⁻¹ [δz(Wᵘ uᴿ) - u δz(Wᵘ)] .
#####
##### The horizontal divergence therefore enters only as the departure of the scheme's
##### reconstruction from the average that continuity carries: nothing with a centered
##### divergence, a dissipative correction with an upwind-biased one. The vertical term is the
##### advection w ∂z u, and leaves a velocity that is uniform in the column unchanged.
##### The kinetic energy Σ V ρᵘ u² / 2 has the same budget as for the momentum density.
#####

function HydrostaticEquations.validate_hydrostatic_momentum_advection(scheme::CVI)
    scheme isa SplitCVI ||
        throw(ArgumentError("HydrostaticDynamics requires a CompressibleVectorInvariant momentum_advection " *
                            "to have vertical_formulation = SplitVerticalAdvection()."))
    return nothing
end

# u [𝒟ᵘ - ℑx(D)]: the divergence term of the scheme less the one of continuity
@inline x_divergence_departure(i, j, k, grid, scheme::CenteredDivergence, vf, u, ρu, ρv, ρw) = zero(grid)
@inline y_divergence_departure(i, j, k, grid, scheme::CenteredDivergence, vf, v, ρu, ρv, ρw) = zero(grid)

@inline function x_divergence_departure(i, j, k, grid, scheme::ReconstructedDivergence, vf, u, ρu, ρv, ρw)
    D = ( ℑxᶠᵃᵃ(i, j, k, grid, x_mass_flux_divergenceᶜᶜᶜ, ρu, ρv, ρw)
          + ℑxᶠᵃᵃ(i, j, k, grid, x_cross_divergenceᶜᶜᶜ, vf, ρu, ρv, ρw))
    return @inbounds x_divergence_term(i, j, k, grid, scheme, vf, u, ρu, ρv, ρw) - u[i, j, k] * D
end

@inline function y_divergence_departure(i, j, k, grid, scheme::ReconstructedDivergence, vf, v, ρu, ρv, ρw)
    D = ( ℑyᵃᶠᵃ(i, j, k, grid, y_mass_flux_divergenceᶜᶜᶜ, ρu, ρv, ρw)
          + ℑyᵃᶠᵃ(i, j, k, grid, y_cross_divergenceᶜᶜᶜ, vf, ρu, ρv, ρw))
    return @inbounds y_divergence_term(i, j, k, grid, scheme, vf, v, ρu, ρv, ρw) - v[i, j, k] * D
end

@inline function HydrostaticEquations.momentum_advection_u(i, j, k, grid, scheme::CVI, ρ, ρw, u, v)
    vf = scheme.vertical_formulation
    ρu = XMomentumDensity(ρ, u)
    ρv = YMomentumDensity(ρ, v)

    δz_ρwu = δzᵃᵃᶜ(i, j, k, grid, vertical_momentum_flux_Wu, scheme, ρw, u)
    δz_ρw = δzᵃᵃᶜ(i, j, k, grid, Az_ℑx_ρwᶠᶜᶠ, scheme, ρw)
    vertical_advection = @inbounds V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δz_ρwu - u[i, j, k] * δz_ρw)

    return ( - x_lamb_term(i, j, k, grid, scheme, ρ, u, v)
             + x_kinetic_energy_gradient(i, j, k, grid, scheme, vf, ρ, u, v, ρw)
             + x_divergence_departure(i, j, k, grid, scheme, vf, u, ρu, ρv, ρw)
             + vertical_advection) / ℑxᶠᵃᵃ(i, j, k, grid, ρ)
end

@inline function HydrostaticEquations.momentum_advection_v(i, j, k, grid, scheme::CVI, ρ, ρw, u, v)
    vf = scheme.vertical_formulation
    ρu = XMomentumDensity(ρ, u)
    ρv = YMomentumDensity(ρ, v)

    δz_ρwv = δzᵃᵃᶜ(i, j, k, grid, vertical_momentum_flux_Wv, scheme, ρw, v)
    δz_ρw = δzᵃᵃᶜ(i, j, k, grid, Az_ℑy_ρwᶜᶠᶠ, scheme, ρw)
    vertical_advection = @inbounds V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δz_ρwv - v[i, j, k] * δz_ρw)

    return ( - y_lamb_term(i, j, k, grid, scheme, ρ, u, v)
             + y_kinetic_energy_gradient(i, j, k, grid, scheme, vf, ρ, u, v, ρw)
             + y_divergence_departure(i, j, k, grid, scheme, vf, v, ρu, ρv, ρw)
             + vertical_advection) / ℑyᵃᶠᵃ(i, j, k, grid, ρ)
end
