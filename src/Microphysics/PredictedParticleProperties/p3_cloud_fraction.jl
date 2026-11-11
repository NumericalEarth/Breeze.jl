#####
##### Cloud fraction diagnostic: C = Φ(Q₁) from the subgrid condensation
#####
#
# The cloud fraction C = Φ(Q₁), Q₁ = s̄ / σₛ, is the one the subgrid (PDF) condensation in
# `p3_subgrid_phase1_rates` uses: the same thermodynamic state (after the vapor-excess
# alignment), the same mean saturation deficit s̄ (`p3_saturation_deficit`), and the same
# closure-supplied statistics. It is evaluated in every cell, including the cold cells where
# the rates keep their grid-mean form. With σₛ = 0 (no subgrid distribution, or a zero spread
# floor and zero subgrid variance) C is 1 in saturated cells (s̄ > 0) and 0 otherwise.

struct P3CloudFractionKernelFunction{P, M, F, D, TH, R, U, CL, CF, S}
    microphysics :: P
    microphysical_fields :: M
    formulation :: F
    dynamics :: D
    thermodynamic_constants :: TH
    density :: R
    velocities :: U
    closure :: CL
    closure_fields :: CF
    subgrid_state :: S
end

Utils.prettysummary(::P3CloudFractionKernelFunction) = "P3CloudFractionKernelFunction"

Adapt.adapt_structure(to, k::P3CloudFractionKernelFunction) =
    P3CloudFractionKernelFunction(Adapt.adapt(to, k.microphysics),
                                  Adapt.adapt(to, k.microphysical_fields),
                                  Adapt.adapt(to, k.formulation),
                                  Adapt.adapt(to, k.dynamics),
                                  Adapt.adapt(to, k.thermodynamic_constants),
                                  Adapt.adapt(to, k.density),
                                  Adapt.adapt(to, k.velocities),
                                  Adapt.adapt(to, k.closure),
                                  Adapt.adapt(to, k.closure_fields),
                                  Adapt.adapt(to, k.subgrid_state))

Microphysics.cloud_fraction(model, p3::P3) =
    KernelFunctionOperation{Center, Center, Center}(
        P3CloudFractionKernelFunction(p3, model.microphysical_fields, model.formulation, model.dynamics,
                                      model.thermodynamic_constants, AM.total_density(model.dynamics),
                                      model.velocities, model.closure, model.closure_fields,
                                      p3_subgrid_state(p3.subgrid, model)),
        model.grid)

# Gaussian cloud fraction; `subgrid_cloud` takes the σₛ → 0 limit itself.
@inline subgrid_cloud_fraction(::DiracStatistics, s̄, a, b, σ₀) = ifelse(s̄ > 0, one(s̄), zero(s̄))

@inline subgrid_cloud_fraction(𝒮, s̄, a, b, σ₀) = subgrid_cloud(𝒮, s̄, a, b, σ₀).C

# The distribution the correlation assigns to the statistics (skewed for a non-zero skewness)
@inline subgrid_cloud_fraction(correlation, 𝒮::DiracStatistics, s̄, a, b, σ₀) = subgrid_cloud_fraction(𝒮, s̄, a, b, σ₀)
@inline subgrid_cloud_fraction(correlation, 𝒮, s̄, a, b, σ₀) = subgrid_cloud(correlation, 𝒮, s̄, a, b, σ₀).C

# The subgrid statistics 𝒮 and mean saturation deficit of cell (i, j, k), from the same state
# `_p3_add_tendencies_kernel!` and `compute_p3_process_rates` build.
@inline function p3_cell_saturation_deficit(i, j, k, grid, d::P3CloudFractionKernelFunction)
    p3 = d.microphysics
    μ = d.microphysical_fields
    constants = d.thermodynamic_constants

    @inbounds begin
        ρ = d.density[i, j, k]
        qᵛᵉ = μ.qᵛ[i, j, k]
    end
    ℳ = AM.grid_microphysical_state(i, j, k, grid, p3, μ, ρ, nothing, d.velocities)
    q = AM.moisture_fractions(p3, ℳ, qᵛᵉ)
    𝒰₀ = AM.diagnose_thermodynamic_state(i, j, k, grid, d.formulation, d.dynamics, q)
    𝒰 = AM.maybe_adjust_thermodynamic_state(𝒰₀, p3, qᵛᵉ, constants)

    T = temperature(𝒰, constants)
    ρᵗʰ = density(𝒰, constants)
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρᵗʰ, constants, PlanarLiquidSurface())
    adjusted = prognostic_vapor_excess_adjustment(p3, ℳ.qᶜˡ, qᵛ⁺ˡ, ℳ.δqᵛ⁺ˡ, T, ρᵗʰ, 𝒰, constants)

    deficit = p3_saturation_deficit(p3.subgrid, adjusted.qᵛ, adjusted.qᵛ⁺ˡ, max(0, adjusted.qᶜˡ),
                                    adjusted.T, 𝒰, constants)
    𝒮 = AM.subgrid_statisticsᶜᶜᶜ(i, j, k, grid, p3.subgrid, d.closure, d.closure_fields, d.subgrid_state)

    return (; 𝒮, deficit...)
end

@inline function (d::P3CloudFractionKernelFunction)(i, j, k, grid)
    cell = p3_cell_saturation_deficit(i, j, k, grid, d)
    return subgrid_cloud_fraction(d.microphysics.subgrid, cell.𝒮, cell.s̄, cell.a, cell.b, cell.σ₀)
end
