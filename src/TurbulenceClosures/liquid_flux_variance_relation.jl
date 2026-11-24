#####
##### The flux of liquid of the latent part of the buoyancy production from a flux–variance relation
#####
##### A component of `DoubleGaussian` (`liquid_flux = LiquidFluxVarianceRelation(...)`).
#####
##### The buoyancy production of the TKE with the energetics of the applied fluxes is the conversion of the fluxes the
##### closure applies plus its latent part Bᴸ, the buoyancy (b̄ᶜ - b̄ᵛ) of the flux of liquid the turbulence carries beyond
##### the condensate flux the closure applies, -Kᶜ ∂z qᶜ, drawn from the internal energy. By default that flux of liquid
##### is the share aᴺ of the flux of the saturation excess down its gradient (`latent_correctionᶜᶜᶠ`). With this relation
##### it is
#####
#####     w′qˡ′ = ℂʳˡ √(w′² qˡ′²),
#####
##### the correlation ℂʳˡ of w with the liquid times the spreads of the distribution, at the faces from the centers:
#####
#####     Bᴸ = (b̄ᶜ - b̄ᵛ) (ℂʳˡ √(w′² qˡ′²) + Kᶜ ∂z qᶜ) + Kᶜ ΔN²ᴸᵒ,
#####
##### with ΔN²ᴸᵒ the part of the latent correction of the ice and the liquid on ice outside the distributed water
##### (zero without a `CondensateFluxFraction`), unchanged. In large-eddy simulations of cumulus the flux of liquid
##### is up wherever there is liquid, while the gradient of the saturation excess that the default reads changes sign
##### within the cloud layer. The energy is drawn from the internal energy as before, so the budget closes. The static
##### stability is not changed.
#####

"""
$(TYPEDEF)

The flux of liquid of the latent part of the buoyancy production of the TKE as a flux–variance relation,

    w′qˡ′ = ℂʳˡ √(w′² qˡ′²),

with the vertical velocity variance and the liquid variance of the [`DoubleGaussian`](@ref) distribution at the
faces: the `liquid_flux` of the distribution. The buoyancy of this flux beyond the condensate flux the closure
applies is drawn from the internal energy. The static stability is unchanged.

The square root makes the flux sensitive to a small liquid variance: an assumed PDF that leaves a residual variance
in clear air (the smooth bounds of `SmoothClipping`, about 10⁻¹² kg² kg⁻² in a dry boundary layer) gives a buoyancy
production of order 10⁻⁵ m² s⁻³ there. [`SkewedScalarDoubleGaussianPDF`](@ref), which has no bounds, gives none.

$(TYPEDFIELDS)
"""
struct LiquidFluxVarianceRelation{FT}
    "Correlation of vertical velocity with the liquid, in [0, 1]"
    ℂʳˡ :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`LiquidFluxVarianceRelation`](@ref). The default correlation, 0.62, is the median that large-eddy
simulations of cumulus (CfSites of the CNRM-CM6-1 amip pool) imply in the cloud layer.
"""
function LiquidFluxVarianceRelation(FT::DataType = Oceananigans.defaults.FloatType; ℂʳˡ = 0.62)
    0 <= ℂʳˡ <= 1 || throw(ArgumentError("ℂʳˡ must lie in [0, 1], got $(ℂʳˡ)."))
    return LiquidFluxVarianceRelation(convert(FT, ℂʳˡ))
end

convert_eltype(::Type{FT}, r::LiquidFluxVarianceRelation) where FT = LiquidFluxVarianceRelation(convert(FT, r.ℂʳˡ))
Adapt.adapt_structure(to, r::LiquidFluxVarianceRelation) = LiquidFluxVarianceRelation(adapt(to, r.ℂʳˡ))
Base.summary(r::LiquidFluxVarianceRelation) = string("LiquidFluxVarianceRelation(ℂʳˡ = ", prettysummary(r.ℂʳˡ), ")")
Base.show(io::IO, r::LiquidFluxVarianceRelation) = print(io, summary(r))

# The fields at the faces: the latent stratification per unit Kᶜ of the applied condensate flux and of the ice outside
# the distributed water, Λᴸ = (b̄ᶜ - b̄ᵛ) ∂z qᶜ + ΔN²ᴸᵒ, the buoyancy of the flux of liquid Bˡᶠ = (b̄ᶜ - b̄ᵛ) w′qˡ′, and ΔN²ᴸᵒ
liquid_flux_fields(grid, ::Nothing) = NamedTuple()
liquid_flux_fields(grid, ::LiquidFluxVarianceRelation) = (Λᴸ = ZFaceField(grid), Bˡᶠ = ZFaceField(grid), ΔN²ᴸᵒ = ZFaceField(grid))

# The part of the latent correction outside the distributed water, stored where the relation reads it
@inline function store_outside_latent_correction!(cl, i, j, k, interior, L)
    hasproperty(cl, :ΔN²ᴸᵒ) || return nothing
    @inbounds cl.ΔN²ᴸᵒ[i, j, k] = ifelse(interior, L.outside, zero(L.outside))
    return nothing
end

# The condensate part of the latent conversion of the solved state per unit Kᶜ: that of the saturated share
# (`condensate`), or, with the relation, the removal of the applied condensate flux (`applied`), whose flux of liquid
# (`liquid_flux_buoyancy`) does not depend on the solved gradients
@inline solved_condensate_conversion(cl, i, j, k, condensate, applied) = hasproperty(cl, :Bˡᶠ) ? applied : condensate
@inline liquid_flux_buoyancy(cl, i, j, k) = hasproperty(cl, :Bˡᶠ) ? @inbounds(cl.Bˡᶠ[i, j, k]) : zero(eltype(cl.B))

@kernel function _compute_liquid_flux_conversion!(cl, grid, closure, s)
    i, j, k = @index(Global, NTuple)
    FT = grid_float_type(grid)
    ℂʳˡ = getclosure(i, j, closure).subgrid_distribution.liquid_flux.ℂʳˡ
    kˢ = max(k, 2)
    interior = (k > 1) & (k < size(grid, 3) + 1)
    @inbounds begin
        bᶜᵛ = (cl.b_qᶜ[i, j, kˢ] + cl.b_qᶜ[i, j, kˢ-1] - cl.b_qᵛ[i, j, kˢ] - cl.b_qᵛ[i, j, kˢ-1]) / 2
        Δqᶜ = distributed_condensateᶜᶜᶜ(i, j, kˢ, grid, s, nothing) - distributed_condensateᶜᶜᶜ(i, j, kˢ-1, grid, s, nothing)
        w′² = (cl.w′²[i, j, kˢ] + cl.w′²[i, j, kˢ-1]) / 2
        qˡ′² = (cl.qˡ′²[i, j, kˢ] + cl.qˡ′²[i, j, kˢ-1]) / 2
        cl.Λᴸ[i, j, k] = ifelse(interior, bᶜᵛ * Δqᶜ / Δzᶜᶜᶠ(i, j, kˢ, grid) + cl.ΔN²ᴸᵒ[i, j, k], zero(FT))
        cl.Bˡᶠ[i, j, k] = ifelse(interior, bᶜᵛ * ℂʳˡ * sqrt(w′² * qˡ′²), zero(FT))
    end
end

compute_liquid_flux_conversion!(fields, closure, model, s) =
    compute_liquid_flux_conversion!(fields, subgrid_distribution(closure), closure, model, s)

compute_liquid_flux_conversion!(fields, distribution, closure, model, s) = nothing

function compute_liquid_flux_conversion!(fields, ::DoubleGaussian{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:LiquidFluxVarianceRelation},
                                         closure, model, s)
    grid = model.grid
    launch!(grid.architecture, grid, (size(grid, 1), size(grid, 2), size(grid, 3) + 1), _compute_liquid_flux_conversion!,
            fields.cloud, grid, closure, s)
    return nothing
end
