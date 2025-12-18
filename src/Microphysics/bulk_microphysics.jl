struct BulkMicrophysics{N, C}
    nucleation :: N
    categories :: C
end

# Bulk microphysics schemes (including those from extensions like CloudMicrophysics)
# use the standard tendency interface, so the model-wide microphysics update is a no-op.
# We forward to the nucleation / saturation-adjustment component to allow specialized
# nucleation schemes to hook into the update cycle.
AtmosphereModels.microphysics_model_update!(bμp::BulkMicrophysics, model) =
    AtmosphereModels.microphysics_model_update!(bμp.nucleation, model)

Base.summary(bμp::BulkMicrophysics) = "BulkMicrophysics"

struct FourCategories{L, I, R, S, C}
    cloud_liquid :: L
    cloud_ice :: I
    rain :: R
    snow :: S
    collisions :: C
end

const FourCategoryBulkMicrophysics = BulkMicrophysics{<:Any, <:FourCategories}
Base.summary(bμp::FourCategoryBulkMicrophysics) = "FourCategoryBulkMicrophysics"

"""
$(TYPEDSIGNATURES)

Return a `BulkMicrophysics` microphysics scheme with `clouds` and `precipitation` microphysics schemes.
"""
function BulkMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                          categories = nothing,
                          nucleation = SaturationAdjustment(FT))

    return BulkMicrophysics(nucleation, categories)
end

# Non-categorical bulk microphysics
const NCBM = BulkMicrophysics{<:Any, Nothing}
const NPBM = NCBM  # Alias: Non-Precipitating Bulk Microphysics

AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰₀, bμp::NCBM, microphysical_fields, qᵗ, constants) =
    adjust_thermodynamic_state(𝒰₀, bμp.nucleation, constants)

AtmosphereModels.prognostic_field_names(::NPBM) = tuple()
AtmosphereModels.materialize_microphysical_fields(bμp::NPBM, grid, bcs) = materialize_microphysical_fields(bμp.nucleation, grid, bcs)

@inline function AtmosphereModels.update_microphysical_fields!(μ, bμp::NPBM, i, j, k, grid, ρ, 𝒰, constants)
    return update_microphysical_fields!(μ, bμp.nucleation, i, j, k, grid, ρ, 𝒰, constants)
end

@inline function AtmosphereModels.compute_moisture_fractions(i, j, k, grid, bμp::NPBM, ρ, qᵗ, μ)
    return compute_moisture_fractions(i, j, k, grid, bμp.nucleation, ρ, qᵗ, μ)
end
