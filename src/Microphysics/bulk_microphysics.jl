using Oceananigans.Utils: prettysummary

struct BulkMicrophysics{N, C}
    nucleation :: N
    categories :: C
end

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
function BulkMicrophysics(FT::DataType = Oceananigans.defaults.FloatType,
                          categories = nothing,
                          nucleation = SaturationAdjustment(FT))

    return BulkMicrophysics(nucleation, categories)
end

function compute_thermodynamic_state(𝒰₀::AbstractThermodynamicState, bμp::BulkMicrophysics, thermo)
    return compute_thermodynamic_state(𝒰₀, bμp.nucleation, thermo)
end

const NPBM = BulkMicrophysics{<:Any, Nothing}

prognostic_field_names(::NPBM) = tuple()
materialize_microphysical_fields(bμp::NPBM, grid, bcs) = materialize_microphysical_fields(bμp.nucleation, grid, bcs)

@inline function update_microphysical_fields!(μ, bμp::NPBM, i, j, k, grid, density, 𝒰, thermo)
    return update_microphysical_fields!(μ, bμp.nucleation, i, j, k, grid, density, 𝒰, thermo)
end
    
@inline function moisture_mass_fractions(i, j, k, grid, bμp::NPBM, density, qᵗ, μ)
    return moisture_mass_fractions(i, j, k, grid, bμp.nucleation, density, qᵗ, μ)
end


# Stub functions defined in extensions
function OneMomentCloudMoisture end
function TwoClassPrecipitation end