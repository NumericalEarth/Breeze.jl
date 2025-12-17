struct BulkMicrophysics{N, C}
    cloud_formation :: N
    categories :: C
end

Base.summary(bμp::BulkMicrophysics) = "BulkMicrophysics"

"""
    NonEquilibriumCloudFormation{L, I}

A cloud formation scheme where cloud liquid and ice are prognostic variables
that evolve via condensation/evaporation and deposition/sublimation tendencies,
rather than being diagnosed instantaneously via saturation adjustment.

The condensation/evaporation rate follows Morrison and Milbrandt (2015),
relaxing toward saturation with timescale `τ_relax`.

# Fields
- `liquid`: Parameters for cloud liquid (contains relaxation timescale `τ_relax`)
- `ice`: Parameters for cloud ice (contains relaxation timescale `τ_relax`), or `nothing` for warm-phase only
"""
struct NonEquilibriumCloudFormation{L, I}
    liquid :: L
    ice :: I
end

Base.summary(::NonEquilibriumCloudFormation) = "NonEquilibriumCloudFormation"

struct FourCategories{L, I, R, S, C, V, A}
    cloud_liquid :: L
    cloud_ice :: I
    rain :: R
    snow :: S
    collisions :: C
    hydrometeor_velocities :: V
    air_properties :: A
end

FourCategories(cloud_liquid, cloud_ice, rain, snow, collisions, hydrometeor_velocities) =
    FourCategories(cloud_liquid, cloud_ice, rain, snow, collisions, hydrometeor_velocities, nothing)

const FourCategoryBulkMicrophysics = BulkMicrophysics{<:Any, <:FourCategories}
Base.summary(bμp::FourCategoryBulkMicrophysics) = "FourCategoryBulkMicrophysics"

"""
$(TYPEDSIGNATURES)

Return a `BulkMicrophysics` microphysics scheme with `clouds` and `precipitation` microphysics schemes.
"""
function BulkMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                          categories = nothing,
                          cloud_formation = SaturationAdjustment(FT))

    return BulkMicrophysics(cloud_formation, categories)
end

# Non-categorical bulk microphysics
const NCBM = BulkMicrophysics{<:Any, Nothing}
const NPBM = NCBM  # Alias: Non-Precipitating Bulk Microphysics

maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::NCBM, ρᵣ, microphysical_fields, qᵗ, constants) =
    adjust_thermodynamic_state(𝒰₀, bμp.cloud_formation, constants)

prognostic_field_names(::NPBM) = tuple()
materialize_microphysical_fields(bμp::NPBM, grid, bcs) = materialize_microphysical_fields(bμp.cloud_formation, grid, bcs)

@inline function update_microphysical_fields!(μ, bμp::NPBM, i, j, k, grid, ρ, 𝒰, constants)
    return update_microphysical_fields!(μ, bμp.cloud_formation, i, j, k, grid, ρ, 𝒰, constants)
end

@inline function compute_moisture_fractions(i, j, k, grid, bμp::NPBM, ρ, qᵗ, μ)
    return compute_moisture_fractions(i, j, k, grid, bμp.cloud_formation, ρ, qᵗ, μ)
end
