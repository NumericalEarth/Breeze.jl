"""
    BulkMicrophysics{N, C, B}

Bulk microphysics scheme with cloud formation and precipitation categories.

# Fields
- `cloud_formation`: Cloud formation scheme (saturation adjustment or non-equilibrium)
- `categories`: Precipitation categories (e.g., rain, snow) or `nothing`
- `precipitation_boundary_condition`: Bottom boundary condition for precipitation sedimentation.
  - `nothing` (default): Precipitation passes through the bottom (open boundary)
  - `ImpenetrableBoundaryCondition()`: Precipitation collects at the bottom (zero terminal velocity at surface)
"""
struct BulkMicrophysics{N, C, B}
    cloud_formation :: N
    categories :: C
    precipitation_boundary_condition :: B
end

Base.summary(bμp::BulkMicrophysics) = "BulkMicrophysics"

struct NonEquilibriumCloudFormation{L, I}
    liquid :: L
    ice :: I

    @doc"""
        NonEquilibriumCloudFormation(liquid, ice=nothing)

    A cloud formation scheme where cloud liquid and ice are prognostic variables
    that evolve via condensation/evaporation and deposition/sublimation tendencies,
    rather than being diagnosed instantaneously via saturation adjustment.

    The condensation/evaporation and deposition/sublimation tendencies are commonly modeled as **relaxation toward
    saturation** with timescale `τ_relax`, including a latent-heat (psychrometric/thermal) correction factor; see
    [Morrison and Grabowski (2008)](@cite Morrison2008novel), Appendix Eq. (A3), and standard cloud microphysics
    texts such as [Pruppacher and Klett (2010)](@cite pruppacher2010microphysics) or
    [Rogers and Yau (1989)](@cite rogers1989short).

    # Fields
    - `liquid`: Parameters for cloud liquid (contains relaxation timescale `τ_relax`)
    - `ice`: Parameters for cloud ice (contains relaxation timescale `τ_relax`), or `nothing` for warm-phase only

    # References
    * Morrison, H. and Grabowski, W. W. (2008). A novel approach for representing ice
        microphysics in models: Description and tests using a kinematic framework.
        J. Atmos. Sci., 65, 1528–1548. https://doi.org/10.1175/2007JAS2491.1
    * Pruppacher, H. R. and Klett, J. D. (2010). Microphysics of Clouds and Precipitation (2nd ed.).
    * Rogers, R. R. and Yau, M. K. (1989). A Short Course in Cloud Physics (3rd ed.).
    """
    function NonEquilibriumCloudFormation(liquid, ice=nothing)
        return new{typeof(liquid), typeof(ice)}(liquid, ice)
    end
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

const FourCategoryBulkMicrophysics = BulkMicrophysics{<:Any, <:FourCategories, <:Any}
Base.summary(bμp::FourCategoryBulkMicrophysics) = "FourCategoryBulkMicrophysics"

"""
$(TYPEDSIGNATURES)

Return a `BulkMicrophysics` microphysics scheme.

# Keyword arguments
- `categories`: Precipitation categories (e.g., rain, snow) or `nothing` for non-precipitating
- `cloud_formation`: Cloud formation scheme (default: `SaturationAdjustment`)
- `precipitation_boundary_condition`: Bottom boundary condition for precipitation sedimentation.
  - `nothing` (default): Precipitation passes through the bottom
  - `ImpenetrableBoundaryCondition()`: Precipitation collects at the bottom
"""
function BulkMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                          categories = nothing,
                          cloud_formation = SaturationAdjustment(FT),
                          precipitation_boundary_condition = nothing)

    return BulkMicrophysics(cloud_formation, categories, precipitation_boundary_condition)
end

# Non-categorical bulk microphysics
const NCBM = BulkMicrophysics{<:Any, Nothing, <:Any}
const NPBM = NCBM  # Alias: Non-Precipitating Bulk Microphysics

maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::NCBM, ρᵣ, microphysical_fields, qᵗ, constants) =
    AtmosphereModels.adjust_thermodynamic_state(𝒰₀, bμp.cloud_formation, constants)

AtmosphereModels.prognostic_field_names(::NPBM) = tuple()
AtmosphereModels.materialize_microphysical_fields(bμp::NPBM, grid, bcs) = materialize_microphysical_fields(bμp.cloud_formation, grid, bcs)

@inline function AtmosphereModels.update_microphysical_fields!(μ, bμp::NPBM, i, j, k, grid, ρ, 𝒰, constants)
    return update_microphysical_fields!(μ, bμp.cloud_formation, i, j, k, grid, ρ, 𝒰, constants)
end

@inline function AtmosphereModels.compute_moisture_fractions(i, j, k, grid, bμp::NPBM, ρ, qᵗ, μ)
    return compute_moisture_fractions(i, j, k, grid, bμp.cloud_formation, ρ, qᵗ, μ)
end
