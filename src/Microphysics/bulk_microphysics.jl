
struct BulkMicrophysics{C, P}
    clouds :: C
    precipitation :: P
end

"""
    $(TYPEDSIGNATURES)

Return a `BulkMicrophysics` microphysics scheme with `clouds` and `precipitation` microphysics schemes.
"""
function BulkMicrophysics(FT::DataType = Oceananigans.defaults.FloatType,
                          clouds = SaturationAdjustment(FT),
                          precipitation = nothing)

    return BulkMicrophysics(clouds, precipitation)
end

function compute_thermodynamic_state(𝒰₀::AbstractThermodynamicState, bμp::BulkMicrophysics, thermo)
    return compute_thermodynamic_state(𝒰₀, bμp.clouds, thermo)
end

const NPBM = BulkMicrophysics{<:Any, Nothing}

prognostic_field_names(::NPBM) = tuple()
materialize_microphysical_fields(bμp::NPBM, grid, bcs) = materialize_microphysical_fields(bμp.clouds, grid, bcs)

@inline function update_microphysical_fields!(μ, bμp::NPBM, i, j, k, grid, density, 𝒰, thermo)
    return update_microphysical_fields!(μ, bμp.clouds, i, j, k, grid, density, 𝒰, thermo)
end
    
@inline function moisture_mass_fractions(i, j, k, grid, bμp::NPBM, density, qᵗ, μ)
    return moisture_mass_fractions(i, j, k, grid, bμp.clouds, density, qᵗ, μ)
end