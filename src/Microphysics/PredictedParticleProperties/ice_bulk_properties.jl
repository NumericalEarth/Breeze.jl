#####
##### Ice Bulk Properties
#####
##### Population-averaged properties computed by integrating over the
##### ice particle size distribution.
#####

"""
    IceBulkProperties

Population-averaged ice properties and diagnostic integrals.
See [`IceBulkProperties`](@ref) constructor for details.
"""
struct IceBulkProperties{FT, EF, DM, RH, RF, LA, MU, SH}
    maximum_mean_diameter :: FT
    minimum_mean_diameter :: FT
    effective_radius :: EF
    mean_diameter :: DM
    mean_density :: RH
    reflectivity :: RF
    slope :: LA
    shape :: MU
    shedding :: SH
end

"""
$(TYPEDSIGNATURES)

Construct `IceBulkProperties` with parameters and quadrature-based integrals.

These integrals compute bulk properties by averaging over the particle 
size distribution. They are used for radiation, radar, and diagnostics.

**Diagnostic integrals:**

- `effective_radius`: Radiation-weighted radius ``r_e = ∫A·N'dD / ∫N'dD``
- `mean_diameter`: Mass-weighted diameter ``D_m = ∫D·m·N'dD / ∫m·N'dD``
- `mean_density`: Mass-weighted density ``ρ̄ = ∫ρ·m·N'dD / ∫m·N'dD``
- `reflectivity`: Radar reflectivity ``Z = ∫D^6·N'dD``

**Distribution parameters (for λ-limiting):**

- `slope`: Slope parameter λ from prognostic constraints
- `shape`: Shape parameter μ from empirical μ-λ relationship

**Process integrals:**

- `shedding`: Rate at which meltwater sheds from large particles

# Keyword Arguments

- `maximum_mean_diameter`: Upper Dm limit [m], default 0.02 (2 cm)
- `minimum_mean_diameter`: Lower Dm limit [m], default 1×10⁻⁵ (10 μm)

# References

[Morrison and Milbrandt (2015a)](@citet Morrison2015parameterization),
[Field et al. (2007)](@citet FieldEtAl2007) for μ-λ relationship.
"""
function IceBulkProperties(FT::Type{<:AbstractFloat} = Float64;
                           maximum_mean_diameter = 2e-2,
                           minimum_mean_diameter = 1e-5)
    return IceBulkProperties(
        FT(maximum_mean_diameter),
        FT(minimum_mean_diameter),
        EffectiveRadius(),
        MeanDiameter(),
        MeanDensity(),
        Reflectivity(),
        SlopeParameter(),
        ShapeParameter(),
        SheddingRate()
    )
end

Base.summary(::IceBulkProperties) = "IceBulkProperties"

function Base.show(io::IO, bp::IceBulkProperties)
    print(io, summary(bp), "(")
    print(io, "Dmax=", bp.maximum_mean_diameter, ", ")
    print(io, "Dmin=", bp.minimum_mean_diameter, ")")
end

