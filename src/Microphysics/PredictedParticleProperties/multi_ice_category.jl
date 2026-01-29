#####
##### Multi-Ice Category Support
#####
##### Extension of P3 to support multiple free ice categories
##### per Milbrandt and Morrison (2016).
#####

export MultiIceCategory

"""
    MultiIceCategory{N, ICE}

Container for multiple ice categories in P3-MC (multi-category P3).

Following [Milbrandt and Morrison (2016)](@citet MilbrandtMorrison2016), each
ice category evolves independently with its own rime fraction, rime density,
liquid fraction, and 3-moment ice state. Categories interact through
ice-ice collection (aggregation between categories).

# Type Parameters
- `N`: Number of ice categories (compile-time constant)
- `ICE`: Type of individual ice property containers

# Fields
- `categories`: NTuple of N `IceProperties` instances, one per category

# Prognostic Fields per Category

For a simulation with N ice categories, the prognostic fields are:

| Field | Description |
|-------|-------------|
| `ρqⁱ_n` | Ice mass for category n |
| `ρnⁱ_n` | Ice number for category n |
| `ρqᶠ_n` | Rime mass for category n |
| `ρbᶠ_n` | Rime volume for category n |
| `ρzⁱ_n` | Ice 6th moment for category n |
| `ρqʷⁱ_n` | Liquid on ice for category n |

where n = 1, 2, ..., N.

# References

[Milbrandt and Morrison (2016)](@cite MilbrandtMorrison2016) Part III:
Multiple ice categories.
"""
struct MultiIceCategory{N, ICE}
    categories :: NTuple{N, ICE}
end

"""
$(TYPEDSIGNATURES)

Construct a multi-category ice container with N identical ice categories.

# Arguments
- `n_categories`: Number of ice categories (default 2)
- `FT`: Floating point type (default Float64)

# Example

```julia
multi_ice = MultiIceCategory(3, Float64)  # 3 ice categories
```
"""
function MultiIceCategory(n_categories::Int = 2, FT::Type{<:AbstractFloat} = Float64)
    categories = ntuple(_ -> IceProperties(FT), n_categories)
    return MultiIceCategory(categories)
end

Base.length(::MultiIceCategory{N}) where N = N
Base.getindex(mic::MultiIceCategory, i::Int) = mic.categories[i]

Base.summary(::MultiIceCategory{N}) where N = "MultiIceCategory{$N}"

function Base.show(io::IO, mic::MultiIceCategory{N}) where N
    print(io, summary(mic), " with ", N, " ice categories")
end

#####
##### Prognostic field names for multi-category ice
#####

"""
$(TYPEDSIGNATURES)

Return prognostic field names for multi-category ice.

Generates suffixed field names for each category: `:ρqⁱ_1`, `:ρqⁱ_2`, etc.
"""
function multi_category_ice_field_names(n_categories::Int)
    names = Symbol[]
    for i in 1:n_categories
        push!(names, Symbol("ρqⁱ_$i"))   # Ice mass
        push!(names, Symbol("ρnⁱ_$i"))   # Ice number
        push!(names, Symbol("ρqᶠ_$i"))   # Rime mass
        push!(names, Symbol("ρbᶠ_$i"))   # Rime volume
        push!(names, Symbol("ρzⁱ_$i"))   # Sixth moment
        push!(names, Symbol("ρqʷⁱ_$i"))  # Liquid on ice
    end
    return Tuple(names)
end

multi_category_ice_field_names(::MultiIceCategory{N}) where N = multi_category_ice_field_names(N)

#####
##### Inter-category interactions
#####

"""
    inter_category_collection(p3, cat1_state, cat2_state, T)

Compute ice-ice collection rate between two ice categories.

Following [Milbrandt and Morrison (2016)](@citet MilbrandtMorrison2016),
particles from different categories can collide and aggregate when
they have similar properties. The collection efficiency depends on
temperature (following aggregation efficiency) and the relative
velocities of particles in each category.

# Arguments
- `p3`: P3 microphysics scheme
- `cat1_state`: State of first ice category (NamedTuple with qⁱ, nⁱ, Fᶠ, etc.)
- `cat2_state`: State of second ice category
- `T`: Temperature [K]

# Returns
- NamedTuple with mass and number transfer rates between categories
"""
@inline function inter_category_collection(p3, cat1_state, cat2_state, T)
    FT = typeof(T)
    prp = p3.process_rates

    # Temperature-dependent collection efficiency (same as aggregation)
    T_low = prp.aggregation_efficiency_temperature_low
    T_high = prp.aggregation_efficiency_temperature_high
    E_max = prp.aggregation_efficiency_max

    # Linear interpolation of efficiency with temperature
    T_clamped = clamp(T, T_low, T_high)
    E = FT(0.1) + (E_max - FT(0.1)) * (T_clamped - T_low) / (T_high - T_low)

    # Collection requires particles from both categories
    qⁱ₁ = clamp_positive(cat1_state.qⁱ)
    qⁱ₂ = clamp_positive(cat2_state.qⁱ)
    nⁱ₁ = clamp_positive(cat1_state.nⁱ)
    nⁱ₂ = clamp_positive(cat2_state.nⁱ)

    # Simplified collection rate (proportional to product of concentrations)
    # Full implementation would compute differential fall speeds and collection kernel
    τ_agg = prp.aggregation_timescale

    # Mass transfer: proportional to product of ice contents
    q_product = sqrt(qⁱ₁ * qⁱ₂)
    mass_rate = E * q_product / τ_agg

    # Number transfer: smaller particles absorbed by larger category
    # Transfer number from category with smaller mean mass
    m̄₁ = safe_divide(qⁱ₁, nⁱ₁, FT(1e-12))
    m̄₂ = safe_divide(qⁱ₂, nⁱ₂, FT(1e-12))

    # Category 1 absorbs 2 if m̄₁ > m̄₂
    cat1_larger = m̄₁ > m̄₂

    # Number rate (always reduces total number)
    n_product = sqrt(nⁱ₁ * nⁱ₂)
    number_rate = E * n_product / τ_agg

    return (
        mass_rate = mass_rate,
        number_rate = number_rate,
        mass_to_cat1 = ifelse(cat1_larger, mass_rate, zero(FT)),
        mass_to_cat2 = ifelse(cat1_larger, zero(FT), mass_rate),
        number_loss_cat1 = ifelse(cat1_larger, zero(FT), number_rate),
        number_loss_cat2 = ifelse(cat1_larger, number_rate, zero(FT))
    )
end
