using Oceananigans: Average, Field, set!, compute!
using Oceananigans.Grids: Center, Face
using Oceananigans.Operators: ∂zᶜᶜᶠ, ℑzᵃᵃᶜ

#####
##### Subsidence forcing types (unmaterialized stubs)
#####

"""
    SubsidenceForcing{W}

Forcing that represents large-scale subsidence advecting horizontally-averaged
fields downward:

```math
F_\\phi = - \\rho_r w^s \\partial_z \\overline{\\phi}
```

where ``w^s`` is the subsidence velocity profile, ``\\rho_r`` is the reference density,
and ``\\overline{\\phi}`` is the horizontal average of the field being forced.

# Fields
- `wˢ`: Either a function of `z` specifying the subsidence velocity profile,
        or a `Field` containing the subsidence velocity.

The horizontal average is computed automatically during `update_state!`.

# Example

```julia
# Using a function for subsidence velocity
wˢ(z) = z < 1500 ? -0.0065 * z / 1500 : -0.0065 * (1 - (z - 1500) / 600)

subsidence = SubsidenceForcing(wˢ)

forcing = (; ρθ=subsidence, ρqᵗ=subsidence)
model = AtmosphereModel(grid; forcing)
```
"""
struct SubsidenceForcing{W}
    wˢ :: W
end

#####
##### Materialized subsidence forcing
#####

"""
    MaterializedSubsidenceForcing{W, R, A, F}

Materialized subsidence forcing containing:
- `wˢ`: Subsidence velocity field (on Face locations in z)
- `ρᵣ`: Reference density field
- `ϕ_avg`: Field storing the horizontal average (computed during `update_state!`)
- `average_operation`: The `Average` operation used to compute `ϕ_avg`
"""
struct MaterializedSubsidenceForcing{W, R, A, F}
    wˢ :: W
    ρᵣ :: R
    ϕ_avg :: A
    average_operation :: F
end

# Kernel function for subsidence forcing
@inline w_dz_ϕ(i, j, k, grid, w, ϕ) = @inbounds w[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, ϕ)

@inline function (forcing::MaterializedSubsidenceForcing)(i, j, k, grid, clock, fields)
    wˢ = forcing.wˢ
    ϕ_avg = forcing.ϕ_avg
    ρᵣ = forcing.ρᵣ
    w_dz_ϕ_avg = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, ϕ_avg)
    return @inbounds - ρᵣ[i, j, k] * w_dz_ϕ_avg
end

#####
##### Materialization function for subsidence forcing
#####

# This is called from AtmosphereModels.atmosphere_model_forcing
# The `averaged_field` is determined by the field name (e.g., :ρu → u, :ρθ → θ)
# and passed in from atmosphere_model_forcing

function materialize_subsidence_forcing(forcing::SubsidenceForcing,
                                        grid,
                                        reference_density,
                                        averaged_field)
    wˢ_input = forcing.wˢ
    ρᵣ = reference_density

    # Create or use the subsidence velocity field
    # Note: wˢ is a 1D column field (Nothing, Nothing, Face), so set! takes a function of z only
    if wˢ_input isa Function
        wˢ = Field{Nothing, Nothing, Face}(grid)
        set!(wˢ, wˢ_input)
    else
        wˢ = wˢ_input
    end

    # Create the horizontal average field and operation
    ϕ_avg = Field{Nothing, Nothing, Center}(grid)
    average_operation = Field(Average(averaged_field, dims=(1, 2)), data=ϕ_avg.data)

    return MaterializedSubsidenceForcing(wˢ, ρᵣ, ϕ_avg, average_operation)
end

#####
##### compute_forcing! for subsidence forcing
#####

"""
    compute_forcing!(forcing::MaterializedSubsidenceForcing)

Compute the horizontal average needed by the subsidence forcing.
This is called automatically during `update_state!`.
"""
function compute_forcing!(forcing::MaterializedSubsidenceForcing)
    compute!(forcing.average_operation)
    return nothing
end

# Fallback for other forcing types - do nothing
compute_forcing!(forcing) = nothing

# Handle tuples of forcings
function compute_forcing!(forcings::Tuple)
    for forcing in forcings
        compute_forcing!(forcing)
    end
    return nothing
end

