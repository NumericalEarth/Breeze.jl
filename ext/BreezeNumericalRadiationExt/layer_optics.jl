#####
##### Per-layer optics of one column
#####
##### The streaming solvers of NumericalRadiation take the optics of a column as a functor
##### `(g, k) -> ...` evaluated per g point and layer. The two functors below compute the layer's
##### state from the model's fields (`column_state.jl`) and call NumericalRadiation's scalar layer
##### API: the gas optics stencil and Planck brackets are formed from that state on the fly, the
##### well-mixed gases from their mole fractions relative to dry air, and cloud phases add their
##### optical depths (a `nothing` phase adds nothing). Everything here runs inside the column
##### kernel: `@inline`, allocation-free, and branching only on types, never on data.
#####

"""
$(TYPEDSIGNATURES)

The scalar gas amounts (mol m⁻²) of `layer` (a [`layer_state`](@ref)) as a `NamedTuple` keyed by
`ECCKD_GAS_NAMES`: its dry air, water vapor and ozone, and the well-mixed gases as
`mole_fractions` times the dry air.
"""
@inline function layer_gas_amounts(layer, mole_fractions)
    n = layer.dry_air
    χ = mole_fractions
    return (composite = n,
            h2o = layer.water_vapor,
            o3 = layer.ozone,
            co2 = χ.co2 * n,
            ch4 = χ.ch4 * n,
            n2o = χ.n2o * n,
            cfc11 = χ.cfc11 * n,
            cfc12 = χ.cfc12 * n)
end

# The gas optics stencil of `layer`, with its H₂O mole fraction guarded as in NumericalRadiation's
# array path; `nothing` for a model without interpolation tables
@inline function layer_stencil(gas_model, layer)
    FT = typeof(layer.dry_air)
    χᵛ = max(0, layer.water_vapor) / max(layer.dry_air, sqrt(eps(FT)))
    return gas_optics_stencil(gas_model, layer.pressure, layer.temperature, χᵛ)
end

# The Planck source of g point `g` at temperature `T`
@inline planck_source(gas_model, g, T) = longwave_source(gas_model, g, T, source_table_bracket(gas_model, T))

# The two cloud phases of a cloud optics container: `nothing` for clear sky. Each phase is folded
# into the layer optics by NumericalRadiation's `add_cloud_scattering_layer` (shortwave) and
# `cloud_absorption_optical_depth` (longwave), whose `Nothing` methods leave the layer untouched,
# so the clear-sky and all-sky kernels are the same code with no branch.
@inline cloud_phases(::Nothing) = (nothing, nothing)
@inline cloud_phases(cloud::NamedTuple) = (cloud.liquid, cloud.ice)

"""
$(TYPEDEF)

Layer optics of column `(i, j)` for NumericalRadiation's streaming solvers. The longwave functor
returns `(τ, Bₖ, Bₖ₊₁)` at `(g, k)`: the gas absorption optical depth of layer `k` at g point `g`
plus the cloud absorption of both phases, and the Planck sources at the layer's top and bottom
interfaces. The shortwave functor returns `(τₐ, τₛ, 𝒢)`: the gas absorption optical depth, the
Rayleigh scattering of the layer's air (the composite amount, which in the dry convention is the
layer's total mass over `mᵈ`, as in NumericalRadiation's array path), and the scattering of both
cloud phases folded in.

Fields:
- `gas_model`: ecCKD gas optics model
- `state`: The [`ColumnState`](@ref)
- `mole_fractions`: Mole fractions of the well-mixed gases relative to dry air
- `liquid_cloud`, `ice_cloud`: Cloud optics of each phase, or `nothing`
- `liquid_bracket`, `ice_bracket`: Effective radius brackets of the cloud optics
- `i`, `j`: The column
"""
struct LayerOptics{S, M, C, L, I, BL, BI, FT}
    gas_model :: M
    state :: C
    mole_fractions :: NamedTuple{(:co2, :ch4, :n2o, :cfc11, :cfc12), NTuple{5, FT}}
    liquid_cloud :: L
    ice_cloud :: I
    liquid_bracket :: BL
    ice_bracket :: BI
    i :: Int
    j :: Int
end

struct Longwave end
struct Shortwave end

@inline function LayerOptics(spectrum, gas_model, state, mole_fractions, cloud, liquid_radius, ice_radius, i, j)
    liquid, ice = cloud_phases(cloud)
    liquid_bracket = effective_radius_bracket(liquid, liquid_radius)
    ice_bracket = effective_radius_bracket(ice, ice_radius)
    M, C, L, I = typeof(gas_model), typeof(state), typeof(liquid), typeof(ice)
    BL, BI, FT = typeof(liquid_bracket), typeof(ice_bracket), eltype(mole_fractions)
    return LayerOptics{typeof(spectrum), M, C, L, I, BL, BI, FT}(gas_model, state, mole_fractions, liquid, ice,
                                                                 liquid_bracket, ice_bracket, i, j)
end

@inline function (optics::LayerOptics{Longwave})(g, k)
    gas_model = optics.gas_model
    state, i, j = optics.state, optics.i, optics.j

    layer = layer_state(state, i, j, k)
    gases = layer_gas_amounts(layer, optics.mole_fractions)
    stencil = layer_stencil(gas_model, layer)

    τ = longwave_optical_depth(gas_model, g, gases, stencil) +
        cloud_absorption_optical_depth(optics.liquid_cloud, g, optics.liquid_bracket, layer.liquid_water_path) +
        cloud_absorption_optical_depth(optics.ice_cloud, g, optics.ice_bracket, layer.ice_water_path)

    Bₖ = planck_source(gas_model, g, interface_temperature(state, i, j, k))
    Bₖ₊₁ = planck_source(gas_model, g, interface_temperature(state, i, j, k + 1))

    return τ, Bₖ, Bₖ₊₁
end

@inline function (optics::LayerOptics{Shortwave})(g, k)
    gas_model = optics.gas_model
    layer = layer_state(optics.state, optics.i, optics.j, k)
    gases = layer_gas_amounts(layer, optics.mole_fractions)

    τₐ = shortwave_optical_depth(gas_model, g, gases, layer_stencil(gas_model, layer))
    τₛ = rayleigh_optical_depth(gas_model, g, gases.composite)
    𝒢 = zero(τₛ)

    τₐ, τₛ, 𝒢 = add_cloud_scattering_layer(τₐ, τₛ, 𝒢, optics.liquid_cloud, g, optics.liquid_bracket, layer.liquid_water_path)
    τₐ, τₛ, 𝒢 = add_cloud_scattering_layer(τₐ, τₛ, 𝒢, optics.ice_cloud, g, optics.ice_bracket, layer.ice_water_path)

    return τₐ, τₛ, 𝒢
end
