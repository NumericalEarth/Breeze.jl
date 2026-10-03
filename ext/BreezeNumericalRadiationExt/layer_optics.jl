#####
##### Per-layer optics of one staged column
#####
##### The streaming solvers of NumericalRadiation take the optics of a column as a functor
##### `(g, k) -> ...` evaluated per g point and layer. The two functors below read the staged
##### column arrays of one column and call NumericalRadiation's scalar layer API: the gas optics
##### stencil and Planck brackets are read back from the arrays kernel C fills once per layer, the
##### well-mixed gases are formed from their mole fractions relative to dry air, and cloud phases
##### add their optical depths (a `nothing` phase adds nothing). Everything here runs inside the
##### column kernel: `@inline`, allocation-free, and branching only on types, never on data.
#####

#####
##### Stencil and Planck bracket storage
#####

"""
$(TYPEDSIGNATURES)

Store the six scalars of the gas optics `stencil` of layer `k` of column `c` in `columns`;
nothing to store for a model without interpolation tables (`stencil::Nothing`).
"""
@inline function store_layer_stencil!(columns::SpectralColumns, c, k, stencil::GasOpticsStencil)
    @inbounds begin
        columns.pressure_index[c, k] = stencil.pressure[1]
        columns.pressure_weight[c, k] = stencil.pressure[3]
        columns.temperature_index[c, k] = stencil.temperature[1]
        columns.temperature_weight[c, k] = stencil.temperature[3]
        columns.water_vapor_index[c, k] = stencil.water_vapor[1]
        columns.water_vapor_weight[c, k] = stencil.water_vapor[3]
    end
    return nothing
end

@inline store_layer_stencil!(columns::SpectralColumns, c, k, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

The gas optics stencil of layer `k` of column `c`, rebuilt from the six scalars stored by
[`store_layer_stencil!`](@ref); `nothing` for a model without interpolation tables.
"""
@inline function layer_stencil(::EcCKDTabulatedGasOpticsModel, columns::SpectralColumns, c, k)
    @inbounds stencil = GasOpticsStencil(columns.pressure_index[c, k], columns.pressure_weight[c, k],
                                         columns.temperature_index[c, k], columns.temperature_weight[c, k],
                                         columns.water_vapor_index[c, k], columns.water_vapor_weight[c, k])
    return stencil
end

@inline layer_stencil(::EcCKDGasOpticsModel, columns::SpectralColumns, c, k) = nothing

"""
$(TYPEDSIGNATURES)

Store the Planck source-table `bracket` of interface `k` of column `c` in `columns`; nothing to
store for a model without a source table (`bracket::Nothing`).
"""
@inline function store_source_bracket!(columns::SpectralColumns, c, k, bracket::Tuple)
    @inbounds begin
        columns.source_index[c, k] = bracket[1]
        columns.source_weight[c, k] = bracket[3]
    end
    return nothing
end

@inline store_source_bracket!(columns::SpectralColumns, c, k, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

The Planck source-table bracket of interface `k` of column `c`, rebuilt from the index and
weight stored by [`store_source_bracket!`](@ref) (the upper index is the lower one plus one, as
NumericalRadiation's `bracket` always returns); `nothing` for a model without a source table,
whose source is the gray `σT⁴`.
"""
@inline function interface_source_bracket(model::EcCKDTabulatedGasOpticsModel, columns::SpectralColumns, c, k)
    # Decided by the model type, so this folds away
    model.longwave_source_table === nothing && return nothing
    @inbounds begin
        i = Int(columns.source_index[c, k])
        w = columns.source_weight[c, k]
    end
    return (i, i + 1, w)
end

@inline interface_source_bracket(::EcCKDGasOpticsModel, columns::SpectralColumns, c, k) = nothing

"""
$(TYPEDSIGNATURES)

Fill the gas optics stencils of the `N` layers and the Planck brackets of the `N + 1`
interfaces of column `c` from its staged pressure, temperature and gas amounts. The H₂O mole
fraction of the stencil is the water vapor amount over the dry-air amount, guarded as in
NumericalRadiation's array path.
"""
@inline function stage_column_stencils!(columns::SpectralColumns, model, c, N)
    FT = eltype(columns)
    @inbounds for k in 1:N
        p = columns.pressure_layers[c, k]
        T = columns.temperature_layers[c, k]
        dry_air_moles = max(columns.dry_air[c, k], sqrt(eps(FT)))
        water_vapor_moles = max(0, columns.water_vapor[c, k])
        store_layer_stencil!(columns, c, k, gas_optics_stencil(model, p, T, water_vapor_moles / dry_air_moles))
    end
    @inbounds for k in 1:N+1
        T = columns.temperature_interfaces[c, k]
        store_source_bracket!(columns, c, k, source_table_bracket(model, T))
    end
    return nothing
end

#####
##### Gas amounts and cloud phases of one layer
#####

"""
$(TYPEDSIGNATURES)

The scalar gas amounts (mol m⁻²) of layer `k` of column `c` as a `NamedTuple` keyed by
`ECCKD_GAS_NAMES`: the staged dry air, water vapor and ozone, and the well-mixed gases as
`mole_fractions` times the dry air.
"""
@inline function layer_gas_amounts(columns::SpectralColumns, mole_fractions, c, k)
    @inbounds begin
        dry_air_moles = columns.dry_air[c, k]
        water_vapor_moles = columns.water_vapor[c, k]
        ozone_moles = columns.ozone[c, k]
    end
    χ = mole_fractions
    return (composite = dry_air_moles,
            h2o = water_vapor_moles,
            o3 = ozone_moles,
            co2 = χ.co2 * dry_air_moles,
            ch4 = χ.ch4 * dry_air_moles,
            n2o = χ.n2o * dry_air_moles,
            cfc11 = χ.cfc11 * dry_air_moles,
            cfc12 = χ.cfc12 * dry_air_moles)
end

# The two cloud phases of a cloud optics container: `nothing` for clear sky. Each phase is folded
# into the layer optics by NumericalRadiation's `add_cloud_scattering_layer` (shortwave) and
# `cloud_absorption_optical_depth` (longwave), whose `Nothing` methods leave the layer untouched,
# so the clear-sky and all-sky kernels are the same code with no branch.
@inline cloud_phases(::Nothing) = (nothing, nothing)
@inline cloud_phases(cloud::NamedTuple) = (cloud.liquid, cloud.ice)

#####
##### Layer optics functors
#####

"""
$(TYPEDEF)

Longwave layer optics of column `column` of `columns` for `NumericalRadiation.streaming_longwave_fluxes!`:
`(g, k)` returns `(τ, Bₖ, Bₖ₊₁)`, the gas absorption optical depth of layer `k` at
g point `g` plus the cloud absorption of both phases, and the Planck sources at the layer's
top and bottom interfaces.

Fields:
- `gas_model`: ecCKD gas optics model
- `columns`: The staged columns
- `mole_fractions`: Mole fractions of the well-mixed gases relative to dry air
- `liquid_cloud`: Longwave liquid cloud optics, or `nothing`
- `ice_cloud`: Longwave ice cloud optics, or `nothing`
- `liquid_bracket`: Effective radius bracket of the liquid cloud optics
- `ice_bracket`: Effective radius bracket of the ice cloud optics
- `column`: Column index `c = i + (j - 1) Nx`
"""
struct LongwaveLayerOptics{M, C, L, I, BL, BI, FT}
    gas_model :: M
    columns :: C
    mole_fractions :: NamedTuple{(:co2, :ch4, :n2o, :cfc11, :cfc12), NTuple{5, FT}}
    liquid_cloud :: L
    ice_cloud :: I
    liquid_bracket :: BL
    ice_bracket :: BI
    column :: Int
end

@inline function (optics::LongwaveLayerOptics)(g, k)
    columns = optics.columns
    model = optics.gas_model
    c = optics.column

    gases = layer_gas_amounts(columns, optics.mole_fractions, c, k)
    stencil = layer_stencil(model, columns, c, k)

    @inbounds begin
        Tₖ = columns.temperature_interfaces[c, k]
        Tₖ₊₁ = columns.temperature_interfaces[c, k+1]
        Wˡ = columns.liquid_water_path[c, k]
        Wⁱ = columns.ice_water_path[c, k]
    end

    top_source_bracket = interface_source_bracket(model, columns, c, k)
    bottom_source_bracket = interface_source_bracket(model, columns, c, k + 1)

    τ = longwave_optical_depth(model, g, gases, stencil) +
        cloud_absorption_optical_depth(optics.liquid_cloud, g, optics.liquid_bracket, Wˡ) +
        cloud_absorption_optical_depth(optics.ice_cloud, g, optics.ice_bracket, Wⁱ)

    Bₖ = longwave_source(model, g, Tₖ, top_source_bracket)
    Bₖ₊₁ = longwave_source(model, g, Tₖ₊₁, bottom_source_bracket)

    return τ, Bₖ, Bₖ₊₁
end

"""
$(TYPEDEF)

Shortwave layer optics of column `column` of `columns` for `NumericalRadiation.streaming_shortwave_fluxes!`:
`(g, k)` returns `(τₐ, τₛ, 𝒢)`, the gas absorption optical depth
of layer `k` at g point `g`, the Rayleigh scattering of the layer's air (the composite
amount, which in the dry convention of the staging kernels is the layer's total mass over `mᵈ`,
as in NumericalRadiation's array path), and the scattering of both cloud phases folded in.

Fields:
- `gas_model`: ecCKD gas optics model
- `columns`: The staged columns
- `mole_fractions`: Mole fractions of the well-mixed gases relative to dry air
- `liquid_cloud`: Shortwave liquid cloud optics, or `nothing`
- `ice_cloud`: Shortwave ice cloud optics, or `nothing`
- `liquid_bracket`: Effective radius bracket of the liquid cloud optics
- `ice_bracket`: Effective radius bracket of the ice cloud optics
- `column`: Column index `c = i + (j - 1) Nx`
"""
struct ShortwaveLayerOptics{M, C, L, I, BL, BI, FT}
    gas_model :: M
    columns :: C
    mole_fractions :: NamedTuple{(:co2, :ch4, :n2o, :cfc11, :cfc12), NTuple{5, FT}}
    liquid_cloud :: L
    ice_cloud :: I
    liquid_bracket :: BL
    ice_bracket :: BI
    column :: Int
end

@inline function (optics::ShortwaveLayerOptics)(g, k)
    columns = optics.columns
    model = optics.gas_model
    c = optics.column

    gases = layer_gas_amounts(columns, optics.mole_fractions, c, k)
    stencil = layer_stencil(model, columns, c, k)

    @inbounds begin
        Wˡ = columns.liquid_water_path[c, k]
        Wⁱ = columns.ice_water_path[c, k]
    end

    τₐ = shortwave_optical_depth(model, g, gases, stencil)
    τₛ = rayleigh_optical_depth(model, g, gases.composite)
    𝒢 = zero(τₛ)

    τₐ, τₛ, 𝒢 = add_cloud_scattering_layer(τₐ, τₛ, 𝒢, optics.liquid_cloud, g, optics.liquid_bracket, Wˡ)
    τₐ, τₛ, 𝒢 = add_cloud_scattering_layer(τₐ, τₛ, 𝒢, optics.ice_cloud, g, optics.ice_bracket, Wⁱ)

    return τₐ, τₛ, 𝒢
end
