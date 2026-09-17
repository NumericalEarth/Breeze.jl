#####
##### Per-layer optics of one staged column
#####
##### The streaming solvers of NumericalRadiation take the optics of a column as a functor
##### `(ig, k) -> ...` evaluated per g point and layer. The two functors below read the staged
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
        columns.stencil_ip[c, k] = stencil.pressure[1]
        columns.stencil_wp[c, k] = stencil.pressure[3]
        columns.stencil_it[c, k] = stencil.temperature[1]
        columns.stencil_wt[c, k] = stencil.temperature[3]
        columns.stencil_ih[c, k] = stencil.h2o[1]
        columns.stencil_wh[c, k] = stencil.h2o[3]
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
    @inbounds stencil = GasOpticsStencil(columns.stencil_ip[c, k], columns.stencil_wp[c, k],
                                         columns.stencil_it[c, k], columns.stencil_wt[c, k],
                                         columns.stencil_ih[c, k], columns.stencil_wh[c, k])
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
        columns.source_is[c, k] = bracket[1]
        columns.source_ws[c, k] = bracket[3]
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
        i = Int(columns.source_is[c, k])
        w = columns.source_ws[c, k]
    end
    return (i, i + 1, w)
end

@inline interface_source_bracket(::EcCKDGasOpticsModel, columns::SpectralColumns, c, k) = nothing

"""
$(TYPEDSIGNATURES)

Fill the gas optics stencils of the `N` layers and the Planck brackets of the `N + 1`
interfaces of column `c` from its staged pressure, temperature and gas amounts. The H₂O mole
fraction of the stencil is `n_h2o / n_dry`, guarded as in NumericalRadiation's array path.
"""
@inline function stage_column_stencils!(columns::SpectralColumns, model, c, N)
    FT = eltype(columns)
    @inbounds for k in 1:N
        p = columns.pressure_layers[c, k]
        T = columns.temperature_layers[c, k]
        n_dry = max(columns.dry_air[c, k], sqrt(eps(FT)))
        n_h2o = max(columns.water_vapor[c, k], zero(FT))
        store_layer_stencil!(columns, c, k, gas_optics_stencil(model, p, T, n_h2o / n_dry))
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
        n_dry = columns.dry_air[c, k]
        n_h2o = columns.water_vapor[c, k]
        n_o3 = columns.ozone[c, k]
    end
    χ = mole_fractions
    return (composite = n_dry,
            h2o = n_h2o,
            o3 = n_o3,
            co2 = χ.co2 * n_dry,
            ch4 = χ.ch4 * n_dry,
            n2o = χ.n2o * n_dry,
            cfc11 = χ.cfc11 * n_dry,
            cfc12 = χ.cfc12 * n_dry)
end

# The two cloud phases of a cloud optics container: `nothing` for clear sky
@inline cloud_phases(::Nothing) = (nothing, nothing)
@inline cloud_phases(cloud::NamedTuple) = (cloud.liquid, cloud.ice)

# Fold a cloud phase into the shortwave scattering optics of a layer; a `nothing` phase leaves
# the layer untouched
@inline add_cloud_scattering(::Nothing, ig, radius_bracket, water_path, τ_absorption, τ_scattering, asymmetry) =
    (τ_absorption, τ_scattering, asymmetry)

@inline function add_cloud_scattering(cloud::SpectralCloudOptics, ig, radius_bracket, water_path,
                                      τ_absorption, τ_scattering, asymmetry)
    κ, ω, g = cloud_layer_optics(cloud, ig, radius_bracket)
    return add_scattering_layer(τ_absorption, τ_scattering, asymmetry, κ, ω, g, water_path)
end

#####
##### Layer optics functors
#####

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Longwave layer optics of column `column` of `columns` for `NumericalRadiation.streaming_longwave_fluxes!`:
`(ig, k)` returns `(τ, B_top, B_bottom)`, the gas absorption optical depth of layer `k` at
g point `ig` plus the cloud absorption of both phases, and the Planck sources at the layer's
top and bottom interfaces.
"""
struct LongwaveLayerOptics{M, C, L, I, BL, BI, FT}
    "ecCKD gas optics model"
    gas_model :: M
    "The staged columns"
    columns :: C
    "Mole fractions of the well-mixed gases relative to dry air"
    mole_fractions :: NamedTuple{(:co2, :ch4, :n2o, :cfc11, :cfc12), NTuple{5, FT}}
    "Longwave liquid cloud optics, or `nothing`"
    liquid_cloud :: L
    "Longwave ice cloud optics, or `nothing`"
    ice_cloud :: I
    "Effective radius bracket of the liquid cloud optics"
    liquid_bracket :: BL
    "Effective radius bracket of the ice cloud optics"
    ice_bracket :: BI
    "Column index `c = i + (j - 1) Nx`"
    column :: Int
end

@inline function (optics::LongwaveLayerOptics)(ig, k)
    columns = optics.columns
    model = optics.gas_model
    c = optics.column

    gases = layer_gas_amounts(columns, optics.mole_fractions, c, k)
    stencil = layer_stencil(model, columns, c, k)

    @inbounds begin
        T_top = columns.temperature_interfaces[c, k]
        T_bottom = columns.temperature_interfaces[c, k+1]
        liquid_water_path = columns.liquid_water_path[c, k]
        ice_water_path = columns.ice_water_path[c, k]
    end

    bracket_top = interface_source_bracket(model, columns, c, k)
    bracket_bottom = interface_source_bracket(model, columns, c, k + 1)

    τ = longwave_optical_depth(model, ig, gases, stencil) +
        cloud_absorption_optical_depth(optics.liquid_cloud, ig, optics.liquid_bracket, liquid_water_path) +
        cloud_absorption_optical_depth(optics.ice_cloud, ig, optics.ice_bracket, ice_water_path)

    B_top = longwave_source(model, ig, T_top, bracket_top)
    B_bottom = longwave_source(model, ig, T_bottom, bracket_bottom)

    return τ, B_top, B_bottom
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Shortwave layer optics of column `column` of `columns` for `NumericalRadiation.streaming_shortwave_fluxes!`:
`(ig, k)` returns `(τ_absorption, τ_scattering, asymmetry)`, the gas absorption optical depth
of layer `k` at g point `ig`, the Rayleigh scattering of the layer's air (dry air plus water
vapor), and the scattering of both cloud phases folded in.
"""
struct ShortwaveLayerOptics{M, C, L, I, BL, BI, FT}
    "ecCKD gas optics model"
    gas_model :: M
    "The staged columns"
    columns :: C
    "Mole fractions of the well-mixed gases relative to dry air"
    mole_fractions :: NamedTuple{(:co2, :ch4, :n2o, :cfc11, :cfc12), NTuple{5, FT}}
    "Shortwave liquid cloud optics, or `nothing`"
    liquid_cloud :: L
    "Shortwave ice cloud optics, or `nothing`"
    ice_cloud :: I
    "Effective radius bracket of the liquid cloud optics"
    liquid_bracket :: BL
    "Effective radius bracket of the ice cloud optics"
    ice_bracket :: BI
    "Column index `c = i + (j - 1) Nx`"
    column :: Int
end

@inline function (optics::ShortwaveLayerOptics)(ig, k)
    columns = optics.columns
    model = optics.gas_model
    c = optics.column

    gases = layer_gas_amounts(columns, optics.mole_fractions, c, k)
    stencil = layer_stencil(model, columns, c, k)

    @inbounds begin
        liquid_water_path = columns.liquid_water_path[c, k]
        ice_water_path = columns.ice_water_path[c, k]
    end

    τ_absorption = shortwave_optical_depth(model, ig, gases, stencil)
    τ_scattering = rayleigh_optical_depth(model, ig, gases.composite + gases.h2o)
    asymmetry = zero(τ_scattering)

    τ_absorption, τ_scattering, asymmetry =
        add_cloud_scattering(optics.liquid_cloud, ig, optics.liquid_bracket, liquid_water_path,
                             τ_absorption, τ_scattering, asymmetry)

    τ_absorption, τ_scattering, asymmetry =
        add_cloud_scattering(optics.ice_cloud, ig, optics.ice_bracket, ice_water_path,
                             τ_absorption, τ_scattering, asymmetry)

    return τ_absorption, τ_scattering, asymmetry
end
