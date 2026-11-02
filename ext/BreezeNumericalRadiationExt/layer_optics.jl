#####
##### Per-layer optics of one column
#####
##### The streaming solvers of NumericalRadiation take the optics of a column as a functor
##### `(g, k) -> ...` evaluated per g point and layer. What depends only on the layer (its gas
##### optics stencil, gas amounts and water paths, and its interface temperatures) is computed
##### once per layer from the model's fields (`column_state.jl`) by the layer pass below, with one
##### work item per layer, and stored in the workspace. The functors read it back per g point and
##### call NumericalRadiation's scalar layer API: the well-mixed gases are formed from their mole
##### fractions relative to dry air, the Planck sources from the interface temperatures, and cloud
##### phases add their optical depths (a `nothing` phase adds nothing). Everything here runs inside
##### kernels: `@inline`, allocation-free, and branching only on types, never on data.
#####

"""
$(TYPEDSIGNATURES)

The scalar gas amounts (mol m⁻²) of `layer` (a [`layer_state`](@ref)) as a `NamedTuple` keyed by
`ECCKD_GAS_NAMES`: its dry air, water vapor and ozone, and the well-mixed gases as
`mole_fractions` times the dry air. The keys are the gas names of the ecCKD tables, which
NumericalRadiation looks the gases up by.
"""
@inline function layer_gas_amounts(layer, mole_fractions)
    n = layer.dry_air
    χ = mole_fractions
    return (dry_air = n,
            H₂O = layer.water_vapor,
            O₃ = layer.ozone,
            CO₂ = χ.CO₂ * n,
            CH₄ = χ.CH₄ * n,
            N₂O = χ.N₂O * n,
            CFC₁₁ = χ.CFC₁₁ * n,
            CFC₁₂ = χ.CFC₁₂ * n)
end

# The gas optics stencil of `layer`, with its H₂O mole fraction guarded as in NumericalRadiation's
# array path; `nothing` for a model without interpolation tables
@inline function layer_stencil(gas_model, layer)
    FT = typeof(layer.dry_air)
    χᵛ = max(0, layer.water_vapor) / max(layer.dry_air, sqrt(eps(FT)))
    return gas_optics_stencil(gas_model, layer.pressure, layer.temperature, χᵛ)
end

#####
##### The layer pass
#####

"""
$(TYPEDSIGNATURES)

Compute the gas optics stencil, gas amounts and water paths of every layer and the temperature of
every interface of the `batch_size = (Nx, rows)` columns of the batch of `state`, and store them in
`workspace` for the column kernel.
"""
function compute_layer_state!(workspace, state, gas_model, batch_size)
    grid = state.grid
    N = number_of_layers(workspace)
    arch = architecture(grid)
    launch!(arch, grid, (batch_size..., N), _compute_layer_state!, workspace, grid, state, gas_model)
    launch!(arch, grid, (batch_size..., N + 1), _compute_interface_temperature!, workspace, grid, state)
    return nothing
end

@kernel function _compute_layer_state!(workspace, grid, state, gas_model)
    i, jᵇ, k = @index(Global, NTuple)
    c = batch_column(i, jᵇ, grid.Nx)
    layer = layer_state(i, jᵇ + state.row_offset, k, grid, state)
    @inbounds begin
        workspace.dry_air[c, k] = layer.dry_air
        workspace.water_vapor[c, k] = layer.water_vapor
        workspace.ozone[c, k] = layer.ozone
    end
    store_water_path!(workspace.liquid_water_path, c, k, layer.liquid_water_path)
    store_water_path!(workspace.ice_water_path, c, k, layer.ice_water_path)
    store_layer_stencil!(workspace.stencil, c, k, layer_stencil(gas_model, layer))
end

@kernel function _compute_interface_temperature!(workspace, grid, state)
    i, jᵇ, k = @index(Global, NTuple)
    c = batch_column(i, jᵇ, grid.Nx)
    @inbounds workspace.interface_temperatures[c, k] = interface_temperature(i, jᵇ + state.row_offset, k, grid, state)
end

@inline store_water_path!(::Nothing, c, k, path) = nothing
@inline store_water_path!(paths, c, k, path) = @inbounds paths[c, k] = path

@inline store_layer_stencil!(::Nothing, c, k, stencil) = nothing

@inline function store_layer_stencil!(stencils::LayerStencils, c, k, stencil::GasOpticsStencil)
    @inbounds begin
        stencils.pressure_index[c, k] = stencil.pressure[1] % Int32
        stencils.pressure_weight[c, k] = stencil.pressure[3]
        stencils.temperature_index[c, k] = stencil.temperature[1] % Int32
        stencils.temperature_weight[c, k] = stencil.temperature[3]
        stencils.water_vapor_index[c, k] = stencil.water_vapor[1] % Int32
        stencils.water_vapor_weight[c, k] = stencil.water_vapor[3]
    end
    return nothing
end

#####
##### Reading the layer pass back
#####

# The gas optics stencil of layer `k` of batch column `c`, rebuilt from its six stored scalars
@inline stored_layer_stencil(c, k, ::Nothing) = nothing

@inline function stored_layer_stencil(c, k, stencils::LayerStencils)
    @inbounds stencil = GasOpticsStencil(stencils.pressure_index[c, k], stencils.pressure_weight[c, k],
                                         stencils.temperature_index[c, k], stencils.temperature_weight[c, k],
                                         stencils.water_vapor_index[c, k], stencils.water_vapor_weight[c, k])
    return stencil
end

# The gas amounts of layer `k` of batch column `c`
@inline stored_layer_gases(c, k, workspace) =
    @inbounds (dry_air = workspace.dry_air[c, k],
               water_vapor = workspace.water_vapor[c, k],
               ozone = workspace.ozone[c, k])

# A clear sky has no stored water paths, and its `nothing` cloud phases ignore them
@inline stored_water_path(c, k, ::Nothing) = false
@inline stored_water_path(c, k, paths) = @inbounds paths[c, k]

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

Layer optics of batch column `c` for NumericalRadiation's streaming solvers, read from the layer
pass's per-layer state in `workspace`. The longwave functor
returns `(τ, Bₖ, Bₖ₊₁)` at `(g, k)`: the gas absorption optical depth of layer `k` at g point `g`
plus the cloud absorption of both phases, and the Planck sources at the layer's top and bottom
interfaces. The shortwave functor returns `(τₐ, τₛ, 𝒢)`: the gas absorption optical depth, the
Rayleigh scattering of the layer's air (the dry-air amount, which in the dry convention is the
layer's total mass over `mᵈ`, as in NumericalRadiation's array path), and the scattering of both
cloud phases folded in.

Fields:
- `gas_model`: ecCKD gas optics model
- `workspace`: The [`ColumnWorkspace`](@ref)
- `mole_fractions`: Mole fractions of the well-mixed gases relative to dry air
- `liquid_cloud`, `ice_cloud`: Cloud optics of each phase, or `nothing`
- `liquid_bracket`, `ice_bracket`: Effective radius brackets of the cloud optics
- `column`: The batch column
"""
struct LayerOptics{S, M, W, L, I, BL, BI, FT}
    gas_model :: M
    workspace :: W
    mole_fractions :: NamedTuple{(:CO₂, :CH₄, :N₂O, :CFC₁₁, :CFC₁₂), NTuple{5, FT}}
    liquid_cloud :: L
    ice_cloud :: I
    liquid_bracket :: BL
    ice_bracket :: BI
    column :: Int
end

struct Longwave end
struct Shortwave end

@inline function LayerOptics(spectrum, gas_model, workspace, mole_fractions, cloud, liquid_radius, ice_radius, column)
    liquid, ice = cloud_phases(cloud)
    liquid_bracket = effective_radius_bracket(liquid, liquid_radius)
    ice_bracket = effective_radius_bracket(ice, ice_radius)
    M, W, L, I = typeof(gas_model), typeof(workspace), typeof(liquid), typeof(ice)
    BL, BI, FT = typeof(liquid_bracket), typeof(ice_bracket), eltype(mole_fractions)
    return LayerOptics{typeof(spectrum), M, W, L, I, BL, BI, FT}(gas_model, workspace, mole_fractions, liquid, ice,
                                                                 liquid_bracket, ice_bracket, column)
end

@inline function (optics::LayerOptics{Longwave})(g, k)
    gas_model = optics.gas_model
    workspace, c = optics.workspace, optics.column

    gases = layer_gas_amounts(stored_layer_gases(c, k, workspace), optics.mole_fractions)
    stencil = stored_layer_stencil(c, k, workspace.stencil)
    liquid_water_path = stored_water_path(c, k, workspace.liquid_water_path)
    ice_water_path = stored_water_path(c, k, workspace.ice_water_path)

    τ = longwave_optical_depth(gas_model, g, gases, stencil) +
        cloud_absorption_optical_depth(optics.liquid_cloud, g, optics.liquid_bracket, liquid_water_path) +
        cloud_absorption_optical_depth(optics.ice_cloud, g, optics.ice_bracket, ice_water_path)

    @inbounds begin
        Bₖ = planck_source(gas_model, g, workspace.interface_temperatures[c, k])
        Bₖ₊₁ = planck_source(gas_model, g, workspace.interface_temperatures[c, k + 1])
    end

    return τ, Bₖ, Bₖ₊₁
end

@inline function (optics::LayerOptics{Shortwave})(g, k)
    gas_model = optics.gas_model
    workspace, c = optics.workspace, optics.column

    gases = layer_gas_amounts(stored_layer_gases(c, k, workspace), optics.mole_fractions)
    stencil = stored_layer_stencil(c, k, workspace.stencil)
    liquid_water_path = stored_water_path(c, k, workspace.liquid_water_path)
    ice_water_path = stored_water_path(c, k, workspace.ice_water_path)

    τₐ = shortwave_optical_depth(gas_model, g, gases, stencil)
    τₛ = rayleigh_optical_depth(gas_model, g, gases.dry_air)
    𝒢 = zero(τₛ)

    τₐ, τₛ, 𝒢 = add_cloud_scattering_layer(τₐ, τₛ, 𝒢, optics.liquid_cloud, g, optics.liquid_bracket, liquid_water_path)
    τₐ, τₛ, 𝒢 = add_cloud_scattering_layer(τₐ, τₛ, 𝒢, optics.ice_cloud, g, optics.ice_bracket, ice_water_path)

    return τₐ, τₛ, 𝒢
end
