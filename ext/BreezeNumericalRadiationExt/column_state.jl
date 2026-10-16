#####
##### The state of the extended columns, read from the model's fields
#####
##### The column solvers index layers and interfaces top-down: with `N = Nz + Nₑ` layers, column
##### layers `1:Nₑ` are the extension (extension layer `m = Nₑ + 1 - k`, counted bottom-up from the
##### grid top) and column layer `k > Nₑ` is grid cell `N + 1 - k`; column interface `k` is the top
##### of column layer `k`, so grid face `kᶠ` is column interface `N + 2 - kᶠ`. Every value is
##### computed from the model's fields (interface temperatures by interpolation of the halo-filled
##### temperature, the surface by its resolved gradient) or from the extension's profiles and
##### per-column pressures, once per layer by the layer pass of `layer_optics.jl`.
#####

"""
$(TYPEDEF)

The model fields, constants and column extension from which [`layer_state`](@ref) and
[`interface_temperature`](@ref) compute the state of the extended columns, and the batch of
columns being solved: the extension pressures are those of the batch whose first row follows
`row_offset` rows of the grid.
"""
struct ColumnState{G, P, T, D, M, MP, MF, O, X, XP, FT}
    grid :: G
    pressure :: P
    temperature :: T
    density :: D
    moisture :: M
    microphysics :: MP
    microphysical_fields :: MF
    ozone :: O
    extension :: X
    extension_pressure :: XP
    row_offset :: Int
    gravitational_acceleration :: FT
    dry_air_molar_mass :: FT
    vapor_molar_mass :: FT
    dry_air_gas_constant :: FT
end

Adapt.adapt_structure(to, state::ColumnState) =
    ColumnState(adapt(to, state.grid),
                adapt(to, state.pressure),
                adapt(to, state.temperature),
                adapt(to, state.density),
                adapt(to, state.moisture),
                adapt(to, state.microphysics),
                adapt(to, state.microphysical_fields),
                adapt(to, state.ozone),
                adapt(to, state.extension),
                adapt(to, state.extension_pressure),
                state.row_offset,
                state.gravitational_acceleration,
                state.dry_air_molar_mass,
                state.vapor_molar_mass,
                state.dry_air_gas_constant)

"""
$(TYPEDSIGNATURES)

The [`ColumnState`](@ref) of `model` with the ozone of `rtm`'s background atmosphere and the
extension and extension pressures of `rtm`'s workspace, for the batch of columns that follows
`row_offset` rows of the grid.
"""
function ColumnState(rtm, model, row_offset)
    constants = model.thermodynamic_constants
    workspace = rtm.atmospheric_state
    FT = eltype(model.grid)
    return ColumnState(model.grid,
                       dynamics_pressure(model.dynamics),
                       model.temperature,
                       total_density(model.dynamics),
                       specific_prognostic_moisture(model),
                       model.microphysics,
                       model.microphysical_fields,
                       rtm.background_atmosphere.O₃,
                       workspace.extension,
                       workspace.extension_pressure,
                       row_offset,
                       convert(FT, constants.gravitational_acceleration),
                       convert(FT, constants.dry_air.molar_mass),
                       convert(FT, constants.vapor.molar_mass),
                       convert(FT, dry_air_gas_constant(constants)))
end

@inline number_of_layers(state::ColumnState) = size(state.grid, 3) + number_of_extension_layers(state.extension)

#####
##### Layers
#####

"""
$(TYPEDSIGNATURES)

Pressure, temperature, molar amounts (mol m⁻²) of dry air, water vapor and ozone, and liquid and
ice water paths (kg m⁻²) of grid cell `k` of column `(i, j)`.

The gas amounts follow the "dry" convention of the ecCKD tables: the composite (dry-air) amount
is the layer's total mass over the dry molar mass, `ρ Δz / mᵈ` (`Δp / (g mᵈ)` for a hydrostatic
layer), and the water vapor `ρ qᵛ Δz / mᵛ` is counted on top of it, so the ratio of the two is
the mole fraction the tables are indexed by. NumericalRadiation's CKDMIP and RFMIP validation
passes only with this convention; the moist form `ρ (1 - qᵗ) Δz / mᵈ` biases the surface
downwelling longwave by -0.2 W m⁻².
"""
@inline function grid_layer_state(state, i, j, k)
    grid = state.grid
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    @inbounds begin
        ρ = state.density[i, j, k]
        p = state.pressure[i, j, k]
        T = state.temperature[i, j, k]
        χ³ = state.ozone[i, j, k]
        q = grid_moisture_fractions(i, j, k, grid, state.microphysics, ρ, state.moisture[i, j, k], state.microphysical_fields)
    end
    dry_air_moles = ρ * Δz / state.dry_air_molar_mass
    return (pressure = p,
            temperature = T,
            dry_air = dry_air_moles,
            water_vapor = ρ * max(0, q.vapor) * Δz / state.vapor_molar_mass,
            ozone = χ³ * dry_air_moles,
            liquid_water_path = ρ * max(0, q.liquid) * Δz,
            ice_water_path = ρ * max(0, q.ice) * Δz)
end

"""
$(TYPEDSIGNATURES)

The state of extension layer `m` of column `(i, j)`, in the form of [`grid_layer_state`](@ref):
the anchored temperature, the log-mean of the hydrostatic pressures on its faces, the dry-air
amount `Δp / (g mᵈ)`, the water vapor and ozone as their mole fractions times it, and no cloud.
"""
@inline function extension_layer_state(state, extension, i, j, m)
    c = batch_column(i, j - state.row_offset, state.grid.Nx)
    T = extension_layer_temperature(state, extension, i, j, m)
    mᵈ = state.dry_air_molar_mass
    @inbounds begin
        q = extension.specific_humidity[m]
        χ³ = extension.ozone[m]
        p_below = state.extension_pressure[c, m]
        p_above = state.extension_pressure[c, m+1]
    end
    χᵛ = q / (1 - q) * mᵈ / state.vapor_molar_mass   # mole fraction relative to dry air
    Δp = p_below - p_above
    dry_air_moles = hydrostatic_air_moles(Δp, state.gravitational_acceleration, mᵈ)
    return (pressure = Δp / log(p_below / p_above),
            temperature = T,
            dry_air = dry_air_moles,
            water_vapor = χᵛ * dry_air_moles,
            ozone = χ³ * dry_air_moles,
            liquid_water_path = zero(dry_air_moles),
            ice_water_path = zero(dry_air_moles))
end

"""
$(TYPEDSIGNATURES)

The state of column layer `k` (top-down) of column `(i, j)`: a grid cell or, above the grid top,
an extension layer.
"""
@inline layer_state(state, i, j, k) = layer_state(state, state.extension, i, j, k)

@inline layer_state(state, ::Nothing, i, j, k) = grid_layer_state(state, i, j, size(state.grid, 3) + 1 - k)

@inline function layer_state(state, extension, i, j, k)
    Nz = size(state.grid, 3)
    Nₑ = number_of_extension_layers(extension)
    # Both are evaluated at valid indices and the one that applies is selected
    grid_layer = grid_layer_state(state, i, j, clamp(Nz + Nₑ + 1 - k, 1, Nz))
    extension_layer = extension_layer_state(state, extension, i, j, clamp(Nₑ + 1 - k, 1, Nₑ))
    in_extension = k ≤ Nₑ
    return map((e, g) -> ifelse(in_extension, e, g), extension_layer, grid_layer)
end

#####
##### Interface temperatures
#####

# The grid's faces interpolate the temperature, whose halos the model fills before every radiation
# update, so the top face takes the top cell's temperature. The bottom face instead continues the
# gradient between the two lowest cells to the surface: the surface layer's gradient is resolved and
# steep, and dropping it cools the bottom face by `Δz/2 ∂T/∂z`, which at 250 m spacing lowers the
# surface downwelling longwave by 2 W m⁻².
@inline function grid_interface_temperature(state, i, j, kᶠ)
    grid = state.grid
    T = state.temperature
    z₁ = znode(i, j, 1, grid, Center(), Center(), Center())
    zᶠ = znode(i, j, 1, grid, Center(), Center(), Face())
    T_bottom = @inbounds T[i, j, 1] - (z₁ - zᶠ) * ∂zᶜᶜᶠ(i, j, 2, grid, T)
    return ifelse(kᶠ == 1, T_bottom, ℑzᵃᵃᶠ(i, j, kᶠ, grid, T))
end

@inline grid_top_temperature(state, i, j) = grid_interface_temperature(state, i, j, size(state.grid, 3) + 1)

# The extension's temperature profile is anchored to the grid top: shifted there by the top
# face's departure from the profile, and relaxed back to it with the anchor weight
@inline extension_anchor(state, extension, i, j) = grid_top_temperature(state, i, j) - extension.join_temperature

@inline extension_layer_temperature(state, extension, i, j, m) =
    @inbounds extension.temperature_layers[m] + extension_anchor(state, extension, i, j) * extension.anchor_weight_layers[m]

# The face above extension layer `m`
@inline extension_face_temperature(state, extension, i, j, m) =
    @inbounds extension.temperature_interfaces[m+1] + extension_anchor(state, extension, i, j) * extension.anchor_weight_interfaces[m+1]

"""
$(TYPEDSIGNATURES)

The temperature on column interface `k` (top-down) of column `(i, j)`: on the grid's faces the
interpolated temperature, above it the anchored extension profile. The top of the atmosphere
radiates at the temperature of the top layer.
"""
@inline interface_temperature(state, i, j, k) = interface_temperature(state, state.extension, i, j, k)

@inline interface_temperature(state, ::Nothing, i, j, k) = grid_interface_temperature(state, i, j, size(state.grid, 3) + 2 - k)

@inline function interface_temperature(state, extension, i, j, k)
    Nz = size(state.grid, 3)
    Nₑ = number_of_extension_layers(extension)
    T_grid = grid_interface_temperature(state, i, j, clamp(Nz + Nₑ + 2 - k, 1, Nz + 1))
    m = clamp(Nₑ + 1 - k, 1, Nₑ)   # the extension layer below interface `k`
    T_extension = ifelse(k == 1, extension_layer_temperature(state, extension, i, j, m),
                                 extension_face_temperature(state, extension, i, j, m))
    return ifelse(k ≤ Nₑ, T_extension, T_grid)
end

#####
##### Extension pressures
#####

"""
$(TYPEDSIGNATURES)

Integrate the extension layers' hydrostatic pressures of the `batch_size = (Nx, rows)` columns of
the batch of `state` upward from the grid's top face, with the virtual temperature of the
anchored extension profile, into `state.extension_pressure`; nothing to do without an extension.
"""
compute_extension_pressure!(state::ColumnState, batch_size) = compute_extension_pressure!(state, state.extension, batch_size)

compute_extension_pressure!(state, ::Nothing, batch_size) = nothing

function compute_extension_pressure!(state, extension, batch_size)
    grid = state.grid
    launch!(architecture(grid), grid, batch_size, _compute_extension_pressure!, state, extension)
    return nothing
end

@kernel function _compute_extension_pressure!(state, extension)
    i, jᵇ = @index(Global, NTuple)

    j = jᵇ + state.row_offset
    c = batch_column(i, jᵇ, state.grid.Nx)
    g = state.gravitational_acceleration
    ϵ = state.dry_air_molar_mass / state.vapor_molar_mass
    p = top_face_pressure(i, j, state.grid, state.pressure, state.density, g)
    @inbounds state.extension_pressure[c, 1] = p

    for m in 1:number_of_extension_layers(extension)
        T = extension_layer_temperature(state, extension, i, j, m)
        @inbounds q = extension.specific_humidity[m]
        @inbounds Δz = extension.Δz[m]
        Tᵛ = T * (1 + (ϵ - 1) * q)
        p = p * exp(-g * Δz / (state.dry_air_gas_constant * Tᵛ))
        @inbounds state.extension_pressure[c, m+1] = p
    end
end

#####
##### Interface pressures (for the host column atmosphere only; the column solvers need none)
#####

@inline function grid_interface_pressure(state, i, j, kᶠ)
    grid = state.grid
    Nz = size(grid, 3)
    g = state.gravitational_acceleration
    kᶠ == 1 && return bottom_face_pressure(i, j, grid, state.pressure, state.density, g)
    kᶠ == Nz + 1 && return top_face_pressure(i, j, grid, state.pressure, state.density, g)
    return ℑzᵃᵃᶠ(i, j, kᶠ, grid, state.pressure)
end

interface_pressure(state, i, j, k) = interface_pressure(state, state.extension, i, j, k)
interface_pressure(state, ::Nothing, i, j, k) = grid_interface_pressure(state, i, j, size(state.grid, 3) + 2 - k)

function interface_pressure(state, extension, i, j, k)
    Nz = size(state.grid, 3)
    Nₑ = number_of_extension_layers(extension)
    k > Nₑ && return grid_interface_pressure(state, i, j, Nz + Nₑ + 2 - k)
    c = batch_column(i, j - state.row_offset, state.grid.Nx)
    return @inbounds state.extension_pressure[c, Nₑ + 2 - k]
end
