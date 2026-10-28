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

The model fields, constants and overlying atmosphere from which [`layer_state`](@ref) and
[`interface_temperature`](@ref) compute the state of the extended columns, and the batch of
columns being solved: the extension pressures are those of the batch whose first row follows
`row_offset` rows of the grid.
"""
struct ColumnState{G, P, T, D, M, MP, MF, O, X, XP, C}
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
    constants :: C
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
                adapt(to, state.constants))

"""
$(TYPEDSIGNATURES)

The [`ColumnState`](@ref) of `model` with the ozone of `rtm`'s background atmosphere and the
extension and extension pressures of `rtm`'s workspace, for the batch of columns that follows
`row_offset` rows of the grid.
"""
function ColumnState(rtm, model, row_offset)
    workspace = rtm.atmospheric_state
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
                       model.thermodynamic_constants)
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
@inline function grid_layer_state(i, j, k, grid, state)
    mᵈ = state.constants.dry_air.molar_mass
    mᵛ = state.constants.vapor.molar_mass
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    @inbounds begin
        ρ = state.density[i, j, k]
        p = state.pressure[i, j, k]
        T = state.temperature[i, j, k]
        χ³ = state.ozone[i, j, k]
        q = grid_moisture_fractions(i, j, k, grid, state.microphysics, ρ, state.moisture[i, j, k], state.microphysical_fields)
    end
    dry_air_moles = ρ * Δz / mᵈ
    return (pressure = p,
            temperature = T,
            dry_air = dry_air_moles,
            water_vapor = ρ * max(0, q.vapor) * Δz / mᵛ,
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
@inline function extension_layer_state(i, j, m, grid, state, extension)
    g = state.constants.gravitational_acceleration
    mᵈ = state.constants.dry_air.molar_mass
    mᵛ = state.constants.vapor.molar_mass
    c = batch_column(i, j - state.row_offset, grid.Nx)
    T = extension_layer_temperature(i, j, m, grid, state, extension)
    @inbounds begin
        qᵛ = extension.qᵛ[m]
        χ³ = extension.O₃[m]
        p_below = state.extension_pressure[c, m]
        p_above = state.extension_pressure[c, m+1]
    end
    χᵛ = qᵛ / (1 - qᵛ) * mᵈ / mᵛ   # mole fraction relative to dry air
    Δp = p_below - p_above
    dry_air_moles = hydrostatic_air_moles(Δp, g, mᵈ)
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
@inline layer_state(i, j, k, grid, state) = layer_state(i, j, k, grid, state, state.extension)

@inline layer_state(i, j, k, grid, state, ::Nothing) = grid_layer_state(i, j, size(grid, 3) + 1 - k, grid, state)

@inline function layer_state(i, j, k, grid, state, extension)
    Nz = size(grid, 3)
    Nₑ = number_of_extension_layers(extension)
    # Both are evaluated at valid indices and the one that applies is selected
    grid_layer = grid_layer_state(i, j, clamp(Nz + Nₑ + 1 - k, 1, Nz), grid, state)
    extension_layer = extension_layer_state(i, j, clamp(Nₑ + 1 - k, 1, Nₑ), grid, state, extension)
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
@inline function grid_interface_temperature(i, j, kᶠ, grid, state)
    T = state.temperature
    z₁ = znode(i, j, 1, grid, Center(), Center(), Center())
    zᶠ = znode(i, j, 1, grid, Center(), Center(), Face())
    T_bottom = @inbounds T[i, j, 1] - (z₁ - zᶠ) * ∂zᶜᶜᶠ(i, j, 2, grid, T)
    return ifelse(kᶠ == 1, T_bottom, ℑzᵃᵃᶠ(i, j, kᶠ, grid, T))
end

@inline grid_top_temperature(i, j, grid, state) = grid_interface_temperature(i, j, size(grid, 3) + 1, grid, state)

# The extension's temperature profile is anchored to the grid top: shifted by the departure of the
# grid's top-face temperature from the profile there, and relaxed back to the profile with height
@inline function extension_layer_temperature(i, j, m, grid, state, extension)
    h = extension.blending_height
    @inbounds begin
        T = extension.T[m]
        z = extension.zᵃᵃᶜ[m]
    end
    anchor = grid_top_temperature(i, j, grid, state) - extension.profile_temperature_at_bottom
    w = ifelse(h > 0, exp(-(z - extension.bottom) / h), zero(h))
    return T + anchor * w
end

# The face above extension layer `m`, linearly interpolated in height between the layers on either
# side of it; the top of the atmosphere, with no layer above, radiates at the top layer's temperature
@inline function extension_face_temperature(i, j, m, grid, state, extension)
    Nₑ = number_of_extension_layers(extension)
    n = min(m + 1, Nₑ)
    @inbounds begin
        Δz₋ = extension.Δzᵃᵃᶜ[m]
        Δz₊ = extension.Δzᵃᵃᶜ[n]
    end
    T₋ = extension_layer_temperature(i, j, m, grid, state, extension)
    T₊ = extension_layer_temperature(i, j, n, grid, state, extension)
    return (Δz₊ * T₋ + Δz₋ * T₊) / (Δz₋ + Δz₊)
end

"""
$(TYPEDSIGNATURES)

The temperature on column interface `k` (top-down) of column `(i, j)`: on the grid's faces the
interpolated temperature, above it the interpolated temperature of the anchored extension layers.
The top of the atmosphere radiates at the temperature of the top layer.
"""
@inline interface_temperature(i, j, k, grid, state) = interface_temperature(i, j, k, grid, state, state.extension)

@inline interface_temperature(i, j, k, grid, state, ::Nothing) = grid_interface_temperature(i, j, size(grid, 3) + 2 - k, grid, state)

@inline function interface_temperature(i, j, k, grid, state, extension)
    Nz = size(grid, 3)
    Nₑ = number_of_extension_layers(extension)
    T_grid = grid_interface_temperature(i, j, clamp(Nz + Nₑ + 2 - k, 1, Nz + 1), grid, state)
    T_extension = extension_face_temperature(i, j, clamp(Nₑ + 1 - k, 1, Nₑ), grid, state, extension)
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
    launch!(architecture(grid), grid, batch_size, _compute_extension_pressure!, grid, state, extension)
    return nothing
end

@kernel function _compute_extension_pressure!(grid, state, extension)
    i, jᵇ = @index(Global, NTuple)

    j = jᵇ + state.row_offset
    c = batch_column(i, jᵇ, grid.Nx)
    constants = state.constants
    g = constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(constants)
    ϵ = constants.dry_air.molar_mass / constants.vapor.molar_mass
    p = top_face_pressure(i, j, grid, state.pressure, state.density, g)
    @inbounds state.extension_pressure[c, 1] = p

    for m in 1:number_of_extension_layers(extension)
        T = extension_layer_temperature(i, j, m, grid, state, extension)
        @inbounds qᵛ = extension.qᵛ[m]
        @inbounds Δz = extension.Δzᵃᵃᶜ[m]
        Tᵛ = T * (1 + (ϵ - 1) * qᵛ)
        p = p * exp(-g * Δz / (Rᵈ * Tᵛ))
        @inbounds state.extension_pressure[c, m+1] = p
    end
end

#####
##### Interface pressures (for the host column atmosphere only; the column solvers need none)
#####

@inline function grid_interface_pressure(i, j, kᶠ, grid, state)
    Nz = size(grid, 3)
    g = state.constants.gravitational_acceleration
    kᶠ == 1 && return bottom_face_pressure(i, j, grid, state.pressure, state.density, g)
    kᶠ == Nz + 1 && return top_face_pressure(i, j, grid, state.pressure, state.density, g)
    return ℑzᵃᵃᶠ(i, j, kᶠ, grid, state.pressure)
end

interface_pressure(i, j, k, grid, state) = interface_pressure(i, j, k, grid, state, state.extension)
interface_pressure(i, j, k, grid, state, ::Nothing) = grid_interface_pressure(i, j, size(grid, 3) + 2 - k, grid, state)

function interface_pressure(i, j, k, grid, state, extension)
    Nz = size(grid, 3)
    Nₑ = number_of_extension_layers(extension)
    k > Nₑ && return grid_interface_pressure(i, j, Nz + Nₑ + 2 - k, grid, state)
    c = batch_column(i, j - state.row_offset, grid.Nx)
    return @inbounds state.extension_pressure[c, Nₑ + 2 - k]
end
