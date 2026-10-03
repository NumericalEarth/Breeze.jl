#####
##### Materialized column extension
#####
##### `ColumnExtension` describes the atmosphere above the grid top with profiles `z -> T`, `z -> q`,
##### `z -> χ_O₃`. The extension is materialized once, on the host, into device vectors sampled on
##### the geometrically stretched extension layers, so that the per-column kernel only reads
##### vectors and applies the temperature anchor of the grid's top face.
#####

"""
$(TYPEDEF)

A [`ColumnExtension`](@ref) sampled on its `Nₑ` layers above the grid top, ready for the column
staging kernels. Layers are indexed bottom-up from the grid top (`m = 1` sits on the grid's top
face); the temperature profile is stored unanchored, and the kernel adds
`(T_top - join_temperature) exp(-(z - base) / blending_height)` with `T_top` the temperature of
the grid's top face in each column.

Fields:
- `Δz`: Layer thickness [m], `(Nₑ,)`
- `z_layer`: Layer center height [m], `(Nₑ,)`
- `temperature_layers`: Unanchored layer temperature `temperature(z_layer)` [K], `(Nₑ,)`
- `temperature_interfaces`: Unanchored interface temperature `temperature(z_face)` [K], `(Nₑ + 1,)`
- `specific_humidity`: Layer specific humidity [kg kg⁻¹], `(Nₑ,)`
- `ozone`: Layer ozone mole fraction [mol mol⁻¹], `(Nₑ,)`
- `blending_height`: Decay height of the temperature anchor [m]; `0` disables it
- `join_temperature`: The profile temperature at the grid top, `temperature(base)` [K]
- `base`: Height of the grid's top face [m]
"""
struct MaterializedColumnExtension{FT, V}
    Δz :: V
    z_layer :: V
    temperature_layers :: V
    temperature_interfaces :: V
    specific_humidity :: V
    ozone :: V
    blending_height :: FT
    join_temperature :: FT
    base :: FT
end

Adapt.adapt_structure(to, extension::MaterializedColumnExtension) =
    MaterializedColumnExtension(adapt(to, extension.Δz),
                                adapt(to, extension.z_layer),
                                adapt(to, extension.temperature_layers),
                                adapt(to, extension.temperature_interfaces),
                                adapt(to, extension.specific_humidity),
                                adapt(to, extension.ozone),
                                extension.blending_height,
                                extension.join_temperature,
                                extension.base)

# The ozone profile of the extension: its own `ozone_mole_fraction` when given, else the
# `O₃` of the background atmosphere, which must be a number or a function of height here — a
# `Field` lives on the grid and says nothing about the atmosphere above it.
extension_ozone_profile(χ::Number, O₃) = z -> χ
extension_ozone_profile(χ, O₃) = χ
extension_ozone_profile(::Nothing, O₃::Number) = z -> O₃
extension_ozone_profile(::Nothing, O₃::AbstractField) =
    throw(ArgumentError("The column extension needs an ozone profile above the grid, but " *
                        "`BackgroundAtmosphere.O₃` is a Field on the grid: pass " *
                        "`ColumnExtension(ozone_mole_fraction = ...)` (a number or a function of height)."))
extension_ozone_profile(::Nothing, O₃) = O₃

"""
$(TYPEDSIGNATURES)

Sample `extension` on the geometrically stretched layers between the top face of `grid` and
`extension.top` (see [`column_extension_faces`](@ref)) into a [`MaterializedColumnExtension`](@ref)
on the grid's architecture, taking the ozone profile from `background` when the extension carries
none. Returns `nothing` when the grid already reaches `extension.top`.
"""
function materialize_column_extension(extension::ColumnExtension, grid, background::BackgroundAtmosphere)
    # Sampled in the grid's float type, whatever the extension's: a default `ColumnExtension()`
    # is Float64, and a Float32 grid must run Float32 kernels
    FT = eltype(grid)
    Nz = size(grid, 3)
    zᶠ = Array(znodes(grid, Face()))
    z_top = FT(zᶠ[Nz+1])
    Δz_top = FT(zᶠ[Nz+1] - zᶠ[Nz])

    faces = FT.(column_extension_faces(extension, z_top, Δz_top))
    Nₑ = length(faces) - 1
    Nₑ == 0 && return nothing

    Δz = diff(faces)
    z_layer = (faces[1:end-1] .+ faces[2:end]) ./ 2
    ozone = extension_ozone_profile(extension.ozone_mole_fraction, background.O₃)

    temperature_layers = FT[extension.temperature(z) for z in z_layer]
    temperature_interfaces = FT[extension.temperature(z) for z in faces]
    specific_humidity = FT[extension.specific_humidity(z) for z in z_layer]
    ozone_mole_fraction = FT[ozone(z) for z in z_layer]
    join_temperature = FT(extension.temperature(z_top))

    arch = architecture(grid)
    return MaterializedColumnExtension(on_architecture(arch, Δz),
                                       on_architecture(arch, z_layer),
                                       on_architecture(arch, temperature_layers),
                                       on_architecture(arch, temperature_interfaces),
                                       on_architecture(arch, specific_humidity),
                                       on_architecture(arch, ozone_mole_fraction),
                                       FT(extension.blending_height),
                                       join_temperature,
                                       z_top)
end

materialize_column_extension(::Nothing, grid, background) = nothing

"""
$(TYPEDSIGNATURES)

The number of extension layers, `Nₑ`.
"""
@inline number_of_extension_layers(extension::MaterializedColumnExtension) = length(extension.Δz)
@inline number_of_extension_layers(::Nothing) = 0

extension_summary(::Nothing) = "none"
extension_summary(extension::MaterializedColumnExtension) =
    string(number_of_extension_layers(extension), " layers from ", prettysummary(extension.base),
           " m to ", prettysummary(extension.base + sum(Array(extension.Δz))), " m")
