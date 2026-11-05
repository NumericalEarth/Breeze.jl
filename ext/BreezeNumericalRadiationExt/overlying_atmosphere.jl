#####
##### Materializing the overlying atmosphere
#####
##### `OverlyingAtmosphereColumn` describes the atmosphere above the grid top with profiles of height. It is
##### materialized once, on the host, by sampling the profiles at the centers of the geometrically
##### stretched extension layers into device vectors, so that the column kernels only read vectors
##### and apply the temperature anchor of the grid's top face. The extension's interface
##### temperatures interpolate its layer temperatures, as the grid's interpolate its cells'.
#####

# A profile is a number, constant with height, or a function of height
profile_value(profile::Number, z) = profile
profile_value(profile, z) = profile(z)

# The extension samples its own ozone profile, or else the `O₃` with which the background
# atmosphere was built. A background `O₃` given as a `Field` exists only on the grid, so it
# cannot be sampled above it.
extension_ozone(ozone_mole_fraction, background_O₃) = ozone_mole_fraction
extension_ozone(::Nothing, background_O₃) = background_O₃
extension_ozone(::Nothing, ::AbstractField) =
    throw(ArgumentError("The overlying atmosphere needs an ozone profile above the grid, but " *
                        "`BackgroundAtmosphere.O₃` is a Field, which exists only on the grid: pass " *
                        "`OverlyingAtmosphereColumn(ozone_mole_fraction = ...)` (a number or a function of height)."))

"""
$(TYPEDSIGNATURES)

Materialize `extension` on the geometrically stretched layers between the top face of `grid` and
`extension.top` (see [`overlying_atmosphere_faces`](@ref)): the layers' thicknesses and centers and the
profiles sampled at the centers, in the grid's float type and on its architecture, with the ozone
profile of `background` when the extension has none. Returns `nothing` when the grid already
reaches `extension.top`.
"""
function materialize_overlying_atmosphere(extension::OverlyingAtmosphereColumn, grid, background::BackgroundAtmosphere)
    # Sampled in the grid's float type, whatever the extension's: a default `OverlyingAtmosphereColumn()`
    # is Float64, and a Float32 grid must run Float32 kernels
    FT = eltype(grid)
    Nz = size(grid, 3)
    zᶠ = Array(znodes(grid, Face()))
    z_top = FT(zᶠ[Nz+1])
    Δz_top = FT(zᶠ[Nz+1] - zᶠ[Nz])

    faces = FT.(overlying_atmosphere_faces(extension, z_top, Δz_top))
    length(faces) == 1 && return nothing

    Δzᵃᵃᶜ = diff(faces)
    zᵃᵃᶜ = (faces[1:end-1] .+ faces[2:end]) ./ 2
    O₃ = extension_ozone(extension.ozone_mole_fraction, background.O₃)
    sample(profile) = on_architecture(architecture(grid), FT[profile_value(profile, z) for z in zᵃᵃᶜ])

    return OverlyingAtmosphereColumn(convert(FT, extension.top),
                           extension.layers,
                           convert(FT, extension.blending_height),
                           extension.temperature,
                           extension.specific_humidity,
                           extension.ozone_mole_fraction,
                           z_top,
                           FT(profile_value(extension.temperature, z_top)),
                           on_architecture(architecture(grid), Δzᵃᵃᶜ),
                           on_architecture(architecture(grid), zᵃᵃᶜ),
                           sample(extension.temperature),
                           sample(extension.specific_humidity),
                           sample(O₃))
end

materialize_overlying_atmosphere(::Nothing, grid, background) = nothing

"""
$(TYPEDSIGNATURES)

The number of extension layers, `Nₑ`.
"""
@inline number_of_extension_layers(extension::OverlyingAtmosphereColumn) = length(extension.Δzᵃᵃᶜ)
@inline number_of_extension_layers(::Nothing) = 0

extension_summary(::Nothing) = "none"
extension_summary(extension::OverlyingAtmosphereColumn) =
    string(number_of_extension_layers(extension), " layers from ", prettysummary(extension.bottom),
           " m to ", prettysummary(extension.top), " m")
