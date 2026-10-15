#####
##### Column workspace: the only per-column storage of the ecCKD radiation
#####
##### The column kernel reads the model's fields directly and accumulates the fluxes directly into
##### the four flux fields of the radiation model, so the only per-column arrays are the scratch the
##### column solvers sweep through (shared by the longwave and shortwave solutions, which run one
##### after the other), the hydrostatic pressures of the column extension (which depend on the
##### column's top pressure and temperature), and the cosine of the solar zenith angle.
#####
##### Columns are rows: column `c = i + (j - 1) Nx`, and with `N = Nz + Nₑ` layers (`Nₑ` extension
##### layers) the scratch is laid out top-down in NumericalRadiation's convention.
#####

"""
$(TYPEDEF)

Scratch and per-column state of the ecCKD radiation of every column of the grid.

Fields:
- `reflectance`, `transmittance`, `direct_reflectance`, `direct_diffuse_transmittance`,
  `direct_flux`: Shortwave adding-method layer scratch, `(Nc, N)`. The longwave solution
  reuses `transmittance` and `reflectance` for its layer transmittance and upward source.
- `stack_albedo`, `source`: Shortwave adding-method interface scratch, `(Nc, N + 1)`
- `extension_pressure`: Hydrostatic pressure on the extension faces, bottom-up from the grid
  top, `(Nc, Nₑ + 1)`, or `nothing` without a column extension
- `cos_zenith`: Cosine of the solar zenith angle, `(Nc,)`
- `extension`: The materialized column extension above the grid top, or `nothing`

Layer and interface arrays carry separate types: Reactant's kernel arrays encode their
dimensions in the type, and interfaces have one more vertical entry.
"""
struct ColumnWorkspace{FT, AF, FF, P, V, X}
    reflectance :: AF
    transmittance :: AF
    direct_reflectance :: AF
    direct_diffuse_transmittance :: AF
    direct_flux :: AF
    stack_albedo :: FF
    source :: FF
    extension_pressure :: P
    cos_zenith :: V
    extension :: X
end

# The float type is that of the layer arrays; every other parameter follows from the fields.
function ColumnWorkspace{FT}(args...) where FT
    fields = NamedTuple{fieldnames(ColumnWorkspace)}(args)
    AF = typeof(fields.reflectance)
    FF = typeof(fields.stack_albedo)
    P = typeof(fields.extension_pressure)
    V = typeof(fields.cos_zenith)
    X = typeof(fields.extension)
    return ColumnWorkspace{FT, AF, FF, P, V, X}(args...)
end

"""
$(TYPEDSIGNATURES)

Allocate a zeroed [`ColumnWorkspace`](@ref) of float type `FT` for `Nc` columns of `N` layers on
architecture `arch`, carrying `extension` (a [`MaterializedColumnExtension`](@ref) or `nothing`).
"""
function ColumnWorkspace(arch, FT, Nc, N, extension)
    layers() = on_architecture(arch, zeros(FT, Nc, N))
    interfaces() = on_architecture(arch, zeros(FT, Nc, N + 1))
    extension_pressure = extension_pressure_array(arch, FT, Nc, extension)
    cos_zenith = on_architecture(arch, zeros(FT, Nc))
    return ColumnWorkspace{FT}(layers(), layers(), layers(), layers(), layers(),
                               interfaces(), interfaces(),
                               extension_pressure, cos_zenith, extension)
end

extension_pressure_array(arch, FT, Nc, ::Nothing) = nothing
extension_pressure_array(arch, FT, Nc, extension) =
    on_architecture(arch, zeros(FT, Nc, number_of_extension_layers(extension) + 1))

Adapt.adapt_structure(to, workspace::ColumnWorkspace{FT}) where FT =
    ColumnWorkspace{FT}(adapt(to, workspace.reflectance),
                        adapt(to, workspace.transmittance),
                        adapt(to, workspace.direct_reflectance),
                        adapt(to, workspace.direct_diffuse_transmittance),
                        adapt(to, workspace.direct_flux),
                        adapt(to, workspace.stack_albedo),
                        adapt(to, workspace.source),
                        adapt(to, workspace.extension_pressure),
                        adapt(to, workspace.cos_zenith),
                        adapt(to, workspace.extension))

Base.eltype(::ColumnWorkspace{FT}) where FT = FT
Base.eltype(::Type{<:ColumnWorkspace{FT}}) where FT = FT

"""
$(TYPEDSIGNATURES)

The number of layers of the extended columns, `N = Nz + Nₑ`.
"""
@inline number_of_layers(workspace::ColumnWorkspace) = size(workspace.reflectance, 2)

"""
$(TYPEDSIGNATURES)

The number of columns, `Nc = Nx Ny`.
"""
@inline number_of_columns(workspace::ColumnWorkspace) = size(workspace.reflectance, 1)

# The scratch of column `c`: the shortwave adding method's, and the longwave solution's
# transmittance and upward source in two of the same rows
@inline shortwave_scratch(workspace, c) = ShortwaveColumnScratch(view(workspace.reflectance, c, :),
                                                                 view(workspace.transmittance, c, :),
                                                                 view(workspace.direct_reflectance, c, :),
                                                                 view(workspace.direct_diffuse_transmittance, c, :),
                                                                 view(workspace.direct_flux, c, :),
                                                                 view(workspace.stack_albedo, c, :),
                                                                 view(workspace.source, c, :))

@inline longwave_scratch(workspace, c) = (view(workspace.transmittance, c, :), view(workspace.reflectance, c, :))
