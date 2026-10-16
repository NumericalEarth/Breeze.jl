#####
##### Column workspace: the per-column storage of the ecCKD radiation, for one batch of columns
#####
##### The columns are solved in batches of whole rows of the grid, `rows` rows of `Nx` columns at a
##### time (`column_batches` of the radiation model), and the workspace holds one batch. Within a batch, column `c = i + (jᵇ - 1) Nx` is the `jᵇ`-th row of the
##### batch, and with `N = Nz + Nₑ` layers (`Nₑ` extension layers) the arrays are laid out top-down
##### in NumericalRadiation's convention. The fluxes are not stored here: the column kernel
##### accumulates them directly into the flux fields of the radiation model.
#####
##### The workspace holds what is expensive to recompute for every g point: the per-layer gas
##### optics stencil, gas amounts and water paths and the interface temperatures, which a layer
##### pass fills with one work item per layer, and the scratch the column solvers sweep through.
#####

"""
$(TYPEDEF)

Per-layer state and solver scratch of the ecCKD radiation of one batch of `Nb = Nx × rows`
columns of `N` layers.

Fields:
- `stencil`: The gas optics stencil of every layer, a [`LayerStencils`](@ref), or `nothing` for a
  gas optics model without interpolation tables
- `dry_air`, `water_vapor`, `ozone`: Layer gas amounts [mol m⁻²], `(Nb, N)`
- `liquid_water_path`, `ice_water_path`: Layer cloud water paths [kg m⁻²], `(Nb, N)`, or `nothing`
  for a clear sky
- `temperature_interfaces`: Interface temperatures [K], `(Nb, N + 1)`
- `reflectance`, `transmittance`, `direct_reflectance`, `direct_diffuse_transmittance`,
  `direct_flux`: Shortwave adding-method layer scratch, `(Nb, N)`. The longwave solution
  reuses `transmittance` and `reflectance` for its layer transmittance and upward source.
- `stack_albedo`, `source`: Shortwave adding-method interface scratch, `(Nb, N + 1)`
- `extension_pressure`: Hydrostatic pressure on the extension faces, bottom-up from the grid
  top, `(Nb, Nₑ + 1)`, or `nothing` without a column extension
- `cos_zenith`: Cosine of the solar zenith angle of every column of the grid, `(Nx Ny,)`
- `extension`: The materialized column extension above the grid top, or `nothing`
- `rows`: The number of grid rows in a batch

Layer and interface arrays carry separate types: Reactant's kernel arrays encode their
dimensions in the type, and interfaces have one more vertical entry.
"""
struct ColumnWorkspace{FT, AF, FF, S, W, P, V, X}
    stencil :: S
    dry_air :: AF
    water_vapor :: AF
    ozone :: AF
    liquid_water_path :: W
    ice_water_path :: W
    temperature_interfaces :: FF
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
    rows :: Int
end

"""
$(TYPEDEF)

The gas optics stencil of every layer of a batch: the lower index and weight of the pressure,
temperature and water vapor brackets, `(Nb, N)` each. The upper index of every bracket is the
lower index plus one.
"""
struct LayerStencils{AI, AF}
    pressure_index :: AI
    pressure_weight :: AF
    temperature_index :: AI
    temperature_weight :: AF
    water_vapor_index :: AI
    water_vapor_weight :: AF
end

Adapt.adapt_structure(to, s::LayerStencils) =
    LayerStencils(adapt(to, s.pressure_index), adapt(to, s.pressure_weight),
                  adapt(to, s.temperature_index), adapt(to, s.temperature_weight),
                  adapt(to, s.water_vapor_index), adapt(to, s.water_vapor_weight))

# The float type is that of the layer arrays; every other parameter follows from the fields.
function ColumnWorkspace{FT}(args...) where FT
    fields = NamedTuple{fieldnames(ColumnWorkspace)}(args)
    AF = typeof(fields.dry_air)
    FF = typeof(fields.temperature_interfaces)
    S = typeof(fields.stencil)
    W = typeof(fields.liquid_water_path)
    P = typeof(fields.extension_pressure)
    V = typeof(fields.cos_zenith)
    X = typeof(fields.extension)
    return ColumnWorkspace{FT, AF, FF, S, W, P, V, X}(args...)
end

"""
$(TYPEDSIGNATURES)

Allocate a zeroed [`ColumnWorkspace`](@ref) of float type `FT` on architecture `arch` for batches
of `rows` rows of `Nx` columns of `N` layers, and the cosine of the solar zenith angle of all `Nc`
columns of the grid. `gas_model` decides whether there are stencils to store, `cloud` whether
there are water paths, and `extension` (a [`MaterializedColumnExtension`](@ref) or `nothing`)
whether there are extension pressures.
"""
function ColumnWorkspace(arch, FT, Nx, rows, Nc, N, gas_model, cloud, extension)
    Nb = Nx * rows
    layers() = on_architecture(arch, zeros(FT, Nb, N))
    interfaces() = on_architecture(arch, zeros(FT, Nb, N + 1))
    stencil = layer_stencils(arch, FT, Nb, N, gas_model)
    liquid_water_path = water_path_array(arch, FT, Nb, N, cloud)
    ice_water_path = water_path_array(arch, FT, Nb, N, cloud)
    extension_pressure = extension_pressure_array(arch, FT, Nb, extension)
    cos_zenith = on_architecture(arch, zeros(FT, Nc))
    return ColumnWorkspace{FT}(stencil, layers(), layers(), layers(),
                               liquid_water_path, ice_water_path, interfaces(),
                               layers(), layers(), layers(), layers(), layers(),
                               interfaces(), interfaces(),
                               extension_pressure, cos_zenith, extension, rows)
end

function layer_stencils(arch, FT, Nb, N, ::EcCKDTabulatedGasOpticsModel)
    indices() = on_architecture(arch, zeros(Int32, Nb, N))
    weights() = on_architecture(arch, zeros(FT, Nb, N))
    return LayerStencils(indices(), weights(), indices(), weights(), indices(), weights())
end

layer_stencils(arch, FT, Nb, N, ::EcCKDGasOpticsModel) = nothing

water_path_array(arch, FT, Nb, N, ::Nothing) = nothing
water_path_array(arch, FT, Nb, N, cloud) = on_architecture(arch, zeros(FT, Nb, N))

extension_pressure_array(arch, FT, Nb, ::Nothing) = nothing
extension_pressure_array(arch, FT, Nb, extension) =
    on_architecture(arch, zeros(FT, Nb, number_of_extension_layers(extension) + 1))

Adapt.adapt_structure(to, workspace::ColumnWorkspace{FT}) where FT =
    ColumnWorkspace{FT}(adapt(to, workspace.stencil),
                        adapt(to, workspace.dry_air),
                        adapt(to, workspace.water_vapor),
                        adapt(to, workspace.ozone),
                        adapt(to, workspace.liquid_water_path),
                        adapt(to, workspace.ice_water_path),
                        adapt(to, workspace.temperature_interfaces),
                        adapt(to, workspace.reflectance),
                        adapt(to, workspace.transmittance),
                        adapt(to, workspace.direct_reflectance),
                        adapt(to, workspace.direct_diffuse_transmittance),
                        adapt(to, workspace.direct_flux),
                        adapt(to, workspace.stack_albedo),
                        adapt(to, workspace.source),
                        adapt(to, workspace.extension_pressure),
                        adapt(to, workspace.cos_zenith),
                        adapt(to, workspace.extension),
                        workspace.rows)

Base.eltype(::ColumnWorkspace{FT}) where FT = FT
Base.eltype(::Type{<:ColumnWorkspace{FT}}) where FT = FT

"""
$(TYPEDSIGNATURES)

The number of layers of the extended columns, `N = Nz + Nₑ`.
"""
@inline number_of_layers(workspace::ColumnWorkspace) = size(workspace.dry_air, 2)

"""
$(TYPEDSIGNATURES)

The number of columns of a batch, `Nb = Nx × rows`.
"""
@inline batch_columns(workspace::ColumnWorkspace) = size(workspace.dry_air, 1)

"""
$(TYPEDSIGNATURES)

The batches of the rows of `grid` that `workspace` holds one at a time: the `(row_offset, size)` of
each, with `row_offset` the number of grid rows before the batch and `size = (Nx, rows)` the columns
it solves. Every batch is full, so the column kernels compile for one size: when the rows do not
divide `Ny`, the final batch overlaps the one before it (as for the RRTMGP optics), and the
overlapping columns are solved twice.
"""
function column_batches(workspace::ColumnWorkspace, grid)
    Nx, Ny = size(grid, 1), size(grid, 2)
    rows = workspace.rows
    return ((min(j₀, Ny - rows), (Nx, rows)) for j₀ in 0:rows:Ny-1)
end

# Column `c` of the batch, row `jᵇ` of it
@inline batch_column(i, jᵇ, Nx) = i + (jᵇ - 1) * Nx

# The scratch of batch column `c`: the shortwave adding method's, and the longwave solution's
# transmittance and upward source in two of the same rows
@inline shortwave_scratch(workspace, c) = ShortwaveColumnScratch(view(workspace.reflectance, c, :),
                                                                 view(workspace.transmittance, c, :),
                                                                 view(workspace.direct_reflectance, c, :),
                                                                 view(workspace.direct_diffuse_transmittance, c, :),
                                                                 view(workspace.direct_flux, c, :),
                                                                 view(workspace.stack_albedo, c, :),
                                                                 view(workspace.source, c, :))

@inline longwave_scratch(workspace, c) = (view(workspace.transmittance, c, :), view(workspace.reflectance, c, :))
