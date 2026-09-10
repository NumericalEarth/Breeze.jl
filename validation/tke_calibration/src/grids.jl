#####
##### Vertical grids and conservative regridding
#####

"""The faces of the LES's uniform vertical grid, from its cell centers `z`."""
function les_faces(z::AbstractVector)
    Δz = z[2] - z[1]
    all(≈(Δz), diff(z)) || error("The LES grid is not uniform")
    return collect(range(z[1] - Δz / 2, z[end] + Δz / 2, length = length(z) + 1))
end

"""Uniform faces with spacing `Δz` from the surface to `Lz`, which `Δz` must divide."""
function uniform_faces(Δz, Lz)
    N = Lz / Δz
    isinteger(N) || error("Δz = $Δz does not divide the depth $Lz")
    return collect(range(0, Lz, length = round(Int, N) + 1))
end

"""
The vertical faces of NumericalEarth's regional hindcast grid — 50 m cells at the surface stretching by
10 percent per level to at most 750 m over 23.2 km, 50 levels in all — truncated to the faces at or
below `top`, the depth of the LES data by default: 24 faces, 23 cells, the topmost at 3977 m.
"""
function hindcast_faces(; top = 4000)
    z = ReferenceToStretchedDiscretization(extent = 23181.81, bias = :left, bias_edge = 0.0,
                                           constant_spacing = 50.0, constant_spacing_extent = 50.0,
                                           maximum_spacing = 750.0, stretching = LinearStretching(0.1))
    faces = [z(k) for k in 1:length(z)+1]
    return faces[faces .≤ top + 1e-6]
end

# Single-column and column-ensemble grids on given faces, on the CPU (regridding is setup work)
column_grid(faces) = RectilinearGrid(CPU(); size = length(faces) - 1, z = faces, topology = (Flat, Flat, Bounded))
column_grid(faces, N₁, N₂) = RectilinearGrid(CPU(); size = ColumnEnsembleSize(Nz = length(faces) - 1, ensemble = (N₁, N₂), Hz = 1),
                                             z = faces, topology = (Flat, Flat, Bounded))

function check_regridding_extent(from, to)
    to[1] ≥ from[1] - 1e-6 && to[end] ≤ from[end] + 1e-6 ||
        error("The target cells [$(to[1]), $(to[end])] extend beyond the source [$(from[1]), $(from[end])]")
    return nothing
end

"""
Conservative regridding of a cell-centered profile from the cells with faces `from` onto the cells with
faces `to` with Oceananigans' `regrid!`: each target value is the mean over the target cell of the
piecewise-constant source profile. The target cells must lie within the source's extent.
"""
function regrid_column(values::AbstractVector, from::AbstractVector, to::AbstractVector)
    check_regridding_extent(from, to)
    source = CenterField(column_grid(from))
    set!(source, reshape(collect(Float64, values), 1, 1, :))
    target = CenterField(column_grid(to))
    regrid!(target, source)
    return vec(Array(interior(target)))
end

"""
[`regrid_column`](@ref) applied at once to every column of an `(N₁, N₂, Nz)` array — the time means of
a column ensemble — on column-ensemble grids of the same shape.
"""
function regrid_columns(values::AbstractArray{<:Any, 3}, from::AbstractVector, to::AbstractVector)
    check_regridding_extent(from, to)
    N₁, N₂ = size(values, 1), size(values, 2)
    source = CenterField(column_grid(from, N₁, N₂))
    set!(source, Float64.(values))
    target = CenterField(column_grid(to, N₁, N₂))
    regrid!(target, source)
    return Array(interior(target))
end
