module VerticalGrids

export PiecewiseStretchedDiscretization, piecewise_linear_regrid!

using DocStringExtensions: TYPEDSIGNATURES
using GPUArraysCore: @allowscalar
using KernelAbstractions: @kernel, @index
using Oceananigans: Center, Face
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: location
using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.Utils: launch!

"""
    PiecewiseStretchedDiscretization(; z, Δz)

Construct a stretched vertical grid where the spacing varies piecewise-linearly
between breakpoints. The grid spacing is specified at breakpoint heights `z`,
and linearly interpolated between them.

Between breakpoints where `Δz` values are equal, the grid is uniform.
Where they differ, the spacing transitions linearly.

The result behaves as a vector of face positions and can be passed directly
to `RectilinearGrid` as a coordinate argument.

# Keyword Arguments

- `z`: sorted vector of breakpoint heights (length ≥ 2)
- `Δz`: vector of grid spacings at each breakpoint (same length as `z`, all positive)

# Examples

A three-region grid with uniform fine spacing, a linear transition, and uniform
coarse spacing (as used for tropical cyclone simulations):

```julia
z = PiecewiseStretchedDiscretization(
    z  = [0, 1000, 3500, 28000],
    Δz = [62.5, 62.5, 2000, 2000]
)
Nz = length(z) - 1
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z)
```

A four-region grid for deep convection (fine near surface, transition to
moderate, uniform through the troposphere, then stretched to the model top):

```julia
z = PiecewiseStretchedDiscretization(
    z  = [0, 1275, 5100, 18000, 27000],
    Δz = [50, 50, 100, 100, 300]
)
```
"""
struct PiecewiseStretchedDiscretization <: AbstractVector{Float64}
    faces :: Vector{Float64}
end

# AbstractVector interface
Base.size(d::PiecewiseStretchedDiscretization) = size(d.faces)
Base.getindex(d::PiecewiseStretchedDiscretization, i::Int) = d.faces[i]

function PiecewiseStretchedDiscretization(; z, Δz)
    N = length(z)
    length(Δz) == N || throw(ArgumentError("`z` and `Δz` must have the same length"))
    N ≥ 2 || throw(ArgumentError("need at least 2 breakpoints"))
    issorted(z) || throw(ArgumentError("`z` must be sorted in increasing order"))
    all(d -> d > 0, Δz) || throw(ArgumentError("all `Δz` must be positive"))

    faces = Float64[z[1]]
    zcur = Float64(z[1])

    for i in 1:(N - 1)
        z_lo = Float64(z[i])
        z_hi = Float64(z[i + 1])
        Δz_lo = Float64(Δz[i])
        Δz_hi = Float64(Δz[i + 1])

        while zcur < z_hi - 1e-6
            frac = (zcur - z_lo) / (z_hi - z_lo)
            spacing = Δz_lo + frac * (Δz_hi - Δz_lo)
            zcur = min(zcur + spacing, z_hi)
            push!(faces, zcur)
        end
    end

    return PiecewiseStretchedDiscretization(faces)
end

function Base.show(io::IO, ::MIME"text/plain", d::PiecewiseStretchedDiscretization)
    Nz = length(d) - 1
    Δz = diff(d.faces)
    print(io, "PiecewiseStretchedDiscretization with $Nz cells and Δz ∈ [$(minimum(Δz)), $(maximum(Δz))]")
end

#####
##### Conservative, second-order regridding of column profiles in z
#####

# TODO: this belongs in Oceananigans, as a second-order (linear-reconstruction) option of `regrid!`, whose
# z-regridding (`_regrid_in_z!`) is piecewise constant. It would then cover x and y, and Face locations through
# the reconstruction on the face-centred grid; here it is the vertical, Center-located case that column
# initialization from observed or LES profiles needs.

"""
$(TYPEDSIGNATURES)

Regrid `source_field` onto the vertical cells of `target_field`, column by column, by exact
integration of a linear reconstruction of the source over each target cell.

Each source cell ``n`` with mean ``ϕₙ`` and center ``zₙ`` is reconstructed as
``ϕₙ + σₙ (z - zₙ)``, with the centered slope ``σₙ = (ϕₙ₊₁ - ϕₙ₋₁) / (zₙ₊₁ - zₙ₋₁)`` (one-sided
in the end cells). A target cell receives the exact mean of this reconstruction over its extent,
so the regridding

* conserves the column integral of the source, cell by cell of the source;
* reproduces a profile that is linear in ``z`` exactly, wherever the target cells are;
* is second-order accurate, rather than first-order like the piecewise-constant
  `Oceananigans.Fields.regrid!`, which places the mean of a coarse target cell at the
  centroid of the source cells it overlaps rather than at its own center.

With `nonnegative = true`, the slope of every source cell with ``ϕₙ ≥ 0`` is limited to
``|σₙ| ≤ 2 ϕₙ / Δzₙ``, so that the reconstruction is nonnegative across the cell and a nonnegative
source maps to a nonnegative target. The slopes are not otherwise limited, so that two profiles
related by a smooth function (potential temperature and temperature, say) remain consistently
mapped. Use it for mass fractions and other nonnegative quantities; signed quantities (velocities,
tendencies) are regridded unlimited by default.

The fields must have `Center` vertical location, the same horizontal size (or a source with a
single column, which is regridded into every target column), and the target column must lie
within the source column.

```jldoctest
using Oceananigans
using Breeze: piecewise_linear_regrid!

topology = (Flat, Flat, Bounded)
source = CenterField(RectilinearGrid(size=4, z=(0, 4), topology=topology))
set!(source, z -> 2z + 1)

target = CenterField(RectilinearGrid(size=2, z=[0, 1.5, 4], topology=topology))
piecewise_linear_regrid!(target, source)

Array(interior(target, 1, 1, :))

# output
2-element Vector{Float64}:
 2.5
 6.5
```
"""
function piecewise_linear_regrid!(target_field, source_field; nonnegative = false)
    location(target_field, 3) === Center && location(source_field, 3) === Center ||
        throw(ArgumentError("piecewise_linear_regrid! requires fields with Center vertical location."))

    target_grid = target_field.grid
    source_grid = source_field.grid
    Nx, Ny, Nz = size(target_field)
    Nxˢ, Nyˢ, Nzˢ = size(source_field)

    (Nxˢ == Nx || Nxˢ == 1) && (Nyˢ == Ny || Nyˢ == 1) ||
        throw(ArgumentError("The source field must have the horizontal size of the target, ($Nx, $Ny), \
                             or a single column; got ($Nxˢ, $Nyˢ)."))

    # Columns are compared at (1, 1): the regridding is per column but the vertical coordinate of a
    # height-coordinate grid is shared by all of them.
    f = Face()
    c = Center()
    zᵇ, zᵗ, zᵇˢ, zᵗˢ = @allowscalar (znode(1, 1, 1, target_grid, c, c, f), znode(1, 1, Nz + 1, target_grid, c, c, f),
                                    znode(1, 1, 1, source_grid, c, c, f), znode(1, 1, Nzˢ + 1, source_grid, c, c, f))
    zᵇˢ ≤ zᵇ && zᵗ ≤ zᵗˢ ||
        throw(ArgumentError("The target column [$zᵇ, $zᵗ] must lie within the source column [$zᵇˢ, $zᵗˢ]."))

    launch!(architecture(target_grid), target_grid, :xy, _piecewise_linear_regrid_in_z!,
            target_field, source_field, target_grid, source_grid, nonnegative)

    return target_field
end

# Linear reconstruction of source cell n in column (i, j), evaluated at height z
@inline function linear_reconstruction(i, j, n, z, source, grid, nonnegative)
    c = Center()
    Nz = size(grid, 3)
    n⁻ = max(n - 1, 1)
    n⁺ = min(n + 1, Nz)

    @inbounds begin
        ϕ = source[i, j, n]
        Δϕ = source[i, j, n⁺] - source[i, j, n⁻]
    end

    zⁿ = znode(i, j, n, grid, c, c, c)
    Δz = znode(i, j, n⁺, grid, c, c, c) - znode(i, j, n⁻, grid, c, c, c)
    σ = ifelse(n⁺ == n⁻, zero(ϕ), Δϕ / Δz)

    # Keep a nonnegative cell's reconstruction nonnegative at both of its faces
    σ₊ = 2 * abs(ϕ) / Δzᶜᶜᶜ(i, j, n, grid)
    σ = ifelse(nonnegative & (ϕ ≥ 0), clamp(σ, -σ₊, σ₊), σ)

    return ϕ + σ * (z - zⁿ)
end

@kernel function _piecewise_linear_regrid_in_z!(target, source, target_grid, source_grid, nonnegative)
    i, j = @index(Global, NTuple)

    c = Center()
    f = Face()
    Nx, Ny, Nz = size(target_grid)
    Nxˢ, Nyˢ, Nzˢ = size(source_grid)
    iˢ = ifelse(Nxˢ == Nx, i, 1)
    jˢ = ifelse(Nyˢ == Ny, j, 1)

    # The source cells overlapping consecutive target cells advance monotonically
    n = 1
    for k in 1:Nz
        z⁻ = znode(i, j, k, target_grid, c, c, f)
        z⁺ = znode(i, j, k + 1, target_grid, c, c, f)

        # First source cell whose top face lies above the target cell's bottom face
        while (n < Nzˢ) & (znode(iˢ, jˢ, n + 1, source_grid, c, c, f) ≤ z⁻)
            n += 1
        end

        integral = zero(eltype(target))
        m = n
        while (m ≤ Nzˢ) & (znode(iˢ, jˢ, m, source_grid, c, c, f) < z⁺)
            ζ⁻ = max(z⁻, znode(iˢ, jˢ, m, source_grid, c, c, f))
            ζ⁺ = min(z⁺, znode(iˢ, jˢ, m + 1, source_grid, c, c, f))
            ϕ̄ = linear_reconstruction(iˢ, jˢ, m, (ζ⁻ + ζ⁺) / 2, source, source_grid, nonnegative)
            integral += (ζ⁺ - ζ⁻) * ϕ̄
            m += 1
        end

        @inbounds target[i, j, k] = integral / (z⁺ - z⁻)
    end
end

end # module
