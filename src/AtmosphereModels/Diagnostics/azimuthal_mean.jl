# Imports are provided by the Diagnostics module.

"""
$(TYPEDSIGNATURES)

Azimuthally average `field` into radial rings about `center = (xc, yc)`, returning a
`Field` on a one-dimensional-in-radius grid — `Bounded` in radius, `Flat` in azimuth, and
`Bounded` in z — with `Nr` uniform rings spanning ``[0, \\texttt{radius}]`` and the same
vertical grid as `field`.

Each Cartesian cell is split into an `m × m` block of sub-cells that are binned by radius,
so a cell contributes to every ring it overlaps — uniform sub-sampling is area-weighting,
which makes this a first-order *conservative* remap onto the radial rings (a pragmatic
stand-in for a reduction on a true cylindrical grid). The kernel runs on the CPU and the
GPU. Larger `m` resolves the rings more finely and keeps near-center rings populated; any
ring that still catches nothing (only when ``\\texttt{radius}/N_r`` is finer than a
sub-cell) is filled with `NaN`, not zero, so it reads as "no data" and doesn't bias a
subsequent radial average.

```jldoctest
using Oceananigans, Breeze

grid = RectilinearGrid(size=(64, 64, 4), x=(-1, 1), y=(-1, 1), z=(0, 1),
                       topology=(Periodic, Periodic, Bounded))

c = CenterField(grid)
set!(c, (x, y, z) -> 5)            # a constant field

c̄ = azimuthal_mean(c; radius=1, Nr=8)
maximum(c̄)                         # the azimuthal mean of a constant is that constant

# output
5.0
```
"""
function azimuthal_mean(field; radius, Nr, center = (0, 0), m = 4)
    grid = field.grid
    ring_grid = RectilinearGrid(architecture(grid), eltype(grid);
                                size = (Nr, size(grid, 3)),
                                x = (0, radius), z = znodes(grid, Face()),
                                topology = (Bounded, Flat, Bounded))
    profile = CenterField(ring_grid)
    return azimuthal_mean!(profile, field; center, m)
end

"""
$(TYPEDSIGNATURES)

Remap `field` (on an ``(x, y, z)`` grid) onto the radial rings of `profile` (on an
``(r, z)`` grid) about `center`, in place, by area-weighted binning of `m × m` sub-cells
per Cartesian cell. `profile` and `field` must share their vertical grid; the radial rings
are `profile`'s uniform `x`-cells.
"""
function azimuthal_mean!(profile, field; center = (0, 0), m = 4)
    ring_grid = profile.grid
    field_grid = field.grid
    Nr = size(ring_grid, 1)
    Nx, Ny = size(field_grid, 1), size(field_grid, 2)
    FT = eltype(ring_grid)
    Δr = xnode(Nr + 1, 1, 1, ring_grid, Face(), c, c) / Nr   # uniform radial spacing
    xc, yc = center
    launch!(architecture(ring_grid), ring_grid, :xyz,
            _azimuthal_mean!, profile, field, field_grid,
            convert(FT, xc), convert(FT, yc), convert(FT, Δr), Nx, Ny, m)
    return profile
end

@kernel function _azimuthal_mean!(profile, field, field_grid, xc, yc, Δr, Nx, Ny, m)
    ir, j, k = @index(Global, NTuple)
    FT = eltype(profile)
    ring_sum = zero(FT)
    ring_count = 0
    @inbounds for jj in 1:Ny, ii in 1:Nx
        x₀ = xnode(ii, jj, k, field_grid, c, c, c) - xc
        y₀ = ynode(ii, jj, k, field_grid, c, c, c) - yc
        Δx = Δxᶜᶜᶜ(ii, jj, k, field_grid)
        Δy = Δyᶜᶜᶜ(ii, jj, k, field_grid)
        fij = field[ii, jj, k]
        # Spread the cell across rings by binning an m × m block of sub-cell centers;
        # uniform sub-sampling weights each ring by the cell area it overlaps.
        for sj in 1:m, si in 1:m
            x = x₀ + (2si - m - 1) * Δx / (2 * m)
            y = y₀ + (2sj - m - 1) * Δy / (2 * m)
            in_ring = unsafe_trunc(Int, sqrt(x^2 + y^2) / Δr) + 1 == ir
            ring_sum += ifelse(in_ring, fij, zero(FT))
            ring_count += ifelse(in_ring, 1, 0)
        end
    end
    # Rings that still catch nothing get NaN (not zero), so they read as "no data" and
    # don't bias a subsequent radial average.
    @inbounds profile[ir, j, k] = ifelse(ring_count > 0, ring_sum / ring_count, convert(FT, NaN))
end
