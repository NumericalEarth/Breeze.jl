# Imports are provided by the Diagnostics module.

"""
$(TYPEDSIGNATURES)

Azimuthally average `field` into radial rings about `center = (xc, yc)`, returning a
`Field` on a one-dimensional-in-radius grid — `Bounded` in radius, `Flat` in azimuth, and
`Bounded` in z — with `Nr` uniform rings spanning ``[0, \\texttt{radius}]`` and the same
vertical grid as `field`.

This bins the Cartesian `field` by radius, a pragmatic stand-in for a reduction on a true
cylindrical grid; the kernel runs on the CPU and the GPU. The radial grid is assumed
uniform. Rings that catch no grid points are filled with `NaN` (not zero) so they read as
"no data" and don't bias a subsequent radial average. Keep the rings coarse enough to stay
populated — roughly ``\\texttt{radius} / N_r \\gtrsim`` the horizontal grid spacing; far
finer `Nr` just produces empty (`NaN`) rings.

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
function azimuthal_mean(field; radius, Nr, center = (0, 0))
    grid = field.grid
    ring_grid = RectilinearGrid(architecture(grid), eltype(grid);
                                size = (Nr, size(grid, 3)),
                                x = (0, radius), z = znodes(grid, Face()),
                                topology = (Bounded, Flat, Bounded))
    profile = CenterField(ring_grid)
    return azimuthal_mean!(profile, field; center)
end

"""
$(TYPEDSIGNATURES)

Bin `field` (on an ``(x, y, z)`` grid) into the radial rings of `profile` (on an
``(r, z)`` grid) about `center`, in place. `profile` and `field` must share their vertical
grid; the radial rings are `profile`'s uniform `x`-cells.
"""
function azimuthal_mean!(profile, field; center = (0, 0))
    ring_grid = profile.grid
    field_grid = field.grid
    Nr = size(ring_grid, 1)
    Nx, Ny = size(field_grid, 1), size(field_grid, 2)
    FT = eltype(ring_grid)
    Δr = xnode(Nr + 1, 1, 1, ring_grid, Face(), c, c) / Nr   # uniform radial spacing
    xc, yc = center
    launch!(architecture(ring_grid), ring_grid, :xyz,
            _azimuthal_mean!, profile, field, field_grid,
            convert(FT, xc), convert(FT, yc), convert(FT, Δr), Nx, Ny)
    return profile
end

@kernel function _azimuthal_mean!(profile, field, field_grid, xc, yc, Δr, Nx, Ny)
    ir, j, k = @index(Global, NTuple)
    FT = eltype(profile)
    ring_sum = zero(FT)
    ring_count = 0
    @inbounds for jj in 1:Ny, ii in 1:Nx
        x = xnode(ii, jj, k, field_grid, c, c, c) - xc
        y = ynode(ii, jj, k, field_grid, c, c, c) - yc
        r = sqrt(x^2 + y^2)
        in_ring = unsafe_trunc(Int, r / Δr) + 1 == ir
        ring_sum += ifelse(in_ring, field[ii, jj, k], zero(FT))
        ring_count += ifelse(in_ring, 1, 0)
    end
    # Empty rings get `NaN`, not zero, so they read as "no data" and don't bias a
    # subsequent radial average.
    @inbounds profile[ir, j, k] = ifelse(ring_count > 0, ring_sum / ring_count, convert(FT, NaN))
end
