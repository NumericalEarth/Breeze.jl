module VerticalGrids

export PiecewiseStretchedDiscretization

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

end # module
