#####
##### Filtered Surface State for Bulk Boundary Conditions
#####
##### Temporal filtering of near-surface velocity and scalar fields
##### to mitigate spurious u*-u' correlations and resolution dependence
##### in bulk flux computations (Shin, Yang & Howland 2025;
##### Nishizawa & Kitamura 2018).
#####

using KernelAbstractions: @kernel, @index
using Oceananigans: architecture
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!

#####
##### FilteredSurfaceVelocities
#####

"""
    FilteredSurfaceVelocities(grid; height=nothing, filter_timescale=Inf)

Two-dimensional fields storing temporally filtered near-surface velocities
for use in bulk flux boundary conditions.

The filtered velocities `ū`, `v̄` are updated each time step via an
exponential (first-order) filter:

    ū ← (ū + ε u_new) / (1 + ε),     ε = Δt / τ

where `τ` is the `filter_timescale`.

# Keyword Arguments

- `height`: Reference height (m) for velocity evaluation. If `nothing` (default),
  the first grid cell center value is used. If a number, velocity is linearly
  interpolated to that height.
- `filter_timescale`: Filter time scale `τ` in seconds (default: `Inf`, no filtering).
"""
struct FilteredSurfaceVelocities{U, V, H, FT}
    u :: U   # Field{Face, Center, Nothing}
    v :: V   # Field{Center, Face, Nothing}
    height :: H
    filter_timescale :: FT
end

function FilteredSurfaceVelocities(grid; height=nothing, filter_timescale=Inf)
    u = Field{Face, Center, Nothing}(grid)
    v = Field{Center, Face, Nothing}(grid)
    FT = typeof(filter_timescale)
    return FilteredSurfaceVelocities(u, v, height, FT(filter_timescale))
end

Adapt.adapt_structure(to, fv::FilteredSurfaceVelocities) =
    FilteredSurfaceVelocities(Adapt.adapt(to, fv.u),
                              Adapt.adapt(to, fv.v),
                              fv.height,
                              fv.filter_timescale)

Base.summary(fv::FilteredSurfaceVelocities) =
    string("FilteredSurfaceVelocities(height=", fv.height,
           ", filter_timescale=", fv.filter_timescale, ")")

#####
##### FilteredSurfaceScalar
#####

"""
    FilteredSurfaceScalar(grid; height=nothing, filter_timescale=Inf)

A two-dimensional field storing a temporally filtered near-surface scalar
for use in bulk flux boundary conditions.

The filter update is the same exponential form as `FilteredSurfaceVelocities`.

# Keyword Arguments

- `height`: Reference height (m) for scalar evaluation. If `nothing`, the
  first grid cell center value is used.
- `filter_timescale`: Filter time scale `τ` in seconds (default: `Inf`).
"""
struct FilteredSurfaceScalar{F, H, FT}
    field :: F   # Field{Center, Center, Nothing}
    height :: H
    filter_timescale :: FT
end

function FilteredSurfaceScalar(grid; height=nothing, filter_timescale=Inf)
    field = Field{Center, Center, Nothing}(grid)
    FT = typeof(filter_timescale)
    return FilteredSurfaceScalar(field, height, FT(filter_timescale))
end

Adapt.adapt_structure(to, fs::FilteredSurfaceScalar) =
    FilteredSurfaceScalar(Adapt.adapt(to, fs.field),
                          fs.height,
                          fs.filter_timescale)

Base.summary(fs::FilteredSurfaceScalar) =
    string("FilteredSurfaceScalar(height=", fs.height,
           ", filter_timescale=", fs.filter_timescale, ")")

#####
##### Height interpolation helper
#####

# No height specified: use first grid cell value
@inline interpolate_or_surface(i, j, grid, field, LX, LY, ::Nothing) =
    @inbounds field[i, j, 1]

# Fixed reference height: linearly interpolate between bracketing z-levels
@inline function interpolate_or_surface(i, j, grid, field, LX, LY, height)
    Nz = size(grid, 3)

    # Find the bracketing levels (evaluation height is typically in first few cells)
    z_below = znode(i, j, 1, grid, LX, LY, Center())
    k = 1
    while k < Nz
        z_above = znode(i, j, k + 1, grid, LX, LY, Center())
        if z_above >= height
            # Linear interpolation
            f_below = @inbounds field[i, j, k]
            f_above = @inbounds field[i, j, k + 1]
            w = (height - z_below) / (z_above - z_below)
            return f_below + w * (f_above - f_below)
        end
        z_below = z_above
        k += 1
    end

    # If height exceeds top cell, return top cell value
    return @inbounds field[i, j, Nz]
end

#####
##### Update kernels
#####

@kernel function _update_filtered_velocities!(u_f, v_f, u, v, grid, height, ε)
    i, j = @index(Global, NTuple)
    u_new = interpolate_or_surface(i, j, grid, u, Face(), Center(), height)
    v_new = interpolate_or_surface(i, j, grid, v, Center(), Face(), height)
    @inbounds u_f[i, j, 1] = (u_f[i, j, 1] + ε * u_new) / (1 + ε)
    @inbounds v_f[i, j, 1] = (v_f[i, j, 1] + ε * v_new) / (1 + ε)
end

@kernel function _update_filtered_scalar!(f_filtered, field_3d, grid, height, ε)
    i, j = @index(Global, NTuple)
    f_new = interpolate_or_surface(i, j, grid, field_3d, Center(), Center(), height)
    @inbounds f_filtered[i, j, 1] = (f_filtered[i, j, 1] + ε * f_new) / (1 + ε)
end

#####
##### Public update! methods
#####

"""
    update!(fv::FilteredSurfaceVelocities, velocities, grid, Δt)

Update the filtered surface velocities using the exponential filter
with time step `Δt`. `velocities` should be a `NamedTuple` with
fields `u` and `v`.
"""
function update!(fv::FilteredSurfaceVelocities, velocities, grid, Δt)
    arch = architecture(grid)
    ε = Δt / fv.filter_timescale
    launch!(arch, grid, :xy, _update_filtered_velocities!,
            fv.u, fv.v, velocities.u, velocities.v, grid, fv.height, ε)
    fill_halo_regions!(fv.u)
    fill_halo_regions!(fv.v)
    return nothing
end

"""
    update!(fs::FilteredSurfaceScalar, field_3d, grid, Δt)

Update the filtered surface scalar using the exponential filter
with time step `Δt`.
"""
function update!(fs::FilteredSurfaceScalar, field_3d, grid, Δt)
    arch = architecture(grid)
    ε = Δt / fs.filter_timescale
    launch!(arch, grid, :xy, _update_filtered_scalar!,
            fs.field, field_3d, grid, fs.height, ε)
    fill_halo_regions!(fs.field)
    return nothing
end
