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
using Oceananigans.Fields: interpolate
using Oceananigans.Grids: xnode, ynode, topology, Flat
using Oceananigans.Utils: launch!, KernelParameters

#####
##### FilteredSurfaceVelocities
#####

"""
    FilteredSurfaceVelocities(grid; height=nothing, filter_timescale=Inf)

Two-dimensional fields storing temporally filtered near-surface velocities
for use in bulk flux boundary conditions.

The filtered velocities `ū`, `v̄` are updated each time step via an
exponential (first-order) filter:

    ū ← (ū + ϵ u_new) / (1 + ϵ),     ϵ = Δt / τ

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
    last_update :: Ref{Tuple{Int, Int}}  # (iteration, stage)
end

function FilteredSurfaceVelocities(grid; height=nothing, filter_timescale=Inf)
    u = Field{Face, Center, Nothing}(grid)
    v = Field{Center, Face, Nothing}(grid)
    FT = typeof(filter_timescale)
    return FilteredSurfaceVelocities(u, v, height, FT(filter_timescale), Ref((0, 0)))
end

Adapt.adapt_structure(to, fv::FilteredSurfaceVelocities) =
    FilteredSurfaceVelocities(Adapt.adapt(to, fv.u),
                              Adapt.adapt(to, fv.v),
                              fv.height,
                              fv.filter_timescale,
                              fv.last_update)

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
    last_update :: Ref{Tuple{Int, Int}}  # (iteration, stage)
end

function FilteredSurfaceScalar(grid; height=nothing, filter_timescale=Inf)
    field = Field{Center, Center, Nothing}(grid)
    FT = typeof(filter_timescale)
    return FilteredSurfaceScalar(field, height, FT(filter_timescale), Ref((0, 0)))
end

Adapt.adapt_structure(to, fs::FilteredSurfaceScalar) =
    FilteredSurfaceScalar(Adapt.adapt(to, fs.field),
                          fs.height,
                          fs.filter_timescale,
                          fs.last_update)

Base.summary(fs::FilteredSurfaceScalar) =
    string("FilteredSurfaceScalar(height=", fs.height,
           ", filter_timescale=", fs.filter_timescale, ")")

#####
##### Kernel parameters for filtered surface fields
#####

# Use 1:N+1 to cover the extra face point, avoiding fill_halo_regions!
filtered_range(::Flat, N) = 1:N
filtered_range(topo, N) = 1:N+1

function filtered_kernel_parameters(grid)
    TX, TY, _ = topology(grid)
    Nx, Ny, _ = size(grid)
    return KernelParameters(filtered_range(TX(), Nx), filtered_range(TY(), Ny))
end

#####
##### Height interpolation helper
#####

# No height specified: use first grid cell value
@inline interpolate_or_surface(i, j, grid, field, ℓx, ℓy, ::Nothing) = @inbounds field[i, j, 1]

# Fixed reference height: interpolate using Oceananigans.Fields.interpolate
@inline function interpolate_or_surface(i, j, grid, field, ℓx, ℓy, z)
    x = xnode(i, j, 1, grid, ℓx, ℓy, Center())
    y = ynode(i, j, 1, grid, ℓx, ℓy, Center())
    return interpolate((x, y, z), field, (ℓx, ℓy, Center()), grid)
end

#####
##### Update kernels
#####

@kernel function _update_filtered_velocities!(û, v̂, u, v, grid, height, ϵ)
    i, j = @index(Global, NTuple)
    uⁿ = interpolate_or_surface(i, j, grid, u, Face(), Center(), height)
    vⁿ = interpolate_or_surface(i, j, grid, v, Center(), Face(), height)
    @inbounds û[i, j, 1] = (û[i, j, 1] + ϵ * uⁿ) / (1 + ϵ)
    @inbounds v̂[i, j, 1] = (v̂[i, j, 1] + ϵ * vⁿ) / (1 + ϵ)
end

@kernel function _update_filtered_scalar!(f̂, field_3d, grid, height, ϵ)
    i, j = @index(Global, NTuple)
    fⁿ = interpolate_or_surface(i, j, grid, field_3d, Center(), Center(), height)
    @inbounds f̂[i, j, 1] = (f̂[i, j, 1] + ϵ * fⁿ) / (1 + ϵ)
end

#####
##### Initialization kernels
#####

@kernel function _initialize_filtered_velocities!(û, v̂, u, v, grid, height)
    i, j = @index(Global, NTuple)
    @inbounds û[i, j, 1] = interpolate_or_surface(i, j, grid, u, Face(), Center(), height)
    @inbounds v̂[i, j, 1] = interpolate_or_surface(i, j, grid, v, Center(), Face(), height)
end

@kernel function _initialize_filtered_scalar!(f̂, field_3d, grid, height)
    i, j = @index(Global, NTuple)
    @inbounds f̂[i, j, 1] = interpolate_or_surface(i, j, grid, field_3d, Center(), Center(), height)
end

#####
##### Public initialize! and update! methods
#####

"""
    initialize!(fv::FilteredSurfaceVelocities, velocities, grid)

Set filtered surface velocities to the current near-surface values.
"""
function initialize!(fv::FilteredSurfaceVelocities, velocities, grid)
    arch = architecture(grid)
    kp = filtered_kernel_parameters(grid)
    launch!(arch, grid, kp, _initialize_filtered_velocities!,
            fv.u, fv.v, velocities.u, velocities.v, grid, fv.height)
    return nothing
end

"""
    initialize!(fs::FilteredSurfaceScalar, field_3d, grid)

Set filtered surface scalar to the current near-surface value.
"""
function initialize!(fs::FilteredSurfaceScalar, field_3d, grid)
    arch = architecture(grid)
    kp = filtered_kernel_parameters(grid)
    launch!(arch, grid, kp, _initialize_filtered_scalar!,
            fs.field, field_3d, grid, fs.height)
    return nothing
end

"""
    update!(fv::FilteredSurfaceVelocities, velocities, grid, Δt)

Update the filtered surface velocities using the exponential filter
with time step `Δt`. `velocities` should be a `NamedTuple` with
fields `u` and `v`.
"""
function update!(fv::FilteredSurfaceVelocities, velocities, grid, Δt)
    arch = architecture(grid)
    kp = filtered_kernel_parameters(grid)
    ϵ = Δt / fv.filter_timescale
    launch!(arch, grid, kp, _update_filtered_velocities!,
            fv.u, fv.v, velocities.u, velocities.v, grid, fv.height, ϵ)
    return nothing
end

"""
    update!(fs::FilteredSurfaceScalar, field_3d, grid, Δt)

Update the filtered surface scalar using the exponential filter
with time step `Δt`.
"""
function update!(fs::FilteredSurfaceScalar, field_3d, grid, Δt)
    arch = architecture(grid)
    kp = filtered_kernel_parameters(grid)
    ϵ = Δt / fs.filter_timescale
    launch!(arch, grid, kp, _update_filtered_scalar!,
            fs.field, field_3d, grid, fs.height, ϵ)
    return nothing
end

#####
##### Deduplication-aware initialize and update helpers
#####

initialize_filtered_surface_state!(::Nothing, model) = nothing
initialize_filtered_surface_state!(::Nothing, source_field, model) = nothing

function initialize_filtered_surface_state!(fv::FilteredSurfaceVelocities, model)
    initialize!(fv, model.velocities, model.grid)
    return nothing
end

function initialize_filtered_surface_state!(fs::FilteredSurfaceScalar, source_field, model)
    initialize!(fs, source_field, model.grid)
    return nothing
end

update_filtered_surface_state!(::Nothing, model) = nothing
update_filtered_surface_state!(::Nothing, source_field, model) = nothing

function update_filtered_surface_state!(fv::FilteredSurfaceVelocities, model)
    key = (model.clock.iteration, model.clock.stage)
    fv.last_update[] == key && return nothing
    Δt = model.clock.last_Δt
    isinf(Δt) && return nothing # no valid Δt yet (before first time step)
    update!(fv, model.velocities, model.grid, Δt)
    fv.last_update[] = key
    return nothing
end

function update_filtered_surface_state!(fs::FilteredSurfaceScalar, source_field, model)
    key = (model.clock.iteration, model.clock.stage)
    fs.last_update[] == key && return nothing
    Δt = model.clock.last_Δt
    isinf(Δt) && return nothing # no valid Δt yet (before first time step)
    update!(fs, source_field, model.grid, Δt)
    fs.last_update[] = key
    return nothing
end
