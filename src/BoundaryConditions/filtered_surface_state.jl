#####
##### Filtered Surface State for Bulk Boundary Conditions
#####
##### Temporal filtering of near-surface velocity and scalar fields
##### to mitigate spurious u*-u' correlations and resolution dependence
##### in bulk flux computations (Shin, Yang & Howland 2025;
##### Nishizawa & Kitamura 2018).
#####
##### `FilteredSurfaceVelocities` holds the filtered velocity components and the
##### filtered virtual potential temperature `θᵥ` used by stability-dependent bulk
##### coefficients. Per-BC scalar differences (θ, e, qᵛ) are held separately in
##### `FilteredSurfaceScalar`.
#####

using KernelAbstractions: @kernel, @index
using Oceananigans: architecture
using Oceananigans.Fields: interpolate
using Oceananigans.Grids: xnode, ynode, topology, Flat
using Oceananigans.Utils: launch!, KernelParameters, prettysummary

#####
##### FilteredSurfaceVelocities
#####

struct FilteredSurfaceVelocities{U, V, Θ, H, FT, R, RΘ}
    u :: U   # Field{Face, Center, Nothing}
    v :: V   # Field{Center, Face, Nothing}
    θᵥ :: Θ  # Field{Center, Center, Nothing} — filtered virtual potential temperature
    height :: H
    filter_timescale :: FT
    last_update :: R     # Ref{Tuple{Int, Int}} on CPU, Tuple{Int, Int} on GPU (for u, v)
    last_θᵥ_update :: RΘ # independent Ref for θᵥ (its source may be nothing)
end

"""
    FilteredSurfaceVelocities(grid; height=nothing, filter_timescale=Inf)

Two-dimensional fields storing temporally filtered near-surface velocities
and virtual potential temperature for use in bulk flux boundary conditions.

The filtered velocities `ū`, `v̄` (and `θ̄ᵥ` when a stability-dependent bulk
coefficient is attached) are updated each time step via an exponential
(first-order) filter:

    ū ← (ū + ϵ u_new) / (1 + ϵ),     ϵ = Δt / τ

where `τ` is the `filter_timescale`.

`θ̄ᵥ` is updated from the virtual-potential-temperature diagnostic owned by any
attached `PolynomialCoefficient`; when the bulk coefficient is a plain `Number`
(no stability correction) the `θ̄ᵥ` field is allocated but unused.

# Keyword Arguments

- `height`: Reference height (m) for velocity evaluation. If `nothing` (default),
  the first grid cell center value is used. If a number, velocity is linearly
  interpolated to that height. (`θ̄ᵥ` is always read at the first cell center,
  matching the height at which `bulk_coefficient` evaluates stability.)
- `filter_timescale`: Filter time scale `τ` in seconds (default: `Inf`, no filtering).
"""
function FilteredSurfaceVelocities(grid; height=nothing, filter_timescale=Inf)
    u  = Field{Face, Center, Nothing}(grid)
    v  = Field{Center, Face, Nothing}(grid)
    θᵥ = Field{Center, Center, Nothing}(grid)
    FT = typeof(filter_timescale)
    return FilteredSurfaceVelocities(u, v, θᵥ, height, FT(filter_timescale),
                                     Ref((0, 0)), Ref((0, 0)))
end

_deref(r::Ref) = r[]
_deref(t::Tuple) = t

Adapt.adapt_structure(to, fv::FilteredSurfaceVelocities) =
    FilteredSurfaceVelocities(Adapt.adapt(to, fv.u),
                              Adapt.adapt(to, fv.v),
                              Adapt.adapt(to, fv.θᵥ),
                              fv.height,
                              fv.filter_timescale,
                              _deref(fv.last_update),
                              _deref(fv.last_θᵥ_update))

Base.summary(fv::FilteredSurfaceVelocities) =
    string("FilteredSurfaceVelocities(height=", fv.height,
           ", filter_timescale=", fv.filter_timescale, ")")

function Base.show(io::IO, fv::FilteredSurfaceVelocities)
    print(io, summary(fv), ":\n",
          "├── u: ", prettysummary(fv.u), '\n',
          "├── v: ", prettysummary(fv.v), '\n',
          "├── θᵥ: ", prettysummary(fv.θᵥ), '\n',
          "├── height: ", prettysummary(fv.height), '\n',
          "├── filter_timescale: ", prettysummary(fv.filter_timescale), '\n',
          "├── last_update: ", fv.last_update, '\n',
          "└── last_θᵥ_update: ", fv.last_θᵥ_update)
end

#####
##### FilteredSurfaceScalar
#####

struct FilteredSurfaceScalar{F, H, FT, R}
    field :: F   # Field{Center, Center, Nothing}
    height :: H
    filter_timescale :: FT
    last_update :: R  # Ref{Tuple{Int, Int}} on CPU, Tuple{Int, Int} on GPU
end

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
function FilteredSurfaceScalar(grid; height=nothing, filter_timescale=Inf)
    field = Field{Center, Center, Nothing}(grid)
    FT = typeof(filter_timescale)
    return FilteredSurfaceScalar(field, height, FT(filter_timescale), Ref((0, 0)))
end

Adapt.adapt_structure(to, fs::FilteredSurfaceScalar) =
    FilteredSurfaceScalar(Adapt.adapt(to, fs.field),
                          fs.height,
                          fs.filter_timescale,
                          _deref(fs.last_update))

Base.summary(fs::FilteredSurfaceScalar) =
    string("FilteredSurfaceScalar(height=", fs.height,
           ", filter_timescale=", fs.filter_timescale, ")")

function Base.show(io::IO, fs::FilteredSurfaceScalar)
    print(io, summary(fs), ":\n",
          "├── field: ", prettysummary(fs.field), '\n',
          "├── height: ", prettysummary(fs.height), '\n',
          "├── filter_timescale: ", prettysummary(fs.filter_timescale), '\n',
          "└── last_update: ", fs.last_update)
end

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

# θᵥ is always read at the first cell (matches the height at which the bulk
# coefficient evaluates stability). The source field may be a
# `KernelFunctionOperation`, for which `interpolate` is not generally defined.
@kernel function _update_filtered_θᵥ!(θ̂ᵥ, θᵥ_source, ϵ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        θⁿᵥ = θᵥ_source[i, j, 1]
        θ̂ᵥ[i, j, 1] = (θ̂ᵥ[i, j, 1] + ϵ * θⁿᵥ) / (1 + ϵ)
    end
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

@kernel function _initialize_filtered_θᵥ!(θ̂ᵥ, θᵥ_source)
    i, j = @index(Global, NTuple)
    @inbounds θ̂ᵥ[i, j, 1] = θᵥ_source[i, j, 1]
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

"""
    initialize_θᵥ!(fv::FilteredSurfaceVelocities, θᵥ_source, grid)

Set the filtered virtual potential temperature to the current first-cell value.
"""
function initialize_θᵥ!(fv::FilteredSurfaceVelocities, θᵥ_source, grid)
    arch = architecture(grid)
    kp = filtered_kernel_parameters(grid)
    launch!(arch, grid, kp, _initialize_filtered_θᵥ!, fv.θᵥ, θᵥ_source)
    return nothing
end

"""
    update_θᵥ!(fv::FilteredSurfaceVelocities, θᵥ_source, grid, Δt)

Apply the exponential filter to the virtual potential temperature.
"""
function update_θᵥ!(fv::FilteredSurfaceVelocities, θᵥ_source, grid, Δt)
    arch = architecture(grid)
    kp = filtered_kernel_parameters(grid)
    ϵ = Δt / fv.filter_timescale
    launch!(arch, grid, kp, _update_filtered_θᵥ!, fv.θᵥ, θᵥ_source, ϵ)
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

# θᵥ filter — dedup-aware variants. No-op when either the FilteredSurfaceVelocities
# or the source field is absent (constant bulk coefficient ⇒ no stability ⇒ no θᵥ).
initialize_filtered_θᵥ!(::Nothing, model) = nothing
initialize_filtered_θᵥ!(::Nothing, source_field, model) = nothing
initialize_filtered_θᵥ!(fv::FilteredSurfaceVelocities, ::Nothing, model) = nothing

function initialize_filtered_θᵥ!(fv::FilteredSurfaceVelocities, θᵥ_source, model)
    initialize_θᵥ!(fv, θᵥ_source, model.grid)
    return nothing
end

update_filtered_θᵥ!(::Nothing, model) = nothing
update_filtered_θᵥ!(::Nothing, source_field, model) = nothing
update_filtered_θᵥ!(fv::FilteredSurfaceVelocities, ::Nothing, model) = nothing

function update_filtered_θᵥ!(fv::FilteredSurfaceVelocities, θᵥ_source, model)
    key = (model.clock.iteration, model.clock.stage)
    fv.last_θᵥ_update[] == key && return nothing
    Δt = model.clock.last_Δt
    isinf(Δt) && return nothing
    update_θᵥ!(fv, θᵥ_source, model.grid, Δt)
    fv.last_θᵥ_update[] = key
    return nothing
end
