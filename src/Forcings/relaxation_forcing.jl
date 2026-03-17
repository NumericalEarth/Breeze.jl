using ..AtmosphereModels: AtmosphereModels
using Oceananigans: Average, Field, set!, compute!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: AbstractField, location, interior
using Oceananigans.Grids: Center, λnodes, φnodes, znode
using Oceananigans.Units: Time
using Oceananigans.Utils: prettysummary
using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt

#####
##### RelaxationForcing struct
#####

struct RelaxationForcing{R, TF, Cl, D, F, RC, TS, ZB}
    reference        :: R    # FieldTimeSeries of the specific variable ϕᵣ
    target           :: TF   # Field of ϕᵣ interpolated to current time, horizontally averaged to a single column in profile mode
    clock            :: Cl   # Model Clock
    density          :: D    # Reference density ρᵣ(z)
    current_field    :: F    # Specific field ϕ (3D) or Average(ϕ) (profile)
    time_scale       :: TS   # Relaxation time scale (seconds)
    reference_column :: RC   # Nothing (3D mode) or NTuple{2,Int} (profile mode)
    z_bottom         :: ZB   # Height below which no nudging is applied (meters)
end

"""
$(TYPEDSIGNATURES)

Forcing that represents Newtonian relaxation (nudging) of a prognostic field
toward a reference [`FieldTimeSeries`](@ref):

```math
F_{ρ ϕ} = -\\frac{ρ_r \\left(\\bar{ϕ} - ϕ_r\\right)}{τ} \\, \\mathbf{1}_{z \\geq z_b}
```

where ``\\bar{ϕ}`` is either the local field value (3D mode) or its horizontal
average (profile mode), ``ϕ_r`` is the time-interpolated reference value,
``ρ_r`` is the reference density, ``τ`` is the relaxation time scale, and
``z_b`` is the cutoff height below which no nudging is applied.

The `reference` [`FieldTimeSeries`](@ref) must contain the specific variable
(e.g. ``u``, ``v``, ``θ``, ``q^t``), not the density-weighted form. The user is
responsible for ensuring the reference variable matches the prognostic field being
forced (e.g., do not apply a ``θ`` reference to ``ρe`` in a
`StaticEnergyFormulation` model). The reference data must be pre-interpolated
onto the simulation's vertical grid.

# Keyword arguments

- `time_scale`: Relaxation time scale in seconds.
- `z_bottom`: Height in meters below which nudging is not applied. Default: 1500 m.
- `reference_position`: Optional `(latitude=..., longitude=...)` `NamedTuple`
  specifying the column to extract from a 3D `reference` for profile nudging.

# Modes

- **Profile mode**: activated when `reference_position` is provided, or when
  `reference` is already a 1×1×Nz column `FieldTimeSeries`. The domain-mean
  horizontal average ``\\overline{ϕ}(k)`` is relaxed toward the reference
  column ``ϕ_r(k, t)``. The same tendency is applied uniformly in (i, j).
- **3D mode**: activated when `reference_position` is `nothing` and `reference`
  has full horizontal extent. Each local value ``ϕ(i, j, k)`` is nudged toward
  ``ϕ_r(i, j, k, t)``. The `reference` must be defined on the simulation grid.

# Example

```jldoctest
using Breeze

grid = RectilinearGrid(size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 3000))
fts = FieldTimeSeries{Center, Center, Center}(grid, [0.0, 3600.0])
nudging = RelaxationForcing(fts; time_scale=3600.0)
nudging.z_bottom

# output
1500
```
"""
function RelaxationForcing(reference; time_scale, z_bottom=1500, reference_position=nothing)
    return RelaxationForcing(reference, nothing, nothing, nothing, nothing,
                             time_scale, reference_position, z_bottom)
end

#####
##### Column resolution for profile mode
#####

"""
$(TYPEDSIGNATURES)

Return the `(i, j)` index in `grid` nearest to
`reference_position = (latitude=..., longitude=...)`.

Performs nearest-neighbor search independently along x (longitude) and y
(latitude). Correct for rectilinear lat-lon grids; a 2D search would be required
for fully curvilinear grids (TODO).
"""
function find_reference_column(grid, reference_position)
    λ₀ = reference_position.longitude
    φ₀ = reference_position.latitude
    λ_nodes = λnodes(grid, Center(); with_halos=false)
    φ_nodes = φnodes(grid, Center(); with_halos=false)
    i_ref = argmin(abs.(λ_nodes .- λ₀))
    j_ref = argmin(abs.(φ_nodes .- φ₀))
    return (i_ref, j_ref)
end

# User provided a lat/lon reference position → profile mode
function resolve_reference_column(fts, reference_position::NamedTuple)
    fts_Nx, fts_Ny, _ = size(fts.grid)
    fts_Nx == 1 && fts_Ny == 1 && return (1, 1)  # already a column; reference_position unused
    return find_reference_column(fts.grid, reference_position)
end

# No reference position given: profile mode if FTS is already a 1D column, else 3D mode
function resolve_reference_column(fts, ::Nothing)
    fts_Nx, fts_Ny, _ = size(fts.grid)
    if fts_Nx == 1 && fts_Ny == 1
        return (1, 1)
    else
        return nothing
    end
end

#####
##### Materialization
#####

function AtmosphereModels.materialize_atmosphere_model_forcing(forcing::RelaxationForcing, field, name,
                                                               model_field_names, context::NamedTuple)
    grid = field.grid

    specific_name = strip_density_prefix(name)
    specific_field = context.specific_fields[specific_name]

    reference_column = resolve_reference_column(forcing.reference, forcing.reference_column)
    profile_mode = !isnothing(reference_column)

    _, _, LZ = location(specific_field)

    if profile_mode
        target = Field{Nothing, Nothing, LZ}(grid)
        current_field = Average(specific_field, dims=(1, 2)) |> Field
    else
        LX, LY, _ = location(specific_field)
        target = Field{LX, LY, LZ}(grid)
        current_field = specific_field
    end

    return RelaxationForcing(forcing.reference, target, context.clock, context.density,
                             current_field, forcing.time_scale, reference_column, forcing.z_bottom)
end

#####
##### compute_forcing!: update target and current field each time step
#####

function AtmosphereModels.compute_forcing!(forcing::RelaxationForcing)
    t = forcing.clock.time
    ref_field = forcing.reference[Time(t)] # interpolate to t
    update_target!(forcing.target, ref_field, forcing.reference_column)
    compute!(forcing.current_field)
    return nothing
end

# 3D mode: copy the full reference field to target
function update_target!(target, ref_field, ::Nothing)
    set!(target, ref_field)
    fill_halo_regions!(target)
    return nothing
end

# Profile mode: extract the column at (i_ref, j_ref) and copy into the 1D target
function update_target!(target, ref_field, (i_ref, j_ref)::NTuple{2, Int})
    Nz = size(target, 3)
    col = interior(ref_field, i_ref, j_ref, :)
    set!(target, reshape(col, 1, 1, Nz))
    fill_halo_regions!(target)
    return nothing
end

#####
##### Kernel callables
#####

# 3D mode (reference_column = Nothing): compare local value to 3D reference
@inline function (f::RelaxationForcing{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing})(i, j, k, grid, clock, fields)
    z  = znode(i, j, k, grid, Center(), Center(), Center())
    ϕ  = @inbounds f.current_field[i, j, k]
    ϕᵣ = @inbounds f.target[i, j, k]
    ρ  = @inbounds f.density[1, 1, k]
    tendency = -ρ * (ϕ - ϕᵣ) / f.time_scale
    return ifelse(z < f.z_bottom, 0, tendency)
end

# Profile mode (reference_column = NTuple{2,Int}): compare horizontal average to reference column
@inline function (f::RelaxationForcing{<:Any, <:Any, <:Any, <:Any, <:Any, <:NTuple{2}})(i, j, k, grid, clock, fields)
    z  = znode(i, j, k, grid, Center(), Center(), Center())
    ϕ̄  = @inbounds f.current_field[1, 1, k]
    ϕᵣ = @inbounds f.target[1, 1, k]
    ρ  = @inbounds f.density[1, 1, k]
    tendency = -ρ * (ϕ̄ - ϕᵣ) / f.time_scale
    return ifelse(z < f.z_bottom, 0, tendency)
end

#####
##### GPU adaptation
#####

Adapt.adapt_structure(to, f::RelaxationForcing) =
    RelaxationForcing(Adapt.adapt(to, f.reference),
                      Adapt.adapt(to, f.target),
                      f.clock,
                      Adapt.adapt(to, f.density),
                      Adapt.adapt(to, f.current_field),
                      f.time_scale,
                      f.reference_column,
                      f.z_bottom)

#####
##### Show
#####

function Base.summary(f::RelaxationForcing)
    isnothing(f.target) && return "RelaxationForcing (pre-materialization)"
    isnothing(f.reference_column) && return "RelaxationForcing (3D mode)"
    return "RelaxationForcing (profile mode)"
end

function Base.show(io::IO, f::RelaxationForcing)
    print(io, summary(f), "\n")
    print(io, "├── time_scale: ", prettysummary(f.time_scale), " seconds\n")
    print(io, "├── z_bottom: ", prettysummary(f.z_bottom), " m\n")
    if isnothing(f.target)
        print(io, "└── reference: ", prettysummary(f.reference))
    else
        print(io, "├── target: ", prettysummary(f.target), "\n")
        print(io, "└── current_field: ", prettysummary(f.current_field))
    end
end
