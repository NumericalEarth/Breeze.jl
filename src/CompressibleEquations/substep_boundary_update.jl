#####
##### SubstepBoundaryUpdate — specified-zone boundary update
#####
##### A scheme attached to the momentum `NormalFlowBoundaryCondition`s. When an
##### open lateral side carries it, the acoustic substep loop treats that side's
##### outermost interior cells as a specified zone: the acoustic
##### perturbation pressure gradient never acts on specified faces, specified
##### cells are excluded from the coupled acoustic update, the specified
##### column's (ρw)′ is closed by a per-substep zero-gradient copy from the
##### nearest interior column, and the specified
##### zone's momentum and scalar perturbations are updated each acoustic substep
##### by an increment, (ρu)′ ← (ρu)′ + Δτ·∂ₜ(ρu)_boundary. The increment —
##### not an overwrite — is load-
##### bearing: see `.agents/substepping.md` for the composition with the
##### stage-entry rewind and the 11/6 secular-drift argument, and for the full
##### WRF/MPAS correspondence.
#####
##### Where a side is specified, the per-substep α relaxation of ρ′,(ρθ)′
##### (issue #738) is superseded on that side and skipped: the update holds the
##### same cells to the time-accurate boundary state directly, so the
##### relaxation's stage-frozen target is redundant. Sides without the scheme
##### keep the relaxation unchanged.
#####

"""
$(TYPEDEF)

Marker scheme for a momentum `NormalFlowBoundaryCondition` that updates the
outermost interior cells (an MPAS-style specified zone) by boundary value +
time-tendency each acoustic substep, instead of freezing them at the
stage-entry state.

The scheme carries no data. The specified-zone time-tendencies
(``∂ₜ(ρu)``, ``∂ₜ(ρv)``, ``∂ₜρᵈ``, ``∂ₜ(ρθ)``, ``∂ₜ(ρqᵛ)``) are held in fields
exposed by [`boundary_tendencies`](@ref), which a driver fills in place
over the specified zone each outer time step (e.g. from a parent model or from
interpolated forcing files). A field left zero holds its variable frozen.

The boundary *value* enters through the `NormalFlowBoundaryCondition`'s own
condition, exactly as without the scheme; the scheme adds the tendency update.

```jldoctest
julia> using Breeze

julia> SubstepBoundaryUpdate()
SubstepBoundaryUpdate()
```
"""
struct SubstepBoundaryUpdate end

Base.summary(::SubstepBoundaryUpdate) = "SubstepBoundaryUpdate"

# Scheme detection on a momentum boundary condition. A side is specified when its
# normal-momentum BC is a `NormalFlow` carrying a `SubstepBoundaryUpdate`.
@inline specified_zone_scheme(bc) = nothing
@inline specified_zone_scheme(bc::BoundaryCondition{<:NormalFlow{<:SubstepBoundaryUpdate}}) =
    bc.classification.scheme

# Which lateral sides are specified (isbits; threaded into the substep kernels).
struct OpenSides
    west  :: Bool
    east  :: Bool
    south :: Bool
    north :: Bool
end

specified_open_sides(ρu_bcs, ρv_bcs) = OpenSides(specified_zone_scheme(ρu_bcs.west) !== nothing,
                                             specified_zone_scheme(ρu_bcs.east) !== nothing,
                                             specified_zone_scheme(ρv_bcs.south) !== nothing,
                                             specified_zone_scheme(ρv_bcs.north) !== nothing)

@inline any_specified(s::OpenSides) = s.west | s.east | s.south | s.north

# The sides whose momentum BCs carry a specified-zone scheme, or `nothing` when none
# does (the type-determined `nothing` lets downstream kernels compile the
# specified-zone branches away).
function active_specified_sides(model)
    ρu_bcs = model.momentum.ρu.boundary_conditions
    ρv_bcs = model.momentum.ρv.boundary_conditions
    open_sides = specified_open_sides(ρu_bcs, ρv_bcs)
    return any_specified(open_sides) ? open_sides : nothing
end

# `(x_specified, y_specified)`: true on the x-/y-faces whose acoustic ∂p′
# stencil reads a specified cell. `∂xᶠᶜᶜ` at face i
# reads centers i, i−1; `∂yᶜᶠᶜ` at face j reads j, j−1; the `:xyz` launch
# writes faces 1:Nx / 1:Ny. (The min-side normal faces i=1/j=1 are inside the
# launch and specified, but the perturbation fields' impenetrability halo fill
# re-pins them to zero after every substep; the wall value re-enters through
# the prognostic momentum's own boundary condition at stage end. The max-side
# normal faces Nx+1/Ny+1 sit outside the launch and take the same fill.)
# The `Nothing` methods let the kernels compile the specified-zone branches
# away entirely when no side is specified.
@inline function specified_zone_faces(i, j, grid, s::OpenSides)
    Nx = size(grid, 1)
    Ny = size(grid, 2)
    x_specified = (s.west  & (i <= 2)) | (s.east  & (i == Nx)) | (s.south & (j == 1)) | (s.north & (j == Ny))
    y_specified = (s.south & (j <= 2)) | (s.north & (j == Ny)) | (s.west  & (i == 1)) | (s.east  & (i == Nx))
    return x_specified, y_specified
end

@inline specified_zone_faces(i, j, grid, ::Nothing) = (false, false)

# `true` if cell (i, j) is a specified cell.
@inline function specified_zone_cell(i, j, grid, s::OpenSides)
    Nx = size(grid, 1)
    Ny = size(grid, 2)
    return (s.west & (i == 1)) | (s.east & (i == Nx)) | (s.south & (j == 1)) | (s.north & (j == Ny))
end

@inline specified_zone_cell(i, j, grid, ::Nothing) = false

# Per-substep specified-zone increment Δτ·∂ₜ. The `Nothing` method keeps kernels free of
# speculative loads (and of the fields entirely) when no scheme is present.
@inline specified_zone_increment(::Nothing, i, j, k, Δτ) = zero(Δτ)
@inline specified_zone_increment(∂ₜ, i, j, k, Δτ) = @inbounds Δτ * ∂ₜ[i, j, k]

# Per-variable access to the bundled specified-zone boundary tendencies. Returns
# the field for a specified substepper (a `NamedTuple` with keys `ρu, ρv, ρᵈ, ρθ,
# ρqᵛ`), or `nothing` when the substepper was built without the scheme (the whole
# bundle is `nothing`) so the substep launches pass `nothing` and the kernels
# dispatch the update away. `Val` keeps the heterogeneous-field access type-stable.
@inline boundary_tendency(bt::NamedTuple, ::Val{name}) where {name} = getproperty(bt, name)
@inline boundary_tendency(::Nothing, ::Val) = nothing

# Close the specified column's (ρw)′ by a zero-gradient copy from the nearest
# interior column: a hard hold would be reflective
# at the specified/interior seam, and the boundary data carries no w. A
# divergent `if` is deliberate here — an `ifelse` would force the (i2, j2)
# gather at every thread. The copy is race-free for min(Nx, Ny) ≥ 3: only
# specified cells write (ρw)′ in the enclosing kernel and (i2, j2) is never a
# specified cell (corners compose diagonally). Specified threads' recovery
# reads of neighboring (ρw)′ may race with these writes, but those reads feed
# only the discarded branch of the recovery `ifelse`.
@inline replace_specified_column_vertical_momentum!(ρw′, i, j, k, grid, ::Nothing) = nothing

@inline function replace_specified_column_vertical_momentum!(ρw′, i, j, k, grid, s::OpenSides)
    if specified_zone_cell(i, j, grid, s)
        Nx = size(grid, 1)
        Ny = size(grid, 2)
        i2 = i + ifelse(s.west  & (i == 1),  1, 0) - ifelse(s.east  & (i == Nx), 1, 0)
        j2 = j + ifelse(s.south & (j == 1),  1, 0) - ifelse(s.north & (j == Ny), 1, 0)
        @inbounds ρw′[i, j, k] = ρw′[i2, j2, k]
    end
    return nothing
end

# Re-impose the specified-zone state U⁰ + βΔt·∂ₜ on the prognostic
# fields. The substep loop leaves the zone exactly there, but post-loop stage
# physics (the vertically-implicit solve, operator-split microphysics, the
# per-stage scalar update of ρqᵛ) is not excluded from the zone; rather than
# gating every such kernel, the zone is restored afterwards — the
# specified-zone contract, where interior physics never acts on the zone.
# (ρw)′ has no boundary data and its zero-gradient closure is a
# perturbation-space relation that cannot be reconstructed post-recovery, so
# ρw is deliberately left to the column-local implicit operator.
@kernel function _reimpose_specified_zone!(ρu, ρv, ρᵈ, ρθ, ρqᵛ, grid,
                                           ρu⁰, ρv⁰, ρᵈ⁰, ρθ⁰, ρqᵛ⁰,
                                           ∂ₜρu, ∂ₜρv, ∂ₜρᵈ, ∂ₜρθ, ∂ₜρqᵛ,
                                           specified_sides, βΔt)
    i, j, k = @index(Global, NTuple)

    x_specified, y_specified = specified_zone_faces(i, j, grid, specified_sides)
    cell_specified = specified_zone_cell(i, j, grid, specified_sides)

    @inbounds begin
        ρu[i, j, k]  = ifelse(x_specified,    ρu⁰[i, j, k]  + βΔt * ∂ₜρu[i, j, k],  ρu[i, j, k])
        ρv[i, j, k]  = ifelse(y_specified,    ρv⁰[i, j, k]  + βΔt * ∂ₜρv[i, j, k],  ρv[i, j, k])
        ρᵈ[i, j, k]  = ifelse(cell_specified, ρᵈ⁰[i, j, k]  + βΔt * ∂ₜρᵈ[i, j, k],  ρᵈ[i, j, k])
        ρθ[i, j, k]  = ifelse(cell_specified, ρθ⁰[i, j, k]  + βΔt * ∂ₜρθ[i, j, k],  ρθ[i, j, k])
        ρqᵛ[i, j, k] = ifelse(cell_specified, ρqᵛ⁰[i, j, k] + βΔt * ∂ₜρqᵛ[i, j, k], ρqᵛ[i, j, k])
    end
end

"""
$(TYPEDSIGNATURES)

Restore the specified prognostics (ρu, ρv, ρᵈ, the thermodynamic density, and
the moisture density) to the specified-zone state `U⁰ + βΔt·∂ₜ` after
post-substep-loop stage physics has acted on the zone. Thin per-side window
launches; corner overlap writes identical values. A no-op when no side is
specified. With `fill_halos = true` (the once-per-step call, after the
operator-split microphysics update) the affected prognostic halos are
refilled, since the boundary-condition fills extrapolate from the re-imposed
zone cells.
"""
function reimpose_specified_zone!(substepper, model, βΔt; fill_halos = false)
    bt = substepper.boundary_tendencies
    bt === nothing && return nothing
    specified_sides = active_specified_sides(model)
    specified_sides === nothing && return nothing

    grid = model.grid
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    U⁰ = model.timestepper.U⁰

    ρᵈ = model.dynamics.dry_density
    ρᵡ = thermodynamic_density(model.formulation)
    ρᵡ_name = thermodynamic_density_name(model.formulation)
    ρqᵛ = model.moisture_density
    ρqᵛ_name = moisture_prognostic_name(model.microphysics)

    args = (model.momentum.ρu, model.momentum.ρv, ρᵈ, ρᵡ, ρqᵛ, grid,
            U⁰.ρu, U⁰.ρv, U⁰.ρᵈ, getproperty(U⁰, ρᵡ_name), getproperty(U⁰, ρqᵛ_name),
            bt.ρu, bt.ρv, bt.ρᵈ, bt.ρθ, bt.ρqᵛ,
            specified_sides, convert(eltype(grid), βΔt))

    # x-face windows span i ≤ 2 / i = Nx (the `x_specified` band); the cell and
    # y-face writes inside each window are decided by the full predicates.
    specified_sides.west  && launch!(arch, grid, KernelParameters(1:2, 1:Ny, 1:Nz),  _reimpose_specified_zone!, args...)
    specified_sides.east  && launch!(arch, grid, KernelParameters(Nx:Nx, 1:Ny, 1:Nz), _reimpose_specified_zone!, args...)
    specified_sides.south && launch!(arch, grid, KernelParameters(1:Nx, 1:2, 1:Nz),  _reimpose_specified_zone!, args...)
    specified_sides.north && launch!(arch, grid, KernelParameters(1:Nx, Ny:Ny, 1:Nz), _reimpose_specified_zone!, args...)

    if fill_halos
        fill_halo_regions!(ρᵈ, boundary_condition_args(model)...)
        fill_halo_regions!(ρᵡ, boundary_condition_args(model)...)
        fill_halo_regions!(ρqᵛ, boundary_condition_args(model)...)
        fill_halo_regions!(model.momentum, boundary_condition_args(model)...)
    end

    return nothing
end

"""
$(TYPEDSIGNATURES)

Return the substepper's bundled boundary tendency fields
`(ρu = ..., ρv = ..., ρᵈ = ..., ρθ = ..., ρqᵛ = ...)` for a model whose
momentum boundary conditions carry a [`SubstepBoundaryUpdate`](@ref) scheme, or
`nothing` for a model built without it. The fields hold ``∂ₜ(ρu)``, ``∂ₜ(ρv)``,
``∂ₜρᵈ``, ``∂ₜ(ρθ)``, ``∂ₜ(ρqᵛ)`` over the specified zone (only their
specified-zone entries are ever read). A driver fills them in place each outer
time step — e.g. from a parent model or from interpolated forcing files — and
the values persist until overwritten. A field left zero holds its variable
frozen.
"""
boundary_tendencies(model) = model.timestepper.substepper.boundary_tendencies
