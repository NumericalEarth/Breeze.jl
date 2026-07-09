#####
##### BoundaryTendencyMarch — MPAS-style specified-zone boundary drive (#825)
#####
##### A scheme attached to the momentum `NormalFlowBoundaryCondition`s. When an
##### open lateral side carries it, the acoustic substep loop treats that side's
##### outermost interior cells as an MPAS specified zone: the acoustic
##### perturbation pressure gradient never acts on specified faces, specified
##### cells are excluded from the coupled acoustic update, the specified
##### column's (ρw)′ is closed by a per-substep zero-gradient copy from the
##### nearest interior column (WRF `zero_grad_bdy` analog), and the specified
##### zone's momentum and scalar perturbations are MARCHED by their boundary
##### time-tendencies each acoustic substep,
#####
#####     (ρu)′ ← (ρu)′ + Δτ · ∂ₜ(ρu)_boundary
#####
##### (MPAS `ru_p += dts·lbc_tend_ru`). Composed with the stage-entry rewind
##### initialization (ρu)′ = U⁰ − Uᴸ_stage, the increment recovers
##### U⁰ + β·Δt·∂ₜ at each RK-stage end. The march must be an increment: an
##### overwrite τ·∂ₜ composes with the per-stage recovery into a secular
##### (β₁+β₂+β₃) = 11/6 over-advance per outer step.
#####
##### Where a side is marched, the per-substep α relaxation of ρ′,(ρθ)′
##### (issue #738) is superseded on that side and skipped: the march holds the
##### same cells to the time-accurate boundary state directly, so the
##### relaxation's stage-frozen target is redundant. Sides without the scheme
##### keep the relaxation unchanged.
#####

"""
$(TYPEDEF)

Scheme for a momentum `NormalFlowBoundaryCondition` that drives the outermost
interior cells (an MPAS-style specified zone) by boundary value + time-tendency
each acoustic substep, instead of freezing it at the stage-entry state.

Tendency sources are callables of `(x, y, z, t)` returning the local time
derivative of the coupled quantity (``∂ₜ(ρu)``, ``∂ₜ(ρv)``, ``∂ₜρᵈ``,
``∂ₜ(ρθ)``, ``∂ₜ(ρqᵛ)``), evaluated over the specified zone once per outer
time step. Sources are evaluated on the device, so they must capture only
isbits values (the standard boundary-condition-function restriction). A source
may be `nothing`, in which case the corresponding tendency field stays zero —
a frozen hold for that variable — unless filled in place through
[`boundary_tendency_fields`](@ref) (for boundary data that cannot be evaluated
on the device, e.g. interpolated forcing files). One scheme instance must be
shared by all marched momentum boundary conditions.

The boundary *value* enters through the `NormalFlowBoundaryCondition`'s own
condition, exactly as without the scheme; the scheme adds the tendency drive.

```jldoctest
julia> using Breeze

julia> BoundaryTendencyMarch()
BoundaryTendencyMarch(ρu_tendency=nothing, ρv_tendency=nothing, ρᵈ_tendency=nothing, ρθ_tendency=nothing, ρqᵛ_tendency=nothing)
```
"""
struct BoundaryTendencyMarch{U, V, R, X, Q}
    ρu_tendency :: U
    ρv_tendency :: V
    ρᵈ_tendency :: R
    ρθ_tendency :: X
    ρqᵛ_tendency :: Q
end

"""
$(TYPEDSIGNATURES)

Build a [`BoundaryTendencyMarch`](@ref) from keyword tendency sources, each a
callable of `(x, y, z, t)` or `nothing` (see the type docstring).
"""
BoundaryTendencyMarch(; ρu_tendency = nothing,
                        ρv_tendency = nothing,
                        ρᵈ_tendency = nothing,
                        ρθ_tendency = nothing,
                        ρqᵛ_tendency = nothing) =
    BoundaryTendencyMarch(ρu_tendency, ρv_tendency, ρᵈ_tendency, ρθ_tendency, ρqᵛ_tendency)

Adapt.adapt_structure(to, scheme::BoundaryTendencyMarch) =
    BoundaryTendencyMarch(adapt(to, scheme.ρu_tendency),
                          adapt(to, scheme.ρv_tendency),
                          adapt(to, scheme.ρᵈ_tendency),
                          adapt(to, scheme.ρθ_tendency),
                          adapt(to, scheme.ρqᵛ_tendency))

Base.summary(scheme::BoundaryTendencyMarch) = "BoundaryTendencyMarch"

tendency_summary(source) = prettysummary(source)
tendency_summary(::Nothing) = "nothing"

Base.show(io::IO, scheme::BoundaryTendencyMarch) =
    print(io, "BoundaryTendencyMarch",
              "(ρu_tendency=", tendency_summary(scheme.ρu_tendency),
              ", ρv_tendency=", tendency_summary(scheme.ρv_tendency),
              ", ρᵈ_tendency=", tendency_summary(scheme.ρᵈ_tendency),
              ", ρθ_tendency=", tendency_summary(scheme.ρθ_tendency),
              ", ρqᵛ_tendency=", tendency_summary(scheme.ρqᵛ_tendency), ")")

# Scheme detection on a momentum boundary condition. A side is marched when its
# normal-momentum BC is a `NormalFlow` carrying a `BoundaryTendencyMarch`.
@inline march_scheme(bc) = nothing
@inline march_scheme(bc::BoundaryCondition{<:NormalFlow{<:BoundaryTendencyMarch}}) =
    bc.classification.scheme

# Which lateral sides are marched (isbits; threaded into the substep kernels).
struct OpenSides
    west  :: Bool
    east  :: Bool
    south :: Bool
    north :: Bool
end

march_open_sides(ρu_bcs, ρv_bcs) = OpenSides(march_scheme(ρu_bcs.west) !== nothing,
                                             march_scheme(ρu_bcs.east) !== nothing,
                                             march_scheme(ρv_bcs.south) !== nothing,
                                             march_scheme(ρv_bcs.north) !== nothing)

@inline any_marched(s::OpenSides) = s.west | s.east | s.south | s.north

# The sides whose momentum BCs carry a march scheme, or `nothing` when none
# does (the type-determined `nothing` lets downstream kernels compile the
# specified-zone branches away).
function active_march_sides(model)
    ρu_bcs = model.momentum.ρu.boundary_conditions
    ρv_bcs = model.momentum.ρv.boundary_conditions
    open_sides = march_open_sides(ρu_bcs, ρv_bcs)
    return any_marched(open_sides) ? open_sides : nothing
end

# The one scheme shared by the marched sides. The tendency fields are filled
# from a single instance, so differing instances would be silently ignored —
# error instead.
function shared_march_scheme(ρu_bcs, ρv_bcs)
    scheme = nothing
    for bc in (ρu_bcs.west, ρu_bcs.east, ρv_bcs.south, ρv_bcs.north)
        side_scheme = march_scheme(bc)
        side_scheme === nothing && continue
        if scheme === nothing
            scheme = side_scheme
        elseif scheme !== side_scheme
            error("All marched momentum boundary conditions must share one " *
                  "BoundaryTendencyMarch instance; found two distinct schemes.")
        end
    end
    return scheme
end

# `(x_specified, y_specified)`: true on the x-/y-faces whose acoustic ∂p′
# stencil reads a specified cell (MPAS `specZoneMaskEdge`). `∂xᶠᶜᶜ` at face i
# reads centers i, i−1; `∂yᶜᶠᶜ` at face j reads j, j−1; the `:xyz` launch
# writes faces 1:Nx / 1:Ny. (The min-side normal faces i=1/j=1 are inside the
# launch and marched, but the perturbation fields' impenetrability halo fill
# re-pins them to zero after every substep; the wall value re-enters through
# the prognostic momentum's own boundary condition at stage end. The max-side
# normal faces Nx+1/Ny+1 sit outside the launch and take the same fill.)
# The `Nothing` methods let the kernels compile the specified-zone branches
# away entirely when no side is marched.
@inline function specified_zone_faces(i, j, grid, s::OpenSides)
    Nx = size(grid, 1)
    Ny = size(grid, 2)
    x_specified = (s.west  & (i <= 2)) | (s.east  & (i == Nx)) | (s.south & (j == 1)) | (s.north & (j == Ny))
    y_specified = (s.south & (j <= 2)) | (s.north & (j == Ny)) | (s.west  & (i == 1)) | (s.east  & (i == Nx))
    return x_specified, y_specified
end

@inline specified_zone_faces(i, j, grid, ::Nothing) = (false, false)

# `true` if cell (i, j) is a specified cell — MPAS `specZoneMaskCell`.
@inline function specified_zone_cell(i, j, grid, s::OpenSides)
    Nx = size(grid, 1)
    Ny = size(grid, 2)
    return (s.west & (i == 1)) | (s.east & (i == Nx)) | (s.south & (j == 1)) | (s.north & (j == Ny))
end

@inline specified_zone_cell(i, j, grid, ::Nothing) = false

# Per-substep march increment Δτ·∂ₜ. The `Nothing` method keeps kernels free of
# speculative loads (and of the fields entirely) when no scheme is present.
@inline march_increment(::Nothing, i, j, k, Δτ) = zero(Δτ)
@inline march_increment(∂ₜ, i, j, k, Δτ) = @inbounds Δτ * ∂ₜ[i, j, k]

# Close the specified column's (ρw)′ by a zero-gradient copy from the nearest
# interior column (WRF `zero_grad_bdy` analog): a hard hold would be reflective
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

@inline tendency_value(::Nothing, x, y, z, t) = zero(x)
@inline tendency_value(f, x, y, z, t) = f(x, y, z, t)

@inline has_tendency_sources(scheme::BoundaryTendencyMarch) =
    !(scheme.ρu_tendency === nothing && scheme.ρv_tendency === nothing &&
      scheme.ρᵈ_tendency === nothing && scheme.ρθ_tendency === nothing &&
      scheme.ρqᵛ_tendency === nothing)

# Evaluate the scheme's tendency sources over the specified zone (zero outside
# it, and zero where the source is `nothing`). Launched once per outer step.
# Each source is bound before its `ifelse` so both branches share one type
# even when the callable's return type differs from the grid float type.
@kernel function _fill_boundary_tendencies!(∂ₜρu, ∂ₜρv, ∂ₜρᵈ, ∂ₜρθ, ∂ₜρqᵛ, grid, scheme, open_sides, t)
    i, j, k = @index(Global, NTuple)

    x_specified, y_specified = specified_zone_faces(i, j, grid, open_sides)
    cell_specified = specified_zone_cell(i, j, grid, open_sides)

    @inbounds begin
        xᶠ = xnode(i, j, k, grid, Face(), Center(), Center())
        xᶜ = xnode(i, j, k, grid, Center(), Center(), Center())
        yᶠ = ynode(i, j, k, grid, Center(), Face(), Center())
        yᶜ = ynode(i, j, k, grid, Center(), Center(), Center())
        zᶜ = znode(i, j, k, grid, Center(), Center(), Center())

        ∂ₜρu_value = tendency_value(scheme.ρu_tendency, xᶠ, yᶜ, zᶜ, t)
        ∂ₜρv_value = tendency_value(scheme.ρv_tendency, xᶜ, yᶠ, zᶜ, t)
        ∂ₜρᵈ_value = tendency_value(scheme.ρᵈ_tendency, xᶜ, yᶜ, zᶜ, t)
        ∂ₜρθ_value = tendency_value(scheme.ρθ_tendency, xᶜ, yᶜ, zᶜ, t)
        ∂ₜρqᵛ_value = tendency_value(scheme.ρqᵛ_tendency, xᶜ, yᶜ, zᶜ, t)

        ∂ₜρu[i, j, k] = ifelse(x_specified,    ∂ₜρu_value, zero(∂ₜρu_value))
        ∂ₜρv[i, j, k] = ifelse(y_specified,    ∂ₜρv_value, zero(∂ₜρv_value))
        ∂ₜρᵈ[i, j, k] = ifelse(cell_specified, ∂ₜρᵈ_value, zero(∂ₜρᵈ_value))
        ∂ₜρθ[i, j, k] = ifelse(cell_specified, ∂ₜρθ_value, zero(∂ₜρθ_value))
        ∂ₜρqᵛ[i, j, k] = ifelse(cell_specified, ∂ₜρqᵛ_value, zero(∂ₜρqᵛ_value))
    end
end

function fill_boundary_tendencies!(substepper, model)
    ∂ₜρu = substepper.boundary_momentum_tendency_u
    ∂ₜρu === nothing && return nothing

    ρu_bcs = model.momentum.ρu.boundary_conditions
    ρv_bcs = model.momentum.ρv.boundary_conditions
    scheme = shared_march_scheme(ρu_bcs, ρv_bcs)
    (scheme === nothing || !has_tendency_sources(scheme)) && return nothing

    grid = model.grid
    launch!(architecture(grid), grid, :xyz, _fill_boundary_tendencies!,
            ∂ₜρu,
            substepper.boundary_momentum_tendency_v,
            substepper.boundary_density_tendency,
            substepper.boundary_density_potential_temperature_tendency,
            substepper.boundary_moisture_tendency,
            grid, scheme, march_open_sides(ρu_bcs, ρv_bcs), model.clock.time)

    return nothing
end

# Re-impose the marched specified-zone state U⁰ + βΔt·∂ₜ on the prognostic
# fields. The substep loop leaves the zone exactly there, but post-loop stage
# physics (the vertically-implicit solve, operator-split microphysics, the
# per-stage scalar update of ρqᵛ) is not excluded from the zone; rather than
# gating every such kernel, the zone is restored afterwards — the WRF
# specified-zone contract, where interior physics never acts on the zone.
# (ρw)′ has no boundary data and its zero-gradient closure is a
# perturbation-space relation that cannot be reconstructed post-recovery, so
# ρw is deliberately left to the column-local implicit operator.
@kernel function _reimpose_specified_zone!(ρu, ρv, ρᵈ, ρθ, ρqᵛ, grid,
                                           ρu⁰, ρv⁰, ρᵈ⁰, ρθ⁰, ρqᵛ⁰,
                                           ∂ₜρu, ∂ₜρv, ∂ₜρᵈ, ∂ₜρθ, ∂ₜρqᵛ,
                                           march_sides, βΔt)
    i, j, k = @index(Global, NTuple)

    x_specified, y_specified = specified_zone_faces(i, j, grid, march_sides)
    cell_specified = specified_zone_cell(i, j, grid, march_sides)

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

Restore the marched prognostics (ρu, ρv, ρᵈ, the thermodynamic density, and
the moisture density) to the specified-zone state `U⁰ + βΔt·∂ₜ` after
post-substep-loop stage physics has acted on the zone. Thin per-side window
launches; corner overlap writes identical values. A no-op when no side is
marched. With `fill_halos = true` (the once-per-step call, after the
operator-split microphysics update) the affected prognostic halos are
refilled, since the boundary-condition fills extrapolate from the re-imposed
zone cells.
"""
function reimpose_specified_zone!(substepper, model, βΔt; fill_halos = false)
    ∂ₜρu = substepper.boundary_momentum_tendency_u
    ∂ₜρu === nothing && return nothing
    march_sides = active_march_sides(model)
    march_sides === nothing && return nothing

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
            ∂ₜρu,
            substepper.boundary_momentum_tendency_v,
            substepper.boundary_density_tendency,
            substepper.boundary_density_potential_temperature_tendency,
            substepper.boundary_moisture_tendency,
            march_sides, convert(eltype(grid), βΔt))

    # x-face windows span i ≤ 2 / i = Nx (the `x_specified` band); the cell and
    # y-face writes inside each window are decided by the full predicates.
    march_sides.west  && launch!(arch, grid, KernelParameters(1:2, 1:Ny, 1:Nz),  _reimpose_specified_zone!, args...)
    march_sides.east  && launch!(arch, grid, KernelParameters(Nx:Nx, 1:Ny, 1:Nz), _reimpose_specified_zone!, args...)
    march_sides.south && launch!(arch, grid, KernelParameters(1:Nx, 1:2, 1:Nz),  _reimpose_specified_zone!, args...)
    march_sides.north && launch!(arch, grid, KernelParameters(1:Nx, Ny:Ny, 1:Nz), _reimpose_specified_zone!, args...)

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

Return the substepper's boundary tendency fields
`(ρu = ..., ρv = ..., ρᵈ = ..., ρθ = ..., ρqᵛ = ...)` for a model whose
momentum boundary conditions carry a [`BoundaryTendencyMarch`](@ref) scheme.
The fields hold ``∂ₜ(ρu)``, ``∂ₜ(ρv)``, ``∂ₜρᵈ``, ``∂ₜ(ρθ)``, ``∂ₜ(ρqᵛ)`` over
the specified zone (zero elsewhere) and may be filled in place by a driver
each outer time step as an alternative to callable tendency sources, e.g.
when the boundary data comes from interpolated files that cannot be evaluated
on the device. When callable sources ARE provided, the fields are refreshed
from them once per outer step and external fills are overwritten.
"""
boundary_tendency_fields(model) =
    (ρu = model.timestepper.substepper.boundary_momentum_tendency_u,
     ρv = model.timestepper.substepper.boundary_momentum_tendency_v,
     ρᵈ = model.timestepper.substepper.boundary_density_tendency,
     ρθ = model.timestepper.substepper.boundary_density_potential_temperature_tendency,
     ρqᵛ = model.timestepper.substepper.boundary_moisture_tendency)
