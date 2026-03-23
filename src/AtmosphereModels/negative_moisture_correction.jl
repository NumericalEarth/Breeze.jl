#####
##### Negative moisture correction (vertical borrowing)
#####
#
# After advection, individual moisture species can become negative because
# the advection operator might not be positive-definite. This correction:
#
#   1. Same-level borrowing: fix negatives by borrowing from related species
#      in the same grid cell, following the chain:
#        rain <- cloud liquid <- vapor  (warm phase)
#   2. Vertical borrowing: remaining negative vapor is fixed by transferring
#      mass from adjacent levels, sweeping top->bottom then one upward step.
#
# Conservation:
#   - Same-level: total moisture at each level is preserved (sum unchanged).
#   - Vertical: column-integrated moisture is preserved (Δz-weighted).
#   - Energy: no explicit adjustment needed because Breeze's thermodynamic
#     prognostics (θ_li or moist static energy) are conserved under phase
#     changes. Temperature is correctly rediagnosed at the next auxiliary
#     variable update.
#####

"""
$(TYPEDSIGNATURES)

Return a tuple of `Field` objects for density-weighted prognostic moisture mass
fields that participate in the negative-moisture correction, ordered from heaviest
hydrometeor to lightest.

Each field borrows from the next in the chain. The lightest field borrows from
the moisture prognostic (vapor or equilibrium moisture, stored in
`model.moisture_density`). Remaining vapor deficits are fixed by vertical borrowing.

Default: empty tuple (no correction).
"""
correction_moisture_fields(microphysics, microphysical_fields) = ()

"""
$(TYPEDSIGNATURES)

Return a tuple of `(number_field, mass_field)` pairs for number concentration
consistency. After same-level mass borrowing, any number field whose corresponding
mass field is non-positive is zeroed to avoid unphysical states (e.g., finite
droplet number with zero mass).

Default: empty tuple (no number fields to correct).
"""
correction_number_mass_pairs(microphysics, microphysical_fields) = ()

"""
$(TYPEDSIGNATURES)

Return a tuple of `Field` objects for density-weighted number concentration
fields that should be clamped to non-negative after advection.

Number concentrations can become negative because the advection scheme
might not be positive-definite. Unlike mass fields (which use borrowing to
preserve conservation), number concentrations are simply zeroed since there
is no meaningful conservation constraint for droplet number.

Default: empty tuple (no number fields to clamp).
"""
correction_number_fields(microphysics, microphysical_fields) = ()

"""
$(TYPEDSIGNATURES)

Fix negative moisture mixing ratios produced by the advection operator.

Operates in two phases:
1. **Same-level borrowing**: at each grid cell, negative hydrometeors borrow
   from lighter species (rain <- cloud <- vapor).
2. **Vertical borrowing**: remaining negative vapor is redistributed vertically
   within each column (top->bottom sweep, then one bottom->top step).

The correction is mass-conserving at each level (phase 1) and column-integrated
(phase 2). No energy adjustment is needed because Breeze's thermodynamic
prognostics are moist-conserved variables.

The borrowing chain is defined by [`correction_moisture_fields`](@ref), which
microphysics schemes extend to specify their prognostic mass fields.
"""
fix_negative_moisture!(model) = fix_negative_moisture!(model.microphysics, model)

fix_negative_moisture!(::Nothing, model) = nothing

negative_moisture_correction(microphysics) = true

function fix_negative_moisture!(microphysics, model)
    negative_moisture_correction(microphysics) || return nothing
    moisture_fields = correction_moisture_fields(microphysics, model.microphysical_fields)
    isempty(moisture_fields) && return nothing

    grid = model.grid
    arch = grid.architecture
    ρ₀ = dynamics_density(model.dynamics)
    ρqᵛᵉ = model.moisture_density
    Nz = size(grid, 3)
    number_mass_pairs = correction_number_mass_pairs(microphysics, model.microphysical_fields)
    number_fields = correction_number_fields(microphysics, model.microphysical_fields)

    launch!(arch, grid, :xy,
            _fix_negative_moisture_column!,
            moisture_fields, number_mass_pairs, number_fields, ρqᵛᵉ, ρ₀, grid, Nz)

    return nothing
end

#####
##### Column-wise kernel
#####

@kernel function _fix_negative_moisture_column!(moisture_fields, number_mass_pairs, number_fields, ρqᵛᵉ, ρ₀, grid, Nz)
    i, j = @index(Global, NTuple)

    # Phase 1: Same-level borrowing at each level
    for k = 1:Nz
        @inbounds ρ = ρ₀[i, j, k]
        same_level_borrow!(i, j, k, ρ, moisture_fields, ρqᵛᵉ)
    end

    # Zero orphaned number concentrations (mass zeroed but number still positive)
    for k = 1:Nz
        zero_orphaned_numbers!(i, j, k, number_mass_pairs)
    end

    # Clamp negative number concentrations to zero
    for k = 1:Nz
        clamp_negative_numbers!(i, j, k, number_fields)
    end

    # Phase 2: Vertical borrowing for vapor/moisture prognostic
    # Sweep from top to bottom, pushing deficit to level below (more moisture there).
    # Breeze convention: k = 1 is bottom, k = Nz is top.
    for k = Nz:-1:2
        @inbounds ρqᵛ_k = ρqᵛᵉ[i, j, k]
        @inbounds ρ = ρ₀[i, j, k]
        qᵛ = ρqᵛ_k / ρ
        Δz_k = Δzᶜᶜᶜ(i, j, k, grid)
        Δz_below = Δzᶜᶜᶜ(i, j, k - 1, grid)

        # Mass deficit [kg/m²] to push downward (positive when qᵛ < 0)
        deficit = ifelse(qᵛ < 0, -ρqᵛ_k * Δz_k, zero(ρqᵛ_k))
        @inbounds ρqᵛᵉ[i, j, k] += deficit / Δz_k          # -> 0 when deficit > 0
        @inbounds ρqᵛᵉ[i, j, k - 1] -= deficit / Δz_below   # receive deficit
    end

    # Phase 2b: If bottom level still negative, borrow from level above.
    # Use ifelse (not if/else) for GPU kernel compatibility.
    # When Nz < 2, clamp indices to 1 so reads are valid but dq_mass = 0.
    k_bot = 1
    k_top = max(2, Nz)  # safe index: equals 2 when Nz >= 2, equals Nz when Nz < 2

    @inbounds ρqᵛ_bot = ρqᵛᵉ[i, j, k_bot]
    @inbounds ρ_bot = ρ₀[i, j, k_bot]
    qᵛ_bot = ρqᵛ_bot / ρ_bot

    @inbounds ρqᵛ_top = ρqᵛᵉ[i, j, k_top]
    @inbounds ρ_top = ρ₀[i, j, k_top]
    qᵛ_top = ρqᵛ_top / ρ_top

    Δz_bot = Δzᶜᶜᶜ(i, j, k_bot, grid)
    Δz_top = Δzᶜᶜᶜ(i, j, k_top, grid)

    can_borrow = (Nz >= 2) & (qᵛ_bot < 0) & (qᵛ_top > 0)
    needed = -ρqᵛ_bot * Δz_bot       # mass needed at bottom [kg/m²]
    available = ρqᵛ_top * Δz_top      # mass available above [kg/m²]
    dq_mass = ifelse(can_borrow, min(needed, available), zero(ρqᵛ_bot))

    @inbounds ρqᵛᵉ[i, j, k_bot] += dq_mass / Δz_bot
    @inbounds ρqᵛᵉ[i, j, k_top] -= dq_mass / Δz_top
end

#####
##### Recursive same-level borrowing helpers
#####

# Two or more fields: heaviest borrows from next lighter, then recurse
@inline function same_level_borrow!(i, j, k, ρ, fields::Tuple{F1, F2, Vararg}, ρqᵛᵉ) where {F1, F2}
    ρq_heavy = fields[1]
    ρq_light = fields[2]

    @inbounds q_heavy = ρq_heavy[i, j, k] / ρ
    @inbounds q_light = ρq_light[i, j, k] / ρ

    # Borrow from lighter species to fix negative heavier species
    sink = ifelse(q_heavy < 0, min(-q_heavy, max(0, q_light)), zero(q_heavy))
    @inbounds ρq_heavy[i, j, k] += ρ * sink
    @inbounds ρq_light[i, j, k] -= ρ * sink

    # Continue down the chain
    same_level_borrow!(i, j, k, ρ, Base.tail(fields), ρqᵛᵉ)
end

# Last field: borrows from moisture prognostic (vapor / equilibrium moisture)
@inline function same_level_borrow!(i, j, k, ρ, fields::Tuple{F1}, ρqᵛᵉ) where {F1}
    ρq = fields[1]

    @inbounds q = ρq[i, j, k] / ρ
    @inbounds qᵛ = ρqᵛᵉ[i, j, k] / ρ

    # Borrow from vapor to fix negative lightest hydrometeor
    sink = ifelse(q < 0, min(-q, max(0, qᵛ)), zero(q))
    @inbounds ρq[i, j, k] += ρ * sink
    @inbounds ρqᵛᵉ[i, j, k] -= ρ * sink
    return nothing
end

# Empty tuple: nothing to do
@inline same_level_borrow!(i, j, k, ρ, ::Tuple{}, ρqᵛᵉ) = nothing

#####
##### Number concentration consistency helpers
#####

# Zero number concentration when corresponding mass is non-positive
@inline function zero_orphaned_numbers!(i, j, k, pairs::Tuple{P, Vararg}) where {P}
    ρn, ρq = pairs[1]
    @inbounds ρn_val = ρn[i, j, k]
    @inbounds ρn[i, j, k] = ifelse(ρq[i, j, k] <= 0, zero(ρn_val), ρn_val)
    zero_orphaned_numbers!(i, j, k, Base.tail(pairs))\
    return nothing
end

@inline zero_orphaned_numbers!(i, j, k, ::Tuple{}) = nothing

#####
##### Negative number concentration clamping
#####

# Clamp negative number concentrations to zero
@inline function clamp_negative_numbers!(i, j, k, fields)
    for f in fields
        @inbounds f[i, j, k] = max(0, f[i, j, k])
    end
end
