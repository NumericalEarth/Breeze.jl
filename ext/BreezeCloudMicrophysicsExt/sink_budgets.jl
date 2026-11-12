#####
##### Sink budgets for the bulk one- and two-moment schemes
#####
#
# Every microphysical process is a transfer out of one reservoir. A bulk scheme keeps its
# reservoirs non-negative by scaling the transfers out of each one to what that reservoir can
# give. What it can give depends on who applies the tendency:
#
#   * `TimescaleSinkBudget`: the gridless budget of `microphysical_tendency`, used by parcel
#     models. Nothing is known about the step that will apply the tendency, so a reservoir
#     loses at most its content, at the state the rates are evaluated at, over the fixed
#     timescale `τⁿᵘᵐ`.
#
#   * `StageSinkBudget`: the budget of a grid cell. The time stepper applies the tendency to
#     the state the coming stage starts from, over the stage interval `Δt`, alongside
#     transport, so each transfer is scaled by the Patankar fraction of the reservoir it draws
#     on (see `AtmosphereModels.patankar_sink_fraction`). That keeps every reservoir
#     non-negative for any `Δt`, and `τⁿᵘᵐ` is left only as the timescale over which a value
#     that transport left negative relaxes back to zero. Before the first time step, when the
#     stage interval is not known, it falls back to the fixed-timescale budget.
#
# Both budgets scale a transfer and its counterpart alike, so they conserve every exchange.

struct TimescaleSinkBudget{FT}
    τ :: FT  # timescale over which a reservoir may be drained
end

struct StageSinkBudget{I, J, K, G, S, FT}
    i :: I
    j :: J
    k :: K
    G :: G      # tendencies accumulated so far, keyed by prognostic name
    stage :: S  # `AtmosphereModels.StageStart` of the same prognostics
    τ :: FT     # relaxation timescale of negative values; drain timescale while `Δt` is unknown
end

# Fraction of a donor's process rates that can be honoured without draining more than
# its content `q` over τⁿᵘᵐ: 1 when the summed sink rate `ΣS` fits, `q⁺ / (τⁿᵘᵐ ΣS)` otherwise.
@inline function donor_limiter(q, ΣS, τⁿᵘᵐ)
    q⁺ = max(0, q)
    # Disabled processes return integer zeros; retain the moisture precision for zero demand.
    demand = oftype(q, τⁿᵘᵐ) * ΣS
    # Protect the eagerly evaluated quotient at zero demand without flooring positive demand.
    denominator = ifelse(iszero(demand), one(demand), demand)
    return ifelse(demand > q⁺, q⁺ / denominator, one(demand))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The fraction of the summed specific sink rate `ΣS` that the prognostic reservoir `name`
(a `Val`), with specific content `q` at the state the rates are evaluated at, can give.
An empty or negative reservoir gives nothing.
"""
@inline sink_fraction(budget::TimescaleSinkBudget, name, ρ, q, ΣS) = donor_limiter(q, ΣS, budget.τ)

@inline function sink_fraction(budget::StageSinkBudget, name::Val{N}, ρ, q, ΣS) where N
    (; i, j, k, stage) = budget
    G = getproperty(budget.G, N)
    φ = AM.patankar_sink_fraction(i, j, k, G, stage, name, ρ * ΣS)
    # A rate evaluated at an empty reservoir carries no information about what it can give.
    φ = ifelse(q > 0, φ, ifelse(ΣS > 0, zero(φ), one(φ)))
    return ifelse(AM.stage_interval_is_known(stage), φ, donor_limiter(q, ΣS, budget.τ))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The fraction of the summed specific sink rate `ΣS` that a *diagnostic* reservoir can give:
one that saturation adjustment partitions out of the prognostic moisture (vapor, cloud
liquid, cloud ice), so that it has no value at the start of the stage and no tendency of its
own. On the grid it is budgeted in Patankar form against its content `q` at the state the
rates are evaluated at, over the stage interval.
"""
@inline diagnostic_sink_fraction(budget::TimescaleSinkBudget, ρ, q, ΣS) = donor_limiter(q, ΣS, budget.τ)

@inline function diagnostic_sink_fraction(budget::StageSinkBudget, ρ, q, ΣS)
    stage = budget.stage
    φ = AM.patankar_fraction(ρ * q, stage.Δt, ρ * ΣS)
    return ifelse(AM.stage_interval_is_known(stage), φ, donor_limiter(q, ΣS, budget.τ))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The fraction of the summed specific rate `ΣS` drawn from the diagnostic reservoirs that the
equilibrium moisture `ρqᵉ` holding them can give. One for the gridless budget, where the
reservoirs it is partitioned into are already budgeted and add up to it.
"""
@inline equilibrium_moisture_sink_fraction(budget::TimescaleSinkBudget, ρ, ΣS) = one(ρ)

@inline function equilibrium_moisture_sink_fraction(budget::StageSinkBudget, ρ, ΣS)
    (; i, j, k, G, stage) = budget
    return AM.patankar_sink_fraction(i, j, k, G.ρqᵉ, stage, Val(:ρqᵉ), ρ * ΣS)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The largest specific rate at which a single process may draw on a reservoir with content
`q`. On the grid the Patankar fraction bounds the drain, so the cap only shuts off a
process whose reservoir is empty at the state its rate is evaluated at.
"""
@inline sink_rate_cap(budget::TimescaleSinkBudget, q) = max(0, q) / budget.τ

@inline function sink_rate_cap(budget::StageSinkBudget, q)
    uncapped = ifelse(q > 0, oftype(q, Inf), zero(q))
    return ifelse(AM.stage_interval_is_known(budget.stage), uncapped, max(0, q) / budget.τ)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The density tendency to add to the process tendency `Gᵖ` of the reservoir `name` so that a
negative value relaxes back to zero over `τⁿᵘᵐ`. The caller takes it from another
reservoir (see [`relaxation_donor_fraction`](@ref)), which conserves the exchange.

The gridless budget replaces the tendency of a negative reservoir with `-ρ q / τⁿᵘᵐ`. On
the grid the relaxation is advanced by backward Euler over the coming stage: with
`c̃ = c⁰ + Δt (Gᵒ + Gᵖ)` what the stage start `c⁰`, every other tendency `Gᵒ` and the
processes leave, the relaxed value is `c̃ / (1 + Δt / τⁿᵘᵐ)`, which lies between `c̃` and
zero for any `Δt`. That is the tendency `-c̃ / (τⁿᵘᵐ + Δt)`, applied where `c̃ < 0`.
"""
@inline negative_value_relaxation(budget::TimescaleSinkBudget, name, ρ, q, Gᵖ) =
    ifelse(q ≥ 0, zero(Gᵖ), -ρ * q / budget.τ - Gᵖ)

@inline function negative_value_relaxation(budget::StageSinkBudget, name::Val{N}, ρ, q, Gᵖ) where N
    (; i, j, k, stage, τ) = budget
    Δt = stage.Δt
    c̃ = AM.stage_remainder(i, j, k, getproperty(budget.G, N), stage, name) + Δt * Gᵖ
    implicit = max(0, -c̃) / (τ + Δt)
    explicit = ifelse(q ≥ 0, zero(Gᵖ), -ρ * q / τ - Gᵖ)
    return ifelse(AM.stage_interval_is_known(stage), implicit, explicit)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The fraction of the density tendency `δ`, relaxing negative values elsewhere, that the
reservoir `name` with process tendency `Gᵖ` can give. One for the gridless budget. On the
grid it is the Patankar fraction against what the stage start, every other tendency and the
processes leave of the reservoir, so a relaxation never takes its donor below zero.
"""
@inline relaxation_donor_fraction(budget::TimescaleSinkBudget, name, Gᵖ, δ) = one(Gᵖ)

@inline function relaxation_donor_fraction(budget::StageSinkBudget, name::Val{N}, Gᵖ, δ) where N
    (; i, j, k, stage) = budget
    Δt = stage.Δt
    c̃ = AM.stage_remainder(i, j, k, getproperty(budget.G, N), stage, name) + Δt * Gᵖ
    φ = AM.patankar_fraction(c̃, Δt, δ)
    return ifelse(AM.stage_interval_is_known(stage), φ, one(φ))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The fractions `(mass, both, number)` applied to the sinks of a reservoir that carries both
a mass and a number, given the fraction `φᵐ` of its mass sinks and `φⁿ` of its number sinks
that can be honoured: `mass` scales the mass transfers, `both` the number rates that
accompany a mass transfer, and `number` the processes that change number alone.

On the grid a transfer is scaled by the fraction of the reservoir it draws on, and a number
rate that accompanies a mass transfer by the smaller of the two, so it never removes more
particles than its mass does mass. The gridless budget scales every sink of the reservoir by
the smaller fraction.
"""
@inline function coupled_sink_fractions(budget::TimescaleSinkBudget, φᵐ, φⁿ)
    φ = min(φᵐ, φⁿ)
    return (φ, φ, φ)
end

@inline function coupled_sink_fractions(budget::StageSinkBudget, φᵐ, φⁿ)
    φᵐ, φⁿ = promote(φᵐ, φⁿ)
    φ = min(φᵐ, φⁿ)
    known = AM.stage_interval_is_known(budget.stage)
    return (ifelse(known, φᵐ, φ), φ, ifelse(known, φⁿ, φ))
end

#####
##### Fused tendencies on the grid
#####
#
# The bulk schemes compute all of their tendencies together (`fused_tendencies`), because
# the sink budgets couple every reservoir. One kernel evaluates the bundle once per cell,
# budgeted against the coming stage, and adds each tendency to its `Gⁿ` field. It runs after
# every other tendency has been accumulated, so `Gⁿ` holds what the budget calls `Gᵒ`.

# The prognostics a bulk scheme's processes act on, the moisture first.
@inline fused_tendency_names(bμp) = (AM.moisture_prognostic_name(bμp), AM.prognostic_field_names(bμp)...)

@inline function fused_tendency_fields(fields, bμp)
    names = fused_tendency_names(bμp)
    return NamedTuple{names}(map(name -> getproperty(fields, name), names))
end

@kernel function _compute_fused_bulk_tendencies!(G, stage, grid, microphysics, dynamics, formulation,
                                                 constants, specific_prognostic_moisture,
                                                 microphysical_fields, velocities)
    i, j, k = @index(Global, NTuple)

    ρ_field = AM.total_density(dynamics)
    @inbounds ρ = ρ_field[i, j, k]
    @inbounds qᵛᵉ = specific_prognostic_moisture[i, j, k]

    # Reconstruct moisture fractions and thermodynamic state at this cell.
    q = AM.grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    𝒰 = AM.diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)
    ℳ = fused_microphysical_state(i, j, k, grid, microphysics, microphysical_fields, ρ, 𝒰, velocities)

    budget = StageSinkBudget(i, j, k, G, stage, microphysics.categories.τⁿᵘᵐ)
    Gᵖ = fused_tendencies(microphysics, ρ, ℳ, 𝒰, constants, budget)

    add_fused_tendencies!(G, i, j, k, Gᵖ)
end

# Recursive iteration over the names: statically unrolled, as the names are in the type.
@inline add_fused_tendencies!(G::NamedTuple{names}, i, j, k, Gᵖ) where names =
    add_fused_tendencies!(G, i, j, k, Gᵖ, map(Val, names))

@inline add_fused_tendencies!(G, i, j, k, Gᵖ, ::Tuple{}) = nothing

@inline function add_fused_tendencies!(G, i, j, k, Gᵖ, names::Tuple)
    add_fused_tendency!(G, i, j, k, Gᵖ, first(names))
    return add_fused_tendencies!(G, i, j, k, Gᵖ, Base.tail(names))
end

@inline function add_fused_tendency!(G, i, j, k, Gᵖ, ::Val{name}) where name
    @inbounds getproperty(G, name)[i, j, k] += getproperty(Gᵖ, name)
    return nothing
end

function compute_fused_bulk_tendencies!(microphysics, model)
    grid = model.grid
    G = fused_tendency_fields(model.timestepper.Gⁿ, microphysics)
    stage = AM.StageStart(model, fields -> fused_tendency_fields(fields, microphysics))

    # Microphysics responds to the physical velocity of the current state, so we pass
    # `model.velocities` rather than `transport_velocities(model)`.
    launch!(grid.architecture, grid, :xyz, _compute_fused_bulk_tendencies!,
            G, stage, grid, microphysics, model.dynamics, model.formulation,
            model.thermodynamic_constants, AM.specific_prognostic_moisture(model),
            model.microphysical_fields, model.velocities)

    return nothing
end
