#####
##### Time discretization types for CompressibleDynamics
#####
##### Two top-level choices:
#####   - SplitExplicitTimeDiscretization — Wicker-Skamarock RK3 outer loop
#####     with an inner substep that evolves linearized acoustic perturbations
#####     about the outer-step-start state. The default is classic centered
#####     Crank-Nicolson with no divergence damping; off-centering and Klemp
#####     2018 damping are available for production runs.
#####   - ExplicitTimeStepping — fully explicit time-stepping. Tendencies
#####     (advection + PGF + buoyancy) are computed together; the time-step
#####     is bounded by the acoustic CFL.
#####

#####
##### Outer scheme interface
#####

"""
$(TYPEDEF)

Abstract supertype for the outer Runge–Kutta scheme that drives the acoustic
substep loop. The current implementation supports a single concrete subtype:

  - [`WickerSkamarock3`](@ref) — three-stage Wicker–Skamarock RK3 with stage
    fractions ``β = (1/3, 1/2, 1)``. This is the only outer scheme supported
    today and the default for [`SplitExplicitTimeDiscretization`](@ref).

The interface exists to make the outer-scheme commitment explicit in the
type system and to provide a clean extension point for a future Multirate
Infinitesimal Step (MIS) outer scheme. A concrete subtype is expected to
provide a [`stage_fractions`](@ref) method returning its stage-fraction tuple.
"""
abstract type AcousticOuterScheme end

"""
$(TYPEDEF)

Three-stage Wicker–Skamarock RK3 outer scheme
([Wicker and Skamarock 2002](@cite WickerSkamarock2002)) with canonical stage
fractions ``β = (1/3, 1/2, 1)``. Each stage resets the prognostic state to
``U^n`` and applies a fraction ``β_k Δt`` of the slow tendency evaluated at
the previous-stage state, while the acoustic substep loop advances
linearized perturbations about the outer-step-start state.
"""
struct WickerSkamarock3 <: AcousticOuterScheme end

"""
$(TYPEDSIGNATURES)

Return the stage-fraction tuple ``(β_1, β_2, β_3)`` for the outer Runge–Kutta
scheme. For [`WickerSkamarock3`](@ref) this is the canonical
``(1/3, 1/2, 1)`` of [Wicker and Skamarock (2002)](@cite WickerSkamarock2002).
"""
stage_fractions(::WickerSkamarock3) = (1//3, 1//2, 1//1)

#####
##### Acoustic substep distribution across the WS-RK3 stages
#####

"""
$(TYPEDEF)

Abstract supertype for the choice of how acoustic substeps are distributed
across the three Wicker–Skamarock RK3 stages.

Concrete subtypes:

  - [`ProportionalSubsteps`](@ref) — every stage uses the same substep size
    ``Δτ = Δt/N``, with stage-dependent substep counts ``Nτ = \\max(1, \\mathrm{round}(β N))``
    (so for the canonical β = (1/3, 1/2, 1) this is N/3, N/2, N substeps in
    stages 1, 2, 3). This is the default.

  - [`MonolithicFirstStage`](@ref) — stage 1 collapses to a single substep of
    size ``Δt/3``; stages 2 and 3 are the same as `ProportionalSubsteps`.
"""
abstract type AcousticSubstepDistribution end

"""
$(TYPEDEF)

Acoustic substep distribution where every stage uses the same substep size
``Δτ = Δt/N`` and the substep counts scale with the WS-RK3 stage fraction
``Nτ = \\max(1, \\mathrm{round}(β_\\mathrm{stage} N))``. For the canonical
β = (1/3, 1/2, 1) this gives ``N/3``, ``N/2``, ``N`` substeps in stages 1,
2, 3 respectively.

This is the default.
"""
struct ProportionalSubsteps <: AcousticSubstepDistribution end

"""
$(TYPEDEF)

Acoustic substep distribution where stage 1 collapses to a single substep
of size ``Δt/3``; stages 2 and 3 are the same as
[`ProportionalSubsteps`](@ref) (``N/2`` and ``N`` substeps of size
``Δτ = Δt/N``).
"""
struct MonolithicFirstStage <: AcousticSubstepDistribution end

#####
##### Acoustic divergence damping strategies
#####

"""
$(TYPEDEF)

Abstract supertype for divergence damping applied inside the substep loop.

Concrete subtypes:

  - [`NoDivergenceDamping`](@ref) — no damping (the default).
  - [`ThermalDivergenceDamping`](@ref) — [Klemp, Skamarock & Ha (2018)](@cite KlempSkamarockHa2018)
    momentum correction using the discrete ``δ_τ(ρθ)`` tendency as the
    divergence proxy. Used for production runs to filter grid-scale
    acoustic divergence over long integrations; not intended as a
    stabilizer for canonical cases.
"""
abstract type AcousticDampingStrategy end

"""
$(TYPEDEF)

No acoustic divergence damping. The substep loop advances perturbation
fields without applying any post-substep momentum correction. **Default.**
"""
struct NoDivergenceDamping <: AcousticDampingStrategy end

"""
$(TYPEDEF)

3-D acoustic divergence damping that uses the **(ρθ)′ tendency as a
discrete proxy for the momentum divergence**. From the linearized
ρθ-continuity equation
``\\partial_t (ρθ)' + \\nabla\\cdot(ρθ^0 u') = 0``, the per-substep
quantity

```math
D \\equiv \\frac{(ρθ)' - (ρθ)'_\\mathrm{old}}{θ^0}
       \\approx -Δτ \\, \\nabla\\cdot(ρu)'
```

is what would otherwise require an extra divergence operator and an
extra kernel pass. Building the correction from `D` reuses the
substep's already-resident `(ρθ)′` snapshots — that's the algorithmic
choice this damping is named for.

Used by
[Klemp, Skamarock & Ha (2018)](@cite KlempSkamarockHa2018) /
[Skamarock & Klemp (1992)](@cite SkamarockKlemp1992) /
[Baldauf (2010)](@cite Baldauf2010). After each acoustic substep, all
three momentum perturbation components ``(ρu)′, (ρv)′, (ρw)′`` pick up
a correction proportional to ``\\nabla D``. The vertical component is the
piece that damps vertical acoustic modes and is REQUIRED for
stability at production ``\\Delta t``: without it, the column tridiag's
anti-symmetric buoyancy off-diagonals and asymmetric (under
stratified ``\\bar\\theta(z)``) PGF off-diagonals make the substep
operator non-normal with ``\\|U\\|_2 \\gg 1``, and a rest atmosphere
amplifies ~1.8× per outer step.

Per-substep momentum correction (Baldauf 2010 §2.d, anisotropic):

```math
Δ(ρu)′ = -α_x · ∂_x D , \\quad
Δ(ρv)′ = -α_y · ∂_y D , \\quad
Δ(ρw)′ = -α_z · ∂_z D
```

with per-direction damping diffusivities
``α_x = β_d Δx²/Δτ``, ``α_y = β_d Δy²/Δτ``, ``α_z = β_d Δz²/Δτ`` —
giving a constant explicit-time Courant number ``β_d`` per direction
that's invariant under Δτ and grid spacing. The combined 3-D
stability bound is ``2 β_d ≤ 1/2 → β_d ≤ 0.25``; the default
``β_d = 0.1`` sits well below the bound. Combined with the
`SplitExplicitTimeDiscretization` default ``\\omega = 0.65``, this
keeps both the rest atmosphere at machine ε at Δt = 20 s and the
DCMIP-2016 dry / moist baroclinic waves stable at their production
``\\Delta t``.

Fields
======

- `coefficient`: Dimensionless damping coefficient ``β_d``. Default `0.1`.
  The vertical part is folded into the column tridiag and is
  unconditionally stable, so this coefficient may be raised above the
  explicit-time bound `β_d ≤ 0.25` to suppress non-normal transient
  amplification of the vertical acoustic mode. The horizontal part
  remains explicit and obeys the usual `2β_d ≤ 1/2` per-direction CFL.
- `length_scale`: Optional override for the dispersion length
  ``ℓ_\\mathrm{disp}``. Default `nothing` (auto = anisotropic
  per-direction grid spacing). Setting `length_scale = ℓ` forces an
  isotropic ``ν = β_d ℓ² / Δτ``.
- `reference_temperature`: Reference temperature for
  ``c_s = \\sqrt{γ^d R^d T_\\mathrm{ref}}``. Default `300` K. The
  damping is a wavenumber-controlled filter; only the dimensional
  scale of ``ν`` depends on this — small variations don't matter.
"""
struct ThermalDivergenceDamping{FT, LS} <: AcousticDampingStrategy
    coefficient :: FT
    length_scale :: LS
    reference_temperature :: FT
end

function ThermalDivergenceDamping(; coefficient = 0.1,
                                    length_scale = nothing,
                                    reference_temperature = 300.0)
    if length_scale === nothing
        FT = promote_type(typeof(coefficient), typeof(reference_temperature))
        return ThermalDivergenceDamping{FT, Nothing}(convert(FT, coefficient), nothing,
                                                     convert(FT, reference_temperature))
    else
        FT = promote_type(typeof(coefficient), typeof(length_scale), typeof(reference_temperature))
        return ThermalDivergenceDamping{FT, FT}(convert(FT, coefficient),
                                                convert(FT, length_scale),
                                                convert(FT, reference_temperature))
    end
end

#####
##### Split-explicit time discretization
#####

"""
$(TYPEDEF)

Split-explicit time discretization for compressible dynamics.

Outer integration is the Wicker–Skamarock RK3 scheme
([Wicker and Skamarock 2002](@cite WickerSkamarock2002)) with stage
fractions ``β = (1/3, 1/2, 1)``. Within each stage, an inner substep loop
evolves linearized acoustic perturbations about the outer-step-start
state. The vertically implicit solve uses an off-centered Crank-Nicolson
scheme with off-centering parameter ``\\omega`` (default 0.5 = classic
centered CN).

The substep distribution across stages is selectable via the
[`AcousticSubstepDistribution`](@ref) interface.

Fields
======

- `substeps`: Number of acoustic substeps ``N`` per outer ``Δt``. Default
  `nothing` adaptively chooses ``N`` from the horizontal acoustic CFL each
  step.
- `forward_weight`: Off-centering parameter ``\\omega`` for the vertically
  implicit solve. ``\\omega = 0.5`` is classic centered Crank-Nicolson;
  the default ``\\omega = 0.65`` adds modest off-centering
  (``\\varepsilon = 2\\omega - 1 = 0.3``). Combined with the default
  divergence damping, it keeps a rest atmosphere at machine ε at
  production ``\\Delta t = 20`` s and survives the DCMIP-2016 dry/moist
  baroclinic-wave smoke tests at production grid.

  **Note on residual non-normality** (Phase 4 of
  `validation/substepping/PRISTINE_SUBSTEPPER_PLAN.md`): the column
  tridiag has anti-symmetric buoyancy off-diagonals (gravity-wave
  physics) and asymmetric PGF off-diagonals on a stratified ``\\bar\\theta(z)``,
  so the substep operator ``U`` has spectral radius ``\\rho(U) = 1``
  but operator norm ``\\|U\\|_2 \\gg 1`` (≈ 44 at ``\\Delta t = 20`` s,
  ``\\omega = 0.55``, no damping). Distributed FP-noise excites the
  non-normal transient-amplification subspace, leading to a
  rest-atmosphere blow-up at default ``\\omega = 0.55``. Klemp 3-D
  divergence damping shrinks ``\\|U^k\\|`` over enough ``k`` and breaks
  the soft outer-step CFL — it is REQUIRED for stability at production
  ``\\Delta t``.
- `damping`: Acoustic divergence damping strategy. Default:
  [`ThermalDivergenceDamping`](@ref) with coefficient 0.1. Required for
  stability at production ``\\Delta t``; passing
  [`NoDivergenceDamping`](@ref) will cause the rest atmosphere to
  amplify ~1.8× per outer step at ``\\Delta t = 20`` s.
- `substep_distribution`: How acoustic substeps are distributed across the
  three WS-RK3 stages.

See also [`ExplicitTimeStepping`](@ref).
"""
struct SplitExplicitTimeDiscretization{N, FT, D <: AcousticDampingStrategy, AD <: AcousticSubstepDistribution}
    substeps :: N
    forward_weight :: FT
    damping :: D
    substep_distribution :: AD
end

function SplitExplicitTimeDiscretization(; substeps = nothing,
                                           forward_weight = 0.65,
                                           damping = ThermalDivergenceDamping(coefficient = 0.1),
                                           substep_distribution = ProportionalSubsteps())

    return SplitExplicitTimeDiscretization(substeps,
                                           forward_weight,
                                           damping,
                                           substep_distribution)
end

"""
$(TYPEDEF)

Standard explicit time discretization for compressible dynamics.

All tendencies (including pressure gradient and acoustic modes) are computed
together and time-stepped explicitly. This requires small time steps limited
by the acoustic CFL condition (sound speed ~340 m/s).

Use [`SplitExplicitTimeDiscretization`](@ref) for more efficient time-stepping
with larger Δt.
"""
struct ExplicitTimeStepping end
