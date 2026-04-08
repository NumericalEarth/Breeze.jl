#####
##### Time discretization types for CompressibleDynamics
#####
##### These types determine how the compressible equations are time-stepped:
##### - SplitExplicitTimeDiscretization: Acoustic substepping (Wicker-Skamarock scheme)
##### - ExplicitTimeStepping: Standard explicit time-stepping (small Δt required)
#####

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
    stages 1, 2, 3). This is the default and matches CM1.

  - [`MonolithicFirstStage`](@ref) — stage 1 collapses to a single substep of
    size ``Δt/3``; stages 2 and 3 are the same as `ProportionalSubsteps`. This
    matches MPAS-A with `config_time_integration_order = 3`.
"""
abstract type AcousticSubstepDistribution end

"""
$(TYPEDEF)

Acoustic substep distribution where every stage uses the same substep size
``Δτ = Δt/N`` and the substep counts scale with the WS-RK3 stage fraction
``Nτ = \\max(1, \\mathrm{round}(β_\\mathrm{stage} N))``. For the canonical
β = (1/3, 1/2, 1) this gives ``N/3``, ``N/2``, ``N`` substeps in stages 1,
2, 3 respectively.

The horizontal acoustic CFL constraint is set by ``Δτ = Δt/N`` — the same in
every stage — so no individual stage imposes a tighter Δt ceiling than the
others.

This is Breeze's default and matches CM1 ([Bryan and Fritsch (2002)](@cite Bryan2002)).
"""
struct ProportionalSubsteps <: AcousticSubstepDistribution end

"""
$(TYPEDEF)

Acoustic substep distribution where stage 1 of WS-RK3 collapses to a single
substep of size ``Δt/3``, while stages 2 and 3 use the same proportional
counts as [`ProportionalSubsteps`](@ref) (``N/2`` and ``N`` substeps of size
``Δt/N``).

Because stage 1 uses a substep of size ``Δt/3`` (rather than the per-substep
``Δt/N`` of stages 2 and 3), the stage-1 horizontal acoustic CFL becomes
``Δt/3 < Δx_\\mathrm{min}/c_s``, which is ``N/3`` times tighter than the
[`ProportionalSubsteps`](@ref) form. This is the dispatch used by MPAS-A
when `config_time_integration_order = 3` (see `mpas_atm_time_integration.F`),
and is provided here for bit-compatible comparisons against MPAS reference
output.
"""
struct MonolithicFirstStage <: AcousticSubstepDistribution end

#####
##### Acoustic divergence damping strategies
#####

"""
$(TYPEDEF)

Abstract supertype for the choice of acoustic divergence damping applied
inside the substep loop.

Concrete subtypes:

  - [`NoDivergenceDamping`](@ref) — no damping. Useful as a baseline for
    "is divergence damping the bottleneck?" experiments.
  - [`ThermodynamicDivergenceDamping`](@ref) — MPAS Klemp–Skamarock–Ha 2018
    momentum correction using the discrete ``(\\rho\\theta)''`` tendency as
    the divergence proxy. This is Breeze's default and matches MPAS-A's
    `atm_divergence_damping_3d`.
  - [`PressureProjectionDamping`](@ref) — literal ERF/CM1/WRF form (Klemp
    2007): forward-extrapolate the diagnosed Exner perturbation, convert
    back to ``(\\rho\\theta)''`` via the linearized EOS.
  - [`ConservativeProjectionDamping`](@ref) — algebraic conservative-variable
    variant of the above; equivalent at the linearized level but skips the
    EOS evaluation.

See `docs/src/appendix/substepping_cleanup_and_damping_plan.md` for the full
strategy design and the empirical comparison plan.
"""
abstract type AcousticDampingStrategy end

"""
$(TYPEDEF)

No acoustic divergence damping. The substep loop advances the perturbation
fields without applying any post-substep momentum correction.
"""
struct NoDivergenceDamping <: AcousticDampingStrategy end

"""
$(TYPEDEF)

MPAS-A Klemp–Skamarock–Ha 2018 acoustic divergence damping
([Klemp et al. 2018](@cite KlempSkamarockHa2018)).

After each acoustic substep, the horizontal momentum perturbations are
corrected by

```math
Δ(\\rho u)'' = \\mathrm{coef}\\,\\partial_x(δ_τ(\\rho\\theta)'') / (2 θ_{m,\\mathrm{edge}})
```

(and similarly in ``y``), where ``δ_τ(\\rho\\theta)'' = (\\rho\\theta)''_\\mathrm{new} - (\\rho\\theta)''_\\mathrm{old}``
is the discrete acoustic ``(\\rho\\theta)''`` tendency, ``\\mathrm{coef} = 2\\,
\\mathrm{smdiv}\\,\\ell_\\mathrm{disp}/Δτ``, and ``\\mathrm{smdiv}`` is the
strategy's `coefficient` field. Using the discrete pressure-tendency proxy
ensures the damping preserves gravity-wave frequencies while filtering
grid-scale acoustic divergence.

Fields
======

- `coefficient`: MPAS `config_smdiv`. Default `0.1`.
- `length_scale`: Optional override for the dispersion length ``\\ell_\\mathrm{disp}`` (MPAS `config_len_disp`). Default `nothing` (Breeze auto-derives ``\\min(Δx, Δy)`` over non-Flat horizontal axes).
"""
struct ThermodynamicDivergenceDamping{FT} <: AcousticDampingStrategy
    coefficient :: FT
    length_scale :: Union{FT, Nothing}
end

function ThermodynamicDivergenceDamping(; coefficient = 0.1, length_scale = nothing)
    FT = length_scale === nothing ? typeof(coefficient) : promote_type(typeof(coefficient), typeof(length_scale))
    coef_FT = convert(FT, coefficient)
    len_FT  = length_scale === nothing ? nothing : convert(FT, length_scale)
    return ThermodynamicDivergenceDamping{FT}(coef_FT, len_FT)
end

"""
$(TYPEDEF)

Conservative pressure-projection damping. Forward-extrapolates the
prognostic ``(\\rho\\theta)''`` perturbation by one substep's worth of
change before it is read by the next substep's horizontal pressure
gradient kernel:

```math
(\\rho\\widetilde{\\theta})''_\\mathrm{for\\ pgf}
    = (\\rho\\theta)'' + \\beta_d \\bigl((\\rho\\theta)'' - (\\rho\\theta)''_\\mathrm{prev}\\bigr)
```

where ``(\\rho\\theta)''_\\mathrm{prev}`` is the value at the start of the
previous substep (already maintained by the substepper as
`previous_rtheta_pp` for the MPAS damping path) and ``\\beta_d`` is the
strategy's `coefficient`.

This is a conservative-variable (algebraic) approximation to the literal
ERF/CM1/WRF [`PressureProjectionDamping`](@ref) — at the strict linearized
level (perturbations small relative to the reference state, EOS map
approximated by its tangent at the reference) the two are equivalent. They
diverge at second order in the perturbations and at the discretization
level.

Fields
======

- `coefficient`: forward-projection weight ``\\beta_d``. Default `0.1`.
- `ρθ″_for_pgf`: scratch `CenterField` written by `apply_pgf_filter!` and read by the next substep's `_mpas_horizontal_forward!`. `nothing` in the user-facing skeleton; allocated by the substepper constructor.
"""
struct ConservativeProjectionDamping{FT, F} <: AcousticDampingStrategy
    coefficient :: FT
    ρθ″_for_pgf :: F
end

function ConservativeProjectionDamping(; coefficient = 0.1)
    FT = typeof(coefficient)
    return ConservativeProjectionDamping{FT, Nothing}(convert(FT, coefficient), nothing)
end

"""
$(TYPEDEF)

Pressure-projection damping in the literal ERF/CM1/WRF form (Klemp et
al. 2007). Each substep diagnoses the Exner perturbation from the
prognostic ``(\\rho\\theta)''``, forward-extrapolates it as
``\\tilde{\\pi}'' = \\pi'' + \\beta_d (\\pi'' - \\pi''_\\mathrm{old})``, and
converts the projected ``\\tilde{\\pi}''`` back into a projected
``(\\rho\\widetilde{\\theta})''_\\mathrm{for\\ pgf}`` via the linearized EOS
so that the existing horizontal pressure gradient kernel
(``-c^2\\,\\Pi_\\mathrm{face}\\,\\partial_x(\\rho\\theta)''``) reads the
filtered field without any kernel-signature change.

The linearized EOS conversion at a cell center is

```math
(\\rho\\widetilde{\\theta})''_\\mathrm{for\\ pgf}
    = (\\rho\\theta)'' + \\frac{c_v}{R}\\,\\frac{(\\rho\\theta)_\\mathrm{stage}}{\\Pi_\\mathrm{stage}}\\,\\beta_d\\,(\\pi'' - \\pi''_\\mathrm{old}),
```

with both the diagnosed ``\\pi''`` (from the current ``(\\rho\\theta)''``)
and the previous-substep ``\\pi''_\\mathrm{old}`` (from the substepper's
`previous_rtheta_pp` field, which is already maintained by the MPAS
divergence-damping path) computed via
``\\pi(\\rho\\theta) = (R\\,\\rho\\theta/p^{st})^{R/c_v}``.

This is the closer-to-ERF / CM1 form of the projection. The cheaper
[`ConservativeProjectionDamping`](@ref) variant is mathematically
equivalent at the linearized level but skips the per-cell EOS
evaluation.

Fields
======

- `coefficient`: forward-projection weight ``\\beta_d``. Default `0.1`.
- `ρθ″_for_pgf`: scratch `CenterField` written by `apply_pgf_filter!` and read by the next substep's `_mpas_horizontal_forward!`. `nothing` in the user-facing skeleton; allocated by the substepper constructor.
"""
struct PressureProjectionDamping{FT, F} <: AcousticDampingStrategy
    coefficient :: FT
    ρθ″_for_pgf :: F
end

function PressureProjectionDamping(; coefficient = 0.1)
    FT = typeof(coefficient)
    return PressureProjectionDamping{FT, Nothing}(convert(FT, coefficient), nothing)
end

"""
$(TYPEDEF)

Split-explicit acoustic substepping for compressible dynamics using the
MPAS-A conservative-perturbation formulation.

The fast prognostic variables — advanced inside the substep loop — are the
horizontal and vertical momentum perturbations ``(\\rho u)''``, ``(\\rho v)''``,
``(\\rho w)''``, the density perturbation ``\\rho''``, and the
``(\\rho\\theta)''`` perturbation. The same family is used by MPAS-A
([Skamarock et al. 2012](@cite Skamarock2012)) and by ERF.

Outer integration is the Wicker–Skamarock RK3 scheme
([Wicker and Skamarock 2002](@cite WickerSkamarock2002)) with stage fractions
``β = (1/3, 1/2, 1)``. The substep distribution across stages is selectable
via the [`AcousticSubstepDistribution`](@ref) interface
([`ProportionalSubsteps`](@ref) or [`MonolithicFirstStage`](@ref)).

The vertically implicit ``(\\rho w)''``–``(\\rho\\theta)''`` coupling is solved
by a Schur-complement tridiagonal sweep at each substep, eliminating the
vertical acoustic CFL constraint. Divergence damping is applied each substep
following the MPAS Klemp–Skamarock–Ha 2018 momentum correction
(see [`acoustic_substepping.jl`](https://github.com/CliMA/Breeze.jl/blob/main/src/CompressibleEquations/acoustic_substepping.jl)).

This allows the outer time step to be set by the advective CFL rather than
the acoustic CFL, typically enabling ~6× larger ``Δt`` than fully explicit
compressible time-stepping.

Fields
======

- `substeps`: Number of acoustic substeps ``N`` per outer ``Δt``. Default `nothing` adaptively chooses ``N`` from the horizontal acoustic CFL each step. With [`ProportionalSubsteps`](@ref) the substep size is ``Δτ = Δt/N`` in every stage; with [`MonolithicFirstStage`](@ref) stage 1 instead uses one substep of size ``Δt/3``.
- `forward_weight`: Off-centering parameter ``ω`` for the vertically implicit ``(\\rho w)''``–``(\\rho\\theta)''`` solve. ``ω > 0.5`` damps vertical acoustic modes; the MPAS off-centering is ``ε = 2ω - 1``. Default: 0.6.
- `damping`: Acoustic divergence damping strategy ([`AcousticDampingStrategy`](@ref)). Default: [`ThermodynamicDivergenceDamping`](@ref) with `coefficient = 0.1` (matches MPAS-A `config_smdiv`). Use [`NoDivergenceDamping`](@ref) to disable damping entirely.
- `acoustic_damping_coefficient`: Optional Klemp 2018 acoustic damping coefficient ``ϰ^{ac}``, applied as a post-implicit-solve velocity correction: ``u -= ϰ^{ac} c_p θ_v ∂Δπ'/∂x``. Default: 0.0.
- `substep_distribution`: How acoustic substeps are distributed across the three WS-RK3 stages. One of [`ProportionalSubsteps`](@ref) (default; constant ``Δτ = Δt/N`` with stage counts ``N/3``, ``N/2``, ``N``) or [`MonolithicFirstStage`](@ref) (single substep of size ``Δt/3`` in stage 1, MPAS-A `config_time_integration_order = 3` form).

See also [`ExplicitTimeStepping`](@ref) and [`VerticallyImplicitTimeStepping`](@ref).
"""
struct SplitExplicitTimeDiscretization{N, FT, D <: AcousticDampingStrategy, AD <: AcousticSubstepDistribution}
    substeps :: N
    forward_weight :: FT
    damping :: D
    acoustic_damping_coefficient :: FT
    substep_distribution :: AD
end

function SplitExplicitTimeDiscretization(; substeps = nothing,
                                           forward_weight = 0.6,
                                           damping = ThermodynamicDivergenceDamping(),
                                           acoustic_damping_coefficient = 0.0,
                                           substep_distribution = ProportionalSubsteps(),
                                           divergence_damping_coefficient = nothing)

    # Backwards-compat: the old `divergence_damping_coefficient` kwarg was
    # silently dropped at runtime (Phase 2 bug fix in
    # docs/src/appendix/substepping_cleanup_and_damping_plan.md). Map it to a
    # ThermodynamicDivergenceDamping when no explicit `damping` was passed and
    # warn loudly so users know to migrate to the new API.
    if divergence_damping_coefficient !== nothing
        Base.depwarn("`divergence_damping_coefficient` is deprecated. " *
                     "Pass `damping = ThermodynamicDivergenceDamping(coefficient = $(divergence_damping_coefficient))` " *
                     "instead. (Note: in prior releases this kwarg was silently ignored at runtime; " *
                     "the substepper used a hardcoded `smdiv = 0.1`.)",
                     :SplitExplicitTimeDiscretization)
        damping = ThermodynamicDivergenceDamping(coefficient = divergence_damping_coefficient)
    end

    return SplitExplicitTimeDiscretization(substeps,
                                           forward_weight,
                                           damping,
                                           acoustic_damping_coefficient,
                                           substep_distribution)
end

"""
$(TYPEDEF)

Standard explicit time discretization for compressible dynamics.

All tendencies (including pressure gradient and acoustic modes) are computed
together and time-stepped explicitly. This requires small time steps limited
by the acoustic CFL condition (sound speed ~340 m/s).

Use [`SplitExplicitTimeDiscretization`](@ref) or
[`VerticallyImplicitTimeStepping`](@ref) for more efficient time-stepping with larger Δt.
"""
struct ExplicitTimeStepping end

"""
$(TYPEDEF)

Vertically implicit time discretization for compressible dynamics.

Treats vertical acoustic propagation implicitly by decomposing the vertical
pressure gradient and vertical ρθ advective flux into linear and nonlinear
parts. The linear vertical acoustic coupling between ρw and ρθ is solved
via a tridiagonal system (backward Euler) after each explicit SSP-RK3 stage,
while all other terms remain explicit.

The tridiagonal equation for the implicit correction is:

```math
\\left[I - (α Δt)^2 \\partial_z (\\mathbb{C}^{ac2} \\partial_z)\\right] (ρθ)^+ = (ρθ)^*
```

followed by a back-solve for ``(ρw)^+``. The linearization state (θ and ℂᵃᶜ²)
comes from the most recent `update_state!` call.

This eliminates the vertical acoustic CFL constraint, allowing time steps limited
only by the horizontal acoustic CFL and advective CFL — typically ~30x larger
than [`ExplicitTimeStepping`](@ref) for kilometer-scale vertical grids.

The parameter `β` controls the implicitness of the acoustic coupling:
  - `β = 0.5` (default): Crank–Nicolson — second-order accurate, moderate acoustic damping
  - `β = 1`: backward Euler — maximum damping of vertical acoustic modes

See also [`ExplicitTimeStepping`](@ref), [`SplitExplicitTimeDiscretization`](@ref).
"""
struct VerticallyImplicitTimeStepping{FT}
    β :: FT
end

VerticallyImplicitTimeStepping(; β=0.5) = VerticallyImplicitTimeStepping(β)
