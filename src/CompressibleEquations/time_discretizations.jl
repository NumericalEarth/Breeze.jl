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

"""
$(TYPEDEF)

Split-explicit time discretization for compressible dynamics using the
Exner pressure formulation following CM1 (Bryan 2002).

Uses acoustic substepping following [Wicker and Skamarock (2002)](@cite WickerSkamarock2002):
- Outer loop: WS-RK3 for slow tendencies (advection, Coriolis, diffusion)
- Inner loop: Forward-backward acoustic substeps for fast tendencies (pressure gradient)

The acoustic loop uses velocity (u, v, w) and Exner pressure perturbation (π') as
prognostic variables, with a vertically implicit w-π' coupling and CM1-style
divergence damping.

This allows using advective CFL time steps (~10-20 m/s) instead of acoustic CFL
time steps (~340 m/s), typically enabling ~6x larger time steps.

Fields
======

- `substeps`: Number of acoustic substeps for the **full** time step (stage 3 of WS-RK3). For WS-RK3, earlier stages take fewer substeps (``Nτ = \\mathrm{round}(β N)``), keeping ``Δτ = Δt/N`` constant. Default: `nothing` (automatically computed from the acoustic CFL condition each time step)
- `forward_weight`: Off-centering parameter ω for the vertically implicit solver. ω > 0.5 damps vertical acoustic modes. Default: 0.55 (gives epssm=0.1, matching MPAS default)
- `divergence_damping_coefficient`: Forward-extrapolation filter coefficient ``ϰ^{di}`` applied to the Exner pressure perturbation: ``π̃' = π' + ϰ^{di} (π' - π'_{old})``. Default: 0.10 (CM1 default)
- `acoustic_damping_coefficient`: Klemp (2018) divergence damping ``ϰ^{ac}``. Post-implicit-solve velocity correction: ``u -= ϰ^{ac} c_p θ_v ∂Δπ'/∂x``. Provides constant damping per outer Δt regardless of substep count. Needed by WS-RK3 at large Δt. Default: 0.0
- `substep_distribution`: How acoustic substeps are distributed across the three WS-RK3 stages. One of [`ProportionalSubsteps`](@ref) (default; constant ``Δτ = Δt/N`` with stage counts ``N/3``, ``N/2``, ``N``) or [`MonolithicFirstStage`](@ref) (single substep of size ``Δt/3`` in stage 1, MPAS-A `config_time_integration_order = 3` form).

See also [`ExplicitTimeStepping`](@ref).
"""
struct SplitExplicitTimeDiscretization{N, FT, AD <: AcousticSubstepDistribution}
    substeps :: N
    forward_weight :: FT
    divergence_damping_coefficient :: FT
    acoustic_damping_coefficient :: FT
    substep_distribution :: AD
end

function SplitExplicitTimeDiscretization(; substeps=nothing,
                                           forward_weight=0.6,
                                           divergence_damping_coefficient=0.10,
                                           acoustic_damping_coefficient=0.0,
                                           substep_distribution=ProportionalSubsteps())
    return SplitExplicitTimeDiscretization(substeps,
                                           forward_weight,
                                           divergence_damping_coefficient,
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
