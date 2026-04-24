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

Abstract supertype for the choice of vertical-face projection (how to
interpolate a cell-centered quantity to an adjacent z-face inside the
acoustic substep kernels). MPAS calls this the "interface projection"; it is
a finite-volume operator choice, not a literal linear interpolation.

On a uniform vertical grid, every subtype collapses to ``\\tfrac{1}{2}(X_k + X_{k-1})``
and behaves identically to Oceananigans' `ℑzᵃᵃᶠ`. The subtypes only differ on
stretched vertical grids.

Concrete subtypes:

  - [`LinearInterpolation`](@ref) — `fzm·Xᵏ + fzp·X⁻` with weights
    `fzm = Δzᶜ(k-1)/total`, `fzp = Δzᶜ(k)/total` (i.e. weight on center `k` is
    proportional to the *opposite* cell's thickness). Coincides with geometric
    linear interpolation if cell centers are at cell midpoints. Matches MPAS's
    default `config_interface_projection = "linear_interpolation"`.
  - [`ArithmeticMean`](@ref) — simple ``\\tfrac{1}{2}(X_k + X_{k-1})``. Matches
    Oceananigans' built-in `ℑzᵃᵃᶠ`. On stretched grids this is a weighted
    average that does *not* correspond to any linear geometry; it is a
    deliberate simplification elsewhere in Breeze and is offered here for
    consistency tests.
"""
abstract type VerticalFaceProjection end

"""
$(TYPEDEF)

MPAS-style `fzm/fzp` face projection: linear interpolation between cell
centers to the intervening z-face. Matches the MPAS
`config_interface_projection = "linear_interpolation"` default.
"""
struct LinearInterpolation <: VerticalFaceProjection end

"""
$(TYPEDEF)

Arithmetic-mean face projection: ``\\tfrac{1}{2}(X_k + X_{k-1})``. Identical
to Oceananigans' `ℑzᵃᵃᶠ`. On a uniform grid, indistinguishable from
[`LinearInterpolation`](@ref).
"""
struct ArithmeticMean <: VerticalFaceProjection end

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
- `length_scale`: Optional override for the dispersion length ``\\ell_\\mathrm{disp}`` (MPAS `config_len_disp`). Default `nothing` (Breeze auto-derives ``\\min(Δx, Δy)`` over non-Flat horizontal axes). The `LS` type parameter is either `Nothing` (auto-derive) or the same float type as `coefficient`, so the struct is fully concretely typed and GPU-isbits.
"""
struct ThermodynamicDivergenceDamping{FT, LS} <: AcousticDampingStrategy
    coefficient :: FT
    length_scale :: LS
end

function ThermodynamicDivergenceDamping(; coefficient = 0.1, length_scale = nothing)
    if length_scale === nothing
        FT = typeof(coefficient)
        coef_FT = convert(FT, coefficient)
        return ThermodynamicDivergenceDamping{FT, Nothing}(coef_FT, nothing)
    else
        FT = promote_type(typeof(coefficient), typeof(length_scale))
        coef_FT = convert(FT, coefficient)
        len_FT  = convert(FT, length_scale)
        return ThermodynamicDivergenceDamping{FT, FT}(coef_FT, len_FT)
    end
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
- `forward_weight`: Off-centering parameter ``ω`` for the vertically implicit ``(\\rho w)''``–``(\\rho\\theta)''`` solve. ``ω > 0.5`` damps vertical acoustic modes; the MPAS off-centering is ``ε = 2ω - 1``. Default: 0.7. (Note: ERF/MPAS canonical ``β_s = 0.1`` corresponds to ``ω = 0.55``, but Breeze's implementation of the tridiagonal coefficients appears to require more off-centering for stability — needs investigation; see `validation/substepping/NOTES.md`.)
- `damping`: Acoustic divergence damping strategy ([`AcousticDampingStrategy`](@ref)). Default: [`PressureProjectionDamping`](@ref) with `coefficient = 0.5`, the literal ERF/CM1/WRF projection form at the empirically-tuned coefficient that produces a clean BCI lifecycle in the DCMIP2016 baroclinic-wave comparison. For small-amplitude wave configurations like the Skamarock-Klemp 1994 inertia-gravity wave, this coefficient is more aggressive than necessary; pass `damping = PressureProjectionDamping(coefficient = 0.1)` for a milder filter. Other options: [`ThermodynamicDivergenceDamping`](@ref) (the MPAS Klemp-Skamarock-Ha 2018 form), [`ConservativeProjectionDamping`](@ref) (cheaper algebraic variant of `PressureProjectionDamping`), or [`NoDivergenceDamping`](@ref) to disable damping entirely.
- `substep_distribution`: How acoustic substeps are distributed across the three WS-RK3 stages. One of [`ProportionalSubsteps`](@ref) (default; constant ``Δτ = Δt/N`` with stage counts ``N/3``, ``N/2``, ``N``) or [`MonolithicFirstStage`](@ref) (single substep of size ``Δt/3`` in stage 1, MPAS-A `config_time_integration_order = 3` form).
- `face_projection`: How to project cell-centered quantities onto adjacent z-faces inside the substep kernels. One of [`LinearInterpolation`](@ref) (default, matches MPAS's `config_interface_projection = "linear_interpolation"`) or [`ArithmeticMean`](@ref) (Oceananigans-style `½·(Xᵏ + X⁻)`). The two are identical on uniform-Δz grids — use a stretched vertical grid to see a difference.

See also [`ExplicitTimeStepping`](@ref).
"""
struct SplitExplicitTimeDiscretization{N, FT, D <: AcousticDampingStrategy, AD <: AcousticSubstepDistribution, FP <: VerticalFaceProjection}
    substeps :: N
    forward_weight :: FT
    damping :: D
    substep_distribution :: AD
    face_projection :: FP
end

function SplitExplicitTimeDiscretization(; substeps = nothing,
                                           forward_weight = 0.7,
                                           damping = PressureProjectionDamping(coefficient = 0.5),
                                           substep_distribution = ProportionalSubsteps(),
                                           face_projection = LinearInterpolation(),
                                           divergence_damping_coefficient = nothing)

    # Backwards-compat: the old `divergence_damping_coefficient` kwarg was
    # silently dropped at runtime in prior releases (the substepper used a
    # hardcoded `smdiv = 0.1`). Map it to a ThermodynamicDivergenceDamping
    # when no explicit `damping` was passed and warn loudly so users know to
    # migrate to the new API.
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
                                           substep_distribution,
                                           face_projection)
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
