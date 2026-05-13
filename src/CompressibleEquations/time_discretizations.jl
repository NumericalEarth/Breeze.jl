#####
##### Time discretization types for CompressibleDynamics
#####
##### Two top-level choices:
#####   - SplitExplicitTimeDiscretization — Wicker-Skamarock RK3 outer loop
#####     with an inner substep that evolves linearized acoustic perturbations
#####     about each RK stage-entry state. The default uses off-centered
#####     Crank-Nicolson and Klemp 2018 horizontal divergence damping.
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
linearized perturbations about each RK stage-entry state.
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

  - [`NoDivergenceDamping`](@ref) — no damping.
  - [`ThermalDivergenceDamping`](@ref) — [Klemp, Skamarock & Ha (2018)](@cite KlempSkamarockHa2018)
    momentum correction using the discrete ``δ_τ(ρθ)`` tendency as the
    divergence proxy. This is the default used by
    [`SplitExplicitTimeDiscretization`](@ref).
"""
abstract type AcousticDampingStrategy end

"""
$(TYPEDEF)

No acoustic divergence damping. The substep loop advances perturbation
fields without applying any post-substep momentum correction.
"""
struct NoDivergenceDamping <: AcousticDampingStrategy end

"""
$(TYPEDEF)

Acoustic divergence damping that uses the **(ρθ)′ tendency as a
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
[Baldauf (2010)](@cite Baldauf2010). After each acoustic substep, the
horizontal momentum perturbation components ``(ρu)′`` and ``(ρv)′`` pick
up an explicit correction proportional to the horizontal gradient of
``D``. If `damp_vertical = true`, the vertical component is folded
implicitly into the column tridiag as a Laplacian on ``(ρw)′``. By
default, `damp_vertical = false` and vertical acoustic damping comes from
the off-centered implicit solve.

Per-substep momentum correction (Klemp, Skamarock & Ha 2018 eq. 36, MPAS form):

```math
Δ(ρu)′ = -γ · ∂_x D , \\quad
Δ(ρv)′ = -γ · ∂_y D .
```

with local per-direction horizontal diffusivities. On a uniform square grid
this is the finite-difference analogue of MPAS's
`coef_divdamp = 2·smdiv·config_len_disp/Δτ`:

```math
γ_x = α \\, Δx^2 / Δτ , \\qquad γ_y = α \\, Δy^2 / Δτ .
```

On anisotropic or latitude-longitude grids, using the local per-direction
spacing keeps the nondimensional explicit damping strength approximately
uniform across the mesh. Pass `length_scale = ℓ` to override the automatic
local scale with the fixed diffusivity ``γ = α ℓ^2 / Δτ`` when a nominal
mesh length is more appropriate. The optional vertical tridiag contribution uses
``γ_z = α Δz² / Δτ`` when `damp_vertical = true`.

``α`` is the dimensionless Klemp 2018 coefficient (= MPAS
`config_smdiv`, default `0.1`). The combined 2-D horizontal
explicit-time stability bound is ``8α ≤ 2 → α ≤ 0.25``; the default
sits well below it. Combined with the `SplitExplicitTimeDiscretization`
default ``\\omega = 0.65``, this keeps the rest atmosphere at machine ε
at Δt = 20 s and provides acoustic-mode damping in the DCMIP-2016
dry / moist baroclinic-wave smoke tests. It should not be read as a
guarantee that every grid-scale balanced-mode growth diagnostic is
physically correct.

Fields
======

- `coefficient`: Dimensionless damping coefficient ``α`` (Klemp 2018 /
  MPAS `config_smdiv`). Default `0.1`. The horizontal part is explicit
  and obeys the usual `8α ≤ 2` 2-D combined CFL. When
  `damp_vertical = true`, the vertical contribution is implicit and is
  folded into the column tridiag.
- `length_scale`: Optional override for the dispersion length ``d``.
  Default `nothing` (auto: local ``γ_x = α Δx^2 / Δτ`` and
  ``γ_y = α Δy^2 / Δτ``). Setting `length_scale = ℓ` forces a fixed
  ``γ = α \\, ℓ² / Δτ`` in both horizontal directions.
- `damp_vertical`: If `true`, the vertical part of the divergence
  damping is folded into the column tridiag (a Laplacian on `(ρw)′`).
  If `false` (default), no extra vertical damping is applied — the
  vertical acoustic modes are damped solely by the off-centering of the
  implicit pressure-gradient solve (``\\omega > 0.5``), which Klemp et
  al. 2018 eq. (32) shows is algebraically equivalent to a vertical
  divergence damping with diffusivity ``γ_z = c² Δτ s/2`` where
  ``s = 2\\omega - 1``.
"""
struct ThermalDivergenceDamping{FT, LS} <: AcousticDampingStrategy
    coefficient :: FT
    length_scale :: LS
    damp_vertical :: Bool
end

function ThermalDivergenceDamping(; coefficient = 0.1,
                                    length_scale = nothing,
                                    damp_vertical = false)
    FT = Oceananigans.defaults.FloatType

    if length_scale === nothing
        return ThermalDivergenceDamping{FT, Nothing}(convert(FT, coefficient), nothing, damp_vertical)
    else
        return ThermalDivergenceDamping{FT, FT}(convert(FT, coefficient),
                                                convert(FT, length_scale),
                                                damp_vertical)
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
evolves linearized acoustic perturbations about each RK stage-entry
state. The vertically implicit solve uses an off-centered Crank-Nicolson
scheme with off-centering parameter ``\\omega`` (default 0.65; ``\\omega = 0.5``
is classic centered CN). In multi-substep stages, the first acoustic
substep includes the frozen stage-entry horizontal pressure gradient but
skips the acoustic perturbation pressure gradient, which is applied on
subsequent substeps following the MPAS forward-backward sequencing.

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

  **Note on residual non-normality**: the column tridiag has
  anti-symmetric buoyancy off-diagonals (gravity-wave
  physics) and asymmetric PGF off-diagonals on a stratified ``\\bar\\theta(z)``,
  so the substep operator ``U`` has spectral radius ``\\rho(U) = 1``
  but operator norm ``\\|U\\|_2 \\gg 1`` (≈ 44 at ``\\Delta t = 20`` s,
  ``\\omega = 0.55``, no damping). Distributed FP-noise can excite the
  non-normal transient-amplification subspace. The stage-rewind
  formulation keeps the exact discrete rest atmosphere bounded even
  without divergence damping, but the default off-centering plus Klemp
  horizontal divergence damping shrinks ``\\|U^k\\|`` over enough ``k``
  and damps acoustic noise in production baroclinic-wave and LES runs.
- `damping`: Acoustic divergence damping strategy. Default:
  [`ThermalDivergenceDamping`](@ref) with coefficient 0.1. The exact
  discrete rest atmosphere is covered by `test/substepper_rest_state.jl`
  even with [`NoDivergenceDamping`](@ref), but noisy acoustic production
  cases still need damping to control grid-scale divergent modes.
- `sponge`: Optional [`UpperSponge`](@ref) that applies implicit
  Rayleigh damping to ``(ρw)′`` inside the substep loop's column tridiag,
  absorbing acoustic / gravity-wave energy in a layer below the rigid
  lid. Default `nothing` (off). Passing `UpperSponge(; ...)` enables it
  with the configured `damping_rate` and `depth`.
- `substep_distribution`: How acoustic substeps are distributed across the
  three WS-RK3 stages.

See also [`ExplicitTimeStepping`](@ref).
"""
convert_acoustic_parameter(::Type{FT}, damping::NoDivergenceDamping) where FT = damping

function convert_acoustic_parameter(::Type{FT}, damping::ThermalDivergenceDamping) where FT
    length_scale = damping.length_scale === nothing ? nothing : convert(FT, damping.length_scale)
    LS = typeof(length_scale)
    return ThermalDivergenceDamping{FT, LS}(convert(FT, damping.coefficient),
                                           length_scale,
                                           damping.damp_vertical)
end

"""
Abstract supertype for upper-sponge ramp shapes. A concrete `AbstractRamp`
is callable as `(ramp)(z, sponge_top, depth)` and returns a value in
``[0, 1]``: zero below ``z_{\\rm sponge\\_top} - \\text{depth}``, rising
to one at the lid ``z = z_{\\rm sponge\\_top}``.
"""
abstract type AbstractRamp end

"""
$(TYPEDEF)

Linear sponge ramp. Cheap but introduces a kink at the bottom of the
sponge layer (nonzero slope at ``z = H − \\text{depth}``), which can
cause small partial reflection of upgoing waves in idealised tests.
WRF's older `damp_opt = 2` form uses this.
"""
struct LinearRamp <: AbstractRamp end

@inline function (::LinearRamp)(z, sponge_top, depth)
    z_start = sponge_top - depth
    return clamp((z - z_start) / depth, zero(z), one(z))
end

"""
$(TYPEDEF)

Hermite cubic "smoothstep" sponge ramp. ``s² (3 − 2s)`` where
``s = \\text{clamp}((z − (H − \\text{depth}))/\\text{depth}, 0, 1)``.

Has zero derivative at both the layer base and the lid, so absorbs
upgoing waves smoothly without the reflective kink of [`LinearRamp`](@ref).
Functionally equivalent to [`Sin2Ramp`](@ref) but ~5–10× cheaper inside
the GPU kernel (no transcendental). Recommended default.
"""
struct CubicRamp <: AbstractRamp end

@inline function (::CubicRamp)(z, sponge_top, depth)
    z_start = sponge_top - depth
    s       = clamp((z - z_start) / depth, zero(z), one(z))
    return s * s * (3 - 2 * s)
end

"""
$(TYPEDEF)

``\\sin^2`` sponge ramp from Klemp, Dudhia & Hassiotis (2008). Same
zero-derivative-at-both-ends behaviour as [`CubicRamp`](@ref), but with
a transcendental call. Provided for parity with WRF (`damp_opt = 3`) /
MPAS-Atmosphere; prefer `CubicRamp` for performance in new code.
"""
struct Sin2Ramp <: AbstractRamp end

@inline function (::Sin2Ramp)(z, sponge_top, depth)
    z_start = sponge_top - depth
    s       = clamp((z - z_start) / depth, zero(z), one(z))
    return sin(π/2 * s)^2
end

"""
$(TYPEDEF)

Implicit upper Rayleigh sponge for the substepper inner loop. Damps ``(ρw)′``
toward zero inside a layer of thickness `depth` below the model lid, with
peak damping rate `damping_rate` (in 1/s) at the lid scaled by `ramp(z)`.

The damping is applied **inside the column tridiag** as a CN-weighted
contribution (paralleling the existing implicit divergence-damping
treatment): ``δτᵐ⁺ × \\text{rate} × \\text{ramp}(z)`` on the LHS diagonal,
``δτˢ⁻ × \\text{rate} × \\text{ramp}(z)`` on the explicit-half RHS. This
matches the Rayleigh-layer form of the Klemp, Dudhia & Hassiotis (2008)
absorbing treatment used in WRF (`damp_opt=3`) and MPAS-Atmosphere. The
profile shape is controlled by `ramp`; use [`Sin2Ramp`](@ref) for the
classic ``\\sin^2`` profile.

# Keyword arguments

- `damping_rate`: peak damping rate at the lid, in 1/s. Default `0.2`.

  Because the damping is **fully implicit** in the inner-loop tridiag, it
  is unconditionally stable for any positive value, so the choice is
  guided by physics rather than CFL. Typical guidance:

  - ``\\text{rate} ≳ N`` (Brunt–Väisälä frequency, ~0.01 /s in the
    stratosphere) is the lower bound at which gravity waves are absorbed
    rather than reflected within the layer crossing time.
  - WRF's `dampcoef` default and Klemp et al.'s recommendation is
    `0.2` (i.e. τ ≈ 5 s at the lid) — comfortably above ``N`` and
    aggressive enough to absorb in one or two crossings.
  - Larger values are fine numerically but produce a sharper "cap"
    near the lid; if the application cares about resolved dynamics
    just below the sponge, prefer ``\\text{rate} ≈ 0.1`` and a deeper
    layer.

- `depth`: sponge-layer thickness below the lid, in metres. Default `5e3`.

  Should span at least ~10 grid cells in the vertical to give the smooth
  profile room to absorb without aliasing; for ``Δz ≈ 1\\,\\text{km}`` the
  default of 5 km gives 5 cells (marginal — bump to 10 km if w-spectrum
  has structure right below the lid).

- `ramp`: an [`AbstractRamp`](@ref) controlling the profile shape.
  Default [`CubicRamp()`](@ref). Other built-ins: [`Sin2Ramp()`](@ref),
  [`LinearRamp()`](@ref). Custom shapes are supported by subtyping
  `AbstractRamp` and defining `(::MyRamp)(z, sponge_top, depth)`.

The ramp is z-only (no horizontal variation), so the sponge does not
break zonal symmetry.
"""
struct UpperSponge{FT, R <: AbstractRamp}
    damping_rate :: FT
    depth :: FT
    ramp :: R
end

function UpperSponge(; damping_rate = 0.2, depth = 5e3, ramp = CubicRamp())
    ramp isa AbstractRamp ||
        throw(ArgumentError("`ramp` must be an `<:AbstractRamp` (e.g. `CubicRamp()`, `Sin2Ramp()`, `LinearRamp()`)"))
    FT = Oceananigans.defaults.FloatType
    return UpperSponge{FT, typeof(ramp)}(convert(FT, damping_rate),
                                           convert(FT, depth),
                                           ramp)
end

convert_acoustic_parameter(::Type{FT}, ::Nothing) where FT = nothing

convert_acoustic_parameter(::Type{FT}, sponge::UpperSponge) where FT =
    UpperSponge{FT, typeof(sponge.ramp)}(convert(FT, sponge.damping_rate),
                                        convert(FT, sponge.depth),
                                        sponge.ramp)

struct SplitExplicitTimeDiscretization{N, FT, D, US, AD <: AcousticSubstepDistribution}
    substeps :: N
    forward_weight :: FT
    damping :: D
    sponge :: US
    substep_distribution :: AD
end

function SplitExplicitTimeDiscretization(FT=Oceananigans.defaults.FloatType;
                                         substeps = nothing,
                                         forward_weight = FT(0.65),
                                         damping = ThermalDivergenceDamping(; coefficient = FT(0.1)),
                                         sponge = nothing,
                                         substep_distribution = ProportionalSubsteps())

    damping isa AcousticDampingStrategy ||
        throw(ArgumentError("`damping` must be an `AcousticDampingStrategy`"))

    sponge isa Union{Nothing, UpperSponge} ||
        throw(ArgumentError("`sponge` must be `nothing` or an `UpperSponge`"))

    return SplitExplicitTimeDiscretization(
        substeps,
        convert(FT, forward_weight),
        convert_acoustic_parameter(FT, damping),
        convert_acoustic_parameter(FT, sponge),
        substep_distribution,
    )
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
