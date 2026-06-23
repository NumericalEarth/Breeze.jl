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

  - [`ProportionalSubsteps`](@ref) — each stage independently covers its own
    interval ``β Δt`` with ``Nτ = ⌈β N⌉`` substeps of size ``Δτ = β Δt / Nτ``
    (count proportional to the stage fraction; size fitted so the substeps exactly
    tile ``β Δt``). This is the default.

  - [`ConstantSubstepSize`](@ref) — every stage uses the same substep size
    ``Δτ = Δt/N`` (``N`` rounded up to a multiple of 6 so ``β N`` is integral),
    with stage-dependent counts ``Nτ = β N``.

  - [`MonolithicFirstStage`](@ref) — stage 1 collapses to a single substep of
    size ``Δt/3``; stages 2 and 3 are the same as `ConstantSubstepSize`.
"""
abstract type AcousticSubstepDistribution end

"""
$(TYPEDEF)

Acoustic substep distribution where each WS-RK3 stage independently covers its
interval ``β_\\mathrm{stage} Δt`` with ``Nτ = ⌈β_\\mathrm{stage} N⌉`` substeps of
size ``Δτ = β_\\mathrm{stage} Δt / Nτ``. The count is proportional to the stage
fraction and the size is fitted so the substeps exactly tile each stage — exact
coverage at the minimum substep count (no global quantization; Δτ may differ
slightly by stage).

This is the default.
"""
struct ProportionalSubsteps <: AcousticSubstepDistribution end

"""
$(TYPEDEF)

Acoustic substep distribution where every stage uses the same substep size
``Δτ = Δt/N``. ``N`` is rounded up to a multiple of 6 (= LCM of the WS-RK3 stage
denominators 2 and 3) so the per-stage count ``Nτ = β_\\mathrm{stage} N`` is an
exact integer and each stage covers exactly ``β Δt`` — uniform Δτ, at the cost of
over-resolving (substep count is the next multiple of 6 ≥ the CFL minimum).
"""
struct ConstantSubstepSize <: AcousticSubstepDistribution end

"""
$(TYPEDEF)

Acoustic substep distribution where stage 1 collapses to a single substep
of size ``Δt/3``; stages 2 and 3 are the same as
[`ConstantSubstepSize`](@ref) (``N/2`` and ``N`` substeps of size
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
``\\partial_t (ρθ)' + \\nabla\\cdot(ρθ^L u') = 0``, the per-substep
quantity

```math
D \\equiv \\frac{(ρθ)' - (ρθ)'_\\mathrm{old}}{θ^L}
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
implicitly into the column tridiag as a Laplacian on the acoustic vertical
momentum perturbation: ``(ρw)′`` for height-coordinate dynamics and
``(ρ\tilde{w})′`` for terrain-following dynamics. By default,
`damp_vertical = false` and vertical acoustic damping comes from the
off-centered implicit solve.

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
default ``\\omega = 0.65``, this preserves the exact discrete rest
atmosphere at Δt = 20 s and damps divergent acoustic noise in production
runs. It should not be read as a guarantee that every grid-scale
balanced-mode growth diagnostic is physically correct.

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
  damping is folded into the column tridiag (a Laplacian on `(ρw)′` in
  height coordinates or `(ρw̃)′` in terrain-following coordinates).
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

"""
$(TYPEDEF)

Acoustic divergence damping that forms the 3-D divergence ``∇·(ρ𝐮)′`` **directly** from the
perturbation momentum, rather than approximating it through the ``(ρθ)′`` substep tendency the way
[`ThermalDivergenceDamping`](@ref) does (Klemp, Skamarock & Ha 2018, their eq. 36). After each acoustic
substep the horizontal perturbation momentum receives the correction

```math
Δ(ρu)′ = α\\, Δx²\\, ∂ₓ[∇·(ρ𝐮)′], \\qquad Δ(ρv)′ = α\\, Δy²\\, ∂_y[∇·(ρ𝐮)′],
```

with the single dimensionless coefficient `α` (MPAS `config_smdiv`, default `0.1`). This is the direct,
"construct the divergence ``D′``" form (KSH18 eq. 7 / their §3). Because the thermal proxy obeys
``(ρθ)′ - (ρθ)′ˢ⁻ ≈ -Δτ\\, θᴸ\\, ∇·(ρ𝐮)′``, the two strategies damp the same leading-order divergence for
matching `α`; the direct form differences the velocity field itself and so carries no ``1/Δτ`` in the
local diffusivity.
"""
struct DivergenceDamping{FT} <: AcousticDampingStrategy
    coefficient :: FT
end

DivergenceDamping(; coefficient = 0.1) =
    DivergenceDamping{Oceananigans.defaults.FloatType}(convert(Oceananigans.defaults.FloatType, coefficient))

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
  step (see `acoustic_cfl`).
- `acoustic_cfl`: Target horizontal acoustic Courant number used by the
  adaptive substep count when `substeps === nothing`. The substep count
  is ``N \\approx \\lceil \\Delta t \\, \\mathbb{C}^{ac} /
  (\\mathrm{acoustic\\_cfl} \\cdot \\Delta x_\\min) \\rceil``, so smaller
  values give more substeps. Default `0.5` (the ERF/WRF target —
  equivalent to the conventional safety factor of `2`). Ignored when
  `substeps` is set explicitly.
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
  ``\\omega = 0.55``, no damping). Perturbations can transiently project
  onto the non-normal amplified subspace. The stage-rewind formulation
  keeps the exact discrete rest atmosphere bounded even without divergence
  damping, while the default off-centering plus Klemp horizontal
  divergence damping damps acoustic noise in production baroclinic-wave
  and LES runs.
- `damping`: Acoustic divergence damping strategy. Default:
  [`ThermalDivergenceDamping`](@ref) with coefficient 0.1. The exact
  discrete rest atmosphere is covered by `test/substepper_rest_state.jl`
  even with [`NoDivergenceDamping`](@ref), but noisy acoustic production
  cases use damping to control grid-scale divergent modes.
- `sponge`: Optional [`UpperSponge`](@ref) that applies implicit
  Rayleigh damping to ``(ρw)′`` inside the substep loop's column tridiag,
  absorbing acoustic / gravity-wave energy in a layer below the rigid
  lid. Default `nothing` (off). Passing `UpperSponge(; ...)` enables it
  with the configured `damping_rate` and `depth`.
- `substep_distribution`: How acoustic substeps are distributed across the
  three WS-RK3 stages.
- `open_boundary_relaxation`: Per-substep relaxation factor ``α \\in (0, 1]``
  applied at the outermost open-boundary cell of ``ρ′,(ρθ)′`` to enforce the
  prescribed wall value across the acoustic substeps. Default ``α = 0.5``,
  matching FV3-LAM's outermost-blend-row weight (``\\approx 0.6``). Without
  this enforcement the perturbation halos reflect, biasing the discrete mass
  balance under transient open-boundary inflow (issue #738). The relaxation is
  a no-op when no side carries an active open BC (periodic, walls, impenetrable
  defaults all skip it).

# Backward integration

Backward integration (`Δt < 0`) is supported. The off-centered
Crank–Nicolson vertical solve with ``ω ∈ [0.5, 1]`` has amplification
factor ``|A|^2 = (1 + ((1-ω) ω_0 Δτ)^2) / (1 + (ω ω_0 Δτ)^2) \\le 1``
for either sign of ``Δτ``, so the linearized acoustic substep is A-stable
in both directions. Horizontal divergence damping is sign-self-consistent
(``γ \\propto Δτ^{-1}`` and ``(ρθ)' - (ρθ)'_\\mathrm{old} \\propto Δτ``
both flip sign with `Δt`). The adaptive substep count uses ``|Δt|``, and
the optional [`UpperSponge`](@ref) keeps its dissipative sign so it acts
as a one-sided regularizer in both directions (i.e. backward integration
through a sponge layer is stable but not an exact inverse of the forward
step inside the sponge).

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

Implicit upper Rayleigh sponge for the substepper inner loop. Damps the
acoustic vertical momentum perturbation toward zero inside a layer of thickness
`depth` below the model lid, with peak damping rate `damping_rate` (in 1/s) at
the lid scaled by `ramp(z)`. The damped variable is ``(ρw)′`` for
height-coordinate dynamics and ``(ρ\tilde{w})′`` for terrain-following
dynamics.

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

- `depth`: sponge-layer thickness below the lid, in metres along the
  reference vertical coordinate. Default `5e3`.

  Should span at least ~10 grid cells in the vertical to give the smooth
  profile room to absorb without aliasing; for ``Δz ≈ 1\\,\\text{km}`` the
  default of 5 km gives 5 cells (marginal — bump to 10 km if w-spectrum
  has structure right below the lid).

- `ramp`: an [`AbstractRamp`](@ref) controlling the profile shape.
  Default [`CubicRamp()`](@ref). Other built-ins: [`Sin2Ramp()`](@ref),
  [`LinearRamp()`](@ref). Custom shapes are supported by subtyping
  `AbstractRamp` and defining `(::MyRamp)(z, sponge_top, depth)`.

The ramp depends only on the reference vertical coordinate (no horizontal
variation), so the sponge does not break zonal symmetry and remains uniform
over terrain-following grids.
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

"""
$(TYPEDEF)

Time discretization for fully compressible dynamics that integrates slow
terms with a Wicker-Skamarock RK3 outer loop and acoustic terms with
split-explicit inner substeps.

The constructor accepts `substeps` or an `acoustic_cfl` for choosing the
number of acoustic substeps, a `forward_weight` for off-centering the acoustic
solve, an acoustic `damping` strategy such as
[`ThermalDivergenceDamping`](@ref), an optional [`UpperSponge`](@ref), and a
`substep_distribution` such as [`ProportionalSubsteps`](@ref).

Backward integration (`time_step!(model, Δt)` with `Δt < 0`) is supported
for the linearized acoustic substep loop. See the field-documentation
docstring for the A-stability argument, sign-handling of the adaptive
substep count, and the irreversibility caveat for the optional
[`UpperSponge`](@ref).
"""
struct SplitExplicitTimeDiscretization{N, FT, D, US, AD <: AcousticSubstepDistribution}
    substeps :: N
    acoustic_cfl :: FT
    forward_weight :: FT
    thermodynamic_tendency_factor :: FT
    vertical_momentum_tendency_factor :: FT
    vertical_pressure_tendency_factor :: FT
    final_stage_vertical_pressure_tendency_factor :: FT
    apply_first_substep_pressure_gradient :: Bool
    damping :: D
    sponge :: US
    substep_distribution :: AD
    open_boundary_relaxation :: FT
end

function SplitExplicitTimeDiscretization(FT=Oceananigans.defaults.FloatType;
                                         substeps = nothing,
                                         acoustic_cfl = FT(0.7),
                                         forward_weight = FT(0.65),
                                         thermodynamic_tendency_factor = FT(1),
                                         vertical_momentum_tendency_factor = FT(1),
                                         vertical_pressure_tendency_factor = FT(1),
                                         final_stage_vertical_pressure_tendency_factor = FT(1),
                                         apply_first_substep_pressure_gradient = false,
                                         damping = ThermalDivergenceDamping(; coefficient = FT(0.1)),
                                         sponge = nothing,
                                         substep_distribution = ProportionalSubsteps(),
                                         open_boundary_relaxation = FT(0.5))

    damping isa AcousticDampingStrategy ||
        throw(ArgumentError("`damping` must be an `AcousticDampingStrategy`"))

    sponge isa Union{Nothing, UpperSponge} ||
        throw(ArgumentError("`sponge` must be `nothing` or an `UpperSponge`"))

    acoustic_cfl > 0 ||
        throw(ArgumentError("`acoustic_cfl` must be positive (got $(acoustic_cfl))"))

    0 < open_boundary_relaxation ≤ 1 ||
        throw(ArgumentError("`open_boundary_relaxation` must be in (0, 1] (got $(open_boundary_relaxation))"))

    return SplitExplicitTimeDiscretization(
        substeps,
        convert(FT, acoustic_cfl),
        convert(FT, forward_weight),
        convert(FT, thermodynamic_tendency_factor),
        convert(FT, vertical_momentum_tendency_factor),
        convert(FT, vertical_pressure_tendency_factor),
        convert(FT, final_stage_vertical_pressure_tendency_factor),
        Bool(apply_first_substep_pressure_gradient),
        convert_acoustic_parameter(FT, damping),
        convert_acoustic_parameter(FT, sponge),
        substep_distribution,
        convert(FT, open_boundary_relaxation),
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
