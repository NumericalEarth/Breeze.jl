#####
##### Acoustic substepping for CompressibleDynamics
#####
##### The substepper evolves linearized acoustic perturbations between WS-RK3
##### stages, with the linearization fixed at the **outer-step-start state**.
##### Prime notation denotes the perturbation about Uᴸ:
#####
#####   ρ′    = ρ   − ρᴸ
#####   (ρθ)′ = ρθ  − ρθᴸ
#####   (ρu)′ = ρu  − ρuᴸ,  (ρv)′ = ρv − ρvᴸ,  (ρw)′ = ρw − ρwᴸ
#####
##### Background quantities θᴸ = ρθᴸ/ρᴸ and Πᴸ = (pᴸ/pˢᵗ)^κ are computed
##### once per outer Δt from the snapshot Uᴸ and reused across all RK stages.
#####
##### The linearized perturbation equations integrated by the substep loop:
#####
#####   ∂t ρ′    +     ∇·((ρu)′, (ρv)′, (ρw)′)       = Gˢρ
#####   ∂t (ρθ)′ +     ∇·(θᴸ · ((ρu)′, (ρv)′, (ρw)′)) = Gˢρθ
#####   ∂t (ρu)′ + γRᵐ · Πᴸ_x ·  ∂x((ρθ)′)           = Gˢρu
#####   ∂t (ρv)′ + γRᵐ · Πᴸ_y ·  ∂y((ρθ)′)           = Gˢρv
#####   ∂t (ρw)′ + γRᵐ · Πᴸ_z ·  ∂z((ρθ)′) + g · ρ′  = Gˢρw
#####
##### Time discretization: horizontal momentum updates are forward-Euler;
##### the vertical ((ρw)′, (ρθ)′, ρ′) coupling is solved implicitly with
##### an off-centered Crank-Nicolson scheme — `forward_weight = 0.5` is
##### classic centered CN (neutrally stable for the linearized inviscid
##### system), `forward_weight > 0.5` adds dissipation. The implicit step
##### reduces to a tridiagonal Schur system in (ρw)′ at z-faces.
#####
##### After each stage's substep loop, the full prognostic state is
##### recovered: ρ = ρᴸ + ρ′, ρθ = ρθᴸ + (ρθ)′, ρu = ρuᴸ + (ρu)′, etc.,
##### and velocities are diagnosed from momenta and density.
#####
##### Notation note (Standard S13): the symbols `σ`, `η` are RESERVED in
##### atmospheric science for vertical coordinates (sigma and hybrid).
##### This module uses prime notation for perturbations and the in-code
##### kernel arguments map as
#####   ρ′    ↔ kernel arg `ρ′`     (struct field: density_perturbation)
#####   (ρθ)′ ↔ kernel arg `ρθ′`    (density_potential_temperature_perturbation)
#####   (ρu)′ ↔ kernel arg `ρu′`    (momentum_perturbation_u)
#####   (ρv)′ ↔ kernel arg `ρv′`    (momentum_perturbation_v)
#####   (ρw)′ ↔ kernel arg `ρw′`    (momentum_perturbation_w)
##### Predictors carry a `★` suffix: `ρ′★`, `ρθ′★`.
#####
##### See `validation/substepping/derivation_phase1.md` for the full
##### derivation from the continuous equations through the column tridiag.
#####
##### References:
#####   - Wicker, L. J. & Skamarock, W. C., 2002. *Time-splitting methods for
#####     elastic models using forward time schemes.* MWR 130, 2088.
#####   - Baldauf, M., 2010. *Linear stability analysis of Runge-Kutta-based
#####     partial time-splitting schemes for the Euler equations.* MWR 138, 4475.
#####   - Klemp, J. B., Skamarock, W. C. & Ha, S.-Y., 2018. *Damping acoustic
#####     modes in compressible HEVI and split-explicit time integration
#####     schemes.* MWR 146, 1911.
#####

using KernelAbstractions: @kernel, @index

using Oceananigans: CenterField, XFaceField, YFaceField, ZFaceField, architecture
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!
using Oceananigans.Operators:
    ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ,
    ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ,
    δxᶜᵃᵃ, δyᵃᶜᵃ,
    div_xyᶜᶜᶜ,
    Δzᶜᶜᶜ, Δzᶜᶜᶠ,
    Axᶠᶜᶜ, Ayᶜᶠᶜ, Vᶜᶜᶜ

using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!

using Oceananigans.Grids: Bounded, Flat, AbstractUnderlyingGrid,
                          Center, peripheral_node,
                          topology,
                          minimum_xspacing, minimum_yspacing, minimum_zspacing

using Adapt: Adapt, adapt

#####
##### Section 1 — Substepper struct
#####

"""
$(TYPEDEF)

Storage and parameters for the split-explicit acoustic substepper.

The substepper evolves linearized acoustic perturbations between WS-RK3
stages, with the linearization fixed at the outer-step-start state ``Uᴸ``.
Background quantities ``ρᴸ, ρθᴸ, pᴸ, Πᴸ, θᴸ`` are snapshotted from ``Uᴸ``
once per outer ``Δt`` and reused across all RK stages.

The vertical implicit solve uses a centered (or off-centered)
Crank-Nicolson scheme that reduces to a tridiagonal Schur system for the
vertical-momentum perturbation ``(ρw)′``.

Fields
======

- `substeps`: Number of acoustic substeps ``N`` per outer ``Δt`` (or
  `nothing` for adaptive).
- `forward_weight`: Off-centering parameter ``\\omega``. Default 0.5 is
  classic centered CN.
- `damping`: Acoustic divergence damping strategy.
- `substep_distribution`: How acoustic substeps are distributed across
  the WS-RK3 stages.

Outer-step-start (linearization point):

- `linearization_density`: ``ρᴸ`` snapshot.
- `linearization_density_potential_temperature`: ``ρθᴸ`` snapshot.
- `linearization_pressure`: ``pᴸ`` diagnosed from the EoS at ``Uᴸ``.
- `linearization_exner`: ``Πᴸ = (pᴸ/pˢᵗ)^κ`` derived from `linearization_pressure`.
- `linearization_potential_temperature`: ``θᴸ = ρθᴸ/ρᴸ`` for the perturbation
  temperature flux ``θᴸ · μ``.

Perturbation prognostics (advanced inside the substep loop):

- `density_perturbation`: ``ρ′ = ρ − ρᴸ``.
- `density_potential_temperature_perturbation`: ``(ρθ)′ = ρθ − ρθᴸ``.
- `momentum_perturbation_u`, `_v`, `_w`: ``(ρu)′, (ρv)′, (ρw)′``.

Per-column scratch (column kernel only):

- `density_predictor`, `density_potential_temperaturepredictor`: explicit
  predictors built before the implicit vertical solve.
- `previous_density_potential_temperature_perturbation`: ``η`` from the
  previous substep, used by Klemp 2018 damping.

Vertical solve:

- `slow_vertical_momentum_tendency`: assembled vertical-momentum slow
  tendency ``Gˢρw`` at z-faces (advection + Coriolis + closure + forcing,
  with PGF and buoyancy excluded — those live in the fast operator).
- `vertical_solver`: `BatchedTridiagonalSolver` for the implicit ``(ρw)′`` update.
"""
struct AcousticSubstepper{N, FT, D, AD, CF, FF, XF, YF, GT, TS}
    substeps :: N
    forward_weight :: FT
    damping :: D
    substep_distribution :: AD

    # Recovery base — FROZEN at outer-step start. Used by
    # `_recover_full_state!` to compute Uᵐ⁺ = U_recovery + (ρ′, (ρθ)′, (ρu)′, (ρv)′, (ρw)′).
    # Required by Wicker–Skamarock RK3 which integrates each stage from
    # the SAME outer-step-start Uᴸ.
    recovery_density :: CF
    recovery_density_potential_temperature :: CF

    # Linearization basic state — REFRESHED at each WS-RK3 stage from the
    # current model state ``U^{(k-1)}``. This keeps the linearized
    # operators (γRᵐ Π ∂(ρθ)′, g·ρ′, ∂x(p − pᵣ)) consistent with the slow
    # tendency `Gⁿ` (also evaluated at U^{(k-1)}) and breaks the soft
    # outer-step CFL limit Δt ≲ Δx/cs.
    linearization_density :: CF
    linearization_density_potential_temperature :: CF
    linearization_pressure :: CF
    linearization_exner :: CF
    linearization_potential_temperature :: CF

    # Moist basic state — snapshotted at outer-step start. Used by the
    # moist-aware linearization (Phase 3 of PRISTINE_SUBSTEPPER_PLAN.md
    # §A3/B1/B2) to evaluate ``Rᵐᴸ, γᵐᴸ, μᵥᴸ`` per cell. For dry runs the
    # three mass-fraction fields are zero; γᵐRᵐᴸ collapses to γᵈRᵈ and
    # μᵥᴸ collapses to 1 — the linearization reduces exactly to the dry case.
    linearization_vapor_mass_fraction :: CF
    linearization_liquid_mass_fraction :: CF
    linearization_ice_mass_fraction :: CF

    # Derived from the moist mass fractions:
    # `linearization_gamma_R_mixture[i,j,k] = γᵐ(i,j,k) · Rᵐ(i,j,k)` enters the
    # linearised PGF (`γᵈRᵈ → γᵐRᵐᴸ`) — Phase 2A of the moist substepper.
    # `linearization_virtual_density_factor[i,j,k] = μᵥᴸ(i,j,k)` is precomputed
    # for use by future moist refinements but is *not* used in the buoyancy
    # term: the conservation-form momentum equation has `g·ρ` (total density),
    # so a virtual-density multiplier on `g·ρ` would be incorrect.
    linearization_gamma_R_mixture :: CF
    linearization_virtual_density_factor :: CF

    # Reference-subtracted pressure perturbation, p − pᵣ (= p when no
    # reference). Refreshed each stage along with linearization_pressure.
    pressure_perturbation :: CF

    density_perturbation :: CF
    density_potential_temperature_perturbation :: CF
    momentum_perturbation_w :: FF
    momentum_perturbation_u :: XF
    momentum_perturbation_v :: YF

    density_predictor :: CF
    density_potential_temperaturepredictor :: CF
    previous_density_potential_temperature_perturbation :: CF

    # WRF/ERF-style pressure extrapolation damping (Klemp 2018, MPAS smdiv form):
    # `lagged_*` stores (ρθ)′ at the END of the previous substep (zero at stage start);
    # `pgf_*` is the forward-biased (ρθ)′ used in the explicit horizontal PGF of
    # the next substep. For dampings other than `PressureExtrapolationDamping`,
    # `pgf_*` is just a copy of `(ρθ)′` and `lagged_*` is unused.
    lagged_density_potential_temperature_perturbation :: CF
    pgf_density_potential_temperature_perturbation :: CF

    # Direct divergence damping workspace:
    # `D = ∂ₓ(ρu)′ + ∂ᵧ(ρv)′` at cell centers. Filled once per substep when
    # `damping isa DivergenceDamping`; otherwise unused.
    horizontal_momentum_divergence :: CF

    slow_vertical_momentum_tendency :: GT
    vertical_solver :: TS
end

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.substeps,
                       a.forward_weight,
                       adapt(to, a.damping),
                       a.substep_distribution,
                       adapt(to, a.recovery_density),
                       adapt(to, a.recovery_density_potential_temperature),
                       adapt(to, a.linearization_density),
                       adapt(to, a.linearization_density_potential_temperature),
                       adapt(to, a.linearization_pressure),
                       adapt(to, a.linearization_exner),
                       adapt(to, a.linearization_potential_temperature),
                       adapt(to, a.linearization_vapor_mass_fraction),
                       adapt(to, a.linearization_liquid_mass_fraction),
                       adapt(to, a.linearization_ice_mass_fraction),
                       adapt(to, a.linearization_gamma_R_mixture),
                       adapt(to, a.linearization_virtual_density_factor),
                       adapt(to, a.pressure_perturbation),
                       adapt(to, a.density_perturbation),
                       adapt(to, a.density_potential_temperature_perturbation),
                       adapt(to, a.momentum_perturbation_w),
                       adapt(to, a.momentum_perturbation_u),
                       adapt(to, a.momentum_perturbation_v),
                       adapt(to, a.density_predictor),
                       adapt(to, a.density_potential_temperaturepredictor),
                       adapt(to, a.previous_density_potential_temperature_perturbation),
                       adapt(to, a.lagged_density_potential_temperature_perturbation),
                       adapt(to, a.pgf_density_potential_temperature_perturbation),
                       adapt(to, a.horizontal_momentum_divergence),
                       adapt(to, a.slow_vertical_momentum_tendency),
                       adapt(to, a.vertical_solver))

#####
##### Section 2 — Constructor
#####

"""
$(TYPEDSIGNATURES)

Construct an `AcousticSubstepper` for the linearized-perturbation
acoustic substep loop.

The optional `prognostic_momentum` keyword carries the prognostic
``ρu``, ``ρv``, ``ρw`` fields whose boundary conditions are inherited by
the substepper's perturbation face fields ``(ρu)′``, ``(ρv)′``, ``(ρw)′``. This is
essential on grids with `Bounded` horizontal topology so that
`fill_halo_regions!` enforces impenetrability on the perturbation
momenta.
"""
function AcousticSubstepper(grid, split_explicit::SplitExplicitTimeDiscretization;
                            prognostic_momentum = nothing)
    Ns = split_explicit.substeps
    FT = eltype(grid)
    ω  = convert(FT, split_explicit.forward_weight)
    damping = split_explicit.damping
    substep_distribution = split_explicit.substep_distribution

    # Recovery base — frozen at outer-step start.
    recovery_density                           = CenterField(grid)
    recovery_density_potential_temperature     = CenterField(grid)

    # Linearization basic state — refreshed each RK stage.
    linearization_density                         = CenterField(grid)
    linearization_density_potential_temperature   = CenterField(grid)
    linearization_pressure                        = CenterField(grid)
    linearization_exner                           = CenterField(grid)
    linearization_potential_temperature           = CenterField(grid)

    # Moist basic state — snapshotted at outer-step start. Phase 3.
    linearization_vapor_mass_fraction             = CenterField(grid)
    linearization_liquid_mass_fraction            = CenterField(grid)
    linearization_ice_mass_fraction               = CenterField(grid)
    linearization_gamma_R_mixture                 = CenterField(grid)
    linearization_virtual_density_factor          = CenterField(grid)

    # Reference-subtracted pressure perturbation (p − pᵣ).
    pressure_perturbation                         = CenterField(grid)

    # Perturbation prognostics. Inherit BCs from the prognostic momenta
    # so impenetrability propagates onto the perturbation momenta.
    bcs_ρu = prognostic_momentum === nothing ? nothing : prognostic_momentum.ρu.boundary_conditions
    bcs_ρv = prognostic_momentum === nothing ? nothing : prognostic_momentum.ρv.boundary_conditions
    bcs_ρw = prognostic_momentum === nothing ? nothing : prognostic_momentum.ρw.boundary_conditions

    _xface(grid, bcs) = bcs === nothing ? XFaceField(grid) : XFaceField(grid; boundary_conditions = bcs)
    _yface(grid, bcs) = bcs === nothing ? YFaceField(grid) : YFaceField(grid; boundary_conditions = bcs)
    _zface(grid, bcs) = bcs === nothing ? ZFaceField(grid) : ZFaceField(grid; boundary_conditions = bcs)

    density_perturbation                          = CenterField(grid)
    density_potential_temperature_perturbation    = CenterField(grid)
    momentum_perturbation_w                       = _zface(grid, bcs_ρw)
    momentum_perturbation_u                       = _xface(grid, bcs_ρu)
    momentum_perturbation_v                       = _yface(grid, bcs_ρv)

    density_predictor                                = CenterField(grid)
    density_potential_temperaturepredictor          = CenterField(grid)
    previous_density_potential_temperature_perturbation = CenterField(grid)
    lagged_density_potential_temperature_perturbation   = CenterField(grid)
    pgf_density_potential_temperature_perturbation      = CenterField(grid)
    horizontal_momentum_divergence                      = CenterField(grid)

    slow_vertical_momentum_tendency = ZFaceField(grid)

    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    scratch = zeros(arch, FT, Nx, Ny, Nz)
    vertical_solver = BatchedTridiagonalSolver(grid;
                                               lower_diagonal = AcousticTridiagLower(),
                                               diagonal       = AcousticTridiagDiagonal(),
                                               upper_diagonal = AcousticTridiagUpper(),
                                               scratch,
                                               tridiagonal_direction = ZDirection())

    return AcousticSubstepper(Ns, ω, damping, substep_distribution,
                              recovery_density,
                              recovery_density_potential_temperature,
                              linearization_density,
                              linearization_density_potential_temperature,
                              linearization_pressure,
                              linearization_exner,
                              linearization_potential_temperature,
                              linearization_vapor_mass_fraction,
                              linearization_liquid_mass_fraction,
                              linearization_ice_mass_fraction,
                              linearization_gamma_R_mixture,
                              linearization_virtual_density_factor,
                              pressure_perturbation,
                              density_perturbation,
                              density_potential_temperature_perturbation,
                              momentum_perturbation_w,
                              momentum_perturbation_u,
                              momentum_perturbation_v,
                              density_predictor,
                              density_potential_temperaturepredictor,
                              previous_density_potential_temperature_perturbation,
                              lagged_density_potential_temperature_perturbation,
                              pgf_density_potential_temperature_perturbation,
                              horizontal_momentum_divergence,
                              slow_vertical_momentum_tendency,
                              vertical_solver)
end

#####
##### Section 3 — Outer-step-start linearization
#####

"""
$(TYPEDSIGNATURES)

Snapshot the outer-step-start state ``Uᴸ`` and compute the background
quantities used by the substepper as the linearization point. Called
once per outer ``Δt`` by the WS-RK3 driver.

After this call:
  - `linearization_density`               = ρᴸ
  - `linearization_density_potential_temperature` = ρθᴸ
  - `linearization_pressure`              = pᴸ (= `model.dynamics.pressure` at outer-step start)
  - `linearization_exner`                 = Πᴸ = (pᴸ/pˢᵗ)^κ
  - `linearization_potential_temperature` = θᴸ = ρθᴸ/ρᴸ
"""
function freeze_linearization_state!(substepper::AcousticSubstepper, model)
    ρθ_field = thermodynamic_density(model.formulation)

    # Snapshot the RECOVERY BASE (frozen across all 3 stages). Used by
    # `_recover_full_state!` so that Uᵐ⁺ = U⁰_outer + (ρ′, (ρθ)′, (ρu)′, (ρv)′, (ρw)′) — the
    # WS-RK3 invariant that each stage starts from Uᴸ.
    parent(substepper.recovery_density)                       .= parent(model.dynamics.density)
    parent(substepper.recovery_density_potential_temperature) .= parent(ρθ_field)

    # Then prime the linearization basic state to Uᴸ for stage 1.
    refresh_linearization_basic_state!(substepper, model)

    fill_halo_regions!(substepper.recovery_density)
    fill_halo_regions!(substepper.recovery_density_potential_temperature)

    return nothing
end

# Refresh the linearization basic state (Πᴸ, θᴸ, pᴸ, p − pᵣ, plus the
# matching ρᴸ, ρθᴸ used by the slow vertical-momentum-tendency assembly)
# from the *current* state. Called once at outer-step start by
# `freeze_linearization_state!` and again before each subsequent RK stage
# by `prepare_acoustic_cache!`.
function refresh_linearization_basic_state!(substepper::AcousticSubstepper, model)
    grid = model.grid
    arch = architecture(grid)
    FT   = eltype(grid)
    constants = model.thermodynamic_constants
    Rᵈ   = convert(FT, dry_air_gas_constant(constants))
    Rᵛ   = convert(FT, vapor_gas_constant(constants))
    cᵖᵈ  = convert(FT, constants.dry_air.heat_capacity)
    cᵖᵛ  = convert(FT, constants.vapor.heat_capacity)
    cˡ   = convert(FT, constants.liquid.heat_capacity)
    cⁱ   = convert(FT, constants.ice.heat_capacity)
    κ    = Rᵈ / cᵖᵈ
    pˢᵗ  = convert(FT, model.dynamics.standard_pressure)

    ρθ_field = thermodynamic_density(model.formulation)

    parent(substepper.linearization_density)                       .= parent(model.dynamics.density)
    parent(substepper.linearization_density_potential_temperature) .= parent(ρθ_field)
    parent(substepper.linearization_pressure)                      .= parent(model.dynamics.pressure)

    # θ_lin = ρθ/ρ and Π_lin = (p/pˢᵗ)^κ from the current state.
    launch!(arch, grid, :xyz, _compute_linearization_exner_and_theta!,
            substepper.linearization_exner,
            substepper.linearization_potential_temperature,
            substepper.linearization_pressure,
            substepper.linearization_density,
            substepper.linearization_density_potential_temperature,
            pˢᵗ, κ)

    # Reference-subtracted pressure perturbation. For ExnerReferenceState
    # the reference depends only on z, so ∂x pᵣ = ∂y pᵣ = 0; the
    # horizontal force is then ∂x(p − pᵣ) = ∂x p. Reference
    # subtraction in z guarantees a hydrostatic rest atmosphere has zero
    # vertical drive, free of FP-rounding noise.
    ref = model.dynamics.reference_state
    if ref isa Nothing
        parent(substepper.pressure_perturbation) .= parent(substepper.linearization_pressure)
    else
        launch!(arch, grid, :xyz, _compute_pressure_perturbation!,
                substepper.pressure_perturbation,
                substepper.linearization_pressure, ref.pressure)
    end

    # Moist basic state snapshot. Vapor is always present (qᵛ field exists
    # even for dry runs — zeroed out). Condensed phases are only snapshotted
    # when the microphysics scheme prognoses them; otherwise the substepper's
    # condensed-phase fields stay zero.
    snapshot_moist_basic_state!(substepper, model)

    # γᵐRᵐᴸ and μᵥᴸ derived from the snapshotted mass fractions. γᵐRᵐᴸ enters
    # the substepper's PGF (γᵈRᵈ → γᵐRᵐᴸ); μᵥᴸ is precomputed for diagnostics
    # / future formulation work but is NOT applied to the buoyancy term — see
    # the field's docstring above for why. For dry runs (qᵛᴸ = qˡᴸ = qⁱᴸ = 0):
    # γᵐRᵐᴸ → γᵈRᵈ bit-identically and μᵥᴸ → 1.
    launch!(arch, grid, :xyz, _compute_linearization_mixture_eos!,
            substepper.linearization_gamma_R_mixture,
            substepper.linearization_virtual_density_factor,
            substepper.linearization_vapor_mass_fraction,
            substepper.linearization_liquid_mass_fraction,
            substepper.linearization_ice_mass_fraction,
            Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, cˡ, cⁱ)

    fill_halo_regions!(substepper.linearization_density)
    fill_halo_regions!(substepper.linearization_density_potential_temperature)
    fill_halo_regions!(substepper.linearization_pressure)
    fill_halo_regions!(substepper.linearization_exner)
    fill_halo_regions!(substepper.linearization_potential_temperature)
    fill_halo_regions!(substepper.linearization_vapor_mass_fraction)
    fill_halo_regions!(substepper.linearization_liquid_mass_fraction)
    fill_halo_regions!(substepper.linearization_ice_mass_fraction)
    fill_halo_regions!(substepper.linearization_gamma_R_mixture)
    fill_halo_regions!(substepper.linearization_virtual_density_factor)
    fill_halo_regions!(substepper.pressure_perturbation)

    return nothing
end

# Copy moisture mass fractions from the model state into the substepper's
# outer-step snapshot. Vapor (`qᵛ`) is always available — for dry runs the
# field exists but is identically zero. Condensed phases are picked up by
# field name from `model.microphysical_fields`:
#   - liquid: `:qˡ` (saturation-adjustment) or `:qᶜˡ` (non-equilibrium /
#     four-category bulk); zero if neither exists
#   - ice: `:qⁱ` or `:qᶜⁱ`; zero if neither exists
# These mass fractions feed `_compute_linearization_mixture_eos!` to derive
# `linearization_gamma_R_mixture` (= γᵐRᵐᴸ) and
# `linearization_virtual_density_factor` (= μᵥᴸ). For dry runs all three are
# zero — γᵐRᵐᴸ collapses to γᵈRᵈ and μᵥᴸ to 1, so dry tests stay
# bit-identical.
function snapshot_moist_basic_state!(substepper::AcousticSubstepper, model)
    qᵛ = specific_prognostic_moisture(model)
    parent(substepper.linearization_vapor_mass_fraction) .= parent(qᵛ)

    fields = model.microphysical_fields
    _copy_or_zero!(substepper.linearization_liquid_mass_fraction, fields, (:qˡ, :qᶜˡ))
    _copy_or_zero!(substepper.linearization_ice_mass_fraction,    fields, (:qⁱ, :qᶜⁱ))

    return nothing
end

@inline function _copy_or_zero!(dest, fields::NamedTuple, candidates::Tuple)
    for name in candidates
        if haskey(fields, name)
            parent(dest) .= parent(fields[name])
            return nothing
        end
    end
    fill!(parent(dest), 0)
    return nothing
end

@kernel function _compute_pressure_perturbation!(p′, pᴸ, pᵣ)
    i, j, k = @index(Global, NTuple)
    @inbounds p′[i, j, k] = pᴸ[i, j, k] - pᵣ[i, j, k]
end

@kernel function _compute_linearization_exner_and_theta!(Π, θ, p, ρ, ρθ, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Π[i, j, k] = (p[i, j, k] / pˢᵗ)^κ
        ρ̂ = ifelse(ρ[i, j, k] == 0, one(eltype(ρ)), ρ[i, j, k])
        θ[i, j, k] = ρθ[i, j, k] / ρ̂
    end
end

# Compute γᵐRᵐᴸ and μᵥᴸ per cell from the snapshotted moisture mass fractions.
#   Rᵐ  = qᵈ Rᵈ + qᵛ Rᵛ                         (mixture gas constant)
#   cᵖᵐ = qᵈ cᵖᵈ + qᵛ cᵖᵛ + qˡ cˡ + qⁱ cⁱ      (mixture heat capacity)
#   cᵛᵐ = cᵖᵐ − Rᵐ
#   γᵐ  = cᵖᵐ / cᵛᵐ
#   μᵥᴸ = 1 + (Rᵛ/Rᵈ − 1) qᵛ − qˡ − qⁱ          (virtual-density factor)
# with qᵈ = 1 − qᵛ − qˡ − qⁱ. For dry inputs (qᵛ = qˡ = qⁱ = 0) these reduce
# to γᵈRᵈ and 1 exactly.
@kernel function _compute_linearization_mixture_eos!(γRᵐ, μᵥ, qᵛ, qˡ, qⁱ,
                                                  Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, cˡ, cⁱ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        qᵛᵢ = qᵛ[i, j, k]
        qˡᵢ = qˡ[i, j, k]
        qⁱᵢ = qⁱ[i, j, k]
        qᵈᵢ = 1 - qᵛᵢ - qˡᵢ - qⁱᵢ

        Rᵐ  = qᵈᵢ * Rᵈ + qᵛᵢ * Rᵛ
        cᵖᵐ = qᵈᵢ * cᵖᵈ + qᵛᵢ * cᵖᵛ + qˡᵢ * cˡ + qⁱᵢ * cⁱ
        cᵛᵐ = cᵖᵐ - Rᵐ

        # Operation order matches the dry-only path's `cᵖᵈ * Rᵈ / (cᵖᵈ - Rᵈ)`
        # so qᵛ = qˡ = qⁱ = 0 reproduces the dry γᵈRᵈ to bit-identical precision.
        γRᵐ[i, j, k] = cᵖᵐ * Rᵐ / cᵛᵐ
        μᵥ[i, j, k]  = 1 + (Rᵛ / Rᵈ - 1) * qᵛᵢ - qˡᵢ - qⁱᵢ
    end
end

"""
$(TYPEDSIGNATURES)

Stage-start cache preparation. Currently a no-op: the linearization
basic state is fixed at the outer-step start. Per-stage refresh has
been tried twice and *empirically NaNs the rest atmosphere at
Δt = 20 s* (T4 of the substepper rest-state test, even though Δt = 0.5 s
passes at machine ε). The mechanism is a recovery-vs-linearization
inconsistency: with refresh on, `linearization_density` ≠
`recovery_density` after stage 1, so the substepper's perturbation
ρ′ = ρ − linearization_ρ doesn't reconcile with `_recover_full_state!`
which reconstructs ρ = recovery_ρ + ρ′. The error is order Δt and
grows stage-to-stage. An SK08-faithful design would need to either
refresh the recovery base too (breaking the WS-RK3 invariant) or
re-derive the perturbations against the refreshed base before each
substep loop.
"""
prepare_acoustic_cache!(::AcousticSubstepper, model) = nothing

#####
##### Section 4 — Adaptive substep computation (acoustic CFL)
#####

"""
$(TYPEDSIGNATURES)

Compute the number of acoustic substeps ``N`` from the horizontal
acoustic CFL: ``N \\approx \\lceil 2 \\Delta t \\, c_s / \\Delta x_\\min \\rceil``,
with ``c_s = \\sqrt{\\gamma^d R^d \\theta_0}`` for a nominal
``\\theta_0 = 300\\,\\mathrm{K}``. The factor of 2 is the standard
ERF/WRF safety factor.
"""
function compute_acoustic_substeps(grid, Δt, thermodynamic_constants)
    FT  = eltype(grid)
    Rᵈ  = dry_air_gas_constant(thermodynamic_constants)
    cᵖᵈ = thermodynamic_constants.dry_air.heat_capacity
    γᵈ  = cᵖᵈ / (cᵖᵈ - Rᵈ)
    cs  = sqrt(γᵈ * Rᵈ * 300)

    Δx_min = let
        TX, TY, _ = topology(grid)
        Δx = TX === Flat ? typemax(FT) : minimum_xspacing(grid)
        Δy = TY === Flat ? typemax(FT) : minimum_yspacing(grid)
        min(Δx, Δy)
    end

    return max(1, ceil(Int, 2 * Δt * cs / Δx_min))
end

@inline acoustic_substeps(N::Int, grid, Δt, constants) = N
@inline acoustic_substeps(::Nothing, grid, Δt, constants) = compute_acoustic_substeps(grid, Δt, constants)

#####
##### Section 5 — Stage substep distribution
#####

# ProportionalSubsteps: every stage uses Δτ = Δt/N, Nτ = round(β·N).
@inline function _stage_substep_count_and_size(::ProportionalSubsteps, β_stage, Δt, N)
    Δτ = Δt / N
    Nτ = max(1, round(Int, β_stage * N))
    return Nτ, Δτ
end

# MonolithicFirstStage: stage 1 collapses to one substep of size Δt/3;
# stages 2 and 3 are the same as ProportionalSubsteps.
@inline function _stage_substep_count_and_size(::MonolithicFirstStage, β_stage, Δt, N)
    if β_stage < (1//3 + 1//2) / 2
        return 1, Δt / 3
    else
        Δτ = Δt / N
        Nτ = max(1, round(Int, β_stage * N))
        return Nτ, Δτ
    end
end

#####
##### Section 6 — Tridiagonal solver coefficient tag types
#####
##### These are stateless tags. The BatchedTridiagonalSolver dispatches on
##### them via `get_coefficient(...)` and computes the entry on the fly.
#####
##### Solver row index k_s aligns with face index k:
#####  - row 1     = bottom-boundary face (b = 1, c = 0, RHS = 0 → (ρw)′[1] = 0)
#####  - rows 2..Nz = interior faces; tridiagonal couples neighbours
#####  - top face (Nz+1) lives outside the solver and is held at 0
#####

struct AcousticTridiagLower    end
struct AcousticTridiagDiagonal end
struct AcousticTridiagUpper    end

import Oceananigans.Solvers: get_coefficient

# At face k, the implicit centered-CN system for `(ρw)′` couples to
# `(ρθ)′` at centers k and k-1 (above and below the face) and to `ρ′`
# at the same centers. Inline coefficient functions:

# Boundary-aware center-to-face z interpolation. At an interior face
# (both adjacent centers are active) this is the standard 2-point average.
# At a boundary face (one of the two adjacent centers is peripheral) the
# peripheral neighbor is replaced by the interior one before averaging,
# giving a one-sided interpolation that returns the interior cell value.
# Mirrors the `ℑbzᵃᵃᶜ` pattern used in Oceananigans CATKE
# (`TKEBasedVerticalDiffusivities.jl`).
@inline function ℑbzᵃᵃᶠ(i, j, k, grid, ψ)
    @inbounds f⁺ = ψ[i, j, k]      # cell ABOVE face k (cell index k)
    @inbounds f⁻ = ψ[i, j, k - 1]  # cell BELOW face k (cell index k-1)

    p⁺ = peripheral_node(i, j, k,     grid, Center(), Center(), Center())
    p⁻ = peripheral_node(i, j, k - 1, grid, Center(), Center(), Center())

    f⁺ = ifelse(p⁺, f⁻, f⁺)
    f⁻ = ifelse(p⁻, f⁺, f⁻)

    return (f⁺ + f⁻) / 2
end

# Off-centered CN tridiag derivation
# ----------------------------------
# At face k, the perturbation `(ρw)′` equation in CN form is
#
#   (ρw)′_n(k) = (ρw)′_o(k) + Δτ Gˢρw(k)
#                - Δτ × γRᵐ × Πᴸ_face(k) × (ωˢ⁻ ∂z (ρθ)′_o + ωᵐ⁺ ∂z (ρθ)′_n) / Δzᶠ(k)
#                - Δτ × g × (ωˢ⁻ ρ′_face_o(k) + ωᵐ⁺ ρ′_face_n(k))
#
# with ωᵐ⁺ = (1+ε)/2, ωˢ⁻ = (1-ε)/2 (ε=0 is centered CN).
#
# The post-solve substitution (matching the column kernel):
#   ρ′_n(k)    = ρ′★(k)  - δτᵐ⁺ × ((ρw)′_n(k+1) - (ρw)′_n(k)) / Δz_c(k)
#   (ρθ)′_n(k) = ρθ′★(k) - δτᵐ⁺ × (θᴸ_face(k+1) (ρw)′_n(k+1)
#                                        - θᴸ_face(k)   (ρw)′_n(k)) / Δz_c(k)
# where δτᵐ⁺ = ωᵐ⁺ Δτ.
#
# Substituting yields the tridiagonal coefficients (ω = ωᵐ⁺):
#
#   A[k,k+1] = -(ω Δτ)² × γRᵐ × Πᴸ_face(k) × θᴸ_face(k+1) × rdz_c(k)   / Δzᶠ(k)
#              - (ω Δτ)² × g          × rdz_c(k)   / 2
#
#   A[k,k]   = 1 + (ω Δτ)² × γRᵐ × Πᴸ_face(k) × θᴸ_face(k)   × (rdz_c(k) + rdz_c(k-1)) / Δzᶠ(k)
#                + (ω Δτ)² × g                              × (rdz_c(k) - rdz_c(k-1)) / 2
#
#   A[k,k-1] = -(ω Δτ)² × γRᵐ × Πᴸ_face(k) × θᴸ_face(k-1) × rdz_c(k-1) / Δzᶠ(k)
#              + (ω Δτ)² × g                              × rdz_c(k-1) / 2
#
# `γᵐRᵐᴸ` is the cell-centered mixture coefficient `γᵐ Rᵐ` evaluated from
# the snapshotted moisture (`linearization_gamma_R_mixture`). It is interpolated
# to z-faces inside the kernel. For dry runs (qᵛ = qˡ = qⁱ = 0) this collapses
# bit-identically to the dry constant `γᵈRᵈ`.
#
# Implicit vertical damping
# -------------------------
# When `damping isa ThermalDivergenceDamping` with `vertical_implicit = true`,
# the vertical part of the divergence damping is folded into the same tridiag.
# Reformulating the kernel correction `Δ(ρw)′ = -γ_z ∂z D` via the linearized
# (ρθ)′ continuity equation gives a discrete vertical Laplacian on `(ρw)′`:
#
#   (ρw)′_n − ω α Δz² ∂z² (ρw)′_n = (ρw)′_o + (1−ω) α Δz² ∂z² (ρw)′_o
#
# At face k the −∂z² stencil contributes (with `dᵐ⁺ ≡ ω α Δz²`):
#
#   A[k,k+1] += -dᵐ⁺ × rdz_c(k)   / Δzᶠ(k)
#   A[k,k]   += +dᵐ⁺ × (rdz_c(k) + rdz_c(k-1)) / Δzᶠ(k)
#   A[k,k-1] += -dᵐ⁺ × rdz_c(k-1) / Δzᶠ(k)
#
# The matching `(1−ω) α Δz² ∂z² (ρw)′_o` term is added to the predictor's
# right-hand side in `_buildpredictors_and_vertical_rhs!`. The constant-Courant
# scaling `γ_z = α Δz² / Δτ` makes `dᵐ⁺` and the RHS prefactor independent
# of Δτ; only `α`, `ω`, and the global vertical spacing `grid.z.Δᵃᵃᶜ` enter.
# When `vertical_implicit = false` (or for `NoDivergenceDamping`), the
# damping factor passed in is zero and the tridiag reduces to the pure
# off-centered CN acoustic system above.

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagLower, p, ::ZDirection,
                                 Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺)
    kᶠ      = k + 1
    Δzᶠ     = Δzᶜᶜᶠ(i, j, kᶠ, grid)
    Δz⁻¹ᵏ⁻  = 1 / Δzᶜᶜᶜ(i, j, kᶠ - 1, grid)

    Πᶜᶜᶠ    = ℑzᵃᵃᶠ(i, j, kᶠ, grid, Πᴸ)
    γRᵐᴸᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, kᶠ, grid, γRᵐᴸ)
    θᵏ⁻     = ℑbzᵃᵃᶠ(i, j, kᶠ - 1, grid, θᴸ)

    pgf_term  = - δτᵐ⁺^2 * γRᵐᴸᶜᶜᶠ * Πᶜᶜᶠ * θᵏ⁻ * Δz⁻¹ᵏ⁻ / Δzᶠ
    buoy_term = + δτᵐ⁺^2 * g                    * Δz⁻¹ᵏ⁻ / 2
    damp_term = - dᵐ⁺                           * Δz⁻¹ᵏ⁻ / Δzᶠ
    return pgf_term + buoy_term + damp_term
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagDiagonal, p, ::ZDirection,
                                 Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺)

    kᶠ      = k
    Δzᶠ     = Δzᶜᶜᶠ(i, j, kᶠ, grid)
    Δz⁻¹ᵏ⁺  = 1 / Δzᶜᶜᶜ(i, j, kᶠ,     grid)
    Δz⁻¹ᵏ⁻  = 1 / Δzᶜᶜᶜ(i, j, kᶠ - 1, grid)

    Πᶜᶜᶠ    = ℑzᵃᵃᶠ(i, j, kᶠ, grid, Πᴸ)
    γRᵐᴸᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, kᶠ, grid, γRᵐᴸ)
    θᶜᶜᶠ    = ℑbzᵃᵃᶠ(i, j, kᶠ, grid, θᴸ)

    pgf_diag  = δτᵐ⁺^2 * γRᵐᴸᶜᶜᶠ * Πᶜᶜᶠ * θᶜᶜᶠ * (Δz⁻¹ᵏ⁺ + Δz⁻¹ᵏ⁻) / Δzᶠ
    buoy_diag = δτᵐ⁺^2 * g                     * (Δz⁻¹ᵏ⁺ - Δz⁻¹ᵏ⁻) / 2
    damp_diag = dᵐ⁺                            * (Δz⁻¹ᵏ⁺ + Δz⁻¹ᵏ⁻) / Δzᶠ

    return one(grid) + (pgf_diag + buoy_diag + damp_diag) * (k > 1)
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagUpper, p, ::ZDirection,
                                 Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺)

    kᶠ      = k
    Δzᶠ     = Δzᶜᶜᶠ(i, j, kᶠ, grid)
    Δz⁻¹ᵏ⁺  = 1 / Δzᶜᶜᶜ(i, j, kᶠ, grid)

    Πᶜᶜᶠ    = ℑzᵃᵃᶠ(i, j, kᶠ, grid, Πᴸ)
    γRᵐᴸᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, kᶠ, grid, γRᵐᴸ)
    θᵏ⁺     = ℑbzᵃᵃᶠ(i, j, kᶠ + 1, grid, θᴸ)

    pgf_term  = - δτᵐ⁺^2 * γRᵐᴸᶜᶜᶠ * Πᶜᶜᶠ * θᵏ⁺ * Δz⁻¹ᵏ⁺ / Δzᶠ
    buoy_term = - δτᵐ⁺^2 * g                    * Δz⁻¹ᵏ⁺ / 2
    damp_term = - dᵐ⁺                           * Δz⁻¹ᵏ⁺ / Δzᶠ

    return (pgf_term + buoy_term + damp_term) * (k > 1)
end

#####
##### Section 7 — Slow vertical-momentum tendency assembly
#####
##### The full vertical-momentum equation is
#####   ∂t (ρw) + ∇·(ρw u) + ∂z p + g ρ = 0
##### The dynamics kernel runs in `SlowTendencyMode` for SplitExplicit,
##### which zeroes the PGF and buoyancy in `Gⁿρw`. We reinstate the
##### **Uᴸ-state** PGF and buoyancy here so the slow ρw tendency has the
##### form
#####   Gˢρw = -∇·(ρw u)  -  ∂z pᴸ  -  g · ρᴸ
##### and the per-substep linearized forces operate on the perturbations:
#####   ∂t (ρw)′ = Gˢρw - γRᵐ · Πᴸ · ∂z((ρθ)′)  -  g · ρ′
##### Total force = Gˢρw + perturbation force = full ∂t(ρw) at the
##### linearization-consistent level.
#####

function assemble_slow_vertical_momentum_tendency!(substepper::AcousticSubstepper, model)
    grid = model.grid
    arch = architecture(grid)
    g    = convert(eltype(grid), model.thermodynamic_constants.gravitational_acceleration)
    Gⁿρw = model.timestepper.Gⁿ.ρw

    ref = model.dynamics.reference_state
    if ref isa Nothing
        launch!(arch, grid, :xyz, _assemble_slow_vertical_momentum_tendency_no_ref!,
                substepper.slow_vertical_momentum_tendency,
                Gⁿρw,
                substepper.linearization_pressure,
                substepper.linearization_density,
                grid, g)
    else
        launch!(arch, grid, :xyz, _assemble_slow_vertical_momentum_tendency!,
                substepper.slow_vertical_momentum_tendency,
                Gⁿρw,
                substepper.linearization_pressure,
                substepper.linearization_density,
                ref.pressure, ref.density,
                grid, g)
    end

    return nothing
end

# Slow-tendency assembly with reference state. Buoyancy uses TOTAL density
# `ρᴸ` (no virtual-density factor): in conservation-form momentum,
# `∂t(ρw) = -∂z p - g ρ`, where `ρ` is total mass density and includes all
# water species. The "virtual" temperature/density transforms only appear
# when one parameterises with *dry* density as the prognostic, which Breeze
# does not do.
@kernel function _assemble_slow_vertical_momentum_tendency!(Gˢρw, Gⁿρw, pᴸ, ρᴸ, pᵣ, ρᵣ, grid, g)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Reference-subtracted PGF and buoyancy: at Uᴸ = reference state
        # both terms are exactly zero by construction of the reference.
        ∂z_p′ = ∂zᶜᶜᶠ(i, j, k, grid, _p_perturbation, pᴸ, pᵣ)
        ρ′ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, _ρ_perturbation, ρᴸ, ρᵣ)

        Gˢρw[i, j, k] = (Gⁿρw[i, j, k] - ∂z_p′ - g * ρ′ᶜᶜᶠ) * (k > 1)
    end
end

@inline _p_perturbation(i, j, k, grid, pᴸ, pᵣ) = @inbounds pᴸ[i, j, k] - pᵣ[i, j, k]
@inline _ρ_perturbation(i, j, k, grid, ρᴸ, ρᵣ) = @inbounds ρᴸ[i, j, k] - ρᵣ[i, j, k]

@kernel function _assemble_slow_vertical_momentum_tendency_no_ref!(Gˢρw, Gⁿρw, pᴸ, ρᴸ, grid, g)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂z_pᴸ  = ∂zᶜᶜᶠ(i, j, k, grid, pᴸ)
        ρᴸᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρᴸ)
        Gˢρw[i, j, k] = (Gⁿρw[i, j, k] - ∂z_pᴸ - g * ρᴸᶜᶜᶠ) * (k > 1)
    end
end

#####
##### Section 8 — Substep kernels
#####

# Reset perturbation prognostics to zero at the start of each WS-RK3
# stage. The substep loop integrates from
# `(ρ′, (ρθ)′, (ρu)′, (ρv)′, (ρw)′) = 0`; the bubble's IC pressure
# imbalance is provided through the slow vertical-momentum tendency's
# reference subtraction (`assemble_slow_vertical_momentum_tendency!`).
function reset_perturbations!(substepper, grid, arch)
    fill!(parent(substepper.density_perturbation), 0)
    fill!(parent(substepper.density_potential_temperature_perturbation), 0)
    fill!(parent(substepper.previous_density_potential_temperature_perturbation), 0)
    fill!(parent(substepper.lagged_density_potential_temperature_perturbation), 0)
    fill!(parent(substepper.pgf_density_potential_temperature_perturbation), 0)
    fill!(parent(substepper.horizontal_momentum_divergence), 0)
    fill!(parent(substepper.momentum_perturbation_u), 0)
    fill!(parent(substepper.momentum_perturbation_v), 0)
    fill!(parent(substepper.momentum_perturbation_w), 0)
    fill!(parent(substepper.density_predictor), 0)
    fill!(parent(substepper.density_potential_temperaturepredictor), 0)
    return nothing
end

# Explicit forward step for horizontal momentum perturbations (ρu)′, (ρv)′.
#
# Linearized at Uᴸ, the full horizontal pressure gradient splits as
#   ∂x p_full = ∂x(pᴸ − pᵣ) + γRᵐ Πᴸ · ∂x((ρθ)′)
# where the first piece is the FROZEN imbalance from the linearization
# point (carried by `pressure_perturbation`) and the second is the
# perturbation force. `Gⁿρu` from `SlowTendencyMode` carries advection
# only (PGF zeroed); we reinstate the frozen horizontal pressure
# perturbation here.
#
# (ρu)′^{τ+Δτ} = (ρu)′^τ + Δτ · (Gⁿρu − ∂x(pᴸ−pᵣ) − γᵐRᵐᴸ Πᴸ_x ∂x((ρθ)′))
# (ρv)′^{τ+Δτ} = (ρv)′^τ + Δτ · (Gⁿρv − ∂y(pᴸ−pᵣ) − γᵐRᵐᴸ Πᴸ_y ∂y((ρθ)′))
@kernel function _explicit_horizontal_step!(ρu′, ρv′, grid, Δτ, ρθ′, Πᴸ, p′,
                                            Gⁿρu, Gⁿρv, γRᵐᴸ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Πᴸ_x   = ℑxᶠᵃᵃ(i, j, k, grid, Πᴸ)
        γRᵐᴸ_x = ℑxᶠᵃᵃ(i, j, k, grid, γRᵐᴸ)
        ∂x_ρθ′ = ∂xᶠᶜᶜ(i, j, k, grid, ρθ′)
        ∂x_p′  = ∂xᶠᶜᶜ(i, j, k, grid, p′)
        ∂x_p   = ∂x_p′ + γRᵐᴸ_x * Πᴸ_x * ∂x_ρθ′

        Πᴸ_y   = ℑyᵃᶠᵃ(i, j, k, grid, Πᴸ)
        γRᵐᴸ_y = ℑyᵃᶠᵃ(i, j, k, grid, γRᵐᴸ)
        ∂y_ρθ′ = ∂yᶜᶠᶜ(i, j, k, grid, ρθ′)
        ∂y_p′  = ∂yᶜᶠᶜ(i, j, k, grid, p′)
        ∂y_p   = ∂y_p′ + γRᵐᴸ_y * Πᴸ_y * ∂y_ρθ′

        not_bdy_x = !on_x_boundary(i, j, k, grid)
        not_bdy_y = !on_y_boundary(i, j, k, grid)

        ρu′[i, j, k] += Δτ * (Gⁿρu[i, j, k] - ∂x_p) * not_bdy_x
        ρv′[i, j, k] += Δτ * (Gⁿρv[i, j, k] - ∂y_p) * not_bdy_y
    end
end

# Boundary-detection helpers — return false on Periodic / Flat, true at
# the Bounded face indices where velocity must vanish.
@inline on_x_boundary(i, j, k, grid) = false
@inline on_y_boundary(i, j, k, grid) = false

const BX_grid = AbstractUnderlyingGrid{FT, Bounded}                                  where FT
const BY_grid = AbstractUnderlyingGrid{FT, <:Any, Bounded}                           where FT

@inline on_x_boundary(i, j, k, grid::BX_grid) = (i == 1) | (i == grid.Nx + 1)
@inline on_y_boundary(i, j, k, grid::BY_grid) = (j == 1) | (j == grid.Ny + 1)

# Build per-column predictors `ρ′★`, `ρθ′★` (cell centers) AND
# the explicit RHS for the tridiagonal `(ρw)′ᵐ⁺` solve at z-faces.
#
# Off-centered Crank–Nicolson with new-side weight ω = forward_weight
# and old-side weight 1−ω. The predictor uses δτˢ⁻ = (1−ω)Δτ on the
# old-step vertical-flux contribution (ω-weighted CN of ∇·m); the
# vertical RHS combines old and pred contributions with their matching
# weights δτˢ⁻ and δτᵐ⁺ respectively. See derivation in
# `validation/substepping/derivation_phase1.md` (eqns. 5, 7, 15).
@kernel function _build_predictors_and_vertical_rhs!(ρw′_rhs,
                                                     ρ′★, ρθ′★,
                                                     ρ′, ρθ′, ρw′, ρu′, ρv′,
                                                     grid, Δτ, δτᵐ⁺, δτˢ⁻,
                                                     Gˢρ, Gˢρθ, Gˢρw,
                                                     θᴸ, Πᴸ,
                                                     γRᵐᴸ, g, dˢ⁻)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Cell-centred predictors `ρ′★`, `ρθ′★`.
        for k in 1:Nz
            V = Vᶜᶜᶜ(i, j, k, grid)

            ∇ʰ_M  = div_xyᶜᶜᶜ(i, j, k, grid, ρu′, ρv′)
            ∇ʰ_θM = (δxᶜᵃᵃ(i, j, k, grid, _theta_face_x_flux, θᴸ, ρu′) +
                     δyᵃᶜᵃ(i, j, k, grid, _theta_face_y_flux, θᴸ, ρv′)) / V

            ρ′★[i, j, k]  = ρ′[i, j, k] +
                                Δτ * (Gˢρ[i, j, k] - ∇ʰ_M) -
                                δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, ρw′)

            ρθ′★[i, j, k] = ρθ′[i, j, k] +
                                Δτ * (Gˢρθ[i, j, k] - ∇ʰ_θM) -
                                δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, _theta_face_z_flux, θᴸ, ρw′)
        end

        # Face-level RHS for `(ρw)′ᵐ⁺` tridiag — split weights for the
        # predictor and old-step contributions per derivation (15).
        # `dˢ⁻ = (1−ω) α Δz²` adds the explicit half of the implicit
        # vertical damping (zero when damping is off or vertical_implicit=false).
        for k in 2:Nz
            Δzᶠ   = Δzᶜᶜᶠ(i, j, k, grid)
            Πᶜᶜᶠ    = ℑzᵃᵃᶠ(i, j, k, grid, Πᴸ)
            γRᵐᴸᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, γRᵐᴸ)

            ∂z_ρθ′★ = ρθ′★[i, j, k] - ρθ′★[i, j, k - 1]
            ∂z_ρθ′ˢ⁻ = ρθ′[i, j, k] - ρθ′[i, j, k - 1]

            sound_force = γRᵐᴸᶜᶜᶠ * Πᶜᶜᶠ / Δzᶠ * (δτˢ⁻ * ∂z_ρθ′ˢ⁻ + δτᵐ⁺ * ∂z_ρθ′★)

            ρ′ᶜᶜᶠ★  = ℑzᵃᵃᶠ(i, j, k, grid, ρ′★)
            ρ′ᶜᶜᶠˢ⁻ = ℑzᵃᵃᶠ(i, j, k, grid, ρ′)
            buoy_force = g * (δτˢ⁻ * ρ′ᶜᶜᶠˢ⁻ + δτᵐ⁺ * ρ′ᶜᶜᶠ★)

            # Explicit (old-step) half of the vertical damping
            # `(1−ω) α Δz² ∂z²(ρw)′ˢ⁻`, evaluated at face k. The face-coupling
            # stencil matches the implicit half folded into the tridiag in
            # `get_coefficient`.
            ∂z²_ρw′ˢ⁻  = ∂zᶜᶜᶠ(i, j, k, grid, ∂zᶜᶜᶜ, ρw′)
            damp_force = - dˢ⁻ * ∂z²_ρw′ˢ⁻

            ρw′_rhs[i, j, k] = ρw′[i, j, k] + Δτ * Gˢρw[i, j, k] -
                               sound_force - buoy_force - damp_force
        end

        # Boundary-row RHS values: f[1] = 0 (matches diagonal b[1] = 1 → (ρw)′[1] = 0).
        ρw′_rhs[i, j, 1] = 0
        # Top face (Nz+1) lives outside the solver; impenetrability w(top) = 0.
        ρw′_rhs[i, j, Nz + 1] = 0
    end
end

# θᴸ · (ρu)′ at an x-face. Used in the area-weighted horizontal
# divergence of the perturbation θ-flux.
@inline _theta_face_x_flux(i, j, k, grid, θᴸ, ρu′) =
    @inbounds Axᶠᶜᶜ(i, j, k, grid) * ℑxᶠᵃᵃ(i, j, k, grid, θᴸ) * ρu′[i, j, k]

@inline _theta_face_y_flux(i, j, k, grid, θᴸ, ρv′) =
    @inbounds Ayᶜᶠᶜ(i, j, k, grid) * ℑyᵃᶠᵃ(i, j, k, grid, θᴸ) * ρv′[i, j, k]

# θᴸ · (ρw)′ at a z-face. Used in the vertical part of the perturbation
# θ-flux divergence; passed to `∂zᶜᶜᶜ` so the divergence is computed at
# cell centers from the face-located product.
@inline _theta_face_z_flux(i, j, k, grid, θᴸ, ρw′) = @inbounds ℑbzᵃᵃᶠ(i, j, k, grid, θᴸ) * ρw′[i, j, k]
@inline ℑb_wθ(i, j, k, grid, w, θ) = @inbounds w[i, j, k] * ℑbzᵃᵃᶠ(i, j, k, grid, θ)

# Post-solve recovery: substitute the tridiag-solved `(ρw)′ᵐ⁺` back
# into the `ρ′★`, `ρθ′★` predictors to get `ρ′ᵐ⁺`, `ρθ′ᵐ⁺`
# (the IMPLICIT half of CN).
#
#   ρ′_n(k)    = ρ′★(k)  - (δτᵐ⁺ / Δz_c(k)) · ((ρw)′_n(k+1) - (ρw)′_n(k))
#   (ρθ)′_n(k) = ρθ′★(k) - (δτᵐ⁺ / Δz_c(k)) · (θᴸ_face(k+1) (ρw)′_n(k+1)
#                                                    - θᴸ_face(k)   (ρw)′_n(k))
@kernel function _post_solve_recovery!(ρ′, ρθ′, ρw′, ρ′★, ρθ′★, grid, δτᵐ⁺, θᴸ)
    i, j, k = @index(Global, NTuple)
    ρ′[i, j, k] = ρ′★[i, j, k] - δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, ρw′)
    ρθ′[i, j, k] = ρθ′★[i, j, k] - δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, ℑb_wθ, ρw′, θᴸ)
end

#####
##### Section 9 — Damping
#####

# No-op default
@inline apply_divergence_damping!(::NoDivergenceDamping, args...) = nothing

# Implicit-vertical-damping prefactors threaded into the column tridiag and
# its RHS. Returns `(dᵐ⁺, dˢ⁻) = (ω, 1−ω) · α · Δz²` for
# `ThermalDivergenceDamping` with `damp_vertical = true`, and `(0, 0)` for
# `NoDivergenceDamping` or when the user opts out via `damp_vertical = false`
# — which makes the tridiag and predictor-RHS additions vanish, recovering
# the pure off-centered CN acoustic system. In the latter case the off-
# centering itself supplies the vertical damping (Klemp et al. 2018 eq. 32).
@inline _implicit_damping_factors(::AcousticDampingStrategy, ω, one_minus_ω, grid, FT) =
    (zero(FT), zero(FT))

@inline function _implicit_damping_factors(damping::ThermalDivergenceDamping, ω, one_minus_ω, grid, FT)
    damping.damp_vertical || return (zero(FT), zero(FT))
    α    = convert(FT, damping.coefficient)
    Δz   = convert(FT, minimum_zspacing(grid))
    base = α * Δz^2
    return (convert(FT, ω) * base, convert(FT, one_minus_ω) * base)
end

# `HyperdiffusiveDivergenceDamping` reuses the same vertical-tridiag
# Laplacian as the 2nd-order Klemp form when `damp_vertical = true`.
@inline function _implicit_damping_factors(damping::HyperdiffusiveDivergenceDamping, ω, one_minus_ω, grid, FT)
    damping.damp_vertical || return (zero(FT), zero(FT))
    α    = convert(FT, damping.coefficient)
    Δz   = convert(FT, minimum_zspacing(grid))
    base = α * Δz^2
    return (convert(FT, ω) * base, convert(FT, one_minus_ω) * base)
end

# Klemp, Skamarock & Ha (2018) 3-D acoustic divergence damping (MPAS form).
# In the linearized acoustic mode,
#   (ρθ)′ − (ρθ)′ˢ⁻ ≈ −Δτ · θᴸ · ∇·((ρu)′, (ρv)′, (ρw)′)
# so D ≡ ((ρθ)′ − (ρθ)′ˢ⁻) / θᴸ is a discrete proxy for −Δτ · ∇·(ρu)′.
# The per-substep momentum correction is
#   Δ(ρu)′ = −γ · ∂x D , Δ(ρv)′ = −γ · ∂y D , Δ(ρw)′ = −γ_z · ∂z D
# with a single isotropic horizontal diffusivity (mirroring MPAS's
# `coef_divdamp = 2·smdiv·config_len_disp/Δτ`):
#   γ = α · d² / Δτ ,    d² ≡ Δx · Δy
#   γ_z = α · Δz² / Δτ   (folded into the column tridiag)
# `α` is the dimensionless Klemp 2018 coefficient (`config_smdiv` in MPAS,
# default 0.1). Linear stability of the explicit forward-Euler horizontal
# step gives `A(k) = 1 − 4α · Σᵢ sin²(kᵢ Δxᵢ/2)`; worst case (2-D Nyquist)
# is `8α ≤ 2 → α ≤ 0.25`; we default to 0.1 for margin. The vertical
# component is essential — without it the rest atmosphere amplifies at
# (Δt = 20 s, ω = 0.55) because the column tridiag's buoyancy off-diagonals
# are anti-symmetric.
function apply_divergence_damping!(damping::ThermalDivergenceDamping, substepper, grid, Δτ, thermodynamic_constants)
    FT    = eltype(grid)
    arch  = architecture(grid)
    α     = convert(FT, damping.coefficient)
    Δτ_FT = convert(FT, Δτ)

    TX, TY, _ = topology(grid)
    Δx = TX === Flat ? zero(FT) : convert(FT, minimum_xspacing(grid))
    Δy = TY === Flat ? zero(FT) : convert(FT, minimum_yspacing(grid))

    # Single isotropic horizontal diffusivity, MPAS-style. Vertical part
    # is folded into the column tridiag via `_implicit_damping_factors`.
    d² = damping.length_scale === nothing ? Δx * Δy : convert(FT, damping.length_scale)^2
    γ  = (TX === Flat && TY === Flat) ? zero(FT) : α * d² / Δτ_FT

    launch!(arch, grid, :xyz, _thermal_divergence_damping!,
            substepper.momentum_perturbation_u,
            substepper.momentum_perturbation_v,
            substepper.density_potential_temperature_perturbation,
            substepper.previous_density_potential_temperature_perturbation,
            substepper.linearization_potential_temperature,
            grid, γ)

    return nothing
end

# Hyperdiffusive (4th-order) variant. Same isotropic-horizontal scaling as
# the 2nd-order form but with one extra factor of d²:
#   γ = α · d⁴ / Δτ ,    d² ≡ Δx · Δy
# Stability bound is tighter (~`α ≤ 2/π⁴ ≈ 0.02`) since the explicit forward-
# Euler bound on `α k⁴` Nyquist is `α · π⁴ · 2 ≤ 2`.
function apply_divergence_damping!(damping::HyperdiffusiveDivergenceDamping, substepper, grid, Δτ, thermodynamic_constants)                                  
    FT    = eltype(grid)
    arch  = architecture(grid)
    α     = convert(FT, damping.coefficient)
    Δτ_FT = convert(FT, Δτ)

    TX, TY, _ = topology(grid)
    Δx = TX === Flat ? zero(FT) : convert(FT, minimum_xspacing(grid))
    Δy = TY === Flat ? zero(FT) : convert(FT, minimum_yspacing(grid))

    d² = damping.length_scale === nothing ? Δx * Δy : convert(FT, damping.length_scale)^2
    γ  = (TX === Flat && TY === Flat) ? zero(FT) : α * d²^2 / Δτ_FT

    launch!(arch, grid, :xyz, _hyperdiffusive_divergence_damping!,
            substepper.momentum_perturbation_u,
            substepper.momentum_perturbation_v,
            substepper.density_potential_temperature_perturbation,
            substepper.previous_density_potential_temperature_perturbation,
            substepper.linearization_potential_temperature,
            grid, γ)

    return nothing
end

@inline dρθ′_over_θ(i, j, k, grid, ρθ′, ρθ′ˢ⁻, θᴸ) =
    @inbounds (ρθ′[i, j, k] - ρθ′ˢ⁻[i, j, k]) / θᴸ[i, j, k]

@inline dρθ′(i, j, k, grid, ρθ′, ρθ′ˢ⁻) = @inbounds ρθ′[i, j, k] - ρθ′ˢ⁻[i, j, k]


# Horizontal divergence damping in the form of Klemp, Skamarock & Ha (2018)
# eq. (36): per-substep momentum correction is the gradient of the (ρθ)′
# tendency, divided by θᴸ at the face,
#   Δ(ρu)′ = −γ · ∂x[(ρθ)′ − (ρθ)′ˢ⁻] / ℑxᶠᵃᵃ(θᴸ)
#   Δ(ρv)′ = −γ · ∂y[(ρθ)′ − (ρθ)′ˢ⁻] / ℑyᵃᶠᵃ(θᴸ)
# The vertical component lives in the column tridiag (it's a Laplacian on
# (ρw)′ folded into the implicit acoustic solve), not here.
@kernel function _thermal_divergence_damping!(ρu′, ρv′, ρθ′, ρθ′ˢ⁻, θᴸ, grid, γ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂x_div = ∂xᶠᶜᶜ(i, j, k, grid, dρθ′, ρθ′, ρθ′ˢ⁻)
        θᴸᶠᶜᶜ  = ℑxᶠᵃᵃ(i, j, k, grid, θᴸ)
        ρu′[i, j, k] -= γ * ∂x_div / θᴸᶠᶜᶜ * !on_x_boundary(i, j, k, grid)

        ∂y_div = ∂yᶜᶠᶜ(i, j, k, grid, dρθ′, ρθ′, ρθ′ˢ⁻)
        θᴸᶜᶠᶜ  = ℑyᵃᶠᵃ(i, j, k, grid, θᴸ)
        ρv′[i, j, k] -= γ * ∂y_div / θᴸᶜᶠᶜ * !on_y_boundary(i, j, k, grid)
    end
end

# Hyperdiffusive (4th-order) horizontal divergence damping. Same structure
# as the 2nd-order Klemp form above, but the proxy is the *horizontal
# Laplacian* of the (ρθ)′ tendency:
#   Δ(ρu)′ = +γ · ∂x[∇_h²((ρθ)′ − (ρθ)′ˢ⁻)] / ℑxᶠᵃᵃ(θᴸ)
#   Δ(ρv)′ = +γ · ∂y[∇_h²((ρθ)′ − (ρθ)′ˢ⁻)] / ℑyᵃᶠᵃ(θᴸ)
# Note the sign: ∇_h² introduces a `−k²` for plane waves, flipping the
# overall sign relative to the 2nd-order Klemp form (which has `−γ ∂x[…]`).
# With the `+γ` here, the spectral form `Δ(ρu)′ ∝ −γ k² (k_x² + k_y²) (ρu)′`
# is *negative-definite* — i.e. damping. Damping rate ∝ k⁴ instead of k²,
# so grid-scale modes are hit much harder than resolved scales.
@kernel function _hyperdiffusive_divergence_damping!(ρu′, ρv′, ρθ′, ρθ′ˢ⁻, θᴸ, grid, γ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂x_lap = ∂xᶠᶜᶜ(i, j, k, grid, ∇²h_dρθ′, ρθ′, ρθ′ˢ⁻)
        θᴸᶠᶜᶜ  = ℑxᶠᵃᵃ(i, j, k, grid, θᴸ)
        ρu′[i, j, k] += γ * ∂x_lap / θᴸᶠᶜᶜ * !on_x_boundary(i, j, k, grid)

        ∂y_lap = ∂yᶜᶠᶜ(i, j, k, grid, ∇²h_dρθ′, ρθ′, ρθ′ˢ⁻)
        θᴸᶜᶠᶜ  = ℑyᵃᶠᵃ(i, j, k, grid, θᴸ)
        ρv′[i, j, k] += γ * ∂y_lap / θᴸᶜᶠᶜ * !on_y_boundary(i, j, k, grid)
    end
end

@inline ∇²h_dρθ′(i, j, k, grid, ρθ′, ρθ′ˢ⁻) =
    ∇²hᶜᶜᶜ(i, j, k, grid, ρθ′) - ∇²hᶜᶜᶜ(i, j, k, grid, ρθ′ˢ⁻)

# Direct divergence damping (Skamarock & Klemp 1992 form, no proxy):
#
#   D = ∂x(ρu)′ + ∂y(ρv)′                  (cell centers, one kernel pass)
#   Δ(ρu)′ = +γ_m · ∂xᶠᶜᶜ(D)               (face centers, second kernel pass)
#   Δ(ρv)′ = +γ_m · ∂yᶜᶠᶜ(D)
#   Δ(ρθ)′ = +γ_θ · ∇²ʰ((ρθ)′)             (optional, third kernel pass)
#
# Two-pass structure is required because we read (ρu)′ at adjacent cells to
# compute D, then write back to (ρu)′ — same for (ρv)′. Computing D into a
# workspace eliminates the read/write hazard.
function apply_divergence_damping!(damping::DivergenceDamping, substepper, grid, Δτ,
                                   thermodynamic_constants)
    FT    = eltype(grid)
    arch  = architecture(grid)
    α_m   = convert(FT, damping.momentum_coefficient)
    α_θ   = convert(FT, damping.rhotheta_coefficient)

    TX, TY, _ = topology(grid)
    Δx = TX === Flat ? zero(FT) : convert(FT, minimum_xspacing(grid))
    Δy = TY === Flat ? zero(FT) : convert(FT, minimum_yspacing(grid))
    d² = Δx * Δy

    # γ has units m² (per-substep diffusion length squared, NOT a diffusivity).
    # K18's `γ = α·d²/Δτ` only has /Δτ because its proxy `(ρθ)'-(ρθ)'_old ≈
    # -Δτ·θ̄·∇·m` already carries an extra Δτ that cancels out. With direct
    # divergence the operator is `α·d² · ∂x(∇·m)` per substep — no /Δτ.
    γ_m = (TX === Flat && TY === Flat) ? zero(FT) : α_m * d²
    γ_θ = (TX === Flat && TY === Flat) ? zero(FT) : α_θ * d²

    # Pass 1: compute horizontal momentum divergence into workspace.
    launch!(arch, grid, :xyz, _compute_horizontal_momentum_divergence!,
            substepper.horizontal_momentum_divergence,
            substepper.momentum_perturbation_u,
            substepper.momentum_perturbation_v,
            grid)
    fill_halo_regions!(substepper.horizontal_momentum_divergence)

    # Pass 2: apply ∂h(D) damping to horizontal momentum.
    launch!(arch, grid, :xyz, _apply_direct_divergence_damping_to_momentum!,
            substepper.momentum_perturbation_u,
            substepper.momentum_perturbation_v,
            substepper.horizontal_momentum_divergence,
            grid, γ_m)

    # Pass 3 (optional): smooth (ρθ)′ via ∇²ʰ((ρθ)′). S&K1992 thermodynamic
    # damping: damps the PE half-cycle of the acoustic mode that the
    # momentum-only damping leaves alone.
    if α_θ != 0
        launch!(arch, grid, :xyz, _apply_rhotheta_smoothing!,
                substepper.density_potential_temperature_perturbation,
                grid, γ_θ)
    end
    return nothing
end

@kernel function _compute_horizontal_momentum_divergence!(D, ρu′, ρv′, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds D[i, j, k] = div_xyᶜᶜᶜ(i, j, k, grid, ρu′, ρv′)
end

@kernel function _apply_direct_divergence_damping_to_momentum!(ρu′, ρv′, D, grid, γ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρu′[i, j, k] += γ * ∂xᶠᶜᶜ(i, j, k, grid, D) * !on_x_boundary(i, j, k, grid)
        ρv′[i, j, k] += γ * ∂yᶜᶠᶜ(i, j, k, grid, D) * !on_y_boundary(i, j, k, grid)
    end
end

@kernel function _apply_rhotheta_smoothing!(ρθ′, grid, γ)
    i, j, k = @index(Global, NTuple)
    @inbounds ρθ′[i, j, k] += γ * ∇²hᶜᶜᶜ(i, j, k, grid, ρθ′)
end

# Pressure-extrapolation damping (WRF/ERF). The post-substep momentum
# correction is a no-op; the damping enters by forward-biasing the
# (ρθ)′ used in the explicit horizontal PGF (`prepare_pgf_rhotheta!`).
@inline apply_divergence_damping!(::PressureExtrapolationDamping, substepper, grid, Δτ,
                                  thermodynamic_constants) = nothing

# Default: the PGF reads the un-biased (ρθ)′. Implemented as a copy so
# the explicit-horizontal-step kernel can always read from the same
# `pgf_*` field regardless of damping strategy.
@inline function prepare_pgf_rhotheta!(::AcousticDampingStrategy, substepper, grid, FT)
    parent(substepper.pgf_density_potential_temperature_perturbation) .=
        parent(substepper.density_potential_temperature_perturbation)
    return nothing
end

# WRF/ERF pre-substep PGF bias:
#   (ρθ)′_pgf = (ρθ)′ + α · ((ρθ)′ - (ρθ)′_lagged)
# `(ρθ)′_lagged` is the value at the END of the previous substep; zero
# at the first substep of an RK stage (set by `reset_perturbations!`).
function prepare_pgf_rhotheta!(damping::PressureExtrapolationDamping, substepper, grid, FT)
    α = convert(FT, damping.coefficient)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _bias_pgf_rhotheta!,
            substepper.pgf_density_potential_temperature_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.lagged_density_potential_temperature_perturbation,
            α)
    fill_halo_regions!(substepper.pgf_density_potential_temperature_perturbation)
    return nothing
end

@kernel function _bias_pgf_rhotheta!(ρθ_pgf, ρθ′, ρθ_lagged, α)
    i, j, k = @index(Global, NTuple)
    @inbounds ρθ_pgf[i, j, k] = ρθ′[i, j, k] + α * (ρθ′[i, j, k] - ρθ_lagged[i, j, k])
end

# Snapshot (ρθ)′ at the END of a substep so the next substep's
# `prepare_pgf_rhotheta!` sees it as the lagged value. No-op for
# damping strategies that don't use the lagged snapshot.
@inline update_lagged_rhotheta!(::AcousticDampingStrategy, substepper) = nothing

@inline function update_lagged_rhotheta!(::PressureExtrapolationDamping, substepper)
    parent(substepper.lagged_density_potential_temperature_perturbation) .=
        parent(substepper.density_potential_temperature_perturbation)
    return nothing
end

#####
##### Section 10 — Full-state recovery at stage end
#####

# After the substep loop completes for a stage, reconstruct the full
# prognostic state ρ, ρu, ρv, ρw, ρθ from the outer-step-start snapshot
# plus the accumulated perturbations:
#   ρᵐ⁺  = ρᴸ  + ρ′
#   ρθᵐ⁺ = ρθᴸ + (ρθ)′
#   ρuᵐ⁺ = ρuᴸ + (ρu)′, etc.
# Velocities are then diagnosed: u = ρu/ρ, etc.
@kernel function _recover_full_state!(ρ, ρθ, m, vel,
                                      ρ′, ρθ′, ρu′, ρv′, ρw′,
                                      ρᴸ, ρuᴸ, ρvᴸ, ρwᴸ, ρθᴸ,
                                      grid)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵐ⁺  = ρᴸ[i, j, k]  + ρ′[i, j, k]
        ρθᵐ⁺ = ρθᴸ[i, j, k] + ρθ′[i, j, k]
        ρuᵐ⁺ = ρuᴸ[i, j, k] + ρu′[i, j, k]
        ρvᵐ⁺ = ρvᴸ[i, j, k] + ρv′[i, j, k]
        ρwᵐ⁺ = ρwᴸ[i, j, k] + ρw′[i, j, k]

        ρ[i, j, k]  = ρᵐ⁺
        ρθ[i, j, k] = ρθᵐ⁺

        m.ρu[i, j, k] = ρuᵐ⁺
        m.ρv[i, j, k] = ρvᵐ⁺
        m.ρw[i, j, k] = ρwᵐ⁺

        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρ̂ᶠᶜᶜ = ifelse(ρᶠᶜᶜ == 0, one(ρᶠᶜᶜ), ρᶠᶜᶜ)
        ρ̂ᶜᶠᶜ = ifelse(ρᶜᶠᶜ == 0, one(ρᶜᶠᶜ), ρᶜᶠᶜ)
        ρ̂ᶜᶜᶠ = ifelse(ρᶜᶜᶠ == 0, one(ρᶜᶜᶠ), ρᶜᶜᶠ)

        vel.u[i, j, k] = ρuᵐ⁺ / ρ̂ᶠᶜᶜ * !on_x_boundary(i, j, k, grid)
        vel.v[i, j, k] = ρvᵐ⁺ / ρ̂ᶜᶠᶜ * !on_y_boundary(i, j, k, grid)
        vel.w[i, j, k] = ρwᵐ⁺ / ρ̂ᶜᶜᶠ * (k > 1)
    end
end

#####
##### Section 11 — Substep loop driver
#####

"""
$(TYPEDSIGNATURES)

Execute one Wicker–Skamarock RK3 stage of the linearized acoustic
substep loop. Number and size of substeps in this stage depend on
`substepper.substep_distribution`.
"""
function acoustic_rk3_substep_loop!(model, substepper, Δt, β_stage, Uᴸ)
    grid = model.grid
    arch = architecture(grid)

    g   = convert(eltype(grid), model.thermodynamic_constants.gravitational_acceleration)
    FT  = eltype(grid)

    ω = FT(substepper.forward_weight)            # CN weight on the new side
    one_minus_ω = FT(1) - ω                       # CN weight on the old side

    # Compute substep count and size for this stage. WS-RK3 stage weights
    # are β = (1/3, 1/2, 1); for ProportionalSubsteps to give integer
    # substep counts at every stage, N must be a multiple of LCM(3, 2) = 6.
    # Floor of 6 ensures sane behavior even for very small Δt where the
    # acoustic-CFL substep count would round to 0 or 1.
    N_raw = acoustic_substeps(substepper.substeps, grid, Δt, model.thermodynamic_constants)
    N = max(6, 6 * cld(N_raw, 6))
    Nτ, Δτ = _stage_substep_count_and_size(substepper.substep_distribution, β_stage, Δt, N)

    # Snapshot Gⁿ.ρw → substepper's slow vertical-momentum tendency.
    # (Pure copy at the moment; assemble_slow_vertical_momentum_tendency!
    # is also called here in case any future linearized PGF/buoyancy
    # contribution needs to be added.)
    assemble_slow_vertical_momentum_tendency!(substepper, model)

    # Reset perturbations to zero at stage start
    reset_perturbations!(substepper, grid, arch)

    Gⁿ = model.timestepper.Gⁿ
    χ_name = thermodynamic_density_name(model.formulation)
    Gˢρθ = getproperty(Gⁿ, χ_name)

    # Substep loop
    for substep in 1:Nτ
        # Step A.0: build the (ρθ)′ used in the explicit horizontal PGF.
        # For `PressureExtrapolationDamping` (WRF/ERF form), this is the
        # forward-biased (ρθ)′ = (ρθ)′ + α·((ρθ)′ - (ρθ)′_lagged); for all
        # other damping strategies it's just a copy of (ρθ)′.
        prepare_pgf_rhotheta!(substepper.damping, substepper, grid, FT)

        # Step A: explicit horizontal forward of (ρu)′, (ρv)′ using the
        # PGF (ρθ)′ (biased or not).
        launch!(arch, grid, :xyz, _explicit_horizontal_step!,
                substepper.momentum_perturbation_u,
                substepper.momentum_perturbation_v,
                grid, FT(Δτ),
                substepper.pgf_density_potential_temperature_perturbation,
                substepper.linearization_exner,
                substepper.pressure_perturbation,
                Gⁿ.ρu, Gⁿ.ρv, substepper.linearization_gamma_R_mixture)

        fill_halo_regions!(substepper.momentum_perturbation_u)
        fill_halo_regions!(substepper.momentum_perturbation_v)

        # Save (ρθ)′ before the column kernel for damping use
        parent(substepper.previous_density_potential_temperature_perturbation) .=
            parent(substepper.density_potential_temperature_perturbation)

        # CN time-step weights for this substep. δτᵐ⁺ = ω·Δτ is the
        # new-side weight (used by the matrix and the post-solve);
        # δτˢ⁻ = (1−ω)·Δτ is the old-side weight (used by the
        # predictor's old-flux contribution and the old part of the
        # vertical RHS). See derivation_phase1.md eqns. (5), (7), (15).
        δτᵐ⁺ = ω * FT(Δτ)
        δτˢ⁻ = one_minus_ω * FT(Δτ)

        # Implicit-vertical-damping prefactors. When the damping strategy
        # is `ThermalDivergenceDamping(vertical_implicit=true)`, the
        # vertical part of the divergence damping is folded into the
        # tridiag with `dᵐ⁺ = ω·α·Δz²` on the LHS and
        # `dˢ⁻ = (1−ω)·α·Δz²` on the predictor RHS. Both reduce to
        # zero for `NoDivergenceDamping` or when the user opts out via
        # `vertical_implicit=false`.
        dᵐ⁺, dˢ⁻ = _implicit_damping_factors(substepper.damping, ω, one_minus_ω, grid, FT)

        # Step B: build predictors `ρ′★`, `ρθ′★` and the tridiag RHS for (ρw)′ᵐ⁺
        launch!(arch, grid, :xy, _build_predictors_and_vertical_rhs!,
                substepper.momentum_perturbation_w,
                substepper.density_predictor,
                substepper.density_potential_temperaturepredictor,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation_w,
                substepper.momentum_perturbation_u, substepper.momentum_perturbation_v,
                grid, FT(Δτ), δτᵐ⁺, δτˢ⁻,
                Gⁿ.ρ, Gˢρθ, substepper.slow_vertical_momentum_tendency,
                substepper.linearization_potential_temperature, substepper.linearization_exner,
                substepper.linearization_gamma_R_mixture, g, dˢ⁻)

        # Step C: implicit tridiag solve for (ρw)′ with implicit-half δτᵐ⁺
        # and (when active) implicit vertical damping prefactor `dᵐ⁺`.
        solve!(substepper.momentum_perturbation_w, substepper.vertical_solver,
               substepper.momentum_perturbation_w,
               substepper.linearization_exner, substepper.linearization_potential_temperature,
               substepper.linearization_gamma_R_mixture, g, δτᵐ⁺, dᵐ⁺)

        # Step D: post-solve recovery of ρ′, (ρθ)′ using new (ρw)′
        launch!(arch, grid, :xyz, _post_solve_recovery!,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation_w,
                substepper.density_predictor,
                substepper.density_potential_temperaturepredictor,
                grid, δτᵐ⁺,
                substepper.linearization_potential_temperature)

        fill_halo_regions!(substepper.density_perturbation)
        fill_halo_regions!(substepper.density_potential_temperature_perturbation)

        # Step E: optional Klemp 2018 post-substep damping (no-op for
        # `PressureExtrapolationDamping`, which damps via the PGF bias instead).
        apply_divergence_damping!(substepper.damping, substepper, grid, FT(Δτ),
                                  model.thermodynamic_constants)

        fill_halo_regions!(substepper.momentum_perturbation_u)
        fill_halo_regions!(substepper.momentum_perturbation_v)

        # Step F: snapshot end-of-substep (ρθ)′ for the WRF/ERF PGF bias of
        # the next substep. No-op when the damping doesn't use it.
        update_lagged_rhotheta!(substepper.damping, substepper)
    end

    # Stage-end: recover the full prognostic state for use in the next
    # stage's slow tendency evaluation. ρ′, (ρθ)′ are stage-local
    # perturbations; the recovery base is U⁰_outer (frozen at outer-step
    # start) so the WS-RK3 invariant U^(k) = U⁰_outer + β_k Δt G holds.
    χ_field = thermodynamic_density(model.formulation)
    launch!(arch, grid, :xyz, _recover_full_state!,
            model.dynamics.density, χ_field,
            model.momentum, model.velocities,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.momentum_perturbation_u,
            substepper.momentum_perturbation_v,
            substepper.momentum_perturbation_w,
            substepper.recovery_density,
            Uᴸ[2], Uᴸ[3], Uᴸ[4],
            substepper.recovery_density_potential_temperature,
            grid)

    return nothing
end
