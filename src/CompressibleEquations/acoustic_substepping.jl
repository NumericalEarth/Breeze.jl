#####
##### Acoustic substepping for CompressibleDynamics
#####
##### The substepper evolves linearized acoustic perturbations between WS-RK3
##### stages, with the linearization fixed at the **outer-step-start state**.
##### Prime notation denotes the perturbation about U⁰:
#####
#####   ρ′    = ρ   − ρ⁰
#####   (ρθ)′ = ρθ  − ρθ⁰
#####   (ρu)′ = ρu  − ρu⁰,  (ρv)′ = ρv − ρv⁰,  (ρw)′ = ρw − ρw⁰
#####
##### Background quantities θ⁰ = ρθ⁰/ρ⁰ and Π⁰ = (p⁰/pˢᵗ)^κ are computed
##### once per outer Δt from the snapshot U⁰ and reused across all RK stages.
#####
##### The linearized perturbation equations integrated by the substep loop:
#####
#####   ∂t ρ′    +     ∇·((ρu)′, (ρv)′, (ρw)′)       = Gˢρ
#####   ∂t (ρθ)′ +     ∇·(θ⁰ · ((ρu)′, (ρv)′, (ρw)′)) = Gˢρθ
#####   ∂t (ρu)′ + γRᵐ · Π⁰_x ·  ∂x((ρθ)′)           = Gˢρu
#####   ∂t (ρv)′ + γRᵐ · Π⁰_y ·  ∂y((ρθ)′)           = Gˢρv
#####   ∂t (ρw)′ + γRᵐ · Π⁰_z ·  ∂z((ρθ)′) + g · ρ′  = Gˢρw
#####
##### Time discretization: horizontal momentum updates are forward-Euler;
##### the vertical ((ρw)′, (ρθ)′, ρ′) coupling is solved implicitly with
##### an off-centered Crank-Nicolson scheme — `forward_weight = 0.5` is
##### classic centered CN (neutrally stable for the linearized inviscid
##### system), `forward_weight > 0.5` adds dissipation. The implicit step
##### reduces to a tridiagonal Schur system in (ρw)′ at z-faces.
#####
##### After each stage's substep loop, the full prognostic state is
##### recovered: ρ = ρ⁰ + ρ′, ρθ = ρθ⁰ + (ρθ)′, ρu = ρu⁰ + (ρu)′, etc.,
##### and velocities are diagnosed from momenta and density.
#####
##### Notation note (Standard S13): the symbols `σ`, `η` are RESERVED in
##### atmospheric science for vertical coordinates (sigma and hybrid).
##### This module uses prime notation for perturbations and the in-code
##### kernel arguments map as
#####   ρ′    ↔ kernel arg `ρp`     (struct field: density_perturbation)
#####   (ρθ)′ ↔ kernel arg `ρθp`    (density_potential_temperature_perturbation)
#####   (ρu)′ ↔ kernel arg `ρup`    (momentum_perturbation_u)
#####   (ρv)′ ↔ kernel arg `ρvp`    (momentum_perturbation_v)
#####   (ρw)′ ↔ kernel arg `ρwp`    (momentum_perturbation_w)
##### Predictors carry a `_pred` suffix: `ρp_pred`, `ρθp_pred`.
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

using Oceananigans.Grids: Periodic, Bounded, Flat,
                          AbstractUnderlyingGrid,
                          topology,
                          minimum_xspacing, minimum_yspacing

using Adapt: Adapt, adapt

#####
##### Section 1 — Substepper struct
#####

"""
$(TYPEDEF)

Storage and parameters for the split-explicit acoustic substepper.

The substepper evolves linearized acoustic perturbations between WS-RK3
stages, with the linearization fixed at the outer-step-start state ``U⁰``.
Background quantities ``ρ⁰, ρθ⁰, p⁰, Π⁰, θ⁰`` are snapshotted from ``U⁰``
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

- `outer_step_density`: ``ρ⁰`` snapshot.
- `outer_step_density_potential_temperature`: ``ρθ⁰`` snapshot.
- `outer_step_pressure`: ``p⁰`` diagnosed from the EoS at ``U⁰``.
- `outer_step_exner`: ``Π⁰ = (p⁰/pˢᵗ)^κ`` derived from `outer_step_pressure`.
- `outer_step_potential_temperature`: ``θ⁰ = ρθ⁰/ρ⁰`` for the perturbation
  temperature flux ``θ⁰ · μ``.

Perturbation prognostics (advanced inside the substep loop):

- `density_perturbation`: ``ρ′ = ρ − ρ⁰``.
- `density_potential_temperature_perturbation`: ``(ρθ)′ = ρθ − ρθ⁰``.
- `momentum_perturbation_u`, `_v`, `_w`: ``(ρu)′, (ρv)′, (ρw)′``.

Per-column scratch (column kernel only):

- `density_predictor`, `density_potential_temperature_predictor`: explicit
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
    # `_recover_full_state!` to compute U_new = U_recovery + (ρ′, (ρθ)′, (ρu)′, (ρv)′, (ρw)′).
    # Required by Wicker–Skamarock RK3 which integrates each stage from
    # the SAME outer-step-start U⁰.
    recovery_density :: CF
    recovery_density_potential_temperature :: CF

    # Linearization basic state — REFRESHED at each WS-RK3 stage from the
    # current model state ``U^{(k-1)}``. This keeps the linearized
    # operators (γRᵐ Π ∂(ρθ)′, g·ρ′, ∂x(p − p_ref)) consistent with the slow
    # tendency `Gⁿ` (also evaluated at U^{(k-1)}) and breaks the soft
    # outer-step CFL limit Δt ≲ Δx/cs.
    outer_step_density :: CF
    outer_step_density_potential_temperature :: CF
    outer_step_pressure :: CF
    outer_step_exner :: CF
    outer_step_potential_temperature :: CF

    # Moist basic state — snapshotted at outer-step start. Used by the
    # moist-aware linearization (Phase 3 of PRISTINE_SUBSTEPPER_PLAN.md
    # §A3/B1/B2) to evaluate ``Rᵐ⁰, γᵐ⁰, μᵥ⁰`` per cell. For dry runs the
    # three mass-fraction fields are zero; γᵐRᵐ⁰ collapses to γᵈRᵈ and
    # μᵥ⁰ collapses to 1 — the linearization reduces exactly to the dry case.
    outer_step_vapor_mass_fraction :: CF
    outer_step_liquid_mass_fraction :: CF
    outer_step_ice_mass_fraction :: CF

    # Derived from the moist mass fractions:
    # `outer_step_gamma_R_mixture[i,j,k] = γᵐ(i,j,k) · Rᵐ(i,j,k)` enters the
    # linearised PGF (`γᵈRᵈ → γᵐRᵐ⁰`) — Phase 2A of the moist substepper.
    # `outer_step_virtual_density_factor[i,j,k] = μᵥ⁰(i,j,k)` is precomputed
    # for use by future moist refinements but is *not* used in the buoyancy
    # term: the conservation-form momentum equation has `g·ρ` (total density),
    # so a virtual-density multiplier on `g·ρ` would be incorrect.
    outer_step_gamma_R_mixture :: CF
    outer_step_virtual_density_factor :: CF

    # Reference-subtracted pressure perturbation, p − p_ref (= p when no
    # reference). Refreshed each stage along with outer_step_pressure.
    pressure_imbalance :: CF

    density_perturbation :: CF
    density_potential_temperature_perturbation :: CF
    momentum_perturbation_w :: FF
    momentum_perturbation_u :: XF
    momentum_perturbation_v :: YF

    density_predictor :: CF
    density_potential_temperature_predictor :: CF
    previous_density_potential_temperature_perturbation :: CF

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
                       adapt(to, a.outer_step_density),
                       adapt(to, a.outer_step_density_potential_temperature),
                       adapt(to, a.outer_step_pressure),
                       adapt(to, a.outer_step_exner),
                       adapt(to, a.outer_step_potential_temperature),
                       adapt(to, a.outer_step_vapor_mass_fraction),
                       adapt(to, a.outer_step_liquid_mass_fraction),
                       adapt(to, a.outer_step_ice_mass_fraction),
                       adapt(to, a.outer_step_gamma_R_mixture),
                       adapt(to, a.outer_step_virtual_density_factor),
                       adapt(to, a.pressure_imbalance),
                       adapt(to, a.density_perturbation),
                       adapt(to, a.density_potential_temperature_perturbation),
                       adapt(to, a.momentum_perturbation_w),
                       adapt(to, a.momentum_perturbation_u),
                       adapt(to, a.momentum_perturbation_v),
                       adapt(to, a.density_predictor),
                       adapt(to, a.density_potential_temperature_predictor),
                       adapt(to, a.previous_density_potential_temperature_perturbation),
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
    outer_step_density                         = CenterField(grid)
    outer_step_density_potential_temperature   = CenterField(grid)
    outer_step_pressure                        = CenterField(grid)
    outer_step_exner                           = CenterField(grid)
    outer_step_potential_temperature           = CenterField(grid)

    # Moist basic state — snapshotted at outer-step start. Phase 3.
    outer_step_vapor_mass_fraction             = CenterField(grid)
    outer_step_liquid_mass_fraction            = CenterField(grid)
    outer_step_ice_mass_fraction               = CenterField(grid)
    outer_step_gamma_R_mixture                 = CenterField(grid)
    outer_step_virtual_density_factor          = CenterField(grid)

    # Reference-subtracted pressure perturbation (p − p_ref).
    pressure_imbalance                         = CenterField(grid)

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
    density_potential_temperature_predictor          = CenterField(grid)
    previous_density_potential_temperature_perturbation = CenterField(grid)

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
                              outer_step_density,
                              outer_step_density_potential_temperature,
                              outer_step_pressure,
                              outer_step_exner,
                              outer_step_potential_temperature,
                              outer_step_vapor_mass_fraction,
                              outer_step_liquid_mass_fraction,
                              outer_step_ice_mass_fraction,
                              outer_step_gamma_R_mixture,
                              outer_step_virtual_density_factor,
                              pressure_imbalance,
                              density_perturbation,
                              density_potential_temperature_perturbation,
                              momentum_perturbation_w,
                              momentum_perturbation_u,
                              momentum_perturbation_v,
                              density_predictor,
                              density_potential_temperature_predictor,
                              previous_density_potential_temperature_perturbation,
                              slow_vertical_momentum_tendency,
                              vertical_solver)
end

#####
##### Section 3 — Outer-step-start linearization
#####

"""
$(TYPEDSIGNATURES)

Snapshot the outer-step-start state ``U⁰`` and compute the background
quantities used by the substepper as the linearization point. Called
once per outer ``Δt`` by the WS-RK3 driver.

After this call:
  - `outer_step_density`               = ρ⁰
  - `outer_step_density_potential_temperature` = ρθ⁰
  - `outer_step_pressure`              = p⁰ (= `model.dynamics.pressure` at outer-step start)
  - `outer_step_exner`                 = Π⁰ = (p⁰/pˢᵗ)^κ
  - `outer_step_potential_temperature` = θ⁰ = ρθ⁰/ρ⁰
"""
function freeze_outer_step_state!(substepper::AcousticSubstepper, model)
    ρθ_field = thermodynamic_density(model.formulation)

    # Snapshot the RECOVERY BASE (frozen across all 3 stages). Used by
    # `_recover_full_state!` so that U_new = U⁰_outer + (ρ′, (ρθ)′, (ρu)′, (ρv)′, (ρw)′) — the
    # WS-RK3 invariant that each stage starts from U⁰.
    parent(substepper.recovery_density)                       .= parent(model.dynamics.density)
    parent(substepper.recovery_density_potential_temperature) .= parent(ρθ_field)

    # Then prime the linearization basic state to U⁰ for stage 1.
    refresh_linearization_basic_state!(substepper, model)

    fill_halo_regions!(substepper.recovery_density)
    fill_halo_regions!(substepper.recovery_density_potential_temperature)

    return nothing
end

# Refresh the linearization basic state (Π⁰, θ⁰, p⁰, p − p_ref, plus the
# matching ρ⁰, ρθ⁰ used by the slow vertical-momentum-tendency assembly)
# from the *current* state. Called once at outer-step start by
# `freeze_outer_step_state!` and again before each subsequent RK stage
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

    parent(substepper.outer_step_density)                       .= parent(model.dynamics.density)
    parent(substepper.outer_step_density_potential_temperature) .= parent(ρθ_field)
    parent(substepper.outer_step_pressure)                      .= parent(model.dynamics.pressure)

    # θ_lin = ρθ/ρ and Π_lin = (p/pˢᵗ)^κ from the current state.
    launch!(arch, grid, :xyz, _compute_outer_step_exner_and_theta!,
            substepper.outer_step_exner,
            substepper.outer_step_potential_temperature,
            substepper.outer_step_pressure,
            substepper.outer_step_density,
            substepper.outer_step_density_potential_temperature,
            pˢᵗ, κ)

    # Reference-subtracted pressure perturbation. For ExnerReferenceState
    # the reference depends only on z, so ∂x p_ref = ∂y p_ref = 0; the
    # horizontal force is then ∂x(p − p_ref) = ∂x p. Reference
    # subtraction in z guarantees a hydrostatic rest atmosphere has zero
    # vertical drive, free of FP-rounding noise.
    ref = model.dynamics.reference_state
    if ref isa Nothing
        parent(substepper.pressure_imbalance) .= parent(substepper.outer_step_pressure)
    else
        launch!(arch, grid, :xyz, _compute_pressure_imbalance!,
                substepper.pressure_imbalance,
                substepper.outer_step_pressure, ref.pressure)
    end

    # Moist basic state snapshot. Vapor is always present (qᵛ field exists
    # even for dry runs — zeroed out). Condensed phases are only snapshotted
    # when the microphysics scheme prognoses them; otherwise the substepper's
    # condensed-phase fields stay zero.
    snapshot_moist_basic_state!(substepper, model)

    # γᵐRᵐ⁰ and μᵥ⁰ derived from the snapshotted mass fractions. γᵐRᵐ⁰ enters
    # the substepper's PGF (γᵈRᵈ → γᵐRᵐ⁰); μᵥ⁰ is precomputed for diagnostics
    # / future formulation work but is NOT applied to the buoyancy term — see
    # the field's docstring above for why. For dry runs (qᵛ⁰ = qˡ⁰ = qⁱ⁰ = 0):
    # γᵐRᵐ⁰ → γᵈRᵈ bit-identically and μᵥ⁰ → 1.
    launch!(arch, grid, :xyz, _compute_outer_step_mixture_eos!,
            substepper.outer_step_gamma_R_mixture,
            substepper.outer_step_virtual_density_factor,
            substepper.outer_step_vapor_mass_fraction,
            substepper.outer_step_liquid_mass_fraction,
            substepper.outer_step_ice_mass_fraction,
            Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, cˡ, cⁱ)

    fill_halo_regions!(substepper.outer_step_density)
    fill_halo_regions!(substepper.outer_step_density_potential_temperature)
    fill_halo_regions!(substepper.outer_step_pressure)
    fill_halo_regions!(substepper.outer_step_exner)
    fill_halo_regions!(substepper.outer_step_potential_temperature)
    fill_halo_regions!(substepper.outer_step_vapor_mass_fraction)
    fill_halo_regions!(substepper.outer_step_liquid_mass_fraction)
    fill_halo_regions!(substepper.outer_step_ice_mass_fraction)
    fill_halo_regions!(substepper.outer_step_gamma_R_mixture)
    fill_halo_regions!(substepper.outer_step_virtual_density_factor)
    fill_halo_regions!(substepper.pressure_imbalance)

    return nothing
end

# Copy moisture mass fractions from the model state into the substepper's
# outer-step snapshot. Vapor (`qᵛ`) is always available — for dry runs the
# field exists but is identically zero. Condensed phases are picked up by
# field name from `model.microphysical_fields`:
#   - liquid: `:qˡ` (saturation-adjustment) or `:qᶜˡ` (non-equilibrium /
#     four-category bulk); zero if neither exists
#   - ice: `:qⁱ` or `:qᶜⁱ`; zero if neither exists
# These mass fractions feed `_compute_outer_step_mixture_eos!` to derive
# `outer_step_gamma_R_mixture` (= γᵐRᵐ⁰) and
# `outer_step_virtual_density_factor` (= μᵥ⁰). For dry runs all three are
# zero — γᵐRᵐ⁰ collapses to γᵈRᵈ and μᵥ⁰ to 1, so dry tests stay
# bit-identical.
function snapshot_moist_basic_state!(substepper::AcousticSubstepper, model)
    qᵛ = specific_prognostic_moisture(model)
    parent(substepper.outer_step_vapor_mass_fraction) .= parent(qᵛ)

    fields = model.microphysical_fields
    _copy_or_zero!(substepper.outer_step_liquid_mass_fraction, fields, (:qˡ, :qᶜˡ))
    _copy_or_zero!(substepper.outer_step_ice_mass_fraction,    fields, (:qⁱ, :qᶜⁱ))

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

@kernel function _compute_pressure_imbalance!(p_imb, p⁰, p_ref)
    i, j, k = @index(Global, NTuple)
    @inbounds p_imb[i, j, k] = p⁰[i, j, k] - p_ref[i, j, k]
end

@kernel function _compute_outer_step_exner_and_theta!(Π, θ, p, ρ, ρθ, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Π[i, j, k] = (p[i, j, k] / pˢᵗ)^κ
        ρ_safe = ifelse(ρ[i, j, k] == 0, one(eltype(ρ)), ρ[i, j, k])
        θ[i, j, k] = ρθ[i, j, k] / ρ_safe
    end
end

# Compute γᵐRᵐ⁰ and μᵥ⁰ per cell from the snapshotted moisture mass fractions.
#   Rᵐ  = qᵈ Rᵈ + qᵛ Rᵛ                         (mixture gas constant)
#   cᵖᵐ = qᵈ cᵖᵈ + qᵛ cᵖᵛ + qˡ cˡ + qⁱ cⁱ      (mixture heat capacity)
#   cᵛᵐ = cᵖᵐ − Rᵐ
#   γᵐ  = cᵖᵐ / cᵛᵐ
#   μᵥ⁰ = 1 + (Rᵛ/Rᵈ − 1) qᵛ − qˡ − qⁱ          (virtual-density factor)
# with qᵈ = 1 − qᵛ − qˡ − qⁱ. For dry inputs (qᵛ = qˡ = qⁱ = 0) these reduce
# to γᵈRᵈ and 1 exactly.
@kernel function _compute_outer_step_mixture_eos!(γRᵐ, μᵥ, qᵛ, qˡ, qⁱ,
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
been tried and *empirically makes the rest-atmosphere amplification
worse* (envelope at Δt=20s, ω=0.55 grows from 1.79 m/s → 41 m/s in
50 steps when refresh is enabled, even with the Phase 2 reference
fix). The mechanism is a separate residual feedback that Phase 4
of `validation/substepping/PRISTINE_SUBSTEPPER_PLAN.md` will
diagnose. Keeping refresh off is the lesser evil for now.
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

# Background θ⁰ at face k. Returns 0 at boundary faces so the kernel can
# call this unconditionally.
@inline function _theta_at_face(i, j, k, grid, θ⁰)
    Nz = size(grid, 3)
    in_interior = (k >= 2) & (k <= Nz)
    k_safe = ifelse(in_interior, k, 2)
    @inbounds val = (θ⁰[i, j, k_safe] + θ⁰[i, j, k_safe - 1]) / 2
    return ifelse(in_interior, val, zero(val))
end

# Off-centered CN tridiag derivation
# ----------------------------------
# At face k, the perturbation `(ρw)′` equation in CN form is
#
#   (ρw)′_n(k) = (ρw)′_o(k) + Δτ Gˢρw(k)
#                - Δτ × γRᵐ × Π⁰_face(k) × (ω_old ∂z (ρθ)′_o + ω_new ∂z (ρθ)′_n) / Δz_face(k)
#                - Δτ × g × (ω_old ρ′_face_o(k) + ω_new ρ′_face_n(k))
#
# with ω_new = (1+ε)/2, ω_old = (1-ε)/2 (ε=0 is centered CN).
#
# The post-solve substitution (matching the column kernel):
#   ρ′_n(k)    = ρp_pred(k)  - δτ_new × ((ρw)′_n(k+1) - (ρw)′_n(k)) / Δz_c(k)
#   (ρθ)′_n(k) = ρθp_pred(k) - δτ_new × (θ⁰_face(k+1) (ρw)′_n(k+1)
#                                        - θ⁰_face(k)   (ρw)′_n(k)) / Δz_c(k)
# where δτ_new = ω_new Δτ.
#
# Substituting yields the tridiagonal coefficients (ω = ω_new):
#
#   A[k,k+1] = -(ω Δτ)² × γRᵐ × Π⁰_face(k) × θ⁰_face(k+1) × rdz_c(k)   / Δz_face(k)
#              - (ω Δτ)² × g          × rdz_c(k)   / 2
#
#   A[k,k]   = 1 + (ω Δτ)² × γRᵐ × Π⁰_face(k) × θ⁰_face(k)   × (rdz_c(k) + rdz_c(k-1)) / Δz_face(k)
#                + (ω Δτ)² × g                              × (rdz_c(k) - rdz_c(k-1)) / 2
#
#   A[k,k-1] = -(ω Δτ)² × γRᵐ × Π⁰_face(k) × θ⁰_face(k-1) × rdz_c(k-1) / Δz_face(k)
#              + (ω Δτ)² × g                              × rdz_c(k-1) / 2
#
# `γᵐRᵐ⁰` is the cell-centered mixture coefficient `γᵐ Rᵐ` evaluated from
# the snapshotted moisture (`outer_step_gamma_R_mixture`). It is interpolated
# to z-faces inside the kernel. For dry runs (qᵛ = qˡ = qⁱ = 0) this collapses
# bit-identically to the dry constant `γᵈRᵈ`.

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagLower, p, ::ZDirection,
                                 Π⁰, θ⁰, γRᵐ⁰, g, δτ_new)
    k_face   = k + 1
    Δz_below = Δzᶜᶜᶜ(i, j, k_face - 1, grid)
    Δz_face  = Δzᶜᶜᶠ(i, j, k_face, grid)
    rdz_c    = 1 / Δz_below

    Π_face    = ℑzᵃᵃᶠ(i, j, k_face, grid, Π⁰)
    γRᵐ⁰_face = ℑzᵃᵃᶠ(i, j, k_face, grid, γRᵐ⁰)
    θ_below   = _theta_at_face(i, j, k_face - 1, grid, θ⁰)

    pgf_term  = - δτ_new^2 * γRᵐ⁰_face * Π_face * θ_below * rdz_c / Δz_face
    buoy_term = + δτ_new^2 * g                            * rdz_c / 2
    return pgf_term + buoy_term
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagDiagonal, p, ::ZDirection,
                                 Π⁰, θ⁰, γRᵐ⁰, g, δτ_new)
    # Bottom-boundary row: (ρw)′[1] = 0 by impenetrability. Trivial b = 1, RHS = 0.
    k == 1 && return one(eltype(γRᵐ⁰))

    k_face   = k
    Δz_face  = Δzᶜᶜᶠ(i, j, k_face, grid)
    Δz_above = Δzᶜᶜᶜ(i, j, k_face,     grid)
    Δz_below = Δzᶜᶜᶜ(i, j, k_face - 1, grid)
    rdz_above = 1 / Δz_above
    rdz_below = 1 / Δz_below

    Π_face    = ℑzᵃᵃᶠ(i, j, k_face, grid, Π⁰)
    γRᵐ⁰_face = ℑzᵃᵃᶠ(i, j, k_face, grid, γRᵐ⁰)
    θ_face    = _theta_at_face(i, j, k_face, grid, θ⁰)

    pgf_diag  = δτ_new^2 * γRᵐ⁰_face * Π_face * θ_face * (rdz_above + rdz_below) / Δz_face
    buoy_diag = δτ_new^2 * g                          * (rdz_above - rdz_below) / 2

    return 1 + pgf_diag + buoy_diag
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagUpper, p, ::ZDirection,
                                 Π⁰, θ⁰, γRᵐ⁰, g, δτ_new)
    # Bottom-boundary row: c[1] = 0 so back-substitution preserves (ρw)′[1] = 0.
    k == 1 && return zero(eltype(γRᵐ⁰))

    k_face   = k
    Δz_face  = Δzᶜᶜᶠ(i, j, k_face, grid)
    Δz_above = Δzᶜᶜᶜ(i, j, k_face, grid)
    rdz_c    = 1 / Δz_above

    Π_face    = ℑzᵃᵃᶠ(i, j, k_face, grid, Π⁰)
    γRᵐ⁰_face = ℑzᵃᵃᶠ(i, j, k_face, grid, γRᵐ⁰)
    θ_above   = _theta_at_face(i, j, k_face + 1, grid, θ⁰)

    pgf_term  = - δτ_new^2 * γRᵐ⁰_face * Π_face * θ_above * rdz_c / Δz_face
    buoy_term = - δτ_new^2 * g                            * rdz_c / 2
    return pgf_term + buoy_term
end

#####
##### Section 7 — Slow vertical-momentum tendency assembly
#####
##### The full vertical-momentum equation is
#####   ∂t (ρw) + ∇·(ρw u) + ∂z p + g ρ = 0
##### The dynamics kernel runs in `SlowTendencyMode` for SplitExplicit,
##### which zeroes the PGF and buoyancy in `Gⁿρw`. We reinstate the
##### **U⁰-state** PGF and buoyancy here so the slow ρw tendency has the
##### form
#####   Gˢρw = -∇·(ρw u)  -  ∂z p⁰  -  g · ρ⁰
##### and the per-substep linearized forces operate on the perturbations:
#####   ∂t (ρw)′ = Gˢρw - γRᵐ · Π⁰ · ∂z((ρθ)′)  -  g · ρ′
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
                substepper.outer_step_pressure,
                substepper.outer_step_density,
                grid, g)
    else
        launch!(arch, grid, :xyz, _assemble_slow_vertical_momentum_tendency!,
                substepper.slow_vertical_momentum_tendency,
                Gⁿρw,
                substepper.outer_step_pressure,
                substepper.outer_step_density,
                ref.pressure, ref.density,
                grid, g)
    end

    return nothing
end

# Slow-tendency assembly with reference state. Buoyancy uses TOTAL density
# `ρ⁰` (no virtual-density factor): in conservation-form momentum,
# `∂t(ρw) = -∂z p - g ρ`, where `ρ` is total mass density and includes all
# water species. The "virtual" temperature/density transforms only appear
# when one parameterises with *dry* density as the prognostic, which Breeze
# does not do.
@kernel function _assemble_slow_vertical_momentum_tendency!(Gˢρw, Gⁿρw, p⁰, ρ⁰, pᵣ, ρᵣ, grid, g)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        if k == 1
            # Bottom face: (ρw)′ = 0 by impenetrability; no slow force needed.
            Gˢρw[i, j, k] = zero(eltype(Gˢρw))
        else
            Δz_face = Δzᶜᶜᶠ(i, j, k, grid)
            # Reference-subtracted PGF and buoyancy: at U⁰ = reference state
            # both terms are exactly zero by construction of the reference.
            p′_above = p⁰[i, j, k]     - pᵣ[i, j, k]
            p′_below = p⁰[i, j, k - 1] - pᵣ[i, j, k - 1]
            ∂z_p′    = (p′_above - p′_below) / Δz_face

            g_ρp_above = g * (ρ⁰[i, j, k]     - ρᵣ[i, j, k])
            g_ρp_below = g * (ρ⁰[i, j, k - 1] - ρᵣ[i, j, k - 1])
            g_ρp_face  = (g_ρp_above + g_ρp_below) / 2

            Gˢρw[i, j, k] = Gⁿρw[i, j, k] - ∂z_p′ - g_ρp_face
        end
    end
end

@kernel function _assemble_slow_vertical_momentum_tendency_no_ref!(Gˢρw, Gⁿρw, p⁰, ρ⁰, grid, g)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        if k == 1
            Gˢρw[i, j, k] = zero(eltype(Gˢρw))
        else
            Δz_face = Δzᶜᶜᶠ(i, j, k, grid)
            ∂z_p⁰   = (p⁰[i, j, k] - p⁰[i, j, k - 1]) / Δz_face
            ρ⁰_face = (ρ⁰[i, j, k] + ρ⁰[i, j, k - 1]) / 2

            Gˢρw[i, j, k] = Gⁿρw[i, j, k] - ∂z_p⁰ - g * ρ⁰_face
        end
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
    fill!(parent(substepper.momentum_perturbation_u), 0)
    fill!(parent(substepper.momentum_perturbation_v), 0)
    fill!(parent(substepper.momentum_perturbation_w), 0)
    fill!(parent(substepper.density_predictor), 0)
    fill!(parent(substepper.density_potential_temperature_predictor), 0)
    return nothing
end

# Explicit forward step for horizontal momentum perturbations (ρu)′, (ρv)′.
#
# Linearized at U⁰, the full horizontal pressure gradient splits as
#   ∂x p_full = ∂x(p⁰ − p_ref) + γRᵐ Π⁰ · ∂x((ρθ)′)
# where the first piece is the FROZEN imbalance from the linearization
# point (carried by `pressure_imbalance`) and the second is the
# perturbation force. `Gⁿρu` from `SlowTendencyMode` carries advection
# only (PGF zeroed); we reinstate the frozen horizontal pressure
# perturbation here.
#
# (ρu)′^{τ+Δτ} = (ρu)′^τ + Δτ · (Gⁿρu − ∂x(p⁰−p_ref) − γᵐRᵐ⁰ Π⁰_x ∂x((ρθ)′))
# (ρv)′^{τ+Δτ} = (ρv)′^τ + Δτ · (Gⁿρv − ∂y(p⁰−p_ref) − γᵐRᵐ⁰ Π⁰_y ∂y((ρθ)′))
@kernel function _explicit_horizontal_step!(ρup, ρvp, grid, Δτ, ρθp, Π⁰, p_imb,
                                            Gⁿρu, Gⁿρv, γRᵐ⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Π⁰_x    = ℑxᶠᵃᵃ(i, j, k, grid, Π⁰)
        γRᵐ⁰_x  = ℑxᶠᵃᵃ(i, j, k, grid, γRᵐ⁰)
        ∂x_ρθp  = ∂xᶠᶜᶜ(i, j, k, grid, ρθp)
        ∂x_pp   = ∂xᶠᶜᶜ(i, j, k, grid, p_imb)
        ∂x_p    = ∂x_pp + γRᵐ⁰_x * Π⁰_x * ∂x_ρθp

        Π⁰_y    = ℑyᵃᶠᵃ(i, j, k, grid, Π⁰)
        γRᵐ⁰_y  = ℑyᵃᶠᵃ(i, j, k, grid, γRᵐ⁰)
        ∂y_ρθp  = ∂yᶜᶠᶜ(i, j, k, grid, ρθp)
        ∂y_pp   = ∂yᶜᶠᶜ(i, j, k, grid, p_imb)
        ∂y_p    = ∂y_pp + γRᵐ⁰_y * Π⁰_y * ∂y_ρθp

        not_bdy_x = !on_x_boundary(i, j, k, grid)
        not_bdy_y = !on_y_boundary(i, j, k, grid)

        ρup[i, j, k] += Δτ * (Gⁿρu[i, j, k] - ∂x_p) * not_bdy_x
        ρvp[i, j, k] += Δτ * (Gⁿρv[i, j, k] - ∂y_p) * not_bdy_y
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

# Build per-column predictors `ρp_pred`, `ρθp_pred` (cell centers) AND
# the explicit RHS for the tridiagonal `(ρw)′_new` solve at z-faces.
#
# Off-centered Crank–Nicolson with new-side weight ω = forward_weight
# and old-side weight 1−ω. The predictor uses δτ_old = (1−ω)Δτ on the
# old-step vertical-flux contribution (ω-weighted CN of ∇·m); the
# vertical RHS combines old and pred contributions with their matching
# weights δτ_old and δτ_new respectively. See derivation in
# `validation/substepping/derivation_phase1.md` (eqns. 5, 7, 15).
@kernel function _build_predictors_and_vertical_rhs!(ρwp_rhs,
                                                      ρp_pred, ρθp_pred,
                                                      ρp, ρθp, ρwp, ρup, ρvp,
                                                      grid, Δτ, δτ_new, δτ_old,
                                                      Gˢρ, Gˢρθ, Gˢρw,
                                                      θ⁰, Π⁰,
                                                      γRᵐ⁰, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Cell-centred predictors `ρp_pred`, `ρθp_pred`.
        for k in 1:Nz
            V = Vᶜᶜᶜ(i, j, k, grid)

            div_h_M  = div_xyᶜᶜᶜ(i, j, k, grid, ρup, ρvp)
            div_h_θM = (δxᶜᵃᵃ(i, j, k, grid, _theta_face_x_flux, θ⁰, ρup) +
                       δyᵃᶜᵃ(i, j, k, grid, _theta_face_y_flux, θ⁰, ρvp)) / V

            Δz_c     = Δzᶜᶜᶜ(i, j, k, grid)
            ρwp_above = ρwp[i, j, k + 1]
            ρwp_here  = ρwp[i, j, k]

            ρp_pred[i, j, k] = ρp[i, j, k] + Δτ * Gˢρ[i, j, k] - Δτ * div_h_M -
                               (δτ_old / Δz_c) * (ρwp_above - ρwp_here)

            θ_face_above = _theta_at_face(i, j, k + 1, grid, θ⁰)
            θ_face_here  = _theta_at_face(i, j, k,     grid, θ⁰)
            ρθp_pred[i, j, k] = ρθp[i, j, k] + Δτ * Gˢρθ[i, j, k] - Δτ * div_h_θM -
                                (δτ_old / Δz_c) *
                                (θ_face_above * ρwp_above - θ_face_here * ρwp_here)
        end

        # Face-level RHS for `(ρw)′_new` tridiag — split weights for the
        # predictor and old-step contributions per derivation (15).
        for k in 2:Nz
            Δz_face   = Δzᶜᶜᶠ(i, j, k, grid)
            Π_face    = ℑzᵃᵃᶠ(i, j, k, grid, Π⁰)
            γRᵐ⁰_face = ℑzᵃᵃᶠ(i, j, k, grid, γRᵐ⁰)

            ∂z_ρθp_pred = ρθp_pred[i, j, k] - ρθp_pred[i, j, k - 1]
            ∂z_ρθp_old  = ρθp[i, j, k]      - ρθp[i, j, k - 1]

            sound_force = γRᵐ⁰_face * Π_face / Δz_face *
                          (δτ_old * ∂z_ρθp_old + δτ_new * ∂z_ρθp_pred)

            ρp_face_pred = (ρp_pred[i, j, k] + ρp_pred[i, j, k - 1]) / 2
            ρp_face_old  = (ρp[i, j, k]      + ρp[i, j, k - 1])      / 2
            buoy_force   = g * (δτ_old * ρp_face_old + δτ_new * ρp_face_pred)

            ρwp_rhs[i, j, k] = ρwp[i, j, k] + Δτ * Gˢρw[i, j, k] -
                               sound_force - buoy_force
        end

        # Boundary-row RHS values: f[1] = 0 (matches diagonal b[1] = 1 → (ρw)′[1] = 0).
        ρwp_rhs[i, j, 1] = 0
        # Top face (Nz+1) lives outside the solver; impenetrability w(top) = 0.
        ρwp_rhs[i, j, Nz + 1] = 0
    end
end

# θ⁰ · (ρu)′ at an x-face. Used in the area-weighted horizontal
# divergence of the perturbation θ-flux.
@inline _theta_face_x_flux(i, j, k, grid, θ⁰, ρup) =
    Axᶠᶜᶜ(i, j, k, grid) * ℑxᶠᵃᵃ(i, j, k, grid, θ⁰) * ρup[i, j, k]

@inline _theta_face_y_flux(i, j, k, grid, θ⁰, ρvp) =
    Ayᶜᶠᶜ(i, j, k, grid) * ℑyᵃᶠᵃ(i, j, k, grid, θ⁰) * ρvp[i, j, k]

# Post-solve recovery: substitute the tridiag-solved `(ρw)′_new` back
# into the `ρp_pred`, `ρθp_pred` predictors to get `ρp_new`, `ρθp_new`
# (the IMPLICIT half of CN).
#
#   ρ′_n(k)    = ρp_pred(k)  - (δτ_new / Δz_c(k)) · ((ρw)′_n(k+1) - (ρw)′_n(k))
#   (ρθ)′_n(k) = ρθp_pred(k) - (δτ_new / Δz_c(k)) · (θ⁰_face(k+1) (ρw)′_n(k+1)
#                                                    - θ⁰_face(k)   (ρw)′_n(k))
@kernel function _post_solve_recovery!(ρp, ρθp, ρwp, ρp_pred, ρθp_pred,
                                        grid, δτ_new, θ⁰)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        for k in 1:Nz
            Δz_c = Δzᶜᶜᶜ(i, j, k, grid)
            θ_face_above = _theta_at_face(i, j, k + 1, grid, θ⁰)
            θ_face_here  = _theta_at_face(i, j, k,     grid, θ⁰)

            ρp[i, j, k] = ρp_pred[i, j, k] -
                          (δτ_new / Δz_c) * (ρwp[i, j, k + 1] - ρwp[i, j, k])
            ρθp[i, j, k] = ρθp_pred[i, j, k] -
                           (δτ_new / Δz_c) * (θ_face_above * ρwp[i, j, k + 1] -
                                               θ_face_here  * ρwp[i, j, k])
        end
    end
end

#####
##### Section 9 — Damping
#####

# No-op default
@inline apply_divergence_damping!(::NoDivergenceDamping, substepper, grid, Δτ,
                                  thermodynamic_constants) = nothing

# Klemp-Skamarock-Ha (2018) / Skamarock-Klemp (1992) / Baldauf (2010)
# 3-D acoustic divergence damping. In the linearized acoustic mode,
#   (ρθ)′ − (ρθ)′_old ≈ −Δτ · θ⁰ · ∇·((ρu)′, (ρv)′, (ρw)′)
# so D ≡ ((ρθ)′ − (ρθ)′_old) / θ⁰ is a discrete proxy for
# −Δτ · ∇·(ρu)′. The per-substep momentum correction is
#   Δ(ρu)′ = −α_x · ∂x D , Δ(ρv)′ = −α_y · ∂y D , Δ(ρw)′ = −α_z · ∂z D
# with **anisotropic** damping diffusivities (Baldauf 2010 §2.d):
#   α_x = β_d · Δx² / Δτ ,  α_y = β_d · Δy² / Δτ ,  α_z = β_d · Δz² / Δτ
# This gives a constant explicit-time Courant number `β_d` per
# direction, independent of Δτ and grid spacing — the right scaling
# across the wide Δτ ranges Breeze users hit (Δτ ~ 1 s for small Δt,
# ~ 40 s for production lat-lon). Stability bound for the combined
# 3-D damping operator is `2β_d ≤ 1/2 → β_d ≤ 0.25`; we default to
# `β_d = 0.1` for margin.
# The vertical component is essential: without it the rest atmosphere
# amplifies at (Δt = 20 s, ω = 0.55) because the column tridiag's
# buoyancy off-diagonals are anti-symmetric.
function apply_divergence_damping!(damping::KlempDivergenceDamping, substepper, grid, Δτ,
                                   thermodynamic_constants)
    FT    = eltype(grid)
    arch  = architecture(grid)
    β_d   = convert(FT, damping.coefficient)
    Δτ_FT = convert(FT, Δτ)

    TX, TY, _ = topology(grid)
    Δx = TX === Flat ? zero(FT) : convert(FT, minimum_xspacing(grid))
    Δy = TY === Flat ? zero(FT) : convert(FT, minimum_yspacing(grid))
    Δz = convert(FT, grid.z.Δᵃᵃᶜ)

    # Optional override via `length_scale` falls back to isotropic ν.
    if damping.length_scale === nothing
        αx = β_d * Δx^2 / Δτ_FT
        αy = β_d * Δy^2 / Δτ_FT
        αz = β_d * Δz^2 / Δτ_FT
    else
        ℓ = convert(FT, damping.length_scale)
        ν = β_d * ℓ^2 / Δτ_FT
        αx = TX === Flat ? zero(FT) : ν
        αy = TY === Flat ? zero(FT) : ν
        αz = ν
    end

    launch!(arch, grid, :xyz, _klemp_divergence_damping!,
            substepper.momentum_perturbation_u, substepper.momentum_perturbation_v,
            substepper.momentum_perturbation_w,
            substepper.density_potential_temperature_perturbation,
            substepper.previous_density_potential_temperature_perturbation,
            substepper.outer_step_potential_temperature,
            grid, αx, αy, αz)
    return nothing
end

@inline _dρθp_over_θ(i, j, k, grid, ρθp, ρθp_old, θ⁰) =
    @inbounds (ρθp[i, j, k] - ρθp_old[i, j, k]) / θ⁰[i, j, k]

# 3-D anisotropic Klemp / Skamarock-Klemp 1992 / Baldauf 2010 divergence
# damping. Per-substep momentum correction:
#   Δ(ρu)′ = −α_x · ∂x[((ρθ)′ − (ρθ)′_old) / θ⁰]
#   Δ(ρv)′ = −α_y · ∂y[((ρθ)′ − (ρθ)′_old) / θ⁰]
#   Δ(ρw)′ = −α_z · ∂z[((ρθ)′ − (ρθ)′_old) / θ⁰]
# The vertical component is the missing piece that damps the vertical
# acoustic modes responsible for the rest-atmosphere blow-up at
# (Δt = 20 s, ω = 0.55) without divergence damping.
@kernel function _klemp_divergence_damping!(ρup, ρvp, ρwp, ρθp, ρθp_old, θ⁰, grid,
                                            αx, αy, αz)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂x_div = ∂xᶠᶜᶜ(i, j, k, grid, _dρθp_over_θ, ρθp, ρθp_old, θ⁰)
        ρup[i, j, k] -= αx * ∂x_div * !on_x_boundary(i, j, k, grid)

        ∂y_div = ∂yᶜᶠᶜ(i, j, k, grid, _dρθp_over_θ, ρθp, ρθp_old, θ⁰)
        ρvp[i, j, k] -= αy * ∂y_div * !on_y_boundary(i, j, k, grid)

        # Vertical damping at z-faces. Skip the bottom face k = 1
        # (impenetrability — (ρw)′[1] ≡ 0).
        if k > 1
            ∂z_div = ∂zᶜᶜᶠ(i, j, k, grid, _dρθp_over_θ, ρθp, ρθp_old, θ⁰)
            ρwp[i, j, k] -= αz * ∂z_div
        end
    end
end

#####
##### Section 10 — Full-state recovery at stage end
#####

# After the substep loop completes for a stage, reconstruct the full
# prognostic state ρ, ρu, ρv, ρw, ρθ from the outer-step-start snapshot
# plus the accumulated perturbations:
#   ρ_new  = ρ⁰  + ρ′
#   ρθ_new = ρθ⁰ + (ρθ)′
#   ρu_new = ρu⁰ + (ρu)′, etc.
# Velocities are then diagnosed: u = ρu/ρ, etc.
@kernel function _recover_full_state!(ρ, ρθ, m, vel,
                                       ρp, ρθp, ρup, ρvp, ρwp,
                                       ρ⁰, ρu⁰, ρv⁰, ρw⁰, ρθ⁰,
                                       grid)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρ_new  = ρ⁰[i, j, k] + ρp[i, j, k]
        ρθ_new = ρθ⁰[i, j, k] + ρθp[i, j, k]
        ρu_new = ρu⁰[i, j, k] + ρup[i, j, k]
        ρv_new = ρv⁰[i, j, k] + ρvp[i, j, k]
        ρw_new = ρw⁰[i, j, k] + ρwp[i, j, k]

        ρ[i, j, k]  = ρ_new
        ρθ[i, j, k] = ρθ_new

        m.ρu[i, j, k] = ρu_new
        m.ρv[i, j, k] = ρv_new
        m.ρw[i, j, k] = ρw_new

        ρ_x = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        ρ_y = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        ρ_z = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρ_x_safe = ifelse(ρ_x == 0, one(ρ_x), ρ_x)
        ρ_y_safe = ifelse(ρ_y == 0, one(ρ_y), ρ_y)
        ρ_z_safe = ifelse(ρ_z == 0, one(ρ_z), ρ_z)

        vel.u[i, j, k] = ρu_new / ρ_x_safe * !on_x_boundary(i, j, k, grid)
        vel.v[i, j, k] = ρv_new / ρ_y_safe * !on_y_boundary(i, j, k, grid)
        vel.w[i, j, k] = ρw_new / ρ_z_safe * (k > 1)
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
function acoustic_rk3_substep_loop!(model, substepper, Δt, β_stage, U⁰)
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
        # Step A: explicit horizontal forward of (ρu)′, (ρv)′ using current (ρθ)′
        launch!(arch, grid, :xyz, _explicit_horizontal_step!,
                substepper.momentum_perturbation_u, substepper.momentum_perturbation_v,
                grid, FT(Δτ),
                substepper.density_potential_temperature_perturbation,
                substepper.outer_step_exner,
                substepper.pressure_imbalance,
                Gⁿ.ρu, Gⁿ.ρv, substepper.outer_step_gamma_R_mixture)

        fill_halo_regions!(substepper.momentum_perturbation_u)
        fill_halo_regions!(substepper.momentum_perturbation_v)

        # Save (ρθ)′ before the column kernel for damping use
        parent(substepper.previous_density_potential_temperature_perturbation) .=
            parent(substepper.density_potential_temperature_perturbation)

        # CN time-step weights for this substep. δτ_new = ω·Δτ is the
        # new-side weight (used by the matrix and the post-solve);
        # δτ_old = (1−ω)·Δτ is the old-side weight (used by the
        # predictor's old-flux contribution and the old part of the
        # vertical RHS). See derivation_phase1.md eqns. (5), (7), (15).
        δτ_new = ω * FT(Δτ)
        δτ_old = one_minus_ω * FT(Δτ)

        # Step B: build predictors `ρp_pred`, `ρθp_pred` and the tridiag RHS for (ρw)′_new
        launch!(arch, grid, :xy, _build_predictors_and_vertical_rhs!,
                substepper.momentum_perturbation_w,
                substepper.density_predictor,
                substepper.density_potential_temperature_predictor,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation_w,
                substepper.momentum_perturbation_u, substepper.momentum_perturbation_v,
                grid, FT(Δτ), δτ_new, δτ_old,
                Gⁿ.ρ, Gˢρθ, substepper.slow_vertical_momentum_tendency,
                substepper.outer_step_potential_temperature, substepper.outer_step_exner,
                substepper.outer_step_gamma_R_mixture, g)

        # Step C: implicit tridiag solve for (ρw)′ with implicit-half δτ_new
        solve!(substepper.momentum_perturbation_w, substepper.vertical_solver,
               substepper.momentum_perturbation_w,
               substepper.outer_step_exner, substepper.outer_step_potential_temperature,
               substepper.outer_step_gamma_R_mixture, g, δτ_new)

        # Step D: post-solve recovery of ρ′, (ρθ)′ using new (ρw)′
        launch!(arch, grid, :xy, _post_solve_recovery!,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation_w,
                substepper.density_predictor,
                substepper.density_potential_temperature_predictor,
                grid, δτ_new,
                substepper.outer_step_potential_temperature)

        fill_halo_regions!(substepper.density_perturbation)
        fill_halo_regions!(substepper.density_potential_temperature_perturbation)

        # Step E: optional Klemp 2018 damping
        apply_divergence_damping!(substepper.damping, substepper, grid, FT(Δτ),
                                  model.thermodynamic_constants)

        fill_halo_regions!(substepper.density_potential_temperature_perturbation)
        fill_halo_regions!(substepper.density_perturbation)
        fill_halo_regions!(substepper.momentum_perturbation_u)
        fill_halo_regions!(substepper.momentum_perturbation_v)
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
            U⁰[2], U⁰[3], U⁰[4],
            substepper.recovery_density_potential_temperature,
            grid)

    return nothing
end
