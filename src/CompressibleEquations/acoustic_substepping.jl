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

- `density★ictor`, `density_potential_temperature★ictor`: explicit
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
    # the SAME outer-step-start U⁰.
    recovery_density :: CF
    recovery_density_potential_temperature :: CF

    # Linearization basic state — REFRESHED at each WS-RK3 stage from the
    # current model state ``U^{(k-1)}``. This keeps the linearized
    # operators (γRᵐ Π ∂(ρθ)′, g·ρ′, ∂x(p − pᵣ)) consistent with the slow
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

    # Reference-subtracted pressure perturbation, p − pᵣ (= p when no
    # reference). Refreshed each stage along with outer_step_pressure.
    pressure_perturbation :: CF

    density_perturbation :: CF
    density_potential_temperature_perturbation :: CF
    momentum_perturbation_w :: FF
    momentum_perturbation_u :: XF
    momentum_perturbation_v :: YF

    density★ictor :: CF
    density_potential_temperature★ictor :: CF
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
                       adapt(to, a.pressure_perturbation),
                       adapt(to, a.density_perturbation),
                       adapt(to, a.density_potential_temperature_perturbation),
                       adapt(to, a.momentum_perturbation_w),
                       adapt(to, a.momentum_perturbation_u),
                       adapt(to, a.momentum_perturbation_v),
                       adapt(to, a.density★ictor),
                       adapt(to, a.density_potential_temperature★ictor),
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

    density★ictor                                = CenterField(grid)
    density_potential_temperature★ictor          = CenterField(grid)
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
                              pressure_perturbation,
                              density_perturbation,
                              density_potential_temperature_perturbation,
                              momentum_perturbation_w,
                              momentum_perturbation_u,
                              momentum_perturbation_v,
                              density★ictor,
                              density_potential_temperature★ictor,
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
    # `_recover_full_state!` so that Uᵐ⁺ = U⁰_outer + (ρ′, (ρθ)′, (ρu)′, (ρv)′, (ρw)′) — the
    # WS-RK3 invariant that each stage starts from U⁰.
    parent(substepper.recovery_density)                       .= parent(model.dynamics.density)
    parent(substepper.recovery_density_potential_temperature) .= parent(ρθ_field)

    # Then prime the linearization basic state to U⁰ for stage 1.
    refresh_linearization_basic_state!(substepper, model)

    fill_halo_regions!(substepper.recovery_density)
    fill_halo_regions!(substepper.recovery_density_potential_temperature)

    return nothing
end

# Refresh the linearization basic state (Π⁰, θ⁰, p⁰, p − pᵣ, plus the
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
    # the reference depends only on z, so ∂x pᵣ = ∂y pᵣ = 0; the
    # horizontal force is then ∂x(p − pᵣ) = ∂x p. Reference
    # subtraction in z guarantees a hydrostatic rest atmosphere has zero
    # vertical drive, free of FP-rounding noise.
    ref = model.dynamics.reference_state
    if ref isa Nothing
        parent(substepper.pressure_perturbation) .= parent(substepper.outer_step_pressure)
    else
        launch!(arch, grid, :xyz, _compute_pressure_perturbation!,
                substepper.pressure_perturbation,
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

@kernel function _compute_pressure_perturbation!(p′, p⁰, pᵣ)
    i, j, k = @index(Global, NTuple)
    @inbounds p′[i, j, k] = p⁰[i, j, k] - pᵣ[i, j, k]
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
#                - Δτ × γRᵐ × Π⁰_face(k) × (ωˢ⁻ ∂z (ρθ)′_o + ωᵐ⁺ ∂z (ρθ)′_n) / Δzᶠ(k)
#                - Δτ × g × (ωˢ⁻ ρ′_face_o(k) + ωᵐ⁺ ρ′_face_n(k))
#
# with ωᵐ⁺ = (1+ε)/2, ωˢ⁻ = (1-ε)/2 (ε=0 is centered CN).
#
# The post-solve substitution (matching the column kernel):
#   ρ′_n(k)    = ρ′★(k)  - δτᵐ⁺ × ((ρw)′_n(k+1) - (ρw)′_n(k)) / Δz_c(k)
#   (ρθ)′_n(k) = ρθ′★(k) - δτᵐ⁺ × (θ⁰_face(k+1) (ρw)′_n(k+1)
#                                        - θ⁰_face(k)   (ρw)′_n(k)) / Δz_c(k)
# where δτᵐ⁺ = ωᵐ⁺ Δτ.
#
# Substituting yields the tridiagonal coefficients (ω = ωᵐ⁺):
#
#   A[k,k+1] = -(ω Δτ)² × γRᵐ × Π⁰_face(k) × θ⁰_face(k+1) × rdz_c(k)   / Δzᶠ(k)
#              - (ω Δτ)² × g          × rdz_c(k)   / 2
#
#   A[k,k]   = 1 + (ω Δτ)² × γRᵐ × Π⁰_face(k) × θ⁰_face(k)   × (rdz_c(k) + rdz_c(k-1)) / Δzᶠ(k)
#                + (ω Δτ)² × g                              × (rdz_c(k) - rdz_c(k-1)) / 2
#
#   A[k,k-1] = -(ω Δτ)² × γRᵐ × Π⁰_face(k) × θ⁰_face(k-1) × rdz_c(k-1) / Δzᶠ(k)
#              + (ω Δτ)² × g                              × rdz_c(k-1) / 2
#
# `γᵐRᵐ⁰` is the cell-centered mixture coefficient `γᵐ Rᵐ` evaluated from
# the snapshotted moisture (`outer_step_gamma_R_mixture`). It is interpolated
# to z-faces inside the kernel. For dry runs (qᵛ = qˡ = qⁱ = 0) this collapses
# bit-identically to the dry constant `γᵈRᵈ`.
#
# Implicit vertical damping
# -------------------------
# When `damping isa ThermalDivergenceDamping` with `vertical_implicit = true`,
# the vertical part of the divergence damping is folded into the same tridiag.
# Reformulating the kernel correction `Δ(ρw)′ = -α_z ∂z D` via the linearized
# (ρθ)′ continuity equation gives a discrete vertical Laplacian on `(ρw)′`:
#
#   (ρw)′_n − ω β_d Δz² ∂z² (ρw)′_n = (ρw)′_o + (1−ω) β_d Δz² ∂z² (ρw)′_o
#
# At face k the −∂z² stencil contributes (with `dᵐ⁺ ≡ ω β_d Δz²`):
#
#   A[k,k+1] += -dᵐ⁺ × rdz_c(k)   / Δzᶠ(k)
#   A[k,k]   += +dᵐ⁺ × (rdz_c(k) + rdz_c(k-1)) / Δzᶠ(k)
#   A[k,k-1] += -dᵐ⁺ × rdz_c(k-1) / Δzᶠ(k)
#
# The matching `(1−ω) β_d Δz² ∂z² (ρw)′_o` term is added to the predictor's
# right-hand side in `_build★ictors_and_vertical_rhs!`. The constant-Courant
# scaling `α_z = β_d Δz² / Δτ` makes `dᵐ⁺` and the RHS prefactor independent
# of Δτ; only `β_d`, `ω`, and the global vertical spacing `grid.z.Δᵃᵃᶜ` enter.
# When `vertical_implicit = false` (or for `NoDivergenceDamping`), the
# damping factor passed in is zero and the tridiag reduces to the pure
# off-centered CN acoustic system above.

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagLower, p, ::ZDirection,
                                 Π⁰, θ⁰, γRᵐ⁰, g, δτᵐ⁺, dᵐ⁺)
    kᶠ      = k + 1
    Δzᶠ     = Δzᶜᶜᶠ(i, j, kᶠ, grid)
    Δz⁻¹ᵏ⁻  = 1 / Δzᶜᶜᶜ(i, j, kᶠ - 1, grid)

    Πᶜᶜᶠ    = ℑzᵃᵃᶠ(i, j, kᶠ, grid, Π⁰)
    γRᵐ⁰ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, kᶠ, grid, γRᵐ⁰)
    θᵏ⁻     = ℑbzᵃᵃᶠ(i, j, kᶠ - 1, grid, θ⁰)

    pgf_term  = - δτᵐ⁺^2 * γRᵐ⁰ᶜᶜᶠ * Πᶜᶜᶠ * θᵏ⁻ * Δz⁻¹ᵏ⁻ / Δzᶠ
    buoy_term = + δτᵐ⁺^2 * g                    * Δz⁻¹ᵏ⁻ / 2
    damp_term = - dᵐ⁺                           * Δz⁻¹ᵏ⁻ / Δzᶠ
    return pgf_term + buoy_term + damp_term
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagDiagonal, p, ::ZDirection,
                                 Π⁰, θ⁰, γRᵐ⁰, g, δτᵐ⁺, dᵐ⁺)

    kᶠ      = k
    Δzᶠ     = Δzᶜᶜᶠ(i, j, kᶠ, grid)
    Δz⁻¹ᵏ⁺  = 1 / Δzᶜᶜᶜ(i, j, kᶠ,     grid)
    Δz⁻¹ᵏ⁻  = 1 / Δzᶜᶜᶜ(i, j, kᶠ - 1, grid)

    Πᶜᶜᶠ    = ℑzᵃᵃᶠ(i, j, kᶠ, grid, Π⁰)
    γRᵐ⁰ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, kᶠ, grid, γRᵐ⁰)
    θᶜᶜᶠ    = ℑbzᵃᵃᶠ(i, j, kᶠ, grid, θ⁰)

    pgf_diag  = δτᵐ⁺^2 * γRᵐ⁰ᶜᶜᶠ * Πᶜᶜᶠ * θᶜᶜᶠ * (Δz⁻¹ᵏ⁺ + Δz⁻¹ᵏ⁻) / Δzᶠ
    buoy_diag = δτᵐ⁺^2 * g                     * (Δz⁻¹ᵏ⁺ - Δz⁻¹ᵏ⁻) / 2
    damp_diag = dᵐ⁺                            * (Δz⁻¹ᵏ⁺ + Δz⁻¹ᵏ⁻) / Δzᶠ

    return one(grid) + (pgf_diag + buoy_diag + damp_diag) * (k > 1)
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagUpper, p, ::ZDirection,
                                 Π⁰, θ⁰, γRᵐ⁰, g, δτᵐ⁺, dᵐ⁺)

    kᶠ      = k
    Δzᶠ     = Δzᶜᶜᶠ(i, j, kᶠ, grid)
    Δz⁻¹ᵏ⁺  = 1 / Δzᶜᶜᶜ(i, j, kᶠ, grid)

    Πᶜᶜᶠ    = ℑzᵃᵃᶠ(i, j, kᶠ, grid, Π⁰)
    γRᵐ⁰ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, kᶠ, grid, γRᵐ⁰)
    θᵏ⁺     = ℑbzᵃᵃᶠ(i, j, kᶠ + 1, grid, θ⁰)

    pgf_term  = - δτᵐ⁺^2 * γRᵐ⁰ᶜᶜᶠ * Πᶜᶜᶠ * θᵏ⁺ * Δz⁻¹ᵏ⁺ / Δzᶠ
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
        # Reference-subtracted PGF and buoyancy: at U⁰ = reference state
        # both terms are exactly zero by construction of the reference.
        ∂z_p′ = ∂zᶜᶜᶠ(i, j, k, grid, _p_perturbation, p⁰, pᵣ)
        ρ′ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, _ρ_perturbation, ρ⁰, ρᵣ)

        Gˢρw[i, j, k] = (Gⁿρw[i, j, k] - ∂z_p′ - g * ρ′ᶜᶜᶠ) * (k > 1)
    end
end

@inline _p_perturbation(i, j, k, grid, p⁰, pᵣ) = @inbounds p⁰[i, j, k] - pᵣ[i, j, k]
@inline _ρ_perturbation(i, j, k, grid, ρ⁰, ρᵣ) = @inbounds ρ⁰[i, j, k] - ρᵣ[i, j, k]

@kernel function _assemble_slow_vertical_momentum_tendency_no_ref!(Gˢρw, Gⁿρw, p⁰, ρ⁰, grid, g)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂z_p⁰  = ∂zᶜᶜᶠ(i, j, k, grid, p⁰)
        ρ⁰ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ⁰)
        Gˢρw[i, j, k] = (Gⁿρw[i, j, k] - ∂z_p⁰ - g * ρ⁰ᶜᶜᶠ) * (k > 1)
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
    fill!(parent(substepper.density★ictor), 0)
    fill!(parent(substepper.density_potential_temperature★ictor), 0)
    return nothing
end

# Explicit forward step for horizontal momentum perturbations (ρu)′, (ρv)′.
#
# Linearized at U⁰, the full horizontal pressure gradient splits as
#   ∂x p_full = ∂x(p⁰ − pᵣ) + γRᵐ Π⁰ · ∂x((ρθ)′)
# where the first piece is the FROZEN imbalance from the linearization
# point (carried by `pressure_perturbation`) and the second is the
# perturbation force. `Gⁿρu` from `SlowTendencyMode` carries advection
# only (PGF zeroed); we reinstate the frozen horizontal pressure
# perturbation here.
#
# (ρu)′^{τ+Δτ} = (ρu)′^τ + Δτ · (Gⁿρu − ∂x(p⁰−pᵣ) − γᵐRᵐ⁰ Π⁰_x ∂x((ρθ)′))
# (ρv)′^{τ+Δτ} = (ρv)′^τ + Δτ · (Gⁿρv − ∂y(p⁰−pᵣ) − γᵐRᵐ⁰ Π⁰_y ∂y((ρθ)′))
@kernel function _explicit_horizontal_step!(ρu′, ρv′, grid, Δτ, ρθ′, Π⁰, p′,
                                            Gⁿρu, Gⁿρv, γRᵐ⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Π⁰_x   = ℑxᶠᵃᵃ(i, j, k, grid, Π⁰)
        γRᵐ⁰_x = ℑxᶠᵃᵃ(i, j, k, grid, γRᵐ⁰)
        ∂x_ρθ′ = ∂xᶠᶜᶜ(i, j, k, grid, ρθ′)
        ∂x_p′  = ∂xᶠᶜᶜ(i, j, k, grid, p′)
        ∂x_p   = ∂x_p′ + γRᵐ⁰_x * Π⁰_x * ∂x_ρθ′

        Π⁰_y   = ℑyᵃᶠᵃ(i, j, k, grid, Π⁰)
        γRᵐ⁰_y = ℑyᵃᶠᵃ(i, j, k, grid, γRᵐ⁰)
        ∂y_ρθ′ = ∂yᶜᶠᶜ(i, j, k, grid, ρθ′)
        ∂y_p′  = ∂yᶜᶠᶜ(i, j, k, grid, p′)
        ∂y_p   = ∂y_p′ + γRᵐ⁰_y * Π⁰_y * ∂y_ρθ′

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
                                                     θ⁰, Π⁰,
                                                     γRᵐ⁰, g, dˢ⁻)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Cell-centred predictors `ρ′★`, `ρθ′★`.
        for k in 1:Nz
            V = Vᶜᶜᶜ(i, j, k, grid)

            ∇ʰ_M  = div_xyᶜᶜᶜ(i, j, k, grid, ρu′, ρv′)
            ∇ʰ_θM = (δxᶜᵃᵃ(i, j, k, grid, _theta_face_x_flux, θ⁰, ρu′) +
                     δyᵃᶜᵃ(i, j, k, grid, _theta_face_y_flux, θ⁰, ρv′)) / V

            ρ′★[i, j, k]  = ρ′[i, j, k] +
                                Δτ * (Gˢρ[i, j, k] - ∇ʰ_M) -
                                δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, ρw′)

            ρθ′★[i, j, k] = ρθ′[i, j, k] +
                                Δτ * (Gˢρθ[i, j, k] - ∇ʰ_θM) -
                                δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, _theta_face_z_flux, θ⁰, ρw′)
        end

        # Face-level RHS for `(ρw)′ᵐ⁺` tridiag — split weights for the
        # predictor and old-step contributions per derivation (15).
        # `dˢ⁻ = (1−ω) β_d Δz²` adds the explicit half of the implicit
        # vertical damping (zero when damping is off or vertical_implicit=false).
        for k in 2:Nz
            Δzᶠ   = Δzᶜᶜᶠ(i, j, k, grid)
            Πᶜᶜᶠ    = ℑzᵃᵃᶠ(i, j, k, grid, Π⁰)
            γRᵐ⁰ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, γRᵐ⁰)

            ∂z_ρθ′★ = ρθ′★[i, j, k] - ρθ′★[i, j, k - 1]
            ∂z_ρθ′ˢ⁻ = ρθ′[i, j, k] - ρθ′[i, j, k - 1]

            sound_force = γRᵐ⁰ᶜᶜᶠ * Πᶜᶜᶠ / Δzᶠ * (δτˢ⁻ * ∂z_ρθ′ˢ⁻ + δτᵐ⁺ * ∂z_ρθ′★)

            ρ′ᶜᶜᶠ★  = ℑzᵃᵃᶠ(i, j, k, grid, ρ′★)
            ρ′ᶜᶜᶠˢ⁻ = ℑzᵃᵃᶠ(i, j, k, grid, ρ′)
            buoy_force = g * (δτˢ⁻ * ρ′ᶜᶜᶠˢ⁻ + δτᵐ⁺ * ρ′ᶜᶜᶠ★)

            # Explicit (old-step) half of the vertical damping
            # `(1−ω) β_d Δz² ∂z²(ρw)′ˢ⁻`, evaluated at face k. The face-coupling
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

# θ⁰ · (ρu)′ at an x-face. Used in the area-weighted horizontal
# divergence of the perturbation θ-flux.
@inline _theta_face_x_flux(i, j, k, grid, θ⁰, ρu′) =
    @inbounds Axᶠᶜᶜ(i, j, k, grid) * ℑxᶠᵃᵃ(i, j, k, grid, θ⁰) * ρu′[i, j, k]

@inline _theta_face_y_flux(i, j, k, grid, θ⁰, ρv′) =
    @inbounds Ayᶜᶠᶜ(i, j, k, grid) * ℑyᵃᶠᵃ(i, j, k, grid, θ⁰) * ρv′[i, j, k]

# θ⁰ · (ρw)′ at a z-face. Used in the vertical part of the perturbation
# θ-flux divergence; passed to `∂zᶜᶜᶜ` so the divergence is computed at
# cell centers from the face-located product.
@inline _theta_face_z_flux(i, j, k, grid, θ⁰, ρw′) =
    @inbounds ℑbzᵃᵃᶠ(i, j, k, grid, θ⁰) * ρw′[i, j, k]

@inline ℑb_wθ(i, j, k, w, θ) = @inbounds w[i, j, k] * ℑbzᵃᵃᶠ(i, j, k, grid, θ⁰)

# Post-solve recovery: substitute the tridiag-solved `(ρw)′ᵐ⁺` back
# into the `ρ′★`, `ρθ′★` predictors to get `ρ′ᵐ⁺`, `ρθ′ᵐ⁺`
# (the IMPLICIT half of CN).
#
#   ρ′_n(k)    = ρ′★(k)  - (δτᵐ⁺ / Δz_c(k)) · ((ρw)′_n(k+1) - (ρw)′_n(k))
#   (ρθ)′_n(k) = ρθ′★(k) - (δτᵐ⁺ / Δz_c(k)) · (θ⁰_face(k+1) (ρw)′_n(k+1)
#                                                    - θ⁰_face(k)   (ρw)′_n(k))
@kernel function _post_solve_recovery!(ρ′, ρθ′, ρw′, ρ′★, ρθ′★,
                                        grid, δτᵐ⁺, θ⁰)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        for k in 1:Nz
            ρ′[i, j, k] = ρ′★[i, j, k] - δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, ρw′)
            ρθ′[i, j, k] = ρθ′★[i, j, k] - δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, ℑb_wθ, ρw′, θ⁰)
        end
    end
end

#####
##### Section 9 — Damping
#####

# No-op default
@inline apply_divergence_damping!(::NoDivergenceDamping, substepper, grid, Δτ,
                                  thermodynamic_constants) = nothing

# Implicit-vertical-damping prefactors threaded into the column tridiag and
# its RHS. Returns `(dᵐ⁺, dˢ⁻) = (ω, 1−ω) · β_d · Δz²` for
# `ThermalDivergenceDamping`, and `(0, 0)` for `NoDivergenceDamping` —
# which makes the tridiag and predictor-RHS additions vanish, recovering
# the pure off-centered CN acoustic system.
@inline _implicit_damping_factors(::AcousticDampingStrategy, ω, one_minus_ω, grid, FT) =
    (zero(FT), zero(FT))

@inline function _implicit_damping_factors(damping::ThermalDivergenceDamping,
                                           ω, one_minus_ω, grid, FT)
    β_d = convert(FT, damping.coefficient)
    Δz  = convert(FT, minimum_zspacing(grid))
    base = β_d * Δz^2
    return (convert(FT, ω) * base, convert(FT, one_minus_ω) * base)
end

# Klemp-Skamarock-Ha (2018) / Skamarock-Klemp (1992) / Baldauf (2010)
# 3-D acoustic divergence damping. In the linearized acoustic mode,
#   (ρθ)′ − (ρθ)′ˢ⁻ ≈ −Δτ · θ⁰ · ∇·((ρu)′, (ρv)′, (ρw)′)
# so D ≡ ((ρθ)′ − (ρθ)′ˢ⁻) / θ⁰ is a discrete proxy for
# −Δτ · ∇·(ρu)′. The per-substep momentum correction is
#   Δ(ρu)′ = −α_x · ∂x D , Δ(ρv)′ = −α_y · ∂y D , Δ(ρw)′ = −α_z · ∂z D
# with **anisotropic** damping diffusivities (Baldauf 2010 §2.d):
#   α_x = β_d · Δx² / Δτ ,  α_y = β_d · Δy² / Δτ ,  α_z = β_d · Δz² / Δτ
# This gives a constant explicit-time Courant number `β_d` per
# direction, independent of Δτ and grid spacing — the right scaling
# across the wide Δτ ranges Breeze users hit (Δτ ~ 1 s for small Δt,
# ~ 40 s for production lat-lon). Linear stability of the explicit
# forward-Euler step gives an amplification factor
#   A(k) = 1 − 4 β_d × Σᵢ sin²(kᵢ Δxᵢ / 2)
# whose worst case (Nyquist excited in all three directions) is
# `12 β_d ≤ 2 → β_d ≤ 1/6 ≈ 0.167`. The 2-D bound (worst case excited
# in only two directions, e.g., lat-lon with Δz ≪ Δx so vertical
# Nyquist is suppressed) is `8 β_d ≤ 2 → β_d ≤ 1/4`. We default to
# `β_d = 0.1` for margin against the strict 3-D bound.
# The vertical component is essential: without it the rest atmosphere
# amplifies at (Δt = 20 s, ω = 0.55) because the column tridiag's
# buoyancy off-diagonals are anti-symmetric.
function apply_divergence_damping!(damping::ThermalDivergenceDamping, substepper, grid, Δτ,
                                   thermodynamic_constants)
    FT    = eltype(grid)
    arch  = architecture(grid)
    β_d   = convert(FT, damping.coefficient)
    Δτ_FT = convert(FT, Δτ)

    TX, TY, _ = topology(grid)
    Δx = TX === Flat ? zero(FT) : convert(FT, minimum_xspacing(grid))
    Δy = TY === Flat ? zero(FT) : convert(FT, minimum_yspacing(grid))

    # Horizontal damping coefficients only — the vertical part is folded
    # into the column tridiag and its RHS via `_implicit_damping_factors`.
    if damping.length_scale === nothing
        αx = β_d * Δx^2 / Δτ_FT
        αy = β_d * Δy^2 / Δτ_FT
    else
        ℓ = convert(FT, damping.length_scale)
        ν = β_d * ℓ^2 / Δτ_FT
        αx = TX === Flat ? zero(FT) : ν
        αy = TY === Flat ? zero(FT) : ν
    end

    launch!(arch, grid, :xyz, _thermal_divergence_damping!,
            substepper.momentum_perturbation_u, substepper.momentum_perturbation_v,
            substepper.density_potential_temperature_perturbation,
            substepper.previous_density_potential_temperature_perturbation,
            substepper.outer_step_potential_temperature,
            grid, αx, αy)
    return nothing
end

@inline _dρθ′_over_θ(i, j, k, grid, ρθ′, ρθ′ˢ⁻, θ⁰) =
    @inbounds (ρθ′[i, j, k] - ρθ′ˢ⁻[i, j, k]) / θ⁰[i, j, k]

# 3-D anisotropic Klemp / Skamarock-Klemp 1992 / Baldauf 2010 divergence
# damping. Per-substep momentum correction:
#   Δ(ρu)′ = −α_x · ∂x[((ρθ)′ − (ρθ)′ˢ⁻) / θ⁰]
#   Δ(ρv)′ = −α_y · ∂y[((ρθ)′ − (ρθ)′ˢ⁻) / θ⁰]
#   Δ(ρw)′ = −α_z · ∂z[((ρθ)′ − (ρθ)′ˢ⁻) / θ⁰]
# The vertical component is the missing piece that damps the vertical
# acoustic modes responsible for the rest-atmosphere blow-up at
# (Δt = 20 s, ω = 0.55) without divergence damping.
@kernel function _thermal_divergence_damping!(ρu′, ρv′, ρθ′, ρθ′ˢ⁻, θ⁰, grid, αx, αy)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂x_div = ∂xᶠᶜᶜ(i, j, k, grid, _dρθ′_over_θ, ρθ′, ρθ′ˢ⁻, θ⁰)
        ρu′[i, j, k] -= αx * ∂x_div * !on_x_boundary(i, j, k, grid)

        ∂y_div = ∂yᶜᶠᶜ(i, j, k, grid, _dρθ′_over_θ, ρθ′, ρθ′ˢ⁻, θ⁰)
        ρv′[i, j, k] -= αy * ∂y_div * !on_y_boundary(i, j, k, grid)
    end
end

#####
##### Section 10 — Full-state recovery at stage end
#####

# After the substep loop completes for a stage, reconstruct the full
# prognostic state ρ, ρu, ρv, ρw, ρθ from the outer-step-start snapshot
# plus the accumulated perturbations:
#   ρᵐ⁺  = ρ⁰  + ρ′
#   ρθᵐ⁺ = ρθ⁰ + (ρθ)′
#   ρuᵐ⁺ = ρu⁰ + (ρu)′, etc.
# Velocities are then diagnosed: u = ρu/ρ, etc.
@kernel function _recover_full_state!(ρ, ρθ, m, vel,
                                      ρ′, ρθ′, ρu′, ρv′, ρw′,
                                      ρ⁰, ρu⁰, ρv⁰, ρw⁰, ρθ⁰,
                                      grid)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵐ⁺  = ρ⁰[i, j, k]  + ρ′[i, j, k]
        ρθᵐ⁺ = ρθ⁰[i, j, k] + ρθ′[i, j, k]
        ρuᵐ⁺ = ρu⁰[i, j, k] + ρu′[i, j, k]
        ρvᵐ⁺ = ρv⁰[i, j, k] + ρv′[i, j, k]
        ρwᵐ⁺ = ρw⁰[i, j, k] + ρw′[i, j, k]

        ρ[i, j, k]  = ρᵐ⁺
        ρθ[i, j, k] = ρθᵐ⁺

        m.ρu[i, j, k] = ρuᵐ⁺
        m.ρv[i, j, k] = ρvᵐ⁺
        m.ρw[i, j, k] = ρwᵐ⁺

        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρ_x_safe = ifelse(ρᶠᶜᶜ == 0, one(ρᶠᶜᶜ), ρᶠᶜᶜ)
        ρ_y_safe = ifelse(ρᶜᶠᶜ == 0, one(ρᶜᶠᶜ), ρᶜᶠᶜ)
        ρ_z_safe = ifelse(ρᶜᶜᶠ == 0, one(ρᶜᶜᶠ), ρᶜᶜᶠ)

        vel.u[i, j, k] = ρuᵐ⁺ / ρ_x_safe * !on_x_boundary(i, j, k, grid)
        vel.v[i, j, k] = ρvᵐ⁺ / ρ_y_safe * !on_y_boundary(i, j, k, grid)
        vel.w[i, j, k] = ρwᵐ⁺ / ρ_z_safe * (k > 1)
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
                substepper.pressure_perturbation,
                Gⁿ.ρu, Gⁿ.ρv, substepper.outer_step_gamma_R_mixture)

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
        # tridiag with `dᵐ⁺ = ω·β_d·Δz²` on the LHS and
        # `dˢ⁻ = (1−ω)·β_d·Δz²` on the predictor RHS. Both reduce to
        # zero for `NoDivergenceDamping` or when the user opts out via
        # `vertical_implicit=false`.
        dᵐ⁺, dˢ⁻ = _implicit_damping_factors(substepper.damping, ω, one_minus_ω, grid, FT)

        # Step B: build predictors `ρ′★`, `ρθ′★` and the tridiag RHS for (ρw)′ᵐ⁺
        launch!(arch, grid, :xy, _build_predictors_and_vertical_rhs!,
                substepper.momentum_perturbation_w,
                substepper.density_predictor,
                substepper.density_potential_temperature★ictor,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation_w,
                substepper.momentum_perturbation_u, substepper.momentum_perturbation_v,
                grid, FT(Δτ), δτᵐ⁺, δτˢ⁻,
                Gⁿ.ρ, Gˢρθ, substepper.slow_vertical_momentum_tendency,
                substepper.outer_step_potential_temperature, substepper.outer_step_exner,
                substepper.outer_step_gamma_R_mixture, g, dˢ⁻)

        # Step C: implicit tridiag solve for (ρw)′ with implicit-half δτᵐ⁺
        # and (when active) implicit vertical damping prefactor `dᵐ⁺`.
        solve!(substepper.momentum_perturbation_w, substepper.vertical_solver,
               substepper.momentum_perturbation_w,
               substepper.outer_step_exner, substepper.outer_step_potential_temperature,
               substepper.outer_step_gamma_R_mixture, g, δτᵐ⁺, dᵐ⁺)

        # Step D: post-solve recovery of ρ′, (ρθ)′ using new (ρw)′
        launch!(arch, grid, :xy, _post_solve_recovery!,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation_w,
                substepper.density★ictor,
                substepper.density_potential_temperature★ictor,
                grid, δτᵐ⁺,
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
