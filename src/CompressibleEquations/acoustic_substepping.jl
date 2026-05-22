#####
##### Acoustic substepping for CompressibleDynamics
#####
##### The substepper evolves linearized acoustic perturbations between WS-RK3
##### stages, with the linearization refreshed to the **stage-entry state**
##### Uᴸ_stage at the start of each RK stage (per Skamarock & Klemp 2008).
##### Prime notation denotes the perturbation about Uᴸ_stage:
#####
#####   ρ′    = ρ   − ρᴸ
#####   (ρθ)′ = ρθ  − ρθᴸ
#####   (ρu)′ = ρu  − ρuᴸ,  (ρv)′ = ρv − ρvᴸ,  (ρw)′ = ρw − ρwᴸ
#####
##### Background quantities θᴸ = ρθᴸ/ρᴸ, Πᴸ = (pᴸ/pˢᵗ)^κ, and γᵐRᵐᴸ are
##### cached per stage; ρᴸ, ρθᴸ, pᴸ, ρuᴸ, ρvᴸ, ρwᴸ are read live from the
##### model since the substep loop never mutates those fields.
#####
##### The linearized perturbation equations integrated by the substep loop:
#####
#####   ∂t ρ′    +     ∇·((ρu)′, (ρv)′, (ρw)′)        = Gˢρ
#####   ∂t (ρθ)′ +     ∇·(θᴸ · ((ρu)′, (ρv)′, (ρw)′)) = Gˢρθ
#####   ∂t (ρu)′ + ∂x pᴸ + ∂x(Cᴸ (ρθ)′)               = Gˢρu
#####   ∂t (ρv)′ + ∂y pᴸ + ∂y(Cᴸ (ρθ)′)               = Gˢρv
#####   ∂t (ρw)′ +         ∂z(Cᴸ (ρθ)′) + g · ρ′      = Gˢρw
#####
##### where Cᴸ = γᵐRᵐᴸΠᴸ. The discrete PGF uses the gradient of the
##### cell-centered product Cᴸ(ρθ)′.
#####
##### Time discretization: horizontal momentum updates are forward-Euler
##### with MPAS-style first-small-step sequencing (the first substep of a
##### multi-substep stage includes the frozen ∇pᴸ force but skips the
##### perturbation acoustic horizontal PGF, which enters on later substeps).
##### The vertical ((ρw)′, (ρθ)′, ρ′) coupling is
##### solved implicitly with an off-centered Crank-Nicolson scheme —
##### `forward_weight = 0.5` is classic centered CN (neutrally stable for
##### the linearized inviscid system), `forward_weight > 0.5` adds
##### dissipation. The implicit step reduces to a tridiagonal Schur system
##### in (ρw)′ at z-faces.
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
#####   (ρu)′ ↔ kernel arg `ρu′`    (momentum_perturbation.u)
#####   (ρv)′ ↔ kernel arg `ρv′`    (momentum_perturbation.v)
#####   (ρw)′ ↔ kernel arg `ρw′`    (momentum_perturbation.w)
##### Predictors carry a `★` suffix: `ρ′★`, `ρθ′★`.
#####
##### See `docs/src/compressible_dynamics.md` for the public derivation
##### from the continuous equations through the column tridiag.
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

using Oceananigans: CenterField, XFaceField, YFaceField, ZFaceField, architecture, interior
using Oceananigans.Grids: ZDirection, rnode
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!
using Oceananigans.Operators:
    ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ,
    ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ,
    δxᶜᵃᵃ, δyᵃᶜᵃ,
    div_xyᶜᶜᶜ,
    Δzᶜᶜᶜ, Δzᶜᶜᶠ,
    Δxᶠᶜᶜ,
    Δyᶜᶠᶜ,
    Axᶠᶜᶜ, Ayᶜᶠᶜ, Vᶜᶜᶜ

using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!

using Oceananigans.Grids: Flat, Center, peripheral_node,
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
stages, with the linearization refreshed to the stage-entry state
``Uᴸ_\\mathrm{stage}`` at the start of each RK stage. ``ρᴸ, ρθᴸ, pᴸ`` and
the stage-entry momenta are read live from `model.dynamics.*` and
`model.momentum.*` (untouched by the substep loop). Three derived
quantities ``Πᴸ = (pᴸ/pˢᵗ)^κ``, ``θᴸ = ρθᴸ/ρᴸ`` and ``γᵐRᵐᴸ`` are
computed once per stage and cached as fields, since recomputing them
inline at each call site is significantly slower on H100.

The vertical implicit solve uses a centered (or off-centered)
Crank-Nicolson scheme that reduces to a tridiagonal Schur system for the
vertical-momentum perturbation ``(ρw)′``.

Fields
======

- `substeps`: Number of acoustic substeps ``N`` per outer ``Δt`` (or
  `nothing` for adaptive — see `acoustic_cfl`).
- `acoustic_cfl`: Target horizontal acoustic Courant number used by the
  adaptive substep count when `substeps === nothing`. Default `0.5`.
- `forward_weight`: Off-centering parameter ``\\omega``. ``\\omega = 0.5``
  is classic centered CN; the default is 0.65.
- `damping`: Acoustic divergence damping strategy.
- `substep_distribution`: How acoustic substeps are distributed across
  the WS-RK3 stages.

Stage-entry linearization point:

The substep loop never mutates `model.dynamics.density`, the formulation's
prognostic ρθ field, or `model.dynamics.pressure` — those are the
**stage-entry** ρᴸ, ρθᴸ, pᴸ throughout the substep loop and serve as the
recovery base for `_recover_full_state!`. No separate snapshot fields
are stored.

- `linearization_exner`: ``Πᴸ = (pᴸ/pˢᵗ)^κ`` cached from the live `pᴸ`.
- `linearization_potential_temperature`: ``θᴸ = ρθᴸ/ρᴸ`` for the perturbation
  temperature flux ``θᴸ · μ``.

Perturbation prognostics (advanced inside the substep loop):

- `density_perturbation`: ``ρ′ = ρ − ρᴸ``.
- `density_potential_temperature_perturbation`: ``(ρθ)′ = ρθ − ρθᴸ``.
- `momentum_perturbation`: ``(ρu)′, (ρv)′, (ρw)′`` fields grouped as
  `.u`, `.v`, and `.w`.

Per-column scratch (column kernel only):

- `density_predictor`, `density_potential_temperature_predictor`: explicit
  predictors built before the implicit vertical solve.
- `previous_density_potential_temperature_perturbation`: ``η`` from the
  previous substep, used by Klemp 2018 damping.

Time-averaged velocity for non-acoustic scalar transport (WRF/MPAS-style
dynamics-transport split):

- `time_averaged_velocities`: velocity tuple returned by
  `transport_velocities(model)` so moisture/tracer/chemistry/TKE tendencies
  advect against the acoustic-mean velocity, not a single snapshot. The slow
  thermodynamic `ρθ` tendency is computed separately with the current RK
  predictor velocity; it is not routed through this cache.

Vertical solve:

- `slow_vertical_momentum_tendency`: assembled vertical-momentum slow
  tendency ``Gˢρw`` at z-faces (advection + Coriolis + closure + forcing,
  with PGF and buoyancy excluded — those live in the fast operator).
- `vertical_solver`: `BatchedTridiagonalSolver` for the implicit ``(ρw)′`` update.
"""
struct AcousticSubstepper{N, FT, D, AD, US, CF, MP, TAV, GT, TS}
    substeps :: N
    acoustic_cfl :: FT
    forward_weight :: FT
    thermodynamic_tendency_factor :: FT
    vertical_momentum_tendency_factor :: FT
    vertical_pressure_tendency_factor :: FT
    final_stage_vertical_pressure_tendency_factor :: FT
    apply_first_substep_pressure_gradient :: Bool
    damping :: D
    substep_distribution :: AD
    sponge :: US

    # Linearization basic state ``Uᴸ`` — Πᴸ and θᴸ derived from the live
    # `model.dynamics.pressure`, `model.dynamics.density`, and the
    # formulation's prognostic ρθ field. Those three model fields are
    # untouched by the substep loop, so they hold ρᴸ, ρθᴸ, pᴸ throughout
    # and also serve as the recovery base in `_recover_full_state!`.
    linearization_exner :: CF
    linearization_potential_temperature :: CF

    # `linearization_gamma_R_mixture[i,j,k] = γᵐ(i,j,k) · Rᵐ(i,j,k)` enters
    # the linearised PGF (`γᵈRᵈ → γᵐRᵐᴸ`) for moist thermodynamics.
    # Recomputed from the live moisture state (vapor, liquid, ice mass
    # fractions read from `specific_prognostic_moisture(model)` and
    # `model.microphysical_fields`) at each per-stage refresh; for dry runs
    # the moisture is identically zero and γᵐRᵐᴸ collapses to γᵈRᵈ exactly.
    linearization_gamma_R_mixture :: CF

    density_perturbation :: CF
    density_potential_temperature_perturbation :: CF
    momentum_perturbation :: MP

    density_predictor :: CF
    density_potential_temperature_predictor :: CF
    previous_density_potential_temperature_perturbation :: CF

    # Time-averaged velocities for non-acoustic scalar advection (WRF/MPAS
    # dynamics-transport split).
    # Read by `transport_velocities(model)` for `AcousticRungeKutta3`-equipped
    # models, so moisture/tracer/chemistry/TKE tendencies advect against the
    # acoustic-mean velocity rather than a snapshot. The slow thermodynamic
    # `ρθ` tendency deliberately uses `model.velocities` instead. Dual-purpose
    # storage:
    #   - At outer-step start, `freeze_linearization_state!` copies
    #     `model.velocities` into these fields (so stage 1's non-acoustic
    #     scalar tendencies use outer-step-start velocities).
    #   - At stage start, `initialize_stage_perturbations!` zeros them; from
    #     here through the end of the substep loop they hold the running sum
    #     of `momentum_perturbation` (units: momentum, not velocity).
    #   - At end of the substep loop, `finalize_time_averaged_velocity!`
    #     overwrites them with `(model.momentum.* + accum/Nτ) / ρᴸ_face`.
    time_averaged_velocities :: TAV

    slow_vertical_momentum_tendency :: GT
    vertical_solver :: TS
end

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.substeps,
                       a.acoustic_cfl,
                       a.forward_weight,
                       a.thermodynamic_tendency_factor,
                       a.vertical_momentum_tendency_factor,
                       a.vertical_pressure_tendency_factor,
                       a.final_stage_vertical_pressure_tendency_factor,
                       a.apply_first_substep_pressure_gradient,
                       adapt(to, a.damping),
                       a.substep_distribution,
                       adapt(to, a.sponge),
                       adapt(to, a.linearization_exner),
                       adapt(to, a.linearization_potential_temperature),
                       adapt(to, a.linearization_gamma_R_mixture),
                       adapt(to, a.density_perturbation),
                       adapt(to, a.density_potential_temperature_perturbation),
                       (u = adapt(to, a.momentum_perturbation.u),
                        v = adapt(to, a.momentum_perturbation.v),
                        w = adapt(to, a.momentum_perturbation.w)),
                       adapt(to, a.density_predictor),
                       adapt(to, a.density_potential_temperature_predictor),
                       adapt(to, a.previous_density_potential_temperature_perturbation),
                       (u = adapt(to, a.time_averaged_velocities.u),
                        v = adapt(to, a.time_averaged_velocities.v),
                        w = adapt(to, a.time_averaged_velocities.w)),
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
    acoustic_cfl = convert(FT, split_explicit.acoustic_cfl)
    thermodynamic_tendency_factor = convert(FT, split_explicit.thermodynamic_tendency_factor)
    vertical_momentum_tendency_factor = convert(FT, split_explicit.vertical_momentum_tendency_factor)
    vertical_pressure_tendency_factor = convert(FT, split_explicit.vertical_pressure_tendency_factor)
    final_stage_vertical_pressure_tendency_factor =
        convert(FT, split_explicit.final_stage_vertical_pressure_tendency_factor)
    apply_first_substep_pressure_gradient = split_explicit.apply_first_substep_pressure_gradient
    damping = split_explicit.damping
    sponge = split_explicit.sponge
    substep_distribution = split_explicit.substep_distribution

    # Linearization basic state — Πᴸ, θᴸ derived from live model fields.
    linearization_exner                           = CenterField(grid)
    linearization_potential_temperature           = CenterField(grid)

    # γᵐRᵐᴸ — the only cached moisture quantity. Recomputed once per stage
    # refresh from the live moisture state.
    linearization_gamma_R_mixture                 = CenterField(grid)

    # Perturbation prognostics. Inherit BCs from the prognostic momenta
    # so impenetrability propagates onto the perturbation momenta.
    bcs_ρu = prognostic_momentum === nothing ? nothing : prognostic_momentum.ρu.boundary_conditions
    bcs_ρv = prognostic_momentum === nothing ? nothing : prognostic_momentum.ρv.boundary_conditions
    bcs_ρw = prognostic_momentum === nothing ? nothing : prognostic_momentum.ρw.boundary_conditions

    xface(grid, bcs) = bcs === nothing ? XFaceField(grid) : XFaceField(grid; boundary_conditions = bcs)
    yface(grid, bcs) = bcs === nothing ? YFaceField(grid) : YFaceField(grid; boundary_conditions = bcs)
    zface(grid, bcs) = bcs === nothing ? ZFaceField(grid) : ZFaceField(grid; boundary_conditions = bcs)

    density_perturbation                          = CenterField(grid)
    density_potential_temperature_perturbation    = CenterField(grid)
    momentum_perturbation = (u = xface(grid, bcs_ρu),
                             v = yface(grid, bcs_ρv),
                             w = zface(grid, bcs_ρw))

    density_predictor                                = CenterField(grid)
    density_potential_temperature_predictor          = CenterField(grid)
    previous_density_potential_temperature_perturbation = CenterField(grid)

    # Time-averaged velocities for scalar transport. Inherit BCs from the
    # prognostic momenta so impenetrability is enforced when these are used
    # for advection at boundaries.
    time_averaged_velocities = (u = xface(grid, bcs_ρu),
                                v = yface(grid, bcs_ρv),
                                w = zface(grid, bcs_ρw))

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

    return AcousticSubstepper(Ns, acoustic_cfl, ω, thermodynamic_tendency_factor,
                              vertical_momentum_tendency_factor,
                              vertical_pressure_tendency_factor,
                              final_stage_vertical_pressure_tendency_factor,
                              apply_first_substep_pressure_gradient,
                              damping, substep_distribution,
                              sponge,
                              linearization_exner,
                              linearization_potential_temperature,
                              linearization_gamma_R_mixture,
                              density_perturbation,
                              density_potential_temperature_perturbation,
                              momentum_perturbation,
                              density_predictor,
                              density_potential_temperature_predictor,
                              previous_density_potential_temperature_perturbation,
                              time_averaged_velocities,
                              slow_vertical_momentum_tendency,
                              vertical_solver)
end

#####
##### Section 3 — Stage-entry linearization
#####

"""
$(TYPEDSIGNATURES)

Compute the background quantities used by the substepper as the first
linearization point of an outer step. Subsequent RK stages call
[`prepare_acoustic_cache!`](@ref), which refreshes the same cached
quantities to the stage-entry state.

After this call:
  - `linearization_exner`                 = Πᴸ = (pᴸ/pˢᵗ)^κ derived from `model.dynamics.pressure`
  - `linearization_potential_temperature` = θᴸ = ρθᴸ/ρᴸ derived from `model.dynamics.density` + ρθ
"""
function freeze_linearization_state!(substepper::AcousticSubstepper, model)
    refresh_linearization_basic_state!(substepper, model)
    velocities = outer_step_start_transport_velocities(model)

    # Seed the time-averaged velocity with the outer-step-start velocities so
    # stage 1's non-acoustic scalar tendencies (which call
    # `transport_velocities(model)` before any substep loop has run) see the
    # outer-step-start state. Terrain-following models seed the vertical
    # component with w̃. Subsequent stages overwrite this with the previous
    # stage's substep-loop time average.
    parent(substepper.time_averaged_velocities.u) .= parent(velocities.u)
    parent(substepper.time_averaged_velocities.v) .= parent(velocities.v)
    parent(substepper.time_averaged_velocities.w) .= parent(velocities.w)

    return nothing
end

outer_step_start_transport_velocities(model) = model.velocities

# Refresh the cached linearization quantities (Πᴸ, θᴸ, γᵐRᵐᴸ) from the
# live model state. Called at outer-step start by `freeze_linearization_state!`
# and at every RK stage by `prepare_acoustic_cache!`. The base-state fields
# ρᴸ, ρθᴸ, pᴸ are `model.dynamics.density`, the formulation's ρθ field, and
# `model.dynamics.pressure` — read directly by the substep kernels.
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

    # θ_lin = ρθ/ρ and Π_lin = (p/pˢᵗ)^κ from the live model state.
    # `model.dynamics.density`, `model.dynamics.pressure`, and `ρθ_field`
    # are not mutated by the substep loop, so they stay equal to ρᴸ, pᴸ,
    # ρθᴸ throughout the stage and double as the recovery base.
    launch!(arch, grid, :xyz, _compute_linearization_exner_and_theta!,
            substepper.linearization_exner,
            substepper.linearization_potential_temperature,
            model.dynamics.pressure,
            model.dynamics.density,
            ρθ_field,
            pˢᵗ, κ)

    # The horizontal pressure-gradient force in `_explicit_horizontal_step!`
    # uses ∂x(pᴸ) directly. With `ExnerReferenceState` the reference depends
    # only on z so ∂x pᵣ ≡ 0, and ∂x(pᴸ − pᵣ) = ∂x pᴸ; with no reference
    # state pᵣ = 0. In both cases no separate `pressure_perturbation` field
    # is needed for the horizontal direction. Vertical reference subtraction
    # for the slow tendency is handled by `assemble_slow_vertical_momentum_tendency!`.

    # γᵐRᵐᴸ recomputed in-place from the live moisture state via
    # `grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛ, μ)`,
    # which dispatches dry/moist transparently. For dry runs (qᵛ = qˡ = qⁱ = 0)
    # this collapses to γᵈ Rᵈ exactly.
    launch!(arch, grid, :xyz, _compute_linearization_mixture_eos!,
            substepper.linearization_gamma_R_mixture,
            grid,
            model.microphysics,
            model.dynamics.density,
            specific_prognostic_moisture(model),
            model.microphysical_fields,
            Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, cˡ, cⁱ)

    fill_halo_regions!(substepper.linearization_exner)
    fill_halo_regions!(substepper.linearization_potential_temperature)
    fill_halo_regions!(substepper.linearization_gamma_R_mixture)

    return nothing
end

@kernel function _compute_linearization_exner_and_theta!(Π, θ, p, ρ, ρθ, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Π[i, j, k] = (p[i, j, k] / pˢᵗ)^κ
        ρ̂ = ifelse(ρ[i, j, k] == 0, one(eltype(ρ)), ρ[i, j, k])
        θ[i, j, k] = ρθ[i, j, k] / ρ̂
    end
end

# Compute γᵐRᵐᴸ per cell from the live moisture state.
#   Rᵐ  = qᵈ Rᵈ + qᵛ Rᵛ                         (mixture gas constant)
#   cᵖᵐ = qᵈ cᵖᵈ + qᵛ cᵖᵛ + qˡ cˡ + qⁱ cⁱ      (mixture heat capacity)
#   cᵛᵐ = cᵖᵐ − Rᵐ
#   γᵐ  = cᵖᵐ / cᵛᵐ
# with qᵈ = 1 − qᵛ − qˡ − qⁱ. `grid_moisture_fractions` dispatches on the
# microphysics scheme to extract (qᵛ, qˡ, qⁱ) at this cell — for dry runs
# the returned fractions are vapor-only with qᵛ = 0 (qᵛ field is zeroed),
# and γᵐRᵐ collapses to γᵈRᵈ exactly.
@kernel function _compute_linearization_mixture_eos!(γRᵐ, grid, microphysics, ρ, qᵛ, μ,
                                                     Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, cˡ, cⁱ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵢ  = ρ[i, j, k]
        qᵛᵢ = qᵛ[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρᵢ, qᵛᵢ, μ)
    @inbounds begin
        qᵈᵢ = 1 - q.vapor - q.liquid - q.ice

        Rᵐ  = qᵈᵢ * Rᵈ + q.vapor * Rᵛ
        cᵖᵐ = qᵈᵢ * cᵖᵈ + q.vapor * cᵖᵛ + q.liquid * cˡ + q.ice * cⁱ
        cᵛᵐ = cᵖᵐ - Rᵐ

        # Operation order matches the dry-only path's `cᵖᵈ * Rᵈ / (cᵖᵈ - Rᵈ)`
        # so qᵛ = qˡ = qⁱ = 0 reproduces the dry γᵈRᵈ to bit-identical precision.
        γRᵐ[i, j, k] = cᵖᵐ * Rᵐ / cᵛᵐ
    end
end

"""
$(TYPEDSIGNATURES)

Stage-start cache preparation. Refreshes the cached linearization
quantities (Πᴸ, θᴸ, γᵐRᵐᴸ) to the **stage-entry state** ``Uᴸ_\\mathrm{stage}``
(per [Skamarock & Klemp 2008](@cite SkamarockKlemp2008) above eq. 16),
recomputing them from the live `model.dynamics.*`. The rewind-perturbation
initialization (`initialize_stage_perturbations!`, called next) handles
the WS-RK3 invariant by setting ``(ρ)′_\\mathrm{init} = Uᴸ_\\mathrm{outer} − Uᴸ_\\mathrm{stage}``
(zero for stage 1; nonzero for stages 2 and 3).
"""
prepare_acoustic_cache!(substepper::AcousticSubstepper, model) =
    refresh_linearization_basic_state!(substepper, model)

#####
##### Section 4 — Adaptive substep computation (acoustic CFL)
#####

"""
$(TYPEDSIGNATURES)

Compute the number of acoustic substeps ``N`` from the horizontal
acoustic CFL:

```math
N \\approx
\\left\\lceil \\frac{\\Delta t \\, \\mathbb{C}^{ac}}{\\nu \\, \\Delta x_\\min} \\right\\rceil ,
```

with ``\\mathbb{C}^{ac} = \\sqrt{γ^d R^d T_r}`` for a nominal reference
temperature ``T_r = 300\\,\\mathrm{K}`` and ``ν`` the target acoustic
Courant number `acoustic_cfl` (default `0.5`, the ERF/WRF target —
equivalent to the conventional safety factor of `2`).
"""
function compute_acoustic_substeps(grid, Δt, thermodynamic_constants, acoustic_cfl)
    FT   = eltype(grid)
    Rᵈ   = convert(FT, dry_air_gas_constant(thermodynamic_constants))
    cᵖᵈ  = convert(FT, thermodynamic_constants.dry_air.heat_capacity)
    γᵈ   = cᵖᵈ / (cᵖᵈ - Rᵈ)
    ℂᵃᶜ  = sqrt(γᵈ * Rᵈ * FT(300))
    ν    = convert(FT, acoustic_cfl)

    Δx_min = let
        TX, TY, _ = topology(grid)
        Δx = TX === Flat ? typemax(FT) : minimum_xspacing(grid)
        Δy = TY === Flat ? typemax(FT) : minimum_yspacing(grid)
        min(Δx, Δy)
    end

    return max(1, ceil(Int, FT(Δt) * ℂᵃᶜ / (ν * Δx_min)))
end

@inline acoustic_substeps(N::Int, grid, Δt, constants, acoustic_cfl) = N
@inline acoustic_substeps(::Nothing, grid, Δt, constants, acoustic_cfl) =
    compute_acoustic_substeps(grid, Δt, constants, acoustic_cfl)

#####
##### Section 5 — Stage substep distribution
#####

# ProportionalSubsteps: every stage uses Δτ = Δt/N, Nτ = round(β·N).
@inline function stage_substep_count_and_size(::ProportionalSubsteps, β_stage, Δt, N)
    Δτ = Δt / N
    Nτ = max(1, round(Int, β_stage * N))
    return Nτ, Δτ
end

# MonolithicFirstStage: stage 1 collapses to one substep of size Δt/3;
# stages 2 and 3 are the same as ProportionalSubsteps.
@inline function stage_substep_count_and_size(::MonolithicFirstStage, β_stage, Δt, N)
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
#                - Δτ × (ωˢ⁻ ∂z p′_o + ωᵐ⁺ ∂z p′_n)
#                - Δτ × g × (ωˢ⁻ ρ′_face_o(k) + ωᵐ⁺ ρ′_face_n(k))
#
# with ωᵐ⁺ = (1+ε)/2, ωˢ⁻ = (1-ε)/2 (ε=0 is centered CN).
# The linearized pressure perturbation is local:
#   p′ = Cᴸ (ρθ)′,  Cᴸ ≡ γRᵐᴸ Πᴸ,
# so the discrete pressure-gradient force is the gradient of the product
# `Cᴸ * (ρθ)′`, not `Cᴸ_face * ∂z(ρθ)′`.
#
# The post-solve substitution (matching the column kernel):
#   ρ′_n(k)    = ρ′★(k)  - δτᵐ⁺ × ((ρw)′_n(k+1) - (ρw)′_n(k)) / Δz_c(k)
#   (ρθ)′_n(k) = ρθ′★(k) - δτᵐ⁺ × (θᴸ_face(k+1) (ρw)′_n(k+1)
#                                        - θᴸ_face(k)   (ρw)′_n(k)) / Δz_c(k)
# where δτᵐ⁺ = ωᵐ⁺ Δτ.
#
# Substituting yields the tridiagonal coefficients (ω = ωᵐ⁺):
#
#   A[k,k+1] = -(ω Δτ)² × Cᴸ(k)   × θᴸ_face(k+1) × rdz_c(k)   / Δzᶠ(k)
#              - (ω Δτ)² × g          × rdz_c(k)   / 2
#
#   A[k,k]   = 1 + (ω Δτ)² × θᴸ_face(k) × (Cᴸ(k) rdz_c(k) + Cᴸ(k-1) rdz_c(k-1)) / Δzᶠ(k)
#                + (ω Δτ)² × g                              × (rdz_c(k) - rdz_c(k-1)) / 2
#
#   A[k,k-1] = -(ω Δτ)² × Cᴸ(k-1) × θᴸ_face(k-1) × rdz_c(k-1) / Δzᶠ(k)
#              + (ω Δτ)² × g                              × rdz_c(k-1) / 2
#
# `γᵐRᵐᴸ` is the cell-centered mixture coefficient `γᵐ Rᵐ` cached in
# `linearization_gamma_R_mixture`, refreshed once per stage from the live
# moisture state. It is interpolated to z-faces inside the kernel. For
# dry runs (qᵛ = qˡ = qⁱ = 0) this collapses bit-identically to the dry
# constant `γᵈRᵈ`.
#
# Implicit vertical damping
# -------------------------
# When `damping isa ThermalDivergenceDamping` with `damp_vertical = true`,
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
# right-hand side in `_build_predictors_and_vertical_rhs!`. The constant-
# Courant scaling `γ_z = α Δz² / Δτ` makes `dᵐ⁺` and the RHS prefactor
# independent of Δτ; only `α`, `ω`, and `minimum_zspacing(grid)` enter.
# When `damp_vertical = false` (or for `NoDivergenceDamping`), the damping
# factor passed in is zero and the tridiag reduces to the pure off-
# centered CN acoustic system above.

# Implicit upper Rayleigh sponge contribution to the column tridiag's
# diagonal. Klemp, Dudhia & Hassiotis (2008): a layer of thickness `depth`
# below the lid where the vertical acoustic momentum perturbation is damped
# at peak rate `damping_rate` (1/s) scaled by the configured ramp shape. The
# ramp uses the reference vertical face coordinate so terrain-following grids
# get a horizontally uniform sponge in ζ. CN-weighted: `δτᵐ⁺ × rate × ramp`
# on the LHS diagonal, matched by `δτˢ⁻ × rate × ramp × ρw_old` subtracted
# on the RHS in `_build_predictors_and_vertical_rhs!`. Local in the vertical
# coordinate, so no off-diagonal coupling.
@inline sponge_term_diag(i, j, k, grid, ::Nothing, δτᵐ⁺) = zero(grid)

@inline function sponge_term_diag(i, j, k, grid, sponge::UpperSponge, δτᵐ⁺)
    z = rnode(k, grid, Face())
    return δτᵐ⁺ * sponge.damping_rate *
           sponge.ramp(z, grid.Lz, sponge.depth)
end

@inline sponge_rhs(i, j, k, grid, ::Nothing, δτˢ⁻, ρw_old) = zero(grid)

@inline function sponge_rhs(i, j, k, grid, sponge::UpperSponge, δτˢ⁻, ρw_old)
    z = rnode(k, grid, Face())
    @inbounds return δτˢ⁻ * sponge.damping_rate *
                     sponge.ramp(z, grid.Lz, sponge.depth) * ρw_old[i, j, k]
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagLower, p, ::ZDirection,
                                 Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
    kᶠ      = k + 1
    Δzᶠ     = Δzᶜᶜᶠ(i, j, kᶠ, grid)
    Δz⁻¹ᵏ⁻  = 1 / Δzᶜᶜᶜ(i, j, kᶠ - 1, grid)

    @inbounds Cᵏ⁻ = γRᵐᴸ[i, j, kᶠ - 1] * Πᴸ[i, j, kᶠ - 1]
    θᵏ⁻     = ℑbzᵃᵃᶠ(i, j, kᶠ - 1, grid, θᴸ)

    pgf_term  = - δτᵐ⁺^2 * Cᵏ⁻ * θᵏ⁻ * Δz⁻¹ᵏ⁻ / Δzᶠ
    buoy_term = + δτᵐ⁺^2 * g                    * Δz⁻¹ᵏ⁻ / 2
    damp_term = - dᵐ⁺                           * Δz⁻¹ᵏ⁻ / Δzᶠ
    # Upper sponge is local in z (Rayleigh-type), so no off-diagonal coupling.
    return pgf_term + buoy_term + damp_term
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagDiagonal, p, ::ZDirection,
                                 Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)

    kᶠ      = k
    Δzᶠ     = Δzᶜᶜᶠ(i, j, kᶠ, grid)
    Δz⁻¹ᵏ⁺  = 1 / Δzᶜᶜᶜ(i, j, kᶠ,     grid)
    Δz⁻¹ᵏ⁻  = 1 / Δzᶜᶜᶜ(i, j, kᶠ - 1, grid)

    @inbounds begin
        Cᵏ⁺ = γRᵐᴸ[i, j, kᶠ]     * Πᴸ[i, j, kᶠ]
        Cᵏ⁻ = γRᵐᴸ[i, j, kᶠ - 1] * Πᴸ[i, j, kᶠ - 1]
    end
    θᶜᶜᶠ    = ℑbzᵃᵃᶠ(i, j, kᶠ, grid, θᴸ)

    pgf_diag    = δτᵐ⁺^2 * θᶜᶜᶠ * (Cᵏ⁺ * Δz⁻¹ᵏ⁺ + Cᵏ⁻ * Δz⁻¹ᵏ⁻) / Δzᶠ
    buoy_diag   = δτᵐ⁺^2 * g                     * (Δz⁻¹ᵏ⁺ - Δz⁻¹ᵏ⁻) / 2
    damp_diag   = dᵐ⁺                            * (Δz⁻¹ᵏ⁺ + Δz⁻¹ᵏ⁻) / Δzᶠ
    spnge_diag  = sponge_term_diag(i, j, k, grid, sponge, δτᵐ⁺)

    return one(grid) + (pgf_diag + buoy_diag + damp_diag + spnge_diag) * (k > 1)
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagUpper, p, ::ZDirection,
                                 Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)

    kᶠ      = k
    Δzᶠ     = Δzᶜᶜᶠ(i, j, kᶠ, grid)
    Δz⁻¹ᵏ⁺  = 1 / Δzᶜᶜᶜ(i, j, kᶠ, grid)

    @inbounds Cᵏ⁺ = γRᵐᴸ[i, j, kᶠ] * Πᴸ[i, j, kᶠ]
    θᵏ⁺     = ℑbzᵃᵃᶠ(i, j, kᶠ + 1, grid, θᴸ)

    pgf_term  = - δτᵐ⁺^2 * Cᵏ⁺ * θᵏ⁺ * Δz⁻¹ᵏ⁺ / Δzᶠ
    buoy_term = - δτᵐ⁺^2 * g                    * Δz⁻¹ᵏ⁺ / 2
    damp_term = - dᵐ⁺                           * Δz⁻¹ᵏ⁺ / Δzᶠ
    # Upper sponge is local in z (Rayleigh-type), so no off-diagonal coupling.

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
#####   Gˢρw = -∇·(ρw u)  -  ∂z(pᴸ - pᵣ)  -  g · (ρᴸ - ρᵣ)   (with reference)
#####   Gˢρw = -∇·(ρw u)  -  ∂z pᴸ        -  g · ρᴸ           (no reference)
##### and the per-substep linearized forces operate on the perturbations:
#####   ∂t (ρw)′ = Gˢρw - γRᵐ · Πᴸ · ∂z((ρθ)′)  -  g · ρ′
##### Total force = Gˢρw + perturbation force = full ∂t(ρw) at the
##### linearization-consistent level. With a hydrostatic-balanced reference
##### state, the reference subtraction makes Gˢρw vanish identically on a
##### resting atmosphere (no FP-rounding noise).
#####

function assemble_slow_vertical_momentum_tendency!(substepper::AcousticSubstepper, model, β_stage = nothing)
    grid = model.grid
    arch = architecture(grid)
    g    = convert(eltype(grid), model.thermodynamic_constants.gravitational_acceleration)
    Gⁿρw = model.timestepper.Gⁿ.ρw

    terrain_reference_pressure = model.dynamics.terrain_reference_pressure
    terrain_reference_density = model.dynamics.terrain_reference_density
    ref = model.dynamics.reference_state

    if terrain_reference_pressure !== nothing && terrain_reference_density !== nothing
        launch!(arch, grid, :xyz, _assemble_slow_vertical_momentum_tendency!,
                substepper.slow_vertical_momentum_tendency,
                Gⁿρw,
                model.dynamics.pressure,
                model.dynamics.density,
                terrain_reference_pressure, terrain_reference_density,
                grid, g)
    elseif ref isa Nothing
        launch!(arch, grid, :xyz, _assemble_slow_vertical_momentum_tendency_no_ref!,
                substepper.slow_vertical_momentum_tendency,
                Gⁿρw,
                model.dynamics.pressure,
                model.dynamics.density,
                grid, g)
    else
        launch!(arch, grid, :xyz, _assemble_slow_vertical_momentum_tendency!,
                substepper.slow_vertical_momentum_tendency,
                Gⁿρw,
                model.dynamics.pressure,
                model.dynamics.density,
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
        ∂z_p′ = ∂zᶜᶜᶠ(i, j, k, grid, p_perturbation, pᴸ, pᵣ)
        ρ′ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ_perturbation, ρᴸ, ρᵣ)

        Gˢρw[i, j, k] = (Gⁿρw[i, j, k] - ∂z_p′ - g * ρ′ᶜᶜᶠ) * (k > 1)
    end
end

@inline p_perturbation(i, j, k, grid, pᴸ, pᵣ) = @inbounds pᴸ[i, j, k] - pᵣ[i, j, k]
@inline ρ_perturbation(i, j, k, grid, ρᴸ, ρᵣ) = @inbounds ρᴸ[i, j, k] - ρᵣ[i, j, k]

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

# Initialize perturbation prognostics at the start of each WS-RK3 stage.
# Per Skamarock & Klemp 2008 (above eq. 16), the substep variables are
# deviations from the **most recent RK3 predictor** (= linearization base
# Uᴸ, refreshed by `prepare_acoustic_cache!` immediately before this).
# But the WS-RK3 invariant ``U^{(k)} = U(t) + β_k Δt R(U^{(k-1)})``
# requires each stage to integrate from U(t) (= the outer-step-start
# state held in `model.timestepper.U⁰` ≡ `Uᴸ_outer`). The standard trick
# (WRF `small_step_prep`, MPAS) is to initialize the perturbations to
# the **rewind term** ``(U_\\mathrm{outer} − Uᴸ)`` so that the substep's
# starting full state ``Uᴸ + (U_\\mathrm{outer} − Uᴸ) = U_\\mathrm{outer}``
# regardless of where Uᴸ was refreshed to. For stage 1 the rewind is
# zero (Uᴸ = U_\\mathrm{outer}); for stages 2 and 3 the rewind picks up
# the previous-stage update. Recovery from `_recover_full_state!` then
# uses the per-stage Uᴸ as the recovery base, and the algebra collapses
# back to ``U_\\mathrm{outer} + Δevolved`` — preserving the WS-RK3 invariant.
#
# Auxiliary perturbation/workspace fields (predictors, divergence
# workspace, K18 / WRF damping snapshots) reset to zero — they don't
# carry stage-to-stage history.
function initialize_stage_perturbations!(substepper, model, Uᴸ_outer)
    grid = model.grid
    arch = architecture(grid)

    # Auxiliary workspaces: zero.
    fill!(parent(substepper.previous_density_potential_temperature_perturbation), 0)
    fill!(parent(substepper.density_predictor), 0)
    fill!(parent(substepper.density_potential_temperature_predictor), 0)

    # Time-averaged-velocity slots switch into accumulator mode for this
    # stage: zero them, then `accumulate_momentum_perturbations!` adds raw
    # `momentum_perturbation` values after each substep, and
    # `finalize_time_averaged_velocity!` normalizes at stage end.
    fill!(parent(substepper.time_averaged_velocities.u), 0)
    fill!(parent(substepper.time_averaged_velocities.v), 0)
    fill!(parent(substepper.time_averaged_velocities.w), 0)

    # Prognostic perturbations: rewind init. The per-stage Uᴸ for ρ and
    # ρθ is held in `model.dynamics.density` and the formulation's ρθ
    # field — untouched by the substep loop, so they equal the per-stage
    # linearization base.
    χ_field = thermodynamic_density(model.formulation)
    χ_name = thermodynamic_density_name(model.formulation)
    launch!(arch, grid, :xyz, _initialize_perturbation_with_rewind!,
            substepper.density_perturbation,
            Uᴸ_outer.ρ, model.dynamics.density)
    launch!(arch, grid, :xyz, _initialize_perturbation_with_rewind!,
            substepper.density_potential_temperature_perturbation,
            Uᴸ_outer[χ_name], χ_field)
    launch!(arch, grid, :xyz, _initialize_perturbation_with_rewind!,
            substepper.momentum_perturbation.u,
            Uᴸ_outer.ρu, model.momentum.ρu)
    launch!(arch, grid, :xyz, _initialize_perturbation_with_rewind!,
            substepper.momentum_perturbation.v,
            Uᴸ_outer.ρv, model.momentum.ρv)
    initialize_vertical_momentum_perturbation!(substepper, model, Uᴸ_outer)

    fill_halo_regions!(substepper.density_perturbation)
    fill_halo_regions!(substepper.density_potential_temperature_perturbation)
    fill_halo_regions!(substepper.momentum_perturbation.u)
    fill_halo_regions!(substepper.momentum_perturbation.v)
    fill_halo_regions!(substepper.momentum_perturbation.w)

    return nothing
end

@kernel function _initialize_perturbation_with_rewind!(perturbation, Uᴸ_outer, Uᴸ_stage)
    i, j, k = @index(Global, NTuple)
    @inbounds perturbation[i, j, k] = Uᴸ_outer[i, j, k] - Uᴸ_stage[i, j, k]
end

function initialize_vertical_momentum_perturbation!(substepper, model, Uᴸ_outer)
    grid = model.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _initialize_perturbation_with_rewind!,
            substepper.momentum_perturbation.w,
            Uᴸ_outer.ρw, model.momentum.ρw)
    return nothing
end

# Explicit forward step for horizontal momentum perturbations (ρu)′, (ρv)′.
#
# Linearized at Uᴸ, the full horizontal pressure gradient splits as
#   ∂x p_full = ∂x pᴸ + ∂x(Cᴸ (ρθ)′),  Cᴸ = γRᵐᴸ Πᴸ
# where the first piece is the FROZEN linearization-point pressure and the
# second is the perturbation force. `ExnerReferenceState` (when present)
# depends only on z, so ∂x pᵣ ≡ 0 and ∂x(pᴸ − pᵣ) = ∂x pᴸ — no separate
# pressure-perturbation field is needed in the horizontal direction.
# `Gⁿρu` from `SlowTendencyMode` carries non-pressure slow terms
# (advection, Coriolis, closures, and forcing; PGF zeroed); we reinstate
# the frozen horizontal pressure here.
#
# (ρu)′^{τ+Δτ} = (ρu)′^τ + Δτ · (Gⁿρu − ∂x pᴸ − ∂x(Cᴸ (ρθ)′))
# (ρv)′^{τ+Δτ} = (ρv)′^τ + Δτ · (Gⁿρv − ∂y pᴸ − ∂y(Cᴸ (ρθ)′))
# The MPAS forward-backward first-small-step sequencing skips only the
# acoustic perturbation PGF, not the frozen large-step PGF. MPAS carries that
# frozen PGF in `tend_u_euler`; here we add it explicitly because
# `SlowTendencyMode` zeros pressure gradients in `Gⁿρu/Gⁿρv`.
@kernel function _explicit_horizontal_step!(ρu′, ρv′, grid, dynamics, Δτ, ρθ′, Πᴸ, p,
                                            Gⁿρu, Gⁿρv, γRᵐᴸ, apply_pressure_gradient)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂x_pᴸ  = acoustic_x_pressure_gradient(i, j, k, grid, dynamics, p)
        ∂x_p′  = acoustic_x_linearized_pressure_gradient(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)
        ∂y_pᴸ  = acoustic_y_pressure_gradient(i, j, k, grid, dynamics, p)
        ∂y_p′  = acoustic_y_linearized_pressure_gradient(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)

        perturbation_pressure_gradient_factor = ifelse(apply_pressure_gradient, one(Δτ), zero(Δτ))
        ∂x_p = ∂x_pᴸ + perturbation_pressure_gradient_factor * ∂x_p′
        ∂y_p = ∂y_pᴸ + perturbation_pressure_gradient_factor * ∂y_p′

        ρu′[i, j, k] += Δτ * (Gⁿρu[i, j, k] - ∂x_p)
        ρv′[i, j, k] += Δτ * (Gⁿρv[i, j, k] - ∂y_p)
    end
end

@inline function cm1_style_x_exner_pressure_acceleration_components(i, j, k, grid, dynamics,
                                                                    ρθ′, Πᴸ, p, ρᴸ, θᴸ, γRᵐᴸ,
                                                                    apply_pressure_gradient,
                                                                    pˢᵗ, κ, cᵖᵈ)
    z = zero(grid)
    return z, z, z, z, z, z, z, z, z
end

@inline function cm1_style_y_exner_pressure_acceleration_components(i, j, k, grid, dynamics,
                                                                    ρθ′, Πᴸ, p, ρᴸ, θᴸ, γRᵐᴸ,
                                                                    apply_pressure_gradient,
                                                                    pˢᵗ, κ, cᵖᵈ)
    z = zero(grid)
    return z, z, z, z, z, z, z, z, z
end

@inline linearized_pressure_perturbation(i, j, k, grid, ρθ′, Πᴸ, γRᵐᴸ) =
    @inbounds γRᵐᴸ[i, j, k] * Πᴸ[i, j, k] * ρθ′[i, j, k]

@inline function full_dry_acoustic_pressure(i, j, k, grid, ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    @inbounds begin
        Rᵈ = κ * cᵖᵈ
        γᵈ = 1 / (1 - κ)
        ρθ = ρᴸ[i, j, k] * θᴸ[i, j, k] + ρθ′[i, j, k]
        return pˢᵗ * (Rᵈ * ρθ / pˢᵗ)^γᵈ
    end
end

@inline function full_dry_acoustic_pressure_perturbation(i, j, k, grid,
                                                         ρθ′, ρᴸ, θᴸ, pᵣ, pˢᵗ, κ, cᵖᵈ)
    @inbounds return full_dry_acoustic_pressure(i, j, k, grid, ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ) -
                      pᵣ[i, j, k]
end

@inline acoustic_x_pressure_gradient(i, j, k, grid, dynamics, p) = ∂xᶠᶜᶜ(i, j, k, grid, p)
@inline acoustic_y_pressure_gradient(i, j, k, grid, dynamics, p) = ∂yᶜᶠᶜ(i, j, k, grid, p)

@inline function acoustic_x_pressure_gradient_components(i, j, k, grid, dynamics, p)
    return acoustic_x_pressure_gradient(i, j, k, grid, dynamics, p), zero(grid)
end

@inline function acoustic_y_pressure_gradient_components(i, j, k, grid, dynamics, p)
    return acoustic_y_pressure_gradient(i, j, k, grid, dynamics, p), zero(grid)
end

@inline acoustic_z_linearized_pressure_gradient(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ) =
    ∂zᶜᶜᶠ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)

@inline acoustic_x_linearized_pressure_gradient(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ) =
    ∂xᶠᶜᶜ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)

@inline acoustic_y_linearized_pressure_gradient(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ) =
    ∂yᶜᶠᶜ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)

@inline function acoustic_x_linearized_pressure_gradient_components(i, j, k, grid,
                                                                    dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    return acoustic_x_linearized_pressure_gradient(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ),
           zero(grid)
end

@inline function acoustic_y_linearized_pressure_gradient_components(i, j, k, grid,
                                                                    dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    return acoustic_y_linearized_pressure_gradient(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ),
           zero(grid)
end

@inline function acoustic_x_full_dry_pressure_gradient(i, j, k, grid, dynamics,
                                                       ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    return ∂xᶠᶜᶜ(i, j, k, grid, full_dry_acoustic_pressure,
                 ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
end

@inline function acoustic_y_full_dry_pressure_gradient(i, j, k, grid, dynamics,
                                                       ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    return ∂yᶜᶠᶜ(i, j, k, grid, full_dry_acoustic_pressure,
                 ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
end

@inline function acoustic_x_full_dry_pressure_gradient_components(i, j, k, grid, dynamics,
                                                                  ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    return acoustic_x_full_dry_pressure_gradient(i, j, k, grid, dynamics,
                                                 ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ),
           zero(grid)
end

@inline function acoustic_y_full_dry_pressure_gradient_components(i, j, k, grid, dynamics,
                                                                  ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    return acoustic_y_full_dry_pressure_gradient(i, j, k, grid, dynamics,
                                                 ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ),
           zero(grid)
end

@inline apply_horizontal_pressure_gradient_substep(substep, Nτ, apply_first_substep_pressure_gradient) =
    apply_first_substep_pressure_gradient | (substep != 1) | (Nτ == 1)

# Build per-column predictors `ρ′★`, `ρθ′★` (cell centers) AND
# the explicit RHS for the tridiagonal `(ρw)′ᵐ⁺` solve at z-faces.
#
# Off-centered Crank–Nicolson with new-side weight ω = forward_weight
# and old-side weight 1−ω. The predictor uses δτˢ⁻ = (1−ω)Δτ on the
# old-step vertical-flux contribution (ω-weighted CN of ∇·m); the
# vertical RHS combines old and pred contributions with their matching
# weights δτˢ⁻ and δτᵐ⁺ respectively. See derivation in
# the split-explicit derivation in `docs/src/compressible_dynamics.md`.
@kernel function _build_predictors_and_vertical_rhs!(ρw′_rhs,
                                                     ρ′★, ρθ′★,
                                                     ρ′, ρθ′, ρw′, ρu′, ρv′,
                                                     grid, dynamics, Δτ, δτᵐ⁺, δτˢ⁻,
                                                     Gˢρ, Gˢρθ, Gˢρw,
                                                     thermodynamic_tendency_factor,
                                                     vertical_momentum_tendency_factor,
                                                     θᴸ, Πᴸ,
                                                     γRᵐᴸ, g, dˢ⁻, sponge)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Cell-centred predictors `ρ′★`, `ρθ′★`.
        for k in 1:Nz
            V = Vᶜᶜᶜ(i, j, k, grid)

            ∇ʰ_M  = div_xyᶜᶜᶜ(i, j, k, grid, ρu′, ρv′)
            ∇ʰ_θM = (δxᶜᵃᵃ(i, j, k, grid, theta_face_x_flux, θᴸ, ρu′) +
                     δyᵃᶜᵃ(i, j, k, grid, theta_face_y_flux, θᴸ, ρv′)) / V

            ρ′★[i, j, k]  = ρ′[i, j, k] +
                                Δτ * (Gˢρ[i, j, k] - ∇ʰ_M) -
                                δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, acoustic_vertical_momentum_flux,
                                               dynamics, ρu′, ρv′, ρw′)

            ρθ′★[i, j, k] = ρθ′[i, j, k] +
                                Δτ * (thermodynamic_tendency_factor * Gˢρθ[i, j, k] - ∇ʰ_θM) -
                                δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, theta_face_z_flux,
                                               θᴸ, dynamics, ρu′, ρv′, ρw′)
        end

        # Face-level RHS for `(ρw)′ᵐ⁺` tridiag — split weights for the
        # predictor and old-step contributions per derivation (15).
        # `dˢ⁻ = (1−ω) α Δz²` adds the explicit half of the implicit
        # vertical damping (zero when damping is off or damp_vertical=false).
        for k in 2:Nz
            Δzᶠ   = Δzᶜᶜᶠ(i, j, k, grid)
            ∂ζ_p′★ = acoustic_z_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                              ρθ′★, Πᴸ, γRᵐᴸ)
            ∂ζ_p′ˢ⁻ = acoustic_z_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                               ρθ′, Πᴸ, γRᵐᴸ)

            sound_force = δτˢ⁻ * ∂ζ_p′ˢ⁻ + δτᵐ⁺ * ∂ζ_p′★

            ρ′ᶜᶜᶠ★  = ℑzᵃᵃᶠ(i, j, k, grid, ρ′★)
            ρ′ᶜᶜᶠˢ⁻ = ℑzᵃᵃᶠ(i, j, k, grid, ρ′)
            buoy_force = g * (δτˢ⁻ * ρ′ᶜᶜᶠˢ⁻ + δτᵐ⁺ * ρ′ᶜᶜᶠ★)

            # Explicit (old-step) half of the vertical damping
            # `(1−ω) α Δz² ∂z²(ρw)′ˢ⁻`, evaluated at face k. The face-coupling
            # stencil matches the implicit half folded into the tridiag in
            # `get_coefficient`.
            ∂z²_ρw′ˢ⁻  = ∂zᶜᶜᶠ(i, j, k, grid, ∂zᶜᶜᶜ, ρw′)
            damp_force = - dˢ⁻ * ∂z²_ρw′ˢ⁻

            # Explicit (old-step) half of the upper Rayleigh sponge:
            # `(1−ω) Δτ × rate × ramp(z) × (ρw)′ˢ⁻` = `δτˢ⁻ × rate × ramp(z) × (ρw)′ˢ⁻`.
            # The matching implicit half on the LHS lives in `get_coefficient`'s
            # diagonal contribution. Local in z, so face-only.
            sponge_force = sponge_rhs(i, j, k, grid, sponge, δτˢ⁻, ρw′)

            ρw′_rhs[i, j, k] =
                ρw′[i, j, k] + Δτ * vertical_momentum_tendency_factor * Gˢρw[i, j, k] -
                               sound_force - buoy_force - damp_force - sponge_force
        end

        # Boundary-row RHS values: f[1] = 0 (matches diagonal b[1] = 1 → (ρw)′[1] = 0).
        ρw′_rhs[i, j, 1] = 0
        # Top face (Nz+1) lives outside the solver; impenetrability w(top) = 0.
        ρw′_rhs[i, j, Nz + 1] = 0
    end
end

# θᴸ · (ρu)′ at an x-face. Used in the area-weighted horizontal
# divergence of the perturbation θ-flux.
@inline theta_face_x_flux(i, j, k, grid, θᴸ, ρu′) =
    @inbounds Axᶠᶜᶜ(i, j, k, grid) * ℑxᶠᵃᵃ(i, j, k, grid, θᴸ) * ρu′[i, j, k]

@inline theta_face_y_flux(i, j, k, grid, θᴸ, ρv′) =
    @inbounds Ayᶜᶠᶜ(i, j, k, grid) * ℑyᵃᶠᵃ(i, j, k, grid, θᴸ) * ρv′[i, j, k]

# θᴸ · (ρw)′ at a z-face. Used in the vertical part of the perturbation
# θ-flux divergence; passed to `∂zᶜᶜᶜ` so the divergence is computed at
# cell centers from the face-located product.
@inline acoustic_vertical_momentum_flux(i, j, k, grid, dynamics, ρu′, ρv′, ρw′) =
    @inbounds ρw′[i, j, k]

@inline theta_face_z_flux(i, j, k, grid, θᴸ, dynamics, ρu′, ρv′, ρw′) =
    @inbounds ℑbzᵃᵃᶠ(i, j, k, grid, θᴸ) *
              acoustic_vertical_momentum_flux(i, j, k, grid, dynamics, ρu′, ρv′, ρw′)

@inline ℑb_wθ(i, j, k, grid, w, θ) = @inbounds w[i, j, k] * ℑbzᵃᵃᶠ(i, j, k, grid, θ)

# Post-solve recovery: substitute the tridiag-solved `(ρw)′ᵐ⁺` back
# into the `ρ′★`, `ρθ′★` predictors to get `ρ′ᵐ⁺`, `ρθ′ᵐ⁺`
# (the IMPLICIT half of CN).
#
#   ρ′_n(k)    = ρ′★(k)  - (δτᵐ⁺ / Δz_c(k)) · ((ρw)′_n(k+1) - (ρw)′_n(k))
#   (ρθ)′_n(k) = ρθ′★(k) - (δτᵐ⁺ / Δz_c(k)) · (θᴸ_face(k+1) (ρw)′_n(k+1)
#                                                    - θᴸ_face(k)   (ρw)′_n(k))
@kernel function _post_solve_recovery!(ρ′, ρθ′, ρw′, ρu′, ρv′, ρ′★, ρθ′★,
                                       grid, dynamics, δτᵐ⁺, θᴸ)
    i, j, k = @index(Global, NTuple)
    ρ′[i, j, k] = ρ′★[i, j, k] -
                  δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, acoustic_vertical_momentum_flux,
                                 dynamics, ρu′, ρv′, ρw′)
    ρθ′[i, j, k] = ρθ′★[i, j, k] -
                   δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, theta_face_z_flux,
                                  θᴸ, dynamics, ρu′, ρv′, ρw′)
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
@inline implicit_damping_factors(::AcousticDampingStrategy, ω, one_minus_ω, grid, FT) =
    (zero(FT), zero(FT))

@inline function implicit_damping_factors(damping::ThermalDivergenceDamping, ω, one_minus_ω, grid, FT)
    damping.damp_vertical || return (zero(FT), zero(FT))
    α    = convert(FT, damping.coefficient)
    Δz   = convert(FT, minimum_zspacing(grid))
    base = α * Δz^2
    return (convert(FT, ω) * base, convert(FT, one_minus_ω) * base)
end

# Klemp, Skamarock & Ha (2018) acoustic divergence damping (MPAS form).
# In the linearized acoustic mode,
#   (ρθ)′ − (ρθ)′ˢ⁻ ≈ −Δτ · θᴸ · ∇·((ρu)′, (ρv)′, (ρw)′)
# so D ≡ ((ρθ)′ − (ρθ)′ˢ⁻) / θᴸ is a discrete proxy for −Δτ · ∇·(ρu)′.
# The default per-substep momentum correction is horizontal:
#   Δ(ρu)′ = −γ · ∂x D , Δ(ρv)′ = −γ · ∂y D
# with per-direction horizontal diffusivities:
#   γˣ = α · Δx² / Δτ,   γʸ = α · Δy² / Δτ
# or, when `length_scale = ℓ` is specified, fixed diffusivity
#   γ = α · ℓ² / Δτ
# in both horizontal directions.
# If `damp_vertical = true`, the vertical contribution
#   γ_z = α · Δz² / Δτ
# is folded into the column tridiag instead of applied as a post-substep
# correction.
# `α` is the dimensionless Klemp 2018 coefficient (`config_smdiv` in MPAS,
# default 0.1). Linear stability of the explicit forward-Euler horizontal
# step gives `A(k) = 1 − 4α · Σᵢ sin²(kᵢ Δxᵢ/2)`; worst case (2-D Nyquist)
# is `8α ≤ 2 → α ≤ 0.25`; we default to 0.1 for margin. The optional
# vertical component is not applied by default; the default vertical acoustic
# damping comes from off-centering (`forward_weight > 0.5`) in the implicit
# column solve.
function apply_divergence_damping!(damping::ThermalDivergenceDamping, substepper, grid, Δτ, thermodynamic_constants)
    FT    = eltype(grid)
    arch  = architecture(grid)
    α     = convert(FT, damping.coefficient)
    Δτ_FT = convert(FT, Δτ)

    TX, TY, _ = topology(grid)
    x_damping_scale = TX === Flat ? NoHorizontalDampingScale() :
                      horizontal_damping_scale(damping, α, Δτ_FT)
    y_damping_scale = TY === Flat ? NoHorizontalDampingScale() :
                      horizontal_damping_scale(damping, α, Δτ_FT)

    launch!(arch, grid, :xyz, _thermal_divergence_damping!,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            substepper.density_potential_temperature_perturbation,
            substepper.previous_density_potential_temperature_perturbation,
            substepper.linearization_potential_temperature,
            grid, x_damping_scale, y_damping_scale)

    return nothing
end

@inline dρθ′(i, j, k, grid, ρθ′, ρθ′ˢ⁻) = @inbounds ρθ′[i, j, k] - ρθ′ˢ⁻[i, j, k]

struct NoHorizontalDampingScale end
struct LocalHorizontalDampingScale{FT}
    coefficient_over_Δτ :: FT
end

struct FixedHorizontalDampingScale{FT}
    diffusivity :: FT
end

@inline horizontal_damping_scale(damping::ThermalDivergenceDamping{FT, Nothing}, α, Δτ) where FT =
    LocalHorizontalDampingScale(α / Δτ)

@inline function horizontal_damping_scale(damping::ThermalDivergenceDamping, α, Δτ)
    ℓ = convert(typeof(α), damping.length_scale)
    return FixedHorizontalDampingScale(α * ℓ^2 / Δτ)
end

@inline x_damping_diffusivity(i, j, k, grid, ::NoHorizontalDampingScale) = zero(grid)
@inline y_damping_diffusivity(i, j, k, grid, ::NoHorizontalDampingScale) = zero(grid)

@inline x_damping_diffusivity(i, j, k, grid, scale::FixedHorizontalDampingScale) =
    scale.diffusivity

@inline y_damping_diffusivity(i, j, k, grid, scale::FixedHorizontalDampingScale) =
    scale.diffusivity

@inline x_damping_diffusivity(i, j, k, grid, scale::LocalHorizontalDampingScale) =
    scale.coefficient_over_Δτ * Δxᶠᶜᶜ(i, j, k, grid)^2

@inline y_damping_diffusivity(i, j, k, grid, scale::LocalHorizontalDampingScale) =
    scale.coefficient_over_Δτ * Δyᶜᶠᶜ(i, j, k, grid)^2


# Horizontal divergence damping in the form of Klemp, Skamarock & Ha (2018)
# eq. (36): per-substep momentum correction is the gradient of the (ρθ)′
# tendency, divided by θᴸ at the face,
#   Δ(ρu)′ = −γˣ · ∂x[(ρθ)′ − (ρθ)′ˢ⁻] / ℑxᶠᵃᵃ(θᴸ)
#   Δ(ρv)′ = −γʸ · ∂y[(ρθ)′ − (ρθ)′ˢ⁻] / ℑyᵃᶠᵃ(θᴸ)
# with local default diffusivities γˣ = α Δx² / Δτ and γʸ = α Δy² / Δτ.
# If the user passes a fixed `length_scale`, both directions use the fixed
# diffusivity γ = α length_scale² / Δτ for backwards-compatible tuning.
# The vertical component lives in the column tridiag (it's a Laplacian on
# (ρw)′ folded into the implicit acoustic solve), not here.
@kernel function _thermal_divergence_damping!(ρu′, ρv′, ρθ′, ρθ′ˢ⁻, θᴸ, grid,
                                              x_damping_scale, y_damping_scale)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂x_div = ∂xᶠᶜᶜ(i, j, k, grid, dρθ′, ρθ′, ρθ′ˢ⁻)
        θᴸᶠᶜᶜ  = ℑxᶠᵃᵃ(i, j, k, grid, θᴸ)
        γˣ = x_damping_diffusivity(i, j, k, grid, x_damping_scale)
        ρu′[i, j, k] -= γˣ * ∂x_div / θᴸᶠᶜᶜ

        ∂y_div = ∂yᶜᶠᶜ(i, j, k, grid, dρθ′, ρθ′, ρθ′ˢ⁻)
        θᴸᶜᶠᶜ  = ℑyᵃᶠᵃ(i, j, k, grid, θᴸ)
        γʸ = y_damping_diffusivity(i, j, k, grid, y_damping_scale)
        ρv′[i, j, k] -= γʸ * ∂y_div / θᴸᶜᶠᶜ
    end
end

#####
##### Section 10 — Time-averaged velocity for non-acoustic scalar transport
#####
##### WRF/MPAS dynamics-transport split for moisture, tracers, chemistry, and
##### TKE: those non-acoustic scalar tendencies advect against the velocity
##### averaged over the stage's substep loop, not a single-snapshot velocity.
##### The slow thermodynamic `ρθ` tendency is part of the acoustic system and
##### is computed separately with `model.velocities` before the substep loop.
##### We accumulate raw `momentum_perturbation` into the
##### `time_averaged_velocities` slots after each substep, then at stage end
##### normalize:
#####
#####   ⟨ρu⟩ = ρuᴸ + (1/Nτ) ∑ₙ (ρu)′(n) = (model.momentum.ρu) + accum/Nτ
#####   ⟨u⟩  ≈ ⟨ρu⟩ / ρᴸ_face
#####
##### `model.momentum.*` is the stage-entry momentum (the substep loop only
##### touches `momentum_perturbation`, not the model's momentum), and
##### `model.dynamics.density` holds the stage-entry density — both serve
##### as the Uᴸ_stage reference. Dividing by ρᴸ ignores the variation of
##### ρ during the substep loop, which is small for acoustic perturbations.
#####

@inline function accumulate_momentum_perturbations!(substepper)
    parent(substepper.time_averaged_velocities.u) .+=
        parent(substepper.momentum_perturbation.u)
    parent(substepper.time_averaged_velocities.v) .+=
        parent(substepper.momentum_perturbation.v)
    parent(substepper.time_averaged_velocities.w) .+=
        parent(substepper.momentum_perturbation.w)
    return nothing
end

function finalize_time_averaged_velocity!(substepper, model, Nτ)
    grid = model.grid
    arch = architecture(grid)
    FT   = eltype(grid)
    inv_Nτ = one(FT) / FT(Nτ)

    # `model.dynamics.density` and `model.momentum.*` are still the
    # stage-entry (Uᴸ) values here — the substep loop only touched
    # substepper-owned perturbation fields. They serve as ρᴸ and ρu/v/wᴸ.
    launch!(arch, grid, :xyz, _finalize_time_averaged_velocity!,
            substepper.time_averaged_velocities.u,
            substepper.time_averaged_velocities.v,
            substepper.time_averaged_velocities.w,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            model.dynamics.density,
            grid, model.dynamics, inv_Nτ)

    fill_halo_regions!(substepper.time_averaged_velocities.u)
    fill_halo_regions!(substepper.time_averaged_velocities.v)
    fill_halo_regions!(substepper.time_averaged_velocities.w)

    return nothing
end

@kernel function _finalize_time_averaged_velocity!(u_avg, v_avg, w_avg,
                                                   ρu_stage, ρv_stage, ρw_stage,
                                                   ρᴸ, grid, dynamics, inv_Nτ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρu_total = ρu_stage[i, j, k] + u_avg[i, j, k] * inv_Nτ
        ρv_total = ρv_stage[i, j, k] + v_avg[i, j, k] * inv_Nτ
        ρw_total = acoustic_stage_vertical_transport_momentum(i, j, k, grid, dynamics,
                                                              ρu_stage, ρv_stage, ρw_stage) +
                   w_avg[i, j, k] * inv_Nτ

        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρᴸ)
        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρᴸ)
        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρᴸ)
        ρ̂ᶠᶜᶜ = ifelse(ρᶠᶜᶜ == 0, one(ρᶠᶜᶜ), ρᶠᶜᶜ)
        ρ̂ᶜᶠᶜ = ifelse(ρᶜᶠᶜ == 0, one(ρᶜᶠᶜ), ρᶜᶠᶜ)
        ρ̂ᶜᶜᶠ = ifelse(ρᶜᶜᶠ == 0, one(ρᶜᶜᶠ), ρᶜᶜᶠ)

        u_avg[i, j, k] = ρu_total / ρ̂ᶠᶜᶜ
        v_avg[i, j, k] = ρv_total / ρ̂ᶜᶠᶜ
        w_avg[i, j, k] = ρw_total / ρ̂ᶜᶜᶠ * (k > 1)
    end
end

@inline function acoustic_stage_vertical_transport_momentum(i, j, k, grid, dynamics,
                                                            ρu_stage, ρv_stage, ρw_stage)
    @inbounds return ρw_stage[i, j, k]
end

#####
##### Section 11 — Full-state recovery at stage end
#####

# After the substep loop completes for a stage, reconstruct the full
# prognostic state ρ, ρu, ρv, ρw, ρθ from the stage-entry linearization
# state plus the accumulated perturbations:
#   ρᵐ⁺  = ρᴸ  + ρ′
#   ρθᵐ⁺ = ρθᴸ + (ρθ)′
#   ρuᵐ⁺ = ρuᴸ + (ρu)′, etc.
#
# Velocity diagnosis is deliberately not done in this kernel. Face velocities
# require neighbor-cell density interpolation; computing them while this same
# kernel writes ρ can read a GPU-scheduling-dependent mix of old and new
# neighbor values. The driver calls AtmosphereModels.compute_velocities! after
# recovery and halo fill.
@kernel function _recover_full_state!(ρ, ρθ, m,
                                      ρ′, ρθ′, ρu′, ρv′, ρw′,
                                      ρᴸ, ρuᴸ, ρvᴸ, ρwᴸ, ρθᴸ,
                                      grid, dynamics)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵐ⁺  = ρᴸ[i, j, k]  + ρ′[i, j, k]
        ρθᵐ⁺ = ρθᴸ[i, j, k] + ρθ′[i, j, k]
        ρuᵐ⁺ = ρuᴸ[i, j, k] + ρu′[i, j, k]
        ρvᵐ⁺ = ρvᴸ[i, j, k] + ρv′[i, j, k]
        ρwᵐ⁺ = acoustic_recovered_vertical_momentum(i, j, k, grid, dynamics,
                                                    ρuᴸ, ρvᴸ, ρwᴸ, ρu′, ρv′, ρw′)

        ρ[i, j, k]  = ρᵐ⁺
        ρθ[i, j, k] = ρθᵐ⁺

        m.ρu[i, j, k] = ρuᵐ⁺
        m.ρv[i, j, k] = ρvᵐ⁺
        m.ρw[i, j, k] = ρwᵐ⁺
    end
end

@inline function acoustic_recovered_vertical_momentum(i, j, k, grid, dynamics,
                                                      ρuᴸ, ρvᴸ, ρwᴸ, ρu′, ρv′, ρw′)
    @inbounds return ρwᴸ[i, j, k] + ρw′[i, j, k]
end

#####
##### Section 12 — Substep loop driver
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

    FT  = eltype(grid)
    g   = convert(FT, model.thermodynamic_constants.gravitational_acceleration)
    constants = model.thermodynamic_constants
    Rᵈ = convert(FT, dry_air_gas_constant(constants))
    cᵖᵈ = convert(FT, constants.dry_air.heat_capacity)
    κ = Rᵈ / cᵖᵈ
    pˢᵗ = convert(FT, model.dynamics.standard_pressure)

    ω = FT(substepper.forward_weight)            # CN weight on the new side
    one_minus_ω = FT(1) - ω                       # CN weight on the old side

    # Compute substep count and size for this stage. WS-RK3 stage weights
    # are β = (1/3, 1/2, 1); for ProportionalSubsteps to give integer
    # substep counts at every stage, N must be a multiple of LCM(3, 2) = 6.
    # Floor of 6 ensures sane behavior even for very small Δt where the
    # acoustic-CFL substep count would round to 0 or 1.
    Δt_FT = FT(Δt)
    N_raw = acoustic_substeps(substepper.substeps, grid, Δt_FT,
                              model.thermodynamic_constants, substepper.acoustic_cfl)
    N = max(6, 6 * cld(N_raw, 6))
    Nτ, Δτ = stage_substep_count_and_size(substepper.substep_distribution, β_stage, Δt_FT, N)

    # Build the slow vertical-momentum tendency Gˢρw at z-faces:
    #   Gˢρw = Gⁿρw − ∂z(pᴸ − pᵣ) − g (ρᴸ − ρᵣ)        (with reference state)
    #   Gˢρw = Gⁿρw − ∂z pᴸ − g ρᴸ                     (no reference state)
    # which the per-substep linearized acoustic forces add to.
    assemble_slow_vertical_momentum_tendency!(substepper, model, β_stage)

    # Initialize perturbations with the SK08 rewind term so the substep
    # effectively starts from U(t) = Uᴸ (the outer-step-start state).
    initialize_stage_perturbations!(substepper, model, Uᴸ)

    Gⁿ = model.timestepper.Gⁿ
    χ_name = thermodynamic_density_name(model.formulation)
    Gˢρθ = getproperty(Gⁿ, χ_name)

    # Substep loop
    for substep in 1:Nτ
        # Step A: explicit horizontal forward of (ρu)′, (ρv)′. Following the
        # MPAS forward-backward acoustic sequence, the first small step in a
        # multi-step stage includes the frozen large-step pressure gradient
        # but skips the acoustic perturbation pressure gradient until
        # mass/thermodynamic perturbations have been advanced once. For
        # degenerate one-substep stages, apply the perturbation pressure
        # gradient immediately so the stage still contains the fast force.
        apply_pressure_gradient =
            apply_horizontal_pressure_gradient_substep(substep, Nτ,
                                                       substepper.apply_first_substep_pressure_gradient)

        launch!(arch, grid, :xyz, _explicit_horizontal_step!,
                substepper.momentum_perturbation.u,
                substepper.momentum_perturbation.v,
                grid, model.dynamics, Δτ,
                substepper.density_potential_temperature_perturbation,
                substepper.linearization_exner,
                model.dynamics.pressure,
                Gⁿ.ρu, Gⁿ.ρv, substepper.linearization_gamma_R_mixture,
                apply_pressure_gradient)

        fill_halo_regions!(substepper.momentum_perturbation.u)
        fill_halo_regions!(substepper.momentum_perturbation.v)


        # Save (ρθ)′ before the column kernel for damping use
        parent(substepper.previous_density_potential_temperature_perturbation) .=
            parent(substepper.density_potential_temperature_perturbation)

        # CN time-step weights for this substep. δτᵐ⁺ = ω·Δτ is the
        # new-side weight (used by the matrix and the post-solve);
        # δτˢ⁻ = (1−ω)·Δτ is the old-side weight (used by the
        # predictor's old-flux contribution and the old part of the
        # vertical RHS). See `docs/src/compressible_dynamics.md`.
        δτᵐ⁺ = ω * Δτ
        δτˢ⁻ = one_minus_ω * Δτ

        # Implicit-vertical-damping prefactors. When the damping strategy
        # is `ThermalDivergenceDamping(damp_vertical=true)`, the
        # vertical part of the divergence damping is folded into the
        # tridiag with `dᵐ⁺ = ω·α·Δz²` on the LHS and
        # `dˢ⁻ = (1−ω)·α·Δz²` on the predictor RHS. Both reduce to
        # zero for `NoDivergenceDamping` or when the user opts out via
        # `damp_vertical=false`.
        dᵐ⁺, dˢ⁻ = implicit_damping_factors(substepper.damping, ω, one_minus_ω, grid, FT)

        # Step B: build predictors `ρ′★`, `ρθ′★` and the tridiag RHS for (ρw)′ᵐ⁺
        launch!(arch, grid, :xy, _build_predictors_and_vertical_rhs!,
                substepper.momentum_perturbation.w,
                substepper.density_predictor,
                substepper.density_potential_temperature_predictor,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation.w,
                substepper.momentum_perturbation.u, substepper.momentum_perturbation.v,
                grid, model.dynamics, Δτ, δτᵐ⁺, δτˢ⁻,
                Gⁿ.ρ, Gˢρθ, substepper.slow_vertical_momentum_tendency,
                substepper.thermodynamic_tendency_factor,
                substepper.vertical_momentum_tendency_factor,
                substepper.linearization_potential_temperature, substepper.linearization_exner,
                substepper.linearization_gamma_R_mixture, g, dˢ⁻,
                substepper.sponge)

        # Step C: implicit tridiag solve for (ρw)′ with implicit-half δτᵐ⁺
        # and (when active) implicit vertical damping prefactor `dᵐ⁺`.
        # `sponge` may add an implicit Rayleigh contribution on the
        # diagonal in a layer below the lid.
        solve!(substepper.momentum_perturbation.w, substepper.vertical_solver,
               substepper.momentum_perturbation.w,
               substepper.linearization_exner, substepper.linearization_potential_temperature,
               substepper.linearization_gamma_R_mixture, g, δτᵐ⁺, dᵐ⁺,
               substepper.sponge)

        # Step D: post-solve recovery of ρ′, (ρθ)′ using new (ρw)′
        launch!(arch, grid, :xyz, _post_solve_recovery!,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation.w,
                substepper.momentum_perturbation.u,
                substepper.momentum_perturbation.v,
                substepper.density_predictor,
                substepper.density_potential_temperature_predictor,
                grid, model.dynamics, δτᵐ⁺,
                substepper.linearization_potential_temperature)

        fill_halo_regions!(substepper.density_perturbation)
        fill_halo_regions!(substepper.density_potential_temperature_perturbation)

        # Step E: optional Klemp 2018 post-substep damping (no-op for
        # `NoDivergenceDamping`).
        apply_divergence_damping!(substepper.damping, substepper, grid, Δτ,
                                  model.thermodynamic_constants)

        fill_halo_regions!(substepper.momentum_perturbation.u)
        fill_halo_regions!(substepper.momentum_perturbation.v)

        # Step F: accumulate (ρu)′, (ρv)′, (ρw)′ for the time-averaged
        # velocity. Normalized to a velocity at stage end by
        # `finalize_time_averaged_velocity!`; consumed by `update_state!`
        # between RK stages for moisture/tracer transport via
        # `transport_velocities`.
        accumulate_momentum_perturbations!(substepper)
    end

    # Stage-end: convert the accumulated momentum perturbations into a
    # time-averaged velocity field. Read by `update_state!` through
    # `transport_velocities(model)` for moisture/tracer tendencies.
    # Done BEFORE `_recover_full_state!` so we read the stage-entry
    # `model.momentum.*` (substep loop hasn't touched it) and stage-entry
    # `model.dynamics.density` as the Uᴸ_stage reference.
    finalize_time_averaged_velocity!(substepper, model, Nτ)

    # Stage-end: recover the full prognostic state in-place. `model.dynamics.density`,
    # `χ_field`, and `model.momentum.*` are still the stage-entry Uᴸ values here
    # (the substep loop only touched substepper.* perturbation fields). The
    # recovery kernel reads them as Uᴸ AND writes the full state back to the
    # same fields — per-thread read-before-write makes this aliasing safe
    # because all reads are local to the same grid point.
    χ_field = thermodynamic_density(model.formulation)
    launch!(arch, grid, :xyz, _recover_full_state!,
            model.dynamics.density, χ_field,
            model.momentum,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            substepper.momentum_perturbation.w,
            model.dynamics.density,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            χ_field,
            grid, model.dynamics)

    fill_halo_regions!(model.dynamics.density)
    fill_halo_regions!(χ_field)
    fill_halo_regions!(model.momentum)
    AtmosphereModels.compute_velocities!(model)

    return nothing
end
