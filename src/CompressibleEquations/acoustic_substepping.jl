#####
##### Acoustic substepping for CompressibleDynamics
#####
##### Evolves linearized acoustic perturbations (primes) about the stage-entry state Uᴸ, refreshed each
##### WS-RK3 stage (Skamarock & Klemp 2008):  ρ′=ρ−ρᴸ, (ρθ)′=ρθ−ρθᴸ, (ρu)′=ρu−ρuᴸ (and v,w).
##### θᴸ=ρθᴸ/ρᴸ, Πᴸ=(pᴸ/pˢᵗ)^κ, γᵐRᵐᴸ are cached per stage; ρᴸ,ρθᴸ,pᴸ,(ρu/v/w)ᴸ are read live (the loop
##### never mutates them) and are the recovery base.  Linearized equations integrated by the loop:
#####
#####   ∂t ρ′    + ∇·((ρu)′,(ρv)′,(ρw)′)      = Gˢρ
#####   ∂t (ρθ)′ + ∇·(θᴸ·((ρu)′,(ρv)′,(ρw)′)) = Gˢρθ
#####   ∂t (ρu)′ + ∂x pᴸ + ∂x(Cᴸ(ρθ)′)        = Gˢρu     (Cᴸ = γᵐRᵐᴸΠᴸ; PGF = gradient of cell-centered Cᴸ(ρθ)′)
#####   ∂t (ρv)′ + ∂y pᴸ + ∂y(Cᴸ(ρθ)′)        = Gˢρv
#####   ∂t (ρw)′ +         ∂z(Cᴸ(ρθ)′) + g·ρ′ = Gˢρw
#####
##### Time discretization: horizontal momentum is forward-Euler, with the frozen ∇pᴸ and the
##### perturbation horizontal PGF applied on every substep. Vertical ((ρw)′,(ρθ)′,ρ′) coupling is off-centered Crank-Nicolson (`forward_weight` ω:
##### 0.5 = centered, >0.5 = dissipative), reducing to a tridiagonal Schur solve for (ρw)′ at z-faces.
##### Each stage then recovers ρ=ρᴸ+ρ′, ρθ=ρθᴸ+(ρθ)′, … and diagnoses velocities.
#####
##### Kernel args ρ′, ρθ′, ρu′, ρv′, ρw′ map to struct fields density_perturbation,
##### density_potential_temperature_perturbation, momentum_perturbation.{u,v,w}; predictors carry ★
##### (ρ′★, ρθ′★). (σ, η are reserved for vertical coordinates, so primes are used throughout.)
##### Public derivation: docs/src/compressible_dynamics.md.
##### Refs: Wicker & Skamarock 2002 (MWR 130, 2088); Klemp et al. 2018 (MWR 146, 1911).
#####

using KernelAbstractions: KernelAbstractions, @kernel, @index, @localmem, @synchronize
using Oceananigans.Architectures: convert_to_device, device
using Oceananigans.Utils: configure_kernel

using Oceananigans: CenterField, XFaceField, YFaceField, ZFaceField, architecture
using Oceananigans.Models: boundary_condition_args
using Oceananigans: fields
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection
using Oceananigans.TimeSteppers: implicit_step!
using Oceananigans.Grids: ZDirection, rnode, znode, RectilinearGrid
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!, masked_multiply
using Oceananigans.Operators:
    ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ,
    ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ,
    δxᶜᵃᵃ, δyᵃᶜᵃ,
    div_xyᶜᶜᶜ,
    Δz⁻¹ᶜᶜᶜ, Δz⁻¹ᶜᶜᶠ,
    Δxᶠᶜᶜ,
    Δyᶜᶠᶜ,
    Axᶠᶜᶜ, Ayᶜᶠᶜ, V⁻¹ᶜᶜᶜ

using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!, BoundaryCondition, NormalFlow

using Oceananigans.Grids: Flat, Bounded, Center, Face, peripheral_node,
                          topology,
                          minimum_xspacing, minimum_yspacing, minimum_zspacing

using Adapt: Adapt, adapt

#####
##### Section 1 — Substepper struct
#####

"""
$(TYPEDEF)

Storage and parameters for the split-explicit acoustic substepper (scheme described in the module
header). Πᴸ=(pᴸ/pˢᵗ)^κ, θᴸ=ρθᴸ/ρᴸ, γᵐRᵐᴸ are cached once per stage (recomputing inline per call is much
slower on H100); ρᴸ, ρθᴸ, pᴸ and the stage-entry momenta are read live from `model.dynamics.*` /
`model.momentum.*` (untouched by the loop) and are the recovery base for `_recover_full_state!` — no
snapshot fields. The vertical solve is a (possibly off-centered) Crank-Nicolson tridiagonal Schur system
for (ρw)′.

Fields:
- `substeps`: acoustic substeps N per Δt (`nothing` ⇒ adaptive via `acoustic_cfl`).
- `acoustic_cfl`: target horizontal acoustic Courant number for the adaptive count (default 0.5).
- `minimum_horizontal_spacing`: the grid's smallest horizontal spacing, which the adaptive count
  measures the acoustic Courant number against; computed once, when the substepper is built.
- `forward_weight`: CN off-centering ω (0.5 = centered; default 0.65).
- `damping`, `substep_distribution`: divergence-damping strategy; substep allocation across WS-RK3 stages.
- `linearization_exner` (Πᴸ), `linearization_potential_temperature` (θᴸ), `linearization_gamma_R_mixture`
  (γᵐRᵐᴸ, the moist PGF coefficient): per-stage caches.
- `density_perturbation` (ρ′), `density_potential_temperature_perturbation` ((ρθ)′),
  `momentum_perturbation` ((ρu/v/w)′ as `.u/.v/.w`): perturbation prognostics advanced in the loop.
- `density_predictor`, `density_potential_temperature_predictor`: explicit predictors before the vertical solve.
- `previous_density_potential_temperature_perturbation`: prior-substep (ρθ)′, for Klemp 2018 damping.
- `time_averaged_velocities`: acoustic-mean velocity for non-acoustic scalar transport (moisture/tracers/
  chemistry/TKE); the slow ρθ tendency uses the current RK predictor velocity instead, not this cache.
- `slow_vertical_momentum_tendency` (Gˢρw, z-faces): advection+Coriolis+closure+forcing (PGF/buoyancy
  excluded — those are in the fast operator).
- `vertical_solver_source_term` (z-faces): explicit RHS of the (ρw)′ tridiagonal system.
- `vertical_solver`: `BatchedTridiagonalSolver` for the implicit (ρw)′ update.
- `vertical_velocity_cache`, `density_cache`: the stage-entry predictor velocity and its carrier
  dry density, frozen for `implicit_substep!` (see `cache_advecting_state!`); `nothing` without
  adaptive-implicit advection.
- `time_averaged_vertical_velocity_cache`: the acoustic-mean transport velocity the moisture and
  tracer tendencies were built with, frozen for `scalar_substep!` and for the thermodynamic
  tendency's condensate sedimentation term (see `cache_transport_velocity!`).
"""
struct AcousticSubstepper{N, FT, D, AD, US, CF, MP, TAV, GT, TS, WC, DC, TWC, PG}
    substeps :: N
    acoustic_cfl :: FT
    minimum_horizontal_spacing :: FT
    forward_weight :: FT
    thermodynamic_tendency_factor :: FT
    vertical_momentum_tendency_factor :: FT
    vertical_pressure_tendency_factor :: FT
    final_stage_vertical_pressure_tendency_factor :: FT
    damping :: D
    substep_distribution :: AD
    sponge :: US
    # Open-boundary relaxation factor α for `ρ′,(ρθ)′` (issue #738).
    open_boundary_relaxation :: FT

    # Linearization basic state Πᴸ, θᴸ, derived per stage from live model fields
    # (`model.dynamics.pressure`, `model.dynamics.dry_density`, and the formulation's
    # prognostic ρθ field — untouched by the substep loop, so they hold ρᴸ, ρθᴸ, pᴸ
    # throughout and serve as the recovery base in `_recover_full_state!`).
    linearization_exner :: CF
    linearization_potential_temperature :: CF

    # γᵐRᵐᴸ = γᵐ·Rᵐ for the moist linearized PGF; per-stage, → γᵈRᵈ for dry runs.
    linearization_gamma_R_mixture :: CF

    density_perturbation :: CF
    density_potential_temperature_perturbation :: CF
    momentum_perturbation :: MP

    density_predictor :: CF
    density_potential_temperature_predictor :: CF
    previous_density_potential_temperature_perturbation :: CF

    # Acoustic-mean velocity for non-acoustic scalar transport (WRF/MPAS split; see docstring).
    time_averaged_velocities :: TAV

    slow_vertical_momentum_tendency :: GT
    vertical_solver_source_term :: GT
    vertical_solver :: TS

    vertical_velocity_cache :: WC                 # stage-entry predictor w, split by momentum and ρθ
    density_cache :: DC                           # stage-entry ρᵈ; `nothing` without adaptive-implicit advection
    time_averaged_vertical_velocity_cache :: TWC  # acoustic-mean w the moisture and tracer tendencies were built with
    pressure_gradients :: PG                      # `TerrainPressureGradients` on terrain-following grids, else `nothing`
end

"""
$(TYPEDEF)

Horizontal pressure gradients stored on terrain-following grids (port of handoff N3 5255302b). There
each gradient is a chain-rule derivative, and without the store several kernels evaluate the same
values again from unchanged inputs:

- `linearization_x`, `linearization_y`: the frozen ∂x pᴸ, ∂y pᴸ, stored once per stage at the start of
  `assemble_slow_vertical_momentum_tendency!` over u-faces `1:Nx+1` × `1:Ny`, v-faces `1:Nx` × `1:Ny+1`
  (one face in a `Flat` direction) and levels `0:Nz`; the slow ρw̃ assembly interpolates them and every
  explicit step of the stage loads them.
- `perturbation_x`, `perturbation_y`: ∂x p′, ∂y p′ of p′ = γRᵐᴸ Πᴸ (ρθ)′ at the substep's old (ρθ)′,
  stored by the explicit step (its faces plus the last face `Nx+1` / `Ny+1`) and interpolated by the
  vertical rhs of the same substep (nothing writes (ρθ)′, Πᴸ, γRᵐᴸ or their halos in between).

Each stored value is the grid-float result of the expression it replaces, so the results are bitwise
those of the in-place evaluation. Switch: `STORED_TERRAIN_PRESSURE_GRADIENTS[]`
(`BREEZE_STORED_TERRAIN_PRESSURE_GRADIENTS=false` at load turns it off).
"""
struct TerrainPressureGradients{X, Y}
    linearization_x :: X
    linearization_y :: Y
    perturbation_x :: X
    perturbation_y :: Y
end

TerrainPressureGradients(grid) = TerrainPressureGradients(XFaceField(grid), YFaceField(grid),
                                                          XFaceField(grid), YFaceField(grid))

Adapt.adapt_structure(to, P::TerrainPressureGradients) =
    TerrainPressureGradients(adapt(to, P.linearization_x), adapt(to, P.linearization_y),
                             adapt(to, P.perturbation_x), adapt(to, P.perturbation_y))

# `TerrainPressureGradients` on terrain-following grids (method in terrain_compressible_physics.jl).
stored_pressure_gradients(grid) = nothing

const STORED_TERRAIN_PRESSURE_GRADIENTS = Ref(true)

# The substepper's stored gradients if the switch is on, else `nothing` (evaluate in place).
active_pressure_gradients(substepper) = STORED_TERRAIN_PRESSURE_GRADIENTS[] ? substepper.pressure_gradients : nothing

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.substeps,
                       a.acoustic_cfl,
                       a.minimum_horizontal_spacing,
                       a.forward_weight,
                       a.thermodynamic_tendency_factor,
                       a.vertical_momentum_tendency_factor,
                       a.vertical_pressure_tendency_factor,
                       a.final_stage_vertical_pressure_tendency_factor,
                       adapt(to, a.damping),
                       a.substep_distribution,
                       adapt(to, a.sponge),
                       a.open_boundary_relaxation,
                       adapt(to, a.linearization_exner),
                       adapt(to, a.linearization_potential_temperature),
                       adapt(to, a.linearization_gamma_R_mixture),
                       adapt(to, a.density_perturbation),
                       adapt(to, a.density_potential_temperature_perturbation),
                       adapt(to, a.momentum_perturbation),
                       adapt(to, a.density_predictor),
                       adapt(to, a.density_potential_temperature_predictor),
                       adapt(to, a.previous_density_potential_temperature_perturbation),
                       adapt(to, a.time_averaged_velocities),
                       adapt(to, a.slow_vertical_momentum_tendency),
                       adapt(to, a.vertical_solver_source_term),
                       adapt(to, a.vertical_solver),
                       adapt(to, a.vertical_velocity_cache),
                       adapt(to, a.density_cache),
                       adapt(to, a.time_averaged_vertical_velocity_cache),
                       adapt(to, a.pressure_gradients))

#####
##### Section 2 — Constructor
#####

"""
$(TYPEDSIGNATURES)

Construct an `AcousticSubstepper`. The perturbation face fields ``(ρu)′, (ρv)′, (ρw)′`` and the
scalar-transport velocities use topology-derived BCs (periodic wrap / impenetrability), **not** the
prognostic momentum's BCs: inheriting them would imprint the full-state wall target onto the perturbation
halo for a nonzero `NormalFlowBoundaryCondition` (issue \\#716) and apply momentum BCs to velocity fields.
The wall target re-enters via the prognostic momentum's own BC after each substep's momentum update.
The `prognostic_momentum` kwarg is retained for backwards compatibility but no longer consulted.
"""
function AcousticSubstepper(grid, split_explicit::SplitExplicitTimeDiscretization;
                            prognostic_momentum = nothing, substep_floattype = eltype(grid),
                            cache_advecting_state = false)
    Ns = split_explicit.substeps
    FT = eltype(grid)
    ω = convert(FT, split_explicit.forward_weight)

    acoustic_cfl = convert(FT, split_explicit.acoustic_cfl)
    Δx_min = convert(FT, minimum_horizontal_spacing(grid))
    thermodynamic_tendency_factor = convert(FT, split_explicit.thermodynamic_tendency_factor)
    vertical_momentum_tendency_factor = convert(FT, split_explicit.vertical_momentum_tendency_factor)
    vertical_pressure_tendency_factor = convert(FT, split_explicit.vertical_pressure_tendency_factor)
    final_stage_vertical_pressure_tendency_factor = convert(FT, split_explicit.final_stage_vertical_pressure_tendency_factor)

    damping = split_explicit.damping
    sponge = split_explicit.sponge
    substep_distribution = split_explicit.substep_distribution
    open_boundary_relaxation = convert(FT, split_explicit.open_boundary_relaxation)

    # `substep_floattype` (default `eltype(grid)`) is the storage type for the acoustic perturbation/predictor/
    # linearization working fields below. Pass a reduced-precision type (e.g. `BFloat16`, where the
    # grid/field types support it) to halve their HBM traffic in the bandwidth-bound substep kernels:
    # kernels read substep_floattype, promote to FT, compute in FT, store substep_floattype. The (ρw)′ solve target, tridiag scratch,
    # primary prognostics, and WENO tendencies stay FT regardless (the solver recurrence and WENO
    # degrade in low precision).

    # Linearization basic state — Πᴸ, θᴸ derived from live model fields.
    linearization_exner = CenterField(grid, substep_floattype)
    linearization_potential_temperature = CenterField(grid, substep_floattype)

    # γᵐRᵐᴸ — the only cached moisture quantity. Recomputed once per stage
    # refresh from the live moisture state.
    linearization_gamma_R_mixture = CenterField(grid, substep_floattype)

    density_perturbation = CenterField(grid, substep_floattype)
    density_potential_temperature_perturbation = CenterField(grid, substep_floattype)

    momentum_perturbation = (u = XFaceField(grid, substep_floattype),
                             v = YFaceField(grid, substep_floattype),
                             w = ZFaceField(grid)) # (ρw)′ stays FT — it is the tridiag solve target

    density_predictor = CenterField(grid, substep_floattype)
    density_potential_temperature_predictor = CenterField(grid, substep_floattype)
    previous_density_potential_temperature_perturbation = CenterField(grid, substep_floattype)

    # Substep-averaged velocities for scalar transport.
    time_averaged_velocities = (u = XFaceField(grid),
                                v = YFaceField(grid),
                                w = ZFaceField(grid))

    slow_vertical_momentum_tendency = ZFaceField(grid)
    vertical_solver_source_term = ZFaceField(grid) # RHS stays FT, like the (ρw)′ solve target

    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    scratch = zeros(arch, FT, Nx, Ny, Nz)
    vertical_solver = BatchedTridiagonalSolver(grid;
                                               lower_diagonal = AcousticTridiagLower(),
                                               diagonal       = AcousticTridiagDiagonal(),
                                               upper_diagonal = AcousticTridiagUpper(),
                                               scratch,
                                               tridiagonal_direction = ZDirection())

    vertical_velocity_cache = cache_advecting_state ? ZFaceField(grid) : nothing
    density_cache = cache_advecting_state ? CenterField(grid) : nothing
    time_averaged_vertical_velocity_cache = ZFaceField(grid)

    return AcousticSubstepper(Ns, acoustic_cfl, Δx_min, ω, thermodynamic_tendency_factor,
                              vertical_momentum_tendency_factor,
                              vertical_pressure_tendency_factor,
                              final_stage_vertical_pressure_tendency_factor,
                              damping, substep_distribution,
                              sponge, open_boundary_relaxation,
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
                              vertical_solver_source_term,
                              vertical_solver,
                              vertical_velocity_cache,
                              density_cache,
                              time_averaged_vertical_velocity_cache,
                              stored_pressure_gradients(grid))
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
  - `linearization_potential_temperature` = θᴸ = ρθᴸ/ρᴸ derived from `model.dynamics.dry_density` + ρθ
"""
function freeze_linearization_state!(substepper::AcousticSubstepper, model)
    # E6 (handoff b97a17ee): stage 1's `prepare_acoustic_cache!` refreshes the same quantities from the
    # same, unchanged outer-step-start state before any substep reads them (every reader is inside the
    # substep loop), so the outer-step refresh is unread. `NO_STEP_LINEARIZATION_REFRESH[] = false`
    # (`BREEZE_NO_STEP_LINEARIZATION_REFRESH=false` at load) restores it.
    NO_STEP_LINEARIZATION_REFRESH[] || refresh_linearization_basic_state!(substepper, model)
    seed_time_averaged_velocities!(substepper, model)
    return nothing
end

const NO_STEP_LINEARIZATION_REFRESH = Ref(true)

"""
$(TYPEDSIGNATURES)

Seed the time-averaged transport velocity with the outer-step-start velocities: stage 1 has no
prior acoustic loop in this outer step to average over. Called at outer-step start by
[`freeze_linearization_state!`](@ref), and by `maybe_prepare_first_time_step!` *before* the
first tendency computation, so the first stage splits a physical velocity instead of the
constructor's zeros.
"""
function seed_time_averaged_velocities!(substepper::AcousticSubstepper, model)
    velocities = outer_step_start_transport_velocities(model)

    # Full parent arrays, halos included. The three staggered components have different array
    # sizes, so each is copied over its own bounds.
    for (avg, src) in zip(substepper.time_averaged_velocities, velocities)
        copyto!(parent(avg), parent(src))
    end

    return nothing
end

outer_step_start_transport_velocities(model) = model.velocities

# Refresh the cached linearization quantities (Πᴸ, θᴸ, γᵐRᵐᴸ) from the
# live model state. Called at outer-step start by `freeze_linearization_state!`
# and at every RK stage by `prepare_acoustic_cache!`. The base-state fields
# ρᴸ, ρθᴸ, pᴸ are `model.dynamics.dry_density`, the formulation's ρθ field, and
# `model.dynamics.pressure` — read directly by the substep kernels.
function refresh_linearization_basic_state!(substepper::AcousticSubstepper, model)
    grid = model.grid
    arch = architecture(grid)
    FT  = eltype(grid)

    constants = model.thermodynamic_constants
    κ   = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    pˢᵗ = convert(FT, model.dynamics.standard_pressure)

    ρθ_field = thermodynamic_density(model.formulation)

    # θ_lin = ρθ/ρ and Π_lin = (p/pˢᵗ)^κ from the live model state.
    # `model.dynamics.dry_density`, `model.dynamics.pressure`, and `ρθ_field`
    # are not mutated by the substep loop, so they stay equal to ρᴸ, pᴸ,
    # ρθᴸ throughout the stage and double as the recovery base.
    launch!(arch, grid, :xyz, _compute_linearization_exner_and_theta!,
            substepper.linearization_exner,
            substepper.linearization_potential_temperature,
            model.dynamics.pressure,
            model.dynamics.dry_density,
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
            model.dynamics.total_density,
            specific_prognostic_moisture(model),
            model.microphysical_fields,
            constants)

    AtmosphereModels.fill_halo_regions_together!((substepper.linearization_exner,
                                                  substepper.linearization_potential_temperature,
                                                  substepper.linearization_gamma_R_mixture))

    return nothing
end

@kernel function _compute_linearization_exner_and_theta!(Π, θ, p, ρ, ρθ, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)
    linearization_exner_and_theta_at!(i, j, k, Π, θ, p, ρ, ρθ, pˢᵗ, κ)
end

@inline function linearization_exner_and_theta_at!(i, j, k, Π, θ, p, ρ, ρθ, pˢᵗ, κ)
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
@kernel function _compute_linearization_mixture_eos!(γRᵐ, grid, microphysics, ρ, qᵛ, μ, constants)
    i, j, k = @index(Global, NTuple)
    linearization_mixture_eos_at!(i, j, k, γRᵐ, grid, microphysics, ρ, qᵛ, μ, constants)
end

@inline function linearization_mixture_eos_at!(i, j, k, γRᵐ, grid, microphysics, ρ, qᵛ, μ, constants)
    @inbounds begin
        ρᵢ  = ρ[i, j, k]
        qᵛᵢ = qᵛ[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρᵢ, qᵛᵢ, μ)
    @inbounds begin
        Rᵈ  = dry_air_gas_constant(constants)
        Rᵛ  = vapor_gas_constant(constants)
        cᵖᵈ = constants.dry_air.heat_capacity
        cᵖᵛ = constants.vapor.heat_capacity
        cˡ  = constants.liquid.heat_capacity
        cⁱ  = constants.ice.heat_capacity

        qᵈᵢ = 1 - q.vapor - q.liquid - q.ice

        Rᵐ  = qᵈᵢ * Rᵈ + q.vapor * Rᵛ
        cᵖᵐ = qᵈᵢ * cᵖᵈ + q.vapor * cᵖᵛ + q.liquid * cˡ + q.ice * cⁱ
        cᵛᵐ = cᵖᵐ - Rᵐ

        # Operation order matches the dry-only path's `cᵖᵈ * Rᵈ / (cᵖᵈ - Rᵈ)`
        # so qᵛ = qˡ = qⁱ = 0 reproduces the dry γᵈRᵈ to bit-identical precision.
        γRᵐ[i, j, k] = cᵖᵐ * Rᵐ / cᵛᵐ
    end
    return nothing
end

#####
##### Linearization folded into the auxiliary pass (handoff E5 64b0409d / C7 in spirit)
#####
# Every reader of Πᴸ, θᴸ, γᵐRᵐᴸ is inside the substep loop, and the stage-entry state the stage refresh reads
# (p, ρᵈ, ρθ, ρ, qᵛ, μ) is the one the preceding `update_state!` left: between it and `prepare_acoustic_cache!` only
# `cache_transport_velocity!` (and, at outer-step start, `store_initial_state!`, the transport seed, and the
# operator-split microphysics update, which ends in its own `update_state!`) run. Where the dynamics' auxiliary
# pass stores the three fields with the same per-cell expressions on the same inputs (terrain fused pass, switch
# `FOLDED_LINEARIZATION`, `BREEZE_FOLDED_LINEARIZATION=false` at load turns it off), the stage refresh is skipped.
const FOLDED_LINEARIZATION = Ref(true)

linearization_in_auxiliary_pass(model) = false

# The arguments of the folded store, or `nothing` when the model has no acoustic substepper.
folded_linearization_target(model) = folded_linearization_target(model, model.timestepper)
folded_linearization_target(model, timestepper) =
    hasproperty(timestepper, :substepper) ? folded_linearization_target(model, timestepper.substepper) : nothing
folded_linearization_target(model, ::Nothing) = nothing

function folded_linearization_target(model, substepper::AcousticSubstepper)
    FT  = eltype(model.grid)
    constants = model.thermodynamic_constants
    κ   = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    pˢᵗ = convert(FT, model.dynamics.standard_pressure)
    return (Π = substepper.linearization_exner,
            θ = substepper.linearization_potential_temperature,
            γRᵐ = substepper.linearization_gamma_R_mixture,
            ρθ = thermodynamic_density(model.formulation),
            pˢᵗ = pˢᵗ, κ = κ)
end

fill_folded_linearization_halos!(::Nothing) = nothing

function fill_folded_linearization_halos!(lin)
    fill_halo_regions!(lin.Π)
    fill_halo_regions!(lin.θ)
    fill_halo_regions!(lin.γRᵐ)
    return nothing
end

@inline store_folded_linearization_at!(i, j, k, ::Nothing, p, ρᵈ, ρ, qᵛ, grid, microphysics, μ, constants) = nothing

@inline function store_folded_linearization_at!(i, j, k, lin, p, ρᵈ, ρ, qᵛ, grid, microphysics, μ, constants)
    linearization_exner_and_theta_at!(i, j, k, lin.Π, lin.θ, p, ρᵈ, lin.ρθ, lin.pˢᵗ, lin.κ)
    linearization_mixture_eos_at!(i, j, k, lin.γRᵐ, grid, microphysics, ρ, qᵛ, μ, constants)
    return nothing
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
    linearization_in_auxiliary_pass(model) ? nothing : refresh_linearization_basic_state!(substepper, model)

#####
##### Section 4 — Adaptive substep computation (acoustic CFL)
#####

"""
$(TYPEDSIGNATURES)

Compute the number of acoustic substeps ``N`` from the horizontal
acoustic CFL:

```math
N \\approx
\\left\\lceil \\frac{|\\Delta t| \\, c^{ac}}{\\nu \\, \\Delta x_\\min} \\right\\rceil ,
```

with ``c^{ac} = \\sqrt{γ^d R^d T_r}`` for a nominal reference
temperature ``T_r = 300\\,\\mathrm{K}`` and ``ν`` the target acoustic
Courant number `acoustic_cfl` (default `0.5`, the conventional ERF/WRF
target — equivalent to a safety factor of `2`).

``\\Delta x_\\min`` is the grid's [`minimum_horizontal_spacing`](@ref); the
substepper computes it once and passes the number in place of `grid`.
"""
function compute_acoustic_substeps(Δx_min::Number, Δt, thermodynamic_constants, acoustic_cfl)
    FT   = typeof(Δx_min)
    Rᵈ   = dry_air_gas_constant(thermodynamic_constants)
    cᵖᵈ  = thermodynamic_constants.dry_air.heat_capacity
    γᵈ   = cᵖᵈ / (cᵖᵈ - Rᵈ)
    cᵃᶜ  = sqrt(γᵈ * Rᵈ * FT(300))

    return max(1, ceil(Int, abs(FT(Δt)) * cᵃᶜ / (acoustic_cfl * Δx_min)))
end

compute_acoustic_substeps(grid, Δt, thermodynamic_constants, acoustic_cfl) =
    compute_acoustic_substeps(minimum_horizontal_spacing(grid), Δt, thermodynamic_constants, acoustic_cfl)

"""
$(TYPEDSIGNATURES)

The smallest horizontal spacing of `grid`, in the grid's float type: the length the acoustic
Courant number of [`compute_acoustic_substeps`](@ref) is measured against. A `Flat` direction
has no spacing and drops out of the minimum. On a GPU this is a reduction over the grid, so the
substepper evaluates it once when it is built rather than at every stage.
"""
function minimum_horizontal_spacing(grid)
    FT = eltype(grid)
    TX, TY, _ = topology(grid)
    Δx = TX === Flat ? typemax(FT) : minimum_xspacing(grid)
    Δy = TY === Flat ? typemax(FT) : minimum_yspacing(grid)
    return min(Δx, Δy)
end

@inline acoustic_substeps(N::Int, Δx_min, Δt, constants, acoustic_cfl) = N
@inline acoustic_substeps(::Nothing, Δx_min, Δt, constants, acoustic_cfl) =
    compute_acoustic_substeps(Δx_min, Δt, constants, acoustic_cfl)

#####
##### Section 5 — Stage substep distribution
#####

# Substeps for a stage covering Δt_stage = β·Δt: ⌈β·N⌉ for a fixed full-step count N,
# or the acoustic-CFL count for the stage interval when adaptive (substeps = nothing).
@inline _stage_substeps(N::Integer, β, Δt_stage, Δx_min, constants, acoustic_cfl) = max(1, ceil(Int, β * N))
@inline _stage_substeps(::Nothing, β, Δt_stage, Δx_min, constants, acoustic_cfl) =
    compute_acoustic_substeps(Δx_min, Δt_stage, constants, acoustic_cfl)

# Uniform full-step count N rounded up to a multiple of 6 so β·N is integral (ConstantSubstepSize,
# MonolithicFirstStage stages 2–3).
@inline function _uniform_substep_count(Δt, Δx_min, constants, acoustic_cfl, substeps)
    N_raw = acoustic_substeps(substeps, Δx_min, Δt, constants, acoustic_cfl)
    return max(6, 6 * cld(N_raw, 6))
end

# ProportionalSubsteps: each stage covers its own interval β·Δt with ⌈β·N⌉ substeps sized to tile
# it exactly (Δτ = β·Δt/Nτ). Exact coverage at the minimum count; Δτ may differ slightly by stage.
@inline function stage_substep_count_and_size(::ProportionalSubsteps, β_stage, Δt, Δx_min, constants, acoustic_cfl, substeps)
    Δt_stage = β_stage * Δt
    Nτ = _stage_substeps(substeps, β_stage, Δt_stage, Δx_min, constants, acoustic_cfl)
    return Nτ, Δt_stage / Nτ
end

# ConstantSubstepSize: one substep size Δτ = Δt/N shared by all stages (N a multiple of 6 ⇒ β·N integral).
@inline function stage_substep_count_and_size(::ConstantSubstepSize, β_stage, Δt, Δx_min, constants, acoustic_cfl, substeps)
    N = _uniform_substep_count(Δt, Δx_min, constants, acoustic_cfl, substeps)
    return max(1, round(Int, β_stage * N)), Δt / N
end

# MonolithicFirstStage: stage 1 collapses to one substep of size Δt/3; stages 2–3 like ConstantSubstepSize.
@inline function stage_substep_count_and_size(::MonolithicFirstStage, β_stage, Δt, Δx_min, constants, acoustic_cfl, substeps)
    β_stage < (1//3 + 1//2) / 2 && return 1, Δt / 3
    N = _uniform_substep_count(Δt, Δx_min, constants, acoustic_cfl, substeps)
    return max(1, round(Int, β_stage * N)), Δt / N
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

#####
##### Off-centered CN tridiag derivation
#####

# At face k the (ρw)′ CN update is
#
#   (ρw)′ₙ(k) = (ρw)′ₒ(k) + Δτ Gˢρw(k) − Δτ(ωˢ⁻ ∂z p′ₒ + ωᵐ⁺ ∂z p′ₙ) − Δτ g(ωˢ⁻ ρ′_faceₒ + ωᵐ⁺ ρ′_faceₙ),
#
# with ωᵐ⁺=(1+ε)/2, ωˢ⁻=(1−ε)/2 (ε=0 centered). p′ = Cᴸ(ρθ)′, Cᴸ≡γRᵐᴸΠᴸ, so the discrete PGF is the
# gradient of the product Cᴸ·(ρθ)′ (not Cᴸ_face·∂z(ρθ)′). Post-solve substitution (δτᵐ⁺=ωᵐ⁺Δτ):
#
#   ρ′ₙ(k)    = ρ′★(k)  − δτᵐ⁺((ρw)′ₙ(k+1) − (ρw)′ₙ(k))/Δz_c(k)
#   (ρθ)′ₙ(k) = ρθ′★(k) − δτᵐ⁺(θᴸ_face(k+1)(ρw)′ₙ(k+1) − θᴸ_face(k)(ρw)′ₙ(k))/Δz_c(k)
#
# yields the tridiag coefficients (ω≡ωᵐ⁺):
#
#   A[k,k+1] = −(ωΔτ)² Cᴸ(k)  θᴸ_face(k+1) rdz_c(k)   /Δzᶠ(k) − (ωΔτ)² g rdz_c(k)/2
#   A[k,k]   = 1 + (ωΔτ)² θᴸ_face(k)(Cᴸ(k)rdz_c(k)+Cᴸ(k−1)rdz_c(k−1))/Δzᶠ(k) + (ωΔτ)² g(rdz_c(k)−rdz_c(k−1))/2
#   A[k,k−1] = −(ωΔτ)² Cᴸ(k−1)θᴸ_face(k−1)rdz_c(k−1)/Δzᶠ(k) + (ωΔτ)² g rdz_c(k−1)/2
#
# γᵐRᵐᴸ (cell-centered γᵐRᵐ cached in `linearization_gamma_R_mixture`, refreshed per stage, interpolated to
# faces in-kernel) collapses bit-identically to dry γᵈRᵈ for qᵛ=qˡ=qⁱ=0.
#
# Implicit vertical damping: for `ThermalDivergenceDamping(damp_vertical=true)`, the vertical divergence
# damping folds into the same tridiag via a discrete vertical Laplacian on (ρw)′:
#
#   (ρw)′ₙ − ωαΔz² ∂z²(ρw)′ₙ = (ρw)′ₒ + (1−ω)αΔz² ∂z²(ρw)′ₒ.
#
# With dᵐ⁺≡ωαΔz², the −∂z² stencil adds  A[k,k±1] += −dᵐ⁺ rdz_c(k or k−1)/Δzᶠ(k),  A[k,k] += +dᵐ⁺(rdz_c(k)
# + rdz_c(k−1))/Δzᶠ(k); the matching (1−ω) term goes on the predictor RHS in `_build_vertical_rhs!`.
# Constant-Courant scaling γ_z=αΔz²/Δτ makes dᵐ⁺ Δτ-independent. `damp_vertical=false`/`NoDivergenceDamping` ⇒ 0.

# Implicit upper Rayleigh sponge → column-tridiag diagonal (Klemp, Dudhia & Hassiotis 2008): a layer of
# thickness `depth` below the lid damps (ρw)′ at peak `damping_rate` (1/s) × ramp shape, using the reference
# face coordinate `rnode` (terrain-following grids get a horizontally uniform sponge in r). CN-weighted:
# `|δτᵐ⁺|·rate·ramp` on the LHS diagonal, matched by `|δτˢ⁻|·rate·ramp·ρw_old` on the RHS in
# `_build_vertical_rhs!`. Local in z (no off-diagonal). `|δτ|` (not δτ) makes it a one-sided
# dissipative regularizer for either integration direction (forward through a sponge is intentionally not
# exactly invertible). Outside the layer the ramp vanishes and the tridiag is unaffected.
@inline sponge_term_diag(i, j, k, grid, ::Nothing, δτᵐ⁺) = zero(grid)

@inline function sponge_term_diag(i, j, k, grid, sponge::UpperSponge, δτᵐ⁺)
    z = rnode(i, j, k, grid, Center(), Center(), Face())
    return abs(δτᵐ⁺) * sponge.damping_rate * sponge.ramp(z, grid.Lz, sponge.depth)
end

@inline sponge_rhs(i, j, k, grid, ::Nothing, δτˢ⁻, ρw_old) = zero(grid)

@inline function sponge_rhs(i, j, k, grid, sponge::UpperSponge, δτˢ⁻, ρw_old)
    z = rnode(i, j, k, grid, Center(), Center(), Face())
    @inbounds return abs(δτˢ⁻) * sponge.damping_rate * sponge.ramp(z, grid.Lz, sponge.depth) * ρw_old[i, j, k]
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagLower, p, ::ZDirection,
                                 Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
    kᶠ     = k + 1
    Δz⁻¹ᶠ  = Δz⁻¹ᶜᶜᶠ(i, j, kᶠ, grid)
    Δz⁻¹ᵏ⁻ = Δz⁻¹ᶜᶜᶜ(i, j, kᶠ - 1, grid)

    @inbounds Cᵏ⁻ = γRᵐᴸ[i, j, kᶠ - 1] * Πᴸ[i, j, kᶠ - 1]
    θᵏ⁻ = ℑbzᵃᵃᶠ(i, j, kᶠ - 1, grid, θᴸ)

    pgf_term  = - δτᵐ⁺^2 * Cᵏ⁻ * θᵏ⁻ * Δz⁻¹ᵏ⁻ * Δz⁻¹ᶠ
    buoy_term = + δτᵐ⁺^2 * g * Δz⁻¹ᵏ⁻ / 2
    damp_term = - dᵐ⁺ * Δz⁻¹ᵏ⁻ * Δz⁻¹ᶠ

    # Upper sponge is local in z (Rayleigh-type), so no off-diagonal coupling.
    return pgf_term + buoy_term + damp_term
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagDiagonal, p, ::ZDirection,
                                 Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)

    Δz⁻¹ᶠ  = Δz⁻¹ᶜᶜᶠ(i, j, k, grid)
    Δz⁻¹ᵏ⁺ = Δz⁻¹ᶜᶜᶜ(i, j, k,     grid)
    Δz⁻¹ᵏ⁻ = Δz⁻¹ᶜᶜᶜ(i, j, k - 1, grid)

    @inbounds begin
        Cᵏ⁺ = γRᵐᴸ[i, j, k]     * Πᴸ[i, j, k]
        Cᵏ⁻ = γRᵐᴸ[i, j, k - 1] * Πᴸ[i, j, k - 1]
    end

    θᶜᶜᶠ = ℑbzᵃᵃᶠ(i, j, k, grid, θᴸ)

    pgf_diag   = δτᵐ⁺^2 * θᶜᶜᶠ * (Cᵏ⁺ * Δz⁻¹ᵏ⁺ + Cᵏ⁻ * Δz⁻¹ᵏ⁻) * Δz⁻¹ᶠ
    buoy_diag  = δτᵐ⁺^2 * g * (Δz⁻¹ᵏ⁺ - Δz⁻¹ᵏ⁻) / 2
    damp_diag  = dᵐ⁺ * (Δz⁻¹ᵏ⁺ + Δz⁻¹ᵏ⁻) * Δz⁻¹ᶠ
    spnge_diag = sponge_term_diag(i, j, k, grid, sponge, δτᵐ⁺)

    return 1 + (pgf_diag + buoy_diag + damp_diag + spnge_diag) * (k > 1)
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagUpper, p, ::ZDirection,
                                 Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)

    Δz⁻¹ᶠ  = Δz⁻¹ᶜᶜᶠ(i, j, k, grid)
    Δz⁻¹ᵏ⁺ = Δz⁻¹ᶜᶜᶜ(i, j, k, grid)

    @inbounds Cᵏ⁺ = γRᵐᴸ[i, j, k] * Πᴸ[i, j, k]
    θᵏ⁺ = ℑbzᵃᵃᶠ(i, j, k + 1, grid, θᴸ)

    pgf_term  = - δτᵐ⁺^2 * Cᵏ⁺ * θᵏ⁺ * Δz⁻¹ᵏ⁺ * Δz⁻¹ᶠ
    buoy_term = - δτᵐ⁺^2 * g * Δz⁻¹ᵏ⁺ / 2
    damp_term = - dᵐ⁺ * Δz⁻¹ᵏ⁺ * Δz⁻¹ᶠ

    # Upper sponge is local in z (Rayleigh-type), so no off-diagonal coupling.
    return (pgf_term + buoy_term + damp_term) * (k > 1)
end

#####
##### Section 7 — Slow vertical-momentum tendency assembly
#####

# The full vertical-momentum equation is
#
# ∂t (ρw) + ∇·(ρw u) + ∂z p + g ρ = 0
#
# The dynamics kernel runs with `SlowDynamics` for SplitExplicit,
# which zeroes the PGF and buoyancy in `Gⁿρw`. We reinstate the
# **Uᴸ-state** PGF and buoyancy here so the slow ρw tendency has the
# form
#
#   Gˢρw = -∇·(ρw u)  -  ∂z(pᴸ - pᵣ)  -  g · (ρᴸ - ρᵣ)   (with reference)
#   Gˢρw = -∇·(ρw u)  -  ∂z pᴸ        -  g · ρᴸ           (no reference)
#
# and the per-substep linearized forces operate on the perturbations:
#
#   ∂t (ρw)′ = Gˢρw - γRᵐ · Πᴸ · ∂z((ρθ)′)  -  g · ρ′
#
# Total force = Gˢρw + perturbation force = full ∂t(ρw) at the
# linearization-consistent level. With a hydrostatic-balanced reference
# state, the reference subtraction makes Gˢρw vanish identically on a
# resting atmosphere (no FP-rounding noise).

# Height-coordinate (non-terrain) slow vertical-momentum tendency. The terrain method is dispatched
# separately (`::TerrainCompressibleModel`), so here the single `reference_state`, when present, is
# a 1D-column `ExnerReferenceState`; `nothing` selects the full-pressure fallback.
function assemble_slow_vertical_momentum_tendency!(substepper::AcousticSubstepper, model, β_stage = nothing)
    grid = model.grid
    arch = architecture(grid)
    g    = convert(eltype(grid), model.thermodynamic_constants.gravitational_acceleration)
    Gⁿρw = model.timestepper.Gⁿ.ρw

    ref = model.dynamics.reference_state

    if ref isa Nothing
        launch!(arch, grid, :xyz, _assemble_slow_vertical_momentum_tendency_no_ref!,
                substepper.slow_vertical_momentum_tendency,
                Gⁿρw,
                model.dynamics.pressure,
                model.dynamics.total_density,
                grid, g)
    else
        launch!(arch, grid, :xyz, _assemble_slow_vertical_momentum_tendency!,
                substepper.slow_vertical_momentum_tendency,
                Gⁿρw,
                model.dynamics.pressure,
                model.dynamics.total_density,
                ref.pressure, ref.density,
                grid, g)
    end

    return nothing
end

# Slow-tendency assembly with reference state. Buoyancy uses the diagnosed TOTAL density
# `ρ = ρᵈ + Σρˣ` as `ρᴸ` (no virtual-density factor): in conservation-form momentum
# `∂t(ρw) = -∂z p - g ρ`, gravity acts on the total mass of all species. The prognostic mass
# variable is the dry density ρᵈ (its continuity has no sedimentation source); total ρ is
# reconstructed each update. Across the acoustic substeps the water densities are frozen, so the
# evolving density perturbation `ρ′ = ρᵈ′` is also the total-density perturbation, and the
# moisture loading enters exactly once — here, through the stage-entry total `ρᴸ`.
@kernel function _assemble_slow_vertical_momentum_tendency!(Gˢρw, Gⁿρw, pᴸ, ρᴸ, pᵣ, ρᵣ, grid, g)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Reference-subtracted PGF and buoyancy: at Uᴸ = reference state
        # both terms are exactly zero by construction of the reference.
        ∂z_p′ = ∂zᶜᶜᶠ(i, j, k, grid, δϕ, pᴸ, pᵣ)
        ρ′ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, δϕ, ρᴸ, ρᵣ)

        Gˢρw[i, j, k] = (Gⁿρw[i, j, k] - ∂z_p′ - g * ρ′ᶜᶜᶠ) * (k > 1)
    end
end

# Field perturbation about a reference (used for both pressure and density).
@inline δϕ(i, j, k, grid, ϕᴸ, ϕᵣ) = @inbounds ϕᴸ[i, j, k] - ϕᵣ[i, j, k]

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

# Initialize perturbation prognostics at each WS-RK3 stage start (Skamarock
# & Klemp 2008, above eq. 16): substep variables are deviations from the
# linearization base Uᴸ (refreshed by `prepare_acoustic_cache!` just before).
# The WS-RK3 invariant ``U^{(k)} = U(t) + β_k Δt R(U^{(k-1)})`` requires each
# stage to integrate from U(t) ≡ Uᴸ_outer. The WRF/MPAS trick: init the
# perturbations to the rewind ``(U_outer − Uᴸ)`` so the substep's starting
# full state ``Uᴸ + (U_outer − Uᴸ) = U_outer`` regardless of Uᴸ. Stage 1
# rewind = 0; stages 2–3 pick up the previous-stage update. `_recover_full_state!`
# then uses per-stage Uᴸ as base, collapsing back to ``U_outer + Δevolved`` —
# preserving the invariant. Auxiliary fields (predictors, divergence/damping
# workspace) reset to zero; they carry no stage-to-stage history.
function initialize_stage_perturbations!(substepper, model, Uᴸ_outer, zero_entry = false, path = nothing)
    grid = model.grid
    arch = architecture(grid)

    # Split column path (A2-bce c/e in spirit, handoff a38e2578): one launch, without the dead workspace zeros.
    if FUSED_STAGE_INITIALIZATION[] && path isa SplitAcousticColumn
        rewind = vertical_momentum_rewind(model, Uᴸ_outer)
        if zero_entry
            launch!(arch, grid, :xyz, _zero_stage_accumulators_and_perturbations!,
                    substepper.time_averaged_velocities.u,
                    substepper.time_averaged_velocities.v,
                    substepper.time_averaged_velocities.w,
                    substepper.density_perturbation,
                    substepper.density_potential_temperature_perturbation,
                    substepper.momentum_perturbation.u,
                    substepper.momentum_perturbation.v,
                    substepper.momentum_perturbation.w)
            fill_halo_regions!(substepper.density_perturbation)
            fill_halo_regions!(substepper.density_potential_temperature_perturbation)
            map(fill_halo_regions!, substepper.momentum_perturbation)
            return nothing
        elseif !isnothing(rewind)
            ρᵡ = thermodynamic_density(model.formulation)
            ρᵡ_name = thermodynamic_density_name(model.formulation)
            launch!(arch, grid, :xyz, _initialize_stage_accumulators_and_perturbations!,
                    substepper.time_averaged_velocities.u,
                    substepper.time_averaged_velocities.v,
                    substepper.time_averaged_velocities.w,
                    substepper.density_perturbation,
                    substepper.density_potential_temperature_perturbation,
                    substepper.momentum_perturbation.u,
                    substepper.momentum_perturbation.v,
                    substepper.momentum_perturbation.w,
                    Uᴸ_outer.ρᵈ, Uᴸ_outer[ρᵡ_name], Uᴸ_outer.ρu, Uᴸ_outer.ρv,
                    model.dynamics.dry_density, ρᵡ, model.momentum.ρu, model.momentum.ρv,
                    rewind)
            fill_halo_regions!(substepper.density_perturbation)
            fill_halo_regions!(substepper.density_potential_temperature_perturbation)
            map(fill_halo_regions!, substepper.momentum_perturbation)
            return nothing
        end
    end

    # Stage 1: Uᴸ_outer is the parent copy of the stage-entry state, so every rewind
    # perturbation is x − x = +0 (finite state). Store the literal +0 in the same launch
    # as the workspaces instead of reading ten fields (bitwise for finite states).
    if zero_entry
        launch!(arch, grid, :xyz, _zero_stage_workspaces_and_perturbations!,
                substepper.previous_density_potential_temperature_perturbation,
                substepper.density_predictor,
                substepper.density_potential_temperature_predictor,
                substepper.time_averaged_velocities.u,
                substepper.time_averaged_velocities.v,
                substepper.time_averaged_velocities.w,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation.u,
                substepper.momentum_perturbation.v,
                substepper.momentum_perturbation.w)

        AtmosphereModels.fill_halo_regions_together!((substepper.density_perturbation,
                                                      substepper.density_potential_temperature_perturbation,
                                                      values(substepper.momentum_perturbation)...))
        return nothing
    end

    # Zero the auxiliary workspaces (predictor/damping scratch) and the
    # time-averaged-velocity accumulator slots in one launch. The velocity slots
    # then accumulate raw `momentum_perturbation` each substep and are normalized
    # by `finalize_time_averaged_velocity!` at stage end.
    launch!(arch, grid, :xyz, _zero_stage_workspaces!,
            substepper.previous_density_potential_temperature_perturbation,
            substepper.density_predictor,
            substepper.density_potential_temperature_predictor,
            substepper.time_averaged_velocities.u,
            substepper.time_averaged_velocities.v,
            substepper.time_averaged_velocities.w)

    # Prognostic perturbations: rewind init. The per-stage Uᴸ for ρ and
    # ρθ is held in `model.dynamics.dry_density` and the formulation's ρθ
    # field — untouched by the substep loop, so they equal the per-stage
    # linearization base.
    ρᵡ = thermodynamic_density(model.formulation)
    ρᵡ_name = thermodynamic_density_name(model.formulation)
    launch!(arch, grid, :xyz, _initialize_stage_perturbations!,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            Uᴸ_outer.ρᵈ, Uᴸ_outer[ρᵡ_name], Uᴸ_outer.ρu, Uᴸ_outer.ρv,
            model.dynamics.dry_density, ρᵡ, model.momentum.ρu, model.momentum.ρv)
    # ρw is dispatched: terrain models initialize the contravariant ρw̃ instead.
    initialize_vertical_momentum_perturbation!(substepper, model, Uᴸ_outer)

    AtmosphereModels.fill_halo_regions_together!((substepper.density_perturbation,
                                                  substepper.density_potential_temperature_perturbation,
                                                  values(substepper.momentum_perturbation)...))

    return nothing
end

"""
Whether the split column path initializes a stage in one launch: the transport accumulators zeroed and the five
perturbations rewound (or zeroed in stage 1), without zeroing the predictors and the (ρθ)′ stash, which
`_build_predictors!` overwrites over the whole interior before any read (their halos are never written by the zeroing).
Default `true`; `BREEZE_FUSED_STAGE_INITIALIZATION=false` at load turns it off.
"""
const FUSED_STAGE_INITIALIZATION = Ref(true)

# The rewind of the vertical momentum perturbation, evaluated per cell: ρw⁰ − ρw; terrain models (with the loaded
# stage-entry ρw̃) override it, and return `nothing` where the rewind needs its own launch.
struct VerticalRewind{O, S}
    outer :: O
    stage :: S
end

Adapt.adapt_structure(to, r::VerticalRewind) = VerticalRewind(adapt(to, r.outer), adapt(to, r.stage))

@inline rewind_value(r::VerticalRewind, i, j, k) = @inbounds r.outer[i, j, k] - r.stage[i, j, k]

vertical_momentum_rewind(model, Uᴸ_outer) = VerticalRewind(Uᴸ_outer.ρw, model.momentum.ρw)

@kernel function _zero_stage_accumulators_and_perturbations!(u_avg, v_avg, w_avg, ρ′, ρθ′, ρu′, ρv′, ρw′)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        u_avg[i, j, k] = 0
        v_avg[i, j, k] = 0
        w_avg[i, j, k] = 0
        ρ′[i, j, k] = 0
        ρθ′[i, j, k] = 0
        ρu′[i, j, k] = 0
        ρv′[i, j, k] = 0
        ρw′[i, j, k] = 0
    end
end

@kernel function _initialize_stage_accumulators_and_perturbations!(u_avg, v_avg, w_avg, ρ′, ρθ′, ρu′, ρv′, ρw′,
                                                                   ρ_outer, ρθ_outer, ρu_outer, ρv_outer,
                                                                   ρ_stage, ρθ_stage, ρu_stage, ρv_stage, rewind)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        u_avg[i, j, k] = 0
        v_avg[i, j, k] = 0
        w_avg[i, j, k] = 0
        ρ′[i, j, k]  = ρ_outer[i, j, k]  - ρ_stage[i, j, k]
        ρθ′[i, j, k] = ρθ_outer[i, j, k] - ρθ_stage[i, j, k]
        ρu′[i, j, k] = ρu_outer[i, j, k] - ρu_stage[i, j, k]
        ρv′[i, j, k] = ρv_outer[i, j, k] - ρv_stage[i, j, k]
        ρw′[i, j, k] = rewind_value(rewind, i, j, k)
    end
end

# Zero the six stage-start workspace/accumulator fields in one launch.
@kernel function _zero_stage_workspaces!(a, b, c, d, e, f)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        a[i, j, k] = 0
        b[i, j, k] = 0
        c[i, j, k] = 0
        d[i, j, k] = 0
        e[i, j, k] = 0
        f[i, j, k] = 0
    end
end

@kernel function _zero_stage_workspaces_and_perturbations!(a, b, c, d, e, f, ρ′, ρθ′, ρu′, ρv′, ρw′)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        a[i, j, k] = 0
        b[i, j, k] = 0
        c[i, j, k] = 0
        d[i, j, k] = 0
        e[i, j, k] = 0
        f[i, j, k] = 0
        ρ′[i, j, k] = 0
        ρθ′[i, j, k] = 0
        ρu′[i, j, k] = 0
        ρv′[i, j, k] = 0
        ρw′[i, j, k] = 0
    end
end

# Rewind-initialize the four non-vertical perturbations (ρ′, ρθ′, ρu′, ρv′) in one launch.
@kernel function _initialize_stage_perturbations!(ρ′, ρθ′, ρu′, ρv′,
                                                  ρ_outer, ρθ_outer, ρu_outer, ρv_outer,
                                                  ρ_stage, ρθ_stage, ρu_stage, ρv_stage)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρ′[i, j, k]  = ρ_outer[i, j, k]  - ρ_stage[i, j, k]
        ρθ′[i, j, k] = ρθ_outer[i, j, k] - ρθ_stage[i, j, k]
        ρu′[i, j, k] = ρu_outer[i, j, k] - ρu_stage[i, j, k]
        ρv′[i, j, k] = ρv_outer[i, j, k] - ρv_stage[i, j, k]
    end
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
# Linearized at Uᴸ, the horizontal pressure gradient splits as
#
#   ∂x p_full = ∂x pᴸ + ∂x(Cᴸ (ρθ)′),  Cᴸ = γRᵐᴸ Πᴸ
#
# the first piece frozen at the linearization point, the second the
# perturbation force. `ExnerReferenceState` depends only on z, so ∂x pᵣ ≡ 0
# and no horizontal pressure-perturbation field is needed.
#
#   (ρu)′^{τ+Δτ} = (ρu)′^τ + Δτ (Gⁿρu − ∂x pᴸ − ∂x(Cᴸ (ρθ)′))
#   (ρv)′^{τ+Δτ} = (ρv)′^τ + Δτ (Gⁿρv − ∂y pᴸ − ∂y(Cᴸ (ρθ)′))
#
# `Gⁿρu` (SlowDynamics) carries non-pressure slow terms with PGF zeroed;
# we reinstate the frozen large-step PGF here (MPAS keeps it in `tend_u_euler`).
@kernel function _explicit_horizontal_step!(ρu′, ρv′, grid, dynamics, Δτ, ρθ′, Πᴸ,
                                            Gⁿρu, Gⁿρv, γRᵐᴸ, pressure_gradients)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂x_p, ∂y_p = horizontal_pressure_gradients!(pressure_gradients, i, j, k, grid, dynamics,
                                                    ρθ′, Πᴸ, γRᵐᴸ)

        ρu′[i, j, k] += Δτ * (Gⁿρu[i, j, k] - ∂x_p)
        ρv′[i, j, k] += Δτ * (Gⁿρv[i, j, k] - ∂y_p)
    end
end

@inline δpᴸ(i, j, k, grid, ρθ′, Πᴸ, γRᵐᴸ) = @inbounds γRᵐᴸ[i, j, k] * Πᴸ[i, j, k] * ρθ′[i, j, k]

@inline ∇ˣp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ) = ∂xᶠᶜᶜ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
@inline ∇ʸp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ) = ∂yᶜᶠᶜ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
@inline ∇ᶻp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ) = ∂zᶜᶜᶠ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)

# The explicit step's total gradients, evaluated in place. The `TerrainPressureGradients` method
# (loads ∂pᴸ, stores ∂p′) is in terrain_compressible_physics.jl.
@inline function horizontal_pressure_gradients!(::Nothing, i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    ∂x_pᴸ  = AtmosphereModels.x_pressure_gradient(i, j, k, grid, dynamics)
    ∂x_p′  = ∇ˣp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    ∂y_pᴸ  = AtmosphereModels.y_pressure_gradient(i, j, k, grid, dynamics)
    ∂y_p′  = ∇ʸp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    return ∂x_pᴸ + ∂x_p′, ∂y_pᴸ + ∂y_p′
end

# The vertical rhs's ∇ᶻp′ of the old (ρθ)′, evaluated in place (stored method in terrain file).
@inline ∇ᶻp′ˢ⁻(::Nothing, i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ) = ∇ᶻp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)

"""
$(TYPEDSIGNATURES)

Run the `Nτ` acoustic substeps of one RK3 stage by calling [`acoustic_substep!`](@ref) for each.

The substeps run under `ReactantCore.@trace`, which is a plain loop on ordinary arrays and a
single traced `while` loop under Reactant, so the compiled program holds one copy of the substep
body rather than `Nτ`.
"""
function acoustic_substep_loop!(Nτ, model, substepper, stage, advection)
    # Without divergence damping nothing changes the momentum perturbations after the last
    # substep's post-solve recovery, so that substep stores no transport accumulators: the
    # finalization reads `accumulator + perturbation` (the very sum the recovery would store).
    # The tiled column kernel always accumulates, so it is never peeled.
    peel = substepper.damping isa NoDivergenceDamping &&
           !(acoustic_column_path(model.grid, model.dynamics, advection) isa TiledAcousticColumn)
    Nloop = peel ? Nτ - 1 : Nτ
    @trace track_numbers=false for _ in 1:Nloop
        acoustic_substep!(model, substepper, stage, advection)
    end
    folded = peel && folds_last_substep_recovery(model, advection)
    peel && acoustic_substep!(model, substepper, stage, advection, folded ? FoldedRecovery() : nothing)

    # Stage-final (ρu)′, (ρv)′ halos, read with horizontal neighbours by the terrain
    # recovery; the per-substep post-damping fill they replace was otherwise dead.
    fill_halo_regions!(substepper.momentum_perturbation.u)
    fill_halo_regions!(substepper.momentum_perturbation.v)
    return peel, folded
end

#####
##### The last substep's post-solve recovery in the stage-end recovery (port of handoff A2-bce a38e2578, part b)
#####

# Marker passed as the accumulators of a peeled last substep whose post-solve recovery is folded.
struct FoldedRecovery end

"""
Whether a peeled last substep (no divergence damping, so the momentum perturbations do not change after its
solve) ends after its solve, with its post-solve recovery of ρ′ and (ρθ)′ and the open-boundary relaxation of
their rim cells evaluated in registers by the stage-end recovery (`_recover_full_state_after_solve!`), the only
other reader before the next stage reinitializes them. Split column path only. Default `true`;
`BREEZE_FOLDED_LAST_SUBSTEP_RECOVERY=false` at load turns it off.
"""
const FOLDED_LAST_SUBSTEP_RECOVERY = Ref(true)

folds_last_substep_recovery(model, advection) =
    FOLDED_LAST_SUBSTEP_RECOVERY[] && acoustic_column_path(model.grid, model.dynamics, advection) isa SplitAcousticColumn

# The open lateral sides `apply_open_boundary_relaxation!` relaxes, with the factor; `nothing` without one.
function open_boundary_rim(substepper, model, grid)
    bcs_u = model.momentum.ρu.boundary_conditions
    bcs_v = model.momentum.ρv.boundary_conditions
    west, east = is_active_open_bc(bcs_u.west), is_active_open_bc(bcs_u.east)
    south, north = is_active_open_bc(bcs_v.south), is_active_open_bc(bcs_v.north)
    (west | east | south | north) || return nothing
    Nx, Ny, _ = size(grid)
    return (; west, east, south, north, Nx, Ny, α = substepper.open_boundary_relaxation)
end

# `c′ += α * ((cᴸ[halo] − cᴸ[boundary]) / 2 − c′)` of `_relax_open_boundary_x!` / `_y!`, rounded to the field type
# as the store would.
@inline relaxed_perturbation(c′::T, cᴴ, cᴮ, α) where T = convert(T, c′ + α * ((cᴴ - cᴮ) / 2 - c′))

@inline relax_rim_perturbations(::Nothing, i, j, k, ρ′, ρθ′, ρᴸ, ρθᴸ) = (ρ′, ρθ′)

# West, east (the x launches), then south, north (the y launches), as `apply_open_boundary_relaxation!` orders them.
@inline function relax_rim_perturbations(rim, i, j, k, ρ′, ρθ′, ρᴸ, ρθᴸ)
    @inbounds begin
        if rim.west & (i == 1)
            ρ′  = relaxed_perturbation(ρ′,  ρᴸ[0, j, k],  ρᴸ[1, j, k],  rim.α)
            ρθ′ = relaxed_perturbation(ρθ′, ρθᴸ[0, j, k], ρθᴸ[1, j, k], rim.α)
        end
        if rim.east & (i == rim.Nx)
            ρ′  = relaxed_perturbation(ρ′,  ρᴸ[rim.Nx + 1, j, k],  ρᴸ[rim.Nx, j, k],  rim.α)
            ρθ′ = relaxed_perturbation(ρθ′, ρθᴸ[rim.Nx + 1, j, k], ρθᴸ[rim.Nx, j, k], rim.α)
        end
        if rim.south & (j == 1)
            ρ′  = relaxed_perturbation(ρ′,  ρᴸ[i, 0, k],  ρᴸ[i, 1, k],  rim.α)
            ρθ′ = relaxed_perturbation(ρθ′, ρθᴸ[i, 0, k], ρθᴸ[i, 1, k], rim.α)
        end
        if rim.north & (j == rim.Ny)
            ρ′  = relaxed_perturbation(ρ′,  ρᴸ[i, rim.Ny + 1, k],  ρᴸ[i, rim.Ny, k],  rim.α)
            ρθ′ = relaxed_perturbation(ρθ′, ρθᴸ[i, rim.Ny + 1, k], ρθᴸ[i, rim.Ny, k], rim.α)
        end
    end
    return ρ′, ρθ′
end

# The two expressions of `post_solve_recovery!` (rounded to the perturbation field type, as stored), the rim
# relaxation, then the sums of `_recover_full_state!`. Reads at other cells only arrays it does not write.
@kernel function _recover_full_state_after_solve!(ρ, ρθ, m, ρw′, ρu′, ρv′, ρ′★, ρθ′★, T′,
                                                  ρᴸ, ρuᴸ, ρvᴸ, ρwᴸ, ρθᴸ,
                                                  grid, dynamics, δτᵐ⁺, θᴸ, rim)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρ′ⁿ  = convert(T′, ρ′★[i, j, k]  - δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, Fʷ, dynamics, ρu′, ρv′, ρw′))
        ρθ′ⁿ = convert(T′, ρθ′★[i, j, k] - δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, θFᶻ, θᴸ, dynamics, ρu′, ρv′, ρw′))
        ρ′ⁿ, ρθ′ⁿ = relax_rim_perturbations(rim, i, j, k, ρ′ⁿ, ρθ′ⁿ, ρᴸ, ρθᴸ)

        ρᵐ⁺  = ρᴸ[i, j, k]  + ρ′ⁿ
        ρθᵐ⁺ = ρθᴸ[i, j, k] + ρθ′ⁿ
        ρuᵐ⁺ = ρuᴸ[i, j, k] + ρu′[i, j, k]
        ρvᵐ⁺ = ρvᴸ[i, j, k] + ρv′[i, j, k]
        ρwᵐ⁺ = acoustic_recovered_vertical_momentum(i, j, k, grid, dynamics, ρwᴸ[i, j, k], ρu′, ρv′, ρw′)

        ρ[i, j, k]  = ρᵐ⁺
        ρθ[i, j, k] = ρθᵐ⁺

        m.ρu[i, j, k] = ρuᵐ⁺
        m.ρv[i, j, k] = ρvᵐ⁺
        m.ρw[i, j, k] = ρwᵐ⁺
    end
end

# Build per-column predictors `ρ′★`, `ρθ′★` (cell centers) AND
# the explicit RHS for the tridiagonal `(ρw)′ᵐ⁺` solve at z-faces.
#
# Off-centered Crank–Nicolson with
#     - ω = forward_weight = "new side" weight
#     - 1 - ω = "old side" weight
#     - δτˢ⁻ = (1 − ω) Δτ
#
# See derivation in the split-explicit derivation in `docs/src/compressible_dynamics.md`.
@kernel function _build_predictors!(ρ′★, ρθ′★, ρθ′ˢ⁻, ρ′, ρθ′, ρw′, ρu′, ρv′,
                                    grid, dynamics, Δτ, δτˢ⁻, Gˢρ, Gˢρᵡ, fθ, θᴸ)

    i, j, k = @index(Global, NTuple)
    build_predictors!(i, j, k, ρ′★, ρθ′★, ρθ′ˢ⁻, ρ′, ρθ′, ρw′, ρu′, ρv′,
                      grid, dynamics, Δτ, δτˢ⁻, Gˢρ, Gˢρᵡ, fθ, θᴸ)
end

@inline function build_predictors!(i, j, k, ρ′★, ρθ′★, ρθ′ˢ⁻, ρ′, ρθ′, ρw′, ρu′, ρv′,
                                   grid, dynamics, Δτ, δτˢ⁻, Gˢρ, Gˢρᵡ, fθ, θᴸ)
    @inbounds begin
        ρθ′ˢ⁻[i, j, k] = ρθ′[i, j, k] # stash old (ρθ)′ for the divergence damping

        V⁻¹ = V⁻¹ᶜᶜᶜ(i, j, k, grid)
        ∇ʰ_M  = div_xyᶜᶜᶜ(i, j, k, grid, ρu′, ρv′)
        ∇ʰ_θM = V⁻¹ * (δxᶜᵃᵃ(i, j, k, grid, θFˣ, θᴸ, ρu′) + δyᵃᶜᵃ(i, j, k, grid, θFʸ, θᴸ, ρv′))

        ρ′★[i, j, k] = ρ′[i, j, k] +
            Δτ * (Gˢρ[i, j, k] - ∇ʰ_M) - δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, Fʷ, dynamics, ρu′, ρv′, ρw′)

        ρθ′★[i, j, k] = ρθ′[i, j, k] +
            Δτ * (fθ * Gˢρᵡ[i, j, k] - ∇ʰ_θM) - δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, θFᶻ, θᴸ, dynamics, ρu′, ρv′, ρw′)
    end
    return nothing
end

# Face-level RHS for `(ρw)′ᵐ⁺`: split weights δτᵐ⁺ (predictor) and δτˢ⁻ (old-step) per
# derivation (15). `dˢ⁻ = (1−ω) α Δz²` is the explicit half of the implicit vertical
# damping (0 when damping off). Boundary rows: f[1] = 0 (matches b[1] = 1 ⇒ (ρw)′[1] = 0);
# top face Nz+1 lives outside the solver (impenetrability w(top) = 0).
@kernel function _build_vertical_rhs!(ρw′_rhs, ρ′★, ρθ′★, ρ′, ρθ′, ρw′,
                                      grid, dynamics, Δτ, δτᵐ⁺, δτˢ⁻, Πᴸ, γRᵐᴸ, g, dˢ⁻,
                                      fw, Gˢρw, sponge, pressure_gradients)
    i, j, k = @index(Global, NTuple)
    @inbounds ρw′_rhs[i, j, k] = vertical_rhs(i, j, k, ρ′★, ρθ′★, ρ′, ρθ′, ρw′,
                                              grid, dynamics, Δτ, δτᵐ⁺, δτˢ⁻, Πᴸ, γRᵐᴸ, g, dˢ⁻,
                                              fw, Gˢρw, sponge, pressure_gradients)
end

#####
##### The vertical rhs as a column launch with its vertical stencil carried (port of handoff bc1c3e6e, D3 S1′)
#####
##### On terrain-following grids each face of `_build_vertical_rhs!` evaluates eight chain-rule gradients of
##### Cᴸ(ρθ)′★ over 24 vertical differences, each evaluated by four faces. `_build_vertical_rhs_carried!` walks
##### each column upward and evaluates per face only what the face below has not (`carried_vertical_rhs`,
##### terrain_compressible_physics.jl): the same expressions on the same operands, so the same bits.

"""
Whether the split substep builds the (ρw)′ rhs with the column launch `_build_vertical_rhs_carried!` where the
dynamics has a carried form (`carries_vertical_rhs`; default `true`, `BREEZE_CARRIED_VERTICAL_RHS=false` at load
turns it off). Needs the stored terrain pressure gradients. Bitwise identical either way.
"""
const CARRIED_VERTICAL_RHS = Ref(true)

carries_vertical_rhs(dynamics, pressure_gradients, grid) = false

# Faces 1 and Nz + 1 hold zero, as `_build_vertical_rhs!` stores them; faces 2…Nz the carried rhs.
@kernel function _build_vertical_rhs_carried!(f, rhs)
    i, j = @index(Global, NTuple)
    Nz = size(rhs.grid, 3)
    @inbounds begin
        state = carried_vertical_rhs_state(rhs, i, j)
        f[i, j, 1] = zero(eltype(f))
        for k = 2:Nz
            rᵏ, state = carried_vertical_rhs(rhs, state, i, j, k)
            f[i, j, k] = rᵏ
        end
        f[i, j, Nz + 1] = zero(eltype(f))
    end
end

@inline function vertical_rhs(i, j, k, ρ′★, ρθ′★, ρ′, ρθ′, ρw′,
                              grid, dynamics, Δτ, δτᵐ⁺, δτˢ⁻, Πᴸ, γRᵐᴸ, g, dˢ⁻,
                              fw, Gˢρw, sponge, pressure_gradients = nothing)
    Nz = size(grid, 3)

    @inbounds begin
        ∂r_p′★  = ∇ᶻp′(i, j, k, grid, dynamics, ρθ′★, Πᴸ, γRᵐᴸ)
        ∂r_p′ˢ⁻ = ∇ᶻp′ˢ⁻(pressure_gradients, i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)
        Gρwᵖ = δτˢ⁻ * ∂r_p′ˢ⁻ + δτᵐ⁺ * ∂r_p′★ # pressure gradient

        ρ′ᶜᶜᶠ★  = ℑzᵃᵃᶠ(i, j, k, grid, ρ′★)
        ρ′ᶜᶜᶠˢ⁻ = ℑzᵃᵃᶠ(i, j, k, grid, ρ′)
        Gρwᵇ = g * (δτˢ⁻ * ρ′ᶜᶜᶠˢ⁻ + δτᵐ⁺ * ρ′ᶜᶜᶠ★) # buoyancy

        ∂z²_ρw′ˢ⁻ = ∂zᶜᶜᶠ(i, j, k, grid, ∂zᶜᶜᶜ, ρw′)
        Gρwᵈ = - dˢ⁻ * ∂z²_ρw′ˢ⁻ # damping

        Gρwˢ = sponge_rhs(i, j, k, grid, sponge, δτˢ⁻, ρw′)

        rhs = ρw′[i, j, k] + Δτ * fw * Gˢρw[i, j, k] - Gρwᵖ - Gρwᵇ - Gρwᵈ - Gρwˢ

        # Interior faces 2:Nz carry the acoustic RHS; boundary faces 1 and Nz+1 are pinned to 0
        # (tridiag b[1] = 1 ⇒ (ρw)′[1] = 0; impenetrability w(top) = 0). Branchless (launched over
        # 1:Nz+1, no warp divergence); the boundary stencils read unfilled k=0/Nz+1 halos but the
        # result is discarded.
        return ifelse((k != 1) & (k != Nz + 1), rhs, zero(rhs))
    end
end

# θᴸ · (ρu)′ at an x-face. Used in the area-weighted horizontal
# divergence of the perturbation θ-flux.
@inline θFˣ(i, j, k, grid, θᴸ, ρu′) = @inbounds Axᶠᶜᶜ(i, j, k, grid) * ℑxᶠᵃᵃ(i, j, k, grid, θᴸ) * ρu′[i, j, k]
@inline θFʸ(i, j, k, grid, θᴸ, ρv′) = @inbounds Ayᶜᶠᶜ(i, j, k, grid) * ℑyᵃᶠᵃ(i, j, k, grid, θᴸ) * ρv′[i, j, k]

# θᴸ · (ρw)′ at a z-face. Used in the vertical part of the perturbation
# θ-flux divergence; passed to `∂zᶜᶜᶜ` so the divergence is computed at
# cell centers from the face-located product.
@inline Fʷ(i, j, k, grid, dynamics, ρu′, ρv′, ρw′) = @inbounds ρw′[i, j, k]
@inline θFᶻ(i, j, k, grid, θᴸ, dynamics, ρu′, ρv′, ρw′) = ℑbzᵃᵃᶠ(i, j, k, grid, θᴸ) * Fʷ(i, j, k, grid, dynamics, ρu′, ρv′, ρw′)
@inline ℑb_wθ(i, j, k, grid, w, θ) = @inbounds w[i, j, k] * ℑbzᵃᵃᶠ(i, j, k, grid, θ)

# Post-solve recovery: substitute the tridiag-solved `(ρw)′ᵐ⁺` back
# into the `ρ′★`, `ρθ′★` predictors to get `ρ′ᵐ⁺`, `ρθ′ᵐ⁺`
#
# NOTE: this runs *before* the divergence damping, so `avg` accumulates the pre-damping (ρu)′,(ρv)′.
# Identical for dry runs (the transport velocity is unused without scalars); for moist
# runs it omits the per-substep damping increment from the transport average.
@kernel function _post_solve_recovery!(ρ′, ρθ′, ρw′, ρu′, ρv′, ρ′★, ρθ′★, avg, grid, dynamics, δτᵐ⁺, θᴸ)
    i, j, k = @index(Global, NTuple)
    post_solve_recovery!(i, j, k, ρ′, ρθ′, ρw′, ρu′, ρv′, ρ′★, ρθ′★, avg, grid, dynamics, δτᵐ⁺, θᴸ)
end

@inline function post_solve_recovery!(i, j, k, ρ′, ρθ′, ρw′, ρu′, ρv′, ρ′★, ρθ′★, avg, grid, dynamics, δτᵐ⁺, θᴸ)
    @inbounds begin
        ρ′[i, j, k]  = ρ′★[i, j, k]  - δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, Fʷ, dynamics, ρu′, ρv′, ρw′)
        ρθ′[i, j, k] = ρθ′★[i, j, k] - δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, θFᶻ, θᴸ, dynamics, ρu′, ρv′, ρw′)
        accumulate_transport!(avg, i, j, k, ρu′, ρv′, ρw′)
    end
    return nothing
end

@inline function accumulate_transport!(avg, i, j, k, ρu′, ρv′, ρw′)
    @inbounds begin
        avg.u[i, j, k] += ρu′[i, j, k]
        avg.v[i, j, k] += ρv′[i, j, k]
        avg.w[i, j, k] += ρw′[i, j, k]
    end
    return nothing
end

@inline accumulate_transport!(::Nothing, i, j, k, ρu′, ρv′, ρw′) = nothing

#####
##### Fused acoustic column substep (flat RectilinearGrid only)
#####
#
# `_fused_acoustic_column_substep!` runs `_build_predictors!` + `_build_vertical_rhs!` + the
# batched tridiagonal `solve!` + `_post_solve_recovery!` in ONE `:xy` launch, one thread per
# column. Each phase calls the same `@inline` per-cell body as the split kernels, in the same
# order, so every stored value is the same expression on the same inputs (bitwise identical):
#
#   1. predictors ρ′★, ρθ′★ (and the (ρθ)′ stash) over cells 1…Nz;
#   2. forward Thomas sweep over rows 1…Nz with the solver's arithmetic (`get_coefficient` on the
#      `AcousticTridiag*` tags, `masked_multiply`, the `|β| > 10 eps` elision). The rhs of row k is
#      formed inside the sweep with `vertical_rhs`, which reads the OLD (ρw)′ at faces k−1, k, k+1.
#      To keep those reads valid, the solution of row k−1 is held in a register and stored to
#      (ρw)′[k−1] only after the rhs of row k is formed (the split path keeps the rhs in a separate
#      field instead). The rhs is never stored;
#   3. back substitution over Nz−1…1 with the stored `t`;
#   4. recovery of ρ′, (ρθ)′ and the ⟨ρu, ρv, ρw⟩ accumulation over cells 1…Nz.
#
# The column phases only read the fused-written fields (ρ′★, ρθ′★, (ρw)′, ρ′, ρθ′) in their own
# column, which holds on a non-terrain RectilinearGrid: the terrain PGF/flux operators read
# horizontal neighbours, and the IMEX implicit advection runs between predictors and rhs.
# Everything else keeps the split sequence (`acoustic_column_path`).

struct SplitAcousticColumn end
struct FusedAcousticColumn end

acoustic_column_path(grid, dynamics, advection) = SplitAcousticColumn()
# The untiled fused column kernel measured slower than the split sequence on the H100 at
# 1800×1080×50 (0.95×) and 960²×250 (0.92×), so it is opt-in.
const USE_FUSED_ACOUSTIC_COLUMN = Ref(false)

# Opt-in, takes precedence over `USE_FUSED_ACOUSTIC_COLUMN[]`: the tiled (shared-memory, k-parallel)
# column kernel `_tiled_acoustic_column_substep!` where a tile fits (else the untiled/split choice above).
const USE_TILED_ACOUSTIC_COLUMN = Ref(true)

# Tiles narrower than this are slower than the split sequence: on the H100 the tiled column ran
# 1.056× at 1800×1080×50 (T = 24) but 0.837× at 960²×250, where only T = 8 fits in shared memory.
const MINIMUM_ACOUSTIC_COLUMN_TILE = Ref(16)

struct TiledAcousticColumn{T, K} end

tile_width(::TiledAcousticColumn{T}) where T = T

function acoustic_column_path(grid::RectilinearGrid, ::CompressibleDynamics{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing}, advection)
    if USE_TILED_ACOUSTIC_COLUMN[]
        tiled = tiled_acoustic_column_layout(grid)
        isnothing(tiled) || tile_width(tiled) < MINIMUM_ACOUSTIC_COLUMN_TILE[] || return tiled
    end
    return USE_FUSED_ACOUSTIC_COLUMN[] ? FusedAcousticColumn() : SplitAcousticColumn()
end
acoustic_column_path(::RectilinearGrid, ::CompressibleDynamics{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                     ::AdaptiveImplicitVerticalAdvection) = SplitAcousticColumn()

# Optional register cap for the fused column launches (CUDA only; tier A: register allocation only,
# the instructions and therefore the stored values are unchanged). `ACOUSTIC_COLUMN_MAXREGS[] = 96`
# reproduces the handoff's FC7 (masonlee: −5..−10 % column work at 336²/672²); 0 (default) leaves the
# launch to KernelAbstractions. KernelAbstractions has no register option, so the capped launch repeats
# what CUDA's KernelAbstractions backend does for a statically sized kernel, plus `maxregs`. CUDA is
# found at run time through Oceananigans' CUDA extension, so Breeze takes no CUDA dependency.
const ACOUSTIC_COLUMN_MAXREGS = Ref(0)

function cuda_core_module()
    ext = Base.get_extension(Oceananigans, :OceananigansCUDAExt)
    isnothing(ext) && return nothing
    CUDA = ext.CUDA
    return isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA
end

is_cuda_kernel(kernel!) = nameof(typeof(KernelAbstractions.backend(kernel!))) === :CUDABackend

# `kernel!` is a KernelAbstractions kernel with static workgroup and static ndrange; `args` are device-converted.
function launch_acoustic_column_kernel!(kernel!, args...)
    maxregs = ACOUSTIC_COLUMN_MAXREGS[]
    if maxregs > 0 && is_cuda_kernel(kernel!)
        launch_with_register_cap!(cuda_core_module(), kernel!, maxregs, args)
    else
        kernel!(args...)
    end
    return nothing
end

function launch_with_register_cap!(cuda, kernel!, maxregs, args)
    ndrange, workgroupsize, iterspace, dynamic = KernelAbstractions.launch_config(kernel!, nothing, nothing)
    ctx = KernelAbstractions.mkcontext(kernel!, ndrange, iterspace)
    threads = length(KernelAbstractions.workitems(iterspace))
    blocks = length(KernelAbstractions.blocks(iterspace))
    blocks == 0 && return nothing
    kernel_args = map(cuda.cudaconvert, (ctx, args...))
    kernel_tt = Tuple{map(Core.Typeof, kernel_args)...}
    backend = KernelAbstractions.backend(kernel!)
    kernel = cuda.cufunction(kernel!.f, kernel_tt; always_inline=backend.always_inline,
                             maxthreads=threads, maxregs)
    kernel(kernel_args...; threads, blocks)
    return nothing
end

@kernel function _fused_acoustic_column_substep!(ρw′, ρ′, ρθ′, ρu′, ρv′, ρ′★, ρθ′★, ρθ′ˢ⁻, avg,
                                                 a, b, c, p, t, grid, dynamics,
                                                 Δτ, δτᵐ⁺, δτˢ⁻, Gˢρ, Gˢρᵡ, fθ, θᴸ,
                                                 Πᴸ, γRᵐᴸ, g, dᵐ⁺, dˢ⁻, fw, Gˢρw, sponge)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    z = ZDirection()
    ε = 10 * eps(eltype(ρw′))

    @inbounds begin
        # 1. Predictors
        for k = 1:Nz
            build_predictors!(i, j, k, ρ′★, ρθ′★, ρθ′ˢ⁻, ρ′, ρθ′, ρw′, ρu′, ρv′,
                              grid, dynamics, Δτ, δτˢ⁻, Gˢρ, Gˢρᵡ, fθ, θᴸ)
        end

        # 2. Forward sweep; ϕᵏ⁻ is the not-yet-stored solution of the previous row
        β  = get_coefficient(i, j, 1, grid, b, p, z, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
        f₁ = vertical_rhs(i, j, 1, ρ′★, ρθ′★, ρ′, ρθ′, ρw′, grid, dynamics, Δτ, δτᵐ⁺, δτˢ⁻,
                          Πᴸ, γRᵐᴸ, g, dˢ⁻, fw, Gˢρw, sponge)
        ϕᵏ⁻ = f₁ / β

        for k = 2:Nz
            fᵏ = vertical_rhs(i, j, k, ρ′★, ρθ′★, ρ′, ρθ′, ρw′, grid, dynamics, Δτ, δτᵐ⁺, δτˢ⁻,
                              Πᴸ, γRᵐᴸ, g, dˢ⁻, fw, Gˢρw, sponge)
            ρw′[i, j, k-1] = ϕᵏ⁻ # old (ρw)′[k−1] no longer needed

            cᵏ⁻¹ = get_coefficient(i, j, k-1, grid, c, p, z, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
            bᵏ   = get_coefficient(i, j, k,   grid, b, p, z, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
            aᵏ⁻¹ = get_coefficient(i, j, k-1, grid, a, p, z, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)

            tᵏ = cᵏ⁻¹ / β
            t[i, j, k] = tᵏ
            β = bᵏ - masked_multiply(aᵏ⁻¹, tᵏ)

            # Not diagonally dominant (β ≈ 0): keep the old (ρw)′[k], as the solver does.
            definitely_diagonally_dominant = abs(β) > ε
            ϕ★ = (fᵏ - masked_multiply(aᵏ⁻¹, ϕᵏ⁻)) / β
            ϕᵏ⁻ = ifelse(definitely_diagonally_dominant, ϕ★, ρw′[i, j, k])
        end
        ρw′[i, j, Nz] = ϕᵏ⁻

        # 3. Back substitution
        for k = Nz-1:-1:1
            ρw′[i, j, k] -= masked_multiply(t[i, j, k+1], ρw′[i, j, k+1])
        end

        # 4. Recovery and transport-velocity accumulation
        for k = 1:Nz
            post_solve_recovery!(i, j, k, ρ′, ρθ′, ρw′, ρu′, ρv′, ρ′★, ρθ′★, avg, grid, dynamics, δτᵐ⁺, θᴸ)
        end
    end
end

#####
##### Tiled fused acoustic column substep (opt-in, `USE_TILED_ACOUSTIC_COLUMN[]`)
#####
#
# Port of the handoff's FC1 (masonlee, "intermediates in shared memory") onto this base. One launch,
# workgroup (T, 1, K) over (Nx, Ny, K): T columns per workgroup, K threads per column. Thread (li, lk)
# owns levels lk, lk+K, … in the k-parallel phases; a single thread per column does the serial sweep.
#
#   A. predictors of every level into the shared tile (P, Pθ) and the (ρθ)′ stash to global; the lid
#      face (ρw)′[Nz+1] into ϕ[Nz+1] (read by the recovery at k = Nz);
#   B. the rhs of faces 1…Nz into ϕ (`vertical_rhs` with the predictors from the tile, ρ′, (ρθ)′,
#      the old (ρw)′ from global — nothing global is written before D);
#   C. the forward Thomas sweep (coefficients from `get_coefficient`, as in the untiled kernel; t into
#      the tile) and back substitution in place in ϕ, one thread per column;
#   D. (ρw)′ ← ϕ (faces 1…Nz) to global, and the recovery of every level from the tile.
#
# Every stored value is the same expression on the same inputs as the split/untiled paths; FT values
# round-trip exactly through shared memory (tier A). The global predictor and rhs fields are not used.
# Shared memory: (4Nz + 1)·T·sizeof(FT) within the 48 KiB static limit.

const tiled_acoustic_column_widths = (32, 24, 16, 8)
const tiled_acoustic_column_shared_bytes = 48 * 1024

function tiled_acoustic_column_layout(grid)
    Nx, Ny, Nz = size(grid)
    FT = eltype(grid)
    for T in tiled_acoustic_column_widths
        fits = (4Nz + 1) * T * sizeof(FT) ≤ tiled_acoustic_column_shared_bytes
        if (Nx % T == 0) & fits
            K = min(Nz, clamp(128 ÷ T, 4, 16))
            return TiledAcousticColumn{T, K}()
        end
    end
    return nothing
end

# A thread's column of a workgroup tile, indexed like the field it stands in for: (i, j, k) ↦ tile[li, k].
# Level 0 (read only by the discarded k = 1 boundary rows of `vertical_rhs`) maps to level 1.
struct TileColumn{A, I}
    tile :: A
    li :: I
end

@inline Base.getindex(c::TileColumn, i, j, k) = @inbounds c.tile[c.li, max(k, 1)]
@inline Base.setindex!(c::TileColumn, v, i, j, k) = @inbounds c.tile[c.li, k] = v

@kernel function _tiled_acoustic_column_substep!(ρw′, ρ′, ρθ′, ρu′, ρv′, ρθ′ˢ⁻, avg,
                                                 a, b, c, p, grid, dynamics,
                                                 Δτ, δτᵐ⁺, δτˢ⁻, Gˢρ, Gˢρᵡ, fθ, θᴸ,
                                                 Πᴸ, γRᵐᴸ, g, dᵐ⁺, dˢ⁻, fw, Gˢρw, sponge, factors,
                                                 ::Val{T}, ::Val{K}, ::Val{Nz}) where {T, K, Nz}
    P  = @localmem eltype(ρw′) (T, Nz)
    Pθ = @localmem eltype(ρw′) (T, Nz)
    ϕ  = @localmem eltype(ρw′) (T, Nz + 1)
    tˢ = @localmem eltype(ρw′) (T, Nz)

    # Each segment between barriers recomputes its indices and tile views (KernelAbstractions' CPU
    # backend runs every segment as its own loop over the workgroup's items).

    # A. Predictors
    i, j, _ = @index(Global, NTuple)
    li, _, lk = @index(Local, NTuple)
    ρ′★  = TileColumn(P, li)
    ρθ′★ = TileColumn(Pθ, li)
    k = lk
    while k ≤ Nz
        build_predictors!(i, j, k, ρ′★, ρθ′★, ρθ′ˢ⁻, ρ′, ρθ′, ρw′, ρu′, ρv′,
                          grid, dynamics, Δτ, δτˢ⁻, Gˢρ, Gˢρᵡ, fθ, θᴸ)
        k += K
    end
    if lk == 1
        @inbounds ϕ[li, Nz + 1] = ρw′[i, j, Nz + 1]
    end

    @synchronize

    # B. rhs of faces 1…Nz
    i, j, _ = @index(Global, NTuple)
    li, _, lk = @index(Local, NTuple)
    ρ′★  = TileColumn(P, li)
    ρθ′★ = TileColumn(Pθ, li)
    k = lk
    while k ≤ Nz
        @inbounds ϕ[li, k] = vertical_rhs(i, j, k, ρ′★, ρθ′★, ρ′, ρθ′, ρw′, grid, dynamics, Δτ, δτᵐ⁺, δτˢ⁻,
                                          Πᴸ, γRᵐᴸ, g, dˢ⁻, fw, Gˢρw, sponge)
        k += K
    end

    @synchronize

    # C. Thomas sweep in place in ϕ (one thread per column)
    i, j, _ = @index(Global, NTuple)
    li, _, lk = @index(Local, NTuple)
    if lk == 1
        tiled_column_sweep!(factors, ϕ, tˢ, li, i, j, ρw′, a, b, c, p, grid, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge, Val(Nz))
    end

    @synchronize

    # D. (ρw)′ to global, recovery from the tile
    i, j, _ = @index(Global, NTuple)
    li, _, lk = @index(Local, NTuple)
    ρ′★  = TileColumn(P, li)
    ρθ′★ = TileColumn(Pθ, li)
    ϕᶜ   = TileColumn(ϕ, li)
    k = lk
    while k ≤ Nz
        @inbounds ρw′[i, j, k] = ϕ[li, k]
        post_solve_recovery!(i, j, k, ρ′, ρθ′, ϕᶜ, ρu′, ρv′, ρ′★, ρθ′★, avg, grid, dynamics, δτᵐ⁺, θᴸ)
        k += K
    end
end

# The untiled kernel's sweep with f and the solution in the tile row ϕ[li, :] and t in tˢ[li, :].
# The keep-old branch reads the old (ρw)′ from global, which is written only after the sweep.
@inline function tiled_column_sweep!(::Nothing, ϕ, tˢ, li, i, j, ρw′, a, b, c, p, grid, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge, ::Val{Nz}) where Nz
    z = ZDirection()
    ε = 10 * eps(eltype(ρw′))
    @inbounds begin
        β = get_coefficient(i, j, 1, grid, b, p, z, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
        ϕᵏ⁻ = ϕ[li, 1] / β
        ϕ[li, 1] = ϕᵏ⁻
        k = 2
        while k ≤ Nz
            fᵏ = ϕ[li, k]
            cᵏ⁻¹ = get_coefficient(i, j, k-1, grid, c, p, z, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
            bᵏ   = get_coefficient(i, j, k,   grid, b, p, z, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
            aᵏ⁻¹ = get_coefficient(i, j, k-1, grid, a, p, z, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
            tᵏ = cᵏ⁻¹ / β
            tˢ[li, k] = tᵏ
            β = bᵏ - masked_multiply(aᵏ⁻¹, tᵏ)
            definitely_diagonally_dominant = abs(β) > ε
            ϕ★ = (fᵏ - masked_multiply(aᵏ⁻¹, ϕᵏ⁻)) / β
            ϕᵏ⁻ = ifelse(definitely_diagonally_dominant, ϕ★, ρw′[i, j, k])
            ϕ[li, k] = ϕᵏ⁻
            k += 1
        end
        k = Nz - 1
        while k ≥ 1
            ϕ[li, k] -= masked_multiply(tˢ[li, k+1], ϕ[li, k+1])
            k -= 1
        end
    end
    return nothing
end

#####
##### Per-stage Thomas factorization (opt-in, `USE_STAGE_FACTORIZED_ACOUSTIC_COLUMN[]`, tiled path only)
#####
#
# The (ρw)′ tridiagonal depends only on the frozen linearization state (Πᴸ, θᴸ, γRᵐᴸ), g, δτᵐ⁺ and
# dᵐ⁺, all constant within a stage's substep loop. `_factorize_acoustic_column!` runs the forward
# sweep's coefficient recurrence once per stage (the same expressions, in the same order, as the
# sweep: t = c[k−1]/β, β = b[k] − a[k−1]·t) and stores a[k−1], β[k], t[k]; every substep's sweep then
# loads three values per row instead of evaluating `get_coefficient` three times (tier A). The
# storage (two extra 3D arrays; t uses the solver's own `t`) is allocated on first use per solver.
const USE_STAGE_FACTORIZED_ACOUSTIC_COLUMN = Ref(false)

const STAGE_FACTORIZATIONS = IdDict{Any, Any}()

function stage_factorization_storage(solver)
    return get!(STAGE_FACTORIZATIONS, solver.t) do
        (; lower = similar(solver.t), pivot = similar(solver.t), t = solver.t)
    end
end

@kernel function _factorize_acoustic_column!(factors, a, b, c, p, grid, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    z = ZDirection()
    @inbounds begin
        β = get_coefficient(i, j, 1, grid, b, p, z, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
        factors.pivot[i, j, 1] = β
        for k = 2:Nz
            cᵏ⁻¹ = get_coefficient(i, j, k-1, grid, c, p, z, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
            bᵏ   = get_coefficient(i, j, k,   grid, b, p, z, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
            aᵏ⁻¹ = get_coefficient(i, j, k-1, grid, a, p, z, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
            tᵏ = cᵏ⁻¹ / β
            β = bᵏ - masked_multiply(aᵏ⁻¹, tᵏ)
            factors.t[i, j, k] = tᵏ
            factors.pivot[i, j, k] = β
            factors.lower[i, j, k] = aᵏ⁻¹
        end
    end
end

@inline function tiled_column_sweep!(factors, ϕ, tˢ, li, i, j, ρw′, a, b, c, p, grid, Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge, ::Val{Nz}) where Nz
    ε = 10 * eps(eltype(ρw′))
    @inbounds begin
        β = factors.pivot[i, j, 1]
        ϕᵏ⁻ = ϕ[li, 1] / β
        ϕ[li, 1] = ϕᵏ⁻
        k = 2
        while k ≤ Nz
            fᵏ = ϕ[li, k]
            aᵏ⁻¹ = factors.lower[i, j, k]
            β = factors.pivot[i, j, k]
            tˢ[li, k] = factors.t[i, j, k]
            definitely_diagonally_dominant = abs(β) > ε
            ϕ★ = (fᵏ - masked_multiply(aᵏ⁻¹, ϕᵏ⁻)) / β
            ϕᵏ⁻ = ifelse(definitely_diagonally_dominant, ϕ★, ρw′[i, j, k])
            ϕ[li, k] = ϕᵏ⁻
            k += 1
        end
        k = Nz - 1
        while k ≥ 1
            ϕ[li, k] -= masked_multiply(tˢ[li, k+1], ϕ[li, k+1])
            k -= 1
        end
    end
    return nothing
end

stage_factors(path, substepper) = nothing
stage_factors(::TiledAcousticColumn, substepper) =
    USE_STAGE_FACTORIZED_ACOUSTIC_COLUMN[] ? stage_factorization_storage(substepper.vertical_solver) : nothing

factorize_stage!(::Nothing, model, substepper, δτᵐ⁺) = nothing

function factorize_stage!(factors, model, substepper, δτᵐ⁺)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    g = convert(FT, model.thermodynamic_constants.gravitational_acceleration)
    ω = FT(substepper.forward_weight)
    dᵐ⁺, _ = implicit_damping_factors(substepper.damping, ω, grid, FT)
    solver = substepper.vertical_solver
    launch!(arch, grid, :xy, _factorize_acoustic_column!, factors,
            solver.a, solver.b, solver.c, solver.parameters, grid,
            substepper.linearization_exner, substepper.linearization_potential_temperature,
            substepper.linearization_gamma_R_mixture, g, δτᵐ⁺, dᵐ⁺, substepper.sponge)
    return nothing
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
@inline implicit_damping_factors(::AcousticDampingStrategy, ω, grid, FT) = (zero(FT), zero(FT))

@inline function implicit_damping_factors(damping::ThermalDivergenceDamping, ω, grid, FT)
    damping.damp_vertical || return (zero(FT), zero(FT))
    α    = convert(FT, damping.coefficient)
    Δz   = convert(FT, minimum_zspacing(grid))
    base = α * Δz^2
    return (convert(FT, ω) * base, convert(FT, 1 - ω) * base)
end

# Klemp, Skamarock & Ha (2018) acoustic divergence damping (MPAS form).
# In the linearized acoustic mode,
#
#   (ρθ)′ − (ρθ)′ˢ⁻ ≈ −Δτ · θᴸ · ∇·((ρu)′, (ρv)′, (ρw)′)
#
# so D ≡ ((ρθ)′ − (ρθ)′ˢ⁻) / θᴸ is a discrete proxy for −Δτ · ∇·(ρu)′.
# The default per-substep momentum correction is horizontal:
#
#   Δ(ρu)′ = −γ · ∂x D , Δ(ρv)′ = −γ · ∂y D
#
# with per-direction horizontal diffusivities:
#
#   γˣ = α · Δx² / Δτ,   γʸ = α · Δy² / Δτ
#
# or, when `length_scale = ℓ` is specified, fixed diffusivity
#
#   γ = α · ℓ² / Δτ
#
# in both horizontal directions.
# If `damp_vertical = true`, the vertical contribution
#
#   γ_z = α · Δz² / Δτ
#
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
    FT   = eltype(grid)
    arch = architecture(grid)
    α    = damping.coefficient

    TX, TY, _ = topology(grid)
    x_damping_scale = TX === Flat ? NoHorizontalDampingScale() : horizontal_damping_scale(damping, α)
    y_damping_scale = TY === Flat ? NoHorizontalDampingScale() : horizontal_damping_scale(damping, α)

    launch!(arch, grid, :xyz, _thermal_divergence_damping!,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            substepper.density_potential_temperature_perturbation,
            substepper.previous_density_potential_temperature_perturbation,
            substepper.linearization_potential_temperature,
            grid, convert(FT, Δτ), x_damping_scale, y_damping_scale)

    return nothing
end

@inline dρθ′(i, j, k, grid, ρθ′, ρθ′ˢ⁻) = @inbounds ρθ′[i, j, k] - ρθ′ˢ⁻[i, j, k]

struct NoHorizontalDampingScale end

# Stores only the dimensionless Klemp coefficient α. Δτ is excluded from the
# struct so that a TracedRNumber substep size never ends up as a struct field
# (embedding traced values in structs causes invalid LLVM IR in Reactant kernels).
# Δτ is passed as a direct scalar argument to the kernel instead.
struct LocalHorizontalDampingScale{FT}
    α :: FT
end

# Stores α · ℓ² (without the 1/Δτ factor) for the same reason.
struct FixedHorizontalDampingScale{FT}
    coefficient :: FT
end

@inline horizontal_damping_scale(damping::ThermalDivergenceDamping{FT, Nothing}, α) where FT =
    LocalHorizontalDampingScale(α)

@inline function horizontal_damping_scale(damping::ThermalDivergenceDamping, α)
    ℓ = convert(typeof(α), damping.length_scale)
    return FixedHorizontalDampingScale(α * ℓ^2)
end

@inline κˣ(i, j, k, grid, ::NoHorizontalDampingScale, Δτ) = zero(grid)
@inline κʸ(i, j, k, grid, ::NoHorizontalDampingScale, Δτ) = zero(grid)

@inline κˣ(i, j, k, grid, scale::FixedHorizontalDampingScale, Δτ) = scale.coefficient / Δτ
@inline κʸ(i, j, k, grid, scale::FixedHorizontalDampingScale, Δτ) = scale.coefficient / Δτ

# Isotropic, mesh-varying horizontal damping (MPAS-style): a single scalar diffusivity
# κ = α/Δτ · ℓ² applied identically in x and y, with ℓ the *smallest* local horizontal
# spacing.  Using min(Δx, Δy) (rather than Δx, Δy separately, or the geometric mean
# √(ΔxΔy), or cbrt(V) which is dragged down by the thin Δz) keeps the explicit-diffusion
# number κΔτ/Δ² ≤ α in both directions everywhere — including the converging meridians at
# high latitude, where √(ΔxΔy) would push it past the α ≤ 0.25 stability bound.
# Smallest local horizontal spacing, excluding Flat dimensions. A Flat direction has degenerate
# unit spacing (Δ = 1), which must not enter the min — otherwise it collapses the diffusivity in the
# non-Flat direction(s). Treating Flat dims as +∞ makes `min` select the physical spacing.
@inline function _min_horizontal_spacing(i, j, k, grid)
    FT = eltype(grid)
    TX, TY, _ = topology(grid)
    Δx = TX === Flat ? FT(Inf) : Δxᶠᶜᶜ(i, j, k, grid)
    Δy = TY === Flat ? FT(Inf) : Δyᶜᶠᶜ(i, j, k, grid)
    return min(Δx, Δy)
end

@inline _κ_local(i, j, k, grid, scale::LocalHorizontalDampingScale, Δτ) =
    scale.α * _min_horizontal_spacing(i, j, k, grid)^2 / Δτ
@inline κˣ(i, j, k, grid, scale::LocalHorizontalDampingScale, Δτ) = _κ_local(i, j, k, grid, scale, Δτ)
@inline κʸ(i, j, k, grid, scale::LocalHorizontalDampingScale, Δτ) = _κ_local(i, j, k, grid, scale, Δτ)


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
@kernel function _thermal_divergence_damping!(ρu′, ρv′, ρθ′, ρθ′ˢ⁻, θᴸ, grid, Δτ,
                                              x_damping_scale, y_damping_scale)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂x_div = ∂xᶠᶜᶜ(i, j, k, grid, dρθ′, ρθ′, ρθ′ˢ⁻)
        θᴸᶠᶜᶜ  = ℑxᶠᵃᵃ(i, j, k, grid, θᴸ)
        γˣ = κˣ(i, j, k, grid, x_damping_scale, Δτ)
        ρu′[i, j, k] -= γˣ * ∂x_div / θᴸᶠᶜᶜ

        ∂y_div = ∂yᶜᶠᶜ(i, j, k, grid, dρθ′, ρθ′, ρθ′ˢ⁻)
        θᴸᶜᶠᶜ  = ℑyᵃᶠᵃ(i, j, k, grid, θᴸ)
        γʸ = κʸ(i, j, k, grid, y_damping_scale, Δτ)
        ρv′[i, j, k] -= γʸ * ∂y_div / θᴸᶜᶠᶜ
    end
end

# Direct divergence damping (`DirectDivergenceDamping`): form the *horizontal* Θ-flux divergence
# δ = ∂x(θᴸ(ρu)′) + ∂y(θᴸ(ρv)′) directly from the velocity — the divergence carried by the (ρθ)′
# continuity equation (Klemp, Skamarock & Ha 2018, eq. 36) — rather than approximating it through the
# (ρθ)′ tendency as `ThermalDivergenceDamping` does. The horizontal momentum is then corrected by the
# θᴸ-scaled gradient:
#   Δ(ρu)′ = α Δx² ∂x[δ] / θᴸ,   Δ(ρv)′ = α Δy² ∂y[δ] / θᴸ,   δ = ∇ʰ·(θᴸ(ρ𝐮)′).
# Horizontal only — by design. Folding the vertical Θ-flux divergence ∂z(θᴸ(ρw)′) into δ damps the
# *resolved* vertical flux and blows up the baroclinic wave; the horizontal divergence is the correct
# quantity for the horizontal filter (cf. PR #794, which fixes the same cold-start blow-up this way).
# Two launches: materialize δ into the (now-free, post-recovery) density-predictor scratch, halo-fill it,
# then take its gradient. The local diffusivity α Δx² carries no 1/Δτ — δ is the divergence itself, not
# the Δτ-scaled (ρθ)′ tendency — which also removes the thermal form's cold-start ∝ α/Δτ spurious force.
function apply_divergence_damping!(damping::DirectDivergenceDamping, substepper, grid, Δτ, thermodynamic_constants)
    arch = architecture(grid)
    α = convert(eltype(grid), damping.coefficient)

    δ   = substepper.density_predictor   # free between recovery and the next predictor build
    θᴸ  = substepper.linearization_potential_temperature
    ρu′ = substepper.momentum_perturbation.u
    ρv′ = substepper.momentum_perturbation.v

    launch!(arch, grid, :xyz, _compute_horizontal_theta_flux_divergence!, δ, grid, θᴸ, ρu′, ρv′)
    fill_halo_regions!(δ)
    launch!(arch, grid, :xyz, _apply_direct_divergence_damping!, ρu′, ρv′, grid, δ, θᴸ, α)

    return nothing
end

# δ = ∇ʰ·(θᴸ(ρ𝐮)′): area-weighted horizontal Θ-flux divergence, using the same flux-form (θFˣ, θFʸ)
# `build_predictors!` uses for ∇ʰ_θM (so the damped divergence matches the one in the Θ equation).
@kernel function _compute_horizontal_theta_flux_divergence!(δ, grid, θᴸ, ρu′, ρv′)
    i, j, k = @index(Global, NTuple)
    @inbounds δ[i, j, k] = (δxᶜᵃᵃ(i, j, k, grid, θFˣ, θᴸ, ρu′) +
                            δyᵃᶜᵃ(i, j, k, grid, θFʸ, θᴸ, ρv′)) * V⁻¹ᶜᶜᶜ(i, j, k, grid)
end

@kernel function _apply_direct_divergence_damping!(ρu′, ρv′, grid, δ, θᴸ, α)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρu′[i, j, k] += α * Δxᶠᶜᶜ(i, j, k, grid)^2 * ∂xᶠᶜᶜ(i, j, k, grid, δ) / ℑxᶠᵃᵃ(i, j, k, grid, θᴸ)
        ρv′[i, j, k] += α * Δyᶜᶠᶜ(i, j, k, grid)^2 * ∂yᶜᶠᶜ(i, j, k, grid, δ) / ℑyᵃᶠᵃ(i, j, k, grid, θᴸ)
    end
end

#####
##### Section 10 — Time-averaged velocity for non-acoustic scalar transport
#####
##### WRF/MPAS dynamics-transport split: non-acoustic scalars (moisture,
##### tracers, chemistry, TKE) advect against the substep-loop-averaged
##### velocity, not a snapshot. (The slow `ρθ` tendency is part of the
##### acoustic system, computed separately before the loop.) We accumulate
##### raw `momentum_perturbation` each substep, then normalize at stage end:
#####
#####   ⟨ρu⟩ = ρuᴸ + (1/Nτ) ∑ₙ (ρu)′(n) = model.momentum.ρu + accum/Nτ
#####   ⟨u⟩  ≈ ⟨ρu⟩ / ρᴸ_face
#####
##### `model.momentum.*` and `model.dynamics.dry_density` are still the stage-entry
##### (Uᴸ_stage) values here. Dividing by ρᴸ ignores ρ's variation over the
##### loop, small for acoustic perturbations.
#####


function finalize_time_averaged_velocity!(substepper, model, Nτ, peeled = false)
    grid = model.grid
    arch = architecture(grid)
    FT   = eltype(grid)
    inv_Nτ = one(FT) / FT(Nτ)

    # `model.dynamics.dry_density` and `model.momentum.*` are still the
    # stage-entry (Uᴸ) values here — the substep loop only touched
    # substepper-owned perturbation fields. They serve as ρᴸ and ρu/v/wᴸ.
    ρw_stage, transport_dynamics = stage_transport_vertical_momentum(model)
    launch!(arch, grid, :xyz, _finalize_time_averaged_velocity!,
            substepper.time_averaged_velocities.u,
            substepper.time_averaged_velocities.v,
            substepper.time_averaged_velocities.w,
            model.momentum.ρu, model.momentum.ρv, ρw_stage,
            model.dynamics.dry_density,
            grid, transport_dynamics, inv_Nτ,
            peeled ? substepper.momentum_perturbation : nothing)

    map(fill_halo_regions!, substepper.time_averaged_velocities)

    return nothing
end

@kernel function _finalize_time_averaged_velocity!(u_avg, v_avg, w_avg,
                                                   ρu_stage, ρv_stage, ρw_stage,
                                                   ρᴸ, grid, dynamics, inv_Nτ, last)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρu_total = ρu_stage[i, j, k] + transport_sum(u_avg, last, :u, i, j, k) * inv_Nτ
        ρv_total = ρv_stage[i, j, k] + transport_sum(v_avg, last, :v, i, j, k) * inv_Nτ
        ρw_total = transport_ρw(i, j, k, grid, dynamics,
                                                              ρu_stage, ρv_stage, ρw_stage) +
                   transport_sum(w_avg, last, :w, i, j, k) * inv_Nτ

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

# Transport sum at stage end: the accumulator, plus the last substep's perturbation when that
# substep was peeled (same operands and order as `accumulate_transport!`).
@inline transport_sum(avg, ::Nothing, name, i, j, k) = @inbounds avg[i, j, k]
@inline transport_sum(avg, last, name, i, j, k) = @inbounds avg[i, j, k] + getproperty(last, name)[i, j, k]

@inline transport_ρw(i, j, k, grid, dynamics, ρu_stage, ρv_stage, ρw_stage) =
    @inbounds ρw_stage[i, j, k]

"""
Whether terrain-following stages load the stage-entry contravariant momentum ρw̃ that `update_state!` stored
(`_compute_contravariant_velocity!`, the same expression on the same momenta and halos) instead of recomputing it
in the perturbation initialization and the transport-velocity finalization (port of handoff N11 84d73a82). Default
`true`; `BREEZE_LOAD_STAGE_CONTRAVARIANT_MOMENTUM=false` at load turns it off. Bitwise identical either way.
"""
const LOAD_STAGE_CONTRAVARIANT_MOMENTUM = Ref(true)

# The finalization's stage-entry vertical transport momentum and the dynamics `transport_ρw` dispatches on:
# (ρw, dynamics) evaluates in place; the terrain method returns (stored ρw̃, nothing), which `transport_ρw` loads.
stage_transport_vertical_momentum(model) = (model.momentum.ρw, model.dynamics)

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
        ρwᵐ⁺ = acoustic_recovered_vertical_momentum(i, j, k, grid, dynamics, ρwᴸ[i, j, k], ρu′, ρv′, ρw′)

        ρ[i, j, k]  = ρᵐ⁺
        ρθ[i, j, k] = ρθᵐ⁺

        m.ρu[i, j, k] = ρuᵐ⁺
        m.ρv[i, j, k] = ρvᵐ⁺
        m.ρw[i, j, k] = ρwᵐ⁺
    end
end

@inline acoustic_recovered_vertical_momentum(i, j, k, grid, dynamics, ρwᴸ_ccf, ρu′, ρv′, ρw′) =
    @inbounds ρwᴸ_ccf + ρw′[i, j, k]

#####
##### Section 12 — Substep loop driver
#####

# Per-substep open-boundary enforcement (Breeze.jl issue #738).
# The perturbation scalars `ρ′,(ρθ)′` carry zero-gradient halos on `Bounded`
# dims, so an open lateral boundary reflects the acoustic pressure perturbation
# back inward — the boundary mass flux is then carried only by the frozen slow
# tendency `Gˢρ`, biasing mass balance under transient inflow. WRF/ERF/MPAS
# instead enforce the specified lateral boundary every substep.
# We mirror that by relaxing the outermost open-boundary cell of `ρ′`, `(ρθ)′`
# toward the prescribed wall value `v` each substep. `update_state!` applied the
# prognostic `ValueBoundaryCondition` to the base at stage entry, so
# `ρᴸ[halo] = 2v − ρᴸ[cell]` and the target perturbation is
# `v − ρᴸ[cell] = (ρᴸ[halo] − ρᴸ[cell]) / 2`, read straight from the base field.
# Relaxation factor `α ∈ (0, 1]` (default 0.5, via
# `SplitExplicitTimeDiscretization(; open_boundary_relaxation = α)`). No-op on any
# side without an active `NormalFlowBoundaryCondition` (periodic/walls/`nothing`
# all skip it) — zero cost when no open lateral BC is present.

@inline is_active_open_bc(bc) = (bc isa BoundaryCondition{<:NormalFlow}) && !(bc.condition isa Nothing)

# Relax ρ′ and (ρθ)′ at the outermost open-boundary cell toward the prescribed
# wall value in a single kernel: target = v − cᴸ[iᴮ] = (cᴸ[iᴴ] − cᴸ[iᴮ]) / 2.
# `iᴮ` is the outermost interior cell index, `iᴴ` the adjacent halo cell index.
@kernel function _relax_open_boundary_x!(ρ′, ρθ′, ρᴸ, ρθᴸ, iᴮ, iᴴ, α)
    j, k = @index(Global, NTuple)
    @inbounds begin
        ρ′[iᴮ, j, k]  += α * ((ρᴸ[iᴴ, j, k]  - ρᴸ[iᴮ, j, k])  / 2 - ρ′[iᴮ, j, k])
        ρθ′[iᴮ, j, k] += α * ((ρθᴸ[iᴴ, j, k] - ρθᴸ[iᴮ, j, k]) / 2 - ρθ′[iᴮ, j, k])
    end
end

@kernel function _relax_open_boundary_y!(ρ′, ρθ′, ρᴸ, ρθᴸ, jᴮ, jᴴ, α)
    i, k = @index(Global, NTuple)
    @inbounds begin
        ρ′[i, jᴮ, k]  += α * ((ρᴸ[i, jᴴ, k]  - ρᴸ[i, jᴮ, k])  / 2 - ρ′[i, jᴮ, k])
        ρθ′[i, jᴮ, k] += α * ((ρθᴸ[i, jᴴ, k] - ρθᴸ[i, jᴮ, k]) / 2 - ρθ′[i, jᴮ, k])
    end
end

function apply_open_boundary_relaxation!(substepper, model, grid, arch)
    bcs_u = model.momentum.ρu.boundary_conditions
    bcs_v = model.momentum.ρv.boundary_conditions
    Nx, Ny, _ = size(grid)
    α   = substepper.open_boundary_relaxation
    ρ′  = substepper.density_perturbation
    ρθ′ = substepper.density_potential_temperature_perturbation
    ρᴸ  = model.dynamics.dry_density
    ρθᴸ = thermodynamic_density(model.formulation)
    if is_active_open_bc(bcs_u.west)
        launch!(arch, grid, :yz, _relax_open_boundary_x!, ρ′, ρθ′, ρᴸ, ρθᴸ, 1, 0, α)
    end
    if is_active_open_bc(bcs_u.east)
        launch!(arch, grid, :yz, _relax_open_boundary_x!, ρ′, ρθ′, ρᴸ, ρθᴸ, Nx, Nx + 1, α)
    end
    if is_active_open_bc(bcs_v.south)
        launch!(arch, grid, :xz, _relax_open_boundary_y!, ρ′, ρθ′, ρᴸ, ρθᴸ, 1, 0, α)
    end
    if is_active_open_bc(bcs_v.north)
        launch!(arch, grid, :xz, _relax_open_boundary_y!, ρ′, ρθ′, ρᴸ, ρθᴸ, Ny, Ny + 1, α)
    end
    return nothing
end

# Per-substep impenetrability enforcement on the wall-normal momentum
# perturbations — the closed-wall companion of `apply_open_boundary_relaxation!`
# above (which handles open lateral boundaries).

@kernel function _zero_x_wall_face!(ρu′, iᶠ)
    j, k = @index(Global, NTuple)
    @inbounds ρu′[iᶠ, j, k] = 0
end

@kernel function _zero_y_wall_face!(ρv′, jᶠ)
    i, k = @index(Global, NTuple)
    @inbounds ρv′[i, jᶠ, k] = 0
end

@inline is_impenetrable_wall(bc) = (bc isa BoundaryCondition{NormalFlow{Nothing}}) && (bc.condition isa Nothing)

function enforce_wall_impenetrability!(substepper, model, grid, arch)
    TX, TY, _ = topology(grid)
    Nx, Ny, _ = size(grid)
    ρu′ = substepper.momentum_perturbation.u
    ρv′ = substepper.momentum_perturbation.v
    bcs_u = model.momentum.ρu.boundary_conditions
    bcs_v = model.momentum.ρv.boundary_conditions
    if TX === Bounded
        is_impenetrable_wall(bcs_u.west) && launch!(arch, grid, :yz, _zero_x_wall_face!, ρu′, 1)
        is_impenetrable_wall(bcs_u.east) && launch!(arch, grid, :yz, _zero_x_wall_face!, ρu′, Nx + 1)
    end
    if TY === Bounded
        is_impenetrable_wall(bcs_v.south) && launch!(arch, grid, :xz, _zero_y_wall_face!, ρv′, 1)
        is_impenetrable_wall(bcs_v.north) && launch!(arch, grid, :xz, _zero_y_wall_face!, ρv′, Ny + 1)
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Advance the implicit half of the IMEX split for the thermodynamic perturbation over one
substep: `(I + Δτ Lⁱ) ρθ′★ⁿᵉʷ = ρθ′★`, solved on the θ predictor between Step B and Step C so
the pressure solve and recovery act on a transport-consistent predictor. Coefficients read the
frozen stage-entry state; the base-state part rides `Gˢρθ`
(see `add_implicit_advection_tendency!`).
"""
implicit_advection_substep!(model, substepper, advection, Δτ) = nothing

function implicit_advection_substep!(model, substepper, advection::AdaptiveImplicitVerticalAdvection, Δτ)
    w = substepper.vertical_velocity_cache
    ρᵈ = substepper.density_cache
    implicit_step!(substepper.density_potential_temperature_predictor,
                   model.timestepper.implicit_solver, nothing, nothing, nothing,
                   model.clock, AtmosphereModels.kernel_fields(model), Δτ, advection, (; w), ρᵈ)
    return nothing
end

function acoustic_column_substep!(::TiledAcousticColumn{T, K}, model, substepper, stage, advection, g, dᵐ⁺, dˢ⁻,
                                 accumulators = substepper.time_averaged_velocities) where {T, K}
    grid = model.grid
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    Gⁿ = model.timestepper.Gⁿ
    (; Δτ, δτᵐ⁺, δτˢ⁻, Gˢρᵡ) = stage
    solver = substepper.vertical_solver

    column! = _tiled_acoustic_column_substep!(device(arch), (T, 1, K), (Nx, Ny, K))
    args = (substepper.momentum_perturbation.w,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            substepper.previous_density_potential_temperature_perturbation,
            substepper.time_averaged_velocities,
            solver.a, solver.b, solver.c, solver.parameters,
            grid, model.dynamics, Δτ, δτᵐ⁺, δτˢ⁻,
            Gⁿ.ρᵈ, Gˢρᵡ, substepper.thermodynamic_tendency_factor,
            substepper.linearization_potential_temperature,
            substepper.linearization_exner, substepper.linearization_gamma_R_mixture,
            g, dᵐ⁺, dˢ⁻, substepper.vertical_momentum_tendency_factor,
            substepper.slow_vertical_momentum_tendency,
            substepper.sponge, stage_factors(TiledAcousticColumn{T, K}(), substepper),
            Val(T), Val(K), Val(Nz))
    launch_acoustic_column_kernel!(column!, convert_to_device(arch, args)...)

    # The (ρθ)′ stash halo is filled in the batched post-column fill (`stash_halo_fields`).
    return nothing
end

"""
$(TYPEDSIGNATURES)

Advance the acoustic perturbation fields of `substepper` by one substep of size `stage.Δτ`.
`stage` holds the loop-invariant quantities of the current RK3 stage: the substep size `Δτ`,
the Crank–Nicolson weights `δτᵐ⁺` and `δτˢ⁻`, and the slow thermodynamic tendency `Gˢρᵡ`.
Everything is passed explicitly so that the loop calling this can be traced by Reactant with
no captured state; see [`acoustic_substep_loop!`](@ref).
"""
function acoustic_substep!(model, substepper, stage, advection,
                           accumulators = substepper.time_averaged_velocities)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    constants = model.thermodynamic_constants
    g = convert(FT, constants.gravitational_acceleration)
    ω = FT(substepper.forward_weight)
    Gⁿ = model.timestepper.Gⁿ
    (; Δτ, δτᵐ⁺, δτˢ⁻, Gˢρᵡ) = stage

    # Step A: explicit horizontal forward of (ρu)′, (ρv)′.
    launch!(arch, grid, :xyz, _explicit_horizontal_step!,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            grid, model.dynamics, Δτ,
            substepper.density_potential_temperature_perturbation,
            substepper.linearization_exner,
            Gⁿ.ρu, Gⁿ.ρv, substepper.linearization_gamma_R_mixture,
            active_pressure_gradients(substepper))

    fill_halo_regions!(substepper.momentum_perturbation.u)
    fill_halo_regions!(substepper.momentum_perturbation.v)

    # Impenetrability on the wall-normal momentum perturbations, before the
    # predictor takes their horizontal divergence (mass conservation).
    enforce_wall_impenetrability!(substepper, model, grid, arch)

    # (old (ρθ)′ is stashed into ρθ′ˢ⁻ inside `_build_predictors!`, then halo-filled.)

    # Implicit-vertical-damping prefactors. When the damping strategy
    # is `ThermalDivergenceDamping(damp_vertical=true)`, the
    # vertical part of the divergence damping is folded into the
    # tridiag with `dᵐ⁺ = ω·α·Δz²` on the LHS and
    # `dˢ⁻ = (1−ω)·α·Δz²` on the predictor RHS. Both reduce to
    # zero for `NoDivergenceDamping` or when the user opts out via
    # `damp_vertical=false`.
    dᵐ⁺, dˢ⁻ = implicit_damping_factors(substepper.damping, ω, grid, FT)

    # Steps B–D: predictors, (ρw)′ tridiag rhs + solve, post-solve recovery. Fused into one
    # column launch on flat RectilinearGrids (see `_fused_acoustic_column_substep!`).
    path = acoustic_column_path(grid, model.dynamics, advection)
    acoustic_column_substep!(path, model, substepper, stage, advection, g, dᵐ⁺, dˢ⁻, accumulators)

    # A last substep whose post-solve recovery and open-boundary relaxation the stage-end recovery
    # performs (`FoldedRecovery`) ends after its solve: nothing below has another reader (the
    # perturbation refill is unread after a last substep, there is no damping, and the wall
    # enforcement rewrites the +0 the one after the explicit step left).
    accumulators isa FoldedRecovery && return nothing

    # Per-substep open-boundary enforcement (issue #738): relax the outermost
    # open-boundary cell of ρ′, (ρθ)′ toward the prescribed wall value, before
    # the halo fill, so the boundary cell tracks the prescribed inflow state.
    apply_open_boundary_relaxation!(substepper, model, grid, arch)

    # One batched fill of the cell-centred perturbations (one launch per side). On the fused
    # path this includes the (ρθ)′ stash, which only the divergence damping reads.
    fill_halo_regions!((substepper.density_perturbation,
                        substepper.density_potential_temperature_perturbation,
                        stash_halo_fields(path, substepper)...))

    # Step E: optional Klemp 2018 post-substep damping (no-op for
    # `NoDivergenceDamping`).
    apply_divergence_damping!(substepper.damping, substepper, grid, Δτ, constants)

    # No (ρu)′, (ρv)′ halo fill here: nothing reads their halos before the next substep's
    # Step A overwrites the interior and refills them. The stage-final fill is done once
    # after the loop (see `acoustic_substep_loop!`).

    # The damping kernel also writes the south/west wall face; re-enforce
    # impenetrability so the next substep's divergence and the accumulated
    # transport velocity see a closed wall (mass conservation).
    enforce_wall_impenetrability!(substepper, model, grid, arch)
    # (time-averaged velocity accumulation is fused into `_post_solve_recovery!` above)
    return nothing
end

function acoustic_column_substep!(::SplitAcousticColumn, model, substepper, stage, advection, g, dᵐ⁺, dˢ⁻,
                                 accumulators = substepper.time_averaged_velocities)
    grid = model.grid
    arch = architecture(grid)
    Gⁿ = model.timestepper.Gⁿ
    (; Δτ, δτᵐ⁺, δτˢ⁻, Gˢρᵡ) = stage

    # Step B: build predictors ρ′★, ρθ′★ (3D), then the (ρw)′ᵐ⁺ tridiag RHS (3D).
    # `_build_predictors!` also stashes old (ρθ)′ into ρθ′ˢ⁻; halo-fill it for the damping.
    launch!(arch, grid, :xyz, _build_predictors!,
            substepper.density_predictor,
            substepper.density_potential_temperature_predictor,
            substepper.previous_density_potential_temperature_perturbation,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.momentum_perturbation.w,
            substepper.momentum_perturbation.u, substepper.momentum_perturbation.v,
            grid, model.dynamics, Δτ, δτˢ⁻,
            Gⁿ.ρᵈ, Gˢρᵡ, substepper.thermodynamic_tendency_factor,
            substepper.linearization_potential_temperature)
    fill_halo_regions!(substepper.previous_density_potential_temperature_perturbation)

    # Implicit half of the IMEX vertical-advection split for the thermodynamic
    # perturbation, applied to the predictor between Step B and Step C so the pressure
    # solve and recovery substitution see a transport-consistent ρθ′★ (issue #897).
    implicit_advection_substep!(model, substepper, advection, Δτ)

    pressure_gradients = active_pressure_gradients(substepper)
    if CARRIED_VERTICAL_RHS[] & carries_vertical_rhs(model.dynamics, pressure_gradients, grid)
        # One thread per column, the vertical stencil carried from level to level (handoff bc1c3e6e).
        rhs = (; ρ′★ = substepper.density_predictor,
                 ρθ′★ = substepper.density_potential_temperature_predictor,
                 ρ′ = substepper.density_perturbation,
                 ρθ′ = substepper.density_potential_temperature_perturbation,
                 ρw′ = substepper.momentum_perturbation.w,
                 grid, Δτ, δτᵐ⁺, δτˢ⁻,
                 Πᴸ = substepper.linearization_exner, γRᵐᴸ = substepper.linearization_gamma_R_mixture,
                 g, dˢ⁻, fw = substepper.vertical_momentum_tendency_factor,
                 Gˢρw = substepper.slow_vertical_momentum_tendency,
                 sponge = substepper.sponge, P = pressure_gradients)
        launch!(arch, grid, :xy, _build_vertical_rhs_carried!, substepper.vertical_solver_source_term, rhs)
    else
        launch!(arch, grid, KernelParameters(1:size(grid, 1), 1:size(grid, 2), 1:size(grid, 3) + 1),
                _build_vertical_rhs!,
                substepper.vertical_solver_source_term,
                substepper.density_predictor,
                substepper.density_potential_temperature_predictor,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation.w,
                grid, model.dynamics, Δτ, δτᵐ⁺, δτˢ⁻,
                substepper.linearization_exner, substepper.linearization_gamma_R_mixture,
                g, dˢ⁻, substepper.vertical_momentum_tendency_factor,
                substepper.slow_vertical_momentum_tendency,
                substepper.sponge, pressure_gradients)
    end

    # Step C: implicit tridiag solve for (ρw)′ with implicit-half δτᵐ⁺
    # and (when active) implicit vertical damping prefactor `dᵐ⁺`.
    # `sponge` may add an implicit Rayleigh contribution on the
    # diagonal in a layer below the lid.
    solve!(substepper.momentum_perturbation.w, substepper.vertical_solver,
           substepper.vertical_solver_source_term,
           substepper.linearization_exner, substepper.linearization_potential_temperature,
           substepper.linearization_gamma_R_mixture, g, δτᵐ⁺, dᵐ⁺,
           substepper.sponge)

    # Step D: post-solve recovery of ρ′, (ρθ)′ using new (ρw)′ (in the stage-end recovery for a folded last substep)
    accumulators isa FoldedRecovery && return nothing
    launch!(arch, grid, :xyz, _post_solve_recovery!,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.momentum_perturbation.w,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            substepper.density_predictor,
            substepper.density_potential_temperature_predictor,
            accumulators,
            grid, model.dynamics, δτᵐ⁺,
            substepper.linearization_potential_temperature)
    return nothing
end

function acoustic_column_substep!(::FusedAcousticColumn, model, substepper, stage, advection, g, dᵐ⁺, dˢ⁻,
                                 accumulators = substepper.time_averaged_velocities)
    grid = model.grid
    arch = architecture(grid)
    Gⁿ = model.timestepper.Gⁿ
    (; Δτ, δτᵐ⁺, δτˢ⁻, Gˢρᵡ) = stage
    solver = substepper.vertical_solver

    column!, _ = configure_kernel(arch, grid, :xy, _fused_acoustic_column_substep!)
    args = (substepper.momentum_perturbation.w,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            substepper.density_predictor,
            substepper.density_potential_temperature_predictor,
            substepper.previous_density_potential_temperature_perturbation,
            accumulators,
            solver.a, solver.b, solver.c, solver.parameters, solver.t,
            grid, model.dynamics, Δτ, δτᵐ⁺, δτˢ⁻,
            Gⁿ.ρᵈ, Gˢρᵡ, substepper.thermodynamic_tendency_factor,
            substepper.linearization_potential_temperature,
            substepper.linearization_exner, substepper.linearization_gamma_R_mixture,
            g, dᵐ⁺, dˢ⁻, substepper.vertical_momentum_tendency_factor,
            substepper.slow_vertical_momentum_tendency,
            substepper.sponge)
    launch_acoustic_column_kernel!(column!, convert_to_device(arch, args)...)

    # The (ρθ)′ stash is read with horizontal neighbours only by the divergence damping;
    # its halo is filled together with ρ′, (ρθ)′ in `acoustic_substep!` (`stash_halo_fields`).
    return nothing
end

# Fields whose halos `acoustic_substep!` fills in its batched post-column fill.
# The split path fills the stash itself, before the implicit advection substep.
stash_halo_fields(::SplitAcousticColumn, substepper) = ()
stash_halo_fields(::FusedAcousticColumn, substepper) =
    (substepper.previous_density_potential_temperature_perturbation,)
stash_halo_fields(::TiledAcousticColumn, substepper) =
    (substepper.previous_density_potential_temperature_perturbation,)

"""
$(TYPEDSIGNATURES)

Execute one Wicker–Skamarock RK3 stage of the linearized acoustic
substep loop. Number and size of substeps in this stage depend on
`substepper.substep_distribution`.
"""
function acoustic_rk3_substep_loop!(model::AtmosphereModel, substepper, Δt, β_stage, Uᴸ, advection = nothing; zero_entry = false)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    constants = model.thermodynamic_constants

    # Substep count Nτ and size Δτ for this stage (WS-RK3 weights β = (1/3, 1/2, 1)).
    # The distribution decides how to split: ProportionalSubsteps fits ⌈β·N⌉ substeps to
    # each stage's β·Δt interval; ConstantSubstepSize/MonolithicFirstStage share one Δτ = Δt/N.
    Δt_FT = FT(Δt)
    Nτ, Δτ = stage_substep_count_and_size(substepper.substep_distribution, β_stage, Δt_FT,
                                          substepper.minimum_horizontal_spacing, constants,
                                          substepper.acoustic_cfl, substepper.substeps)

    # CN time-step weights for this substep. δτᵐ⁺ = ω·Δτ is the
    # new-side weight (used by the matrix and the post-solve);
    # δτˢ⁻ = (1−ω)·Δτ is the old-side weight (used by the
    # predictor's old-flux contribution and the old part of the
    # vertical RHS). See `docs/src/compressible_dynamics.md`.
    ω = FT(substepper.forward_weight) # CN weight on the new side
    δτᵐ⁺ = ω * Δτ
    δτˢ⁻ = (1 - ω) * Δτ

    # Build the slow vertical-momentum tendency Gˢρw at z-faces:
    #   Gˢρw = Gⁿρw − ∂z(pᴸ − pᵣ) − g (ρᴸ − ρᵣ)        (with reference state)
    #   Gˢρw = Gⁿρw − ∂z pᴸ − g ρᴸ                     (no reference state)
    # which the per-substep linearized acoustic forces add to.
    assemble_slow_vertical_momentum_tendency!(substepper, model, β_stage)

    # Initialize perturbations with the SK08 rewind term so the substep
    # effectively starts from U(t) = Uᴸ (the outer-step-start state).
    initialize_stage_perturbations!(substepper, model, Uᴸ, zero_entry,
                                    acoustic_column_path(grid, model.dynamics, advection))

    Gⁿ = model.timestepper.Gⁿ
    ρᵡ_name = thermodynamic_density_name(model.formulation)
    Gˢρᵡ = getproperty(Gⁿ, ρᵡ_name)

    stage = (; Δτ, δτᵐ⁺, δτˢ⁻, Gˢρᵡ)
    factors = stage_factors(acoustic_column_path(grid, model.dynamics, advection), substepper)
    factorize_stage!(factors, model, substepper, δτᵐ⁺)
    peeled, folded = acoustic_substep_loop!(Nτ, model, substepper, stage, advection)

    # Stage-end: convert the accumulated momentum perturbations into a
    # time-averaged velocity field. Read by `update_state!` through
    # `transport_velocities(model)` for moisture/tracer tendencies.
    # Done BEFORE `_recover_full_state!` so we read the stage-entry
    # `model.momentum.*` (substep loop hasn't touched it) and stage-entry
    # `model.dynamics.dry_density` as the Uᴸ_stage reference.
    finalize_time_averaged_velocity!(substepper, model, Nτ, peeled)

    # Stage-end: recover the full prognostic state in-place. `model.dynamics.dry_density`,
    # `ρᵡ`, and `model.momentum.*` are still the stage-entry Uᴸ values here
    # (the substep loop only touched substepper.* perturbation fields). The
    # recovery kernel reads them as Uᴸ AND writes the full state back to the
    # same fields; per-thread read-before-write makes this aliasing safe
    # because all reads are local to the same grid point.
    ρᵡ = thermodynamic_density(model.formulation)
    if folded
        launch!(arch, grid, :xyz, _recover_full_state_after_solve!,
                model.dynamics.dry_density, ρᵡ,
                model.momentum,
                substepper.momentum_perturbation.w,
                substepper.momentum_perturbation.u,
                substepper.momentum_perturbation.v,
                substepper.density_predictor,
                substepper.density_potential_temperature_predictor,
                eltype(substepper.density_perturbation),
                model.dynamics.dry_density,
                model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
                ρᵡ,
                grid, model.dynamics, δτᵐ⁺,
                substepper.linearization_potential_temperature,
                open_boundary_rim(substepper, model, grid))
    else
        launch!(arch, grid, :xyz, _recover_full_state!,
                model.dynamics.dry_density, ρᵡ,
                model.momentum,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation.u,
                substepper.momentum_perturbation.v,
                substepper.momentum_perturbation.w,
                model.dynamics.dry_density,
                model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
                ρᵡ,
                grid, model.dynamics)
    end

    # Thread clock + model fields so time-dependent Open BCs on the recovered
    # prognostic state dispatch correctly in `getbc` (see #717).
    fill_halo_regions!(model.dynamics.dry_density, AtmosphereModels.kernel_boundary_condition_args(model)...)
    fill_halo_regions!(ρᵡ, AtmosphereModels.kernel_boundary_condition_args(model)...)
    fill_halo_regions!(model.momentum, AtmosphereModels.kernel_boundary_condition_args(model)...)
    AtmosphereModels.compute_velocities!(model)

    return nothing
end

Oceananigans.prognostic_state(substepper::AcousticSubstepper) =
    (time_averaged_velocities = Oceananigans.prognostic_state(substepper.time_averaged_velocities),
     time_averaged_vertical_velocity_cache = Oceananigans.prognostic_state(substepper.time_averaged_vertical_velocity_cache))

function Oceananigans.restore_prognostic_state!(restored::AcousticSubstepper, from)
    Oceananigans.restore_prognostic_state!(restored.time_averaged_velocities, from.time_averaged_velocities)
    Oceananigans.restore_prognostic_state!(restored.time_averaged_vertical_velocity_cache,
                                           from.time_averaged_vertical_velocity_cache)
    return restored
end

Oceananigans.restore_prognostic_state!(substepper::AcousticSubstepper, ::Nothing) = substepper
