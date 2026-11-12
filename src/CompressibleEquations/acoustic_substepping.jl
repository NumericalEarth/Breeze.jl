#####
##### Acoustic substepping for CompressibleDynamics
#####
##### Evolves linearized acoustic perturbations (primes) about the stage-entry state Uᴸ, refreshed each
##### WS-RK3 stage (Skamarock & Klemp 2008):  ρ′=ρ−ρᴸ, (ρu)′=ρu−ρuᴸ (and v,w), and the pressure p′=p−pᴸ.
##### θᴸ=ρθᴸ/ρᴸ and the pressure coefficient Cᴸ=∂p/∂(ρθ) (see `refresh_linearization_basic_state!`) are
##### cached per stage; ρᴸ,ρθᴸ,pᴸ,(ρu/v/w)ᴸ are read live (the loop never mutates them) and are the
##### recovery base.
#####
##### The loop's thermodynamic unknown is the pressure perturbation in units of the thermodynamic
##### density, ϖ = p′/Cᴸ, written (ρθ)′ in the kernels. The equation of state, linearized at Uᴸ in
##### every prognostic density it depends on, gives
#####
#####   ϖ = (ρθ)′ + W ,      W = − Σₐ Λₐ (ρqₐ)′ + Λᵈ ρᵈ′      (a: the moisture densities)
#####
##### so ϖ is the perturbation of ρθ plus the content W that the pressure reads from the other
##### densities, and it is (ρθ)′ itself in dry air (see `PressureProjection`). The loop moves no
##### water, so W is known along the slow path of the stage: the state the stage starts from, less
##### the stage-entry state, plus the ramp of the slow tendencies. Linearized equations integrated
##### by the loop:
#####
#####   ∂t ρ′    + ∇·((ρu)′,(ρv)′,(ρw)′)      = Gˢρ
#####   ∂t ϖ     + ∇·(θᴸ·((ρu)′,(ρv)′,(ρw)′)) = Gˢρθ − Σₐ Λₐ Gˢρqₐ + Λᵈ Gˢρ
#####   ∂t (ρu)′ + ∂x pᴸ + ∂x(Cᴸ ϖ)           = Gˢρu     (PGF = gradient of cell-centered Cᴸ ϖ)
#####   ∂t (ρv)′ + ∂y pᴸ + ∂y(Cᴸ ϖ)           = Gˢρv
#####   ∂t (ρw)′ +         ∂z(Cᴸ ϖ) + g·ρ′    = Gˢρw
#####
##### Time discretization: horizontal momentum is forward-Euler, with the frozen ∇pᴸ and the
##### perturbation horizontal PGF applied on every substep. Vertical ((ρw)′,(ρθ)′,ρ′) coupling is off-centered Crank-Nicolson (`forward_weight` ω:
##### 0.5 = centered, >0.5 = dissipative), reducing to a tridiagonal Schur solve for (ρw)′ at z-faces.
##### Each stage then recovers ρ=ρᴸ+ρ′, ρθ=ρθᴸ+ϖ−W, … and diagnoses velocities: the thermodynamic
##### density has taken its own tendency and the flux divergence the loop applied, like any scalar.
#####
##### Kernel args ρ′, ρθ′, ρu′, ρv′, ρw′ map to struct fields density_perturbation,
##### density_potential_temperature_perturbation, momentum_perturbation.{u,v,w}; predictors carry ★
##### (ρ′★, ρθ′★). (σ, η are reserved for vertical coordinates, so primes are used throughout.)
##### Public derivation: docs/src/compressible_dynamics.md.
##### Refs: Wicker & Skamarock 2002 (MWR 130, 2088); Klemp et al. 2018 (MWR 146, 1911).
#####

using KernelAbstractions: @kernel, @index

using Oceananigans: Field, CenterField, XFaceField, YFaceField, ZFaceField, architecture, location
using Oceananigans.Models: boundary_condition_args
using Oceananigans: fields
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection
using Oceananigans.Grids: ZDirection, rnode, znode, node
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Solvers: BatchedTridiagonalSolver, compute_factorization!, masked_multiply, solve!
using Oceananigans.TurbulenceClosures: vertically_implicit_closures, VerticallyImplicitDiffusionLowerDiagonal,
                                      VerticallyImplicitDiffusionDiagonal, VerticallyImplicitDiffusionUpperDiagonal
using Oceananigans.Operators:
    ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ,
    ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ,
    δxTᶠᵃᵃ, δyTᵃᶠᵃ, δxTᶜᵃᵃ, δyTᵃᶜᵃ,
    Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ,
    Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ,
    Δz⁻¹ᶜᶜᶜ, Δz⁻¹ᶜᶜᶠ,
    Δxᶠᶜᶜ,
    Δyᶜᶠᶜ,
    Axᶠᶜᶜ, Ayᶜᶠᶜ, V⁻¹ᶜᶜᶜ

using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!, BoundaryCondition, NormalFlow, needs_implicit_solver

using Oceananigans.Architectures: CPU, GPU
using Oceananigans.Grids: AbstractGrid, Flat, Bounded, Center, Face, peripheral_node,
                          topology,
                          minimum_xspacing, minimum_yspacing, minimum_zspacing

using Adapt: Adapt, adapt

#####
##### Section 1 — Substepper struct
#####

"""
$(TYPEDEF)

Storage and parameters for the split-explicit acoustic substepper (scheme described in the module
header). θᴸ=ρθᴸ/ρᴸ and Cᴸ are cached once per stage (recomputing inline per call is much slower on
H100); ρᴸ, ρθᴸ, pᴸ and the stage-entry momenta are read live from `model.dynamics.*` /
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
- `pressure_coefficient`: the `AcousticPressureCoefficient` selecting how Cᴸ is computed (the
  frozen-composition slope, or the phase-equilibrium slope in saturated cells).
- `linearization_potential_temperature` (θᴸ), `linearization_pressure_coefficient` (Cᴸ, the slope
  ∂p/∂(ρθ) of the equation of state at the stage-entry state): per-stage caches, see
  `refresh_linearization_basic_state!`.
- `pressure_projection`: the [`PressureProjection`](@ref) of the moisture onto the loop's pressure:
  the moisture sensitivities of the equation of state, cached with θᴸ and Cᴸ, and the stage's
  thermodynamic forcing and moisture content of the pressure.
- `density_perturbation` (ρ′), `density_potential_temperature_perturbation` (the pressure
  perturbation in units of the thermodynamic density, ϖ = p′/Cᴸ, which is (ρθ)′ in dry air),
  `momentum_perturbation` ((ρu/v/w)′ as `.u/.v/.w`): perturbation prognostics advanced in the loop.
- `density_predictor`, `density_potential_temperature_predictor`: explicit predictors before the vertical solve.
- `previous_density_potential_temperature_perturbation`: prior-substep (ρθ)′, for Klemp 2018 damping.
- `time_averaged_velocities`: acoustic-mean velocity for non-acoustic scalar transport (moisture/tracers/
  chemistry/TKE); the slow ρθ tendency uses the current RK predictor velocity instead, not this cache.
- `slow_vertical_momentum_tendency` (Gˢρw, z-faces): advection+Coriolis+closure+forcing (PGF/buoyancy
  excluded — those are in the fast operator).
- `vertical_solver_source_term` (z-faces): explicit RHS of the (ρw)′ tridiagonal system.
- `vertical_solver`: factorized `BatchedTridiagonalSolver` of the implicit (ρw)′ update, factored once
  per stage (`factor_vertical_solver!`) so that each substep solves by substitution.
- `vertical_velocity_cache`, `density_cache`: the stage-entry predictor velocity and the dry
  density that weights it, frozen for `implicit_substep!` (see `cache_advecting_state!`); `nothing` without
  adaptive-implicit advection.
- `time_averaged_vertical_velocity_cache`: the acoustic-mean transport velocity the moisture and
  tracer tendencies were built with, frozen for `scalar_substep!` and for the thermodynamic
  tendency's condensate sedimentation term (see `cache_transport_velocity!`).
"""
struct AcousticSubstepper{N, FT, D, AD, US, PC, CF, PP, MP, TAV, GT, TS, WC, DC, TWC, IS}
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
    pressure_coefficient :: PC

    # Linearization basic state θᴸ and Cᴸ, derived per stage from live model fields
    # (`model.dynamics.dry_density`, `model.dynamics.total_density`, the moisture and the
    # formulation's prognostic ρθ field — untouched by the substep loop, so they hold the
    # stage-entry state throughout and serve as the recovery base in `_recover_full_state!`).
    linearization_potential_temperature :: CF
    linearization_pressure_coefficient :: CF
    pressure_projection :: PP

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
    implicit_advection_solver :: IS               # factorized solver of the (ρθ)′ implicit-advection system, factored per stage; `nothing` without AIVA
end

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
                       a.pressure_coefficient,
                       adapt(to, a.linearization_potential_temperature),
                       adapt(to, a.linearization_pressure_coefficient),
                       adapt(to, a.pressure_projection),
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
                       adapt(to, a.implicit_advection_solver))

#####
##### Halo fills of the substep loop
#####

# The substep kernels take every horizontal difference of the fields the loop writes with
# Oceananigans' topology-aware operators (`δxTᶠᵃᵃ`, `δxTᶜᵃᵃ`, `δyTᵃᶠᵃ`, `δyTᵃᶜᵃ`), as the
# split-explicit barotropic substepping does: they read the periodic image, or apply the
# no-flux / impenetrability condition the perturbations carry on a `Bounded` side, from the
# interior. So between substeps a single-architecture grid needs no halo fill; a distributed grid
# exchanges the halos of its connected sides, where the operators fall back to reading them.
# The terrain dynamics fills (see `terrain_compressible_physics.jl`).
fill_substep_halos!(fields, model) = fill_substep_halos!(fields, model.grid)
fill_substep_halos!(fields, grid::AbstractGrid) = fill_architecture_halos!(fields, architecture(grid))
fill_architecture_halos!(fields, ::Union{CPU, GPU}) = nothing
fill_architecture_halos!(fields, arch) = fill_halo_regions!(fields)

# TODO: Oceananigans has no immersed-grid method for `δxTᶠᵃᵃ`, `δyTᵃᶠᵃ` (they fall back to
# `δxᶠᵃᵃ`, which reads halos); it should provide immersed topology-aware operators. Until then
# the substep kernels call these wrappers, which pass the underlying grid on an immersed grid.
@inline topology_aware_δxᶠᵃᵃ(i, j, k, grid, args...) = δxTᶠᵃᵃ(i, j, k, grid, args...)
@inline topology_aware_δyᵃᶠᵃ(i, j, k, grid, args...) = δyTᵃᶠᵃ(i, j, k, grid, args...)
@inline topology_aware_δxᶠᵃᵃ(i, j, k, ibg::ImmersedBoundaryGrid, args...) = δxTᶠᵃᵃ(i, j, k, ibg.underlying_grid, args...)
@inline topology_aware_δyᵃᶠᵃ(i, j, k, ibg::ImmersedBoundaryGrid, args...) = δyTᵃᶠᵃ(i, j, k, ibg.underlying_grid, args...)

#####
##### Section 2 — Constructor
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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
    # Evaluate the ramp into a 1D z-face profile now, so the substep kernels never touch it.
    sponge = materialize_sponge(grid, split_explicit.sponge)
    substep_distribution = split_explicit.substep_distribution
    open_boundary_relaxation = convert(FT, split_explicit.open_boundary_relaxation)
    pressure_coefficient = split_explicit.pressure_coefficient

    # `substep_floattype` (default `eltype(grid)`) is the storage type for the acoustic perturbation/predictor/
    # linearization working fields below. Pass a reduced-precision type (e.g. `BFloat16`, where the
    # grid/field types support it) to halve their HBM traffic in the bandwidth-bound substep kernels:
    # kernels read substep_floattype, promote to FT, compute in FT, store substep_floattype. The (ρw)′ solve target, tridiag scratch,
    # primary prognostics, and WENO tendencies stay FT regardless (the solver recurrence and WENO
    # degrade in low precision).

    # Linearization basic state — θᴸ and Cᴸ derived from live model fields once per stage.
    linearization_potential_temperature = CenterField(grid, substep_floattype)
    linearization_pressure_coefficient = CenterField(grid, substep_floattype)
    pressure_projection = PressureProjection(grid, substep_floattype)

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

    # Factored once per stage; the substeps solve by substitution
    vertical_solver = BatchedTridiagonalSolver(grid;
                                               lower_diagonal = AcousticTridiagLower(),
                                               diagonal       = AcousticTridiagDiagonal(),
                                               upper_diagonal = AcousticTridiagUpper(),
                                               tridiagonal_direction = ZDirection(),
                                               factorized = true)

    vertical_velocity_cache = cache_advecting_state ? ZFaceField(grid) : nothing
    density_cache = cache_advecting_state ? CenterField(grid) : nothing
    time_averaged_vertical_velocity_cache = ZFaceField(grid)
    # The coefficients of the model's implicit solver (`implicit_diffusion_solver`), in a factorized solver of its own
    implicit_advection_solver = cache_advecting_state ? factorized_implicit_solver(grid) : nothing

    return AcousticSubstepper(Ns, acoustic_cfl, Δx_min, ω, thermodynamic_tendency_factor,
                              vertical_momentum_tendency_factor,
                              vertical_pressure_tendency_factor,
                              final_stage_vertical_pressure_tendency_factor,
                              damping, substep_distribution,
                              sponge, open_boundary_relaxation,
                              pressure_coefficient,
                              linearization_potential_temperature,
                              linearization_pressure_coefficient,
                              pressure_projection,
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
                              implicit_advection_solver)
end

#####
##### Section 3 — Stage-entry linearization
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Compute the background quantities used by the substepper as the first
linearization point of an outer step. Subsequent RK stages call
[`prepare_acoustic_cache!`](@ref), which refreshes the same cached
quantities to the stage-entry state.

After this call:
  - `linearization_potential_temperature` = θᴸ = ρθᴸ/ρᴸ derived from `model.dynamics.dry_density` + ρθ
  - `linearization_pressure_coefficient`  = Cᴸ, the slope ∂p/∂(ρθ) of the equation of state at that
                                            state (see `refresh_linearization_basic_state!`)
  - the moisture sensitivities of the [`PressureProjection`](@ref) = Λᵐ, Λˡ, Λⁱ at that state
"""
function freeze_linearization_state!(substepper::AcousticSubstepper, model)
    refresh_linearization_basic_state!(substepper, model)
    seed_time_averaged_velocities!(substepper, model)
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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

# Refresh the cached linearization quantities (θᴸ, Cᴸ, and the moisture sensitivities Λ of the
# pressure, see `PressureProjection`) from the live model state. Called at
# outer-step start by `freeze_linearization_state!` and at every RK stage by
# `prepare_acoustic_cache!`. The base-state fields ρᴸ, ρθᴸ, pᴸ are `model.dynamics.dry_density`,
# the formulation's ρθ field, and `model.dynamics.pressure` — read directly by the substep kernels.
#
# The loop's pressure perturbation is p′ = Cᴸ ϖ, with ϖ = (ρθ)′ − Σₐ Λₐ (ρqₐ)′ the pressure in units
# of the thermodynamic density. Cᴸ is the slope ∂p/∂(ρθ), at fixed dry density and composition, of
# the equation of state p = ρ Rᵐ T that the core closes with `compressible_density_state`, and
# −Cᴸ Λₐ its slope ∂p/∂(ρqₐ) at fixed dry and thermodynamic densities. The kernel differentiates
# the selected temperature inversion for Cᴸ, so stopped and non-iterated solves use the slope of
# the temperature they return; the Λ are those of the converged root. Derivation, closed forms and
# the limits of this linearization: docs/src/compressible_dynamics.md.
# TODO: carry ∂p/∂ρᵈ = Cᴸ θ Hᶜ / (cᵖᵐ T − Hᶜ), which the loop omits in cloud, as a ρ′ term of p′
# (cached per stage next to Cᴸ, differentiated through the selected inversion). It enters the
# horizontal pressure gradient and the vertical Schur complement next to Cᴸ θᴸ.
# TODO: weight the pressure gradient and gravity on ρᵈu by ρᵈ / ρ throughout the compressible core
# (as WRF-ARW does; v4 technical note, §2.2), NumericalEarth/Breeze.jl#1210. The acoustic inertia
# is ρᵈ today, so the loop's sound speed is (Cᴸ θ)^½ = (γᵉ p / ρᵈ)^½ rather than (γᵉ Rᵐ T)^½.
function refresh_linearization_basic_state!(substepper::AcousticSubstepper, model)
    grid = model.grid
    arch = architecture(grid)

    projection = substepper.pressure_projection
    microphysics = model.microphysics
    condensate_names = condensate_field_names(microphysics)

    launch!(arch, grid, :xyz, _compute_linearization_basic_state!,
            substepper.linearization_potential_temperature,
            substepper.linearization_pressure_coefficient,
            projection.prognostic_moisture_sensitivity,
            projection.liquid_sensitivity,
            projection.ice_sensitivity,
            projection.dry_density_sensitivity,
            grid,
            substepper.pressure_coefficient,
            model.formulation,
            model.dynamics,
            model.dynamics.dry_density,
            model.dynamics.total_density,
            specific_prognostic_moisture(model),
            microphysics,
            model.microphysical_fields,
            condensate_liquid_fractions(microphysics, condensate_names),
            values(model.microphysical_fields[condensate_names]),
            model.thermodynamic_constants)

    # The horizontal pressure-gradient force in `_explicit_horizontal_step!`
    # uses ∂x(pᴸ) directly. With `ExnerReferenceState` the reference depends
    # only on z so ∂x pᵣ ≡ 0, and ∂x(pᴸ − pᵣ) = ∂x pᴸ; with no reference
    # state pᵣ = 0. In both cases no separate `pressure_perturbation` field
    # is needed for the horizontal direction. Vertical reference subtraction
    # for the slow tendency is handled by `assemble_slow_vertical_momentum_tendency!`.

    # One tuple fill: fields that share a location and halo-filling kernels fill in one launch per side.
    fill_halo_regions!((substepper.linearization_potential_temperature,
                        substepper.linearization_pressure_coefficient))

    return nothing
end

# θᴸ, Cᴸ and the moisture sensitivities per cell, from the same moisture and state the core closes
# the thermodynamics with (`_compute_temperature_and_pressure!`). The frozen-composition slope is
# Cᶠ = ρ Rᵐ (θ ∂T/∂θ) / (ρᵈ θ); `acoustic_pressure_coefficient` replaces it in saturated cells when the
# phase-equilibrium coefficient is selected. The divisors are zero only before `set!`; there they are
# replaced by one, and θ = 0 has zero sensitivity. The sensitivity Λᵐ of the prognostic moisture
# density is that of the phases a change of it takes (`prognostic_moisture_phase_fractions`): the
# vapor, unless the scheme diagnoses condensate from it. The sensitivity Λᵈ of the dry density, at
# fixed thermodynamic and moisture densities, is Λᴴ + Σₐ Λₐ rₐ over the prognostic moisture
# densities, rₐ = ρqₐ / ρᵈ.
@kernel function _compute_linearization_basic_state!(θᴸ, Cᴸ, Λᵐ, Λˡ, Λⁱ, Λᵈ, grid, pressure_coefficient, formulation,
                                                     dynamics, dry_density, total_density, specific_prognostic_moisture,
                                                     microphysics, microphysical_fields,
                                                     liquid_fractions, condensate_densities, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵈ  = dry_density[i, j, k]
        ρ   = total_density[i, j, k]
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
    end

    q  = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    ρ̂ᵈ = ifelse(ρᵈ == 0, one(ρᵈ), ρᵈ)
    𝒰  = compressible_density_state(i, j, k, formulation, dynamics, ρ̂ᵈ, ρ, q)
    θ  = 𝒰.potential_temperature
    θ̂  = ifelse(θ == 0, one(θ), θ)
    T, θ∂T∂θ = temperature_and_sensitivity(𝒰, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    Cᶠ = ρ * Rᵐ * θ∂T∂θ / (ρ̂ᵈ * θ̂)
    qᵖ = precipitating_mass_fraction(i, j, k, microphysics, microphysical_fields)

    λᵛ, λˡ, λⁱ, λᴴ = moisture_pressure_sensitivities(formulation, 𝒰, T, ρᵈ, constants)
    fᵛ, fˡ, fⁱ = prognostic_moisture_phase_fractions(i, j, k, grid, microphysics, microphysical_fields, q)
    λᵐ = fᵛ * λᵛ + fˡ * λˡ + fⁱ * λⁱ
    Σλρq = λᵐ * ρ * qᵛᵉ + condensate_sensitivity_sum(i, j, k, λˡ, λⁱ, liquid_fractions, condensate_densities)

    @inbounds begin
        θᴸ[i, j, k] = θ
        Cᴸ[i, j, k] = acoustic_pressure_coefficient(pressure_coefficient, Cᶠ, 𝒰, T, qᵖ, ρ̂ᵈ, θ̂,
                                                         microphysics, constants)
        Λᵐ[i, j, k] = λᵐ
        Λˡ[i, j, k] = λˡ
        Λⁱ[i, j, k] = λⁱ
        Λᵈ[i, j, k] = λᴴ + Σλρq / ρ̂ᵈ
    end
end

# The cached coefficient: the frozen-composition slope Cᶠ, or in saturated cells (cloud condensate,
# in excess of the precipitating fraction qᵖ, at the linearization state) the phase-equilibrium slope.
@inline acoustic_pressure_coefficient(::FrozenCompositionPressureCoefficient, Cᶠ, 𝒰, T, qᵖ, ρ̂ᵈ, θ̂,
                                           microphysics, constants) = Cᶠ

@inline function acoustic_pressure_coefficient(::PhaseEquilibriumPressureCoefficient, Cᶠ, 𝒰, T, qᵖ, ρ̂ᵈ, θ̂,
                                                    microphysics, constants)
    q = 𝒰.moisture_mass_fractions
    saturated = q.liquid + q.ice - qᵖ > 0
    equilibrium = microphysics_phase_equilibrium(microphysics)
    Cᵉ = equilibrium_pressure_coefficient(𝒰, T, ρ̂ᵈ, θ̂, equilibrium, constants)
    return ifelse(saturated, Cᵉ, Cᶠ)
end

# The slope Cᵉ = ∂p/∂(ρθ) of the saturation-adjusted pressure along a uniform compression of a
# saturated cell, at the converged root of the θˡⁱ → T inversion.
#
# The acoustic loop compresses ρᵈ and ρθ together, and the water follows with the loop-mean velocity,
# so the stage-end state that saturation adjustment re-diagnoses is a uniform compression: every
# density scales by the same factor, θ and the total water qᵗ are unchanged, and (ρθ)′ = ρᵈ θ ds with
# ds = d ln ρ. Along that path the cell stays saturated, with the vapor at qᵛ⁺(T, ρ) = pᵛ⁺(T) / (ρ Rᵛ T)
# and the condensate qᶜ = qᵗ − qᵛ⁺ − qᵖ taking up the difference. Compression warms the air and raises
# qᵛ⁺, so condensate evaporates and its latent heat is drawn from the air: the pressure rises less than
# at frozen composition. Writing H = ĉᵖ Π θ = cᵖᵐ T − Hᶜ for the dry-coupled part of the energy, the
# inversion cᵖᵐ T = H + Hᶜ is differentiated at fixed θ and qᵗ, with the phase change dqᶜ = −dqᵛ
# entering through cᵖᵐ (dcᵖᵐ = −Δcᵖ dqᶜ), Rᵐ (dRᵐ = −Rᵛ dqᶜ) and Hᶜ (dHᶜ = ℒ₀ dqᶜ), while ĉᵖ and κ̂
# depend on qᵗ alone:
#
#   (cᵖᵐ − κ̂ H / T) dT = κ̂ H ds + (ℒ − κ̂ H Rᵛ / Rᵐ) dqᶜ ,        ℒ = ℒ₀ + Δcᵖ T ,
#
# which is the phase-exchange response that `Microphysics.saturation_response` differentiates, plus
# the compression term κ̂ H ds. Saturation closes it, by Clausius–Clapeyron d ln pᵛ⁺ / dT = ℒ / (Rᵛ T²):
#
#   dqᶜ = −dqᵛ⁺ = qᵛ⁺ ds − A dT ,        A = qᵛ⁺ (ℒ / (Rᵛ T²) − 1 / T) .
#
# With D = cᵖᵐ − κ̂ H / T and B = ℒ − κ̂ H Rᵛ / Rᵐ, the compression response is
#
#   dT / ds = (κ̂ H + B qᵛ⁺) / (D + A B) ,        dqᶜ / ds = qᵛ⁺ − A dT / ds ,
#
# and the pressure p = ρ Rᵐ T responds with
#
#   d ln p / ds = 1 + (1 / T) dT / ds − (Rᵛ / Rᵐ) dqᶜ / ds ,        Cᵉ = p (d ln p / ds) / (ρᵈ θ) .
#
# Without vapor (qᵛ⁺ → 0, A → 0) this is the frozen compression slope, dT / ds = κ̂ H / D, which in dry
# air is γᵈ p / (ρθ). The frozen slope that the loop otherwise uses in cloud is the part of that at
# fixed ρᵈ, Cᶠ = γᵉ p / (ρθ), and the ∂p/∂ρᵈ remainder (see `refresh_linearization_basic_state!`) is
# included here, since the saturated response is derived along the compression. Over the equilibrium
# of a mixed-phase `MixedPhaseEquilibrium`, ℒ and pᵛ⁺ are those of the liquid fraction at T, and its
# temperature dependence is neglected. T = 0 (before `set!`) is replaced by one; the cell is then not
# saturated and the frozen slope is returned.
@inline function equilibrium_pressure_coefficient(𝒰, T, ρ̂ᵈ, θ̂, equilibrium, constants)
    q   = 𝒰.moisture_mass_fractions
    ρ   = 𝒰.density
    T̂   = ifelse(T == 0, one(T), T)
    Rᵛ  = vapor_gas_constant(constants)
    Rᵐ  = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    κ̂   = exner_exponent(𝒰.exner_exponent, q, constants)

    surface = equilibrated_surface(equilibrium, T̂)
    ℒ   = absolute_zero_latent_heat(constants, surface) + specific_heat_difference(constants, surface) * T̂
    qᵛ⁺ = saturation_specific_humidity(T̂, ρ, constants, surface)

    H = cᵖᵐ * T̂ - condensate_latent_energy(q, constants)
    D = cᵖᵐ - κ̂ * H / T̂
    B = ℒ - κ̂ * H * Rᵛ / Rᵐ
    A = qᵛ⁺ * (ℒ / (Rᵛ * T̂^2) - 1 / T̂)

    dT_ds  = (κ̂ * H + B * qᵛ⁺) / (D + A * B)
    dqᶜ_ds = qᵛ⁺ - A * dT_ds
    dlnp_ds = 1 + dT_ds / T̂ - Rᵛ / Rᵐ * dqᶜ_ds

    p = ρ * Rᵐ * T̂
    return p * dlnp_ds / (ρ̂ᵈ * θ̂)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Stage-start cache preparation. Refreshes the cached linearization
quantities (θᴸ, Cᴸ and the moisture sensitivities of the pressure) to the **stage-entry state** ``Uᴸ_\\mathrm{stage}``
(per [Skamarock & Klemp 2008](@cite SkamarockKlemp2008) above eq. 16),
recomputing them from the live `model.dynamics.*`. The rewind-perturbation
initialization (`initialize_stage_perturbations!`, called next) handles
the WS-RK3 invariant by setting ``(ρ)′_\\mathrm{init} = Uᴸ_\\mathrm{outer} − Uᴸ_\\mathrm{stage}``
(zero for stage 1; nonzero for stages 2 and 3).
"""
prepare_acoustic_cache!(substepper::AcousticSubstepper, model) =
    refresh_linearization_basic_state!(substepper, model)

#####
##### Section 3b — The pressure the loop integrates, and the moisture's part of it
#####
##### The loop's thermodynamic unknown is the pressure perturbation p′ = p − pᴸ about the
##### stage-entry pressure, held in units of the thermodynamic density: ϖ = p′ / Cᴸ. The equation
##### of state p = (ρᵈ Rᵈ + ρᵛ Rᵛ) T, linearized in every density it depends on, is
#####
#####   p′ = Cᴸ [(ρθ)′ + Λᵈ ρᵈ′ − Λᵛ (ρᵛ)′ − Λˡ (ρˡ)′ − Λⁱ (ρⁱ)′] ,
#####
##### with (ρˣ)′ the perturbations of the partial densities of vapor, liquid and ice, Λˣ the
##### thermodynamic density that a unit of phase x holds at fixed pressure, and Λᵈ the same for a
##### unit of dry air (`moisture_pressure_sensitivities`). The model's prognostic moisture
##### densities ρqₐ, the moisture density and the condensate masses, are each one phase or a fixed
##### mix of them, with sensitivity Λₐ, so that
#####
#####   ϖ = (ρθ)′ + W ,        W = − Σₐ Λₐ (ρqₐ)′ + Λᵈ ρᵈ′ ,        Λᵈ = Λᴴ + Σₐ Λₐ rₐ ,
#####
##### where rₐ = ρqₐ / ρᵈ are the mixing ratios and Λᴴ = θ Hᶜ / (cᵖᵐ T − Hᶜ) is the dependence on the
##### dry density at fixed composition. Dry air that carries its moisture along at fixed mixing
##### ratios therefore changes the pressure, besides through ρθ, only through Λᴴ.
#####
##### W is the content that the pressure reads from the densities other than ρθ, in units of ρθ.
##### The loop transports no water, so the moisture's part of W is not an unknown: over a stage
##### the moisture goes from the state U⁰ the loop starts from along its slow tendencies Gˢ, and
##### the stage-entry state Uᴸ is the base of the perturbations. The dry density's part is taken
##### along the same slow path,
#####
#####   W(τ) = − Σₐ Λₐ (ρqₐ⁰ − ρqₐᴸ) + Λᵈ (ρᵈ⁰ − ρᵈᴸ) − τ (Σₐ Λₐ Gˢρqₐ − Λᵈ Gˢρ) .
#####
##### Hence the loop starts from ϖ(0) = (ρθ⁰ − ρθᴸ) + W(0), the pressure of U⁰ to first order, and
##### forces ϖ with Gˢρθ + dW/dτ: the projection Σ (∂p/∂U)(·) / Cᴸ of the start state and of every
##### slow tendency onto the pressure (`project_moisture_onto_pressure!`). A tendency that moves no
##### pressure, such as condensate sedimenting, mixing or being advected at one temperature and
##### pressure, or a phase change at fixed pressure, projects to zero whichever process it comes
##### from, and the momentum equation does not see it. After the loop, ρθ = ρθᴸ + ϖ − W: the
##### thermodynamic density has taken its own tendency and the flux divergence −∇·(θᴸ (ρu)′) that
##### the loop applied, as a moisture density takes its tendency and its own flux divergence.
#####
##### The projection of vapor is the moist potential temperature that WRF substeps (Xiao et al.
##### 2015): with Λᵛ ≈ −θ Rᵛ / Rᵈ, ϖ ≈ (ρᵈ θᵐ)′, θᵐ = θ (1 + rᵛ Rᵛ / Rᵈ).
#####
##### The coefficients are cached with θᴸ and Cᴸ, and both projections are linear in the state
##### difference and in the tendencies. An outer scheme that starts a stage from a combination of
##### earlier stages, forces it with a combination of their tendencies and freezes the linearization
##### over a step (a multirate infinitesimal step) projects those combinations with the same
##### coefficients.
#####
##### Not in the loop's pressure: the dry density perturbation that the loop's own mass flux
##### builds, −∇·(ρu)′ (see the TODO at `refresh_linearization_basic_state!`); the response of a
##### diagnosed condensate partition to the temperature (saturation adjustment: the vapor is held
##### at the saturation of the linearization state); and what the vertically-implicit solves that
##### follow the loop do to the moisture in the same stage. The next stage's base pressure holds
##### all three.
#####
##### Without moisture every ρqₐ, Gˢρqₐ and Λᵈ is exactly zero, so W = 0, ϖ = (ρθ)′, and the loop
##### is bitwise what it is without the projection.
#####

"""
$(TYPEDEF)

The projection of the moisture onto the pressure that the acoustic substep loop integrates:
``ϖ = (ρθ)′ + W`` with ``W = − Σₐ Λₐ (ρqₐ)′ + Λᵈ ρᵈ′``, the content that the pressure reads from
the moisture and the dry density along the slow path of a stage (see the section comment above).

Fields:
- `prognostic_moisture_sensitivity` (Λᵐ), `liquid_sensitivity` (Λˡ), `ice_sensitivity` (Λⁱ): the
  thermodynamic density that a unit of the prognostic moisture density, of liquid and of ice holds
  at fixed pressure, at the linearization state; cached with θᴸ and Cᴸ
  (`refresh_linearization_basic_state!`).
- `dry_density_sensitivity` (Λᵈ): the thermodynamic density that a unit of dry air holds at fixed
  pressure and moisture densities, ``Λᴴ + Σₐ Λₐ rₐ``; cached with them.
- `tendency`: the slow forcing of ϖ over the stage, ``Gˢρθ − Σₐ Λₐ Gˢρqₐ + Λᵈ Gˢρ``.
- `content`: the moisture content ``W`` of the pressure that comes off ϖ when the stage recovers
  the thermodynamic density, and that an open boundary adds to its target. It is the content at
  the end of the stage; under adaptive implicit vertical advection, whose per-substep solve takes
  the content off its predictor, it is advanced with the loop to the end of the current substep.
"""
struct PressureProjection{SF, CF}
    prognostic_moisture_sensitivity :: SF
    liquid_sensitivity :: SF
    ice_sensitivity :: SF
    dry_density_sensitivity :: SF
    tendency :: CF
    content :: CF
end

# The sensitivities are linearization fields, stored like θᴸ and Cᴸ. The tendency and the content
# keep the grid's precision: they stand in for the slow tendency and for the thermodynamic density.
PressureProjection(grid, substep_floattype) =
    PressureProjection(CenterField(grid, substep_floattype),
                       CenterField(grid, substep_floattype),
                       CenterField(grid, substep_floattype),
                       CenterField(grid, substep_floattype),
                       CenterField(grid),
                       CenterField(grid))

Adapt.adapt_structure(to, p::PressureProjection) =
    PressureProjection(adapt(to, p.prognostic_moisture_sensitivity),
                       adapt(to, p.liquid_sensitivity),
                       adapt(to, p.ice_sensitivity),
                       adapt(to, p.dry_density_sensitivity),
                       adapt(to, p.tendency),
                       adapt(to, p.content))

# The moisture sensitivities (Λᵛ, Λˡ, Λⁱ) of the pressure p = (ρᵈ Rᵈ + ρᵛ Rᵛ) T for the
# thermodynamic density ρθ = ρᵈ θˡⁱ. At fixed ρᵈ, with θˡⁱ a function of T, p and the composition,
#
#   d(ρθ) = ρᵈ (∂θ/∂T dT + ∂θ/∂p dp) + Σₓ (ρᵈ / ρ) χˣ dρˣ ,        dp = ρ Rᵐ dT + Rᵛ T dρᵛ ,
#
# with χˣ the derivative of θˡⁱ at fixed T and p along q̂ˣ − q, the change of the mass fractions per
# unit of phase x added to a unit of mass (q̂ˣ: all mass in x). Eliminating dT, and with Cᴸ the
# slope dp / d(ρθ) at fixed composition,
#
#   dp = Cᴸ [d(ρθ) − Σₓ Λˣ dρˣ] ,        Λˣ = (ρᵈ / ρ) χˣ − δˣᵛ (ρᵈ / ρ) (∂θ/∂T) Rᵛ T / Rᵐ ,
#
# and ∂θ/∂T = θ cᵖᵐ / H at fixed p and composition, H = cᵖᵐ T − Hᶜ = ĉᵖ Π θˡⁱ. Condensate holds
# Λ ≈ −ℒ / (ĉᵖ Π): adding it at fixed ρθ raises the temperature by the latent heat θˡⁱ credits it
# with. Vapor holds Λᵛ ≈ −θ Rᵛ / Rᵈ: it adds pressure like Rᵛ / Rᵈ of its mass in dry air, which
# is the moist potential temperature θᵐ = θ (1 + rᵛ Rᵛ / Rᵈ) of Xiao et al. (2015). These are
# the slopes at the converged root of the temperature inversion. The divisors vanish only in a
# state that was never set, where they are replaced by one.
#
# The fourth value is the sensitivity to the dry density at fixed composition (mass fractions)
# and ρθ, where the pressure depends on ρᵈ only through the condensate latent energy:
# ∂p/∂ρᵈ = Cᴸ Λᴴ with Λᴴ = θ Hᶜ / H. At fixed partial densities of the moisture instead, a unit
# of dry air also dilutes the mass fractions, and ∂p/∂ρᵈ = Cᴸ (Λᴴ + Σₓ Λˣ ρˣ / ρᵈ).
@inline function moisture_pressure_sensitivities(formulation::LiquidIcePotentialTemperatureFormulation,
                                                 𝒰, T, ρᵈ, constants)
    q = 𝒰.moisture_mass_fractions
    θ = 𝒰.potential_temperature
    ρ = 𝒰.density
    κ̂ = 𝒰.exner_exponent

    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    H = cᵖᵐ * T - condensate_latent_energy(q, constants)
    Ĥ = ifelse(H == 0, one(H), H)
    θ̂ = ifelse(θ == 0, one(θ), θ)
    ρ̂ = ifelse(ρ == 0, one(ρ), ρ)
    Π = Ĥ / (ĉᵖ * θ̂)

    χᵛ = phase_potential_temperature_content(Val(:vapor), κ̂, q, constants, T, θ, ĉᵖ, Π)
    χˡ = phase_potential_temperature_content(Val(:liquid), κ̂, q, constants, T, θ, ĉᵖ, Π)
    χⁱ = phase_potential_temperature_content(Val(:ice), κ̂, q, constants, T, θ, ĉᵖ, Π)

    density_ratio = ρᵈ / ρ̂
    gas_pressure_content = θ * (cᵖᵐ * T / Ĥ) * vapor_gas_constant(constants) / Rᵐ
    Λᴴ = θ * condensate_latent_energy(q, constants) / Ĥ

    return density_ratio * (χᵛ - gas_pressure_content), density_ratio * χˡ, density_ratio * χⁱ, Λᴴ
end

# The change q̂ˣ − q of the mass fractions per unit of phase x added to a unit of mass, at fixed
# partial densities of the rest.
@inline added_phase_increment(::Val{:vapor}, q) = MoistureMassFractions(1 - q.vapor, - q.liquid, - q.ice)
@inline added_phase_increment(::Val{:liquid}, q) = MoistureMassFractions(- q.vapor, 1 - q.liquid, - q.ice)
@inline added_phase_increment(::Val{:ice}, q) = MoistureMassFractions(- q.vapor, - q.liquid, 1 - q.ice)

@inline phase_potential_temperature_content(phase, κ̂, q, constants, T, θ, ĉᵖ, Π) =
    PotentialTemperatureFormulations.potential_temperature_content(added_phase_increment(phase, q), κ̂, q,
                                                                   constants, T, θ, ĉᵖ, Π)

# The liquid fraction of each condensate mass the scheme carries: the phase its sensitivity is
# that of. A scheme declares it per mass with `condensate_liquid_fraction`.
function condensate_liquid_fractions(microphysics, names)
    fractions = map(name -> condensate_liquid_fraction(microphysics, Val(name)), names)

    for (name, fraction) in zip(names, fractions)
        isnothing(fraction) &&
            throw(ArgumentError("The condensate mass $name of $(nameof(typeof(microphysics))) declares no " *
                                "condensate_liquid_fraction, which the acoustic substep loop needs to " *
                                "project its tendency onto the pressure: define " *
                                "AtmosphereModels.condensate_liquid_fraction(microphysics, ::Val{:$name})."))
    end

    return fractions
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Project the moisture of a stage onto the pressure the acoustic loop integrates, once the stage's
perturbations are initialized and its slow tendencies `Gⁿ` are complete:

- add to the thermodynamic perturbation the content of the pressure at the state `U⁰` the loop
  starts from, that of its moisture and dry densities relative to the stage-entry ones, so that
  the loop starts from the pressure of `U⁰` to first order;
- store the slow forcing of that perturbation, the thermodynamic tendency plus the projection of
  the moisture and dry density tendencies, in the projection's `tendency`;
- store the content of the stage's start, ramped by those tendencies over `ramp_interval`, in the
  projection's `content`.

`ramp_interval` is the interval the loop integrates the forcing over, or zero when the content is
advanced with the loop instead (see `content_ramp_interval`).
"""
function project_moisture_onto_pressure!(substepper::AcousticSubstepper, model, U⁰, Gⁿ, ramp_interval)
    grid = model.grid
    projection = substepper.pressure_projection
    microphysics = model.microphysics

    ρᵡ_name = thermodynamic_density_name(model.formulation)
    moisture_name = moisture_prognostic_name(microphysics)
    condensate_names = condensate_field_names(microphysics)
    liquid_fractions = condensate_liquid_fractions(microphysics, condensate_names)

    launch!(architecture(grid), grid, :xyz, _project_moisture_onto_pressure!,
            substepper.density_potential_temperature_perturbation,
            substepper.previous_density_potential_temperature_perturbation,
            projection.tendency,
            projection.content,
            Gⁿ[ρᵡ_name],
            projection.prognostic_moisture_sensitivity,
            projection.liquid_sensitivity,
            projection.ice_sensitivity,
            projection.dry_density_sensitivity,
            liquid_fractions, ramp_interval,
            U⁰.ρᵈ, model.dynamics.dry_density, Gⁿ.ρᵈ,
            U⁰[moisture_name], model.moisture_density, Gⁿ[moisture_name],
            values(U⁰[condensate_names]),
            values(model.microphysical_fields[condensate_names]),
            values(Gⁿ[condensate_names]))

    fill_halo_regions!(substepper.density_potential_temperature_perturbation)

    return nothing
end

@kernel function _project_moisture_onto_pressure!(ϖ, ϖˢ⁻, Gϖ, W, Gˢρᵡ, Λᵐ, Λˡ, Λⁱ, Λᵈ, liquid_fractions, ramp_interval,
                                                  ρᵈ⁰, ρᵈᴸ, Gˢρᵈ,
                                                  ρqᵐ⁰, ρqᵐᴸ, Gˢρqᵐ, ρqᶜ⁰, ρqᶜᴸ, Gˢρqᶜ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        λᵐ = Λᵐ[i, j, k]
        λˡ = Λˡ[i, j, k]
        λⁱ = Λⁱ[i, j, k]
        λᵈ = Λᵈ[i, j, k]

        # The content of the moisture and dry densities the loop starts from, relative to the
        # stage-entry ones, and the rate at which their slow tendencies add to it.
        W⁰ = - λᵐ * (ρqᵐ⁰[i, j, k] - ρqᵐᴸ[i, j, k]) -
               (condensate_sensitivity_sum(i, j, k, λˡ, λⁱ, liquid_fractions, ρqᶜ⁰) -
                condensate_sensitivity_sum(i, j, k, λˡ, λⁱ, liquid_fractions, ρqᶜᴸ)) +
               λᵈ * (ρᵈ⁰[i, j, k] - ρᵈᴸ[i, j, k])

        Gᵂ = - λᵐ * Gˢρqᵐ[i, j, k] - condensate_sensitivity_sum(i, j, k, λˡ, λⁱ, liquid_fractions, Gˢρqᶜ) +
               λᵈ * Gˢρᵈ[i, j, k]

        ϖ⁰ = ϖ[i, j, k] + W⁰
        ϖ[i, j, k] = ϖ⁰
        # The divergence damping measures the change of ϖ over a substep from the stash ϖˢ⁻, which
        # stage initialization sets equal to the perturbation: the content added here is not such a
        # change, so the stash takes it too and the first substep's damping stays zero.
        ϖˢ⁻[i, j, k] = ϖ⁰
        Gϖ[i, j, k] = Gˢρᵡ[i, j, k] + Gᵂ
        W[i, j, k] = W⁰ + ramp_interval * Gᵂ
    end
end

# Σ over the condensate masses of their sensitivity times the value of `fields` at (i, j, k). A
# mass with liquid fraction f holds f Λˡ + (1 − f) Λⁱ.
@inline condensate_sensitivity_sum(i, j, k, λˡ, λⁱ, ::Tuple{}, ::Tuple{}) = zero(λˡ)

@inline function condensate_sensitivity_sum(i, j, k, λˡ, λⁱ, liquid_fractions::Tuple, fields::Tuple)
    f = first(liquid_fractions)
    c = @inbounds first(fields)[i, j, k]
    return (f * λˡ + (1 - f) * λⁱ) * c +
           condensate_sensitivity_sum(i, j, k, λˡ, λⁱ, Base.tail(liquid_fractions), Base.tail(fields))
end

# The interval over which `project_moisture_onto_pressure!` ramps the content it stores: the
# loop's, so that the stored content is the stage's last, unless the content is advanced with the
# loop (`remove_moisture_content!`).
content_ramp_interval(advection, interval) = interval
content_ramp_interval(::AdaptiveImplicitVerticalAdvection, interval) = zero(interval)

# Advance the moisture content by the ramp of the moisture tendencies over `Δτ`, the difference of
# the forcing of ϖ and the thermodynamic tendency, and take it off `ϖ★`; `_restore_moisture_content!`
# puts it back.
@kernel function _remove_moisture_content!(ϖ★, W, Gϖ, Gˢρᵡ, Δτ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Wᵐ⁺ = W[i, j, k] + Δτ * (Gϖ[i, j, k] - Gˢρᵡ[i, j, k])
        W[i, j, k] = Wᵐ⁺
        ϖ★[i, j, k] -= Wᵐ⁺
    end
end

@kernel function _restore_moisture_content!(ϖ★, W)
    i, j, k = @index(Global, NTuple)
    @inbounds ϖ★[i, j, k] += W[i, j, k]
end

#####
##### Section 4 — Adaptive substep computation (acoustic CFL)
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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
#
# When adaptive, a stage covering more than half of Δt takes at least two substeps. A single
# acoustic substep over most of the outer step is unstable even below the acoustic CFL target:
# SSPMIS43 (first stage 0.56 Δt) on a 5 km tropical-cyclone grid grew a near-surface mode and
# failed for every Δt between 11 and 12.5 s, where its first stage got one substep of 6.2–7 s,
# while Δt ≥ 13 s, with two substeps, ran 6 h. Short stages keep the CFL count.
@inline minimum_stage_substeps(β) = ifelse(β > 1//2, 2, 1)

@inline _stage_substeps(N::Integer, β, Δt_stage, Δx_min, constants, acoustic_cfl) = max(1, ceil(Int, β * N))
@inline _stage_substeps(::Nothing, β, Δt_stage, Δx_min, constants, acoustic_cfl) =
    max(minimum_stage_substeps(β), compute_acoustic_substeps(Δx_min, Δt_stage, constants, acoustic_cfl))

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
# with ωᵐ⁺=(1+ε)/2, ωˢ⁻=(1−ε)/2 (ε=0 centered). p′ = Cᴸ(ρθ)′ with the cell-centered Cᴸ, so the
# discrete PGF is the gradient of the product Cᴸ·(ρθ)′ (not Cᴸ_face·∂z(ρθ)′). Post-solve
# substitution (δτᵐ⁺=ωᵐ⁺Δτ):
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
# Implicit vertical damping: for `ThermalDivergenceDamping(damp_vertical=true)`, the vertical divergence
# damping folds into the same tridiag via a discrete vertical Laplacian on (ρw)′:
#
#   (ρw)′ₙ − ωαΔz² ∂z²(ρw)′ₙ = (ρw)′ₒ + (1−ω)αΔz² ∂z²(ρw)′ₒ.
#
# With dᵐ⁺≡ωαΔz², the −∂z² stencil adds  A[k,k±1] += −dᵐ⁺ rdz_c(k or k−1)/Δzᶠ(k),  A[k,k] += +dᵐ⁺(rdz_c(k)
# + rdz_c(k−1))/Δzᶠ(k); the matching (1−ω) term goes on the predictor RHS in `_build_vertical_rhs!`.
# Constant-Courant scaling γ_z=αΔz²/Δτ makes dᵐ⁺ Δτ-independent. `damp_vertical=false`/`NoDivergenceDamping` ⇒ 0.

# Implicit upper Rayleigh sponge (Klemp, Dudhia & Hassiotis 2008): a layer of thickness `depth`
# below the lid damps the **total** vertical momentum at peak `damping_rate` (1/s) × ramp shape,
# using the reference face coordinate `rnode` (terrain-following grids get a horizontally uniform
# sponge in r). Outside the layer the ramp vanishes and nothing is touched.
#
# Within a substep the total splits as ρw = ρwᴸ + (ρw)′, so the damping −γ(ρwᴸ + (ρw)′) splits too:
#
#   * the stage-entry part −γ ρwᴸ is a *known* constant over the substep loop, so it rides along in
#     the slow tendency Gˢρw (`sponge_slow_tendency`, assembled once per stage);
#   * the acoustic part −γ (ρw)′ is CN-weighted into the column tridiag: `|δτᵐ⁺|·γ` on the LHS
#     diagonal (`sponge_term_diag`), matched by `|δτˢ⁻|·γ·ρw_old` on the RHS in
#     `_build_vertical_rhs!` (`sponge_rhs`). Local in z, so no off-diagonal coupling.
#
# Damping only the perturbation would leave an established (quasi-steady) gravity wave untouched —
# (ρw)′ measures the *change* over a stage, not the wave itself — and the absorbing layer would be a
# no-op at the very moment it is needed. `|δτ|` (not δτ) makes the acoustic half a one-sided
# dissipative regularizer for either integration direction (forward through a sponge is
# intentionally not exactly invertible).
#
# γ = rate × ramp(z) is a function of the reference face coordinate alone, so it is precomputed
# once per z-face by `materialize_sponge` and read here with a single load. It is read twice per
# face per acoustic substep: from the tridiag diagonal, inside the serial-in-k Thomas sweep, and
# from `_build_vertical_rhs!`.
@inline sponge_damping(i, j, k, grid, ::Nothing) = zero(grid)
@inline sponge_damping(i, j, k, grid, sponge::UpperSponge) = @inbounds sponge.damping_profile[i, j, k]

materialize_sponge(grid, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Build the grid-dependent form of `sponge` by evaluating its ramp into a 1D profile of the damping
rate ``γ(z) = \\text{rate} × \\text{ramp}(z)`` over z-faces. Called by the
[`AcousticSubstepper`](@ref) constructor; a `nothing` sponge returns `nothing`.

The user-facing [`UpperSponge`](@ref) constructor has no grid, so it produces a skeleton with
`damping_profile = nothing` — this is the matching materialization step.
"""
function materialize_sponge(grid, sponge::UpperSponge)
    isnothing(sponge.mask) || return materialize_masked_sponge(grid, sponge)

    # Converting to the *grid* float type is what keeps γ out of the tridiag's promotion path: a
    # Float64 γ on a Float32 grid would widen the whole Thomas recurrence, per column per substep.
    FT = eltype(grid)
    damping_rate = convert(FT, sponge.damping_rate)
    depth = convert(FT, sponge.depth)

    # Faces 1:Nz+1 are the whole read range (the tridiag covers 1:Nz and `_build_vertical_rhs!`
    # is launched over 1:Nz+1), so the halo values are left at the field's initial zero.
    damping_profile = Field{Nothing, Nothing, Face}(grid)

    launch!(architecture(grid), grid, tuple(size(grid, 3) + 1), _compute_sponge_damping_profile!,
            damping_profile, grid, damping_rate, depth, sponge.ramp)

    return UpperSponge(damping_rate, depth, sponge.ramp, nothing, damping_profile)
end

# With a `mask` the damping rate varies horizontally, so it is evaluated into a full z-face field:
# γ = rate × max(ramp(r), mask(x, y, z)), read at the same (i, j, k) as the 1D profile.
function materialize_masked_sponge(grid, sponge::UpperSponge)
    FT = eltype(grid)
    damping_rate = convert(FT, sponge.damping_rate)
    depth = convert(FT, sponge.depth)
    damping_profile = ZFaceField(grid)
    Nx, Ny, Nz = size(grid)

    launch!(architecture(grid), grid, (Nx, Ny, Nz + 1), _compute_masked_sponge_damping!,
            damping_profile, grid, damping_rate, depth, sponge.ramp, sponge.mask)

    return UpperSponge(damping_rate, depth, sponge.ramp, nothing, damping_profile)
end

@kernel function _compute_masked_sponge_damping!(γ, grid, damping_rate, depth, ramp, mask)
    i, j, k = @index(Global, NTuple)
    r = rnode(i, j, k, grid, Center(), Center(), Face())
    # `node` gives the coordinates `set!` passes to functions: (λ, φ, z) on a LatitudeLongitudeGrid
    # (`xnode`/`ynode` would give distances in metres there), (x, y, z) on a RectilinearGrid.
    m = mask(node(i, j, k, grid, Center(), Center(), Face())...)
    @inbounds γ[i, j, k] = damping_rate * max(ramp(r, grid.Lz, depth), m)
end

# `ramp` is an arbitrary user-supplied callable, so this is the only place it is evaluated,
# once per face at construction rather than per substep.
@kernel function _compute_sponge_damping_profile!(γ, grid, damping_rate, depth, ramp)
    k = @index(Global)
    z = rnode(1, 1, k, grid, Center(), Center(), Face())
    @inbounds γ[1, 1, k] = damping_rate * ramp(z, grid.Lz, depth)
end

@inline sponge_term_diag(i, j, k, grid, sponge, δτᵐ⁺) =
    abs(δτᵐ⁺) * sponge_damping(i, j, k, grid, sponge)

# `_build_vertical_rhs!` loads ρw′ at this face anyway, so there is nothing for a `::Nothing`
# specialization to save here (unlike `sponge_slow_tendency` below).
@inline sponge_rhs(i, j, k, grid, sponge, δτˢ⁻, ρw_old) =
    @inbounds abs(δτˢ⁻) * sponge_damping(i, j, k, grid, sponge) * ρw_old[i, j, k]

# Rayleigh damping of the stage-entry vertical momentum, folded into the slow tendency Gˢρw. `ρwᴸ`
# is the **Cartesian** ρw on both height-coordinate and terrain-following grids (see the terrain
# assembly for why targeting ρw̃ would force the mean flow). `direction = sign(Δτ)` keeps the term
# dissipative when integrating backwards, since `_build_vertical_rhs!` scales Gˢρw by the *signed*
# Δτ (the acoustic half gets the same treatment through its `abs(δτ)` weights). Riding in Gˢρw also
# means this half is scaled by `vertical_momentum_tendency_factor` along with the rest of the slow
# tendency — irrelevant at its default of 1, but worth knowing if that knob is ever turned down.
# The `::Nothing` method is what keeps ρwᴸ out of the sponge-free assembly kernels entirely.
@inline sponge_slow_tendency(i, j, k, grid, ::Nothing, ρwᴸ, direction) = zero(grid)

@inline sponge_slow_tendency(i, j, k, grid, sponge::UpperSponge, ρwᴸ, direction) =
    @inbounds -direction * sponge_damping(i, j, k, grid, sponge) * ρwᴸ[i, j, k]

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagLower, p, ::ZDirection,
                                 θᴸ, Cᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
    kᶠ     = k + 1
    Δz⁻¹ᶠ  = Δz⁻¹ᶜᶜᶠ(i, j, kᶠ, grid)
    Δz⁻¹ᵏ⁻ = Δz⁻¹ᶜᶜᶜ(i, j, kᶠ - 1, grid)

    @inbounds Cᵏ⁻ = Cᴸ[i, j, kᶠ - 1]
    θᵏ⁻ = ℑbzᵃᵃᶠ(i, j, kᶠ - 1, grid, θᴸ)

    pgf_term  = - δτᵐ⁺^2 * Cᵏ⁻ * θᵏ⁻ * Δz⁻¹ᵏ⁻ * Δz⁻¹ᶠ
    buoy_term = + δτᵐ⁺^2 * g * Δz⁻¹ᵏ⁻ / 2
    damp_term = - dᵐ⁺ * Δz⁻¹ᵏ⁻ * Δz⁻¹ᶠ

    # Upper sponge is local in z (Rayleigh-type), so no off-diagonal coupling.
    return pgf_term + buoy_term + damp_term
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagDiagonal, p, ::ZDirection,
                                 θᴸ, Cᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)

    Δz⁻¹ᶠ  = Δz⁻¹ᶜᶜᶠ(i, j, k, grid)
    Δz⁻¹ᵏ⁺ = Δz⁻¹ᶜᶜᶜ(i, j, k,     grid)
    Δz⁻¹ᵏ⁻ = Δz⁻¹ᶜᶜᶜ(i, j, k - 1, grid)

    @inbounds begin
        Cᵏ⁺ = Cᴸ[i, j, k]
        Cᵏ⁻ = Cᴸ[i, j, k - 1]
    end

    θᶜᶜᶠ = ℑbzᵃᵃᶠ(i, j, k, grid, θᴸ)

    pgf_diag   = δτᵐ⁺^2 * θᶜᶜᶠ * (Cᵏ⁺ * Δz⁻¹ᵏ⁺ + Cᵏ⁻ * Δz⁻¹ᵏ⁻) * Δz⁻¹ᶠ
    buoy_diag  = δτᵐ⁺^2 * g * (Δz⁻¹ᵏ⁺ - Δz⁻¹ᵏ⁻) / 2
    damp_diag  = dᵐ⁺ * (Δz⁻¹ᵏ⁺ + Δz⁻¹ᵏ⁻) * Δz⁻¹ᶠ
    spnge_diag = sponge_term_diag(i, j, k, grid, sponge, δτᵐ⁺)

    return 1 + (pgf_diag + buoy_diag + damp_diag + spnge_diag) * (k > 1)
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagUpper, p, ::ZDirection,
                                 θᴸ, Cᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)

    Δz⁻¹ᶠ  = Δz⁻¹ᶜᶜᶠ(i, j, k, grid)
    Δz⁻¹ᵏ⁺ = Δz⁻¹ᶜᶜᶜ(i, j, k, grid)

    @inbounds Cᵏ⁺ = Cᴸ[i, j, k]
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
#   ∂t (ρw)′ = Gˢρw - ∂z(Cᴸ · (ρθ)′)  -  g · ρ′
#
# Total force = Gˢρw + perturbation force = full ∂t(ρw) at the
# linearization-consistent level. With a hydrostatic-balanced reference
# state, the reference subtraction makes Gˢρw vanish identically on a
# resting atmosphere (no FP-rounding noise).

# Height-coordinate (non-terrain) slow vertical-momentum tendency. The terrain method is dispatched
# separately (`::TerrainCompressibleModel`), so here the single `reference_state`, when present, is
# a 1D-column `ExnerReferenceState`; `nothing` selects the full-pressure fallback.
function assemble_slow_vertical_momentum_tendency!(substepper::AcousticSubstepper, model,
                                                  β_stage = nothing, Δτ = one(eltype(model.grid)))
    grid = model.grid
    arch = architecture(grid)
    FT   = eltype(grid)
    g    = convert(FT, model.thermodynamic_constants.gravitational_acceleration)
    Gⁿρw = model.timestepper.Gⁿ.ρw
    direction = convert(FT, sign(Δτ))

    ref = model.dynamics.reference_state

    if ref isa Nothing
        launch!(arch, grid, :xyz, _assemble_slow_vertical_momentum_tendency_no_ref!,
                substepper.slow_vertical_momentum_tendency,
                Gⁿρw,
                model.dynamics.pressure,
                model.dynamics.total_density,
                model.momentum.ρw,
                grid, g, substepper.sponge, direction)
    else
        launch!(arch, grid, :xyz, _assemble_slow_vertical_momentum_tendency!,
                substepper.slow_vertical_momentum_tendency,
                Gⁿρw,
                model.dynamics.pressure,
                model.dynamics.total_density,
                ref.pressure, ref.density,
                model.momentum.ρw,
                grid, g, substepper.sponge, direction)
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
@kernel function _assemble_slow_vertical_momentum_tendency!(Gˢρw, Gⁿρw, pᴸ, ρᴸ, pᵣ, ρᵣ, ρwᴸ,
                                                            grid, g, sponge, direction)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Reference-subtracted PGF and buoyancy: at Uᴸ = reference state
        # both terms are exactly zero by construction of the reference.
        ∂z_p′ = ∂zᶜᶜᶠ(i, j, k, grid, δϕ, pᴸ, pᵣ)
        ρ′ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, δϕ, ρᴸ, ρᵣ)
        Gˢρwˢᵖ = sponge_slow_tendency(i, j, k, grid, sponge, ρwᴸ, direction)

        Gˢρw[i, j, k] = (Gⁿρw[i, j, k] - ∂z_p′ - g * ρ′ᶜᶜᶠ + Gˢρwˢᵖ) * (k > 1)
    end
end

# Field perturbation about a reference (used for both pressure and density).
@inline δϕ(i, j, k, grid, ϕᴸ, ϕᵣ) = @inbounds ϕᴸ[i, j, k] - ϕᵣ[i, j, k]

@kernel function _assemble_slow_vertical_momentum_tendency_no_ref!(Gˢρw, Gⁿρw, pᴸ, ρᴸ, ρwᴸ,
                                                                   grid, g, sponge, direction)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂z_pᴸ  = ∂zᶜᶜᶠ(i, j, k, grid, pᴸ)
        ρᴸᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρᴸ)
        Gˢρwˢᵖ = sponge_slow_tendency(i, j, k, grid, sponge, ρwᴸ, direction)
        Gˢρw[i, j, k] = (Gⁿρw[i, j, k] - ∂z_pᴸ - g * ρᴸᶜᶜᶠ + Gˢρwˢᵖ) * (k > 1)
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
function initialize_stage_perturbations!(substepper, model, Uᴸ_outer)
    grid = model.grid
    arch = architecture(grid)

    # Zero the auxiliary workspaces (predictor/damping scratch) and the
    # time-averaged-velocity accumulator slots in one launch. The velocity slots
    # then accumulate raw `momentum_perturbation` each substep and are normalized
    # by `finalize_time_averaged_velocity!` at stage end.
    launch!(arch, grid, :xyz, _zero_stage_workspaces!,
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
    # The stashed (ρθ)′ˢ⁻ starts equal to (ρθ)′: the divergence damping measures the change of
    # (ρθ)′ over a substep, and none has run. A damping applied at the start of the first
    # substep (see `deferred_divergence_damping`) then adds exactly zero.
    launch!(arch, grid, :xyz, _initialize_stage_perturbations!,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.previous_density_potential_temperature_perturbation,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            Uᴸ_outer.ρᵈ, Uᴸ_outer[ρᵡ_name], Uᴸ_outer.ρu, Uᴸ_outer.ρv,
            model.dynamics.dry_density, ρᵡ, model.momentum.ρu, model.momentum.ρv)
    # ρw is dispatched: terrain models initialize the contravariant ρw̃ instead.
    initialize_vertical_momentum_perturbation!(substepper, model, Uᴸ_outer)

    # The only fill of the ρ′ halos in the stage: no substep kernel reads ρ′ off its own column, and the
    # k = 0, Nz+1 values that `_build_vertical_rhs!` reads only feed its discarded boundary rows. Filled
    # here, with (ρθ)′ and its stash in one launch per side, so those discarded reads stay finite.
    fill_halo_regions!((substepper.density_perturbation,
                        substepper.density_potential_temperature_perturbation,
                        substepper.previous_density_potential_temperature_perturbation))
    map(fill_halo_regions!, substepper.momentum_perturbation)

    return nothing
end

# Zero the five stage-start workspace/accumulator fields in one launch.
@kernel function _zero_stage_workspaces!(b, c, d, e, f)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        b[i, j, k] = 0
        c[i, j, k] = 0
        d[i, j, k] = 0
        e[i, j, k] = 0
        f[i, j, k] = 0
    end
end

# Rewind-initialize the four non-vertical perturbations (ρ′, ρθ′, ρu′, ρv′) in one launch.
@kernel function _initialize_stage_perturbations!(ρ′, ρθ′, ρθ′ˢ⁻, ρu′, ρv′,
                                                  ρ_outer, ρθ_outer, ρu_outer, ρv_outer,
                                                  ρ_stage, ρθ_stage, ρu_stage, ρv_stage)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρ′[i, j, k]  = ρ_outer[i, j, k]  - ρ_stage[i, j, k]
        ρθ′[i, j, k] = ρθ_outer[i, j, k] - ρθ_stage[i, j, k]
        ρθ′ˢ⁻[i, j, k] = ρθ_outer[i, j, k] - ρθ_stage[i, j, k]
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
#   ∂x p_full = ∂x pᴸ + ∂x(Cᴸ (ρθ)′)
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
@kernel function _explicit_horizontal_step!(ρu′, ρv′, grid, dynamics, Δτ, ρθ′, Cᴸ, Gⁿρu, Gⁿρv)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂x_pᴸ  = AtmosphereModels.x_pressure_gradient(i, j, k, grid, dynamics)
        ∂x_p′  = ∇ˣp′(i, j, k, grid, dynamics, ρθ′, Cᴸ)
        ∂y_pᴸ  = AtmosphereModels.y_pressure_gradient(i, j, k, grid, dynamics)
        ∂y_p′  = ∇ʸp′(i, j, k, grid, dynamics, ρθ′, Cᴸ)

        ∂x_p = ∂x_pᴸ + ∂x_p′
        ∂y_p = ∂y_pᴸ + ∂y_p′

        ρu′[i, j, k] += Δτ * (Gⁿρu[i, j, k] - ∂x_p)
        ρv′[i, j, k] += Δτ * (Gⁿρv[i, j, k] - ∂y_p)
    end
end

# The same step preceded by the `ThermalDivergenceDamping` correction of the substep before,
#
#   (ρu)′^{τ+Δτ} = [(ρu)′^τ − γˣ ∂x D / θᴸ] + Δτ (Gⁿρu − ∂x pᴸ − ∂x(Cᴸ (ρθ)′)),   D = (ρθ)′ − (ρθ)′ˢ⁻,
#
# in the order the two separate kernels apply them, so the result is the same. Nothing reads
# (ρu)′, (ρv)′ between that damping and this step, and (ρθ)′, (ρθ)′ˢ⁻ and θᴸ do not change in
# between, so the damping can wait for this kernel, which already reads and writes (ρu)′, (ρv)′.
# The first substep of a stage finds D = 0 (`initialize_stage_perturbations!`); the damping of
# the last is applied by `finish_horizontal_momentum_perturbations!`.
@kernel function _damped_explicit_horizontal_step!(ρu′, ρv′, grid, dynamics, Δτ, ρθ′, Cᴸ, Gⁿρu, Gⁿρv,
                                                   ρθ′ˢ⁻, θᴸ, x_damping_scale, y_damping_scale)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂x_pᴸ  = AtmosphereModels.x_pressure_gradient(i, j, k, grid, dynamics)
        ∂x_p′  = ∇ˣp′(i, j, k, grid, dynamics, ρθ′, Cᴸ)
        ∂y_pᴸ  = AtmosphereModels.y_pressure_gradient(i, j, k, grid, dynamics)
        ∂y_p′  = ∇ʸp′(i, j, k, grid, dynamics, ρθ′, Cᴸ)

        ∂x_p = ∂x_pᴸ + ∂x_p′
        ∂y_p = ∂y_pᴸ + ∂y_p′

        ρu′ᵈ = ρu′[i, j, k] - x_thermal_divergence_damping(i, j, k, grid, ρθ′, ρθ′ˢ⁻, θᴸ, Δτ, x_damping_scale)
        ρv′ᵈ = ρv′[i, j, k] - y_thermal_divergence_damping(i, j, k, grid, ρθ′, ρθ′ˢ⁻, θᴸ, Δτ, y_damping_scale)

        ρu′[i, j, k] = ρu′ᵈ + Δτ * (Gⁿρu[i, j, k] - ∂x_p)
        ρv′[i, j, k] = ρv′ᵈ + Δτ * (Gⁿρv[i, j, k] - ∂y_p)
    end
end

# Step A of a substep, with the damping of the substep before when it is deferred to this step
# (`deferred_divergence_damping`).
function explicit_horizontal_step!(::NoDivergenceDamping, substepper, model, Δτ)
    grid = model.grid
    Gⁿ = model.timestepper.Gⁿ
    launch!(architecture(grid), grid, :xyz, _explicit_horizontal_step!,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            grid, model.dynamics, Δτ,
            substepper.density_potential_temperature_perturbation,
            substepper.linearization_pressure_coefficient,
            Gⁿ.ρu, Gⁿ.ρv)
    return nothing
end

function explicit_horizontal_step!(damping::ThermalDivergenceDamping, substepper, model, Δτ)
    grid = model.grid
    Gⁿ = model.timestepper.Gⁿ
    x_damping_scale, y_damping_scale = horizontal_damping_scales(damping, grid)
    launch!(architecture(grid), grid, :xyz, _damped_explicit_horizontal_step!,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            grid, model.dynamics, Δτ,
            substepper.density_potential_temperature_perturbation,
            substepper.linearization_pressure_coefficient,
            Gⁿ.ρu, Gⁿ.ρv,
            substepper.previous_density_potential_temperature_perturbation,
            substepper.linearization_potential_temperature,
            x_damping_scale, y_damping_scale)
    return nothing
end

@inline δpᴸ(i, j, k, grid, ρθ′, Cᴸ) = @inbounds Cᴸ[i, j, k] * ρθ′[i, j, k]

@inline ∇ˣp′(i, j, k, grid, dynamics, ρθ′, Cᴸ) = topology_aware_δxᶠᵃᵃ(i, j, k, grid, δpᴸ, ρθ′, Cᴸ) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline ∇ʸp′(i, j, k, grid, dynamics, ρθ′, Cᴸ) = topology_aware_δyᵃᶠᵃ(i, j, k, grid, δpᴸ, ρθ′, Cᴸ) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
@inline ∇ᶻp′(i, j, k, grid, dynamics, ρθ′, Cᴸ) = ∂zᶜᶜᶠ(i, j, k, grid, δpᴸ, ρθ′, Cᴸ)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Run the `Nτ` acoustic substeps of one RK3 stage by calling [`acoustic_substep!`](@ref) for each.

The substeps run under `ReactantCore.@trace`, which is a plain loop on ordinary arrays and a
single traced `while` loop under Reactant, so the compiled program holds one copy of the substep
body rather than `Nτ`. Reverse-mode differentiation of that loop caches a minimal cut of its
values (`mincut`).
"""
function acoustic_substep_loop!(Nτ, model, substepper, stage, advection)
    @trace mincut=true track_numbers=false for _ in 1:Nτ
        acoustic_substep!(model, substepper, stage, advection)
    end
    finish_horizontal_momentum_perturbations!(model, substepper, stage)
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Apply the divergence damping of the last acoustic substep of a stage, when it is deferred to the
explicit step of the substep after (`deferred_divergence_damping`), then fill the halos of ``(ρu)′``
and ``(ρv)′`` and enforce impenetrability on them. The divergence damping that ends each substep
writes their interior, and inside the loop the next reader of their halos and wall faces is the
predictor of the following substep, which comes after that substep's own explicit step, fill and
enforcement. So only the last substep's damped values need them, for the recovery of the full state.
"""
function finish_horizontal_momentum_perturbations!(model, substepper, stage)
    grid = model.grid
    apply_divergence_damping!(deferred_divergence_damping(substepper.damping), substepper, grid, stage.Δτ,
                              model.thermodynamic_constants)
    fill_halo_regions!(substepper.momentum_perturbation.u)
    fill_halo_regions!(substepper.momentum_perturbation.v)
    enforce_wall_impenetrability!(substepper, model, grid, architecture(grid))
    return nothing
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

    @inbounds begin
        ρθ′ˢ⁻[i, j, k] = ρθ′[i, j, k] # stash old (ρθ)′ for the divergence damping

        V⁻¹ = V⁻¹ᶜᶜᶜ(i, j, k, grid)
        ∇ʰ_M  = V⁻¹ * (δxTᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, ρu′) + δyTᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, ρv′))
        ∇ʰ_θM = V⁻¹ * (δxTᶜᵃᵃ(i, j, k, grid, θFˣ, θᴸ, ρu′) + δyTᵃᶜᵃ(i, j, k, grid, θFʸ, θᴸ, ρv′))

        ρ′★[i, j, k] = ρ′[i, j, k] +
            Δτ * (Gˢρ[i, j, k] - ∇ʰ_M) - δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, Fʷ, dynamics, ρu′, ρv′, ρw′)

        ρθ′★[i, j, k] = ρθ′[i, j, k] +
            Δτ * (fθ * Gˢρᵡ[i, j, k] - ∇ʰ_θM) - δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, θFᶻ, θᴸ, dynamics, ρu′, ρv′, ρw′)
    end
end

# Face-level RHS for `(ρw)′ᵐ⁺`: split weights δτᵐ⁺ (predictor) and δτˢ⁻ (old-step) per
# derivation (15). `dˢ⁻ = (1−ω) α Δz²` is the explicit half of the implicit vertical
# damping (0 when damping off). Boundary rows: f[1] = 0 (matches b[1] = 1 ⇒ (ρw)′[1] = 0);
# top face Nz+1 lives outside the solver (impenetrability w(top) = 0).
@kernel function _build_vertical_rhs!(ρw′_rhs, ρ′★, ρθ′★, ρ′, ρθ′, ρw′,
                                      grid, dynamics, Δτ, δτᵐ⁺, δτˢ⁻, Cᴸ, g, dˢ⁻,
                                      fw, Gˢρw, sponge)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        ∂r_p′★  = ∇ᶻp′(i, j, k, grid, dynamics, ρθ′★, Cᴸ)
        ∂r_p′ˢ⁻ = ∇ᶻp′(i, j, k, grid, dynamics, ρθ′,  Cᴸ)
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
        ρw′_rhs[i, j, k] = ifelse((k != 1) & (k != Nz + 1), rhs, zero(rhs))
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
    @inbounds begin
        ρ′[i, j, k]  = ρ′★[i, j, k]  - δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, Fʷ, dynamics, ρu′, ρv′, ρw′)
        ρθ′ᵐ⁺ = ρθ′★[i, j, k] - δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, θFᶻ, θᴸ, dynamics, ρu′, ρv′, ρw′)
        ρθ′[i, j, k] = ρθ′ᵐ⁺
        avg.u[i, j, k] += ρu′[i, j, k]
        avg.v[i, j, k] += ρv′[i, j, k]
        avg.w[i, j, k] += ρw′[i, j, k]
    end
end

#####
##### Section 9 — Damping
#####

# No-op default
@inline apply_divergence_damping!(::NoDivergenceDamping, args...) = nothing

# The damping that each substep's explicit horizontal step applies for the substep before
# (`explicit_horizontal_step!`), and the damping applied right after each substep: one of the two
# is `NoDivergenceDamping`. Only `ThermalDivergenceDamping` can be deferred: its correction is
# a single pass over fields that stay put until the next substep's predictors.
deferred_divergence_damping(damping) = NoDivergenceDamping()
deferred_divergence_damping(damping::ThermalDivergenceDamping) = damping

immediate_divergence_damping(damping) = damping
immediate_divergence_damping(::ThermalDivergenceDamping) = NoDivergenceDamping()

# The fields whose halos are filled after `_post_solve_recovery!`, in one launch per side: (ρθ)′,
# whose horizontal and vertical neighbors the next substep reads, and what the damping reads next
# to it. Nothing else read off-column is written in a substep except (ρu)′, (ρv)′.
post_recovery_halo_fields(damping, substepper) = (substepper.density_potential_temperature_perturbation,)

# `ThermalDivergenceDamping` differentiates (ρθ)′ − (ρθ)′ˢ⁻ horizontally, and reads no halo of
# (ρθ)′ˢ⁻ before this fill.
post_recovery_halo_fields(::ThermalDivergenceDamping, substepper) =
    (substepper.density_potential_temperature_perturbation,
     substepper.previous_density_potential_temperature_perturbation)

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
    x_damping_scale, y_damping_scale = horizontal_damping_scales(damping, grid)

    launch!(arch, grid, :xyz, _thermal_divergence_damping!,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            substepper.density_potential_temperature_perturbation,
            substepper.previous_density_potential_temperature_perturbation,
            substepper.linearization_potential_temperature,
            grid, convert(FT, Δτ), x_damping_scale, y_damping_scale)

    return nothing
end

function horizontal_damping_scales(damping::ThermalDivergenceDamping, grid)
    α = damping.coefficient
    TX, TY, _ = topology(grid)
    x_damping_scale = TX === Flat ? NoHorizontalDampingScale() : horizontal_damping_scale(damping, α)
    y_damping_scale = TY === Flat ? NoHorizontalDampingScale() : horizontal_damping_scale(damping, α)
    return x_damping_scale, y_damping_scale
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
        ρu′[i, j, k] -= x_thermal_divergence_damping(i, j, k, grid, ρθ′, ρθ′ˢ⁻, θᴸ, Δτ, x_damping_scale)
        ρv′[i, j, k] -= y_thermal_divergence_damping(i, j, k, grid, ρθ′, ρθ′ˢ⁻, θᴸ, Δτ, y_damping_scale)
    end
end

# The corrections γˣ ∂x D / θᴸ and γʸ ∂y D / θᴸ that the damping subtracts from (ρu)′ and (ρv)′.
@inline function x_thermal_divergence_damping(i, j, k, grid, ρθ′, ρθ′ˢ⁻, θᴸ, Δτ, x_damping_scale)
    ∂x_div = topology_aware_δxᶠᵃᵃ(i, j, k, grid, dρθ′, ρθ′, ρθ′ˢ⁻) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    θᴸᶠᶜᶜ  = ℑxᶠᵃᵃ(i, j, k, grid, θᴸ)
    γˣ = κˣ(i, j, k, grid, x_damping_scale, Δτ)
    return γˣ * ∂x_div / θᴸᶠᶜᶜ
end

@inline function y_thermal_divergence_damping(i, j, k, grid, ρθ′, ρθ′ˢ⁻, θᴸ, Δτ, y_damping_scale)
    ∂y_div = topology_aware_δyᵃᶠᵃ(i, j, k, grid, dρθ′, ρθ′, ρθ′ˢ⁻) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    θᴸᶜᶠᶜ  = ℑyᵃᶠᵃ(i, j, k, grid, θᴸ)
    γʸ = κʸ(i, j, k, grid, y_damping_scale, Δτ)
    return γʸ * ∂y_div / θᴸᶜᶠᶜ
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
    fill_substep_halos!(δ, grid)
    launch!(arch, grid, :xyz, _apply_direct_divergence_damping!, ρu′, ρv′, grid, δ, θᴸ, α)

    return nothing
end

# δ = ∇ʰ·(θᴸ(ρ𝐮)′): area-weighted horizontal Θ-flux divergence, using the same flux-form (θFˣ, θFʸ)
# `build_predictors!` uses for ∇ʰ_θM (so the damped divergence matches the one in the Θ equation).
@kernel function _compute_horizontal_theta_flux_divergence!(δ, grid, θᴸ, ρu′, ρv′)
    i, j, k = @index(Global, NTuple)
    @inbounds δ[i, j, k] = (δxTᶜᵃᵃ(i, j, k, grid, θFˣ, θᴸ, ρu′) +
                            δyTᵃᶜᵃ(i, j, k, grid, θFʸ, θᴸ, ρv′)) * V⁻¹ᶜᶜᶜ(i, j, k, grid)
end

@kernel function _apply_direct_divergence_damping!(ρu′, ρv′, grid, δ, θᴸ, α)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρu′[i, j, k] += α * Δxᶠᶜᶜ(i, j, k, grid)^2 * topology_aware_δxᶠᵃᵃ(i, j, k, grid, δ) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid) / ℑxᶠᵃᵃ(i, j, k, grid, θᴸ)
        ρv′[i, j, k] += α * Δyᶜᶠᶜ(i, j, k, grid)^2 * topology_aware_δyᵃᶠᵃ(i, j, k, grid, δ) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid) / ℑyᵃᶠᵃ(i, j, k, grid, θᴸ)
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
##### ρ′ sees the off-centered vertical flux (1 − ω)(ρw)′(n−1) + ω(ρw)′(n), so ⟨ρw⟩ must too.
##### The vertical solve is the only writer of (ρw)′ inside the loop, so the sum of that flux over
##### n = 1…Nτ telescopes onto the raw accumulation:
#####
#####   ∑ₙ [(1 − ω)(ρw)′(n−1) + ω(ρw)′(n)] = ∑ₙ (ρw)′(n) − (1 − ω)[(ρw)′(Nτ) − (ρw)′(0)]
#####
##### The rewind (ρw)′(0) is zero in stage 1 but not in stages 2–3, so it is recomputed
##### from Uᴸ_outer.
#####
##### `model.momentum.*` and `model.dynamics.dry_density` are still the stage-entry
##### (Uᴸ_stage) values here. Dividing by ρᴸ ignores ρ's variation over the
##### loop, small for acoustic perturbations.
#####


function finalize_time_averaged_velocity!(substepper, model, Nτ, Uᴸ_outer)
    grid = model.grid
    arch = architecture(grid)
    FT   = eltype(grid)
    inv_Nτ = one(FT) / FT(Nτ)
    ωˢ⁻ = 1 - substepper.forward_weight

    # `model.dynamics.dry_density` and `model.momentum.*` are still the
    # stage-entry (Uᴸ) values here — the substep loop only touched
    # substepper-owned perturbation fields. They serve as ρᴸ and ρu/v/wᴸ.
    launch!(arch, grid, :xyz, _finalize_time_averaged_velocity!,
            substepper.time_averaged_velocities.u,
            substepper.time_averaged_velocities.v,
            substepper.time_averaged_velocities.w,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            model.dynamics.dry_density,
            substepper.momentum_perturbation.w,
            Uᴸ_outer.ρu, Uᴸ_outer.ρv, Uᴸ_outer.ρw,
            grid, model.dynamics, inv_Nτ, ωˢ⁻)

    map(fill_halo_regions!, substepper.time_averaged_velocities)

    return nothing
end

@kernel function _finalize_time_averaged_velocity!(u_avg, v_avg, w_avg,
                                                   ρu_stage, ρv_stage, ρw_stage,
                                                   ρᴸ, ρw′, ρu_outer, ρv_outer, ρw_outer,
                                                   grid, dynamics, inv_Nτ, ωˢ⁻)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρwᴸ = transport_ρw(i, j, k, grid, dynamics, ρu_stage, ρv_stage, ρw_stage)
        ρw′⁰ = transport_ρw(i, j, k, grid, dynamics, ρu_outer, ρv_outer, ρw_outer) - ρwᴸ
        ρu_total = ρu_stage[i, j, k] + u_avg[i, j, k] * inv_Nτ
        ρv_total = ρv_stage[i, j, k] + v_avg[i, j, k] * inv_Nτ
        ρw_total = ρwᴸ + (w_avg[i, j, k] - ωˢ⁻ * (ρw′[i, j, k] - ρw′⁰)) * inv_Nτ

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

@inline transport_ρw(i, j, k, grid, dynamics, ρu_stage, ρv_stage, ρw_stage) =
    @inbounds ρw_stage[i, j, k]

#####
##### Section 11 — Full-state recovery at stage end
#####

# After the substep loop completes for a stage, reconstruct the full
# prognostic state ρ, ρu, ρv, ρw, ρθ from the stage-entry linearization
# state plus the accumulated perturbations:
#   ρᵐ⁺  = ρᴸ  + ρ′
#   ρθᵐ⁺ = ρθᴸ + (ϖ − W)
#   ρuᵐ⁺ = ρuᴸ + (ρu)′, etc.
#
# The loop integrated the pressure ϖ = (ρθ)′ + W; the moisture content W that the stage's
# projection put into it comes off (see `PressureProjection`), which leaves the thermodynamic
# density with its own tendency and the flux divergence the loop applied.
#
# Velocity diagnosis is deliberately not done in this kernel. Face velocities
# require neighbor-cell density interpolation; computing them while this same
# kernel writes ρ can read a GPU-scheduling-dependent mix of old and new
# neighbor values. The driver calls AtmosphereModels.compute_velocities! after
# recovery and halo fill.
#
# Each sum is rounded with the round-off treatment `rounding` (see `RoundingTreatments`);
# `key` seeds its random numbers, offset by one per prognostic.
@kernel function _recover_full_state!(ρ, ρθ, m,
                                      ρ′, ρθ′, W, ρu′, ρv′, ρw′,
                                      ρᴸ, ρuᴸ, ρvᴸ, ρwᴸ, ρθᴸ,
                                      grid, dynamics, rounding, key)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵐ⁺  = rounded_add(rounding, ρᴸ[i, j, k],  ρ′[i, j, k],  i, j, k, key + 0x00000001)
        ρθᵐ⁺ = rounded_add(rounding, ρθᴸ[i, j, k], ρθ′[i, j, k] - W[i, j, k], i, j, k, key + 0x00000002)
        ρuᵐ⁺ = rounded_add(rounding, ρuᴸ[i, j, k], ρu′[i, j, k], i, j, k, key + 0x00000003)
        ρvᵐ⁺ = rounded_add(rounding, ρvᴸ[i, j, k], ρv′[i, j, k], i, j, k, key + 0x00000004)
        ρwᵐ⁺ = rounded_recovered_vertical_momentum(rounding, i, j, k, grid, dynamics, ρwᴸ[i, j, k],
                                                   ρu′, ρv′, ρw′, key + 0x00000005)

        ρ[i, j, k]  = ρᵐ⁺
        ρθ[i, j, k] = ρθᵐ⁺

        m.ρu[i, j, k] = ρuᵐ⁺
        m.ρv[i, j, k] = ρvᵐ⁺
        m.ρw[i, j, k] = ρwᵐ⁺
    end
end

@inline acoustic_recovered_vertical_momentum(i, j, k, grid, dynamics, ρwᴸ_ccf, ρu′, ρv′, ρw′) =
    @inbounds ρwᴸ_ccf + ρw′[i, j, k]

# Round-to-nearest keeps the original summation order (bitwise unchanged); other treatments
# round the sum of the stage-entry value and the whole recovered increment.
@inline rounded_recovered_vertical_momentum(::RoundToNearest, i, j, k, grid, dynamics, ρwᴸ_ccf, ρu′, ρv′, ρw′, key) =
    acoustic_recovered_vertical_momentum(i, j, k, grid, dynamics, ρwᴸ_ccf, ρu′, ρv′, ρw′)

@inline function rounded_recovered_vertical_momentum(rounding, i, j, k, grid, dynamics, ρwᴸ_ccf, ρu′, ρv′, ρw′, key)
    δ = acoustic_recovered_vertical_momentum(i, j, k, grid, dynamics, 0, ρu′, ρv′, ρw′)
    return rounded_add(rounding, ρwᴸ_ccf, δ, i, j, k, key)
end

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
# The loop's thermodynamic unknown is the pressure ϖ = (ρθ)′ + W, so its target is that of (ρθ)′
# plus the moisture content W of the cell (see `PressureProjection`): the recovery takes W off
# again, and the thermodynamic density of the boundary cell tracks its prescribed value.
# Relaxation factor `α ∈ (0, 1]` (default 0.5, via
# `SplitExplicitTimeDiscretization(; open_boundary_relaxation = α)`). No-op on any
# side without an active `NormalFlowBoundaryCondition` (periodic/walls/`nothing`
# all skip it) — zero cost when no open lateral BC is present.

@inline is_active_open_bc(bc) = (bc isa BoundaryCondition{<:NormalFlow}) && !(bc.condition isa Nothing)

# Relax ρ′ and ϖ at the outermost open-boundary cell toward the prescribed
# wall value in a single kernel: target = v − cᴸ[iᴮ] = (cᴸ[iᴴ] − cᴸ[iᴮ]) / 2, plus the moisture
# content W of the cell for ϖ. `iᴮ` is the outermost interior cell index, `iᴴ` the adjacent halo
# cell index.
@kernel function _relax_open_boundary_x!(ρ′, ρθ′, ρᴸ, ρθᴸ, W, iᴮ, iᴴ, α)
    j, k = @index(Global, NTuple)
    @inbounds begin
        ρ′[iᴮ, j, k]  += α * ((ρᴸ[iᴴ, j, k]  - ρᴸ[iᴮ, j, k])  / 2 - ρ′[iᴮ, j, k])
        ρθ′[iᴮ, j, k] += α * ((ρθᴸ[iᴴ, j, k] - ρθᴸ[iᴮ, j, k]) / 2 + W[iᴮ, j, k] - ρθ′[iᴮ, j, k])
    end
end

@kernel function _relax_open_boundary_y!(ρ′, ρθ′, ρᴸ, ρθᴸ, W, jᴮ, jᴴ, α)
    i, k = @index(Global, NTuple)
    @inbounds begin
        ρ′[i, jᴮ, k]  += α * ((ρᴸ[i, jᴴ, k]  - ρᴸ[i, jᴮ, k])  / 2 - ρ′[i, jᴮ, k])
        ρθ′[i, jᴮ, k] += α * ((ρθᴸ[i, jᴴ, k] - ρθᴸ[i, jᴮ, k]) / 2 + W[i, jᴮ, k] - ρθ′[i, jᴮ, k])
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
    W   = substepper.pressure_projection.content
    if is_active_open_bc(bcs_u.west)
        launch!(arch, grid, :yz, _relax_open_boundary_x!, ρ′, ρθ′, ρᴸ, ρθᴸ, W, 1, 0, α)
    end
    if is_active_open_bc(bcs_u.east)
        launch!(arch, grid, :yz, _relax_open_boundary_x!, ρ′, ρθ′, ρᴸ, ρθᴸ, W, Nx, Nx + 1, α)
    end
    if is_active_open_bc(bcs_v.south)
        launch!(arch, grid, :xz, _relax_open_boundary_y!, ρ′, ρθ′, ρᴸ, ρθᴸ, W, 1, 0, α)
    end
    if is_active_open_bc(bcs_v.north)
        launch!(arch, grid, :xz, _relax_open_boundary_y!, ρ′, ρθ′, ρᴸ, ρθᴸ, W, Ny, Ny + 1, α)
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

function enforce_wall_impenetrability!(substepper, model, grid, arch)
    TX, TY, _ = topology(grid)
    Nx, Ny, _ = size(grid)
    ρu′ = substepper.momentum_perturbation.u
    ρv′ = substepper.momentum_perturbation.v
    # The perturbations carry impenetrability on every `Bounded` side, open ones included (an open
    # boundary acts through `apply_open_boundary_relaxation!` on ρ′, (ρθ)′): their halo fill zeroed
    # these faces, and the topology-aware divergence (`δxTᶜᵃᵃ`) assumes it.
    if TX === Bounded
        launch!(arch, grid, :yz, _zero_x_wall_face!, ρu′, 1)
        launch!(arch, grid, :yz, _zero_x_wall_face!, ρu′, Nx + 1)
    end
    if TY === Bounded
        launch!(arch, grid, :xz, _zero_y_wall_face!, ρv′, 1)
        launch!(arch, grid, :xz, _zero_y_wall_face!, ρv′, Ny + 1)
    end
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Advance the implicit half of the IMEX split for the thermodynamic perturbation over one
substep: `(I + Δτ Lⁱ) ρθ′★ⁿᵉʷ = ρθ′★`, solved on the θ predictor between Step B and Step C so
the pressure solve and recovery act on a transport-consistent predictor. Coefficients read the
frozen stage-entry state; the base-state part rides `Gˢρθ`
(see `add_implicit_advection_tendency!`).

The predictor the loop holds is that of the pressure, ``ϖ★ = (ρθ)′★ + W``, while the split
transports the thermodynamic density. The moisture content ``W`` of the pressure at the end of
this substep (see [`PressureProjection`](@ref)) is therefore taken off the predictor for the
solve and put back after it.
"""
implicit_advection_substep!(model, substepper, advection, Δτ) = nothing

function implicit_advection_substep!(model, substepper, advection::AdaptiveImplicitVerticalAdvection, Δτ)
    grid = model.grid
    arch = architecture(grid)
    predictor = substepper.density_potential_temperature_predictor
    projection = substepper.pressure_projection
    Gˢρᵡ = model.timestepper.Gⁿ[thermodynamic_density_name(model.formulation)]

    solver = substepper.implicit_advection_solver
    Nz = size(grid, 3)

    launch!(arch, grid, :xy, _solve_implicit_advection_without_moisture_content!,
            predictor, projection.content, projection.tendency, Gˢρᵡ,
            Δτ * substepper.thermodynamic_tendency_factor,
            solver.factors.β, solver.factors.a, solver.t, Nz)

    return nothing
end

# The implicit-advection solve of one column, with the moisture content `W` taken off the predictor
# `ϖ★` as the right-hand side is read and put back as the solution is written: the removal
# (`_remove_moisture_content!`), the factored substitution (`solve!` with the stored factorization of
# `factor_implicit_advection!`) and the restoration (`_restore_moisture_content!`) in one pass over
# the column, with the same arithmetic as the three separate passes. `β`, `a` and `t` are the
# stored pivots, lower diagonal and elimination ratios of the factorization.
@kernel function _solve_implicit_advection_without_moisture_content!(ϖ★, W, Gϖ, Gˢρᵡ, Δτ, β, a, t, Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(ϖ★)

    @inbounds begin
        Wᵏ = W[i, j, 1] + Δτ * (Gϖ[i, j, 1] - Gˢρᵡ[i, j, 1])
        W[i, j, 1] = Wᵏ
        f = convert(FT, ϖ★[i, j, 1] - Wᵏ)
        ϕ⁻ = convert(FT, f / β[i, j, 1])
        ϖ★[i, j, 1] = ϕ⁻

        for k = 2:Nz
            Wᵏ = W[i, j, k] + Δτ * (Gϖ[i, j, k] - Gˢρᵡ[i, j, k])
            W[i, j, k] = Wᵏ
            f = convert(FT, ϖ★[i, j, k] - Wᵏ)
            βᵏ = β[i, j, k]
            definitely_diagonally_dominant = abs(βᵏ) > 10 * eps(FT)
            ϕ★ = (f - masked_multiply(a[i, j, k-1], ϕ⁻)) / βᵏ
            ϕ⁻ = convert(FT, ifelse(definitely_diagonally_dominant, ϕ★, f))
            ϖ★[i, j, k] = ϕ⁻
        end

        ϕ⁺ = ϕ⁻
        ϖ★[i, j, Nz] = convert(FT, ϕ⁺ + W[i, j, Nz])
        for k = Nz-1:-1:1
            ϕ⁺ = convert(FT, ϖ★[i, j, k] - masked_multiply(t[i, j, k+1], ϕ⁺))
            ϖ★[i, j, k] = convert(FT, ϕ⁺ + W[i, j, k])
        end
    end
end

# The (ρθ)′ implicit-advection system is the same on every substep of a stage — its coefficients depend only on
# the frozen stage-entry w and ρᵈ caches and on Δτ — so it is factored once per stage (`factor_implicit_advection!`)
# and each substep only substitutes. The factorization reproduces `implicit_step!` bitwise: it evaluates the
# implicit solver's own coefficients with the arguments `implicit_step!` passes.
factor_implicit_advection!(model, substepper, advection, Δτ) = nothing

function factor_implicit_advection!(model, substepper, advection::AdaptiveImplicitVerticalAdvection, Δτ)
    field = substepper.density_potential_temperature_predictor
    args = implicit_step_solver_arguments(field, nothing, nothing, nothing, model.clock, fields(model),
                                          Δτ, advection, substepper.vertical_velocity_cache, substepper.density_cache)
    compute_factorization!(substepper.implicit_advection_solver, args...)
    return nothing
end

# A factorized solver with the coefficients of the model's implicit solver (`implicit_diffusion_solver`)
factorized_implicit_solver(grid) =
    BatchedTridiagonalSolver(grid;
                             lower_diagonal = VerticallyImplicitDiffusionLowerDiagonal(),
                             diagonal       = VerticallyImplicitDiffusionDiagonal(),
                             upper_diagonal = VerticallyImplicitDiffusionUpperDiagonal(),
                             factorized = true)

# The coefficient arguments `Oceananigans.TimeSteppers.implicit_step!` passes to `solve!`: the vertically
# implicit closures (`nothing` for the substep's solve, which integrates advection alone), the advection
# scheme with its velocity and density, and the boundary conditions of `field`.
# TODO: this mirrors `implicit_step!`; an Oceananigans helper returning these arguments would remove the duplication.
function implicit_step_solver_arguments(field, closure, closure_fields, tracer_index, clock, model_fields, Δt,
                                        advection, w, density)
    implicit_closure, implicit_closure_fields = implicit_step_closures(closure, closure_fields)
    LX, LY, LZ = location(field)
    bcs = field.boundary_conditions
    return (implicit_closure, implicit_closure_fields, tracer_index, LX(), LY(), LZ(), Δt, clock, model_fields,
            advection, w, density, bcs.top, bcs.bottom, bcs.immersed, field)
end

# The vertically implicit closures among `closure`, as `implicit_step!` selects them: `nothing` when there are none.
function implicit_step_closures(closure, closure_fields)
    implicit_closure, implicit_closure_fields = vertically_implicit_closures(closure, closure_fields)
    return implicit_closure === () ? (nothing, nothing) : (implicit_closure, implicit_closure_fields)
end

# The per-stage implicit steps (scalars; momentum and the thermodynamic variable) solve up to this
# many fields per column thread (`solve_tridiagonal_columns_in_place!`); 1 solves each field with a
# kernel of its own. The limits are the number of scratch arrays (`implicit_step_scratch`, 4) and
# the size of a GPU kernel's arguments, which each field of a batch adds to (CUDA allows 4 KiB,
# and the model's fields, passed once, take much of that): lower it if a kernel fails to compile
# with a formal-parameter-space overflow.
const column_solve_batch_size = Ref(3)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Advance each of `requests` over `Δt` by its vertically implicit terms — closure diffusion, the implicit
remainder of adaptive implicit vertical advection, implicit boundary fluxes — as
`Oceananigans.TimeSteppers.implicit_step!` does for one field, with the same coefficients and
arithmetic. A request is a tuple `(field, tracer_index, advection, velocities)`; `closure`,
`closure_fields`, `clock`, `model_fields`, `Δt` and `density` are common to all. Like `implicit_step!`,
a field about which nothing is vertically implicit is left alone.

The column sweeps keep the Thomas recurrence in registers and solve up to `column_solve_batch_size[]`
fields per thread (see `solve_tridiagonal_columns_in_place!`), with `scratch` holding one array per
field of a batch. The coefficients of every field are evaluated from the state on entry, except for
its own boundary rows: no field's coefficients may depend on another field of `requests`.

`Δt` enters every coefficient of every level, so pass it in the float type of the grid
(`kernel_time_step`): a `Float64` step on a `Float32` grid makes the whole sweep double precision.
"""
function implicit_steps_in_place!(requests, implicit_solver::BatchedTridiagonalSolver, scratch,
                                  closure, closure_fields, clock, model_fields, Δt, density)
    implicit_closure, implicit_closure_fields = implicit_step_closures(closure, closure_fields)
    shared = (; closure = implicit_closure, closure_fields = implicit_closure_fields,
                Δt, clock, model_fields, density)

    needed = filter(request -> needs_implicit_step(implicit_closure, request...), requests)
    isempty(needed) && return nothing

    batch_size = clamp(column_solve_batch_size[], 1, length(scratch))
    one_by_one = batch_size == 1 || size(implicit_solver.grid, 3) < 2

    for batch in Iterators.partition(needed, one_by_one ? 1 : batch_size)
        fields = Tuple(first(request) for request in batch)
        columns = Tuple(implicit_step_column(request...) for request in batch)
        if one_by_one
            args = implicit_step_arguments(shared, first(columns), first(fields))
            ϕ = first(fields)
            solve!(ϕ, implicit_solver, ϕ, args...)
        else
            solve_tridiagonal_columns_in_place!(fields, implicit_solver, scratch, implicit_step_arguments,
                                                shared, columns)
        end
    end

    return nothing
end

implicit_steps_in_place!(requests, ::Nothing, args...) = nothing

needs_implicit_step(implicit_closure, field, tracer_index, advection, velocities) =
    !isnothing(implicit_closure) || needs_implicit_solver(advection) || needs_implicit_solver(field.boundary_conditions)

# What the coefficients of one field's implicit step take beyond the arguments common to a batch.
function implicit_step_column(field, tracer_index, advection, velocities)
    LX, LY, LZ = location(field)
    bcs = field.boundary_conditions
    w = isnothing(velocities) ? nothing : velocities.w
    return (; tracer_index, ℓx = LX(), ℓy = LY(), ℓz = LZ(), advection, w,
              top = bcs.top, bottom = bcs.bottom, immersed = bcs.immersed)
end

# The coefficient arguments `implicit_step!` passes to `solve!` (see `implicit_step_solver_arguments`),
# from the arguments common to a batch and those of one field.
@inline implicit_step_arguments(shared, column, ϕ) =
    (shared.closure, shared.closure_fields, column.tracer_index, column.ℓx, column.ℓy, column.ℓz,
     shared.Δt, shared.clock, shared.model_fields, column.advection, column.w, shared.density,
     column.top, column.bottom, column.immersed, ϕ)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Scratch arrays for the column sweeps of the implicit steps that follow a stage's acoustic loop: the
implicit solver's own, and the stored factorization of the loop's ``(ρw)′`` solver, which is free once
the loop has run and is recomputed before the next one (`factor_vertical_solver!`).
"""
function implicit_step_scratch(substepper, implicit_solver)
    solver = substepper.vertical_solver
    return (implicit_solver.t, solver.factors.β, solver.t, solver.factors.a)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Factor the column system of the implicit ``(ρw)′`` update for the stage about to run. Its coefficients
depend on the linearization state ``θᴸ``, ``Cᴸ``, on the implicit substep weight `δτᵐ⁺`, and on the
vertical damping and sponge, all fixed over the substeps of a stage, so each substep only substitutes.
"""
function factor_vertical_solver!(substepper, grid, thermodynamic_constants, δτᵐ⁺)
    FT = eltype(grid)
    g = convert(FT, thermodynamic_constants.gravitational_acceleration)
    ω = FT(substepper.forward_weight)
    dᵐ⁺, _ = implicit_damping_factors(substepper.damping, ω, grid, FT)
    args = (substepper.linearization_potential_temperature,
            substepper.linearization_pressure_coefficient, g, δτᵐ⁺, dᵐ⁺,
            substepper.sponge)
    compute_factorization!(substepper.vertical_solver, args...)
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Advance the acoustic perturbation fields of `substepper` by one substep of size `stage.Δτ`.
`stage` holds the loop-invariant quantities of the current RK3 stage: the substep size `Δτ`,
the Crank–Nicolson weights `δτᵐ⁺` and `δτˢ⁻`, and the slow forcing `Gˢρᵡ` of the thermodynamic
perturbation (the pressure in units of the thermodynamic density, see [`PressureProjection`](@ref)).
Everything is passed explicitly so that the loop calling this can be traced by Reactant with
no captured state; see [`acoustic_substep_loop!`](@ref).
"""
function acoustic_substep!(model, substepper, stage, advection)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    constants = model.thermodynamic_constants
    g = convert(FT, constants.gravitational_acceleration)
    ω = FT(substepper.forward_weight)
    Gⁿ = model.timestepper.Gⁿ
    (; Δτ, δτᵐ⁺, δτˢ⁻, Gˢρᵡ) = stage

    # Step A: explicit horizontal forward of (ρu)′, (ρv)′, after the divergence damping of the
    # substep before when that is deferred to this step.
    explicit_horizontal_step!(deferred_divergence_damping(substepper.damping), substepper, model, Δτ)

    # No fills on a single-architecture grid (`fill_substep_halos!`), here and below.
    fill_substep_halos!(substepper.momentum_perturbation.u, model)
    fill_substep_halos!(substepper.momentum_perturbation.v, model)

    # Impenetrability on the wall-normal momentum perturbations, before the
    # predictor takes their horizontal divergence (mass conservation).
    enforce_wall_impenetrability!(substepper, model, grid, arch)

    # (old (ρθ)′ is stashed into ρθ′ˢ⁻ inside `_build_predictors!`; its halos are filled with
    # those of the new (ρθ)′ after Step D, before the damping that reads them.)

    # Implicit-vertical-damping prefactors. When the damping strategy
    # is `ThermalDivergenceDamping(damp_vertical=true)`, the
    # vertical part of the divergence damping is folded into the
    # tridiag with `dᵐ⁺ = ω·α·Δz²` on the LHS and
    # `dˢ⁻ = (1−ω)·α·Δz²` on the predictor RHS. Both reduce to
    # zero for `NoDivergenceDamping` or when the user opts out via
    # `damp_vertical=false`.
    _, dˢ⁻ = implicit_damping_factors(substepper.damping, ω, grid, FT)

    # Step B: build predictors ρ′★, ρθ′★ (3D), then the (ρw)′ᵐ⁺ tridiag RHS (3D).
    # `_build_predictors!` also stashes old (ρθ)′ into ρθ′ˢ⁻.
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

    # Implicit half of the IMEX vertical-advection split for the thermodynamic
    # perturbation, applied to the predictor between Step B and Step C so the pressure
    # solve and recovery substitution see a transport-consistent ρθ′★ (issue #897).
    implicit_advection_substep!(model, substepper, advection, Δτ)

    launch!(arch, grid, KernelParameters(1:size(grid, 1), 1:size(grid, 2), 1:size(grid, 3) + 1),
            _build_vertical_rhs!,
            substepper.vertical_solver_source_term,
            substepper.density_predictor,
            substepper.density_potential_temperature_predictor,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.momentum_perturbation.w,
            grid, model.dynamics, Δτ, δτᵐ⁺, δτˢ⁻,
            substepper.linearization_pressure_coefficient,
            g, dˢ⁻, substepper.vertical_momentum_tendency_factor,
            substepper.slow_vertical_momentum_tendency,
            substepper.sponge)

    # Step C: implicit tridiag solve for (ρw)′ with implicit-half δτᵐ⁺
    # and (when active) implicit vertical damping prefactor `dᵐ⁺`.
    # `sponge` may add an implicit Rayleigh contribution on the
    # diagonal in a layer below the lid. The system is the same on every substep
    # of the stage, so it was factored once (`factor_vertical_solver!`).
    solve!(substepper.momentum_perturbation.w, substepper.vertical_solver, substepper.vertical_solver_source_term)

    # Step D: post-solve recovery of ρ′, (ρθ)′ using new (ρw)′
    launch!(arch, grid, :xyz, _post_solve_recovery!,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.momentum_perturbation.w,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            substepper.density_predictor,
            substepper.density_potential_temperature_predictor,
            substepper.time_averaged_velocities,
            grid, model.dynamics, δτᵐ⁺,
            substepper.linearization_potential_temperature)

    # Per-substep open-boundary enforcement (issue #738): relax the outermost
    # open-boundary cell of ρ′, (ρθ)′ toward the prescribed wall value, before
    # the halo fill, so the boundary cell tracks the prescribed inflow state.
    apply_open_boundary_relaxation!(substepper, model, grid, arch)

    # ρ′ takes no fill: nothing in the loop reads its halos (see `initialize_stage_perturbations!`).
    fill_substep_halos!(post_recovery_halo_fields(substepper.damping, substepper), model)

    # Step E: optional Klemp 2018 post-substep damping (no-op for
    # `NoDivergenceDamping`). It writes (ρu)′, (ρv)′, including the south/west wall face, after
    # this substep's last reader of them: the next substep's explicit step adds to them and then
    # fills their halos and enforces impenetrability, and the last substep's are finished by
    # `finish_horizontal_momentum_perturbations!`.
    apply_divergence_damping!(immediate_divergence_damping(substepper.damping), substepper, grid, Δτ, constants)
    # (time-averaged velocity accumulation is fused into `_post_solve_recovery!` above)
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Execute one Wicker–Skamarock RK3 stage of the linearized acoustic
substep loop. Number and size of substeps in this stage depend on
`substepper.substep_distribution`. The recovery of the full state from the stage-entry state and
the accumulated perturbations is rounded with the round-off treatment `rounding`.
"""
function acoustic_rk3_substep_loop!(model::AtmosphereModel, substepper, Δt, β_stage, Uᴸ, advection = nothing;
                                    rounding = RoundToNearest())
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
    assemble_slow_vertical_momentum_tendency!(substepper, model, β_stage, Δτ)

    # Initialize perturbations with the SK08 rewind term so the substep
    # effectively starts from U(t) = Uᴸ (the outer-step-start state).
    initialize_stage_perturbations!(substepper, model, Uᴸ)

    # The loop integrates the pressure: add the moisture's part of it to the thermodynamic
    # perturbation the loop starts from and to the slow tendency that forces it. The content the
    # projection stores is ramped over the interval the loop applies that forcing over.
    Gⁿ = model.timestepper.Gⁿ
    forcing_interval = Nτ * Δτ * substepper.thermodynamic_tendency_factor
    project_moisture_onto_pressure!(substepper, model, Uᴸ, Gⁿ, content_ramp_interval(advection, forcing_interval))
    Gˢρᵡ = substepper.pressure_projection.tendency

    stage = (; Δτ, δτᵐ⁺, δτˢ⁻, Gˢρᵡ)
    factor_implicit_advection!(model, substepper, advection, Δτ)
    factor_vertical_solver!(substepper, grid, constants, δτᵐ⁺)
    acoustic_substep_loop!(Nτ, model, substepper, stage, advection)

    # Stage-end: convert the accumulated momentum perturbations into a
    # time-averaged velocity field. Read by `update_state!` through
    # `transport_velocities(model)` for moisture/tracer tendencies.
    # Done BEFORE `_recover_full_state!` so we read the stage-entry
    # `model.momentum.*` (substep loop hasn't touched it) and stage-entry
    # `model.dynamics.dry_density` as the Uᴸ_stage reference.
    finalize_time_averaged_velocity!(substepper, model, Nτ, Uᴸ)

    # Stage-end: recover the full prognostic state in-place. `model.dynamics.dry_density`,
    # `ρᵡ`, and `model.momentum.*` are still the stage-entry Uᴸ values here
    # (the substep loop only touched substepper.* perturbation fields). The
    # recovery kernel reads them as Uᴸ AND writes the full state back to the
    # same fields; per-thread read-before-write makes this aliasing safe
    # because all reads are local to the same grid point. The thermodynamic density is recovered
    # from the pressure the loop integrated, less the moisture content of the stage's end. The sums are rounded with `rounding`;
    # field number 0 keeps their keys apart from those of the scalar updates.
    ρᵡ = thermodynamic_density(model.formulation)
    key = rounding_key(rounding, model.clock.iteration, model.clock.stage, 0)
    launch!(arch, grid, :xyz, _recover_full_state!,
            model.dynamics.dry_density, ρᵡ,
            model.momentum,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.pressure_projection.content,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            substepper.momentum_perturbation.w,
            model.dynamics.dry_density,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            ρᵡ,
            grid, model.dynamics, rounding, key)

    # Thread clock + model fields so time-dependent Open BCs on the recovered
    # prognostic state dispatch correctly in `getbc` (see #717).
    fill_halo_regions!((model.dynamics.dry_density, ρᵡ), boundary_condition_args(model)...)
    fill_halo_regions!(model.momentum, boundary_condition_args(model)...)
    AtmosphereModels.compute_velocities!(model)

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Compute into `C` the stage-entry thermodynamic flux divergence in the discretization of the
substep loop, ``C = -∇·(θᴸ (ρu)ᴸ)`` with ``θᴸ`` the substepper's cached linearization potential
temperature (so [`prepare_acoustic_cache!`](@ref) must have run for this stage). This is the
part of the thermodynamic tendency that the acoustic loop integrates as fast: its perturbation
about the stage-entry state is ``-∇·(θᴸ (ρu)′)``. Multirate steppers subtract it from the full
tendency to isolate the slow remainder.
"""
function compute_linearized_thermodynamic_flux_divergence!(C, substepper::AcousticSubstepper, model)
    grid = model.grid
    launch!(architecture(grid), grid, :xyz, _linearized_thermodynamic_flux_divergence!,
            C, grid, model.dynamics, substepper.linearization_potential_temperature,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw)
    return nothing
end

@kernel function _linearized_thermodynamic_flux_divergence!(C, grid, dynamics, θᴸ, ρu, ρv, ρw)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        V⁻¹ = V⁻¹ᶜᶜᶜ(i, j, k, grid)
        ∇ʰ_θM = V⁻¹ * (δxᶜᵃᵃ(i, j, k, grid, θFˣ, θᴸ, ρu) + δyᵃᶜᵃ(i, j, k, grid, θFʸ, θᴸ, ρv))
        C[i, j, k] = - ∇ʰ_θM - ∂zᶜᶜᶜ(i, j, k, grid, θFᶻ, θᴸ, dynamics, ρu, ρv, ρw)
    end
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
