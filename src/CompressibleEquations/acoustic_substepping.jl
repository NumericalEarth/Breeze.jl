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
##### Time discretization: horizontal momentum is forward-Euler with MPAS first-small-step sequencing
##### (first substep applies frozen ∇pᴸ but skips the perturbation horizontal PGF; it enters on later
##### substeps). Vertical ((ρw)′,(ρθ)′,ρ′) coupling is off-centered Crank-Nicolson (`forward_weight` ω:
##### 0.5 = centered, >0.5 = dissipative), reducing to a tridiagonal Schur solve for (ρw)′ at z-faces.
##### Each stage then recovers ρ=ρᴸ+ρ′, ρθ=ρθᴸ+(ρθ)′, … and diagnoses velocities.
#####
##### Kernel args ρ′, ρθ′, ρu′, ρv′, ρw′ map to struct fields density_perturbation,
##### density_potential_temperature_perturbation, momentum_perturbation.{u,v,w}; predictors carry ★
##### (ρ′★, ρθ′★). (σ, η are reserved for vertical coordinates, so primes are used throughout.)
##### Public derivation: docs/src/compressible_dynamics.md.
##### Refs: Wicker & Skamarock 2002 (MWR 130, 2088); Klemp et al. 2018 (MWR 146, 1911).
#####

using KernelAbstractions: @kernel, @index

using Oceananigans: CenterField, XFaceField, YFaceField, ZFaceField, architecture
using Oceananigans.Models: boundary_condition_args
using Oceananigans.Grids: ZDirection, rnode, znode
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
using Oceananigans.BoundaryConditions: fill_halo_regions!, BoundaryCondition, NormalFlow

using Oceananigans.Grids: Flat, Center, Face, peripheral_node,
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
- `vertical_solver`: `BatchedTridiagonalSolver` for the implicit (ρw)′ update.
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
    # Open-boundary relaxation factor α for `ρ′,(ρθ)′` (issue #738).
    open_boundary_relaxation :: FT

    # Linearization basic state Πᴸ, θᴸ, derived per stage from live model fields.
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
                       adapt(to, a.vertical_solver))

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
                            prognostic_momentum = nothing, substep_floattype = eltype(grid))
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

    # Seed the time-averaged velocity field with the outer-step-start velocities.
    grid = model.grid
    arch = architecture(grid)
    avg = map(parent, substepper.time_averaged_velocities)
    src = map(parent, velocities)
    sz = max.(map(size, avg)...)
    launch!(arch, grid, KernelParameters(1:sz[1], 1:sz[2], 1:sz[3]),
            _seed_time_averaged_velocity!, avg, src)

    return nothing
end

# Copy the three velocity components (full parent arrays, halos included) in one launch.
@kernel function _seed_time_averaged_velocity!(avg, src)
    i, j, k = @index(Global, NTuple)
    checkbounds(Bool, avg.u, i, j, k) && @inbounds (avg.u[i, j, k] = src.u[i, j, k])
    checkbounds(Bool, avg.v, i, j, k) && @inbounds (avg.v[i, j, k] = src.v[i, j, k])
    checkbounds(Bool, avg.w, i, j, k) && @inbounds (avg.w[i, j, k] = src.w[i, j, k])
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
    κ    = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
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
            constants)

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
@kernel function _compute_linearization_mixture_eos!(γRᵐ, grid, microphysics, ρ, qᵛ, μ, constants)
    i, j, k = @index(Global, NTuple)
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
\\left\\lceil \\frac{|\\Delta t| \\, \\mathbb{C}^{ac}}{\\nu \\, \\Delta x_\\min} \\right\\rceil ,
```

with ``\\mathbb{C}^{ac} = \\sqrt{γ^d R^d T_r}`` for a nominal reference
temperature ``T_r = 300\\,\\mathrm{K}`` and ``ν`` the target acoustic
Courant number `acoustic_cfl` (default `0.5`, the ERF/WRF target —
equivalent to the conventional safety factor of `2`).
"""
function compute_acoustic_substeps(grid, Δt, thermodynamic_constants, acoustic_cfl)
    FT   = eltype(grid)
    Rᵈ   = dry_air_gas_constant(thermodynamic_constants)
    cᵖᵈ  = thermodynamic_constants.dry_air.heat_capacity
    γᵈ   = cᵖᵈ / (cᵖᵈ - Rᵈ)
    ℂᵃᶜ  = sqrt(γᵈ * Rᵈ * FT(300))

    Δx_min = let
        TX, TY, _ = topology(grid)
        Δx = TX === Flat ? typemax(FT) : minimum_xspacing(grid)
        Δy = TY === Flat ? typemax(FT) : minimum_yspacing(grid)
        min(Δx, Δy)
    end

    return max(1, ceil(Int, abs(FT(Δt)) * ℂᵃᶜ / (acoustic_cfl * Δx_min)))
end

@inline acoustic_substeps(N::Int, grid, Δt, constants, acoustic_cfl) = N
@inline acoustic_substeps(::Nothing, grid, Δt, constants, acoustic_cfl) =
    compute_acoustic_substeps(grid, Δt, constants, acoustic_cfl)

#####
##### Section 5 — Stage substep distribution
#####

# Substeps for a stage covering Δt_stage = β·Δt: ⌈β·N⌉ for a fixed full-step count N,
# or the acoustic-CFL count for the stage interval when adaptive (substeps = nothing).
@inline _stage_substeps(N::Integer, β, Δt_stage, grid, constants, acoustic_cfl) = max(1, ceil(Int, β * N))
@inline _stage_substeps(::Nothing, β, Δt_stage, grid, constants, acoustic_cfl) =
    compute_acoustic_substeps(grid, Δt_stage, constants, acoustic_cfl)

# Uniform full-step count N rounded up to a multiple of 6 so β·N is integral (ConstantSubstepSize,
# MonolithicFirstStage stages 2–3).
@inline function _uniform_substep_count(Δt, grid, constants, acoustic_cfl, substeps)
    N_raw = acoustic_substeps(substeps, grid, Δt, constants, acoustic_cfl)
    return max(6, 6 * cld(N_raw, 6))
end

# ProportionalSubsteps: each stage covers its own interval β·Δt with ⌈β·N⌉ substeps sized to tile
# it exactly (Δτ = β·Δt/Nτ). Exact coverage at the minimum count; Δτ may differ slightly by stage.
@inline function stage_substep_count_and_size(::ProportionalSubsteps, β_stage, Δt, grid, constants, acoustic_cfl, substeps)
    Δt_stage = β_stage * Δt
    Nτ = _stage_substeps(substeps, β_stage, Δt_stage, grid, constants, acoustic_cfl)
    return Nτ, Δt_stage / Nτ
end

# ConstantSubstepSize: one substep size Δτ = Δt/N shared by all stages (N a multiple of 6 ⇒ β·N integral).
@inline function stage_substep_count_and_size(::ConstantSubstepSize, β_stage, Δt, grid, constants, acoustic_cfl, substeps)
    N = _uniform_substep_count(Δt, grid, constants, acoustic_cfl, substeps)
    return max(1, round(Int, β_stage * N)), Δt / N
end

# MonolithicFirstStage: stage 1 collapses to one substep of size Δt/3; stages 2–3 like ConstantSubstepSize.
@inline function stage_substep_count_and_size(::MonolithicFirstStage, β_stage, Δt, grid, constants, acoustic_cfl, substeps)
    β_stage < (1//3 + 1//2) / 2 && return 1, Δt / 3
    N = _uniform_substep_count(Δt, grid, constants, acoustic_cfl, substeps)
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

# Off-centered CN tridiag derivation
# ----------------------------------
# At face k the (ρw)′ CN update is
#   (ρw)′ₙ(k) = (ρw)′ₒ(k) + Δτ Gˢρw(k) − Δτ(ωˢ⁻ ∂z p′ₒ + ωᵐ⁺ ∂z p′ₙ) − Δτ g(ωˢ⁻ ρ′_faceₒ + ωᵐ⁺ ρ′_faceₙ),
# with ωᵐ⁺=(1+ε)/2, ωˢ⁻=(1−ε)/2 (ε=0 centered). p′ = Cᴸ(ρθ)′, Cᴸ≡γRᵐᴸΠᴸ, so the discrete PGF is the
# gradient of the product Cᴸ·(ρθ)′ (not Cᴸ_face·∂z(ρθ)′). Post-solve substitution (δτᵐ⁺=ωᵐ⁺Δτ):
#   ρ′ₙ(k)    = ρ′★(k)  − δτᵐ⁺((ρw)′ₙ(k+1) − (ρw)′ₙ(k))/Δz_c(k)
#   (ρθ)′ₙ(k) = ρθ′★(k) − δτᵐ⁺(θᴸ_face(k+1)(ρw)′ₙ(k+1) − θᴸ_face(k)(ρw)′ₙ(k))/Δz_c(k)
# yields the tridiag coefficients (ω≡ωᵐ⁺):
#   A[k,k+1] = −(ωΔτ)² Cᴸ(k)  θᴸ_face(k+1) rdz_c(k)   /Δzᶠ(k) − (ωΔτ)² g rdz_c(k)/2
#   A[k,k]   = 1 + (ωΔτ)² θᴸ_face(k)(Cᴸ(k)rdz_c(k)+Cᴸ(k−1)rdz_c(k−1))/Δzᶠ(k) + (ωΔτ)² g(rdz_c(k)−rdz_c(k−1))/2
#   A[k,k−1] = −(ωΔτ)² Cᴸ(k−1)θᴸ_face(k−1)rdz_c(k−1)/Δzᶠ(k) + (ωΔτ)² g rdz_c(k−1)/2
# γᵐRᵐᴸ (cell-centered γᵐRᵐ cached in `linearization_gamma_R_mixture`, refreshed per stage, interpolated to
# faces in-kernel) collapses bit-identically to dry γᵈRᵈ for qᵛ=qˡ=qⁱ=0.
#
# Implicit vertical damping: for `ThermalDivergenceDamping(damp_vertical=true)`, the vertical divergence
# damping folds into the same tridiag via a discrete vertical Laplacian on (ρw)′:
#   (ρw)′ₙ − ωαΔz² ∂z²(ρw)′ₙ = (ρw)′ₒ + (1−ω)αΔz² ∂z²(ρw)′ₒ.
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
    return abs(δτᵐ⁺) * sponge.damping_rate *
           sponge.ramp(z, grid.Lz, sponge.depth)
end

@inline sponge_rhs(i, j, k, grid, ::Nothing, δτˢ⁻, ρw_old) = zero(grid)

@inline function sponge_rhs(i, j, k, grid, sponge::UpperSponge, δτˢ⁻, ρw_old)
    z = rnode(i, j, k, grid, Center(), Center(), Face())
    @inbounds return abs(δτˢ⁻) * sponge.damping_rate * sponge.ramp(z, grid.Lz, sponge.depth) * ρw_old[i, j, k]
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagLower, p, ::ZDirection,
                                 Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
    kᶠ      = k + 1
    Δzᶠ     = Δzᶜᶜᶠ(i, j, kᶠ, grid)
    Δz⁻¹ᵏ⁻  = 1 / Δzᶜᶜᶜ(i, j, kᶠ - 1, grid)

    @inbounds Cᵏ⁻ = γRᵐᴸ[i, j, kᶠ - 1] * Πᴸ[i, j, kᶠ - 1]
    θᵏ⁻     = ℑbzᵃᵃᶠ(i, j, kᶠ - 1, grid, θᴸ)

    pgf_term  = - δτᵐ⁺^2 * Cᵏ⁻ * θᵏ⁻ * Δz⁻¹ᵏ⁻ / Δzᶠ
    buoy_term = + δτᵐ⁺^2 * g * Δz⁻¹ᵏ⁻ / 2
    damp_term = - dᵐ⁺ * Δz⁻¹ᵏ⁻ / Δzᶠ
    # Upper sponge is local in z (Rayleigh-type), so no off-diagonal coupling.
    return pgf_term + buoy_term + damp_term
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagDiagonal, p, ::ZDirection,
                                 Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)

    Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
    Δz⁻¹ᵏ⁺ = 1 / Δzᶜᶜᶜ(i, j, k,     grid)
    Δz⁻¹ᵏ⁻ = 1 / Δzᶜᶜᶜ(i, j, k - 1, grid)

    @inbounds begin
        Cᵏ⁺ = γRᵐᴸ[i, j, k]     * Πᴸ[i, j, k]
        Cᵏ⁻ = γRᵐᴸ[i, j, k - 1] * Πᴸ[i, j, k - 1]
    end
    
    θᶜᶜᶠ = ℑbzᵃᵃᶠ(i, j, k, grid, θᴸ)

    pgf_diag   = δτᵐ⁺^2 * θᶜᶜᶠ * (Cᵏ⁺ * Δz⁻¹ᵏ⁺ + Cᵏ⁻ * Δz⁻¹ᵏ⁻) / Δzᶠ
    buoy_diag  = δτᵐ⁺^2 * g * (Δz⁻¹ᵏ⁺ - Δz⁻¹ᵏ⁻) / 2
    damp_diag  = dᵐ⁺ * (Δz⁻¹ᵏ⁺ + Δz⁻¹ᵏ⁻) / Δzᶠ
    spnge_diag = sponge_term_diag(i, j, k, grid, sponge, δτᵐ⁺)

    return one(grid) + (pgf_diag + buoy_diag + damp_diag + spnge_diag) * (k > 1)
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagUpper, p, ::ZDirection,
                                 Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)

    Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
    Δz⁻¹ᵏ⁺  = 1 / Δzᶜᶜᶜ(i, j, k, grid)

    @inbounds Cᵏ⁺ = γRᵐᴸ[i, j, k] * Πᴸ[i, j, k]
    θᵏ⁺ = ℑbzᵃᵃᶠ(i, j, k + 1, grid, θᴸ)

    pgf_term  = - δτᵐ⁺^2 * Cᵏ⁺ * θᵏ⁺ * Δz⁻¹ᵏ⁺ / Δzᶠ
    buoy_term = - δτᵐ⁺^2 * g * Δz⁻¹ᵏ⁺ / 2
    damp_term = - dᵐ⁺ * Δz⁻¹ᵏ⁺ / Δzᶠ
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
function initialize_stage_perturbations!(substepper, model, Uᴸ_outer)
    grid = model.grid
    arch = architecture(grid)

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
    # ρθ is held in `model.dynamics.density` and the formulation's ρθ
    # field — untouched by the substep loop, so they equal the per-stage
    # linearization base.
    χ_field = thermodynamic_density(model.formulation)
    χ_name = thermodynamic_density_name(model.formulation)
    launch!(arch, grid, :xyz, _initialize_stage_perturbations!,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            Uᴸ_outer.ρ, Uᴸ_outer[χ_name], Uᴸ_outer.ρu, Uᴸ_outer.ρv,
            model.dynamics.density, χ_field, model.momentum.ρu, model.momentum.ρv)
    # ρw is dispatched: terrain models initialize the contravariant ρw̃ instead.
    initialize_vertical_momentum_perturbation!(substepper, model, Uᴸ_outer)

    fill_halo_regions!(substepper.density_perturbation)
    fill_halo_regions!(substepper.density_potential_temperature_perturbation)
    map(fill_halo_regions!, substepper.momentum_perturbation)

    return nothing
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
#   ∂x p_full = ∂x pᴸ + ∂x(Cᴸ (ρθ)′),  Cᴸ = γRᵐᴸ Πᴸ
# the first piece frozen at the linearization point, the second the
# perturbation force. `ExnerReferenceState` depends only on z, so ∂x pᵣ ≡ 0
# and no horizontal pressure-perturbation field is needed.
#   (ρu)′^{τ+Δτ} = (ρu)′^τ + Δτ (Gⁿρu − ∂x pᴸ − ∂x(Cᴸ (ρθ)′))
#   (ρv)′^{τ+Δτ} = (ρv)′^τ + Δτ (Gⁿρv − ∂y pᴸ − ∂y(Cᴸ (ρθ)′))
# `Gⁿρu` (SlowTendencyMode) carries non-pressure slow terms with PGF zeroed;
# we reinstate the frozen large-step PGF here (MPAS keeps it in `tend_u_euler`).
# Forward-backward sequencing skips only the acoustic perturbation PGF.
@kernel function _explicit_horizontal_step!(ρu′, ρv′, grid, dynamics, Δτ, ρθ′, Πᴸ,
                                            Gⁿρu, Gⁿρv, γRᵐᴸ, apply_pressure_gradient)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ∂x_pᴸ  = AtmosphereModels.x_pressure_gradient(i, j, k, grid, dynamics)
        ∂x_p′  = ∇ˣp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)
        ∂y_pᴸ  = AtmosphereModels.y_pressure_gradient(i, j, k, grid, dynamics)
        ∂y_p′  = ∇ʸp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)

        perturbation_pressure_gradient_factor = ifelse(apply_pressure_gradient, one(Δτ), zero(Δτ))
        ∂x_p = ∂x_pᴸ + perturbation_pressure_gradient_factor * ∂x_p′
        ∂y_p = ∂y_pᴸ + perturbation_pressure_gradient_factor * ∂y_p′

        ρu′[i, j, k] += Δτ * (Gⁿρu[i, j, k] - ∂x_p)
        ρv′[i, j, k] += Δτ * (Gⁿρv[i, j, k] - ∂y_p)
    end
end

@inline δpᴸ(i, j, k, grid, ρθ′, Πᴸ, γRᵐᴸ) = @inbounds γRᵐᴸ[i, j, k] * Πᴸ[i, j, k] * ρθ′[i, j, k]

# `slope_correction` gates the terrain horizontal slope correction (see the
# `TerrainCompressibleDynamics` method in `terrain_compressible_physics.jl`).
# On a flat grid there is no horizontal correction, so the factor is ignored here.
@inline ∇ˣp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ) = ∂xᶠᶜᶜ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
@inline ∇ʸp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ) = ∂yᶜᶠᶜ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
@inline ∇ᶻp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ, slope_correction) = ∂zᶜᶜᶠ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)

@inline apply_horizontal_pressure_gradient_substep(substep, Nτ, apply_first_substep_pressure_gradient) =
    apply_first_substep_pressure_gradient | (substep != 1) | (Nτ == 1)

@inline apply_horizontal_pressure_gradient_substep(substep, Nτ) =
    apply_horizontal_pressure_gradient_substep(substep, Nτ, false)

# Build per-column predictors `ρ′★`, `ρθ′★` (cell centers) AND
# the explicit RHS for the tridiagonal `(ρw)′ᵐ⁺` solve at z-faces.
#
# Off-centered Crank–Nicolson with new-side weight ω = forward_weight
# and old-side weight 1−ω. The predictor uses δτˢ⁻ = (1−ω)Δτ on the
# old-step vertical-flux contribution (ω-weighted CN of ∇·m); the
# vertical RHS combines old and pred contributions with their matching
# weights δτˢ⁻ and δτᵐ⁺ respectively. See derivation in
# the split-explicit derivation in `docs/src/compressible_dynamics.md`.
# Build the cell-centred predictors ρ′★, ρθ′★ in one 3D kernel, then the face-level
# tridiag RHS for (ρw)′ᵐ⁺ in a second 3D kernel that reads them. Split (vs one column
# kernel) so both run `:xyz` for full occupancy; the kernel-launch boundary is the
# global sync that lets the RHS read predictor values at k±1. The predictor kernel also
# stashes the old (ρθ)′ into ρθ′ˢ⁻ (for the divergence damping), folding in what was a
# separate full-field copy. (Flat grids only read predictors vertically here; terrain's
# horizontal slope read would need a ρθ′★ halo fill between the two kernels — as the
# former single column kernel already required across columns.)
@kernel function _build_predictors!(ρ′★, ρθ′★, ρθ′ˢ⁻,
                                    ρ′, ρθ′, ρw′, ρu′, ρv′,
                                    grid, dynamics, Δτ, δτˢ⁻, Gˢρ, Gˢρθ, fθ, θᴸ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρθ′ˢ⁻[i, j, k] = ρθ′[i, j, k]   # stash old (ρθ)′ for the divergence damping

        V⁻¹ = 1 / Vᶜᶜᶜ(i, j, k, grid)
        ∇ʰ_M  = div_xyᶜᶜᶜ(i, j, k, grid, ρu′, ρv′)
        ∇ʰ_θM = (δxᶜᵃᵃ(i, j, k, grid, θFˣ, θᴸ, ρu′) +
                 δyᵃᶜᵃ(i, j, k, grid, θFʸ, θᴸ, ρv′)) * V⁻¹

        ρ′★[i, j, k]  = ρ′[i, j, k] + Δτ * (Gˢρ[i, j, k] - ∇ʰ_M) -
                            δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, Fʷ, dynamics, ρu′, ρv′, ρw′)
        ρθ′★[i, j, k] = ρθ′[i, j, k] + Δτ * (fθ * Gˢρθ[i, j, k] - ∇ʰ_θM) -
                            δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, θFᶻ, θᴸ, dynamics, ρu′, ρv′, ρw′)
    end
end

# Face-level RHS for `(ρw)′ᵐ⁺`: split weights δτᵐ⁺ (predictor) and δτˢ⁻ (old-step) per
# derivation (15). `dˢ⁻ = (1−ω) α Δz²` is the explicit half of the implicit vertical
# damping (0 when damping off). Boundary rows: f[1] = 0 (matches b[1] = 1 ⇒ (ρw)′[1] = 0);
# top face Nz+1 lives outside the solver (impenetrability w(top) = 0).
@kernel function _build_vertical_rhs!(ρw′_rhs, ρ′★, ρθ′★, ρ′, ρθ′, ρw′,
                                      grid, dynamics, Δτ, δτᵐ⁺, δτˢ⁻, Πᴸ, γRᵐᴸ, g, dˢ⁻,
                                      fw, Gˢρw, sponge, apply_pressure_gradient)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    # Gate the terrain horizontal slope correction in lockstep with the MPAS
    # first-small-step gate (no effect on a flat grid; always applied on terrain).
    slope_correction = ifelse(apply_pressure_gradient, one(Δτ), zero(Δτ))

    @inbounds begin
        ∂r_p′★  = ∇ᶻp′(i, j, k, grid, dynamics, ρθ′★, Πᴸ, γRᵐᴸ, slope_correction)
        ∂r_p′ˢ⁻ = ∇ᶻp′(i, j, k, grid, dynamics, ρθ′,  Πᴸ, γRᵐᴸ, slope_correction)
        sound_force = δτˢ⁻ * ∂r_p′ˢ⁻ + δτᵐ⁺ * ∂r_p′★

        ρ′ᶜᶜᶠ★  = ℑzᵃᵃᶠ(i, j, k, grid, ρ′★)
        ρ′ᶜᶜᶠˢ⁻ = ℑzᵃᵃᶠ(i, j, k, grid, ρ′)
        buoy_force = g * (δτˢ⁻ * ρ′ᶜᶜᶠˢ⁻ + δτᵐ⁺ * ρ′ᶜᶜᶠ★)

        ∂z²_ρw′ˢ⁻  = ∂zᶜᶜᶠ(i, j, k, grid, ∂zᶜᶜᶜ, ρw′)
        damp_force = - dˢ⁻ * ∂z²_ρw′ˢ⁻

        sponge_force = sponge_rhs(i, j, k, grid, sponge, δτˢ⁻, ρw′)

        rhs = ρw′[i, j, k] + Δτ * fw * Gˢρw[i, j, k] -
              sound_force - buoy_force - damp_force - sponge_force

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
# (the IMPLICIT half of CN).
#
#   ρ′_n(k)    = ρ′★(k)  - (δτᵐ⁺ / Δz_c(k)) · ((ρw)′_n(k+1) - (ρw)′_n(k))
#   (ρθ)′_n(k) = ρθ′★(k) - (δτᵐ⁺ / Δz_c(k)) · (θᴸ_face(k+1) (ρw)′_n(k+1)
#                                                    - θᴸ_face(k)   (ρw)′_n(k))
# Recovers ρ′, (ρθ)′ from the solved (ρw)′, and folds in the time-averaged-velocity
# accumulation (Step F) since the momentum components are already loaded here. NOTE: this
# runs *before* the divergence damping, so `avg` accumulates the pre-damping (ρu)′,(ρv)′.
# Identical for dry runs (the transport velocity is unused without scalars); for moist
# runs it omits the per-substep damping increment from the transport average.
@kernel function _post_solve_recovery!(ρ′, ρθ′, ρw′, ρu′, ρv′, ρ′★, ρθ′★, avg, grid, dynamics, δτᵐ⁺, θᴸ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρ′[i, j, k]  = ρ′★[i, j, k]  - δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, Fʷ, dynamics, ρu′, ρv′, ρw′)
        ρθ′[i, j, k] = ρθ′★[i, j, k] - δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, θFᶻ, θᴸ, dynamics, ρu′, ρv′, ρw′)
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
    x_damping_scale = TX === Flat ? NoHorizontalDampingScale() : horizontal_damping_scale(damping, α, Δτ_FT)
    y_damping_scale = TY === Flat ? NoHorizontalDampingScale() : horizontal_damping_scale(damping, α, Δτ_FT)

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
    α_over_Δτ :: FT
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

@inline κˣ(i, j, k, grid, ::NoHorizontalDampingScale) = zero(grid)
@inline κʸ(i, j, k, grid, ::NoHorizontalDampingScale) = zero(grid)

@inline κˣ(i, j, k, grid, scale::FixedHorizontalDampingScale) = scale.diffusivity
@inline κʸ(i, j, k, grid, scale::FixedHorizontalDampingScale) = scale.diffusivity

@inline κˣ(i, j, k, grid, scale::LocalHorizontalDampingScale) = scale.α_over_Δτ * Δxᶠᶜᶜ(i, j, k, grid)^2
@inline κʸ(i, j, k, grid, scale::LocalHorizontalDampingScale) = scale.α_over_Δτ * Δyᶜᶠᶜ(i, j, k, grid)^2


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
        γˣ = κˣ(i, j, k, grid, x_damping_scale)
        ρu′[i, j, k] -= γˣ * ∂x_div / θᴸᶠᶜᶜ

        ∂y_div = ∂yᶜᶠᶜ(i, j, k, grid, dρθ′, ρθ′, ρθ′ˢ⁻)
        θᴸᶜᶠᶜ  = ℑyᵃᶠᵃ(i, j, k, grid, θᴸ)
        γʸ = κʸ(i, j, k, grid, y_damping_scale)
        ρv′[i, j, k] -= γʸ * ∂y_div / θᴸᶜᶠᶜ
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
##### `model.momentum.*` and `model.dynamics.density` are still the stage-entry
##### (Uᴸ_stage) values here. Dividing by ρᴸ ignores ρ's variation over the
##### loop, small for acoustic perturbations.
#####


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

    map(fill_halo_regions!, substepper.time_averaged_velocities)

    return nothing
end

@kernel function _finalize_time_averaged_velocity!(u_avg, v_avg, w_avg,
                                                   ρu_stage, ρv_stage, ρw_stage,
                                                   ρᴸ, grid, dynamics, inv_Nτ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρu_total = ρu_stage[i, j, k] + u_avg[i, j, k] * inv_Nτ
        ρv_total = ρv_stage[i, j, k] + v_avg[i, j, k] * inv_Nτ
        ρw_total = transport_ρw(i, j, k, grid, dynamics,
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

@inline transport_ρw(i, j, k, grid, dynamics, ρu_stage, ρv_stage, ρw_stage) =
    @inbounds ρw_stage[i, j, k]

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
        ρwᵐ⁺ = acoustic_recovered_vertical_momentum(i, j, k, grid, dynamics, ρuᴸ, ρvᴸ, ρwᴸ, ρu′, ρv′, ρw′)

        ρ[i, j, k]  = ρᵐ⁺
        ρθ[i, j, k] = ρθᵐ⁺

        m.ρu[i, j, k] = ρuᵐ⁺
        m.ρv[i, j, k] = ρvᵐ⁺
        m.ρw[i, j, k] = ρwᵐ⁺
    end
end

@inline acoustic_recovered_vertical_momentum(i, j, k, grid, dynamics, ρuᴸ, ρvᴸ, ρwᴸ, ρu′, ρv′, ρw′) =
    @inbounds ρwᴸ[i, j, k] + ρw′[i, j, k]

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
    ρᴸ  = model.dynamics.density
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

"""
$(TYPEDSIGNATURES)

Execute one Wicker–Skamarock RK3 stage of the linearized acoustic
substep loop. Number and size of substeps in this stage depend on
`substepper.substep_distribution`.
"""
function acoustic_rk3_substep_loop!(model::AtmosphereModel, substepper, Δt, β_stage, Uᴸ)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    constants = model.thermodynamic_constants
    g = convert(FT, constants.gravitational_acceleration)

    # Substep count Nτ and size Δτ for this stage (WS-RK3 weights β = (1/3, 1/2, 1)).
    # The distribution decides how to split: ProportionalSubsteps fits ⌈β·N⌉ substeps to
    # each stage's β·Δt interval; ConstantSubstepSize/MonolithicFirstStage share one Δτ = Δt/N.
    Δt_FT = FT(Δt)
    Nτ, Δτ = stage_substep_count_and_size(substepper.substep_distribution, β_stage, Δt_FT,
                                          grid, constants, substepper.acoustic_cfl, substepper.substeps)

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
        apply_pressure_gradient = apply_horizontal_pressure_gradient_substep(substep, Nτ,
            substepper.apply_first_substep_pressure_gradient)

        launch!(arch, grid, :xyz, _explicit_horizontal_step!,
                substepper.momentum_perturbation.u,
                substepper.momentum_perturbation.v,
                grid, model.dynamics, Δτ,
                substepper.density_potential_temperature_perturbation,
                substepper.linearization_exner,
                Gⁿ.ρu, Gⁿ.ρv, substepper.linearization_gamma_R_mixture,
                apply_pressure_gradient)

        fill_halo_regions!(substepper.momentum_perturbation.u)
        fill_halo_regions!(substepper.momentum_perturbation.v)

        # (old (ρθ)′ is stashed into ρθ′ˢ⁻ inside `_build_predictors!`, then halo-filled.)

        # Implicit-vertical-damping prefactors. When the damping strategy
        # is `ThermalDivergenceDamping(damp_vertical=true)`, the
        # vertical part of the divergence damping is folded into the
        # tridiag with `dᵐ⁺ = ω·α·Δz²` on the LHS and
        # `dˢ⁻ = (1−ω)·α·Δz²` on the predictor RHS. Both reduce to
        # zero for `NoDivergenceDamping` or when the user opts out via
        # `damp_vertical=false`.
        dᵐ⁺, dˢ⁻ = implicit_damping_factors(substepper.damping, ω, grid, FT)

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
                Gⁿ.ρ, Gˢρθ, substepper.thermodynamic_tendency_factor,
                substepper.linearization_potential_temperature)
        fill_halo_regions!(substepper.previous_density_potential_temperature_perturbation)

        launch!(arch, grid, KernelParameters(1:size(grid, 1), 1:size(grid, 2), 1:size(grid, 3) + 1),
                _build_vertical_rhs!,
                substepper.momentum_perturbation.w,
                substepper.density_predictor,
                substepper.density_potential_temperature_predictor,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation.w,
                grid, model.dynamics, Δτ, δτᵐ⁺, δτˢ⁻,
                substepper.linearization_exner, substepper.linearization_gamma_R_mixture,
                g, dˢ⁻, substepper.vertical_momentum_tendency_factor,
                substepper.slow_vertical_momentum_tendency,
                substepper.sponge, apply_pressure_gradient)

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
                substepper.time_averaged_velocities,
                grid, model.dynamics, δτᵐ⁺,
                substepper.linearization_potential_temperature)

        # Per-substep open-boundary enforcement (issue #738): relax the outermost
        # open-boundary cell of ρ′, (ρθ)′ toward the prescribed wall value, before
        # the halo fill, so the boundary cell tracks the prescribed inflow state.
        apply_open_boundary_relaxation!(substepper, model, grid, arch)

        fill_halo_regions!(substepper.density_perturbation)
        fill_halo_regions!(substepper.density_potential_temperature_perturbation)

        # Step E: optional Klemp 2018 post-substep damping (no-op for
        # `NoDivergenceDamping`).
        apply_divergence_damping!(substepper.damping, substepper, grid, Δτ, constants)

        fill_halo_regions!(substepper.momentum_perturbation.u)
        fill_halo_regions!(substepper.momentum_perturbation.v)
        # (time-averaged velocity accumulation is fused into `_post_solve_recovery!` above)
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
            model.dynamics.density, χ_field, model.momentum,
            substepper.density_perturbation,
            substepper.density_potential_temperature_perturbation,
            substepper.momentum_perturbation.u,
            substepper.momentum_perturbation.v,
            substepper.momentum_perturbation.w,
            model.dynamics.density,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            χ_field, grid, model.dynamics)

    # Thread clock + model fields so time-dependent Open BCs on the recovered
    # prognostic state dispatch correctly in `getbc` (see #717).
    fill_halo_regions!(model.dynamics.density, boundary_condition_args(model)...)
    fill_halo_regions!(χ_field, boundary_condition_args(model)...)
    fill_halo_regions!(model.momentum, boundary_condition_args(model)...)
    AtmosphereModels.compute_velocities!(model)

    return nothing
end
