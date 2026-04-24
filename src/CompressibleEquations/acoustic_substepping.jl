#####
##### Acoustic Substepping for CompressibleDynamics
#####
##### MPAS-A conservative-perturbation split-explicit acoustic substepper
##### (Skamarock et al. 2012; Wicker–Skamarock 2002 RK3 outer loop).
#####
##### Active path:
#####   - Fast prognostics:  (ρu)″, (ρv)″, (ρw)″, ρ″, (ρθ)″   (MPAS ru_p, rv_p, rw_p,
#####                                                          rho_pp, rtheta_pp)
#####   - Outer scheme:      Wicker–Skamarock RK3 with β = (1/3, 1/2, 1)
#####   - Substep schedule:  selectable via AcousticSubstepDistribution
#####                        (ProportionalSubsteps default; MonolithicFirstStage for
#####                        bit-compatible MPAS `config_time_integration_order = 3`)
#####   - Vertical solve:    Schur-complement tridiagonal in (ρw)″, (ρθ)″ with
#####                        forward-weight off-centering ε = 2ω - 1
#####   - Divergence damping: selectable via the AcousticDampingStrategy interface.
#####                         Default PressureProjectionDamping(coefficient=0.5)
#####                         is the literal ERF/CM1/WRF projection form, tuned on
#####                         the DCMIP2016 baroclinic wave. Other strategies:
#####                         ThermodynamicDivergenceDamping (MPAS Klemp 2018),
#####                         ConservativeProjectionDamping (algebraic projection),
#####                         NoDivergenceDamping.
#####   - Boundary semantics: the substep loop uses *both* topology-safe
#####                         interpolation/difference operators (which return
#####                         zero on Flat axes and skip out-of-bounds faces) and
#####                         per-substep `fill_halo_regions!` on u, v, ρu″, ρv″,
#####                         ρθ″, ρ″ to enforce physical BCs and exchange MPI
#####                         halos.
#####
##### File layout (top-down):
#####   1.  Topology-safe interpolation and difference operators
#####   2.  AcousticSubstepper struct + constructor + Adapt
#####   3.  Damping strategy materialization + Adapt
#####   4.  Adaptive substep computation
#####   5.  Stage cache preparation (frozen θ_v, π₀, frozen pressure)
#####   6.  Slow tendency conversion ρu/ρv/ρw → u/v/w
#####   7.  MPAS horizontal forward kernel
#####   8.  MPAS divergence damping kernel + apply_pgf_filter! / apply_divergence_damping!
#####       strategy dispatch
#####   9.  Tridiagonal coefficient helpers (cofwz, cofwr, cofwt, coftz)
#####   10. Column kernel (_build_acoustic_rhs!)
#####   11. Tridiagonal solver coefficient tag types
#####   12. Post-solve diagnostics (recover ρ″, ρθ″ from new ρw″)
#####   13. WS-RK3 stage substep loop driver
#####   14. Recovery (ρ_new, w_new, momentum)
#####

using KernelAbstractions: @kernel, @index

using Oceananigans: CenterField, XFaceField, YFaceField, ZFaceField, architecture
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!
using Oceananigans.Operators:
    ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ,
    ℑxᶠᵃᵃ, ℑxᶜᵃᵃ, ℑyᵃᶠᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ,
    δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ,
    δzᵃᵃᶜ, δzᵃᵃᶠ,
    divᶜᶜᶜ, div_xyᶜᶜᶜ,
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
##### Section 1: Topology-aware interpolation and difference operators
#####
##### These avoid halo access for frozen fields during acoustic substeps.
#####

# Fallback: use standard interpolation
@inline ℑxTᶠᵃᵃ(i, j, k, grid, f::AbstractArray) = ℑxᶠᵃᵃ(i, j, k, grid, f)
@inline ℑyTᵃᶠᵃ(i, j, k, grid, f::AbstractArray) = ℑyᵃᶠᵃ(i, j, k, grid, f)
@inline ℑzTᵃᵃᶠ(i, j, k, grid, f::AbstractArray) = ℑzᵃᵃᶠ(i, j, k, grid, f)
@inline ℑzTᵃᵃᶠ(i, j, k, grid, f, args...)       = ℑzᵃᵃᶠ(i, j, k, grid, f, args...)
@inline ℑzTᵃᵃᶜ(i, j, k, grid, f::AbstractArray) = ℑzᵃᵃᶜ(i, j, k, grid, f)

# Fallback: use standard difference
@inline δzTᵃᵃᶠ(i, j, k, grid, f::AbstractArray) = δzᵃᵃᶠ(i, j, k, grid, f)
@inline δzTᵃᵃᶜ(i, j, k, grid, f::AbstractArray) = δzᵃᵃᶜ(i, j, k, grid, f)

@inline δzTᵃᵃᶠ(i, j, k, grid, f, args...) = δzᵃᵃᶠ(i, j, k, grid, f, args...)
@inline δzTᵃᵃᶜ(i, j, k, grid, f, args...) = δzᵃᵃᶜ(i, j, k, grid, f, args...)

# Periodic: wrap at i=1 / j=1
const PX = AbstractUnderlyingGrid{FT, Periodic} where FT
const PY = AbstractUnderlyingGrid{FT, <:Any, Periodic} where FT

# Bounded horizontal: boundary faces where velocity = 0
const BX = AbstractUnderlyingGrid{FT, Bounded} where FT
const BY = AbstractUnderlyingGrid{FT, <:Any, Bounded} where FT

# For periodic topologies, no boundary faces exist
@inline on_x_boundary(i, j, k, grid) = false
@inline on_y_boundary(i, j, k, grid) = false

# For bounded topologies, face fields have boundary faces on BOTH sides:
# faces at i=1 and i=Nx+1 are the west and east x-boundaries respectively.
# Both must be masked, otherwise the substep updates ru_p at the east face
# as if it were interior and the boundary value drifts away from zero.
@inline on_x_boundary(i, j, k, grid::BX) = (i == 1) | (i == grid.Nx + 1)
@inline on_y_boundary(i, j, k, grid::BY) = (j == 1) | (j == grid.Ny + 1)

@inline function ℑxTᶠᵃᵃ(i, j, k, grid::PX, f::AbstractArray)
    wrapped_ℑx_f = @inbounds (f[1, j, k] + f[grid.Nx, j, k]) / 2
    return ifelse(i == 1, wrapped_ℑx_f, ℑxᶠᵃᵃ(i, j, k, grid, f))
end

@inline function ℑyTᵃᶠᵃ(i, j, k, grid::PY, f::AbstractArray)
    wrapped_ℑy_f = @inbounds (f[i, 1, k] + f[i, grid.Ny, k]) / 2
    return ifelse(j == 1, wrapped_ℑy_f, ℑyᵃᶠᵃ(i, j, k, grid, f))
end

const BZ = AbstractUnderlyingGrid{FT, <:Any, <:Any, Bounded} where FT

@inline function ℑzTᵃᵃᶠ(i, j, k, grid::BZ, f::AbstractArray)
    Nz = size(grid, 3)
    bottom = k == 1
    top = k == Nz + 1
    return @inbounds ifelse(bottom, f[i, j, 1],
                     ifelse(top, f[i, j, Nz],
                            ℑzᵃᵃᶠ(i, j, k, grid, f)))
end

@inline function ℑzTᵃᵃᶠ(i, j, k, grid::BZ, f, args...)
    Nz = size(grid, 3)
    bottom = k == 1
    top = k == Nz + 1
    return ifelse(bottom, f(i, j, 1, grid, args...),
            ifelse(top, f(i, j, Nz, grid, args...),
                   ℑzᵃᵃᶠ(i, j, k, grid, f, args...)))
end

@inline function δzTᵃᵃᶠ(i, j, k, grid::BZ, f::AbstractArray)
    Nz = size(grid, 3)
    bottom = k == 1
    top = k == Nz + 1
    return @inbounds ifelse(bottom, zero(eltype(f)),
                     ifelse(top, zero(eltype(f)),
                            δzᵃᵃᶠ(i, j, k, grid, f)))
end

#####
##### Section 2: AcousticSubstepper struct (MPAS conservative-perturbation form)
#####

"""
    AcousticSubstepper

Storage and parameters for the MPAS-A conservative-perturbation acoustic
substepper. The fast prognostic variables advanced inside the substep loop
are ``(\\rho u)''``, ``(\\rho v)''``, ``(\\rho w)''``, ``\\rho''``, and
``(\\rho\\theta)''`` — the same family used by MPAS-A's
`atm_advance_acoustic_step` and by ERF.

Each substep performs:

1. **Horizontal forward** of ``(\\rho u)''``, ``(\\rho v)''`` from the
   horizontal pressure-gradient force ``-c^2 \\Pi_\\mathrm{face}\\,
   \\partial_x(\\rho\\theta)''`` plus the frozen slow horizontal tendency.
2. **Vertical Schur-complement solve** for ``(\\rho w)''``, ``(\\rho\\theta)''``,
   ``\\rho''`` along each column with forward-weight off-centering
   ``\\varepsilon = 2\\omega - 1``. The tridiagonal coefficients (MPAS
   `cofwz`, `cofwr`, `cofwt`, `coftz`) are computed inline at the linearization
   point cached in `frozen_pressure`.
3. **Divergence damping** — selectable via the typed
   [`AcousticDampingStrategy`](@ref) interface. The default is
   [`PressureProjectionDamping`](@ref) at ``\\beta_d = 0.5`` (the literal
   ERF/CM1/WRF projection form, tuned on the DCMIP2016 baroclinic wave).
   The MPAS Klemp–Skamarock–Ha 2018 momentum correction is also available
   as [`ThermodynamicDivergenceDamping`](@ref).

Fields
======

- `substeps`: Number of acoustic substeps ``N`` per outer ``Δt`` (or `nothing` for adaptive).
- `forward_weight`: Off-centering parameter ``\\omega`` for the implicit solve. ``\\omega > 0.5`` damps vertical acoustic modes. ``\\varepsilon = 2\\omega - 1`` is the MPAS off-centering.
- `damping`: Acoustic divergence damping strategy ([`AcousticDampingStrategy`](@ref)). Default [`PressureProjectionDamping`](@ref) at ``\\beta_d = 0.5``, the literal ERF/CM1/WRF projection form tuned on the DCMIP2016 baroclinic wave.
- `substep_distribution`: How acoustic substeps are distributed across the WS-RK3 stages. One of [`ProportionalSubsteps`](@ref) or [`MonolithicFirstStage`](@ref).
- `virtual_potential_temperature`: Stage-frozen ``\\theta_v`` (CenterField, MPAS `t`).
- `reference_exner_function`: Reference ``\\Pi_0 = (p_\\mathrm{ref}/p^{st})^{R/c_p}`` (CenterField, MPAS `pb`).

The horizontal acoustic PGF uses the dry-γ linearization
``\\partial_x p'' \\approx \\gamma R^d \\, \\Pi \\, \\partial_x (\\rho\\theta)''``
(Klemp 2007). For a moist atmosphere with Breeze's
`LiquidIcePotentialTemperatureFormulation` this is consistent only in the dry
limit; see `validation/substepping/NOTES.md` for the open problem.
- `ρθ″_predictor`: Per-column explicit predictor for ``(\\rho\\theta)''`` at cell centers, assembled inside the column kernel before the implicit vertical solve (CenterField; MPAS `ts`).
- `ρ″_predictor`: Per-column explicit predictor for ``\\rho''`` at cell centers, assembled inside the column kernel before the implicit vertical solve (CenterField; MPAS `rs`).
- `previous_ρθ″`: ``(\\rho\\theta)''`` snapshot before the column kernel — used by divergence damping (CenterField).
- `ρ″`:  acoustic ``\\rho`` perturbation (CenterField, MPAS `rho_pp`).
- `ρθ″`: acoustic ``(\\rho\\theta)`` perturbation (CenterField, MPAS `rtheta_pp`).
- `ρw″`: acoustic ``(\\rho w)`` perturbation (ZFaceField, MPAS `rw_p`).
- `ρu″`: acoustic ``(\\rho u)`` perturbation (XFaceField, MPAS `ru_p`).
- `ρv″`: acoustic ``(\\rho v)`` perturbation (YFaceField, MPAS `rv_p`).
- `gamma_tri`: Thomas sweep scratch in the column kernel (ZFaceField).
- `Gˢρw_total`: Stage-frozen vertical slow **momentum** tendency (kg/(m²·s²)) — assembles ``G^n.\\rho w`` (advection + Coriolis + diffusion + forcing) with the linearized vertical PGF + density-perturbation buoyancy in reference-state-subtracted form ``-\\partial_z p' - g\\rho'``. Passed directly into the column kernel as the momentum tendency; no divide by ``\\rho`` anywhere. Horizontal momentum tendencies are read directly from ``G^n.\\rho u`` / ``G^n.\\rho v``.
- `vertical_solver`: BatchedTridiagonalSolver for the implicit ``(\\rho w)''`` acoustic update.
- `frozen_pressure`: Snapshot of `model.dynamics.pressure` taken once per outer step. Used as the linearization point for the implicit Schur coefficients so that the substepper sees the same `exner` at every WS-RK3 stage of the outer step (matches MPAS, where `diag%exner` is only recomputed at `rk_step == 3`).

The `cofwz`, `cofwr`, `cofwt`, `coftz` MPAS coefficients are computed inline by
helper functions inside the column kernel — no separate fields are stored.
"""
struct AcousticSubstepper{N, FT, D, AD, CF, FF, XF, YF, GT, TS}
    substeps :: N
    forward_weight :: FT                       # Off-centering ω → epssm = 2ω - 1
    damping :: D                               # AcousticDampingStrategy
    substep_distribution :: AD                 # ProportionalSubsteps or MonolithicFirstStage
    virtual_potential_temperature :: CF        # Stage-frozen θᵥ (MPAS `t`)
    reference_exner_function :: CF             # Π₀ from reference state
    ρθ″_predictor :: CF                        # Per-column explicit predictor for ρθ″ (MPAS `ts`)
    ρ″_predictor :: CF                         # Per-column explicit predictor for ρ″  (MPAS `rs`)
    previous_ρθ″ :: CF                   # (ρθ)″ snapshot for divergence damping
    ρ″  :: CF                                  # MPAS rho_pp     — acoustic ρ perturbation
    ρθ″ :: CF                                  # MPAS rtheta_pp  — acoustic ρθ perturbation
    ρw″ :: FF                                  # MPAS rw_p       — acoustic ρw perturbation (z-face, with momentum BCs)
    ρu″ :: XF                                  # MPAS ru_p       — acoustic ρu perturbation (x-face, with momentum BCs)
    ρv″ :: YF                                  # MPAS rv_p       — acoustic ρv perturbation (y-face, with momentum BCs)
    gamma_tri :: GT                            # LU decomp scratch (z-face, default BCs)
    Gˢρw_total :: GT                           # Stage-frozen vertical slow momentum tendency (ERF-style, z-face, default BCs)
    vertical_solver :: TS                      # BatchedTridiagonalSolver for implicit ρw″ update
    frozen_pressure :: CF                      # Snapshot of model.dynamics.pressure at outer-step start
end

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.substeps,
                       a.forward_weight,
                       adapt(to, a.damping),
                       a.substep_distribution,
                       adapt(to, a.virtual_potential_temperature),
                       adapt(to, a.reference_exner_function),
                       adapt(to, a.ρθ″_predictor),
                       adapt(to, a.ρ″_predictor),
                       adapt(to, a.previous_ρθ″),
                       adapt(to, a.ρ″),
                       adapt(to, a.ρθ″),
                       adapt(to, a.ρw″),
                       adapt(to, a.ρu″),
                       adapt(to, a.ρv″),
                       adapt(to, a.gamma_tri),
                       adapt(to, a.Gˢρw_total),
                       adapt(to, a.vertical_solver),
                       adapt(to, a.frozen_pressure))

"""
$(TYPEDSIGNATURES)

Construct an `AcousticSubstepper` for the MPAS-A conservative-perturbation
acoustic substep loop.

The optional `prognostic_momentum` keyword carries the prognostic ``\\rho u``,
``\\rho v``, ``\\rho w`` fields whose boundary conditions are inherited by the
substepper's perturbation face fields ``(\\rho u)''``, ``(\\rho v)''``,
``(\\rho w)''``. This is essential on grids with `Bounded` horizontal topology
so that `fill_halo_regions!` enforces impenetrability (``v = 0`` at the
south/north walls, ``u = 0`` at the east/west walls) on the perturbation momenta.
Without this, the substepper's halo fills use default Periodic/NoFlux BCs and
boundary cells drift away from zero.
"""
function AcousticSubstepper(grid, split_explicit::SplitExplicitTimeDiscretization;
                            prognostic_momentum = nothing)
    Ns = split_explicit.substeps
    FT = eltype(grid)
    ω = convert(FT, split_explicit.forward_weight)
    damping = materialize_damping(grid, _convert_damping(FT, split_explicit.damping))
    substep_distribution = split_explicit.substep_distribution

    virtual_potential_temperature = CenterField(grid)
    reference_exner_function = CenterField(grid)
    ρθ″_predictor = CenterField(grid)
    ρ″_predictor  = CenterField(grid)
    previous_ρθ″ = CenterField(grid)

    # Inherit boundary conditions from the prognostic momentum so that
    # `fill_halo_regions!` enforces impenetrability on the perturbation fields.
    bcs_ρu = prognostic_momentum === nothing ? nothing : prognostic_momentum.ρu.boundary_conditions
    bcs_ρv = prognostic_momentum === nothing ? nothing : prognostic_momentum.ρv.boundary_conditions
    bcs_ρw = prognostic_momentum === nothing ? nothing : prognostic_momentum.ρw.boundary_conditions

    _xface(grid, bcs) = bcs === nothing ? XFaceField(grid) : XFaceField(grid; boundary_conditions = bcs)
    _yface(grid, bcs) = bcs === nothing ? YFaceField(grid) : YFaceField(grid; boundary_conditions = bcs)
    _zface(grid, bcs) = bcs === nothing ? ZFaceField(grid) : ZFaceField(grid; boundary_conditions = bcs)

    # MPAS perturbation variables (Eq. 3.23: V''_h, Ω'', Θ''_m, ρ̃''_d).
    # Only the prognostic-like perturbation momenta inherit the BCs of the
    # model's prognostic momentum fields, so that fill_halo_regions! enforces
    # impenetrability on the south/north and east/west walls. Tendencies and
    # scratch fields don't need BCs — they get overwritten on every update.
    ρ″  = CenterField(grid)
    ρθ″ = CenterField(grid)
    ρw″ = _zface(grid, bcs_ρw)
    ρu″ = _xface(grid, bcs_ρu)
    ρv″ = _yface(grid, bcs_ρv)

    # Scratch / tendency fields use plain default BCs.
    gamma_tri_field = ZFaceField(grid)

    # Horizontal slow momentum tendencies come directly from Gⁿ.ρu/Gⁿ.ρv.
    # Gˢρw_total is an assembled vertical momentum tendency (kg/(m²·s²)):
    #   Gⁿ.ρw (advection + Coriolis + diffusion + forcing) + linearized PGF + buoyancy perturbation.
    # Stored here because the assembly is non-trivial and stage-frozen.
    Gˢρw_total = ZFaceField(grid)

    # Vertical tridiagonal solver. Coefficients are computed on the fly via
    # `get_coefficient` dispatch on the stateless tag types — no array storage.
    # The solver only needs `scratch` (for the Thomas γ values) and `rhs`.
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    scratch = zeros(arch, FT, Nx, Ny, Nz)

    vertical_solver = BatchedTridiagonalSolver(grid;
                                               lower_diagonal = AcousticTridiagLower(),
                                               diagonal       = AcousticTridiagDiagonal(),
                                               upper_diagonal = AcousticTridiagUpper(),
                                               scratch,
                                               tridiagonal_direction = ZDirection())

    frozen_pressure = CenterField(grid)

    return AcousticSubstepper(Ns, ω, damping,
                              substep_distribution,
                              virtual_potential_temperature,
                              reference_exner_function,
                              ρθ″_predictor,
                              ρ″_predictor,
                              previous_ρθ″,
                              ρ″, ρθ″, ρw″, ρu″, ρv″,
                              gamma_tri_field,
                              Gˢρw_total,
                              vertical_solver,
                              frozen_pressure)
end

# Promote damping strategy fields to the grid's float type so the substepper's
# concrete type parameters match the grid float type. The methods below cover
# every concrete `AcousticDampingStrategy`. New strategies must add a method.
@inline _convert_damping(::Type, d::NoDivergenceDamping) = d
@inline _convert_damping(::Type{FT}, d::ThermodynamicDivergenceDamping) where FT =
    ThermodynamicDivergenceDamping(coefficient = convert(FT, d.coefficient),
                                   length_scale = d.length_scale === nothing ? nothing : convert(FT, d.length_scale))
@inline _convert_damping(::Type{FT}, d::ConservativeProjectionDamping) where FT =
    ConservativeProjectionDamping{FT, typeof(d.ρθ″_for_pgf)}(convert(FT, d.coefficient), d.ρθ″_for_pgf)
@inline _convert_damping(::Type{FT}, d::PressureProjectionDamping) where FT =
    PressureProjectionDamping{FT, typeof(d.ρθ″_for_pgf)}(convert(FT, d.coefficient), d.ρθ″_for_pgf)

# Materialize the damping strategy by allocating any per-strategy scratch
# fields it needs. Called once at substepper construction. Default no-op for
# strategies that carry no scratch fields. Concrete projection strategies
# allocate a CenterField (`ρθ″_for_pgf`) at this point.
@inline materialize_damping(grid, d::NoDivergenceDamping) = d
@inline materialize_damping(grid, d::ThermodynamicDivergenceDamping) = d

function materialize_damping(grid, d::ConservativeProjectionDamping{FT}) where FT
    ρθ″_for_pgf = CenterField(grid)
    return ConservativeProjectionDamping{FT, typeof(ρθ″_for_pgf)}(d.coefficient, ρθ″_for_pgf)
end

function materialize_damping(grid, d::PressureProjectionDamping{FT}) where FT
    ρθ″_for_pgf = CenterField(grid)
    return PressureProjectionDamping{FT, typeof(ρθ″_for_pgf)}(d.coefficient, ρθ″_for_pgf)
end

# Adapt methods so projection strategies survive a CPU → GPU adapt.
Adapt.adapt_structure(to, d::NoDivergenceDamping) = d
Adapt.adapt_structure(to, d::ThermodynamicDivergenceDamping) = d
function Adapt.adapt_structure(to, d::ConservativeProjectionDamping{FT}) where FT
    adapted = adapt(to, d.ρθ″_for_pgf)
    return ConservativeProjectionDamping{FT, typeof(adapted)}(d.coefficient, adapted)
end
function Adapt.adapt_structure(to, d::PressureProjectionDamping{FT}) where FT
    adapted = adapt(to, d.ρθ″_for_pgf)
    return PressureProjectionDamping{FT, typeof(adapted)}(d.coefficient, adapted)
end

"""
$(TYPEDSIGNATURES)

Snapshot `model.dynamics.pressure` into `substepper.frozen_pressure`. Called once
per outer time step to freeze the linearization point used by the substepper's
`acoustic_pgf_coefficient` and `buoyancy_linearization_coefficient` helpers, so
that the implicit Schur coefficients seen by the substep loop are identical at
every WS-RK3 stage. This matches MPAS, where `diag%exner` is only recomputed at
`rk_step==3` (end of outer step) and the substepper sees the same `exner` at
every stage of an outer step.
"""
function freeze_outer_step_state!(substepper::AcousticSubstepper, model)
    parent(substepper.frozen_pressure) .= parent(model.dynamics.pressure)
    return nothing
end

# Default for non-substepping models — does nothing.
freeze_outer_step_state!(substepper, model) = nothing

#####
##### Section 2b: Adaptive substep computation
#####

using Breeze.AtmosphereModels: thermodynamic_density, thermodynamic_density_name
using Breeze.Thermodynamics: dry_air_gas_constant

"""
$(TYPEDSIGNATURES)

Compute the number of acoustic substeps from the horizontal acoustic CFL condition.

Uses a conservative sound speed estimate ``ℂᵃᶜ = (γ Rᵈ Tᵣ)^{1/2}`` with ``Tᵣ = 300\\;\\mathrm{K}``
(giving ``ℂᵃᶜ ≈ 347\\;\\mathrm{m/s}``) and the minimum horizontal grid spacing. The vertical
CFL is not needed because the ``(\\rho w)''``–``(\\rho\\theta)''`` coupling is
vertically implicit.

The substep count is chosen so that ``ℂᵃᶜ Δτ / Δx_{min} ≤ 1/\\text{safety\\_factor}`` where
``Δτ = Δt / N`` is the acoustic substep size. The default safety factor of 2.0 targets
acoustic CFL ≈ 0.5, which matches the ERF/WRF best-practice guidance
(Klemp-Skamarock-Dudhia 2007, Skamarock-Klemp 2008, Baldauf 2010) for
forward-backward acoustic substepping with divergence damping.
"""
function compute_acoustic_substeps(grid, Δt, thermodynamic_constants)
    cᵖᵈ = thermodynamic_constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(thermodynamic_constants)
    cᵛᵈ = cᵖᵈ - Rᵈ
    γ = cᵖᵈ / cᵛᵈ
    Tᵣ = 300 # Conservative reference temperature (surface conditions)
    ℂᵃᶜ = sqrt(γ * Rᵈ * Tᵣ) # ≈ 347 m/s

    # Minimum horizontal grid spacing (skip Flat dimensions)
    TX, TY, _ = topology(grid)
    Δx_min = TX === Flat ? Inf : minimum_xspacing(grid)
    Δy_min = TY === Flat ? Inf : minimum_yspacing(grid)
    Δh_min = min(Δx_min, Δy_min)

    # Target acoustic CFL ≤ 0.5 (ERF/WRF best practice).
    safety_factor = 2.0
    return ceil(Int, safety_factor * Δt * ℂᵃᶜ / Δh_min)
end

# When substeps is specified, use it directly
@inline acoustic_substeps(N::Int, grid, Δt, constants) = N

# When substeps is nothing, compute from acoustic CFL
@inline acoustic_substeps(::Nothing, grid, Δt, constants) = compute_acoustic_substeps(grid, Δt, constants)

#####
##### Section 3: Cache preparation (once per RK stage)
#####

"""
$(TYPEDSIGNATURES)

Prepare the stage-frozen cache read by the acoustic substep loop.

Computes:

1. Virtual potential temperature ``\\theta_v`` (frozen during the substep loop).
2. Reference Exner function ``\\Pi_0`` from the model's reference state, used
   by the implicit solve as the linearization point.

The remaining stage-frozen quantities (``\\rho``, ``\\theta_v``, the implicit
Schur coefficients) are read directly from the model's prognostic fields and
from `substepper.frozen_pressure`, which is snapshotted once per outer Δt by
[`freeze_outer_step_state!`](@ref).
"""
function prepare_acoustic_cache!(substepper, model)
    grid = model.grid
    arch = architecture(grid)

    # Compute stage-frozen θᵥ (the only field this routine still produces).
    pˢᵗ = model.dynamics.standard_pressure
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    Rᵈ  = dry_air_gas_constant(model.thermodynamic_constants)
    κ   = Rᵈ / cᵖᵈ  # R/cp

    launch!(arch, grid, :xyz, _prepare_virtual_theta!,
            substepper.virtual_potential_temperature,
            model.dynamics.density,
            model.dynamics.pressure,
            model.temperature,
            specific_prognostic_moisture(model),
            grid,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants,
            pˢᵗ, κ)

    # Cache Π₀ for use by `convert_slow_tendencies!`.
    _set_exner_reference!(substepper, model, model.dynamics.reference_state)

    fill_halo_regions!(substepper.virtual_potential_temperature)

    return nothing
end

@kernel function _prepare_virtual_theta!(θᵥ_field, ρ, p, T,
                                          specific_prognostic_moisture, grid,
                                          microphysics, microphysical_fields, constants, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρⁱ = ρ[i, j, k]
        pⁱ = p[i, j, k]
        Tⁱ = T[i, j, k]
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
    end

    # Build moisture fractions (vapor / liquid / ice) and use the mixture gas
    # constant to form the virtual temperature, Tᵥ = T · Rᵐ / Rᵈ. To first order
    # in the mass fractions this is T (1 + (Rᵛ/Rᵈ - 1) qᵛ - qˡ - qⁱ), the standard
    # MPAS / WRF / Klemp et al. 2007 definition that buoyancy and the linearized
    # PGF in the substep loop expect.
    q  = grid_moisture_fractions(i, j, k, grid, microphysics, ρⁱ, qᵛᵉ, microphysical_fields)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵐ = mixture_gas_constant(q, constants)
    Tᵥ = Tⁱ * Rᵐ / Rᵈ

    # Virtual potential temperature θᵥ = Tᵥ / Π. We use the dry Exner function
    # Π = (p/pˢᵗ)^κ with the dry κ = Rᵈ/cᵖᵈ to be consistent with the reference
    # Exner state Π₀ used by `_set_exner_reference!` and the slow PGF below.
    Πⁱ = (pⁱ / pˢᵗ)^κ
    @inbounds θᵥ_field[i, j, k] = Tᵥ / Πⁱ
end

##### Set the Exner reference state for the acoustic loop.
##### Dispatches on reference state type to use the most accurate π₀.

function _set_exner_reference!(substepper, model, ref::ExnerReferenceState)
    parent(substepper.reference_exner_function) .= parent(ref.exner_function)
    return nothing
end

function _set_exner_reference!(substepper, model, ::Nothing)
    fill!(parent(substepper.reference_exner_function), 0)
    return nothing
end

@kernel function _compute_reference_exner_from_pressure!(πᵣ, p_ref, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)
    @inbounds πᵣ[i, j, k] = (p_ref[i, j, k] / pˢᵗ)^κ
end

@inline reference_exner(i, j, k, ::Nothing, pˢᵗ, κ) = zero(pˢᵗ)

@inline function reference_exner(i, j, k, ref::ExnerReferenceState, pˢᵗ, κ)
    @inbounds return ref.exner_function[i, j, k]
end

#####
##### Section 4: Convert slow tendencies to velocity/pressure form
#####

"""
$(TYPEDSIGNATURES)

Convert slow momentum tendencies to velocity form and add the MPAS linearized
pressure gradient and buoyancy (§5-8 of mpas_algorithm_complete.md).

For `SplitExplicitTimeDiscretization`, the dynamics kernel zeroes the vertical
PGF, buoyancy, and horizontal PGF. These are added back here using:
- §5: p′(k) = p_frozen(k) − p_ref(k)
- §6: dpdz(k) = -g (ρ(k) - ρ₀(k))
- §7: tend_w_euler = -rdzu (p′_k - p′_{k-1}) + fzm dpdz_k + fzp dpdz_{k-1}
- §8: tend_u_euler = -(p′(i,j,k) - p′(i-1,j,k)) / Δx

All computed from U⁰ (step-start state), frozen across all RK stages.
"""
function convert_slow_tendencies!(substepper, model, U⁰)
    grid = model.grid
    arch = architecture(grid)
    g    = model.thermodynamic_constants.gravitational_acceleration
    Gⁿ   = model.timestepper.Gⁿ

    ref = model.dynamics.reference_state
    ρᵣ = ref isa Nothing ? model.dynamics.density  : ref.density
    pᵣ = ref isa Nothing ? model.dynamics.pressure : ref.pressure

    launch!(arch, grid, :xyz, _convert_slow_tendencies!,
            substepper.Gˢρw_total,
            Gⁿ.ρw,
            grid, g,
            U⁰.ρ, substepper.frozen_pressure,
            pᵣ, ρᵣ)
    return nothing
end

##### Perturbation pressure pp = p - p_base at a cell center.
#####
##### Earlier versions of this kernel used the linearized form
#####     pp(k) = Rᵈ (Π(k) ρθ_p(k) + ρθ_base(k) (Π(k) - Π_base(k)))
##### with Π recomputed from ρθ⁰ via the **dry-air** EOS,
##### `Π = (Rᵈ ρθ⁰ / pˢᵗ)^(Rᵈ/cᵥ)`. For dry air this is exactly equal to
##### `(p/pˢᵗ)^κ` (so pp = p − p_base reduces to the linearized form). For a
##### moist atmosphere — where p = ρ Rᵐ T with Rᵐ ≈ Rᵈ(1 + 0.608 qᵛ − qˡ −
##### qⁱ) — that identity breaks at O(qᵛ ε_v) ≈ a few percent, leaving a
##### residual `tend_w_euler ≠ 0` even for a hydrostatically-balanced moist
##### state. That residual integrates over the substeps into a large w
##### startup transient that destabilizes the moist BW within ~3 outer steps.
#####
##### To eliminate that EOS mismatch we use the cached, **moist** pressure
##### directly: `pp = p_frozen − p_base`. This is exactly zero in any
##### hydrostatic balance (because both p_frozen and p_base solve hydrostatics
##### using whichever EOS produced them) and is exact for any equation of
##### state, dry or moist. The cancellation between −∂(pp)/∂z and the
##### gravity-density `dpdz = -g (ρ⁰ - ρ₀)` in §7 still holds: in
##### balance, ∂p_frozen/∂z = -ρ⁰ g and ∂p_base/∂z = -ρ₀ g, so the two
##### terms exactly cancel.
@inline function perturbation_pressure(i, j, k, p_frozen, pᵣ)
    @inbounds return p_frozen[i, j, k] - pᵣ[i, j, k]
end

@kernel function _convert_slow_tendencies!(Gˢρw_total, Gⁿρw, grid, g,
                                           ρ⁰, p_frozen, pᵣ, ρᵣ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Vertical slow momentum tendency in MOMENTUM FORM (kg/(m²·s²)):
        #   ∂ₜ(ρw) = (advection, etc.) − ∂p/∂z − ρ g
        #          = Gⁿρw              − ∂(p−pᵣ)/∂z − (ρ−ρᵣ) g     (using ∂pᵣ/∂z = −ρᵣ g)
        #          = Gⁿρw              − ∂z_p′      − g·ρ′
        # where ′ denotes deviation from the reference. Dynamics has zeroed the
        # vertical PGF + buoyancy in Gⁿρw for SplitExplicit, so we reinstate it
        # here at the z-face, evaluated from U⁰ and frozen across the RK stage.
        # No divide-by-ρ here: the column kernel consumes this directly as a
        # momentum tendency, matching ERF/WRF.
        if k > 1
            # Perturbation pressure p′ = p_frozen − pᵣ at cell centers, ∂z(p′) at the face.
            p′ᵏ = perturbation_pressure(i, j, k,     p_frozen, pᵣ)
            p′⁻ = perturbation_pressure(i, j, k - 1, p_frozen, pᵣ)
            Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
            ∂z_p′ = (p′ᵏ - p′⁻) / Δzᶠ

            # Density-perturbation buoyancy force per unit volume, g·ρ′, at centers,
            # arithmetically averaged to the z-face (matches Breeze's anelastic
            # `buoyancy_forceᶜᶜᶠ`, Oceananigans' `ℑzᵃᵃᶠ`, and ERF).
            gρ′ᵏ = g * (ρ⁰[i, j, k]     - ρᵣ[i, j, k])
            gρ′⁻ = g * (ρ⁰[i, j, k - 1] - ρᵣ[i, j, k - 1])
            gρ′ᶜᶜᶠ = (gρ′ᵏ + gρ′⁻) / 2

            # Full vertical slow forcing at the z-face (momentum units):
            #   advection + Coriolis + diffusion + forcing (Gⁿρw) + linearized PGF + buoyancy perturbation
            Gˢρw_total[i, j, k] = Gⁿρw[i, j, k] - ∂z_p′ - gρ′ᶜᶜᶠ
        else
            Gˢρw_total[i, j, k] = zero(eltype(Gˢρw_total))
        end
    end
end

#####
##### MPAS-style horizontal forward step using ρθ perturbation PGF.
##### MPAS: u += dts * (Gˢu - γRᵈ * Πᶠᶜᶜ * ∂x(ρθ″) * cqw / zz_face)
##### For dry air, no terrain: cqw=1, zz=1.

# Exner function at a cell center, for interpolation via function composition.
@inline _exner_from_p(i, j, k, grid, p, pˢᵗ, κ) = (p[i, j, k] / pˢᵗ)^κ

@kernel function _mpas_horizontal_forward!(ρu″, ρv″, grid, Δτ,
                                            ρθ″, pressure,
                                            Gⁿρu, Gⁿρv,
                                            cᵖᵈ, Rᵈ, pˢᵗ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        cᵛᵈ = cᵖᵈ - Rᵈ
        γRᵈ = cᵖᵈ * Rᵈ / cᵛᵈ  # = γ·Rᵈ, the Klemp 2007 PGF prefactor
        κ   = Rᵈ / cᵖᵈ

        # ERF-style horizontal acoustic momentum update in MOMENTUM form:
        #   ∂x_p″ = γRᵈ · Πᶠᶜᶜ · ∂x(ρθ″)              (Klemp 2007 linearization, dry-γ form)
        #   ρu″  += Δτ · (Gⁿρu − ∂x_p″)               (momentum perturbation only)
        # We do NOT update u/v during substeps; velocities are reconstructed
        # once at the end of the substep loop via ρu_new = ρu⁰ + ρu″, then
        # u_new = ρu_new / ρ_new. This matches ERF's momentum-only prognostic
        # convention (see ERF docs, "Acoustic Sub-stepping"), which is a
        # natural fit since Breeze's outer prognostic is ρu, not u.
        # NOTE: moist-consistent only in the dry limit at present — see
        # `validation/substepping/NOTES.md` ("Open: moist acoustic PGF").
        Πᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, _exner_from_p, pressure, pˢᵗ, κ)
        ∂x_ρθ″ = ∂xᶠᶜᶜ(i, j, k, grid, ρθ″)
        ∂x_p″  = γRᵈ * Πᶠᶜᶜ * ∂x_ρθ″

        not_bdy_x = !on_x_boundary(i, j, k, grid)
        ρu″[i, j, k] += Δτ * (Gⁿρu[i, j, k] - ∂x_p″) * not_bdy_x

        Πᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, _exner_from_p, pressure, pˢᵗ, κ)
        ∂y_ρθ″ = ∂yᶜᶠᶜ(i, j, k, grid, ρθ″)
        ∂y_p″  = γRᵈ * Πᶜᶠᶜ * ∂y_ρθ″

        not_bdy_y = !on_y_boundary(i, j, k, grid)
        ρv″[i, j, k] += Δτ * (Gⁿρv[i, j, k] - ∂y_p″) * not_bdy_y
    end
end

#####
##### Section 8a: MPAS divergence damping (Klemp, Skamarock & Ha 2018)
#####
##### Applied after each acoustic substep to the horizontal momentum
##### perturbations as
#####
#####   ru_p += coef * δx(divΘ) / (2 θ_m_edge)
#####   rv_p += coef * δy(divΘ) / (2 θ_m_edge)
#####
##### where divΘ = -(rtheta_pp_new - rtheta_pp_old) is the discrete acoustic
##### (ρθ)″ tendency (used as the divergence proxy) and
##### coef = 2 * smdiv * len_disp / Δτ. Using the (ρθ)″ tendency as the
##### proxy preserves gravity-wave frequencies while damping the grid-scale
##### acoustic divergence.
#####

# Divergence proxy: divCell = -(ρθ″_new - ρθ″_old).
@inline _neg_δΘ(i, j, k, grid, ρθ″, ρθ″_old) =
    -(ρθ″[i, j, k] - ρθ″_old[i, j, k])

@kernel function _mpas_divergence_damping!(ρu″, ρv″,
                                            ρθ″, ρθ″_old, θᵥ,
                                            grid, coef_div_damp)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # MPAS (lines 3059-3062): divCell = -(ρθ″_new - ρθ″_old)
        # ρu″ += coef_divdamp * δx(divCell) / (θᵥ_cell1 + θᵥ_cell2)
        # Uses topology-safe operators: ∂xᶠᶜᶜ returns 0 for Flat x.

        # x-direction: gradient of divΘ at u-face, divided by θᵥ sum
        ∂x_divΘ = δxᶠᵃᵃ(i, j, k, grid, _neg_δΘ, ρθ″, ρθ″_old)
        θᵥ_sumᶠᶜᶜ = 2 * ℑxᶠᵃᵃ(i, j, k, grid, θᵥ)
        θᵥ_sumᶠᶜᶜ_safe = ifelse(θᵥ_sumᶠᶜᶜ == 0, one(θᵥ_sumᶠᶜᶜ), θᵥ_sumᶠᶜᶜ)
        ρu″[i, j, k] += coef_div_damp * ∂x_divΘ / θᵥ_sumᶠᶜᶜ_safe *
                          !on_x_boundary(i, j, k, grid)

        # y-direction: gradient of divΘ at v-face, divided by θᵥ sum
        ∂y_divΘ = δyᵃᶠᵃ(i, j, k, grid, _neg_δΘ, ρθ″, ρθ″_old)
        θᵥ_sumᶜᶠᶜ = 2 * ℑyᵃᶠᵃ(i, j, k, grid, θᵥ)
        θᵥ_sumᶜᶠᶜ_safe = ifelse(θᵥ_sumᶜᶠᶜ == 0, one(θᵥ_sumᶜᶠᶜ), θᵥ_sumᶜᶠᶜ)
        ρv″[i, j, k] += coef_div_damp * ∂y_divΘ / θᵥ_sumᶜᶠᶜ_safe *
                          !on_y_boundary(i, j, k, grid)
    end
end

#####
##### Divergence damping strategy dispatch
#####
##### Each substep calls `apply_divergence_damping!(strategy, substepper, grid, Δτ)`
##### immediately after the column kernel + ρθ″ halo fill. The strategy is the
##### `damping :: AcousticDampingStrategy` field carried by the substepper.
##### Phase 3 of the cleanup plan adds two pressure-projection variants
##### (`PressureProjectionDamping`, `ConservativeProjectionDamping`); their
##### dispatch methods will be added here when implemented.
#####

#####
##### PGF source dispatch
#####
##### `pgf_source_field(damping, substepper)` returns the (ρθ)″ field that the
##### horizontal forward kernel will read as the PGF source. For non-projection
##### strategies it is `substepper.ρθ″` directly. For projection strategies it
##### is the strategy's own `ρθ″_for_pgf` scratch CenterField, which is filled
##### by `apply_pgf_filter!` at the start of each substep.
#####

@inline pgf_source_field(::AcousticDampingStrategy, substepper) = substepper.ρθ″
@inline pgf_source_field(damping::ConservativeProjectionDamping, substepper) = damping.ρθ″_for_pgf
@inline pgf_source_field(damping::PressureProjectionDamping, substepper) = damping.ρθ″_for_pgf

#####
##### Pre-substep filter dispatch
#####
##### Called at the start of every substep, before the horizontal forward step.
##### Default for `NoDivergenceDamping` and `ThermodynamicDivergenceDamping` is
##### a no-op (the horizontal forward kernel reads `substepper.ρθ″` directly).
##### Projection strategies launch a kernel that writes `damping.ρθ″_for_pgf`
##### from the current `(ρθ)″` and the previous-substep snapshot
##### `substepper.previous_ρθ″`.
#####

@inline apply_pgf_filter!(::AcousticDampingStrategy, substepper, model, ρθ_stage) = nothing

function apply_pgf_filter!(damping::ConservativeProjectionDamping, substepper, model, ρθ_stage)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    β = convert(FT, damping.coefficient)
    launch!(arch, grid, :xyz, _conservative_projection_filter!,
            damping.ρθ″_for_pgf, substepper.ρθ″, substepper.previous_ρθ″, β)
    # The horizontal forward kernel reads ρθ_for_pgf via ∂xᶠᶜᶜ / ∂yᶜᶠᶜ, which
    # accesses the i+1 / j+1 halo cells.
    fill_halo_regions!(damping.ρθ″_for_pgf)
    return nothing
end

@kernel function _conservative_projection_filter!(ρθ_for_pgf, ρθ″, ρθ″_old, β)
    i, j, k = @index(Global, NTuple)
    @inbounds ρθ_for_pgf[i, j, k] = ρθ″[i, j, k] + β * (ρθ″[i, j, k] - ρθ″_old[i, j, k])
end

function apply_pgf_filter!(damping::PressureProjectionDamping, substepper, model, ρθ_stage)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    β = convert(FT, damping.coefficient)
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    Rᵈ  = dry_air_gas_constant(model.thermodynamic_constants)
    pˢᵗ = FT(model.dynamics.standard_pressure)
    rcv = FT(Rᵈ / (cᵖᵈ - Rᵈ))   # R/cv (the Π exponent on (Rρθ/pˢᵗ))
    cv_over_R = FT((cᵖᵈ - Rᵈ) / Rᵈ)
    launch!(arch, grid, :xyz, _pressure_projection_filter!,
            damping.ρθ″_for_pgf, substepper.ρθ″, substepper.previous_ρθ″,
            ρθ_stage, β, FT(Rᵈ), pˢᵗ, rcv, cv_over_R)
    fill_halo_regions!(damping.ρθ″_for_pgf)
    return nothing
end

@kernel function _pressure_projection_filter!(ρθ_for_pgf, ρθ″, ρθ″_old,
                                              ρθ_stage, β, R, pˢᵗ, rcv, cv_over_R)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # Dry-γ linearization of Π about the stage-frozen state:
        # Π_stage = (R · ρθ_stage / pˢᵗ)^(R/cv),
        # Π_curr  = (R · (ρθ_stage + ρθ″)     / pˢᵗ)^(R/cv),
        # Π_old   = (R · (ρθ_stage + ρθ″_old) / pˢᵗ)^(R/cv).
        # See `validation/substepping/NOTES.md` for the moist-EOS caveat.
        ρθ_st  = ρθ_stage[i, j, k]
        ρθ_st_safe = ifelse(ρθ_st == 0, one(ρθ_st), ρθ_st)
        Π_stage = (R * ρθ_st / pˢᵗ)^rcv
        Π_stage_safe = ifelse(Π_stage == 0, one(Π_stage), Π_stage)
        Π_curr = (R * (ρθ_st + ρθ″[i, j, k])     / pˢᵗ)^rcv
        Π_old  = (R * (ρθ_st + ρθ″_old[i, j, k]) / pˢᵗ)^rcv
        π″_curr = Π_curr - Π_stage
        π″_old  = Π_old  - Π_stage
        # Linearized EOS conversion factor (cv/R) · ρθ_stage / Π_stage —
        # i.e. d(ρθ)/dπ at the frozen state.
        conversion = cv_over_R * ρθ_st_safe / Π_stage_safe
        ρθ_for_pgf[i, j, k] = ρθ″[i, j, k] + conversion * β * (π″_curr - π″_old)
    end
end

@inline apply_divergence_damping!(::NoDivergenceDamping, substepper, grid, Δτ) = nothing
# Projection strategies do their work in `apply_pgf_filter!` instead — the
# post-substep momentum correction is a no-op for them.
@inline apply_divergence_damping!(::ConservativeProjectionDamping, substepper, grid, Δτ) = nothing
@inline apply_divergence_damping!(::PressureProjectionDamping,    substepper, grid, Δτ) = nothing

function apply_divergence_damping!(damping::ThermodynamicDivergenceDamping, substepper, grid, Δτ)
    arch = architecture(grid)
    FT = eltype(grid)

    # MPAS `config_len_disp` is a user-set scalar nominal grid resolution. By
    # default we derive it as the minimum horizontal cell spacing, skipping
    # Flat axes; on a 3D periodic-periodic grid this is min(Δx, Δy); on a
    # 2D periodic-flat grid it falls back to Δx. Users can override this via
    # `damping.length_scale`.
    if damping.length_scale === nothing
        TX, TY, _ = topology(grid)
        Δx_eff = TX === Flat ? FT(Inf) : FT(minimum_xspacing(grid))
        Δy_eff = TY === Flat ? FT(Inf) : FT(minimum_yspacing(grid))
        len_disp_raw = min(Δx_eff, Δy_eff)
        len_disp = isfinite(len_disp_raw) ? len_disp_raw : one(FT)
    else
        len_disp = convert(FT, damping.length_scale)
    end

    smdiv = convert(FT, damping.coefficient)
    coef_div_damp = 2 * smdiv * len_disp / Δτ

    launch!(arch, grid, :xyz, _mpas_divergence_damping!,
            substepper.ρu″, substepper.ρv″,
            substepper.ρθ″, substepper.previous_ρθ″,
            substepper.virtual_potential_temperature,
            grid, coef_div_damp)
    return nothing
end

##### MPAS acoustic substep: verbatim translation of Sections 3-8.
##### Area-weighted θᵥ fluxes for the topology-safe divergence computation.
@inline Axθᵥρu″(i, j, k, grid, θᵥ, ρu″) = Axᶠᶜᶜ(i, j, k, grid) * ℑxᶠᵃᵃ(i, j, k, grid, θᵥ) * ρu″[i, j, k]
@inline Ayθᵥρv″(i, j, k, grid, θᵥ, ρv″) = Ayᶜᶠᶜ(i, j, k, grid) * ℑyᵃᶠᵃ(i, j, k, grid, θᵥ) * ρv″[i, j, k]

#####
##### Inline tridiagonal coefficients for the MPAS acoustic substep.
#####
##### These return the per-unit-Δτₛ value of each MPAS coefficient at a single
##### face/center point. The runtime substep kernel multiplies by Δτₛ where
##### needed. They replace the cofwz/cofwr/cofwt/coftz fields that were
##### previously precomputed and cached on the substepper.
#####

# Vertical-face projection of a center-valued scalar from adjacent levels
# Acoustic PGF coefficient at face k: γRᵈ · Π_face / Δzᶠ.
# Π is interpolated from centers k-1 and k via arithmetic mean (matches the
# rest of Breeze, Oceananigans' `ℑzᵃᵃᶠ`, and ERF's acoustic substep).
@inline function acoustic_pgf_coefficient(i, j, k, grid, pressure, γRᵈ, pˢᵗ, κ)
    Δzᶠ  = Δzᶜᶜᶠ(i, j, k, grid)
    Πₖ   = (pressure[i, j, k]     / pˢᵗ)^κ
    Π⁻   = (pressure[i, j, k - 1] / pˢᵗ)^κ
    Πᶜᶜᶠ = (Πₖ + Π⁻) / 2
    return γRᵈ / Δzᶠ * Πᶜᶜᶠ
end

# Gravity coefficient at face k (MPAS `cofwr / dtseps`). Collapses to g/2.
@inline buoyancy_coefficient(g) = g / 2

# θᵥ interpolated to the z-face at k via arithmetic mean. Multiplied by ρw″
# gives the vertical θᵥ-flux Jθᵥ at the face (MPAS `coftz` up to the Δτᵋ factor).
# Returns 0 at the bottom face (k=1) and the top face (k=Nz+1) so the kernel
# can call this helper unconditionally even at boundary indices.
@inline function θᵥ_at_face(i, j, k, grid, θᵥ)
    Nz = size(grid, 3)
    in_interior = (k >= 2) & (k <= Nz)
    k_safe = ifelse(in_interior, k, 2)
    @inbounds val = (θᵥ[i, j, k_safe] + θᵥ[i, j, k_safe - 1]) / 2
    return ifelse(in_interior, val, zero(val))
end

# Buoyancy linearization coefficient at center k (MPAS cofwt / dtseps):
#   cofwt(k) = (R/cᵥ)/2 × g × ρ₀(k) × Π(k) / [ρθ(k) × Π₀(k)]
@inline function buoyancy_linearization_coefficient(i, j, k, grid,
                                                     pressure, ρ₀, Π₀_field,
                                                     ρθ_stage, pˢᵗ, κ, rcv, g)
    ρθ = ρθ_stage[i, j, k]
    ρθ_safe = ifelse(ρθ == 0, one(ρθ), ρθ)
    Π₀ = Π₀_field[i, j, k]
    Π₀_safe = ifelse(Π₀ == 0, one(Π₀), Π₀)
    Πₖ = (pressure[i, j, k] / pˢᵗ)^κ
    return rcv / 2 * g * ρ₀[i, j, k] * Πₖ / (ρθ_safe * Π₀_safe)
end

#####
##### Inline helpers used by the column kernel below.
#####

# Explicit (forward) update for ρw″ at face k. Combines:
#   - slow momentum tendency Δτ · Gˢρw (momentum units, kg/(m²·s²))
#   - acoustic θ-difference  (pgf_coeff)
#   - gravity-density       (buoy_coeff)
#   - linearized buoyancy    (buoy_lin_coeff)
# Superscripts ᵏ and ⁻ denote "at face/center k" and "at k-1" respectively.
# `ρθ″_pred*` and `ρ″_pred*` are the explicit predictors from the preceding
# Section-4 accumulation (see `_build_acoustic_rhs!`); `ρθ″_old*`, `ρ″_old*`
# are the values from the previous substep.
@inline function _explicit_ρw″_face_update(ρw″_oldᵏ,
                                           Δτ, Gˢρwᵏ,
                                           pgf_coeffᵏ, buoy_coeffᵏ,
                                           buoy_lin_coeffᵏ, buoy_lin_coeff⁻,
                                           ρθ″_predᵏ, ρθ″_pred⁻,
                                           ρ″_predᵏ,  ρ″_pred⁻,
                                           ρθ″_oldᵏ,  ρθ″_old⁻,
                                           ρ″_oldᵏ,   ρ″_old⁻,
                                           backward_weight)
    return ρw″_oldᵏ + Δτ * Gˢρwᵏ -
           pgf_coeffᵏ  * ((ρθ″_predᵏ - ρθ″_pred⁻) +
                          backward_weight * (ρθ″_oldᵏ - ρθ″_old⁻)) -
           buoy_coeffᵏ * ((ρ″_predᵏ + ρ″_pred⁻) +
                          backward_weight * (ρ″_oldᵏ + ρ″_old⁻)) +
           buoy_lin_coeffᵏ * (ρθ″_predᵏ + backward_weight * ρθ″_oldᵏ) +
           buoy_lin_coeff⁻ * (ρθ″_pred⁻ + backward_weight * ρθ″_old⁻)
end

# Tridiagonal coefficients (a, b, c) at face k.
# Names follow the (a, b, c) Thomas-algorithm convention; see Doc C for the full
# Schur-complement derivation. Superscripts ᵏ/⁻/⁺ denote levels k/k-1/k+1.
# `Jθ*` := θᵥ_at_face · Δτᵋ (vertical θᵥ-flux per unit ρw, scaled by Δτᵋ).
@inline function _tridiag_a_at_face(pgf_coeffᵏ, buoy_coeffᵏ, buoy_lin_coeff⁻,
                                    Jθ⁻, cofrz⁻, rdzw_below)
    return -pgf_coeffᵏ * Jθ⁻ * rdzw_below +
            buoy_coeffᵏ * cofrz⁻ -
            buoy_lin_coeff⁻ * Jθ⁻ * rdzw_below
end

@inline function _tridiag_b_at_face(pgf_coeffᵏ, buoy_coeffᵏ,
                                    buoy_lin_coeffᵏ, buoy_lin_coeff⁻,
                                    Jθᵏ, cofrzᵏ, cofrz⁻,
                                    rdzw_above, rdzw_below)
    return 1 +
           pgf_coeffᵏ * (Jθᵏ * rdzw_above + Jθᵏ * rdzw_below) -
           Jθᵏ * (buoy_lin_coeffᵏ * rdzw_above - buoy_lin_coeff⁻ * rdzw_below) +
           buoy_coeffᵏ * (cofrzᵏ - cofrz⁻)
end

@inline function _tridiag_c_at_face(pgf_coeffᵏ, buoy_coeffᵏ, buoy_lin_coeffᵏ,
                                    Jθ⁺, cofrzᵏ, rdzw_above)
    return -pgf_coeffᵏ * Jθ⁺ * rdzw_above -
            buoy_coeffᵏ * cofrzᵏ +
            buoy_lin_coeffᵏ * Jθ⁺ * rdzw_above
end

#####
##### Functional coefficient types for the BatchedTridiagonalSolver.
#####
##### These are stateless tag structs. The solver dispatches `get_coefficient`
##### on them, and the methods compute the tridiagonal entry on the fly using
##### the inline coefficient helpers above.
#####
##### Conventions (NO-shift mapping: face index = solver row index):
##### - The model's z-grid has Nz cells, so ρw″ has Nz+1 face entries.
##### - The acoustic system has unknowns ρw″[k] for k = 2..Nz with boundary
#####   conditions ρw″[1] = 0 and ρw″[Nz+1] = 0.
##### - We solve on the model grid (Nz rows). Solver row k_s = face k = k_s.
##### - Solver row 1 is the bottom-boundary face: b[1] = 1, c[1] = 0, f[1] = 0
#####   so the solver computes ρw″[1] = 0. The trivial solve is cheap.
##### - The top boundary face Nz+1 lives outside the solver and stays at its
#####   initialized value (= 0).
##### - Inside `get_coefficient`, the third argument k corresponds to:
#####     * a (lower): solver row k_s − 1, so face = k + 1
#####     * b (diag): solver row k_s,     so face = k
#####     * c (upper): solver row k_s − 1, so face = k
#####
##### These conventions are verified by test/batched_tridiagonal_vs_mpas_thomas.jl.
#####

struct AcousticTridiagLower end
struct AcousticTridiagDiagonal end
struct AcousticTridiagUpper end

# Per-substep inputs are passed via solve!(ϕ, solver, rhs, args...) and arrive
# here as the variadic tail. Order: pressure, ρ₀, Π₀_field, ρθ_stage,
# θᵥ, γRᵈ, pˢᵗ, κ, rcv, g, Δτᵋ.

import Oceananigans.Solvers: get_coefficient

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagLower, p, ::ZDirection,
                                 pressure, ρ₀, Π₀_field, ρθ_stage, θᵥ,
                                 γRᵈ, pˢᵗ, κ, rcv, g, Δτᵋ)
    # Lower at face k_face = k + 1
    k_face = k + 1
    Δzᶜ_below = Δzᶜᶜᶜ(i, j, k_face - 1, grid)
    rdzw_below = 1 / Δzᶜ_below
    cofrz⁻ = Δτᵋ * rdzw_below

    pgf_coeffᵏ      = acoustic_pgf_coefficient(i, j, k_face, grid, pressure, γRᵈ, pˢᵗ, κ) * Δτᵋ
    buoy_coeffᵏ     = buoyancy_coefficient(g) * Δτᵋ
    buoy_lin_coeff⁻ = buoyancy_linearization_coefficient(i, j, k_face - 1, grid,
                                                         pressure, ρ₀, Π₀_field,
                                                         ρθ_stage, pˢᵗ, κ, rcv, g) * Δτᵋ
    Jθ⁻    = θᵥ_at_face(i, j, k_face - 1, grid, θᵥ) * Δτᵋ

    return _tridiag_a_at_face(pgf_coeffᵏ, buoy_coeffᵏ, buoy_lin_coeff⁻,
                              Jθ⁻, cofrz⁻, rdzw_below)
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagDiagonal, p, ::ZDirection,
                                 pressure, ρ₀, Π₀_field, ρθ_stage, θᵥ,
                                 γRᵈ, pˢᵗ, κ, rcv, g, Δτᵋ)
    # Bottom-boundary row: trivial b = 1, paired with f[1] = 0 → ρw″[1] = 0.
    k == 1 && return one(γRᵈ)

    # Otherwise face = k, build the diagonal at face k.
    k_face = k
    Δzᶜ_above = Δzᶜᶜᶜ(i, j, k_face,     grid)
    Δzᶜ_below = Δzᶜᶜᶜ(i, j, k_face - 1, grid)
    rdzw_above = 1 / Δzᶜ_above
    rdzw_below = 1 / Δzᶜ_below
    cofrzᵏ     = Δτᵋ * rdzw_above
    cofrz⁻     = Δτᵋ * rdzw_below

    pgf_coeffᵏ      = acoustic_pgf_coefficient(i, j, k_face, grid, pressure, γRᵈ, pˢᵗ, κ) * Δτᵋ
    buoy_coeffᵏ     = buoyancy_coefficient(g) * Δτᵋ
    buoy_lin_coeffᵏ = buoyancy_linearization_coefficient(i, j, k_face,     grid,
                                                         pressure, ρ₀, Π₀_field,
                                                         ρθ_stage, pˢᵗ, κ, rcv, g) * Δτᵋ
    buoy_lin_coeff⁻ = buoyancy_linearization_coefficient(i, j, k_face - 1, grid,
                                                         pressure, ρ₀, Π₀_field,
                                                         ρθ_stage, pˢᵗ, κ, rcv, g) * Δτᵋ
    Jθᵏ    = θᵥ_at_face(i, j, k_face, grid, θᵥ) * Δτᵋ

    return _tridiag_b_at_face(pgf_coeffᵏ, buoy_coeffᵏ,
                              buoy_lin_coeffᵏ, buoy_lin_coeff⁻,
                              Jθᵏ, cofrzᵏ, cofrz⁻,
                              rdzw_above, rdzw_below)
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagUpper, p, ::ZDirection,
                                 pressure, ρ₀, Π₀_field, ρθ_stage, θᵥ,
                                 γRᵈ, pˢᵗ, κ, rcv, g, Δτᵋ)
    # Bottom-boundary row: c[1] must be 0 so the back-substitution preserves
    # ρw″[1] = 0. (γ_1 = c[1] / β_1 = 0/1 = 0; ρw″[1] -= γ_1 * ρw″[2] = 0.)
    k == 1 && return zero(γRᵈ)

    # Otherwise face = k, build the upper at face k.
    k_face = k
    Δzᶜ_above = Δzᶜᶜᶜ(i, j, k_face, grid)
    rdzw_above = 1 / Δzᶜ_above
    cofrzᵏ     = Δτᵋ * rdzw_above

    pgf_coeffᵏ      = acoustic_pgf_coefficient(i, j, k_face, grid, pressure, γRᵈ, pˢᵗ, κ) * Δτᵋ
    buoy_coeffᵏ     = buoyancy_coefficient(g) * Δτᵋ
    buoy_lin_coeffᵏ = buoyancy_linearization_coefficient(i, j, k_face, grid,
                                                         pressure, ρ₀, Π₀_field,
                                                         ρθ_stage, pˢᵗ, κ, rcv, g) * Δτᵋ
    Jθ⁺    = θᵥ_at_face(i, j, k_face + 1, grid, θᵥ) * Δτᵋ

    return _tridiag_c_at_face(pgf_coeffᵏ, buoy_coeffᵏ, buoy_lin_coeffᵏ,
                              Jθ⁺, cofrzᵏ, rdzw_above)
end

##### This kernel does ONE complete substep for ONE column (i,j).
##### Launched with :xy worksize. Sequential k-loops match MPAS exactly.

##### Builds the explicit ρw″ predictor (the right-hand side of the tridiagonal)
##### in place on ρw″ at faces k = 2..Nz, and writes the ρ″ / ρθ″ explicit
##### predictors at all centers k = 1..Nz. Does NOT do the Thomas sweep —
##### that step lives in the BatchedTridiagonalSolver call back in the
##### substep loop.
@kernel function _build_acoustic_rhs!(ρw″, ρ″, ρθ″,
                                       ρθ″_predictor, ρ″_predictor,
                                       ρu″, ρv″,
                                       grid, Δτ, Δτᵋ, backward_weight, ε,
                                       Gˢρw_total, Gˢρ, Gˢρθ,
                                       θᵥ, ρ,
                                       pressure, ρ₀, Π₀_field, ρθ_stage,
                                       γRᵈ, pˢᵗ, κ, rcv, g,
                                       is_first_substep)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        ## ── MPAS Section 3: Initialize on first substep ──
        if is_first_substep
            for k in 1:Nz
                ρ″[i, j, k]  = 0
                ρθ″[i, j, k] = 0
                ρw″[i, j, k] = 0
            end
            ρw″[i, j, Nz + 1] = 0
        end

        ## ── MPAS Section 4: build ρ″ and ρθ″ explicit predictors at cell centers ──
        for k in 1:Nz
            V = Vᶜᶜᶜ(i, j, k, grid)

            # Horizontal flux divergences: mass flux Jρ″ = (ρu″, ρv″),
            # θᵥ-flux Jθᵥρ″ = (θᵥ_face · ρu″, θᵥ_face · ρv″) area-weighted.
            div_Jρ″_h   = div_xyᶜᶜᶜ(i, j, k, grid, ρu″, ρv″)
            div_Jθᵥρ″_h = (δxᶜᵃᵃ(i, j, k, grid, Axθᵥρu″, θᵥ, ρu″) +
                           δyᵃᶜᵃ(i, j, k, grid, Ayθᵥρv″, θᵥ, ρv″)) / V

            Δzᶜ    = Δzᶜᶜᶜ(i, j, k, grid)
            cofrzᵏ = Δτᵋ / Δzᶜ
            ρw″⁺   = ρw″[i, j, k + 1]
            ρw″ᵏ   = ρw″[i, j, k]

            # ρ″ predictor: old ρ″ + Δτ · slow tendency − Δτ · ∇ₕ·Jρ″ − backward-weighted vertical divergence.
            ρ″_predᵏ = ρ″[i, j, k] + Δτ * Gˢρ[i, j, k] - Δτ * div_Jρ″_h -
                       cofrzᵏ * backward_weight * (ρw″⁺ - ρw″ᵏ)

            # ρθ″ predictor: same structure with θᵥ-weighted vertical mass-flux.
            θᵥ_face⁺ = θᵥ_at_face(i, j, k + 1, grid, θᵥ)
            θᵥ_faceᵏ = θᵥ_at_face(i, j, k,     grid, θᵥ)
            ρθ″_predᵏ = ρθ″[i, j, k] + Δτ * Gˢρθ[i, j, k] - Δτ * div_Jθᵥρ″_h -
                        backward_weight / Δzᶜ * (θᵥ_face⁺ * Δτᵋ * ρw″⁺ - θᵥ_faceᵏ * Δτᵋ * ρw″ᵏ)

            ρθ″_predictor[i, j, k] = ρθ″_predᵏ
            ρ″_predictor[i, j, k]  = ρ″_predᵏ
        end

        ## ── MPAS Section 5: explicit ρw″ predictor update ──
        ## After this loop, ρw″[i, j, k] for k = 2..Nz holds the explicit predictor
        ## that the BatchedTridiagonalSolver will use as its right-hand side.
        buoy_coeff_raw = buoyancy_coefficient(g)
        for k in 2:Nz
            ρθ″_predᵏ = ρθ″_predictor[i, j, k]
            ρθ″_pred⁻ = ρθ″_predictor[i, j, k - 1]
            ρ″_predᵏ  = ρ″_predictor[i, j, k]
            ρ″_pred⁻  = ρ″_predictor[i, j, k - 1]

            ρθ″_oldᵏ = ρθ″[i, j, k]
            ρθ″_old⁻ = ρθ″[i, j, k - 1]
            ρ″_oldᵏ  = ρ″[i, j, k]
            ρ″_old⁻  = ρ″[i, j, k - 1]

            pgf_coeffᵏ      = acoustic_pgf_coefficient(i, j, k, grid, pressure, γRᵈ, pˢᵗ, κ) * Δτᵋ
            buoy_coeffᵏ     = buoy_coeff_raw * Δτᵋ
            buoy_lin_coeffᵏ = buoyancy_linearization_coefficient(i, j, k,     grid, pressure, ρ₀, Π₀_field, ρθ_stage, pˢᵗ, κ, rcv, g) * Δτᵋ
            buoy_lin_coeff⁻ = buoyancy_linearization_coefficient(i, j, k - 1, grid, pressure, ρ₀, Π₀_field, ρθ_stage, pˢᵗ, κ, rcv, g) * Δτᵋ

            ρw″[i, j, k] = _explicit_ρw″_face_update(ρw″[i, j, k],
                                                     Δτ, Gˢρw_total[i, j, k],
                                                     pgf_coeffᵏ, buoy_coeffᵏ,
                                                     buoy_lin_coeffᵏ, buoy_lin_coeff⁻,
                                                     ρθ″_predᵏ, ρθ″_pred⁻,
                                                     ρ″_predᵏ, ρ″_pred⁻,
                                                     ρθ″_oldᵏ, ρθ″_old⁻,
                                                     ρ″_oldᵏ, ρ″_old⁻,
                                                     backward_weight)
        end
    end
end

##### Post-solve diagnostics: substitute the new ρw″ into the ρ″ and ρθ″
##### predictors to recover the new ρ″ and ρθ″ at cell centers.
@kernel function _post_acoustic_solve_diagnostics!(ρ″, ρθ″, ρw″,
                                                    ρθ″_predictor, ρ″_predictor,
                                                    grid, Δτᵋ,
                                                    θᵥ)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        for k in 1:Nz
            Δzᶜ    = Δzᶜᶜᶜ(i, j, k, grid)
            cofrzᵏ = Δτᵋ / Δzᶜ
            Jθ⁺    = θᵥ_at_face(i, j, k + 1, grid, θᵥ) * Δτᵋ
            Jθᵏ    = θᵥ_at_face(i, j, k,     grid, θᵥ) * Δτᵋ

            ρ″_predᵏ  = ρ″_predictor[i, j, k]
            ρθ″_predᵏ = ρθ″_predictor[i, j, k]

            ρ″[i, j, k]  = ρ″_predᵏ  - cofrzᵏ * (ρw″[i, j, k + 1] - ρw″[i, j, k])
            ρθ″[i, j, k] = ρθ″_predᵏ - (1 / Δzᶜ) * (Jθ⁺ * ρw″[i, j, k + 1] -
                                                    Jθᵏ * ρw″[i, j, k])
        end
    end
end

##### MPAS-style direct ρθ recovery: ρθ_new = ρθ⁰ + ρθ″.
##### Density from θ⁺ = θⁿ + Δt_stage Gˢθ, then ρ = ρθ / θ⁺.

@kernel function _mpas_recovery_wsrk3!(ρ, ρχ, ρθ″, ρ″,
                                        θᵥ, Gˢρχ, Gˢρ,
                                        ρ⁰, ρχ⁰, Δt_stage)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # ρθ_new = ρθ⁰ + ρθ″ (direct, no EOS conversion needed)
        ρχ⁰_ijk = ρχ⁰[i, j, k]
        ρχ⁺ = ρχ⁰_ijk + ρθ″[i, j, k]
        ρχ[i, j, k] = ρχ⁺

        # ρ_new = ρ⁰ + ρ″ (direct from MPAS acoustic density perturbation)
        ρ[i, j, k] = ρ⁰[i, j, k] + ρ″[i, j, k]
    end
end

##### Convert ρw″ (momentum perturbation) to velocity w.
##### MPAS recovery for w (line 3331-3334):
#####   ρw_new(k) = ρw⁰(k) + ρw″(k)
#####   w(k) = ρw_new(k) / (fzm·ρ(k) + fzp·ρ(k-1))
@kernel function _convert_ρw″_to_w!(w, ρw″, ρw⁰, ρ, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρᶜᶜᶠ_safe = ifelse(ρᶜᶜᶠ == 0, one(ρᶜᶜᶠ), ρᶜᶜᶠ)
        ρw_new = ρw⁰[i, j, k] + ρw″[i, j, k]
        w[i, j, k] = ρw_new / ρᶜᶜᶠ_safe * (k > 1)
    end
end


#####
##### Section 9: WS-RK3 substep loop
#####

# Stage substep count and size for ProportionalSubsteps:
# Δτ is the same in every stage; Nτ scales with the WS-RK3 stage fraction.
# For β = (1/3, 1/2, 1) and N a multiple of 6 this gives N/3, N/2, N substeps.
@inline function _stage_substep_count_and_size(::ProportionalSubsteps, β_stage, Δt, N)
    Δτ = Δt / N
    Nτ = max(1, round(Int, β_stage * N))
    return Nτ, Δτ
end

# Stage substep count and size for MonolithicFirstStage:
# Stage 1 collapses to a single substep of size Δt/3 (matching MPAS-A
# `config_time_integration_order = 3`); stages 2 and 3 are identical to
# ProportionalSubsteps. Stage 1 is identified by β_stage being closer to
# 1/3 than to 1/2 — a robust comparison that avoids any Float ↔ Rational
# round-trip in the inner loop.
@inline function _stage_substep_count_and_size(::MonolithicFirstStage, β_stage, Δt, N)
    if β_stage < (1//3 + 1//2) / 2   # β_stage is the canonical 1/3
        return 1, Δt / 3
    else
        Δτ = Δt / N
        Nτ = max(1, round(Int, β_stage * N))
        return Nτ, Δτ
    end
end

"""
$(TYPEDSIGNATURES)

Execute one Wicker–Skamarock RK3 stage of the MPAS conservative-perturbation
acoustic substep loop. The number and size of substeps in this stage depend
on `substepper.substep_distribution`:

  - [`ProportionalSubsteps`](@ref) (default): every stage uses
    ``Δτ = Δt/N`` and ``Nτ = \\max(\\mathrm{round}(β N), 1)`` substeps
    (so for ``β = 1/3, 1/2, 1`` this gives ``N/3``, ``N/2``, ``N`` substeps).
  - [`MonolithicFirstStage`](@ref): stage 1 collapses to a single substep
    of size ``Δt/3``; stages 2 and 3 are the same as `ProportionalSubsteps`.

``N`` is rounded up to a multiple of 6 so that ``N/3`` and ``N/2`` are both
integers.
"""
function acoustic_rk3_substep_loop!(model, substepper, Δt, β_stage, U⁰)
    grid = model.grid
    arch = architecture(grid)
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity

    # Compute substep count (adaptive when substeps === nothing). Round up to
    # a multiple of 6 so that N/3 (stage 1, ProportionalSubsteps) and N/2
    # (stage 2) are both integers.
    N_raw = acoustic_substeps(substepper.substeps, grid, Δt, model.thermodynamic_constants)
    N = max(6, 6 * cld(N_raw, 6))

    # Stage substep count and size — dispatched on the AcousticSubstepDistribution
    # type carried by the substepper.
    Nτ, Δτ = _stage_substep_count_and_size(substepper.substep_distribution, β_stage, Δt, N)

    # Convert slow tendencies to velocity/pressure form.
    # MPAS: tend_w_euler (vertical PGF + buoyancy) is computed ONLY at rk_step=1
    # Uses U⁰ for the linearized pp PGF+buoyancy (§5-8). Since U⁰ is frozen
    # across all stages, tend_w_euler and tend_u_euler are the same at every stage.
    convert_slow_tendencies!(substepper, model, U⁰)

    # No halo fill on `Gˢρw_total`: the substep kernels only
    # read these face fields at the same `(i, j, k)` as the kernel index, so the
    # halo values are never consumed.

    # Reset perturbation variables at each stage start.
    # MPAS accumulates across stages (only resets at stage 1), but this requires
    # the accumulated perturbation to be consistent with the velocity reset and
    # slow tendency re-evaluation. Per-stage reset is the standard WS-RK3 approach.
    fill!(parent(substepper.ρθ″_predictor), 0)
    fill!(parent(substepper.ρ″_predictor),  0)
    fill!(parent(substepper.previous_ρθ″), 0)
    fill!(parent(substepper.ρ″), 0)
    fill!(parent(substepper.ρθ″), 0)
    fill!(parent(substepper.ρw″), 0)
    fill!(parent(substepper.ρu″), 0)
    fill!(parent(substepper.ρv″), 0)

    u = model.velocities.u
    v = model.velocities.v
    w = model.velocities.w

    # WS-RK3: reset velocities to Uⁿ (U⁰) at the start of each stage.
    # Each stage computes U_new = U⁰ + β·Δt·R(eval_state), so the acoustic
    # loop must start from U⁰ velocities — not the previous stage's result.
    # The slow velocity tendencies (computed above from the evaluation state)
    # are added as forcing during the acoustic substeps.
    launch!(arch, grid, :xyz, _reset_velocities_to_U0!,
            u, v, w, U⁰[2], U⁰[3], U⁰[4], U⁰[1], grid)

    # Fill halos for all stage-frozen fields read with horizontal neighbor access
    fill_halo_regions!(u)
    fill_halo_regions!(v)

    ω = substepper.forward_weight

    Gⁿ = model.timestepper.Gⁿ
    χ_name = thermodynamic_density_name(model.formulation)
    Gˢρθ = getproperty(Gⁿ, χ_name)
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    g = model.thermodynamic_constants.gravitational_acceleration
    FT = eltype(grid)
    ε = FT(2 * ω - 1)                              # MPAS off-centering parameter
    Δτᵋ = FT(0.5) * FT(Δτ) * (1 + ε)            # off-centered effective substep (MPAS dtseps)
    backward_weight = (1 - ε) / (1 + ε)           # MPAS resm

    # Constants needed by the inline coefficient helpers in the column kernel.
    κ      = FT(Rᵈ / cᵖᵈ)        # R / cp
    rcv    = FT(Rᵈ / (cᵖᵈ - Rᵈ)) # R / cv
    γRᵈ    = FT(cᵖᵈ * rcv)       # = cp · R/cv = γ·Rᵈ, the Klemp 2007 PGF prefactor
    pˢᵗ_FT = FT(model.dynamics.standard_pressure)

    ref = model.dynamics.reference_state
    ρ₀    = ref isa Nothing ? model.dynamics.density              : ref.density
    Π₀_field = ref isa Nothing ? substepper.reference_exner_function : ref.exner_function

    for substep in 1:Nτ
        # Step 0: pre-PGF filter — projection-style strategies write
        # `damping.ρθ″_for_pgf` from the current `(ρθ)″` and the previous
        # substep's snapshot. Default strategies are no-ops.
        apply_pgf_filter!(substepper.damping, substepper, model, U⁰[5])

        # Step 1: Horizontal forward — update u, v from PGF and slow tendency.
        # MPAS conservative-perturbation form: the horizontal PGF reads the
        # accumulated (ρθ)″ perturbation,
        #   ∂x_p″ = γRᵈ · Πᶠᶜᶜ · ∂x(ρθ″)
        # which provides horizontal acoustic coupling through the accumulated
        # (ρθ)″ field. Projection strategies replace the PGF source with the
        # filtered scratch field returned by `pgf_source_field`.
        pˢᵗ = model.dynamics.standard_pressure
        ρθ_for_pgf = pgf_source_field(substepper.damping, substepper)
        launch!(arch, grid, :xyz, _mpas_horizontal_forward!,
                substepper.ρu″, substepper.ρv″, grid, Δτ,
                ρθ_for_pgf, substepper.frozen_pressure,
                Gⁿ.ρu, Gⁿ.ρv,
                cᵖᵈ, Rᵈ, pˢᵗ)

        # Fill halos after horizontal forward: ρu″/ρv″ were updated in the
        # interior; the column kernel reads ρu″[i+1] via div_xyᶜᶜᶜ.
        # u, v are not touched during substeps (ERF-style momentum-only), so
        # their halos stay fresh from the stage-start `_reset_velocities_to_U0!`.
        fill_halo_regions!(substepper.ρu″)
        fill_halo_regions!(substepper.ρv″)

        # Save ρθ″ before substep for divergence damping (δ_τρθ″ computation)
        ρθ″_old = substepper.previous_ρθ″
        parent(ρθ″_old) .= parent(substepper.ρθ″)

        # Steps 2-5: build θflux/mflux scratches and the explicit ρw″ predictor.
        # The result lives in ρw″[i, j, k] for k = 2..Nz; ρw″[1] and ρw″[Nz+1]
        # remain at the boundary value 0.
        launch!(arch, grid, :xy, _build_acoustic_rhs!,
                substepper.ρw″, substepper.ρ″, substepper.ρθ″,
                substepper.ρθ″_predictor, substepper.ρ″_predictor,
                substepper.ρu″, substepper.ρv″,
                grid, FT(Δτ), Δτᵋ, FT(backward_weight), ε,
                substepper.Gˢρw_total, Gⁿ.ρ, Gˢρθ,
                substepper.virtual_potential_temperature, model.dynamics.density,
                substepper.frozen_pressure, ρ₀, Π₀_field, U⁰[5],
                γRᵈ, pˢᵗ_FT, κ, rcv, FT(g),
                substep == 1)

        # Step 6: BatchedTridiagonalSolver. The coefficients (a, b, c) are
        # computed on the fly via the AcousticTridiag* tag types' get_coefficient
        # dispatch (see top of this file). Pass the per-substep state through
        # `args...` so the dispatch can read it without rebuilding the solver.
        # In-place solve: ρw″ is both the RHS and the output. The Thomas forward
        # sweep reads f[k] before writing ϕ[k] at each iteration, so this is safe.
        solve!(substepper.ρw″, substepper.vertical_solver, substepper.ρw″,
               substepper.frozen_pressure, ρ₀, Π₀_field, U⁰[5],
               substepper.virtual_potential_temperature,
               γRᵈ, pˢᵗ_FT, κ, rcv, FT(g), Δτᵋ)

        # Step 8: post-solve diagnostics — recover ρ″ and ρθ″ from the new ρw″.
        launch!(arch, grid, :xy, _post_acoustic_solve_diagnostics!,
                substepper.ρ″, substepper.ρθ″, substepper.ρw″,
                substepper.ρθ″_predictor, substepper.ρ″_predictor,
                grid, Δτᵋ,
                substepper.virtual_potential_temperature)

        # Fill ρθ″ halos before divergence damping reads δx(ρθ″_new - ρθ″_old).
        # The column kernel updated interior ρθ″ but not halos; stale halos create
        # a spurious gradient at periodic boundaries that feeds back into ρu″.
        fill_halo_regions!(substepper.ρθ″)

        # Divergence damping — strategy is dispatched on `substepper.damping`.
        apply_divergence_damping!(substepper.damping, substepper, grid, FT(Δτ))

        # MPAS halo exchanges (lines 1279-1322): communicate ρ″, ρθ″, ρu″
        # after each substep so the next substep's horizontal forward step
        # and horizontal flux use up-to-date neighbor values.
        fill_halo_regions!(substepper.ρθ″)
        fill_halo_regions!(substepper.ρ″)
        fill_halo_regions!(substepper.ρu″)
        fill_halo_regions!(substepper.ρv″)
    end

    # MPAS recovery order: ρ and ρθ first, then w from recovered density.
    # This matches MPAS lines 3319-3334: rho_zz updated before w = rw/rho_face.
    Δt_stage = Nτ * Δτ
    ρχ = thermodynamic_density(model.formulation)
    Gⁿ = model.timestepper.Gⁿ
    χ_name = thermodynamic_density_name(model.formulation)
    Gˢρχ = getproperty(Gⁿ, χ_name)

    launch!(arch, grid, :xyz, _mpas_recovery_wsrk3!,
            model.dynamics.density, ρχ,
            substepper.ρθ″, substepper.ρ″,
            substepper.virtual_potential_temperature, Gˢρχ, Gⁿ.ρ,
            U⁰[1], U⁰[5], Δt_stage)

    # ERF-style momentum recovery: ρu_new = ρu⁰ + ρu″, then u_new = ρu_new / ρ_new.
    # Replaces the two-step MPAS-style recovery (ρw″→w via ρ_new, then ρu = ρ·u
    # from substep-updated velocities), which required maintaining u, v in parallel
    # with ρu″, ρv″ inside the substep loop.
    launch!(arch, grid, :xyz, _erf_recover_momentum_and_velocity!,
            model.momentum, model.velocities,
            substepper.ρu″, substepper.ρv″, substepper.ρw″,
            U⁰[2], U⁰[3], U⁰[4],
            model.dynamics.density, grid)

    return nothing
end


@kernel function _reset_velocities_to_U0!(u, v, w, ρu⁰, ρv⁰, ρw⁰, ρ⁰, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ⁰)
        u[i, j, k] = ρu⁰[i, j, k] / ρᶠᶜᶜ * !on_x_boundary(i, j, k, grid)

        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ⁰)
        v[i, j, k] = ρv⁰[i, j, k] / ρᶜᶠᶜ * !on_y_boundary(i, j, k, grid)

        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ⁰)
        w[i, j, k] = ρw⁰[i, j, k] / ρᶜᶜᶠ * (k > 1)
    end
end


@kernel function _recover_momentum!(m, ρ, vel, grid)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        m.ρu[i, j, k] = ℑxᶠᵃᵃ(i, j, k, grid, ρ) * vel.u[i, j, k]
        m.ρv[i, j, k] = ℑyᵃᶠᵃ(i, j, k, grid, ρ) * vel.v[i, j, k]
        m.ρw[i, j, k] = ℑzᵃᵃᶠ(i, j, k, grid, ρ) * vel.w[i, j, k]
    end
end

# ERF-style recovery: full momentum is the direct sum of stage-start momentum
# and the accumulated perturbation. Velocity is then diagnosed from the new
# momentum and the already-updated density. Replaces the MPAS-style pair
# (`_convert_ρw″_to_w!` + `_recover_momentum!`) which multiplied the
# substep-updated velocities back by the new density.
@kernel function _erf_recover_momentum_and_velocity!(m, vel,
                                                     ρu″, ρv″, ρw″,
                                                     ρu⁰, ρv⁰, ρw⁰,
                                                     ρ, grid)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρu_new = ρu⁰[i, j, k] + ρu″[i, j, k]
        ρv_new = ρv⁰[i, j, k] + ρv″[i, j, k]
        ρw_new = ρw⁰[i, j, k] + ρw″[i, j, k]

        m.ρu[i, j, k] = ρu_new
        m.ρv[i, j, k] = ρv_new
        m.ρw[i, j, k] = ρw_new

        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρᶠᶜᶜ_safe = ifelse(ρᶠᶜᶜ == 0, one(ρᶠᶜᶜ), ρᶠᶜᶜ)
        ρᶜᶠᶜ_safe = ifelse(ρᶜᶠᶜ == 0, one(ρᶜᶠᶜ), ρᶜᶠᶜ)
        ρᶜᶜᶠ_safe = ifelse(ρᶜᶜᶠ == 0, one(ρᶜᶜᶠ), ρᶜᶜᶠ)

        vel.u[i, j, k] = ρu_new / ρᶠᶜᶜ_safe * !on_x_boundary(i, j, k, grid)
        vel.v[i, j, k] = ρv_new / ρᶜᶠᶜ_safe * !on_y_boundary(i, j, k, grid)
        vel.w[i, j, k] = ρw_new / ρᶜᶜᶠ_safe * (k > 1)
    end
end
