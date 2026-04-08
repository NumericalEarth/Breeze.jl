#####
##### Acoustic Substepping for CompressibleDynamics — Exner Pressure Formulation
#####
##### Implements split-explicit time integration following CM1 (Bryan 2002),
##### Wicker-Skamarock (2002), and Klemp et al. (2007):
##### - Forward-backward acoustic substeps with (velocity, Exner pressure) variables
##### - Vertically implicit w-π coupling with off-centering (always on)
##### - Forward-extrapolation filter (ϰᵈⁱ) on the pressure variable
##### - Constant acoustic substep size Δτ = Δt/N across all RK stages
##### - Topology-aware operators (no halo filling between substeps)
#####

using KernelAbstractions: @kernel, @index

using Oceananigans: CenterField, XFaceField, YFaceField, ZFaceField, architecture
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!
using Oceananigans.Operators:
    ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ,
    ℑxᶠᵃᵃ, ℑxᶜᵃᵃ, ℑyᵃᶠᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ,
    δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ,
    δxTᶠᵃᵃ, δyTᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ,
    divᶜᶜᶜ, div_xyᶜᶜᶜ,
    Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ,
    Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶜ, Δzᶜᶜᶠ,
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
##### Section 2: AcousticSubstepper struct (Exner pressure formulation)
#####

"""
    AcousticSubstepper

Storage and parameters for acoustic substepping using the Exner pressure
formulation, following CM1's `sound.F`.

The acoustic loop uses velocity (u, v, w) and Exner pressure perturbation (π')
as prognostic variables, forming a stable 2-variable system that avoids the
density-buoyancy instability of the (ρu, ρ, ρθ) formulation.

The forward-backward scheme updates:
1. **Forward**: Velocity from Exner pressure gradient: ``u += Δτ (u_{ten} - cᵖ θᵥ ∂π'_d/∂x)``
2. **Backward**: Exner pressure from velocity divergence: ``π' += Δτ (π_{ten} - S ∇·u) + w_{terms}``
3. **Implicit**: Vertically implicit w-π' coupling (tridiagonal solve)
4. **Filtering**: ``π̃' = π' + ϰᵈⁱ (π' - π'_{old})``

Fields
======

- `substeps`: Number of acoustic substeps for the full time step (Δt)
- `forward_weight`: Off-centering parameter ω (default 0.55, → ε = 2ω−1 = 0.1, MPAS default)
- `divergence_damping_coefficient`: Forward-extrapolation filter ϰᵈⁱ for π' (default 0.10)
- `acoustic_damping_coefficient`: Klemp 2018 ϰᵃᶜ for velocity damping
- `virtual_potential_temperature`: Stage-frozen θᵥ (CenterField, MPAS `theta_m`/`t`)
- `reference_exner_function`: Reference π₀ = (p_ref/pˢᵗ)^(R/cᵖ) (CenterField, MPAS `pb`)
- `theta_flux_scratch`: ts accumulator in the column kernel (CenterField, MPAS `ts`)
- `mass_flux_scratch`: rs accumulator in the column kernel (CenterField, MPAS `rs`)
- `previous_rtheta_pp`: (ρθ)″ snapshot before the column kernel — used by divergence damping (CenterField)
- `ρ″`:  acoustic ρ perturbation (CenterField, MPAS `rho_pp`)
- `ρθ″`: acoustic ρθ perturbation (CenterField, MPAS `rtheta_pp`)
- `ρw″`: acoustic ρw perturbation (ZFaceField, MPAS `rw_p`)
- `ρu″`: acoustic ρu perturbation (XFaceField, MPAS `ru_p`)
- `ρv″`: acoustic ρv perturbation (YFaceField, MPAS `rv_p`)
- `gamma_tri`: Thomas sweep scratch in the column kernel (ZFaceField)
- `averaged_velocities`: Time-averaged velocities for scalar advection (NamedTuple of u, v, w fields)
- `slow_tendencies`: Frozen slow tendencies (velocity, exner_pressure). Momentum tendencies
  are stored in the outer timestepper's `Gⁿ` fields; density and thermodynamic density
  tendencies are also read directly from `Gⁿ`.
- `vertical_solver`: BatchedTridiagonalSolver for the implicit ρw″ acoustic update
- `frozen_pressure`: Snapshot of `model.dynamics.pressure` taken once per outer step.
  Used as the linearization point for `acoustic_pgf_coefficient` and
  `buoyancy_linearization_coefficient` so that the cofwz/cofwt coefficients seen by
  the substepper are *frozen across all WS-RK3 stages of an outer step*. This
  matches MPAS's behavior: MPAS stores `exner` in the diagnostics pool and only
  recomputes it at `rk_step==3` (end of outer step), so the substepper's
  linearization point is identical at every stage. Without this snapshot, Breeze's
  `update_state!` (called between stages) would update `model.dynamics.pressure`
  to the post-stage value, mismatching MPAS and introducing per-outer-step
  numerical drift.

The `cofwz`, `cofwr`, `cofwt`, `coftz` MPAS coefficients are computed inline by
helper functions inside the column kernel — no fields are stored.
"""
struct AcousticSubstepper{N, FT, AD, CF, FF, XF, YF, GT, AV, ST, TS}
    substeps :: N
    forward_weight :: FT                       # Off-centering ω → epssm = 2ω - 1
    divergence_damping_coefficient :: FT
    acoustic_damping_coefficient :: FT
    substep_distribution :: AD                 # ProportionalSubsteps or MonolithicFirstStage
    virtual_potential_temperature :: CF        # Stage-frozen θ_m (MPAS `t`)
    reference_exner_function :: CF             # π₀ from reference state
    theta_flux_scratch :: CF                   # ts_scratch in column kernel
    mass_flux_scratch :: CF                    # rs_scratch in column kernel
    previous_rtheta_pp :: CF                   # (ρθ)″ snapshot for divergence damping
    ρ″  :: CF                                  # MPAS rho_pp     — acoustic ρ perturbation
    ρθ″ :: CF                                  # MPAS rtheta_pp  — acoustic ρθ perturbation
    ρw″ :: FF                                  # MPAS rw_p       — acoustic ρw perturbation (z-face, with momentum BCs)
    ρu″ :: XF                                  # MPAS ru_p       — acoustic ρu perturbation (x-face, with momentum BCs)
    ρv″ :: YF                                  # MPAS rv_p       — acoustic ρv perturbation (y-face, with momentum BCs)
    gamma_tri :: GT                            # LU decomp scratch (z-face, default BCs)
    averaged_velocities :: AV
    slow_tendencies :: ST
    vertical_solver :: TS                      # BatchedTridiagonalSolver for implicit ρw″ update
    frozen_pressure :: CF                      # Snapshot of model.dynamics.pressure at outer-step start
end

function _adapt_slow_tendencies(to, st)
    return (velocity = map(f -> adapt(to, f), st.velocity),
            exner_pressure = adapt(to, st.exner_pressure))
end

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.substeps,
                       a.forward_weight,
                       a.divergence_damping_coefficient,
                       a.acoustic_damping_coefficient,
                       a.substep_distribution,
                       adapt(to, a.virtual_potential_temperature),
                       adapt(to, a.reference_exner_function),
                       adapt(to, a.theta_flux_scratch),
                       adapt(to, a.mass_flux_scratch),
                       adapt(to, a.previous_rtheta_pp),
                       adapt(to, a.ρ″),
                       adapt(to, a.ρθ″),
                       adapt(to, a.ρw″),
                       adapt(to, a.ρu″),
                       adapt(to, a.ρv″),
                       adapt(to, a.gamma_tri),
                       map(f -> adapt(to, f), a.averaged_velocities),
                       _adapt_slow_tendencies(to, a.slow_tendencies),
                       adapt(to, a.vertical_solver),
                       adapt(to, a.frozen_pressure))

"""
$(TYPEDSIGNATURES)

Construct an `AcousticSubstepper` using the Exner pressure formulation.

The optional `prognostic_momentum` keyword carries the prognostic ρu/ρv/ρw
fields whose boundary conditions are inherited by the substepper's
perturbation face fields. This is essential on grids with Bounded horizontal
topology so that `fill_halo_regions!` enforces impenetrability (v=0 at the
south/north walls, u=0 at the east/west walls) on the perturbation momenta.
Without this, the substepper's halo fills use default Periodic/NoFlux BCs and
boundary cells drift away from zero.
"""
function AcousticSubstepper(grid, split_explicit::SplitExplicitTimeDiscretization;
                            prognostic_momentum = nothing)
    Ns = split_explicit.substeps
    FT = eltype(grid)
    ω = convert(FT, split_explicit.forward_weight)
    ϰᵈⁱ = convert(FT, split_explicit.divergence_damping_coefficient)
    ϰᵃᶜ = convert(FT, split_explicit.acoustic_damping_coefficient)
    substep_distribution = split_explicit.substep_distribution

    virtual_potential_temperature = CenterField(grid)
    reference_exner_function = CenterField(grid)
    theta_flux_scratch = CenterField(grid)
    mass_flux_scratch = CenterField(grid)
    previous_rtheta_pp = CenterField(grid)

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

    averaged_velocities = (u = XFaceField(grid),
                           v = YFaceField(grid),
                           w = ZFaceField(grid))

    slow_tendencies = (velocity = (u = XFaceField(grid),
                                   v = YFaceField(grid),
                                   w = ZFaceField(grid)),
                       exner_pressure = CenterField(grid))

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

    return AcousticSubstepper(Ns, ω, ϰᵈⁱ, ϰᵃᶜ,
                              substep_distribution,
                              virtual_potential_temperature,
                              reference_exner_function,
                              theta_flux_scratch,
                              mass_flux_scratch,
                              previous_rtheta_pp,
                              ρ″, ρθ″, ρw″, ρu″, ρv″,
                              gamma_tri_field,
                              averaged_velocities,
                              slow_tendencies,
                              vertical_solver,
                              frozen_pressure)
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

Uses a conservative sound speed estimate `ℂᵃᶜ = √(γ Rᵈ Tᵣ)` with `Tᵣ = 300 K`
(giving `ℂᵃᶜ ≈ 347 m/s`) and the minimum horizontal grid spacing. The vertical
CFL is not needed because the w-π' coupling is vertically implicit.

Following CM1, the substep count satisfies `Δτ · ℂᵃᶜ / Δx_min ≤ 1` where
`Δτ = Δt / N` is the acoustic substep size. A safety factor of 1.2 is applied
to ensure stability with the forward-backward splitting.
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

    safety_factor = 1.2
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

Prepare the acoustic cache for an RK stage.

Computes stage-frozen coefficients for the Exner pressure acoustic loop:
1. Virtual potential temperature θᵥ (frozen during acoustic loop)
2. Pressure tendency coefficient S = c²/(cᵖ ρ₀ θᵥ²)
3. Exner pressure perturbation π' = (p/pˢᵗ)^(R/cᵖ) - π₀
4. Reference Exner function π₀ from the reference state

Following CM1's `sound.F`, the acoustic loop prognostics velocity and
Exner pressure perturbation, with density diagnosed from the equation
of state after the loop.
"""
function prepare_acoustic_cache!(substepper, model)
    grid = model.grid
    arch = architecture(grid)

    # Compute stage-frozen θᵥ (the only field this routine still produces).
    pˢᵗ = model.dynamics.standard_pressure
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖ  # R/cp

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

    # Cache π₀ for use by `convert_slow_tendencies!`.
    _set_exner_reference!(substepper, model, model.dynamics.reference_state)

    fill_halo_regions!(substepper.virtual_potential_temperature)

    return nothing
end

@kernel function _prepare_virtual_theta!(θᵥ_field, ρ, p, T, specific_prognostic_moisture, grid,
                                          microphysics, microphysical_fields, constants, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρⁱ = ρ[i, j, k]
        pⁱ = p[i, j, k]
        Tⁱ = T[i, j, k]
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
    end

    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρⁱ, qᵛᵉ, microphysical_fields)
    # (mixture properties currently unused; kept here in case future moist closures need them)
    _ = mixture_gas_constant(q, constants)
    _ = mixture_heat_capacity(q, constants)

    # Virtual potential temperature: θᵥ = T / π where π = (p/pˢᵗ)^κ
    πⁱ = (pⁱ / pˢᵗ)^κ
    @inbounds θᵥ_field[i, j, k] = Tⁱ / πⁱ
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
- §5: pp(k) = Rᵈ (Π(k) ρθ_p(k) + ρθ_base(k) (Π(k) - Π_base(k)))
- §6: dpdz(k) = -g (ρ(k) - ρ_base(k))
- §7: tend_w_euler = -rdzu (pp_k - pp_{k-1}) + fzm dpdz_k + fzp dpdz_{k-1}
- §8: tend_u_euler = -(pp(i,j,k) - pp(i-1,j,k)) / Δx

All computed from U⁰ (step-start state), frozen across all RK stages.
"""
function convert_slow_tendencies!(substepper, model, U⁰)
    grid = model.grid
    arch = architecture(grid)
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    g = model.thermodynamic_constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖᵈ
    Gⁿ = model.timestepper.Gⁿ
    pˢᵗ = model.dynamics.standard_pressure

    ref = model.dynamics.reference_state
    ρᵣ = ref isa Nothing ? model.dynamics.density : ref.density
    pᵣ = ref isa Nothing ? model.dynamics.pressure : ref.pressure

    launch!(arch, grid, :xyz, _convert_slow_tendencies!,
            substepper.slow_tendencies.velocity.u,
            substepper.slow_tendencies.velocity.v,
            substepper.slow_tendencies.velocity.w,
            substepper.slow_tendencies.exner_pressure,
            Gⁿ.ρu, Gⁿ.ρv, Gⁿ.ρw,
            model.dynamics.density,
            model.velocities.u,
            model.velocities.v,
            model.velocities.w,
            substepper.theta_flux_scratch,
            substepper.reference_exner_function,
            substepper.virtual_potential_temperature,
            grid, κ, cᵖᵈ, g,
            U⁰.ρθ, U⁰.ρ, pˢᵗ, ρᵣ, pᵣ)

    return nothing
end

##### Compute pp (linearized perturbation pressure, §5) at a cell center.
##### pp(k) = Rᵈ (Π(k) ρθ_p(k) + ρθ_base(k) (Π(k) - Π_base(k)))
@inline function linearized_pp(i, j, k, ρθ⁰, πᵣ, pᵣ, Rᵈ, rcv, pˢᵗ)
    @inbounds begin
        Π_base = πᵣ[i, j, k]
        Π_base_safe = ifelse(Π_base == 0, one(Π_base), Π_base)
        ρθ_base = pᵣ[i, j, k] / (Rᵈ * Π_base_safe)
        ρθ_p = ρθ⁰[i, j, k] - ρθ_base
        Π = (Rᵈ * ρθ⁰[i, j, k] / pˢᵗ)^rcv
        return Rᵈ * (Π * ρθ_p + ρθ_base * (Π - Π_base))
    end
end

@kernel function _convert_slow_tendencies!(Gˢu, Gˢv, Gˢw, Gˢπ,
                                           Gˢρu, Gˢρv, Gˢρw,
                                           ρ, u, v, w,
                                           π′, πᵣ, θᵥ,
                                           grid, κ, cᵖᵈ, g,
                                           ρθ⁰, ρ⁰, pˢᵗ, ρᵣ, pᵣ)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)
    Rᵈ = κ * cᵖᵈ
    rcv = κ / (1 - κ)   # Rᵈ/cᵥ

    @inbounds begin
        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)

        # Horizontal tendencies: Gˢu = Gˢρu/ρ, Gˢv = Gˢρv/ρ.
        # Gˢρu from dynamics includes the FULL horizontal PGF (∂p/∂x) which is
        # recomputed at each RK stage from the current state's pressure.
        # This matches MPAS where tend_u_euler is recomputed per stage.
        Gˢu[i, j, k] = Gˢρu[i, j, k] / ρᶠᶜᶜ * !on_x_boundary(i, j, k, grid)
        Gˢv[i, j, k] = Gˢρv[i, j, k] / ρᶜᶠᶜ * !on_y_boundary(i, j, k, grid)

        # ── §5-7: Vertical PGF + buoyancy from linearized pp and dpdz ──
        # Gˢρw from dynamics has vertical PGF + buoyancy ZEROED (SplitExplicit).
        # Compute tend_w_euler from U⁰ (frozen across stages, matching MPAS rk_step=1).
        if k > 1
            pp_k = linearized_pp(i, j, k, ρθ⁰, πᵣ, pᵣ, Rᵈ, rcv, pˢᵗ)
            pp_km1 = linearized_pp(i, j, k - 1, ρθ⁰, πᵣ, pᵣ, Rᵈ, rcv, pˢᵗ)

            # §7: PGF = rdzu * (pp_k - pp_{k-1})
            Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
            pgf = (pp_k - pp_km1) / Δzᶠ

            # §6: dpdz = -g * (ρ⁰ - ρ_base) at centers
            dpdz_k = -g * (ρ⁰[i, j, k] - ρᵣ[i, j, k])
            dpdz_km1 = -g * (ρ⁰[i, j, k - 1] - ρᵣ[i, j, k - 1])

            # §7: tend_w_euler = -pgf + fzm*dpdz_k + fzp*dpdz_{k-1}
            # fzm = Δz_below / total, fzp = Δz_above / total (MPAS convention)
            Δzᶜ_above = Δzᶜᶜᶜ(i, j, k, grid)
            Δzᶜ_below = Δzᶜᶜᶜ(i, j, k - 1, grid)
            fzm = Δzᶜ_below / (Δzᶜ_above + Δzᶜ_below)
            fzp = Δzᶜ_above / (Δzᶜ_above + Δzᶜ_below)
            tend_w_euler = -pgf + fzm * dpdz_k + fzp * dpdz_km1

            # Total: Gˢw = advection/ρ + tend_w_euler/ρ
            Gˢw[i, j, k] = Gˢρw[i, j, k] / ρᶜᶜᶠ + tend_w_euler / ρᶜᶜᶠ
        else
            Gˢw[i, j, k] = zero(eltype(Gˢw))
        end

        # ── Slow Exner pressure tendency: Gˢπ = -u · ∇π ──
        # Use topology-safe operators: ℑxᶜᵃᵃ(∂xᶠᶜᶜ(π')) gives the centered
        # gradient (f[i+1]-f[i-1])/(2Δx) on uniform grids; automatically zero for Flat.
        uᶜ = ℑxᶜᵃᵃ(i, j, k, grid, u)
        ∂π_∂x = ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, π′)

        vᶜ = ℑyᵃᶜᵃ(i, j, k, grid, v)
        ∂π_∂y = ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, π′)

        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)
        π_above = ifelse(k == Nz, πᵣ[i, j, k] + π′[i, j, k],
                         πᵣ[i, j, k + 1] + π′[i, j, k + 1])
        π_below = ifelse(k == 1, πᵣ[i, j, k] + π′[i, j, k],
                         πᵣ[i, j, k - 1] + π′[i, j, k - 1])
        wᶜ = ifelse(k == 1, w[i, j, k + 1] / 2,
              ifelse(k == Nz, w[i, j, k] / 2,
                     (w[i, j, k] + w[i, j, k + 1]) / 2))
        ∂π_∂z = (π_above - π_below) / (2 * Δzᶜ)

        Gˢπ[i, j, k] = -(uᶜ * ∂π_∂x + vᶜ * ∂π_∂y + wᶜ * ∂π_∂z)
    end
end

#####
##### MPAS-style horizontal forward step using ρθ perturbation PGF.
##### MPAS: u += dts * (Gˢu - c2 * Π_face * ∂(rtheta_pp)/∂x * cqw / zz_face)
##### For dry air, no terrain: cqw=1, zz=1.

# Exner function at a cell center, for interpolation via function composition.
@inline _exner_from_p(i, j, k, grid, p, pˢᵗ, κ) = (p[i, j, k] / pˢᵗ)^κ

@kernel function _mpas_horizontal_forward!(u, v, ru_p, rv_p, grid, Δτ,
                                            rtheta_pp, pressure, ρ,
                                            Gˢu, Gˢv,
                                            cₚ, Rᵈ, pˢᵗ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        rcv = Rᵈ / (cₚ - Rᵈ)
        c2 = cₚ * rcv  # cp * R/cv = γR_d (Eq. 3.25)
        κ = Rᵈ / cₚ

        # MPAS horizontal momentum update (line 2808-2812):
        #   pgrad = c2 * Π_avg * Δ(rtheta_pp) / dcEdge
        #   ru_p += dts * (tend_ru - pgrad)
        # Exner interpolated to faces via function composition (topology-safe).
        Π_u_face = ℑxᶠᵃᵃ(i, j, k, grid, _exner_from_p, pressure, pˢᵗ, κ)
        ∂x_ρθ = ∂xᶠᶜᶜ(i, j, k, grid, rtheta_pp)
        pgrad_u = c2 * Π_u_face * ∂x_ρθ

        ρ_u_face = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        ρ_u_safe = ifelse(ρ_u_face == 0, one(ρ_u_face), ρ_u_face)
        not_bdy_x = !on_x_boundary(i, j, k, grid)
        u[i, j, k] += Δτ * (Gˢu[i, j, k] - pgrad_u / ρ_u_safe) * not_bdy_x
        ru_p[i, j, k] += Δτ * (ρ_u_face * Gˢu[i, j, k] - pgrad_u) * not_bdy_x

        Π_v_face = ℑyᵃᶠᵃ(i, j, k, grid, _exner_from_p, pressure, pˢᵗ, κ)
        ∂y_ρθ = ∂yᶜᶠᶜ(i, j, k, grid, rtheta_pp)
        pgrad_v = c2 * Π_v_face * ∂y_ρθ

        ρ_v_face = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        ρ_v_safe = ifelse(ρ_v_face == 0, one(ρ_v_face), ρ_v_face)
        not_bdy_y = !on_y_boundary(i, j, k, grid)
        v[i, j, k] += Δτ * (Gˢv[i, j, k] - pgrad_v / ρ_v_safe) * not_bdy_y
        rv_p[i, j, k] += Δτ * (ρ_v_face * Gˢv[i, j, k] - pgrad_v) * not_bdy_y
    end
end

##### MPAS-style ts/rtheta_pp tracking for gravity-wave coupling
#####
##### ts accumulates the total ρθ perturbation at each substep:
#####   ts = rtheta_pp_old + Δτ Gˢρθ - θ_m Δτ div_h(u)
#####
##### After the w solve, rtheta_pp is updated:
#####   rtheta_pp_new = ts - θ_m Δτ (w_new(k+1) - w_new(k)) / Δz
#####
##### The cofwt term uses ts to provide the gravity-wave restoring force.
#####

##### MPAS divergence damping (Eq. 3.45, Klemp, Skamarock & Ha 2018)
##### Applied after each acoustic substep to V''_h (horizontal momentum perturbation):
#####   V''_h += (γ_D Δx / Θ^t_m) δ_τ Θ''_m
##### where δ_τ Θ'' = (rtheta_pp_new - rtheta_pp_old) / Δτ is the acoustic θ tendency.
##### Using δ_τ Θ'' as the divergence proxy ensures numerical consistency with the
##### discrete pressure equation, preventing corruption of gravity wave frequencies.

##### MPAS divergence damping (Eq. 3.45, atm_divergence_damping_3d)
##### Applied after each acoustic substep to V''_h (horizontal momentum perturbation):
#####   ru_p += coef * (rtheta_pp_new - rtheta_pp_old) * dvEdge / (dcEdge * 2 * θ_m_edge)
##### where coef = 2 * smdiv * len_disp / dts.
##### Using δ_τ Θ'' as the divergence proxy ensures numerical consistency with the
##### discrete pressure equation (Klemp, Skamarock & Ha 2018).

##### Inline helpers for function composition with topology-safe operators.

# Exner perturbation change per substep (for legacy divergence damping).
@inline _Δπ(i, j, k, grid, π′, π′⁻) = π′[i, j, k] - π′⁻[i, j, k]

##### Divergence proxy for MPAS damping: divCell = -(rtheta_pp - rtheta_pp_old)
@inline _neg_δΘ(i, j, k, grid, rtheta_pp, rtheta_pp_old) =
    -(rtheta_pp[i, j, k] - rtheta_pp_old[i, j, k])

@kernel function _mpas_divergence_damping!(ru_p, rv_p,
                                            rtheta_pp, rtheta_pp_old, θ_m,
                                            grid, coef_div_damp)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # MPAS (lines 3059-3062): divCell = -(rtheta_pp_new - rtheta_pp_old)
        # ru_p += coef_divdamp * δx(divCell) / (θ_cell1 + θ_cell2)
        # Uses topology-safe operators: ∂xᶠᶜᶜ returns 0 for Flat x.

        # x-direction: gradient of divΘ at u-face, divided by θ sum
        ∂x_divΘ = δxᶠᵃᵃ(i, j, k, grid, _neg_δΘ, rtheta_pp, rtheta_pp_old)
        θ_sum_u = 2 * ℑxᶠᵃᵃ(i, j, k, grid, θ_m)
        θ_sum_u_safe = ifelse(θ_sum_u == 0, one(θ_sum_u), θ_sum_u)
        ru_p[i, j, k] += coef_div_damp * ∂x_divΘ / θ_sum_u_safe *
                          !on_x_boundary(i, j, k, grid)

        # y-direction: gradient of divΘ at v-face, divided by θ sum
        ∂y_divΘ = δyᵃᶠᵃ(i, j, k, grid, _neg_δΘ, rtheta_pp, rtheta_pp_old)
        θ_sum_v = 2 * ℑyᵃᶠᵃ(i, j, k, grid, θ_m)
        θ_sum_v_safe = ifelse(θ_sum_v == 0, one(θ_sum_v), θ_sum_v)
        rv_p[i, j, k] += coef_div_damp * ∂y_divΘ / θ_sum_v_safe *
                          !on_y_boundary(i, j, k, grid)
    end
end

##### MPAS acoustic substep: verbatim translation of Sections 3-8.
##### Area-weighted θ fluxes for the topology-safe divergence computation.
@inline Ax_θ_ru(i, j, k, grid, θ_m, ru_p) = Axᶠᶜᶜ(i, j, k, grid) * ℑxᶠᵃᵃ(i, j, k, grid, θ_m) * ru_p[i, j, k]
@inline Ay_θ_rv(i, j, k, grid, θ_m, rv_p) = Ayᶜᶠᶜ(i, j, k, grid) * ℑyᵃᶠᵃ(i, j, k, grid, θ_m) * rv_p[i, j, k]

#####
##### Inline tridiagonal coefficients for the MPAS acoustic substep.
#####
##### These return the per-unit-Δτₛ value of each MPAS coefficient at a single
##### face/center point. The runtime substep kernel multiplies by Δτₛ where
##### needed. They replace the cofwz/cofwr/cofwt/coftz fields that were
##### previously precomputed and cached on the substepper.
#####

# Vertical-face Δz fraction weights (MPAS fzm, fzp).
# fzm(k) = Δz_below(k) / [Δz_above(k) + Δz_below(k)] — weight on center k
# fzp(k) = Δz_above(k) / [Δz_above(k) + Δz_below(k)] — weight on center k-1
@inline function _face_z_weights(i, j, k, grid)
    Δzᶜ_above = Δzᶜᶜᶜ(i, j, k, grid)
    Δzᶜ_below = Δzᶜᶜᶜ(i, j, k - 1, grid)
    inv_total = 1 / (Δzᶜ_above + Δzᶜ_below)
    fzm = Δzᶜ_below * inv_total
    fzp = Δzᶜ_above * inv_total
    return fzm, fzp
end

# Acoustic PGF coefficient at face k (MPAS cofwz / dtseps):
#   cofwz(k) = c² × Δzᶠ⁻¹ × Π_face(k)
# where Π_face = fzm Π(k) + fzp Π(k-1).
@inline function acoustic_pgf_coefficient(i, j, k, grid, pressure, c², pˢᵗ, κ)
    Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
    fzm, fzp = _face_z_weights(i, j, k, grid)
    Πₖ   = (pressure[i, j, k]     / pˢᵗ)^κ
    Πₖ₋₁ = (pressure[i, j, k - 1] / pˢᵗ)^κ
    Π_face = fzm * Πₖ + fzp * Πₖ₋₁
    return c² / Δzᶠ * Π_face
end

# Buoyancy / gravity coefficient at face k (MPAS cofwr / dtseps).
# With zz=1 (no terrain) this collapses to the trivial constant g/2.
@inline buoyancy_coefficient(g) = g / 2

# θ-flux coefficient at face k (MPAS coftz / dtseps):
#   coftz(k) = fzm θ(k) + fzp θ(k-1)
# Returns 0 at the bottom face (k=1) and the top face (k=Nz+1) so that the
# kernel can call this helper unconditionally even at boundary indices.
@inline function theta_flux_coefficient(i, j, k, grid, θᵥ)
    Nz = size(grid, 3)
    in_interior = (k >= 2) & (k <= Nz)
    k_safe = ifelse(in_interior, k, 2)
    fzm, fzp = _face_z_weights(i, j, k_safe, grid)
    val = fzm * θᵥ[i, j, k_safe] + fzp * θᵥ[i, j, k_safe - 1]
    return ifelse(in_interior, val, zero(val))
end

# Buoyancy linearization coefficient at center k (MPAS cofwt / dtseps):
#   cofwt(k) = (R/cᵥ)/2 × g × ρ_base(k) × Π(k) / [ρθ(k) × Π_base(k)]
@inline function buoyancy_linearization_coefficient(i, j, k, grid,
                                                     pressure, ρ_base, exner_base,
                                                     ρθ_stage, pˢᵗ, κ, rcv, g)
    ρθ = ρθ_stage[i, j, k]
    ρθ_safe = ifelse(ρθ == 0, one(ρθ), ρθ)
    Π_base = exner_base[i, j, k]
    Π_base_safe = ifelse(Π_base == 0, one(Π_base), Π_base)
    Πₖ = (pressure[i, j, k] / pˢᵗ)^κ
    return rcv / 2 * g * ρ_base[i, j, k] * Πₖ / (ρθ_safe * Π_base_safe)
end

#####
##### Inline helpers used by the column kernel below.
#####

# Explicit (forward) update for ρw″ at face k. Combines:
#   - slow tendency Δτ ρ_face Gˢw
#   - acoustic θ-difference  (pgf_coeff)
#   - gravity-density       (buoy_coeff)
#   - linearized buoyancy    (buoy_lin_coeff)
@inline function _explicit_ρw″_face_update(ρw″_old_k,
                                           Δτ, ρ_face_k, Gˢw_k,
                                           pgf_coeff_k, buoy_coeff_k,
                                           buoy_lin_coeff_k, buoy_lin_coeff_km1,
                                           θflux_k, θflux_km1,
                                           mflux_k, mflux_km1,
                                           ρθ″_old_k, ρθ″_old_km1,
                                           ρ″_old_k, ρ″_old_km1,
                                           backward_weight)
    return ρw″_old_k + Δτ * ρ_face_k * Gˢw_k -
           pgf_coeff_k  * ((θflux_k - θflux_km1) +
                           backward_weight * (ρθ″_old_k - ρθ″_old_km1)) -
           buoy_coeff_k * ((mflux_k + mflux_km1) +
                           backward_weight * (ρ″_old_k + ρ″_old_km1)) +
           buoy_lin_coeff_k   * (θflux_k   + backward_weight * ρθ″_old_k) +
           buoy_lin_coeff_km1 * (θflux_km1 + backward_weight * ρθ″_old_km1)
end

# Tridiagonal coefficients (a, b, c) at face k.
# Names follow the (a, b, c) Thomas-algorithm convention; see Doc C for the full
# Schur-complement derivation.
@inline function _tridiag_a_at_face(pgf_coeff_k, buoy_coeff_k, buoy_lin_coeff_km1,
                                    θflux_coeff_km1, cofrz_km1, rdzw_below)
    return -pgf_coeff_k * θflux_coeff_km1 * rdzw_below +
            buoy_coeff_k * cofrz_km1 -
            buoy_lin_coeff_km1 * θflux_coeff_km1 * rdzw_below
end

@inline function _tridiag_b_at_face(pgf_coeff_k, buoy_coeff_k,
                                    buoy_lin_coeff_k, buoy_lin_coeff_km1,
                                    θflux_coeff_k, cofrz_k, cofrz_km1,
                                    rdzw_above, rdzw_below)
    return 1 +
           pgf_coeff_k * (θflux_coeff_k * rdzw_above + θflux_coeff_k * rdzw_below) -
           θflux_coeff_k * (buoy_lin_coeff_k * rdzw_above - buoy_lin_coeff_km1 * rdzw_below) +
           buoy_coeff_k * (cofrz_k - cofrz_km1)
end

@inline function _tridiag_c_at_face(pgf_coeff_k, buoy_coeff_k, buoy_lin_coeff_k,
                                    θflux_coeff_kp1, cofrz_k, rdzw_above)
    return -pgf_coeff_k * θflux_coeff_kp1 * rdzw_above -
            buoy_coeff_k * cofrz_k +
            buoy_lin_coeff_k * θflux_coeff_kp1 * rdzw_above
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
# here as the variadic tail. Order: pressure, ρ_base, exner_base, ρθ_stage,
# θᵥ, c², pˢᵗ, κ, rcv, g, Δτᵋ.

import Oceananigans.Solvers: get_coefficient

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagLower, p, ::ZDirection,
                                 pressure, ρ_base, exner_base, ρθ_stage, θᵥ,
                                 c², pˢᵗ, κ, rcv, g, Δτᵋ)
    # Lower at face k_face = k + 1
    k_face = k + 1
    Δzᶜ_below = Δzᶜᶜᶜ(i, j, k_face - 1, grid)
    rdzw_below = 1 / Δzᶜ_below
    cofrz_km1 = Δτᵋ * rdzw_below

    pgf_coeff_k       = acoustic_pgf_coefficient(i, j, k_face, grid, pressure, c², pˢᵗ, κ) * Δτᵋ
    buoy_coeff_k      = buoyancy_coefficient(g) * Δτᵋ
    buoy_lin_coeff_km1 = buoyancy_linearization_coefficient(i, j, k_face - 1, grid,
                                                             pressure, ρ_base, exner_base,
                                                             ρθ_stage, pˢᵗ, κ, rcv, g) * Δτᵋ
    θflux_coeff_km1   = theta_flux_coefficient(i, j, k_face - 1, grid, θᵥ) * Δτᵋ

    return _tridiag_a_at_face(pgf_coeff_k, buoy_coeff_k, buoy_lin_coeff_km1,
                              θflux_coeff_km1, cofrz_km1, rdzw_below)
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagDiagonal, p, ::ZDirection,
                                 pressure, ρ_base, exner_base, ρθ_stage, θᵥ,
                                 c², pˢᵗ, κ, rcv, g, Δτᵋ)
    # Bottom-boundary row: trivial b = 1, paired with f[1] = 0 → ρw″[1] = 0.
    k == 1 && return one(c²)

    # Otherwise face = k, build the diagonal at face k.
    k_face = k
    Δzᶜ_above = Δzᶜᶜᶜ(i, j, k_face,     grid)
    Δzᶜ_below = Δzᶜᶜᶜ(i, j, k_face - 1, grid)
    rdzw_above = 1 / Δzᶜ_above
    rdzw_below = 1 / Δzᶜ_below
    cofrz_k    = Δτᵋ * rdzw_above
    cofrz_km1  = Δτᵋ * rdzw_below

    pgf_coeff_k        = acoustic_pgf_coefficient(i, j, k_face, grid, pressure, c², pˢᵗ, κ) * Δτᵋ
    buoy_coeff_k       = buoyancy_coefficient(g) * Δτᵋ
    buoy_lin_coeff_k   = buoyancy_linearization_coefficient(i, j, k_face,     grid,
                                                             pressure, ρ_base, exner_base,
                                                             ρθ_stage, pˢᵗ, κ, rcv, g) * Δτᵋ
    buoy_lin_coeff_km1 = buoyancy_linearization_coefficient(i, j, k_face - 1, grid,
                                                             pressure, ρ_base, exner_base,
                                                             ρθ_stage, pˢᵗ, κ, rcv, g) * Δτᵋ
    θflux_coeff_k      = theta_flux_coefficient(i, j, k_face, grid, θᵥ) * Δτᵋ

    return _tridiag_b_at_face(pgf_coeff_k, buoy_coeff_k,
                              buoy_lin_coeff_k, buoy_lin_coeff_km1,
                              θflux_coeff_k, cofrz_k, cofrz_km1,
                              rdzw_above, rdzw_below)
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagUpper, p, ::ZDirection,
                                 pressure, ρ_base, exner_base, ρθ_stage, θᵥ,
                                 c², pˢᵗ, κ, rcv, g, Δτᵋ)
    # Bottom-boundary row: c[1] must be 0 so the back-substitution preserves
    # ρw″[1] = 0. (γ_1 = c[1] / β_1 = 0/1 = 0; ρw″[1] -= γ_1 * ρw″[2] = 0.)
    k == 1 && return zero(c²)

    # Otherwise face = k, build the upper at face k.
    k_face = k
    Δzᶜ_above = Δzᶜᶜᶜ(i, j, k_face, grid)
    rdzw_above = 1 / Δzᶜ_above
    cofrz_k    = Δτᵋ * rdzw_above

    pgf_coeff_k        = acoustic_pgf_coefficient(i, j, k_face, grid, pressure, c², pˢᵗ, κ) * Δτᵋ
    buoy_coeff_k       = buoyancy_coefficient(g) * Δτᵋ
    buoy_lin_coeff_k   = buoyancy_linearization_coefficient(i, j, k_face, grid,
                                                             pressure, ρ_base, exner_base,
                                                             ρθ_stage, pˢᵗ, κ, rcv, g) * Δτᵋ
    θflux_coeff_kp1    = theta_flux_coefficient(i, j, k_face + 1, grid, θᵥ) * Δτᵋ

    return _tridiag_c_at_face(pgf_coeff_k, buoy_coeff_k, buoy_lin_coeff_k,
                              θflux_coeff_kp1, cofrz_k, rdzw_above)
end

##### This kernel does ONE complete substep for ONE column (i,j).
##### Launched with :xy worksize. Sequential k-loops match MPAS exactly.

##### Builds the explicit ρw″ predictor (the right-hand side of the tridiagonal)
##### in place on ρw″ at faces k = 2..Nz, and writes the θflux/mflux scratches
##### at all centers k = 1..Nz. Does NOT do the Thomas sweep — that step now
##### lives in the BatchedTridiagonalSolver call back in the substep loop.
@kernel function _build_acoustic_rhs!(ρw″, ρ″, ρθ″,
                                       θflux_scratch, mflux_scratch,
                                       ρu″, ρv″,
                                       grid, Δτ, Δτᵋ, backward_weight, ε,
                                       Gˢw, Gˢρ, Gˢρθ,
                                       θᵥ, ρ,
                                       pressure, ρ_base, exner_base, ρθ_stage,
                                       c², pˢᵗ, κ, rcv, g,
                                       ū, inv_Nτ,
                                       is_first_substep)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        ## ── MPAS Section 3: Initialize on first substep ──
        if is_first_substep
            for k in 1:Nz
                ρ″[i, j, k] = 0
                ρθ″[i, j, k] = 0
                ρw″[i, j, k] = 0
            end
            ρw″[i, j, Nz + 1] = 0
        end

        ## ── MPAS Section 4: accumulate θflux and mflux into the column scratch ──
        for k in 1:Nz
            V = Vᶜᶜᶜ(i, j, k, grid)

            mass_flux_div  = div_xyᶜᶜᶜ(i, j, k, grid, ρu″, ρv″)
            theta_flux_div = (δxᶜᵃᵃ(i, j, k, grid, Ax_θ_ru, θᵥ, ρu″) +
                              δyᵃᶜᵃ(i, j, k, grid, Ay_θ_rv, θᵥ, ρv″)) / V

            mflux_k = -Δτ * mass_flux_div
            θflux_k = -Δτ * theta_flux_div

            Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)
            cofrz_k = Δτᵋ / Δzᶜ

            ρw″_top = ρw″[i, j, k + 1]
            ρw″_bot = ρw″[i, j, k]

            mflux_k = ρ″[i, j, k] + Δτ * Gˢρ[i, j, k] + mflux_k -
                      cofrz_k * backward_weight * (ρw″_top - ρw″_bot)

            θflux_top = theta_flux_coefficient(i, j, k + 1, grid, θᵥ)
            θflux_bot = theta_flux_coefficient(i, j, k,     grid, θᵥ)
            θflux_k = ρθ″[i, j, k] + Δτ * Gˢρθ[i, j, k] + θflux_k -
                      backward_weight / Δzᶜ * (θflux_top * Δτᵋ * ρw″_top - θflux_bot * Δτᵋ * ρw″_bot)

            θflux_scratch[i, j, k] = θflux_k
            mflux_scratch[i, j, k] = mflux_k
        end

        ## ── MPAS Section 5: time-averaged w (pre-solve) + explicit w update ──
        ## After this loop, ρw″[i, j, k] for k = 2..Nz holds the explicit predictor
        ## that the BatchedTridiagonalSolver will use as its right-hand side.
        buoy_coeff_raw = buoyancy_coefficient(g)
        for k in 2:Nz
            ū.w[i, j, k] = ū.w[i, j, k] + (1 - ε) / 2 * ρw″[i, j, k] * inv_Nτ

            θflux_k   = θflux_scratch[i, j, k]
            θflux_km1 = θflux_scratch[i, j, k - 1]
            mflux_k   = mflux_scratch[i, j, k]
            mflux_km1 = mflux_scratch[i, j, k - 1]

            ρθ″_old_k   = ρθ″[i, j, k]
            ρθ″_old_km1 = ρθ″[i, j, k - 1]
            ρ″_old_k    = ρ″[i, j, k]
            ρ″_old_km1  = ρ″[i, j, k - 1]

            pgf_coeff_k        = acoustic_pgf_coefficient(i, j, k, grid, pressure, c², pˢᵗ, κ) * Δτᵋ
            buoy_coeff_k       = buoy_coeff_raw * Δτᵋ
            buoy_lin_coeff_k   = buoyancy_linearization_coefficient(i, j, k,     grid, pressure, ρ_base, exner_base, ρθ_stage, pˢᵗ, κ, rcv, g) * Δτᵋ
            buoy_lin_coeff_km1 = buoyancy_linearization_coefficient(i, j, k - 1, grid, pressure, ρ_base, exner_base, ρθ_stage, pˢᵗ, κ, rcv, g) * Δτᵋ

            ρ_face_k = (ρ[i, j, k] + ρ[i, j, k - 1]) / 2
            ρw″[i, j, k] = _explicit_ρw″_face_update(ρw″[i, j, k],
                                                     Δτ, ρ_face_k, Gˢw[i, j, k],
                                                     pgf_coeff_k, buoy_coeff_k,
                                                     buoy_lin_coeff_k, buoy_lin_coeff_km1,
                                                     θflux_k, θflux_km1,
                                                     mflux_k, mflux_km1,
                                                     ρθ″_old_k, ρθ″_old_km1,
                                                     ρ″_old_k, ρ″_old_km1,
                                                     backward_weight)
        end
    end
end

##### Post-solve diagnostics: substitute the new ρw″ back into the mass and θ
##### flux equations to recover ρ″ and ρθ″, and accumulate the post-solve
##### contribution to the time-averaged w.
@kernel function _post_acoustic_solve_diagnostics!(ρ″, ρθ″, ρw″,
                                                    θflux_scratch, mflux_scratch,
                                                    grid, Δτᵋ, ε,
                                                    θᵥ, ū, inv_Nτ)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        for k in 2:Nz
            ū.w[i, j, k] = ū.w[i, j, k] + (1 + ε) / 2 * ρw″[i, j, k] * inv_Nτ
        end

        for k in 1:Nz
            Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)
            cofrz_k         = Δτᵋ / Δzᶜ
            θflux_coeff_kp1 = theta_flux_coefficient(i, j, k + 1, grid, θᵥ) * Δτᵋ
            θflux_coeff_k   = theta_flux_coefficient(i, j, k,     grid, θᵥ) * Δτᵋ

            mflux_k = mflux_scratch[i, j, k]
            θflux_k = θflux_scratch[i, j, k]

            ρ″[i, j, k]  = mflux_k - cofrz_k * (ρw″[i, j, k + 1] - ρw″[i, j, k])
            ρθ″[i, j, k] = θflux_k - (1 / Δzᶜ) * (θflux_coeff_kp1 * ρw″[i, j, k + 1] -
                                                  θflux_coeff_k   * ρw″[i, j, k])
        end
    end
end

##### MPAS-style direct ρθ recovery: ρθ_new = ρθ⁰ + rtheta_pp.
##### Density from θ⁺ = θⁿ + Δt_stage Gˢθ, then ρ = ρθ / θ⁺.

@kernel function _mpas_recovery_wsrk3!(ρ, ρχ, rtheta_pp, rho_pp,
                                        θᵥ, Gˢρχ, Gˢρ,
                                        ρ⁰, ρχ⁰, Δt_stage)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # ρθ_new = ρθ⁰ + rtheta_pp (direct, no EOS conversion needed)
        ρχ⁰_ijk = ρχ⁰[i, j, k]
        ρχ⁺ = ρχ⁰_ijk + rtheta_pp[i, j, k]
        ρχ[i, j, k] = ρχ⁺

        # ρ_new = ρ⁰ + rho_pp (direct from MPAS acoustic density perturbation)
        ρ[i, j, k] = ρ⁰[i, j, k] + rho_pp[i, j, k]
    end
end

##### Convert rw_p (momentum perturbation) to velocity w.
##### MPAS recovery for w (line 3331-3334):
#####   rw(k) = rw_save(k) + rw_p(k)
#####   w(k) = rw(k) / (fzm*rho_zz(k) + fzp*rho_zz(k-1))
##### rw_save is the vertical momentum at step start (ρw⁰ from U⁰).
@kernel function _convert_rw_p_to_w!(w, rw_p, ρw⁰, ρ, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρᶠ_safe = ifelse(ρᶠ == 0, one(ρᶠ), ρᶠ)
        rw_total = ρw⁰[i, j, k] + rw_p[i, j, k]
        w[i, j, k] = rw_total / ρᶠ_safe * (k > 1)
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

Execute the acoustic substep loop for a Wicker-Skamarock RK3 stage using
the Exner pressure formulation. The number and size of substeps in this
stage depends on `substepper.substep_distribution`:

  - [`ProportionalSubsteps`](@ref) (default): every stage uses
    ``Δτ = Δt/N`` and ``Nτ = \\max(\\mathrm{round}(β N), 1)`` substeps
    (so for β = 1/3, 1/2, 1 this gives N/3, N/2, N substeps).
  - [`MonolithicFirstStage`](@ref): stage 1 collapses to a single substep
    of size ``Δt/3``; stages 2 and 3 are the same as `ProportionalSubsteps`.

`N` is rounded up to a multiple of 6 so that N/3 and N/2 are both integers.
"""
function acoustic_rk3_substep_loop!(model, substepper, Δt, β_stage, U⁰)
    grid = model.grid
    arch = architecture(grid)
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity

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

    # Fill halos for slow tendencies (read by horizontal forward at boundary faces)
    fill_halo_regions!(substepper.slow_tendencies.velocity.u)
    fill_halo_regions!(substepper.slow_tendencies.velocity.v)
    fill_halo_regions!(substepper.slow_tendencies.velocity.w)
    fill_halo_regions!(substepper.slow_tendencies.exner_pressure)

    # Initialize time-averaged velocities to zero
    ū = substepper.averaged_velocities
    launch!(arch, grid, :xyz, _zero_avg_velocities!, ū)

    # Reset perturbation variables at each stage start.
    # MPAS accumulates across stages (only resets at stage 1), but this requires
    # the accumulated perturbation to be consistent with the velocity reset and
    # slow tendency re-evaluation. Per-stage reset is the standard WS-RK3 approach.
    fill!(parent(substepper.theta_flux_scratch), 0)
    fill!(parent(substepper.mass_flux_scratch), 0)
    fill!(parent(substepper.previous_rtheta_pp), 0)
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
    ω̄ = 1 - ω
    ϰᵈⁱ = substepper.divergence_damping_coefficient
    ϰᵃᶜ = substepper.acoustic_damping_coefficient

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
    κ = FT(Rᵈ / cᵖ)             # R / cp
    rcv = FT(Rᵈ / (cᵖ - Rᵈ))    # R / cv
    c² = FT(cᵖ * rcv)           # cp R / cv
    pˢᵗ_FT = FT(model.dynamics.standard_pressure)

    ref = model.dynamics.reference_state
    ρ_base    = ref isa Nothing ? model.dynamics.density              : ref.density
    exner_base = ref isa Nothing ? substepper.reference_exner_function : ref.exner_function

    for substep in 1:Nτ
        # Step 1: Horizontal forward — update u, v from PGF and slow tendency.
        # MPAS uses the ρθ perturbation gradient for the horizontal PGF:
        #   pgf_x = -c2 * Π_face * ∂(rtheta_pp)/∂x
        # NOT the Exner perturbation gradient. This provides horizontal
        # acoustic coupling through the accumulated ρθ perturbation.
        pˢᵗ = model.dynamics.reference_state.standard_pressure
        launch!(arch, grid, :xyz, _mpas_horizontal_forward!,
                u, v, substepper.ρu″, substepper.ρv″, grid, Δτ,
                substepper.ρθ″,
                substepper.frozen_pressure, model.dynamics.density,
                substepper.slow_tendencies.velocity.u,
                substepper.slow_tendencies.velocity.v,
                cᵖ, Rᵈ, pˢᵗ)

        # Fill halos after horizontal forward: ru_p/rv_p were updated in the
        # interior; the column kernel reads ru_p[i+1] via div_xyᶜᶜᶜ.
        # Also fill u/v halos since the column kernel may indirectly need them.
        fill_halo_regions!(substepper.ρu″)
        fill_halo_regions!(substepper.ρv″)
        fill_halo_regions!(u)
        fill_halo_regions!(v)

        # Save rtheta_pp before substep for divergence damping (δ_τΘ'' computation)
        rtheta_pp_old = substepper.previous_rtheta_pp
        parent(rtheta_pp_old) .= parent(substepper.ρθ″)

        # Steps 2-5: build θflux/mflux scratches and the explicit ρw″ predictor.
        # The result lives in ρw″[i, j, k] for k = 2..Nz; ρw″[1] and ρw″[Nz+1]
        # remain at the boundary value 0.
        launch!(arch, grid, :xy, _build_acoustic_rhs!,
                substepper.ρw″, substepper.ρ″, substepper.ρθ″,
                substepper.theta_flux_scratch, substepper.mass_flux_scratch,
                substepper.ρu″, substepper.ρv″,
                grid, FT(Δτ), Δτᵋ, FT(backward_weight), ε,
                substepper.slow_tendencies.velocity.w, Gⁿ.ρ, Gˢρθ,
                substepper.virtual_potential_temperature, model.dynamics.density,
                substepper.frozen_pressure, ρ_base, exner_base, U⁰[5],
                c², pˢᵗ_FT, κ, rcv, FT(g),
                ū, 1 / Nτ,
                substep == 1)

        # Step 6: BatchedTridiagonalSolver. The coefficients (a, b, c) are
        # computed on the fly via the AcousticTridiag* tag types' get_coefficient
        # dispatch (see top of this file). Pass the per-substep state through
        # `args...` so the dispatch can read it without rebuilding the solver.
        # In-place solve: ρw″ is both the RHS and the output. The Thomas forward
        # sweep reads f[k] before writing ϕ[k] at each iteration, so this is safe.
        solve!(substepper.ρw″, substepper.vertical_solver, substepper.ρw″,
               substepper.frozen_pressure, ρ_base, exner_base, U⁰[5],
               substepper.virtual_potential_temperature,
               c², pˢᵗ_FT, κ, rcv, FT(g), Δτᵋ)

        # Step 8: post-solve diagnostics — recover ρ″ and ρθ″ from the new ρw″
        # and accumulate the post-solve contribution to the time-averaged w.
        launch!(arch, grid, :xy, _post_acoustic_solve_diagnostics!,
                substepper.ρ″, substepper.ρθ″, substepper.ρw″,
                substepper.theta_flux_scratch, substepper.mass_flux_scratch,
                grid, Δτᵋ, ε,
                substepper.virtual_potential_temperature, ū, 1 / Nτ)

        # Fill rtheta_pp halos before divergence damping reads δx(rtheta_pp_new - rtheta_pp_old).
        # The column kernel updated interior rtheta_pp but not halos; stale halos create
        # a spurious gradient at periodic boundaries that feeds back into ru_p.
        fill_halo_regions!(substepper.ρθ″)

        # MPAS divergence damping (atm_divergence_damping_3d):
        #   coef = 2 * smdiv * len_disp / dts
        #   ru_p += coef * (δΘ_cell2 - δΘ_cell1) * dvEdge / (dcEdge * 2 * θ_edge)
        # config_smdiv = 0.1, config_len_disp = grid resolution.
        smdiv = FT(0.1)  # MPAS default
        # MPAS `config_len_disp` is a user-set scalar nominal grid resolution.
        # We derive it as the minimum horizontal cell spacing, skipping Flat
        # axes. On a 3D periodic-periodic grid this is min(Δx, Δy); on a
        # 2D periodic-flat grid it falls back to Δx. (Previously this used
        # minimum_yspacing on non-Flat-y grids, which on a RectilinearGrid
        # with Δy ≠ Δx made `coef_div_damp` mismatched and caused a
        # bottom-corner instability — see test_sk94_igw_3d.jl.)
        TX, TY, _ = topology(grid)
        Δx_eff = TX === Flat ? FT(Inf) : FT(minimum_xspacing(grid))
        Δy_eff = TY === Flat ? FT(Inf) : FT(minimum_yspacing(grid))
        len_disp_raw = min(Δx_eff, Δy_eff)
        len_disp = isfinite(len_disp_raw) ? len_disp_raw : FT(1)
        coef_div_damp = 2 * smdiv * len_disp / FT(Δτ)
        launch!(arch, grid, :xyz, _mpas_divergence_damping!,
                substepper.ρu″, substepper.ρv″,
                substepper.ρθ″, rtheta_pp_old,
                substepper.virtual_potential_temperature,
                grid, coef_div_damp)

        # MPAS halo exchanges (lines 1279-1322): communicate rho_pp, rtheta_pp,
        # ru_p after each substep so the next substep's horizontal forward step
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

    # Convert rw_p to w velocity: w = (ρw⁰ + rw_p) / ρ_recovered (MPAS line 3331-3334).
    launch!(arch, grid, :xyz, _convert_rw_p_to_w!,
            w, substepper.ρw″, U⁰[4], model.dynamics.density, grid)  # U⁰[4] = ρw⁰

    # Reconstruct momentum from updated density and velocity
    launch!(arch, grid, :xyz, _recover_momentum!,
            model.momentum, model.dynamics.density, model.velocities, grid)

    return nothing
end


@kernel function _zero_avg_velocities!(ū)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ū.u[i, j, k] = 0
        ū.v[i, j, k] = 0
        ū.w[i, j, k] = 0
    end
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

