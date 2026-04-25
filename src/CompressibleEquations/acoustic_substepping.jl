#####
##### Acoustic Substepping for CompressibleDynamics
#####
##### Baldauf-2010 framework: split-explicit Wicker–Skamarock RK3.
#####
##### Slow operator P_A = advection + Coriolis + closure (handled by
##### `compute_slow_*_tendencies!` in TimeSteppers/acoustic_substep_helpers.jl).
##### Fast operator P_F = PGF + buoyancy + acoustic damping, integrated by the
##### substep loop in this file. The fast operator is linearized about the
##### time-independent hydrostatic reference state (ρᵣ, θᵣ, Πᵣ) carried on
##### `model.dynamics.reference_state`.
#####
##### Substepper working variables (departures from reference):
#####   σ = ρ − ρᵣ                          (CenterField)
#####   η = ρθ − ρθᵣ  with ρθᵣ = ρᵣ θᵣ    (CenterField)
#####   ρu, ρv, ρw                          (face fields; reference momenta = 0)
#####
##### Pressure linearization (p depends on potential temperature, not on η):
#####   p = ρ R Πᵣ θ  ⇒  p′ ≈ ρᵣ R Πᵣ θ′   where θ′ = θ − θᵣ.
##### Approximating θ′ ≈ η/ρᵣ − θᵣ σ/ρᵣ (valid to first order at fixed Π) gives
#####   p′ ≈ R Πᵣ (η − θᵣ σ).
##### The pressure-gradient force on momentum is thus
#####   ∂_t (ρu) = -R Πᵣ ∂_x η + (slow advection forcing).
##### (We use the simpler `-R Πᵣ ∂η` PGF rather than the chain-rule γRΠ form,
##### because pressure depends on potential temperature, not on `ρθ` directly.)
#####
##### Equations integrated each substep:
#####   ∂σ/∂t  = -∇·M    + Gˢ_ρ
#####   ∂η/∂t  = -∇·(θᵣ M) + Gˢ_ρθ
#####   ∂ρu/∂t = -R Πᵣᶠᶜᶜ ∂x(η) + Gˢ_ρu
#####   ∂ρv/∂t = -R Πᵣᶜᶠᶜ ∂y(η) + Gˢ_ρv
#####   ∂ρw/∂t = -R Πᵣᶜᶜᶠ ∂z(η) − g σ      + Gˢ_ρw
##### where M = (ρu, ρv, ρw).
#####
##### Time discretization (forward-backward off-centered):
#####   horizontal forward Euler for ρu, ρv;
#####   implicit Schur tridiagonal in ρw with off-centering ω;
#####   damping (configurable via AcousticDampingStrategy) at each substep.
#####
##### Outer driver: WS-RK3 (acoustic_runge_kutta_3.jl) recomputes slow tendencies
##### each stage; nothing is frozen across stages (the linearization point is
##### already time-independent).
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
using Oceananigans.Grids: Periodic, Bounded, Flat, topology, minimum_xspacing, minimum_yspacing

using Adapt: Adapt, adapt

# Topology-safe boundary checks for momentum BCs.
@inline _x_topo(grid) = topology(grid)[1]
@inline _y_topo(grid) = topology(grid)[2]
@inline on_x_boundary(i, j, k, grid) = (_x_topo(grid) === Bounded) & ((i == 1) | (i == size(grid, 1) + 1))
@inline on_y_boundary(i, j, k, grid) = (_y_topo(grid) === Bounded) & ((j == 1) | (j == size(grid, 2) + 1))

#####
##### Substep distribution and damping types are defined in
##### `time_discretizations.jl` (included before this file). We use them here
##### for substepper construction and dispatch.
##### The damping field-name convention from time_discretizations.jl is
##### `ρθ″_for_pgf` (the perturbation ρθ for the PGF source); we re-use that
##### name even though our working perturbation in this file is η = ρθ - ρθᵣ
##### (departure from reference, not departure from stage state).
#####

#####
##### AcousticSubstepper — substep state and linearization-reference fields.
##### All reference fields are time-independent; populated once at construction.
#####

struct AcousticSubstepper{N, FT, D, AD, CF, FF, XF, YF, GT, TS}
    substeps :: N
    forward_weight :: FT                 # off-centering ω (1 = fully implicit, 0.5 = centered)
    damping :: D
    substep_distribution :: AD

    # Reference fields (time-independent, populated by `freeze_outer_step_state!`)
    reference_density :: CF              # ρᵣ
    reference_ρθ      :: CF              # ρθᵣ = ρᵣ θᵣ
    reference_pressure :: CF             # pᵣ
    reference_exner_function :: CF       # Πᵣ

    # Substep working state (perturbations from reference)
    σ  :: CF                             # σ = ρ - ρᵣ  (continuous in time, advanced each substep)
    η  :: CF                             # η = ρθ - ρθᵣ
    ρu :: XF                             # full ρu (since ρuᵣ = 0)
    ρv :: YF
    ρw :: FF

    # Predictor scratch (used inside the column kernel for the explicit part)
    σ_pred :: CF
    η_pred :: CF
    η_prev :: CF                         # previous-substep η for damping

    # Solver scratch
    gamma_tri :: GT                      # Thomas-sweep workspace
    vertical_solver :: TS
end

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.substeps,
                       a.forward_weight,
                       adapt(to, a.damping),
                       a.substep_distribution,
                       adapt(to, a.reference_density),
                       adapt(to, a.reference_ρθ),
                       adapt(to, a.reference_pressure),
                       adapt(to, a.reference_exner_function),
                       adapt(to, a.σ),
                       adapt(to, a.η),
                       adapt(to, a.ρu),
                       adapt(to, a.ρv),
                       adapt(to, a.ρw),
                       adapt(to, a.σ_pred),
                       adapt(to, a.η_pred),
                       adapt(to, a.η_prev),
                       adapt(to, a.gamma_tri),
                       adapt(to, a.vertical_solver))

# Adapt for damping types (concrete strategies).
Adapt.adapt_structure(to, d::NoDivergenceDamping) = d
Adapt.adapt_structure(to, d::ThermodynamicDivergenceDamping) = d
for T in (:PressureProjectionDamping, :ConservativeProjectionDamping)
    @eval Adapt.adapt_structure(to, d::$T{FT}) where FT =
        $T{FT, typeof(adapt(to, d.ρθ″_for_pgf))}(d.coefficient, adapt(to, d.ρθ″_for_pgf))
end

# Float-type promotion helpers for damping coefficients.
@inline _convert_damping(::Type, d::NoDivergenceDamping) = d
@inline _convert_damping(::Type{FT}, d::ThermodynamicDivergenceDamping) where FT =
    ThermodynamicDivergenceDamping(coefficient = convert(FT, d.coefficient),
                                   length_scale = d.length_scale === nothing ? nothing : convert(FT, d.length_scale))
@inline _convert_damping(::Type{FT}, d::PressureProjectionDamping) where FT =
    PressureProjectionDamping{FT, typeof(d.ρθ″_for_pgf)}(convert(FT, d.coefficient), d.ρθ″_for_pgf)
@inline _convert_damping(::Type{FT}, d::ConservativeProjectionDamping) where FT =
    ConservativeProjectionDamping{FT, typeof(d.ρθ″_for_pgf)}(convert(FT, d.coefficient), d.ρθ″_for_pgf)

# Materialization (allocate scratch fields per damping strategy).
@inline materialize_damping(grid, d::NoDivergenceDamping) = d
@inline materialize_damping(grid, d::ThermodynamicDivergenceDamping) = d
function materialize_damping(grid, d::PressureProjectionDamping{FT}) where FT
    ρθ″_for_pgf = CenterField(grid)
    return PressureProjectionDamping{FT, typeof(ρθ″_for_pgf)}(d.coefficient, ρθ″_for_pgf)
end
function materialize_damping(grid, d::ConservativeProjectionDamping{FT}) where FT
    ρθ″_for_pgf = CenterField(grid)
    return ConservativeProjectionDamping{FT, typeof(ρθ″_for_pgf)}(d.coefficient, ρθ″_for_pgf)
end

#####
##### Tridiagonal coefficient tag types (stateless).
#####

struct AcousticTridiagLower    end
struct AcousticTridiagDiagonal end
struct AcousticTridiagUpper    end

#####
##### AcousticSubstepper constructor.
#####

function AcousticSubstepper(grid, split_explicit;
                            prognostic_momentum = nothing)
    Ns = split_explicit.substeps
    FT = eltype(grid)
    ω  = convert(FT, split_explicit.forward_weight)

    damping = materialize_damping(grid, _convert_damping(FT, split_explicit.damping))
    substep_distribution = split_explicit.substep_distribution

    # Reference and working fields.
    ref_ρ  = CenterField(grid)
    ref_ρθ = CenterField(grid)
    ref_p  = CenterField(grid)
    ref_Π  = CenterField(grid)

    σ = CenterField(grid)
    η = CenterField(grid)

    # Inherit BCs from prognostic momentum so impenetrability is enforced.
    bcs_ρu = prognostic_momentum === nothing ? nothing : prognostic_momentum.ρu.boundary_conditions
    bcs_ρv = prognostic_momentum === nothing ? nothing : prognostic_momentum.ρv.boundary_conditions
    bcs_ρw = prognostic_momentum === nothing ? nothing : prognostic_momentum.ρw.boundary_conditions
    _xface(g, b) = b === nothing ? XFaceField(g) : XFaceField(g; boundary_conditions = b)
    _yface(g, b) = b === nothing ? YFaceField(g) : YFaceField(g; boundary_conditions = b)
    _zface(g, b) = b === nothing ? ZFaceField(g) : ZFaceField(g; boundary_conditions = b)

    ρu = _xface(grid, bcs_ρu)
    ρv = _yface(grid, bcs_ρv)
    ρw = _zface(grid, bcs_ρw)

    σ_pred = CenterField(grid)
    η_pred = CenterField(grid)
    η_prev = CenterField(grid)

    gamma_tri = ZFaceField(grid)

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
                              ref_ρ, ref_ρθ, ref_p, ref_Π,
                              σ, η, ρu, ρv, ρw,
                              σ_pred, η_pred, η_prev,
                              gamma_tri, vertical_solver)
end

#####
##### Idempotent population of reference fields. Called once at outer-step start
##### but reads the time-independent `model.dynamics.reference_state` so it can
##### be a no-op after the first call. Kept as a function for explicitness.
#####

function freeze_outer_step_state!(substepper::AcousticSubstepper, model)
    ref = model.dynamics.reference_state
    grid = model.grid
    arch = architecture(grid)
    if ref === nothing
        # No reference state: fall back to the current state. Acoustic
        # substepping isn't really meaningful in this configuration but we
        # avoid crashing.
        parent(substepper.reference_density)        .= parent(model.dynamics.density)
        parent(substepper.reference_pressure)       .= parent(model.dynamics.pressure)
        parent(substepper.reference_ρθ)             .= parent(thermodynamic_density(model.formulation))
        Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
        pˢᵗ = model.dynamics.standard_pressure
        cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
        κ = Rᵈ / cᵖᵈ
        launch!(arch, grid, :xyz, _populate_Π_from_p!,
                substepper.reference_exner_function,
                substepper.reference_pressure, pˢᵗ, κ)
    else
        parent(substepper.reference_density)        .= parent(ref.density)
        parent(substepper.reference_pressure)       .= parent(ref.pressure)
        parent(substepper.reference_exner_function) .= parent(ref.exner_function)
        Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
        launch!(arch, grid, :xyz, _populate_reference_ρθ!,
                substepper.reference_ρθ, ref.pressure, ref.exner_function, Rᵈ)
    end
    return nothing
end

freeze_outer_step_state!(substepper, model) = nothing

@kernel function _populate_reference_ρθ!(ρθᵣ, pᵣ, Πᵣ, Rᵈ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Π = Πᵣ[i, j, k]
        Π_safe = ifelse(Π == 0, one(Π), Π)
        # ρθᵣ = pᵣ / (Rᵈ Πᵣ) — from the dry-air EOS p = ρ R T = ρ R Π θ ⇒ ρθ = p / (R Π).
        ρθᵣ[i, j, k] = pᵣ[i, j, k] / (Rᵈ * Π_safe)
    end
end

@kernel function _populate_Π_from_p!(Π, p, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)
    @inbounds Π[i, j, k] = (p[i, j, k] / pˢᵗ)^κ
end

#####
##### Adaptive substep computation (Klemp 2007 acoustic CFL).
#####

const ACOUSTIC_SAFETY_FACTOR = 2.0

function compute_acoustic_substeps(grid, Δt, constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    Rᵈ  = dry_air_gas_constant(constants)
    γ   = cᵖᵈ / (cᵖᵈ - Rᵈ)
    Tᵣ  = 250.0
    cₛ  = sqrt(γ * Rᵈ * Tᵣ)
    TX, TY, _ = topology(grid)
    Δx_min = TX === Flat ? Inf : minimum_xspacing(grid)
    Δy_min = TY === Flat ? Inf : minimum_yspacing(grid)
    Δh_min = min(Δx_min, Δy_min)
    return ceil(Int, ACOUSTIC_SAFETY_FACTOR * Δt * cₛ / Δh_min)
end

@inline acoustic_substeps(N::Int, grid, Δt, constants) = N
@inline acoustic_substeps(::Nothing, grid, Δt, constants) = compute_acoustic_substeps(grid, Δt, constants)

# Per-stage substep count and size.
@inline function _stage_substep_count_and_size(::ProportionalSubsteps, β_stage, Δt, N)
    Δτ = Δt / N
    Nτ = max(1, round(Int, β_stage * N))
    return Nτ, Δτ
end
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
##### Stage-frozen cache (currently unused in this Baldauf form; kept as a
##### no-op to preserve the API expected by acoustic_runge_kutta_3.jl).
#####

prepare_acoustic_cache!(substepper, model) = nothing

#####
##### Substep kernels — Baldauf-2010 linearized fast operator.
#####

# Cell-center ρθᵣ at face k (arithmetic mean — matches Oceananigans' ℑzᵃᵃᶠ).
@inline function _ρθᵣ_at_face(i, j, k, grid, ρθᵣ)
    Nz = size(grid, 3)
    in_interior = (k >= 2) & (k <= Nz)
    k_safe = ifelse(in_interior, k, 2)
    @inbounds val = (ρθᵣ[i, j, k_safe] + ρθᵣ[i, j, k_safe - 1]) / 2
    return ifelse(in_interior, val, zero(val))
end

# θᵣ_face = ρθᵣ_face / ρᵣ_face. Used as the "θ-flux coefficient" on vertical
# mass flux contributions to η.
@inline function _θᵣ_at_face(i, j, k, grid, ρθᵣ, ρᵣ)
    Nz = size(grid, 3)
    in_interior = (k >= 2) & (k <= Nz)
    k_safe = ifelse(in_interior, k, 2)
    @inbounds begin
        ρθ_f = (ρθᵣ[i, j, k_safe] + ρθᵣ[i, j, k_safe - 1]) / 2
        ρ_f  = (ρᵣ[i, j, k_safe]  + ρᵣ[i, j, k_safe - 1])  / 2
    end
    val = ρθ_f / ifelse(ρ_f == 0, one(ρ_f), ρ_f)
    return ifelse(in_interior, val, zero(val))
end

# Horizontal forward step: update ρu, ρv from current η.
# ρu_new = ρu + Δτ · (Gⁿ.ρu - R Πᵣᶠᶜᶜ · ∂x(η))
@kernel function _horizontal_forward!(ρu, ρv, grid, Δτ, ρθ″_for_pgf, Πᵣ, Gρu, Gρv, Rᵈ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Πᶠᶜᶜ  = ℑxᶠᵃᵃ(i, j, k, grid, Πᵣ)
        Πᶜᶠᶜ  = ℑyᵃᶠᵃ(i, j, k, grid, Πᵣ)
        # PGF source: -R Πᵣ ∂η  (linearization of p = ρRΠθ at fixed Π and ρ_ref).
        # Note: this is NOT γRΠ ∂(ρθ); see header comment.
        ∂x_p′ = Rᵈ * Πᶠᶜᶜ * ∂xᶠᶜᶜ(i, j, k, grid, ρθ″_for_pgf)
        ∂y_p′ = Rᵈ * Πᶜᶠᶜ * ∂yᶜᶠᶜ(i, j, k, grid, ρθ″_for_pgf)
        ρu[i, j, k] += Δτ * (Gρu[i, j, k] - ∂x_p′) * !on_x_boundary(i, j, k, grid)
        ρv[i, j, k] += Δτ * (Gρv[i, j, k] - ∂y_p′) * !on_y_boundary(i, j, k, grid)
    end
end

# Predictors for σ, η using the horizontal mass flux divergence at the
# new horizontal momentum and the explicit (1-ω) part of the vertical flux
# divergence at the OLD ρw. Used to build the tridiagonal RHS for ρw.
@kernel function _build_σ_η_predictors!(σ_pred, η_pred,
                                         σ, η, ρu, ρv, ρw,
                                         grid, Δτ, ω,
                                         Gρ, Gρθ, ρθᵣ, ρᵣ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        V = Vᶜᶜᶜ(i, j, k, grid)

        div_M_h = div_xyᶜᶜᶜ(i, j, k, grid, ρu, ρv)

        # Horizontal θ-flux divergence: ∂x(θᵣ_face × ρu) + ∂y(θᵣ_face × ρv).
        div_θM_h_x = δxᶜᵃᵃ(i, j, k, grid, _Axθᵣρu, ρθᵣ, ρᵣ, ρu) / V
        div_θM_h_y = δyᵃᶜᵃ(i, j, k, grid, _Ayθᵣρv, ρθᵣ, ρᵣ, ρv) / V
        div_θM_h   = div_θM_h_x + div_θM_h_y

        # Explicit (1-ω) part of vertical flux divergence using OLD ρw.
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)
        ρw_above = ρw[i, j, k + 1]
        ρw_below = ρw[i, j, k]
        d_ρw_dz   = (ρw_above - ρw_below) / Δzᶜ
        θᵣ_above = _θᵣ_at_face(i, j, k + 1, grid, ρθᵣ, ρᵣ)
        θᵣ_below = _θᵣ_at_face(i, j, k,     grid, ρθᵣ, ρᵣ)
        d_θρw_dz  = (θᵣ_above * ρw_above - θᵣ_below * ρw_below) / Δzᶜ

        # Mass continuity and θ-conservation are FULLY in the fast operator
        # (-∇·M and -∇·(θᵣM)). The slow Gρ, Gρθ already include these via
        # -∇·(ρU) at stage state — including them here too would double-count.
        # So we use ONLY the fast operator's continuity here.
        σ_pred[i, j, k] = σ[i, j, k] + Δτ * (- div_M_h   - (1 - ω) * d_ρw_dz)
        η_pred[i, j, k] = η[i, j, k] + Δτ * (- div_θM_h  - (1 - ω) * d_θρw_dz)
    end
end

@inline _Axθᵣρu(i, j, k, grid, ρθᵣ, ρᵣ, ρu) =
    Axᶠᶜᶜ(i, j, k, grid) * (@inbounds (ρθᵣ[i, j, k] + ρθᵣ[i - 1, j, k]) / 2 /
                            max((ρᵣ[i, j, k] + ρᵣ[i - 1, j, k]) / 2, one(eltype(grid)))) *
    ρu[i, j, k]

@inline _Ayθᵣρv(i, j, k, grid, ρθᵣ, ρᵣ, ρv) =
    Ayᶜᶠᶜ(i, j, k, grid) * (@inbounds (ρθᵣ[i, j, k] + ρθᵣ[i, j - 1, k]) / 2 /
                            max((ρᵣ[i, j, k] + ρᵣ[i, j - 1, k]) / 2, one(eltype(grid)))) *
    ρv[i, j, k]

# Build the tridiagonal RHS for ρw at faces.
# At face k (interior, k = 2..Nz):
#   f(k) = ρw(k) + Δτ Gρw(k)
#        - Δτ R Πᵣ_face(k) · [(1-ω) δz η + ω δz η_pred] / Δzᶠ
#        - Δτ g · [(1-ω) avg(σ) + ω avg(σ_pred)]
# The implicit (ω) parts use the predictors (which themselves contain only the
# explicit (1-ω) ρw vertical flux), and the FULL ω vertical-flux dependence on
# ρw_new is handled inside the tridiag coefficients.
@kernel function _build_ρw_rhs!(ρw, σ, η, σ_pred, η_pred,
                                 grid, Δτ, ω, Gρw, Πᵣ, Rᵈ, g)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        if k > 1 && k <= size(grid, 3)
            Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
            Π_f = ℑzᵃᵃᶠ(i, j, k, grid, Πᵣ)
            # PGF: -R Πᵣ_face × ∂z(η). Off-centered on (1-ω) old + ω predictor.
            δz_η_old  = (η[i, j, k]      - η[i, j, k - 1])      / Δzᶠ
            δz_η_pred = (η_pred[i, j, k] - η_pred[i, j, k - 1]) / Δzᶠ
            pgf = Rᵈ * Π_f * ((1 - ω) * δz_η_old + ω * δz_η_pred)

            # Buoyancy: -g·σ at face. Off-centered on (1-ω) old + ω predictor.
            σ_face_old  = (σ[i, j, k]      + σ[i, j, k - 1])      / 2
            σ_face_pred = (σ_pred[i, j, k] + σ_pred[i, j, k - 1]) / 2
            buoy = g * ((1 - ω) * σ_face_old + ω * σ_face_pred)

            ρw[i, j, k] = ρw[i, j, k] + Δτ * (Gρw[i, j, k] - pgf - buoy)
        end
    end
end

# Tridiag coefficient helpers for the implicit ω-dependence of ρw_new on σ, η
# updates. From the substituted continuity equations:
#   σ(k) ← σ_pred(k) - ω Δτ/Δzᶜ_k · (ρw_new(k+1) - ρw_new(k))
#   η(k) ← η_pred(k) - ω Δτ/Δzᶜ_k · (θᵣ_above(k) ρw_new(k+1) - θᵣ_below(k) ρw_new(k))
# These contributions, when fed back into the ρw equation at face k, produce
# a tridiagonal in ρw_new at faces k-1, k, k+1.

import Oceananigans.Solvers: get_coefficient

# Solver `args...` order matches the `solve!` call in the substep loop:
# (Πᵣ, ρθᵣ, ρᵣ, Δτᵋ, Rᵈ, g, Δτ, ω). We access Πᵣ, ρθᵣ, ρᵣ.
@inline function _tridiag_pgf_coeff(i, j, k, grid, Πᵣ, ρθᵣ, ρᵣ, Δτ, ω, Rᵈ)
    # Coefficient: ω² R Πᵣ_face / Δzᶜ_center · θᵣ_at_face, scaled by Δτ²
    # (one Δτ from ρw equation outer, one from σ/η continuity).
    Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
    Π_f = ℑzᵃᵃᶠ(i, j, k, grid, Πᵣ)
    return ω * ω * Δτ * Δτ * Rᵈ * Π_f / Δzᶠ
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagLower, p, ::ZDirection,
                                 Πᵣ, ρθᵣ, ρᵣ, Δτ, Rᵈ, g, ω)
    k_face = k + 1
    Δzᶜ_below = Δzᶜᶜᶜ(i, j, k_face - 1, grid)
    θᵣ_below_face_below = _θᵣ_at_face(i, j, k_face - 1, grid, ρθᵣ, ρᵣ)
    pgf_factor = _tridiag_pgf_coeff(i, j, k_face, grid, Πᵣ, ρθᵣ, ρᵣ, Δτ, ω, Rᵈ)
    # PGF coupling: ρw_new(k_face-1) → η at center k_face-1 → face k_face PGF.  Sign: -
    pgf_coupling  = -pgf_factor * θᵣ_below_face_below / Δzᶜ_below
    # Buoyancy coupling: ρw_new(k_face-1) → σ at center k_face-1 → face k_face buoy.  Sign: +
    buoy_coupling = +ω * ω * Δτ * Δτ * g / (2 * Δzᶜ_below)
    return pgf_coupling + buoy_coupling
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagDiagonal, p, ::ZDirection,
                                 Πᵣ, ρθᵣ, ρᵣ, Δτ, Rᵈ, g, ω)
    k == 1 && return one(Δτ)
    k_face = k
    Δzᶜ_above = Δzᶜᶜᶜ(i, j, k_face,     grid)
    Δzᶜ_below = Δzᶜᶜᶜ(i, j, k_face - 1, grid)
    θᵣ_face   = _θᵣ_at_face(i, j, k_face, grid, ρθᵣ, ρᵣ)
    pgf_factor = _tridiag_pgf_coeff(i, j, k_face, grid, Πᵣ, ρθᵣ, ρᵣ, Δτ, ω, Rᵈ)
    pgf_coupling  = +pgf_factor * θᵣ_face * (1 / Δzᶜ_above + 1 / Δzᶜ_below)
    # Buoyancy diagonal: gω²Δτ²/2 × (1/Δzᶜ_above − 1/Δzᶜ_below). Zero on uniform grid.
    buoy_coupling = +ω * ω * Δτ * Δτ * g / 2 * (1 / Δzᶜ_above - 1 / Δzᶜ_below)
    return one(Δτ) + pgf_coupling + buoy_coupling
end

@inline function get_coefficient(i, j, k, grid, ::AcousticTridiagUpper, p, ::ZDirection,
                                 Πᵣ, ρθᵣ, ρᵣ, Δτ, Rᵈ, g, ω)
    k == 1 && return zero(Δτ)
    k_face = k
    Δzᶜ_above = Δzᶜᶜᶜ(i, j, k_face, grid)
    θᵣ_above_face_above = _θᵣ_at_face(i, j, k_face + 1, grid, ρθᵣ, ρᵣ)
    pgf_factor = _tridiag_pgf_coeff(i, j, k_face, grid, Πᵣ, ρθᵣ, ρᵣ, Δτ, ω, Rᵈ)
    pgf_coupling  = -pgf_factor * θᵣ_above_face_above / Δzᶜ_above
    buoy_coupling = -ω * ω * Δτ * Δτ * g / (2 * Δzᶜ_above)
    return pgf_coupling + buoy_coupling
end

# Post-solve update of σ, η using new ρw.
@kernel function _post_solve_update!(σ, η, σ_pred, η_pred, ρw,
                                      grid, Δτ, ω, ρθᵣ, ρᵣ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)
        ρw_above = ρw[i, j, k + 1]
        ρw_below = ρw[i, j, k]
        θᵣ_above = _θᵣ_at_face(i, j, k + 1, grid, ρθᵣ, ρᵣ)
        θᵣ_below = _θᵣ_at_face(i, j, k,     grid, ρθᵣ, ρᵣ)

        σ[i, j, k] = σ_pred[i, j, k] - ω * Δτ * (ρw_above - ρw_below) / Δzᶜ
        η[i, j, k] = η_pred[i, j, k] - ω * Δτ * (θᵣ_above * ρw_above - θᵣ_below * ρw_below) / Δzᶜ
    end
end

#####
##### Damping kernels.
#####

# Pressure-projection filter: forward-project η for the next substep's PGF.
# ρθ″_for_pgf = η + β · (η - η_prev)
@inline apply_pgf_filter!(::AcousticDampingStrategy, substepper, model) = nothing

function apply_pgf_filter!(damping::ConservativeProjectionDamping, substepper, model)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    β = convert(FT, damping.coefficient)
    launch!(arch, grid, :xyz, _pressure_projection_filter!,
            damping.ρθ″_for_pgf, substepper.η, substepper.η_prev, β)
    fill_halo_regions!(damping.ρθ″_for_pgf)
    return nothing
end

function apply_pgf_filter!(damping::PressureProjectionDamping, substepper, model)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    β = convert(FT, damping.coefficient)
    launch!(arch, grid, :xyz, _pressure_projection_filter!,
            damping.ρθ″_for_pgf, substepper.η, substepper.η_prev, β)
    fill_halo_regions!(damping.ρθ″_for_pgf)
    return nothing
end

@kernel function _pressure_projection_filter!(ρθ″_for_pgf, η, η_prev, β)
    i, j, k = @index(Global, NTuple)
    @inbounds ρθ″_for_pgf[i, j, k] = η[i, j, k] + β * (η[i, j, k] - η_prev[i, j, k])
end

# Selects which field the horizontal_forward kernel reads as the PGF source.
@inline pgf_source_field(::AcousticDampingStrategy, substepper) = substepper.η
@inline pgf_source_field(damping::PressureProjectionDamping, substepper)    = damping.ρθ″_for_pgf
@inline pgf_source_field(damping::ConservativeProjectionDamping, substepper) = damping.ρθ″_for_pgf

# Klemp-Skamarock-Ha 2018 thermodynamic damping (post-substep, on ρu, ρv).
@inline apply_divergence_damping!(::NoDivergenceDamping, substepper, grid, Δτ) = nothing
@inline apply_divergence_damping!(::PressureProjectionDamping, substepper, grid, Δτ) = nothing
@inline apply_divergence_damping!(::ConservativeProjectionDamping, substepper, grid, Δτ) = nothing

function apply_divergence_damping!(damping::ThermodynamicDivergenceDamping, substepper, grid, Δτ)
    arch = architecture(grid)
    FT = eltype(grid)
    if damping.length_scale === nothing
        TX, TY, _ = topology(grid)
        Δx_eff = TX === Flat ? FT(Inf) : FT(minimum_xspacing(grid))
        Δy_eff = TY === Flat ? FT(Inf) : FT(minimum_yspacing(grid))
        len_disp = isfinite(min(Δx_eff, Δy_eff)) ? min(Δx_eff, Δy_eff) : one(FT)
    else
        len_disp = convert(FT, damping.length_scale)
    end
    smdiv = convert(FT, damping.coefficient)
    coef = 2 * smdiv * len_disp / Δτ
    launch!(arch, grid, :xyz, _thermodynamic_damping!,
            substepper.ρu, substepper.ρv,
            substepper.η, substepper.η_prev,
            grid, coef)
    return nothing
end

@inline _neg_δη(i, j, k, grid, η, η_prev) = -(@inbounds η[i, j, k] - η_prev[i, j, k])

@kernel function _thermodynamic_damping!(ρu, ρv, η, η_prev, grid, coef)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ∂x_div = δxᶠᵃᵃ(i, j, k, grid, _neg_δη, η, η_prev)
        ∂y_div = δyᵃᶠᵃ(i, j, k, grid, _neg_δη, η, η_prev)
        ρu[i, j, k] += coef * ∂x_div * !on_x_boundary(i, j, k, grid)
        ρv[i, j, k] += coef * ∂y_div * !on_y_boundary(i, j, k, grid)
    end
end

#####
##### Recovery: prognostic state ← reference + perturbation.
#####

@kernel function _recover_prognostics!(ρ_prog, ρθ_prog, m_ρu, m_ρv, m_ρw,
                                        v_u, v_v, v_w,
                                        σ, η, ρu, ρv, ρw,
                                        ρᵣ, ρθᵣ, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρ_new  = ρᵣ[i, j, k]  + σ[i, j, k]
        ρθ_new = ρθᵣ[i, j, k] + η[i, j, k]
        ρ_prog[i, j, k]  = ρ_new
        ρθ_prog[i, j, k] = ρθ_new
        m_ρu[i, j, k] = ρu[i, j, k]
        m_ρv[i, j, k] = ρv[i, j, k]
        m_ρw[i, j, k] = ρw[i, j, k]

        ρᶠ = ℑxᶠᵃᵃ(i, j, k, grid, ρ_prog)
        v_u[i, j, k] = ρu[i, j, k] / ifelse(ρᶠ == 0, one(ρᶠ), ρᶠ) * !on_x_boundary(i, j, k, grid)
        ρᶠ = ℑyᵃᶠᵃ(i, j, k, grid, ρ_prog)
        v_v[i, j, k] = ρv[i, j, k] / ifelse(ρᶠ == 0, one(ρᶠ), ρᶠ) * !on_y_boundary(i, j, k, grid)
        ρᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ_prog)
        v_w[i, j, k] = ρw[i, j, k] / ifelse(ρᶠ == 0, one(ρᶠ), ρᶠ) * (k > 1)
    end
end

#####
##### WS-RK3 stage substep loop driver.
#####

function acoustic_rk3_substep_loop!(model, substepper, Δt, β_stage, U⁰)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)

    N_raw = acoustic_substeps(substepper.substeps, grid, Δt, model.thermodynamic_constants)
    N = max(6, 6 * cld(N_raw, 6))
    Nτ, Δτ = _stage_substep_count_and_size(substepper.substep_distribution, β_stage, Δt, N)
    Δτ_FT = FT(Δτ)
    ω = substepper.forward_weight

    Rᵈ  = dry_air_gas_constant(model.thermodynamic_constants)
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    γ   = cᵖᵈ / (cᵖᵈ - Rᵈ)              # = cp/cv for dry air ≈ 1.4
    γRᵈ = γ * Rᵈ                         # PGF prefactor: ∂p/∂(ρθ) = γRᵈ Π
    g   = model.thermodynamic_constants.gravitational_acceleration

    # Slow tendencies.
    Gⁿ = model.timestepper.Gⁿ
    χ_name = thermodynamic_density_name(model.formulation)
    Gρθ = getproperty(Gⁿ, χ_name)
    Gρ  = Gⁿ.ρ
    Gρu = Gⁿ.ρu
    Gρv = Gⁿ.ρv
    Gρw = Gⁿ.ρw

    # Seed perturbations from U⁰ minus reference at stage start.
    ρθ_prog = thermodynamic_density(model.formulation)
    parent(substepper.σ)  .= parent(U⁰[1]) .- parent(substepper.reference_density)
    parent(substepper.η)  .= parent(U⁰[5]) .- parent(substepper.reference_ρθ)
    parent(substepper.ρu) .= parent(U⁰[2])
    parent(substepper.ρv) .= parent(U⁰[3])
    parent(substepper.ρw) .= parent(U⁰[4])
    parent(substepper.η_prev) .= parent(substepper.η)
    fill_halo_regions!(substepper.σ)
    fill_halo_regions!(substepper.η)
    fill_halo_regions!(substepper.ρu)
    fill_halo_regions!(substepper.ρv)
    fill_halo_regions!(substepper.ρw)

    for substep in 1:Nτ
        # Save η before this substep for damping (δ_τ η).
        parent(substepper.η_prev) .= parent(substepper.η)

        # Optional pressure-projection damping fills `damping.ρθ″_for_pgf`.
        apply_pgf_filter!(substepper.damping, substepper, model)
        ρθ″_for_pgf = pgf_source_field(substepper.damping, substepper)

        # 1. Forward step for ρu, ρv.
        launch!(arch, grid, :xyz, _horizontal_forward!,
                substepper.ρu, substepper.ρv, grid, Δτ_FT,
                ρθ″_for_pgf, substepper.reference_exner_function,
                Gρu, Gρv, FT(γRᵈ))
        fill_halo_regions!(substepper.ρu)
        fill_halo_regions!(substepper.ρv)

        # 2. Predictors for σ, η (using NEW ρu, ρv and OLD ρw).
        launch!(arch, grid, :xyz, _build_σ_η_predictors!,
                substepper.σ_pred, substepper.η_pred,
                substepper.σ, substepper.η,
                substepper.ρu, substepper.ρv, substepper.ρw,
                grid, Δτ_FT, FT(ω),
                Gρ, Gρθ,
                substepper.reference_ρθ, substepper.reference_density)

        # 3. Build RHS for ρw at faces.
        launch!(arch, grid, :xyz, _build_ρw_rhs!,
                substepper.ρw, substepper.σ, substepper.η,
                substepper.σ_pred, substepper.η_pred,
                grid, Δτ_FT, FT(ω), Gρw,
                substepper.reference_exner_function, FT(γRᵈ), FT(g))

        # 4. Tridiagonal solve: ρw at faces.
        solve!(substepper.ρw, substepper.vertical_solver, substepper.ρw,
               substepper.reference_exner_function,
               substepper.reference_ρθ,
               substepper.reference_density,
               Δτ_FT, FT(γRᵈ), FT(g), FT(ω))

        # 5. Update σ, η using new ρw.
        launch!(arch, grid, :xyz, _post_solve_update!,
                substepper.σ, substepper.η,
                substepper.σ_pred, substepper.η_pred, substepper.ρw,
                grid, Δτ_FT, FT(ω),
                substepper.reference_ρθ, substepper.reference_density)

        fill_halo_regions!(substepper.σ)
        fill_halo_regions!(substepper.η)
        fill_halo_regions!(substepper.ρw)

        # 6. Damping (post-substep momentum filter, if active).
        apply_divergence_damping!(substepper.damping, substepper, grid, Δτ_FT)
        fill_halo_regions!(substepper.ρu)
        fill_halo_regions!(substepper.ρv)
    end

    # Recover prognostic state from substepper's σ, η, ρu, ρv, ρw.
    launch!(arch, grid, :xyz, _recover_prognostics!,
            model.dynamics.density, ρθ_prog,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            model.velocities.u, model.velocities.v, model.velocities.w,
            substepper.σ, substepper.η, substepper.ρu, substepper.ρv, substepper.ρw,
            substepper.reference_density, substepper.reference_ρθ,
            grid)

    return nothing
end
