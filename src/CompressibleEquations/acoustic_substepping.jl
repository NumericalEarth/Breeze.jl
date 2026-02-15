#####
##### Acoustic Substepping for CompressibleDynamics — Exner Pressure Formulation
#####
##### Implements split-explicit time integration following CM1 (Bryan 2002),
##### Wicker-Skamarock (2002), and Klemp et al. (2007):
##### - Forward-backward acoustic substeps with (velocity, Exner pressure) variables
##### - Vertically implicit w-π coupling with off-centering (always on)
##### - CM1-style divergence damping (kdiv) on the pressure variable
##### - Constant acoustic substep size Δτ = Δt/N across all RK stages
##### - Topology-aware operators (no halo filling between substeps)
#####

using KernelAbstractions: @kernel, @index

using Oceananigans: CenterField, XFaceField, YFaceField, ZFaceField, architecture
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!
using Oceananigans.Operators:
    ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ,
    ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ,
    δxTᶠᵃᵃ, δyTᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ,
    divᶜᶜᶜ,
    Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶜ, Δzᶜᶜᶠ

using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!

using Oceananigans.Grids: Periodic, Bounded,
                          AbstractUnderlyingGrid

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
@inline ℑzTᵃᵃᶠ(i, j, k, grid, f, args...)        = ℑzᵃᵃᶠ(i, j, k, grid, f, args...)
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

# For periodic/flat topologies, no boundary faces exist
@inline on_x_boundary(i, j, k, grid) = false
@inline on_y_boundary(i, j, k, grid) = false

# For bounded topologies, face i=1 / j=1 are boundary faces
@inline on_x_boundary(i, j, k, grid::BX) = (i == 1)
@inline on_y_boundary(i, j, k, grid::BY) = (j == 1)

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
4. **Damping**: ``π'_d = π' + κ_{div} (π' - π'_{old})``

Fields
======

- `substeps`: Number of acoustic substeps for the full time step
- `forward_weight`: Off-centering parameter α (0.6 = CM1 default); β = 1 - α
- `divergence_damping_coefficient`: CM1-style `kdiv` for π' damping (default 0.10)
- `θᵥ`: Stage-frozen virtual potential temperature (CenterField)
- `ppterm`: Pressure tendency coefficient c²/(cᵖ ρ₀ θᵥ²) (CenterField)
- `π_ref`: Reference Exner function π₀ = (p_ref/pˢᵗ)^(R/cᵖ) (CenterField, z-only)
- `u₀, v₀, w₀`: Velocity at time-step start (for WS-RK3 reset)
- `π′`: Exner pressure perturbation (CenterField)
- `π′_old`: Previous-substep π' for divergence damping (CenterField)
- `π′_damped`: Damped Exner pressure for PGF (CenterField)
- `averaged_velocities`: Time-averaged velocities for scalar advection
- `slow_momentum_tendencies`: Slow ρu, ρv, ρw tendencies (frozen during acoustic loop)
- `slow_velocity_tendencies`: Slow u, v, w tendencies (computed from momentum/density)
- `Gˢρ`: Slow density tendency
- `Gˢρχ`: Slow thermodynamic tendency (ρθ)
- `ppten`: Slow Exner pressure tendency (CenterField)
- `vertical_solver`: BatchedTridiagonalSolver for implicit w-π' coupling
- `rhs`: Right-hand side storage for tridiagonal solve
"""
struct AcousticSubstepper{N, FT, CF, ZF, AV, SM, SV, TS}
    # Number of acoustic substeps for the full time step (stage 3)
    substeps :: N

    # Off-centering: α = forward weight (CM1 default 0.6), β = 1 - α
    forward_weight :: FT

    # CM1-style divergence damping coefficient (kdiv)
    divergence_damping_coefficient :: FT

    # Klemp 2018 acoustic damping coefficient (β_d)
    acoustic_damping_coefficient :: FT

    # Stage-frozen thermodynamic coefficients
    θᵥ     :: CF  # Virtual potential temperature (frozen)
    ppterm :: CF  # c²/(cᵖ ρ₀ θᵥ²) — pressure tendency coefficient

    # Reference Exner function (from reference state, z-only)
    π_ref :: ZF

    # Exner pressure perturbation fields
    π′       :: CF  # Current Exner pressure perturbation
    π′_old   :: CF  # Previous-substep value (for damping)
    π′_damped :: CF  # Damped value used in PGF

    # Stage-frozen reference state (for recovery and vertical PGF)
    ρᵣ  :: CF  # Stage-frozen density
    ρχᵣ :: CF  # Stage-frozen ρθ
    pᵣ  :: CF  # Reference-subtracted pressure: p_stage - p_ref (exact discrete balance)
    Bᵣ  :: CF  # Reference buoyancy: -g(ρ_stage - ρ_ref) / ρ_stage

    # Time-averaged velocities for scalar advection
    averaged_velocities :: AV

    # Slow tendencies in momentum form (from outer RK)
    slow_momentum_tendencies :: SM  # NamedTuple with ρu, ρv, ρw
    Gˢρ  :: CF  # Slow density tendency
    Gˢρχ :: CF  # Slow thermodynamic tendency (ρθ)

    # Slow tendencies in velocity/pressure form (converted)
    slow_velocity_tendencies :: SV  # NamedTuple with u, v, w
    ppten :: CF  # Slow Exner pressure tendency

    # Vertical tridiagonal solver for implicit w-π' coupling
    vertical_solver :: TS

    # Right-hand side storage for tridiagonal solve
    rhs :: CF
end

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.substeps,
                       a.forward_weight,
                       a.divergence_damping_coefficient,
                       a.acoustic_damping_coefficient,
                       adapt(to, a.θᵥ),
                       adapt(to, a.ppterm),
                       adapt(to, a.π_ref),
                       adapt(to, a.π′),
                       adapt(to, a.π′_old),
                       adapt(to, a.π′_damped),
                       adapt(to, a.ρᵣ),
                       adapt(to, a.ρχᵣ),
                       adapt(to, a.pᵣ),
                       adapt(to, a.Bᵣ),
                       map(f -> adapt(to, f), a.averaged_velocities),
                       map(f -> adapt(to, f), a.slow_momentum_tendencies),
                       adapt(to, a.Gˢρ),
                       adapt(to, a.Gˢρχ),
                       map(f -> adapt(to, f), a.slow_velocity_tendencies),
                       adapt(to, a.ppten),
                       adapt(to, a.vertical_solver),
                       adapt(to, a.rhs))

"""
$(TYPEDSIGNATURES)

Construct an `AcousticSubstepper` using the Exner pressure formulation.
"""
function AcousticSubstepper(grid, split_explicit::SplitExplicitTimeDiscretization)
    Ns = split_explicit.substeps
    FT = eltype(grid)
    α = convert(FT, split_explicit.forward_weight)
    kdiv = convert(FT, split_explicit.divergence_damping_coefficient)
    β_d = convert(FT, split_explicit.acoustic_damping_coefficient)

    # Stage-frozen coefficients
    θᵥ = CenterField(grid)
    ppterm = CenterField(grid)

    # Reference Exner function (z-only for no-terrain case)
    π_ref = CenterField(grid)

    # Exner pressure perturbation
    π′ = CenterField(grid)
    π′_old = CenterField(grid)
    π′_damped = CenterField(grid)

    # Stage-frozen reference state
    ρᵣ = CenterField(grid)
    ρχᵣ = CenterField(grid)
    pᵣ = CenterField(grid)
    Bᵣ = CenterField(grid)

    # Time-averaged velocities
    averaged_velocities = (u = XFaceField(grid),
                           v = YFaceField(grid),
                           w = ZFaceField(grid))

    # Slow momentum tendencies
    slow_momentum_tendencies = (ρu = XFaceField(grid),
                                ρv = YFaceField(grid),
                                ρw = ZFaceField(grid))
    Gˢρ = CenterField(grid)
    Gˢρχ = CenterField(grid)

    # Slow velocity/pressure tendencies (converted from momentum form)
    slow_velocity_tendencies = (u = XFaceField(grid),
                                v = YFaceField(grid),
                                w = ZFaceField(grid))
    ppten = CenterField(grid)

    # Vertical tridiagonal solver (always allocated for Exner formulation)
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    lower_diagonal = zeros(arch, FT, Nx, Ny, Nz)
    diagonal = zeros(arch, FT, Nx, Ny, Nz)
    upper_diagonal = zeros(arch, FT, Nx, Ny, Nz)
    scratch = zeros(arch, FT, Nx, Ny, Nz)

    vertical_solver = BatchedTridiagonalSolver(grid;
                                               lower_diagonal,
                                               diagonal,
                                               upper_diagonal,
                                               scratch,
                                               tridiagonal_direction = ZDirection())

    rhs = CenterField(grid)

    return AcousticSubstepper(Ns, α, kdiv, β_d,
                              θᵥ, ppterm, π_ref,
                              π′, π′_old, π′_damped,
                              ρᵣ, ρχᵣ, pᵣ, Bᵣ,
                              averaged_velocities,
                              slow_momentum_tendencies,
                              Gˢρ, Gˢρχ,
                              slow_velocity_tendencies,
                              ppten,
                              vertical_solver,
                              rhs)
end

#####
##### Section 3: Cache preparation (once per RK stage)
#####

using Breeze.AtmosphereModels: thermodynamic_density
using Breeze.Thermodynamics: dry_air_gas_constant

"""
$(TYPEDSIGNATURES)

Prepare the acoustic cache for an RK stage.

Computes stage-frozen coefficients for the Exner pressure acoustic loop:
1. Virtual potential temperature θᵥ (frozen during acoustic loop)
2. Pressure tendency coefficient ppterm = c²/(cᵖ ρ₀ θᵥ²)
3. Exner pressure perturbation π' = (p/pˢᵗ)^(R/cᵖ) - π₀
4. Reference Exner function π₀ from the reference state

Following CM1's `sound.F`, the acoustic loop prognostics velocity and
Exner pressure perturbation, with density diagnosed from the equation
of state after the loop.
"""
function prepare_acoustic_cache!(substepper, model)
    grid = model.grid
    arch = architecture(grid)

    # Store stage-frozen reference state (for recovery)
    χ = thermodynamic_density(model.formulation)
    parent(substepper.ρᵣ) .= parent(model.dynamics.density)
    parent(substepper.ρχᵣ) .= parent(χ)

    # Compute stage-frozen coefficients
    pˢᵗ = model.dynamics.standard_pressure
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖ  # R/cp

    launch!(arch, grid, :xyz, _prepare_exner_cache!,
            substepper.θᵥ,
            substepper.ppterm,
            substepper.π′,
            substepper.π′_damped,
            substepper.π_ref,
            model.dynamics.density,
            model.dynamics.pressure,
            model.temperature,
            model.specific_moisture,
            grid,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants,
            model.dynamics.reference_state,
            pˢᵗ, cᵖ, κ)

    # Use the ExnerReferenceState's π₀ directly (exact discrete Exner hydrostatic balance),
    # or build from stage θᵥ for standard ReferenceState.
    _set_exner_reference!(substepper, model, model.dynamics.reference_state, pˢᵗ, κ)

    return nothing
end

@kernel function _recompute_pi_prime!(π′, π′_damped, p, π_ref, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        πⁱ = (p[i, j, k] / pˢᵗ)^κ
        π′[i, j, k] = πⁱ - π_ref[i, j, k]
        π′_damped[i, j, k] = π′[i, j, k]
    end
end

@kernel function _prepare_exner_cache!(θᵥ_field, ppterm_field, π′_field, π′_damped_field,
                                       π_ref_field,
                                       ρ, p, T, qᵗ, grid,
                                       microphysics, microphysical_fields,
                                       constants, reference_state, pˢᵗ, cᵖ, κ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρⁱ = ρ[i, j, k]
        pⁱ = p[i, j, k]
        Tⁱ = T[i, j, k]
        qᵗⁱ = qᵗ[i, j, k]
    end

    # Compute moisture fractions and mixture properties
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρⁱ, qᵗⁱ, microphysical_fields)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    cᵛᵐ = cᵖᵐ - Rᵐ
    γᵐ = cᵖᵐ / cᵛᵐ

    # Virtual potential temperature: θᵥ = T / π where π = (p/pˢᵗ)^κ
    πⁱ = (pⁱ / pˢᵗ)^κ
    θᵥⁱ = Tⁱ / πⁱ

    # Pressure tendency coefficient: ppterm = (γ-1) π₀
    # Derivation: from the ideal gas law and continuity equation,
    # ∂π'/∂t = -(γ-1) π₀ ∇·u. Combined with the momentum equation
    # ∂u/∂t = -cᵖ θᵥ ∂π'/∂x, this gives the correct acoustic wave speed:
    # c_eff² = cᵖ θᵥ ppterm = cᵖ θᵥ (γ-1) π₀ = γ Rᵐ T = c²
    pptermⁱ = (γᵐ - 1) * πⁱ

    # Exner pressure perturbation: π' = π - π_ref
    π_refⁱ = _get_reference_exner(i, j, k, reference_state, pˢᵗ, κ)

    @inbounds begin
        θᵥ_field[i, j, k] = θᵥⁱ
        ppterm_field[i, j, k] = pptermⁱ
        π′_field[i, j, k] = πⁱ - π_refⁱ
        π′_damped_field[i, j, k] = πⁱ - π_refⁱ
        π_ref_field[i, j, k] = π_refⁱ
    end
end

##### Set the Exner reference state for the acoustic loop.
##### Dispatches on reference state type to use the most accurate π₀.

function _set_exner_reference!(substepper, model, ref::ExnerReferenceState, pˢᵗ, κ)
    grid = model.grid
    arch = architecture(grid)
    # Use the stored π₀ directly (exact discrete Exner hydrostatic balance)
    parent(substepper.π_ref) .= parent(ref.exner)
    # Compute π' = π_actual - π₀
    launch!(arch, grid, :xyz, _recompute_pi_prime!,
            substepper.π′, substepper.π′_damped,
            model.dynamics.pressure, substepper.π_ref, pˢᵗ, κ)
    return nothing
end

function _set_exner_reference!(substepper, model, ref::ReferenceState, pˢᵗ, κ)
    grid = model.grid
    arch = architecture(grid)
    # Build π_ref from reference pressure (not exact Exner balance)
    launch!(arch, grid, :xyz, _set_bottom_exner!,
            substepper.π_ref, ref.pressure, pˢᵗ, κ)
    launch!(arch, grid, :xyz, _recompute_pi_prime!,
            substepper.π′, substepper.π′_damped,
            model.dynamics.pressure, substepper.π_ref, pˢᵗ, κ)
    return nothing
end

function _set_exner_reference!(substepper, model, ::Nothing, pˢᵗ, κ)
    grid = model.grid
    arch = architecture(grid)
    fill!(parent(substepper.π_ref), 0)
    launch!(arch, grid, :xyz, _recompute_pi_prime!,
            substepper.π′, substepper.π′_damped,
            model.dynamics.pressure, substepper.π_ref, pˢᵗ, κ)
    return nothing
end

##### Compute reference-subtracted pressure and buoyancy for the vertical PGF.
##### These satisfy exact discrete hydrostatic balance: ∂pᵣ/∂z + g avg(ρ-ρ_ref) = 0
##### to machine precision, avoiding the hydrostatic imbalance that plagues the
##### pure Exner formulation.

function _compute_vertical_reference!(substepper, model)
    ref = model.dynamics.reference_state
    _compute_vertical_reference!(substepper, model, ref)
end

_compute_vertical_reference!(substepper, model, ::Nothing) = nothing

function _compute_vertical_reference!(substepper, model, ref::Union{ReferenceState, ExnerReferenceState})
    grid = model.grid
    arch = architecture(grid)
    g = model.thermodynamic_constants.gravitational_acceleration

    # pᵣ = p_stage - p_ref
    parent(substepper.pᵣ) .= parent(model.dynamics.pressure)
    parent(substepper.pᵣ) .-= parent(ref.pressure)

    # Bᵣ = -g(ρ_stage - ρ_ref) / ρ_stage (buoyancy in velocity form)
    launch!(arch, grid, :xyz, _compute_reference_buoyancy_velocity!,
            substepper.Bᵣ, model.dynamics.density, ref.density, g)
    return nothing
end

@kernel function _compute_reference_buoyancy_velocity!(Bᵣ, ρ, ρ_ref, g)
    i, j, k = @index(Global, NTuple)
    @inbounds Bᵣ[i, j, k] = -g * (ρ[i, j, k] - ρ_ref[i, j, k]) / ρ[i, j, k]
end

function _build_stage_hydrostatic_exner!(π_ref, θᵥ, model)
    grid = model.grid
    arch = architecture(grid)
    g = model.thermodynamic_constants.gravitational_acceleration
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity
    pˢᵗ = model.dynamics.standard_pressure
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖ

    # Set bottom value from actual stage pressure
    launch!(arch, grid, :xyz, _set_bottom_exner!,
            π_ref, model.dynamics.pressure, pˢᵗ, κ)

    # Integrate upward using STAGE θᵥ (same as used by acoustic loop)
    Nx, Ny, Nz = size(grid)
    launch!(arch, grid, :xy, _integrate_stage_hydrostatic_exner!,
            π_ref, θᵥ, grid, g, cᵖ, Nz)

    return nothing
end

@kernel function _set_bottom_exner!(π_ref, p, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)
    @inbounds π_ref[i, j, k] = (p[i, j, k] / pˢᵗ)^κ
end

@kernel function _integrate_stage_hydrostatic_exner!(π_ref, θᵥ, grid, g, cᵖ, Nz)
    i, j = @index(Global, NTuple)
    for k in 2:Nz
        @inbounds begin
            Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
            # Use the SAME face average as the w-update kernel: avg(θᵥ[k-1], θᵥ[k])
            θᵥᶠ = (θᵥ[i, j, k] + θᵥ[i, j, k-1]) / 2
            π_ref[i, j, k] = π_ref[i, j, k-1] - g * Δzᶠ / (cᵖ * θᵥᶠ)
        end
    end
end

@inline _get_reference_exner(i, j, k, ::Nothing, pˢᵗ, κ) = zero(pˢᵗ)

@inline function _get_reference_exner(i, j, k, ref::ReferenceState, pˢᵗ, κ)
    @inbounds p_ref = ref.pressure[i, j, k]
    return (p_ref / pˢᵗ)^κ
end

@inline function _get_reference_exner(i, j, k, ref::ExnerReferenceState, pˢᵗ, κ)
    @inbounds return ref.exner[i, j, k]
end

# Old build_discrete_hydrostatic_exner! removed — replaced by ExnerReferenceState.

#####
##### Section 4: Convert slow tendencies to velocity/pressure form
#####

"""
$(TYPEDSIGNATURES)

Convert slow momentum tendencies (Gˢρu, Gˢρv, Gˢρw) to slow velocity
tendencies (uten, vten, wten) and slow pressure tendency (ppten).

The velocity tendency is: uten ≈ Gˢρu / ρ
The pressure tendency is: ppten = -u · ∇π

These are frozen during the acoustic substep loop.
"""
function convert_slow_tendencies!(substepper, model)
    grid = model.grid
    arch = architecture(grid)
    pˢᵗ = model.dynamics.standard_pressure
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity
    g = model.thermodynamic_constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖ

    launch!(arch, grid, :xyz, _convert_slow_tendencies!,
            substepper.slow_velocity_tendencies.u,
            substepper.slow_velocity_tendencies.v,
            substepper.slow_velocity_tendencies.w,
            substepper.ppten,
            substepper.slow_momentum_tendencies.ρu,
            substepper.slow_momentum_tendencies.ρv,
            substepper.slow_momentum_tendencies.ρw,
            substepper.Gˢρχ,
            substepper.Gˢρ,
            model.dynamics.density,
            model.velocities.u,
            model.velocities.v,
            model.velocities.w,
            substepper.ρχᵣ,
            substepper.π′,
            substepper.π_ref,
            substepper.θᵥ,
            grid, κ, cᵖ, g)

    return nothing
end

@kernel function _convert_slow_tendencies!(uten, vten, wten, ppten,
                                           Gˢρu, Gˢρv, Gˢρw, Gˢρχ, Gˢρ,
                                           ρ, u, v, w,
                                           ρχᵣ, π′, π_ref, θᵥ,
                                           grid, κ, cᵖ, g)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Velocity tendencies from momentum tendencies: uten = Gˢρu / ρ
        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        uten[i, j, k] = Gˢρu[i, j, k] / ρᶠᶜᶜ * !on_x_boundary(i, j, k, grid)

        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        vten[i, j, k] = Gˢρv[i, j, k] / ρᶜᶠᶜ * !on_y_boundary(i, j, k, grid)

        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)

        # Buoyancy from the Exner pressure reference-state splitting:
        # Full vertical acceleration = -cᵖ θᵥ ∂π/∂z - g
        #   = -cᵖ θᵥ ∂π₀/∂z - cᵖ θᵥ ∂π'/∂z - g
        #   = B - cᵖ θᵥ ∂π'/∂z
        # where B = -cᵖ θᵥ_face δz(π₀)/Δz - g captures the buoyancy from the
        # mismatch between actual θᵥ and the reference θ₀ used to build π₀.
        # The acoustic loop provides -cᵖ θᵥ ∂π'/∂z; we add B as a slow tendency.
        θᵥᶠ = ℑzTᵃᵃᶠ(i, j, k, grid, θᵥ)
        δz_π₀ = δzTᵃᵃᶠ(i, j, k, grid, π_ref)
        Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
        B = -cᵖ * θᵥᶠ * δz_π₀ / Δzᶠ - g

        wten[i, j, k] = (Gˢρw[i, j, k] / ρᶜᶜᶠ + B) * (k > 1)

        # Slow Exner pressure tendency: ppten = -u · ∇π
        #
        # The full π equation splits into slow and fast parts:
        #   ∂π'/∂t = ppten - ppterm · ∇·u
        # where ppterm·∇·u is the fast (acoustic) compression handled by the
        # acoustic backward step, and ppten captures slow advection of π.
        #
        # From π = (ρθ Rᵈ/p₀)^(Rᵈ/cᵥᵈ), the chain rule gives:
        #   dπ/dt = (R/cᵥ)(π/ρθ) · d(ρθ)/dt
        # The slow part is (R/cᵥ)(π/ρθ)·(-u·∇ρθ) = -u·∇π (no extra factor).
        #
        # Computing ppten = -u·∇π directly (rather than from Gˢρθ) avoids a
        # discretization mismatch: Gˢρθ uses WENO flux-divergence while the
        # compression correction ρθ·∇·u uses centered differences.
        #
        # We use centered differences for ∇π. Since π_ref varies only in z,
        # horizontal derivatives involve only π'.
        Δx = Δxᶠᶜᶜ(i, j, k, grid)
        Δy = Δyᶜᶠᶜ(i, j, k, grid)
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)

        # u · ∂π/∂x: centered average of u to cell center × centered π' gradient
        uᶜ = (u[i, j, k] + u[i + 1, j, k]) / 2
        ∂π_∂x = (π′[i + 1, j, k] - π′[i - 1, j, k]) / (2 * Δx)

        # v · ∂π/∂y: (zero for Flat y; centered for Periodic/Bounded)
        vᶜ = (v[i, j, k] + v[i, j + 1, k]) / 2
        ∂π_∂y = (π′[i, j + 1, k] - π′[i, j - 1, k]) / (2 * Δy)

        # w · ∂π/∂z: full π = π_ref + π', using centered differences
        # At boundaries, π values from halos; w→0 at solid boundaries
        π_above = ifelse(k == Nz, π_ref[i, j, k] + π′[i, j, k],
                         π_ref[i, j, k + 1] + π′[i, j, k + 1])
        π_below = ifelse(k == 1, π_ref[i, j, k] + π′[i, j, k],
                         π_ref[i, j, k - 1] + π′[i, j, k - 1])
        wᶜ = ifelse(k == 1, w[i, j, k + 1] / 2,
              ifelse(k == Nz, w[i, j, k] / 2,
                     (w[i, j, k] + w[i, j, k + 1]) / 2))
        ∂π_∂z = (π_above - π_below) / (2 * Δzᶜ)

        u_dot_grad_π = uᶜ * ∂π_∂x + vᶜ * ∂π_∂y + wᶜ * ∂π_∂z

        # ppten = -u · ∇π (no extra R/cᵥ factor needed)
        # The chain rule already accounts for it: u·∇π = (R/cᵥ)(π/ρθ)·u·∇(ρθ),
        # so -u·∇π = (R/cᵥ)(π/ρθ)·(-u·∇ρθ) which is the correct slow π tendency.
        ppten[i, j, k] = -u_dot_grad_π
    end
end

#####
##### Section 5: Acoustic forward step — horizontal velocity only
#####
##### CM1 equivalent: sound.F lines 320-340 (no-terrain Cartesian)
##### u += dts * (uten - cp*0.5*(ppd[i]-ppd[i-1])/dx * (thv[i]+thv[i-1]))
#####
##### The vertical velocity w is handled by the implicit tridiagonal solver
##### (Section 7), following CM1's always-implicit approach.
#####

@kernel function _acoustic_horizontal_forward!(u, v, grid, Δτ, cᵖ,
                                               π′_damped, θᵥ, uten, vten)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # x-velocity: u += Δτ * (uten - cp * avg(θᵥ) * ∂π'_d/∂x)
        θᵥᶠᶜᶜ = ℑxTᶠᵃᵃ(i, j, k, grid, θᵥ)
        ∂x_π = δxTᶠᵃᵃ(i, j, k, grid, π′_damped) / Δxᶠᶜᶜ(i, j, k, grid)
        u[i, j, k] += Δτ * (uten[i, j, k] - cᵖ * θᵥᶠᶜᶜ * ∂x_π) * !on_x_boundary(i, j, k, grid)

        # y-velocity: v += Δτ * (vten - cp * avg(θᵥ) * ∂π'_d/∂y)
        θᵥᶜᶠᶜ = ℑyTᵃᶠᵃ(i, j, k, grid, θᵥ)
        ∂y_π = δyTᵃᶠᵃ(i, j, k, grid, π′_damped) / Δyᶜᶠᶜ(i, j, k, grid)
        v[i, j, k] += Δτ * (vten[i, j, k] - cᵖ * θᵥᶜᶠᶜ * ∂y_π) * !on_y_boundary(i, j, k, grid)
    end
end

#####
##### Section 6: Compute fpk — explicit Exner pressure tendency
#####
##### CM1 equivalent: sound.F lines 490-510 (no-terrain Cartesian)
##### fpk includes: slow tendency, horizontal divergence, β-weighted old-w vertical terms
#####

@kernel function _compute_fpk!(fpk, grid, Δτ, β,
                               u, v, w, ppterm, ppten)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Horizontal velocity divergence (using NEW u, v from forward step)
        div_h = (u[i+1, j, k] - u[i, j, k]) / Δxᶠᶜᶜ(i, j, k, grid) +
                (v[i, j+1, k] - v[i, j, k]) / Δyᶜᶠᶜ(i, j, k, grid)

        # Slow tendency + horizontal divergence
        fpk_val = Δτ * (ppten[i, j, k] - ppterm[i, j, k] * div_h)

        # β-weighted vertical divergence from OLD w
        w_bot = ifelse(k == 1, zero(eltype(w)), w[i, j, k])
        w_top = ifelse(k == Nz, zero(eltype(w)), w[i, j, k + 1])
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)
        fpk_val -= β * Δτ * ppterm[i, j, k] * (w_top - w_bot) / Δzᶜ

        fpk[i, j, k] = fpk_val
    end
end

#####
##### Section 7: Implicit vertical w-π' solve
#####
##### CM1 equivalent: sound.F lines 660-730 (vertically implicit solver)
##### Solves a tridiagonal system for w that arises from coupling the
##### vertical momentum equation (w depends on π') with the pressure
##### equation (π' depends on w).
#####
##### The off-centering parameter α provides damping of vertical acoustic modes.
##### The tridiagonal system is: aa[k]*w[k-1] + bb[k]*w[k] + cc[k]*w[k+1] = dd[k]
#####

"""
$(TYPEDSIGNATURES)

Solve the vertically implicit w-π' system, then update w.

Instead of solving a tridiagonal system for w (which requires face-indexed
arrays), we solve for π' at cell centers (matching the solver dimensions),
then back-solve for w.

The approach:
1. Substitute ``w_{new}[k] = w[k] + Δτ w_{ten}[k] - Δτ mm[k] δz(π'_{new})``
   into the pressure equation ``π'_{new} = π' + fpk - α Δτ ppterm ∂w_{new}/∂z``
2. This gives a tridiagonal system in π'_new at center locations
3. After solving for π'_new, update w from the new pressure gradient
"""
function implicit_w_solve!(w, substepper, model, Δτ, fpk)
    grid = model.grid
    arch = architecture(grid)
    α = substepper.forward_weight
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity
    g = model.thermodynamic_constants.gravitational_acceleration
    solver = substepper.vertical_solver

    # Build tridiagonal system for π' and solve
    launch!(arch, grid, :xyz, _build_pi_tridiagonal!,
            solver.a, solver.b, solver.c, substepper.rhs,
            grid, α, Δτ, cᵖ,
            w, substepper.π′, fpk,
            substepper.θᵥ, substepper.ppterm,
            substepper.slow_velocity_tendencies.w)

    # Solve: A * π'_new = rhs → result goes into π'
    solve!(substepper.π′, solver, substepper.rhs)

    # Update w from the new pressure + reference PGF (off-centered)
    launch!(arch, grid, :xyz, _update_w_from_pressure!,
            w, grid, α, Δτ, cᵖ, g,
            substepper.π′, substepper.π′_old, substepper.θᵥ,
            substepper.pᵣ, substepper.Bᵣ, substepper.ρᵣ,
            substepper.slow_velocity_tendencies.w)

    return nothing
end

@kernel function _build_pi_tridiagonal!(lower, diag, upper, rhs_field,
                                        grid, α, Δτ, cᵖ,
                                        w, π′, fpk,
                                        θᵥ, ppterm,
                                        wten)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)

        # mm at face k and k+1 (vertical PGF coefficient for w from π')
        Δzᶠ_bot = Δzᶜᶜᶠ(i, j, k, grid)
        Δzᶠ_top = Δzᶜᶜᶠ(i, j, k + 1, grid)
        θᵥᶠ_bot = ℑzTᵃᵃᶠ(i, j, k, grid, θᵥ)
        θᵥᶠ_top = ℑzTᵃᵃᶠ(i, j, k + 1, grid, θᵥ)
        mm_bot = cᵖ * θᵥᶠ_bot / Δzᶠ_bot
        mm_top = cᵖ * θᵥᶠ_top / Δzᶠ_top

        # Coupling coefficients
        pptermⁱ = ppterm[i, j, k]
        Q_bot = α * α * Δτ * Δτ * pptermⁱ * mm_bot / Δzᶜ
        Q_top = α * α * Δτ * Δτ * pptermⁱ * mm_top / Δzᶜ

        Q_bot = ifelse(k == 1, zero(Q_bot), Q_bot)
        Q_top = ifelse(k == Nz, zero(Q_top), Q_top)

        lower[i, j, k] = -Q_bot
        upper[i, j, k] = -Q_top
        diag[i, j, k] = 1 + Q_bot + Q_top

        # w_explicit at faces: includes slow tendency + acoustic π' perturbation
        # The vertical PGF is handled through π' (which has zero hydrostatic residual
        # because π_ref was built using the same θᵥ averaging as the w-update).
        δz_π_bot = ifelse(k == 1, zero(eltype(π′)), π′[i, j, k] - π′[i, j, k - 1])
        δz_π_top = ifelse(k == Nz, zero(eltype(π′)), π′[i, j, k + 1] - π′[i, j, k])

        β = 1 - α
        w_exp_bot = ifelse(k == 1, zero(eltype(w)),
                           w[i, j, k] + Δτ * wten[i, j, k] - β * Δτ * mm_bot * δz_π_bot)
        w_exp_top = ifelse(k == Nz, zero(eltype(w)),
                           w[i, j, k + 1] + Δτ * wten[i, j, k + 1] - β * Δτ * mm_top * δz_π_top)

        div_w_exp = (w_exp_top - w_exp_bot) / Δzᶜ

        rhs_val = π′[i, j, k] + fpk[i, j, k] - α * Δτ * pptermⁱ * div_w_exp
        rhs_field[i, j, k] = rhs_val
    end
end

@kernel function _update_w_from_pressure!(w, grid, α, Δτ, cᵖ, g,
                                          π′_new, π′_old, θᵥ,
                                          pᵣ, Bᵣ, ρᵣ,
                                          wten)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
        θᵥᶠ = ℑzTᵃᵃᶠ(i, j, k, grid, θᵥ)
        mm = cᵖ * θᵥᶠ / Δzᶠ

        # Off-centered vertical PGF: β*δz(π'_old) + α*δz(π'_new)
        β = 1 - α
        δz_π_old = δzTᵃᵃᶠ(i, j, k, grid, π′_old)
        δz_π_new = δzTᵃᵃᶠ(i, j, k, grid, π′_new)

        # w = w + Δτ*wten - Δτ*mm*(β*δz(π'_old) + α*δz(π'_new))
        w_new = w[i, j, k] + Δτ * wten[i, j, k] - Δτ * mm * (β * δz_π_old + α * δz_π_new)
        w[i, j, k] = w_new * (k > 1)
    end
end

#####
##### Section 8: Update π' with new w, apply damping, accumulate averages
#####
##### After the implicit w solve, update π' using the NEW w (α-weighted)
##### and apply the CM1-style divergence damping.
#####

@kernel function _update_pressure_and_average!(π′, π′_damped, π′_old,
                                               u, v, w, ū,
                                               grid, kdiv, avg_weight)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # π' was already updated by the implicit solve.
        # Apply divergence damping: π'_damped = π' + kdiv * (π' - π'_old)
        π′_new = π′[i, j, k]
        π′_old_val = π′_old[i, j, k]
        π′_damped[i, j, k] = π′_new + kdiv * (π′_new - π′_old_val)

        # Save current π' as old for next substep
        π′_old[i, j, k] = π′_new

        # Accumulate time-averaged velocities
        ū.u[i, j, k] += avg_weight * u[i, j, k]
        ū.v[i, j, k] += avg_weight * v[i, j, k]
        ū.w[i, j, k] += avg_weight * w[i, j, k]
    end
end

@kernel function _acoustic_divergence_damping!(u, v, π′, π′_old, θᵥ, grid, β_d, cᵖ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Klemp (2018) divergence damping: damp velocity proportional to the
        # PGF-scaled change in π' per substep. This provides constant damping
        # per outer Δt regardless of substep count N, stabilizing WS-RK3.
        #
        # u -= β_d * cₚ * θᵥ_face * ∂(Δπ')/∂x
        # v -= β_d * cₚ * θᵥ_face * ∂(Δπ')/∂y
        #
        # The cₚ·θᵥ factor matches the PGF scaling so that β_d is a
        # dimensionless O(1) coefficient (β_d ∈ [2, 10] typical).
        Δπ_i   = π′[i, j, k]     - π′_old[i, j, k]
        Δπ_im1 = π′[i - 1, j, k] - π′_old[i - 1, j, k]
        Δx = Δxᶠᶜᶜ(i, j, k, grid)
        θᵥᶠᶜᶜ = ℑxTᶠᵃᵃ(i, j, k, grid, θᵥ)
        u[i, j, k] -= β_d * cᵖ * θᵥᶠᶜᶜ * (Δπ_i - Δπ_im1) / Δx * !on_x_boundary(i, j, k, grid)

        Δπ_j   = π′[i, j, k]     - π′_old[i, j, k]
        Δπ_jm1 = π′[i, j - 1, k] - π′_old[i, j - 1, k]
        Δy = Δyᶜᶠᶜ(i, j, k, grid)
        θᵥᶜᶠᶜ = ℑyTᵃᶠᵃ(i, j, k, grid, θᵥ)
        v[i, j, k] -= β_d * cᵖ * θᵥᶜᶠᶜ * (Δπ_j - Δπ_jm1) / Δy * !on_y_boundary(i, j, k, grid)
    end
end

#####
##### Section 8: Zero fields
#####

@kernel function _zero_acoustic_fields!(ū, π′, π′_old, π′_damped)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ū.u[i, j, k] = 0
        ū.v[i, j, k] = 0
        ū.w[i, j, k] = 0
        π′[i, j, k] = 0
        π′_old[i, j, k] = 0
        π′_damped[i, j, k] = 0
    end
end

#####
##### Section 9: WS-RK3 substep loop
#####

"""
$(TYPEDSIGNATURES)

Execute the acoustic substep loop for a Wicker-Skamarock RK3 stage
using the Exner pressure formulation.

The acoustic substep size is constant: ``Δτ = Δt / N``.
Each stage takes ``Nτ = \\max(\\mathrm{round}(β N), 1)`` substeps.
"""
function acoustic_rk3_substep_loop!(model, substepper, Δt, β_stage, U⁰)
    N = substepper.substeps
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity

    # Constant acoustic substep size across all stages
    Δτ = Δt / N

    # Substep count varies per stage: Nτ ≈ β * N
    Nτ = max(round(Int, β_stage * N), 1)

    grid = model.grid
    arch = architecture(grid)

    # Convert slow tendencies to velocity/pressure form
    convert_slow_tendencies!(substepper, model)

    # Initialize time-averaged velocities to zero
    ū = substepper.averaged_velocities
    launch!(arch, grid, :xyz, _zero_avg_velocities!, ū)

    # WS-RK3: reset π' to π'(Uⁿ), not π'(U_eval).
    # The acoustic loop must start from a CONSISTENT Uⁿ state (both velocity
    # AND pressure from Uⁿ). Starting π' from U_eval while velocities are from
    # Uⁿ creates an imbalance that destabilizes the acoustic loop at large Δt.
    # θᵥ, ppterm, π_ref remain from U_eval (frozen thermodynamic quantities).
    pˢᵗ = model.dynamics.standard_pressure
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖ
    launch!(arch, grid, :xyz, _reset_pi_prime_to_U0!,
            substepper.π′, substepper.π_ref, U⁰[5], pˢᵗ, Rᵈ, κ)

    parent(substepper.π′_damped) .= parent(substepper.π′)
    parent(substepper.π′_old) .= parent(substepper.π′)

    # Save π'_initial in ρχᵣ for the perturbation recovery.
    # With π' reset to π'(Uⁿ), the recovery computes:
    #   π_new = π(Uⁿ) + Δπ' = π_ref + π'_final  (they cancel)
    # so ρθ_new = EOS(π_ref + π'_final).
    parent(substepper.ρχᵣ) .= parent(substepper.π′)

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

    α = substepper.forward_weight
    β = 1 - α
    kdiv = substepper.divergence_damping_coefficient
    β_d = substepper.acoustic_damping_coefficient

    fpk = CenterField(grid)  # TODO: pre-allocate this

    for n in 1:Nτ
        # Step 1: Horizontal forward — update u, v
        launch!(arch, grid, :xyz, _acoustic_horizontal_forward!,
                u, v, grid, Δτ, cᵖ,
                substepper.π′_damped, substepper.θᵥ,
                substepper.slow_velocity_tendencies.u,
                substepper.slow_velocity_tendencies.v)

        # Step 2: Compute fpk — explicit π' tendency (horizontal div + β*old-w)
        launch!(arch, grid, :xyz, _compute_fpk!,
                fpk, grid, Δτ, β,
                u, v, w, substepper.ppterm, substepper.ppten)

        # Save π' before implicit solve (for damping)
        parent(substepper.π′_old) .= parent(substepper.π′)

        # Step 3: Implicit solve — tridiagonal for π' + update w
        implicit_w_solve!(w, substepper, model, Δτ, fpk)

        # Step 3b: Klemp 2018 divergence damping (if β_d > 0)
        # Damp horizontal velocities proportional to ∂(Δπ')/∂x where
        # Δπ' = π'_new - π'_old is the change from this substep's solve.
        # Total damping per outer Δt is constant regardless of N.
        if β_d > 0
            launch!(arch, grid, :xyz, _acoustic_divergence_damping!,
                    u, v, substepper.π′, substepper.π′_old,
                    substepper.θᵥ, grid, β_d, cᵖ)
        end

        # Step 4: Apply kdiv damping + accumulate velocity averages
        launch!(arch, grid, :xyz, _update_pressure_and_average!,
                substepper.π′, substepper.π′_damped, substepper.π′_old,
                u, v, w, ū,
                grid, kdiv, 1 / Nτ)
    end

    # Recovery: convert acoustic variables back to Breeze prognostic fields.
    # Pass the stage time (β·Δt) for slow θ evolution in recovery.
    Δt_stage = Nτ * Δτ
    recover_full_fields!(model, substepper, U⁰, Δt_stage)

    return nothing
end

@kernel function _reset_pi_prime_to_U0!(π′, π_ref, ρχ⁰, pˢᵗ, Rᵈ, κ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # Compute π(Uⁿ) from ρθⁿ via the equation of state: π = (Rd·ρθ/p₀)^(R/cv)
        R_over_cv = κ / (1 - κ)
        πⁿ = (Rᵈ * ρχ⁰[i, j, k] / pˢᵗ)^R_over_cv
        π′[i, j, k] = πⁿ - π_ref[i, j, k]
    end
end

@kernel function _zero_avg_velocities!(ū)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ū.u[i, j, k] = 0
        ū.v[i, j, k] = 0
        ū.w[i, j, k] = 0
    end
end

@kernel function _accumulate_density_change!(Δρ, grid, Δτ, ρ₀, u, v, w)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Velocity divergence: ∂u/∂x + ∂v/∂y + ∂w/∂z
        # Must use the SAME discrete divergence operator as the pressure equation
        # (in _compute_fpk!) so that ρ and ρθ evolve consistently.
        # Using ∇·(ρ₀u) instead would introduce an extra u·∇ρ₀ term not present
        # in the pressure equation, causing ρ-ρθ inconsistency and drift.
        div_u = (u[i + 1, j, k] - u[i, j, k]) / Δxᶠᶜᶜ(i, j, k, grid) +
                (v[i, j + 1, k] - v[i, j, k]) / Δyᶜᶠᶜ(i, j, k, grid)

        w_top = ifelse(k == Nz, zero(eltype(w)), w[i, j, k + 1])
        w_bot = ifelse(k == 1, zero(eltype(w)), w[i, j, k])
        div_u += (w_top - w_bot) / Δzᶜᶜᶜ(i, j, k, grid)

        # Linearized continuity equation: ∂ρ/∂t = -ρ₀ ∇·u
        Δρ[i, j, k] -= Δτ * ρ₀[i, j, k] * div_u
    end
end

@kernel function _compute_slow_theta_tendency!(Gˢθ_out, Gˢρχ, Gˢρ, θᵥ, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # Material derivative of θ: Gˢθ = (Gˢρθ - θ · Gˢρ) / ρ
        Gˢθ_out[i, j, k] = (Gˢρχ[i, j, k] - θᵥ[i, j, k] * Gˢρ[i, j, k]) / ρ[i, j, k]
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

@kernel function _reset_pi_prime_to_U0!(π′, π′_damped, ρ⁰, ρθ⁰, π_ref,
                                        constants, pˢᵗ, κ, cᵖ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρⁱ = ρ⁰[i, j, k]
        ρθⁱ = ρθ⁰[i, j, k]
        θⁱ = ρθⁱ / ρⁱ
        Rᵈ = cᵖ * κ
        # T = θ^γ (ρ R / pˢᵗ)^(γ-1) — standard formula
        γ = cᵖ / (cᵖ - Rᵈ)
        Tⁱ = θⁱ^γ * (ρⁱ * Rᵈ / pˢᵗ)^(γ - 1)
        pⁱ = ρⁱ * Rᵈ * Tⁱ
        πⁱ = (pⁱ / pˢᵗ)^κ
        π′_val = πⁱ - π_ref[i, j, k]
        π′[i, j, k] = π′_val
        π′_damped[i, j, k] = π′_val
    end
end

#####
##### Section 10: Recovery kernels
#####

"""
$(TYPEDSIGNATURES)

Recover full fields from Exner pressure acoustic variables.

After the acoustic loop, convert the updated velocity and Exner pressure
back to Breeze's prognostic variables (ρ, ρu, ρv, ρw, ρθ).

For WS-RK3: The recovery uses U⁰ as the base state and adds the
change computed by the acoustic loop. The velocity fields were modified
in-place during the loop, so we need to compute the change and apply
the WS-RK3 formula.
"""
function recover_full_fields!(model, substepper, U⁰, Δt_stage)
    grid = model.grid
    arch = architecture(grid)
    ρχ = thermodynamic_density(model.formulation)
    pˢᵗ = model.dynamics.standard_pressure
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖ

    # Nonlinear recovery for WS-RK3:
    # ρθ: π'-perturbation approach — apply WS-RK3 perturbation in π'-space,
    #   then convert once via the equation of state. Avoids nonlinear splitting.
    # ρ: diagnosed from ρ = ρθ / θ_new where θ_new = θⁿ + Δt_stage · Gˢθ.
    #   θⁿ = ρθⁿ/ρⁿ from U⁰ (initial state), NOT θᵥ from the evaluation state.
    #   Using θᵥ would double-count the θ change from earlier stages.
    #   Gˢθ = (Gˢρθ - θᵥ·Gˢρ)/ρ is the slow θ tendency at the evaluation state.
    # π'_initial is saved in ρχᵣ; Gˢρχ/Gˢρ hold slow tendencies.
    launch!(arch, grid, :xyz, _nonlinear_recovery_wsrk3!,
            model.dynamics.density, ρχ,
            substepper.π′, substepper.ρχᵣ,
            substepper.π_ref,
            substepper.θᵥ, substepper.Gˢρχ, substepper.Gˢρ,
            U⁰[1], U⁰[5], pˢᵗ, Rᵈ, κ, Δt_stage)

    # Reconstruct momentum from updated density and velocity
    launch!(arch, grid, :xyz, _recover_momentum!,
            model.momentum, model.dynamics.density, model.velocities, grid)

    return nothing
end

@kernel function _linearized_recovery!(ρ, ρχ, π′, π′_initial, ρᵣ, ρχᵣ, π_ref, θᵥ, ρ⁰, ρχ⁰, κ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Linearized conversion from Δπ' to ρθ perturbation:
        # π ∝ (ρθ)^(R/cv), so δπ = (R/cv)(π/ρθ) δ(ρθ)
        # Invert: δ(ρθ) = (cv/R)(ρθ/π) δπ = ((1-κ)/κ)(ρθ/π) Δπ'
        # Use the CHANGE in π' (not total) to avoid double-counting.
        rcv_over_R = (1 - κ) / κ  # cv/R
        πᵣ = π_ref[i, j, k]
        Δπ = π′[i, j, k] - π′_initial[i, j, k]
        ρχ_perturbation = rcv_over_R * ρχᵣ[i, j, k] / πᵣ * Δπ

        # WS-RK3: U_new = U⁰ + perturbation
        ρχ[i, j, k] = ρχ⁰[i, j, k] + ρχ_perturbation
        ρ[i, j, k] = ρχ[i, j, k] / θᵥ[i, j, k]
    end
end

@kernel function _nonlinear_recovery_wsrk3!(ρ, ρχ, π′_final, π′_initial, π_ref,
                                             θᵥ, Gˢρχ, Gˢρ,
                                             ρ⁰, ρχ⁰, pˢᵗ, Rᵈ, κ, Δt_stage)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        cv_over_R = (1 - κ) / κ
        R_over_cv = κ / (1 - κ)
        coeff = pˢᵗ / Rᵈ

        # WS-RK3 perturbation applied to π' (the natural acoustic variable).
        # π'_new = π'⁰ + (π'_final - π'_initial), then convert to ρθ.
        Δπ′ = π′_final[i, j, k] - π′_initial[i, j, k]

        # Compute π⁰ from ρθ⁰ via the equation of state
        ρχ⁰_ijk = ρχ⁰[i, j, k]
        π⁰ = (Rᵈ * ρχ⁰_ijk / pˢᵗ)^R_over_cv

        # Apply WS-RK3 perturbation in π'-space, then convert to ρθ
        π_new = π⁰ + Δπ′
        ρχ_new = coeff * π_new^cv_over_R
        ρχ[i, j, k] = ρχ_new

        # Density: ρ = ρθ / θ_new, where θ_new = θⁿ + Δt_stage · Gˢθ.
        # WS-RK3 requires the θ BASE to be from the initial state Uⁿ (not the
        # evaluation state U*). Using θᵥ from U* would double-count the θ
        # change from earlier stages (θ(U*) = θⁿ + β₁·Δt·Gˢθ already).
        # The slow Gˢθ is evaluated at U* (correct for WS-RK3).
        ρ_eval = ρ[i, j, k]
        θᵥ_eval = θᵥ[i, j, k]
        θⁿ_ijk = ρχ⁰_ijk / ρ⁰[i, j, k]
        θ_new = θⁿ_ijk + Δt_stage * (Gˢρχ[i, j, k] - θᵥ_eval * Gˢρ[i, j, k]) / ρ_eval
        ρ[i, j, k] = ρχ_new / θ_new
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

#####
##### Section 11: SSP-RK3 substep loop (delegates to same acoustic loop)
#####

"""
$(TYPEDSIGNATURES)

Execute the acoustic substep loop for an SSP RK3 stage.
Delegates to the same Exner pressure acoustic loop used by WS-RK3.
"""
function acoustic_substep_loop!(model, substepper, Δt, α_ssp, U⁰)
    # For SSP-RK3, all stages use Ns substeps
    Ns = substepper.substeps
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity

    Δτ = Δt / Ns
    Nτ = Ns

    grid = model.grid
    arch = architecture(grid)

    # Convert slow tendencies
    convert_slow_tendencies!(substepper, model)

    # Initialize time-averaged velocities to zero
    ū = substepper.averaged_velocities
    launch!(arch, grid, :xyz, _zero_avg_velocities!, ū)

    parent(substepper.π′_damped) .= parent(substepper.π′)
    parent(substepper.π′_old) .= parent(substepper.π′)

    u = model.velocities.u
    v = model.velocities.v
    w = model.velocities.w

    # Off-centering parameter for implicit solver (NOT the SSP coefficient)
    α_implicit = substepper.forward_weight
    β_implicit = 1 - α_implicit
    kdiv = substepper.divergence_damping_coefficient
    β_d = substepper.acoustic_damping_coefficient
    fpk = CenterField(grid)  # TODO: pre-allocate

    for n in 1:Nτ
        launch!(arch, grid, :xyz, _acoustic_horizontal_forward!,
                u, v, grid, Δτ, cᵖ,
                substepper.π′_damped, substepper.θᵥ,
                substepper.slow_velocity_tendencies.u,
                substepper.slow_velocity_tendencies.v)

        launch!(arch, grid, :xyz, _compute_fpk!,
                fpk, grid, Δτ, β_implicit,
                u, v, w, substepper.ppterm, substepper.ppten)

        parent(substepper.π′_old) .= parent(substepper.π′)
        implicit_w_solve!(w, substepper, model, Δτ, fpk)

        if β_d > 0
            launch!(arch, grid, :xyz, _acoustic_divergence_damping!,
                    u, v, substepper.π′, substepper.π′_old,
                    substepper.θᵥ, grid, β_d, cᵖ)
        end

        launch!(arch, grid, :xyz, _update_pressure_and_average!,
                substepper.π′, substepper.π′_damped, substepper.π′_old,
                u, v, w, ū,
                grid, kdiv, 1 / Nτ)
    end

    # Recovery uses π'_final: convert back to prognostic fields
    # with SSP convex combination (uses SSP coefficient, not forward_weight)
    recover_full_fields_ssp!(model, substepper, α_ssp, U⁰, Δt)

    return nothing
end

"""
$(TYPEDSIGNATURES)

SSP-RK3 recovery: ``U_{new} = (1 - α) U⁰ + α U_{acoustic}``

Uses nonlinear recovery from π' to ρθ via the equation of state:
``ρθ_{acoustic} = (p_{st}/R_d) (π_{ref} + π'_{final})^{c_v/R}``

This is exact for dry air and avoids linearization errors that could
accumulate when the acoustic loop runs many substeps.
"""
function recover_full_fields_ssp!(model, substepper, α, U⁰, Δt)
    grid = model.grid
    arch = architecture(grid)
    ρχ = thermodynamic_density(model.formulation)
    pˢᵗ = model.dynamics.standard_pressure
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖ

    m⁰ = (ρu = U⁰[2], ρv = U⁰[3], ρw = U⁰[4])

    # Nonlinear recovery from π' to ρθ using the equation of state:
    # ρθ = (pˢᵗ/Rᵈ) * π^(cv/R) where π = π_ref + π'
    # Density is diagnosed from ρ = ρθ/θ_new where θ_new includes the
    # slow advective θ tendency accumulated over Δt.
    launch!(arch, grid, :xyz, _nonlinear_recovery!,
            model.dynamics.density, ρχ,
            substepper.π′, substepper.π_ref, substepper.θᵥ,
            substepper.Gˢρχ, substepper.Gˢρ,
            pˢᵗ, Rᵈ, κ, Δt)

    # Reconstruct momentum from acoustic velocity and recovered density
    launch!(arch, grid, :xyz, _recover_momentum!,
            model.momentum, model.dynamics.density, model.velocities, grid)

    # Apply SSP convex combination:
    # U_final = (1-α) U⁰ + α U_acoustic
    launch!(arch, grid, :xyz, _ssp_convex_combination!,
            model.momentum, model.dynamics.density, ρχ,
            m⁰, U⁰[1], U⁰[5], α)

    return nothing
end

@kernel function _nonlinear_recovery!(ρ, ρχ, π′, π_ref, θᵥ, Gˢρχ, Gˢρ, pˢᵗ, Rᵈ, κ, Δt)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Nonlinear equation of state: ρθ = (pˢᵗ/Rᵈ) * π^(cv/R)
        # where cv/R = (1-κ)/κ and π = π_ref + π'
        cv_over_R = (1 - κ) / κ
        π_total = π_ref[i, j, k] + π′[i, j, k]
        ρχ_new = (pˢᵗ / Rᵈ) * π_total^cv_over_R
        ρχ[i, j, k] = ρχ_new

        # Update θ with slow advection: θ_new = θᵥ + Δt · Gˢθ
        # where Gˢθ = (Gˢρθ - θ · Gˢρ) / ρ is the material derivative of θ.
        # Without this, θ = ρθ/ρ = θᵥ = frozen, preventing θ evolution.
        ρ_old = ρ[i, j, k]
        θᵥ_ijk = θᵥ[i, j, k]
        θ_new = θᵥ_ijk + Δt * (Gˢρχ[i, j, k] - θᵥ_ijk * Gˢρ[i, j, k]) / ρ_old
        ρ[i, j, k] = ρχ_new / θ_new
    end
end

@kernel function _ssp_convex_combination!(m, ρ, ρχ, m⁰, ρ⁰, ρχ⁰, α)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        m.ρu[i, j, k] = α * m.ρu[i, j, k] + (1 - α) * m⁰.ρu[i, j, k]
        m.ρv[i, j, k] = α * m.ρv[i, j, k] + (1 - α) * m⁰.ρv[i, j, k]
        m.ρw[i, j, k] = α * m.ρw[i, j, k] + (1 - α) * m⁰.ρw[i, j, k]
        ρ[i, j, k]    = α * ρ[i, j, k]    + (1 - α) * ρ⁰[i, j, k]
        ρχ[i, j, k]   = α * ρχ[i, j, k]   + (1 - α) * ρχ⁰[i, j, k]
    end
end

#####
##### Section 12: Unused legacy functions (kept for API compatibility)
#####

add_base_state_pressure_correction!(substepper, model) = nothing
add_base_state_pressure_correction!(substepper, model, ::Nothing) = nothing
add_base_state_pressure_correction!(substepper, model, ref) = nothing
