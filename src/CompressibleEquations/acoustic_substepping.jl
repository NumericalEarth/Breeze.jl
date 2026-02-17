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
- `acoustic_damping_coefficient`: Klemp 2018 β_d for velocity damping
- `virtual_potential_temperature`: Stage-frozen θᵥ (CenterField)
- `acoustic_compression`: Coefficient `(γ-1)π₀` converting `∇·u` to `∂π'/∂t` (CenterField)
- `reference_exner_function`: Reference π₀ = (p_ref/pˢᵗ)^(R/cᵖ) (CenterField)
- `exner_perturbation`: Current Exner pressure perturbation π' = π - π₀ (CenterField)
- `previous_exner_perturbation`: Previous-substep π' for divergence damping (CenterField)
- `damped_exner_perturbation`: Damped π' used in PGF (CenterField)
- `stage_thermodynamic_density`: Stage-frozen ρθ (CenterField)
- `averaged_velocities`: Time-averaged velocities for scalar advection
- `slow_tendencies`: Frozen slow tendencies (momentum, density, thermodynamic_density, velocity, exner_pressure)
- `vertical_solver`: BatchedTridiagonalSolver for implicit w-π' coupling
- `rhs`: Right-hand side storage for tridiagonal solve
"""
struct AcousticSubstepper{N, FT, CF, AV, ST, TS}
    substeps :: N                              # Number of acoustic substeps per full Δt
    forward_weight :: FT                       # Off-centering α (CM1 default 0.6), β = 1 - α
    divergence_damping_coefficient :: FT       # CM1-style kdiv for π' damping
    acoustic_damping_coefficient :: FT         # Klemp 2018 β_d
    virtual_potential_temperature :: CF        # Stage-frozen θᵥ
    acoustic_compression :: CF                 # (γ-1)π₀ — converts ∇·u to ∂π'/∂t
    reference_exner_function :: CF             # π₀ from reference state
    exner_perturbation :: CF                   # Current π' = π - π₀
    previous_exner_perturbation :: CF          # Previous-substep π' (for damping)
    damped_exner_perturbation :: CF            # Damped π' used in PGF
    stage_thermodynamic_density :: CF          # Stage-frozen ρθ
    averaged_velocities :: AV                  # Time-averaged velocities for scalar advection
    slow_tendencies :: ST                      # Frozen slow tendencies (NamedTuple)
    vertical_solver :: TS                      # BatchedTridiagonalSolver for implicit w-π' coupling
    rhs :: CF                                  # Right-hand side storage for tridiagonal solve
end

function _adapt_slow_tendencies(to, st)
    return (momentum = map(f -> adapt(to, f), st.momentum),
            density = adapt(to, st.density),
            thermodynamic_density = adapt(to, st.thermodynamic_density),
            velocity = map(f -> adapt(to, f), st.velocity),
            exner_pressure = adapt(to, st.exner_pressure))
end

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.substeps,
                       a.forward_weight,
                       a.divergence_damping_coefficient,
                       a.acoustic_damping_coefficient,
                       adapt(to, a.virtual_potential_temperature),
                       adapt(to, a.acoustic_compression),
                       adapt(to, a.reference_exner_function),
                       adapt(to, a.exner_perturbation),
                       adapt(to, a.previous_exner_perturbation),
                       adapt(to, a.damped_exner_perturbation),
                       adapt(to, a.stage_thermodynamic_density),
                       map(f -> adapt(to, f), a.averaged_velocities),
                       _adapt_slow_tendencies(to, a.slow_tendencies),
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

    virtual_potential_temperature = CenterField(grid)
    acoustic_compression = CenterField(grid)
    reference_exner_function = CenterField(grid)
    exner_perturbation = CenterField(grid)
    previous_exner_perturbation = CenterField(grid)
    damped_exner_perturbation = CenterField(grid)
    stage_thermodynamic_density = CenterField(grid)

    averaged_velocities = (u = XFaceField(grid),
                           v = YFaceField(grid),
                           w = ZFaceField(grid))

    slow_tendencies = (momentum = (ρu = XFaceField(grid),
                                   ρv = YFaceField(grid),
                                   ρw = ZFaceField(grid)),
                       density = CenterField(grid),
                       thermodynamic_density = CenterField(grid),
                       velocity = (u = XFaceField(grid),
                                   v = YFaceField(grid),
                                   w = ZFaceField(grid)),
                       exner_pressure = CenterField(grid))

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
                              virtual_potential_temperature,
                              acoustic_compression,
                              reference_exner_function,
                              exner_perturbation,
                              previous_exner_perturbation,
                              damped_exner_perturbation,
                              stage_thermodynamic_density,
                              averaged_velocities,
                              slow_tendencies,
                              vertical_solver,
                              rhs)
end

#####
##### Section 2b: Adaptive substep computation
#####

using Breeze.AtmosphereModels: thermodynamic_density
using Breeze.Thermodynamics: dry_air_gas_constant

"""
$(TYPEDSIGNATURES)

Compute the number of acoustic substeps from the horizontal acoustic CFL condition.

Uses a conservative sound speed estimate `cₛ = √(γ Rᵈ Tᵣ)` with `Tᵣ = 300 K`
(giving `cₛ ≈ 347 m/s`) and the minimum horizontal grid spacing. The vertical
CFL is not needed because the w-π' coupling is vertically implicit.

Following CM1, the substep count satisfies `Δτ · cₛ / Δx_min ≤ 1` where
`Δτ = Δt / N` is the acoustic substep size. A safety factor of 1.2 is applied
to ensure stability with the forward-backward splitting.
"""
function compute_acoustic_substeps(grid, Δt, thermodynamic_constants)
    cᵖ = thermodynamic_constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(thermodynamic_constants)
    cᵥ = cᵖ - Rᵈ
    γ = cᵖ / cᵥ
    Tᵣ = 300 # Conservative reference temperature (surface conditions)
    cₛ = sqrt(γ * Rᵈ * Tᵣ) # ≈ 347 m/s

    # Minimum horizontal grid spacing (skip Flat dimensions)
    TX, TY, _ = topology(grid)
    Δx_min = TX === Flat ? Inf : minimum_xspacing(grid)
    Δy_min = TY === Flat ? Inf : minimum_yspacing(grid)
    Δh_min = min(Δx_min, Δy_min)

    safety_factor = 1.2
    return ceil(Int, safety_factor * Δt * cₛ / Δh_min)
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

    # Store stage-frozen thermodynamic density (for recovery)
    χ = thermodynamic_density(model.formulation)
    parent(substepper.stage_thermodynamic_density) .= parent(χ)

    # Compute stage-frozen coefficients
    pˢᵗ = model.dynamics.standard_pressure
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖ  # R/cp

    launch!(arch, grid, :xyz, _prepare_exner_cache!,
            substepper.virtual_potential_temperature,
            substepper.acoustic_compression,
            substepper.exner_perturbation,
            substepper.damped_exner_perturbation,
            substepper.reference_exner_function,
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

    # Use the ExnerReferenceState's π₀ directly (exact discrete Exner hydrostatic balance).
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

@kernel function _prepare_exner_cache!(θᵥ_field, acoustic_compression_field, π′_field, π′_damped_field,
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

    # Pressure tendency coefficient: S = (γ-1) π₀
    # Derivation: from the ideal gas law and continuity equation,
    # ∂π'/∂t = -(γ-1) π₀ ∇·u. Combined with the momentum equation
    # ∂u/∂t = -cᵖ θᵥ ∂π'/∂x, this gives the correct acoustic wave speed:
    # c_eff² = cᵖ θᵥ S = cᵖ θᵥ (γ-1) π₀ = γ Rᵐ T = c²
    Sⁱ = (γᵐ - 1) * πⁱ

    # Exner pressure perturbation: π' = π - π_ref
    π_refⁱ = _get_reference_exner(i, j, k, reference_state, pˢᵗ, κ)

    @inbounds begin
        θᵥ_field[i, j, k] = θᵥⁱ
        acoustic_compression_field[i, j, k] = Sⁱ
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
    parent(substepper.reference_exner_function) .= parent(ref.exner_function)
    # Compute π' = π_actual - π₀
    launch!(arch, grid, :xyz, _recompute_pi_prime!,
            substepper.exner_perturbation, substepper.damped_exner_perturbation,
            model.dynamics.pressure, substepper.reference_exner_function, pˢᵗ, κ)
    return nothing
end

function _set_exner_reference!(substepper, model, ::Nothing, pˢᵗ, κ)
    grid = model.grid
    arch = architecture(grid)
    fill!(parent(substepper.reference_exner_function), 0)
    launch!(arch, grid, :xyz, _recompute_pi_prime!,
            substepper.exner_perturbation, substepper.damped_exner_perturbation,
            model.dynamics.pressure, substepper.reference_exner_function, pˢᵗ, κ)
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

function _compute_vertical_reference!(substepper, model, ref::ExnerReferenceState)
    grid = model.grid
    arch = architecture(grid)
    g = model.thermodynamic_constants.gravitational_acceleration

    # pᵣ = p_stage - p_ref
    parent(substepper.stage_pressure) .= parent(model.dynamics.pressure)
    parent(substepper.stage_pressure) .-= parent(ref.pressure)

    # bᵣ = -g(ρ_stage - ρ_ref) / ρ_stage (buoyancy in velocity form)
    launch!(arch, grid, :xyz, _compute_reference_buoyancy_velocity!,
            substepper.stage_buoyancy, model.dynamics.density, ref.density, g)
    return nothing
end

@kernel function _compute_reference_buoyancy_velocity!(bᵣ, ρ, ρ_ref, g)
    i, j, k = @index(Global, NTuple)
    @inbounds bᵣ[i, j, k] = -g * (ρ[i, j, k] - ρ_ref[i, j, k]) / ρ[i, j, k]
end

function _build_stage_hydrostatic_exner!(πᵣ, θᵥ, model)
    grid = model.grid
    arch = architecture(grid)
    g = model.thermodynamic_constants.gravitational_acceleration
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    pˢᵗ = model.dynamics.standard_pressure
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖᵈ

    # Set bottom value from actual stage pressure
    launch!(arch, grid, :xyz, _set_bottom_exner!,
            πᵣ, model.dynamics.pressure, pˢᵗ, κ)

    # Integrate upward using STAGE θᵥ (same as used by acoustic loop)
    Nz = size(grid, 3)
    launch!(arch, grid, :xy, _integrate_stage_hydrostatic_exner!,
            πᵣ, θᵥ, grid, g, cᵖᵈ, Nz)

    return nothing
end

@kernel function _set_bottom_exner!(πᵣ, p, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)
    @inbounds πᵣ[i, j, k] = (p[i, j, k] / pˢᵗ)^κ
end

@kernel function _integrate_stage_hydrostatic_exner!(πᵣ, θᵥ, grid, g, cᵖᵈ, Nz)
    i, j = @index(Global, NTuple)
    for k in 2:Nz
        @inbounds begin
            Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
            # Use the SAME face average as the w-update kernel: avg(θᵥ[k-1], θᵥ[k])
            θᵥᶠ = (θᵥ[i, j, k] + θᵥ[i, j, k-1]) / 2
            πᵣ[i, j, k] = πᵣ[i, j, k-1] - g * Δzᶠ / (cᵖᵈ * θᵥᶠ)
        end
    end
end

@inline _get_reference_exner(i, j, k, ::Nothing, pˢᵗ, κ) = zero(pˢᵗ)

@inline function _get_reference_exner(i, j, k, ref::ExnerReferenceState, pˢᵗ, κ)
    @inbounds return ref.exner_function[i, j, k]
end

#####
##### Section 4: Convert slow tendencies to velocity/pressure form
#####

"""
$(TYPEDSIGNATURES)

Convert slow momentum tendencies (Gˢρu, Gˢρv, Gˢρw) to slow velocity
tendencies (uten, vten, wten) and slow pressure tendency (Gˢπ).

The velocity tendency is: uten ≈ Gˢρu / ρ
The pressure tendency is: Gˢπ = -u · ∇π

These are frozen during the acoustic substep loop.
"""
function convert_slow_tendencies!(substepper, model)
    grid = model.grid
    arch = architecture(grid)
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    g = model.thermodynamic_constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖᵈ

    launch!(arch, grid, :xyz, _convert_slow_tendencies!,
            substepper.slow_tendencies.velocity.u,
            substepper.slow_tendencies.velocity.v,
            substepper.slow_tendencies.velocity.w,
            substepper.slow_tendencies.exner_pressure,
            substepper.slow_tendencies.momentum.ρu,
            substepper.slow_tendencies.momentum.ρv,
            substepper.slow_tendencies.momentum.ρw,
            substepper.slow_tendencies.thermodynamic_density,
            substepper.slow_tendencies.density,
            model.dynamics.density,
            model.velocities.u,
            model.velocities.v,
            model.velocities.w,
            substepper.stage_thermodynamic_density,
            substepper.exner_perturbation,
            substepper.reference_exner_function,
            substepper.virtual_potential_temperature,
            grid, κ, cᵖᵈ, g)

    return nothing
end

@kernel function _convert_slow_tendencies!(Gˢu, Gˢv, Gˢw, Gˢπ,
                                           Gˢρu, Gˢρv, Gˢρw, Gˢρχ, Gˢρ,
                                           ρ, u, v, w,
                                           ρχᵣ, π′, πᵣ, θᵥ,
                                           grid, κ, cᵖᵈ, g)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Velocity tendencies from momentum tendencies: Gˢu = Gˢρu / ρ
        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        Gˢu[i, j, k] = Gˢρu[i, j, k] / ρᶠᶜᶜ * !on_x_boundary(i, j, k, grid)

        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        Gˢv[i, j, k] = Gˢρv[i, j, k] / ρᶜᶠᶜ * !on_y_boundary(i, j, k, grid)

        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)

        # Buoyancy from the Exner pressure reference-state splitting:
        # Full vertical acceleration = -cᵖ θᵥ ∂π/∂z - g
        #   = -cᵖ θᵥ ∂π₀/∂z - cᵖ θᵥ ∂π'/∂z - g
        #   = b - cᵖ θᵥ ∂π'/∂z
        # where b = -cᵖ θᵥ_face δz(π₀)/Δz - g captures the buoyancy from the
        # mismatch between actual θᵥ and the reference θ₀ used to build π₀.
        # The acoustic loop provides -cᵖ θᵥ ∂π'/∂z; we add b as a slow tendency.
        θᵥᶠ = ℑzTᵃᵃᶠ(i, j, k, grid, θᵥ)
        δz_πᵣ = δzTᵃᵃᶠ(i, j, k, grid, πᵣ)
        Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
        b = -cᵖᵈ * θᵥᶠ * δz_πᵣ / Δzᶠ - g

        Gˢw[i, j, k] = (Gˢρw[i, j, k] / ρᶜᶜᶠ + b) * (k > 1)

        # Slow Exner pressure tendency: Gˢπ = -u · ∇π
        #
        # The full π equation splits into slow and fast parts:
        #   ∂π'/∂t = Gˢπ - S · ∇·u
        # where S·∇·u is the fast (acoustic) compression handled by the
        # acoustic backward step, and Gˢπ captures slow advection of π.
        #
        # From π = (ρθ Rᵈ/p₀)^(Rᵈ/cᵥᵈ), the chain rule gives:
        #   dπ/dt = (R/cᵥ)(π/ρθ) · d(ρθ)/dt
        # The slow part is (R/cᵥ)(π/ρθ)·(-u·∇ρθ) = -u·∇π (no extra factor).
        #
        # Computing Gˢπ = -u·∇π directly (rather than from Gˢρθ) avoids a
        # discretization mismatch: Gˢρθ uses WENO flux-divergence while the
        # compression correction ρθ·∇·u uses centered differences.
        #
        # We use centered differences for ∇π. Since πᵣ varies only in z,
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

        # w · ∂π/∂z: full π = πᵣ + π', using centered differences
        # At boundaries, π values from halos; w→0 at solid boundaries
        π_above = ifelse(k == Nz, πᵣ[i, j, k] + π′[i, j, k],
                         πᵣ[i, j, k + 1] + π′[i, j, k + 1])
        π_below = ifelse(k == 1, πᵣ[i, j, k] + π′[i, j, k],
                         πᵣ[i, j, k - 1] + π′[i, j, k - 1])
        wᶜ = ifelse(k == 1, w[i, j, k + 1] / 2,
              ifelse(k == Nz, w[i, j, k] / 2,
                     (w[i, j, k] + w[i, j, k + 1]) / 2))
        ∂π_∂z = (π_above - π_below) / (2 * Δzᶜ)

        u_dot_grad_π = uᶜ * ∂π_∂x + vᶜ * ∂π_∂y + wᶜ * ∂π_∂z

        # Gˢπ = -u · ∇π (no extra R/cᵥ factor needed)
        # The chain rule already accounts for it: u·∇π = (R/cᵥ)(π/ρθ)·u·∇(ρθ),
        # so -u·∇π = (R/cᵥ)(π/ρθ)·(-u·∇ρθ) which is the correct slow π tendency.
        Gˢπ[i, j, k] = -u_dot_grad_π
    end
end

#####
##### Section 5: Acoustic forward step — horizontal velocity only
#####
##### Updates u, v from the PGF (∂π'/∂x, ∂π'/∂y) and slow tendency (Gˢu, Gˢv).
##### The vertical velocity w is handled by the implicit tridiagonal solver
##### (Section 7).
#####

@kernel function _acoustic_horizontal_forward!(u, v, grid, Δτ, cᵖ,
                                               π′_damped, θᵥ, Gˢu, Gˢv)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # u += Δτ (Gˢu - cᵖ θᵥ ∂π'/∂x)
        θᵥᶠᶜᶜ = ℑxTᶠᵃᵃ(i, j, k, grid, θᵥ)
        ∂x_π = δxTᶠᵃᵃ(i, j, k, grid, π′_damped) / Δxᶠᶜᶜ(i, j, k, grid)
        u[i, j, k] += Δτ * (Gˢu[i, j, k] - cᵖ * θᵥᶠᶜᶜ * ∂x_π) * !on_x_boundary(i, j, k, grid)

        # v += Δτ (Gˢv - cᵖ θᵥ ∂π'/∂y)
        θᵥᶜᶠᶜ = ℑyTᵃᶠᵃ(i, j, k, grid, θᵥ)
        ∂y_π = δyTᵃᶠᵃ(i, j, k, grid, π′_damped) / Δyᶜᶠᶜ(i, j, k, grid)
        v[i, j, k] += Δτ * (Gˢv[i, j, k] - cᵖ * θᵥᶜᶠᶜ * ∂y_π) * !on_y_boundary(i, j, k, grid)
    end
end

#####
##### Section 6: Compute explicit Exner pressure forcing
#####
##### Explicit contribution to the π' update: slow tendency Gˢπ, horizontal
##### divergence from the updated u/v, and β-weighted vertical divergence
##### from the old (pre-solve) w.
#####

@kernel function _compute_π′_forcing!(π′_forcing, grid, Δτ, β,
                                      u, v, w, S, Gˢπ)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Horizontal velocity divergence (using updated u⁺, v⁺ from forward step)
        ∇ₕ_u = (u[i+1, j, k] - u[i, j, k]) / Δxᶠᶜᶜ(i, j, k, grid) +
                (v[i, j+1, k] - v[i, j, k]) / Δyᶜᶠᶜ(i, j, k, grid)

        # β-weighted vertical divergence from old w (before implicit solve)
        w⁻_bot = ifelse(k == 1, zero(eltype(w)), w[i, j, k])
        w⁻_top = ifelse(k == Nz, zero(eltype(w)), w[i, j, k + 1])
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)

        # π′_forcing = Δτ (Gˢπ - S ∇ₕ·u⁺) - β Δτ S ∂w⁻/∂z
        π′_forcing[i, j, k] = Δτ * (Gˢπ[i, j, k] - S[i, j, k] * ∇ₕ_u) -
                               β * Δτ * S[i, j, k] * (w⁻_top - w⁻_bot) / Δzᶜ
    end
end

#####
##### Section 7: Implicit vertical w-π' solve
#####
##### Solves a tridiagonal system coupling the vertical momentum equation
##### (w depends on π') with the pressure equation (π' depends on w).
##### The off-centering parameter α provides damping of vertical acoustic modes.
#####

"""
$(TYPEDSIGNATURES)

Solve the vertically implicit w-π' system, then update w.

Instead of solving a tridiagonal system for w (which requires face-indexed
arrays), we solve for π' at cell centers (matching the solver dimensions),
then back-solve for w.

The approach:
1. Substitute ``w⁺[k] = w[k] + Δτ Gˢw[k] - Δτ (cᵖ θᵥ / Δz) δz(π'⁺)``
   into the pressure equation ``π'⁺ = π' + π'_{forcing} - α Δτ S ∂w⁺/∂z``
2. This gives a tridiagonal system in π'⁺ at center locations
3. After solving for π'⁺, back-solve for w⁺ from the new pressure gradient
"""
function implicit_w_solve!(w, substepper, model, Δτ, π′_forcing)
    grid = model.grid
    arch = architecture(grid)
    α = substepper.forward_weight
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    solver = substepper.vertical_solver

    # Build tridiagonal system for π' and solve
    launch!(arch, grid, :xyz, _build_π′_tridiagonal!,
            solver.a, solver.b, solver.c, substepper.rhs,
            grid, α, Δτ, cᵖᵈ,
            w, substepper.exner_perturbation, π′_forcing,
            substepper.virtual_potential_temperature, substepper.acoustic_compression,
            substepper.slow_tendencies.velocity.w)

    # Solve: A π'⁺ = rhs → result overwrites π'
    solve!(substepper.exner_perturbation, solver, substepper.rhs)

    # Back-solve: w⁺ from the off-centered pressure gradient
    launch!(arch, grid, :xyz, _update_w_from_pressure!,
            w, grid, α, Δτ, cᵖᵈ,
            substepper.exner_perturbation, substepper.previous_exner_perturbation,
            substepper.virtual_potential_temperature,
            substepper.slow_tendencies.velocity.w)

    return nothing
end

@kernel function _build_π′_tridiagonal!(lower, diag, upper, rhs_field,
                                        grid, α, Δτ, cᵖᵈ,
                                        w, π′, π′_forcing,
                                        θᵥ, S,
                                        Gˢw)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)

        # Mᵖ = cᵖ θᵥ / Δz: vertical PGF coefficient (converts δπ' to acceleration)
        Δzᶠ_bot = Δzᶜᶜᶠ(i, j, k, grid)
        Δzᶠ_top = Δzᶜᶜᶠ(i, j, k + 1, grid)
        θᵥᶠ_bot = ℑzTᵃᵃᶠ(i, j, k, grid, θᵥ)
        θᵥᶠ_top = ℑzTᵃᵃᶠ(i, j, k + 1, grid, θᵥ)
        Mᵖ_bot = cᵖᵈ * θᵥᶠ_bot / Δzᶠ_bot
        Mᵖ_top = cᵖᵈ * θᵥᶠ_top / Δzᶠ_top

        # Tridiagonal coupling coefficients: Q = α² Δτ² S Mᵖ / Δz
        Sⁱ = S[i, j, k]
        Q_bot = α * α * Δτ * Δτ * Sⁱ * Mᵖ_bot / Δzᶜ
        Q_top = α * α * Δτ * Δτ * Sⁱ * Mᵖ_top / Δzᶜ

        Q_bot = ifelse(k == 1, zero(Q_bot), Q_bot)
        Q_top = ifelse(k == Nz, zero(Q_top), Q_top)

        lower[i, j, k] = -Q_bot
        upper[i, j, k] = -Q_top
        diag[i, j, k] = 1 + Q_bot + Q_top

        # Explicit w at faces: wᵉ = w + Δτ Gˢw - β Δτ Mᵖ δz(π')
        # π_ref has zero hydrostatic residual (built with same θᵥ averaging).
        δz_π_bot = ifelse(k == 1, zero(eltype(π′)), π′[i, j, k] - π′[i, j, k - 1])
        δz_π_top = ifelse(k == Nz, zero(eltype(π′)), π′[i, j, k + 1] - π′[i, j, k])

        β = 1 - α
        wᵉ_bot = ifelse(k == 1, zero(eltype(w)),
                         w[i, j, k] + Δτ * Gˢw[i, j, k] - β * Δτ * Mᵖ_bot * δz_π_bot)
        wᵉ_top = ifelse(k == Nz, zero(eltype(w)),
                         w[i, j, k + 1] + Δτ * Gˢw[i, j, k + 1] - β * Δτ * Mᵖ_top * δz_π_top)

        ∂z_wᵉ = (wᵉ_top - wᵉ_bot) / Δzᶜ

        # RHS = π' + π′_forcing - α Δτ S ∂wᵉ/∂z
        rhs_field[i, j, k] = π′[i, j, k] + π′_forcing[i, j, k] - α * Δτ * Sⁱ * ∂z_wᵉ
    end
end

@kernel function _update_w_from_pressure!(w, grid, α, Δτ, cᵖᵈ,
                                          π′⁺, π′⁻, θᵥ,
                                          Gˢw)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
        θᵥᶠ = ℑzTᵃᵃᶠ(i, j, k, grid, θᵥ)
        Mᵖ = cᵖᵈ * θᵥᶠ / Δzᶠ  # vertical PGF coefficient

        # Off-centered vertical PGF: β δz(π'⁻) + α δz(π'⁺)
        β = 1 - α
        δz_π⁻ = δzTᵃᵃᶠ(i, j, k, grid, π′⁻)
        δz_π⁺ = δzTᵃᵃᶠ(i, j, k, grid, π′⁺)

        # w⁺ = w + Δτ Gˢw - Δτ Mᵖ (β δz(π'⁻) + α δz(π'⁺))
        w⁺ = w[i, j, k] + Δτ * Gˢw[i, j, k] - Δτ * Mᵖ * (β * δz_π⁻ + α * δz_π⁺)
        w[i, j, k] = w⁺ * (k > 1)
    end
end

#####
##### Section 8: Update π' with new w, apply damping, accumulate averages
#####
##### After the implicit w solve, update π' using the NEW w (α-weighted)
##### and apply the CM1-style divergence damping.
#####

@kernel function _update_pressure_and_average!(π′, π′_damped, π′⁻,
                                               u, v, w, ū,
                                               grid, kdiv, avg_weight)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Apply divergence damping: π'_damped = π'⁺ + kdiv (π'⁺ - π'⁻)
        π′⁺ = π′[i, j, k]
        π′_damped[i, j, k] = π′⁺ + kdiv * (π′⁺ - π′⁻[i, j, k])

        # Save current π' as previous for next substep
        π′⁻[i, j, k] = π′⁺

        # Accumulate time-averaged velocities
        ū.u[i, j, k] += avg_weight * u[i, j, k]
        ū.v[i, j, k] += avg_weight * v[i, j, k]
        ū.w[i, j, k] += avg_weight * w[i, j, k]
    end
end

@kernel function _acoustic_divergence_damping!(u, v, π′, π′⁻, θᵥ, grid, β_d, cᵖ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Klemp (2018) divergence damping: damp velocity proportional to the
        # PGF-scaled change in π' per substep. This provides constant damping
        # per outer Δt regardless of substep count N, stabilizing WS-RK3.
        #
        # u -= β_d cᵖ θᵥ ∂(Δπ')/∂x,  v -= β_d cᵖ θᵥ ∂(Δπ')/∂y
        #
        # The cᵖ θᵥ factor matches the PGF scaling so that β_d is a
        # dimensionless O(1) coefficient (β_d ∈ [2, 10] typical).
        Δπ_i   = π′[i, j, k]     - π′⁻[i, j, k]
        Δπ_im1 = π′[i - 1, j, k] - π′⁻[i - 1, j, k]
        Δx = Δxᶠᶜᶜ(i, j, k, grid)
        θᵥᶠᶜᶜ = ℑxTᶠᵃᵃ(i, j, k, grid, θᵥ)
        u[i, j, k] -= β_d * cᵖ * θᵥᶠᶜᶜ * (Δπ_i - Δπ_im1) / Δx * !on_x_boundary(i, j, k, grid)

        Δπ_j   = π′[i, j, k]     - π′⁻[i, j, k]
        Δπ_jm1 = π′[i, j - 1, k] - π′⁻[i, j - 1, k]
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
    grid = model.grid
    arch = architecture(grid)
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity

    # Compute substep count (adaptive when substeps === nothing)
    N = acoustic_substeps(substepper.substeps, grid, Δt, model.thermodynamic_constants)

    # Constant acoustic substep size across all stages
    Δτ = Δt / N

    # Substep count varies per stage: Nτ ≈ β * N
    Nτ = max(round(Int, β_stage * N), 1)

    # Convert slow tendencies to velocity/pressure form
    convert_slow_tendencies!(substepper, model)

    # Initialize time-averaged velocities to zero
    ū = substepper.averaged_velocities
    launch!(arch, grid, :xyz, _zero_avg_velocities!, ū)

    # WS-RK3: reset π' to π'(Uⁿ), not π'(U_eval).
    # The acoustic loop must start from a CONSISTENT Uⁿ state (both velocity
    # AND pressure from Uⁿ). Starting π' from U_eval while velocities are from
    # Uⁿ creates an imbalance that destabilizes the acoustic loop at large Δt.
    # θᵥ, S, π_ref remain from U_eval (frozen thermodynamic quantities).
    pˢᵗ = model.dynamics.standard_pressure
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖ
    launch!(arch, grid, :xyz, _reset_pi_prime_to_U0!,
            substepper.exner_perturbation, substepper.reference_exner_function, U⁰[5], pˢᵗ, Rᵈ, κ)

    parent(substepper.damped_exner_perturbation) .= parent(substepper.exner_perturbation)
    parent(substepper.previous_exner_perturbation) .= parent(substepper.exner_perturbation)

    # Save π'_initial in ρχᵣ for the perturbation recovery.
    # With π' reset to π'(Uⁿ), the recovery computes:
    #   π_new = π(Uⁿ) + Δπ' = π_ref + π'_final  (they cancel)
    # so ρθ_new = EOS(π_ref + π'_final).
    parent(substepper.stage_thermodynamic_density) .= parent(substepper.exner_perturbation)

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

    π′_forcing = CenterField(grid)  # TODO: pre-allocate this

    for _ in 1:Nτ
        # Step 1: Forward — update u, v from PGF and slow tendency
        launch!(arch, grid, :xyz, _acoustic_horizontal_forward!,
                u, v, grid, Δτ, cᵖ,
                substepper.damped_exner_perturbation, substepper.virtual_potential_temperature,
                substepper.slow_tendencies.velocity.u,
                substepper.slow_tendencies.velocity.v)

        # Step 2: Explicit π' forcing (Gˢπ + horizontal divergence + β·∂w⁻/∂z)
        launch!(arch, grid, :xyz, _compute_π′_forcing!,
                π′_forcing, grid, Δτ, β,
                u, v, w, substepper.acoustic_compression, substepper.slow_tendencies.exner_pressure)

        # Save π' before implicit solve (for damping)
        parent(substepper.previous_exner_perturbation) .= parent(substepper.exner_perturbation)

        # Step 3: Implicit solve — tridiagonal for π'⁺, back-solve for w⁺
        implicit_w_solve!(w, substepper, model, Δτ, π′_forcing)

        # Step 3b: Klemp (2018) divergence damping (if β_d > 0)
        # Damp u, v proportional to ∂(π'⁺ - π'⁻)/∂x.
        # Total damping per outer Δt is constant regardless of N.
        if β_d > 0
            launch!(arch, grid, :xyz, _acoustic_divergence_damping!,
                    u, v, substepper.exner_perturbation, substepper.previous_exner_perturbation,
                    substepper.virtual_potential_temperature, grid, β_d, cᵖ)
        end

        # Step 4: Apply kdiv forward-extrapolation + accumulate velocity averages
        launch!(arch, grid, :xyz, _update_pressure_and_average!,
                substepper.exner_perturbation, substepper.damped_exner_perturbation, substepper.previous_exner_perturbation,
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

@kernel function _compute_slow_theta_tendency!(Gˢθ, Gˢρχ, Gˢρ, θᵥ, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # Material derivative of θ: Gˢθ = (Gˢρθ - θ Gˢρ) / ρ
        Gˢθ[i, j, k] = (Gˢρχ[i, j, k] - θᵥ[i, j, k] * Gˢρ[i, j, k]) / ρ[i, j, k]
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
            substepper.exner_perturbation, substepper.stage_thermodynamic_density,
            substepper.reference_exner_function,
            substepper.virtual_potential_temperature, substepper.slow_tendencies.thermodynamic_density, substepper.slow_tendencies.density,
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
        cᵥ_over_R = (1 - κ) / κ
        πᵣ = π_ref[i, j, k]
        Δπ = π′[i, j, k] - π′_initial[i, j, k]
        ρχ_perturbation = cᵥ_over_R * ρχᵣ[i, j, k] / πᵣ * Δπ

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
        cᵥ_over_R = (1 - κ) / κ
        R_over_cᵥ = κ / (1 - κ)

        # WS-RK3 perturbation applied in π'-space (the natural acoustic variable).
        # π⁺ = π⁰ + (π'_final - π'_initial), then convert to ρθ via EOS.
        Δπ′ = π′_final[i, j, k] - π′_initial[i, j, k]

        # Compute π⁰ from ρθ⁰ via the equation of state
        ρχ⁰_ijk = ρχ⁰[i, j, k]
        π⁰ = (Rᵈ * ρχ⁰_ijk / pˢᵗ)^R_over_cᵥ

        # Apply perturbation in π'-space, then convert to ρθ
        π⁺ = π⁰ + Δπ′
        ρχ⁺ = (pˢᵗ / Rᵈ) * π⁺^cᵥ_over_R
        ρχ[i, j, k] = ρχ⁺

        # Density: ρ = ρθ / θ⁺, where θ⁺ = θⁿ + Δt_stage Gˢθ.
        # WS-RK3 requires the θ BASE to be from the initial state Uⁿ (not the
        # evaluation state U*). Using θᵥ from U* would double-count the θ
        # change from earlier stages (θ(U*) = θⁿ + β₁ Δt Gˢθ already).
        # The slow Gˢθ is evaluated at U* (correct for WS-RK3).
        ρ_eval = ρ[i, j, k]
        θᵥ_eval = θᵥ[i, j, k]
        θⁿ = ρχ⁰_ijk / ρ⁰[i, j, k]
        θ⁺ = θⁿ + Δt_stage * (Gˢρχ[i, j, k] - θᵥ_eval * Gˢρ[i, j, k]) / ρ_eval
        ρ[i, j, k] = ρχ⁺ / θ⁺
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
    grid = model.grid
    arch = architecture(grid)
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity

    # For SSP-RK3, all stages use Ns substeps (adaptive when substeps === nothing)
    Ns = acoustic_substeps(substepper.substeps, grid, Δt, model.thermodynamic_constants)
    Δτ = Δt / Ns
    Nτ = Ns

    # Convert slow tendencies
    convert_slow_tendencies!(substepper, model)

    # Initialize time-averaged velocities to zero
    ū = substepper.averaged_velocities
    launch!(arch, grid, :xyz, _zero_avg_velocities!, ū)

    parent(substepper.damped_exner_perturbation) .= parent(substepper.exner_perturbation)
    parent(substepper.previous_exner_perturbation) .= parent(substepper.exner_perturbation)

    u = model.velocities.u
    v = model.velocities.v
    w = model.velocities.w

    # Off-centering parameter for implicit solver (NOT the SSP coefficient)
    α_implicit = substepper.forward_weight
    β_implicit = 1 - α_implicit
    kdiv = substepper.divergence_damping_coefficient
    β_d = substepper.acoustic_damping_coefficient
    π′_forcing = CenterField(grid)  # TODO: pre-allocate

    for _ in 1:Nτ
        launch!(arch, grid, :xyz, _acoustic_horizontal_forward!,
                u, v, grid, Δτ, cᵖ,
                substepper.damped_exner_perturbation, substepper.virtual_potential_temperature,
                substepper.slow_tendencies.velocity.u,
                substepper.slow_tendencies.velocity.v)

        launch!(arch, grid, :xyz, _compute_π′_forcing!,
                π′_forcing, grid, Δτ, β_implicit,
                u, v, w, substepper.acoustic_compression, substepper.slow_tendencies.exner_pressure)

        parent(substepper.previous_exner_perturbation) .= parent(substepper.exner_perturbation)
        implicit_w_solve!(w, substepper, model, Δτ, π′_forcing)

        if β_d > 0
            launch!(arch, grid, :xyz, _acoustic_divergence_damping!,
                    u, v, substepper.exner_perturbation, substepper.previous_exner_perturbation,
                    substepper.virtual_potential_temperature, grid, β_d, cᵖ)
        end

        launch!(arch, grid, :xyz, _update_pressure_and_average!,
                substepper.exner_perturbation, substepper.damped_exner_perturbation, substepper.previous_exner_perturbation,
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
            substepper.exner_perturbation, substepper.reference_exner_function, substepper.virtual_potential_temperature,
            substepper.slow_tendencies.thermodynamic_density, substepper.slow_tendencies.density,
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
        # Nonlinear EOS: ρθ = (pˢᵗ/Rᵈ) π^(cᵥ/R), where π = πᵣ + π'
        cᵥ_over_R = (1 - κ) / κ
        π_total = π_ref[i, j, k] + π′[i, j, k]
        ρχ⁺ = (pˢᵗ / Rᵈ) * π_total^cᵥ_over_R
        ρχ[i, j, k] = ρχ⁺

        # Update θ with slow advection: θ⁺ = θᵥ + Δt Gˢθ
        # where Gˢθ = (Gˢρθ - θ Gˢρ) / ρ is the material derivative of θ.
        # Without this, θ = ρθ/ρ = θᵥ = frozen, preventing θ evolution.
        ρ_eval = ρ[i, j, k]
        θᵥ_ijk = θᵥ[i, j, k]
        θ⁺ = θᵥ_ijk + Δt * (Gˢρχ[i, j, k] - θᵥ_ijk * Gˢρ[i, j, k]) / ρ_eval
        ρ[i, j, k] = ρχ⁺ / θ⁺
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
