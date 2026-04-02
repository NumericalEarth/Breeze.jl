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
    ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ,
    δxTᶠᵃᵃ, δyTᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ,
    divᶜᶜᶜ,
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
4. **Filtering**: ``π̃' = π' + ϰᵈⁱ (π' - π'_{old})``

Fields
======

- `substeps`: Number of acoustic substeps for the full time step
- `forward_weight`: Off-centering parameter ω (0.6 = CM1 default)
- `divergence_damping_coefficient`: Forward-extrapolation filter ϰᵈⁱ for π' (default 0.10)
- `acoustic_damping_coefficient`: Klemp 2018 ϰᵃᶜ for velocity damping
- `virtual_potential_temperature`: Stage-frozen θᵥ (CenterField)
- `acoustic_compression`: Coefficient `(γ-1)π₀` converting `∇·u` to `∂π'/∂t` (CenterField)
- `reference_exner_function`: Reference π₀ = (p_ref/pˢᵗ)^(R/cᵖ) (CenterField)
- `exner_perturbation`: Current Exner pressure perturbation π' = π - π₀ (CenterField)
- `previous_exner_perturbation`: Previous-substep π' for divergence damping (CenterField)
- `filtered_exner_perturbation`: Filtered π̃' used in PGF (CenterField)
- `stage_thermodynamic_density`: Stage-frozen ρθ (CenterField)
- `averaged_velocities`: Time-averaged velocities for scalar advection
- `slow_tendencies`: Frozen slow tendencies (velocity, exner_pressure). Momentum tendencies
  are stored in the outer timestepper's `Gⁿ` fields; density and thermodynamic density
  tendencies are also read directly from `Gⁿ`.
- `rtheta_pp`: Acoustic ρθ perturbation at cell centers (MPAS-style tracking)
- `vertical_solver`: BatchedTridiagonalSolver for implicit w-π' coupling
- `rhs`: Right-hand side storage for tridiagonal solve
"""
struct AcousticSubstepper{N, FT, CF, AV, ST, TS}
    substeps :: N                              # Number of acoustic substeps per full Δt
    forward_weight :: FT                       # Off-centering ω (CM1 default 0.6)
    divergence_damping_coefficient :: FT       # Forward-extrapolation filter ϰᵈⁱ
    acoustic_damping_coefficient :: FT         # Klemp 2018 ϰᵃᶜ
    virtual_potential_temperature :: CF        # Stage-frozen θᵥ
    acoustic_compression :: CF                 # (γ-1)π₀ — converts ∇·u to ∂π'/∂t
    reference_exner_function :: CF             # π₀ from reference state
    exner_perturbation :: CF                   # Current π' = π - π₀
    previous_exner_perturbation :: CF          # Previous-substep π' (for damping)
    filtered_exner_perturbation :: CF            # Filtered π̃' used in PGF
    stage_thermodynamic_density :: CF          # Stage-frozen ρθ
    rtheta_pp :: CF                            # Acoustic ρθ perturbation (MPAS ts/rtheta_pp)
    averaged_velocities :: AV                  # Time-averaged velocities for scalar advection
    slow_tendencies :: ST                      # Frozen slow tendencies (NamedTuple)
    vertical_solver :: TS                      # BatchedTridiagonalSolver for implicit w-π' coupling
    rhs :: CF                                  # Right-hand side storage for tridiagonal solve
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
                       adapt(to, a.virtual_potential_temperature),
                       adapt(to, a.acoustic_compression),
                       adapt(to, a.reference_exner_function),
                       adapt(to, a.exner_perturbation),
                       adapt(to, a.previous_exner_perturbation),
                       adapt(to, a.filtered_exner_perturbation),
                       adapt(to, a.stage_thermodynamic_density),
                       adapt(to, a.rtheta_pp),
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
    ω = convert(FT, split_explicit.forward_weight)
    ϰᵈⁱ = convert(FT, split_explicit.divergence_damping_coefficient)
    ϰᵃᶜ = convert(FT, split_explicit.acoustic_damping_coefficient)

    virtual_potential_temperature = CenterField(grid)
    acoustic_compression = CenterField(grid)
    reference_exner_function = CenterField(grid)
    exner_perturbation = CenterField(grid)
    previous_exner_perturbation = CenterField(grid)
    filtered_exner_perturbation = CenterField(grid)
    stage_thermodynamic_density = CenterField(grid)
    rtheta_pp = CenterField(grid)

    averaged_velocities = (u = XFaceField(grid),
                           v = YFaceField(grid),
                           w = ZFaceField(grid))

    slow_tendencies = (velocity = (u = XFaceField(grid),
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

    return AcousticSubstepper(Ns, ω, ϰᵈⁱ, ϰᵃᶜ,
                              virtual_potential_temperature,
                              acoustic_compression,
                              reference_exner_function,
                              exner_perturbation,
                              previous_exner_perturbation,
                              filtered_exner_perturbation,
                              stage_thermodynamic_density,
                              rtheta_pp,
                              averaged_velocities,
                              slow_tendencies,
                              vertical_solver,
                              rhs)
end

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
    cᵖ = thermodynamic_constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(thermodynamic_constants)
    cᵥ = cᵖ - Rᵈ
    γ = cᵖ / cᵥ
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
            substepper.filtered_exner_perturbation,
            substepper.reference_exner_function,
            model.dynamics.density,
            model.dynamics.pressure,
            model.temperature,
            specific_prognostic_moisture(model),
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

@kernel function _recompute_pi_prime!(π′, π̃′, p, πᵣ, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        πⁱ = (p[i, j, k] / pˢᵗ)^κ
        π′[i, j, k] = πⁱ - πᵣ[i, j, k]
        π̃′[i, j, k] = π′[i, j, k]
    end
end

@kernel function _prepare_exner_cache!(θᵥ_field, acoustic_compression_field, π′_field, π̃′_field,
                                       πᵣ_field,
                                       ρ, p, T, specific_prognostic_moisture, grid,
                                       microphysics, microphysical_fields,
                                       constants, reference_state, pˢᵗ, cᵖ, κ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρⁱ = ρ[i, j, k]
        pⁱ = p[i, j, k]
        Tⁱ = T[i, j, k]
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
    end

    # Compute moisture fractions and mixture properties
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρⁱ, qᵛᵉ, microphysical_fields)
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

    # Exner pressure perturbation: π' = π - πᵣ
    πᵣⁱ = reference_exner(i, j, k, reference_state, pˢᵗ, κ)

    @inbounds begin
        θᵥ_field[i, j, k] = θᵥⁱ
        acoustic_compression_field[i, j, k] = Sⁱ
        π′_field[i, j, k] = πⁱ - πᵣⁱ
        π̃′_field[i, j, k] = πⁱ - πᵣⁱ
        πᵣ_field[i, j, k] = πᵣⁱ
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
            substepper.exner_perturbation, substepper.filtered_exner_perturbation,
            model.dynamics.pressure, substepper.reference_exner_function, pˢᵗ, κ)
    return nothing
end

function _set_exner_reference!(substepper, model, ::Nothing, pˢᵗ, κ)
    grid = model.grid
    arch = architecture(grid)
    fill!(parent(substepper.reference_exner_function), 0)
    launch!(arch, grid, :xyz, _recompute_pi_prime!,
            substepper.exner_perturbation, substepper.filtered_exner_perturbation,
            model.dynamics.pressure, substepper.reference_exner_function, pˢᵗ, κ)
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
    Gⁿ = model.timestepper.Gⁿ

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
            substepper.exner_perturbation,
            substepper.reference_exner_function,
            substepper.virtual_potential_temperature,
            grid, κ, cᵖᵈ, g)

    return nothing
end

@kernel function _convert_slow_tendencies!(Gˢu, Gˢv, Gˢw, Gˢπ,
                                           Gˢρu, Gˢρv, Gˢρw,
                                           ρ, u, v, w,
                                           π′, πᵣ, θᵥ,
                                           grid, κ, cᵖᵈ, g)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # MPAS-style: slow velocity tendencies include the FULL horizontal
        # PGF at the current RK stage state. The momentum tendencies Gˢρu, Gˢρv
        # exclude PGF (zeroed by SlowTendencyMode), so we add it here using
        # the full Exner π = π₀ + π'. Since π₀ is 1D, ∂π₀/∂x = ∂π₀/∂y = 0,
        # so the horizontal PGF comes entirely from π'.
        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        θᵥᶠᶜᶜ = ℑxTᶠᵃᵃ(i, j, k, grid, θᵥ)
        ∂x_π = δxTᶠᵃᵃ(i, j, k, grid, π′) / Δxᶠᶜᶜ(i, j, k, grid)
        Gˢu[i, j, k] = (Gˢρu[i, j, k] / ρᶠᶜᶜ - cᵖᵈ * θᵥᶠᶜᶜ * ∂x_π) * !on_x_boundary(i, j, k, grid)

        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        θᵥᶜᶠᶜ = ℑyTᵃᶠᵃ(i, j, k, grid, θᵥ)
        ∂y_π = δyTᵃᶠᵃ(i, j, k, grid, π′) / Δyᶜᶠᶜ(i, j, k, grid)
        Gˢv[i, j, k] = (Gˢρv[i, j, k] / ρᶜᶠᶜ - cᵖᵈ * θᵥᶜᶠᶜ * ∂y_π) * !on_y_boundary(i, j, k, grid)

        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)

        # MPAS-style: slow w tendency includes the FULL Exner PGF + gravity,
        # evaluated at the current RK stage state. The acoustic loop only
        # provides the small acoustic perturbation PGF from π' changes
        # that develop DURING the substep loop (π' starts at zero).
        #
        # Full vertical acceleration = -cₚ θᵥ ∂(π₀+π')/∂z - g
        θᵥᶠ = ℑzTᵃᵃᶠ(i, j, k, grid, θᵥ)
        δz_πᵣ = δzTᵃᵃᶠ(i, j, k, grid, πᵣ)
        δz_π′ = δzTᵃᵃᶠ(i, j, k, grid, π′)
        Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
        full_pgf_grav = -cᵖᵈ * θᵥᶠ * (δz_πᵣ + δz_π′) / Δzᶠ - g

        Gˢw[i, j, k] = (Gˢρw[i, j, k] / ρᶜᶜᶠ + full_pgf_grav) * (k > 1)

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

@kernel function _compute_ts!(ts, grid, Δτ, u, v, rtheta_pp, Gˢρθ, θ_m)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        V = Vᶜᶜᶜ(i, j, k, grid)

        # Horizontal ρθ flux divergence: -θ_m div_h(u)
        θ_mᵢ = θ_m[i, j, k]
        div_h = (Axᶠᶜᶜ(i + 1, j, k, grid) * u[i + 1, j, k] - Axᶠᶜᶜ(i, j, k, grid) * u[i, j, k] +
                 Ayᶜᶠᶜ(i, j + 1, k, grid) * v[i, j + 1, k] - Ayᶜᶠᶜ(i, j, k, grid) * v[i, j, k]) / V

        # ts = previous rtheta_pp + slow tendency + horizontal flux
        ts[i, j, k] = rtheta_pp[i, j, k] + Δτ * Gˢρθ[i, j, k] - θ_mᵢ * Δτ * div_h
    end
end

@kernel function _update_rtheta_pp!(rtheta_pp, ts, grid, Δτ, w, θ_m)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Vertical θ-flux divergence: θ_m (w(k+1) - w(k)) / Δz
        w_top = ifelse(k < Nz, w[i, j, k + 1], zero(eltype(w)))
        w_bot = ifelse(k > 1,  w[i, j, k],     zero(eltype(w)))
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)
        θ_mᵢ = θ_m[i, j, k]

        # rtheta_pp = ts - θ_m Δτ ∂w/∂z
        rtheta_pp[i, j, k] = ts[i, j, k] - θ_mᵢ * Δτ * (w_top - w_bot) / Δzᶜ
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
                                               π̃′, θᵥ, Gˢu, Gˢv)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # u += Δτ (Gˢu - cᵖ θᵥ ∂π'/∂x)
        θᵥᶠᶜᶜ = ℑxTᶠᵃᵃ(i, j, k, grid, θᵥ)
        ∂x_π = δxTᶠᵃᵃ(i, j, k, grid, π̃′) / Δxᶠᶜᶜ(i, j, k, grid)
        u[i, j, k] += Δτ * (Gˢu[i, j, k] - cᵖ * θᵥᶠᶜᶜ * ∂x_π) * !on_x_boundary(i, j, k, grid)

        # v += Δτ (Gˢv - cᵖ θᵥ ∂π'/∂y)
        θᵥᶜᶠᶜ = ℑyTᵃᶠᵃ(i, j, k, grid, θᵥ)
        ∂y_π = δyTᵃᶠᵃ(i, j, k, grid, π̃′) / Δyᶜᶠᶜ(i, j, k, grid)
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

@kernel function _compute_π′_forcing!(π′_forcing, grid, Δτ, ω̄,
                                      u, v, w, S, Gˢπ)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Area-weighted horizontal velocity divergence: V⁻¹ (Ax u|ᵢ₊₁ - Ax u|ᵢ + Ay v|ⱼ₊₁ - Ay v|ⱼ)
        # Required for LatitudeLongitudeGrid where cell areas vary with latitude.
        V = Vᶜᶜᶜ(i, j, k, grid)
        ∇ₕ_u = (Axᶠᶜᶜ(i + 1, j, k, grid) * u[i + 1, j, k] - Axᶠᶜᶜ(i, j, k, grid) * u[i, j, k] +
                 Ayᶜᶠᶜ(i, j + 1, k, grid) * v[i, j + 1, k] - Ayᶜᶠᶜ(i, j, k, grid) * v[i, j, k]) / V

        # (1-ω)-weighted vertical divergence from old w (before implicit solve)
        w⁻_bot = ifelse(k == 1, zero(eltype(w)), w[i, j, k])
        w⁻_top = ifelse(k == Nz, zero(eltype(w)), w[i, j, k + 1])
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)

        # π′_forcing = Δτ (Gˢπ - S ∇ₕ·u⁺) - (1-ω) Δτ S ∂w⁻/∂z
        π′_forcing[i, j, k] = Δτ * (Gˢπ[i, j, k] - S[i, j, k] * ∇ₕ_u) -
                               ω̄ * Δτ * S[i, j, k] * (w⁻_top - w⁻_bot) / Δzᶜ
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
    ω = substepper.forward_weight
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cᵖᵈ
    g = model.thermodynamic_constants.gravitational_acceleration
    solver = substepper.vertical_solver

    # Build w-primary tridiagonal (cofwz + cofwt in matrix).
    # Face k=2..Nz mapped to center index k-1=1..Nz-1.
    # Index Nz stores the π' forcing for the pressure update.
    launch!(arch, grid, :xyz, _build_w_tridiagonal!,
            solver.a, solver.b, solver.c, substepper.rhs,
            grid, ω, Δτ, cᵖᵈ, g, κ,
            w, substepper.exner_perturbation,
            substepper.virtual_potential_temperature,
            substepper.acoustic_compression,
            substepper.stage_thermodynamic_density,
            model.dynamics.density,
            substepper.slow_tendencies.velocity.w)

    # Solve the w tridiagonal. Result goes into exner_perturbation (scratch).
    solve!(substepper.exner_perturbation, solver, substepper.rhs)

    # Copy solved w back to face field and update π'
    # w_solved is in exner_perturbation (scratch), π'_old in previous_exner_perturbation
    launch!(arch, grid, :xyz, _recover_w_and_update_pi!,
            w, substepper.rhs,  # rhs is now free, use it for new π'
            substepper.exner_perturbation, substepper.previous_exner_perturbation, π′_forcing,
            grid, ω, Δτ, substepper.acoustic_compression)

    # Copy new π' from rhs back to exner_perturbation
    parent(substepper.exner_perturbation) .= parent(substepper.rhs)

    return nothing
end

##### MPAS-style w-primary tridiagonal with cofwt in the matrix.
#####
##### The w equation at face k:
#####   w⁺(k) = wᵉ(k) - ω Δτ Mᵖ δz(π'⁺) + ω Δτ cofwt_coef * ts_new
#####
##### where π'⁺ depends on ∂w⁺/∂z (pressure equation) and ts_new depends
##### on ∂(θw⁺)/∂z (ρθ flux). Substituting creates a tridiagonal for w.
#####
##### Face k=2..Nz → center index k-1=1..Nz-1 for the BatchedTridiagonalSolver.
##### Index Nz is unused (padded with identity row).

@kernel function _build_w_tridiagonal!(lower, diag, upper, rhs_field,
                                        grid, ω, Δτ, cᵖᵈ, g, κ,
                                        w, π′,
                                        θᵥ, S, ts, ρ,
                                        Gˢw)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)
    kf = k + 1  # face index (k=1 in solver → face k=2)

    @inbounds begin
        rcv = κ / (1 - κ)
        ω̄ = 1 - ω

        if k >= Nz
            # Padding row: identity (unused face Nz+1 or beyond)
            lower[i, j, k] = 0
            diag[i, j, k] = 1
            upper[i, j, k] = 0
            rhs_field[i, j, k] = 0
        else
            # Face kf = k+1 (interior face, k goes 1..Nz-1)
            Δzᶠ = Δzᶜᶜᶠ(i, j, kf, grid)
            θᵥᶠ = (θᵥ[i, j, kf] + θᵥ[i, j, kf - 1]) / 2
            Mᵖ = cᵖᵈ * θᵥᶠ / Δzᶠ

            # Acoustic coupling: cofwz through Schur complement
            # w depends on π' gradient, π' depends on ∂w/∂z
            Δzᶜ_above = Δzᶜᶜᶜ(i, j, kf, grid)
            Δzᶜ_below = Δzᶜᶜᶜ(i, j, kf - 1, grid)
            S_above = S[i, j, kf]
            S_below = S[i, j, kf - 1]

            Qz_above = ω * ω * Δτ * Δτ * S_above * Mᵖ / Δzᶜ_above
            Qz_below = ω * ω * Δτ * Δτ * S_below * Mᵖ / Δzᶜ_below
            Qz_above = ifelse(kf == Nz, zero(Qz_above), Qz_above)

            # cofwt coupling: w depends on ts_new, ts_new depends on ∂(θw)/∂z
            θ_above = θᵥ[i, j, kf]
            θ_below = θᵥ[i, j, kf - 1]
            ρ_above_safe = ifelse(ρ[i, j, kf] == 0, one(eltype(ρ)), ρ[i, j, kf])
            ρ_below_safe = ifelse(ρ[i, j, kf - 1] == 0, one(eltype(ρ)), ρ[i, j, kf - 1])
            θ_above_safe = ifelse(θ_above == 0, one(θ_above), θ_above)
            θ_below_safe = ifelse(θ_below == 0, one(θ_below), θ_below)

            cofwt_above = rcv / 2 * g / (ρ_above_safe * θ_above_safe)
            cofwt_below = rcv / 2 * g / (ρ_below_safe * θ_below_safe)

            Qw_above = ω * Δτ * cofwt_above * θ_above * ω * Δτ / Δzᶜ_above
            Qw_below = ω * Δτ * cofwt_below * θ_below * ω * Δτ / Δzᶜ_below
            Qw_above = ifelse(kf == Nz, zero(Qw_above), Qw_above)

            # Tridiagonal coefficients (face k → solver index k-1)
            lower[i, j, k] = ifelse(kf == 2, zero(Qz_below), -(Qz_below + Qw_below))
            upper[i, j, k] = -(Qz_above + Qw_above)
            diag[i, j, k] = 1 + Qz_above + Qz_below + Qw_above + Qw_below

            # RHS: explicit w with acoustic PGF + cofwt buoyancy from ts.
            # The slow tendency Gˢw already includes the full PGF+gravity.
            # The acoustic π' provides the implicit-time correction only.
            δz_π = π′[i, j, kf] - π′[i, j, kf - 1]
            ts_face = (ts[i, j, kf] + ts[i, j, kf - 1]) / 2
            ρ_face = (ρ[i, j, kf] + ρ[i, j, kf - 1]) / 2
            ρ_face_safe = ifelse(ρ_face == 0, one(ρ_face), ρ_face)
            θᵥᶠ_safe = ifelse(θᵥᶠ == 0, one(θᵥᶠ), θᵥᶠ)

            wᵉ = w[i, j, kf] + Δτ * Gˢw[i, j, kf] - ω̄ * Δτ * Mᵖ * δz_π +
                  rcv / 2 * g * ts_face / (ρ_face_safe * θᵥᶠ_safe)

            rhs_field[i, j, k] = wᵉ
        end
    end
end

@kernel function _recover_w_and_update_pi!(w, π′, w_solved, π′_old, π′_forcing,
                                            grid, ω, Δτ, S)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Copy solved w from center-indexed array to face field
        # Solver index k=1..Nz-1 → face k+1=2..Nz
        kf = k + 1
        if k < Nz
            w[i, j, kf] = w_solved[i, j, k]
        end
        # Boundaries
        if k == 1
            w[i, j, 1] = 0
        end

        # Update π' from pressure equation:
        # π'⁺(k) = π'_old(k) + π'_forcing(k) - ω Δτ S(k) (w⁺(k+1) - w⁺(k)) / Δz(k)
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)
        w_top = ifelse(k < Nz, w_solved[i, j, k], zero(eltype(w)))  # w at face k+1
        w_bot = ifelse(k > 1,  w_solved[i, j, k - 1], zero(eltype(w)))  # w at face k
        π′[i, j, k] = π′_old[i, j, k] + π′_forcing[i, j, k] - ω * Δτ * S[i, j, k] * (w_top - w_bot) / Δzᶜ
    end
end

##### Legacy column solve (kept for reference)
##### MPAS-style column solve: tridiagonal for w at faces, then update π'.
##### The tridiagonal includes cofwz (acoustic PGF), cofwr (gravity on ρ'),
##### and cofwt (buoyancy from θ'), all from the implicit coupling between
##### w, ρθ_pp, and ρ_pp through vertical flux divergence.
#####
##### After solving for w, π' is updated from the pressure equation:
#####   π'⁺ = π' + π'_forcing - ω Δτ S ∂w⁺/∂z

@kernel function _mpas_implicit_w_column_solve!(w, π′,
                                                 grid, ω, Δτ, cᵖᵈ, g, κ,
                                                 π′⁻, π′_forcing,
                                                 θᵥ, S, ts, ρ, Gˢw)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    # R/cv for cofwt
    rcv = κ / (1 - κ)

    ## ── Step 1: Build explicit w at each face ──
    ## wᵉ(k) = w_old(k) + Δτ Gˢw(k) - (1-ω) Δτ Mᵖ δz(π') + cofwt(ts)
    ## Also compute tridiagonal coefficients for the implicit part.

    @inbounds begin
        ω̄ = 1 - ω

        ## Forward elimination arrays (stored on stack for the column)
        ## For faces k=2..Nz, the tridiagonal is:
        ##   a(k) w(k-1) + b(k) w(k) + c(k) w(k+1) = rhs(k)
        ## with w(1) = 0 and w(Nz+1) = 0.
        ##
        ## The implicit coupling comes from:
        ## - cofwz: w depends on π' gradient → π' depends on ∂w/∂z
        ## - cofwt: w depends on ts → ts depends on ∂(θ w)/∂z

        for k in 2:Nz
            Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
            θᵥᶠ = (θᵥ[i, j, k] + θᵥ[i, j, k - 1]) / 2
            Mᵖ = cᵖᵈ * θᵥᶠ / Δzᶠ

            # Explicit w: PGF from old π' + slow tendency + explicit cofwt
            δz_π = π′[i, j, k] - π′[i, j, k - 1]
            ts_face = (ts[i, j, k] + ts[i, j, k - 1]) / 2
            ρ_face = (ρ[i, j, k] + ρ[i, j, k - 1]) / 2
            θᵥᶠ_safe = ifelse(θᵥᶠ == 0, one(θᵥᶠ), θᵥᶠ)
            ρ_safe = ifelse(ρ_face == 0, one(ρ_face), ρ_face)

            wᵉ = w[i, j, k] + Δτ * Gˢw[i, j, k] - ω̄ * Δτ * Mᵖ * δz_π +
                  rcv / 2 * g * ts_face / (ρ_safe * θᵥᶠ_safe)

            # Implicit coupling: cofwz gives w ∝ -ω Δτ Mᵖ δz(π'_new)
            # and π'_new = π' + forcing - ω Δτ S ∂w_new/∂z
            # Substituting: the tridiagonal for w has coefficients from
            # ω² Δτ² Mᵖ S / Δz at each face-center-face stencil.
            #
            # cofwt adds: w ∝ cofwt_coef * ts_new, where
            # ts_new(k) = ts(k) - θ_m(k) ω Δτ (w_new(k+1) - w_new(k)) / Δz(k)
            # This creates additional coupling to w(k-1), w(k), w(k+1)
            # through the θ_m weighted vertical divergence.

            Δzᶜ_above = Δzᶜᶜᶜ(i, j, k, grid)
            Δzᶜ_below = Δzᶜᶜᶜ(i, j, k - 1, grid)
            S_above = S[i, j, k]
            S_below = S[i, j, k - 1]
            θ_above = θᵥ[i, j, k]
            θ_below = θᵥ[i, j, k - 1]

            # cofwz coupling: ω² Δτ² Mᵖ S / Δz (standard acoustic)
            # At face k, π' at centers k and k-1 couple through Mᵖ.
            # The Schur complement gives diagonal and off-diagonal terms.
            Qz_above = ω * ω * Δτ * Δτ * S_above * Mᵖ / Δzᶜ_above
            Qz_below = ω * ω * Δτ * Δτ * S_below * Mᵖ / Δzᶜ_below

            # cofwt coupling through ts → vertical θ-divergence of w.
            # At center k: ts_new(k) = ts(k) - θ(k) ω Δτ (w(k+1) - w(k)) / Δz(k)
            # cofwt at face k from center k: cofwt_coef(k) * ts_new(k)
            #   = cofwt_coef(k) * [ts(k) - θ(k) ω Δτ (w(k+1) - w(k)) / Δz(k)]
            # The w-dependent part: -cofwt_coef(k) θ(k) ω Δτ / Δz(k) * (w(k+1) - w(k))
            cofwt_k   = rcv / 2 * g / (ρ[i, j, k] * θ_above)
            cofwt_km1 = rcv / 2 * g / (ρ[i, j, k - 1] * θ_below)

            Qw_above = ω * Δτ * cofwt_k * θ_above * ω * Δτ / Δzᶜ_above
            Qw_below = ω * Δτ * cofwt_km1 * θ_below * ω * Δτ / Δzᶜ_below

            # Tridiagonal coefficients at face k:
            # a(k) = coupling to w(k-1)
            # b(k) = diagonal
            # c(k) = coupling to w(k+1)
            a_k = ifelse(k == 2, zero(Qz_below),
                         -Qz_below - Qw_below)
            c_k = ifelse(k == Nz, zero(Qz_above),
                         -Qz_above - Qw_above)
            b_k = 1 + Qz_above + Qz_below + Qw_above + Qw_below

            # Thomas algorithm: forward elimination
            if k == 2
                # First row: no a_k
                alpha = 1 / b_k
                w[i, j, k] = wᵉ * alpha
                # Store gamma for back-substitution
                π′[i, j, k - 1] = c_k * alpha  # reuse π'[k-1] as gamma storage
            else
                gamma_prev = π′[i, j, k - 2]  # gamma from previous row
                denom = b_k - a_k * gamma_prev
                alpha = 1 / denom
                w[i, j, k] = (wᵉ - a_k * w[i, j, k - 1]) * alpha
                π′[i, j, k - 1] = c_k * alpha
            end
        end

        ## ── Step 2: Back-substitution ──
        for k in (Nz - 1):-1:2
            gamma_k = π′[i, j, k - 1]
            w[i, j, k] = w[i, j, k] - gamma_k * w[i, j, k + 1]
        end

        # Boundary conditions
        w[i, j, 1] = 0
        # w[i, j, Nz+1] is already 0 (face field)

        ## ── Step 3: Update π' from pressure equation ──
        ## π'⁺(k) = π'_old(k) + π'_forcing(k) - ω Δτ S(k) (w⁺(k+1) - w⁺(k)) / Δz(k)
        for k in 1:Nz
            Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)
            w_top = ifelse(k < Nz, w[i, j, k + 1], zero(eltype(w)))
            w_bot = ifelse(k > 1,  w[i, j, k],     zero(eltype(w)))
            π′[i, j, k] = π′⁻[i, j, k] + π′_forcing[i, j, k] - ω * Δτ * S[i, j, k] * (w_top - w_bot) / Δzᶜ
        end
    end
end

##### Legacy π'-primary tridiagonal (kept for reference/fallback)
@kernel function _build_π′_tridiagonal!(lower, diag, upper, rhs_field,
                                        grid, ω, Δτ, cᵖᵈ, g, κ,
                                        w, π′, π′_forcing,
                                        θᵥ, S, ts, ρ,
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

        # Tridiagonal coupling coefficients: Q = ω² Δτ² S Mᵖ / Δz
        Sⁱ = S[i, j, k]
        Q_bot = ω * ω * Δτ * Δτ * Sⁱ * Mᵖ_bot / Δzᶜ
        Q_top = ω * ω * Δτ * Δτ * Sⁱ * Mᵖ_top / Δzᶜ

        Q_bot = ifelse(k == 1, zero(Q_bot), Q_bot)
        Q_top = ifelse(k == Nz, zero(Q_top), Q_top)

        lower[i, j, k] = -Q_bot
        upper[i, j, k] = -Q_top
        diag[i, j, k] = 1 + Q_bot + Q_top

        # cofwt: MPAS-style buoyancy feedback from accumulated ρθ perturbation (ts).
        #
        # ts carries the total ρθ perturbation from horizontal fluxes, slow
        # tendency, and previous acoustic corrections. The buoyancy at face k
        # is g ts_face / ρ_face (the θ perturbation times g).
        # MPAS coefficient: cofwt ≈ 0.5 ω Δτ (R/cv) g ρ_base / ρθ_total
        # Simplified: cofwt * ts / ρ_face ≈ ω Δτ g ts_face / (ρ_face θ_face)
        ρᶠ_bot = ifelse(k == 1, one(eltype(ρ)), (ρ[i, j, k] + ρ[i, j, k - 1]) / 2)
        ρᶠ_top = ifelse(k == Nz, one(eltype(ρ)), (ρ[i, j, k + 1] + ρ[i, j, k]) / 2)
        ts_bot = ifelse(k == 1, zero(eltype(ts)), (ts[i, j, k] + ts[i, j, k - 1]) / 2)
        ts_top = ifelse(k == Nz, zero(eltype(ts)), (ts[i, j, k + 1] + ts[i, j, k]) / 2)

        # MPAS cofwt: 0.5 ω Δτ (R/cv) g / θ, applied to ts
        # In velocity: Δw = 0.5 ω Δτ (R/cv) g ts_face / (ρ_face θ_face)
        rcv = κ / (1 - κ)
        cofwt_bot = rcv / 2 * ω * Δτ * g * ts_bot / (ρᶠ_bot * θᵥᶠ_bot)
        cofwt_top = rcv / 2 * ω * Δτ * g * ts_top / (ρᶠ_top * θᵥᶠ_top)

        # Explicit w at faces: wᵉ = w + Δτ Gˢw - (1-ω) Δτ Mᵖ δz(π') + cofwt
        δz_π_bot = ifelse(k == 1, zero(eltype(π′)), π′[i, j, k] - π′[i, j, k - 1])
        δz_π_top = ifelse(k == Nz, zero(eltype(π′)), π′[i, j, k + 1] - π′[i, j, k])

        ω̄ = 1 - ω
        wᵉ_bot = ifelse(k == 1, zero(eltype(w)),
                         w[i, j, k] + Δτ * Gˢw[i, j, k] - ω̄ * Δτ * Mᵖ_bot * δz_π_bot + cofwt_bot)
        wᵉ_top = ifelse(k == Nz, zero(eltype(w)),
                         w[i, j, k + 1] + Δτ * Gˢw[i, j, k + 1] - ω̄ * Δτ * Mᵖ_top * δz_π_top + cofwt_top)

        ∂z_wᵉ = (wᵉ_top - wᵉ_bot) / Δzᶜ

        # RHS = π' + π′_forcing - ω Δτ S ∂wᵉ/∂z
        rhs_field[i, j, k] = π′[i, j, k] + π′_forcing[i, j, k] - ω * Δτ * Sⁱ * ∂z_wᵉ
    end
end

## Back-solve is no longer needed — _recover_w_and_update_pi! handles both
## w recovery and π' update after the tridiagonal solve.

#####
##### Section 8: Update π' with new w, apply damping, accumulate averages
#####
##### After the implicit w solve, update π' using the NEW w (α-weighted)
##### and apply the forward-extrapolation filter.
#####

@kernel function _update_pressure_and_average!(π′, π̃′, π′⁻,
                                               u, v, w, ū,
                                               grid, ϰᵈⁱ, avg_weight)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Forward-extrapolation filter: π̃′ = π'⁺ + ϰᵈⁱ (π'⁺ - π'⁻)
        π′⁺ = π′[i, j, k]
        π̃′[i, j, k] = π′⁺ + ϰᵈⁱ * (π′⁺ - π′⁻[i, j, k])

        # Save current π' as previous for next substep
        π′⁻[i, j, k] = π′⁺

        # Accumulate time-averaged velocities
        ū.u[i, j, k] += avg_weight * u[i, j, k]
        ū.v[i, j, k] += avg_weight * v[i, j, k]
        ū.w[i, j, k] += avg_weight * w[i, j, k]
    end
end

@kernel function _acoustic_divergence_damping!(u, v, π′, π′⁻, θᵥ, grid, ϰᵃᶜ, cᵖ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Klemp (2018) divergence damping: damp velocity proportional to the
        # PGF-scaled change in π' per substep. This provides constant damping
        # per outer Δt regardless of substep count N, stabilizing WS-RK3.
        #
        # u -= ϰᵃᶜ cᵖ θᵥ ∂(Δπ')/∂x,  v -= ϰᵃᶜ cᵖ θᵥ ∂(Δπ')/∂y
        #
        # The cᵖ θᵥ factor matches the PGF scaling so that ϰᵃᶜ is a
        # dimensionless O(1) coefficient (ϰᵃᶜ ∈ [2, 10] typical).
        Δπ_i   = π′[i, j, k]     - π′⁻[i, j, k]
        Δπ_im1 = π′[i - 1, j, k] - π′⁻[i - 1, j, k]
        Δx = Δxᶠᶜᶜ(i, j, k, grid)
        θᵥᶠᶜᶜ = ℑxTᶠᵃᵃ(i, j, k, grid, θᵥ)
        u[i, j, k] -= ϰᵃᶜ * cᵖ * θᵥᶠᶜᶜ * (Δπ_i - Δπ_im1) / Δx * !on_x_boundary(i, j, k, grid)

        Δπ_j   = π′[i, j, k]     - π′⁻[i, j, k]
        Δπ_jm1 = π′[i, j - 1, k] - π′⁻[i, j - 1, k]
        Δy = Δyᶜᶠᶜ(i, j, k, grid)
        θᵥᶜᶠᶜ = ℑyTᵃᶠᵃ(i, j, k, grid, θᵥ)
        v[i, j, k] -= ϰᵃᶜ * cᵖ * θᵥᶜᶠᶜ * (Δπ_j - Δπ_jm1) / Δy * !on_y_boundary(i, j, k, grid)
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

    # MPAS-style: reset all acoustic perturbation variables to ZERO.
    fill!(parent(substepper.exner_perturbation), 0)
    fill!(parent(substepper.filtered_exner_perturbation), 0)
    fill!(parent(substepper.previous_exner_perturbation), 0)
    fill!(parent(substepper.rtheta_pp), 0)
    fill!(parent(substepper.stage_thermodynamic_density), 0)

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

    ω = substepper.forward_weight
    ω̄ = 1 - ω
    ϰᵈⁱ = substepper.divergence_damping_coefficient
    ϰᵃᶜ = substepper.acoustic_damping_coefficient

    π′_forcing = CenterField(grid)  # TODO: pre-allocate this

    Gⁿ = model.timestepper.Gⁿ
    χ_name = thermodynamic_density_name(model.formulation)
    Gˢρθ = getproperty(Gⁿ, χ_name)
    g = model.thermodynamic_constants.gravitational_acceleration

    for _ in 1:Nτ
        # Step 1: Forward — update u, v from PGF and slow tendency
        launch!(arch, grid, :xyz, _acoustic_horizontal_forward!,
                u, v, grid, Δτ, cᵖ,
                substepper.filtered_exner_perturbation, substepper.virtual_potential_temperature,
                substepper.slow_tendencies.velocity.u,
                substepper.slow_tendencies.velocity.v)

        # Step 1b: Accumulate ts (MPAS-style ρθ perturbation).
        # ts = rtheta_pp_old + Δτ Gˢρθ - θ_m Δτ div_h(u)
        # This carries the gravity-wave signal through the substep loop.
        # Stored in stage_thermodynamic_density (scratch during substep loop).
        ts = substepper.stage_thermodynamic_density
        launch!(arch, grid, :xyz, _compute_ts!,
                ts, grid, Δτ,
                u, v, substepper.rtheta_pp, Gˢρθ,
                substepper.virtual_potential_temperature)

        # Step 2: Explicit π' forcing (Gˢπ + horizontal divergence + (1-ω)·∂w⁻/∂z)
        launch!(arch, grid, :xyz, _compute_π′_forcing!,
                π′_forcing, grid, Δτ, ω̄,
                u, v, w, substepper.acoustic_compression, substepper.slow_tendencies.exner_pressure)

        # Save π' before implicit solve (for damping)
        parent(substepper.previous_exner_perturbation) .= parent(substepper.exner_perturbation)

        # Step 3: Implicit solve — tridiagonal for π'⁺, back-solve for w⁺
        implicit_w_solve!(w, substepper, model, Δτ, π′_forcing)

        # Step 3b: Update rtheta_pp from ts and new w (vertical θ-flux divergence)
        launch!(arch, grid, :xyz, _update_rtheta_pp!,
                substepper.rtheta_pp, ts, grid, Δτ, w,
                substepper.virtual_potential_temperature)

        # Step 3c: Klemp (2018) divergence damping (if ϰᵃᶜ > 0)
        if ϰᵃᶜ > 0
            launch!(arch, grid, :xyz, _acoustic_divergence_damping!,
                    u, v, substepper.exner_perturbation, substepper.previous_exner_perturbation,
                    substepper.virtual_potential_temperature, grid, ϰᵃᶜ, cᵖ)
        end

        # Step 4: Apply ϰᵈⁱ forward-extrapolation + accumulate velocity averages
        launch!(arch, grid, :xyz, _update_pressure_and_average!,
                substepper.exner_perturbation, substepper.filtered_exner_perturbation, substepper.previous_exner_perturbation,
                u, v, w, ū,
                grid, ϰᵈⁱ, 1 / Nτ)
    end

    # Restore π'_initial = 0 for recovery (stage_thermodynamic_density was used as ts scratch)
    fill!(parent(substepper.stage_thermodynamic_density), 0)

    # Recovery: convert acoustic variables back to Breeze prognostic fields.
    Δt_stage = Nτ * Δτ
    recover_full_fields!(model, substepper, U⁰, Δt_stage)

    return nothing
end

@kernel function _reset_pi_prime_to_U0!(π′, πᵣ, ρχ⁰, pˢᵗ, Rᵈ, κ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # Compute π(Uⁿ) from ρθⁿ via the equation of state: π = (Rd·ρθ/p₀)^(R/cv)
        R_over_cv = κ / (1 - κ)
        πⁿ = (Rᵈ * ρχ⁰[i, j, k] / pˢᵗ)^R_over_cv
        π′[i, j, k] = πⁿ - πᵣ[i, j, k]
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

    # Read slow tendencies directly from Gⁿ (no substepper copy needed)
    Gⁿ = model.timestepper.Gⁿ
    χ_name = thermodynamic_density_name(model.formulation)
    Gˢρχ = getproperty(Gⁿ, χ_name)

    # Nonlinear recovery for WS-RK3:
    # ρθ: π'-perturbation approach — apply WS-RK3 perturbation in π'-space,
    #   then convert once via the equation of state. Avoids nonlinear splitting.
    # ρ: diagnosed from ρ = ρθ / θ_new where θ_new = θⁿ + Δt_stage · Gˢθ.
    #   θⁿ = ρθⁿ/ρⁿ from U⁰ (initial state), NOT θᵥ from the evaluation state.
    #   Using θᵥ would double-count the θ change from earlier stages.
    #   Gˢθ = (Gˢρθ - θᵥ·Gˢρ)/ρ is the slow θ tendency at the evaluation state.
    # π'_initial is saved in stage_thermodynamic_density.
    launch!(arch, grid, :xyz, _nonlinear_recovery_wsrk3!,
            model.dynamics.density, ρχ,
            substepper.exner_perturbation, substepper.stage_thermodynamic_density,
            substepper.reference_exner_function,
            substepper.virtual_potential_temperature, Gˢρχ, Gⁿ.ρ,
            U⁰[1], U⁰[5], pˢᵗ, Rᵈ, κ, Δt_stage)

    # Reconstruct momentum from updated density and velocity
    launch!(arch, grid, :xyz, _recover_momentum!,
            model.momentum, model.dynamics.density, model.velocities, grid)

    return nothing
end

@kernel function _nonlinear_recovery_wsrk3!(ρ, ρχ, π′_final, π′_initial, πᵣ,
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

    parent(substepper.filtered_exner_perturbation) .= parent(substepper.exner_perturbation)
    parent(substepper.previous_exner_perturbation) .= parent(substepper.exner_perturbation)

    u = model.velocities.u
    v = model.velocities.v
    w = model.velocities.w

    # Off-centering parameter for implicit solver (NOT the SSP coefficient)
    ω = substepper.forward_weight
    ω̄ = 1 - ω
    ϰᵈⁱ = substepper.divergence_damping_coefficient
    ϰᵃᶜ = substepper.acoustic_damping_coefficient
    π′_forcing = CenterField(grid)  # TODO: pre-allocate

    for _ in 1:Nτ
        launch!(arch, grid, :xyz, _acoustic_horizontal_forward!,
                u, v, grid, Δτ, cᵖ,
                substepper.filtered_exner_perturbation, substepper.virtual_potential_temperature,
                substepper.slow_tendencies.velocity.u,
                substepper.slow_tendencies.velocity.v)

        launch!(arch, grid, :xyz, _compute_π′_forcing!,
                π′_forcing, grid, Δτ, ω̄,
                u, v, w, substepper.acoustic_compression, substepper.slow_tendencies.exner_pressure)

        parent(substepper.previous_exner_perturbation) .= parent(substepper.exner_perturbation)
        implicit_w_solve!(w, substepper, model, Δτ, π′_forcing)

        if ϰᵃᶜ > 0
            launch!(arch, grid, :xyz, _acoustic_divergence_damping!,
                    u, v, substepper.exner_perturbation, substepper.previous_exner_perturbation,
                    substepper.virtual_potential_temperature, grid, ϰᵃᶜ, cᵖ)
        end

        launch!(arch, grid, :xyz, _update_pressure_and_average!,
                substepper.exner_perturbation, substepper.filtered_exner_perturbation, substepper.previous_exner_perturbation,
                u, v, w, ū,
                grid, ϰᵈⁱ, 1 / Nτ)
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

    # Read slow tendencies directly from Gⁿ (no substepper copy needed)
    Gⁿ = model.timestepper.Gⁿ
    χ_name = thermodynamic_density_name(model.formulation)
    Gˢρχ = getproperty(Gⁿ, χ_name)

    # Nonlinear recovery from π' to ρθ using the equation of state:
    # ρθ = (pˢᵗ/Rᵈ) * π^(cv/R) where π = πᵣ + π'
    # Density is diagnosed from ρ = ρθ/θ_new where θ_new includes the
    # slow advective θ tendency accumulated over Δt.
    launch!(arch, grid, :xyz, _nonlinear_recovery!,
            model.dynamics.density, ρχ,
            substepper.exner_perturbation, substepper.reference_exner_function, substepper.virtual_potential_temperature,
            Gˢρχ, Gⁿ.ρ,
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

@kernel function _nonlinear_recovery!(ρ, ρχ, π′, πᵣ, θᵥ, Gˢρχ, Gˢρ, pˢᵗ, Rᵈ, κ, Δt)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Nonlinear EOS: ρθ = (pˢᵗ/Rᵈ) π^(cᵥ/R), where π = πᵣ + π'
        cᵥ_over_R = (1 - κ) / κ
        π_total = πᵣ[i, j, k] + π′[i, j, k]
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
