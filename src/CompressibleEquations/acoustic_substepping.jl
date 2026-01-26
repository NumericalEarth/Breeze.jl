#####
##### Acoustic Substepping for CompressibleDynamics
#####
##### Implements split-explicit time integration following CM1/Wicker-Skamarock,
##### with optimizations from Oceananigans.SplitExplicitFreeSurface:
##### - On-the-fly pressure gradient computation
##### - Pre-converted kernel arguments
##### - Topology-aware operators (no halo filling between substeps)
#####

using KernelAbstractions: @kernel, @index

using Oceananigans: CenterField, XFaceField, YFaceField, ZFaceField, architecture
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: BatchedTridiagonalSolver
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, divᶜᶜᶜ
using Oceananigans.Utils: launch!

using Adapt: Adapt, adapt

"""
    AcousticSubstepper

Storage and parameters for acoustic substepping within each RK stage.

Follows performance patterns from Oceananigans.SplitExplicitFreeSurface:
- Precomputed thermodynamic coefficients (ψ = Rᵐ T, c²) for on-the-fly pressure
- Time-averaged velocity fields for scalar advection
- Slow tendency storage
- Reference density for divergence damping
- Vertical tridiagonal solver

Fields
======

- `Ns`: Number of acoustic substeps per full time step
- `α`: Implicit weight for vertical solve (0.5 for Crank-Nicolson)
- `κᵈ`: Divergence damping coefficient (typically 0.05-0.1)
- `ψ`: Pressure coefficient ψ = Rᵐ T, so p = ψ ρ (CenterField)
- `c²`: Moist sound speed squared c² = γᵐ ψ (CenterField)
- `ū, v̄, w̄`: Time-averaged velocities for scalar advection
- `Gˢρu, Gˢρv, Gˢρw`: Slow tendencies (fixed during acoustic loop)
- `ρᵣ`: Reference density at start of acoustic loop (for damping)
- `vertical_solver`: BatchedTridiagonalSolver for w-ρ implicit coupling
- `rhs`: Right-hand side storage for tridiagonal solve
"""
struct AcousticSubstepper{N, FT, CF, UF, VF, WF, TS, RHS}
    # Number of acoustic substeps per full time step
    Ns :: N

    # Implicitness parameter (Crank-Nicolson: α = 0.5)
    α :: FT

    # Divergence damping coefficient
    κᵈ :: FT

    # Precomputed thermodynamic coefficients (computed once per RK stage)
    ψ  :: CF  # Pressure coefficient: p = ψ ρ = Rᵐ T ρ
    c² :: CF  # Sound speed squared: c² = γᵐ ψ

    # Time-averaged velocities for scalar advection
    ū :: UF  # XFaceField
    v̄ :: VF  # YFaceField
    w̄ :: WF  # ZFaceField

    # Slow tendencies (computed once per RK stage, held fixed during acoustic loop)
    Gˢρu :: UF
    Gˢρv :: VF
    Gˢρw :: WF

    # Reference density at start of acoustic loop (for divergence damping)
    ρᵣ :: CF

    # Vertical tridiagonal solver for implicit w-ρ coupling
    vertical_solver :: TS

    # Right-hand side storage for tridiagonal solve
    rhs :: RHS
end

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.Ns,
                       a.α,
                       a.κᵈ,
                       adapt(to, a.ψ),
                       adapt(to, a.c²),
                       adapt(to, a.ū),
                       adapt(to, a.v̄),
                       adapt(to, a.w̄),
                       adapt(to, a.Gˢρu),
                       adapt(to, a.Gˢρv),
                       adapt(to, a.Gˢρw),
                       adapt(to, a.ρᵣ),
                       adapt(to, a.vertical_solver),
                       adapt(to, a.rhs))

"""
    AcousticSubstepper(grid; Ns=6, α=0.5, κᵈ=0.05)

Construct an `AcousticSubstepper` for acoustic substepping on `grid`.

Keyword Arguments
=================

- `Ns`: Number of acoustic substeps per full time step. Default: 6
- `α`: Implicitness parameter for vertical solve. Default: 0.5 (Crank-Nicolson)
- `κᵈ`: Divergence damping coefficient. Default: 0.05
"""
function AcousticSubstepper(grid; Ns::N=6, α=0.5, κᵈ=0.05) where N
    FT = eltype(grid)
    arch = architecture(grid)

    α = convert(FT, α)
    κᵈ = convert(FT, κᵈ)

    # Thermodynamic coefficients
    ψ = CenterField(grid)
    c² = CenterField(grid)

    # Time-averaged velocities
    ū = XFaceField(grid)
    v̄ = YFaceField(grid)
    w̄ = ZFaceField(grid)

    # Slow tendencies
    Gˢρu = XFaceField(grid)
    Gˢρv = YFaceField(grid)
    Gˢρw = ZFaceField(grid)

    # Reference density
    ρᵣ = CenterField(grid)

    # Vertical tridiagonal solver
    vertical_solver = build_acoustic_vertical_solver(grid)

    # RHS storage for tridiagonal solve
    rhs = ZFaceField(grid)

    return AcousticSubstepper(Ns, α, κᵈ,
                              ψ, c²,
                              ū, v̄, w̄,
                              Gˢρu, Gˢρv, Gˢρw,
                              ρᵣ,
                              vertical_solver,
                              rhs)
end

"""
Build the vertical tridiagonal solver for the implicit w-ρ coupling.
"""
function build_acoustic_vertical_solver(grid)
    arch = architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)

    # Diagonal coefficients vary in space (3D arrays)
    lower_diagonal = zeros(arch, FT, Nx, Ny, Nz)
    diagonal = zeros(arch, FT, Nx, Ny, Nz)
    upper_diagonal = zeros(arch, FT, Nx, Ny, Nz)
    scratch = zeros(arch, FT, Nx, Ny, Nz)

    return BatchedTridiagonalSolver(grid;
                                    lower_diagonal,
                                    diagonal,
                                    upper_diagonal,
                                    scratch,
                                    tridiagonal_direction = ZDirection())
end

#####
##### Acoustic substep count per RK stage (following CM1)
#####

"""
    acoustic_substeps_per_stage(stage, Ns)

Number of acoustic substeps for RK `stage` given total `Ns` substeps per time step.

Following CM1's convention for the Wicker-Skamarock SSP RK3 scheme:
- Stage 1: Ns/3 substeps
- Stage 2: Ns/2 substeps
- Stage 3: Ns substeps

Arguments
=========

- `stage`: RK stage number (1, 2, or 3)
- `Ns`: Total number of acoustic substeps per full time step
"""
@inline function acoustic_substeps_per_stage(stage, Ns)
    if stage == 1
        return max(1, div(Ns, 3))
    elseif stage == 2
        return max(1, div(Ns, 2))
    else  # stage == 3
        return Ns
    end
end

#####
##### Compute thermodynamic coefficients (once per RK stage)
#####

"""
Compute ψ = Rᵐ T and c² = γᵐ ψ for the acoustic substep loop.

These coefficients are held fixed during acoustic substepping since
temperature evolves via slow tendencies only.
"""
function compute_acoustic_coefficients!(acoustic, model)
    grid = model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _compute_acoustic_coefficients!,
            acoustic.ψ, acoustic.c²,
            model.dynamics.density,
            model.specific_moisture,
            model.temperature,
            grid,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    return nothing
end

@kernel function _compute_acoustic_coefficients!(ψ, c², ρ_field, qᵗ_field, T_field,
                                                  grid, microphysics, microphysical_fields, constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ = ρ_field[i, j, k]
        qᵗ = qᵗ_field[i, j, k]
        T = T_field[i, j, k]
    end

    # Compute moisture fractions
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ, microphysical_fields)

    # Mixture thermodynamic properties
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    cᵛᵐ = cᵖᵐ - Rᵐ
    γᵐ = cᵖᵐ / cᵛᵐ

    @inbounds begin
        # Pressure coefficient: p = ψ ρ
        ψ[i, j, k] = Rᵐ * T

        # Moist sound speed squared: c² = γᵐ ψ = γᵐ Rᵐ T
        c²[i, j, k] = γᵐ * ψ[i, j, k]
    end
end

#####
##### Horizontal momentum update (explicit, on-the-fly pressure gradient)
#####

"""
Update horizontal momentum with fast pressure gradient (explicit).

Uses on-the-fly pressure gradient: ∂p/∂x = ψ ∂ρ/∂x where ψ = Rᵐ T.
"""
function acoustic_horizontal_momentum_step!(model, acoustic, Δtˢ)
    grid = model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _acoustic_horizontal_momentum!,
            model.momentum.ρu, model.momentum.ρv, grid, Δtˢ,
            model.dynamics.density, acoustic.ψ)

    return nothing
end

@kernel function _acoustic_horizontal_momentum!(ρu, ρv, grid, Δtˢ, ρ, ψ)
    i, j, k = @index(Global, NTuple)

    # Fast pressure gradient: ∂ₓp = ψ ∂ₓρ where ψ = Rᵐ T
    # Note: ψ is at cell centers, must interpolate to faces
    @inbounds begin
        # u-component: pressure gradient at (Face, Center, Center)
        ψᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ψ)
        ∂ₓρ = ∂xᶠᶜᶜ(i, j, k, grid, ρ)
        ∂ₓp = ψᶠᶜᶜ * ∂ₓρ

        ρu[i, j, k] -= Δtˢ * ∂ₓp

        # v-component: pressure gradient at (Center, Face, Center)
        ψᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ψ)
        ∂ᵧρ = ∂yᶜᶠᶜ(i, j, k, grid, ρ)
        ∂ᵧp = ψᶜᶠᶜ * ∂ᵧρ

        ρv[i, j, k] -= Δtˢ * ∂ᵧp
    end
end

#####
##### Vertical momentum update (semi-implicit)
#####

"""
Update vertical momentum with fast pressure gradient and buoyancy.

For now, use explicit vertical pressure gradient.
The full implicit solve will be added in a future iteration.
"""
function acoustic_vertical_momentum_step!(model, acoustic, Δtˢ, g)
    grid = model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _acoustic_vertical_momentum!,
            model.momentum.ρw, grid, Δtˢ, g,
            model.dynamics.density, acoustic.ψ)

    return nothing
end

@kernel function _acoustic_vertical_momentum!(ρw, grid, Δtˢ, g, ρ, ψ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Pressure gradient at (Center, Center, Face)
        ψᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ψ)
        ∂zρ = ∂zᶜᶜᶠ(i, j, k, grid, ρ)
        ∂zp = ψᶜᶜᶠ * ∂zρ

        # Buoyancy: b = -g at cell faces
        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρb = -g * ρᶜᶜᶠ

        # Fast terms: pressure gradient + buoyancy
        ρw[i, j, k] += Δtˢ * (-∂zp + ρb)
    end
end

#####
##### Density update from compression + velocity averaging
#####

"""
Update density from compression and accumulate time-averaged velocities.

The compression term is the fast (acoustic) part of continuity:
∂ₜρ = -ρ ∇·u
"""
function acoustic_density_step!(model, acoustic, Δtˢ, n, Nsₛₜₐgₑ)
    grid = model.grid
    arch = architecture(grid)

    χᵗ = 1 / Nsₛₜₐgₑ  # Uniform time-averaging weight

    launch!(arch, grid, :xyz, _acoustic_density_and_averaging!,
            model.dynamics.density, grid, Δtˢ, χᵗ,
            model.velocities.u, model.velocities.v, model.velocities.w,
            acoustic.ρᵣ, acoustic.κᵈ,
            acoustic.ū, acoustic.v̄, acoustic.w̄)

    return nothing
end

@kernel function _acoustic_density_and_averaging!(ρ, grid, Δtˢ, χᵗ, u, v, w, ρᵣ, κᵈ, ū, v̄, w̄)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Velocity divergence
        ∇u = divᶜᶜᶜ(i, j, k, grid, u, v, w)

        # Density update from compression: ∂ₜρ = -ρ ∇·u
        ρ[i, j, k] -= Δtˢ * ρ[i, j, k] * ∇u

        # Divergence damping: nudge density toward reference
        ρ[i, j, k] -= κᵈ * (ρ[i, j, k] - ρᵣ[i, j, k])

        # Accumulate time-averaged velocities
        ū[i, j, k] += χᵗ * u[i, j, k]
        v̄[i, j, k] += χᵗ * v[i, j, k]
        w̄[i, j, k] += χᵗ * w[i, j, k]
    end
end

#####
##### Update velocities from momentum
#####

"""
Update velocity fields from momentum and density: u = ρu / ρ
"""
function update_velocities_from_momentum!(model)
    grid = model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _update_velocities!,
            model.velocities.u, model.velocities.v, model.velocities.w,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            model.dynamics.density, grid)

    return nothing
end

@kernel function _update_velocities!(u, v, w, ρu, ρv, ρw, ρ, grid)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)

        u[i, j, k] = ρu[i, j, k] / ρᶠᶜᶜ
        v[i, j, k] = ρv[i, j, k] / ρᶜᶠᶜ
        w[i, j, k] = ρw[i, j, k] / ρᶜᶜᶠ
    end
end

#####
##### Main acoustic substep loop
#####

"""
    acoustic_substep_loop!(model, acoustic, stage, Δt)

Execute the acoustic substep loop for RK `stage`.

This function:
1. Applies slow tendencies to momentum once (for the full RK stage)
2. Precomputes thermodynamic coefficients (ψ, c²)
3. Initializes time-averaged velocities
4. Loops over acoustic substeps, updating momentum and density with fast terms
5. Uses time-averaged velocities for scalar advection (handled in outer RK loop)

Arguments
=========

- `model`: The `AtmosphereModel`
- `acoustic`: The `AcousticSubstepper` containing storage and parameters
- `stage`: RK stage number (1, 2, or 3)
- `Δt`: Time step for this RK stage (α × full Δt)
"""
function acoustic_substep_loop!(model, acoustic, stage, Δt)
    grid = model.grid
    arch = architecture(grid)
    Ns = acoustic.Ns
    g = model.thermodynamic_constants.gravitational_acceleration

    # Number of substeps for this RK stage
    Nsₛₜₐgₑ = acoustic_substeps_per_stage(stage, Ns)
    Δtˢ = Δt / Nsₛₜₐgₑ  # Acoustic substep time step

    # === PRECOMPUTE PHASE (once per RK stage) ===

    # Apply slow tendencies to momentum ONCE for the full RK stage
    apply_slow_momentum_tendencies!(model, acoustic, Δt)

    # Compute thermodynamic coefficients: ψ = Rᵐ T, c² = γᵐ ψ
    compute_acoustic_coefficients!(acoustic, model)

    # Store density reference for divergence damping
    parent(acoustic.ρᵣ) .= parent(model.dynamics.density)

    # Initialize time-averaged velocities
    fill!(acoustic.ū, 0)
    fill!(acoustic.v̄, 0)
    fill!(acoustic.w̄, 0)

    # === ACOUSTIC SUBSTEP LOOP ===
    for n = 1:Nsₛₜₐgₑ
        # Update momentum from fast terms (pressure gradient + buoyancy)
        acoustic_horizontal_momentum_step!(model, acoustic, Δtˢ)
        acoustic_vertical_momentum_step!(model, acoustic, Δtˢ, g)

        # Update velocities from momentum
        update_velocities_from_momentum!(model)

        # Update density from compression + accumulate averaged velocities
        acoustic_density_step!(model, acoustic, Δtˢ, n, Nsₛₜₐgₑ)
    end

    return nothing
end

"""
Apply slow momentum tendencies once per RK stage.

The slow tendencies (advection, Coriolis, diffusion - everything except
pressure gradient and buoyancy) are integrated over the full stage Δt.
"""
function apply_slow_momentum_tendencies!(model, acoustic, Δt)
    grid = model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _apply_slow_momentum_tendencies!,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            acoustic.Gˢρu, acoustic.Gˢρv, acoustic.Gˢρw,
            Δt)

    return nothing
end

@kernel function _apply_slow_momentum_tendencies!(ρu, ρv, ρw, Gˢρu, Gˢρv, Gˢρw, Δt)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρu[i, j, k] += Δt * Gˢρu[i, j, k]
        ρv[i, j, k] += Δt * Gˢρv[i, j, k]
        ρw[i, j, k] += Δt * Gˢρw[i, j, k]
    end
end
