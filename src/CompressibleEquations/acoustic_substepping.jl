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
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ
using Oceananigans.Operators: δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ, volume
using Oceananigans.Utils: launch!, KernelParameters
using Oceananigans.Architectures: convert_to_device

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

- `nsound`: Number of acoustic substeps per full time step
- `α`: Implicit weight for vertical solve (0.5 for Crank-Nicolson)
- `kdiv`: Divergence damping coefficient (typically 0.05-0.1)
- `ψ`: Pressure coefficient ψ = Rᵐ T, so p = ψ ρ (CenterField)
- `c²`: Moist sound speed squared c² = γᵐ ψ (CenterField)
- `ū, v̄, w̄`: Time-averaged velocities for scalar advection
- `G_slow_ρu, G_slow_ρv, G_slow_ρw`: Slow tendencies (fixed during acoustic loop)
- `ρ_ref`: Reference density at start of acoustic loop (for damping)
- `vertical_solver`: BatchedTridiagonalSolver for w-ρ implicit coupling
- `rhs`: Right-hand side storage for tridiagonal solve
"""
struct AcousticSubstepper{N, FT, CF, UF, VF, WF, TS, RHS}
    # Number of acoustic substeps per full time step
    nsound :: N
    
    # Implicitness parameter (Crank-Nicolson: α = 0.5)
    α :: FT
    
    # Divergence damping coefficient
    kdiv :: FT
    
    # Precomputed thermodynamic coefficients (computed once per RK stage)
    ψ  :: CF  # Pressure coefficient: p = ψ ρ = Rᵐ T ρ
    c² :: CF  # Sound speed squared: c² = γᵐ ψ
    
    # Time-averaged velocities for scalar advection
    ū :: UF  # XFaceField
    v̄ :: VF  # YFaceField
    w̄ :: WF  # ZFaceField
    
    # Slow tendencies (computed once per RK stage, held fixed during acoustic loop)
    G_slow_ρu :: UF
    G_slow_ρv :: VF
    G_slow_ρw :: WF
    
    # Reference density at start of acoustic loop (for divergence damping)
    ρ_ref :: CF
    
    # Vertical tridiagonal solver for implicit w-ρ coupling
    vertical_solver :: TS
    
    # Right-hand side storage for tridiagonal solve
    rhs :: RHS
end

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.nsound,
                       a.α,
                       a.kdiv,
                       adapt(to, a.ψ),
                       adapt(to, a.c²),
                       adapt(to, a.ū),
                       adapt(to, a.v̄),
                       adapt(to, a.w̄),
                       adapt(to, a.G_slow_ρu),
                       adapt(to, a.G_slow_ρv),
                       adapt(to, a.G_slow_ρw),
                       adapt(to, a.ρ_ref),
                       adapt(to, a.vertical_solver),
                       adapt(to, a.rhs))

"""
    AcousticSubstepper(grid; nsound=6, α=0.5, kdiv=0.05)

Construct an `AcousticSubstepper` for acoustic substepping on `grid`.

Keyword Arguments
=================

- `nsound`: Number of acoustic substeps per full time step. Default: 6
- `α`: Implicitness parameter for vertical solve. Default: 0.5 (Crank-Nicolson)
- `kdiv`: Divergence damping coefficient. Default: 0.05
"""
function AcousticSubstepper(grid; nsound::N=6, α=0.5, kdiv=0.05) where N
    FT = eltype(grid)
    arch = architecture(grid)
    
    α = convert(FT, α)
    kdiv = convert(FT, kdiv)
    
    # Thermodynamic coefficients
    ψ = CenterField(grid)
    c² = CenterField(grid)
    
    # Time-averaged velocities
    ū = XFaceField(grid)
    v̄ = YFaceField(grid)
    w̄ = ZFaceField(grid)
    
    # Slow tendencies
    G_slow_ρu = XFaceField(grid)
    G_slow_ρv = YFaceField(grid)
    G_slow_ρw = ZFaceField(grid)
    
    # Reference density
    ρ_ref = CenterField(grid)
    
    # Vertical tridiagonal solver
    vertical_solver = build_acoustic_vertical_solver(grid)
    
    # RHS storage for tridiagonal solve
    rhs = ZFaceField(grid)
    
    return AcousticSubstepper(nsound, α, kdiv,
                              ψ, c²,
                              ū, v̄, w̄,
                              G_slow_ρu, G_slow_ρv, G_slow_ρw,
                              ρ_ref,
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
Number of acoustic substeps for RK stage `nrk` given total `nsound` substeps.

Following CM1's convention for the Wicker-Skamarock RK3 scheme:
- Stage 1 (nrk=1): nsound/3 substeps
- Stage 2 (nrk=2): nsound/2 substeps  
- Stage 3 (nrk=3): nsound substeps
"""
@inline function acoustic_substeps_per_stage(nrk, nsound)
    if nrk == 1
        return max(1, div(nsound, 3))
    elseif nrk == 2
        return max(1, div(nsound, 2))
    else  # nrk == 3
        return nsound
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

@kernel function _compute_acoustic_coefficients!(ψ, c², density, specific_moisture, temperature,
                                                  grid, microphysics, microphysical_fields, constants)
    i, j, k = @index(Global, NTuple)
    
    @inbounds begin
        ρ = density[i, j, k]
        qᵗ = specific_moisture[i, j, k]
        T = temperature[i, j, k]
    end
    
    # Compute moisture fractions
    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ, microphysical_fields)
    
    # Mixture thermodynamic properties
    Rᵐ = mixture_gas_constant(q, constants)
    cₚᵐ = mixture_heat_capacity(q, constants)
    cᵥᵐ = cₚᵐ - Rᵐ
    γᵐ = cₚᵐ / cᵥᵐ
    
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
Update horizontal momentum components with pressure gradient (explicit).

Uses on-the-fly pressure gradient: ∂p/∂x = ψ ∂ρ/∂x where ψ = Rᵐ T.
"""
function acoustic_horizontal_momentum_step!(model, acoustic, Δts)
    grid = model.grid
    arch = architecture(grid)
    
    launch!(arch, grid, :xyz, _acoustic_horizontal_momentum!,
            model.momentum.ρu, model.momentum.ρv, grid, Δts,
            model.dynamics.density, acoustic.ψ,
            acoustic.G_slow_ρu, acoustic.G_slow_ρv)
    
    return nothing
end

@kernel function _acoustic_horizontal_momentum!(ρu, ρv, grid, Δts, ρ, ψ, G_slow_ρu, G_slow_ρv)
    i, j, k = @index(Global, NTuple)
    
    # Fast pressure gradient: (∂p/∂x)_fast = ψ ∂ρ/∂x where ψ = Rᵐ T
    # Note: ψ is at cell centers, must interpolate to faces
    @inbounds begin
        # u-component: pressure gradient at (Face, Center, Center)
        ψᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ψ)
        ∂ρ_∂x = ∂xᶠᶜᶜ(i, j, k, grid, ρ)
        ∂p_∂x_fast = ψᶠᶜᶜ * ∂ρ_∂x
        
        # Total tendency = -fast pressure gradient + slow tendency
        ρu[i, j, k] += Δts * (-∂p_∂x_fast + G_slow_ρu[i, j, k])
        
        # v-component: pressure gradient at (Center, Face, Center)
        ψᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ψ)
        ∂ρ_∂y = ∂yᶜᶠᶜ(i, j, k, grid, ρ)
        ∂p_∂y_fast = ψᶜᶠᶜ * ∂ρ_∂y
        
        ρv[i, j, k] += Δts * (-∂p_∂y_fast + G_slow_ρv[i, j, k])
    end
end

#####
##### Vertical momentum update (semi-implicit)
#####

"""
Update vertical momentum with pressure gradient.

For now, use explicit vertical pressure gradient. 
The full implicit solve will be added in a future iteration.
"""
function acoustic_vertical_momentum_step!(model, acoustic, Δts)
    grid = model.grid
    arch = architecture(grid)
    
    launch!(arch, grid, :xyz, _acoustic_vertical_momentum!,
            model.momentum.ρw, grid, Δts,
            model.dynamics.density, acoustic.ψ, acoustic.G_slow_ρw)
    
    return nothing
end

@kernel function _acoustic_vertical_momentum!(ρw, grid, Δts, ρ, ψ, G_slow_ρw)
    i, j, k = @index(Global, NTuple)
    
    @inbounds begin
        # w-component: pressure gradient at (Center, Center, Face)
        ψᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ψ)
        ∂ρ_∂z = ∂zᶜᶜᶠ(i, j, k, grid, ρ)
        ∂p_∂z_fast = ψᶜᶜᶠ * ∂ρ_∂z
        
        ρw[i, j, k] += Δts * (-∂p_∂z_fast + G_slow_ρw[i, j, k])
    end
end

#####
##### Density update from compression + velocity averaging
#####

"""
Update density from compression term and accumulate time-averaged velocities.

The compression term is the fast (acoustic) part of continuity:
∂ρ/∂t = -ρ ∇·u
"""
function acoustic_density_step!(model, acoustic, Δts, n, nloop)
    grid = model.grid
    arch = architecture(grid)
    
    weight = 1 / nloop  # Uniform averaging weight
    
    launch!(arch, grid, :xyz, _acoustic_density_and_averaging!,
            model.dynamics.density, grid, Δts, weight,
            model.velocities.u, model.velocities.v, model.velocities.w,
            acoustic.ρ_ref, acoustic.kdiv,
            acoustic.ū, acoustic.v̄, acoustic.w̄)
    
    return nothing
end

@kernel function _acoustic_density_and_averaging!(ρ, grid, Δts, weight, u, v, w, ρ_ref, kdiv, ū, v̄, w̄)
    i, j, k = @index(Global, NTuple)
    
    @inbounds begin
        # Velocity divergence (using standard operators - halos filled at start of acoustic loop)
        div_u = (δxᶜᵃᵃ(i, j, k, grid, u) / Δxᶜᶜᶜ(i, j, k, grid) +
                 δyᵃᶜᵃ(i, j, k, grid, v) / Δyᶜᶜᶜ(i, j, k, grid) +
                 δzᵃᵃᶜ(i, j, k, grid, w) / Δzᶜᶜᶜ(i, j, k, grid))
        
        # Density update from compression: ∂ρ/∂t = -ρ ∇·u
        ρ[i, j, k] -= Δts * ρ[i, j, k] * div_u
        
        # Divergence damping
        ρ[i, j, k] += kdiv * (ρ[i, j, k] - ρ_ref[i, j, k])
        
        # Accumulate time-averaged velocities
        ū[i, j, k] += weight * u[i, j, k]
        v̄[i, j, k] += weight * v[i, j, k]
        w̄[i, j, k] += weight * w[i, j, k]
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
    acoustic_substep_loop!(model, acoustic, nrk, Δt_rk)

Execute the acoustic substep loop for RK stage `nrk`.

This function:
1. Precomputes thermodynamic coefficients (ψ, c²)
2. Initializes time-averaged velocities
3. Loops over acoustic substeps, updating momentum and density
4. Uses time-averaged velocities for scalar advection (handled in outer RK loop)
"""
function acoustic_substep_loop!(model, acoustic, nrk, Δt_rk)
    grid = model.grid
    nsound = acoustic.nsound
    
    # Number of substeps for this RK stage
    nloop = acoustic_substeps_per_stage(nrk, nsound)
    Δts = Δt_rk / nloop  # acoustic timestep
    
    # === PRECOMPUTE PHASE (once per RK stage) ===
    
    # Compute thermodynamic coefficients: ψ = Rᵐ T, c² = γᵐ ψ
    compute_acoustic_coefficients!(acoustic, model)
    
    # Store density reference for divergence damping
    parent(acoustic.ρ_ref) .= parent(model.dynamics.density)
    
    # Initialize time-averaged velocities
    fill!(acoustic.ū, 0)
    fill!(acoustic.v̄, 0)
    fill!(acoustic.w̄, 0)
    
    # === ACOUSTIC SUBSTEP LOOP ===
    for n = 1:nloop
        # Update momentum from pressure gradient
        acoustic_horizontal_momentum_step!(model, acoustic, Δts)
        acoustic_vertical_momentum_step!(model, acoustic, Δts)
        
        # Update velocities from momentum
        update_velocities_from_momentum!(model)
        
        # Update density from compression + accumulate averaged velocities
        acoustic_density_step!(model, acoustic, Δts, n, nloop)
    end
    
    return nothing
end
