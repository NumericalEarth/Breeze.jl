#####
##### Acoustic Substepping for CompressibleDynamics
#####
##### Implements split-explicit time integration following CM1/Wicker-Skamarock,
##### with optimizations from Oceananigans.SplitExplicitTimeDiscretizationFreeSurface:
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

Follows performance patterns from Oceananigans.SplitExplicitTimeDiscretizationFreeSurface:
- Precomputed thermodynamic coefficients (ψ = Rᵐ T, c²) for on-the-fly pressure
- Stage-frozen reference state (ρᵣ, χᵣ) for perturbation pressure gradient
- Time-averaged velocity fields for scalar advection
- Slow tendency storage for momentum, density, and thermodynamic variable

The acoustic substepping implements the forward-backward scheme from
[Wicker and Skamarock (2002)](@cite WickerSkamarock2002) and
[Klemp, Skamarock, and Dudhia (2007)](@cite KlempSkamarockDudhia2007):

1. **Forward step**: Update momentum using perturbation pressure gradient
2. **Backward step**: Update density and thermodynamic variable using new velocities
3. Accumulate time-averaged velocities for scalar transport

The vertical acoustic step is either fully explicit (`vertical_time_discretization = nothing`)
or vertically implicit (`vertical_time_discretization = VerticallyImplicit(α)`).

Fields
======

- `substeps`: Number of acoustic substeps per full time step
- `vertical_time_discretization`: Vertical vertical_time_discretization strategy (`nothing` or [`VerticallyImplicit`](@ref))
- `κᵈ`: Divergence damping coefficient (typically 0.05-0.1)
- `ψ`: Pressure coefficient ψ = Rᵐ T, so p = ψ ρ (CenterField)
- `c²`: Moist sound speed squared c² = γᵐ ψ (CenterField)
- `ū, v̄, w̄`: Time-averaged velocities for scalar advection
- `Gˢρu, Gˢρv, Gˢρw`: Slow momentum tendencies (fixed during acoustic loop)
- `Gˢρ`: Slow density tendency (fixed during acoustic loop)
- `Gˢχ`: Slow thermodynamic tendency (fixed during acoustic loop)
- `ρᵣ`: Stage-frozen reference density
- `χᵣ`: Stage-frozen reference thermodynamic variable (ρθ or ρe)
- `vertical_solver`: BatchedTridiagonalSolver for w-ρ implicit coupling (`nothing` when explicit)
- `rhs`: Right-hand side storage for tridiagonal solve (`nothing` when explicit)
"""
struct AcousticSubstepper{N, SS, FT, CF, UF, VF, WF, TS, RHS}
    # Number of acoustic substeps per full time step
    substeps :: N

    # Vertical vertical_time_discretization: nothing (explicit) or VerticallyImplicit(α)
    vertical_time_discretization :: SS

    # Divergence damping coefficient
    κᵈ :: FT

    # Precomputed thermodynamic coefficients (computed once per RK stage)
    ψ  :: CF  # Pressure coefficient: p = ψ ρ = Rᵐ T ρ
    c² :: CF  # Sound speed squared: c² = γᵐ ψ

    # Time-averaged velocities for scalar advection
    ū :: UF  # XFaceField
    v̄ :: VF  # YFaceField
    w̄ :: WF  # ZFaceField

    # Slow momentum tendencies (computed once per RK stage, held fixed during acoustic loop)
    Gˢρu :: UF
    Gˢρv :: VF
    Gˢρw :: WF

    # Slow density tendency (fixed during acoustic loop)
    Gˢρ :: CF

    # Slow thermodynamic tendency (fixed during acoustic loop)
    Gˢχ :: CF

    # Stage-frozen reference density (for perturbation pressure gradient and damping)
    ρᵣ :: CF

    # Stage-frozen reference thermodynamic variable (ρθ or ρe)
    χᵣ :: CF

    # Vertical tridiagonal solver for implicit w-ρ coupling (nothing when explicit)
    vertical_solver :: TS

    # Right-hand side storage for tridiagonal solve (nothing when explicit)
    rhs :: RHS
end

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.substeps,
                       a.vertical_time_discretization,
                       a.κᵈ,
                       adapt(to, a.ψ),
                       adapt(to, a.c²),
                       adapt(to, a.ū),
                       adapt(to, a.v̄),
                       adapt(to, a.w̄),
                       adapt(to, a.Gˢρu),
                       adapt(to, a.Gˢρv),
                       adapt(to, a.Gˢρw),
                       adapt(to, a.Gˢρ),
                       adapt(to, a.Gˢχ),
                       adapt(to, a.ρᵣ),
                       adapt(to, a.χᵣ),
                       adapt(to, a.vertical_solver),
                       adapt(to, a.rhs))

"""
    AcousticSubstepper(grid, split_explicit::SplitExplicitTimeDiscretization)

Construct an `AcousticSubstepper` for acoustic substepping on `grid`,
using parameters from `split_explicit`.
"""
function AcousticSubstepper(grid, split_explicit::SplitExplicitTimeDiscretization)
    Ns = split_explicit.substeps
    vertical_time_discretization = split_explicit.vertical_time_discretization
    FT = eltype(grid)
    κᵈ = convert(FT, split_explicit.κᵈ)

    # Thermodynamic coefficients
    ψ = CenterField(grid)
    c² = CenterField(grid)

    # Time-averaged velocities
    ū = XFaceField(grid)
    v̄ = YFaceField(grid)
    w̄ = ZFaceField(grid)

    # Slow momentum tendencies
    Gˢρu = XFaceField(grid)
    Gˢρv = YFaceField(grid)
    Gˢρw = ZFaceField(grid)

    # Slow density tendency
    Gˢρ = CenterField(grid)

    # Slow thermodynamic tendency
    Gˢχ = CenterField(grid)

    # Stage-frozen reference density
    ρᵣ = CenterField(grid)

    # Stage-frozen reference thermodynamic variable
    χᵣ = CenterField(grid)

    # Vertical tridiagonal solver (only allocated for implicit vertical stepping)
    vertical_solver = build_acoustic_vertical_solver(grid, vertical_time_discretization)
    rhs = build_acoustic_vertical_rhs(grid, vertical_time_discretization)

    return AcousticSubstepper(Ns, vertical_time_discretization, κᵈ,
                              ψ, c²,
                              ū, v̄, w̄,
                              Gˢρu, Gˢρv, Gˢρw,
                              Gˢρ, Gˢχ,
                              ρᵣ, χᵣ,
                              vertical_solver,
                              rhs)
end

# No vertical solver or RHS for explicit vertical stepping
build_acoustic_vertical_solver(grid, ::Nothing) = nothing
build_acoustic_vertical_rhs(grid, ::Nothing) = nothing

"""
Build the vertical tridiagonal solver for the implicit w-ρ coupling.
"""
function build_acoustic_vertical_solver(grid, ::VerticallyImplicit)
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

build_acoustic_vertical_rhs(grid, ::VerticallyImplicit) = ZFaceField(grid)

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
##### Prepare acoustic cache (once per RK stage)
#####

using Breeze.AtmosphereModels: thermodynamic_density

"""
$(TYPEDSIGNATURES)

Prepare the acoustic cache for an RK stage by storing the stage-frozen
reference state and computing linearized EOS coefficients.

This function:
1. Stores the stage-frozen reference density ρᵣ and thermodynamic variable χᵣ
2. Computes the linearized acoustic pressure coefficient ψ = Rᵐ T and sound
   speed squared c² = γᵐ ψ, held fixed during acoustic substepping

The perturbation pressure gradient during acoustic substeps uses:
``p' ≈ ψ ρ' = Rᵐ T (ρ - ρᵣ)``

For the potential temperature formulation, this is equivalent to the
linearized EOS ``p' = c̄² ρ'`` since ``θ`` is materially conserved
to leading order in acoustic perturbations.
"""
function prepare_acoustic_cache!(substepper, model)
    grid = model.grid
    arch = architecture(grid)

    # Store stage-frozen reference state
    χ = thermodynamic_density(model.formulation)
    parent(substepper.ρᵣ) .= parent(model.dynamics.density)
    parent(substepper.χᵣ) .= parent(χ)

    # Compute thermodynamic coefficients: ψ = Rᵐ T, c² = γᵐ ψ
    launch!(arch, grid, :xyz, _compute_acoustic_coefficients!,
            substepper.ψ, substepper.c²,
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
function acoustic_horizontal_momentum_step!(model, substepper, Δτ)
    grid = model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _acoustic_horizontal_momentum!,
            model.momentum.ρu, model.momentum.ρv, grid, Δτ,
            model.dynamics.density, substepper.ρᵣ, substepper.ψ)

    return nothing
end

@kernel function _acoustic_horizontal_momentum!(ρu, ρv, grid, Δτ, ρ, ρᵣ, ψ)
    i, j, k = @index(Global, NTuple)

    # Fast pressure gradient: ∂ₓp' = ψ ∂ₓρ' where ψ = Rᵐ T and ρ' = ρ - ρᵣ
    # We use the perturbation pressure gradient to avoid amplifying hydrostatic imbalance
    @inbounds begin
        # u-component: pressure gradient at (Face, Center, Center)
        ψᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ψ)
        ∂ₓρ′ = ∂xᶠᶜᶜ(i, j, k, grid, ρ) - ∂xᶠᶜᶜ(i, j, k, grid, ρᵣ)
        ∂ₓp′ = ψᶠᶜᶜ * ∂ₓρ′

        ρu[i, j, k] -= Δτ * ∂ₓp′

        # v-component: pressure gradient at (Center, Face, Center)
        ψᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ψ)
        ∂ᵧρ′ = ∂yᶜᶠᶜ(i, j, k, grid, ρ) - ∂yᶜᶠᶜ(i, j, k, grid, ρᵣ)
        ∂ᵧp′ = ψᶜᶠᶜ * ∂ᵧρ′

        ρv[i, j, k] -= Δτ * ∂ᵧp′
    end
end

#####
##### Thermodynamic variable update (backward step)
#####

"""
$(TYPEDSIGNATURES)

Update the thermodynamic variable χ (ρθ or ρe) during an acoustic substep.

Following [Klemp, Skamarock, and Dudhia (2007)](@cite KlempSkamarockDudhia2007) Eq. 15,
the thermodynamic variable is updated using a linearized flux divergence:

```math
χ^{τ+Δτ} = χ^τ - Δτ \\, \\boldsymbol{∇·}(\\bar{s} \\, \\boldsymbol{m}^{τ+Δτ})
    + Δτ \\, Π^{\\mathrm{ac}}(\\bar{U}) \\, \\boldsymbol{∇·u}^{τ+Δτ}
    + Δτ \\, G_{χ,\\mathrm{slow}}
```

where ``\\bar{s} = \\bar{χ} / \\bar{ρ}`` is the stage-frozen specific thermodynamic
variable (θ or e) and ``\\boldsymbol{m} = (ρu, ρv, ρw)`` is momentum.

The linearization advects the reference-level specific variable by the current
momentum, rather than computing the full nonlinear flux. This is critical for
stability of the acoustic substepping scheme.

For `LiquidIcePotentialTemperatureFormulation`, ``Π^{\\mathrm{ac}} = 0`` because
``θ`` is materially conserved.
"""
function acoustic_thermodynamic_step!(model, substepper, Δτ)
    grid = model.grid
    arch = architecture(grid)
    χ = thermodynamic_density(model.formulation)

    launch!(arch, grid, :xyz, _acoustic_thermodynamic_step!,
            χ, grid, Δτ,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            substepper.χᵣ, substepper.ρᵣ,
            substepper.Gˢχ)

    return nothing
end

@kernel function _acoustic_thermodynamic_step!(χ, grid, Δτ, ρu, ρv, ρw, χᵣ, ρᵣ, Gˢχ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Stage-frozen specific thermodynamic variable: s̄ = χ̄ / ρ̄
        s̄ = χᵣ[i, j, k] / ρᵣ[i, j, k]

        # Linearized flux divergence: ∇·(s̄ m) where m = (ρu, ρv, ρw) is the
        # newly updated momentum (backward step). The specific variable s̄ is
        # interpolated to cell centers via s̄ · ∇·m ≈ s̄ · div(ρu, ρv, ρw)
        # since s̄ varies slowly compared to the acoustic perturbations.
        div_m = divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)

        # Thermodynamic update: χ = χ - Δτ s̄ ∇·m + Δτ Gˢχ
        # Note: For ρθ formulation, Π^ac = 0, so no compression source term.
        # The compression source for static energy formulation will be added
        # via dispatch when that formulation supports split-explicit stepping.
        χ[i, j, k] += Δτ * (-s̄ * div_m + Gˢχ[i, j, k])
    end
end

#####
##### Vertical momentum update (semi-implicit)
#####

"""
Update vertical momentum with fast perturbation pressure gradient and buoyancy.

Uses perturbation quantities (ρ' = ρ - ρᵣ) to avoid amplifying hydrostatic imbalance.
For now, use explicit vertical pressure gradient.
The full implicit solve will be added in a future iteration.
"""
function acoustic_vertical_momentum_step!(model, substepper, Δτ, g)
    grid = model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _acoustic_vertical_momentum!,
            model.momentum.ρw, grid, Δτ, g,
            model.dynamics.density, substepper.ρᵣ, substepper.ψ)

    return nothing
end

@kernel function _acoustic_vertical_momentum!(ρw, grid, Δτ, g, ρ, ρᵣ, ψ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Perturbation pressure gradient at (Center, Center, Face): ∂p'/∂z = ψ ∂ρ'/∂z
        ψᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ψ)
        ∂zρ′ = ∂zᶜᶜᶠ(i, j, k, grid, ρ) - ∂zᶜᶜᶠ(i, j, k, grid, ρᵣ)
        ∂zp′ = ψᶜᶜᶠ * ∂zρ′

        # Perturbation buoyancy: b' = -g ρ' = -g (ρ - ρᵣ)
        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρᵣᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρᵣ)
        ρ′ = ρᶜᶜᶠ - ρᵣᶜᶜᶠ
        b′ = -g * ρ′

        # Fast terms: perturbation pressure gradient + perturbation buoyancy
        # ∂(ρw)/∂t = ... - ∂p'/∂z + b' = ... - ∂p'/∂z - g*ρ'
        ρw[i, j, k] += Δτ * (-∂zp′ + b′)
    end
end

#####
##### Density update (backward step, conservative form)
#####

"""
$(TYPEDSIGNATURES)

Update density using the conservative mass flux divergence (backward step).

The density is updated using the newly computed momentum (backward step):

```math
ρ^{τ+Δτ} = ρ^τ - Δτ \\, \\boldsymbol{∇·m}^{τ+Δτ} + Δτ \\, G_{ρ,\\mathrm{slow}}
```

where ``\\boldsymbol{m} = (ρu, ρv, ρw)`` is the momentum updated in the forward step.
Divergence damping is applied to suppress spurious acoustic oscillations.
"""
function acoustic_density_step!(model, substepper, Δτ)
    grid = model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _acoustic_density_step!,
            model.dynamics.density, grid, Δτ,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            substepper.ρᵣ, substepper.κᵈ, substepper.Gˢρ)

    return nothing
end

@kernel function _acoustic_density_step!(ρ, grid, Δτ, ρu, ρv, ρw, ρᵣ, κᵈ, Gˢρ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Mass flux divergence: ∇·m = ∇·(ρu, ρv, ρw) using new momentum
        div_m = divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)

        # Density update: ∂ₜρ = -∇·m + Gˢρ
        ρ[i, j, k] += Δτ * (-div_m + Gˢρ[i, j, k])

        # Divergence damping: nudge density toward stage-frozen reference
        ρ[i, j, k] -= κᵈ * (ρ[i, j, k] - ρᵣ[i, j, k])
    end
end

#####
##### Horizontal-only density update (for implicit vertical vertical_time_discretization)
#####

using Oceananigans.Operators: δxᶜᵃᵃ, δyᵃᶜᵃ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Vᶜᶜᶜ

"""
$(TYPEDSIGNATURES)

Update density using only horizontal mass flux divergence.

Used when the vertical density update is handled inside the implicit w-ρ solve.
"""
function acoustic_density_horizontal_step!(model, substepper, Δτ)
    grid = model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _acoustic_density_horizontal_step!,
            model.dynamics.density, grid, Δτ,
            model.momentum.ρu, model.momentum.ρv,
            substepper.ρᵣ, substepper.κᵈ, substepper.Gˢρ)

    return nothing
end

@kernel function _acoustic_density_horizontal_step!(ρ, grid, Δτ, ρu, ρv, ρᵣ, κᵈ, Gˢρ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Horizontal mass flux divergence only: ∂(ρu)/∂x + ∂(ρv)/∂y
        Vⁱ = Vᶜᶜᶜ(i, j, k, grid)
        div_m_h = (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, ρu) +
                   δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, ρv)) / Vⁱ

        # Density update: ∂ₜρ = -∇ₕ·m + Gˢρ
        ρ[i, j, k] += Δτ * (-div_m_h + Gˢρ[i, j, k])

        # Divergence damping
        ρ[i, j, k] -= κᵈ * (ρ[i, j, k] - ρᵣ[i, j, k])
    end
end

#####
##### Vertically implicit w-ρ solve
#####

using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, δzᵃᵃᶠ, δzᵃᵃᶜ, Az_qᶜᶜᶠ
using Oceananigans.Solvers: solve!

"""
$(TYPEDSIGNATURES)

Compute tridiagonal coefficients for the vertically implicit w-ρ solve.

The tridiagonal system arises from coupling the vertical momentum equation
(which depends on ∂ρ'/∂z) with the continuity equation (which depends on ∂(ρw)/∂z).
Eliminating ρ yields a tridiagonal system for ρw.

Following CM1 (`sound.F`, lines 661-718), the coefficients have an α² Δτ² factor
from the product of implicit weights in the w and ρ equations.

Called once per RK stage (coefficients depend only on frozen quantities).
"""
function compute_implicit_vertical_coefficients!(substepper, Δτ)
    solver = substepper.vertical_solver
    α = substepper.vertical_time_discretization.α
    α² = α * α
    grid = substepper.ψ.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _compute_implicit_vertical_coefficients!,
            solver.a, solver.b, solver.c,
            grid, α², Δτ, substepper.ψ)

    return nothing
end

@kernel function _compute_implicit_vertical_coefficients!(lower, diag, upper,
                                                          grid, α², Δτ, ψ)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Grid spacings
        Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)  # Face spacing (between centers k-1 and k)

        # Coupling coefficient: α² Δτ² ψ_face / Δz_face
        ψᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ψ)
        Q = α² * Δτ * Δτ * ψᶠ / Δzᶠ

        if k == 1 || k == Nz + 1
            # Boundary: w = 0 at top and bottom
            lower[i, j, k] = 0
            diag[i, j, k] = 1
            upper[i, j, k] = 0
        else
            # Interior: coupling to ρw at k-1, k, k+1
            Δzᶜk   = Δzᶜᶜᶜ(i, j, k, grid)     # Center spacing at k (between faces k and k+1)
            Δzᶜkm1 = Δzᶜᶜᶜ(i, j, k - 1, grid)  # Center spacing at k-1

            # Lower diagonal: coupling to (ρw)_{k-1} via ρ at center k-1
            lower[i, j, k] = -Q / Δzᶜkm1

            # Upper diagonal: coupling to (ρw)_{k+1} via ρ at center k
            upper[i, j, k] = -Q / Δzᶜk

            # Diagonal: self-coupling
            diag[i, j, k] = 1 + Q * (1 / Δzᶜk + 1 / Δzᶜkm1)
        end
    end
end

"""
$(TYPEDSIGNATURES)

Compute RHS of the tridiagonal system for vertical momentum and solve.

The RHS includes:
1. Current ρw
2. Off-centered explicit vertical pressure gradient and buoyancy
3. Implicit vertical pressure gradient using partially-updated density (after horizontal step)

After solving for (ρw)^{n+1}, updates density with the vertical mass flux divergence.
"""
function acoustic_implicit_vertical_step!(model, substepper, Δτ, g)
    grid = model.grid
    arch = architecture(grid)
    α = substepper.vertical_time_discretization.α
    β = 1 - α
    rhs = substepper.rhs

    # Build the RHS of the tridiagonal system
    launch!(arch, grid, :xyz, _compute_implicit_vertical_rhs!,
            rhs, grid, Δτ, α, β, g,
            model.momentum.ρw,
            model.dynamics.density, substepper.ρᵣ, substepper.ψ)

    # Solve the tridiagonal system: A (ρw)^{n+1} = rhs
    # Note: the solver handles k=1:Nz; the top face k=Nz+1 must be set to 0
    solve!(model.momentum.ρw, substepper.vertical_solver, rhs)

    # Enforce w=0 at top boundary (k=Nz+1 is outside the solver's range)
    Nz = size(grid, 3)
    ρw = model.momentum.ρw
    view(parent(ρw), :, :, Nz + 1 + ρw.data.offsets[3]) .= 0

    # Update density with the vertical mass flux divergence using new ρw
    launch!(arch, grid, :xyz, _acoustic_density_vertical_step!,
            model.dynamics.density, grid, Δτ,
            model.momentum.ρw)

    return nothing
end

@kernel function _compute_implicit_vertical_rhs!(rhs, grid, Δτ, α, β, g,
                                                  ρw, ρ, ρᵣ, ψ)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        if k == 1 || k == Nz + 1
            # Boundary: w = 0
            rhs[i, j, k] = 0
        else
            # Perturbation pressure gradient at face k: ψ ∂ρ'/∂z
            ψᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ψ)
            ∂zρ′ = ∂zᶜᶜᶠ(i, j, k, grid, ρ) - ∂zᶜᶜᶠ(i, j, k, grid, ρᵣ)
            ∂zp′ = ψᶠ * ∂zρ′

            # Perturbation buoyancy at face k: -g ρ'
            ρ′ᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ) - ℑzᵃᵃᶠ(i, j, k, grid, ρᵣ)

            # Vertical force: -∂p'/∂z - g ρ'
            F_vert = -∂zp′ - g * ρ′ᶠ

            # RHS = current ρw + full Δτ vertical force
            # (The tridiagonal matrix handles the implicit correction)
            rhs[i, j, k] = ρw[i, j, k] + Δτ * F_vert
        end
    end
end

@kernel function _acoustic_density_vertical_step!(ρ, grid, Δτ, ρw)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Vertical mass flux divergence: ∂(ρw)/∂z at cell center
        Vⁱ = Vᶜᶜᶜ(i, j, k, grid)
        div_ρw_z = δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, ρw) / Vⁱ

        # Update density with vertical divergence
        ρ[i, j, k] -= Δτ * div_ρw_z
    end
end

#####
##### Accumulate time-averaged velocities
#####

"""
$(TYPEDSIGNATURES)

Accumulate time-averaged velocities for scalar transport.

Time-averaged velocities from the acoustic loop are used for advection
of scalars (θ, moisture, tracers) in the outer RK loop, ensuring
mass-consistent transport following [Klemp, Skamarock, and Dudhia (2007)](@cite KlempSkamarockDudhia2007).
"""
function accumulate_time_averaged_velocities!(substepper, model, Nτ)
    grid = model.grid
    arch = architecture(grid)

    χᵗ = 1 / Nτ  # Uniform time-averaging weight

    launch!(arch, grid, :xyz, _accumulate_time_averaged_velocities!,
            substepper.ū, substepper.v̄, substepper.w̄,
            model.velocities.u, model.velocities.v, model.velocities.w,
            χᵗ)

    return nothing
end

@kernel function _accumulate_time_averaged_velocities!(ū, v̄, w̄, u, v, w, χᵗ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
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
    acoustic_substep_loop!(model, substepper, stage, Δt)

Execute the acoustic substep loop for RK `stage`.

Implements the forward-backward scheme from
[Wicker and Skamarock (2002)](@cite WickerSkamarock2002) and
[Klemp, Skamarock, and Dudhia (2007)](@cite KlempSkamarockDudhia2007):

1. Prepare acoustic cache (stage-frozen reference state and linearized EOS)
2. Apply slow momentum tendencies once for the full stage
3. Loop over acoustic substeps:
   a. **Forward step**: Update momentum from perturbation pressure gradient + buoyancy
   b. Update velocities from momentum
   c. **Backward step**: Update density from mass flux divergence (new momentum)
   d. **Backward step**: Update thermodynamic variable from linearized flux divergence
   e. Accumulate time-averaged velocities

Arguments
=========

- `model`: The `AtmosphereModel`
- `substepper`: The `AcousticSubstepper` containing storage and parameters
- `stage`: RK stage number (1, 2, or 3)
- `Δt`: Time step for this RK stage (α × full Δt)
"""
function acoustic_substep_loop!(model, substepper, stage, Δt)
    Ns = substepper.substeps
    g = model.thermodynamic_constants.gravitational_acceleration

    # Number of substeps for this RK stage
    Nτ = acoustic_substeps_per_stage(stage, Ns)
    Δτ = Δt / Nτ  # Acoustic substep time step

    # === PRECOMPUTE PHASE (once per RK stage) ===
    # Note: prepare_acoustic_cache! is called before this function
    # (in acoustic_ssp_rk3_substep!) because the slow tendency correction
    # needs the stage-frozen reference state.

    # Apply slow momentum tendencies to momentum ONCE for the full RK stage
    apply_slow_momentum_tendencies!(model, substepper, Δt)

    # Compute implicit vertical coefficients (once per stage, if implicit)
    compute_implicit_vertical_coefficients!(substepper, Δτ, substepper.vertical_time_discretization)

    # Initialize time-averaged velocities
    fill!(substepper.ū, 0)
    fill!(substepper.v̄, 0)
    fill!(substepper.w̄, 0)

    # === ACOUSTIC SUBSTEP LOOP ===
    for n = 1:Nτ
        acoustic_substep!(model, substepper, Δτ, g, Nτ, substepper.vertical_time_discretization)
    end

    return nothing
end

# No-op for explicit vertical vertical_time_discretization
compute_implicit_vertical_coefficients!(substepper, Δτ, ::Nothing) = nothing

# Compute coefficients for implicit vertical vertical_time_discretization
function compute_implicit_vertical_coefficients!(substepper, Δτ, vis::VerticallyImplicit)
    compute_implicit_vertical_coefficients!(substepper, Δτ)
    return nothing
end

#####
##### Explicit acoustic substep (vertical_time_discretization = nothing)
#####

"""
Perform one acoustic substep with explicit vertical stepping.
"""
function acoustic_substep!(model, substepper, Δτ, g, Nτ, ::Nothing)
    # (A) Forward step: update momentum from fast pressure gradient + buoyancy
    acoustic_horizontal_momentum_step!(model, substepper, Δτ)
    acoustic_vertical_momentum_step!(model, substepper, Δτ, g)

    # (B) Update velocities from momentum
    update_velocities_from_momentum!(model)

    # (C) Backward step: update density using new momentum (full 3D mass flux divergence)
    acoustic_density_step!(model, substepper, Δτ)

    # (D) Backward step: update thermodynamic variable using new momentum
    acoustic_thermodynamic_step!(model, substepper, Δτ)

    # (E) Accumulate time-averaged velocities for scalar transport
    accumulate_time_averaged_velocities!(substepper, model, Nτ)

    return nothing
end

#####
##### Implicit acoustic substep (vertical_time_discretization = VerticallyImplicit)
#####

"""
Perform one acoustic substep with vertically implicit w-ρ solve.

The step ordering differs from the explicit case:
1. Horizontal momentum update (explicit)
2. Horizontal density update (explicit, using new ρu, ρv)
3. Implicit vertical w-ρ solve (tridiagonal) + vertical density update
4. Update velocities from momentum
5. Thermodynamic variable update (backward step)
6. Accumulate time-averaged velocities
"""
function acoustic_substep!(model, substepper, Δτ, g, Nτ, ::VerticallyImplicit)
    # (A) Forward step: update horizontal momentum from fast pressure gradient
    acoustic_horizontal_momentum_step!(model, substepper, Δτ)

    # (B) Update density with horizontal divergence only (before implicit vertical solve)
    acoustic_density_horizontal_step!(model, substepper, Δτ)

    # (C) Implicit vertical w-ρ solve: builds RHS, solves tridiagonal, updates density vertically
    acoustic_implicit_vertical_step!(model, substepper, Δτ, g)

    # (D) Update velocities from momentum
    update_velocities_from_momentum!(model)

    # (E) Backward step: update thermodynamic variable using new momentum
    acoustic_thermodynamic_step!(model, substepper, Δτ)

    # (F) Accumulate time-averaged velocities for scalar transport
    accumulate_time_averaged_velocities!(substepper, model, Nτ)

    return nothing
end

"""
Apply slow momentum tendencies once per RK stage.

The slow tendencies (advection, Coriolis, diffusion - everything except
pressure gradient and buoyancy) are integrated over the full stage Δt.
"""
function apply_slow_momentum_tendencies!(model, substepper, Δt)
    grid = model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _apply_slow_momentum_tendencies!,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            substepper.Gˢρu, substepper.Gˢρv, substepper.Gˢρw,
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
