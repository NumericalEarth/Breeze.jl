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
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, divᶜᶜᶜ,
                              δxTᶠᵃᵃ, δyTᵃᶠᵃ, δxTᶜᵃᵃ, δyTᵃᶜᵃ,
                              Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Az_qᶜᶜᶠ, Vᶜᶜᶜ,
                              Δxᶠᶜᶜ, Δyᶜᶠᶜ,
                              δzᵃᵃᶜ
using Oceananigans.Utils: launch!

using Oceananigans.Grids: Periodic, Bounded,
                          AbstractUnderlyingGrid

using Adapt: Adapt, adapt

#####
##### Topology-aware interpolation operators (to be moved to Oceananigans)
#####
##### These avoid halo access for frozen fields during acoustic substeps.
##### Convention: ℑxTᶠᵃᵃ is the topology-aware version of ℑxᶠᵃᵃ.
#####

# Fallback: use standard interpolation
@inline ℑxTᶠᵃᵃ(i, j, k, grid, f::AbstractArray) = ℑxᶠᵃᵃ(i, j, k, grid, f)
@inline ℑyTᵃᶠᵃ(i, j, k, grid, f::AbstractArray) = ℑyᵃᶠᵃ(i, j, k, grid, f)

# Periodic: wrap at i=1 / j=1
const PX = AbstractUnderlyingGrid{FT, Periodic} where FT
const PY = AbstractUnderlyingGrid{FT, <:Any, Periodic} where FT

@inline function ℑxTᶠᵃᵃ(i, j, k, grid::PX, f::AbstractArray)
    @inbounds ifelse(i == 1,
                     (f[1, j, k] + f[grid.Nx, j, k]) / 2,
                     ℑxᶠᵃᵃ(i, j, k, grid, f))
end

@inline function ℑyTᵃᶠᵃ(i, j, k, grid::PY, f::AbstractArray)
    @inbounds ifelse(j == 1,
                     (f[i, 1, k] + f[i, grid.Ny, k]) / 2,
                     ℑyᵃᶠᵃ(i, j, k, grid, f))
end

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

The vertical acoustic step is either fully explicit (`time_discretization = nothing`)
or vertically implicit (`time_discretization = VerticallyImplicit(α)`).

Fields
======

- `substeps`: Number of acoustic substeps per full time step
- `time_discretization`: Vertical time_discretization strategy (`nothing` or [`VerticallyImplicit`](@ref))
- `divergence_damping_coefficient`: Divergence damping coefficient (typically 0.05-0.1)
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

    # Vertical time_discretization: nothing (explicit) or VerticallyImplicit(α)
    time_discretization :: SS

    # Divergence damping coefficient
    divergence_damping_coefficient :: FT

    # Precomputed thermodynamic coefficients (computed once per RK stage)
    ψ  :: CF  # Pressure coefficient: p = ψ ρ = Rᵐ T ρ
    c² :: CF  # Sound speed squared: c² = γᵐ ψ

    # Time-averaged velocities for scalar advection
    ū :: UF  # XFaceField
    v̄ :: VF  # YFaceField
    w̄ :: WF  # ZFaceField

    # Slow tendencies (full RHS R^t, computed once per RK stage)
    Gˢρu :: UF  # Slow x-momentum tendency
    Gˢρv :: VF  # Slow y-momentum tendency
    Gˢρw :: WF  # Slow z-momentum tendency
    Gˢρ  :: CF  # Slow density tendency = -∇·m^t
    Gˢχ  :: CF  # Slow thermodynamic tendency

    # Stage-frozen reference state
    ρᵣ :: CF  # Stage-frozen density
    χᵣ :: CF  # Stage-frozen thermodynamic variable (ρθ or ρe)

    # Perturbation fields (advanced during acoustic loop, start at zero each stage)
    ρu″ :: UF  # Perturbation x-momentum
    ρv″ :: VF  # Perturbation y-momentum
    ρw″ :: WF  # Perturbation z-momentum
    ρ″  :: CF  # Perturbation density
    χ″  :: CF  # Perturbation thermodynamic variable

    # Vertical tridiagonal solver for implicit w-ρ coupling (nothing when explicit)
    vertical_solver :: TS

    # Right-hand side storage for tridiagonal solve (nothing when explicit)
    rhs :: RHS
end

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.substeps,
                       a.time_discretization,
                       a.divergence_damping_coefficient,
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
                       adapt(to, a.ρu″),
                       adapt(to, a.ρv″),
                       adapt(to, a.ρw″),
                       adapt(to, a.ρ″),
                       adapt(to, a.χ″),
                       adapt(to, a.vertical_solver),
                       adapt(to, a.rhs))

"""
    AcousticSubstepper(grid, split_explicit::SplitExplicitTimeDiscretization)

Construct an `AcousticSubstepper` for acoustic substepping on `grid`,
using parameters from `split_explicit`.
"""
function AcousticSubstepper(grid, split_explicit::SplitExplicitTimeDiscretization)
    Ns = split_explicit.substeps
    time_discretization = split_explicit.time_discretization
    FT = eltype(grid)
    divergence_damping_coefficient = convert(FT, split_explicit.divergence_damping_coefficient)

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

    # Stage-frozen reference state
    ρᵣ = CenterField(grid)
    χᵣ = CenterField(grid)

    # Perturbation fields (zeroed at start of each RK stage)
    ρu″ = XFaceField(grid)
    ρv″ = YFaceField(grid)
    ρw″ = ZFaceField(grid)
    ρ″ = CenterField(grid)
    χ″ = CenterField(grid)

    # Vertical tridiagonal solver (only allocated for implicit vertical stepping)
    vertical_solver = build_acoustic_vertical_solver(grid, time_discretization)
    rhs = build_acoustic_vertical_rhs(grid, time_discretization)

    return AcousticSubstepper(Ns, time_discretization, divergence_damping_coefficient,
                              ψ, c²,
                              ū, v̄, w̄,
                              Gˢρu, Gˢρv, Gˢρw,
                              Gˢρ, Gˢχ,
                              ρᵣ, χᵣ,
                              ρu″, ρv″, ρw″, ρ″, χ″,
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

#####
##### Combined forward kernel: all momentum perturbation updates
#####

function acoustic_forward_step!(substepper, Δτ, g)
    grid = substepper.ψ.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _acoustic_forward_step!,
            substepper.ρu″, substepper.ρv″, substepper.ρw″,
            grid, Δτ, g,
            substepper.ρ″, substepper.ψ,
            substepper.Gˢρu, substepper.Gˢρv, substepper.Gˢρw)

    return nothing
end

@kernel function _acoustic_forward_step!(ρu″, ρv″, ρw″, grid, Δτ, g,
                                         ρ″, ψ, Gˢρu, Gˢρv, Gˢρw)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # x-momentum: topology-aware pressure gradient and interpolation
        ψᶠᶜᶜ = ℑxTᶠᵃᵃ(i, j, k, grid, ψ)
        ∂ₓp″ = ψᶠᶜᶜ * δxTᶠᵃᵃ(i, j, k, grid, ρ″) / Δxᶠᶜᶜ(i, j, k, grid)
        ρu″[i, j, k] += Δτ * (Gˢρu[i, j, k] - ∂ₓp″)

        # y-momentum: topology-aware pressure gradient and interpolation
        ψᶜᶠᶜ = ℑyTᵃᶠᵃ(i, j, k, grid, ψ)
        ∂ᵧp″ = ψᶜᶠᶜ * δyTᵃᶠᵃ(i, j, k, grid, ρ″) / Δyᶜᶠᶜ(i, j, k, grid)
        ρv″[i, j, k] += Δτ * (Gˢρv[i, j, k] - ∂ᵧp″)

        # z-momentum: skip bottom boundary face k=1 (w=0 there;
        # top face k=Nz+1 is outside the kernel range since launch is :xyz over centers)
        ψᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ψ)
        ∂zp″ = ψᶜᶜᶠ * ∂zᶜᶜᶠ(i, j, k, grid, ρ″)
        ρ″ᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ″)
        Δρw″ = Δτ * (Gˢρw[i, j, k] - ∂zp″ - g * ρ″ᶠ)
        ρw″[i, j, k] += Δρw″ * (k > 1)
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
#####
##### Combined backward kernel: density, thermodynamic, and velocity averaging
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
function acoustic_backward_step!(substepper, model, Δτ, Nτ)
    grid = substepper.ψ.grid
    arch = architecture(grid)
    χᵗ = 1 / Nτ

    launch!(arch, grid, :xyz, _acoustic_backward_step!,
            substepper.ρ″, substepper.χ″,
            substepper.ū, substepper.v̄, substepper.w̄,
            grid, Δτ, χᵗ, substepper.divergence_damping_coefficient,
            substepper.ρu″, substepper.ρv″, substepper.ρw″,
            substepper.χᵣ, substepper.ρᵣ,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            model.dynamics.density,
            substepper.Gˢρ, substepper.Gˢχ)

    return nothing
end

@kernel function _acoustic_backward_step!(ρ″, χ″, ū, v̄, w̄,
                                           grid, Δτ, χᵗ, κᵈ,
                                           ρu″, ρv″, ρw″,
                                           χᵣ, ρᵣ,
                                           ρu, ρv, ρw, ρ,
                                           Gˢρ, Gˢχ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Topology-aware perturbation momentum divergence
        Vⁱ = Vᶜᶜᶜ(i, j, k, grid)
        div_m″ = (δxTᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, ρu″) +
                  δyTᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, ρv″) +
                  δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, ρw″)) / Vⁱ

        # --- Density perturbation update ---
        ρ″[i, j, k] += Δτ * (Gˢρ[i, j, k] - div_m″)
        ρ″[i, j, k] *= (1 - κᵈ)

        # --- Thermodynamic perturbation update ---
        s̄ = χᵣ[i, j, k] / ρᵣ[i, j, k]
        χ″[i, j, k] += Δτ * (Gˢχ[i, j, k] - s̄ * div_m″)

        # --- Accumulate time-averaged velocities ---
        # Topology-aware interpolation for perturbation density (no halos filled)
        # Base density ρ has valid halos from update_state!
        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ) + ℑxTᶠᵃᵃ(i, j, k, grid, ρ″)
        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ) + ℑyTᵃᶠᵃ(i, j, k, grid, ρ″)
        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ) + ℑzᵃᵃᶠ(i, j, k, grid, ρ″)

        ū[i, j, k] += χᵗ * (ρu[i, j, k] + ρu″[i, j, k]) / ρᶠᶜᶜ
        v̄[i, j, k] += χᵗ * (ρv[i, j, k] + ρv″[i, j, k]) / ρᶜᶠᶜ
        w̄[i, j, k] += χᵗ * (ρw[i, j, k] + ρw″[i, j, k]) / ρᶜᶜᶠ
    end
end

#####
##### Horizontal-only density update (for implicit vertical time_discretization)
#####

##### (Operators Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Vᶜᶜᶜ etc. imported above)

function acoustic_density_horizontal_step!(substepper, Δτ)
    grid = substepper.ψ.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _acoustic_density_horizontal_step!,
            substepper.ρ″, grid, Δτ,
            substepper.ρu″, substepper.ρv″,
            substepper.divergence_damping_coefficient, substepper.Gˢρ)

    return nothing
end

@kernel function _acoustic_density_horizontal_step!(ρ″, grid, Δτ, ρu″, ρv″, κᵈ, Gˢρ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Vⁱ = Vᶜᶜᶜ(i, j, k, grid)
        div_m_h″ = (δxTᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, ρu″) +
                    δyTᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, ρv″)) / Vⁱ

        ρ″[i, j, k] += Δτ * (Gˢρ[i, j, k] - div_m_h″)
        ρ″[i, j, k] *= (1 - κᵈ)
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
    α = substepper.time_discretization.implicit_weight
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
function acoustic_implicit_vertical_step!(substepper, Δτ, g)
    grid = substepper.ψ.grid
    arch = architecture(grid)
    rhs = substepper.rhs

    # Build the RHS of the tridiagonal system using perturbation fields
    launch!(arch, grid, :xyz, _compute_implicit_vertical_rhs!,
            rhs, grid, Δτ, g,
            substepper.ρw″, substepper.ρ″, substepper.ψ, substepper.Gˢρw)

    # Solve the tridiagonal system: A ρw″^{n+1} = rhs
    solve!(substepper.ρw″, substepper.vertical_solver, rhs)

    # Enforce w=0 at top boundary (k=Nz+1 is outside the solver's range)
    Nz = size(grid, 3)
    ρw″ = substepper.ρw″
    view(parent(ρw″), :, :, Nz + 1 + ρw″.data.offsets[3]) .= 0

    # Update perturbation density with vertical perturbation momentum divergence
    launch!(arch, grid, :xyz, _acoustic_density_vertical_step!,
            substepper.ρ″, grid, Δτ, substepper.ρw″)

    return nothing
end

@kernel function _compute_implicit_vertical_rhs!(rhs, grid, Δτ, g,
                                                  ρw″, ρ″, ψ, Gˢρw)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        if k == 1 || k == Nz + 1
            rhs[i, j, k] = 0
        else
            # Perturbation pressure gradient: ψ ∂ρ″/∂z
            ψᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ψ)
            ∂zp″ = ψᶠ * ∂zᶜᶜᶠ(i, j, k, grid, ρ″)

            # Perturbation buoyancy: -g ρ″
            ρ″ᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ″)

            # RHS = current ρw″ + Δτ * (G_slow - ∂p″/∂z - g ρ″)
            rhs[i, j, k] = ρw″[i, j, k] + Δτ * (Gˢρw[i, j, k] - ∂zp″ - g * ρ″ᶠ)
        end
    end
end

@kernel function _acoustic_density_vertical_step!(ρ″, grid, Δτ, ρw″)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Vⁱ = Vᶜᶜᶜ(i, j, k, grid)
        div_ρw″_z = δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, ρw″) / Vⁱ
        ρ″[i, j, k] -= Δτ * div_ρw″_z
    end
end

##### (Velocity averaging is combined into _acoustic_backward_step!)
##### (update_velocities_from_momentum! removed -- velocities are computed
#####  from model fields during update_state!)

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
    # (in acoustic_ssp_rk3_substep!) because slow tendencies need
    # the stage-frozen reference state.

    # Compute implicit vertical coefficients (once per stage, if implicit)
    compute_implicit_vertical_coefficients!(substepper, Δτ, substepper.time_discretization)

    # Initialize perturbation fields to zero
    fill!(substepper.ρu″, 0)
    fill!(substepper.ρv″, 0)
    fill!(substepper.ρw″, 0)
    fill!(substepper.ρ″, 0)
    fill!(substepper.χ″, 0)

    # Initialize time-averaged velocities to zero
    fill!(substepper.ū, 0)
    fill!(substepper.v̄, 0)
    fill!(substepper.w̄, 0)

    # === ACOUSTIC SUBSTEP LOOP (advances perturbation variables) ===
    for n = 1:Nτ
        acoustic_substep!(model, substepper, Δτ, g, Nτ, substepper.time_discretization)
    end

    # === RECOVERY: add perturbations to model fields ===
    recover_full_fields!(model, substepper)

    return nothing
end

"""
Recover full fields from perturbation variables at the end of the substep loop.

Adds the perturbation to the stage-level model fields:
``U = U^t + U''``
"""
function recover_full_fields!(model, substepper)
    grid = model.grid
    arch = architecture(grid)
    χ = thermodynamic_density(model.formulation)

    launch!(arch, grid, :xyz, _recover_full_fields!,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            model.dynamics.density, χ,
            substepper.ρu″, substepper.ρv″, substepper.ρw″,
            substepper.ρ″, substepper.χ″)

    return nothing
end

@kernel function _recover_full_fields!(ρu, ρv, ρw, ρ, χ, ρu″, ρv″, ρw″, ρ″, χ″)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρu[i, j, k] += ρu″[i, j, k]
        ρv[i, j, k] += ρv″[i, j, k]
        ρw[i, j, k] += ρw″[i, j, k]
        ρ[i, j, k] += ρ″[i, j, k]
        χ[i, j, k] += χ″[i, j, k]
    end
end

# No-op for explicit vertical time_discretization
compute_implicit_vertical_coefficients!(substepper, Δτ, ::Nothing) = nothing

# Compute coefficients for implicit vertical time_discretization
function compute_implicit_vertical_coefficients!(substepper, Δτ, vis::VerticallyImplicit)
    compute_implicit_vertical_coefficients!(substepper, Δτ)
    return nothing
end

#####
##### Explicit acoustic substep (time_discretization = nothing)
#####

"""
Perform one acoustic substep with explicit vertical stepping.
Two kernel launches: forward (momentum) and backward (density + thermo + averaging).
"""
function acoustic_substep!(model, substepper, Δτ, g, Nτ, ::Nothing)
    # Forward: update all perturbation momentum (ρu″, ρv″, ρw″)
    acoustic_forward_step!(substepper, Δτ, g)

    # Backward: update ρ″, χ″, and accumulate ū, v̄, w̄
    acoustic_backward_step!(substepper, model, Δτ, Nτ)

    return nothing
end

#####
##### Implicit acoustic substep (time_discretization = VerticallyImplicit)
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
    # (A) Forward: update perturbation horizontal momentum only
    acoustic_forward_step!(substepper, Δτ, g)
    # TODO: for the implicit path, only the horizontal momentum should be
    # updated in the forward step. The vertical momentum is solved implicitly.
    # For now we update all momentum explicitly, then the implicit solve
    # overwrites ρw″.

    # (B) Horizontal density update (before implicit vertical solve)
    acoustic_density_horizontal_step!(substepper, Δτ)

    # (C) Implicit vertical w-ρ solve on perturbation fields
    acoustic_implicit_vertical_step!(substepper, Δτ, g)

    # (D) Backward: thermodynamic + velocity averaging
    acoustic_backward_step!(substepper, model, Δτ, Nτ)

    return nothing
end

##### (apply_slow_momentum_tendencies! removed -- slow tendencies are now
##### applied each substep through the perturbation-variable approach)
