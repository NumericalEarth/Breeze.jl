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
using Oceananigans.Operators:
    ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ,
    ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ,
    ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ,
    δxTᶠᵃᵃ, δyTᵃᶠᵃ, δxTᶜᵃᵃ, δyTᵃᶜᵃ,
    Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Az_qᶜᶜᶠ, Vᶜᶜᶜ, divᶜᶜᶜ,
    Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶜ,
    δzᵃᵃᶜ, δzᵃᵃᶠ

using Oceananigans.Utils: launch!, configure_kernel
using Oceananigans.Architectures: convert_to_device

using Oceananigans.Grids: Periodic, Bounded,
                          AbstractUnderlyingGrid

using Adapt: Adapt, adapt

#####
##### Topology-aware interpolation and difference operators
#####
##### These avoid halo access for frozen fields during acoustic substeps.
##### Convention: ℑxTᶠᵃᵃ is the topology-aware version of ℑxᶠᵃᵃ.
##### Convention: δxTᶠᵃᵃ is the topology-aware version of δxTᶠᵃᵃ.
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
# (face Nx+1 / Ny+1 are outside the :xyz kernel range)
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

# Bounded vertical: handle boundary faces k=1 and k=Nz+1
# For a bounded vertical grid, the vertical interpolation to face k
# averages values at cell centers k-1 and k.
# At k=1 (bottom face), there is no k-1 center in the interior, so use k=1 center value.
# At k=Nz+1 (top face), there is no k+1 center in the interior, so use k=Nz center value.
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

# Vertical difference for bounded grids: δz f at face k = f[k] - f[k-1]
# At k=1 (bottom face), there is no k-1 center, so return 0.
# At k=Nz+1 (top face), there is no k center, so return 0.
@inline function δzTᵃᵃᶠ(i, j, k, grid::BZ, f::AbstractArray)
    Nz = size(grid, 3)
    bottom = k == 1
    top = k == Nz + 1
    return @inbounds ifelse(bottom, zero(eltype(f)),
                     ifelse(top, zero(eltype(f)),
                            δzᵃᵃᶠ(i, j, k, grid, f)))
end

"""
    AcousticSubstepper

Storage and parameters for acoustic substepping within each RK stage.

Follows performance patterns from Oceananigans.SplitExplicitTimeDiscretizationFreeSurface:
- Precomputed thermodynamic coefficients (ψ = Rᵐ T, c²) for on-the-fly pressure
- Stage-frozen reference state (ρᵣ, ρχᵣ) for perturbation pressure gradient
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
- `divergence_damping_coefficient`: Divergence damping coefficient. Must satisfy `(1-κᵈ)^Ns < 0.1` for stability with base-state pressure correction (e.g., κᵈ=0.2 for Ns=12)
- `pressure_coefficient`: Pressure coefficient ψ = Rᵐ T, so p = ψ ρ (CenterField)
- `sound_speed_squared`: Moist acoustic sound speed squared cᵃᶜ² = γᵐ ψ (CenterField)
- `averaged_velocities`: NamedTuple with (u, v, w) time-averaged velocities for scalar advection
- `slow_momentum_tendencies`: NamedTuple with (ρu, ρv, ρw) slow momentum tendencies (fixed during acoustic loop)
- `Gˢρ`: Slow density tendency (fixed during acoustic loop)
- `Gˢρχ`: Slow thermodynamic tendency (fixed during acoustic loop)
- `ρᵣ`: Stage-frozen reference density
- `ρχᵣ`: Stage-frozen reference thermodynamic variable (ρθ or ρe)
- `perturbation_momentum`: NamedTuple with (ρu, ρv, ρw) perturbation momentum
- `ρ″`: Perturbation density
- `ρχ″`: Perturbation thermodynamic variable
- `vertical_solver`: BatchedTridiagonalSolver for w-ρ implicit coupling (`nothing` when explicit)
- `rhs`: Right-hand side storage for tridiagonal solve (`nothing` when explicit)
"""
struct AcousticSubstepper{N, SS, FT, CF, AV, SM, PM, TS, RHS}
    # Number of acoustic substeps per full time step
    substeps :: N

    # Vertical time_discretization: nothing (explicit) or VerticallyImplicit(α)
    time_discretization :: SS

    # Divergence damping coefficient
    divergence_damping_coefficient :: FT

    # Precomputed thermodynamic coefficients (computed once per RK stage)
    pressure_coefficient :: CF  # ψ = Rᵐ T, so p = ψ ρ
    sound_speed_squared  :: CF  # cᵃᶜ² = γᵐ ψ = γᵐ Rᵐ T (acoustic sound speed squared)

    # Time-averaged velocities for scalar advection (NamedTuple with u, v, w)
    averaged_velocities :: AV

    # Slow tendencies (full RHS R^t, computed once per RK stage)
    slow_momentum_tendencies :: SM  # NamedTuple with ρu, ρv, ρw
    Gˢρ  :: CF  # Slow density tendency = -∇·m^t
    Gˢρχ :: CF  # Slow thermodynamic tendency

    # Stage-frozen reference state
    ρᵣ  :: CF  # Stage-frozen density
    ρχᵣ :: CF  # Stage-frozen thermodynamic variable (ρθ or ρe)

    # Perturbation fields (advanced during acoustic loop, start at zero each stage)
    perturbation_momentum :: PM  # NamedTuple with ρu, ρv, ρw
    ρ″  :: CF  # Perturbation density
    ρχ″ :: CF  # Perturbation thermodynamic variable

    # Vertical tridiagonal solver for implicit w-ρ coupling (nothing when explicit)
    vertical_solver :: TS

    # Right-hand side storage for tridiagonal solve (nothing when explicit)
    rhs :: RHS
end

Adapt.adapt_structure(to, a::AcousticSubstepper) =
    AcousticSubstepper(a.substeps,
                       a.time_discretization,
                       a.divergence_damping_coefficient,
                       adapt(to, a.pressure_coefficient),
                       adapt(to, a.sound_speed_squared),
                       map(f -> adapt(to, f), a.averaged_velocities),
                       map(f -> adapt(to, f), a.slow_momentum_tendencies),
                       adapt(to, a.Gˢρ),
                       adapt(to, a.Gˢρχ),
                       adapt(to, a.ρᵣ),
                       adapt(to, a.ρχᵣ),
                       map(f -> adapt(to, f), a.perturbation_momentum),
                       adapt(to, a.ρ″),
                       adapt(to, a.ρχ″),
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
    pressure_coefficient = CenterField(grid)
    sound_speed_squared = CenterField(grid)

    # Time-averaged velocities (for scalar advection)
    averaged_velocities = (u = XFaceField(grid),
                           v = YFaceField(grid),
                           w = ZFaceField(grid))

    # Slow momentum tendencies
    slow_momentum_tendencies = (ρu = XFaceField(grid),
                                ρv = YFaceField(grid),
                                ρw = ZFaceField(grid))

    # Slow density tendency
    Gˢρ = CenterField(grid)

    # Slow thermodynamic tendency
    Gˢρχ = CenterField(grid)

    # Stage-frozen reference state
    ρᵣ = CenterField(grid)
    ρχᵣ = CenterField(grid)

    # Perturbation fields (zeroed at start of each RK stage)
    perturbation_momentum = (ρu = XFaceField(grid),
                             ρv = YFaceField(grid),
                             ρw = ZFaceField(grid))
    ρ″ = CenterField(grid)
    ρχ″ = CenterField(grid)

    # Vertical tridiagonal solver (only allocated for implicit vertical stepping)
    vertical_solver = build_acoustic_vertical_solver(grid, time_discretization)
    rhs = build_acoustic_vertical_rhs(grid, time_discretization)

    return AcousticSubstepper(Ns, time_discretization, divergence_damping_coefficient,
                              pressure_coefficient, sound_speed_squared,
                              averaged_velocities,
                              slow_momentum_tendencies,
                              Gˢρ, Gˢρχ,
                              ρᵣ, ρχᵣ,
                              perturbation_momentum, ρ″, ρχ″,
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
##### Prepare acoustic cache (once per RK stage)
#####

using Breeze.AtmosphereModels: thermodynamic_density

"""
$(TYPEDSIGNATURES)

Prepare the acoustic cache for an RK stage by storing the stage-frozen
reference state and computing linearized EOS coefficients.

This function:
1. Stores the stage-frozen reference density ρᵣ and thermodynamic variable ρχᵣ
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
    parent(substepper.ρχᵣ) .= parent(χ)

    # Compute thermodynamic coefficients: ψ = Rᵐ T, c² = γᵐ ψ
    launch!(arch, grid, :xyz, _compute_acoustic_coefficients!,
            substepper.pressure_coefficient, substepper.sound_speed_squared,
            model.dynamics.density,
            model.specific_moisture,
            model.temperature,
            grid,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    return nothing
end

@kernel function _compute_acoustic_coefficients!(pressure_coefficient, sound_speed_squared,
                                                  ρ, qᵗ, T,
                                                  grid, microphysics, microphysical_fields, constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρⁱ = ρ[i, j, k]
        qᵗⁱ = qᵗ[i, j, k]
        Tⁱ = T[i, j, k]
    end

    # Compute moisture fractions
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρⁱ, qᵗⁱ, microphysical_fields)

    # Mixture thermodynamic properties
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    cᵛᵐ = cᵖᵐ - Rᵐ
    γᵐ = cᵖᵐ / cᵛᵐ

    # Pressure coefficient ψ = Rᵐ T (so p = ψ ρ)
    ψⁱ = Rᵐ * Tⁱ

    @inbounds begin
        pressure_coefficient[i, j, k] = ψⁱ
        # Acoustic sound speed squared: cᵃᶜ² = γᵐ ψ = γᵐ Rᵐ T
        sound_speed_squared[i, j, k] = γᵐ * ψⁱ
    end
end

#####
##### Horizontal momentum update (explicit, on-the-fly pressure gradient)
#####

#####
##### Combined forward kernel: all momentum perturbation updates
#####

function acoustic_forward_step!(substepper, Δτ, g)
    grid = substepper.pressure_coefficient.grid
    arch = architecture(grid)
    m″ = substepper.perturbation_momentum
    Gˢm = substepper.slow_momentum_tendencies

    launch!(arch, grid, :xyz, _acoustic_forward_step!,
            m″.ρu, m″.ρv, m″.ρw,
            grid, Δτ, g,
            substepper.ρ″, substepper.pressure_coefficient,
            Gˢm.ρu, Gˢm.ρv, Gˢm.ρw)

    return nothing
end

@kernel function _acoustic_forward_step!(ρu″, ρv″, ρw″, grid, Δτ, g,
                                         ρ″, pressure_coefficient, Gˢρu, Gˢρv, Gˢρw)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # x-momentum: topology-aware pressure gradient and interpolation
        # Skip boundary face i=1 for bounded x topology (i=Nx+1 is outside :xyz range)
        ψᶠᶜᶜ = ℑxTᶠᵃᵃ(i, j, k, grid, pressure_coefficient)
        ∂x_p″ = ψᶠᶜᶜ * δxTᶠᵃᵃ(i, j, k, grid, ρ″) / Δxᶠᶜᶜ(i, j, k, grid)
        ρu″[i, j, k] += Δτ * (Gˢρu[i, j, k] - ∂x_p″) * !on_x_boundary(i, j, k, grid)

        # y-momentum: topology-aware pressure gradient and interpolation
        # Skip boundary face j=1 for bounded y topology (j=Ny+1 is outside :xyz range)
        ψᶜᶠᶜ = ℑyTᵃᶠᵃ(i, j, k, grid, pressure_coefficient)
        ∂y_p″ = ψᶜᶠᶜ * δyTᵃᶠᵃ(i, j, k, grid, ρ″) / Δyᶜᶠᶜ(i, j, k, grid)
        ρv″[i, j, k] += Δτ * (Gˢρv[i, j, k] - ∂y_p″) * !on_y_boundary(i, j, k, grid)

        # z-momentum: topology-aware pressure gradient and interpolation
        ψᶜᶜᶠ = ℑzTᵃᵃᶠ(i, j, k, grid, pressure_coefficient)
        ∂z_p″ = ψᶜᶜᶠ * δzTᵃᵃᶜ(i, j, k, grid, ρ″) / Δzᶜᶜᶜ(i, j, k, grid)
        ρ″ᶠ = ℑzTᵃᵃᶠ(i, j, k, grid, ρ″)
        Δρw″ = Δτ * (Gˢρw[i, j, k] - ∂z_p″ - g * ρ″ᶠ)

        # skip bottom boundary face k=1 (w=0 there;
        # top face k=Nz+1 is outside the kernel range since launch is :xyz over centers)
        ρw″[i, j, k] += Δρw″ * (k > 1)
    end
end

#####
##### Horizontal-only momentum update (for implicit vertical time_discretization)
#####

"""
$(TYPEDSIGNATURES)

Update only horizontal momentum perturbations during an acoustic substep.
Used by the vertically implicit path, where vertical momentum (ρw″) is
computed by the implicit tridiagonal solve rather than the explicit forward step.
"""
function acoustic_horizontal_forward_step!(substepper, Δτ)
    grid = substepper.pressure_coefficient.grid
    arch = architecture(grid)
    m″ = substepper.perturbation_momentum
    Gˢm = substepper.slow_momentum_tendencies

    launch!(arch, grid, :xyz, _acoustic_horizontal_forward_step!,
            m″.ρu, m″.ρv,
            grid, Δτ,
            substepper.ρ″, substepper.pressure_coefficient,
            Gˢm.ρu, Gˢm.ρv)

    return nothing
end

@kernel function _acoustic_horizontal_forward_step!(ρu″, ρv″, grid, Δτ,
                                                     ρ″, pressure_coefficient, Gˢρu, Gˢρv)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # x-momentum: topology-aware pressure gradient and interpolation
        ψᶠᶜᶜ = ℑxTᶠᵃᵃ(i, j, k, grid, pressure_coefficient)
        ∂x_p″ = ψᶠᶜᶜ * δxTᶠᵃᵃ(i, j, k, grid, ρ″) / Δxᶠᶜᶜ(i, j, k, grid)
        ρu″[i, j, k] += Δτ * (Gˢρu[i, j, k] - ∂x_p″) * !on_x_boundary(i, j, k, grid)

        # y-momentum: topology-aware pressure gradient and interpolation
        ψᶜᶠᶜ = ℑyTᵃᶠᵃ(i, j, k, grid, pressure_coefficient)
        ∂y_p″ = ψᶜᶠᶜ * δyTᵃᶠᵃ(i, j, k, grid, ρ″) / Δyᶜᶠᶜ(i, j, k, grid)
        ρv″[i, j, k] += Δτ * (Gˢρv[i, j, k] - ∂y_p″) * !on_y_boundary(i, j, k, grid)
    end
end

#####
##### Thermodynamic variable update (backward step)
#####

"""
$(TYPEDSIGNATURES)

Update the thermodynamic variable χ during an acoustic substep.

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
    grid = substepper.pressure_coefficient.grid
    arch = architecture(grid)
    averaging_weight = 1 / Nτ

    launch!(arch, grid, :xyz, _acoustic_backward_step!,
            substepper.ρ″, substepper.ρχ″,
            substepper.averaged_velocities,
            grid, Δτ, averaging_weight, substepper.divergence_damping_coefficient,
            substepper.perturbation_momentum,
            substepper.ρχᵣ, substepper.ρᵣ,
            model.momentum, model.dynamics.density,
            substepper.Gˢρ, substepper.Gˢρχ)

    return nothing
end

@kernel function _acoustic_backward_step!(ρ″, ρχ″, ū,
                                          grid, Δτ, averaging_weight, κᵈ,
                                          m″, ρχᵣ, ρᵣ,
                                          m, ρ,
                                          Gˢρ, Gˢρχ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Topology-aware perturbation momentum divergence
        Vⁱ = Vᶜᶜᶜ(i, j, k, grid)
        div_m″ = (δxTᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, m″.ρu) +
                  δyTᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, m″.ρv) +
                  δzTᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, m″.ρw)) / Vⁱ

        # --- Density perturbation update ---
        ρ″[i, j, k] += Δτ * (Gˢρ[i, j, k] - div_m″)
        ρ″[i, j, k] *= (1 - κᵈ)

        # --- Thermodynamic perturbation update ---
        s̄ = ρχᵣ[i, j, k] / ρᵣ[i, j, k]
        ρχ″[i, j, k] += Δτ * (Gˢρχ[i, j, k] - s̄ * div_m″)

        # --- Accumulate time-averaged velocities ---
        # Topology-aware interpolation for perturbation density (no halos filled)
        # Base density ρ has valid halos from update_state!
        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ) + ℑxTᶠᵃᵃ(i, j, k, grid, ρ″)
        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ) + ℑyTᵃᶠᵃ(i, j, k, grid, ρ″)
        ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ) + ℑzTᵃᵃᶠ(i, j, k, grid, ρ″)

        ū.u[i, j, k] += averaging_weight * (m.ρu[i, j, k] + m″.ρu[i, j, k]) / ρᶠᶜᶜ
        ū.v[i, j, k] += averaging_weight * (m.ρv[i, j, k] + m″.ρv[i, j, k]) / ρᶜᶠᶜ
        ū.w[i, j, k] += averaging_weight * (m.ρw[i, j, k] + m″.ρw[i, j, k]) / ρᶜᶜᶠ
    end
end

#####
##### Horizontal-only density update (for implicit vertical time_discretization)
#####

##### (Operators Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Vᶜᶜᶜ etc. imported above)

function acoustic_density_horizontal_step!(substepper, Δτ)
    grid = substepper.pressure_coefficient.grid
    arch = architecture(grid)
    m″ = substepper.perturbation_momentum

    launch!(arch, grid, :xyz, _acoustic_density_horizontal_step!,
            substepper.ρ″, grid, Δτ,
            m″.ρu, m″.ρv,
            substepper.divergence_damping_coefficient, substepper.Gˢρ)

    return nothing
end

@kernel function _acoustic_density_horizontal_step!(ρ″, grid, Δτ, ρu″, ρv″, κᵈ, Gˢρ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Vⁱ = Vᶜᶜᶜ(i, j, k, grid)

        # Horizontal momentum divergence perturbation: ∂(ρu″)/∂x + ∂(ρv″)/∂y
        div_ρuₕ″ = (δxTᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, ρu″) +
                    δyTᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, ρv″)) / Vⁱ

        ρ″[i, j, k] += Δτ * (Gˢρ[i, j, k] - div_ρuₕ″)
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
    grid = substepper.pressure_coefficient.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _compute_implicit_vertical_coefficients!,
            solver.a, solver.b, solver.c,
            grid, α², Δτ, substepper.pressure_coefficient)

    return nothing
end

@kernel function _compute_implicit_vertical_coefficients!(lower, diag, upper,
                                                          grid, α², Δτ, pressure_coefficient)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Grid spacings
        Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)  # Face spacing (between centers k-1 and k)

        # Coupling coefficient: α² Δτ² ψ_face / Δz_face
        ψᶠ = ℑzᵃᵃᶠ(i, j, k, grid, pressure_coefficient)
        Q = α² * Δτ^2 * ψᶠ / Δzᶠ

        not_boundary = (k > 1)

        # Interior: coupling to ρw at k-1, k, k+1
        Δzᶜᵏ = Δzᶜᶜᶜ(i, j, k, grid)     # Center spacing at k (between faces k and k+1)
        Δzᶜ⁻ = Δzᶜᶜᶜ(i, j, k - 1, grid)  # Center spacing at k-1

        # Lower diagonal: coupling to (ρw)_{k-1} via ρ at center k-1
        lower[i, j, k] = -Q / Δzᶜ⁻ * not_boundary

        # Upper diagonal: coupling to (ρw)_{k+1} via ρ at center k
        upper[i, j, k] = -Q / Δzᶜᵏ * not_boundary

        # Diagonal: self-coupling
        diag[i, j, k] = 1 + Q * (1 / Δzᶜᵏ + 1 / Δzᶜ⁻) * not_boundary
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
    grid = substepper.pressure_coefficient.grid
    arch = architecture(grid)
    rhs = substepper.rhs
    m″ = substepper.perturbation_momentum
    Gˢm = substepper.slow_momentum_tendencies

    # Build the RHS of the tridiagonal system using perturbation fields
    launch!(arch, grid, :xyz, _compute_implicit_vertical_rhs!,
            rhs, grid, Δτ, g,
            m″.ρw, substepper.ρ″, substepper.pressure_coefficient, Gˢm.ρw)

    # Solve the tridiagonal system: A ρw″^{n+1} = rhs
    solve!(m″.ρw, substepper.vertical_solver, rhs)

    # Enforce w=0 at boundaries and update perturbation density
    # with vertical perturbation momentum divergence
    launch!(arch, grid, :xyz, _acoustic_density_vertical_step!,
            substepper.ρ″, grid, Δτ, m″.ρw)

    return nothing
end

@kernel function _compute_implicit_vertical_rhs!(rhs, grid, Δτ, g,
                                                 ρw″, ρ″, pressure_coefficient, Gˢρw)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    not_boundary = (k > 1)
    @inbounds begin
        # Perturbation pressure gradient: ψ ∂ρ″/∂z
        ψᶠ = ℑzᵃᵃᶠ(i, j, k, grid, pressure_coefficient)
        ∂z_p″ = ψᶠ * ∂zᶜᶜᶠ(i, j, k, grid, ρ″)

        # Perturbation buoyancy: -g ρ″
        ρ″ᶠ = ℑzTᵃᵃᶠ(i, j, k, grid, ρ″)

        # RHS = current ρw″ + Δτ * (G_slow - ∂p″/∂z - g ρ″)
        # Zero at bottom boundary (k=1) to enforce w=0
        rhs_ijk = ρw″[i, j, k] + Δτ * (Gˢρw[i, j, k] - ∂z_p″ - g * ρ″ᶠ)
        rhs[i, j, k] = rhs_ijk * not_boundary
    end
end

@kernel function _acoustic_density_vertical_step!(ρ″, grid, Δτ, ρw″)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Enforce w=0 at top boundary (face Nz+1 is outside the solver's range).
        # Only the thread at k=Nz reads this face (via δz), so no race condition.
        if k == Nz
            ρw″[i, j, Nz + 1] = zero(eltype(ρw″))
        end

        Vⁱ = Vᶜᶜᶜ(i, j, k, grid)
        div_ρw″_z = δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, ρw″) / Vⁱ
        ρ″[i, j, k] -= Δτ * div_ρw″_z
    end
end

##### (Velocity averaging is combined into _acoustic_backward_step!)
##### (update_velocities_from_momentum! removed -- velocities are computed
#####  from model fields during update_state!)

@kernel function _zero_acoustic_fields!(m″, ρ″, ρχ″, ū)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        m″.ρu[i, j, k] = 0
        m″.ρv[i, j, k] = 0
        m″.ρw[i, j, k] = 0
        ρ″[i, j, k] = 0
        ρχ″[i, j, k] = 0
        ū.u[i, j, k] = 0
        ū.v[i, j, k] = 0
        ū.w[i, j, k] = 0
    end
end

#####
##### Main acoustic substep loop
#####

"""
    acoustic_substep_loop!(model, substepper, Δt, α, U⁰)

Execute the acoustic substep loop for an SSP RK3 stage with weight `α`.

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

The acoustic substep size `Δτ = Δt / Ns` is constant across all RK stages to maintain
stability. The number of substeps varies with the SSP RK3 stage weight `α`:
- Stage 1 (α=1): Ns substeps
- Stage 2 (α=1/4): Ns/4 substeps
- Stage 3 (α=2/3): 2Ns/3 substeps

After the acoustic loop, the SSP RK3 convex combination is applied:
``U_{new} = α (U + U'') + (1 - α) U⁰``

Arguments
=========

- `model`: The `AtmosphereModel`
- `substepper`: The `AcousticSubstepper` containing storage and parameters
- `Δt`: Full time step (not scaled by α)
- `α`: SSP RK3 stage weight (1, 1/4, or 2/3)
- `U⁰`: Initial state at beginning of time step (for SSP RK3 convex combination)
"""
function acoustic_substep_loop!(model, substepper, Δt, α, U⁰)
    Ns = substepper.substeps
    g = model.thermodynamic_constants.gravitational_acceleration

    # Constant acoustic substep size (same for all RK stages)
    Δτ = Δt / Ns

    # All stages use Ns substeps to compute the full Δt tendency.
    # The SSP RK3 formula applies the α weight to (U + Δt*L), not to the tendency.
    Nτ = Ns

    # === PRECOMPUTE PHASE (once per RK stage) ===
    # Note: prepare_acoustic_cache! is called before this function
    # (in acoustic_ssp_rk3_substep!) because slow tendencies need
    # the stage-frozen reference state.

    # Compute implicit vertical coefficients (once per stage, if implicit)
    compute_implicit_vertical_coefficients!(substepper, Δτ, substepper.time_discretization)

    # Initialize perturbation fields and time-averaged velocities to zero
    m″ = substepper.perturbation_momentum
    ū = substepper.averaged_velocities
    launch!(architecture(model.grid), model.grid, :xyz, _zero_acoustic_fields!,
            m″, substepper.ρ″, substepper.ρχ″, ū)

    # === ACOUSTIC SUBSTEP LOOP (advances perturbation variables) ===
    # Dispatch to optimized explicit or implicit loop
    acoustic_substep_loop!(model, substepper, Δτ, g, Nτ, substepper.time_discretization)

    # === RECOVERY: apply SSP RK3 convex combination ===
    # U_new = α * (U + U'') + (1 - α) * U⁰
    recover_full_fields_ssp!(model, substepper, α, U⁰)

    return nothing
end

#####
##### Optimized explicit acoustic substep loop
#####

"""
Optimized acoustic substep loop for explicit vertical stepping.

Pre-configures kernels and pre-converts arguments to minimize overhead
during the substep loop, following the pattern from Oceananigans'
split-explicit free surface solver.
"""
function acoustic_substep_loop!(model, substepper, Δτ, g, Nτ, ::Nothing)
    grid = substepper.pressure_coefficient.grid
    arch = architecture(grid)
    averaging_weight = 1 / Nτ  # velocity averaging weight
    κᵈ = substepper.divergence_damping_coefficient
    m″ = substepper.perturbation_momentum
    Gˢm = substepper.slow_momentum_tendencies
    ū = substepper.averaged_velocities

    # Pre-configure kernels (once per RK stage)
    forward_kernel!, _ = configure_kernel(arch, grid, :xyz, _acoustic_forward_step!)
    backward_kernel!, _ = configure_kernel(arch, grid, :xyz, _acoustic_backward_step!)

    # Pack kernel arguments (order must match kernel signatures)
    forward_args = (m″.ρu, m″.ρv, m″.ρw,
                    grid, Δτ, g,
                    substepper.ρ″, substepper.pressure_coefficient,
                    Gˢm.ρu, Gˢm.ρv, Gˢm.ρw)

    backward_args = (substepper.ρ″, substepper.ρχ″,
                     substepper.averaged_velocities,
                     grid, Δτ, averaging_weight, κᵈ,
                     substepper.perturbation_momentum,
                     substepper.ρχᵣ, substepper.ρᵣ,
                     model.momentum, model.dynamics.density,
                     substepper.Gˢρ, substepper.Gˢρχ)

    # Pre-convert arguments to device-compatible format (important for GPU)
    # GC.@preserve prevents garbage collection during the substep loop
    GC.@preserve forward_args backward_args begin
        converted_forward_args = convert_to_device(arch, forward_args)
        converted_backward_args = convert_to_device(arch, backward_args)

        # Execute substep loop with pre-converted arguments
        for n = 1:Nτ
            forward_kernel!(converted_forward_args...)
            backward_kernel!(converted_backward_args...)
        end
    end

    return nothing
end

#####
##### Implicit acoustic substep loop
#####

"""
Acoustic substep loop for vertically implicit stepping.
Not yet optimized with pre-configured kernels (the tridiagonal solve
dominates the cost, so kernel launch overhead is less significant).
"""
function acoustic_substep_loop!(model, substepper, Δτ, g, Nτ, ::VerticallyImplicit)
    for n = 1:Nτ
        acoustic_substep!(model, substepper, Δτ, g, Nτ, substepper.time_discretization)
    end
    return nothing
end

"""
Recover full fields from perturbation variables with SSP RK3 convex combination.

Applies the SSP RK3 update formula:
``U_{new} = α (U + U'') + (1 - α) U⁰``

This correctly implements the SSP RK3 scheme where each stage blends
the updated state with the initial state.
"""
function recover_full_fields_ssp!(model, substepper, α, U⁰)
    grid = model.grid
    arch = architecture(grid)
    ρχ = thermodynamic_density(model.formulation)

    # prognostic_fields returns: (ρ, ρu, ρv, ρw, ρθ, ρqᵗ, ...)
    # For CompressibleDynamics: ρ=U⁰[1], ρu=U⁰[2], ρv=U⁰[3], ρw=U⁰[4], ρχ(ρθ)=U⁰[5]
    m⁰ = (ρu = U⁰[2], ρv = U⁰[3], ρw = U⁰[4])

    launch!(arch, grid, :xyz, _recover_full_fields_ssp!,
            model.momentum, model.dynamics.density, ρχ,
            substepper.perturbation_momentum, substepper.ρ″, substepper.ρχ″,
            m⁰, U⁰[1], U⁰[5], α)

    return nothing
end

@kernel function _recover_full_fields_ssp!(m, ρ, ρχ, m″, ρ″, ρχ″, m⁰, ρ⁰, ρχ⁰, α)
    i, j, k = @index(Global, NTuple)

    # SSP RK3 convex combination: U_new = α * (U + U'') + (1 - α) * U⁰
    @inbounds begin
        m.ρu[i, j, k] = α * (m.ρu[i, j, k] + m″.ρu[i, j, k]) + (1 - α) * m⁰.ρu[i, j, k]
        m.ρv[i, j, k] = α * (m.ρv[i, j, k] + m″.ρv[i, j, k]) + (1 - α) * m⁰.ρv[i, j, k]
        m.ρw[i, j, k] = α * (m.ρw[i, j, k] + m″.ρw[i, j, k]) + (1 - α) * m⁰.ρw[i, j, k]
        ρ[i, j, k]    = α * (ρ[i, j, k]    + ρ″[i, j, k])     + (1 - α) * ρ⁰[i, j, k]
        ρχ[i, j, k]   = α * (ρχ[i, j, k]   + ρχ″[i, j, k])    + (1 - α) * ρχ⁰[i, j, k]
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

    # Backward: update ρ″, ρχ″, and accumulate ū, v̄, w̄
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
    # (A) Forward: update only horizontal perturbation momentum (ρu″, ρv″)
    # Vertical momentum (ρw″) is computed by the implicit solve in step (C)
    acoustic_horizontal_forward_step!(substepper, Δτ)

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

#####
##### Wicker-Skamarock RK3 recovery (no convex combination)
#####

"""
Recover full fields from perturbation variables (Wicker-Skamarock RK3).

Each RK stage resets to the initial state ``U⁰`` and adds the total
perturbation accumulated during the acoustic substep loop:

``U_{new} = U⁰ + U''``

Unlike [`recover_full_fields_ssp!`](@ref) there is no convex combination.
"""
function recover_full_fields!(model, substepper, U⁰)
    grid = model.grid
    arch = architecture(grid)
    ρχ = thermodynamic_density(model.formulation)

    # prognostic_fields returns: (ρ, ρu, ρv, ρw, ρθ, ρqᵗ, ...)
    m⁰ = (ρu = U⁰[2], ρv = U⁰[3], ρw = U⁰[4])

    launch!(arch, grid, :xyz, _recover_full_fields!,
            model.momentum, model.dynamics.density, ρχ,
            substepper.perturbation_momentum, substepper.ρ″, substepper.ρχ″,
            m⁰, U⁰[1], U⁰[5])

    return nothing
end

@kernel function _recover_full_fields!(m, ρ, ρχ, m″, ρ″, ρχ″, m⁰, ρ⁰, ρχ⁰)
    i, j, k = @index(Global, NTuple)

    # Wicker-Skamarock RK3: U_new = U⁰ + U''
    @inbounds begin
        m.ρu[i, j, k] = m⁰.ρu[i, j, k] + m″.ρu[i, j, k]
        m.ρv[i, j, k] = m⁰.ρv[i, j, k] + m″.ρv[i, j, k]
        m.ρw[i, j, k] = m⁰.ρw[i, j, k] + m″.ρw[i, j, k]
        ρ[i, j, k]    = ρ⁰[i, j, k]    + ρ″[i, j, k]
        ρχ[i, j, k]   = ρχ⁰[i, j, k]   + ρχ″[i, j, k]
    end
end

#####
##### Wicker-Skamarock RK3 acoustic substep loop
#####

"""
    acoustic_rk3_substep_loop!(model, substepper, Δt_stage, U⁰)

Execute the acoustic substep loop for a Wicker-Skamarock RK3 stage.

The effective stage timestep `Δt_stage = β * Δt` determines the acoustic
substep size `Δτ = Δt_stage / Ns`. Each stage uses `Ns` substeps but with
different `Δτ`:
- Stage 1 (β=1/3): `Δτ = Δt / (3 Ns)`
- Stage 2 (β=1/2): `Δτ = Δt / (2 Ns)`
- Stage 3 (β=1):   `Δτ = Δt / Ns`

After the acoustic loop, recovery applies `U = U⁰ + U''` (no convex combination).
"""
function acoustic_rk3_substep_loop!(model, substepper, Δt_stage, U⁰)
    Ns = substepper.substeps
    g = model.thermodynamic_constants.gravitational_acceleration

    # Acoustic substep size varies per stage (Δτ = β * Δt / Ns)
    Δτ = Δt_stage / Ns
    Nτ = Ns

    # Compute implicit vertical coefficients (depend on Δτ, so recomputed per stage)
    compute_implicit_vertical_coefficients!(substepper, Δτ, substepper.time_discretization)

    # Initialize perturbation fields and time-averaged velocities to zero
    m″ = substepper.perturbation_momentum
    ū = substepper.averaged_velocities
    launch!(architecture(model.grid), model.grid, :xyz, _zero_acoustic_fields!,
            m″, substepper.ρ″, substepper.ρχ″, ū)

    # Inner acoustic substep loop (reuses explicit/implicit dispatch)
    acoustic_substep_loop!(model, substepper, Δτ, g, Nτ, substepper.time_discretization)

    # Recovery: U = U⁰ + perturbation (no convex combination)
    recover_full_fields!(model, substepper, U⁰)

    return nothing
end

#####
##### Base-state pressure correction for slow momentum tendencies
#####

"""
    add_base_state_pressure_correction!(substepper, model)

Add the temperature-driven horizontal pressure gradient to slow momentum tendencies.

The acoustic substep computes perturbation pressure gradients `-ψ ∂ρ″/∂x` where
ρ″ starts at zero each RK stage and ψ is frozen. This only captures pressure
changes from density variations *during* the acoustic loop. The full horizontal
pressure gradient decomposes as `∂p/∂x = ψ*∂ρ/∂x + ρ*∂ψ/∂x`, and the
temperature-driven term `ρ*∂ψ/∂x` is missing from the acoustic loop.

This correction adds the temperature-driven pressure gradient to the slow
momentum tendencies:

```math
Gˢ_{ρu} -= ρ \\, ∂ψ/∂x, \\quad Gˢ_{ρv} -= ρ \\, ∂ψ/∂y
```

where `ψ = Rᵐ T` is the pressure coefficient (so `p = ψ ρ`).

The vertical pressure gradient and buoyancy are handled entirely by the acoustic
forward step through `ψ*∂ρ″/∂z` and `g*ρ″`.

!!! note "Stability constraint"
    The divergence damping must satisfy `(1 - κᵈ)^Ns < 0.1` for stability,
    where `κᵈ` is the divergence damping coefficient and `Ns` is the number
    of acoustic substeps.

!!! note "Vertically implicit substepping"
    This correction requires `VerticallyImplicit` acoustic substepping for
    stability. With explicit vertical substepping, the horizontal correction
    drives vertical motion that the explicit vertical step cannot stabilize.
"""
add_base_state_pressure_correction!(substepper, model) =
    add_base_state_pressure_correction!(substepper, model, model.dynamics.reference_state)

# No-op when no reference state is provided
add_base_state_pressure_correction!(substepper, model, ::Nothing) = nothing

function add_base_state_pressure_correction!(substepper, model, ref::ReferenceState)
    grid = model.grid
    arch = architecture(grid)
    g = model.thermodynamic_constants.gravitational_acceleration
    Gˢm = substepper.slow_momentum_tendencies

    launch!(arch, grid, :xyz, _add_base_state_pressure_correction!,
            Gˢm.ρu, Gˢm.ρv, Gˢm.ρw,
            grid, g,
            model.dynamics.density,
            ref.density,
            ref.pressure,
            substepper.pressure_coefficient)

    return nothing
end

@inline perturbation_pressureᶜᶜᶜ(i, j, k, grid, ψ, ρ, p̄) =
    @inbounds ψ[i, j, k] * ρ[i, j, k] - p̄[i, j, k]

@inline perturbation_densityᶜᶜᶜ(i, j, k, grid, ρ, ρ̄) =
    @inbounds ρ[i, j, k] - ρ̄[i, j, k]

@kernel function _add_base_state_pressure_correction!(Gˢρu, Gˢρv, Gˢρw,
                                                       grid, g, ρ, ρ̄, p̄,
                                                       pressure_coefficient)
    i, j, k = @index(Global, NTuple)
    ψ = pressure_coefficient

    @inbounds begin
        # --- Horizontal: -ρ * ∂ψ/∂x (temperature-driven pressure gradient) ---
        # The full horizontal pressure gradient decomposes as:
        #   ∂p/∂x = ψ*∂ρ/∂x + ρ*∂ψ/∂x
        #
        # The density-driven term ψ*∂ρ/∂x evolves on the acoustic timescale.
        # The acoustic forward step handles density changes through ψ*∂ρ″/∂x.
        # Only the temperature-driven term ρ*∂ψ/∂x belongs in the slow tendency.
        ρᶠᶜᶜ = ℑxTᶠᵃᵃ(i, j, k, grid, ρ)
        ∂x_ψ = δxTᶠᵃᵃ(i, j, k, grid, pressure_coefficient) / Δxᶠᶜᶜ(i, j, k, grid)
        Gˢρu[i, j, k] -= ρᶠᶜᶜ * ∂x_ψ * !on_x_boundary(i, j, k, grid)

        ρᶜᶠᶜ = ℑyTᵃᶠᵃ(i, j, k, grid, ρ)
        ∂y_ψ = δyTᵃᶠᵃ(i, j, k, grid, pressure_coefficient) / Δyᶜᶠᶜ(i, j, k, grid)
        Gˢρv[i, j, k] -= ρᶜᶠᶜ * ∂y_ψ * !on_y_boundary(i, j, k, grid)

        # No vertical correction — handled by acoustic forward step through
        # ψ*∂ρ″/∂z and g*ρ″ (perturbations relative to the stage-start state).
    end
end
