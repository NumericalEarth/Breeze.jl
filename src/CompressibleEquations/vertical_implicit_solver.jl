#####
##### Vertically Implicit Acoustic Solver for CompressibleDynamics
#####
##### Treats vertical acoustic propagation implicitly by solving a tridiagonal
##### system for ρθ after each explicit SSP-RK3 stage, then back-solving for ρw.
#####
##### The linear vertical acoustic mode couples ρw and ρθ through:
#####   ∂(ρw)/∂t = ... - (ℂᵃᶜ²/θ) ∂(ρθ)/∂z   (linear PGF)
#####   ∂(ρθ)/∂t = ... - ∂(ρθ w)/∂z            (linear vertical flux)
#####
##### The tridiagonal system (backward Euler):
#####   [I - (αΔt)² ∂z(ℂᵃᶜ² ∂z)] (ρθ)⁺ = (ρθ)*
#####
##### Back-solve:
#####   (ρw)⁺ = (ρw)* - αΔt (ℂᵃᶜ²/θ) ∂(ρθ)⁺/∂z
#####

using KernelAbstractions: @kernel, @index

using Adapt: Adapt, adapt

using Oceananigans: CenterField, architecture
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!
using Oceananigans.Operators: ℑzᵃᵃᶠ, δzᵃᵃᶠ, Δzᶜᶜᶜ, Δzᶜᶜᶠ
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!

#####
##### VerticalAcousticSolver struct
#####

"""
$(TYPEDEF)

Solver for the vertically implicit acoustic correction in compressible dynamics.

Created during model materialization when [`VerticallyImplicitTimeStepping`](@ref)
is used. Holds workspace fields for the tridiagonal solve that couples ρw and ρθ
through the vertical acoustic mode.

Fields
======

- `acoustic_speed_squared`: ℂᵃᶜ² = γᵐ Rᵐ T at cell centers, updated each time step
- `rhs`: Scratch field for the tridiagonal right-hand side
- `vertical_solver`: `BatchedTridiagonalSolver` for Nₓ × Nᵧ independent column solves
"""
struct VerticalAcousticSolver{F, S}
    acoustic_speed_squared :: F
    rhs                    :: F
    vertical_solver        :: S
end

function VerticalAcousticSolver(grid)
    FT = eltype(grid)
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)

    acoustic_speed_squared = CenterField(grid)
    rhs = CenterField(grid)

    lower_diagonal = zeros(arch, FT, Nx, Ny, Nz)
    diagonal       = zeros(arch, FT, Nx, Ny, Nz)
    upper_diagonal = zeros(arch, FT, Nx, Ny, Nz)
    scratch        = zeros(arch, FT, Nx, Ny, Nz)

    vertical_solver = BatchedTridiagonalSolver(grid;
                                               lower_diagonal,
                                               diagonal,
                                               upper_diagonal,
                                               scratch,
                                               tridiagonal_direction = ZDirection())

    return VerticalAcousticSolver(acoustic_speed_squared, rhs, vertical_solver)
end

Adapt.adapt_structure(to, s::VerticalAcousticSolver) =
    VerticalAcousticSolver(adapt(to, s.acoustic_speed_squared),
                           adapt(to, s.rhs),
                           adapt(to, s.vertical_solver))

#####
##### Materialization dispatch
#####

materialize_vertical_acoustic_solver(::ExplicitTimeStepping, grid) = nothing
materialize_vertical_acoustic_solver(::SplitExplicitTimeDiscretization, grid) = nothing
materialize_vertical_acoustic_solver(::VerticallyImplicitTimeStepping, grid) = VerticalAcousticSolver(grid)

#####
##### Compute ℂᵃᶜ² = γᵐ Rᵐ T
#####

_compute_ℂᵃᶜ²!(model, ::Nothing) = nothing

function _compute_ℂᵃᶜ²!(model, solver::VerticalAcousticSolver)
    grid = model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _compute_acoustic_speed_squared!,
            solver.acoustic_speed_squared,
            model.temperature,
            model.dynamics.density,
            specific_prognostic_moisture(model),
            grid,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    fill_halo_regions!(solver.acoustic_speed_squared)

    return nothing
end

@kernel function _compute_acoustic_speed_squared!(ℂᵃᶜ²_field, temperature_field,
                                                   density, specific_prognostic_moisture,
                                                   grid, microphysics,
                                                   microphysical_fields, constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ = density[i, j, k]
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
        T = temperature_field[i, j, k]
    end

    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    cᵛᵐ = cᵖᵐ - Rᵐ
    γᵐ = cᵖᵐ / cᵛᵐ

    @inbounds ℂᵃᶜ²_field[i, j, k] = γᵐ * Rᵐ * T
end

#####
##### Tendency corrections — subtract linear vertical acoustic terms
#####
##### These are @inline dispatch functions that compile to zero for non-VITS dynamics.
#####

## ρw correction: +(ℂᵃᶜ²/θ)ᶠ ∂(ρθ)/∂z — cancels linear vertical PGF
## Extends the fallback defined in AtmosphereModels.dynamics_kernel_functions
@inline AtmosphereModels.vertical_acoustic_correction_ρw(i, j, k, grid, dynamics::CompressibleDynamics, formulation) =
    _vac_ρw(i, j, k, grid, dynamics.vertical_acoustic_solver, formulation)

@inline _vac_ρw(i, j, k, grid, ::Nothing, formulation) = zero(grid)

@inline function _vac_ρw(i, j, k, grid, solver::VerticalAcousticSolver, formulation)
    ρθ = formulation.potential_temperature_density
    θ = formulation.potential_temperature
    ℂᵃᶜ² = solver.acoustic_speed_squared
    ℂ²ᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ℂᵃᶜ²)
    θᶠ = ℑzᵃᵃᶠ(i, j, k, grid, θ)
    δz_ρθ = δzᵃᵃᶠ(i, j, k, grid, ρθ)
    Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
    return ℂ²ᶠ / θᶠ * δz_ρθ / Δzᶠ * (k > 1)
end

## ρθ correction: +∂(ρθ w)/∂z — cancels linear vertical advective flux
## Extends the fallback defined in AtmosphereModels.dynamics_kernel_functions
@inline AtmosphereModels.vertical_acoustic_correction_ρθ(i, j, k, grid, dynamics::CompressibleDynamics, formulation, velocities) =
    _vac_ρθ(i, j, k, grid, dynamics.vertical_acoustic_solver, formulation, velocities)

@inline _vac_ρθ(i, j, k, grid, ::Nothing, formulation, velocities) = zero(grid)

@inline function _vac_ρθ(i, j, k, grid, ::VerticalAcousticSolver, formulation, velocities)
    ρθ = formulation.potential_temperature_density
    w = velocities.w
    Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)
    @inbounds begin
        ρθᶠ_top = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρθ)
        w_top = w[i, j, k + 1]
        ρθᶠ_bot = ℑzᵃᵃᶠ(i, j, k, grid, ρθ)
        w_bot = w[i, j, k]
    end
    flux_top = ρθᶠ_top * w_top
    flux_bot = ρθᶠ_bot * w_bot
    return (flux_top - flux_bot) / Δzᶜ
end

#####
##### Tridiagonal build kernel
#####
##### Solves: [I - (αΔt)² ∂z(ℂᵃᶜ² ∂z)] (ρθ)⁺ = (ρθ)*
#####

@kernel function _build_ρθ_tridiagonal!(lower, diag, upper, rhs_field,
                                         grid, αΔt, ℂᵃᶜ², ρθ)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)

        ## ℂᵃᶜ² interpolated to bottom and top faces of cell k
        ℂ²_bot = ℑzᵃᵃᶠ(i, j, k, grid, ℂᵃᶜ²)
        ℂ²_top = ℑzᵃᵃᶠ(i, j, k + 1, grid, ℂᵃᶜ²)
        Δzᶠ_bot = Δzᶜᶜᶠ(i, j, k, grid)
        Δzᶠ_top = Δzᶜᶜᶠ(i, j, k + 1, grid)

        ## Coupling coefficients: Q = (αΔt)² ℂ² / (Δzᶠ Δzᶜ)
        αΔt² = αΔt * αΔt
        Q_bot = αΔt² * ℂ²_bot / (Δzᶠ_bot * Δzᶜ)
        Q_top = αΔt² * ℂ²_top / (Δzᶠ_top * Δzᶜ)

        ## w = 0 at boundaries: no flux coupling at k=1 bottom or k=Nz top
        Q_bot = ifelse(k == 1, zero(Q_bot), Q_bot)
        Q_top = ifelse(k == Nz, zero(Q_top), Q_top)

        lower[i, j, k] = -Q_bot
        upper[i, j, k] = -Q_top
        diag[i, j, k] = 1 + Q_bot + Q_top

        ## RHS = (ρθ)* (copy into separate field for solve!)
        rhs_field[i, j, k] = ρθ[i, j, k]
    end
end

#####
##### Back-solve kernel for ρw
#####
##### After solving for (ρθ)⁺, compute:
#####   (ρw)⁺ = (ρw)* - αΔt (ℂᵃᶜ²/θ)ᶠ ∂(ρθ)⁺/∂z
#####
##### θ and ℂᵃᶜ² are from the previous update_state! (the linearization state).
#####

@kernel function _back_solve_ρw!(ρw, grid, αΔt, ℂᵃᶜ², θ, ρθ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ℂ²ᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ℂᵃᶜ²)
        θᶠ = ℑzᵃᵃᶠ(i, j, k, grid, θ)
        δz_ρθ = δzᵃᵃᶠ(i, j, k, grid, ρθ)
        Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)

        ## (ρw)⁺ = (ρw)* - αΔt (ℂᵃᶜ²/θ)ᶠ ∂(ρθ)⁺/∂z
        ## Zero at k=1 (bottom boundary: w=0)
        ρw[i, j, k] = (ρw[i, j, k] - αΔt * ℂ²ᶠ / θᶠ * δz_ρθ / Δzᶠ) * (k > 1)
    end
end

#####
##### Entry point: vertical acoustic implicit step
#####

"""
$(TYPEDSIGNATURES)

Apply the vertically implicit acoustic correction after an explicit SSP-RK3 substep.

Dispatches on `model.dynamics.vertical_acoustic_solver`: no-op when `nothing`
(i.e., for [`ExplicitTimeStepping`](@ref) or [`SplitExplicitTimeDiscretization`](@ref)).
"""
vertical_acoustic_implicit_step!(model, αΔt) =
    _vertical_acoustic_implicit_step!(model, model.dynamics, αΔt)

_vertical_acoustic_implicit_step!(model, dynamics, αΔt) = nothing

function _vertical_acoustic_implicit_step!(model,
                                           dynamics::CompressibleDynamics{<:VerticallyImplicitTimeStepping},
                                           αΔt)
    grid = model.grid
    arch = architecture(grid)
    solver_cache = dynamics.vertical_acoustic_solver
    solver = solver_cache.vertical_solver

    ρθ = model.formulation.potential_temperature_density
    θ = model.formulation.potential_temperature
    ρw = model.momentum.ρw
    ℂᵃᶜ² = solver_cache.acoustic_speed_squared

    ## Build tridiagonal system: [I - (αΔt)² ∂z(ℂᵃᶜ² ∂z)] (ρθ)⁺ = (ρθ)*
    launch!(arch, grid, :xyz, _build_ρθ_tridiagonal!,
            solver.a, solver.b, solver.c, solver_cache.rhs,
            grid, αΔt, ℂᵃᶜ², ρθ)

    ## Solve tridiagonal system: (ρθ)⁺ overwrites the ρθ field
    solve!(ρθ, solver, solver_cache.rhs)

    ## Back-solve: (ρw)⁺ from the updated ρθ
    launch!(arch, grid, :xyz, _back_solve_ρw!,
            ρw, grid, αΔt, ℂᵃᶜ², θ, ρθ)

    return nothing
end
