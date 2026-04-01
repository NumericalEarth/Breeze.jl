#####
##### Vertically Implicit Acoustic Solver for CompressibleDynamics
#####
##### IMEX split: subtract vertical PGF + buoyancy + vertical ρθ flux from
##### explicit tendencies, solve implicitly after each RK stage.
#####
##### In hydrostatic balance the PGF and buoyancy corrections cancel exactly
##### (no reference state needed), leaving zero residual in the explicit step.
#####
##### Helmholtz (cell centers, for δρθ):
#####   [I - (αΔt)² ∂z(ℂᵃᶜ² ∂z)] δρθ = -αΔt ∂z(θᶠ ρw*)
#####
#####   Solving for the perturbation δρθ (not the full ρθ) ensures the
#####   hydrostatically balanced state is a fixed point (flux = 0 → δρθ = 0).
#####   Then (ρθ)⁺ = (ρθ)* + δρθ.
#####
##### Back-solve (z-faces, for ρw) using the CHANGE in ρθ only:
#####   (ρw)⁺ = (ρw)* - αΔt (ℂᵃᶜ²/θ)ᶠ ∂(δρθ)/∂z
#####
##### where δρθ = (ρθ)⁺ − (ρθ)* preserves hydrostatic balance.
#####

using KernelAbstractions: @kernel, @index

using Adapt: Adapt, adapt

using Oceananigans: CenterField, architecture
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!
using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶜ, Δzᶜᶜᶠ, Azᶜᶜᶠ, Vᶜᶜᶜ
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!

#####
##### VerticalAcousticSolver struct
#####

"""
$(TYPEDEF)

Solver for the vertically implicit acoustic correction in compressible dynamics.

Fields
======

- `acoustic_speed_squared`: ℂᵃᶜ² = γᵐ Rᵐ T at cell centers, updated each time step
- `rhs`: Scratch field for the tridiagonal right-hand side
- `ρθ_scratch`: Scratch field storing (ρθ)* before the solve for computing δρθ
- `vertical_solver`: `BatchedTridiagonalSolver` for Nₓ × Nᵧ independent column solves
"""
struct VerticalAcousticSolver{F, S}
    acoustic_speed_squared :: F
    rhs                    :: F
    ρθ_scratch             :: F
    vertical_solver        :: S
end

function VerticalAcousticSolver(grid)
    FT = eltype(grid)
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)

    acoustic_speed_squared = CenterField(grid)
    rhs = CenterField(grid)
    ρθ_scratch = CenterField(grid)

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

    return VerticalAcousticSolver(acoustic_speed_squared, rhs, ρθ_scratch, vertical_solver)
end

Adapt.adapt_structure(to, s::VerticalAcousticSolver) =
    VerticalAcousticSolver(adapt(to, s.acoustic_speed_squared),
                           adapt(to, s.rhs),
                           adapt(to, s.ρθ_scratch),
                           nothing)

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

##### IMPORTANT for IMEX-RK time steppers (ARS343, SSP332):
##### The explicit tendency fᴱ MUST exclude vertical PGF + buoyancy + vertical
##### density flux, because these are handled by the implicit tendency fᴵ.
##### Including them in both fᴱ and fᴵ double-counts and causes instability.
#####
##### For the SSP-RK3 time stepper with additive VITS correction (non-IMEX),
##### the explicit step keeps the full PGF+buoyancy and the implicit solve
##### provides a small correction. This only works at Δt ≤ Δz/cs.
#####
##### The dispatches below zero the vertical PGF+buoyancy from the explicit
##### tendency when using VITS. This is required for proper IMEX-RK.

@inline AtmosphereModels.explicit_z_pressure_gradient(i, j, k, grid,
        ::CompressibleDynamics{<:VerticallyImplicitTimeStepping}) = zero(grid)

@inline AtmosphereModels.explicit_buoyancy_forceᶜᶜᶠ(i, j, k, grid,
        ::CompressibleDynamics{<:VerticallyImplicitTimeStepping}, args...) = zero(grid)

#####
##### Helmholtz: [I - (αΔt)² ∂z(ℂᵃᶜ² ∂z)] δρθ = -αΔt ∂z(θᶠ ρw*)
#####

@kernel function _build_ρθ_tridiagonal!(lower, diag, upper, rhs_field,
                                         grid, αΔt, ℂᵃᶜ², ρθ, θ, ρw)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        ## Volume and face areas for proper metric-consistent divergence
        V = Vᶜᶜᶜ(i, j, k, grid)
        Az_bot = Azᶜᶜᶠ(i, j, k, grid)
        Az_top = Azᶜᶜᶠ(i, j, k + 1, grid)
        ℂ²_bot = ℑzᵃᵃᶠ(i, j, k, grid, ℂᵃᶜ²)
        ℂ²_top = ℑzᵃᵃᶠ(i, j, k + 1, grid, ℂᵃᶜ²)
        Δzᶠ_bot = Δzᶜᶜᶠ(i, j, k, grid)
        Δzᶠ_top = Δzᶜᶜᶠ(i, j, k + 1, grid)

        ## Helmholtz coefficients: V⁻¹ δz(Az ℂ² δz(·)/Δzᶠ)
        αΔt² = αΔt * αΔt
        Q_bot = αΔt² * Az_bot * ℂ²_bot / (Δzᶠ_bot * V)
        Q_top = αΔt² * Az_top * ℂ²_top / (Δzᶠ_top * V)

        Q_bot = ifelse(k == 1, zero(Q_bot), Q_bot)
        Q_top = ifelse(k == Nz, zero(Q_top), Q_top)

        ## Solver convention: a[k] is the sub-diagonal for row k+1.
        ## Q_bot(k+1) uses the same face as Q_top(k) but V at k+1.
        V_above = Vᶜᶜᶜ(i, j, k + 1, grid)
        Q_lower = αΔt² * Az_top * ℂ²_top / (Δzᶠ_top * V_above)
        Q_lower = ifelse(k >= Nz, zero(Q_lower), Q_lower)

        lower[i, j, k] = -Q_lower
        upper[i, j, k] = -Q_top
        diag[i, j, k] = 1 + Q_bot + Q_top

        ## RHS: V⁻¹ δz(Az θᶠ ρw) — area-weighted vertical flux divergence
        θᶠ_top = ℑzᵃᵃᶠ(i, j, k + 1, grid, θ)
        θᶠ_bot = ℑzᵃᵃᶠ(i, j, k, grid, θ)

        ## Enforce w = 0 at impenetrable boundary faces (may be non-zero
        ## after the explicit substep before BCs are re-applied).
        ρw_top = ρw[i, j, k + 1] * (k < Nz)
        ρw_bot = ρw[i, j, k] * (k > 1)

        rhs_field[i, j, k] = -αΔt / V * (Az_top * θᶠ_top * ρw_top - Az_bot * θᶠ_bot * ρw_bot)
    end
end

#####
##### Back-solve: (ρw)⁺ = (ρw)* - αΔt (ℂᵃᶜ²/θ)ᶠ ∂(δρθ)/∂z
#####
##### Uses only the CHANGE δρθ = (ρθ)⁺ − (ρθ)*, not the full (ρθ)⁺.
##### This preserves hydrostatic balance: if δρθ ≈ 0, then (ρw)⁺ ≈ (ρw)*.
#####

@kernel function _back_solve_ρw!(ρw, grid, αΔt, ℂᵃᶜ², θ, ρθ, ρθ_scratch)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ℂ²ᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ℂᵃᶜ²)
        θᶠ = ℑzᵃᵃᶠ(i, j, k, grid, θ)

        δρθ_above = ρθ[i, j, k] - ρθ_scratch[i, j, k]
        δρθ_below = ρθ[i, j, k - 1] - ρθ_scratch[i, j, k - 1]
        Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
        δz_δρθ = (δρθ_above - δρθ_below) / Δzᶠ

        ρw[i, j, k] = (ρw[i, j, k] - αΔt * ℂ²ᶠ / θᶠ * δz_δρθ) * (k > 1)
    end
end

##### Update ρ from the vertical divergence of the corrected ρw.
##### The explicit density tendency has only horizontal divergence for VITS,
##### so the vertical part ∂(ρw)/∂z is applied here after the ρw back-solve.
##### Following Klemp et al. (2018, eq. 23).

@kernel function _update_density_vertical!(ρ, grid, αΔt, ρw)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        ## Area-weighted vertical divergence: V⁻¹ δz(Az ρw)
        ## Consistent with div_xyᶜᶜᶜ so that div_xy + div_z = divᶜᶜᶜ.
        ρw_top = ifelse(k < Nz, ρw[i, j, k + 1], zero(eltype(grid)))
        ρw_bot = ifelse(k > 1,  ρw[i, j, k],     zero(eltype(grid)))
        Az_top = Azᶜᶜᶠ(i, j, k + 1, grid)
        Az_bot = Azᶜᶜᶠ(i, j, k, grid)
        V = Vᶜᶜᶜ(i, j, k, grid)

        ρ[i, j, k] = ρ[i, j, k] - αΔt * (Az_top * ρw_top - Az_bot * ρw_bot) / V
    end
end

#####
##### Save (ρθ)* before solve
#####

@kernel function _copy_field!(dst, src)
    i, j, k = @index(Global, NTuple)
    @inbounds dst[i, j, k] = src[i, j, k]
end

@kernel function _add_field!(dst, src)
    i, j, k = @index(Global, NTuple)
    @inbounds dst[i, j, k] += src[i, j, k]
end

#####
##### Entry point
#####

"""
$(TYPEDSIGNATURES)

Apply the vertically implicit acoustic correction after an explicit SSP-RK3 substep.

Following Gardner et al. (2018, GMD), the minimum HEVI implicit terms are:
1. Vertical PGF + buoyancy in ρw equation (subtracted from explicit, restored here)
2. Vertical density flux in continuity equation (corrected here via ρ update)

The implicit solve proceeds:
1. Helmholtz solve for δ(ρθ) from the vertical ρw flux
2. Update ρθ
3. Back-solve ρw from the ρθ change
4. Update ρ from the vertical divergence of the ρw change
"""
vertical_acoustic_implicit_step!(model, αΔt) =
    _vertical_acoustic_implicit_step!(model, model.dynamics, αΔt)

_vertical_acoustic_implicit_step!(model, dynamics, αΔt) = nothing

function _vertical_acoustic_implicit_step!(model,
                                           dynamics::CompressibleDynamics{<:VerticallyImplicitTimeStepping},
                                           αΔt)
    grid = model.grid
    arch = architecture(grid)
    sc = dynamics.vertical_acoustic_solver
    solver = sc.vertical_solver
    β = dynamics.time_discretization.β

    ρθ = model.formulation.potential_temperature_density
    θ = model.formulation.potential_temperature
    ρw = model.momentum.ρw
    ℂᵃᶜ² = sc.acoustic_speed_squared

    ## β·αΔt is the effective implicit time scale.
    ## β = 1 (backward Euler): maximum acoustic damping.
    ## β = 0.5 (Crank–Nicolson): second-order, less damping.
    βαΔt = β * αΔt

    ## 1. Save (ρθ)* for computing δρθ after the solve
    launch!(arch, grid, :xyz, _copy_field!, sc.ρθ_scratch, ρθ)

    ## 2. Build and solve: [I - (β αΔt)² Op] δρθ = -β αΔt div_z(θᶠ ρw*)
    launch!(arch, grid, :xyz, _build_ρθ_tridiagonal!,
            solver.a, solver.b, solver.c, sc.rhs,
            grid, βαΔt, ℂᵃᶜ², ρθ, θ, ρw)

    solve!(ρθ, solver, sc.rhs)  # ρθ now holds δρθ

    ## 3. Recover (ρθ)⁺ = (ρθ)* + δρθ
    launch!(arch, grid, :xyz, _add_field!, ρθ, sc.ρθ_scratch)

    ## 4. Back-solve: ρw⁺ = ρw* - β αΔt (ℂ²/θ)ᶠ ∂(δρθ)/∂z
    launch!(arch, grid, :xyz, _back_solve_ρw!,
            ρw, grid, βαΔt, ℂᵃᶜ², θ, ρθ, sc.ρθ_scratch)

    ## No ρ update needed: the explicit step already has the full 3D divergence.
    ## The implicit correction to ρw is captured in the next time step's divergence.

    return nothing
end
