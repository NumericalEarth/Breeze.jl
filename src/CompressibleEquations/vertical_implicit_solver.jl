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
- `mean_pressure`: p̄(z) horizontal-mean pressure, updated each stage
- `mean_density`: ρ̄(z) horizontal-mean density, updated each stage
- `vertical_solver`: `BatchedTridiagonalSolver` for Nₓ × Nᵧ independent column solves
"""
struct VerticalAcousticSolver{F, S}
    acoustic_speed_squared  :: F
    rhs                     :: F
    ρθ_scratch              :: F
    mean_pressure           :: F
    mean_density            :: F
    vertical_solver         :: S
end

function VerticalAcousticSolver(grid)
    FT = eltype(grid)
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)

    acoustic_speed_squared = CenterField(grid)
    rhs = CenterField(grid)
    ρθ_scratch = CenterField(grid)
    mean_pressure = CenterField(grid)
    mean_density = CenterField(grid)

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

    return VerticalAcousticSolver(acoustic_speed_squared, rhs, ρθ_scratch,
                                  mean_pressure, mean_density, vertical_solver)
end

Adapt.adapt_structure(to, s::VerticalAcousticSolver) =
    VerticalAcousticSolver(adapt(to, s.acoustic_speed_squared),
                           adapt(to, s.rhs),
                           adapt(to, s.ρθ_scratch),
                           adapt(to, s.mean_pressure),
                           adapt(to, s.mean_density),
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

#####
##### Compute discrete hydrostatic pressure from current density
#####
##### Integrates ρg downward column-by-column so that the discrete
##### finite-difference gradient exactly satisfies hydrostatic balance:
#####
#####   (pₕ[k] - pₕ[k-1]) / Δzᶠ[k] + ℑz(ρ)[k] · g = 0
#####
##### Starting from pₕ[Nz] = p[Nz] (match EOS pressure at top center).
##### The non-hydrostatic perturbation p' = p - pₕ is then small
##### everywhere, independent of horizontal temperature gradients.
#####

@kernel function _compute_discrete_hydrostatic_pressure!(pₕ, grid, p, ρ, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    ## Start from top: pₕ[Nz] = p[Nz]
    @inbounds pₕ[i, j, Nz] = p[i, j, Nz]

    ## Integrate downward: pₕ[k] = pₕ[k+1] + ℑz(ρ)[k+1] · g · Δzᶠ[k+1]
    for k in Nz-1 : -1 : 1
        @inbounds begin
            ρᶠ = (ρ[i, j, k] + ρ[i, j, k + 1]) / 2
            Δzᶠ = Δzᶜᶜᶠ(i, j, k + 1, grid)
            pₕ[i, j, k] = pₕ[i, j, k + 1] + ρᶠ * g * Δzᶠ
        end
    end
end

function compute_discrete_hydrostatic_pressure!(pₕ, grid, p, ρ, g)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _compute_discrete_hydrostatic_pressure!, pₕ, grid, p, ρ, g)
    fill_halo_regions!(pₕ)
    return nothing
end

#####
##### Compute horizontal-mean reference state (p̄, ρ̄) for the Helmholtz
#####
##### Updated every implicit stage from the current 3D fields.
##### Perturbations p-p̄ and ρ-ρ̄ have zero horizontal mean by construction,
##### so the gravity RHS has no constant-in-x bias.
#####

@kernel function _set_horizontal_mean!(mean_field, grid, field)
    i, j, k = @index(Global, NTuple)
    Nx = size(grid, 1)
    Ny = max(1, size(grid, 2))
    inv_N = 1 / (Nx * Ny)
    s = zero(eltype(grid))
    for jj in 1:Ny
        for ii in 1:Nx
            @inbounds s += field[ii, jj, k]
        end
    end
    @inbounds mean_field[i, j, k] = s * inv_N
end

function compute_mean_reference_state!(sc, grid, p, ρ)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _set_horizontal_mean!, sc.mean_pressure, grid, p)
    launch!(arch, grid, :xyz, _set_horizontal_mean!, sc.mean_density,  grid, ρ)
    fill_halo_regions!(sc.mean_pressure)
    fill_halo_regions!(sc.mean_density)
    return nothing
end

#####
##### Zero vertical PGF and buoyancy from the explicit tendency.
#####
##### In the IMEX-ARK framework, PGF+buoyancy are part of fᴵ (the implicit
##### function). They are evaluated explicitly via evaluate_implicit_tendency!
##### at stage 1 and provided by the Helmholtz solve residual at subsequent
##### stages. Keeping them in fᴱ would double-count the acoustic forcing.
#####

@inline AtmosphereModels.explicit_z_pressure_gradient(i, j, k, grid,
        ::CompressibleDynamics{<:VerticallyImplicitTimeStepping}) = zero(grid)

@inline AtmosphereModels.explicit_buoyancy_forceᶜᶜᶠ(i, j, k, grid,
        ::CompressibleDynamics{<:VerticallyImplicitTimeStepping}, args...) = zero(grid)

#####
##### Schur complement Helmholtz for δρθ (acoustic + gravity):
#####
#####   [I - τ² L - τ²g ∂/∂z] δρθ = -τ θ₀ div_z(ρw*)
#####                                 + τ² div_z(ℂ² ∂ρθ*/∂z + ρθ* g)
#####
##### L = V⁻¹ δz(Az ℂ² δ(·)/Δzᶠ)   (acoustic, symmetric second derivative)
##### τ²g ∂/∂z                        (gravity, skew-symmetric first derivative)
#####
##### The gravity ∂/∂z term comes from the Schur complement: eliminating δρ
##### from the coupled (ρw, ρθ, ρ) system using δρ = δρθ/θ₀ introduces
##### a buoyancy feedback -τ(g/θ₀)δρθ in the ρw equation. Substituting
##### into the δρθ equation gives the first-derivative gravity operator.
#####
##### The back-solve gives the FULL linearized fᴵ_ρw:
#####   δρw = -τ(ℂ²/θ)∂δρθ/∂z - τ(g/θ₀)δρθ - τ[(ℂ²/θ)∂ρθ*/∂z + ρg]
#####

@kernel function _build_ρθ_tridiagonal!(lower, diag, upper, rhs_field,
                                         grid, αΔt, ℂᵃᶜ², ρθ, θ, ρw, ρ, p, g, p̄, ρ̄)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        V = Vᶜᶜᶜ(i, j, k, grid)
        Az_bot = Azᶜᶜᶠ(i, j, k, grid)
        Az_top = Azᶜᶜᶠ(i, j, k + 1, grid)
        ℂ²_bot = ℑzᵃᵃᶠ(i, j, k, grid, ℂᵃᶜ²)
        ℂ²_top = ℑzᵃᵃᶠ(i, j, k + 1, grid, ℂᵃᶜ²)
        Δzᶠ_bot = Δzᶜᶜᶠ(i, j, k, grid)
        Δzᶠ_top = Δzᶜᶜᶠ(i, j, k + 1, grid)

        ## Acoustic Helmholtz: V⁻¹ δz(Az ℂ² δz(·)/Δzᶠ)
        αΔt² = αΔt * αΔt
        Q_bot = αΔt² * Az_bot * ℂ²_bot / (Δzᶠ_bot * V)
        Q_top = αΔt² * Az_top * ℂ²_top / (Δzᶠ_top * V)
        Q_bot = ifelse(k == 1, zero(Q_bot), Q_bot)
        Q_top = ifelse(k == Nz, zero(Q_top), Q_top)

        V_above = Vᶜᶜᶜ(i, j, k + 1, grid)
        Q_lower = αΔt² * Az_top * ℂ²_top / (Δzᶠ_top * V_above)
        Q_lower = ifelse(k >= Nz, zero(Q_lower), Q_lower)

        ## Gravity first-derivative: ±τ²g/(2Δz) from Schur complement
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)
        G_grav = αΔt² * g / (2 * Δzᶜ)

        ## Acoustic (symmetric) + gravity (skew-symmetric) tridiagonal
        lower[i, j, k] = -Q_lower + G_grav
        upper[i, j, k] = -Q_top - G_grav
        diag[i, j, k] = 1 + Q_bot + Q_top

        ## RHS term 1: acoustic flux from predictor ρw
        θᶠ_top = ℑzᵃᵃᶠ(i, j, k + 1, grid, θ)
        θᶠ_bot = ℑzᵃᵃᶠ(i, j, k, grid, θ)
        ρw_top = ρw[i, j, k + 1] * (k < Nz)
        ρw_bot = ρw[i, j, k] * (k > 1)

        flux_rhs = -αΔt / V * (Az_top * θᶠ_top * ρw_top - Az_bot * θᶠ_bot * ρw_bot)

        ## RHS term 2: gravity from perturbation ∂(p-p̄)/∂z + g(ρ-ρ̄).
        ## p̄, ρ̄ are horizontal means → perturbations have zero horizontal mean
        ## → no constant-in-x spurious forcing.
        ρᶠ_top = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ)
        ρᶠ_bot = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρ̄ᶠ_top = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ̄)
        ρ̄ᶠ_bot = ℑzᵃᵃᶠ(i, j, k, grid, ρ̄)

        pgf_top = (p[i,j,k+1] - p̄[i,j,k+1] - p[i,j,k] + p̄[i,j,k]) / Δzᶠ_top + g * (ρᶠ_top - ρ̄ᶠ_top)
        pgf_bot = (p[i,j,k] - p̄[i,j,k] - p[i,j,k-1] + p̄[i,j,k-1]) / Δzᶠ_bot + g * (ρᶠ_bot - ρ̄ᶠ_bot)
        pgf_top = ifelse(k == Nz, zero(pgf_top), pgf_top)
        pgf_bot = ifelse(k == 1, zero(pgf_bot), pgf_bot)

        grav_rhs = αΔt² / V * (Az_top * θᶠ_top * pgf_top - Az_bot * θᶠ_bot * pgf_bot)

        rhs_field[i, j, k] = flux_rhs + grav_rhs
    end
end

#####
##### Back-solve: full linearized fᴵ_ρw
#####
#####   δρw = -τ (ℂ²/θ)ᶠ ∂(δρθ)/∂z  -  τ [(ℂ²/θ)ᶠ ∂(ρθ*)/∂z + ρᶠ g]
#####
##### The first term is the acoustic perturbation PGF from the Helmholtz.
##### The second term is the mean linearized PGF+gravity at the predictor.
##### In hydrostatic balance the second term ≈ 0.
#####

@kernel function _back_solve_ρw!(ρw, grid, αΔt, ℂᵃᶜ², θ, ρθ, ρθ_scratch, ρ, p, g, p̄, ρ̄)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ℂ²ᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ℂᵃᶜ²)
        θᶠ = ℑzᵃᵃᶠ(i, j, k, grid, θ)
        Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)

        δρθ_above = ρθ[i, j, k] - ρθ_scratch[i, j, k]
        δρθ_below = ρθ[i, j, k - 1] - ρθ_scratch[i, j, k - 1]
        δρθ_face = (δρθ_above + δρθ_below) / 2

        ## 1. Perturbation PGF: -(ℂ²/θ) ∂(δρθ)/∂z
        perturbation_pgf = -ℂ²ᶠ / θᶠ * (δρθ_above - δρθ_below) / Δzᶠ

        ## 2. Gravity buoyancy from δρ = δρθ/θ₀: -(g/θ₀) δρθ
        gravity_buoyancy = -(g / θᶠ) * δρθ_face

        ## 3. Mean PGF+gravity from horizontal-mean reference: -(∂(p-p̄)/∂z + g(ρ-ρ̄))
        ρᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρ̄ᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ̄)
        mean_pgf_grav = -((p[i,j,k] - p̄[i,j,k] - p[i,j,k-1] + p̄[i,j,k-1]) / Δzᶠ + g * (ρᶠ - ρ̄ᶠ))

        ρw[i, j, k] = (ρw[i, j, k] + αΔt * (perturbation_pgf + gravity_buoyancy + mean_pgf_grav)) * (k > 1)
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

##### Update ρθ from the vertical flux divergence of the corrected ρw.
##### This ensures ρθ and ρ use the SAME ρw (post-back-solve), so
##### θ = ρθ/ρ is conserved by the implicit step.

@kernel function _update_ρθ_from_ρw!(ρθ, ρθ_scratch, grid, αΔt, θ, ρw)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        θᶠ_top = ℑzᵃᵃᶠ(i, j, k + 1, grid, θ)
        θᶠ_bot = ℑzᵃᵃᶠ(i, j, k, grid, θ)
        ρw_top = ifelse(k < Nz, ρw[i, j, k + 1], zero(eltype(grid)))
        ρw_bot = ifelse(k > 1,  ρw[i, j, k],     zero(eltype(grid)))
        Az_top = Azᶜᶜᶠ(i, j, k + 1, grid)
        Az_bot = Azᶜᶜᶠ(i, j, k, grid)
        V = Vᶜᶜᶜ(i, j, k, grid)

        ## ρθ⁺ = (ρθ)* - αΔt V⁻¹ δz(Az θ ρw)
        ρθ[i, j, k] = ρθ_scratch[i, j, k] - αΔt * (Az_top * θᶠ_top * ρw_top - Az_bot * θᶠ_bot * ρw_bot) / V
    end
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
    ρ = dynamics.density
    ℂᵃᶜ² = sc.acoustic_speed_squared

    ## β·αΔt is the effective implicit time scale.
    ## β = 1 (backward Euler): maximum acoustic damping.
    ## β = 0.5 (Crank–Nicolson): second-order, less damping.
    βαΔt = β * αΔt

    ## 1. Save (ρθ)* for computing δρθ after the solve
    launch!(arch, grid, :xyz, _copy_field!, sc.ρθ_scratch, ρθ)

    g = model.thermodynamic_constants.gravitational_acceleration
    p = dynamics.pressure

    ## 2. Compute horizontal-mean reference state (p̄, ρ̄)
    compute_mean_reference_state!(sc, grid, p, ρ)

    ## 3. Helmholtz (acoustic + gravity with mean-subtracted RHS)
    launch!(arch, grid, :xyz, _build_ρθ_tridiagonal!,
            solver.a, solver.b, solver.c, sc.rhs,
            grid, βαΔt, ℂᵃᶜ², ρθ, θ, ρw, ρ, p, g, sc.mean_pressure, sc.mean_density)

    solve!(ρθ, solver, sc.rhs)  # ρθ now holds δρθ

    ## 4. Recover (ρθ)⁺ = (ρθ)* + δρθ
    launch!(arch, grid, :xyz, _add_field!, ρθ, sc.ρθ_scratch)

    ## 5. Back-solve with mean-subtracted PGF+gravity
    launch!(arch, grid, :xyz, _back_solve_ρw!,
            ρw, grid, βαΔt, ℂᵃᶜ², θ, ρθ, sc.ρθ_scratch, ρ, p, g, sc.mean_pressure, sc.mean_density)

    ## 6. Update ρ from the vertical divergence of the corrected ρw.
    launch!(arch, grid, :xyz, _update_density_vertical!,
            ρ, grid, αΔt, ρw)

    return nothing
end
