#####
##### Common utilities for IMEX Additive Runge-Kutta (ARK) time steppers
#####
##### Proper IMEX-ARK methods evaluate the implicit function fᴵ at the
##### SAME stage value as fᴱ. The implicit system couples ρw ↔ ρθ through
##### the vertical acoustic mode; ρ is updated diagnostically from ρw.
#####
##### References:
#####   Ascher, Ruuth, Spiteri (1997). Implicit-explicit Runge-Kutta methods
#####     for time-dependent partial differential equations.
#####   Gardner et al. (2018, GMD). Implicit-explicit (IMEX) Runge-Kutta
#####     methods for non-hydrostatic atmospheric models.
#####

using KernelAbstractions: @kernel, @index

using Oceananigans: prognostic_fields, fields, architecture
using Oceananigans.Utils: launch!, time_difference_seconds
using Oceananigans.Operators: Azᶜᶜᶠ, Vᶜᶜᶜ
using Oceananigans.Solvers: solve!
using Oceananigans.TimeSteppers: update_state!, tick_stage!, compute_flux_bc_tendencies!,
                                  step_lagrangian_particles!, maybe_prepare_first_time_step!

using Breeze.AtmosphereModels: AtmosphereModel, compute_pressure_correction!, make_pressure_correction!
using Breeze.CompressibleEquations: CompressibleEquations,
                                    VerticallyImplicitTimeStepping, CompressibleDynamics,
                                    _compute_ℂᵃᶜ²!, VerticalAcousticSolver,
                                    _build_ρθ_tridiagonal!, _back_solve_ρw!,
                                    _copy_field!, _add_field!

#####
##### GPU-compatible kernels
#####

@kernel function _set_to_initial!(u, u⁰)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = u⁰[i, j, k]
end

@kernel function _accumulate_tendency!(u, h_coeff, G)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = u[i, j, k] + h_coeff * G[i, j, k]
end

## Area-weighted vertical divergence: -V⁻¹ δz(Az ρw)
## Used for both the density update in the implicit solve
## and the implicit tendency fᴵ_ρ stored for the stage predictor.
@inline function _vertical_divergence_ρw(i, j, k, grid, ρw, Nz)
    ρw_top = ifelse(k < Nz, ρw[i, j, k + 1], zero(eltype(grid)))
    ρw_bot = ifelse(k > 1,  ρw[i, j, k],     zero(eltype(grid)))
    Az_top = Azᶜᶜᶠ(i, j, k + 1, grid)
    Az_bot = Azᶜᶜᶠ(i, j, k, grid)
    V = Vᶜᶜᶜ(i, j, k, grid)
    return -(Az_top * ρw_top - Az_bot * ρw_bot) / V
end

@kernel function _store_vertical_divergence!(fI_ρ, grid, ρw)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)
    @inbounds fI_ρ[i, j, k] = _vertical_divergence_ρw(i, j, k, grid, ρw, Nz)
end

@kernel function _update_density!(ρ, grid, τ, ρw)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)
    @inbounds ρ[i, j, k] = ρ[i, j, k] + τ * _vertical_divergence_ρw(i, j, k, grid, ρw, Nz)
end

@kernel function _compute_implicit_tendency!(fI, field_after, field_before, inv_τ)
    i, j, k = @index(Global, NTuple)
    @inbounds fI[i, j, k] = (field_after[i, j, k] - field_before[i, j, k]) * inv_τ
end

@kernel function _zero_boundary_ρw!(ρw, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ρw[i, j, 1] = 0
        ρw[i, j, Nz + 1] = 0
    end
end

#####
##### Implicit acoustic solve: Helmholtz for δρθ, back-solve for δρw, update ρ
#####
##### Solves the coupled implicit system at a single ARK stage:
#####   ρθ⁺ = ρθ* - τ div_z(θ ρw⁺)         (linearized vertical ρθ flux)
#####   ρw⁺ = ρw* - τ (ℂ²/θ) ∂(ρθ⁺)/∂z    (linearized vertical PGF)
#####   ρ⁺  = ρ*  + τ fᴵ_ρ(ρw⁺)            (vertical mass flux divergence)
#####
##### where τ = γᵢ h is the SDIRK diagonal element times Δt.
#####

function imex_acoustic_solve!(model, τ)
    τ == 0 && return nothing

    grid = model.grid
    arch = architecture(grid)
    dynamics = model.dynamics
    sc = dynamics.vertical_acoustic_solver
    sc isa Nothing && return nothing

    solver = sc.vertical_solver
    ρθ = model.formulation.potential_temperature_density
    θ = model.formulation.potential_temperature
    ρw = model.momentum.ρw
    ρ = dynamics.density
    ℂᵃᶜ² = sc.acoustic_speed_squared

    g = model.thermodynamic_constants.gravitational_acceleration

    ## 1. Save (ρθ)* before solve
    launch!(arch, grid, :xyz, _copy_field!, sc.ρθ_scratch, ρθ)

    p = dynamics.pressure

    ## 2. Compute horizontal-mean reference state (p̄, ρ̄)
    CompressibleEquations.compute_mean_reference_state!(sc, grid, p, dynamics.density)

    ## 3. Helmholtz (acoustic + gravity with mean-subtracted RHS)
    launch!(arch, grid, :xyz, _build_ρθ_tridiagonal!,
            solver.a, solver.b, solver.c, sc.rhs,
            grid, τ, ℂᵃᶜ², ρθ, θ, ρw, dynamics.density, p, g,
            sc.mean_pressure, sc.mean_density)

    solve!(ρθ, solver, sc.rhs)  # ρθ now holds δρθ

    ## 3. Recover ρθ⁺ = (ρθ)* + δρθ
    launch!(arch, grid, :xyz, _add_field!, ρθ, sc.ρθ_scratch)

    ## 5. Back-solve with mean-subtracted PGF+gravity
    launch!(arch, grid, :xyz, _back_solve_ρw!,
            ρw, grid, τ, ℂᵃᶜ², θ, ρθ, sc.ρθ_scratch, dynamics.density, p, g,
            sc.mean_pressure, sc.mean_density)

    ## 6. Update ρ from vertical divergence of corrected ρw
    launch!(arch, grid, :xyz, _update_density!, ρ, grid, τ, ρw)

    return nothing
end

#####
##### Zero ρw at impenetrable boundary faces
#####
##### After the stage predictor accumulates tendencies, ρw at boundary faces
##### (k=1 bottom, k=Nz+1 top) may be nonzero. These must be zeroed before
##### the implicit solve since w = 0 at impenetrable boundaries.
#####

function zero_boundary_ρw!(model)
    grid = model.grid
    Nz = size(grid, 3)
    launch!(architecture(grid), grid, :xy, _zero_boundary_ρw!, model.momentum.ρw, Nz)
    return nothing
end

#####
##### Evaluate the implicit function fᴵ at a given state (no solve needed)
#####
##### Used at stages where γᵢ = 0 (no Helmholtz solve). The implicit function
##### values are still needed by subsequent stages through aᴵᵢⱼ fᴵ(zⱼ).
#####
##### fᴵ_ρw = -(ℂ²/θ)∂ρθ/∂z - ρg   (linearized PGF + gravity)
##### fᴵ_ρθ = -V⁻¹ δz(Az θ ρw)      (vertical ρθ flux)
##### fᴵ_ρ  = -V⁻¹ δz(Az ρw)         (vertical density flux)
#####

using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶠ, ∂zᶜᶜᶠ

## fᴵ_ρw = -∂p/∂z - ρg  (using the ACTUAL EOS pressure, not linearized)
## This avoids the O(Δz²) mismatch between (ℂ²/θ)∂ρθ/∂z and ∂p/∂z
## that causes secular drift when accumulated through the ARK predictor.
@kernel function _evaluate_fI_ρw!(fI_ρw, grid, pressure, ρ, g)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ∂z_p = ∂zᶜᶜᶠ(i, j, k, grid, pressure)
        ρᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        fI_ρw[i, j, k] = -(∂z_p + ρᶠ * g) * (k > 1)
    end
end

@kernel function _evaluate_fI_ρθ!(fI_ρθ, grid, θ, ρw)
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
        fI_ρθ[i, j, k] = -(Az_top * θᶠ_top * ρw_top - Az_bot * θᶠ_bot * ρw_bot) / V
    end
end

function evaluate_implicit_tendency!(fI_tuple, model)
    grid = model.grid
    arch = architecture(grid)
    fI_ρθ, fI_ρw, fI_ρ = fI_tuple
    g = model.thermodynamic_constants.gravitational_acceleration

    ## fᴵ_ρw = -(∂p/∂z + ρg): full EOS PGF+gravity.
    ## This exactly matches what `explicit_z_pressure_gradient` and
    ## `explicit_buoyancy_forceᶜᶜᶠ` zeroed from fᴱ.
    launch!(arch, grid, :xyz, _evaluate_fI_ρw!,
            fI_ρw, grid, model.dynamics.pressure, model.dynamics.density, g)

    ## fᴵ_ρθ = -div_z(θ ρw): vertical ρθ flux (acoustic compressibility).
    ## The caller must subtract this from fᴱ_ρθ to avoid double-counting
    ## (since fᴱ includes full 3D advection of ρθ).
    launch!(arch, grid, :xyz, _evaluate_fI_ρθ!,
            fI_ρθ, grid, model.formulation.potential_temperature, model.momentum.ρw)

    ## fᴵ_ρ = -div_z(ρw): vertical density flux, matches the
    ## horizontal-only divergence in fᴱ_ρ.
    launch!(arch, grid, :xyz, _store_vertical_divergence!,
            fI_ρ, grid, model.momentum.ρw)

    return nothing
end

#####
##### Remove double-counted vertical ρθ flux from explicit tendency
#####
##### The explicit ρθ tendency includes full 3D advection (horizontal + vertical).
##### The implicit fᴵ_ρθ also contains the vertical ρθ flux. To ensure
##### fᴱ + fᴵ = f (total tendency), subtract fᴵ_ρθ from fᴱ_ρθ:
#####   fᴱ_adj = fᴱ_full - fᴵ_ρθ  →  fᴱ_adj + fᴵ_ρθ = fᴱ_full
#####

function remove_vertical_ρθ_flux_from_explicit!(fE, fI_ρθ, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _accumulate_tendency!,
            fE.ρθ, -1, fI_ρθ)
    return nothing
end

#####
##### Store implicit tendency from solve residual
#####
##### fᴵ_ρθ(zⱼ) = (ρθⱼ - ρθⱼ*) / τ     (from Helmholtz + add-back)
##### fᴵ_ρw(zⱼ) = (ρwⱼ - ρwⱼ*) / τ     (from back-solve)
##### fᴵ_ρ(zⱼ)  = -div_z(ρwⱼ)           (vertical mass flux at post-solve state)
#####

function store_implicit_tendency!(fI_ρθ, fI_ρw, fI_ρ, model, τ, ρw_scratch)
    ## All from solve residual: (field_after - field_before) / τ.
    ## The back-solve now includes reference-subtracted mean PGF+gravity,
    ## so the ρw residual captures the full perturbation fᴵ_ρw.
    grid = model.grid
    arch = architecture(grid)
    sc = model.dynamics.vertical_acoustic_solver
    inv_τ = 1 / τ

    launch!(arch, grid, :xyz, _compute_implicit_tendency!,
            fI_ρθ, model.formulation.potential_temperature_density, sc.ρθ_scratch, inv_τ)

    launch!(arch, grid, :xyz, _compute_implicit_tendency!,
            fI_ρw, model.momentum.ρw, ρw_scratch, inv_τ)

    launch!(arch, grid, :xyz, _store_vertical_divergence!,
            fI_ρ, grid, model.momentum.ρw)

    return nothing
end

#####
##### Stage predictor
#####
##### Sets model state to the ARK predictor value:
#####   z*ᵢ = yₙ + h Σⱼ₌₁ⁱ⁻¹ [aᴱᵢⱼ fᴱ(zⱼ) + aᴵᵢⱼ fᴵ(zⱼ)]
#####

function set_stage_predictor!(model, U⁰, h, explicit_tendencies, implicit_tendencies,
                               aE_row, aI_row, n_completed_stages)
    grid = model.grid
    arch = architecture(grid)

    # Reset all prognostic fields to yₙ
    for (u, u⁰) in zip(prognostic_fields(model), U⁰)
        launch!(arch, grid, :xyz, _set_to_initial!, u, u⁰)
    end

    # Accumulate explicit tendency contributions: h aᴱᵢⱼ fᴱ(zⱼ)
    for j in 1:n_completed_stages
        aE_row[j] == 0 && continue
        h_coeff = h * aE_row[j]
        for (u, G) in zip(prognostic_fields(model), explicit_tendencies[j])
            launch!(arch, grid, :xyz, _accumulate_tendency!, u, h_coeff, G)
        end
    end

    # Accumulate implicit tendency contributions: h aᴵᵢⱼ fᴵ(zⱼ)
    # Only ρθ, ρw, and ρ have nonzero fᴵ.
    for j in 1:n_completed_stages
        aI_row[j] == 0 && continue
        h_coeff = h * aI_row[j]
        fI_ρθ, fI_ρw, fI_ρ = implicit_tendencies[j]

        launch!(arch, grid, :xyz, _accumulate_tendency!,
                model.formulation.potential_temperature_density, h_coeff, fI_ρθ)
        launch!(arch, grid, :xyz, _accumulate_tendency!,
                model.momentum.ρw, h_coeff, fI_ρw)
        launch!(arch, grid, :xyz, _accumulate_tendency!,
                model.dynamics.density, h_coeff, fI_ρ)
    end

    return nothing
end

#####
##### Final combination (non-stiffly-accurate methods only)
#####
##### y_{n+1} = yₙ + h Σⱼ [bᴱⱼ fᴱ(zⱼ) + bᴵⱼ fᴵ(zⱼ)]
#####

function apply_final_combination!(model, U⁰, h, explicit_tendencies, implicit_tendencies,
                                   bE, bI, n_stages)
    grid = model.grid
    arch = architecture(grid)

    for (u, u⁰) in zip(prognostic_fields(model), U⁰)
        launch!(arch, grid, :xyz, _set_to_initial!, u, u⁰)
    end

    for j in 1:n_stages
        bE[j] == 0 && continue
        h_coeff = h * bE[j]
        for (u, G) in zip(prognostic_fields(model), explicit_tendencies[j])
            launch!(arch, grid, :xyz, _accumulate_tendency!, u, h_coeff, G)
        end
    end

    for j in 1:n_stages
        bI[j] == 0 && continue
        h_coeff = h * bI[j]
        fI_ρθ, fI_ρw, fI_ρ = implicit_tendencies[j]

        launch!(arch, grid, :xyz, _accumulate_tendency!,
                model.formulation.potential_temperature_density, h_coeff, fI_ρθ)
        launch!(arch, grid, :xyz, _accumulate_tendency!,
                model.momentum.ρw, h_coeff, fI_ρw)
        launch!(arch, grid, :xyz, _accumulate_tendency!,
                model.dynamics.density, h_coeff, fI_ρ)
    end

    return nothing
end
