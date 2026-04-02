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
using Breeze.Thermodynamics: dry_air_gas_constant
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

    cₚᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cₚᵈ
    g = model.thermodynamic_constants.gravitational_acceleration
    pˢᵗ = dynamics.standard_pressure

    ## 1. Save (ρθ)* before solve
    launch!(arch, grid, :xyz, _copy_field!, sc.ρθ_scratch, ρθ)

    ## 2. Exner perturbation Π' = (p/pˢᵗ)^κ - Π₀ from fixed reference
    ref = dynamics.reference_state
    CompressibleEquations.compute_exner_perturbation!(sc, grid, dynamics.pressure,
                                                       ref.exner_function, pˢᵗ, κ)

    ## 3. Purely acoustic Helmholtz with perturbation Exner RHS
    launch!(arch, grid, :xyz, _build_ρθ_tridiagonal!,
            solver.a, solver.b, solver.c, sc.rhs,
            grid, τ, ℂᵃᶜ², ρθ, θ, ρw, sc.exner_perturbation, cₚᵈ)

    solve!(ρθ, solver, sc.rhs)  # ρθ now holds δρθ

    ## 4. Recover ρθ⁺ = (ρθ)* + δρθ
    launch!(arch, grid, :xyz, _add_field!, ρθ, sc.ρθ_scratch)

    ## 5. Back-solve with perturbation Exner PGF (no gravity)
    launch!(arch, grid, :xyz, _back_solve_ρw!,
            ρw, grid, τ, ℂᵃᶜ², θ, ρθ, sc.ρθ_scratch, sc.exner_perturbation, cₚᵈ)

    ## No density update — ρ uses full 3D divergence in the explicit tendency.

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
##### Exner splitting: fᴵ_ρw = -cₚ (ρθ)_face ∂Π'/∂z (perturbation Exner PGF)
##### fᴵ_ρθ = -V⁻¹ δz(Az θ ρw)      (vertical ρθ flux)
##### fᴵ_ρ  = -V⁻¹ δz(Az ρw)         (vertical density flux)
#####

using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶠ, ∂zᶜᶜᶠ

## fᴵ_ρw = -cₚ (ρθ)_face ∂Π'/∂z  (perturbation Exner PGF, small)
## Gravity is in the explicit buoyancy b = -(cₚ θᵥ ∂Π₀/∂z + g), not here.
@kernel function _evaluate_fI_ρw!(fI_ρw, grid, ρθ, Π′, cₚᵈ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρθᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρθ)
        Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
        dΠ′ = (Π′[i, j, k] - Π′[i, j, k - 1]) / Δzᶠ
        fI_ρw[i, j, k] = (-cₚᵈ * ρθᶠ * dΠ′) * (k > 1)
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
    dynamics = model.dynamics
    sc = dynamics.vertical_acoustic_solver

    cₚᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cₚᵈ
    g = model.thermodynamic_constants.gravitational_acceleration
    pˢᵗ = dynamics.standard_pressure

    ## Exner perturbation from fixed reference
    CompressibleEquations.compute_exner_perturbation!(sc, grid, dynamics.pressure,
                                                       dynamics.reference_state.exner_function, pˢᵗ, κ)

    ## fᴵ_ρw = -cₚ (ρθ)_face ∂Π'/∂z: perturbation Exner PGF.
    ## Gravity is in the explicit buoyancy, not in fᴵ.
    launch!(arch, grid, :xyz, _evaluate_fI_ρw!,
            fI_ρw, grid, model.formulation.potential_temperature_density, sc.exner_perturbation, cₚᵈ)

    ## fᴵ_ρθ = 0 and fᴵ_ρ = 0: only ρw is split between fᴱ/fᴵ.
    ## ρθ and ρ use full 3D tendencies in fᴱ with uniform Butcher weights.
    fill!(parent(fI_ρθ), 0)
    fill!(parent(fI_ρ), 0)

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

## With the Exner splitting, fᴵ_ρθ = 0 and fᴱ_ρθ keeps full 3D advection.
## No subtraction needed — avoids split-weighting instability.
function remove_vertical_ρθ_flux_from_explicit!(fE, fI_ρθ, grid)
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
    grid = model.grid
    arch = architecture(grid)
    sc = model.dynamics.vertical_acoustic_solver
    inv_τ = 1 / τ

    ## Only ρw has nonzero fᴵ (Exner PGF splitting).
    ## ρθ and ρ use full 3D tendencies in fᴱ — no splitting.
    fill!(parent(fI_ρθ), 0)

    launch!(arch, grid, :xyz, _compute_implicit_tendency!,
            fI_ρw, model.momentum.ρw, ρw_scratch, inv_τ)

    fill!(parent(fI_ρ), 0)

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
