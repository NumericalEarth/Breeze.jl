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

using Oceananigans: CenterField, Average, Field, architecture, compute!, set!
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

Uses the Exner-function splitting (same as acoustic substepping / WRF / MPAS):
buoyancy goes in the explicit tendency, perturbation Exner PGF goes in the
implicit solve. The Helmholtz is purely acoustic (no gravity operator).

Fields
======

- `acoustic_speed_squared`: ℂᵃᶜ² = γᵐ Rᵐ T at cell centers, updated each time step
- `rhs`: Scratch field for the tridiagonal right-hand side
- `ρθ_scratch`: Scratch field storing (ρθ)* before the solve for computing δρθ
- `exner_perturbation`: Π' = Π - Π₀ at cell centers, updated each implicit stage
- `vertical_solver`: `BatchedTridiagonalSolver` for Nₓ × Nᵧ independent column solves
"""
struct VerticalAcousticSolver{F, S}
    acoustic_speed_squared  :: F
    rhs                     :: F
    ρθ_scratch              :: F
    exner_perturbation      :: F
    vertical_solver         :: S
end

function VerticalAcousticSolver(grid)
    FT = eltype(grid)
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)

    acoustic_speed_squared = CenterField(grid)
    rhs = CenterField(grid)
    ρθ_scratch = CenterField(grid)
    exner_perturbation = CenterField(grid)

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
                                  exner_perturbation, vertical_solver)
end

Adapt.adapt_structure(to, s::VerticalAcousticSolver) =
    VerticalAcousticSolver(adapt(to, s.acoustic_speed_squared),
                           adapt(to, s.rhs),
                           adapt(to, s.ρθ_scratch),
                           adapt(to, s.exner_perturbation),
                           nothing)

#####
##### Materialization dispatch
#####

materialize_vertical_acoustic_solver(::ExplicitTimeStepping, grid) = nothing
materialize_vertical_acoustic_solver(::SplitExplicitTimeDiscretization, grid) = nothing
materialize_vertical_acoustic_solver(::VerticallyImplicitTimeStepping, grid) =
    VerticalAcousticSolver(grid)

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
##### Compute Exner perturbation: Π' = (p/pˢᵗ)^κ - Π₀
#####
##### Π₀ is the fixed ExnerReferenceState (may be 1D or latitude-dependent 3D).
##### Π' is small when the reference matches the atmosphere's θ(φ,z) structure.
#####

@kernel function _compute_exner_perturbation!(Π′, grid, p, Π₀, pˢᵗ, κ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Πⁱ = (p[i, j, k] / pˢᵗ)^κ
        Π′[i, j, k] = Πⁱ - Π₀[i, j, k]
    end
end

function compute_exner_perturbation!(sc, grid, p, Π₀, pˢᵗ, κ)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _compute_exner_perturbation!, sc.exner_perturbation, grid, p, Π₀, pˢᵗ, κ)
    fill_halo_regions!(sc.exner_perturbation)
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
##### Exner buoyancy: ρ·b where b = -(cₚᵈ θᵥ ∂Π₀/∂z + g)
#####
##### The ExnerReferenceState guarantees cₚᵈ θ₀_face (Π₀[k]-Π₀[k-1])/Δz = -g
##### exactly. When θᵥ ≈ θ₀, buoyancy b ≈ 0. For latitude-dependent references,
##### b is small everywhere. This is the slow buoyancy forcing (explicit tendency).
#####

@inline function AtmosphereModels.exner_buoyancy_forceᶜᶜᶠ(i, j, k, grid,
        dynamics::CompressibleDynamics{<:VerticallyImplicitTimeStepping},
        temperature, specific_prognostic_moisture, microphysics, microphysical_fields, constants)

    ref = dynamics.reference_state
    ref === nothing && return zero(grid)

    Π₀ = ref.exner_function
    ρ = dynamics.density
    p = dynamics.pressure
    g = constants.gravitational_acceleration
    cₚᵈ = constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(constants)
    κ = Rᵈ / cₚᵈ
    pˢᵗ = dynamics.standard_pressure

    ## Virtual potential temperature at the face (dry: θᵥ = T/Π)
    @inbounds begin
        T_above = temperature[i, j, k]
        T_below = temperature[i, j, k - 1]
        p_above = p[i, j, k]
        p_below = p[i, j, k - 1]
    end
    Π_above = (p_above / pˢᵗ)^κ
    Π_below = (p_below / pˢᵗ)^κ
    θᵥ_above = T_above / Π_above
    θᵥ_below = T_below / Π_below
    θᵥᶠ = (θᵥ_above + θᵥ_below) / 2

    ## ∂Π₀/∂z at the face
    Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
    @inbounds δz_Π₀ = Π₀[i, j, k] - Π₀[i, j, k - 1]

    ## Buoyancy: b = -(cₚ θᵥ ∂Π₀/∂z + g)
    b = -cₚᵈ * θᵥᶠ * δz_Π₀ / Δzᶠ - g

    ## ρ at the face
    ρᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)

    return ρᶠ * b * (k > 1)
end

#####
##### Purely acoustic Helmholtz for δρθ (Exner splitting):
#####
#####   [I - τ² div_z(ℂᵃᶜ² ∂/∂z)] δρθ = -τ div_z(θ (ρw)*)
#####                                      + τ² div_z(ℂᵃᶜ² ρθ* ∂Π'/∂z)
#####
##### No gravity operator on the LHS — gravity is entirely in the explicit
##### buoyancy b = -(cₚ θᵥ ∂Π₀/∂z + g). The Helmholtz is purely acoustic:
##### L = V⁻¹ δz(Az ℂ² δ(·)/Δzᶠ).
#####
##### The RHS perturbation Exner term comes from the mean fᴵ_ρw at the
##### predictor: -ρ cₚ θᵥ ∂Π'/∂z = -ℂᵃᶜ² ρθ ∂Π'/∂z / (θ p) × p ≈ ...
##### Simplified using ℂᵃᶜ² = γᵐ Rᵐ T and ∂p/∂(ρθ) = ℂᵃᶜ²/θ,
##### the perturbation Exner PGF at the predictor is -(ℂᵃᶜ²/θ) ρθ ∂Π'/∂z.
#####
##### Back-solve (2 terms, no gravity):
#####   δρw = -τ (ℂ²/θ) ∂(δρθ)/∂z  -  τ cₚ (ρθ)_face ∂Π'/∂z
#####

@kernel function _build_ρθ_tridiagonal!(lower, diag, upper, rhs_field,
                                         grid, αΔt, ℂᵃᶜ², ρθ, θ, ρw, Π′, cₚᵈ)
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

        ## Purely acoustic Helmholtz: V⁻¹ δz(Az ℂ² δz(·)/Δzᶠ)
        αΔt² = αΔt * αΔt
        Q_bot = αΔt² * Az_bot * ℂ²_bot / (Δzᶠ_bot * V)
        Q_top = αΔt² * Az_top * ℂ²_top / (Δzᶠ_top * V)
        Q_bot = ifelse(k == 1, zero(Q_bot), Q_bot)
        Q_top = ifelse(k == Nz, zero(Q_top), Q_top)

        V_above = Vᶜᶜᶜ(i, j, k + 1, grid)
        Q_lower = αΔt² * Az_top * ℂ²_top / (Δzᶠ_top * V_above)
        Q_lower = ifelse(k >= Nz, zero(Q_lower), Q_lower)

        ## Symmetric acoustic tridiagonal (no gravity first-derivative)
        lower[i, j, k] = -Q_lower
        upper[i, j, k] = -Q_top
        diag[i, j, k] = 1 + Q_bot + Q_top

        ## RHS term 1: acoustic flux from predictor ρw
        θᶠ_top = ℑzᵃᵃᶠ(i, j, k + 1, grid, θ)
        θᶠ_bot = ℑzᵃᵃᶠ(i, j, k, grid, θ)
        ρw_top = ρw[i, j, k + 1] * (k < Nz)
        ρw_bot = ρw[i, j, k] * (k > 1)

        flux_rhs = -αΔt / V * (Az_top * θᶠ_top * ρw_top - Az_bot * θᶠ_bot * ρw_bot)

        ## RHS term 2: perturbation Exner PGF at predictor.
        ## fᴵ_ρw(z*) = -cₚ (ρθ)_face ∂Π'/∂z (small, no cancellation).
        ρθᶠ_top = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρθ)
        ρθᶠ_bot = ℑzᵃᵃᶠ(i, j, k, grid, ρθ)

        dΠ′_top = (Π′[i, j, k + 1] - Π′[i, j, k]) / Δzᶠ_top * (k < Nz)
        dΠ′_bot = (Π′[i, j, k] - Π′[i, j, k - 1]) / Δzᶠ_bot * (k > 1)

        fᴵ_ρw_top = -cₚᵈ * ρθᶠ_top * dΠ′_top
        fᴵ_ρw_bot = -cₚᵈ * ρθᶠ_bot * dΠ′_bot

        exner_rhs = αΔt² / V * (Az_top * θᶠ_top * fᴵ_ρw_top - Az_bot * θᶠ_bot * fᴵ_ρw_bot)

        rhs_field[i, j, k] = flux_rhs + exner_rhs
    end
end

#####
##### Back-solve: perturbation PGF + mean perturbation Exner PGF
#####
#####   δρw = -τ (ℂ²/θ)ᶠ ∂(δρθ)/∂z  -  τ cₚ (ρθ)_face ∂Π'*/∂z
#####
##### Term 1: acoustic PGF from the Helmholtz correction δρθ.
##### Term 2: perturbation Exner PGF at the predictor (small, no cancellation).
##### No gravity-density coupling — gravity is in the explicit buoyancy.
#####

@kernel function _back_solve_ρw!(ρw, grid, αΔt, ℂᵃᶜ², θ, ρθ, ρθ_scratch, Π′, cₚᵈ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ℂ²ᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ℂᵃᶜ²)
        θᶠ = ℑzᵃᵃᶠ(i, j, k, grid, θ)
        Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)

        δρθ_above = ρθ[i, j, k] - ρθ_scratch[i, j, k]
        δρθ_below = ρθ[i, j, k - 1] - ρθ_scratch[i, j, k - 1]

        ## 1. Perturbation PGF from Helmholtz: -(ℂ²/θ) ∂(δρθ)/∂z
        perturbation_pgf = -ℂ²ᶠ / θᶠ * (δρθ_above - δρθ_below) / Δzᶠ

        ## 2. Perturbation Exner PGF at predictor: -cₚ (ρθ)_face ∂Π'/∂z
        ρθᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρθ_scratch)
        dΠ′ = (Π′[i, j, k] - Π′[i, j, k - 1]) / Δzᶠ
        mean_exner_pgf = -cₚᵈ * ρθᶠ * dΠ′

        ρw[i, j, k] = (ρw[i, j, k] + αΔt * (perturbation_pgf + mean_exner_pgf)) * (k > 1)
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

    cₚᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(model.thermodynamic_constants)
    κ = Rᵈ / cₚᵈ
    g = model.thermodynamic_constants.gravitational_acceleration
    pˢᵗ = dynamics.standard_pressure

    ## β·αΔt is the effective implicit time scale.
    βαΔt = β * αΔt

    ## 1. Save (ρθ)* for computing δρθ after the solve
    launch!(arch, grid, :xyz, _copy_field!, sc.ρθ_scratch, ρθ)

    ## 2. Exner perturbation Π' = (p/pˢᵗ)^κ - Π₀ from fixed reference
    ref = dynamics.reference_state
    compute_exner_perturbation!(sc, grid, dynamics.pressure, ref.exner_function, pˢᵗ, κ)

    ## 3. Purely acoustic Helmholtz with perturbation Exner RHS
    launch!(arch, grid, :xyz, _build_ρθ_tridiagonal!,
            solver.a, solver.b, solver.c, sc.rhs,
            grid, βαΔt, ℂᵃᶜ², ρθ, θ, ρw, sc.exner_perturbation, cₚᵈ)

    solve!(ρθ, solver, sc.rhs)  # ρθ now holds δρθ

    ## 4. Recover (ρθ)⁺ = (ρθ)* + δρθ
    launch!(arch, grid, :xyz, _add_field!, ρθ, sc.ρθ_scratch)

    ## 5. Back-solve with perturbation Exner PGF (no gravity)
    launch!(arch, grid, :xyz, _back_solve_ρw!,
            ρw, grid, βαΔt, ℂᵃᶜ², θ, ρθ, sc.ρθ_scratch, sc.exner_perturbation, cₚᵈ)

    ## No density update here — ρ uses full 3D divergence in the explicit tendency.
    ## This avoids split-weighting instability from different Butcher coefficients
    ## on horizontal vs vertical density divergence.

    return nothing
end
