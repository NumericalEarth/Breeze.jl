#####
##### Fused acoustic substep — EXPERIMENT, ABANDONED (measured −24% on H100). Kept for the record.
#####
##### The baseline substep launches 3 column-structured kernels in sequence:
#####   _build_predictors_and_vertical_rhs!  (:xy)  → ρ★, ρθ★, RHS(=ρw′)
#####   solve!  (BatchedTridiagonalSolver, ZDirection)
#####   _post_solve_recovery!  (:xyz)              → ρ′, ρθ′
##### This file fuses them into ONE :xy (thread-per-column) kernel, reusing the exact operator
##### bodies + solve_batched_tridiagonal_system_z! + get_coefficient (correct-by-construction).
#####
##### RESULT: validated bit-identical but 0.76× (24% SLOWER) — the fused kernel is 6.17 ms/call vs
##### build 1.86 + tridiag 1.22 + recover 0.47 = 3.55 ms for the 3 separate (1.74× slower). Cause:
##### register-spill / occupancy collapse (combined live set; recovery loses its :xyz parallelism),
##### with no compensating bandwidth saving (intermediates kept global). Holding intermediates in
##### per-thread local arrays would only deepen the register pressure. Conclusion: the 3 substep
##### kernels are already occupancy-tuned; fusion trades occupancy for bandwidth and loses on this
##### GPU/problem. DO NOT enable for production. Opt-in via FUSE_ACOUSTIC=1.

using Oceananigans.Solvers: solve_batched_tridiagonal_system_z!
using Oceananigans.Grids: ZDirection

# build (ρ★, ρθ★) + tridiag RHS for one column (i,j); body copied from _build_predictors_and_vertical_rhs!
@inline function _build_predictors_column!(i, j, ρw′_rhs, ρ′★, ρθ′★, ρ′, ρθ′, ρw′, ρu′, ρv′,
                                           grid, dynamics, Δτ, δτᵐ⁺, δτˢ⁻, Gˢρ, Gˢρθ, Gˢρw,
                                           thermodynamic_tendency_factor,
                                           vertical_momentum_tendency_factor, θᴸ, Πᴸ, γRᵐᴸ, g,
                                           dˢ⁻, sponge, apply_pressure_gradient)
    Nz = size(grid, 3)
    slope_correction = ifelse(apply_pressure_gradient, one(Δτ), zero(Δτ))
    @inbounds begin
        for k in 1:Nz
            V     = Vᶜᶜᶜ(i, j, k, grid)
            ∇ʰ_M  = div_xyᶜᶜᶜ(i, j, k, grid, ρu′, ρv′)
            ∇ʰ_θM = (δxᶜᵃᵃ(i, j, k, grid, theta_face_x_flux, θᴸ, ρu′) +
                     δyᵃᶜᵃ(i, j, k, grid, theta_face_y_flux, θᴸ, ρv′)) / V
            ρ′★[i, j, k]  = ρ′[i, j, k] + Δτ * (Gˢρ[i, j, k] - ∇ʰ_M) -
                            δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, acoustic_vertical_momentum_flux,
                                          dynamics, ρu′, ρv′, ρw′)
            ρθ′★[i, j, k] = ρθ′[i, j, k] + Δτ * (thermodynamic_tendency_factor * Gˢρθ[i, j, k] - ∇ʰ_θM) -
                            δτˢ⁻ * ∂zᶜᶜᶜ(i, j, k, grid, theta_face_z_flux,
                                          θᴸ, dynamics, ρu′, ρv′, ρw′)
        end
        for k in 2:Nz
            ∂r_p′★  = z_linearized_pressure_gradient(i, j, k, grid, dynamics, ρθ′★, Πᴸ, γRᵐᴸ, slope_correction)
            ∂r_p′ˢ⁻ = z_linearized_pressure_gradient(i, j, k, grid, dynamics, ρθ′,  Πᴸ, γRᵐᴸ, slope_correction)
            sound_force = δτˢ⁻ * ∂r_p′ˢ⁻ + δτᵐ⁺ * ∂r_p′★
            ρ′ᶜᶜᶠ★  = ℑzᵃᵃᶠ(i, j, k, grid, ρ′★)
            ρ′ᶜᶜᶠˢ⁻ = ℑzᵃᵃᶠ(i, j, k, grid, ρ′)
            buoy_force  = g * (δτˢ⁻ * ρ′ᶜᶜᶠˢ⁻ + δτᵐ⁺ * ρ′ᶜᶜᶠ★)
            ∂z²_ρw′ˢ⁻   = ∂zᶜᶜᶠ(i, j, k, grid, ∂zᶜᶜᶜ, ρw′)
            damp_force  = - dˢ⁻ * ∂z²_ρw′ˢ⁻
            sponge_force = sponge_rhs(i, j, k, grid, sponge, δτˢ⁻, ρw′)
            ρw′_rhs[i, j, k] = ρw′[i, j, k] + Δτ * vertical_momentum_tendency_factor * Gˢρw[i, j, k] -
                               sound_force - buoy_force - damp_force - sponge_force
        end
        ρw′_rhs[i, j, 1]      = 0
        ρw′_rhs[i, j, Nz + 1] = 0
    end
end

# recovery for one column (i,j); body copied from _post_solve_recovery!, looped over k
@inline function _recover_column!(i, j, ρ′, ρθ′, ρw′, ρu′, ρv′, ρ′★, ρθ′★, grid, dynamics, δτᵐ⁺, θᴸ)
    Nz = size(grid, 3)
    @inbounds for k in 1:Nz
        ρ′[i, j, k]  = ρ′★[i, j, k] -
                       δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, acoustic_vertical_momentum_flux, dynamics, ρu′, ρv′, ρw′)
        ρθ′[i, j, k] = ρθ′★[i, j, k] -
                       δτᵐ⁺ * ∂zᶜᶜᶜ(i, j, k, grid, theta_face_z_flux, θᴸ, dynamics, ρu′, ρv′, ρw′)
    end
end

@kernel function _fused_acoustic_substep!(ρ′, ρθ′, ρw′, ρu′, ρv′, ρ′★, ρθ′★, t,
                                          grid, dynamics, Δτ, δτᵐ⁺, δτˢ⁻, dᵐ⁺, dˢ⁻,
                                          Gˢρ, Gˢρθ, Gˢρw, thermodynamic_tendency_factor,
                                          vertical_momentum_tendency_factor, θᴸ, Πᴸ, γRᵐᴸ, g,
                                          sponge, apply_pressure_gradient)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    _build_predictors_column!(i, j, ρw′, ρ′★, ρθ′★, ρ′, ρθ′, ρw′, ρu′, ρv′,
                              grid, dynamics, Δτ, δτᵐ⁺, δτˢ⁻, Gˢρ, Gˢρθ, Gˢρw,
                              thermodynamic_tendency_factor, vertical_momentum_tendency_factor,
                              θᴸ, Πᴸ, γRᵐᴸ, g, dˢ⁻, sponge, apply_pressure_gradient)
    args = (Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, sponge)
    solve_batched_tridiagonal_system_z!(i, j, Nz, ρw′, AcousticTridiagLower(), AcousticTridiagDiagonal(),
                                        AcousticTridiagUpper(), ρw′, t, grid, nothing, args, ZDirection())
    _recover_column!(i, j, ρ′, ρθ′, ρw′, ρu′, ρv′, ρ′★, ρθ′★, grid, dynamics, δτᵐ⁺, θᴸ)
end
