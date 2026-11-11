#####
##### Preconditioned conjugate-gradient anelastic pressure solver for immersed boundaries
#####
##### On an ImmersedBoundaryGrid the default Fourier–tridiagonal solve ignores the immersed
##### boundary, so the projected momentum is divergent in cells next to it and mass leaks
##### through solid faces. This solver iterates on the operator of the masked grid instead,
#####
#####     V ∇·(ρᵣ ∇φ) = V ∇·(ρŨ) / Δt,
#####
##### with the gradient zeroed across immersed faces by the ImmersedBoundaryGrid operators
##### (homogeneous Neumann on the body). It is preconditioned by the Fourier–tridiagonal
##### anelastic solve on the underlying grid. φ = αᵣp′ is the kinematic pressure that
##### `make_pressure_correction!` applies as ρu -= Δt ρᵣ ∇φ.
#####
##### TODO: Oceananigans' ConjugateGradientPoissonSolver hard-codes the constant-coefficient
##### Laplacian; a variable-coefficient operator hook there would let Breeze reuse its fused kernels.
#####

struct AnelasticConjugateGradientSolver{T, I}
    reltol :: T
    abstol :: T
    maxiter :: I
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Skeleton for a preconditioned conjugate-gradient anelastic pressure solve on an
`ImmersedBoundaryGrid`; on other grids the exact Fourier–tridiagonal solve is used.
Pass it as `AnelasticDynamics(reference_state; pressure_solver=...)`.
`reltol` and `abstol` apply to the volume-normalized residual; `maxiter` caps the iterations per solve.
"""
AnelasticConjugateGradientSolver(; reltol=1e-8, abstol=1e-10, maxiter=100) =
    AnelasticConjugateGradientSolver(promote(reltol, abstol)..., maxiter)

Base.summary(s::AnelasticConjugateGradientSolver) =
    string("AnelasticConjugateGradientSolver(reltol=", s.reltol, ", maxiter=", s.maxiter, ")")

struct ConjugateGradientAnelasticPressureSolver{G, R, S, P, D}
    grid :: G
    right_hand_side :: R
    conjugate_gradient_solver :: S
    preconditioner :: P
    reference_density :: D
end

Architectures.architecture(solver::ConjugateGradientAnelasticPressureSolver) = architecture(solver.grid)
Base.summary(::ConjugateGradientAnelasticPressureSolver) = "ConjugateGradientAnelasticPressureSolver"

#####
##### Operator: V ∇·(ρᵣ ∇φ)
#####

# No flux through faces on or inside the immersed boundary (the same faces whose velocity is masked)
@inline open_face(i, j, k, grid, ℓx, ℓy, ℓz) = !peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz)

@inline ρAx_∂x(i, j, k, grid, ρᵣ, φ) =
    @inbounds open_face(i, j, k, grid, Face(), Center(), Center()) * ρᵣ[i, j, k] * Axᶠᶜᶜ(i, j, k, grid) * ∂xᶠᶜᶜ(i, j, k, grid, φ)
@inline ρAy_∂y(i, j, k, grid, ρᵣ, φ) =
    @inbounds open_face(i, j, k, grid, Center(), Face(), Center()) * ρᵣ[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid) * ∂yᶜᶠᶜ(i, j, k, grid, φ)
@inline ρAz_∂z(i, j, k, grid, ρᵣ, φ) =
    open_face(i, j, k, grid, Center(), Center(), Face()) * ℑzᵃᵃᶠ(i, j, k, grid, ρᵣ) * Azᶜᶜᶠ(i, j, k, grid) * ∂zᶜᶜᶠ(i, j, k, grid, φ)

@inline anelastic_V_divergence_of_gradient(i, j, k, grid, ρᵣ, φ) =
    δxᶜᵃᵃ(i, j, k, grid, ρAx_∂x, ρᵣ, φ) +
    δyᵃᶜᵃ(i, j, k, grid, ρAy_∂y, ρᵣ, φ) +
    δzᵃᵃᶜ(i, j, k, grid, ρAz_∂z, ρᵣ, φ)

@kernel function _anelastic_operator!(Aφ, grid, ρᵣ, φ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    @inbounds Aφ[i, j, k] = active * anelastic_V_divergence_of_gradient(i, j, k, grid, ρᵣ, φ)
end

# Signature required by Oceananigans' ConjugateGradientSolver: linear_operation!(Ax, x, args...)
function anelastic_operator!(Aφ, φ, ρᵣ)
    grid = φ.grid
    fill_halo_regions!(φ)
    launch!(architecture(grid), grid, :xyz, _anelastic_operator!, Aφ, grid, ρᵣ, φ)
    return nothing
end

#####
##### Preconditioner: the Fourier–tridiagonal anelastic solve on the underlying grid
#####

struct AnelasticFourierPreconditioner{S}
    fourier_solver :: S
end

Base.summary(::AnelasticFourierPreconditioner) = "AnelasticFourierPreconditioner"

# The CG residual r is volume-weighted (V × source). The Fourier–tridiagonal source term is Δz × source,
# so it is r / (Δx Δy) = r Δz / V.
@kernel function _volume_to_fourier_source!(rhs, grid, r)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = r[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid) * V⁻¹ᶜᶜᶜ(i, j, k, grid)
end

function Solvers.precondition!(z, preconditioner::AnelasticFourierPreconditioner, r, args...)
    solver = preconditioner.fourier_solver
    grid = solver.grid
    launch!(architecture(grid), grid, :xyz, _volume_to_fourier_source!, solver.source_term, grid, r)
    fill!(solver.storage, 0)
    solve!(z, solver)
    mask_immersed_field!(z, zero(eltype(z)))
    return z
end

#####
##### Construction and solve
#####

function AtmosphereModels.dynamics_pressure_solver(dynamics::AnelasticDynamics{<:Any, <:Any, <:Any, <:AnelasticConjugateGradientSolver},
                                                   grid::ImmersedBoundaryGrid)
    method = dynamics.pressure_solver
    ρᵣ = dynamics.reference_state.density
    fourier_solver = anelastic_pressure_solver(dynamics.pressure_solver_float_type, grid.underlying_grid, ρᵣ)
    preconditioner = AnelasticFourierPreconditioner(fourier_solver)
    rhs = CenterField(grid)

    conjugate_gradient_solver = ConjugateGradientSolver(anelastic_operator!;
                                                        template_field = rhs,
                                                        reltol = method.reltol,
                                                        abstol = method.abstol,
                                                        maxiter = method.maxiter,
                                                        preconditioner,
                                                        enforce_gauge_condition! = ZeroMeanGaugeCondition(grid),
                                                        residual_norm = VolumeInverseNorm(grid))

    return ConjugateGradientAnelasticPressureSolver(grid, rhs, conjugate_gradient_solver, preconditioner, ρᵣ)
end

AtmosphereModels.update_pressure_solver!(solver::ConjugateGradientAnelasticPressureSolver, dynamics::AnelasticDynamics) =
    AtmosphereModels.update_pressure_solver!(solver.preconditioner.fourier_solver, dynamics)

@kernel function _volume_weighted_anelastic_source!(rhs, grid, ρŨ, Δt)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    ρu, ρv, ρw = ρŨ
    @inbounds rhs[i, j, k] = active * Vᶜᶜᶜ(i, j, k, grid) * divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw) / Δt
end

function solve_for_anelastic_pressure!(φ, solver::ConjugateGradientAnelasticPressureSolver, ρŨ, Δt)
    grid = solver.grid
    arch = architecture(grid)
    kernel_Δt = kernel_time_step(arch, grid, Δt)
    launch!(arch, grid, :xyz, _volume_weighted_anelastic_source!, solver.right_hand_side, grid, ρŨ, kernel_Δt)
    # Warm-started from the previous pressure φ
    solve!(φ, solver.conjugate_gradient_solver, solver.right_hand_side, solver.reference_density)
    return φ
end
