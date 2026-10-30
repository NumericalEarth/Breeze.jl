include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: AtmosphereModels, closure_implicit_upper_diagonal, closure_implicit_lower_diagonal,
                               closure_implicit_linear_coefficient, closure_implicit_diagonal, closure_scalar_index, implicit_advection_density,
                               thermodynamic_density
using Breeze.TimeSteppers: implicit_stage_step!
using Oceananigans
using Oceananigans.Grids: Center
using Oceananigans.Operators: Δzᶜᶜᶜ, ℑzᵃᵃᶠ
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, VerticallyImplicitTimeDiscretization
using GPUArraysCore: @allowscalar
using Test

#####
##### A closure that transports the closure scalar with index `I` by first-order upwind advection at
##### the face velocity `wₖ = w × profile(k)`, entirely through the implicit coefficients: the flux
##### through the interior face k is Fₖ = ρᶠₖ [max(wₖ, 0) cₖ₋₁ + min(wₖ, 0) cₖ], with c = q / ρ, and
##### it vanishes at the bottom and top faces. Such a velocity is the compensating subsidence of a
##### mass-flux closure. `Uniform` is constant on interior faces; `Sinusoidal` is w sin(π (k - 1) / Nz),
##### which is divergent, like a return flow. `conservative = false` drops the linear coefficient,
##### giving the advective form.
#####

struct Uniform end
struct Sinusoidal end

struct UpwindTransport{I, P, FT} <: AbstractTurbulenceClosure{VerticallyImplicitTimeDiscretization, 1}
    w :: FT
    profile :: P
    conservative :: Bool
end

UpwindTransport{I}(w, profile = Uniform(); conservative = true) where I =
    UpwindTransport{I, typeof(profile), typeof(w)}(w, profile, conservative)

@inline face_velocity(k, grid, w, ::Uniform) = w * (1 < k) * (k ≤ grid.Nz)
@inline face_velocity(k, grid, w, ::Sinusoidal) = w * sin(π * (k - 1) / grid.Nz) * (1 < k) * (k ≤ grid.Nz)
@inline face_velocity(k, grid, closure::UpwindTransport) = face_velocity(k, grid, closure.w, closure.profile)

# A closure that does not extend the hook
struct InertClosure <: AbstractTurbulenceClosure{VerticallyImplicitTimeDiscretization, 1} end

const TestClosure = Union{UpwindTransport, InertClosure}

@inline AtmosphereModels.∇_dot_Jᶜ(i, j, k, grid, ρ, ::TestClosure, args...) = zero(grid)
@inline AtmosphereModels.∂ⱼ_𝒯₁ⱼ(i, j, k, grid, ρ, ::TestClosure, args...) = zero(grid)
@inline AtmosphereModels.∂ⱼ_𝒯₂ⱼ(i, j, k, grid, ρ, ::TestClosure, args...) = zero(grid)
@inline AtmosphereModels.∂ⱼ_𝒯₃ⱼ(i, j, k, grid, ρ, ::TestClosure, args...) = zero(grid)

# Row k, column k + 1: the downward (min) part of the flux through face k + 1. Zero for k = Nz.
@inline function AtmosphereModels.closure_implicit_upper_diagonal(i, j, k, grid, closure::UpwindTransport{I}, K, ::Val{I},
                                                                  ℓx, ℓy, ::Center, Δt, clock, fields, ρ) where I
    w = face_velocity(k+1, grid, closure)
    ρᶠ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ)
    return @inbounds Δt * ρᶠ * min(w, 0) / (Δzᶜᶜᶜ(i, j, k, grid) * ρ[i, j, k+1])
end

# Row k′ + 1, column k′: the upward (max) part of the flux through face k′ + 1. Zero for k′ = 0.
@inline function AtmosphereModels.closure_implicit_lower_diagonal(i, j, k′, grid, closure::UpwindTransport{I}, K, ::Val{I},
                                                                  ℓx, ℓy, ::Center, Δt, clock, fields, ρ) where I
    k = k′ + 1
    w = face_velocity(k, grid, closure)
    ρᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
    return @inbounds - Δt * ρᶠ * max(w, 0) / (Δzᶜᶜᶜ(i, j, k, grid) * ρ[i, j, k′])
end

# λ = - ∂ŵ/∂z, the density-weighted divergence of the closure velocity (see the docstring)
@inline function AtmosphereModels.closure_implicit_linear_coefficient(i, j, k, grid, closure::UpwindTransport{I}, K, ::Val{I},
                                                                      ℓx, ℓy, ::Center, Δt, clock, fields, ρ) where I
    w⁺ = face_velocity(k+1, grid, closure)
    w⁻ = face_velocity(k,   grid, closure)
    ρᶠ⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ)
    ρᶠ⁻ = ℑzᵃᵃᶠ(i, j, k,   grid, ρ)
    ŵ⁺ = @inbounds ρᶠ⁺ * (max(w⁺, 0) / ρ[i, j, k] + min(w⁺, 0) / ρ[i, j, k+1])
    ŵ⁻ = @inbounds ρᶠ⁻ * (max(w⁻, 0) / ρ[i, j, k-1] + min(w⁻, 0) / ρ[i, j, k])
    return - closure.conservative * (ŵ⁺ - ŵ⁻) / Δzᶜᶜᶜ(i, j, k, grid)
end

function upwind_column_model(closure; Nz = 16, H = 4000)
    grid = RectilinearGrid(default_arch; size = Nz, z = (0, H), topology = (Flat, Flat, Bounded))
    model = AtmosphereModel(grid; closure, advection = nothing, tracers = :ρc)
    set!(model; θ = z -> 300 + 3e-3 * z, ρc = z -> exp(-(z - H / 2)^2 / (H / 8)^2))
    return model
end

# The backward-Euler upwind matrix A of (I + Δt L) q = q⋆, written out from the flux
function upwind_matrix(grid, ρ, w, Δt; profile = Uniform())
    Nz = grid.Nz
    ρᶜ = Array(interior(ρ))[1, 1, :]
    ρᶠ = [(ρᶜ[k - 1] + ρᶜ[k]) / 2 for k in 2:Nz]  # interior faces 2, …, Nz
    Δz = [Δzᶜᶜᶜ(1, 1, k, grid) for k in 1:Nz]
    A = zeros(Nz, Nz)
    for k in 1:Nz
        A[k, k] = 1
    end
    # Face f (2 ≤ f ≤ Nz) carries ρᶠ [max(w, 0) q_{f-1}/ρ_{f-1} + min(w, 0) q_f/ρ_f] out of cell f-1 into cell f
    for f in 2:Nz
        F = ρᶠ[f - 1]
        wᶠ = face_velocity(f, grid, w, profile)
        for (m, coefficient) in ((f - 1, max(wᶠ, 0) / ρᶜ[f - 1]), (f, min(wᶠ, 0) / ρᶜ[f]))
            A[f - 1, m] += Δt * F * coefficient / Δz[f - 1]
            A[f,     m] -= Δt * F * coefficient / Δz[f]
        end
    end
    return A
end

column(field) = Array(interior(field))[1, 1, :]

@testset "Closure-supplied implicit coefficients" begin
    @testset "the default contributes nothing" begin
        model = upwind_column_model(TKEBasedTurbulenceClosure())
        grid = model.grid
        ρ = implicit_advection_density(model.dynamics, model.formulation, :ρc)
        args = (Center(), Center(), Center(), 120.0, model.clock, fields(model), ρ)
        for closure in (nothing, model.closure, InertClosure(), (model.closure, InertClosure()))
            closure_fields = closure isa Tuple ? (model.closure_fields, nothing) : model.closure_fields
            for coefficient in (closure_implicit_upper_diagonal, closure_implicit_lower_diagonal,
                                closure_implicit_linear_coefficient)
                @test @allowscalar(coefficient(1, 1, 3, grid, closure, closure_fields, Val(1), args...)) === zero(eltype(grid))
            end
            # The derived diagonal is -0, an exact identity under addition
            @test @allowscalar(closure_implicit_diagonal(1, 1, 3, grid, closure, closure_fields, Val(1), args...)) === -zero(eltype(grid))
        end

        # A closure that does not extend the hook leaves the TKE closure's solution bitwise unchanged
        baseline = upwind_column_model(TKEBasedTurbulenceClosure())
        augmented = upwind_column_model((TKEBasedTurbulenceClosure(), InertClosure()))
        for _ in 1:5
            time_step!(baseline, 60)
            time_step!(augmented, 60)
        end
        @test column(augmented.tracers.ρc) == column(baseline.tracers.ρc)
        @test column(augmented.tracers.ρe) == column(baseline.tracers.ρe)
        @test column(thermodynamic_density(augmented.formulation)) ==
              column(thermodynamic_density(baseline.formulation))
    end

    # The Courant number |w| Δt / Δz is 4: far beyond the explicit limit
    for w in (-1.0, 1.0)
        @testset "upwind transport at w = $w matches the implicit upwind solution" begin
            Δt = 1000.0
            name = :ρc
            # The closure acts on ρc alone, whose index among the closure scalars it is built with
            index_model = upwind_column_model(nothing)
            I = typeof(closure_scalar_index(index_model, name)).parameters[1]
            closure = UpwindTransport{I}(w)

            model = upwind_column_model(closure)
            grid = model.grid
            ρ = implicit_advection_density(model.dynamics, model.formulation, name)

            q⁰ = column(model.tracers.ρc)
            A = upwind_matrix(grid, ρ, w, Δt)
            expected = A \ q⁰

            implicit_stage_step!(model, name, model.tracers.ρc, Δt)
            q¹ = column(model.tracers.ρc)

            Δz = [Δzᶜᶜᶜ(1, 1, k, grid) for k in 1:grid.Nz]
            @test q¹ ≈ expected rtol = 1e-12
            @test all(q¹ .≥ 0)
            @test sum(q¹ .* Δz) ≈ sum(q⁰ .* Δz) rtol = 1e-12

            # The center of mass moves with the velocity
            z = Array(znodes(grid, Center()))
            @test sign(sum(z .* q¹ .* Δz) - sum(z .* q⁰ .* Δz)) == sign(w)

            # A full SSP-RK3 step composes three implicit solves over Δt, Δt / 4 and 2Δt / 3
            model = upwind_column_model(closure)
            time_step!(model, Δt)
            S(α) = upwind_matrix(grid, ρ, w, α * Δt)
            q⁽¹⁾ = S(1) \ q⁰
            q⁽²⁾ = S(1/4) \ (3/4 * q⁰ + 1/4 * q⁽¹⁾)
            q⁽³⁾ = S(2/3) \ (1/3 * q⁰ + 2/3 * q⁽²⁾)
            @test column(model.tracers.ρc) ≈ q⁽³⁾ rtol = 1e-10
        end
    end

    # A divergent return flow, w sin(π (k - 1) / Nz), at a peak Courant number of 4
    for w in (-1.0, 1.0)
        @testset "divergent upwind transport at w = $w conserves the column" begin
            Δt = 1000.0
            name = :ρc
            index_model = upwind_column_model(nothing)
            I = typeof(closure_scalar_index(index_model, name)).parameters[1]

            model = upwind_column_model(UpwindTransport{I}(w, Sinusoidal()))
            grid = model.grid
            ρ = implicit_advection_density(model.dynamics, model.formulation, name)
            Δz = [Δzᶜᶜᶜ(1, 1, k, grid) for k in 1:grid.Nz]

            q⁰ = column(model.tracers.ρc)
            expected = upwind_matrix(grid, ρ, w, Δt; profile = Sinusoidal()) \ q⁰
            implicit_stage_step!(model, name, model.tracers.ρc, Δt)
            q¹ = column(model.tracers.ρc)

            @test q¹ ≈ expected rtol = 1e-12
            @test all(q¹ .≥ 0)
            @test sum(q¹ .* Δz) ≈ sum(q⁰ .* Δz) rtol = 1e-13

            # Without the linear coefficient the solve is in advective form: it does not conserve
            advective = upwind_column_model(UpwindTransport{I}(w, Sinusoidal(); conservative = false))
            implicit_stage_step!(advective, name, advective.tracers.ρc, Δt)
            q¹ₐ = column(advective.tracers.ρc)
            @test abs(sum(q¹ₐ .* Δz) - sum(q⁰ .* Δz)) > 1e-3 * sum(q⁰ .* Δz)

            # ...but its rows sum to one, so it preserves a constant q
            set!(advective; ρc = 1)
            q⁰ₐ = column(advective.tracers.ρc)
            implicit_stage_step!(advective, name, advective.tracers.ρc, Δt)
            @test column(advective.tracers.ρc) ≈ q⁰ₐ rtol = 1e-12
        end
    end
end
