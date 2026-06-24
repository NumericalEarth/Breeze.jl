using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using Oceananigans.Grids: xnodes, ynodes
using CUDA
using Reactant
using Reactant: @trace
using GPUArraysCore: @allowscalar
using Enzyme
using Test

if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
end

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

#####
##### Physical parameters
#####

const _A   = 2.0
const _σ₀  = 20.0
const _κ   = 40.0
const _U₀  = 10.0
const _L   = 200.0
const _t_f = 1.0
const _CFL = 0.5
const _c_s = sqrt(1.4 * 287.0 * 300.0)

#####
##### Analytical reference
#####
#
# For ∂T/∂t + U₀ ∂T/∂x = κ ∇²T with Gaussian IC, the L² objective
# J = ∫ T² dx dy = π A² σ₀⁴ / s_f (s_f = σ₀² + 2κ t_f) has exact gradients:
#   ∂J/∂A  = 2J/A
#   ∂J/∂σ₀ = 2π A² σ₀³ (σ₀² + 4κ t_f) / s_f²
#   ∂J/∂U₀ = 0  (L² norm is translation-invariant)

function analytical_values(A, σ₀, κ, t_f)
    s_f    = σ₀^2 + 2κ * t_f
    J      = π * A^2 * σ₀^4 / s_f
    ∂J_∂A  =  2π * A   * σ₀^4 / s_f
    ∂J_∂σ₀ =  2π * A^2 * σ₀^3 * (σ₀^2 + 4κ * t_f) / s_f^2
    ∂J_∂U₀ =  0.0
    return (; J, ∂J_∂A, ∂J_∂σ₀, ∂J_∂U₀)
end

#####
##### Model builders and loss
#####

function build_case(N, L, κ)
    grid = RectilinearGrid(ReactantState();
        size = (N, N), x = (-L/2, L/2), y = (-L/2, L/2),
        topology = (Periodic, Periodic, Flat))

    model = AtmosphereModel(grid;
        dynamics  = CompressibleDynamics(),
        advection = WENO(order = 5),
        closure   = ScalarDiffusivity(κ = Float64(κ)),
        tracers   = :ρc)

    T⁰  = CenterField(grid)
    dT⁰ = CenterField(grid)
    set!(dT⁰, 0.0)
    dmodel = Enzyme.make_zero(model)

    xc = Reactant.to_rarray(Array(xnodes(grid, Center())))
    yc = Reactant.to_rarray(Array(ynodes(grid, Center())))

    return model, dmodel, T⁰, dT⁰, xc, yc, L / N
end

function loss(model, T⁰, θ, xc, yc, Δt, Nₛ, Δx)
    A_  = @allowscalar θ[1]
    σ₀_ = @allowscalar θ[2]
    U₀_ = @allowscalar θ[3]

    X = reshape(xc, :, 1)
    Y = reshape(yc, 1, :)
    r² = X .^ 2 .+ Y .^ 2
    T_vals = A_ .* exp.(-r² ./ (2 * σ₀_^2))
    interior(T⁰) .= reshape(T_vals, size(interior(T⁰)))

    set!(model; ρc = T⁰, ρ = 1.0, θ = 300.0, u = U₀_, v = 0.0, w = 0.0)
    @trace track_numbers = false mincut = true checkpointing = false for _ in 1:Nₛ
        time_step!(model, Δt)
    end
    return Δx^2 * sum(interior(model.tracers.ρc) .^ 2)
end

function grad_loss(model, dmodel, T⁰, dT⁰, θ, dθ, xc, yc, Δt, Nₛ, Δx)
    parent(dT⁰) .= 0
    dθ .= 0
    _, J = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(T⁰,   dT⁰),
        Enzyme.Duplicated(θ, dθ),
        Enzyme.Const(xc),
        Enzyme.Const(yc),
        Enzyme.Const(Δt),
        Enzyme.Const(Nₛ),
        Enzyme.Const(Δx))
    return dθ, J
end

#####
##### Test
#####

@testset "advection_diffusion_ad" begin
    N_list = [16, 32, 64]
    exact  = analytical_values(_A, _σ₀, _κ, _t_f)

    @show @__LINE__

    results = map(N_list) do N
        @show N

        Δx = _L / N
        Δt = _CFL * Δx / (_c_s + _U₀)
        Nₛ = ceil(Int, _t_f / Δt)
        Δt = _t_f / Nₛ

        @show @__LINE__

        model, dmodel, T⁰, dT⁰, xc, yc, _ = build_case(N, _L, _κ)
        θ  = Reactant.to_rarray(Float64[_A, _σ₀, _U₀])
        dθ = Reactant.to_rarray(zeros(3))

        @show @__LINE__

        compiled = Reactant.@compile raise_first = true raise = true sync = true grad_loss(
            model, dmodel, T⁰, dT⁰, θ, dθ, xc, yc, Δt, Nₛ, Δx)

        @show @__LINE__

        dθ_result, J_ad = compiled(model, dmodel, T⁰, dT⁰, θ, dθ, xc, yc, Δt, Nₛ, Δx)
        J_ad  = Float64(J_ad)
        grads = Array(dθ_result)

        @show @__LINE__

        rel_J  = abs(J_ad - exact.J) / abs(exact.J)
        rel_A  = abs(grads[1] - exact.∂J_∂A)  / abs(exact.∂J_∂A)
        rel_σ₀ = abs(grads[2] - exact.∂J_∂σ₀) / abs(exact.∂J_∂σ₀)
        rel_U₀ = abs(grads[3]) / abs(exact.J / _U₀)

        @show @__LINE__

        (; N, J_ad, grads, rel_J, rel_A, rel_σ₀, rel_U₀)
    end

    @show @__LINE__
end
