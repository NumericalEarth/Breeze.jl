using Enzyme
using CairoMakie

# ══════════════════════════════════════════════════════════════════════════════
# Generic adjoint accumulation infrastructure
#
# The user provides:
#   step!(model)          — mutates model state in-place
#   loss(model) -> scalar — terminal objective
#   get_state(model)      — extract the state vector (returns a copy)
#   set_state!(model, u)  — restore the state vector
# ══════════════════════════════════════════════════════════════════════════════

function vjp_step(model, λ, step!_fn)
    u_prev = get_state(model)

    function f_dot_λ(u0, v, m)
        set_state!(m, u0)
        step!_fn(m)
        s = get_state(m)
        return sum(s .* v)
    end

    du = zero(u_prev)
    Enzyme.autodiff(
        Enzyme.Reverse, Enzyme.Const(f_dot_λ), Enzyme.Active,
        Enzyme.Duplicated(copy(u_prev), du),
        Enzyme.Const(λ),
        Enzyme.Duplicated(model, Enzyme.make_zero(model))
    )

    set_state!(model, u_prev)
    return du
end

function collect_adjoints(model, u0, K, step!_fn, loss_fn)
    set_state!(model, u0)

    snapshots = Vector{typeof(u0)}(undef, K + 1)
    snapshots[1] = copy(u0)
    for k in 1:K
        step!_fn(model)
        snapshots[k + 1] = get_state(model)
    end

    λs = Vector{typeof(u0)}(undef, K + 1)

    set_state!(model, snapshots[end])
    dmodel = Enzyme.make_zero(model)
    Enzyme.autodiff(
        Enzyme.Reverse, Enzyme.Const(loss_fn), Enzyme.Active,
        Enzyme.Duplicated(model, dmodel)
    )
    λs[K + 1] = get_state(dmodel)

    for k in K:-1:1
        set_state!(model, snapshots[k])
        λs[k] = vjp_step(model, λs[k + 1], step!_fn)
    end

    return snapshots, λs
end

# ══════════════════════════════════════════════════════════════════════════════
# Toy model — swap this out for any model that implements the interface above
# ══════════════════════════════════════════════════════════════════════════════

mutable struct ToyModel
    u::Vector{Float64}
    κ::Float64
end

get_state(m::ToyModel) = copy(m.u)
set_state!(m::ToyModel, u) = (m.u .= u; nothing)

function toy_step!(model::ToyModel)
    u = model.u
    κ = model.κ
    N = length(u)
    u_new = similar(u)
    for i in 1:N
        ip = mod1(i + 1, N)
        im = mod1(i - 1, N)
        u_new[i] = u[i] + κ * (u[ip] - 2u[i] + u[im]) + 0.05 * sin(u[i])
    end
    model.u .= u_new
    return nothing
end

toy_loss(model::ToyModel) = sum(model.u .^ 2)

# ── Run and plot ─────────────────────────────────────────────────────────────

N  = 64
K  = 100
κ  = 0.1
u0 = [exp(-((i - N/2)^2) / (2 * 5^2)) for i in 1:N]

model = ToyModel(copy(u0), κ)
snapshots, λs = collect_adjoints(model, u0, K, toy_step!, toy_loss)

fig = Figure(; size=(900, 400))

ax1 = Axis(fig[1, 1]; xlabel="grid index", ylabel="u", title="Forward states uₖ")
for k in [1, K ÷ 4, K ÷ 2, 3K ÷ 4, K + 1]
    lines!(ax1, snapshots[k]; label="k=$(k-1)")
end
axislegend(ax1; position=:rt)

ax2 = Axis(fig[1, 2]; xlabel="grid index", ylabel="dg/duₖ", title="Adjoint accumulation λₖ = dg/duₖ")
for k in [K + 1, 3K ÷ 4 + 1, K ÷ 2 + 1, K ÷ 4 + 1, 1]
    lines!(ax2, λs[k]; label="k=$(k-1)")
end
axislegend(ax2; position=:rt)

save("adjoint_accumulation.png", fig)
