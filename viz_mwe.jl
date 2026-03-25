using Enzyme
using Reactant
using CairoMakie
Reactant.allowscalar(true)

# ══════════════════════════════════════════════════════════════════════════════
# Generic adjoint accumulation infrastructure (Reactant-compiled)
# ══════════════════════════════════════════════════════════════════════════════

function get_state end
function set_state! end

function collect_adjoints(model, u0, K, step!_fn, loss_fn)
    set_state!(model, u0)

    compiled_step! = @compile step!_fn(model)

    function grad_loss(m, dm)
        _, J = Enzyme.autodiff(
            Enzyme.ReverseWithPrimal, Enzyme.Const(loss_fn), Enzyme.Active,
            Enzyme.Duplicated(m, dm)
        )
        return J
    end
    compiled_grad_loss = @compile raise=true raise_first=true grad_loss(model, Enzyme.make_zero(model))

    function vjp_kernel(u_prev, du, λ, m, dm)
        set_state!(m, u_prev)
        step!_fn(m)
        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const((x, v, mdl) -> begin
                set_state!(mdl, x)
                step!_fn(mdl)
                return sum(get_state(mdl) .* v)
            end),
            Enzyme.Active,
            Enzyme.Duplicated(u_prev, du),
            Enzyme.Const(λ),
            Enzyme.Duplicated(m, dm)
        )
        return nothing
    end
    compiled_vjp = @compile raise=true raise_first=true vjp_kernel(
        Reactant.to_rarray(Array(u0)),
        Enzyme.make_zero(u0),
        Reactant.to_rarray(ones(Float64, size(Array(u0)))),
        model,
        Enzyme.make_zero(model),
    )

    # ── forward sweep ───────────────────────────────────────────────────────
    snapshots = Vector{Vector{Float64}}(undef, K + 1)
    snapshots[1] = Array(u0)
    for k in 1:K
        compiled_step!(model)
        snapshots[k + 1] = Array(get_state(model))
    end

    # ── terminal adjoint ────────────────────────────────────────────────────
    λs = Vector{Vector{Float64}}(undef, K + 1)
    set_state!(model, snapshots[end])
    dmodel = Enzyme.make_zero(model)
    compiled_grad_loss(model, dmodel)
    λs[K + 1] = Array(get_state(dmodel))

    # ── backward sweep ──────────────────────────────────────────────────────
    for k in K:-1:1
        u_prev_r = Reactant.to_rarray(snapshots[k])
        du_r     = Reactant.to_rarray(zeros(Float64, size(Array(u0))))
        λ_r      = Reactant.to_rarray(λs[k + 1])
        dm_r     = Enzyme.make_zero(model)
        compiled_vjp(u_prev_r, du_r, λ_r, model, dm_r)
        λs[k] = Array(du_r)
    end

    return snapshots, λs
end

# ══════════════════════════════════════════════════════════════════════════════
# Toy model — swap this out for any model that implements the interface
# ══════════════════════════════════════════════════════════════════════════════

mutable struct ToyModel{A<:AbstractVector}
    u::A
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
K  = 10
κ  = 0.1
u0 = [exp(-((i - N/2)^2) / (2 * 5^2)) for i in 1:N]

u0_r  = Reactant.to_rarray(u0)
model = ToyModel(Reactant.to_rarray(copy(u0)), κ)
snapshots, λs = collect_adjoints(model, u0_r, K, toy_step!, toy_loss)

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
