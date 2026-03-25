using Breeze
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: xnodes, znodes
using Enzyme
using CUDA
using Reactant
using CairoMakie
Reactant.allowscalar(true)

# ══════════════════════════════════════════════════════════════════════════════
# Generic adjoint accumulation infrastructure (from viz_mwe.jl)
#
# The user provides:
#   step!(model)          — mutates model state in-place
#   loss(model) -> scalar — terminal objective
#   get_state(model)      — extract the state as a flat 1-D vector (copy)
#   set_state!(model, u)  — restore the state from a flat 1-D vector
# ══════════════════════════════════════════════════════════════════════════════

function get_state end
function set_state! end

function collect_adjoints(model, u0, K, step!_fn, loss_fn)
    set_state!(model, u0)

    @info "  Compiling forward step …"
    compiled_step! = @compile raise=true raise_first=true donated_args=:none step!_fn(model)

    function grad_loss(m, dm)
        _, J = Enzyme.autodiff(
            Enzyme.ReverseWithPrimal, Enzyme.Const(loss_fn), Enzyme.Active,
            Enzyme.Duplicated(m, dm)
        )
        return J
    end
    @info "  Compiling terminal gradient (grad_loss) …"
    compiled_grad_loss = @compile raise=true raise_first=true donated_args=:none grad_loss(model, Enzyme.make_zero(model))

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
    @info "  Compiling VJP kernel …"
    compiled_vjp = @compile raise=true raise_first=true donated_args=:none vjp_kernel(
        Reactant.to_rarray(Array(u0)),
        Enzyme.make_zero(u0),
        Reactant.to_rarray(ones(Float64, size(Array(u0)))),
        model,
        Enzyme.make_zero(model),
    )

    # ── forward sweep ───────────────────────────────────────────────────────
    @info "  Forward sweep: $K steps …"
    snapshots = Vector{Vector{Float64}}(undef, K + 1)
    snapshots[1] = Array(u0)
    for k in 1:K
        compiled_step!(model)
        snapshots[k + 1] = Array(get_state(model))
        @info "    step $k/$K done"
    end

    # ── terminal adjoint ────────────────────────────────────────────────────
    @info "  Computing terminal adjoint (dg/d state_K) …"
    λs = Vector{Vector{Float64}}(undef, K + 1)
    set_state!(model, snapshots[end])
    dmodel = Enzyme.make_zero(model)
    compiled_grad_loss(model, dmodel)
    λs[K + 1] = Array(get_state(dmodel))

    # ── backward sweep ──────────────────────────────────────────────────────
    @info "  Backward sweep: $K VJP steps …"
    for k in K:-1:1
        u_prev_r = Reactant.to_rarray(snapshots[k])
        du_r     = Reactant.to_rarray(zeros(Float64, size(Array(u0))))
        λ_r      = Reactant.to_rarray(λs[k + 1])
        dm_r     = Enzyme.make_zero(model)
        compiled_vjp(u_prev_r, du_r, λ_r, model, dm_r)
        λs[k] = Array(du_r)
        @info "    VJP step $k/$K done"
    end

    @info "  Adjoint accumulation complete."
    return snapshots, λs
end

# ══════════════════════════════════════════════════════════════════════════════
# AtmosphereModel interface — flatten/unflatten prognostic fields
#
# Prognostic field order (CompressibleDynamics + LiquidIcePotentialTemperature):
#   ρ, ρu, ρv, ρw, ρθ, ρqᵛ
#
# All are (Nx, 1, Nz) except ρw which is (Nx, 1, Nz+1) for Bounded z.
# ══════════════════════════════════════════════════════════════════════════════

function get_state(model::AtmosphereModel)
    return vcat(
        vec(copy(interior(model.dynamics.density))),
        vec(copy(interior(model.momentum.ρu))),
        vec(copy(interior(model.momentum.ρv))),
        vec(copy(interior(model.momentum.ρw))),
        vec(copy(interior(model.formulation.potential_temperature_density))),
        vec(copy(interior(model.moisture_density))),
    )
end

function set_state!(model::AtmosphereModel, u)
    sz_c = size(interior(model.dynamics.density))
    sz_w = size(interior(model.momentum.ρw))
    sz_v = size(interior(model.momentum.ρv))
    nc = prod(sz_c)
    nw = prod(sz_w)
    nv = prod(sz_v)

    o = 0
    interior(model.dynamics.density) .= reshape(u[o+1:o+nc], sz_c);                                     o += nc
    interior(model.momentum.ρu)      .= reshape(u[o+1:o+nc], sz_c);                                     o += nc
    interior(model.momentum.ρv)      .= reshape(u[o+1:o+nv], sz_v);                                     o += nv
    interior(model.momentum.ρw)      .= reshape(u[o+1:o+nw], sz_w);                                     o += nw
    interior(model.formulation.potential_temperature_density) .= reshape(u[o+1:o+nc], sz_c);             o += nc
    interior(model.moisture_density) .= reshape(u[o+1:o+nc], sz_c)
    return nothing
end

"""Compute byte offsets into the flat state vector for each prognostic field."""
function state_layout(model)
    nc = prod(size(interior(model.dynamics.density)))
    nv = prod(size(interior(model.momentum.ρv)))
    nw = prod(size(interior(model.momentum.ρw)))
    offsets = Dict{Symbol, UnitRange{Int}}()
    o = 0
    offsets[:ρ]   = (o+1):(o+nc); o += nc
    offsets[:ρu]  = (o+1):(o+nc); o += nc
    offsets[:ρv]  = (o+1):(o+nv); o += nv
    offsets[:ρw]  = (o+1):(o+nw); o += nw
    offsets[:ρθ]  = (o+1):(o+nc); o += nc
    offsets[:ρqᵛ] = (o+1):(o+nc); o += nc
    return offsets
end

# ══════════════════════════════════════════════════════════════════════════════
# Model setup: warm bubble in a compressible atmosphere
# ══════════════════════════════════════════════════════════════════════════════

Nx, Nz = 32, 16
Lx, Lz = 1000.0, 500.0

@info "Building grid (Nx=$Nx, Nz=$Nz) and AtmosphereModel …"
grid = RectilinearGrid(ReactantState();
    size = (Nx, Nz),
    x = (-Lx/2, Lx/2), z = (0, Lz),
    topology = (Periodic, Flat, Bounded))

model = AtmosphereModel(grid; dynamics = CompressibleDynamics(ExplicitTimeStepping()))

constants = model.thermodynamic_constants
θ₀  = 300.0
p₀  = 101325.0
pˢᵗ = 1e5

Rᵈ  = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
γ   = cᵖᵈ / (cᵖᵈ - Rᵈ)
ℂᵃᶜ = sqrt(γ * Rᵈ * θ₀)

Δθ = 2.0
zc = Lz / 2
σ  = 50.0

θᵢ(x, z) = θ₀ + Δθ * exp(-(x^2 + (z - zc)^2) / (2σ^2))
ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)

@info "Setting initial conditions: Gaussian θ perturbation (Δθ=$Δθ K, σ=$σ m, zc=$zc m)"
set!(model; ρ = ρᵢ, θ = θᵢ)

# ══════════════════════════════════════════════════════════════════════════════
# Step and loss
# ══════════════════════════════════════════════════════════════════════════════

CFL = 0.3
Δx  = Lx / Nx
Δz  = Lz / Nz
Δt  = CFL * min(Δx, Δz) / ℂᵃᶜ

breeze_step!(m) = time_step!(m, Δt)

function breeze_loss(m)
    ρθ = interior(m.formulation.potential_temperature_density)
    ρ  = interior(m.dynamics.density)
    θ  = ρθ ./ ρ
    return sum(θ .^ 2)
end

# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

K  = 5
u0 = get_state(model)

@info "State vector length: $(length(Array(u0)))"
@info "Starting adjoint accumulation: K=$K steps, Nx=$Nx, Nz=$Nz, Δt=$(round(Δt; sigdigits=3))s"
snapshots, λs = collect_adjoints(model, u0, K, breeze_step!, breeze_loss)

# ══════════════════════════════════════════════════════════════════════════════
# Visualize
# ══════════════════════════════════════════════════════════════════════════════

@info "Preparing visualization …"
layout = state_layout(model)
sz_c   = (Nx, Nz)

function slice_θ(snap)
    ρ  = reshape(snap[layout[:ρ]],  sz_c)
    ρθ = reshape(snap[layout[:ρθ]], sz_c)
    return ρθ ./ ρ
end

slice_λρθ(λ) = reshape(λ[layout[:ρθ]], sz_c)

xc = Array(xnodes(grid, Center()))
zc_nodes = Array(znodes(grid, Center()))

plot_ks = [1, max(1, K ÷ 2), K + 1]

fig = Figure(; size = (1000, 400 * length(plot_ks)))

for (row, k) in enumerate(plot_ks)
    θ_field  = slice_θ(snapshots[k]) .- θ₀
    λ_field  = slice_λρθ(λs[k])

    ax1 = Axis(fig[row, 1]; xlabel = "x [m]", ylabel = "z [m]",
               title = "θ′ = θ − θ₀  (k=$(k-1))")
    hm1 = heatmap!(ax1, xc, zc_nodes, θ_field)
    Colorbar(fig[row, 2], hm1)

    ax2 = Axis(fig[row, 3]; xlabel = "x [m]", ylabel = "z [m]",
               title = "λ_ρθ = dg/dρθ  (k=$(k-1))")
    hm2 = heatmap!(ax2, xc, zc_nodes, λ_field)
    Colorbar(fig[row, 4], hm2)
end

save("compressible_adjoint_viz.png", fig)
@info "Saved compressible_adjoint_viz.png"
