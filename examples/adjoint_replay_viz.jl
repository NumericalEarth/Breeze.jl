using Breeze
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: xnodes, znodes
using Oceananigans.Fields: CenterField, XFaceField
using Enzyme
using CUDA
using Reactant
using Reactant: @trace
using CairoMakie
Reactant.allowscalar(true)

# ══════════════════════════════════════════════════════════════════════════════
# Model setup — same numerics as compressible_adjoint_viz.jl
# ══════════════════════════════════════════════════════════════════════════════

Nx, Nz = 128, 64
Lx, Lz = 1000.0, 200.0

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

U₀ = 20.0
ℓ  = 1.0
Uᵢ(z) = U₀ * log((z + ℓ) / ℓ)

δρ = 0.01
σ  = 20.0

gaussian(x, z) = exp(-(x^2 + z^2) / (2σ^2))
ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants) + δρ * gaussian(x, z)
uᵢ(x, z) = Uᵢ(z)

CFL = 0.3
Δx  = Lx / Nx
Δz  = Lz / Nz
Δt  = CFL * min(Δx, Δz) / (ℂᵃᶜ + Uᵢ(Lz))

K = 25

target_i = round(Int, 0.75Nx)
target_k = round(Int, 0.35Nz)

@info "Parameters: K=$K steps, Δt=$(round(Δt; sigdigits=3))s, CFL=$CFL"
@info "Observation point: grid index ($target_i, $target_k)"

# ══════════════════════════════════════════════════════════════════════════════
# State interface (same as compressible_adjoint_viz.jl)
# ══════════════════════════════════════════════════════════════════════════════

function get_state(m::AtmosphereModel)
    return vcat(
        vec(copy(interior(m.dynamics.density))),
        vec(copy(interior(m.momentum.ρu))),
        vec(copy(interior(m.momentum.ρv))),
        vec(copy(interior(m.momentum.ρw))),
        vec(copy(interior(m.formulation.potential_temperature_density))),
        vec(copy(interior(m.moisture_density))),
    )
end

function state_layout(m)
    nc = prod(size(interior(m.dynamics.density)))
    nv = prod(size(interior(m.momentum.ρv)))
    nw = prod(size(interior(m.momentum.ρw)))
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

breeze_step!(m) = time_step!(m, Δt)

function breeze_loss(m)
    return interior(m.dynamics.density)[target_i, 1, target_k]^2
end

@info "Setting initial conditions …"
set!(model; ρ = ρᵢ, θ = θ₀, u = uᵢ)

u0 = get_state(model)
N_state = length(Array(u0))
layout  = state_layout(model)
sz_c    = (Nx, Nz)

@info "State vector length: $N_state"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Compile kernels
#
# We need three compiled kernels:
#   1. A forward step (for replay and the forward sweep)
#   2. The terminal gradient  dJ/d(model state at final time)
#   3. A VJP kernel that does NOT use set_state! internally — instead,
#      the model is brought to the correct state by replaying forward
#      BEFORE calling the VJP.
# ══════════════════════════════════════════════════════════════════════════════

@info "Phase 1 — Compiling forward step …"
compiled_step! = @compile raise=true raise_first=true donated_args=:none time_step!(model, Δt)

@info "Phase 1 — Compiling terminal gradient …"
function grad_loss(m, dm)
    _, J = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, Enzyme.Const(breeze_loss), Enzyme.Active,
        Enzyme.Duplicated(m, dm))
    return J
end
dmodel = Enzyme.make_zero(model)
compiled_grad_loss = @compile raise=true raise_first=true donated_args=:none grad_loss(model, dmodel)

@info "Phase 1 — Compiling VJP kernel …"
function vjp_step!(m, dm, λ_next)
    Enzyme.autodiff(
        Enzyme.Reverse,
        Enzyme.Const((mdl, v) -> begin
            breeze_step!(mdl)
            return sum(get_state(mdl) .* v)
        end),
        Enzyme.Active,
        Enzyme.Duplicated(m, dm),
        Enzyme.Const(λ_next))
    return nothing
end
compiled_vjp_step = @compile raise=true raise_first=true donated_args=:none vjp_step!(
    model,
    Enzyme.make_zero(model),
    Reactant.to_rarray(ones(Float64, N_state)))

@info "Phase 1 complete — all kernels compiled."

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Forward sweep — save density snapshots for visualization
# ══════════════════════════════════════════════════════════════════════════════

@info "Phase 2 — Forward sweep ($K steps) …"
set!(model; ρ = ρᵢ, θ = θ₀, u = uᵢ)

fwd_ρ = Vector{Matrix{Float64}}(undef, K + 1)
fwd_ρ[1] = Array(interior(model.dynamics.density, :, 1, :))

for k in 1:K
    compiled_step!(model, Δt)
    fwd_ρ[k + 1] = Array(interior(model.dynamics.density, :, 1, :))
    @info "  step $k/$K done"
end
@info "Phase 2 complete."

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Terminal adjoint  λ_K = dJ/d(u_K)
# ══════════════════════════════════════════════════════════════════════════════

@info "Phase 3 — Computing terminal adjoint …"
dmodel = Enzyme.make_zero(model)
compiled_grad_loss(model, dmodel)
λ_K = Array(get_state(dmodel))
@info "  ‖λ_K‖∞ = $(maximum(abs, λ_K))"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Backward sweep with O(K²) replay
#
# For each k = K-1, K-2, …, 0:
#   1. Replay forward from IC  (set! → k compiled steps)
#      This reconstructs ALL model state at step k, including internal
#      tendencies, diagnostics, and any multi-stage RK buffers.
#   2. Call the compiled VJP with λ_{k+1} to get λ_k.
#
# Total cost: O(K²) forward steps + K VJP calls.
# ══════════════════════════════════════════════════════════════════════════════

@info "Phase 4 — Backward sweep with O(K²) replay …"

λs = Vector{Vector{Float64}}(undef, K + 1)
λs[K + 1] = λ_K

for k in K:-1:1
    set!(model; ρ = ρᵢ, θ = θ₀, u = uᵢ)
    for j in 1:k-1
        compiled_step!(model, Δt)
    end

    local dm = Enzyme.make_zero(model)
    local λ_r = Reactant.to_rarray(λs[k + 1])
    compiled_vjp_step(model, dm, λ_r)
    λs[k] = Array(get_state(dm))

    local λ_ρ = @view λs[k][layout[:ρ]]
    @info "  k=$k  replay=$(k-1) steps  ‖λ‖₁=$(round(sum(abs, λs[k]); sigdigits=4))  ‖λ‖∞=$(round(maximum(abs, λs[k]); sigdigits=4))  ‖λ_ρ‖∞=$(round(maximum(abs, λ_ρ); sigdigits=4))"
end

@info "Phase 4 complete — backward sweep done."

# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: e2e gradient for comparison at k=0
# ══════════════════════════════════════════════════════════════════════════════

@info "Phase 5 — Building fresh model for e2e gradient …"

grid_ad = RectilinearGrid(ReactantState();
    size = (Nx, Nz),
    x = (-Lx/2, Lx/2), z = (0, Lz),
    topology = (Periodic, Flat, Bounded))
model_ad = AtmosphereModel(grid_ad; dynamics = CompressibleDynamics(ExplicitTimeStepping()))

δρᵢ_field  = CenterField(grid_ad)
dδρᵢ_field = CenterField(grid_ad)
set!(δρᵢ_field, (x, z) -> δρ * gaussian(x, z))
set!(dδρᵢ_field, 0)

ρᵇᵍ_field = CenterField(grid_ad)
uᵇᵍ_field = XFaceField(grid_ad)
set!(ρᵇᵍ_field, (x, z) -> adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants))
set!(uᵇᵍ_field, (x, z) -> Uᵢ(z))

ρᵗ_field  = CenterField(grid_ad)
dρᵗ_field = CenterField(grid_ad)
dmodel_ad = Enzyme.make_zero(model_ad)

function e2e_loss(mdl, δρ_f, ρᵗ_f, ρᵇᵍ_f, uᵇᵍ_f, θ_ref, dt, nsteps, it, kt)
    interior(ρᵗ_f) .= interior(ρᵇᵍ_f) .+ interior(δρ_f)
    set!(mdl; ρ = ρᵗ_f, θ = θ_ref, u = uᵇᵍ_f, v = 0.0, w = 0.0)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(mdl, dt)
    end
    return @allowscalar interior(mdl.dynamics.density)[it, 1, kt]^2
end

function e2e_grad(mdl, dmdl, δρ_f, dδρ_f,
                  ρᵗ_f, dρᵗ_f, ρᵇᵍ_f, uᵇᵍ_f, θ_ref, dt, nsteps, it, kt)
    parent(dδρ_f) .= 0
    parent(dρᵗ_f) .= 0
    _, J = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        e2e_loss, Enzyme.Active,
        Enzyme.Duplicated(mdl, dmdl),
        Enzyme.Duplicated(δρ_f, dδρ_f),
        Enzyme.Duplicated(ρᵗ_f, dρᵗ_f),
        Enzyme.Const(ρᵇᵍ_f),
        Enzyme.Const(uᵇᵍ_f),
        Enzyme.Const(θ_ref),
        Enzyme.Const(dt),
        Enzyme.Const(nsteps),
        Enzyme.Const(it),
        Enzyme.Const(kt))
    return dδρ_f, J
end

@info "Phase 5 — Compiling e2e gradient …"
compiled_e2e_grad = Reactant.@compile raise=true raise_first=true sync=true e2e_grad(
    model_ad, dmodel_ad, δρᵢ_field, dδρᵢ_field,
    ρᵗ_field, dρᵗ_field, ρᵇᵍ_field, uᵇᵍ_field, θ₀, Δt, K, target_i, target_k)

@info "Phase 5 — Running e2e gradient …"
dδρ_result, J_e2e = compiled_e2e_grad(
    model_ad, dmodel_ad, δρᵢ_field, dδρᵢ_field,
    ρᵗ_field, dρᵗ_field, ρᵇᵍ_field, uᵇᵍ_field, θ₀, Δt, K, target_i, target_k)

e2e_sensitivity = Array(interior(dδρ_result, :, 1, :))
@info "Phase 5 complete — J = $(Float64(only(J_e2e)))  ‖∂J/∂δρ‖∞ = $(maximum(abs, e2e_sensitivity))"

# ── Reconstruct total dJ/d(δρ) from manual adjoint components at k=0 ─────

@info "Reconstructing total dJ/d(δρ) from replay adjoint at k=0 …"

slice_field(λ, rng) = reshape(λ[rng], sz_c)

replay_λρ  = slice_field(λs[1], layout[:ρ])
replay_λρu = slice_field(λs[1], layout[:ρu])
replay_λρθ = slice_field(λs[1], layout[:ρθ])

xc = Array(xnodes(grid, Center()))
zc_nodes = Array(znodes(grid, Center()))
u0_field = [Uᵢ(z) for _ in xc, z in zc_nodes]

reconstructed = replay_λρ .+ replay_λρu .* u0_field .+ replay_λρθ .* θ₀

@info "  ‖λ_ρ‖∞          = $(maximum(abs, replay_λρ))"
@info "  ‖λ_{ρu}·u₀‖∞    = $(maximum(abs, replay_λρu .* u0_field))"
@info "  ‖λ_{ρθ}·θ₀‖∞    = $(maximum(abs, replay_λρθ .* θ₀))"
@info "  ‖reconstructed‖∞ = $(maximum(abs, reconstructed))"
@info "  ‖e2e‖∞           = $(maximum(abs, e2e_sensitivity))"

Δgrad = reconstructed .- e2e_sensitivity
@info "  ‖recon − e2e‖∞   = $(maximum(abs, Δgrad))"

# ══════════════════════════════════════════════════════════════════════════════
# Visualize
# ══════════════════════════════════════════════════════════════════════════════

@info "Preparing visualization …"

x_obs = xc[target_i]
z_obs = zc_nodes[target_k]

ρ_bg = [adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants) for _ in xc, z in zc_nodes]
ρ_fields = [fwd_ρ[k] .- ρ_bg for k in 1:K+1]
λ_fields = [slice_field(λs[k], layout[:ρ]) for k in 1:K+1]

domain_aspect = Lx / Lz

ρ_cmax = maximum(maximum(abs, f) for f in ρ_fields)
λ_cmax = maximum(maximum(abs, f) for f in λ_fields)
ρ_clims = (-ρ_cmax, ρ_cmax)
λ_clims = (-λ_cmax, λ_cmax)

# ── Static snapshots ──────────────────────────────────────────────────────

plot_ks = unique([1, max(1, K ÷ 4), max(1, K ÷ 2), max(1, 3K ÷ 4), K + 1])
n_snaps = length(plot_ks)

@info "Saving static snapshots ($(n_snaps) rows) …"

fig = Figure(; size = (1200, 250 * n_snaps))

for (row, k) in enumerate(plot_ks)
    local ax1 = Axis(fig[row, 1]; xlabel = "x [m]", ylabel = "z [m]",
                      aspect = domain_aspect,
                      title = "ρ′  (step $(k-1))")
    local hm1 = heatmap!(ax1, xc, zc_nodes, ρ_fields[k];
                          colormap = :balance, colorrange = ρ_clims)
    scatter!(ax1, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
    Colorbar(fig[row, 2], hm1)

    local ax2 = Axis(fig[row, 3]; xlabel = "x [m]", ylabel = "z [m]",
                      aspect = domain_aspect,
                      title = "λ_ρ = ∂J/∂ρ  (step $(k-1))")
    local hm2 = heatmap!(ax2, xc, zc_nodes, λ_fields[k];
                          colormap = :balance, colorrange = λ_clims)
    scatter!(ax2, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
    Colorbar(fig[row, 4], hm2)
end

save("adjoint_replay_viz.png", fig)
@info "Saved adjoint_replay_viz.png"

# ── Comparison plot: reconstructed vs e2e at k=0 ─────────────────────────

@info "Plotting replay adjoint vs e2e comparison …"

cmax_recon = maximum(abs, reconstructed)
cmax_e2e   = maximum(abs, e2e_sensitivity)
cmax_diff  = max(maximum(abs, Δgrad), eps())

fig_cmp = Figure(; size = (1200, 700))

local cmp_ax1 = Axis(fig_cmp[1, 1]; xlabel = "x [m]", ylabel = "z [m]",
                      aspect = domain_aspect,
                      title = "Replay: λ_ρ + λ_{ρu}·u₀ + λ_{ρθ}·θ₀  (k=0)")
local cmp_hm1 = heatmap!(cmp_ax1, xc, zc_nodes, reconstructed;
                          colormap = :balance, colorrange = (-cmax_recon, cmax_recon))
scatter!(cmp_ax1, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
Colorbar(fig_cmp[1, 2], cmp_hm1)

local cmp_ax2 = Axis(fig_cmp[2, 1]; xlabel = "x [m]", ylabel = "z [m]",
                      aspect = domain_aspect,
                      title = "End-to-end  dJ/d(δρ)")
local cmp_hm2 = heatmap!(cmp_ax2, xc, zc_nodes, e2e_sensitivity;
                          colormap = :balance, colorrange = (-cmax_e2e, cmax_e2e))
scatter!(cmp_ax2, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
Colorbar(fig_cmp[2, 2], cmp_hm2)

local cmp_ax3 = Axis(fig_cmp[3, 1]; xlabel = "x [m]", ylabel = "z [m]",
                      aspect = domain_aspect,
                      title = "Difference  (recon − e2e),  max|Δ| = $(round(cmax_diff; sigdigits=3))")
local cmp_hm3 = heatmap!(cmp_ax3, xc, zc_nodes, Δgrad;
                          colormap = :balance, colorrange = (-cmax_diff, cmax_diff))
scatter!(cmp_ax3, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
Colorbar(fig_cmp[3, 2], cmp_hm3)

save("adjoint_replay_comparison.png", fig_cmp)
@info "Saved adjoint_replay_comparison.png"

# ── Animated GIF ──────────────────────────────────────────────────────────

@info "Rendering GIF …"

obs_ρ     = Observable(ρ_fields[1])
obs_λ     = Observable(λ_fields[1])
obs_title = Observable("step 0")

fig_gif = Figure(; size = (1200, 550))

gif_ax1 = Axis(fig_gif[1, 1]; xlabel = "x [m]", ylabel = "z [m]",
               aspect = domain_aspect,
               title = @lift("ρ′  " * $obs_title))
gif_hm1 = heatmap!(gif_ax1, xc, zc_nodes, obs_ρ;
                    colormap = :balance, colorrange = ρ_clims)
scatter!(gif_ax1, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
Colorbar(fig_gif[1, 2], gif_hm1)

gif_ax2 = Axis(fig_gif[2, 1]; xlabel = "x [m]", ylabel = "z [m]",
               aspect = domain_aspect,
               title = @lift("λ_ρ  " * $obs_title))
gif_hm2 = heatmap!(gif_ax2, xc, zc_nodes, obs_λ;
                    colormap = :balance, colorrange = λ_clims)
scatter!(gif_ax2, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
Colorbar(fig_gif[2, 2], gif_hm2)

CairoMakie.record(fig_gif, "adjoint_replay_viz.gif", 1:K+1; framerate = 4) do k
    obs_ρ[]     = ρ_fields[k]
    obs_λ[]     = λ_fields[k]
    obs_title[] = "step $(k-1)"
end

@info "Saved adjoint_replay_viz.gif"
@info "Done."
