using Breeze
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: xnodes, znodes
using Enzyme
using CUDA
using Reactant
using CairoMakie
Reactant.allowscalar(true)

# ══════════════════════════════════════════════════════════════════════════════
# AtmosphereModel state interface — flatten/unflatten prognostic fields
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
# Model setup: acoustic pulse in a sheared compressible atmosphere
# (following examples/acoustic_wave.jl)
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

@info "Setting initial conditions: acoustic density pulse (δρ=$δρ kg/m³, σ=$σ m) + log-layer wind (U₀=$U₀ m/s)"
set!(model; ρ = ρᵢ, θ = θ₀, u = uᵢ)

# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

CFL = 0.3
Δx  = Lx / Nx
Δz  = Lz / Nz
Δt  = CFL * min(Δx, Δz) / (ℂᵃᶜ + Uᵢ(Lz))
K   = 1

target_i = round(Int, 0.75Nx)
target_k = round(Int, 0.35Nz)

breeze_step!(m) = time_step!(m, Δt)

function breeze_loss(m)
    return interior(m.dynamics.density)[target_i, 1, target_k]^2
end

u0 = get_state(model)
N_state = length(Array(u0))

@info "State vector length: $N_state"
@info "Adjoint parameters: K=$K steps, Δt=$(round(Δt; sigdigits=3))s"
@info "Observation point: grid index ($target_i, $target_k)"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Compile all kernels up front (before any execution)
# ══════════════════════════════════════════════════════════════════════════════

@info "Phase 1 — Compiling set_state! …"
function restore_state!(m, u)
    set_state!(m, u)
    return nothing
end
compiled_set_state! = @compile raise=true raise_first=true donated_args=:none restore_state!(model, u0)

@info "Phase 1 — Compiling forward step + snapshot …"
function step_and_save!(m)
    breeze_step!(m)
    return get_state(m)
end
compiled_step_and_save! = @compile raise=true raise_first=true donated_args=:none step_and_save!(model)

@info "Phase 1 — Compiling terminal gradient …"
function grad_loss(m, dm)
    _, J = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, Enzyme.Const(breeze_loss), Enzyme.Active,
        Enzyme.Duplicated(m, dm)
    )
    return J
end
dmodel = Enzyme.make_zero(model)
compiled_grad_loss = @compile raise=true raise_first=true donated_args=:none grad_loss(model, dmodel)

@info "Phase 1 — Compiling VJP kernel …"
function vjp_kernel(u_prev, du, λ, m, dm)
    set_state!(m, u_prev)
    breeze_step!(m)
    Enzyme.autodiff(
        Enzyme.Reverse,
        Enzyme.Const((x, v, mdl) -> begin
            set_state!(mdl, x)
            breeze_step!(mdl)
            return sum(get_state(mdl) .* v)
        end),
        Enzyme.Active,
        Enzyme.Duplicated(u_prev, du),
        Enzyme.Const(λ),
        Enzyme.Duplicated(m, dm)
    )
    return nothing
end
compiled_vjp = @compile raise=true raise_first=true donated_args=:none vjp_kernel(
    Reactant.to_rarray(Array(u0)),
    Enzyme.make_zero(u0),
    Reactant.to_rarray(ones(Float64, N_state)),
    model,
    Enzyme.make_zero(model),
)

@info "Phase 1 complete — all kernels compiled."

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Forward sweep (checkpoint K snapshots)
# ══════════════════════════════════════════════════════════════════════════════

@info "Phase 2 — Forward sweep: $K steps …"
compiled_set_state!(model, u0)

snapshots = Vector{Vector{Float64}}(undef, K + 1)
snapshots[1] = Array(u0)
for k in 1:K
    snapshots[k + 1] = Array(compiled_step_and_save!(model))
    @info "  step $k/$K done"
end

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Terminal adjoint  dg/d(state_K)
# ══════════════════════════════════════════════════════════════════════════════

@info "Phase 3 — Computing terminal adjoint …"
compiled_set_state!(model, Reactant.to_rarray(snapshots[end]))

λs = Vector{Vector{Float64}}(undef, K + 1)
dmodel = Enzyme.make_zero(model)
compiled_grad_loss(model, dmodel)
λs[K + 1] = Array(get_state(dmodel))
@info "  terminal loss gradient ‖λ_K‖ = $(sum(abs, λs[K+1]))"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Backward sweep (VJP at each checkpoint)
# ══════════════════════════════════════════════════════════════════════════════

layout_bw = state_layout(model)
ρ_range   = layout_bw[:ρ]

@info "Phase 4 — Backward sweep: $K VJP steps …"
for k in K:-1:1
    u_prev_r = Reactant.to_rarray(snapshots[k])
    du_r     = Reactant.to_rarray(zeros(Float64, N_state))
    λ_r      = Reactant.to_rarray(λs[k + 1])
    dm_r     = Enzyme.make_zero(model)
    compiled_vjp(u_prev_r, du_r, λ_r, model, dm_r)
    λs[k] = Array(du_r)
    λ_ρ = @view λs[k][ρ_range]
    @info "  VJP step $k/$K done  ‖λ‖₁=$(sum(abs, λs[k]))  ‖λ‖∞=$(maximum(abs, λs[k]))  ‖λ_ρ‖∞=$(maximum(abs, λ_ρ))"
end

@info "Adjoint accumulation complete."

# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: End-to-end gradient via @trace checkpointing for comparison
#
# Following the acoustic_wave.jl pattern: build a *fresh* model + fields so
# that the tracer starts from clean buffers (the Phase 1-4 model has been
# mutated by many compiled calls).
# ══════════════════════════════════════════════════════════════════════════════

using Oceananigans.Fields: CenterField, XFaceField
using Reactant: @trace

@info "Phase 5 — Building fresh model for end-to-end gradient …"

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

ρᵗ_field   = CenterField(grid_ad)
dρᵗ_field  = CenterField(grid_ad)
dmodel_ad  = Enzyme.make_zero(model_ad)

Nsteps_e2e = K
@info "Phase 5 — Nsteps for e2e: $Nsteps_e2e (= K)"

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

@info "Phase 5 — Compiling end-to-end gradient (this may take a while) …"
compiled_e2e_grad = Reactant.@compile raise=true raise_first=true sync=true e2e_grad(
    model_ad, dmodel_ad, δρᵢ_field, dδρᵢ_field,
    ρᵗ_field, dρᵗ_field, ρᵇᵍ_field, uᵇᵍ_field, θ₀, Δt, Nsteps_e2e, target_i, target_k)

@info "Phase 5 — Running end-to-end gradient …"
dδρ_result, J_e2e = compiled_e2e_grad(
    model_ad, dmodel_ad, δρᵢ_field, dδρᵢ_field,
    ρᵗ_field, dρᵗ_field, ρᵇᵍ_field, uᵇᵍ_field, θ₀, Δt, Nsteps_e2e, target_i, target_k)

e2e_sensitivity = Array(interior(dδρ_result, :, 1, :))
@info "Phase 5 complete — J = $(Float64(only(J_e2e)))  ‖∂J/∂δρ‖∞ = $(maximum(abs, e2e_sensitivity))"

# ══════════════════════════════════════════════════════════════════════════════
# Visualize
# ══════════════════════════════════════════════════════════════════════════════

using Statistics: median

@info "Preparing visualization …"
layout = state_layout(model)
sz_c   = (Nx, Nz)

slice_ρ(snap) = reshape(snap[layout[:ρ]], sz_c)
slice_λρ(λ)  = reshape(λ[layout[:ρ]], sz_c)

xc = Array(xnodes(grid, Center()))
zc_nodes = Array(znodes(grid, Center()))

x_obs = xc[target_i]
z_obs = zc_nodes[target_k]

ρ_bg = [adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants) for _ in xc, z in zc_nodes]

ρ_fields = [slice_ρ(snapshots[k]) .- ρ_bg for k in 1:K+1]
λ_fields = [slice_λρ(λs[k])               for k in 1:K+1]

function balanced_around_median(fields)
    all_vals = vcat(vec.(fields)...)
    med = median(all_vals)
    half = maximum(abs.(all_vals .- med))
    return (med - half, med + half)
end

ρ_clims = balanced_around_median(ρ_fields)
λ_clims = balanced_around_median(λ_fields)

domain_aspect = Lx / Lz

# ── Static snapshot figure ────────────────────────────────────────────────
# Left column: ρ′ at selected time steps.  Right column: λ_ρ at the same steps.

plot_ks = [1, max(1, K ÷ 2), K + 1]
n_snaps = length(plot_ks)

fig = Figure(; size = (1200, 250 * n_snaps))

for (row, k) in enumerate(plot_ks)
    local ax1 = Axis(fig[row, 1]; xlabel = "x [m]", ylabel = "z [m]",
                      aspect = domain_aspect,
                      title = "ρ′ = ρ − ρ_bg  (k=$(k-1))")
    local hm1 = heatmap!(ax1, xc, zc_nodes, ρ_fields[k];
                          colormap = :balance, colorrange = ρ_clims)
    scatter!(ax1, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
    Colorbar(fig[row, 2], hm1)

    local ax2 = Axis(fig[row, 3]; xlabel = "x [m]", ylabel = "z [m]",
                      aspect = domain_aspect,
                      title = "λ_ρ = ∂J/∂ρ  (k=$(k-1))")
    local hm2 = heatmap!(ax2, xc, zc_nodes, λ_fields[k];
                          colormap = :balance, colorrange = λ_clims)
    scatter!(ax2, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
    Colorbar(fig[row, 4], hm2)
end

save("compressible_adjoint_viz.png", fig)
@info "Saved compressible_adjoint_viz.png"

# ── Animated GIF (one frame per time-step) ────────────────────────────────
# ρ′ on top, λ_ρ on bottom, both with domain-proportioned aspect ratio.

@info "Rendering GIF …"

obs_ρ = Observable(ρ_fields[1])
obs_λ = Observable(λ_fields[1])
obs_title = Observable("k = 0")

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

CairoMakie.record(fig_gif, "compressible_adjoint_viz.gif", 1:K+1; framerate = 8) do k
    obs_ρ[]     = ρ_fields[k]
    obs_λ[]     = λ_fields[k]
    obs_title[] = "k = $(k-1)"
end

@info "Saved compressible_adjoint_viz.gif"

# ── Comparison: manual adjoint vs end-to-end gradient ─────────────────────
#
# The e2e differentiates through set!(mdl; ρ = ρᵗ, θ = θ₀, u = u₀) which
# converts primitive → conservative:  ρθ = ρ·θ₀,  ρu = ρ·u₀.
# So dJ/d(δρ) = λ_ρ + λ_{ρu}·u₀ + λ_{ρθ}·θ₀.
#
# The manual adjoint's λ_ρ alone is ∂J/∂ρ₀ at fixed ρu, ρθ — a different
# quantity.  We reconstruct the total derivative from the manual adjoint
# components to verify consistency.

@info "Reconstructing total dJ/d(δρ) from manual adjoint components …"

manual_λρ  = slice_λρ(λs[1])
manual_λρu = reshape(λs[1][layout_bw[:ρu]], sz_c)
manual_λρθ = reshape(λs[1][layout_bw[:ρθ]], sz_c)

u0_field = [Uᵢ(z) for _ in xc, z in zc_nodes]

reconstructed = manual_λρ .+ manual_λρu .* u0_field .+ manual_λρθ .* θ₀

@info "  ‖λ_ρ‖∞          = $(maximum(abs, manual_λρ))"
@info "  ‖λ_{ρu}·u₀‖∞    = $(maximum(abs, manual_λρu .* u0_field))"
@info "  ‖λ_{ρθ}·θ₀‖∞    = $(maximum(abs, manual_λρθ .* θ₀))"
@info "  ‖reconstructed‖∞ = $(maximum(abs, reconstructed))"
@info "  ‖e2e‖∞           = $(maximum(abs, e2e_sensitivity))"

# ── Comparison plot: 4 panels ─────────────────────────────────────────────

@info "Plotting adjoint comparison …"

Δgrad = reconstructed .- e2e_sensitivity
Δgrad_max = maximum(abs, Δgrad)

cmax_partial = maximum(abs, manual_λρ)
cmax_recon   = maximum(abs, reconstructed)
cmax_e2e     = maximum(abs, e2e_sensitivity)
cmax_diff    = max(Δgrad_max, eps())

fig_cmp = Figure(; size = (1200, 950))

ax1 = Axis(fig_cmp[1, 1]; xlabel = "x [m]", ylabel = "z [m]",
           aspect = domain_aspect, title = "Manual  λ_ρ  (∂J/∂ρ₀  at fixed ρu,ρθ)")
hm1 = heatmap!(ax1, xc, zc_nodes, manual_λρ;
               colormap = :balance, colorrange = (-cmax_partial, cmax_partial))
scatter!(ax1, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
Colorbar(fig_cmp[1, 2], hm1)

ax2 = Axis(fig_cmp[2, 1]; xlabel = "x [m]", ylabel = "z [m]",
           aspect = domain_aspect, title = "Reconstructed  λ_ρ + λ_{ρu}·u₀ + λ_{ρθ}·θ₀")
hm2 = heatmap!(ax2, xc, zc_nodes, reconstructed;
               colormap = :balance, colorrange = (-cmax_recon, cmax_recon))
scatter!(ax2, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
Colorbar(fig_cmp[2, 2], hm2)

ax3 = Axis(fig_cmp[3, 1]; xlabel = "x [m]", ylabel = "z [m]",
           aspect = domain_aspect, title = "End-to-end  dJ/d(δρ)")
hm3 = heatmap!(ax3, xc, zc_nodes, e2e_sensitivity;
               colormap = :balance, colorrange = (-cmax_e2e, cmax_e2e))
scatter!(ax3, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
Colorbar(fig_cmp[3, 2], hm3)

ax4 = Axis(fig_cmp[4, 1]; xlabel = "x [m]", ylabel = "z [m]",
           aspect = domain_aspect,
           title = "Difference  (recon − e2e),  max|Δ| = $(round(Δgrad_max; sigdigits=3))")
hm4 = heatmap!(ax4, xc, zc_nodes, Δgrad;
               colormap = :balance, colorrange = (-cmax_diff, cmax_diff))
scatter!(ax4, [x_obs], [z_obs]; color = :black, marker = :star5, markersize = 14)
Colorbar(fig_cmp[4, 2], hm4)

save("adjoint_comparison.png", fig_cmp)
rel_err = Δgrad_max / max(cmax_recon, cmax_e2e, eps())
@info "Saved adjoint_comparison.png  —  max|Δ| = $Δgrad_max,  relative = $(round(rel_err; sigdigits=3))"
