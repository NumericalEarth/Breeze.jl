#####
##### MWE: Checkpointing with multiple heterogeneously-shaped fields on BBB
#####
# Previous MWEs showed:
#   - Single field + checkpointing + halo fill → PASSES
#   - Full AtmosphereModel + checkpointing + time_step! → FAILS
#
# Hypothesis: the checkpoint save/restore of multiple fields with
# different parent array shapes (CCC: 12×12×12, CCF: 12×12×13, etc.)
# triggers the stablehlo.pad shape mismatch.
#
# This MWE builds up incrementally from multi-field to model-like operations.
#
# Run: julia --project -e 'include("test/bbb-backward-pad-shape/mwe_checkpoint_multifield.jl")'

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean

Reactant.set_default_backend("cpu")

N = 6
nsteps = 4

grid = RectilinearGrid(ReactantState();
    size = (N, N, N), extent = (1e3, 1e3, 1e3),
    topology = (Bounded, Bounded, Bounded))

# ──────────────────────────────────────────────────────────────────
# Level 1: Two fields (CCC + CCF) in a NamedTuple, checkpointed loop
# ──────────────────────────────────────────────────────────────────

c1 = CenterField(grid); set!(c1, 1.0)                       # CCC → parent 12×12×12
w1 = Field{Center, Center, Face}(grid); set!(w1, 1.0)       # CCF → parent 12×12×13

function loss_level1(c, w, nsteps)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        parent(c) .*= 0.99
        parent(w) .*= 0.99
        fill_halo_regions!(c)
        fill_halo_regions!(w)
    end
    return mean(interior(c).^2) + mean(interior(w).^2)
end

function grad_level1(c, dc, w, dw, nsteps)
    parent(dc) .= 0
    parent(dw) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_level1, Enzyme.Active,
        Enzyme.Duplicated(c, dc),
        Enzyme.Duplicated(w, dw),
        Enzyme.Const(nsteps))
    return lv
end

dc1 = Enzyme.make_zero(c1)
dw1 = Enzyme.make_zero(w1)

@info "[Level 1] Two fields (CCC+CCF), checkpointed..."
try
    compiled = Reactant.@compile raise=true raise_first=true sync=true grad_level1(c1, dc1, w1, dw1, nsteps)
    @info "[Level 1] ✓ PASS"
catch e
    @warn "[Level 1] ✗ FAIL" exception=(e, catch_backtrace())
end

# ──────────────────────────────────────────────────────────────────
# Level 2: Four fields (FCC + CFC + CCF + CCC) — all momentum-like
# locations plus a scalar, with store/restore + substep pattern
# ──────────────────────────────────────────────────────────────────

ρu = Field{Face, Center, Center}(grid); set!(ρu, 1.0)    # FCC → 13×12×12
ρv = Field{Center, Face, Center}(grid); set!(ρv, 1.0)    # CFC → 12×13×12
ρw = Field{Center, Center, Face}(grid); set!(ρw, 1.0)    # CCF → 12×12×13
ρθ = CenterField(grid); set!(ρθ, 300.0)                  # CCC → 12×12×12

ρu⁰ = Field{Face, Center, Center}(grid)
ρv⁰ = Field{Center, Face, Center}(grid)
ρw⁰ = Field{Center, Center, Face}(grid)
ρθ⁰ = CenterField(grid)

function loss_level2(ρu, ρv, ρw, ρθ, ρu⁰, ρv⁰, ρw⁰, ρθ⁰, nsteps)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        # store initial state (like store_initial_state!)
        parent(ρu⁰) .= parent(ρu)
        parent(ρv⁰) .= parent(ρv)
        parent(ρw⁰) .= parent(ρw)
        parent(ρθ⁰) .= parent(ρθ)

        # substep-like update: u = (1-α)*u⁰ + α*(u + Δt*G)
        # using α=1 and G = -0.01*u for simplicity
        parent(ρu) .= parent(ρu) .* 0.99
        parent(ρv) .= parent(ρv) .* 0.99
        parent(ρw) .= parent(ρw) .* 0.99
        parent(ρθ) .= parent(ρθ) .* 0.99

        fill_halo_regions!(ρu)
        fill_halo_regions!(ρv)
        fill_halo_regions!(ρw)
        fill_halo_regions!(ρθ)
    end
    return mean(interior(ρθ).^2)
end

function grad_level2(ρu, dρu, ρv, dρv, ρw, dρw, ρθ, dρθ, ρu⁰, dρu⁰, ρv⁰, dρv⁰, ρw⁰, dρw⁰, ρθ⁰, dρθ⁰, nsteps)
    for df in (dρu, dρv, dρw, dρθ, dρu⁰, dρv⁰, dρw⁰, dρθ⁰)
        parent(df) .= 0
    end
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_level2, Enzyme.Active,
        Enzyme.Duplicated(ρu, dρu),
        Enzyme.Duplicated(ρv, dρv),
        Enzyme.Duplicated(ρw, dρw),
        Enzyme.Duplicated(ρθ, dρθ),
        Enzyme.Duplicated(ρu⁰, dρu⁰),
        Enzyme.Duplicated(ρv⁰, dρv⁰),
        Enzyme.Duplicated(ρw⁰, dρw⁰),
        Enzyme.Duplicated(ρθ⁰, dρθ⁰),
        Enzyme.Const(nsteps))
    return lv
end

dρu = Enzyme.make_zero(ρu)
dρv = Enzyme.make_zero(ρv)
dρw = Enzyme.make_zero(ρw)
dρθ = Enzyme.make_zero(ρθ)
dρu⁰ = Enzyme.make_zero(ρu⁰)
dρv⁰ = Enzyme.make_zero(ρv⁰)
dρw⁰ = Enzyme.make_zero(ρw⁰)
dρθ⁰ = Enzyme.make_zero(ρθ⁰)

@info "[Level 2] Four fields with store/restore + substep, checkpointed..."
try
    compiled = Reactant.@compile raise=true raise_first=true sync=true grad_level2(
        ρu, dρu, ρv, dρv, ρw, dρw, ρθ, dρθ,
        ρu⁰, dρu⁰, ρv⁰, dρv⁰, ρw⁰, dρw⁰, ρθ⁰, dρθ⁰, nsteps)
    @info "[Level 2] ✓ PASS"
catch e
    @warn "[Level 2] ✗ FAIL" exception=(e, catch_backtrace())
end

# ──────────────────────────────────────────────────────────────────
# Level 3: Cross-field reads (u = ρu / ρ pattern) + halo fills
# ──────────────────────────────────────────────────────────────────

ρ = CenterField(grid); set!(ρ, 1.0)
u_vel = Field{Face, Center, Center}(grid)
w_vel = Field{Center, Center, Face}(grid)

ρu3 = Field{Face, Center, Center}(grid); set!(ρu3, 1.0)
ρw3 = Field{Center, Center, Face}(grid); set!(ρw3, 1.0)
ρθ3 = CenterField(grid); set!(ρθ3, 300.0)

function loss_level3(ρ, ρu, ρw, ρθ, u_vel, w_vel, nsteps)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        # "compute velocities" (cross-field division)
        parent(u_vel) .= parent(ρu) ./ parent(ρ)[1:size(parent(ρu), 1), :, :]
        parent(w_vel) .= parent(ρw) ./ parent(ρ)[:, :, 1:size(parent(ρw), 3)]

        # decay
        parent(ρu) .*= 0.99
        parent(ρw) .*= 0.99
        parent(ρθ) .*= 0.99

        # fill halos
        fill_halo_regions!(ρu)
        fill_halo_regions!(ρw)
        fill_halo_regions!(ρθ)
        fill_halo_regions!(u_vel)
        fill_halo_regions!(w_vel)
    end
    return mean(interior(ρθ).^2)
end

function grad_level3(ρ, dρ, ρu, dρu, ρw, dρw, ρθ, dρθ, u_vel, du_vel, w_vel, dw_vel, nsteps)
    for df in (dρ, dρu, dρw, dρθ, du_vel, dw_vel)
        parent(df) .= 0
    end
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_level3, Enzyme.Active,
        Enzyme.Duplicated(ρ, dρ),
        Enzyme.Duplicated(ρu, dρu),
        Enzyme.Duplicated(ρw, dρw),
        Enzyme.Duplicated(ρθ, dρθ),
        Enzyme.Duplicated(u_vel, du_vel),
        Enzyme.Duplicated(w_vel, dw_vel),
        Enzyme.Const(nsteps))
    return lv
end

dρ = Enzyme.make_zero(ρ)
dρu3 = Enzyme.make_zero(ρu3)
dρw3 = Enzyme.make_zero(ρw3)
dρθ3 = Enzyme.make_zero(ρθ3)
du_vel = Enzyme.make_zero(u_vel)
dw_vel = Enzyme.make_zero(w_vel)

@info "[Level 3] Cross-field operations + halo fills, checkpointed..."
try
    compiled = Reactant.@compile raise=true raise_first=true sync=true grad_level3(
        ρ, dρ, ρu3, dρu3, ρw3, dρw3, ρθ3, dρθ3, u_vel, du_vel, w_vel, dw_vel, nsteps)
    @info "[Level 3] ✓ PASS"
catch e
    @warn "[Level 3] ✗ FAIL" exception=(e, catch_backtrace())
end
