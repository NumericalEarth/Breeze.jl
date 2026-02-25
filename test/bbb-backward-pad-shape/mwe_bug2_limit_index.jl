#=
BUG 2: limit index > dimension size (BBB backward + duplicate update_state!)
═════════════════════════════════════════════════════════════════════════════

ERROR:
  limit index 5 is larger than dimension size 4 in dimension 0

  Occurs in _fill_flux_top_halo! on velocity field u at (Face,Center,Center)
  with Open{Nothing} BCs and OffsetStaticSize{(1:5, 1:4)}.

TRIGGER: Enzyme reverse-mode through a checkpointed @trace loop where the
  loop body contains an ssp_rk3_substep! followed by TWO update_state! calls.
  The adjoint of the Open BC halo fill on Face fields gets confused about
  array dimensions when the same kernel appears twice in the loop body.

MINIMAL TRIGGER CONDITIONS (all required):
  1. Topology = (Bounded, Bounded, Bounded)
  2. CompressibleDynamics (diagnostic velocity with Open BCs on Face fields)
  3. checkpointing=true on the @trace loop
  4. Enzyme reverse mode
  5. At least one ssp_rk3_substep! BEFORE the duplicate update_state! calls
  6. TWO or more update_state!(compute_tendencies=true) calls in one loop body

PASS/FAIL matrix from bisection (mwe_topdown_narrow.jl):
  N1: 2× update_state! (no substep)              → PASS
  N2: 2× substep (no update_state!)              → PASS
  N3: 1× substep + 2× update_state!             → FAIL ← this pattern
  N4: 2× substep + 1× update_state!             → PASS
  N5: substep + update + substep (no 2nd update) → PASS

  → The trigger is: substep then 2+ update_state! calls. Neither alone suffices.

RELATIONSHIP TO BUG 1:
  Both are BBB backward + checkpointing halo-fill adjoint failures on Face
  fields. Bug 1 hits the full time_step! (3 RK stages) and gets a pad shape
  error. Bug 2 hits a simpler pattern (substep + 2× update) and gets a limit
  index error. They may share a root cause in the checkpointing transform's
  handling of repeated halo-fill kernels on asymmetric Face fields.

Run: julia --check-bounds=no --project -e 'include("test/bbb-backward-pad-shape/mwe_bug2_limit_index.jl")'
=#

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: update_state!
using Breeze.TimeSteppers: store_initial_state!, ssp_rk3_substep!
using Reactant, Reactant: @trace
using Enzyme
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size=(4, 4, 4), extent=(1e3, 1e3, 1e3),
    topology=(Bounded, Bounded, Bounded))

model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
FT = eltype(grid)

dmodel = Enzyme.make_zero(model)
θ_init  = CenterField(grid); set!(θ_init,  (args...) -> FT(300))
dθ_init = CenterField(grid); set!(dθ_init, FT(0))

function loss(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=one(FT))
    α¹ = model.timestepper.α¹
    @trace checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies=true)
        update_state!(model; compute_tendencies=true)   # ← second call triggers bug
    end
    return mean(interior(model.temperature).^2)
end

function grad_loss(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    parent(dθ_init) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt), Enzyme.Const(nsteps))
    return dθ_init, lv
end

Δt = FT(0.02)
nsteps = 4

@info "Bug 2: Expecting 'limit index > dimension size' error..."
compiled = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model, dmodel, θ_init, dθ_init, Δt, nsteps)
