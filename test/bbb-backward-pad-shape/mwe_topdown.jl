# B.6.10 top-down bisection: single time_step! passes, now bisect loop/set!/checkpointing
# Run: julia --check-bounds=no -O0 --project -e 'include("test/bbb-backward-pad-shape/mwe_topdown.jl")'

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean

Reactant.set_default_backend("cpu")

FT = Float64

function fresh_model()
    grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), extent=(1, 1, 1),
                           topology=(Bounded, Bounded, Bounded))
    model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
    set!(model; θ=FT(300), ρ=one(FT))
    return model
end

function fresh_init_fields(grid)
    θ_init = CenterField(grid)
    set!(θ_init, (args...) -> FT(300))
    dθ_init = CenterField(grid)
    set!(dθ_init, FT(0))
    return θ_init, dθ_init
end

# ── L1: single time_step!, no loop, no set! inside loss (KNOWN PASS) ──

function loss_L1(model)
    time_step!(model, FT(0.001))
    return mean(interior(model.temperature).^2)
end

function grad_L1(model, dmodel)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_L1, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel))
    return lv
end

@info "L1: single time_step! (no loop, no set!)"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_L1(model, dmodel)
    lv = compiled(model, dmodel)
    @info "L1 passed — loss=$lv"
end

# ── L2: set! inside loss + single time_step! ──

function loss_L2(model, θ_init)
    set!(model; θ=θ_init, ρ=one(FT))
    time_step!(model, FT(0.001))
    return mean(interior(model.temperature).^2)
end

function grad_L2(model, dmodel, θ_init, dθ_init)
    parent(dθ_init) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_L2, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init))
    return lv
end

@info "L2: set! + single time_step!"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    θ_init, dθ_init = fresh_init_fields(model.grid)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_L2(model, dmodel, θ_init, dθ_init)
    lv = compiled(model, dmodel, θ_init, dθ_init)
    @info "L2 passed — loss=$lv"
end

# ── L3: @trace loop (nsteps=4), no checkpointing, no set! ──

function loss_L3(model)
    @trace track_numbers=false for _ in 1:4
        time_step!(model, FT(0.001))
    end
    return mean(interior(model.temperature).^2)
end

function grad_L3(model, dmodel)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_L3, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel))
    return lv
end

@info "L3: @trace loop (4 steps), no checkpointing"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_L3(model, dmodel)
    lv = compiled(model, dmodel)
    @info "L3 passed — loss=$lv"
end

# ── L4: set! + @trace loop (nsteps=4), no checkpointing ──

function loss_L4(model, θ_init)
    set!(model; θ=θ_init, ρ=one(FT))
    @trace track_numbers=false for _ in 1:4
        time_step!(model, FT(0.001))
    end
    return mean(interior(model.temperature).^2)
end

function grad_L4(model, dmodel, θ_init, dθ_init)
    parent(dθ_init) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_L4, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init))
    return lv
end

@info "L4: set! + @trace loop (4 steps), no checkpointing"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    θ_init, dθ_init = fresh_init_fields(model.grid)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_L4(model, dmodel, θ_init, dθ_init)
    lv = compiled(model, dmodel, θ_init, dθ_init)
    @info "L4 passed — loss=$lv"
end

# ── L5: set! + @trace loop with checkpointing (matches actual test) ──

function loss_L5(model, θ_init)
    set!(model; θ=θ_init, ρ=one(FT))
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:4
        time_step!(model, FT(0.001))
    end
    return mean(interior(model.temperature).^2)
end

function grad_L5(model, dmodel, θ_init, dθ_init)
    parent(dθ_init) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_L5, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init))
    return lv
end

@info "L5: set! + @trace loop with checkpointing (full test pattern)"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    θ_init, dθ_init = fresh_init_fields(model.grid)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_L5(model, dmodel, θ_init, dθ_init)
    lv = compiled(model, dmodel, θ_init, dθ_init)
    @info "L5 passed — loss=$lv"
end

@info "All levels passed."
