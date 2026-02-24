# B.6.10 fine bisection: isolate checkpointing vs mincut vs set!
# Requires Breeze model (simple field-only MWE passed — full model context needed)
#
# Run: julia --check-bounds=no -O0 --project -e 'include("test/bbb-backward-pad-shape/mwe_checkpoint_bisect.jl")'

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
    θ = CenterField(grid); set!(θ, (args...) -> FT(300))
    dθ = CenterField(grid); set!(dθ, FT(0))
    return θ, dθ
end

# ── A: checkpointing only (no mincut), with set! ──

# function loss_A(model, θ_init)
#     set!(model; θ=θ_init, ρ=one(FT))
#     @trace checkpointing=true track_numbers=false for _ in 1:4
#         time_step!(model, FT(0.001))
#     end
#     return mean(interior(model.temperature).^2)
# end

# function grad_A(model, dmodel, θ_init, dθ_init)
#     parent(dθ_init) .= 0
#     _, lv = Enzyme.autodiff(
#         Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
#         loss_A, Enzyme.Active,
#         Enzyme.Duplicated(model, dmodel),
#         Enzyme.Duplicated(θ_init, dθ_init))
#     return lv
# end

# @info "A: set! + checkpointing=true (no mincut)"
# let model = fresh_model(), dmodel = Enzyme.make_zero(model)
#     θ, dθ = fresh_init_fields(model.grid)
#     @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_A(model, dmodel, θ, dθ)
#     lv = compiled(model, dmodel, θ, dθ)
#     @info "A passed — loss=$lv"
# end

# ── B: mincut only (no checkpointing), with set! ──

function loss_B(model, θ_init)
    set!(model; θ=θ_init, ρ=one(FT))
    @trace mincut=true track_numbers=false for _ in 1:4
        time_step!(model, FT(0.001))
    end
    return mean(interior(model.temperature).^2)
end

function grad_B(model, dmodel, θ_init, dθ_init)
    parent(dθ_init) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_B, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init))
    return lv
end

@info "B: set! + mincut=true (no checkpointing)"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    θ, dθ = fresh_init_fields(model.grid)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_B(model, dmodel, θ, dθ)
    lv = compiled(model, dmodel, θ, dθ)
    @info "B passed — loss=$lv"
end

# ── C: checkpointing + mincut, NO set! (is set! needed?) ──

function loss_C(model)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:4
        time_step!(model, FT(0.001))
    end
    return mean(interior(model.temperature).^2)
end

function grad_C(model, dmodel)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_C, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel))
    return lv
end

@info "C: checkpointing + mincut, NO set! (is set! needed?)"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_C(model, dmodel)
    lv = compiled(model, dmodel)
    @info "C passed — loss=$lv"
end

# ── D: checkpointing + mincut + set! (= L5, known FAIL, confirmation) ──

function loss_D(model, θ_init)
    set!(model; θ=θ_init, ρ=one(FT))
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:4
        time_step!(model, FT(0.001))
    end
    return mean(interior(model.temperature).^2)
end

function grad_D(model, dmodel, θ_init, dθ_init)
    parent(dθ_init) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_D, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init))
    return lv
end

@info "D: set! + checkpointing + mincut (= L5, known FAIL)"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    θ, dθ = fresh_init_fields(model.grid)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_D(model, dmodel, θ, dθ)
    lv = compiled(model, dmodel, θ, dθ)
    @info "D passed — loss=$lv"
end

@info "All tests done."
