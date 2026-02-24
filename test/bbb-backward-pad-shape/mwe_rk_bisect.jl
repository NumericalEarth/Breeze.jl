# B.6.10 bisection: what part of the RK machinery (beyond update_state!) triggers the error?
# Known: update_state! alone passes. Full time_step! fails. Pressure correction is no-op.
# Test the store_initial_state! + substep pattern using broadcasts to isolate the pattern.
#
# Run: julia --check-bounds=no -O0 --project -e 'include("test/bbb-backward-pad-shape/mwe_rk_bisect.jl")'

using Breeze
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: update_state!
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
    time_step!(model, FT(0.001))
    return model
end

# ── I: store + broadcast substep + update_state (single stage) ──

function loss_I(model)
    ts = model.timestepper
    @trace checkpointing=true track_numbers=false for _ in 1:4
        for (u⁰, u) in zip(ts.U⁰, prognostic_fields(model))
            parent(u⁰) .= parent(u)
        end
        for (u, u⁰, G) in zip(prognostic_fields(model), ts.U⁰, ts.Gⁿ)
            parent(u) .= parent(u⁰) .+ FT(0.001) .* parent(G)
        end
        update_state!(model; compute_tendencies=true)
    end
    return mean(interior(model.temperature).^2)
end

function grad_I(model, dmodel)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_I, Enzyme.Active, Enzyme.Duplicated(model, dmodel))
    return lv
end

@info "I: store + broadcast substep + update_state! (single stage sim)"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_I(model, dmodel)
    lv = compiled(model, dmodel)
    @info "I passed — loss=$lv"
end

# ── J: just broadcast mutation of prognostic fields + update_state! ──

function loss_J(model)
    @trace checkpointing=true track_numbers=false for _ in 1:4
        for u in prognostic_fields(model)
            parent(u) .= parent(u) .* FT(0.999)
        end
        update_state!(model; compute_tendencies=true)
    end
    return mean(interior(model.temperature).^2)
end

function grad_J(model, dmodel)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_J, Enzyme.Active, Enzyme.Duplicated(model, dmodel))
    return lv
end

@info "J: broadcast field mutation + update_state!"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_J(model, dmodel)
    lv = compiled(model, dmodel)
    @info "J passed — loss=$lv"
end

# ── K: 3× (store + broadcast substep + update_state!) — full 3-stage sim ──

function loss_K(model)
    ts = model.timestepper
    α = (FT(1), FT(1//4), FT(2//3))
    @trace checkpointing=true track_numbers=false for _ in 1:4
        for (u⁰, u) in zip(ts.U⁰, prognostic_fields(model))
            parent(u⁰) .= parent(u)
        end
        for stage in 1:3
            for (u, u⁰, G) in zip(prognostic_fields(model), ts.U⁰, ts.Gⁿ)
                parent(u) .= (1 - α[stage]) .* parent(u⁰) .+ α[stage] .* (parent(u) .+ FT(0.001) .* parent(G))
            end
            update_state!(model; compute_tendencies=true)
        end
    end
    return mean(interior(model.temperature).^2)
end

function grad_K(model, dmodel)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_K, Enzyme.Active, Enzyme.Duplicated(model, dmodel))
    return lv
end

@info "K: 3× (store + broadcast substep + update_state!) — full 3-stage sim"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_K(model, dmodel)
    lv = compiled(model, dmodel)
    @info "K passed — loss=$lv"
end

@info "All tests done."
