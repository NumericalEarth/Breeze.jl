# B.6.10 bisection: what part of time_step! triggers the error under checkpointing?
# Known: checkpointing=true is the sole flag needed. Now isolate which piece of the
# SSPRK3 time_step! body is required.
#
# Run: julia --check-bounds=no -O0 --project -e 'include("test/bbb-backward-pad-shape/mwe_timestep_bisect.jl")'

using Breeze
using Breeze.AtmosphereModels: compute_velocities!
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
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
    time_step!(model, FT(0.001))  # populate all fields (velocities, tendencies, etc.)
    return model
end

# ── E: fill_halo_regions! on velocity fields only ──

function loss_E(model)
    @trace checkpointing=true track_numbers=false for _ in 1:4
        fill_halo_regions!(model.velocities)
    end
    return mean(interior(model.velocities.u).^2)
end

function grad_E(model, dmodel)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_E, Enzyme.Active, Enzyme.Duplicated(model, dmodel))
    return lv
end

@info "E: fill_halo_regions!(model.velocities) in checkpointed loop"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_E(model, dmodel)
    lv = compiled(model, dmodel)
    @info "E passed — loss=$lv"
end

# ── F: compute_velocities! (KA kernel + halo fills on Face fields) ──

function loss_F(model)
    @trace checkpointing=true track_numbers=false for _ in 1:4
        compute_velocities!(model)
    end
    return mean(interior(model.velocities.u).^2)
end

function grad_F(model, dmodel)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_F, Enzyme.Active, Enzyme.Duplicated(model, dmodel))
    return lv
end

@info "F: compute_velocities!(model) in checkpointed loop"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_F(model, dmodel)
    lv = compiled(model, dmodel)
    @info "F passed — loss=$lv"
end

# ── G: update_state! without tendencies ──

function loss_G(model)
    @trace checkpointing=true track_numbers=false for _ in 1:4
        update_state!(model; compute_tendencies=false)
    end
    return mean(interior(model.temperature).^2)
end

function grad_G(model, dmodel)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_G, Enzyme.Active, Enzyme.Duplicated(model, dmodel))
    return lv
end

@info "G: update_state!(model; compute_tendencies=false) in checkpointed loop"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_G(model, dmodel)
    lv = compiled(model, dmodel)
    @info "G passed — loss=$lv"
end

# ── H: update_state! with tendencies ──

function loss_H(model)
    @trace checkpointing=true track_numbers=false for _ in 1:4
        update_state!(model; compute_tendencies=true)
    end
    return mean(interior(model.temperature).^2)
end

function grad_H(model, dmodel)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_H, Enzyme.Active, Enzyme.Duplicated(model, dmodel))
    return lv
end

@info "H: update_state!(model; compute_tendencies=true) in checkpointed loop"
let model = fresh_model(), dmodel = Enzyme.make_zero(model)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_H(model, dmodel)
    lv = compiled(model, dmodel)
    @info "H passed — loss=$lv"
end

@info "All tests done."
