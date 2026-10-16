include(joinpath(dirname(@__DIR__), "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: first_time_step!, reset!
using Reactant
using Enzyme
using CUDA
using Statistics: mean
using Test

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

function anelastic_model(arch)
    grid = RectilinearGrid(arch, Float64; size=(8, 8, 8), extent=(800, 800, 800), halo=(5, 5, 5),
                           topology=(Periodic, Periodic, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(Float64);
                                     base_pressure=101500, potential_temperature=290)
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state),
                            momentum_advection=WENO(order=5), scalar_advection=WENO(order=5))
    return Base.inferencebarrier(model)
end

initial_θ(x, y, z) = 290 + z / 250 + 3 * sinpi(x / 400) * cospi(y / 400) * sinpi(z / 800) / 10
perturbation(x, y, z) = (1 + cospi(x / 400)) * sinpi(z / 800)

function vertical_kinetic_energy(model, θ₀, Δt)
    reset!(model.clock)
    set!(model; θ=θ₀, u=0, v=0, w=0)
    first_time_step!(model, Δt)
    return mean(interior(model.velocities.w) .^ 2)
end

function vertical_kinetic_energy_gradient!(model, dmodel, θ₀, dθ₀, Δt)
    Enzyme.remake_zero!(dmodel)
    parent(dθ₀) .= 0
    _, value = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                               vertical_kinetic_energy, Enzyme.Active,
                               Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(θ₀, dθ₀), Enzyme.Const(Δt))
    return dθ₀, value
end

# Each reference evaluation uses a freshly constructed model, so that no velocity
# or clock state carries over from a previous forecast.
function reference_vertical_kinetic_energy(ε, Δt)
    model = anelastic_model(CPU())
    θ₀ = CenterField(model.grid)
    direction = CenterField(model.grid)
    set!(θ₀, initial_θ)
    set!(direction, perturbation)
    set!(θ₀, interior(θ₀) .+ ε .* interior(direction))
    return vertical_kinetic_energy(model, θ₀, Δt), Array(interior(direction))
end

@testset "Anelastic projection on ReactantState" begin
    u₀(x, y, z) = sinpi(x / 400) * sinpi(z / 800)
    w₀(x, y, z) = cospi(y / 400) * sinpi(z / 800)
    models = map((ReactantState(), CPU())) do arch
        model = anelastic_model(arch)
        set!(model; u=u₀, w=w₀)
        model
    end
    for name in (:u, :v, :w)
        projected = Array(interior(getproperty(models[1].velocities, name)))
        expected = Array(interior(getproperty(models[2].velocities, name)))
        @test projected ≈ expected rtol=1e-12 atol=1e-14
    end
end

@testset "Anelastic reverse-mode gradient" begin
    Δt = 1.0
    model = anelastic_model(ReactantState())
    θ₀ = CenterField(model.grid)
    dθ₀ = CenterField(model.grid)
    set!(θ₀, initial_θ)
    dmodel = Enzyme.make_zero(model)
    compiled = @with_stack_size 256 << 20 Reactant.@compile raise=true raise_first=true sync=true vertical_kinetic_energy_gradient!(
        model, dmodel, θ₀, dθ₀, Δt)

    reference, direction = reference_vertical_kinetic_energy(0, Δt)
    ε = 1e-4
    plus, _ = reference_vertical_kinetic_energy(+ε, Δt)
    minus, _ = reference_vertical_kinetic_energy(-ε, Δt)
    finite_difference = (plus - minus) / 2ε

    # Reusing the executable must restore the forecast and clear the previous adjoint.
    for repetition in 1:2
        gradient, value = compiled(model, dmodel, θ₀, dθ₀, Δt)
        gradient = Array(interior(gradient))
        @test Float64(value) ≈ reference rtol=1e-10
        @test all(isfinite, gradient)
        @test sum(gradient .* direction) ≈ finite_difference rtol=1e-6
    end
end
