# Differentiability of Lagrangian droplet microphysics with Enzyme (and Reactant), CPU.
#
#   julia --project=~/droplet-tasks-env droplet_differentiability.jl [levels]
#
# Level a: single-droplet functions; level b: time_step! of a small box with droplets;
# level c: Reactant compilation of time_step! with droplets.

using Breeze
using Breeze.ParticleMicrophysics: diffusional_growth_rate, condensational_growth_step, droplet_water_mass
using Breeze.Thermodynamics: saturation_vapor_pressure, vapor_gas_constant, dry_air_gas_constant, PlanarLiquidSurface
using Oceananigans
using Enzyme
using CUDA # needed by Reactant to raise KernelAbstractions kernels
get(ENV, "ENZYME_LOOSE", "0") == "1" && Enzyme.API.looseTypeAnalysis!(true)
get(ENV, "ENZYME_VERBOSE", "0") == "1" && (Enzyme.Compiler.VERBOSE_ERRORS[] = true)
using Reactant
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!, KernelParameters
using Oceananigans.Grids: halo_size
using Oceananigans.Models.LagrangianParticleTracking: advect_lagrangian_particles!
using Printf
using Random

levels = length(ARGS) ≥ 1 ? ARGS[1] : "abc"

fd(f, x; h=1e-6 * abs(x)) = (f(x + h) - f(x - h)) / 2h
report(name, ad, fdv) = @printf("%-55s AD = % .8e  FD = % .8e  rel.err = %.2e\n", name, ad, fdv, abs(ad - fdv) / max(abs(fdv), 1e-300))

function try_report(f, name)
    try
        f()
    catch err
        msg = sprint(showerror, err)
        println("FAILED ", name, ": ", first(msg, parse(Int, get(ENV, "ERRLEN", "1500"))))
    end
end

constants = ThermodynamicConstants(Float64)
physics = DropletPhysicsParameters(Float64)
T, p = 282.0, 85000.0
r₀, rᵈ₀, κ₀, 𝒮₀, Δt = 5e-6, 5e-8, 0.6, 0.01, 0.1

if 'a' in levels
    println("=== Level a: single droplet")
    Φ(r, rᵈ, κ, 𝒮) = diffusional_growth_rate(r, rᵈ, κ, 𝒮, 1.0, T, p, constants, physics)
    for (name, i) in (("r", 1), ("rᵈ", 2), ("κ", 3), ("𝒮", 4))
        try_report("dΦ/d$name reverse") do
            args = [r₀, rᵈ₀, κ₀, 𝒮₀]
            g = Enzyme.autodiff(Reverse, Φ, Active, Active.(args)...)[1][i]
            f = x -> (a = copy(args); a[i] = x; Φ(a...))
            report("diffusional_growth_rate dΦ/d$name (reverse)", g, fd(f, args[i]))
        end
    end

    for (𝒮, Δt, label) in ((0.01, 0.1, "growing"), (-0.05, 0.1, "evaporating"), (0.01, 10.0, "large Δt"),
                           (0.0005, 1.0, "haze (S < Sc)"))
        step(r, rᵈ, κ, S) = condensational_growth_step(r, rᵈ, κ, S, 1.0, T, p, Δt, constants, physics).radius
        r = label == "haze (S < Sc)" ? 1.5e-7 : r₀
        for (name, i) in (("r", 1), ("rᵈ", 2), ("κ", 3), ("𝒮", 4))
            try_report("step $label d/d$name") do
                args = [r, rᵈ₀, κ₀, 𝒮]
                g = Enzyme.autodiff(Reverse, step, Active, Active.(args)...)[1][i]
                gf = Enzyme.autodiff(Forward, step, Duplicated, ntuple(j -> j == i ? Duplicated(args[j], 1.0) : Const(args[j]), 4)...)[1]
                f = x -> (a = copy(args); a[i] = x; step(a...))
                report("growth step [$label] dr⁺/d$name (reverse)", g, fd(f, args[i]))
                report("growth step [$label] dr⁺/d$name (forward)", gf, fd(f, args[i]))
            end
        end
    end
    # Multi-step parcel: 100 implicit steps at fixed 𝒮, scalar = final radius
    try_report("100 steps") do
        function hundred(r)
            for _ in 1:100
                r = condensational_growth_step(r, rᵈ₀, κ₀, 𝒮₀, 1.0, T, p, 0.05, constants, physics).radius
            end
            return r
        end
        g = Enzyme.autodiff(Reverse, hundred, Active, Active(r₀))[1][1]
        report("100 growth steps dr/dr₀ (reverse)", g, fd(hundred, r₀))
    end
end

L = 0.02
function droplet_box(arch; Nd=100, M=8)
    grid = RectilinearGrid(arch; size=(M, M, M), x=(0, L), y=(0, L), z=(0, L), topology=(Periodic, Periodic, Periodic))
    dynamics = BoussinesqDynamics(grid; pressure=p, temperature=T)
    rng = Xoshiro(1)
    droplets = (x = L .* rand(rng, Nd), y = L .* rand(rng, Nd), z = L .* rand(rng, Nd),
                r = 5e-6 .+ 5e-6 .* rand(rng, Nd), rᵈ = fill(5e-8, Nd), κ = fill(0.6, Nd))
    closure = ScalarDiffusivity(; ν=1.6e-5, κ=(ρθ=2.22e-5, ρqᵛ=2.55e-5))
    model = AtmosphereModel(grid; dynamics, closure, microphysics=LagrangianMicrophysics(; droplets))
    return model
end

function vapor_mass_fraction(S)
    pᵛ = (1 + S) * saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    return ε * pᵛ / (p - (1 - ε) * pᵛ)
end

# The state is reset by copying saved parent arrays (`set!` is not Enzyme-differentiable:
# it fails in `interpolation_operator` on a `getglobal`). The scalar is the total droplet volume
# after n steps, as a function of a relative perturbation δ of the initial vapor density and a
# scaling a of the initial droplet radii.
saved_state(model) = (; ρqᵛ = copy(parent(model.moisture_density)), ρθ = copy(parent(model.formulation.potential_temperature_density)),
                        u = copy(parent(model.velocities.u)), r = copy(model.microphysics.droplets.properties.r),
                        x = copy(model.microphysics.droplets.properties.x), y = copy(model.microphysics.droplets.properties.y),
                        z = copy(model.microphysics.droplets.properties.z))

function reset!(model, s, δ, a)
    parent(model.moisture_density) .= (1 + δ) .* s.ρqᵛ
    parent(model.formulation.potential_temperature_density) .= s.ρθ
    parent(model.velocities.u) .= s.u
    d = model.microphysics.droplets.properties
    d.r .= a .* s.r
    d.x .= s.x; d.y .= s.y; d.z .= s.z
    model.clock.time = 0
    model.clock.iteration = 0
    return nothing
end

function loss(model, δ, a, s, n)
    reset!(model, s, δ, a)
    for _ in 1:n
        time_step!(model, 0.01)
    end
    r = model.microphysics.droplets.properties.r
    J = 0.0
    for i in eachindex(r)
        J += r[i]^3
    end
    return J
end

function update_loss(model, δ, a, s)
    reset!(model, s, δ, a)
    model.clock.last_Δt = 0.01
    Breeze.AtmosphereModels.microphysics_model_update!(model.microphysics, model)
    r = model.microphysics.droplets.properties.r
    J = 0.0
    for i in eachindex(r)
        J += r[i]^3
    end
    return J
end

if 'b' in levels
    println("=== Level b: time_step! of an 8³ box with 100 droplets (CPU)")
    model = droplet_box(CPU())
    set!(model; T=T, qᵗ=vapor_mass_fraction(0.01), u=0.01)
    s = saved_state(model)
    fdδ = fd(δ -> update_loss(model, δ, 1.0, s), 0.0; h=1e-6)
    @printf("update only: FD dJ/dδ = %.6e\n", fdδ)
    try_report("microphysics_model_update! alone reverse") do
        dmodel = Enzyme.make_zero(model)
        g = autodiff(set_runtime_activity(Reverse), update_loss, Active, Duplicated(model, dmodel), Active(0.0), Const(1.0), Const(s))
        report("microphysics_model_update! dΣr³/dδ (reverse)", g[1][2], fdδ)
    end
    try_report("microphysics_model_update! alone forward") do
        dmodel = Enzyme.make_zero(model)
        g = autodiff(set_runtime_activity(Forward), update_loss, Duplicated(model, dmodel), Duplicated(0.0, 1.0), Const(1.0), Const(s))
        report("microphysics_model_update! dΣr³/dδ (forward)", g[1], fdδ)
    end
    for n in (1, 3)
        fdδ = fd(δ -> loss(model, δ, 1.0, s, n), 0.0; h=1e-6)
        fda = fd(a -> loss(model, 0.0, a, s, n), 1.0; h=1e-6)
        @printf("n=%d: J = %.6e, FD dJ/dδ = %.6e, FD dJ/da = %.6e\n", n, loss(model, 0.0, 1.0, s, n), fdδ, fda)
        try_report("Enzyme reverse, n=$n") do
            dmodel = Enzyme.make_zero(model)
            g = autodiff(set_runtime_activity(Reverse), loss, Active, Duplicated(model, dmodel), Active(0.0), Active(1.0), Const(s), Const(n))
            report("time_step! ×$n dΣr³/dδ (Enzyme reverse)", g[1][2], fdδ)
            report("time_step! ×$n dΣr³/da (Enzyme reverse)", g[1][3], fda)
        end
        try_report("Enzyme forward, n=$n") do
            dmodel = Enzyme.make_zero(model)
            g = autodiff(set_runtime_activity(Forward), loss, Duplicated(model, dmodel), Duplicated(0.0, 1.0), Const(1.0), Const(s), Const(n))
            report("time_step! ×$n dΣr³/dδ (Enzyme forward)", g[1], fdδ)
        end
    end
end

# The droplet kernels of `microphysics_model_update!` without the `update_state!` that follows
const PM = Breeze.ParticleMicrophysics
function droplet_kernels_loss(model, δ, a, s)
    reset!(model, s, δ, a)
    μ = model.microphysics
    grid = model.grid
    arch = architecture(grid)
    droplets = μ.droplets.properties
    constants = model.thermodynamic_constants
    ρ = Breeze.AtmosphereModels.dynamics_density(model.dynamics)
    p = Breeze.AtmosphereModels.dynamics_pressure(model.dynamics)
    pˢᵗ = Breeze.AtmosphereModels.standard_pressure(model.dynamics)
    C = model.microphysical_fields.C
    fill!(parent(C), 0)
    si = PM.self_interaction_parameters(μ.self_interaction, model.closure, grid, 0.01)
    launch!(arch, grid, KernelParameters(1:length(droplets)), PM._condense_droplets!, parent(C), halo_size(grid), droplets, grid,
            0.01, model.temperature, ρ, model.moisture_density, p, constants, μ.physics, si)
    launch!(arch, grid, :xyz, PM._apply_condensation!, model.formulation.potential_temperature_density,
            model.moisture_density, C, model.temperature, ρ, p, pˢᵗ, constants)
    J = 0.0
    r = droplets.r
    for i in eachindex(r)
        J += r[i]^3
    end
    ρqᵛ = parent(model.moisture_density) # include the grid so that the scatter (atomic adds) is differentiated
    for i in eachindex(ρqᵛ)
        J += 1e-12 * ρqᵛ[i]
    end
    return J
end

# Particle advection alone: one RK3 stage, scalar Σ x
function advection_loss(model, U, s)
    reset!(model, s, 0.0, 1.0)
    parent(model.velocities.u) .= U .* s.u
    Oceananigans.Models.LagrangianParticleTracking.resize_particle_time_stepper!(model.microphysics.droplets.timestepper, length(s.x))
    advect_lagrangian_particles!(model.microphysics.droplets, model, 0.01, Oceananigans.Models.LagrangianParticleTracking.particle_stage(model.timestepper, 1, true))
    x = model.microphysics.droplets.properties.x
    J = 0.0
    for i in eachindex(x)
        J += x[i]
    end
    return J
end

if 'k' in levels
    println("=== Level b': droplet kernels without update_state! (CPU)")
    model = droplet_box(CPU())
    set!(model; T=T, qᵗ=vapor_mass_fraction(0.01), u=0.01)
    s = saved_state(model)
    fdδ = fd(δ -> droplet_kernels_loss(model, δ, 1.0, s), 0.0; h=1e-6)
    fda = fd(a -> droplet_kernels_loss(model, 0.0, a, s), 1.0; h=1e-6)
    for mode in (Reverse, Forward)
        try_report("droplet kernels $mode") do
            dmodel = Enzyme.make_zero(model)
            if mode === Reverse
                g = autodiff(set_runtime_activity(Reverse), droplet_kernels_loss, Active, Duplicated(model, dmodel), Active(0.0), Active(1.0), Const(s))
                report("condense+apply kernels dJ/dδ (reverse)", g[1][2], fdδ)
                report("condense+apply kernels dJ/da (reverse)", g[1][3], fda)
            else
                g = autodiff(set_runtime_activity(Forward), droplet_kernels_loss, Duplicated(model, dmodel), Duplicated(0.0, 1.0), Const(1.0), Const(s))
                report("condense+apply kernels dJ/dδ (forward)", g[1], fdδ)
            end
        end
    end
    fdU = fd(U -> advection_loss(model, U, s), 1.0; h=1e-6)
    try_report("advection reverse") do
        dmodel = Enzyme.make_zero(model)
        g = autodiff(set_runtime_activity(Reverse), advection_loss, Active, Duplicated(model, dmodel), Active(1.0), Const(s))
        report("particle advection stage dΣx/dU (reverse)", g[1][2], fdU)
    end
end

if 'c' in levels
    println("=== Level c: Reactant")
    try_report("Reactant compile") do
        model = droplet_box(Oceananigans.Architectures.ReactantState())
        f = Reactant.@compile time_step!(model, 0.01)
        println("Reactant compiled time_step! with droplets")
    end
end
