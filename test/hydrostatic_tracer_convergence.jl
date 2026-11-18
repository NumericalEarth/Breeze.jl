include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.HydrostaticEquations: VerletSubsteps, ForwardBackwardSubsteps
using Oceananigans
using Test

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

# Accuracy of the transport of a tracer that is not uniform. The other tracer tests use a
# uniform mixing ratio, which tests that the mass fluxes that carry the tracers are those of
# continuity, and says nothing about the accuracy of the transport. Every expectation and
# tolerance below was written before the first run of the test it belongs to.

const tableaux = (:MIS3C, :MIS4a, :WickerSkamarock3)
const inner_solvers = (VerletSubsteps(), ForwardBackwardSubsteps(divergence_damping=0.1))
const configurations = [(tableau, inner_solver) for tableau in tableaux for inner_solver in inner_solvers]

# Relative L² norm of the difference of two arrays of mixing ratio on a uniform grid
relative_l2(χ, reference) = sqrt(sum(abs2, χ .- reference) / sum(abs2, reference))

function mixing_ratio(model, name)
    ρ = Array(interior(continuity_density(model.dynamics)))
    return Array(interior(model.tracers[name])) ./ ρ
end

function set_mixing_ratios!(model, χ)
    field = CenterField(model.grid)
    set!(field, χ)
    ρ = continuity_density(model.dynamics)
    set!(model; c = field * ρ, d = field * ρ)
    return nothing
end

#####
##### (1) A smooth blob in a uniform flow that accelerates: convergence in space and time
#####
#
# A resting isentropic atmosphere, uniform along x, moving with u(t) = U + ΔU sin(2π t / P) under
# the uniform acceleration ∂ₜu = (2π ΔU / P) cos(2π t / P). The flow is uniform and without
# divergence, so the mixing ratio is carried without change of shape,
#
#   χ(x, z, t) = χ₀(x − X(t), z),      X(t) = U t + ΔU P (1 − cos(2π t / P)) / 2π ,
#
# which is the exact solution. χ₀ = exp(4 [cos(2π (x − x₀) / L) − 1]) (1 + z): a smooth periodic
# blob about 250 km wide in a channel of L = 3200 km, different on every level. U = 20 m s⁻¹,
# ΔU = 10 m s⁻¹, P = T = 80 000 s, so that the blob crosses half the channel.
#
# The acceleration makes the velocity differ from its value at the start of each time step,
# so the split schemes transport the tracer along both of their paths: the full tendency, and
# the transport of the mixing ratio frozen at the start of the step by the accumulated mass
# flux of the velocity perturbation. The perturbation flux is uniform here; a perturbation flux
# with divergence is the subject of (2).
#
# The grid and the time step are refined together: Nx = 32, 64, 128 with Δt = 1600, 800, 400 s
# for the split schemes (Courant number 0.48 at 30 m s⁻¹) and Δt = 200, 100, 50 s for SSP RK3,
# whose time step the gravity waves bound (2 c / Δx: 247 s at Nx = 32). Tracer `c` is advected
# by the second-order centered scheme and `d` by WENO of order 5.
#
# Expectations for the order of the L² error, from the two finer grids:
#
#   * `c`, every scheme: 2 ± 0.3. The centered scheme's dispersion error, (k Δx)² / 6 in phase
#     speed, dominates the time integration error at these Courant numbers.
#   * `d`, SSP RK3: at least 2.7. The scheme is fifth order in space where the mixing ratio is
#     smooth and monotone and third order in time, at a Courant number of 0.06.
#   * `d`, split schemes: at least 1.7. The mixing ratio that the perturbation flux carries is
#     the one at the start of the step, which is second order in Δt; whether that error or the
#     spatial error is the larger at these resolutions is not known in advance.
#   * Split against unsplit, tracer `c` on the finest grid: the split error is within a factor
#     of 2 of the SSP RK3 error, at a time step 8 times as long, because the two share the
#     spatial discretization that dominates. The ratio is logged for both tracers.
#
# The errors must also decrease from each grid to the next, for every scheme and tracer.

const channel_length = 3.2e6
const duration = 8e4
const mean_speed = 20.0
const speed_variation = 10.0

blob(x, z) = exp(4 * (cos(2π * (x - 8e5) / channel_length) - 1)) * (1 + z)
displacement(t) = mean_speed * t + speed_variation * duration * (1 - cos(2π * t / duration)) / 2π
uniform_acceleration(x, z, t) = 2π * speed_variation / duration * cos(2π * t / duration)

function blob_errors(Nx, Δt; kw...)
    grid = RectilinearGrid(default_arch, FT; size=(Nx, 4), x=(0, channel_length), z=(0, 1),
                           topology=(Periodic, Flat, Bounded))
    model = hydrostatic_model(grid; dynamics=HydrostaticDynamics(top_pressure=200), tracers=(:c, :d),
                              scalar_advection=(; d=WENO(FT; order=5)), forcing=(; u=uniform_acceleration), kw...)
    set!(model; θ=300, u=mean_speed)
    set_mixing_ratios!(model, blob)

    for _ in 1:round(Int, duration / Δt)
        time_step!(model, Δt)
    end

    @test model.clock.time ≈ duration

    # The flow stays uniform and returns to U: the tracer is carried by a known displacement
    u = Array(interior(model.velocities.u))
    @test maximum(abs, u .- mean_speed) < 1e-6
    @test max_abs(model.velocities.w) < 1e-12

    exact = CenterField(grid)
    set!(exact, (x, z) -> blob(x - displacement(duration), z))
    exact = Array(interior(exact))

    return (c = relative_l2(mixing_ratio(model, :c), exact), d = relative_l2(mixing_ratio(model, :d), exact))
end

const resolutions = (32, 64, 128)

@testset "Blob in an accelerating uniform flow: convergence in space and time" begin
    unsplit = [blob_errors(Nx, Δt) for (Nx, Δt) in zip(resolutions, (200, 100, 50))]
    unsplit_orders = (c = log2(unsplit[2].c / unsplit[3].c), d = log2(unsplit[2].d / unsplit[3].d))
    @info "Tracer blob, SSPRungeKutta3 (Δt = 200, 100, 50 s): L² errors of c $([e.c for e in unsplit]), " *
          "of d $([e.d for e in unsplit]); orders on the two finer grids $unsplit_orders"

    @test unsplit[1].c > unsplit[2].c > unsplit[3].c
    @test unsplit[1].d > unsplit[2].d > unsplit[3].d
    @test abs(unsplit_orders.c - 2) < 0.3
    @test unsplit_orders.d > 2.7

    @testset "$(configuration_name(tableau, inner_solver))" for (tableau, inner_solver) in configurations
        split = [blob_errors(Nx, Δt; split_explicit(tableau, inner_solver)...) for (Nx, Δt) in zip(resolutions, (1600, 800, 400))]
        orders = (c = log2(split[2].c / split[3].c), d = log2(split[2].d / split[3].d))
        ratios = (c = split[3].c / unsplit[3].c, d = split[3].d / unsplit[3].d)
        @info "Tracer blob, $(configuration_name(tableau, inner_solver)) (Δt = 1600, 800, 400 s): L² errors of c " *
              "$([e.c for e in split]), of d $([e.d for e in split]); orders on the two finer grids $orders; " *
              "ratio to the SSP RK3 errors on the finest grid $ratios"

        @test split[1].c > split[2].c > split[3].c
        @test split[1].d > split[2].d > split[3].d
        @test abs(orders.c - 2) < 0.3
        @test orders.d > 1.7
        @test ratios.c < 2
    end
end

#####
##### (2) A blob in a flow with divergence that changes in time: convergence in time
#####
#
# The flow of the temporal convergence test of `hydrostatic_split_explicit.jl` (a wave in the
# surface pressure, a divergent sheared velocity, a stratified atmosphere) on a mean flow of
# 10 m s⁻¹, with the blob of (1). The velocity perturbation of a time step now has divergence
# and a vertical mass flux, and it varies along the tracer's gradient: this is the part of the
# split schemes' tracer transport that (1) does not exercise. There is no exact solution, so the
# reference is SSP RK3 at Δt = 12.5 s on the same grid, and the L² error measured is that of the
# time integration alone. T = 4800 s.
#
# Expectations for the order, from the two finer time steps, within ± 0.3:
#
#   * SSP RK3 (Δt = 100, 50 s): 3, for both tracers.
#   * Split schemes (Δt = 400, 200, 100 s, with the substeps per time step of
#     `hydrostatic_split_explicit.jl`): the order of the velocity that carries the tracer,
#     which is 2 with Verlet substeps and 1 with forward-backward substeps.
#   * The order of u for MIS4a with Verlet substeps is not established in that test (recorded
#     there as broken), so the order of the tracers it carries is not expected either: it is
#     measured, logged, and recorded as broken if it is outside the bound.
#
# WENO's weights depend on the solution, which may blur the order of `d`; the bound is kept
# and an order outside it is a finding.

const tableau_substeps = (MIS3C = 10, MIS4a = 11, WickerSkamarock3 = 6)

@testset "Blob in a divergent unsteady flow: convergence in time" begin
    Nx, Nz = 32, 8
    k = 2π / channel_length
    grid = RectilinearGrid(default_arch, FT; size=(Nx, Nz), x=(0, channel_length), z=(0, 1),
                           topology=(Periodic, Flat, Bounded))

    function solution(Δt; kw...)
        model = hydrostatic_model(grid; dynamics=HydrostaticDynamics(top_pressure=200), tracers=(:c, :d),
                                  scalar_advection=(; d=WENO(FT; order=5)), kw...)
        set!(model; pˢ = x -> 1e5 + 200 * cos(k * x),
                    θ = (x, z) -> 300 + 40 * z,
                    u = (x, z) -> 10 + 5 * (1 + z) * sin(k * x))
        set_mixing_ratios!(model, blob)
        initial = mixing_ratio(model, :c)

        for _ in 1:round(Int, 4800 / Δt)
            time_step!(model, Δt)
        end

        @test model.clock.time ≈ 4800
        return (c = mixing_ratio(model, :c), d = mixing_ratio(model, :d), initial = initial,
                w = max_abs(model.velocities.w))
    end

    reference = solution(12.5)
    @test reference.w > 0                                        # mass crosses the coordinate surfaces
    @test relative_l2(reference.c, reference.initial) > 1e-2     # the tracer is transported

    function orders(solutions, label)
        return map((:c, :d)) do name
            errors = [relative_l2(solution[name], reference[name]) for solution in solutions]
            order = log2(errors[end-1] / errors[end])
            @info "Tracer blob in a divergent flow, $label, $name: L² errors $errors, order on the two finer steps $order"
            @test errors[end] > 1e3 * eps(FT) # above rounding
            order
        end
    end

    @testset "SSPRungeKutta3" begin
        for order in orders([solution(Δt) for Δt in (100, 50)], "SSPRungeKutta3")
            @test abs(order - 3) < 0.3
        end
    end

    @testset "$(configuration_name(tableau, inner_solver))" for (tableau, inner_solver) in configurations
        substeps = tableau_substeps[tableau]
        solutions = [solution(Δt; split_explicit(tableau, inner_solver; substeps)...) for Δt in (400, 200, 100)]
        expected_order = inner_solver isa VerletSubsteps ? 2 : 1
        not_established = tableau === :MIS4a && inner_solver isa VerletSubsteps

        for order in orders(solutions, configuration_name(tableau, inner_solver))
            within_bound = abs(order - expected_order) < 0.3
            if not_established && !within_bound
                @test_broken within_bound
            else
                @test within_bound
            end
        end
    end
end
