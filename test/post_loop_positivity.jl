include(joinpath(@__DIR__, "setup.jl"))

#####
##### Positivity of post-loop scalars under a multirate infinitesimal step
#####
##### Each stage of a coefficient set whose `β` is not diagonal transports the moisture and the
##### tracers with the mass flux Fⱼ + Rⱼ: its own loop's plus the correction that makes the
##### recombined fluxes carry each stage's own (`ScalarMassFluxCorrection`). One bounds-preserving
##### reconstruction carries both, and nothing else limits the correction. Rⱼ is a combination of
##### differences of the stages' mass fluxes, so it is largest where the flow accelerates and
##### diverges. A warm bubble that accelerates from rest in a sheared, convergent wind makes it so,
##### and sharp-edged scalars next to zeros show any flux that the reconstruction does not bound.
#####
##### The bounds-preserving limiter does not make the scalars exactly non-negative at these Courant
##### numbers: the undershoots below are those of the bounded scheme at a horizontal Courant number
##### near one, an order of magnitude and more below those of the unbounded scheme.
#####

using Breeze
using Breeze.AtmosphereModels: total_density
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: update_state!
using Test

const qᵛ₀ = 0.012

# `courant_number` is that of the initial wind, whose maximum is 25 m/s. The transport velocity
# reaches a Courant number about 15% higher as the bubble rises.
function sheared_bubble_scalars(; coefficients, courant_number, uniform = false, stop_time = 240,
                                scalar_advection = WENO(order = 5, bounds = (0, 1)))
    Nx, Nz, Δ = 32, 24, 200
    Lx, Lz = Nx * Δ, Nz * Δ
    grid = RectilinearGrid(default_arch, Float64; size = (Nx, Nz), halo = (5, 5), x = (-Lx / 2, Lx / 2), z = (0, Lz),
                           topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5,
                                    reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, momentum_advection = WENO(order = 5), scalar_advection, tracers = :ρc,
                            timestepper = :MultirateInfinitesimalStep,
                            timestepper_kwargs = (; coefficients, post_loop_scalar_tendencies = true))

    bubble(x, z) = cos(π * min(1, sqrt(x^2 + (z - 1200)^2) / 1000) / 2)^2
    uᵢ(x, z) = 20 * tanh((z - 2000) / 600) - 5 * sin(2π * x / Lx) * exp(-z / 1500)
    # A slab of vapor with sharp edges, and a tracer strip two cells wide.
    qᵢ(x, z) = uniform ? qᵛ₀ : ifelse(abs(x + 800) < 1000 && z < 1800, qᵛ₀, 0.0)
    cᵢ(x, z) = uniform ? 1.0 : ifelse(abs(x - 300) < Δ && 600 < z < 2400, 1.0, 0.0)
    set!(model; θ = (x, z) -> 300 + 8 * bubble(x, z), qᵛ = qᵢ, ρ = HydrostaticallyBalancedDensity(), u = uᵢ)
    ρ = total_density(model.dynamics)
    c = CenterField(grid)
    set!(c, cᵢ)
    parent(model.tracers.ρc) .= parent(c) .* parent(ρ)
    update_state!(model)

    Δt = courant_number * Δ / maximum(abs, interior(model.velocities.u))
    mass_fractions() = (q = Array(interior(prognostic_fields(model).ρqᵛ)) ./ Array(interior(ρ)),
                        c = Array(interior(model.tracers.ρc)) ./ Array(interior(ρ)))

    lowest = (q = Inf, c = Inf)
    highest = (q = -Inf, c = -Inf)
    for _ in 1:ceil(Int, stop_time / Δt)
        time_step!(model, Δt)
        fractions = mass_fractions()
        lowest = map((l, f) -> min(l, minimum(f)), lowest, fractions)
        highest = map((h, f) -> max(h, maximum(f)), highest, fractions)
    end

    return (; lowest, highest, maximum_w = maximum(abs, interior(model.velocities.w)))
end

@testset "Post-loop MIS scalars stay positive in an accelerating, sheared flow [$coefficients]" for
        coefficients in (:SSPMIS43, :TVDMISB)
    scalars = sheared_bubble_scalars(; coefficients, courant_number = 0.75)
    @test scalars.maximum_w > 5 # the bubble has accelerated

    # Lowest mass fractions over the run, relative to the initial maximum: -1.0e-4 (`:SSPMIS43`)
    # and -2.1e-4 (`:TVDMISB`) for the vapor, -1.2e-6 and -2.6e-6 for the tracer. The unbounded
    # scheme gives -5.2e-3 and -1.6e-4 or lower, and a Courant number of 1.4 gives -2.4e-2 and -3.7e-3.
    @test scalars.lowest.q ≥ -1e-3 * qᵛ₀
    @test scalars.lowest.c ≥ -2e-5

    # A uniform mass fraction stays uniform to round-off in the same flow.
    uniform = sheared_bubble_scalars(; coefficients, courant_number = 0.75, uniform = true, stop_time = 120)
    @test max(abs(uniform.lowest.q - qᵛ₀), abs(uniform.highest.q - qᵛ₀)) < 1e-12 * qᵛ₀
    @test max(abs(uniform.lowest.c - 1), abs(uniform.highest.c - 1)) < 1e-12
end

@testset "The positivity bound of the post-loop MIS scalars needs the bounded scheme" begin
    unbounded = sheared_bubble_scalars(; coefficients = :SSPMIS43, courant_number = 0.75, scalar_advection = WENO(order = 5))
    @test unbounded.lowest.q < -1e-3 * qᵛ₀
    @test unbounded.lowest.c < -2e-5
end
