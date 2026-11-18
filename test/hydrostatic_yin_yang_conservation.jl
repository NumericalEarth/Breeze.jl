include(joinpath(@__DIR__, "setup.jl"))

#####
##### HydrostaticDynamics on a YinYangGrid: the owned integrals of the layer mass, of Θ and of
##### the tracers, without the conservation fixer (the default) and with it
#####

using Breeze
using Oceananigans
using Oceananigans.MultiRegion: YinYangGrid
using Test

include(joinpath(@__DIR__, "hydrostatic_yin_yang_utils.jl"))

# A flow with horizontal divergence, vertical shear and a surface pressure wave, which crosses
# the seam of the panels everywhere, with a non-uniform potential temperature, a uniform tracer
# `c` (second-order centered) and a tracer blob `d` (fifth-order WENO), on six hybrid layers
function crossing_flow_model(grid; kw...)
    A, B = hybrid_coefficients(size(grid, 3))
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B)
    model = hydrostatic_model(grid; dynamics, coriolis=HydrostaticSphericalCoriolis(FT; rotation_rate),
                              tracers=(:c, :d), scalar_advection=(; ρθ=WENO(FT; order=5), d=WENO(FT; order=5)),
                              thermodynamic_constants=reference_model_constants(), kw...)

    set!(model; pˢ = (λ, φ) -> 1e5 + 500 * cosd(2λ) * cosd(φ)^2,
                θ = (λ, φ, z) -> 290 + 60 * z + 2 * sind(λ) * sind(2φ),
                u = (λ, φ, z) -> 20 * (1 + z) * cosd(φ) + 5 * sind(λ),
                v = (λ, φ, z) -> 5 * z * sind(2λ) * cosd(φ))

    ρ = continuity_density(model.dynamics)
    set!(model; c = 3 * ρ)
    set!(model; d = ρ * horizontal_field(grid, (λ, φ) -> blob(λ, φ, (-60.0, 20.0))))

    return model
end

#####
##### Tolerances, written before these runs
#####
#
# MEASUREMENT (recorded; finiteness asserted): without the fixer, the relative drift per day of
# the owned integrals of the layer mass, of Θ and of the two tracers over 20 time steps. The
# panels step all of their cells with their own fluxes and disagree on the flux through the
# curve that divides the sphere, so these integrals are not conserved. This is the behavior of
# the core as it is, and the number that matters for it.
#
# GATES (asserted), with the fixer:
#
#   1. Each of the four integrals after 20 time steps is within 1e-12 of its initial value,
#      relative, for SSP RK3 and for both split-explicit candidates. Rationale: the fixer
#      returns the seam budget of every stage, which it obtains from owned sums over ~ 2e4
#      cells; each sum and each stage update rounds at a few eps(Float64) = 2e-16, and 20 steps
#      take 60 stages: ~ 1e-14, with a margin of 100. This is the bound of the conservation
#      tests of the core on single grids (1e-13 unsplit, 1e-12 split).
#   2. The uniform mixing ratio stays uniform within 1e-11, relative: the fixer gives every
#      layer the scalar that the mass it adds carries, so it must not break what the
#      uncorrected scheme preserves.
#
# The fixer is global: it restores the integrals and does nothing for the fluxes at the seam.

@testset "Owned integrals [$(scheme.name), fixer $(fixer ? "on" : "off")]" for scheme in schemes, fixer in (false, true)
    resolution = first(resolutions)
    grid = yin_yang_grid(resolution; Nz=6)
    cells = sphere_cells(grid)
    Δt = time_step(scheme, resolution)
    steps = 20

    model = crossing_flow_model(grid; scheme.kw...)
    callbacks = fixer ? conservation_fixer_callbacks(model) : []
    pˢ⁰ = panels(model.dynamics.surface_pressure)
    C⁰ = contents(model)

    for _ in 1:steps
        time_step!(model, Δt; callbacks)
    end

    change = map((C, C⁰) -> C / C⁰ - 1, contents(model), C⁰)
    constancy = owned_maximum(map(χ -> χ ./ 3 .- 1, mixing_ratio(model, :c)), cells.owned)
    case = "$(resolution.name), $(scheme.name), Δt=$Δt, fixer $(fixer ? "on" : "off")"

    for name in keys(change)
        record("conservation", case, "relative change of $name over $steps steps", change[name])
        record("conservation", case, "relative drift per day of $name", change[name] * 86400 / (steps * Δt))
        @test isfinite(change[name])
        fixer && @test abs(change[name]) < 1e-12        # gate 1
    end

    record("conservation", case, "uniform mixing ratio, largest relative departure", constancy)
    @test constancy < 1e-11                             # gate 2, and without the fixer too

    # The state evolved
    pˢ = panels(model.dynamics.surface_pressure)
    @test maximum(r -> maximum(abs, pˢ[r] .- pˢ⁰[r]), eachindex(pˢ)) > 1
end

@testset "The conservation fixer of a simulation" begin
    grid = yin_yang_grid(first(resolutions); Nz=6)
    model = crossing_flow_model(grid)
    C⁰ = contents(model)
    simulation = Simulation(model; Δt=300, stop_iteration=5, verbose=false)
    add_conservation_fixer!(simulation)
    @test simulation.callbacks[:conservation_fixer].func isa HydrostaticConservationFixer
    run!(simulation)
    @test all(abs(C / C⁰ - 1) < 1e-12 for (C, C⁰) in zip(contents(model), C⁰))
end
