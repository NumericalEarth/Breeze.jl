include(joinpath(@__DIR__, "setup.jl"))

#####
##### HydrostaticDynamics on a YinYangGrid: construction, and the resting isentropic atmosphere
##### over flat ground and over a mountain that lies across the seam of the two panels
#####

using Breeze
using Breeze.HydrostaticEquations: HydrostaticModel
using Oceananigans
using Test

include(joinpath(@__DIR__, "hydrostatic_yin_yang_utils.jl"))

@testset "HydrostaticDynamics on a YinYangGrid: construction" begin
    grid = yin_yang_grid(first(resolutions); Nz=4)
    model = hydrostatic_model(grid; dynamics=HydrostaticDynamics(top_pressure=200), tracers=:c)
    @test model isa HydrostaticModel
    @test keys(Oceananigans.prognostic_fields(model)) == (:pˢ, :u, :v, :ρθ, :ρqᵛ, :c)

    # Velocities are set from their geographic components; the prognostic variables are the
    # velocities, so there is no momentum density to set
    @test_throws ArgumentError set!(model; ρu=1)
    @test_throws ArgumentError set!(model; w=1)

    # A zonal flow has both components on the Yang panel, whose basis is rotated
    set!(model; θ=300, u = (λ, φ, z) -> 10 * cosd(φ))
    u, v = panels(model.velocities.u), panels(model.velocities.v)
    @test maximum(abs, v[1]) < 1e-12
    @test maximum(abs, v[2]) > 1
    # The speed is that of the zonal flow on both panels, as far as components that are
    # staggered by half a cell can tell
    cells = sphere_cells(grid)
    speed = map((u, v) -> sqrt.(u .^ 2 .+ v .^ 2), u, v)
    @test 9.9 < owned_maximum(speed, cells.owned) < 10.1

    # The owned areas of the two panels tile the sphere, and the seam band is a part of them
    @test sum(sum, cells.area) ≈ 4π * sphere_radius^2 rtol=1e-12
    band_area = sum(sum(cells.area[r] .* cells.band[r]) for r in 1:2)
    @test 0 < band_area < sum(sum, cells.area) / 2
    @info "The seam band (owned cells within 2 cells of the dividing curve) holds $(band_area / sum(sum, cells.area)) of the sphere at $(first(resolutions).name)"
end

#####
##### (a) Resting isentropic atmosphere
#####
#
# With uniform θ = θ₀ the pressure-gradient acceleration is -∇(Φ + cᵖᵈ θ₀ Π), and Φ + cᵖᵈ θ₀ Π
# = Φˢ + cᵖᵈ θ₀ Πˢ in every layer, which the balanced surface pressure makes uniform over any
# terrain. On a Yin-Yang grid the halos of the geopotential Φ and of the layer Exner function Π
# are interpolated from the other panel. The interpolation is linear with weights that sum to
# one, so it returns the uniform value of Φ + cᵖᵈ θ₀ Π: the state is at rest on the two panels,
# with the mountain across their seam, to rounding error. (Diagnosing Φ and Π in the halo
# columns from an interpolated surface pressure would not do this: Π is not linear in pˢ.)
#
# Tolerance rationale, written before the test was run for 100 steps. Φ + cᵖᵈ θ₀ Π ≈ 3e5 m² s⁻²,
# so rounding in the diagnostics and in the interpolation (weights of order one, a few eps each)
# leaves differences of ~ 1e-10 m² s⁻² between neighbors: an acceleration of ~ 1e-10 / 1.4e5 m
# ≈ 1e-15 m s⁻² at 2°, allowed to be ten times larger. If it acted in one direction for the
# 100 steps of 1200 s of the longest run, it would give 1e-14 × 1.2e5 ≈ 1e-9 m s⁻¹. The bound is
# 1e-8 m s⁻¹ for both velocity components, for every scheme, grid and resolution, and 1e-5 Pa
# for the change of the surface pressure, which the divergence of such a velocity produces
# (∂ₜpˢ ~ pˢ ∇·u ~ 1e5 × 1e-8 / 1.4e5 ≈ 1e-8 Pa s⁻¹ × 1.2e5 s ≈ 1e-3 Pa if every column
# converged at once; rounding is incoherent, so two orders less).
#
# Disclosure: a development probe had run this state for 21 steps at 4° before the tolerance
# was written (largest |u| between 1.5e-12 and 3.8e-12 m s⁻¹ for the three schemes).
#
# What is recorded, not gated: the ratio of the departure on the Yin-Yang grid to that of the
# same state on the latitude-longitude grid.

# `case` names the run whose largest velocity is recorded every ten steps, for figures
function departure_from_rest(model, steps, Δt; case=nothing)
    pˢ⁰ = panels(model.dynamics.surface_pressure)
    for step in 1:steps
        time_step!(model, Δt)
        if !isnothing(case) && step % 10 == 0
            speed = max(maximum(abs, model.velocities.u), maximum(abs, model.velocities.v))
            record("rest series", case, "step $step largest velocity component (m/s)", speed)
        end
    end
    pˢ = panels(model.dynamics.surface_pressure)
    return (u = maximum(abs, model.velocities.u),
            v = maximum(abs, model.velocities.v),
            w = maximum(abs, model.velocities.w),
            pˢ = maximum(r -> maximum(abs, pˢ[r] .- pˢ⁰[r]), eachindex(pˢ)))
end

@testset "Resting isentropic atmosphere [$(resolution.name), $terrain, $(scheme.name)]" for
        resolution in resolutions, terrain in (:flat, :mountain), scheme in schemes

    Δt = time_step(scheme, resolution)
    departures = map((yin_yang_grid(resolution), latitude_longitude_grid(resolution))) do grid
        model = resting_isentropic_model(grid, terrain; scheme.kw...)

        if terrain === :mountain
            pˢ = model.dynamics.surface_pressure
            @test maximum(pˢ) - minimum(pˢ) > 1e4 # the terrain is felt
        end

        case = "$(grid_name(grid)), $(resolution.name), $terrain, $(scheme.name), Δt=$Δt"
        departure = departure_from_rest(model, 100, Δt; case)
        record("rest", case, departure)

        @test departure.u < 1e-8
        @test departure.v < 1e-8
        @test departure.pˢ < 1e-5
        @test isfinite(departure.w)
        departure
    end

    yin_yang, latitude_longitude = departures
    speed(d) = max(d.u, d.v)
    ratio = record("rest", "$(resolution.name), $terrain, $(scheme.name), Δt=$Δt",
                   "Yin-Yang / latitude-longitude largest velocity", speed(yin_yang) / speed(latitude_longitude))
    @test isfinite(ratio) || speed(latitude_longitude) == 0 # both exactly at rest
end
