include(joinpath(@__DIR__, "setup.jl"))

#####
##### HydrostaticDynamics on a YinYangGrid: a gravity wave that crosses the seam of the panels,
##### against the same wave propagating in the interior of a panel
#####

using Breeze
using Oceananigans
using Test

include(joinpath(@__DIR__, "hydrostatic_yin_yang_utils.jl"))

#####
##### The wave
#####
#
# A resting isothermal atmosphere at 300 K over flat ground on a planet at rest, with a bump of
# 100 Pa in its surface pressure: a Gaussian of the great-circle angle to its center, 12° in
# radius. It radiates gravity waves, the external mode first. Without rotation the solution
# depends on the great-circle angle d to the center alone, wherever the center is, so the same
# wave started at two places gives the same profile pˢ′(d) except for what the grid does to it:
#
#   * "interior": centered at (0°, 0°), the middle of the Yin panel. In two hours the crest of
#     the leading ring is 24° from the center and its front has not reached the seam, 45° away.
#   * "north": centered at (0°, 30°N). Northward the wave meets the seam after 15° and the crest
#     travels its last 9° on the Yang panel; southward it stays in the interior of the Yin panel.
#   * "south": centered at (0°, 30°S), the mirror image. Southward the wave crosses the
#     southern edge of the Yin panel, which is the western edge of the Yang panel there. These
#     are the edges where the pressure-gradient force of the first faces reads the halos; at the
#     northern edge of Yin and the eastern edge of Yang, which the "north" wave crosses, the
#     velocities on the edge are themselves halo points. (Added, with the gates below applied
#     to it unchanged, when that difference was noticed after the first runs.)
#
# (The duration was three hours when the tolerances below were written, on the estimate that
# the wave would travel 34°. The first run showed the crest of the ring, which starts 12° wide,
# at 36°: at the edge of the 4° to 38° over which the profiles are compared, with its front in
# the seam of the interior wave. The duration was shortened so that the whole ring is compared
# and the interior wave is free of the seam. The tolerances were not changed.)
#
# Profiles are the area-weighted kernel average of pˢ′ over the owned cells of a 90° sector of
# bearings, with a Gaussian kernel in d of width half a cell. The profile of a sector is
# compared with the northward profile of the interior wave by the shift δ in d and the amplitude
# factor s that minimize || P(d) - s P_interior(d - δ) || over 4° ≤ d ≤ 38°.

const wave_radius = 12.0     # degrees
const wave_amplitude = 100.0 # Pa
const wave_duration = 7200   # s
const wave_centers = (interior = (0.0, 0.0), north = (0.0, 30.0), south = (0.0, -30.0))
const background_temperature = 300.0

function gravity_wave_model(grid, center; kw...)
    A, B = hybrid_coefficients(size(grid, 3))
    dynamics = HydrostaticDynamics(surface_pressure=pˢᵗ, pressure_coefficient=A, surface_pressure_coefficient=B)
    model = hydrostatic_model(grid; dynamics, thermodynamic_constants=reference_model_constants(), kw...)

    # The potential temperature of the isothermal atmosphere at rest, kept while the surface
    # pressure is perturbed
    θ = CenterField(grid)
    set!(θ, background_temperature * (pˢᵗ / dynamics_pressure(model.dynamics))^κ)

    bump(λ, φ) = pˢᵗ + wave_amplitude * exp(-(great_circle_angle(λ, φ, center...) / wave_radius)^2)
    set!(model; pˢ=bump, θ=θ)

    return model
end

# Bearing from `center` to (λ, φ), in degrees from north
function bearing(λ, φ, center)
    λ₀, φ₀ = center
    return atand(sind(λ - λ₀) * cosd(φ), cosd(φ₀) * sind(φ) - sind(φ₀) * cosd(φ) * cosd(λ - λ₀))
end

in_sector(λ, φ, center, direction) = abs(rem(bearing(λ, φ, center) - direction, 360, RoundNearest)) ≤ 45

# Kernel average of the panel arrays `values` over the owned cells of the sector of `center`
# towards `direction`, at the great-circle angles `d`
function sector_profile(values, cells, center, direction, d, width)
    numerator, denominator = zero(d), zero(d)
    for r in eachindex(values), index in CartesianIndices(cells.area[r])
        a = cells.area[r][index]
        λ, φ = cells.λ[r][index], cells.φ[r][index]
        (a > 0 && in_sector(λ, φ, center, direction)) || continue
        dᶜ = great_circle_angle(λ, φ, center...)
        kernel = @. a * exp(-((d - dᶜ) / width)^2 / 2)
        numerator .+= kernel .* values[r][index, 1]
        denominator .+= kernel
    end
    return numerator ./ denominator
end

linear_interpolation(d, P, x) = (n = clamp(searchsortedlast(d, x), 1, length(d) - 1);
                                 P[n] + (P[n+1] - P[n]) * (x - d[n]) / (d[n+1] - d[n]))

# Shift δ (degrees, positive when `P` leads) and amplitude factor s of `P` against `reference`
function shift_and_amplitude(d, P, reference, Δ)
    fit = findall(x -> 4 ≤ x ≤ 38, d)
    best = (misfit = Inf, shift = NaN, amplitude = NaN)
    for δ in range(-2Δ, 2Δ, length=401)
        shifted = [linear_interpolation(d, reference, d[n] - δ) for n in fit]
        s = sum(P[fit] .* shifted) / sum(abs2, shifted)
        misfit = sqrt(sum(abs2, P[fit] .- s .* shifted) / sum(abs2, P[fit]))
        misfit < best.misfit && (best = (; misfit, shift = δ, amplitude = s))
    end
    return best
end

function run_gravity_wave(grid, center, scheme, Δt)
    model = gravity_wave_model(grid, center; scheme.kw...)
    C⁰ = (mass = total(continuity_density(model.dynamics)), Θ = total(model.formulation.potential_temperature_density))

    for _ in 1:Int(wave_duration ÷ Δt)
        time_step!(model, Δt)
    end

    C = (mass = total(continuity_density(model.dynamics)), Θ = total(model.formulation.potential_temperature_density))
    drift = map((C, C⁰) -> (C / C⁰ - 1) * 86400 / wave_duration, C, C⁰)
    pˢ′ = map(p -> p .- pˢᵗ, panels(model.dynamics.surface_pressure))
    return pˢ′, drift
end

#####
##### Tolerances, written before any of these runs
#####
#
# GATES (asserted), for the profile of each wave that crosses the seam, in the direction in
# which it crosses, against the northward profile of the interior wave:
#
#   1. Phase: the shift |δ| is at most half a cell, Δ / 2. Rationale: on the C-grid a wave of
#      wavenumber k is slow by (k Δ)² / 24, which for k = 1 / (12°) is 0.5 % at Δ = 4.09° and
#      0.12 % at 2°: 0.17° and 0.04° over the 34° travelled. Both profiles carry that error, and
#      differ only by its dependence on the direction and on the latitude of the path, so they
#      should agree within a fraction of it. Half a cell (2° and 1°) is what the kernel average,
#      half a cell wide, can resolve with confidence; a seam that delayed or advanced the wave
#      by more would be visible in any plot.
#
#   2. Amplitude: the factor s is within 10 % of one. Rationale: the spreading of the wave over
#      the sphere depends on d alone, and the bicubic interpolation of the halos has no
#      dissipation of its own; partial reflection or absorption at the seam of more than a tenth
#      of the wave would be a defect of the coupling of the panels.
#
# MEASUREMENTS (recorded; finiteness asserted): the misfit that remains after the shift and the
# amplitude factor; the same three numbers for the southward profile of the "north" wave,
# which never leaves the interior of the Yin panel and so measures what the comparison shows
# without a seam; the relative change per day of the owned mass and Θ integrals of the "north"
# wave, without the conservation fixer.

@testset "Gravity wave across the seam [$(resolution.name), $(scheme.name)]" for
        resolution in resolutions, scheme in schemes

    grid = yin_yang_grid(resolution)
    cells = sphere_cells(grid)
    Δ = 90 / resolution.Nφ
    Δt = time_step(scheme, resolution)
    d = collect(0:Δ/8:48)
    case = "$(resolution.name), $(scheme.name), Δt=$Δt"

    interior_wave, _ = run_gravity_wave(grid, wave_centers.interior, scheme, Δt)
    north_wave, drift = run_gravity_wave(grid, wave_centers.north, scheme, Δt)
    south_wave, _ = run_gravity_wave(grid, wave_centers.south, scheme, Δt)

    reference = sector_profile(interior_wave, cells, wave_centers.interior, 0, d, Δ / 2)
    across_north = sector_profile(north_wave, cells, wave_centers.north, 0, d, Δ / 2)
    across_south = sector_profile(south_wave, cells, wave_centers.south, 180, d, Δ / 2)
    inward = sector_profile(north_wave, cells, wave_centers.north, 180, d, Δ / 2)

    # The wave has left its center and is resolved by the profile
    @test maximum(abs, reference) > wave_amplitude / 100
    record("gravity wave", case, "largest |pˢ′| of the interior profile (Pa)", maximum(abs, reference))
    record("gravity wave", case, "angle of the largest |pˢ′| of the interior profile (degrees)", d[argmax(abs.(reference))])

    # The profiles themselves, for figures
    directory = results_directory()
    if !isnothing(directory)
        open(joinpath(directory, "gravity_wave_profiles.csv"), "a") do io
            for (name, profile) in (("interior", reference), ("across the northern edge of Yin", across_north),
                                    ("across the southern edge of Yin", across_south), ("interior control", inward))
                for n in eachindex(d)
                    println(io, join((resolution.name, scheme.name, Δt, name, d[n], profile[n]), ','))
                end
            end
        end
    end

    north = shift_and_amplitude(d, across_north, reference, Δ)
    south = shift_and_amplitude(d, across_south, reference, Δ)
    control = shift_and_amplitude(d, inward, reference, Δ)

    for (name, fit) in (("across the northern edge of Yin", north), ("across the southern edge of Yin", south),
                        ("interior control", control))
        record("gravity wave", case, "$name: shift (degrees)", fit.shift)
        record("gravity wave", case, "$name: shift / cell", fit.shift / Δ)
        record("gravity wave", case, "$name: amplitude factor", fit.amplitude)
        record("gravity wave", case, "$name: remaining misfit", fit.misfit)
        @test all(isfinite, values(fit))
    end

    for across in (north, south)
        @test abs(across.shift) ≤ Δ / 2          # gate 1
        @test abs(across.amplitude - 1) ≤ 0.1    # gate 2
    end

    for name in keys(drift)
        record("drift", "gravity wave, " * case, "relative drift per day of $name", drift[name])
        @test isfinite(drift[name])
    end
end
