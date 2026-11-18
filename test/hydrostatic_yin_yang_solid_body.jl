include(joinpath(@__DIR__, "setup.jl"))

#####
##### HydrostaticDynamics on a YinYangGrid: balanced solid-body rotation for five days, in two
##### orientations, with a uniform tracer and a tracer blob that cross the seam of the panels.
##### The conservation fixer is off: these are the results of the core as it is.
#####

using Breeze
using Breeze.HydrostaticEquations: accumulate_state_checks
using NCDatasets: NCDataset, defDim, defVar
using Oceananigans
using Test

include(joinpath(@__DIR__, "hydrostatic_yin_yang_utils.jl"))

#####
##### The flows
#####
#
#   * "zonal": the axis of the flow is the pole of the grid, on a planet at rest. The flow runs
#     along the northern and southern edges of the Yin panel and crosses its eastern and western
#     edges at right angles.
#   * "tilted": the same flow about an axis tilted by 45°, which crosses every edge obliquely.
#     Its dynamics are those of "zonal": only the orientation with respect to the grid differs.
#   * "rotating zonal": the zonal flow on the rotating planet, in geostrophic balance.
#
# The zonal flows also run on the latitude-longitude grid, between walls along which they flow.
# The blob starts on the equator of the flow and travels 155° in five days. In the zonal flows
# it starts at 100°E, leaves the Yin panel through its eastern edge (135°E) after a day, crosses
# the Yang panel, and enters the Yin panel again through its western edge (135°W) after four
# days: the two kinds of edge, since the velocities on the western and southern faces of a panel
# are prognostic and those on its eastern and northern faces are halo points. In the tilted flow
# it starts at 90°W and crosses the northern edge of the Yin panel at (0°, 45°N) after three days.
# (When the tolerances below were written the blob of the zonal flows started at 60°E and
# crossed the eastern edge only. The start was moved before any run of the complete test, when
# the difference between the two kinds of edge was noticed; the tolerances were not changed.)

const flows = ((name = "zonal", state = solid_body_rotation(α = 0, Ω = 0), center = (100.0, 0.0), latitude_longitude = true),
               (name = "tilted", state = solid_body_rotation(α = π / 4, Ω = 0), center = (-90.0, 0.0), latitude_longitude = false),
               (name = "rotating zonal", state = solid_body_rotation(α = 0, Ω = rotation_rate), center = (100.0, 0.0), latitude_longitude = true))

const days = 5

function run_solid_body(grid, flow, scheme, Δt)
    s = flow.state
    model = solid_body_model(grid, s; blob_center=flow.center, scheme.kw...)
    cells = sphere_cells(grid)
    u, v = model.velocities.u, model.velocities.v
    pˢ = model.dynamics.surface_pressure

    u⁰, v⁰, pˢ⁰ = panels(u), panels(v), panels(pˢ)
    C⁰ = contents(model)
    checks = [hydrostatic_state_check(model)]
    speed, pressure = [], []

    for day in 1:days
        for _ in 1:Int(86400 ÷ Δt)
            time_step!(model, Δt)
        end

        push!(checks, hydrostatic_state_check(model))
        push!(speed, seam_statistics(speed_error(panels(u), panels(v), u⁰, v⁰), cells))
        push!(pressure, seam_statistics(map((p, p⁰) -> abs.(p .- p⁰), panels(pˢ), pˢ⁰), cells))
    end

    drift = map((C, C⁰) -> (C / C⁰ - 1) / days, contents(model), C⁰)

    # The uniform mixing ratio and the blob, against the blob carried rigidly by the flow
    constancy = owned_maximum(map(χ -> χ ./ 3 .- 1, mixing_ratio(model, :c)), cells.owned)
    center = blob_center(s, flow.center, days * 86400)
    d = mixing_ratio(model, :d)
    exact = map((λ, φ) -> blob.(λ, φ, Ref(center)), cells.λ, cells.φ)
    difference = map((d, exact) -> d .- exact, d, exact)
    blob_error = masked_rms(difference, cells.area, cells.owned) / masked_rms(exact, cells.area, cells.owned)
    blob_minimum = minimum(minimum(d[r][cells.owned[r], :]) for r in eachindex(d))
    blob_maximum = owned_maximum(d, cells.owned)

    # The error of the velocity at day 5, for a map of one representative run
    error_map = map(e -> sqrt.(sum(abs2, e, dims=3)[:, :, 1] ./ size(e, 3)), speed_error(panels(u), panels(v), u⁰, v⁰))

    return (; valid = accumulate_state_checks(checks).valid, speed, pressure, drift,
              constancy, blob_error, blob_minimum, blob_maximum, error_map, cells)
end

# One representative field for figures: the error of the velocity at day 5, root-mean-square
# over the layers, on both panels, with the cells' coordinates, owned areas and seam band
function save_error_map(directory, name, result, attributes)
    Nx, Ny = size(first(result.error_map))
    NCDataset(joinpath(directory, name), "c"; attrib=attributes) do dataset
        defDim(dataset, "panel", length(result.error_map))
        defDim(dataset, "i", Nx)
        defDim(dataset, "j", Ny)
        stack3(panels) = permutedims(cat(panels...; dims=3), (3, 1, 2))
        variable(name, panels, units, long_name) =
            defVar(dataset, name, Float64.(stack3(panels)), ("panel", "i", "j"); attrib=["units" => units, "long_name" => long_name])
        variable("longitude", result.cells.λ, "degrees_east", "geographic longitude of the cell center")
        variable("latitude", result.cells.φ, "degrees_north", "geographic latitude of the cell center")
        variable("owned_area", result.cells.area, "m2", "area of the cell that its panel owns (zero: owned by the other panel)")
        variable("seam_band", result.cells.band, "1", "1 for owned cells within 2 cells of the curve dividing the panels")
        variable("speed_error", result.error_map, "m s-1",
                 "|u - u(0)| at day 5 of the steady solid-body rotation, root-mean-square over the layers; panel 1 is Yin, panel 2 Yang")
    end
    return nothing
end

function record_solid_body(case, result, U)
    for day in 1:days
        for (quantity, statistics) in (("speed error (m/s)", result.speed[day]), ("pˢ error (Pa)", result.pressure[day]))
            for name in keys(statistics)
                record("solid body", case, "day $day $quantity $name", statistics[name])
            end
        end
    end

    record("solid body", case, "speed error growth, day $days / day 1", result.speed[days].rms / result.speed[1].rms)
    record("solid body", case, "day $days largest speed error / U", result.speed[days].maximum / U)

    for name in keys(result.drift)
        record("drift", case, "relative drift per day of $name", result.drift[name])
    end

    record("tracers", case, "uniform mixing ratio, largest relative departure", result.constancy)
    record("tracers", case, "blob relative L2 error", result.blob_error)
    record("tracers", case, "blob minimum", result.blob_minimum)
    record("tracers", case, "blob maximum", result.blob_maximum)
    return nothing
end

#####
##### Tolerances, written before any of these runs
#####
#
# GATES (asserted):
#
#   1. Every daily state check is valid: every prognostic value is a number, every layer has
#      mass, and the temperature stays within (150, 350) K. Rationale: a steady flow that
#      cannot be held for five days is a failure whatever its error.
#
#   2. The largest error of the velocity at day 5 is below 5 % of U = 40 m s⁻¹, that is 2 m s⁻¹,
#      on both grids and at both resolutions. Rationale: the state is steady, so its truncation
#      error is a steady imbalance rather than a phase error that accumulates. For the
#      second-order operators the imbalance of the acceleration U² / a = 2.5e-4 m s⁻² of a field
#      of wavenumber 2 is (k Δ)² / 6 ≈ 3e-3 of it at Δ = 4° (0.07 rad): 8e-7 m s⁻². If it were
#      never balanced by an adjustment of the flow, five days of it would give 0.35 m s⁻¹, 0.9 %
#      of U. The bound is five times that, and the same at 2°, where the estimate is four times
#      smaller: the convergence is recorded rather than gated, because the vertical
#      discretization of the pressure-gradient force of an isothermal atmosphere on ten layers
#      leaves an imbalance that horizontal resolution does not reduce.
#
#   3. A uniform mixing ratio stays uniform through the seam: its largest relative departure at
#      day 5 is below 1e-11. Rationale: the halos of a uniform mixing ratio are interpolated to
#      the uniform value within a few eps(Float64), and each stage preserves ρc = χ ρ to
#      rounding. The longest run takes 3 × 2880 stages: 8640 × 2e-16 ≈ 2e-12 if every rounding
#      error had the same sign, and the bound is five times that.
#
#   4. The blob that the zonal flows carry through the seam has, at day 5, a relative L2 error on
#      the Yin-Yang grid below twice its error in the same run on the latitude-longitude grid.
#      Rationale: the operators are the same (fifth-order WENO with the mass fluxes of
#      continuity) and the cells along the path are within 2.3 % of the same size, which is
#      12 % in a fifth-order error; the Yin-Yang run adds one passage through the seam, whose
#      bicubic interpolation is of fourth order. A factor of two leaves room for that and
#      for the different truncation of the balanced flow that carries the blob; more would mean
#      that the seam dominates the error.
#
# MEASUREMENTS (recorded; only their finiteness is asserted):
#
#   * the error of the velocity and of the surface pressure at days 1 and 5: largest value,
#     root-mean-square over the sphere, over the seam band (the owned cells within 2 cells of
#     the curve that divides the sphere between the panels) and over the interior (the other
#     owned cells), the ratio band / interior, and the growth from day 1 to day 5;
#   * the ratio of the Yin-Yang error to the latitude-longitude error of the zonal flows;
#   * the relative drift per day of the owned integrals of the layer mass, of Θ, of the uniform
#     tracer and of the blob, without the conservation fixer;
#   * the error of the blob of the tilted flow, which has no latitude-longitude counterpart, and
#     the extrema of the blob.

@testset "Solid-body rotation [$(resolution.name), $(flow.name), $(scheme.name)]" for
        resolution in resolutions, flow in flows, scheme in schemes

    Δt = time_step(scheme, resolution)
    U = flow.state.U
    label(grid) = "$(grid_name(grid)), $(resolution.name), $(flow.name), $(scheme.name), Δt=$Δt"

    function gated_run(grid)
        result = run_solid_body(grid, flow, scheme, Δt)
        record_solid_body(label(grid), result, U)

        @test result.valid                                # gate 1
        @test result.speed[days].maximum < 0.05 * U       # gate 2
        @test all(isfinite, values(result.drift))
        @test isfinite(result.blob_error)
        return result
    end

    yin_yang = gated_run(yin_yang_grid(resolution))

    directory = results_directory()
    if !isnothing(directory) && flow.name == "tilted" && scheme.name == "SSP RK3"
        save_error_map(directory, "speed_error_day5_yin_yang_$(resolution.Nφ)_tilted_ssp_rk3.nc", yin_yang,
                       ["grid" => "YinYangGrid, Nφ = $(resolution.Nφ) ($(resolution.name)), 10 layers",
                        "flow" => "solid-body rotation, U = 40 m/s, axis tilted 45°, planet at rest",
                        "scheme" => scheme.name, "time_step_seconds" => Δt, "kind" => "measured",
                        "conservation_fixer" => "off"])
    end

    @test yin_yang.constancy < 1e-11                      # gate 3
    @test isfinite(yin_yang.speed[days].ratio)
    @test isfinite(yin_yang.pressure[days].ratio)

    if flow.latitude_longitude
        latitude_longitude = gated_run(latitude_longitude_grid(resolution))
        case = "$(resolution.name), $(flow.name), $(scheme.name), Δt=$Δt"

        record("solid body", case, "Yin-Yang / latitude-longitude day $days speed error rms",
               yin_yang.speed[days].rms / latitude_longitude.speed[days].rms)

        blob_ratio = record("tracers", case, "Yin-Yang / latitude-longitude blob error",
                            yin_yang.blob_error / latitude_longitude.blob_error)

        @test latitude_longitude.constancy < 1e-11
        @test blob_ratio < 2                              # gate 4
    end
end
