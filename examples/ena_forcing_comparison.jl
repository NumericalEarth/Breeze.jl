# # A common LES for 18 July 2017 at ARM's Eastern North Atlantic site
#
# Each run uses one constructor and loads external forcing with `set!`.
# The initial sounding, grid, microphysics, and radiation treatment are shared.
# This is a coarse forward experiment; differentiation is a later step.
# All cases use P3 with its default prescribed droplet number (200 cm⁻³),
# Khairoutdinov–Kogan warm-rain processes, and prognostic rain and ice moments.
#
# Radiation uses a CPU RRTMGP column driven by horizontal mean LES profiles.
# Dynamics and microphysics run on Metal. The radiation column extends the
# observed sounding above the LES lid, but neglects horizontal cloud variability.
#
# Run from the repository root with `julia --project=examples examples/ena_forcing_comparison.jl`.
# This runs Covert plus four LASSO forcing datasets for 06–15 UTC on 18 July 2017.
# LASSO inputs must first be extracted into `examples/data/ena/lasso/<forcing>/`.
# Each directory needs the archived `lsf` and `sfc` files. Request the four
# `morr-aer2-flxsst` input bundles at https://adc.arm.gov/lasso/#/ena for this date.
# `examples/download_lasso_ena.jl <ftp-directory>` downloads and extracts them.
# The default `era5` and `merra2` products average over 5°; `s1n0` uses 1°.
# `ENA_CASES` selects comma-separated case names; `ENA_DURATION_SECONDS` shortens runs.
#
# This controlled comparison uses prescribed archived H and LE, common stress,
# and no wind nudging. LASSO originally computes surface fluxes online and nudges
# mean winds over two hours. Its geostrophic winds come from lsf columns 8–9;
# Covert's large-scale winds are used as geostrophic targets. Initial conditions,
# surface stress, microphysics, domain, and radiation algorithm are common.
# These choices test sensitivity to forcing, rather than reproduce the parent LES.
#
# After running, execute `examples/download_ena_observations.jl` (ARM credentials
# in the environment), then `examples/compare_ena_results.jl`, with the same project.
# LWP is compared in matched 30-minute bins during 09–12 UTC, after three hours
# of spin-up. The 11:38 sounding provides a profile comparison, but is not an
# independent initialization check: it also underlies the idealized Covert sounding.
# Archived SAM results for the four LASSO forcings can also be compared by running
# `examples/compare_ena_models.jl`; see that file for data requirements and differences
# between the archived SAM configurations and these Breeze experiments.

using Breeze
using Oceananigans
using Metal

include("setups/ena_simulation.jl")
include("setups/ena_diagnostics.jl")

function run_ena_case(data; size=(64, 64, 75), stop_time=9*3600, run_until=stop_time,
                      output_directory=joinpath(@__DIR__, "ena_output", data.name))
    0 < run_until ≤ stop_time || throw(ArgumentError("run_until must lie in (0, stop_time]."))
    Oceananigans.defaults.FloatType = Float32
    simulation = ena_simulation(GPU(Metal.MetalBackend()); size, stop_time, Δt=2)
    set!(simulation, data)
    add_ena_diagnostics!(simulation, output_directory)
    conjure_time_step_wizard!(simulation; cfl=0.6, max_Δt=5)
    simulation.callbacks[:progress] = Callback(TimeInterval(1800)) do sim
        @info "ENA forecast" case=data.name time=sim.model.clock.time Δt=sim.Δt
        flush(stderr)
    end
    write(joinpath(output_directory, "source.txt"), data.source * "\n")
    write(joinpath(output_directory, "configuration.txt"),
          "date = 2017-07-18\nstart_UTC = 06:00\nduration_seconds = $stop_time\n" *
          "grid = $size\nextent_m = (6400, 6400, 3000)\narchitecture = Metal\n" *
          "initial_state = Covert sounding, shared\nmicrophysics = P3\n" *
          "cloud_droplet_number_m3 = $(simulation.model.microphysics.cloud.number_concentration)\n" *
          "warm_rain = $(summary(simulation.model.microphysics.warm_rain_scheme))\n" *
          "radiation = CPU RRTMGP mean column, 60 s updates\n" *
          "surface = prescribed H and LE; common Covert stress\nwind_nudging = none\n")
    simulation.stop_time = run_until
    run!(simulation)
    simulation.stop_time = stop_time # A pilot can resume without rebuilding or truncating forcing.
    return simulation
end

if abspath(PROGRAM_FILE) == @__FILE__
    duration = parse(Float64, get(ENV, "ENA_DURATION_SECONDS", "32400"))
    names = split(get(ENV, "ENA_CASES", "covert,era5,era5s1n0,merra2,merra2s1n0"), ',')
    cases = [name == "covert" ? covert_ena_case() : lasso_ena_case(name) for name in names]
    for data in cases
        run_ena_case(data; stop_time=duration)
        GC.gc()
    end
end
