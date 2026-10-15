# Compare exported moist-air mass fractions on the common grid and output times.
# All figures are generated with Julia/CairoMakie.
using DelimitedFiles
using Statistics
using Printf
using CairoMakie

output_directory = get(ENV, "P3_COMPARISON_OUTPUT",
                       normpath(joinpath(@__DIR__, "../../validation_output/p3_kin1d")))

function read_profiles(filename)
    rows = readdlm(filename, Float64; comments=true)
    all(isfinite, rows) || error("Nonfinite reference output: $filename")
    times = sort(unique(rows[:, 1]))
    heights = sort(unique(rows[:, 2]))
    times == collect(0:30:3600) || error("Incomplete output in $filename")
    size(rows) == (length(times) * length(heights), 15) || error("Incorrect output shape")
    data = Array{Float64}(undef, length(heights), length(times), 15)
    for (n, t) in enumerate(times)
        snapshot = rows[rows[:, 1] .== t, :]
        data[:, n, :] .= snapshot[sortperm(snapshot[:, 2]), :]
    end
    return (; times, heights, data)
end

reference = read_profiles(joinpath(output_directory, "kin1d_dt1.0/profiles.dat"))
reference_half = read_profiles(joinpath(output_directory, "kin1d_dt0.5/profiles.dat"))
breeze = read_profiles(joinpath(output_directory, "breeze_fixed_dt1.0.dat"))
breeze_half = read_profiles(joinpath(output_directory, "breeze_fixed_dt0.5.dat"))
upwind = read_profiles(joinpath(output_directory, "breeze_upwind_dt1.0.dat"))
reservoir = read_profiles(joinpath(output_directory, "breeze_reservoir_dt1.0.dat"))
reference_options = read_profiles(joinpath(output_directory, "breeze_reference_options_dt1.0.dat"))
dry_reference = read_profiles(joinpath(output_directory, "dry/kin1d_dt1.0/profiles.dat"))
dry_breeze = read_profiles(joinpath(output_directory, "dry/breeze_dry_dt1.0.dat"))
@assert dry_breeze.data[:, 1, 3] ≈ dry_reference.data[:, 1, 3] rtol=1e-12
@assert dry_breeze.heights == dry_reference.heights == reference.heights
@assert all(iszero, dry_breeze.data[:, :, 6:13])
@assert all(iszero, dry_reference.data[:, :, 6:13])

for run in (reference, reference_half, breeze, breeze_half, upwind, reservoir, reference_options)
    @assert run.heights == reference.heights
    @assert run.times == reference.times
    @assert run.data[:, 1, 3] ≈ reference.data[:, 1, 3] rtol=1e-12
    @assert run.data[:, 1, 4] ≈ reference.data[:, 1, 4] rtol=1e-12
    @assert run.data[:, 1, 6] ≈ reference.data[:, 1, 6] rtol=1e-12
    for c in 6:13
        @assert minimum(run.data[:, :, c]) >= -64eps(maximum(abs, run.data[:, :, c]))
    end
end
@assert length(reference.heights) == 100
@assert all(diff(reference.heights) .== 200)

# Use the shared initial reference density for every integrated profile. This
# isolates differences in mass fraction from the hosts' different density rules;
# these are common-density column integrals, not a precipitation budget closure.
weights = 200 .* breeze.data[:, 1, 5]
column(run, c) = vec(sum(weights .* run.data[:, :, c]; dims=1))
weighted_error(run, base, c) =
    sum(weights .* abs.(run.data[:, 2:end, c] - base.data[:, 2:end, c])) /
    sum(weights .* abs.(base.data[:, 2:end, c]))
profile_rms(run, base, c) = sqrt(mean(abs2, run.data[:, 2:end, c] - base.data[:, 2:end, c]))
fields = ((6, "vapor"), (7, "cloud"), (8, "rain"), (9, "frozen ice"), (10, "coating"))

open(joinpath(output_directory, "metrics.txt"), "w") do io
    println(io, "P3 kin1d comparison: 100 cells, 200 m spacing, 2 m/s ascent, 3600 s")
    println(io, "All runs finite; identical initial T, pressure, vapor fraction, grid and output times verified.")
    println(io, "Errors exclude t=0. Mass error = density-weighted space-time L1 error / reference L1 norm.")
    println(io, "Column amounts use the same initial density in both models (kg/m²).")
    @printf(io, "Dry control temperature RMS difference: %.6g K; all water remains exactly zero.\n",
            profile_rms(dry_breeze, dry_reference, 3))
    @printf(io, "Dry control maximum temperature difference: %.6g K\n",
            maximum(abs, dry_breeze.data[:, :, 3] - dry_reference.data[:, :, 3]))
    for (label, run, base) in (("Breeze fixed aerosol vs kin1d", breeze, reference),
                                ("Breeze upwind vs kin1d", upwind, reference),
                                ("Breeze original reservoir vs kin1d", reservoir, reference),
                                ("Breeze reference options vs kin1d", reference_options, reference),
                                ("kin1d dt=0.5 vs dt=1", reference_half, reference),
                                ("Breeze dt=0.5 vs dt=1", breeze_half, breeze))
        println(io, "\n", label)
        @printf(io, "Temperature RMS difference: %.6g K\n", profile_rms(run, base, 3))
        @printf(io, "%-12s %15s %19s %19s\n", "field", "L1 error (%)", "run final column", "base final column")
        for (c, name) in fields
            @printf(io, "%-12s %15.6g %19.8g %19.8g\n", name,
                    100weighted_error(run, base, c), last(column(run, c)), last(column(base, c)))
        end
    end
    for (label, run) in (("Breeze", breeze), ("kin1d", reference))
        @printf(io, "\n%s max |prognostic - diagnosed Ssat|: %.8g kg/kg\n", label,
                maximum(abs, run.data[:, :, 14] - run.data[:, :, 15]))
        for (c, name) in fields
            @printf(io, "%s %s extrema: %.8g %.8g\n", label, name, extrema(run.data[:, :, c])...)
        end
    end
end

set_theme!(fontsize=20, linewidth=3)
runs = ((reference, "kin1d P3", :black, :solid),
        (breeze, "Breeze · fixed aerosol", :dodgerblue, :solid),
        (upwind, "Breeze · first-order advection", :orangered, :dash),
        (reservoir, "Breeze · aerosol reservoir", :purple, :dot))
figure = Figure(size=(1400, 1050), figure_padding=25)
column_axes = Axis[]
Label(figure[0, 1:2], "P3 kinematic column comparison"; fontsize=28)
for (index, (c, name)) in enumerate(fields[2:end])
    row, col = (index - 1) ÷ 2 + 1, (index - 1) % 2 + 1
    ax = Axis(figure[row, col]; title=uppercasefirst(name), xlabel="Time (s)",
              ylabel="Common-density column amount (kg/m²)")
    push!(column_axes, ax)
    for (run, label, color, linestyle) in runs
        lines!(ax, run.times, column(run, c); label, color, linestyle)
    end
end
Legend(figure[3, 1:2], first(column_axes); orientation=:horizontal, nbanks=2, tellheight=true)
save(joinpath(output_directory, "column_comparison.png"), figure; px_per_unit=2)
save(joinpath(output_directory, "column_comparison.svg"), figure)

profiles = Figure(size=(1600, 900), figure_padding=25)
profile_axes = Axis[]
for (index, (c, name)) in enumerate(((3, "Temperature (K)"), (7, "Cloud (kg/kg)"),
                                    (8, "Rain (kg/kg)"), (9, "Frozen ice (kg/kg)")))
    ax = Axis(profiles[1, index]; xlabel=name, ylabel="Height (km)", title="t = 3600 s")
    push!(profile_axes, ax)
    for (run, label, color, linestyle) in runs
        lines!(ax, run.data[:, end, c], run.heights / 1000; label, color, linestyle)
    end
end
Legend(profiles[2, 1:4], first(profile_axes); orientation=:horizontal, nbanks=2)
save(joinpath(output_directory, "final_profiles.png"), profiles; px_per_unit=2)
save(joinpath(output_directory, "final_profiles.svg"), profiles)

sensitivity = Figure(size=(1400, 1050), figure_padding=25)
sensitivity_axes = Axis[]
Label(sensitivity[0, 1:2], "Sensitivity to saturation formula and relaxation times"; fontsize=26)
for (index, (c, name)) in enumerate(fields[2:end])
    row, col = (index - 1) ÷ 2 + 1, (index - 1) % 2 + 1
    ax = Axis(sensitivity[row, col]; title=uppercasefirst(name), xlabel="Time (s)",
              ylabel="Common-density column amount (kg/m²)")
    push!(sensitivity_axes, ax)
    for (run, label, color, linestyle) in ((reference, "kin1d P3", :black, :solid),
            (upwind, "Breeze · first-order advection", :orangered, :dash),
            (reference_options, "Breeze · Flatau + 1 s relaxation", :dodgerblue, :solid))
        lines!(ax, run.times, column(run, c); label, color, linestyle)
    end
end
Legend(sensitivity[3, 1:2], first(sensitivity_axes); orientation=:horizontal, nbanks=2)
save(joinpath(output_directory, "options_sensitivity.png"), sensitivity; px_per_unit=2)
save(joinpath(output_directory, "options_sensitivity.svg"), sensitivity)
print(read(joinpath(output_directory, "metrics.txt"), String))
