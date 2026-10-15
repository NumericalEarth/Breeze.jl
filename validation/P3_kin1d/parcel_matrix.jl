# A factorial experiment using the same production ParcelModel setup as the
# two ascent benchmarks. No new advancement scheme or comparison adapter.
include("parcel_compare.jl")

function parcel_matrix_cases()
    constants = parcel_thermodynamic_constants("reference_constants")
    rows = Vector{Float64}[]
    for T in (288., 300.), humidity in (0.6, 0.8, 1.0), speed in (0.5, 1., 2.)
        e = humidity * TD.saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
        vapor_ratio = 0.622e / (101325-e)
        total_water = vapor_ratio / (1+vapor_ratio)
        push!(rows, [length(rows)+1, T, humidity, speed, total_water])
    end
    return permutedims(hcat(rows...))
end

root = normpath(joinpath(@__DIR__, "../../validation_output/p3_matrix"))
mkpath(root)
cases = parcel_matrix_cases()
case_path = joinpath(root, "cases.dat")
if isfile(case_path)
    @assert readdlm(case_path) ≈ cases "Existing matrix has a different configuration."
else
    writedlm(case_path, cases)
end
write(joinpath(root, "case_columns.txt"), "case_id initial_temperature_K initial_relative_humidity ascent_m_s total_water_kg_kg\n")

if get(ARGS, 1, "") != "prepare"
    dt = parse(Float64, get(ARGS, 1, "0.01"))
    @assert isfinite(dt) && dt > 0
    identifiers = length(ARGS) > 1 ? parse.(Int, split(ARGS[2], ',')) : collect(1:size(cases,1))
    directory = joinpath(root, "breeze_dt$(dt)")
    mkpath(directory)
    write(joinpath(directory, "configuration.txt"),
          "Julia $VERSION; production ParcelModel SSP RK3; dt=$dt s; fixed tau=0.025 s.\n" *
          "Reference constants via public constructors; native mixture thermodynamics and prognostic supersaturation.\n" *
          "5400 s; p0=101325 Pa; common dry 300 K reference pressure profile.\n" *
          "Conserved theta_li and total water; no entrainment/fallout; fixed aerosol 3e8 per kg dry air.\n" *
          "Initial conditions: ../cases.dat. Remove cached output before rerunning changed physics or setup.\n")
    for identifier in identifiers
        _, initial_temperature, humidity, speed, total_water = cases[identifier,:]
        path = joinpath(directory, "case_$(lpad(identifier,3,'0')).dat")
        if isfile(path)
            println("Keeping completed case $identifier: $path")
            continue
        end
        elapsed = @elapsed data = run_production_parcel(speed; dt, timescale=0.025,
            configuration="reference_constants", initial_temperature, total_water, case_identifier=identifier)
        # Retain failures as well, rather than silently excluding them from the matrix.
        writedlm(path, data)
        finite = all(isfinite, data)
        @printf("Case %2d T0=%.0f RH=%.0f%% w=%.1f: %.1f s elapsed, finite=%s, final T=%.5f K\n",
                identifier, initial_temperature, 100humidity, speed, elapsed, finite, data[end,4])
        flush(stdout)
    end
end
