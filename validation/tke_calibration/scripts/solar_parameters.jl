# The fixed-sun parameters of the LES protocol for each cfSite and month: the 2004–2008 monthly-mean TOA insolation
# and the insolation-weighted cosine of the solar zenith angle (PyCLES prescribes both from the GCM and sets a fixed
# sun with cos θ_z = coszen and TOA flux = insolation, i.e. an adjusted solar constant insolation / coszen).
# Computed astronomically from the site coordinates with Breeze.CelestialMechanics; added to the GCM-column file.
#
#     julia --project scripts/solar_parameters.jl [data/gcm_columns_CNRM-CM6-1_amip.nc]
using Breeze.CelestialMechanics: cos_solar_zenith_angle
using NCDatasets, Dates, Statistics, Printf

path = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "gcm_columns_CNRM-CM6-1_amip.nc")
S₀ = 1361.0
ds = NCDataset(path, "a")
sites, months = ds["site"][:], ds["month"][:]
lat, lon = ds["latitude"][:], ds["longitude"][:]
insolation = zeros(length(sites), length(months)); coszen = zeros(length(sites), length(months))
for (si, s) in enumerate(sites), (mi, m) in enumerate(months)
    μs = Float64[]
    for y in 2004:2008
        t = DateTime(y, m, 1)
        while month(t) == m
            push!(μs, max(0.0, cos_solar_zenith_angle(t, lon[si], lat[si])))
            t += Minute(30)
        end
    end
    insolation[si, mi] = S₀ * mean(μs)
    coszen[si, mi] = sum(μs .^ 2) / sum(μs)             # insolation-weighted mean cos θ_z
end
haskey(ds, "insolation") || defVar(ds, "insolation", insolation, ("site", "month"); attrib = Dict("units" => "W m-2", "long_name" => "monthly-mean TOA insolation, S₀ = 1361 W m⁻², astronomical"))
haskey(ds, "coszen") || defVar(ds, "coszen", coszen, ("site", "month"); attrib = Dict("long_name" => "insolation-weighted cosine of the solar zenith angle"))
close(ds)
@printf "site 17 (California): Jan insolation %.0f W/m², coszen %.3f; Jul %.0f, %.3f\n" insolation[findfirst(==(17), sites), 1] coszen[findfirst(==(17), sites), 1] insolation[findfirst(==(17), sites), 3] coszen[findfirst(==(17), sites), 3]
println("added insolation and coszen to $path")
