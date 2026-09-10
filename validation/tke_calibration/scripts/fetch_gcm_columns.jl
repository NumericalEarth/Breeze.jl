# Monthly-mean CNRM-CM6-1 AMIP columns (temperature, specific humidity, pressure) at the cfSites of the Shen
# et al. (2022) library, 2004–2008, from the CMIP6 CFsubhr output over OPeNDAP (CEDA). These are the GCM
# columns the LES was forced with; the column model needs them above the LES top for interactive radiation.
#
#     SSL_CERT_FILE=/etc/ssl/cert.pem julia --project scripts/fetch_gcm_columns.jl ta        # temperature + ps over OPeNDAP → data/gcm_ta.jld2
#     julia --project scripts/fetch_gcm_columns.jl hus                                          # humidity from data/hus_*.nc (downloaded) → data/gcm_hus.jld2
#     julia --project scripts/fetch_gcm_columns.jl combine [data/gcm_columns.nc]                # both → NetCDF with pressure and heights
#     julia --project scripts/solar_parameters.jl                                               # then add the fixed-sun parameters
#
# The result, data/gcm_columns_CNRM-CM6-1_amip.nc (0.3 MB), is committed; these steps only regenerate it.
#
# Site index in the CFsubhr files equals the cfSite number (checked: 17 → 35°N 235°E, 2 → 20°S 287.5°E).
# Temperature streams from CEDA's OPeNDAP server; the humidity files are not served there (and DKRZ's OPeNDAP is
# down), so they are downloaded whole from DKRZ's file server into data/.
using NCDatasets, Dates, Statistics, Printf, JLD2

step = length(ARGS) ≥ 1 ? ARGS[1] : "combine"
data = joinpath(@__DIR__, "..", "data")
output = length(ARGS) ≥ 2 ? ARGS[2] : joinpath(data, "gcm_columns_CNRM-CM6-1_amip.nc")
base = "https://esgf.ceda.ac.uk/thredds/dodsC/esg_cmip6/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/amip/r1i1p1f2/CFsubhr"
files = Dict("ta"  => ["$base/ta/gn/v20181203/ta_CFsubhr_CNRM-CM6-1_amip_r1i1p1f2_gn_20040101003000-20090101000000.nc"],
             "hus" => [joinpath(data, "hus_CFsubhr_CNRM-CM6-1_amip_r1i1p1f2_gn_20030101003000-20070101000000.nc"),
                       joinpath(data, "hus_CFsubhr_CNRM-CM6-1_amip_r1i1p1f2_gn_20070101003000-20110101000000.nc")])
sites = [2:15; 17:23]
months = [1, 4, 7, 10]
years = 2004:2008

# Time indices of every (year, month) in a file's time axis
function month_ranges(times)
    ranges = Dict{Tuple{Int, Int}, UnitRange{Int}}()
    for y in years, m in months
        idx = findall(t -> year(t) == y && month(t) == m, times)
        isempty(idx) || (ranges[(y, m)] = first(idx):last(idx))
    end
    return ranges
end

Nlev = 91
ta = fill(NaN, length(sites), length(months), Nlev)
hus = fill(NaN, length(sites), length(months), Nlev)
ps = fill(NaN, length(sites), length(months))
lat = zeros(length(sites)); lon = zeros(length(sites)); ap = zeros(Nlev); b = zeros(Nlev)
counts = Dict("ta" => zeros(Int, length(sites), length(months)), "hus" => zeros(Int, length(sites), length(months)))

if step in ("ta", "hus")
for url in files[step]
    name = step
    @info "opening $(basename(url))"
    ds = NCDataset(url)
    times = ds["time"][:]
    ranges = month_ranges(times)
    if name == "ta"
        lat .= ds["latitude"][sites]; lon .= ds["longitude"][sites]; ap .= ds["ap"][:]; b .= ds["b"][:]
    end
    for (si, s) in enumerate(sites), (mi, m) in enumerate(months)
        for y in years
            haskey(ranges, (y, m)) || continue
            r = ranges[(y, m)]
            v = ds[name][s, :, r]                       # (lev, time) for this site and month
            # Accumulate sums weighted by the number of samples: a month can be split across two files
            target = name == "ta" ? ta : hus
            n = counts[name][si, mi]
            target[si, mi, :] .= (n == 0 ? 0 : target[si, mi, :]) .+ vec(sum(v, dims = 2))
            if name == "ta"
                p = ds["ps"][s, r]
                ps[si, mi] = (n == 0 ? 0 : ps[si, mi]) + sum(p)
            end
            counts[name][si, mi] = n + length(r)
        end
        @printf "  %s site %2d month %02d: %d half-hourly samples\n" name s m counts[name][si, mi]
        flush(stdout)
    end
    close(ds)
end
# Means from the sums
n = counts[step]
if step == "ta"
    ta .= ta ./ n; ps .= ps ./ n
else
    hus .= hus ./ n
end
jldsave(joinpath(data, "gcm_$step.jld2"); ta, hus, ps, lat, lon, ap, b, counts = n, sites, months)
println("wrote data/gcm_$step.jld2")
exit()
end

# combine
t = load(joinpath(data, "gcm_ta.jld2")); h = load(joinpath(data, "gcm_hus.jld2"))
ta, ps, lat, lon, ap, b = t["ta"], t["ps"], t["lat"], t["lon"], t["ap"], t["b"]
hus = h["hus"]
expected = 5 * 48 * 30   # about five months of half-hourly samples (30–31 days)
all(t["counts"] .≥ 0.95expected) && all(h["counts"] .≥ 0.95expected) || @warn "some (site, month) has fewer samples than five months" extrema(t["counts"]) extrema(h["counts"])

# Hybrid-sigma full-level pressure, and heights by hydrostatic integration from the surface with virtual temperature
pfull = [ap[k] + b[k] * ps[si, mi] for si in eachindex(sites), mi in eachindex(months), k in 1:Nlev]
Rᵈ = 287.04; g = 9.81
zg = similar(pfull)
for si in eachindex(sites), mi in eachindex(months)
    p = pfull[si, mi, :]; T = ta[si, mi, :]; q = hus[si, mi, :]
    order = sortperm(p; rev = true)                    # from the surface upward
    z = 0.0; p_below = ps[si, mi]
    for k in order
        Tᵛ = T[k] * (1 + 0.608 * q[k])
        z += Rᵈ * Tᵛ / g * log(p_below / p[k]); zg[si, mi, k] = z; p_below = p[k]
    end
end

rm(output; force = true)
NCDataset(output, "c") do out
    defDim(out, "site", length(sites)); defDim(out, "month", length(months)); defDim(out, "lev", Nlev)
    defVar(out, "site", sites, ("site",)); defVar(out, "month", months, ("month",))
    defVar(out, "latitude", lat, ("site",)); defVar(out, "longitude", lon, ("site",))
    defVar(out, "ta", ta, ("site", "month", "lev"); attrib = Dict("units" => "K", "long_name" => "2004–2008 monthly-mean temperature"))
    defVar(out, "hus", hus, ("site", "month", "lev"); attrib = Dict("units" => "kg/kg", "long_name" => "2004–2008 monthly-mean specific humidity"))
    defVar(out, "pfull", pfull, ("site", "month", "lev"); attrib = Dict("units" => "Pa", "long_name" => "ap + b ps with the monthly-mean surface pressure"))
    defVar(out, "zg", zg, ("site", "month", "lev"); attrib = Dict("units" => "m", "long_name" => "hydrostatic height above the surface from the monthly means"))
    defVar(out, "ps", ps, ("site", "month"); attrib = Dict("units" => "Pa"))
    out.attrib["source"] = "CMIP6 CNRM-CM6-1 amip r1i1p1f2 CFsubhr (ta, hus, ps) via https://esgf.ceda.ac.uk OPeNDAP; monthly means 2004–2008"
end
println("wrote $output")
