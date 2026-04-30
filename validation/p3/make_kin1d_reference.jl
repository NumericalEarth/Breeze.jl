using Dates
using NCDatasets

p3_repo = get(ENV, "P3_REPO", "")
isempty(p3_repo) && error("Set P3_REPO to the path of your P3-microphysics clone.")

input_path = get(ENV, "P3_KIN1D_OUT", joinpath(p3_repo, "kin1d", "src", "out_p3.dat"))
output_path = get(ENV, "P3_KIN1D_NETCDF", joinpath(@__DIR__, "kin1d_reference.nc"))

nk = parse(Int, get(ENV, "P3_NK", "41"))
outfreq_min = parse(Float64, get(ENV, "P3_OUTFREQ_MIN", "1"))

lines = readlines(input_path)
rows = [parse.(Float64, split(strip(line))) for line in lines if !isempty(strip(line))]
nrows = length(rows)
ncols = length(rows[1])

if nrows % nk != 0
    error("Row count $nrows is not divisible by nk=$nk. Check P3_NK or input file.")
end

nt = div(nrows, nk)
data = Array{Float64}(undef, nt, nk, ncols)
for idx in 1:nrows
    t = div(idx - 1, nk) + 1
    k = mod(idx - 1, nk) + 1
    data[t, k, :] = rows[idx]
end

z = data[1, :, 1]
time = collect(1:nt) .* outfreq_min .* 60.0

isfile(output_path) && rm(output_path)
ds = NCDataset(output_path, "c")

defDim(ds, "time", nt)
defDim(ds, "z", nk)

vtime = defVar(ds, "time", Float64, ("time",))
vtime.attrib["units"] = "s"
vtime.attrib["long_name"] = "time"
vtime[:] = time

vz = defVar(ds, "z", Float64, ("z",))
vz.attrib["units"] = "m"
vz.attrib["long_name"] = "height"
vz[:] = z

function put_2d(name, col, units, long_name)
    v = defVar(ds, name, Float64, ("time", "z"))
    v.attrib["units"] = units
    v.attrib["long_name"] = long_name
    v[:, :] = data[:, :, col]
    return nothing
end

function put_1d(name, col, units, long_name)
    v = defVar(ds, name, Float64, ("time",))
    v.attrib["units"] = units
    v.attrib["long_name"] = long_name
    v[:] = data[:, 1, col]
    return nothing
end

# Column mapping for nCat=1 from kin1d/src/cld1d.f90
put_2d("w", 2, "m s-1", "vertical_velocity")
put_1d("prt_liq", 3, "mm h-1", "surface_precipitation_rate_liquid")
put_1d("prt_sol", 4, "mm h-1", "surface_precipitation_rate_solid")
put_2d("reflectivity", 5, "dBZ", "radar_reflectivity")
put_2d("temperature", 6, "C", "temperature_celsius")
put_2d("q_cloud", 7, "kg kg-1", "cloud_liquid_mixing_ratio")
put_2d("q_rain", 8, "kg kg-1", "rain_mixing_ratio")
put_2d("n_cloud", 9, "kg-1", "cloud_droplet_number_mixing_ratio")
put_2d("n_rain", 10, "kg-1", "rain_number_mixing_ratio")
put_2d("q_ice", 11, "kg kg-1", "total_ice_mixing_ratio")
put_2d("n_ice", 12, "kg-1", "ice_number_mixing_ratio")
put_2d("rime_fraction", 13, "1", "rime_mass_fraction")
put_2d("liquid_fraction", 14, "1", "liquid_mass_fraction_on_ice")
put_2d("drm", 15, "m", "rain_mean_volume_diameter")
put_2d("q_ice_cat1", 16, "kg kg-1", "ice_mixing_ratio_category1")
put_2d("q_rime_cat1", 17, "kg kg-1", "rime_mixing_ratio_category1")
put_2d("q_liquid_on_ice_cat1", 18, "kg kg-1", "liquid_on_ice_mixing_ratio_category1")
put_2d("n_ice_cat1", 19, "kg-1", "ice_number_mixing_ratio_category1")
put_2d("b_rime_cat1", 20, "m3 m-3", "rime_volume_density_category1")
put_2d("z_ice_cat1", 21, "m6 m-3", "ice_reflectivity_moment_category1")
put_2d("rho_ice_cat1", 22, "kg m-3", "ice_bulk_density_category1")
put_2d("d_ice_cat1", 23, "m", "ice_mean_diameter_category1")

commit = get(ENV, "P3_COMMIT", "")
if isempty(commit)
    try
        commit = readchomp(`git -C $p3_repo rev-parse HEAD`)
    catch
        commit = "unknown"
    end
end

ds.attrib["source"] = input_path
ds.attrib["p3_repo_commit"] = commit
ds.attrib["p3_version_param"] = get(ENV, "P3_VERSION_PARAM", "v5.3.14")
ds.attrib["p3_init_version"] = get(ENV, "P3_INIT_VERSION", "v5.5.0")
ds.attrib["nCat"] = get(ENV, "P3_NCAT", "1")
ds.attrib["triple_moment_ice"] = get(ENV, "P3_TRPL_MOM_ICE", "true")
ds.attrib["liquid_fraction"] = get(ENV, "P3_LIQ_FRAC", "true")
ds.attrib["dt_seconds"] = get(ENV, "P3_DT_SECONDS", "10")
ds.attrib["outfreq_minutes"] = string(outfreq_min)
ds.attrib["total_minutes"] = get(ENV, "P3_TOTAL_MINUTES", "90")
ds.attrib["sounding"] = get(ENV, "P3_SOUNDING", "snd_input.KOUN_00z1june2008.data")
ds.attrib["driver"] = get(ENV, "P3_DRIVER", "kin1d cld1d.f90")
ds.attrib["created"] = string(now())

close(ds)

println(output_path)
