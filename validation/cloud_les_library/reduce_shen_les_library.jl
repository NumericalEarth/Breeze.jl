# # Reduce the Shen et al. (2022) cloud LES library for single-column evaluation
#
# The library of GCM-forced large-eddy simulations of Shen et al. (2022),
# https://doi.org/10.22002/D1.20052 (CC0), holds one PyCLES `Stats` file per member —
# `Stats.cfsite<site>_<gcm>_<experiment>_2004-2008.<month>.nc` — of 200–330 MB each: 22 sites
# along the GPCI Pacific transect, three GCMs, two climates (`amip`, `amip4K`) and four months.
# Each file carries 10-minute horizontal means, second moments, fluxes and the TKE budget on
# 200 levels (Δz = 20 m, to 4 km) over 3.7 days, together with everything that forced the LES:
# the large-scale subsidence and horizontal advective tendencies (time-invariant), the radiative
# heating (diurnal), the nudging tendencies, and the surface fluxes.
#
# This script downloads the members of one GCM and reduces each to the ~0.25 MB a single-column
# model needs: the forcing, the initial and reference profiles, the surface time series, hourly
# radiative heating, and time-mean target profiles over the final two days. The reduced files
# are what `examples/single_column_tke_boundary_layer.jl` reads, packaged as the lazy artifact
# `shen_et_al_2022_les_profiles` (see `build_artifact.jl`). Run from this directory with
#
#     julia --project reduce_shen_les_library.jl --gcm CNRM-CM6-1 --output reduced
#
# Files are downloaded one at a time to `--download` (default: a temporary directory) and deleted
# after reduction unless `--keep` is given.

using NCDatasets
using Downloads
using SHA
using Statistics
using Printf

const RECORD = "j8mw7-fm491"
const DOI = "10.22002/D1.20052"
const API = "https://data.caltech.edu/api/records/$RECORD/files"

#####
##### Command line
#####

function parse_arguments(args)
    options = Dict{String, Any}("gcm" => "CNRM-CM6-1",
                                "output" => "reduced",
                                "download" => mktempdir(),
                                "keep" => false,
                                "sites" => 2:23,
                                "experiments" => ("amip", "amip4K"),
                                "months" => ("01", "04", "07", "10"))
    i = 1
    while i ≤ length(args)
        arg = args[i]
        if arg == "--keep"
            options["keep"] = true
        elseif arg in ("--gcm", "--output", "--download")
            options[arg[3:end]] = args[i+1]
            i += 1
        elseif arg == "--sites"
            options["sites"] = parse.(Int, split(args[i+1], ","))
            i += 1
        elseif arg == "--experiments"
            options["experiments"] = Tuple(split(args[i+1], ","))
            i += 1
        elseif arg == "--months"
            options["months"] = Tuple(split(args[i+1], ","))
            i += 1
        else
            error("Unknown argument $arg")
        end
        i += 1
    end
    return options
end

#####
##### Download
#####

source_name(site, gcm, experiment, month) = "Stats.cfsite$(site)_$(gcm)_$(experiment)_2004-2008.$(month).nc"
reduced_name(site, gcm, experiment, month) = "cfsite$(lpad(site, 2, '0'))_$(gcm)_$(experiment)_$(month).nc"

function download_member(name, directory)
    path = joinpath(directory, name)
    isfile(path) && return path
    url = "$API/$name/content"
    for attempt in 1:3
        try
            Downloads.download(url, path)
            return path
        catch err
            # A member that does not exist in the library returns 404 on the first attempt
            (err isa Downloads.RequestError && err.response.status == 404) && return nothing
            @warn "Download of $name failed (attempt $attempt)" err
            rm(path; force=true)
            attempt == 3 && rethrow()
        end
    end
end

#####
##### Reduction
#####

const HOURLY = 6            # the LES writes every 600 s
const TARGET_WINDOW = 2 * 86400 # the final two days

# Time-mean profiles over the final `TARGET_WINDOW` seconds
const TARGET_PROFILES = ("thetali_mean", "qt_mean", "qv_mean", "ql_mean", "qr_mean", "temperature_mean",
                         "theta_rho_mean", "u_mean", "v_mean", "cloud_fraction", "rh_mean",
                         "tke_mean", "tke_prod_S", "tke_prod_B", "tke_prod_D", "tke_prod_T", "tke_prod_P", "tke_prod_A",
                         "qt_flux_z", "qt_sgs_flux_z", "s_flux_z", "s_sgs_flux_z",
                         "u_sgs_flux_z", "v_sgs_flux_z", "w_sgs_flux_z",
                         "w_mean2", "qt_mean2", "thetali_mean2", "buoyancy_frequency_mean",
                         "diffusivity_mean", "viscosity_mean")

# Time-invariant large-scale forcing profiles (asserted below)
const FORCING_PROFILES = ("ls_subsidence", "dtdt_hadv", "dqtdt_hadv", "dtdt_fluc", "dqtdt_fluc")

# Hourly profiles: the radiative heating that forces a single-column model, and the cloud fraction
const HOURLY_PROFILES = ("dtdt_rad", "cloud_fraction")

# Hourly surface and integrated time series
const TIMESERIES = ("shf_surface_mean", "lhf_surface_mean", "surface_temperature", "friction_velocity_mean",
                    "uw_surface_mean", "vw_surface_mean", "buoyancy_flux_surface_mean", "obukhov_length_mean",
                    "lwp", "cloud_fraction", "cloud_base", "cloud_top")

# Profiles at the initial time
const INITIAL_PROFILES = ("thetali_mean", "qt_mean", "u_mean", "v_mean", "temperature_mean", "ql_mean")

# Whole-run time-mean profiles: the natural nudging targets for a single-column model, and the
# nudging tendencies the LES applied
const NUDGING_TARGETS = ("u_mean", "v_mean", "thetali_mean", "qt_mean")
const NUDGING_TENDENCIES = ("dudt_nudge", "dvdt_nudge", "dtdt_nudge", "dqtdt_nudge")

const REFERENCE_PROFILES = ("p0", "rho0", "temperature0", "qv0")

timemean(x, sel) = vec(mean(view(x, :, sel), dims=2))

function reduce_member!(output_path, source_path, source_name; site, gcm, experiment, month)
    ds = NCDataset(source_path)
    P = ds.group["profiles"]
    S = ds.group["timeseries"]
    R = ds.group["reference"]

    z = Float32.(P["z_half"][:])       # cell centers, where the profiles live
    t = P["t"][:]
    Nt = length(t)
    hourly = 1:HOURLY:Nt
    target = findall(τ -> τ ≥ t[end] - TARGET_WINDOW, t)

    NCDataset(output_path, "c") do out
        out.attrib["title"] = "Reduced Shen et al. (2022) LES: $source_name"
        out.attrib["source"] = "https://data.caltech.edu/records/$RECORD/files/$source_name"
        out.attrib["source_sha256"] = bytes2hex(open(sha256, source_path))
        out.attrib["source_size_bytes"] = filesize(source_path)
        out.attrib["doi"] = DOI
        out.attrib["license"] = "CC0-1.0"
        out.attrib["reference"] = "Shen, Sridhar, Tan, Jaruga & Schneider (2022), J. Adv. Model. Earth Syst. 14, e2021MS002631"
        out.attrib["cfsite"] = site
        out.attrib["gcm"] = gcm
        out.attrib["experiment"] = experiment
        out.attrib["month"] = month
        out.attrib["target_window_start"] = t[end] - TARGET_WINDOW
        out.attrib["target_window_end"] = t[end]
        out.attrib["description"] = "Profiles are horizontal means at cell centers z. *_mean profiles are " *
                                    "time means over [target_window_start, target_window_end]; *_initial are the " *
                                    "initial profiles; *_nudge are whole-run time means; subsidence and the " *
                                    "advective and fluctuation tendencies are time-invariant in the LES."

        defDim(out, "z", length(z))
        defDim(out, "time", length(hourly))
        defVar(out, "z", z, ("z",); attrib = Dict("units" => "m", "long_name" => "height of cell centers"))
        defVar(out, "time", Float32.(t[hourly]), ("time",); attrib = Dict("units" => "s"))

        profile(name, data; attrib = Dict{String, String}()) =
            defVar(out, name, Float32.(data), ("z",); deflatelevel = 5, attrib)
        series(name, data; attrib = Dict{String, String}()) =
            defVar(out, name, Float32.(data), ("time",); deflatelevel = 5, attrib)

        for name in REFERENCE_PROFILES
            profile(name, R[name][:]; attrib = Dict("units" => string(get(R[name].attrib, "units", ""))))
        end

        for name in FORCING_PROFILES
            x = P[name][:, :]
            steady = maximum(std(x, dims=2)) ≤ 1e-8 * max(maximum(abs, x), eps(Float32))
            steady || @warn "$name is not time-invariant in $source_name; storing its time mean"
            profile(name, timemean(x, 1:Nt))
        end

        for name in INITIAL_PROFILES
            profile(name * "_initial", P[name][:, 1])
        end

        for name in NUDGING_TARGETS
            profile(name * "_nudge", timemean(P[name][:, :], 1:Nt))
        end

        for name in NUDGING_TENDENCIES
            profile(name, timemean(P[name][:, :], 1:Nt))
        end

        for name in TARGET_PROFILES
            haskey(P, name) || continue
            label = endswith(name, "_mean") ? name : name * "_mean"
            profile(label, timemean(P[name][:, :], target))
        end

        for name in HOURLY_PROFILES
            defVar(out, name * "_hourly", Float32.(P[name][:, hourly]), ("z", "time"); deflatelevel = 5)
        end

        for name in TIMESERIES
            series(name, S[name][hourly])
        end
    end

    close(ds)
    return output_path
end

#####
##### Main
#####

function main(args)
    options = parse_arguments(args)
    gcm = options["gcm"]
    output = options["output"]
    mkpath(output)
    mkpath(options["download"])

    members = [(site, experiment, month) for site in options["sites"], experiment in options["experiments"], month in options["months"]]
    @info "Reducing up to $(length(members)) members of $gcm into $output"

    n_done = 0
    for (site, experiment, month) in members
        reduced = joinpath(output, reduced_name(site, gcm, experiment, month))
        if isfile(reduced)
            n_done += 1
            continue
        end
        name = source_name(site, gcm, experiment, month)
        path = download_member(name, options["download"])
        if path === nothing
            @info "  $name is not in the library; skipping"
            continue
        end
        try
            reduce_member!(reduced, path, name; site, gcm, experiment, month)
            n_done += 1
            @info @sprintf("  %-55s → %s (%.2f MB)", name, basename(reduced), filesize(reduced) / 1e6)
        catch err
            @error "Reduction of $name failed" err
            rm(reduced; force=true)
        finally
            options["keep"] || rm(path; force=true)
        end
    end
    @info "Reduced $n_done members into $output"
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
