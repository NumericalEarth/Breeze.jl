# Resolution study: NCAR (pseudo-spectral) and Breeze (finite volume) in the same 32³ NCAR box
# (L = 41.4 mm, ν = 1.6e-5, ε = 0.05), all diagnostics computed with the same functions (NCARDNS.jl).
#
#   julia -t 8 --project=validation/ncar_dns validation/ncar_dns/compare_resolution.jl [decay|forced|all]
#
# Inputs: NCAR runs in ~/ncar-dns/runs/R (fixed-box builds, BOXN=32) plus the original 32³ runs
# (~/ncar-dns/runs/L1a/n32_seed1_decay, ~/ncar-dns/runs/L1b/n32_seed*), and the Breeze outputs of
# resolution_runs.jl in validation/ncar_dns/output/resolution (decay) and output/resolution/forced.
# Outputs (CSV + summary text) in validation/ncar_dns/output/resolution.

include(joinpath(@__DIR__, "NCARDNS.jl"))
using .NCARDNS
using DelimitedFiles
using Printf
using Statistics
using FFTW

FFTW.set_num_threads(max(1, Threads.nthreads()))

const RUNS = expanduser("~/ncar-dns/runs")
const OUT = joinpath(@__DIR__, "output", "resolution")
const BOX_N = 32
const L = ncar_parameters(32).domain_length
const η₀ = ncar_parameters(32).kolmogorov_length   # η at ε = 0.05 (forced, stationary)

#####
##### Readers / per-snapshot diagnostics
#####

function read_table(path)
    data, header = readdlm(path, ',', header=true)
    return Dict(String(h) => Float64.(data[:, i]) for (i, h) in enumerate(vec(header)))
end

"All diagnostics of one NCAR Zk dump (common code path)."
function zk_diagnostics(path; box_N=BOX_N)
    z = read_zk(path; box_N)
    F = fourier_velocity_from_zk(z)
    s = turbulence_statistics(F; skewness=false)
    g = gradient_moments(F)
    e = shell_spectrum(F)
    return (; time=z.time, K=s.kinetic_energy, ε=s.dissipation, Reλ=s.taylor_reynolds, u′=s.rms_velocity,
              S=g.skewness, F=g.flatness, FT=g.transverse_flatness, ratio=g.transverse_variance / g.longitudinal_variance,
              k=e.k, E=e.energy, Ef=e.energy[1], N=z.parameters.N, field=F)
end

"Zk dumps of an NCAR case directory in time order: Zk.in.ncf (t₀) then Zk1…Zkn."
function zk_files(dir)
    files = [joinpath(dir, "Zk.in.ncf")]
    n = 1
    while isfile(joinpath(dir, "Zk$(n).out.ncf"))
        push!(files, joinpath(dir, "Zk$(n).out.ncf")); n += 1
    end
    return filter(isfile, files)
end

"Diagnostics of all dumps of an NCAR run, cached in a CSV (scalars) + spectra CSV; fields kept if `keep`."
function ncar_series(name, dir; keep=false, box_N=BOX_N, skip_input=false)
    files = zk_files(dir)
    skip_input && (files = files[2:end])
    snaps = []
    for f in files
        d = try
            zk_diagnostics(f; box_N)
        catch err
            @warn "skipping unreadable dump (run still in progress?)" f err
            nothing
        end
        d === nothing && continue
        !isempty(snaps) && d.time ≤ snaps[end].time && continue   # pre-created, not yet written dumps
        push!(snaps, keep ? d : merge(d, (; field=nothing)))
    end
    t₀ = snaps[1].time
    write_csv(joinpath(OUT, "ncar_$(name)_timeseries.csv"),
              (time=[s.time for s in snaps], kinetic_energy=[s.K for s in snaps], dissipation=[s.ε for s in snaps],
               taylor_reynolds=[s.Reλ for s in snaps], skewness=[s.S for s in snaps], flatness=[s.F for s in snaps],
               transverse_flatness=[s.FT for s in snaps], gradient_variance_ratio=[s.ratio for s in snaps],
               forced_band_energy=[s.Ef for s in snaps]))
    write_csv(joinpath(OUT, "ncar_$(name)_spectra.csv"),
              (time=reduce(vcat, [fill(s.time, length(s.k)) for s in snaps]), k=reduce(vcat, [s.k for s in snaps]),
               energy=reduce(vcat, [s.E for s in snaps])))
    return snaps
end

"Breeze run as a vector of snapshots from the CSVs of resolution_runs.jl (only rows with gradient moments)."
function breeze_series(name; dir=OUT, all_rows=false)
    ts = read_table(joinpath(dir, "breeze_$(name)_timeseries.csv"))
    sp = read_table(joinpath(dir, "breeze_$(name)_spectra.csv"))
    snaps = []
    for i in eachindex(ts["time"])
        t = ts["time"][i]
        (all_rows || !isnan(ts["skewness"][i])) || continue
        sel = sp["time"] .== t
        push!(snaps, (; time=t, K=ts["kinetic_energy"][i], ε=ts["dissipation"][i], Reλ=ts["taylor_reynolds"][i],
                        u′=ts["rms_velocity"][i], S=ts["skewness"][i], F=ts["flatness"][i], FT=ts["transverse_flatness"][i],
                        ratio=ts["gradient_variance_ratio"][i], S_fd=ts["skewness_fd"][i], F_fd=ts["flatness_fd"][i],
                        FT_fd=ts["transverse_flatness_fd"][i], ε_fd=ts["dissipation_fd"][i],
                        k=Int.(sp["k"][sel]), E=sp["energy"][sel], Ef=ts["forced_band_energy"][i],
                        P=ts["injection_power"][i]))
    end
    return snaps
end

#####
##### Statistics helpers
#####

"Integrated autocorrelation time (in samples) with Geyer's initial positive sequence."
function autocorrelation_samples(x)
    n = length(x)
    n < 4 && return 1.0
    y = x .- mean(x)
    c0 = sum(abs2, y) / n
    c0 == 0 && return 1.0
    ρ(k) = sum(y[1:n-k] .* y[1+k:n]) / n / c0
    τ = 1.0
    k = 1
    while k + 1 < n ÷ 2
        Γ = ρ(k) + ρ(k + 1)
        Γ ≤ 0 && break
        τ += 2Γ
        k += 2
    end
    return max(τ, 1.0)
end

"Mean and standard error accounting for autocorrelation (samples equally spaced in time)."
function mean_se(x)
    x = filter(!isnan, x)
    n = length(x)
    τ = autocorrelation_samples(x)
    return mean(x), std(x) * sqrt(τ / n), std(x), τ
end

"Mean and standard error of an ensemble of runs: pooled mean of run means, SE from the per-run SEs."
function ensemble_mean_se(series)
    ms = [mean_se(x) for x in series]
    m = mean(first.(ms))
    se = sqrt(sum(abs2, [v[2] for v in ms])) / length(ms)
    spread = length(ms) > 1 ? std(first.(ms)) / sqrt(length(ms)) : NaN
    return m, max(se, isnan(spread) ? 0.0 : spread), [v[1] for v in ms]
end

pct(x, ref) = 100 * (x / ref - 1)

"Smallest M at which |error| falls (and stays) below `threshold` %, log-log interpolated between neighbours."
function required_resolution(Ms, errors, threshold)
    order = sortperm(Ms)
    Ms = Ms[order]; e = abs.(errors[order])
    i = findfirst(j -> all(e[j:end] .< threshold), eachindex(e))
    i === nothing && return NaN
    i == 1 && return Ms[1] # already below at the coarsest
    e₁, e₂ = e[i-1], e[i]
    (e₂ ≤ 0 || e₁ ≤ e₂) && return Ms[i]
    p = log(e₁ / e₂) / log(Ms[i] / Ms[i-1])          # local convergence order
    return Ms[i-1] * (e₁ / threshold)^(1 / p)
end

#####
##### Decaying turbulence (L1a)
#####

function compare_decay()
    io = open(joinpath(OUT, "decay_summary.txt"), "w")
    log(args...) = (println(io, args...); println(args...))
    log("Resolution study, decaying turbulence (L1a) in the NCAR 32³ box, L = $(round(L, digits=6)) m")
    log("NCAR fixed-box runs (BOXN=32) from the zero-padded L1b/n32_seed1/Zk10 field; Breeze from the same field imported exactly.")
    log("All diagnostics computed with NCARDNS.jl from the spectral coefficients (NCAR) or the C-grid arrays (Breeze).\n")

    ncar = Dict{String, Any}()
    for (name, dir) in (("n32", joinpath(RUNS, "L1a", "n32_seed1_decay")), ("n64", joinpath(RUNS, "R", "L1a_n64")),
                        ("n64_dt0.25", joinpath(RUNS, "R", "L1a_n64_dt0.25")), ("n128", joinpath(RUNS, "R", "L1a_n128")))
        isfile(joinpath(dir, "Zk1.out.ncf")) || (log("missing NCAR run $dir"); continue)
        ncar[name] = ncar_series("decay_" * name, dir; keep=true)
        @info "NCAR $name: $(length(ncar[name])) dumps"
    end
    reference = ncar["n128"]
    t₀ = reference[1].time
    tref = [s.time - t₀ for s in reference]

    breeze_names = sort(filter(n -> isfile(joinpath(OUT, "breeze_$(n)_timeseries.csv")),
                               ["R_M$(M)_$(a)" for M in (32, 48, 64, 96, 128, 192, 256) for a in ("centered2", "centered4", "centered6")]))
    breeze = Dict(n => breeze_series(n) for n in breeze_names)

    # match times (Breeze times start at 0)
    find_time(series, t; offset=0.0) = findfirst(s -> abs(s.time - offset - t) < 1e-5, series)

    metrics = (("K", s -> s.K), ("ε", s -> s.ε), ("Re_λ", s -> s.Reλ), ("S", s -> s.S), ("F", s -> s.F),
               ("F_T", s -> s.FT))
    fd_metrics = (("ε_FD", s -> s.ε_fd, r -> r.ε), ("S_FD", s -> s.S_fd, r -> r.S), ("F_FD", s -> s.F_fd, r -> r.F),
                  ("F_T_FD", s -> s.FT_fd, r -> r.FT))
    rows = (run=String[], code=String[], M=Int[], scheme=String[], metric=String[], time=Float64[], value=Float64[],
            reference=Float64[], error_pct=Float64[])
    pointwise = (run=String[], time=Float64[], velocity_error=Float64[], gradient_error=Float64[], gradient_correlation=Float64[])

    runs = Any[]
    for name in ("n32", "n64", "n64_dt0.25", "n128")
        haskey(ncar, name) || continue
        N = ncar[name][1].N
        push!(runs, ("NCAR_" * name, "NCAR", N, name == "n64_dt0.25" ? "spectral_dt/2" : "spectral", ncar[name], t₀))
    end
    for n in breeze_names
        M = parse(Int, match_regex(r"M(\d+)_", n)); scheme = split(n, '_')[3]
        push!(runs, ("Breeze_" * n, "Breeze", M, scheme, breeze[n], 0.0))
    end

    for (label, code, M, scheme, series, offset) in runs
        for (i, t) in enumerate(tref)
            j = find_time(series, t; offset)
            j === nothing && continue
            s = series[j]; r = reference[i]
            for (m, f) in metrics
                push!(rows.run, label); push!(rows.code, code); push!(rows.M, M); push!(rows.scheme, scheme)
                push!(rows.metric, m); push!(rows.time, t); push!(rows.value, f(s)); push!(rows.reference, f(r))
                push!(rows.error_pct, pct(f(s), f(r)))
            end
            if code == "Breeze" # Breeze's own second-order differences vs the exact reference values
                for (m, f, g) in fd_metrics
                    push!(rows.run, label); push!(rows.code, code); push!(rows.M, M); push!(rows.scheme, scheme)
                    push!(rows.metric, m); push!(rows.time, t); push!(rows.value, f(s)); push!(rows.reference, g(r))
                    push!(rows.error_pct, pct(f(s), g(r)))
                end
            end
            if code == "Breeze" && t > 0
                n = parse(Int, match_regex(r"M(\d+)_", label))
                path = joinpath(OUT, "snapshots", @sprintf("%s_n%02d.nc", replace(label, "Breeze_" => ""), j - 1))
                if isfile(path)
                    G, tt = load_fourier_velocity(path)
                    @assert abs(tt - t) < 1e-5
                    d = field_difference(G, r.field)
                    push!(pointwise.run, label); push!(pointwise.time, t); push!(pointwise.velocity_error, d.velocity_error)
                    push!(pointwise.gradient_error, d.gradient_error); push!(pointwise.gradient_correlation, d.gradient_correlation)
                end
            elseif code == "NCAR" && t > 0 && series !== reference && s.field !== nothing
                d = field_difference(s.field, r.field)
                push!(pointwise.run, label); push!(pointwise.time, t); push!(pointwise.velocity_error, d.velocity_error)
                push!(pointwise.gradient_error, d.gradient_error); push!(pointwise.gradient_correlation, d.gradient_correlation)
            end
        end
    end
    write_csv(joinpath(OUT, "decay_errors.csv"), rows)
    write_csv(joinpath(OUT, "decay_pointwise.csv"), pointwise)

    # Spectra: ratio to the reference at selected times
    spec = (run=String[], time=Float64[], k=Int[], energy=Float64[], reference=Float64[], ratio=Float64[])
    for (label, code, M, scheme, series, offset) in runs, t in tref[2:end]
        j = find_time(series, t; offset); j === nothing && continue
        i = findfirst(==(t), tref)
        s = series[j]; r = reference[i]
        for (q, k) in enumerate(s.k)
            k > length(r.k) && continue
            push!(spec.run, label); push!(spec.time, t); push!(spec.k, k); push!(spec.energy, s.E[q])
            push!(spec.reference, r.E[k]); push!(spec.ratio, s.E[q] / r.E[k])
        end
    end
    write_csv(joinpath(OUT, "decay_spectra_ratio.csv"), spec)
    # Largest shell k such that every shell ≤ k is within 1 % / 5 % of NCAR 128³ at every dump time
    log("Spectral accuracy: largest k (units 2π/L; k η₀ in brackets) up to which E(k) is within 1 % / 5 % of NCAR 128³ at all 20 dumps:")
    for lab in unique(spec.run)
        lab == "NCAR_n128" && continue
        out = String[]
        for thr in (0.01, 0.05)
            kgood = 0
            for k in 1:64
                sel = [q for q in eachindex(spec.run) if spec.run[q] == lab && spec.k[q] == k]
                (isempty(sel) || any(q -> abs(spec.ratio[q] - 1) ≥ thr, sel)) && break
                kgood = k
            end
            push!(out, @sprintf("%2d%%: k ≤ %2d (kη₀ ≤ %.2f)", round(Int, 100thr), kgood, kgood * 2π / L * η₀))
        end
        log(@sprintf("  %-28s %s", lab, join(out, "   ")))
    end

    # Summary tables: errors averaged over the decay (rms over the 20 dump times) and at selected times
    labels = unique(rows.run)
    log(@sprintf("Reference: NCAR 128³ (kmax η = %.2f at t₀).", reference[1].N / 2 * 2π / L * (1.6e-5^3 / reference[1].ε)^0.25))
    log("\nReference values (NCAR 128³) at the dump times:")
    log(@sprintf("%8s %12s %12s %8s %8s %8s %8s %8s", "t", "K", "ε", "Re_λ", "S", "F", "F_T", "ratio"))
    for (i, r) in enumerate(reference)
        log(@sprintf("%8.4f %12.5e %12.5e %8.3f %8.4f %8.4f %8.4f %8.4f", tref[i], r.K, r.ε, r.Reλ, r.S, r.F, r.FT, r.ratio))
    end
    for (window, sel) in (("all dumps t ∈ (0, 0.426] s", t -> t > 0), ("early t ≤ 0.107 s", t -> 0 < t ≤ 0.107),
                          ("late t ≥ 0.213 s", t -> t ≥ 0.213 - 1e-6))
        log("\nRMS relative error (%) vs NCAR 128³ over $window:")
        allm = vcat([m for (m, _) in metrics], [m for (m, _, _) in fd_metrics])
        log(@sprintf("%-28s", "run") * join([@sprintf("%9s", m) for m in allm]))
        for lab in labels
            line = @sprintf("%-28s", lab)
            for m in allm
                e = [rows.error_pct[q] for q in eachindex(rows.run) if rows.run[q] == lab && rows.metric[q] == m && sel(rows.time[q])]
                line *= isempty(e) ? @sprintf("%9s", "-") : @sprintf("%9.3f", sqrt(mean(abs2, e)))
            end
            log(line)
        end
    end
    # Breeze resolution needed for 1 % / 5 % RMS error (all dumps), log-log interpolated
    log("\nBreeze M needed for an RMS error (all dumps) below 1 % / 5 % (log-log interpolation between the runs);")
    log("equivalent Δ/η uses η = $(round(η₀ * 1e3, digits=4)) mm (stationary, ε = 0.05), i.e. the t₀ state:")
    req = (scheme=String[], metric=String[], threshold=Float64[], M=Float64[], Δ_over_η=Float64[], kmax_η=Float64[],
           ncar32_error=Float64[])
    for scheme in ("centered4", "centered2", "centered6"), m in vcat([m for (m, _) in metrics], [m for (m, _, _) in fd_metrics])
        labs = [l for l in labels if occursin("_" * scheme, l) && startswith(l, "Breeze")]
        length(labs) < 2 && continue
        Ms = [parse(Int, match_regex(r"M(\d+)_", l)) for l in labs]
        errs = [sqrt(mean(abs2, [rows.error_pct[q] for q in eachindex(rows.run) if rows.run[q] == l && rows.metric[q] == m && rows.time[q] > 0])) for l in labs]
        e32 = [rows.error_pct[q] for q in eachindex(rows.run) if rows.run[q] == "NCAR_n32" && rows.metric[q] == replace(m, "_FD" => "") && rows.time[q] > 0]
        for threshold in (1.0, 5.0)
            Mreq = required_resolution(Ms, errs, threshold)
            push!(req.scheme, scheme); push!(req.metric, m); push!(req.threshold, threshold); push!(req.M, Mreq)
            push!(req.Δ_over_η, L / Mreq / η₀); push!(req.kmax_η, Mreq / 2 * 2π / L * η₀)
            push!(req.ncar32_error, isempty(e32) ? NaN : sqrt(mean(abs2, e32)))
        end
        scheme == "centered4" && log(@sprintf("  %-10s %-7s 1%%: M ≈ %6.1f (Δ/η = %.2f)   5%%: M ≈ %6.1f (Δ/η = %.2f)   [NCAR 32³ RMS error %.3f %%; Breeze M at which it is matched: %.1f]",
                     scheme, m, req.M[end-1], req.Δ_over_η[end-1], req.M[end], req.Δ_over_η[end], req.ncar32_error[end],
                     isempty(e32) ? NaN : required_resolution(Ms, errs, sqrt(mean(abs2, e32)))))
    end
    write_csv(joinpath(OUT, "decay_required_resolution.csv"), req)

    log("\nPointwise difference to NCAR 128³ (relative L² error of ∇u; correlation of ∇u) at selected times:")
    times_sel = tref[[2, 3, 5, 9, 21]]
    log(@sprintf("%-28s", "run") * join([@sprintf("  t=%.3f      ", t) for t in times_sel]))
    for lab in unique(pointwise.run)
        line = @sprintf("%-28s", lab)
        for t in times_sel
            q = findfirst(x -> pointwise.run[x] == lab && abs(pointwise.time[x] - t) < 1e-6, eachindex(pointwise.run))
            line *= q === nothing ? @sprintf("%16s", "-") : @sprintf("  %.4f %.5f", pointwise.gradient_error[q], pointwise.gradient_correlation[q])
        end
        log(line)
    end
    close(io)
    return rows
end

match_regex(r, s) = match(r, s).captures[1]

#####
##### Forced turbulence (L1b)
#####

function compare_forced(; window_start=0.5)
    io = open(joinpath(OUT, "forced_summary.txt"), "w")
    log(args...) = (println(io, args...); println(args...))
    FDIR = joinpath(OUT, "forced")
    log("Resolution study, forced stationary turbulence (L1b) in the NCAR 32³ box; statistics over t ≥ $(window_start) s")
    log("after the start from the (padded) L1b/n32_seed<s>/Zk10 field. ± = standard error (autocorrelation-corrected;")
    log("for ensembles the larger of the pooled SE and the seed spread / √n).\n")

    groups = Dict{String, Vector{Any}}()   # label => vector of runs (each a vector of snapshots)
    function add!(label, snaps)
        haskey(groups, label) || (groups[label] = Any[])
        push!(groups[label], snaps)
    end
    # NCAR fixed-box runs
    for N in (64, 128), s in 1:3
        dir = joinpath(RUNS, "R", "L1b_n$(N)_s$(s)")
        info = joinpath(dir, "run_info.txt")
        (isfile(info) && occursin("wall_s", read(info, String))) || continue   # finished runs only
        snaps = ncar_series("forced_n$(N)_s$(s)", dir; skip_input=false)
        t₀ = snaps[1].time
        add!("NCAR_n$(N)", [merge(x, (; time=x.time - t₀)) for x in snaps])
    end
    # NCAR original 32³ (dumps every 0.974 s after the 1-s spin-up of a random start)
    for s in 1:3
        dir = joinpath(RUNS, "L1b", "n32_seed$(s)")
        snaps = ncar_series("forced_n32_s$(s)", dir; skip_input=true)
        add!("NCAR_n32", [merge(x, (; time=x.time)) for x in snaps if x.time ≥ 1.0])
    end
    # Breeze
    for f in readdir(FDIR)
        m = match(r"^breeze_(.+)_timeseries\.csv$", f)
        m === nothing && continue
        name = m.captures[1]                       # e.g. F_s1_M128_centered4
        mm = match(r"_M(\d+)_(centered\d)", name)
        add!("Breeze_M$(mm.captures[1])_$(mm.captures[2])", breeze_series(name; dir=FDIR, all_rows=true))
    end

    metrics = (("K", s -> s.K, false), ("ε", s -> s.ε, false), ("Re_λ", s -> s.Reλ, false), ("u′", s -> s.u′, false),
               ("S", s -> s.S, true), ("F", s -> s.F, true), ("F_T", s -> s.FT, true), ("E_f", s -> s.Ef, false))
    table = (label=String[], metric=String[], mean=Float64[], se=Float64[], n_runs=Int[], n_samples=Int[], run_means=String[])
    labels = sort(collect(keys(groups)); by=l -> (startswith(l, "NCAR") ? 0 : 1, parse(Int, match(r"(\d+)", l).captures[1]), l))
    for lab in labels, (m, f, slow) in metrics
        series = [[f(x) for x in run if x.time ≥ window_start && !isnan(f(x))] for run in groups[lab]]
        series = filter(x -> length(x) ≥ 3, series)
        isempty(series) && continue
        μ, se, means = ensemble_mean_se(series)
        push!(table.label, lab); push!(table.metric, m); push!(table.mean, μ); push!(table.se, se)
        push!(table.n_runs, length(series)); push!(table.n_samples, sum(length, series))
        push!(table.run_means, join([@sprintf("%.5g", x) for x in means], ";"))
    end
    write_csv(joinpath(OUT, "forced_statistics.csv"), table)
    get(lab, m) = (q = findfirst(i -> table.label[i] == lab && table.metric[i] == m, eachindex(table.label));
                   q === nothing ? (NaN, NaN) : (table.mean[q], table.se[q]))
    ref = haskey(groups, "NCAR_n128") ? "NCAR_n128" : "NCAR_n64"
    log("Reference: $ref")
    log(@sprintf("%-22s", "run") * join([@sprintf("%22s", m) for (m, _, _) in metrics]))
    for lab in labels
        line = @sprintf("%-22s", lab)
        for (m, _, _) in metrics
            μ, se = get(lab, m)
            line *= isnan(μ) ? @sprintf("%22s", "-") : @sprintf("%22s", @sprintf("%.4g ± %.2g", μ, se))
        end
        log(line)
    end
    log("\nDifference to $ref in % (± combined SE in %):")
    for lab in labels
        line = @sprintf("%-22s", lab)
        for (m, _, _) in metrics
            μ, se = get(lab, m); μr, ser = get(ref, m)
            if isnan(μ) || isnan(μr)
                line *= @sprintf("%22s", "-")
            else
                line *= @sprintf("%22s", @sprintf("%+.2f ± %.2f", pct(μ, μr), 100 * sqrt((se / μr)^2 + (μ * ser / μr^2)^2)))
            end
        end
        log(line)
    end

    # time-mean spectra and ratio to the reference
    spec = (label=String[], k=Int[], energy=Float64[], se=Float64[], ratio=Float64[])
    mean_spectrum(lab) = begin
        runs = groups[lab]
        kmax = minimum(maximum(x.k) for run in runs for x in run)
        Es = [reduce(hcat, [x.E[1:kmax] for x in run if x.time ≥ window_start]) for run in runs]
        μ = mean(reduce(hcat, Es); dims=2)[:, 1]
        se = [ensemble_mean_se([E[k, :] for E in Es])[2] for k in 1:kmax]
        μ, se
    end
    Eref, _ = mean_spectrum(ref)
    for lab in labels
        E, se = mean_spectrum(lab)
        for k in eachindex(E)
            push!(spec.label, lab); push!(spec.k, k); push!(spec.energy, E[k]); push!(spec.se, se[k])
            push!(spec.ratio, k ≤ length(Eref) ? E[k] / Eref[k] : NaN)
        end
    end
    write_csv(joinpath(OUT, "forced_spectra.csv"), spec)
    close(io)
    return table
end

if abspath(PROGRAM_FILE) == @__FILE__
    what = isempty(ARGS) ? "all" : ARGS[1]
    mkpath(OUT)
    what in ("decay", "all") && compare_decay()
    what in ("forced", "all") && compare_forced()
end
