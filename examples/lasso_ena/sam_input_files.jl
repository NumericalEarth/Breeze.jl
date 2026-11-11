#####
##### Readers for the traditional SAM text input files (`snd`, `lsf`, `sfc`) and the
##### Fortran namelist (`prm`). The formats follow `setforcing.f90` / `setdata.f90` of the
##### LASSO-ENA SAM (branch `lasso_ena_noice`, commit 12d0244). The `lsf` reader accepts
##### both the 7-column standard layout and the 9-column LASSO layout that appends the
##### geostrophic wind pair (`read_in_geostrophic_wind = .true.`).
#####

using SHA: sha256

"""
    SAMSounding

One time record of a SAM `snd` file. `z` is `missing` (SAM: `-9999`) when the record is
given on pressure levels. `θ` is the potential temperature (SAM column `tp[K]`), `q` the
total non-precipitating water in kg/kg (the file stores g/kg), `p` in Pa (file: hPa).
"""
struct SAMSounding{FT}
    day :: FT               # fractional day of year
    surface_pressure :: FT  # Pa
    z :: Vector{FT}         # m (NaN when the record is on pressure levels)
    p :: Vector{FT}         # Pa
    θ :: Vector{FT}         # K
    q :: Vector{FT}         # kg/kg
    u :: Vector{FT}         # m/s
    v :: Vector{FT}         # m/s
end

"""
    SAMLargeScaleForcing

One time record of a SAM `lsf` file. `tls` [K/s] is the large-scale tendency SAM adds to
its liquid/ice static energy variable `t`, `qls` [kg/kg/s] is added to vapor,
`uls`/`vls` [m/s] are the wind targets of the domain-mean nudging, `wls` [m/s] is the
large-scale vertical velocity used for full-field vertical advection, and `ug`/`vg` [m/s]
are the geostrophic winds entering the Coriolis term. When the file has no geostrophic
columns, `ug`/`vg` alias `uls`/`vls`, which is exactly what the LASSO SAM does when
`read_in_geostrophic_wind = .false.` (see `forcing.f90`).
"""
struct SAMLargeScaleForcing{FT}
    day :: FT
    surface_pressure :: FT
    z :: Vector{FT}
    p :: Vector{FT}
    tls :: Vector{FT}
    qls :: Vector{FT}
    uls :: Vector{FT}
    vls :: Vector{FT}
    wls :: Vector{FT}
    ug :: Vector{FT}
    vg :: Vector{FT}
    has_geostrophic_columns :: Bool
end

"""
    SAMSurfaceForcing

The SAM `sfc` time series: day, SST [K], sensible heat flux H [W/m²], latent heat flux
LE [W/m²], and kinematic surface stress τ [m²/s²].
"""
struct SAMSurfaceForcing{FT}
    day :: Vector{FT}
    sst :: Vector{FT}
    sensible_heat_flux :: Vector{FT}
    latent_heat_flux :: Vector{FT}
    kinematic_stress :: Vector{FT}
end

const SAM_MISSING_HEIGHT = -9999

is_comment_or_blank(line) = isempty(strip(line)) || startswith(strip(line), "!")

function tokens(line)
    return split(replace(line, ',' => ' '))
end

# Fortran writes double-precision literals with a `D` exponent (`-4.0D-05`).
parse_fortran(FT, s) = parse(FT, replace(s, r"[dD]" => "e"))

# Parse a record header of the form "day nlev pres0 ..." — pres0 may be absent in the
# `snd` header of some SAM versions, in which case we return `NaN` for it.
function parse_record_header(line, FT)
    t = tokens(line)
    day = parse_fortran(FT, t[1])
    nlev = parse(Int, t[2])
    pres0 = length(t) ≥ 3 ? tryparse(FT, replace(t[3], r"[dD]" => "e")) : nothing
    pres0 = isnothing(pres0) ? FT(NaN) : pres0
    return day, nlev, pres0
end

"""
    read_sam_sounding(path; FT=Float64)

Read every time record of a SAM `snd` file into a vector of [`SAMSounding`](@ref).
Pressures are converted from hPa to Pa and moisture from g/kg to kg/kg.
"""
function read_sam_sounding(path; FT=Float64)
    lines = filter(!is_comment_or_blank, readlines(path))
    records = SAMSounding{FT}[]
    i = 2 # skip column header line
    while i ≤ length(lines)
        day, nlev, pres0 = parse_record_header(lines[i], FT)
        z = zeros(FT, nlev); p = similar(z); θ = similar(z); q = similar(z); u = similar(z); v = similar(z)
        for n in 1:nlev
            t = tokens(lines[i + n])
            length(t) ≥ 6 || error("snd line $(i + n) has fewer than 6 columns: $(lines[i + n])")
            zn = parse_fortran(FT, t[1])
            z[n] = zn ≈ SAM_MISSING_HEIGHT ? FT(NaN) : zn
            p[n] = 100 * parse_fortran(FT, t[2])
            θ[n] = parse_fortran(FT, t[3])
            q[n] = 1e-3 * parse_fortran(FT, t[4])
            u[n] = parse_fortran(FT, t[5])
            v[n] = parse_fortran(FT, t[6])
        end
        push!(records, SAMSounding(day, 100 * pres0, z, p, θ, q, u, v))
        i += nlev + 1
    end
    isempty(records) && error("no sounding records found in $path")
    return records
end

"""
    read_sam_large_scale_forcing(path; FT=Float64)

Read every time record of a SAM `lsf` file into a vector of [`SAMLargeScaleForcing`](@ref).
Detects whether the geostrophic wind columns (`ug`, `vg`) are present from the number of
columns on the first data line.
"""
function read_sam_large_scale_forcing(path; FT=Float64)
    lines = filter(!is_comment_or_blank, readlines(path))
    records = SAMLargeScaleForcing{FT}[]
    i = 2
    while i ≤ length(lines)
        day, nlev, pres0 = parse_record_header(lines[i], FT)
        ncol = length(tokens(lines[i + 1]))
        has_geostrophic = ncol ≥ 9
        ncol ∈ (7, 9) || error("lsf line $(i + 1) has $ncol columns; expected 7 or 9")
        z = zeros(FT, nlev); p = similar(z)
        tls = similar(z); qls = similar(z); uls = similar(z); vls = similar(z); wls = similar(z)
        ug = similar(z); vg = similar(z)
        for n in 1:nlev
            t = tokens(lines[i + n])
            length(t) == ncol || error("lsf line $(i + n) has $(length(t)) columns; expected $ncol")
            zn = parse_fortran(FT, t[1])
            z[n] = zn ≈ SAM_MISSING_HEIGHT ? FT(NaN) : zn
            p[n] = 100 * parse_fortran(FT, t[2])
            tls[n] = parse_fortran(FT, t[3])
            qls[n] = parse_fortran(FT, t[4])
            uls[n] = parse_fortran(FT, t[5])
            vls[n] = parse_fortran(FT, t[6])
            wls[n] = parse_fortran(FT, t[7])
            if has_geostrophic
                ug[n] = parse_fortran(FT, t[8])
                vg[n] = parse_fortran(FT, t[9])
            else
                ug[n] = uls[n]
                vg[n] = vls[n]
            end
        end
        push!(records, SAMLargeScaleForcing(day, 100 * pres0, z, p, tls, qls, uls, vls, wls, ug, vg, has_geostrophic))
        i += nlev + 1
    end
    isempty(records) && error("no large-scale forcing records found in $path")
    return records
end

"""
    read_sam_surface_forcing(path; FT=Float64)

Read a SAM `sfc` file (day, SST, H, LE, τ) into a [`SAMSurfaceForcing`](@ref).
"""
function read_sam_surface_forcing(path; FT=Float64)
    lines = filter(!is_comment_or_blank, readlines(path))
    rows = [[parse_fortran(FT, x) for x in tokens(l)] for l in lines[2:end]]
    all(length(r) ≥ 5 for r in rows) || error("sfc rows must have 5 columns (day sst H LE tau)")
    day = [r[1] for r in rows]
    sst = [r[2] for r in rows]
    H = [r[3] for r in rows]
    LE = [r[4] for r in rows]
    τ = [r[5] for r in rows]
    return SAMSurfaceForcing(day, sst, H, LE, τ)
end

#####
##### Fortran namelist (`prm`) parser: returns Dict{String, Any} of lower-case keys.
#####

function parse_namelist_value(s)
    s = strip(s)
    low = lowercase(s)
    if low ∈ (".true.", "t", ".t.", "true")
        return true
    elseif low ∈ (".false.", "f", ".f.", "false")
        return false
    elseif startswith(s, "'") || startswith(s, "\"")
        return String(strip(s, ['\'', '"']))
    else
        v = tryparse(Int, s)
        isnothing(v) || return v
        v = tryparse(Float64, replace(s, r"[dD]" => "e"))
        isnothing(v) || return v
        return String(s)
    end
end

"""
    read_sam_namelist(path)

Parse a SAM `prm` Fortran namelist into a `Dict{String, NamelistValue}` keyed by lower-case
parameter name. Handles `key = value` pairs separated by commas or newlines, `!` comments,
logicals, integers, floats (including Fortran `d` exponents), and quoted strings.
"""
const NamelistValue = Union{Bool, Int, Float64, String}

function read_sam_namelist(path)
    params = Dict{String, NamelistValue}()
    text = read(path, String)
    lines = String[]
    for raw in split(text, '\n')
        line = strip(first(split(raw, '!')))
        isempty(line) && continue
        (startswith(line, "&") || startswith(line, "/") || startswith(line, "\$")) && continue
        push!(lines, line)
    end
    body = join(lines, " ")
    for m in eachmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*('[^']*'|\"[^\"]*\"|[^,\s]+)", body)
        params[lowercase(m.captures[1])] = parse_namelist_value(m.captures[2])
    end
    return params
end

#####
##### Heights from pressure levels, following `setdata.f90`:
#####   T = θ (p / 1000 hPa)^(R/cp) ; z₁ = (R/g) T₁ ln(p₀/p₁) ;
#####   zᵢ = zᵢ₋₁ + (R/2g)(Tᵢ + Tᵢ₋₁) ln(pᵢ₋₁/pᵢ)
##### SAM integrates with the dry temperature (no virtual correction) and a constant
##### R = 287, cp = 1004, g = 9.81; we keep those constants here so that the converted
##### heights match the SAM run that consumed the same file.
#####

"""
    sam_hydrostatic_heights(p, θ, surface_pressure; R=287, cp=1004, g=9.81, standard_pressure=1e5)

Return the heights [m] of pressure levels `p` [Pa] with potential temperatures `θ` [K]
using the SAM `setdata.f90` recipe. Levels below the surface (p > surface_pressure)
get negative heights, exactly as in SAM.
"""
function sam_hydrostatic_heights(p, θ, surface_pressure; R=287.0, cp=1004.0, g=9.81, standard_pressure=1e5)
    n = length(p)
    z = zeros(promote_type(eltype(p), eltype(θ)), n)
    T = @. θ * (p / standard_pressure)^(R / cp)
    z[1] = R / g * T[1] * log(surface_pressure / p[1])
    for i in 2:n
        z[i] = z[i-1] + R / (2g) * (T[i] + T[i-1]) * log(p[i-1] / p[i])
    end
    return z
end

"""
    record_heights(record)

Heights of the levels of a `snd`/`lsf` record: the file heights when they are given,
otherwise the SAM hydrostatic conversion of its pressure levels. For an `lsf` record
(which carries no temperature), `θ` must be supplied.
"""
function record_heights(record::SAMSounding)
    if all(isfinite, record.z)
        return copy(record.z)
    else
        return sam_hydrostatic_heights(record.p, record.θ, record.surface_pressure)
    end
end

function record_heights(record::SAMLargeScaleForcing, sounding::SAMSounding)
    if all(isfinite, record.z)
        return copy(record.z)
    else
        # Interpolate the sounding potential temperature onto the forcing pressure levels
        # (in log p), then convert hydrostatically. The forcing file shares the surface
        # pressure convention of the sounding.
        θ = interpolate_profile(log.(sounding.p), sounding.θ, log.(record.p))
        return sam_hydrostatic_heights(record.p, θ, record.surface_pressure)
    end
end

"""
    initial_sounding(soundings, day; tolerance=1e-6)

The sounding SAM's `setdata.f90` initializes from: the two consecutive `snd` records
bracketing `day` (= `day0`), interpolated linearly in time (levels, θ, q, u, v and the
surface pressure; pressure-level records keep their `NaN` heights). A record within
`tolerance` days of `day` is returned as is, and a single-record file is used as is (SAM
itself needs two records). `day` outside the records is an error, as in SAM.
"""
function initial_sounding(soundings::Vector{<:SAMSounding}, day; tolerance=1e-6)
    length(soundings) == 1 && return soundings[1]
    for record in soundings
        abs(record.day - day) ≤ tolerance && return record
    end
    for n in 1:length(soundings)-1
        a, b = soundings[n], soundings[n+1]
        if a.day ≤ day ≤ b.day
            length(a.z) == length(b.z) ||
                error("snd records at days $(a.day) and $(b.day) have different level counts; SAM assumes one layout")
            w = (day - a.day) / (b.day - a.day)
            blend(x, y) = @. (1 - w) * x + w * y
            z = all(isfinite, a.z) && all(isfinite, b.z) ? blend(a.z, b.z) : copy(a.z)
            return SAMSounding(convert(typeof(a.day), day), blend(a.surface_pressure, b.surface_pressure),
                               z, blend(a.p, b.p), blend(a.θ, b.θ), blend(a.q, b.q), blend(a.u, b.u), blend(a.v, b.v))
        end
    end
    error("day $day is beyond the sounding time range $(first(soundings).day)–$(last(soundings).day); SAM's setdata.f90 aborts here too")
end

"""
    interpolate_profile(x, y, xq)

Piecewise-linear interpolation of `y(x)` at query points `xq`, with constant
extrapolation beyond the ends of `x`. `x` may be increasing or decreasing.
"""
function interpolate_profile(x, y, xq::AbstractVector)
    return [interpolate_profile(x, y, ξ) for ξ in xq]
end

function interpolate_profile(x, y, ξ::Number)
    if x[1] > x[end]
        x = reverse(x); y = reverse(y)
    end
    n = length(x)
    ξ ≤ x[1] && return y[1]
    ξ ≥ x[n] && return y[n]
    i = searchsortedlast(x, ξ)
    i = clamp(i, 1, n - 1)
    w = (ξ - x[i]) / (x[i+1] - x[i])
    return (1 - w) * y[i] + w * y[i+1]
end

"""
    mass_fraction_from_mixing_ratio(r)

SAM's moisture variables are mixing ratios (kg per kg *dry* air; `sat.f90`: `qsatw =
0.622 e/(p - e)`), Breeze's are mass fractions of total air: `q = r / (1 + r)`.
"""
mass_fraction_from_mixing_ratio(r) = r / (1 + r)

"""
    day_to_seconds(day, day0)

Seconds elapsed since the fractional day-of-year `day0`.
"""
day_to_seconds(day, day0) = (day - day0) * 86400

"""
    file_sha256(path)

Hex SHA-256 digest of a file, for the provenance manifest.
"""
file_sha256(path) = bytes2hex(open(sha256, path))

"""
    read_sam_grd(path)

Read a SAM `grd` file (one scalar-level height per line, optionally more lines than the
model uses) and return the cell-center heights.
"""
function read_sam_grd(path)
    values = Float64[]
    for line in eachline(path)
        t = split(line)
        isempty(t) && continue
        v = tryparse(Float64, replace(t[1], r"[dD]" => "e"))
        isnothing(v) || push!(values, v)
    end
    isempty(values) && error("no heights found in $path")
    return values
end

parse_caseid(caseid) = parse.(Int, split(lowercase(String(caseid)), "x"))

"""
    epoch_from_day_of_year(day0; year=2017)

UTC `DateTime` of the fractional day-of-year `day0` (SAM `day0`; 1.0 = 1 January 00 UTC):
199.25 → 2017-07-18T06:00:00.
"""
function epoch_from_day_of_year(day0; year=2017)
    whole = floor(Int, day0)
    seconds = round(Int, (day0 - whole) * 86400)
    return DateTime(year, 1, 1) + Dates.Day(whole - 1) + Dates.Second(seconds)
end
