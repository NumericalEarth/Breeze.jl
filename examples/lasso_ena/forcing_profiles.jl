#####
##### Time-height forcing profiles as `FieldTimeSeries` on the model grid.
#####
##### Every profile is stored as a `FieldTimeSeries{Nothing, Nothing, Center}` (one column,
##### horizontally uniform) whose `times` are seconds since the case epoch `day0`. Forcing
##### kernels read `fts[1, 1, k, Time(t)]`, which interpolates linearly in time on the fly —
##### the same linear-in-time, linear-in-height treatment SAM applies in `forcing.f90`.
#####

using Oceananigans: FieldTimeSeries, Center, Face, set!
using Oceananigans.Units: Time

"""
    LargeScaleForcingProfiles

The time-height large-scale forcing of one case converted to the model grid. Keeps the
background wind (`uls`, `vls`: nudging targets), geostrophic wind (`ug`, `vg`: Coriolis
term), horizontal advective tendencies (`tls`, `qls`) and large-scale vertical velocity
(`wls`) as distinct fields, as the LASSO SAM does.
"""
struct LargeScaleForcingProfiles{FT, T}
    times :: Vector{FT}
    tls :: T
    qls :: T
    uls :: T
    vls :: T
    wls :: T
    ug :: T
    vg :: T
    has_geostrophic_columns :: Bool
end

Base.summary(p::LargeScaleForcingProfiles) =
    string("LargeScaleForcingProfiles over ", length(p.times), " times (",
           p.has_geostrophic_columns ? "with" : "without", " separate geostrophic columns)")

Base.show(io::IO, p::LargeScaleForcingProfiles) = print(io, summary(p))

"""
    sam_interpolate_column(x, y, xq; pressure_grid, above=:hold)

Interpolate the record `y(x)` onto the model levels `xq` exactly as SAM's `forcing.f90`
does: linearly in height for height-coordinate records (`pressure_grid=false`, `x`
ascending) or linearly in *pressure* for pressure-coordinate records (`pressure_grid=true`,
`x` descending, `xq` the model reference pressure). Below the first record level SAM's
formula extrapolates linearly through the first two levels. Above the last record level
the behaviour is variable-specific: `above=:zero` (tls, qls, wls) or `above=:hold` (uls,
vls, ug, vg keep the value of the level below).
"""
function sam_interpolate_column(x, y, xq; pressure_grid, above=:hold)
    n = length(x)
    out = zeros(promote_type(eltype(y), eltype(xq)), length(xq))
    for (k, ξ) in enumerate(xq)
        found = false
        for i in 2:n
            inside = pressure_grid ? (ξ ≥ x[i]) : (ξ ≤ x[i])
            if inside
                coef = (ξ - x[i-1]) / (x[i] - x[i-1])
                out[k] = y[i-1] + (y[i] - y[i-1]) * coef
                found = true
                break
            end
        end
        if !found
            out[k] = above === :zero ? zero(eltype(out)) : (k > 1 ? out[k-1] : y[n])
        end
    end
    return out
end

is_pressure_grid(record) = !all(isfinite, record.z)

"""
    profile_time_series(grid, times, columns; location=Center)

Build a `FieldTimeSeries{Nothing, Nothing, location}` on `grid` at `times` from a list of
column values `columns[n]` (one value per model level, bottom to top).
"""
function profile_time_series(grid, times, columns; location=Center)
    FT = eltype(grid)
    fts = FieldTimeSeries{Nothing, Nothing, location}(grid, FT.(times))
    Nz = size(grid, 3)
    for n in eachindex(times)
        column = columns[n]
        length(column) == Nz || throw(ArgumentError("column $n has $(length(column)) levels, grid has $Nz"))
        set!(fts[n], reshape(FT.(column), 1, 1, Nz))
    end
    return fts
end

"""
    LargeScaleForcingProfiles(grid, records, z_centers, reference_pressure; day0)

Convert `lsf` records to grid profiles following `forcing.f90`: height-coordinate records
are interpolated in z onto the cell centers `z_centers`; pressure-coordinate records are
interpolated linearly in pressure onto the model reference pressure `reference_pressure`
(both vectors over the model levels). Above the record top, tls/qls/wls are zero while the
wind profiles hold their last value.
"""
function LargeScaleForcingProfiles(grid, records::Vector{<:SAMLargeScaleForcing}, z_centers, reference_pressure; day0)
    FT = eltype(grid)
    times = FT[day_to_seconds(r.day, day0) for r in records]
    function columns(getter, above)
        map(records) do r
            pressure_grid = is_pressure_grid(r)
            x = pressure_grid ? r.p : r.z
            xq = pressure_grid ? reference_pressure : z_centers
            sam_interpolate_column(x, getter(r), xq; pressure_grid, above)
        end
    end
    make(getter, above) = profile_time_series(grid, times, columns(getter, above))
    tls = make(r -> r.tls, :zero)
    qls = make(r -> r.qls, :zero)
    wls = make(r -> r.wls, :zero)
    uls = make(r -> r.uls, :hold)
    vls = make(r -> r.vls, :hold)
    ug = make(r -> r.ug, :hold)
    vg = make(r -> r.vg, :hold)
    return LargeScaleForcingProfiles(times, tls, qls, uls, vls, wls, ug, vg, records[1].has_geostrophic_columns)
end

"""
    surface_time_series(grid, sfc, day0)

The `sfc` file as `FieldTimeSeries{Center, Center, Nothing}` (horizontally uniform 2D
fields at each time) for SST, sensible heat flux, latent heat flux and kinematic stress.
"""
function surface_time_series(grid, sfc::SAMSurfaceForcing, day0)
    FT = eltype(grid)
    times = FT[day_to_seconds(d, day0) for d in sfc.day]
    make(values) = begin
        fts = FieldTimeSeries{Center, Center, Nothing}(grid, times)
        for n in eachindex(times)
            set!(fts[n], FT(values[n]))
        end
        fts
    end
    return (; times,
              sst = make(sfc.sst),
              sensible_heat_flux = make(sfc.sensible_heat_flux),
              latent_heat_flux = make(sfc.latent_heat_flux),
              kinematic_stress = make(sfc.kinematic_stress))
end

"""
    interpolate_time_series(times, values, t)

Linear interpolation in time with constant extrapolation (SAM's convention for `sfc`).
"""
interpolate_time_series(times, values, t) = interpolate_profile(times, values, t)

"""
    SoundingTargetProfiles(grid, soundings, z_centers, reference_pressure; day0, R=287, cp=1004)

The observed-sounding targets of SAM's nudging/upper-boundary relaxation as profile time
series: `T` (SAM `tg0`, absolute temperature `θ_snd (p/1000 hPa)^(R/cp)` with the model
reference pressure) and `q` (SAM `qg0`, total water). Pressure-coordinate records are
interpolated in pressure and height-coordinate records in height, as `forcing.f90` does;
above the sounding top the last value is held.
"""
function SoundingTargetProfiles(grid, soundings::Vector{<:SAMSounding}, z_centers, reference_pressure;
                                day0, R=287.0, cp=1004.0, standard_pressure=1e5, moisture_basis=:mixing_ratio)
    FT = eltype(grid)
    times = FT[day_to_seconds(r.day, day0) for r in soundings]
    if length(soundings) == 1
        # SAM pads a single record so that the profile holds for all times
        times = FT[times[1], times[1] + 86400 * 1000]
        soundings = [soundings[1], soundings[1]]
    end
    Π = (reference_pressure ./ standard_pressure) .^ (R / cp)
    function column(r, getter)
        pressure_grid = is_pressure_grid(r)
        x = pressure_grid ? r.p : r.z
        xq = pressure_grid ? reference_pressure : z_centers
        sam_interpolate_column(x, getter(r), xq; pressure_grid, above=:hold)
    end
    T_columns = [column(r, r -> r.θ) .* Π for r in soundings]
    convert_q = moisture_basis === :mixing_ratio ? mass_fraction_from_mixing_ratio :
                moisture_basis === :mass_fraction ? identity :
                throw(ArgumentError("moisture_basis must be :mixing_ratio or :mass_fraction"))
    q_columns = [convert_q.(column(r, r -> r.q)) for r in soundings]
    return (; times, T = profile_time_series(grid, times, T_columns), q = profile_time_series(grid, times, q_columns))
end

"""
    shifted_profile_time_series(fts, shift)

A copy of the profile time series `fts` with the constant `shift` subtracted (used to move
wind profiles into SAM's translating frame).
"""
function shifted_profile_time_series(fts, shift)
    grid = fts.grid
    FT = eltype(grid)
    out = FieldTimeSeries{Nothing, Nothing, Center}(grid, fts.times)
    for n in eachindex(fts.times)
        set!(out[n], fts[n])
        parent(out[n]) .-= FT(shift)
    end
    return out
end

"""
    sounding_nudging_targets(grid, soundings, z_centers; day0, moisture_basis=:mixing_ratio, frame_velocity=(0, 0))

The domain-mean nudging targets of a multi-record `snd` as profile time series on the model
levels: `θ` (the file's potential temperature, which is the liquid-ice potential temperature
of condensate-free air), `q` (the file's total water as a Breeze mass fraction) and `u`, `v`
(minus `frame_velocity`, SAM's translating frame). Every record is interpolated *in height*
exactly as the initial state is ([`SoundingProfiles`](@ref): pressure-level records are
converted to heights with the `setdata.f90` recipe, height-coordinate records are used as they
are), so the target of the record at `day0` equals the initial condition. Between records the
targets are linear in time; before the first and after the last record they hold the end
record (`FieldTimeSeries` `Clamp` indexing). A single record is held for all times.
"""
function sounding_nudging_targets(grid, soundings::Vector{<:SAMSounding}, z_centers;
                                  day0, moisture_basis=:mixing_ratio, frame_velocity=(0, 0))
    FT = eltype(grid)
    times = FT[day_to_seconds(r.day, day0) for r in soundings]
    issorted(times) && allunique(times) || throw(ArgumentError("snd record times must increase strictly"))
    if length(soundings) == 1
        times = FT[times[1], times[1] + 86400 * 1000]
        soundings = [soundings[1], soundings[1]]
    end
    profiles = [SoundingProfiles(r; moisture_basis) for r in soundings]
    column(p, name) = [p(name, z) for z in z_centers]
    uᶠ, vᶠ = frame_velocity
    make(f) = profile_time_series(grid, times, [f(p) for p in profiles])
    return (; times,
              θ = make(p -> column(p, :θ)),
              q = make(p -> column(p, :qᵗ)),
              u = make(p -> column(p, :u) .- uᶠ),
              v = make(p -> column(p, :v) .- vᶠ))
end
