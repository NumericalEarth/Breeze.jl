"""
    BreezeCalibration

Calibrate Breeze's `TKEBasedTurbulenceClosure` against the Shen et al. (2022) library of GCM-forced
large-eddy simulations with ensemble Kalman inversion, using Breeze's column-ensemble mode: every
combination of a parameter set and an LES member is one column of a single `AtmosphereModel`, so one
forward map — one model run — evaluates the whole ensemble on all training members at once.

Each column reproduces the LES's own single-column protocol (Shen et al. 2022, §2): the member's subsidence
(upwind, as in the LES), horizontal-advection and eddy-flux tendencies, its surface fluxes, Kessler warm-rain
microphysics, relaxation of the winds toward the GCM profiles the LES was relaxed toward (6 h) and of θ and qᵗ
above 3 km (24 h), and interactive all-sky RRTMGP radiation with the LES's fixed sun. Because RRTMGP needs the
atmosphere above, the column extends to 25 km on a stretched grid and is relaxed strongly toward the monthly-mean
GCM column above the LES top. Columns are scored by the RMSE of their time-mean θˡ, qᵗ, qˡ, u and v below 3 km
over the LES's target window, as means over 100 m observation cells so that every model grid is scored alike.
"""
module BreezeCalibration

export PROTOCOL_VERSION,
       LESMember, load_member, member_path, library_members,
       ColumnEnsembleProblem, MultiResolutionProblem, forward_map, observations, run_ensemble,
       les_faces, uniform_faces, hindcast_faces, regrid_column, regrid_columns,
       observable_variables, default_variables, default_observation_noise,
       ParameterSpace, RiDependentSpace, ConstantSpace, named, space_of,
       Parameters, default_parameters, catke_calibration_parameters, prior_center, closure_from, parameter_names,
       run_eki, evaluate, rmse_table, replay, prior_distribution,
       DataMisfitController, NesterovAccelerator, DefaultAccelerator, SECNice, NoLocalization

using Breeze
# Explicit imports: Breeze re-exports several of these Oceananigans names, and an unqualified `using`
# of both would make them ambiguous inside this module
using Oceananigans: RectilinearGrid, Flat, Bounded, Center, Face, CPU,
                    Field, CenterField, XFaceField, YFaceField, ZFaceField, FieldTimeSeries,
                    set!, interior, compute!, Simulation, run!, add_callback!, TimeInterval, time,
                    Relaxation, Forcing, FluxBoundaryCondition, FieldBoundaryConditions,
                    regrid!, ReferenceToStretchedDiscretization, LinearStretching
using Breeze.TurbulenceClosures: ConstantStabilityFunctions
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula, RadiativeTransferModel, AllSkyOptics, FixedCosineZenith
using RRTMGP, ClimaComms # loading RRTMGP activates Breeze's radiative-transfer extension
using Breeze.AtmosphereModels: moisture_specific_name
using Oceananigans.Advection: UpwindBiased
using Oceananigans.Units
using Oceananigans.Grids: ColumnEnsembleSize
using Oceananigans.Grids: znode
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using NCDatasets
using Statistics
using LinearAlgebra
using Random
using Printf
using JLD2
using Pkg.Artifacts: ensure_artifact_installed
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers: SECNice, NoLocalization
const EKP = EnsembleKalmanProcesses

#####
##### The protocol version
#####

"""
The version of the forward map that `run_ensemble` implements. A checkpoint records it, and resuming
into a different version is refused: EKI resumes by *replaying saved forward maps* through the update,
so a checkpoint whose `G` came from different physics would be mixed with new evaluations and the
posterior would mean nothing. Neither the parameter count nor the recorded grids and members catch
this — the physics can change with all of them fixed.

**Bump this whenever the map from parameters to observations changes**, in `run_ensemble` or in the
Breeze physics it exercises, and add a line here.

- `1` — the protocol of PR #975 as merged into this study: Kessler warm rain with Tetens saturation,
  first-order upwind subsidence, relaxation toward the `*_mean_initial` GCM profiles, interactive
  all-sky RRTMGP on a column extended to 25 km. Superseded protocols (saturation adjustment, centered
  subsidence, replayed radiation) predate the marker and have no version recorded at all.
- `2` — subsidence of cloud liquid and rain as mass fractions, relaxation of nonprecipitating
  total water, condensate removal in the upper extension, and exclusion of rain from the
  closure's saturation test (with all water retained in its dry-air denominator).
- `3` — the branch merged forward: the SSP RK3 third-stage tendency is evaluated at tⁿ + Δt/2
  (#980), which changes the time integration and so every trajectory; the energy flux and forcing
  move to the formulation-agnostic `ρE`/`E` keys (#974), a re-keying that leaves the tendency —
  still divided by cᵖᵐ Π for a `ρθ` formulation — unchanged.
"""
const PROTOCOL_VERSION = 3

#####
##### The LES library
#####

library_path() = ensure_artifact_installed("shen_et_al_2022_les_profiles", joinpath(pkgdir(Breeze), "Artifacts.toml"))

member_path(site, month; gcm = "CNRM-CM6-1", climate = "amip") =
    joinpath(library_path(), "cfsite$(lpad(site, 2, '0'))_$(gcm)_$(climate)_$(month).nc")

"""All (site, month) pairs of the `amip` CNRM-CM6-1 members present in the library."""
function library_members(; climate = "amip")
    members = Tuple{Int, String}[]
    for site in [2:15; 17:23], month in ("01", "04", "07", "10")
        isfile(member_path(site, month; climate)) && push!(members, (site, month))
    end
    return members
end

"""
One reduced LES member: everything the single-column protocol needs, on the LES's own vertical grid
(`z`, cell centers) and hourly time axis (`t`), plus the time-mean targets over the LES's final two
days.
"""
struct LESMember
    site :: Int
    month :: String
    z :: Vector{Float64}          # cell centers of the LES grid
    t :: Vector{Float64}          # hourly times of the surface series and radiation
    window :: Tuple{Float64, Float64}
    p₀ :: Float64                 # reference pressure extrapolated to the surface
    θ₀ :: Float64                 # initial mixed-layer potential temperature
    initial :: NamedTuple         # θ, qᵗ, u, v profiles
    subsidence :: Vector{Float64} # wˢ at the LES centers
    dTdt :: Vector{Float64}       # cᵖᵈ (hadv + fluc) heating, J kg⁻¹ s⁻¹
    dqdt :: Vector{Float64}       # hadv + fluc moistening, kg kg⁻¹ s⁻¹
    heating :: Matrix{Float64}    # cᵖᵈ × radiative heating, (Nz, Nt)
    nudging :: NamedTuple         # u, v, θ, qᵗ relaxation targets (the GCM profiles by default)
    surface :: NamedTuple         # shf, lhf, uw, vw hourly series; ℒᵛ
    targets :: NamedTuple         # θˡ, qᵗ, qˡ, qʳ, u, v time means; max cloud fraction
    gcm :: Union{Nothing, NamedTuple} # the GCM column the LES was forced with: z, θ, qᵛ, p, insolation, coszen (for the atmosphere above the LES)
end

const default_gcm_columns = joinpath(@__DIR__, "..", "data", "gcm_columns_CNRM-CM6-1_amip.nc")

"""
The monthly-mean CNRM-CM6-1 column at cfSite `site` for `month` from `path` (see `scripts/fetch_gcm_columns.jl`):
heights `z` above the surface, dry potential temperature `θ` (referenced to 10⁵ Pa), specific humidity `qᵛ`,
pressure `p`, ordered from the surface up, plus the monthly-mean TOA `insolation` and insolation-weighted `coszen`.
`nothing` if the file is absent.
"""
function load_gcm_column(site, month; path = default_gcm_columns, constants = ThermodynamicConstants())
    isfile(path) || return nothing
    ds = NCDataset(path)
    si = findfirst(==(site), ds["site"][:]); mi = findfirst(==(parse(Int, month)), ds["month"][:])
    (isnothing(si) || isnothing(mi)) && (close(ds); return nothing)
    z = Float64.(ds["zg"][si, mi, :]); T = Float64.(ds["ta"][si, mi, :]); q = Float64.(ds["hus"][si, mi, :]); p = Float64.(ds["pfull"][si, mi, :])
    insolation = haskey(ds, "insolation") ? Float64(ds["insolation"][si, mi]) : NaN
    coszen = haskey(ds, "coszen") ? Float64(ds["coszen"][si, mi]) : NaN
    close(ds)
    Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants); cᵖᵈ = constants.dry_air.heat_capacity
    θ = T .* (1e5 ./ p) .^ (Rᵈ / cᵖᵈ)
    order = sortperm(z)
    return (z = z[order], θ = θ[order], qᵛ = q[order], p = p[order], insolation, coszen)
end

"""
Load one LES member. The relaxation targets are the GCM profiles the LES itself was relaxed toward — its initial
profiles, since Shen et al. (2022) initialize from the 5-year-mean GCM state and apply time-invariant forcing. (The
file's `*_mean_nudge` profiles are the LES's whole-run means, not the targets: `u_mean_initial ≈ u_mean_nudge +
6 h × dudt_nudge` to 0.01 m s⁻¹ across the library.)
"""
function load_member(site, month; climate = "amip", constants = ThermodynamicConstants(), gcm_columns = default_gcm_columns)
    ds = NCDataset(member_path(site, month; climate))
    z = Float64.(ds["z"][:]); t = Float64.(ds["time"][:])
    g = constants.gravitational_acceleration
    cᵖᵈ = constants.dry_air.heat_capacity
    p₀ = ds["p0"][1] + ds["rho0"][1] * g * z[1]
    θ₀ = Float64(ds["thetali_mean_initial"][1])
    prof(name) = Float64.(ds[name][:])
    initial = (θ = prof("thetali_mean_initial"), qᵗ = prof("qt_mean_initial"), qˡ = prof("ql_mean_initial"), u = prof("u_mean_initial"), v = prof("v_mean_initial"))
    subsidence = prof("ls_subsidence")
    dTdt = cᵖᵈ .* (prof("dtdt_hadv") .+ prof("dtdt_fluc"))
    dqdt = prof("dqtdt_hadv") .+ prof("dqtdt_fluc")
    heating = cᵖᵈ .* Float64.(ds["dtdt_rad_hourly"][:, :])
    nudging = (u = prof("u_mean_initial"), v = prof("v_mean_initial"), θ = prof("thetali_mean_initial"), qᵗ = prof("qt_mean_initial"))
    ℒᵛ = Breeze.Thermodynamics.liquid_latent_heat(Float64(ds["surface_temperature"][1]), constants)
    surface = (shf = prof("shf_surface_mean"), lhf = prof("lhf_surface_mean"), uw = prof("uw_surface_mean"), vw = prof("vw_surface_mean"),
               Tₛ = prof("surface_temperature"), ℒᵛ = ℒᵛ)
    window = (Float64(ds.attrib["target_window_start"]), Float64(ds.attrib["target_window_end"]))
    targets = (θˡ = prof("thetali_mean"), qᵗ = prof("qt_mean"), qˡ = prof("ql_mean"), qʳ = prof("qr_mean"), u = prof("u_mean"), v = prof("v_mean"),
               cloud_fraction = maximum(Float64.(ds["cloud_fraction_mean"][:])))
    close(ds)
    gcm = load_gcm_column(site, month; path = gcm_columns, constants)
    return LESMember(site, month, z, t, window, p₀, θ₀, initial, subsidence, dTdt, dqdt, heating, nudging, surface, targets, gcm)
end

# Linear interpolation of a profile onto new heights, constant beyond the ends
function interpolate_profile(zs, vs, znew)
    return map(znew) do z
        z ≤ zs[1] && return vs[1]
        z ≥ zs[end] && return vs[end]
        k = searchsortedlast(zs, z)
        w = (z - zs[k]) / (zs[k+1] - zs[k])
        (1 - w) * vs[k] + w * vs[k+1]
    end
end

#####
##### Parameter spaces
#####

# The physical parameters of the closure: the coefficients of its stability functions — twelve for the
# Richardson-number-dependent functions (momentum, tracers, TKE and dissipation, each in unstable air, at
# neutral and in the stable limit, with the onset and width of their stable transition) or four constants —
# and the wall coefficient of the mixing length. Two more parameters set a surface flux of turbulent kinetic
# energy, CATKE's Jᵉ = Cᵂu★ u★³ + Cᵂʷ wΔ³ with wΔ³ = Δz Jᵇ the convective velocity of the first cell: the
# production inside the first grid cell that the resolved shear and buoyancy cannot carry. Zero flux is the
# default closure.

"""
A set of free parameters of `TKEBasedTurbulenceClosure`: names, defaults, the prior's center and the closure
they build. Every space includes the wall coefficient Cˢ of the gradient-limited mixing length and the two
surface-TKE-flux coefficients.
"""
abstract type ParameterSpace end

"""
The 17 parameters of the Richardson-number-dependent closure: the twelve endpoints of
`RiDependentStabilityFunctions`, the transition Ri⁰ and Riᵟ, Cˢ, Cᵂu★ and Cᵂʷ.
"""
struct RiDependentSpace <: ParameterSpace end

"""
The 7 parameters of the constant-coefficient closure, `ConstantStabilityFunctions` in Nakanishi–Niino's
form: Cᵘ, Cᶜ, Cᵉ, Cᴰ, Cˢ, Cᵂu★ and Cᵂʷ.
"""
struct ConstantSpace <: ParameterSpace end

Base.summary(::RiDependentSpace) = "RiDependentSpace"
Base.summary(::ConstantSpace) = "ConstantSpace"
Base.show(io::IO, space::ParameterSpace) = print(io, summary(space))

parameter_names(::RiDependentSpace) = (:Cᵘ⁻, :Cᵘ⁰, :Cᵘ⁺, :Cᶜ⁻, :Cᶜ⁰, :Cᶜ⁺, :Cᵉ⁻, :Cᵉ⁰, :Cᵉ⁺, :Cᴰ⁻, :Cᴰ⁰, :Cᴰ⁺, :Ri⁰, :Riᵟ, :Cˢ, :Cᵂu★, :Cᵂʷ)
parameter_names(::ConstantSpace) = (:Cᵘ, :Cᶜ, :Cᵉ, :Cᴰ, :Cˢ, :Cᵂu★, :Cᵂʷ)
parameter_names() = parameter_names(RiDependentSpace())

"""The named tuple of `values` in `space`."""
named(space::ParameterSpace, values) = NamedTuple{parameter_names(space)}(Tuple(Float64.(collect(values))))

"""The parameter space with `n` parameters."""
space_of(n::Int) = n == 17 ? RiDependentSpace() : n == 7 ? ConstantSpace() : error("No parameter space has $n parameters")
space_of(p::NamedTuple) = space_of(length(p))
space_of(v::AbstractVector) = space_of(length(v))

"""
Nakanishi–Niino's constants in Breeze's normalization — at all three endpoints for the Ri-dependent space,
so that its stability functions are constant and the closure is Breeze's default — with CATKE's
transition Ri⁰ and Riᵟ (then inert), the default wall coefficient and zero surface TKE flux.
"""
default_parameters(space::RiDependentSpace) = named(space, (0.149, 0.149, 0.149, 0.201, 0.201, 0.201, 0.298, 0.298, 0.298, 0.388, 0.388, 0.388, 0.254, 1.02, 1.316, 0.0, 0.0))
default_parameters(space::ConstantSpace) = named(space, (0.149, 0.201, 0.298, 0.388, 1.316, 0.0, 0.0))
default_parameters() = default_parameters(RiDependentSpace())

const Parameters = typeof(default_parameters(RiDependentSpace()))
Parameters(v::AbstractVector) = named(RiDependentSpace(), v)

"""CATKE's parameters, as `catke_parameters()` supplies them, with CATKE's surface TKE flux coefficients."""
catke_calibration_parameters() = Parameters((0.370, 0.361, 0.242, 0.572, 0.369, 0.098, 1.447, 7.863, 0.548, 0.923, 1.604, 0.579, 0.254, 1.02, 1.131, 3.72, 1.10))

"""
The center of the prior: the defaults, except that the surface TKE flux coefficients — zero in the default
closure, which a positive prior cannot be centered on — start at one.
"""
prior_center(space::ParameterSpace) = merge(default_parameters(space), (Cᵂu★ = 1.0, Cᵂʷ = 1.0))
prior_center() = prior_center(RiDependentSpace())

parameter_index(space::ParameterSpace, name) = findfirst(==(name), parameter_names(space))

"""The closure for one parameter set of `space`, with the gradient-limited mixing length."""
function closure_from(space::RiDependentSpace, values; static_stability = MoistStaticStability())
    p = named(space, values)
    stability_functions = RiDependentStabilityFunctions(; Cᵘ⁻ = p.Cᵘ⁻, Cᵘ⁰ = p.Cᵘ⁰, Cᵘ⁺ = p.Cᵘ⁺,
                                                          Cᶜ⁻ = p.Cᶜ⁻, Cᶜ⁰ = p.Cᶜ⁰, Cᶜ⁺ = p.Cᶜ⁺,
                                                          Cᵉ⁻ = p.Cᵉ⁻, Cᵉ⁰ = p.Cᵉ⁰, Cᵉ⁺ = p.Cᵉ⁺,
                                                          Cᴰ⁻ = p.Cᴰ⁻, Cᴰ⁰ = p.Cᴰ⁰, Cᴰ⁺ = p.Cᴰ⁺,
                                                          Ri⁰ = p.Ri⁰, Riᵟ = p.Riᵟ)
    return TKEBasedTurbulenceClosure(; mixing_length = GradientLimitedMixingLength(Cˢ = p.Cˢ), stability_functions, static_stability)
end

function closure_from(space::ConstantSpace, values; static_stability = MoistStaticStability())
    p = named(space, values)
    stability_functions = ConstantStabilityFunctions(Cᵘ = p.Cᵘ, Cᶜ = p.Cᶜ, Cᵉ = p.Cᵉ, Cᴰ = p.Cᴰ)
    return TKEBasedTurbulenceClosure(; mixing_length = GradientLimitedMixingLength(Cˢ = p.Cˢ), stability_functions, static_stability)
end

closure_from(p::NamedTuple; kw...) = closure_from(space_of(p), collect(p); kw...)
closure_from(v::AbstractVector; kw...) = closure_from(space_of(v), v; kw...)

#####
##### The column-ensemble problem
#####

"""
The training members and everything about them precomputed for one model grid: one `ColumnEnsembleProblem`
serves every forward map on that grid. The model's vertical faces `z_faces` default to the LES grid (uniform
20 m, 200 levels, shared by all members); pass `Δz` for a uniform grid of that spacing over the LES depth,
or `z_faces` for any other, such as [`hindcast_faces`](@ref). The LES forcing, initial and target profiles
are regridded conservatively onto the model cells with Oceananigans' `regrid!`. Observations are the
time-mean θˡ (K) and qᵗ (g kg⁻¹) as means over the cells of the observation grid `observation_faces` —
100 m cells up to 3 km by default — onto which model and LES profiles are both regridded, so that the
misfit is independent of the model grid. `variables` selects the observed time means among `θˡ` (K), `qᵗ`,
`qˡ` and `qʳ` (g kg⁻¹), `u` and `v` (m s⁻¹); by default all but `qʳ` (see `default_variables`), so that every
prognostic field constrains the closure — θˡ and qᵗ alone leave the momentum coefficients free.

The column extends above the LES data to `top` (25 km by default; `nothing` for a column ending at the LES top)
on faces growing by the factor `stretching` per cell, where the state is the member's monthly-mean GCM column and
is relaxed strongly toward it — the atmosphere interactive radiation needs above the boundary layer.
"""
struct ColumnEnsembleProblem
    members :: Vector{LESMember}
    zf :: Vector{Float64}             # the model's vertical faces
    zc :: Vector{Float64}
    les_zf :: Vector{Float64}         # the LES's vertical faces, the source of forcing and targets
    observation_zf :: Vector{Float64} # the faces of the observation cells
    times :: Vector{Float64}          # the common hourly axis, from the first sample to the latest window end
    static_stability :: Any
    variables :: Tuple{Vararg{Symbol}} # the observed fields, among (:θˡ, :qᵗ, :qˡ, :qʳ, :u, :v)
    les_top :: Float64                  # the top of the LES data; above it the column is relaxed strongly to the GCM
end

const observable_variables = (:θˡ, :qᵗ, :qˡ, :qʳ, :u, :v)
# The fields scored by default. Rain water is diagnosed but not scored: the LES's horizontal-mean rain water
# (a few mg kg⁻¹, from patchy showers) is not the same quantity as a mean-field Kessler column's (tens of
# mg kg⁻¹), and scoring it drove a calibration away from the thermodynamic fields.
const default_variables = (:θˡ, :qᵗ, :qˡ, :u, :v)
# Scales taking model units to the observation units K, g kg⁻¹ and m s⁻¹
const observation_scales = (θˡ = 1.0, qᵗ = 1e3, qˡ = 1e3, qʳ = 1e3, u = 1.0, v = 1.0)
# Default noise per observation cell in those units; the LES's time-mean rain water is a few mg kg⁻¹
const default_observation_noise = (θˡ = 0.25, qᵗ = 0.25, qˡ = 0.1, qʳ = 0.002, u = 0.5, v = 0.5)

function ColumnEnsembleProblem(members::Vector{LESMember}; Δz = nothing, z_faces = nothing,
                               observation_faces = collect(0.0:100.0:3000.0), static_stability = MoistStaticStability(),
                               variables = default_variables, top = 25_000, stretching = 1.12)
    all(v -> v in observable_variables, variables) || error("variables must be among $observable_variables")
    z = members[1].z
    all(m -> m.z ≈ z, members) || error("The members do not share a vertical grid")
    les_zf = les_faces(z)
    zf = !isnothing(z_faces) ? collect(Float64, z_faces) : isnothing(Δz) ? les_zf : uniform_faces(Δz, les_zf[end])
    # A tall column: the faces above the LES top grow geometrically to `top`, and the GCM column supplies the state there
    if !isnothing(top) && top > les_zf[end] + 1e-6
        all(m -> !isnothing(m.gcm), members) || error("A column above the LES needs the GCM columns; run scripts/fetch_gcm_columns.jl")
        zf = extend_faces(zf, top, stretching)
    end
    zf[end] ≤ les_zf[end] + 1e-6 || zf[end] ≈ top || error("The model grid extends above the LES data, whose top is at $(les_zf[end]) m")
    observation_faces[end] ≤ zf[end] + 1e-6 || error("The observation cells extend above the model grid")
    zc = (zf[1:end-1] .+ zf[2:end]) ./ 2
    # The members run for different lengths (3.7 to 6 days) but share the hourly sampling; the
    # ensemble steps to the latest window end and every column is scored over its own window
    Δt = members[1].t[2] - members[1].t[1]
    all(m -> m.t[1] ≈ members[1].t[1] && all(≈(Δt), diff(m.t)), members) || error("The members do not share the hourly sampling")
    t_end = maximum(m.window[2] for m in members)
    times = collect(members[1].t[1]:Δt:t_end)
    times[end] < t_end && push!(times, t_end)
    return ColumnEnsembleProblem(members, zf, zc, les_zf, collect(Float64, observation_faces), times, static_stability, Tuple(variables), les_zf[end])
end

ColumnEnsembleProblem(members::Vector{Tuple{Int, String}}; kw...) = ColumnEnsembleProblem([load_member(s, m) for (s, m) in members]; kw...)

# Uniform hourly series interpolated at time t, column j: F is (N_members, Nt)
@inline function series_value(F, t₀, Δt, Nt, t, j)
    s = (t - t₀) / Δt
    n = clamp(floor(Int, s) + 1, 1, Nt - 1)
    w = clamp(s - (n - 1), 0, 1)
    return @inbounds (1 - w) * F[j, n] + w * F[j, n+1]
end

@inline prescribed_flux(i, j, grid, clock, fields, p) = series_value(p.F, p.t₀, p.Δt, p.Nt, clock.time, j)

# The LES relaxes nonprecipitating total water. Apply that source to vapor while
# retaining cloud water, so d(qᵛ + qᶜˡ)/dt = rate × (qₙ - qᵛ - qᶜˡ).
@inline function total_water_relaxation(i, j, k, grid, clock, fields, p)
    z = znode(i, j, k, grid, Center(), Center(), Center())
    @inbounds qᵗ = (fields.ρqᵛ[i, j, k] + fields.ρqᶜˡ[i, j, k]) / p.density[i, j, k]
    return @inbounds p.rate * p.mask(z) * (p.target[i, j, k] - qᵗ)
end

# The surface flux of ρe: ρ₀ (Cᵂu★ u★³ + Cᵂʷ wΔ³) with the coefficients of parameter set i and the
# hourly u★³ and wΔ³ series of member j
@inline function surface_tke_flux(i, j, grid, clock, fields, p)
    u★³ = series_value(p.u★³, p.t₀, p.Δt, p.Nt, clock.time, j)
    wΔ³ = series_value(p.wΔ³, p.t₀, p.Δt, p.Nt, clock.time, j)
    return @inbounds p.ρ₀[j] * (p.Cᵂu★[i] * u★³ + p.Cᵂʷ[i] * wΔ³)
end

"""Faces above `faces[end]` growing by the factor `stretching` per cell from the last LES spacing until `top` (the last face lands on `top`)."""
function extend_faces(faces, top, stretching)
    zf = copy(faces); Δ = zf[end] - zf[end-1]
    while zf[end] < top - 1e-6
        Δ *= stretching
        push!(zf, min(zf[end] + Δ, top))
    end
    return zf
end

# A profile over the whole column as (faces, cell values): the LES profile on the LES cells and, above the
# LES top, `above` sampled on 50 m cells — the member's GCM column (`:θ`, `:qᵛ`), the LES top value held
# constant (`:hold`, for winds), or zero (`:zero`, for the LES's forcing tendencies).
function composite_profile(m::LESMember, les_values, les_zf, top, above)
    top > les_zf[end] + 1e-6 || return les_zf, collect(Float64, les_values)
    fine = collect(range(les_zf[end], top, length = max(2, round(Int, (top - les_zf[end]) / 50) + 1)))
    zc = (fine[1:end-1] .+ fine[2:end]) ./ 2
    upper = above == :zero ? zeros(length(zc)) :
            above == :hold ? fill(Float64(les_values[end]), length(zc)) :
            interpolate_profile(m.gcm.z, getproperty(m.gcm, above), zc)
    return vcat(les_zf, fine[2:end]), vcat(collect(Float64, les_values), upper)
end

# The LES's `qt` is vapor plus cloud liquid and its `ql` the cloud liquid; rain is separate. Kessler's fields
# in that partition, as interior arrays on the host.
function moisture_arrays(model)
    μ = model.microphysical_fields
    qᶜˡ = Array(interior(μ.qᶜˡ))
    return Array(interior(μ.qᵛ)) .+ qᶜˡ, qᶜˡ, Array(interior(μ.qʳ))
end

"""
Run the column ensemble: parameter set `i` (column of `params`, ordered as `parameter_names(space)`)
against member `j`, for every pair, on `architecture` (`CPU()` or `GPU()`). Radiation is `:interactive`
(RRTMGP every `radiation_interval`) or `:prescribed` (the LES's own hourly heating replayed). Returns the
time-mean θˡ, qᵗ (vapor + cloud liquid, as the LES's `qt`), qˡ (cloud liquid), qʳ (rain), u and v over the
target window as arrays `(N_ens, N_members, Nz)`, and the model.

The third return value times the run: `(; setup_seconds, integration_seconds, steps,
seconds_per_step)`, with the integration measured around `run!` alone.

None of these is a reliable cost per step on its own, and neither is differencing two runs. Setup
varies with contention for the device; `run!` carries its own one-offs (the first `update_state!`,
the first radiation call, late specialization) worth tens of seconds at production size; and a
short run is the first at its column shape, so it is warmed differently from a long one. Each of
those has produced a wrong answer here, twice a negative one. To measure cost per step, time a
block *inside one run* — `sample_callback` fires on the accumulation schedule, so it can stamp
`CUDA.synchronize(); time_ns()` at two iterations and difference them — over enough steps to
contain several radiation calls.

`sample_callback(model, t, active)`, if given, is called on every accumulation — the same instants that
enter the time means, with `active[j]` saying whether member `j`'s scored window is open — so a caller can
record closure diagnostics (diffusivities, mixing length, N², the TKE budget) averaged over exactly the
scored window. That is not the same as diagnosing the mean state: the stability functions and the
saturation switch are nonlinear, so the mean of the closure and the closure of the mean differ.
"""
function run_ensemble(problem::ColumnEnsembleProblem, params::AbstractMatrix;
                      space = space_of(size(params, 1)),
                      Δt = 1minute, architecture = CPU(), stop_time = nothing, verbose = false,
                      averaging_window = nothing,
                      subsidence_advection = UpwindBiased(order = 1),
                      radiation = :interactive,
                      radiation_interval = 10minutes,
                      upper_relaxation_rate = 1 / 600,
                      sample_callback = nothing)
    setup_start = time_ns()
    members = problem.members
    microphysics = DCMIP2016KesslerMicrophysics()   # the LES's warm-rain scheme, written for Tetens' saturation vapor pressure
    constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())
    device(x) = on_architecture(architecture, x)    # arrays read inside kernels (boundary-condition parameters) live on the device
    N_ens, N_mem = size(params, 2), length(members)
    zc, zf, les_zf = problem.zc, problem.zf, problem.les_zf
    Nz = length(zf) - 1
    FT = Float64
    top = zf[end]
    tall = top > problem.les_top + 1e-6
    radiation in (:prescribed, :interactive) || error("radiation must be :prescribed or :interactive")
    radiation == :interactive && !tall && @warn "Interactive radiation on a column that ends at the LES top: no atmosphere above $(top) m"
    # Profiles over the whole column, regridded conservatively onto the model cells
    onto_centers(v, m, above) = (src = composite_profile(m, v, les_zf, top, above); regrid_column(src[2], src[1], zf))

    grid = RectilinearGrid(architecture; size = ColumnEnsembleSize(Nz = Nz, ensemble = (N_ens, N_mem), Hz = 3),
                           z = zf, topology = (Flat, Flat, Bounded))

    # Per-column profiles broadcast over the parameter dimension: regrid each member once. Regridding is
    # not cheap — `onto_centers` builds two grids and two fields and runs a `regrid!` — so `f` must be
    # evaluated once per member and only then indexed, never inside the (i, j, k) comprehension.
    function column_array(f)
        profiles = [f(members[j]) for j in 1:N_mem]
        return FT[profiles[j][k] for i in 1:N_ens, j in 1:N_mem, k in 1:Nz]
    end

    p₀ = [m.p₀ for i in 1:N_ens, m in members]
    θ₀ = [m.θ₀ for i in 1:N_ens, m in members]
    # The reference state: adiabatic per column for a column within the LES; for a tall column each column's
    # reference follows its own initial potential temperature (LES below, GCM above) so the reference pressure
    # stays realistic through the stratosphere
    θᵣ = tall ? column_array(m -> onto_centers(m.initial.θ, m, :θ)) : θ₀
    reference_state = ReferenceState(grid, constants; surface_pressure = p₀, potential_temperature = θᵣ)
    dynamics = AnelasticDynamics(reference_state)

    closures = [closure_from(space, view(params, :, i); static_stability = problem.static_stability) for i in 1:N_ens, j in 1:N_mem]

    # The subsidence velocity, a point value at the faces, is interpolated (zero above the LES top)
    onto_faces(v, m) = [z ≤ les_zf[end] ? interpolate_profile(m.z, v, [z])[1] : 0.0 for z in zf]

    wˢ = ZFaceField(grid)
    wˢ_profiles = [onto_faces(members[j].subsidence, members[j]) for j in 1:N_mem]
    set!(wˢ, FT[wˢ_profiles[j][k] for i in 1:N_ens, j in 1:N_mem, k in 1:Nz+1])
    fill_halo_regions!(wˢ)
    subsidence = SubsidenceForcing(wˢ; advection = subsidence_advection)

    dTdt = CenterField(grid); set!(dTdt, column_array(m -> onto_centers(m.dTdt, m, :zero)))
    dqdt = CenterField(grid); set!(dqdt, column_array(m -> onto_centers(m.dqdt, m, :zero)))

    times = problem.times
    t_end = times[end]
    # Each member's hourly heating on the model cells, held at its last value beyond the end of its own
    # record. Only `radiation = :prescribed` reads it, and the series is one field per hour over the whole
    # ensemble grid — hundreds of megabytes at production ensemble sizes — so build it only when it is used.
    dTdt_rad = if radiation == :prescribed
        zf_les = zf[zf .≤ les_zf[end] + 1e-6]
        heating = [regrid_columns(reshape(permutedims(m.heating), size(m.heating, 2), 1, :), les_zf, zf_les) for m in members] # (Nt, 1, Nz_les) each
        Nz_les = size(heating[1], 3)
        fts = FieldTimeSeries{Center, Center, Center}(grid, times)
        for n in eachindex(times)
            set!(fts[n], FT[k ≤ Nz_les ? heating[j][min(n, length(members[j].t)), 1, k] : 0 for i in 1:N_ens, j in 1:N_mem, k in 1:Nz])
        end
        fts
    else
        nothing
    end

    # Relaxation targets: the GCM profiles the LES was relaxed toward (its initial profiles), continued by the
    # GCM column above the LES top (winds held at their LES-top value there)
    target(name, loc, above) = (f = loc(grid); set!(f, column_array(m -> onto_centers(getproperty(m.nudging, name), m, above))); fill_halo_regions!(f); f)
    uₙ, vₙ = target(:u, XFaceField, :hold), target(:v, YFaceField, :hold)
    θₙ, qₙ = target(:θ, CenterField, :θ), target(:qᵗ, CenterField, :qᵛ)
    ramp(z) = z < 3000 ? 0.0 : z > 3500 ? 1.0 : (1 - cos(π * (z - 3000) / 500)) / 2
    relax_u = Relaxation(rate = 1 / 6hours, target = uₙ)
    relax_v = Relaxation(rate = 1 / 6hours, target = vₙ)
    relax_θ = Relaxation(rate = 1 / 24hours, mask = ramp, target = θₙ)
    relax_q = Forcing(total_water_relaxation; discrete_form = true,
                     parameters = (; rate = 1 / 24hours, mask = ramp, target = qₙ, density = reference_state.density))
    # Above the LES top the column is not free: relax it strongly to the GCM column (a 200 m cosine onset)
    les_top = problem.les_top
    upper(z) = z < les_top ? 0.0 : z > les_top + 200 ? 1.0 : (1 - cos(π * (z - les_top) / 200)) / 2
    upper_relaxation(target) = Relaxation(; rate = upper_relaxation_rate, mask = upper, target)

    # Advect every Kessler moisture category with the large-scale velocity. The LES's
    # prescribed moistening and total-water relaxation are sources of vapor; the upper
    # extension relaxes vapor to the GCM humidity and removes condensate on the same timescale.
    qname = moisture_specific_name(microphysics)
    aloft(f, target) = tall ? (f..., upper_relaxation(target)) : f
    energy_forcing = radiation == :prescribed ? (Forcing(dTdt), Forcing(dTdt_rad)) : (Forcing(dTdt),)
    forcing = merge((u = aloft((subsidence, relax_u), uₙ),
                     v = aloft((subsidence, relax_v), vₙ),
                     θ = aloft((subsidence, relax_θ), θₙ),
                     E = energy_forcing,
                     qᶜˡ = tall ? (subsidence, upper_relaxation(0)) : subsidence,
                     qʳ = tall ? (subsidence, upper_relaxation(0)) : subsidence),
                    NamedTuple{(qname,)}((aloft((subsidence, Forcing(dqdt), relax_q), qₙ),)))

    # Interactive radiation: RRTMGP all-sky on every column with the LES protocol's fixed sun (the GCM's
    # insolation-weighted zenith angle and matching TOA flux), the LES's sea surface temperature, ocean albedo
    # 0.06 and emissivity 0.95, updated every `radiation_interval`
    radiative_transfer = if radiation == :interactive
        sst = Field{Center, Center, Nothing}(grid)
        set!(sst, FT[mean(members[j].surface.Tₛ) for i in 1:N_ens, j in 1:N_mem])
        coszen = FT[members[j].gcm.coszen for i in 1:N_ens, j in 1:N_mem]
        insolation = FT[members[j].gcm.insolation for i in 1:N_ens, j in 1:N_mem]
        all(isfinite, coszen) && all(isfinite, insolation) || error("The GCM column file lacks insolation/coszen; run scripts/solar_parameters.jl")
        RadiativeTransferModel(grid, AllSkyOptics(), constants; surface_temperature = sst, surface_albedo = 0.06, surface_emissivity = 0.95,
                               solar_position = FixedCosineZenith(coszen), solar_constant = insolation ./ coszen,
                               schedule = TimeInterval(radiation_interval))
    else
        nothing
    end

    # Surface fluxes: hourly series per member on the common axis, held at their last value beyond
    # the end of each member's record, linearly interpolated in time
    ρ₀ = Array(interior(reference_state.density))[1, :, 1]
    Nt = length(times)
    padded(v, n) = v[min(n, length(v))]
    series(f) = FT[f(members[j], n) for j in 1:N_mem, n in 1:Nt]
    parameters(F) = (; F = device(F), t₀ = FT(times[1]), Δt = FT(times[2] - times[1]), Nt)
    flux_bc(F) = FluxBoundaryCondition(prescribed_flux; discrete_form = true, parameters = parameters(F))
    # The surface TKE flux from the prescribed stress and virtual heat flux: u★³ = (uw² + vw²)^{3/4} and
    # wΔ³ = Δz Jᵇ, with the surface buoyancy flux Jᵇ = (g/θ₀) [shf / (ρ₀ cᵖᵈ) + 0.61 θ₀ lhf / (ρ₀ ℒᵛ)] (clipped at zero)
    g = constants.gravitational_acceleration
    cᵖᵈ = constants.dry_air.heat_capacity
    Δz₁ = zf[2] - zf[1]
    u★³ = FT[(padded(members[j].surface.uw, n)^2 + padded(members[j].surface.vw, n)^2)^(3/4) for j in 1:N_mem, n in 1:Nt]
    wΔ³ = FT[max(0, Δz₁ * g / members[j].θ₀ * (padded(members[j].surface.shf, n) / (ρ₀[j] * cᵖᵈ) +
                                                   0.61 * members[j].θ₀ * padded(members[j].surface.lhf, n) / (ρ₀[j] * members[j].surface.ℒᵛ)))
             for j in 1:N_mem, n in 1:Nt]
    tke_parameters = (; u★³ = device(u★³), wΔ³ = device(wΔ³), ρ₀ = device(FT.(ρ₀)),
                        Cᵂu★ = device(FT.(params[parameter_index(space, :Cᵂu★), :])), Cᵂʷ = device(FT.(params[parameter_index(space, :Cᵂʷ), :])),
                        t₀ = FT(times[1]), Δt = FT(times[2] - times[1]), Nt)
    tke_bc = FluxBoundaryCondition(surface_tke_flux; discrete_form = true, parameters = tke_parameters)

    ρqname = Symbol(:ρ, qname)
    boundary_conditions = merge((ρE = FieldBoundaryConditions(bottom = flux_bc(series((m, n) -> padded(m.surface.shf, n)))),
                           ρu = FieldBoundaryConditions(bottom = flux_bc(FT[ρ₀[j] * padded(members[j].surface.uw, n) for j in 1:N_mem, n in 1:Nt])),
                           ρv = FieldBoundaryConditions(bottom = flux_bc(FT[ρ₀[j] * padded(members[j].surface.vw, n) for j in 1:N_mem, n in 1:Nt])),
                           ρe = FieldBoundaryConditions(bottom = tke_bc)),
                          NamedTuple{(ρqname,)}((FieldBoundaryConditions(bottom = flux_bc(series((m, n) -> padded(m.surface.lhf, n) / m.surface.ℒᵛ))),)))

    model = AtmosphereModel(grid; dynamics, microphysics, closure = closures, forcing, boundary_conditions, advection = nothing,
                            thermodynamic_constants = constants, radiation = radiative_transfer)

    set!(model; θ = column_array(m -> onto_centers(m.initial.θ, m, :θ)),
                u = column_array(m -> onto_centers(m.initial.u, m, :hold)),
                v = column_array(m -> onto_centers(m.initial.v, m, :hold)),
                ρe = 1e-3)
    # Initial moisture: the LES's cloud water, and its total water less the cloud water as vapor
    qᵗ₀ = column_array(m -> onto_centers(m.initial.qᵗ, m, :qᵛ)); qˡ₀ = column_array(m -> onto_centers(m.initial.qˡ, m, :zero))
    set!(model; qᶜˡ = qˡ₀, qᵛ = max.(0, qᵗ₀ .- qˡ₀))
    set!(model.tracers.ρe, reference_state.density * model.tracers.ρe)

    stop = isnothing(stop_time) ? t_end : stop_time
    simulation = Simulation(model; Δt, stop_time = stop, verbose)

    # Time means over each member's own target window
    θˡ = model.formulation.potential_temperature
    u, v = model.velocities.u, model.velocities.v
    sums = NamedTuple{observable_variables}(Tuple(zeros(FT, N_ens, N_mem, Nz) for v in observable_variables))
    counts = zeros(Int, N_mem)
    windows = isnothing(averaging_window) ? [m.window for m in members] : [averaging_window for m in members]
    function accumulate!(sim)
        t = time(sim)
        active = [w[1] ≤ t ≤ w[2] for w in windows]
        any(active) || return nothing
        compute!(θˡ)
        θ_now = Array(interior(θˡ)); q_now, l_now, r_now = moisture_arrays(model)
        u_now = Array(interior(u)); v_now = Array(interior(v))
        for j in findall(active)
            sums.θˡ[:, j, :] .+= θ_now[:, j, :]; sums.qᵗ[:, j, :] .+= q_now[:, j, :]; sums.qˡ[:, j, :] .+= l_now[:, j, :]; sums.qʳ[:, j, :] .+= r_now[:, j, :]
            sums.u[:, j, :] .+= u_now[:, j, :]; sums.v[:, j, :] .+= v_now[:, j, :]
            counts[j] += 1
        end
        # Diagnostics are sampled here, on exactly the instants and the columns that enter the score,
        # so a time-mean diagnostic is the mean of the closure over the scored window rather than the
        # closure evaluated on the mean state — a different quantity for anything nonlinear, which
        # every stability function and the saturation switch are. `nothing` costs a branch per sample.
        isnothing(sample_callback) || sample_callback(model, t, active)
        return nothing
    end
    add_callback!(simulation, accumulate!, TimeInterval(10minutes))

    if verbose
        add_callback!(simulation, sim -> @info(@sprintf("t = %.1f h, wall %.0f s", time(sim) / 3600, sim.run_wall_time)), TimeInterval(6hours))
    end

    # Timing is measured here rather than by differencing two whole runs of different length: setup
    # is tens of seconds and varies with GPU contention, so a difference of two wall times can be
    # swamped by it — badly enough to return a negative cost per step.
    setup_seconds = (time_ns() - setup_start) / 1e9
    integration_seconds = @elapsed run!(simulation)
    steps = model.clock.iteration

    # Every column must have contributed at least one sample, or its "time mean" is a zero profile that
    # looks like a finite observation. This is what a `stop_time` short of a member's target window does.
    if any(iszero, counts)
        empty = findall(iszero, counts)
        error("Members $([(members[j].site, members[j].month) for j in empty]) contributed no samples to " *
              "their time mean: the run stopped at t = $(time(simulation)) s and their averaging windows are " *
              "$([windows[j] for j in empty]). Pass `averaging_window` (and a `stop_time` that reaches it) " *
              "when running for less than a member's own target window.")
    end

    n = reshape(counts, 1, N_mem, 1)
    timing = (; setup_seconds, integration_seconds, steps,
                seconds_per_step = integration_seconds / max(steps, 1))
    return map(x -> x ./ n, sums), model, timing
end

"""
The observation vector of one column: for each of `problem.variables` in turn, the time-mean profile as means
over the observation cells in observation units, from `profiles` (a named tuple of profiles on the cells with
faces `from` — the model's by default, or the LES's).
"""
function observation_vector(problem::ColumnEnsembleProblem, profiles::NamedTuple; from = problem.zf)
    to = problem.observation_zf
    return vcat([observation_scales[v] .* regrid_column(profiles[v], from, to) for v in problem.variables]...)
end

function forward_map(problem::ColumnEnsembleProblem, params::AbstractMatrix; kw...)
    means, _ = run_ensemble(problem, params; kw...)
    to = problem.observation_zf
    observed = NamedTuple{problem.variables}(Tuple(observation_scales[v] .* regrid_columns(means[v], problem.zf, to) for v in problem.variables))
    N_ens, N_mem = size(means.θˡ, 1), size(means.θˡ, 2)
    column(i, j) = vcat([observed[v][i, j, :] for v in problem.variables]...)
    G = hcat([vcat([column(i, j) for j in 1:N_mem]...) for i in 1:N_ens]...)
    return G, means
end

"""
The LES targets as the observation vector, and a diagonal noise covariance with, on every observation
cell, the noise `σ[v]` of each observed variable (`default_observation_noise`: 0.25 K, 0.25 g kg⁻¹ of qᵗ,
0.1 g kg⁻¹ of qˡ, 0.5 m s⁻¹).
"""
function observations(problem::ColumnEnsembleProblem; σ = default_observation_noise)
    y = vcat([observation_vector(problem, m.targets; from = problem.les_zf) for m in problem.members]...)
    nk = length(problem.observation_zf) - 1
    σ_member = vcat([fill(Float64(σ[v]), nk) for v in problem.variables]...)
    return y, Diagonal(vcat([σ_member for m in problem.members]...) .^ 2)
end

#####
##### Scores
#####

"""
RMSE over the observation cells of one column's time means (`profiles`, a named tuple with θˡ, qᵗ, qˡ, u, v) against
its member: θˡ in K, qᵗ and qˡ in g kg⁻¹, u and v in m s⁻¹, and `wind = √(uᵉʳʳ² + vᵉʳʳ²)`.
"""
function rmse(problem::ColumnEnsembleProblem, profiles::NamedTuple, m::LESMember)
    to = problem.observation_zf
    err(v) = observation_scales[v] * sqrt(mean((regrid_column(profiles[v], problem.zf, to) .- regrid_column(m.targets[v], problem.les_zf, to)) .^ 2))
    errors = NamedTuple{observable_variables}(Tuple(err(v) for v in observable_variables))
    return merge(errors, (wind = sqrt(errors.u^2 + errors.v^2),))
end

"""
Evaluate one or more parameter sets (columns of `params`) on a problem, or on `members` on the LES grid:
per-member RMSEs `(N_sets, N_members)` from one column-ensemble run, and the time means.
"""
function evaluate(params::AbstractMatrix, problem::ColumnEnsembleProblem; kw...)
    means, _ = run_ensemble(problem, params; kw...)
    members = problem.members
    N_ens, N_mem = size(means.θˡ, 1), size(means.θˡ, 2)
    column(i, j) = NamedTuple{observable_variables}(Tuple(means[v][i, j, :] for v in observable_variables))
    scores = [rmse(problem, column(i, j), members[j]) for i in 1:N_ens, j in 1:N_mem]
    return scores, means
end

evaluate(params::AbstractMatrix, members::Vector{LESMember}; kw...) = evaluate(params, ColumnEnsembleProblem(members); kw...)
evaluate(p::NamedTuple, x; kw...) = evaluate(reshape(collect(Float64, p), :, 1), x; kw...)

function rmse_table(io::IO, scores::AbstractMatrix, labels; members = nothing)
    for i in axes(scores, 1)
        med(v) = median([s[v] for s in scores[i, :]])
        θ = [s.θˡ for s in scores[i, :]]
        @printf(io, "%-28s median RMSE θˡ %.2f K  qᵗ %.2f g/kg  qˡ %.3f g/kg  wind %.2f m/s   (θˡ mean %.2f, worst %.2f K)\n", labels[i], med(:θˡ), med(:qᵗ), med(:qˡ), med(:wind), mean(θ), maximum(θ))
    end
end
rmse_table(scores, labels; kw...) = rmse_table(stdout, scores, labels; kw...)

include("grids.jl")
include("multiresolution.jl")
include("inversion.jl")

end # module
