#####
##### Large-scale forcings that reproduce the LASSO-ENA SAM protocol
##### (lasso_sam_sbm, branch lasso_ena_noice, commit 12d0244):
#####
#####   * `TimeVaryingGeostrophicForcing` — Coriolis/pressure-gradient term with the
#####     time-height geostrophic wind `ug/vg` (coriolis.f90 uses ug0/vg0)
#####   * `MeanProfileNudging` — domain-mean wind relaxed to `uls/vls` (nudging.f90 uses ul0/vl0)
#####   * `LargeScaleVerticalAdvection` — full-field upwind −wls ∂zϕ (subsidence.f90)
#####   * `LargeScaleEnergyForcing` / `LargeScaleMoistureForcing` — the horizontal
#####     advective tendencies tls/qls as the energy source cᵖᵐ tls of the liquid-ice
#####     potential-temperature formulation and a vapor source (forcing.f90 adds tls to SAM's
#####     `t` and qls to vapor)
#####   * `SAMSponge` — the upper-30% geometric Rayleigh damping (damping.f90)
#####
##### All kernels return *specific* tendencies and are meant to be supplied under the
##### specific prognostic keys (`u`, `v`, `w`, `θ`, the energy key `E`, `qᵛ`, `qʳ`, ...); Breeze wraps them in
##### `SpecificForcing` and multiplies by the density at kernel time.
#####

using Adapt: Adapt, adapt
using Oceananigans: Field, Average, compute!, set!
using Oceananigans.Grids: Center, Face, XDirection, YDirection, znodes, znode
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶜᶜᶜ
using Oceananigans.Units: Time
using Oceananigans.Utils: prettysummary
using Breeze.AtmosphereModels: AtmosphereModels, grid_moisture_fractions
using Breeze.Thermodynamics: mixture_heat_capacity, dry_air_mass_fraction

# Read a horizontally uniform profile time series at level k and time t.
@inline profile_value(fts, k, t) = @inbounds fts[1, 1, k, Time(t)]

# Look a model field up by name at kernel time; `name` is a `Val` so the lookup is static.
@inline field_by_name(fields, ::Val{name}) where name = getproperty(fields, name)

#####
##### Time-varying geostrophic forcing
#####

"""
    TimeVaryingGeostrophicForcing

Specific momentum tendency from the large-scale pressure gradient expressed through the
geostrophic wind: `-f vᵍ(z, t)` for `u` and `+f uᵍ(z, t)` for `v`. Unlike Breeze's
`GeostrophicForcing`, the profile is a `FieldTimeSeries` read at the current time.
Construct the pair with [`time_varying_geostrophic_forcings`](@ref).
"""
struct TimeVaryingGeostrophicForcing{D, V, F}
    direction :: D
    geostrophic_velocity :: V  # FieldTimeSeries of the *other* component
    coriolis_parameter :: F
end

Adapt.adapt_structure(to, f::TimeVaryingGeostrophicForcing) =
    TimeVaryingGeostrophicForcing(f.direction, adapt(to, f.geostrophic_velocity), adapt(to, f.coriolis_parameter))

Base.summary(f::TimeVaryingGeostrophicForcing) =
    string("TimeVaryingGeostrophicForcing{", nameof(typeof(f.direction)), "}",
           isnothing(f.coriolis_parameter) ? "" : "(f=$(prettysummary(f.coriolis_parameter)))")
Base.show(io::IO, f::TimeVaryingGeostrophicForcing) = print(io, summary(f))

"""
    time_varying_geostrophic_forcings(uᵍ, vᵍ)

Return `(; u, v)` geostrophic forcings from the profile time series `uᵍ(z, t)`, `vᵍ(z, t)`.
The Coriolis parameter is taken from the model's `coriolis` at materialization.
"""
function time_varying_geostrophic_forcings(uᵍ, vᵍ)
    Fu = TimeVaryingGeostrophicForcing(XDirection(), vᵍ, nothing)
    Fv = TimeVaryingGeostrophicForcing(YDirection(), uᵍ, nothing)
    return (; u=Fu, v=Fv)
end

@inline function (f::TimeVaryingGeostrophicForcing{<:XDirection})(i, j, k, grid, clock, fields)
    vᵍ = profile_value(f.geostrophic_velocity, k, clock.time)
    return - f.coriolis_parameter * vᵍ
end

@inline function (f::TimeVaryingGeostrophicForcing{<:YDirection})(i, j, k, grid, clock, fields)
    uᵍ = profile_value(f.geostrophic_velocity, k, clock.time)
    return + f.coriolis_parameter * uᵍ
end

function AtmosphereModels.materialize_atmosphere_model_forcing(f::TimeVaryingGeostrophicForcing,
                                                               field, name, model_field_names, context::NamedTuple)
    FT = eltype(field.grid)
    fᶜ = convert(FT, context.coriolis.f)
    return TimeVaryingGeostrophicForcing(f.direction, f.geostrophic_velocity, fᶜ)
end

#####
##### Domain-mean profile nudging
#####

"""
    MeanProfileNudging(target; timescale, vertical_weight=nothing)

Relax the *horizontal mean* of a prognostic variable toward the profile time series
`target(z, t)` with rate `1 / timescale`, applying the same tendency to every column:

    Fϕ(z, t) = - 𝒲(z) (⟨ϕ⟩(z, t) - target(z, t)) / timescale

This is the LASSO `n0` wind nudging (`nudging.f90`: `dudt -= (u0 - ul0) / tauls`), and
deliberately not a pointwise `Relaxation`, which would damp resolved eddies.

`vertical_weight` 𝒲(z) is `nothing` (𝒲 = 1, the full column), a vector of weights on the
cell centers, or a function `𝒲(z)` evaluated at the cell centers when the forcing is
materialized (see [`nudging_height_ramp`](@ref) for the free-troposphere-only nudging of the
Covert sonde-forcing runs).
"""
struct MeanProfileNudging{T, A, R, W}
    target :: T
    averaged_field :: A
    rate :: R
    vertical_weight :: W
end

MeanProfileNudging(target; timescale, vertical_weight=nothing) =
    MeanProfileNudging(target, nothing, 1 / timescale, vertical_weight)

Adapt.adapt_structure(to, f::MeanProfileNudging) =
    MeanProfileNudging(adapt(to, f.target), adapt(to, f.averaged_field), adapt(to, f.rate), adapt(to, f.vertical_weight))

Base.summary(f::MeanProfileNudging) = string("MeanProfileNudging(timescale=", prettysummary(1 / f.rate), " s",
                                             isnothing(f.vertical_weight) ? "" : ", height-weighted", ")")
Base.show(io::IO, f::MeanProfileNudging) = print(io, summary(f))

@inline nudging_weight(::Nothing, k) = 1
@inline nudging_weight(𝒲, k) = @inbounds 𝒲[1, 1, k]

@inline function (f::MeanProfileNudging)(i, j, k, grid, clock, fields)
    ϕ̄ = @inbounds f.averaged_field[1, 1, k]
    ϕᵗ = profile_value(f.target, k, clock.time)
    𝒲 = nudging_weight(f.vertical_weight, k)
    return - 𝒲 * f.rate * (ϕ̄ - ϕᵗ)
end

"""
    nudging_height_ramp(z₁, z₂)

The vertical weight `𝒲(z)` of a nudging confined to heights above `z₁`: 0 below `z₁`, 1 above
`z₂`, and `sin²(π/2 (z - z₁)/(z₂ - z₁))` in between (continuous with a continuous derivative).
"""
function nudging_height_ramp(z₁, z₂)
    z₂ > z₁ || throw(ArgumentError("nudging_height_ramp needs z₂ > z₁, got ($z₁, $z₂)"))
    return z -> sin(π / 2 * clamp((z - z₁) / (z₂ - z₁), 0, 1))^2
end

materialize_vertical_weight(::Nothing, grid) = nothing

function materialize_vertical_weight(𝒲::Function, grid)
    z = Array(znodes(grid, Center()))
    return materialize_vertical_weight(𝒲.(z), grid)
end

function materialize_vertical_weight(𝒲::AbstractVector, grid)
    Nz = size(grid, 3)
    length(𝒲) == Nz || throw(ArgumentError("vertical_weight has $(length(𝒲)) values, the grid has $Nz levels"))
    weight = Field{Nothing, Nothing, Center}(grid)
    set!(weight, reshape(eltype(grid).(𝒲), 1, 1, Nz))
    return weight
end

function AtmosphereModels.materialize_atmosphere_model_forcing(f::MeanProfileNudging,
                                                               field, name, model_field_names, context::NamedTuple)
    startswith(string(name), "ρ") &&
        throw(ArgumentError("MeanProfileNudging returns a specific tendency; supply it under the specific key (e.g. `u`)"))
    specific_field = context.specific_fields[name]
    averaged_field = Field(Average(specific_field, dims=(1, 2)))
    FT = eltype(field.grid)
    weight = materialize_vertical_weight(f.vertical_weight, field.grid)
    return MeanProfileNudging(f.target, averaged_field, convert(FT, f.rate), weight)
end

function AtmosphereModels.compute_forcing!(f::MeanProfileNudging)
    compute!(f.averaged_field)
    return nothing
end

#####
##### Full-field large-scale vertical advection (SAM subsidence.f90)
#####

"""
    LargeScaleVerticalAdvection(wls)

Specific tendency `-wls(z, t) ∂z ϕ` applied to the *instantaneous three-dimensional* field
`ϕ`, with the first-order upwind vertical derivative of `subsidence.f90`:

    wls ≥ 0 :  ∂zϕ ≈ (ϕₖ - ϕₖ₋₁) / (zₖ - zₖ₋₁)
    wls < 0 :  ∂zϕ ≈ (ϕₖ₊₁ - ϕₖ) / (zₖ₊₁ - zₖ)

and no tendency in the bottom and top cells (SAM loops `k = 2, nzm-1`). `wls` is a
profile time series on cell centers. Supply the same object under every specific key it
should act on (`u`, `v`, `s`, `qᵛ`, and the microphysical species); the field is looked up
by name at kernel time. This differs from Breeze's `SubsidenceForcing`, which advects the
horizontal-mean profile.
"""
struct LargeScaleVerticalAdvection{N, W}
    name :: N
    velocity :: W
end

LargeScaleVerticalAdvection(wls) = LargeScaleVerticalAdvection(nothing, wls)

Adapt.adapt_structure(to, f::LargeScaleVerticalAdvection) =
    LargeScaleVerticalAdvection(f.name, adapt(to, f.velocity))

Base.summary(f::LargeScaleVerticalAdvection) =
    string("LargeScaleVerticalAdvection", isnothing(f.name) ? "" : "($(f.name))")
Base.show(io::IO, f::LargeScaleVerticalAdvection) = print(io, summary(f))

@inline function (f::LargeScaleVerticalAdvection)(i, j, k, grid, clock, fields)
    ϕ = field_by_name(fields, f.name)
    w = profile_value(f.velocity, k, clock.time)
    Nz = size(grid, 3)
    k⁻ = max(k - 1, 1)
    k⁺ = min(k + 1, Nz)
    @inbounds begin
        ϕᵏ = ϕ[i, j, k]
        ϕ⁻ = ϕ[i, j, k⁻]
        ϕ⁺ = ϕ[i, j, k⁺]
    end
    ∂zϕ⁻ = (ϕᵏ - ϕ⁻) / Δzᶜᶜᶠ(i, j, k, grid)      # between centers k-1 and k
    ∂zϕ⁺ = (ϕ⁺ - ϕᵏ) / Δzᶜᶜᶠ(i, j, k + 1, grid)  # between centers k and k+1
    ∂zϕ = ifelse(w ≥ 0, ∂zϕ⁻, ∂zϕ⁺)
    interior = (k > 1) & (k < Nz)
    return ifelse(interior, - w * ∂zϕ, zero(w))
end

function AtmosphereModels.materialize_atmosphere_model_forcing(f::LargeScaleVerticalAdvection,
                                                               field, name, model_field_names, context::NamedTuple)
    startswith(string(name), "ρ") &&
        throw(ArgumentError("LargeScaleVerticalAdvection returns a specific tendency; supply it under the specific key"))
    name ∈ model_field_names ||
        throw(ArgumentError("LargeScaleVerticalAdvection needs the specific field `$name` among the model fields $(model_field_names)"))
    return LargeScaleVerticalAdvection(Val(name), f.velocity)
end

#####
##### Horizontal advective tendencies: tls (K/s) and qls (kg/kg/s)
#####

"""
    large_scale_thermodynamic_forcings(tls, qls; microphysics, thermodynamic_constants, moisture_name,
                                       moisture_basis=:mixing_ratio)

Return `(; E = energy_forcing, <moisture_name> = moisture_forcing)` implementing SAM's
`forcing.f90`, which adds `tls` to the temperature-like variable `t` (SAM: `t = T + gz/cp
- (Lv/cp) qˡ - (Ls/cp) qⁱ` with constant `cp = 1004`) and `qls` to the vapor. Supply the
energy forcing under the energy key `E` of a liquid-ice potential-temperature model (the only
formulation BreezeLab uses) and the moisture forcing under `moisture_name`.

SAM's vapor is a mixing ratio (per kg dry air) while Breeze carries mass fractions: SAM
imposes `drᵛ/dt = qls`. **Approximation:** SAM holds the *dry-basis* condensate ratios fixed
while the vapor source acts, whereas this forcing leaves Breeze's condensate mass fractions
untouched; with `qᵛ + qᵈ = 1 - qᶜ` fixed the vapor source maps to

    dqᵛ/dt = qᵈ² / (1 - qᶜ) R,   qᶜ = qˡ + qⁱ

(`moisture_basis = :mixing_ratio`, the default; `:mass_fraction` applies `qls` verbatim).
In clear air the two conventions coincide; in cloudy cells they differ at O(qᶜ) ≈ 0.1 %.

**Physical-temperature invariant.** Over a forcing-only step the state must satisfy
`dT/dt = tls` and `dqᵛ/dt` as above at fixed condensate and height. Breeze's potential-temperature
formulation converts an energy source `F` into `dθ/dt = F / (cᵖᵐ Π)`, i.e. `cᵖᵐ dT/dt = F`, and a
vapor source at fixed θ leaves `T` unchanged (up to the composition dependence of the Exner
function, which vanishes at the standard pressure), so the energy forcing is `F = cᵖᵐ(q) tls`
and the vapor source needs no energy counterpart. `test/runtests.jl` checks both in
forcing-only steps.
"""
function large_scale_thermodynamic_forcings(tls, qls; microphysics, thermodynamic_constants, moisture_name,
                                            moisture_basis=:mixing_ratio)
    basis = moisture_basis === :mixing_ratio ? Val(:mixing_ratio) :
            moisture_basis === :mass_fraction ? Val(:mass_fraction) :
            throw(ArgumentError("moisture_basis must be :mixing_ratio or :mass_fraction"))
    energy = LargeScaleEnergyForcing(tls, microphysics, thermodynamic_constants, nothing, Val(moisture_name))
    moisture = LargeScaleMoistureForcing(qls, microphysics, nothing, Val(moisture_name), basis)
    return NamedTuple{(:E, moisture_name)}((energy, moisture))
end

# Vapor mass-fraction rate implied by a SAM vapor mixing-ratio source R with the condensate
# mass fractions held fixed (qᵛ + qᵈ = 1 - qᶜ): dqᵛ/dt = qᵈ² / (1 - qᶜ) R.
@inline function moisture_rates(::Val{:mixing_ratio}, q, R)
    qᵈ = dry_air_mass_fraction(q)
    qᶜ = q.liquid + q.ice
    return (; vapor = qᵈ^2 / (1 - qᶜ) * R)
end
@inline moisture_rates(::Val{:mass_fraction}, q, R) = (; vapor = R)

"""
    LargeScaleEnergyForcing

The heating `cᵖᵐ(q) tls` [W kg⁻¹] of a large-scale temperature tendency `tls(z, t)`, with the
mixture heat capacity of the local moisture state; supplied under the energy key `E`.
"""
struct LargeScaleEnergyForcing{T, M, C, D, N}
    tls :: T
    microphysics :: M
    thermodynamic_constants :: C
    density :: D
    moisture_name :: N
end

Adapt.adapt_structure(to, f::LargeScaleEnergyForcing) =
    LargeScaleEnergyForcing(adapt(to, f.tls), adapt(to, f.microphysics), adapt(to, f.thermodynamic_constants),
                            adapt(to, f.density), f.moisture_name)

Base.summary(f::LargeScaleEnergyForcing) = "LargeScaleEnergyForcing(cᵖᵐ tls)"
Base.show(io::IO, f::LargeScaleEnergyForcing) = print(io, summary(f))

@inline function (f::LargeScaleEnergyForcing)(i, j, k, grid, clock, fields)
    tls = profile_value(f.tls, k, clock.time)
    ρ = @inbounds f.density[i, j, k]
    qᵛᵉ = @inbounds field_by_name(fields, f.moisture_name)[i, j, k]
    q = grid_moisture_fractions(i, j, k, grid, f.microphysics, ρ, qᵛᵉ, fields)
    cᵖᵐ = mixture_heat_capacity(q, f.thermodynamic_constants)
    return cᵖᵐ * tls
end

function AtmosphereModels.materialize_atmosphere_model_forcing(f::LargeScaleEnergyForcing,
                                                               field, name, model_field_names, context::NamedTuple)
    name === :E || throw(ArgumentError("LargeScaleEnergyForcing must be supplied under the energy key `E`, got $name"))
    # The scheme's lookup tables must live on the device, as Breeze does for model.microphysics
    microphysics = on_architecture(architecture(field.grid), f.microphysics)
    return LargeScaleEnergyForcing(f.tls, microphysics, f.thermodynamic_constants, context.total_density, f.moisture_name)
end

struct LargeScaleMoistureForcing{Q, M, D, N, B}
    qls :: Q
    microphysics :: M
    density :: D
    moisture_name :: N
    moisture_basis :: B
end

Adapt.adapt_structure(to, f::LargeScaleMoistureForcing) =
    LargeScaleMoistureForcing(adapt(to, f.qls), adapt(to, f.microphysics), adapt(to, f.density), f.moisture_name, f.moisture_basis)
Base.summary(f::LargeScaleMoistureForcing) = string("LargeScaleMoistureForcing(qls, ", f.moisture_basis, ")")
Base.show(io::IO, f::LargeScaleMoistureForcing) = print(io, summary(f))

@inline function (f::LargeScaleMoistureForcing)(i, j, k, grid, clock, fields)
    R = profile_value(f.qls, k, clock.time)
    ρ = @inbounds f.density[i, j, k]
    qᵛᵉ = @inbounds field_by_name(fields, f.moisture_name)[i, j, k]
    q = grid_moisture_fractions(i, j, k, grid, f.microphysics, ρ, qᵛᵉ, fields)
    return moisture_rates(f.moisture_basis, q, R).vapor
end

function AtmosphereModels.materialize_atmosphere_model_forcing(f::LargeScaleMoistureForcing,
                                                               field, name, model_field_names, context::NamedTuple)
    microphysics = on_architecture(architecture(field.grid), f.microphysics)
    return LargeScaleMoistureForcing(f.qls, microphysics, context.total_density, f.moisture_name, f.moisture_basis)
end

#####
##### SAM upper sponge (damping.f90)
#####

"""
    SAMSponge(; damping_depth_fraction=0.3, minimum_timescale=60, maximum_timescale=1800)

Rayleigh damping over the upper `damping_depth_fraction` of the domain (measured with the
top cell-center height, as SAM does) with a timescale that varies geometrically from
`maximum_timescale` at the sponge base to `minimum_timescale` at the top:

    τ(z) = τmin (τmax / τmin)^((ztop - z) / (ztop - zbase))

`u` and `v` are damped toward their *horizontal means* and `w` toward zero
(`damping.f90`: `dudt -= (u - u0)/τ`, `dwdt -= w/τ`). Supply the same object under `u`,
`v` and `w`.
"""
struct SAMSponge{N, R, A, P}
    name :: N
    rate :: R              # Field{Nothing, Nothing, Center/Face} with 1/τ(z)
    averaged_field :: A    # horizontal mean of the damped field (u, v); nothing for w
    parameters :: P        # (; damping_depth_fraction, minimum_timescale, maximum_timescale)
end

SAMSponge(; damping_depth_fraction=0.3, minimum_timescale=60, maximum_timescale=1800) =
    SAMSponge(nothing, nothing, nothing, (; damping_depth_fraction, minimum_timescale, maximum_timescale))

Adapt.adapt_structure(to, f::SAMSponge) =
    SAMSponge(f.name, adapt(to, f.rate), adapt(to, f.averaged_field), nothing)

Base.summary(f::SAMSponge) = string("SAMSponge", isnothing(f.name) ? "" : "($(f.name))")
Base.show(io::IO, f::SAMSponge) = print(io, summary(f))

"""
    sam_sponge_rates(z_centers, z_query; damping_depth_fraction, minimum_timescale, maximum_timescale)

Inverse damping timescale at heights `z_query` for the SAM sponge defined on the cell
centers `z_centers` (levels with `ztop - z < damping_depth_fraction * ztop` are inside the
sponge; the level just below them carries `1 / maximum_timescale`).
"""
function sam_sponge_rates(z_centers, z_query; damping_depth_fraction=0.3, minimum_timescale=60, maximum_timescale=1800)
    Nz = length(z_centers)
    z_top = z_centers[Nz]
    n_damp = 0
    for k in Nz:-1:1
        if z_top - z_centers[k] < damping_depth_fraction * z_top
            n_damp = Nz - k + 1
        end
    end
    k_base = max(Nz - n_damp, 1)
    z_base = z_centers[k_base]
    rates = zeros(length(z_query))
    for (n, z) in enumerate(z_query)
        if z ≥ z_base
            ξ = clamp((z_top - z) / (z_top - z_base), 0, 1)
            τ = minimum_timescale * (maximum_timescale / minimum_timescale)^ξ
            rates[n] = 1 / τ
        end
    end
    return rates
end

@inline function (f::SAMSponge)(i, j, k, grid, clock, fields)
    ϕ = @inbounds field_by_name(fields, f.name)[i, j, k]
    ϕ̄ = sponge_reference(f.averaged_field, k)
    r = @inbounds f.rate[1, 1, k]
    return - r * (ϕ - ϕ̄)
end

@inline sponge_reference(::Nothing, k) = 0
@inline sponge_reference(ϕ̄, k) = @inbounds ϕ̄[1, 1, k]

function AtmosphereModels.materialize_atmosphere_model_forcing(f::SAMSponge,
                                                               field, name, model_field_names, context::NamedTuple)
    name ∈ (:u, :v, :w) || throw(ArgumentError("SAMSponge acts on `u`, `v`, `w`; got $name"))
    grid = field.grid
    p = f.parameters
    z_centers = Array(znodes(grid, Center()))
    center_rates = sam_sponge_rates(z_centers, z_centers; p.damping_depth_fraction, p.minimum_timescale, p.maximum_timescale)
    if name === :w
        # damping.f90 applies the cell-k rate τ(z(k)) to w(i, j, k), the face below center k;
        # the top face (w = 0) repeats the top cell's rate.
        rate = Field{Nothing, Nothing, Face}(grid)
        rates = vcat(center_rates, center_rates[end])
        averaged_field = nothing
    else
        rate = Field{Nothing, Nothing, Center}(grid)
        rates = center_rates
        averaged_field = Field(Average(context.specific_fields[name], dims=(1, 2)))
    end
    set!(rate, reshape(rates, 1, 1, length(rates)))
    return SAMSponge(Val(name), rate, averaged_field, nothing)
end

AtmosphereModels.compute_forcing!(f::SAMSponge) = compute_sponge_average!(f.averaged_field)
compute_sponge_average!(::Nothing) = nothing
compute_sponge_average!(ϕ̄) = (compute!(ϕ̄); nothing)

#####
##### SAM upper-boundary relaxation (upperbound.f90 with dolargescale = .true.)
#####

"""
    upper_boundary_relaxation_forcings(T_target, q_target; microphysics, thermodynamic_constants,
                                       moisture_name, timescale=3600, levels=2)

SAM's `upperbound.f90` (`doupperbound = .true.` with `dolargescale`): in the top `levels`
scalar levels, `t` is relaxed toward `tg0 + gamaz` and the vapor toward `qg0` with
`tau_nudging = 3600 s`, where `tg0(z, t)` and `qg0(z, t)` are the observed sounding
(`snd`) interpolated in time. With constant `cp` this imposes `dT/dt = -(T - Tg0)/τ` and
`dqᵛ/dt = -(qᵛ - qg0)/τ`. In the potential-temperature formulation the energy forcing
`F = -cᵖᵐ (T - Tg0)/τ` gives that temperature response (`cᵖᵐ dT = F`; the simultaneous vapor
relaxation at fixed θ leaves `T` unchanged). Returns `(; E, <moisture_name>)`. This is
distinct from the momentum sponge.
"""
function upper_boundary_relaxation_forcings(T_target, q_target; microphysics, thermodynamic_constants,
                                            moisture_name, timescale=3600, levels=2)
    energy = UpperBoundaryEnergyRelaxation(T_target, microphysics, thermodynamic_constants, nothing,
                                           Val(moisture_name), 1 / timescale, levels)
    moisture = UpperBoundaryMoistureRelaxation(q_target, Val(moisture_name), 1 / timescale, levels)
    return NamedTuple{(:E, moisture_name)}((energy, moisture))
end

struct UpperBoundaryEnergyRelaxation{T, M, C, D, N, R}
    target :: T             # Tg0(z, t)
    microphysics :: M
    thermodynamic_constants :: C
    density :: D
    moisture_name :: N
    rate :: R
    levels :: Int
end

Adapt.adapt_structure(to, f::UpperBoundaryEnergyRelaxation) =
    UpperBoundaryEnergyRelaxation(adapt(to, f.target), adapt(to, f.microphysics), adapt(to, f.thermodynamic_constants),
                                  adapt(to, f.density), f.moisture_name, f.rate, f.levels)

Base.summary(f::UpperBoundaryEnergyRelaxation) = string("UpperBoundaryEnergyRelaxation(top ", f.levels, " levels, τ=", 1 / f.rate, " s)")
Base.show(io::IO, f::UpperBoundaryEnergyRelaxation) = print(io, summary(f))

@inline function (f::UpperBoundaryEnergyRelaxation)(i, j, k, grid, clock, fields)
    Nz = size(grid, 3)
    active = k > Nz - f.levels
    Tᵗ = profile_value(f.target, k, clock.time)
    ρ = @inbounds f.density[i, j, k]
    qᵛᵉ = @inbounds field_by_name(fields, f.moisture_name)[i, j, k]
    T = @inbounds fields.T[i, j, k]
    q = grid_moisture_fractions(i, j, k, grid, f.microphysics, ρ, qᵛᵉ, fields)
    cᵖᵐ = mixture_heat_capacity(q, f.thermodynamic_constants)
    dTdt = - f.rate * (T - Tᵗ)
    return ifelse(active, cᵖᵐ * dTdt, zero(T))
end

function AtmosphereModels.materialize_atmosphere_model_forcing(f::UpperBoundaryEnergyRelaxation,
                                                               field, name, model_field_names, context::NamedTuple)
    name === :E || throw(ArgumentError("UpperBoundaryEnergyRelaxation must be supplied under the energy key `E`, got $name"))
    FT = eltype(field.grid)
    microphysics = on_architecture(architecture(field.grid), f.microphysics)
    return UpperBoundaryEnergyRelaxation(f.target, microphysics, f.thermodynamic_constants,
                                         context.total_density, f.moisture_name, convert(FT, f.rate), f.levels)
end

struct UpperBoundaryMoistureRelaxation{T, N, R}
    target :: T
    moisture_name :: N
    rate :: R
    levels :: Int
end

Adapt.adapt_structure(to, f::UpperBoundaryMoistureRelaxation) =
    UpperBoundaryMoistureRelaxation(adapt(to, f.target), f.moisture_name, f.rate, f.levels)

Base.summary(f::UpperBoundaryMoistureRelaxation) = string("UpperBoundaryMoistureRelaxation(top ", f.levels, " levels, τ=", 1 / f.rate, " s)")
Base.show(io::IO, f::UpperBoundaryMoistureRelaxation) = print(io, summary(f))

@inline function (f::UpperBoundaryMoistureRelaxation)(i, j, k, grid, clock, fields)
    Nz = size(grid, 3)
    active = k > Nz - f.levels
    qᵗ = profile_value(f.target, k, clock.time)
    q = @inbounds field_by_name(fields, f.moisture_name)[i, j, k]
    return ifelse(active, - f.rate * (q - qᵗ), zero(q))
end

function AtmosphereModels.materialize_atmosphere_model_forcing(f::UpperBoundaryMoistureRelaxation,
                                                               field, name, model_field_names, context::NamedTuple)
    FT = eltype(field.grid)
    return UpperBoundaryMoistureRelaxation(f.target, f.moisture_name, convert(FT, f.rate), f.levels)
end

#####
##### SBM-style aerosol replenishment for the prognostic P3 aerosol reservoir
#####

"""
    AerosolReplenishment(n_initial; timescale=60)

**Labelled approximation / sensitivity.** The LASSO HUJI-SBM (`diagCCN = .true.`)
rebuilds its CCN spectrum from the initial spectrum minus the current droplet number at
every microphysics call, so the total aerosol + droplet number never falls below the
initial setting. Breeze's prognostic P3 aerosol reservoir only depletes. This forcing on
`nᵃ` restores the deficit by relaxation,

    F_nᵃ = max(0, n_initial - (nᵃ + nᶜˡ + nʳ)) / timescale

with `n_initial` the total aerosol number mixing ratio [kg⁻¹] of the activation modes
(constant with height, as the SBM initializes it). The relaxation `timescale` is not an
archived LASSO parameter; the exact SBM rule is [`DiagnosticCCNProjection`](@ref).
Supply under the `nᵃ` key.
"""
struct AerosolReplenishment{FT}
    n_initial :: FT
    rate :: FT
end

AerosolReplenishment(n_initial; timescale=60) = AerosolReplenishment(promote(n_initial, 1 / timescale)...)

Base.summary(f::AerosolReplenishment) = string("AerosolReplenishment(n_initial=", f.n_initial, " kg⁻¹, τ=", 1 / f.rate, " s)")
Base.show(io::IO, f::AerosolReplenishment) = print(io, summary(f))

@inline function (f::AerosolReplenishment)(i, j, k, grid, clock, fields)
    @inbounds begin
        nᵃ = fields.nᵃ[i, j, k]
        nᶜˡ = fields.nᶜˡ[i, j, k]
        nʳ = fields.nʳ[i, j, k]
    end
    deficit = max(0, f.n_initial - (nᵃ + nᶜˡ + nʳ))
    return f.rate * deficit
end

function AtmosphereModels.materialize_atmosphere_model_forcing(f::AerosolReplenishment,
                                                               field, name, model_field_names, context::NamedTuple)
    name === :nᵃ || throw(ArgumentError("AerosolReplenishment must be supplied under the `nᵃ` key, got $name"))
    FT = eltype(field.grid)
    return AerosolReplenishment(convert(FT, f.n_initial), convert(FT, f.rate))
end

"""
    DiagnosticCCNProjection(model, n_initial)

Callback reproducing the LASSO HUJI-SBM `diagCCN = .true.` rule: the unactivated aerosol
reservoir is *rebuilt* from the initial spectrum minus the particles currently carried by
droplets and rain, never below zero,

    ρnᵃ ← max(0, ρ n_initial - ρnᶜˡ - ρnʳ)

(`MICRO_HUJISBM/microphysics.f90`: `FCCN = FCCN0` reduced by `ndrop`). `n_initial` is the
total number mixing ratio [kg⁻¹] of the activation modes. **Timing approximation:** SAM
applies this at every microphysics call; Breeze exposes no per-stage microphysics hook, so
the projection runs once per time step (before the step, via a callback), and the
reservoir seen by activation inside the Runge-Kutta stages is the projected value of the
previous step.
"""
struct DiagnosticCCNProjection{A, C, R, D, FT}
    ρnᵃ :: A
    ρnᶜˡ :: C
    ρnʳ :: R
    ρ :: D
    n_initial :: FT
end

function DiagnosticCCNProjection(model, n_initial)
    μ = model.microphysical_fields
    ρ = model.dynamics.reference_state.density
    return DiagnosticCCNProjection(μ.ρnᵃ, μ.ρnᶜˡ, μ.ρnʳ, ρ, convert(eltype(model.grid), n_initial))
end

function (projection::DiagnosticCCNProjection)(simulation)
    ρnᵃ = parent(projection.ρnᵃ)
    ρnᶜˡ = parent(projection.ρnᶜˡ)
    ρnʳ = parent(projection.ρnʳ)
    ρ = parent(projection.ρ)
    ρnᵃ .= max.(0, projection.n_initial .* ρ .- ρnᶜˡ .- ρnʳ)
    return nothing
end

"""
    compact_forcing(forcing::AbstractDict)

The `NamedTuple` of forcing tuples Breeze's `AtmosphereModel` takes, with absent (`nothing`)
terms dropped and fields without any term left out.
"""
function compact_forcing(forcing::AbstractDict)
    terms = Dict(name => Tuple(f for f in fs if !isnothing(f)) for (name, fs) in forcing)
    return NamedTuple(name => fs for (name, fs) in terms if !isempty(fs))
end
