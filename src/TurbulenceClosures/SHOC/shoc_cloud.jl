#####
##### SHOC's moisture: the model's water, mixed as the microphysics holds it
#####
##### SHOC mixes total water `qᵗ` (its `qw`) and the liquid-ice potential temperature. Its subgrid
##### liquid `shoc_ql` and cloud fraction `shoc_cldfrac` are diagnostics: the microphysics condenses
##### its own cloud liquid (it may read SHOC's distribution through `TurbulenceClosureStatistics`).
#####
##### With saturation adjustment and no prognostic microphysical fields, the model's moisture
##### density is SHOC's total water. Otherwise SHOC's total water is the moisture density plus the
##### cloud liquid when the microphysics carries it (`ρqᶜˡ`), and every prognostic microphysical
##### field is diffused with SHOC's heat diffusivity in the same implicit solve as `qw`, as Fortran
##### SHOC diffuses its `qtracers`. The solve is linear, so the vapor stored after the step,
##### `ρ qw - ρqᶜˡ`, is the vapor diffused with the surface moisture flux, and the water of every
##### species is conserved.
#####

"""
$(TYPEDEF)

The water SHOC mixes when the microphysics carries prognostic fields: the model's moisture
density, the cloud liquid density that SHOC's total water adds to it (`nothing` if the
microphysics has none), and the prognostic microphysical densities that SHOC diffuses.

$(TYPEDFIELDS)
"""
struct SHOCWater{M, L, T}
    "The model's moisture density, which receives SHOC's surface moisture flux"
    moisture :: M
    "The cloud liquid density in SHOC's total water, or `nothing`"
    cloud_liquid :: L
    "The prognostic microphysical densities diffused with SHOC's heat diffusivity"
    tracers :: T
end

Adapt.adapt_structure(to, w::SHOCWater) =
    SHOCWater(adapt(to, w.moisture), adapt(to, w.cloud_liquid), adapt(to, w.tracers))

function shoc_moisture(model)
    microphysics = model.microphysics
    names = AtmosphereModels.prognostic_field_names(microphysics)
    isempty(names) && return model.moisture_density

    μ = model.microphysical_fields
    tracers = Tuple(μ[name] for name in names)
    moisture_is_vapor = AtmosphereModels.moisture_prognostic_name(microphysics) === :ρqᵛ
    cloud_liquid = moisture_is_vapor && (:ρqᶜˡ ∈ names) ? μ.ρqᶜˡ : nothing

    return SHOCWater(model.moisture_density, cloud_liquid, tracers)
end

# The moisture density whose bottom boundary condition is SHOC's surface moisture flux
@inline surface_moisture_density(ρqᵗ) = ρqᵗ
@inline surface_moisture_density(water::SHOCWater) = water.moisture

@inline total_water_density(i, j, k, ρqᵗ) = @inbounds ρqᵗ[i, j, k]
@inline total_water_density(i, j, k, water::SHOCWater) = @inbounds water.moisture[i, j, k] + cloud_liquid_density(i, j, k, water.cloud_liquid)

@inline cloud_liquid_density(i, j, k, ::Nothing) = 0
@inline cloud_liquid_density(i, j, k, ρqᶜˡ) = @inbounds ρqᶜˡ[i, j, k]

@inline function store_total_water!(i, j, k, ρqᵗ, ρ, qw, ql)
    @inbounds ρqᵗ[i, j, k] = ρ * qw
    return nothing
end

# Called after the microphysical densities have been diffused
@inline function store_total_water!(i, j, k, water::SHOCWater, ρ, qw, ql)
    @inbounds water.moisture[i, j, k] = ρ * qw - cloud_liquid_density(i, j, k, water.cloud_liquid)
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Diffuse the prognostic microphysical densities of column `(i, j)` implicitly over `dtime` with
SHOC's heat diffusivity `tkh` of the step and no surface flux: the operator of SHOC's `θˡ` and
`qw` solve, applied to the specific values.
"""
@inline mix_microphysical_tracers!(i, j, Nz, dtime, closure, fields, ρ, ρqᵗ) = nothing

@inline function mix_microphysical_tracers!(i, j, Nz, dtime, closure, fields, ρ, water::SHOCWater)
    tracers = water.tracers
    host, scratch = fields.host, fields.scratch
    du, dl, d = scratch.du, scratch.dl, scratch.d
    no_surface_flux = zero(eltype(du))

    vd_shoc_decomp!(i, j, Nz, dtime, closure.constants, host.zt_grid, host.zi_grid, host.pdel,
                    fields.diagnostics.tkh, no_surface_flux, du, dl, d)

    @inbounds for k in 1:Nz
        ρᵏ = ρ[i, j, k]
        map(ρc -> (ρc[i, j, k] = ρc[i, j, k] / ρᵏ), tracers)
    end

    vd_shoc_solve!(i, j, Nz, du, dl, d, tracers)

    @inbounds for k in 1:Nz
        ρᵏ = ρ[i, j, k]
        map(c -> (c[i, j, k] = ρᵏ * c[i, j, k]), tracers)
    end

    return nothing
end
