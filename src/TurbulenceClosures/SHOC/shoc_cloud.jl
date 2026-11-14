#####
##### SHOC's moisture: the moisture density of the model
#####
##### SHOC mixes the prognostic moisture density of the model. Its subgrid liquid `shoc_ql` and
##### cloud fraction `shoc_cldfrac` are diagnostics: the microphysics condenses its own cloud
##### liquid (it may read SHOC's distribution through `TurbulenceClosureStatistics`).
#####

shoc_moisture(model) = model.moisture_density

# The moisture density whose bottom boundary condition is SHOC's surface moisture flux
@inline surface_moisture_density(ρqᵗ) = ρqᵗ

@inline total_water_density(i, j, k, ρqᵗ) = @inbounds ρqᵗ[i, j, k]

@inline function store_total_water!(i, j, k, ρqᵗ, ρ, qw, ql)
    @inbounds ρqᵗ[i, j, k] = ρ * qw
    return nothing
end
