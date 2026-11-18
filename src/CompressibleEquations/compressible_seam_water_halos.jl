#####
##### Halos of the water of CompressibleDynamics across the seam of a Yin-Yang grid
#####
#
# PROVISIONAL (AMIP pilot). Not included by Breeze: a driver evaluates this file in `Breeze.HydrostaticEquations`,
# whose limited interpolation of a mixing ratio it reuses unchanged (`moist_seam_water_halos.jl`:
# `_fill_water_density_seam_halo!`, `_fill_total_density_seam_halo!`, `water_densities`, `water_mass_fractions`).
#
# The cubic interpolation that fills the halos of a panel is not convex, so the halo of a water density that is
# nowhere negative can be negative next to a sharp edge (measured: the first negative rain of a compressible
# model on a Yin-Yang grid is in halo cells, while every owned cell is non-negative). For the water only, the
# halos are filled again after the prognostic halos and before the diagnosis of the state, at every stage:
# the mixing ratio per unit DRY mass r = ρc / ρᵈ is interpolated with the stencils of the grid and limited to
# the range of its donors, the halo of the water density is ρᵈ r with the dry density of the halo cell, the
# halo of the total density is ρᵈ + Σ ρc and the halos of the mass fractions are ρc / ρ. Only halo cells are
# written: nothing is clipped and no mass is moved.

const CompressibleYinYangModel = AtmosphereModels.AtmosphereModel{<:parentmodule(@__MODULE__).CompressibleEquations.CompressibleDynamics, <:Any, <:Any, <:Any, <:YinYangGrid}

function fill_compressible_water_seam_halos!(model::CompressibleYinYangModel)
    isnothing(model.microphysics) && return nothing
    grid = model.grid
    dynamics = model.dynamics
    ρᵈ = dynamics.dry_density
    ρ = dynamics.total_density
    ρq = water_densities(model)
    q = water_mass_fractions(model)
    table = interpolation_table(yin_yang_interpolation(grid), Center(), Center())

    for receiver in 1:2
        donor = 3 - receiver
        panel = getregion(grid, receiver)
        data(field, r) = getregion(field.data, r)
        ρᵈʳ = data(ρᵈ, receiver)
        ks = axes(ρᵈʳ, 3)

        for ρc in ρq
            launch_yin_yang_interpolation!(panel, ks, table, _fill_water_density_seam_halo!,
                                           data(ρc, receiver), ρᵈʳ, data(ρc, donor), data(ρᵈ, donor), table)
        end

        launch_yin_yang_interpolation!(panel, ks, table, _fill_total_density_seam_halo!,
                                       data(ρ, receiver), map(f -> data(f, receiver), q), ρᵈʳ,
                                       map(f -> data(f, receiver), ρq), table)
    end

    return nothing
end

# After the halos of the prognostic fields are filled and before the state is diagnosed from them, and again
# after it, so that the mass fractions the advection reconstructs in halo cells are those of the limited water
function AtmosphereModels.compute_auxiliary_variables!(model::CompressibleYinYangModel)
    fill_compressible_water_seam_halos!(model)
    invoke(AtmosphereModels.compute_auxiliary_variables!, Tuple{AtmosphereModels.AtmosphereModel}, model)
    fill_compressible_water_seam_halos!(model)
    return nothing
end
