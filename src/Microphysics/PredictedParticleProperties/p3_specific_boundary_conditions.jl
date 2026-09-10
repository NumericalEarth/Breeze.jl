using Oceananigans.Models: boundary_condition_args

# P3 number, mass and property diagnostics all use TOTAL-air specific units.
function AM.fill_microphysical_boundary_halos!(p3::P3, model)
    density = AM.total_density(model.dynamics)
    args = boundary_condition_args(model)
    for name in AM.prognostic_field_names(p3)
        specific = model.microphysical_fields[AM.specific_field_name(name)]
        raw = model.microphysical_fields[name]
        fill_p3_specific_boundary_halos!(specific, raw, density, args...)
    end
    return nothing
end

fill_p3_specific_boundary_halos!(args...) = AM.fill_density_specific_boundary_halos!(args...)
