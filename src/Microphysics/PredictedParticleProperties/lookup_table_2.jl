# src/Microphysics/PredictedParticleProperties/lookup_table_2.jl
function build_lookup_table_2(ice::IceProperties, rain::RainProperties, arch, params::LookupTable2Parameters)
    FT = typeof(params.minimum_log_mean_particle_mass)
    mass_eval = build_ice_rain_family_entry(IceRainMassCollection(), rain; quadrature_points = params.number_of_quadrature_points)
    number_eval = build_ice_rain_family_entry(IceRainNumberCollection(), rain; quadrature_points = params.number_of_quadrature_points)
    sixth_eval = build_ice_rain_family_entry(IceRainSixthMomentCollection(), rain; quadrature_points = params.number_of_quadrature_points)

    range = table_range((params.minimum_log_mean_particle_mass, params.maximum_log_mean_particle_mass),
                        (params.minimum_log_rain_slope_parameter, params.maximum_log_rain_slope_parameter),
                        (zero(FT), one(FT)), (zero(FT), one(FT)),
                        (params.minimum_rime_density, params.maximum_rime_density))
    points = (params.number_of_mass_points, params.number_of_rain_size_points,
              params.number_of_rime_fraction_points, params.number_of_liquid_fraction_points,
              params.number_of_rime_density_points)

    mass_table = build_tabulated_function(mass_eval, arch, FT, range, points)
    number_table = build_tabulated_function(number_eval, arch, FT, range, points)
    sixth_table = build_tabulated_function(sixth_eval, arch, FT, range, points)

    return P3LookupTable2(mass_table, number_table, sixth_table)
end

function build_ice_rain_family_entry(integral, rain::RainProperties; quadrature_points)
    FT = typeof(rain.maximum_mean_diameter)
    ice_eval = P3IntegralEvaluator(integral, FT; number_of_quadrature_points = quadrature_points)
    return function (log_q, log_λr, Fᶠ, Fˡ, ρᶠ)
        ice_component = ice_eval(log_q, Fᶠ, Fˡ, ρᶠ)
        rain_component = if integral isa IceRainNumberCollection
            rain.velocity_number(log_λr)
        else
            rain.velocity_mass(log_λr)
        end
        return ice_component * rain_component
    end
end
