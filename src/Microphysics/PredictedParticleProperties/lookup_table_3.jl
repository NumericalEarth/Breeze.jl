# src/Microphysics/PredictedParticleProperties/lookup_table_3.jl

@inline function shape_parameter_lookup(table::P3LookupTable3, L_ice, N_ice, Z_ice, Fᶠ, Fˡ, ρᶠ)
    z = log_znorm(Z_ice, L_ice)
    q = log_qnorm(L_ice, N_ice)
    return table.shape(z, ρᶠ, q, Fᶠ, Fˡ)
end

@inline function slope_parameter_lookup(table::P3LookupTable3, L_ice, N_ice, Z_ice, Fᶠ, Fˡ, ρᶠ)
    z = log_znorm(Z_ice, L_ice)
    q = log_qnorm(L_ice, N_ice)
    return table.slope(z, ρᶠ, q, Fᶠ, Fˡ)
end

function build_lookup_table_3(ice::IceProperties, arch, params::LookupTable3Parameters)
    # H15: Pass liquid fraction (Fˡ) into the 3-moment solve so that the
    # bulk-density diagnostic blends liquid water density (Fortran convention).
    μ_eval(log_z, ρᶠ, log_q, Fᶠ, Fˡ) = begin
        L_ice = 10.0^log_q
        N_ice = 1.0
        Z_ice = 10.0^(log_z + log_q)
        solve_shape_parameter(L_ice, N_ice, Z_ice, Fᶠ, ρᶠ;
                              liquid_fraction = Fˡ, closure = ThreeMomentClosure())
    end

    λ_eval(log_z, ρᶠ, log_q, Fᶠ, Fˡ) = begin
        L_ice = 10.0^log_q
        N_ice = 1.0
        Z_ice = 10.0^(log_z + log_q)
        μ = solve_shape_parameter(L_ice, N_ice, Z_ice, Fᶠ, ρᶠ;
                                  liquid_fraction = Fˡ, closure = ThreeMomentClosure())
        exp(solve_lambda(L_ice, N_ice, Z_ice, Fᶠ, ρᶠ, μ))
    end

    range = table_range((params.minimum_log_znorm, params.maximum_log_znorm),
                        (params.minimum_rime_density, params.maximum_rime_density),
                        (params.minimum_log_qnorm, params.maximum_log_qnorm),
                        (0.0, 1.0), (0.0, 1.0))

    points = (params.number_of_znorm_points,
              params.number_of_rime_density_points,
              params.number_of_qnorm_points,
              params.number_of_rime_fraction_points,
              params.number_of_liquid_fraction_points)

    μ_table = build_tabulated_function(μ_eval, arch, Float64, range, points)
    λ_table = build_tabulated_function(λ_eval, arch, Float64, range, points)

    return P3LookupTable3(μ_table, λ_table, nothing)
end
