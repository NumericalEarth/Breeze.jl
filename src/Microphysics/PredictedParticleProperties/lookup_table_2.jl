# src/Microphysics/PredictedParticleProperties/lookup_table_2.jl
function build_lookup_table_2(ice::IceProperties, rain::RainProperties, arch, params::LookupTable2Parameters)
    FT = typeof(params.minimum_log_mean_particle_mass)
    mass_eval = build_ice_rain_family_entry(IceRainMassCollection(), FT; quadrature_points = params.number_of_quadrature_points)
    number_eval = build_ice_rain_family_entry(IceRainNumberCollection(), FT; quadrature_points = params.number_of_quadrature_points)
    sixth_eval = build_ice_rain_family_entry(IceRainSixthMomentCollection(), FT; quadrature_points = params.number_of_quadrature_points)

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

# M4: Full double integral for ice-rain collection matching Fortran P3 convention.
#
# Fortran create_p3_lookupTable_2.f90 computes:
#   ∫∫ (√A_ice + √A_rain)² × |V_ice − V_rain| × w(D_ice) × N_ice(D_ice) × N_rain(D_rain) dD_ice dD_rain
#
# where w(D_ice) = m_ice(D_ice) for mass collection, 1 for number.
# The geometric cross-section kernel (√A_ice + √A_rain)² and differential fall speed
# |V_ice − V_rain| couple the two PSDs, making the integral non-separable.
function build_ice_rain_family_entry(integral, FT::Type{<:AbstractFloat};
                                     quadrature_points = 64,
                                     rain_quadrature_points = 40,
                                     rain_density = FT(997))
    ice_eval = P3IntegralEvaluator(integral, FT; number_of_quadrature_points = quadrature_points)

    return function (log_q, log_λr, Fᶠ, Fˡ, ρᶠ)
        λ_r = FT(10)^log_λr
        mean_particle_mass = FT(10)^log_q

        # Build ice PSD state
        state = state_from_mean_particle_mass(ice_eval, mean_particle_mass, Fᶠ, Fˡ; rime_density = ρᶠ)
        thresholds = regime_thresholds_from_state(FT, state)
        λ_ice = state.slope

        # Chebyshev-Gauss nodes for ice dimension
        ice_nodes = ice_eval.nodes
        ice_weights = ice_eval.weights
        n_ice = length(ice_nodes)

        # Chebyshev-Gauss nodes for rain dimension
        rain_nodes, rain_weights = chebyshev_gauss_nodes_weights(FT, rain_quadrature_points)

        result = zero(FT)

        for j in 1:rain_quadrature_points
            x_r = rain_nodes[j]
            w_r = rain_weights[j]

            D_r = transform_to_diameter(x_r, λ_r)
            J_r = jacobian_diameter_transform(x_r, λ_r)

            # Rain properties at D_r
            V_r = rain_fall_speed(D_r, one(FT))
            A_r = FT(π) / 4 * D_r^2
            N_r = exp(-λ_r * D_r)  # exponential PSD (μ_r = 0), no N0 normalization

            for i in 1:n_ice
                x_i = @inbounds ice_nodes[i]
                w_i = @inbounds ice_weights[i]

                D_i = transform_to_diameter(x_i, λ_ice)
                J_i = jacobian_diameter_transform(x_i, λ_ice)

                # Ice properties at D_i
                V_i = terminal_velocity(D_i, state, thresholds)
                A_i = particle_area(D_i, state, thresholds)
                N_i = size_distribution(D_i, state)

                # Geometric cross-section kernel: (√A_ice + √A_rain)²
                cross_section = (sqrt(A_i) + sqrt(A_r))^2

                # Differential fall speed
                dV = abs(V_i - V_r)

                # Per-integral weighting factor
                w_factor = _ice_rain_weight(integral, D_i, state, thresholds)

                result += w_i * w_r * J_i * J_r * cross_section * dV * w_factor * N_i * N_r
            end
        end

        return result
    end
end

# Per-integral weighting: m_ice for mass collection, 1 for number
@inline _ice_rain_weight(::IceRainMassCollection, D, state, thresholds) = particle_mass(D, state, thresholds)
@inline _ice_rain_weight(::IceRainNumberCollection, D, state, thresholds) = one(typeof(D))

# M4/D6: Sixth moment ice-rain collection uses Fortran's relative-variance formula.
#
# Fortran create_p3_lookupTable_2.f90 computes:
#   sum3 = ∫∫ kernel × 6D_ice^5 / dmdD × m_rain × N_ice × N_rain  dD_ice dD_rain
#   sum4 = ∫∫ kernel × 3D_ice^2 / dmdD × m_rain × N_ice × N_rain  dD_ice dD_rain
#   mom3 = N₀ × Γ(μ+4) / λ^(μ+4)
#   mom6 = N₀ × Γ(μ+7) / λ^(μ+7)
#   m6collr = sum3/mom3² − 2 × mom6/mom3³ × sum4
#
# This is the dG/dt formula for relative variance G = M6/M3², exactly the same
# structure as `sixth_moment_relative_variance` used for single-integral processes.
function build_ice_rain_family_entry(::IceRainSixthMomentCollection, FT::Type{<:AbstractFloat};
                                     quadrature_points = 64,
                                     rain_quadrature_points = 40,
                                     rain_density = FT(997))
    ice_eval = P3IntegralEvaluator(IceRainSixthMomentCollection(), FT;
                                    number_of_quadrature_points = quadrature_points)

    return function (log_q, log_λr, Fᶠ, Fˡ, ρᶠ)
        λ_r = FT(10)^log_λr
        mean_particle_mass = FT(10)^log_q

        # Build ice PSD state
        state = state_from_mean_particle_mass(ice_eval, mean_particle_mass, Fᶠ, Fˡ; rime_density = ρᶠ)
        thresholds = regime_thresholds_from_state(FT, state)
        λ_ice = state.slope

        # Chebyshev-Gauss nodes for ice dimension
        ice_nodes = ice_eval.nodes
        ice_weights = ice_eval.weights
        n_ice = length(ice_nodes)

        # Chebyshev-Gauss nodes for rain dimension
        rain_nodes, rain_weights = chebyshev_gauss_nodes_weights(FT, rain_quadrature_points)

        sum3 = zero(FT)
        sum4 = zero(FT)

        for j in 1:rain_quadrature_points
            x_r = rain_nodes[j]
            w_r = rain_weights[j]

            D_r = transform_to_diameter(x_r, λ_r)
            J_r = jacobian_diameter_transform(x_r, λ_r)

            # Rain properties at D_r
            V_r = rain_fall_speed(D_r, one(FT))
            A_r = FT(π) / 4 * D_r^2
            N_r = exp(-λ_r * D_r)  # exponential PSD (μ_r = 0), no N0 normalization
            m_r = FT(π) / 6 * rain_density * D_r^3

            for i in 1:n_ice
                x_i = @inbounds ice_nodes[i]
                w_i = @inbounds ice_weights[i]

                D_i = transform_to_diameter(x_i, λ_ice)
                J_i = jacobian_diameter_transform(x_i, λ_ice)

                # Ice properties at D_i
                V_i = terminal_velocity(D_i, state, thresholds)
                A_i = particle_area(D_i, state, thresholds)
                N_i = size_distribution(D_i, state)
                dmdD = particle_mass_derivative(D_i, state, thresholds)

                # Geometric cross-section kernel: (√A_ice + √A_rain)²
                cross_section = (sqrt(A_i) + sqrt(A_r))^2

                # Differential fall speed
                dV = abs(V_i - V_r)

                # Common factor: quadrature weights × Jacobians × kernel × PSD × rain
                common = w_i * w_r * J_i * J_r * cross_section * dV * N_i * N_r * m_r / dmdD

                # sum3: 6 D_ice^5 / dmdD weighting (dmdD included in common)
                sum3 += common * 6 * D_i^5

                # sum4: 3 D_ice^2 / dmdD weighting (dmdD included in common)
                sum4 += common * 3 * D_i^2
            end
        end

        # PSD moments: mom_k = N₀ × Γ(μ + k + 1) / λ^(μ + k + 1)
        μ = state.shape
        N₀ = state.intercept
        mom3 = N₀ * FT(gamma(μ + 4)) / λ_ice^(μ + 4)
        mom6 = N₀ * FT(gamma(μ + 7)) / λ_ice^(μ + 7)
        mom3_safe = max(mom3, eps(FT))

        # Relative-variance formula: dG/dt for G = M6/M3²
        return sum3 / mom3_safe^2 - 2 * mom6 / mom3_safe^3 * sum4
    end
end
