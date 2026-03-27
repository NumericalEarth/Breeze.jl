export LookupTable1Parameters, LookupTable2Parameters, LookupTable3Parameters,
       P3TabulationParameters

struct LookupTable1Parameters{FT}
    number_of_mass_points :: Int
    number_of_rime_fraction_points :: Int
    number_of_liquid_fraction_points :: Int
    number_of_rime_density_points :: Int
    minimum_log_mean_particle_mass :: FT
    maximum_log_mean_particle_mass :: FT
    minimum_rime_density :: FT
    maximum_rime_density :: FT
    number_of_quadrature_points :: Int
    shape_parameter_override :: FT
end

function LookupTable1Parameters(FT::Type{<:AbstractFloat} = Float64; kwargs...)
    parameters = (; kwargs...)
    return LookupTable1Parameters(
        Int(get(parameters, :number_of_mass_points, 150)),
        Int(get(parameters, :number_of_rime_fraction_points, 8)),
        Int(get(parameters, :number_of_liquid_fraction_points, 4)),
        Int(get(parameters, :number_of_rime_density_points, 10)),
        FT(get(parameters, :minimum_log_mean_particle_mass, -14.8)),
        FT(get(parameters, :maximum_log_mean_particle_mass, -0.6)),
        FT(get(parameters, :minimum_rime_density, 50)),
        FT(get(parameters, :maximum_rime_density, 900)),
        Int(get(parameters, :number_of_quadrature_points, 64)),
        FT(get(parameters, :shape_parameter_override, NaN)))
end

struct LookupTable2Parameters{FT}
    number_of_mass_points :: Int
    number_of_rain_size_points :: Int
    number_of_rime_fraction_points :: Int
    number_of_liquid_fraction_points :: Int
    number_of_rime_density_points :: Int
    minimum_log_mean_particle_mass :: FT
    maximum_log_mean_particle_mass :: FT
    minimum_log_rain_slope_parameter :: FT
    maximum_log_rain_slope_parameter :: FT
    minimum_rime_density :: FT
    maximum_rime_density :: FT
    number_of_quadrature_points :: Int
end

function LookupTable2Parameters(FT::Type{<:AbstractFloat} = Float64; kwargs...)
    parameters = (; kwargs...)
    return LookupTable2Parameters(
        Int(get(parameters, :number_of_mass_points, 50)),
        Int(get(parameters, :number_of_rain_size_points, 30)),
        Int(get(parameters, :number_of_rime_fraction_points, 4)),
        Int(get(parameters, :number_of_liquid_fraction_points, 4)),
        Int(get(parameters, :number_of_rime_density_points, 10)),
        FT(get(parameters, :minimum_log_mean_particle_mass, -14.8)),
        FT(get(parameters, :maximum_log_mean_particle_mass, -0.6)),
        FT(get(parameters, :minimum_log_rain_slope_parameter, 2.5)),
        FT(get(parameters, :maximum_log_rain_slope_parameter, 5.5)),
        FT(get(parameters, :minimum_rime_density, 50)),
        FT(get(parameters, :maximum_rime_density, 900)),
        Int(get(parameters, :number_of_quadrature_points, 64)))
end

struct LookupTable3Parameters{FT}
    number_of_znorm_points :: Int
    number_of_rime_density_points :: Int
    number_of_qnorm_points :: Int
    number_of_rime_fraction_points :: Int
    number_of_liquid_fraction_points :: Int
    minimum_log_znorm :: FT
    maximum_log_znorm :: FT
    minimum_log_qnorm :: FT
    maximum_log_qnorm :: FT
    minimum_rime_density :: FT
    maximum_rime_density :: FT
    number_of_quadrature_points :: Int
end

function LookupTable3Parameters(FT::Type{<:AbstractFloat} = Float64; kwargs...)
    parameters = (; kwargs...)
    return LookupTable3Parameters(
        Int(get(parameters, :number_of_znorm_points, 80)),
        Int(get(parameters, :number_of_rime_density_points, 10)),
        Int(get(parameters, :number_of_qnorm_points, 50)),
        Int(get(parameters, :number_of_rime_fraction_points, 4)),
        Int(get(parameters, :number_of_liquid_fraction_points, 4)),
        FT(get(parameters, :minimum_log_znorm, -23)),
        FT(get(parameters, :maximum_log_znorm, 3)),
        FT(get(parameters, :minimum_log_qnorm, -14.8)),
        FT(get(parameters, :maximum_log_qnorm, -0.6)),
        FT(get(parameters, :minimum_rime_density, 50)),
        FT(get(parameters, :maximum_rime_density, 900)),
        Int(get(parameters, :number_of_quadrature_points, 64)))
end

struct P3TabulationParameters{FT, LT1, LT2, LT3}
    lookup_table_1 :: LT1
    lookup_table_2 :: LT2
    lookup_table_3 :: LT3
    rain_lambda_points :: Int
    rain_log_lambda_range :: Tuple{FT, FT}
end

function P3TabulationParameters(FT::Type{<:AbstractFloat} = Float64; kwargs...)
    parameters = (; kwargs...)

    lookup_table_1 = LookupTable1Parameters(FT; kwargs...)
    lookup_table_2 = LookupTable2Parameters(FT; kwargs...)
    lookup_table_3 = LookupTable3Parameters(FT; kwargs...)

    return P3TabulationParameters(
        lookup_table_1,
        lookup_table_2,
        lookup_table_3,
        get(parameters, :rain_lambda_points, 200),
        (FT(get(parameters, :minimum_log_rain_slope_parameter, 2.5)),
         FT(get(parameters, :maximum_log_rain_slope_parameter, 5.5))))
end
