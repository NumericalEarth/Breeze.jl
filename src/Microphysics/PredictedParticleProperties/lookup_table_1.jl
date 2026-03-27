# src/Microphysics/PredictedParticleProperties/lookup_table_1.jl
function build_lookup_table_1(ice::IceProperties, arch, params::LookupTable1Parameters)
    fall_speed = tabulate(ice.fall_speed, arch, params)
    deposition = tabulate(ice.deposition, arch, params)
    bulk = tabulate(ice.bulk_properties, arch, params)
    collection = tabulate(ice.collection, arch, params)
    sixth = tabulate(ice.sixth_moment, arch, params)
    limiter = tabulate(ice.lambda_limiter, arch, params)
    ice_rain = tabulate(ice.ice_rain, arch, params)

    return P3LookupTable1(fall_speed, deposition, bulk, collection, sixth, limiter, ice_rain)
end
