# src/Microphysics/PredictedParticleProperties/lookup_table_3.jl

@inline function shape_parameter_lookup(table::P3ThreeMomentShapeTable, L_ice, N_ice, Z_ice, Fᶠ, Fˡ, ρᶠ)
    z = log_znorm(Z_ice, L_ice)
    q = log_qnorm(L_ice, N_ice)
    return table.shape(z, ρᶠ, q, Fᶠ, Fˡ)
end
