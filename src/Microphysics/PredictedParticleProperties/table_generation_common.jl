# src/Microphysics/PredictedParticleProperties/table_generation_common.jl
@inline log_mean_particle_mass(L_ice, N_ice) = log10(L_ice / N_ice)
@inline log_qnorm(L_ice, N_ice) = log10(L_ice / N_ice)
@inline log_znorm(Z_ice, L_ice) = log10(Z_ice / L_ice)
