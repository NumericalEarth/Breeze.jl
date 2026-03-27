# src/Microphysics/PredictedParticleProperties/table_generation_common.jl
@inline log_mean_particle_mass(L_ice, N_ice) = log10(max(L_ice / max(N_ice, eps(L_ice)), eps(L_ice)))
@inline log_qnorm(L_ice, N_ice) = log10(max(L_ice / max(N_ice, eps(L_ice)), eps(L_ice)))
@inline log_znorm(Z_ice, L_ice) = log10(max(Z_ice / max(L_ice, eps(Z_ice)), eps(Z_ice)))
