# One step of the droplet box: is the change of ρθ the latent heating of the condensed mass C?
using Breeze, Oceananigans, Statistics, Printf, CUDA
using Breeze.ParticleMicrophysics: potential_temperature_increment
using Breeze.Thermodynamics: liquid_latent_heat, mixture_heat_capacity, MoistureMassFractions
include(joinpath(@__DIR__, "box_precision_setup.jl"))

for (FT, TF) in ((Float64, Float64), (Float32, Float64), (Float32, Float32))
    m = box(FT; thermodynamic_float_type=TF, velocity=false)
    c = m.thermodynamic_constants
    for n in 1:3
        ρθ₀ = Float64.(Array(interior(m.formulation.potential_temperature_density)))
        ρq₀ = Float64.(Array(interior(m.moisture_density)))
        T₀ = Float64.(Array(interior(m.temperature)))
        time_step!(m, 1e-2)
        ρθ₁ = Float64.(Array(interior(m.formulation.potential_temperature_density)))
        ρq₁ = Float64.(Array(interior(m.moisture_density)))
        C = Float64.(Array(interior(m.microphysical_fields.C)))
        ρ = Float64(m.dynamics.reference_state.density[1, 1, 1]); p = Float64(m.dynamics.reference_state.pressure[1, 1, 1])
        c64 = ThermodynamicConstants(Float64)
        expected = [ρ * potential_temperature_increment(T₀[i], liquid_latent_heat(T₀[i], c64) * C[i] /
                    (ρ * mixture_heat_capacity(MoistureMassFractions(ρq₀[i] / ρ - C[i] / ρ), c64)), p, 1e5, ρq₀[i] / ρ, -C[i] / ρ, c64)
                    for i in eachindex(C)]
        @info @sprintf("%s/%s step %d: ΣΔρθ = %.6e, expected from C %.6e (ratio %.5f); ΣΔρqᵛ = %.6e, -ΣC = %.6e, max|Δρθ-exp| = %.2e",
                       FT, TF, n, sum(ρθ₁ .- ρθ₀), sum(expected), sum(ρθ₁ .- ρθ₀) / sum(expected), sum(ρq₁ .- ρq₀), -sum(C),
                       maximum(abs, ρθ₁ .- ρθ₀ .- expected))
    end
end
