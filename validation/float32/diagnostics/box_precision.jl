# Where does a Float32 droplet box condense more than Float64? Component-by-component comparison.
using Breeze, Oceananigans, Statistics, Printf, CUDA
using Breeze.ParticleMicrophysics: condensational_growth_step, diffusional_growth_coefficients
using Breeze.Thermodynamics: saturation_vapor_pressure, vapor_gas_constant, dry_air_gas_constant, PlanarLiquidSurface

for FT in (Float64, Float32)
    c = ThermodynamicConstants(FT); P = DropletPhysicsParameters(FT)
    s = condensational_growth_step(FT(5e-6), FT(5e-8), FT(0.6), FT(0.005), one(FT), FT(281), FT(9e4), FT(1e-2), c, P)
    @info "growth step $FT" s.radius - 5e-6 s.condensed_mass
end

include(joinpath(@__DIR__, "box_precision_setup.jl"))

for (name, kw) in (("default", (;)), ("no self-interaction", (; self_interaction=nothing)), ("no flow", (; velocity=false)),
                   ("no flow, no self-interaction", (; self_interaction=nothing, velocity=false)))
    results = map(((Float64, Float64), (Float32, Float64), (Float32, Float32))) do (FT, TF)
        m = box(FT; thermodynamic_float_type=TF, kw...)
        θ₀ = mean(Float64.(Array(interior(m.formulation.potential_temperature))))
        q₀ = mean(Float64.(Array(interior(m.moisture_density))))
        r₀ = mean(Float64.(Array(m.microphysics.droplets.properties.r)))
        for _ in 1:10; time_step!(m, 1e-2); end
        Δθ = mean(Float64.(Array(interior(m.formulation.potential_temperature)))) - θ₀
        Δq = mean(Float64.(Array(interior(m.moisture_density)))) - q₀
        Δr = mean(Float64.(Array(m.microphysics.droplets.properties.r))) - r₀
        (FT, TF, Δθ, Δq, Δr)
    end
    for (FT, TF, Δθ, Δq, Δr) in results
        @info @sprintf("%-28s grid %s thermo %s: Δθ = %.6e  Δρqᵛ = %.6e  Δr̄ = %.6e", name, FT, TF, Δθ, Δq, Δr)
    end
end
