# Where does a Float32 droplet box condense more than Float64? Component-by-component comparison.
using Breeze, Oceananigans, Statistics, Printf, CUDA
using Breeze.ParticleMicrophysics: condensational_growth_step, diffusional_growth_coefficients
using Breeze.Thermodynamics: saturation_vapor_pressure, vapor_gas_constant, dry_air_gas_constant, PlanarLiquidSurface

for FT in (Float64, Float32)
    c = ThermodynamicConstants(FT); P = DropletPhysicsParameters(FT)
    s = condensational_growth_step(FT(5e-6), FT(5e-8), FT(0.6), FT(0.005), one(FT), FT(281), FT(9e4), FT(1e-2), c, P)
    @info "growth step $FT" s.radius - 5e-6 s.condensed_mass
end

quasirandom(n, k=1) = mod(n * sqrt(2 + k), 1)
function box(FT; thermodynamic_float_type=FT, self_interaction=DiffusiveSelfInteraction(), velocity=true)
    L = FT(0.02); Nd = 200
    grid = RectilinearGrid(GPU(), FT; size=(6, 6, 6), x=(0, L), y=(0, L), z=(0, L), topology=(Periodic, Periodic, Periodic))
    dynamics = BoussinesqDynamics(grid; pressure=90000, temperature=281)
    n = 1:Nd
    droplets = (x = L .* quasirandom.(n, 1), y = L .* quasirandom.(n, 2), z = L .* quasirandom.(n, 3),
                r = fill(5e-6, Nd), rᵈ = fill(5e-8, Nd), κ = fill(0.6, Nd))
    microphysics = LagrangianMicrophysics(FT; droplets, self_interaction)
    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_float_type)
    constants = model.thermodynamic_constants
    pᵛ = 1.005 * saturation_vapor_pressure(281.0, ThermodynamicConstants(Float64), PlanarLiquidSurface())
    ε = 287.0 / 461.5
    ε = dry_air_gas_constant(ThermodynamicConstants(Float64)) / vapor_gas_constant(ThermodynamicConstants(Float64))
    qᵛ₀ = ε * pᵛ / (90000 - (1 - ε) * pᵛ)
    if velocity
        set!(model; T=281, qᵗ=qᵛ₀, u=(x, y, z) -> sin(2π * x / L) / 20, v=(x, y, z) -> -cos(2π * y / L) / 20)
    else
        set!(model; T=281, qᵗ=qᵛ₀)
    end
    return model
end

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
