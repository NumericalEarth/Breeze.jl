include(joinpath(@__DIR__, "setup.jl"))

#####
##### The moisture's part of the pressure that the acoustic substep loop integrates.
#####
##### A sharp vapor jump under an inversion, advected by a uniform wind over small potential
##### temperature noise (the case of NumericalEarth/Breeze.jl#1093, after Xiao et al. 2015,
##### Fig. 7). With a loop that integrates the moist potential temperature density, the pressure
##### feels the advection of the jump, so the vertical velocity in the jump layer stays at the level
##### the noise sets. A loop that reads the perturbation of the thermodynamic density as its
##### pressure, and so sees the vapor one stage late, amplifies it there several-fold within ten
##### minutes.
#####

using Breeze
using Breeze.CompressibleEquations: MoistPotentialTemperatureDensity
using Oceananigans
using Test

interpolate_sounding(zs, vs, z) = (k = clamp(searchsortedlast(zs, z), 1, length(zs) - 1);
                                   vs[k] + (vs[k+1] - vs[k]) * clamp((z - zs[k]) / (zs[k+1] - zs[k]), 0, 1))

sounding_θ(z) = interpolate_sounding([0, 430, 1000, 1200, 1400, 1600, 2000, 3000, 5000],
                                     [295.1, 297.4, 303.1, 307.7, 312.6, 314.2, 314.6, 315.0, 315.4], z)

function sounding_q(z)
    r = 1e-3 * interpolate_sounding([0, 1150, 1600, 5000], [13, 13, 1, 1], z)
    return r / (1 + r)
end

@testset "Vapor jump advected under an inversion [$FT]" for FT in (Float64,)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (40, 250), halo = (5, 5), x = (0, 4e3), z = (0, 5e3),
                           topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps = 6, thermodynamic_variable = MoistPotentialTemperatureDensity());
                                    base_pressure = 97e3,
                                    reference_potential_temperature = sounding_θ,
                                    reference_vapor_mass_fraction = sounding_q)
    model = AtmosphereModel(grid; dynamics, advection = WENO(order = 5))

    # Deterministic noise of amplitude 0.1 K below 2 km.
    noise(x, z) = 2 * mod(43758.5453 * sin(0.129898x + 3.91165z), 1) - 1
    set!(model; θ = (x, z) -> sounding_θ(z) + 0.1 * (z < 2e3) * noise(x, z), qᵛ = (x, z) -> sounding_q(z),
         u = 12, ρ = Breeze.HydrostaticallyBalancedDensity())

    for _ in 1:1200
        time_step!(model, 0.5)
    end

    w = Array(interior(model.velocities.w))
    jump_layer = findall(z -> 1e3 <= z <= 1.8e3, znodes(grid, Face()))
    wʲ = w[:, 1, jump_layer]
    w′ = wʲ .- sum(wʲ; dims = 1) / size(wʲ, 1)

    @test maximum(abs, w′) < 0.08
end
