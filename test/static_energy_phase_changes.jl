using Breeze
using Breeze.Thermodynamics: Thermodynamics as TD
using Test

@testset "Static-energy phase-change heating" begin
    for FT in (Float32, Float64), T in FT.((240, 270, 290))
        constants = ThermodynamicConstants(FT)
        q = TD.MoistureMassFractions(FT(0.01), FT(0.001), FT(0.001))
        δq = FT(1e-4)
        # Condensation, deposition, freezing, and their reverse processes. The
        # exact finite-transfer identity uses the final mixture heat capacity.
        for (vapor, liquid, ice) in ((-1, 1, 0), (-1, 0, 1), (0, -1, 1)), sign in (-1, 1)
            Δq = sign * δq
            after = TD.MoistureMassFractions(q.vapor + vapor * Δq,
                                            q.liquid + liquid * Δq,
                                            q.ice + ice * Δq)
            state = TD.with_temperature(TD.StaticEnergyState(zero(FT), q, FT(1000), FT(80000)), T, constants)
            updated = TD.with_moisture(state, after)
            actual = @inferred TD.temperature(updated, constants)
            heating = Δq * (liquid * TD.liquid_latent_heat(T, constants) +
                             ice * TD.ice_latent_heat(T, constants))
            expected = T + heating / TD.mixture_heat_capacity(after, constants)
            @test actual ≈ expected rtol=4eps(FT)
            @test updated.static_energy == state.static_energy
            @test actual isa FT
        end
    end
end
