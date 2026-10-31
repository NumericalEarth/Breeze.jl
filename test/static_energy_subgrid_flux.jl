include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Test

# A dry column at rest, reference θᵣ = 300 K, set to uniform θ = 300 K + Δθ: N² = 0 everywhere, while T ≠ Tᵣ.
function dry_column_model(formulation, closure, FT; Δθ)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=20, z=(0, 10e3), topology=(Flat, Flat, Bounded))
    reference_state = ReferenceState(grid; potential_temperature=300)
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), formulation, closure)
    set!(model; θ=300 + Δθ)
    return model
end

interior_array(field) = Array(interior(field))

@testset "Subgrid mixing of ρs is neutral where N² vanishes [$FT]" for FT in test_float_types()
    # The tolerance scales with the flux that mixing ∂z s alone would drive. Uniform θ has ∂z s = g (1 - T / Tᵣ),
    # which is 0.65 m/s² for Δθ = -20 K. With κ = 100 m²/s, the top and bottom cells then change by
    # κ ∂z s / (cᵖ Δz) ≈ 1.3e-4 K/s, or 0.13 K over the 1000 s below.
    closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=100, κ=100)
    tolerance = FT == Float64 ? 1e-3 : 2e-2

    @testset "Δθ = $Δθ K" for Δθ in (-20, 20)
        ΔT = map((:StaticEnergy, :LiquidIcePotentialTemperature)) do formulation
            model = dry_column_model(formulation, closure, FT; Δθ)
            T₀ = interior_array(model.temperature)
            for _ in 1:100
                time_step!(model, 10)
            end
            maximum(abs, interior_array(model.temperature) .- T₀)
        end
        @test ΔT[1] < tolerance # the static energy formulation leaves the state alone
        @test ΔT[2] < tolerance # as the potential temperature formulation does
    end

    @testset "∫ρs is unaffected" begin
        model = dry_column_model(:StaticEnergy, closure, FT; Δθ=-20)
        ρs = model.formulation.energy_density
        ∫ρs₀ = sum(Float64, interior_array(ρs))
        for _ in 1:100
            time_step!(model, 10)
        end
        @test abs(sum(Float64, interior_array(ρs)) - ∫ρs₀) / abs(∫ρs₀) < (FT == Float64 ? 1e-13 : 1e-5)
    end
    Oceananigans.defaults.FloatType = Float64
end

@testset "TKE decays in a neutral column colder than the reference state [$FT]" for FT in test_float_types()
    # Mixing ∂z s alone relaxes this column to N² = g² (T - Tᵣ) / (cᵖ T Tᵣ) < 0, against which the TKE grows from
    # 1 m²/s² to a steady ~100 m²/s² within half a day. The potential temperature formulation is the control.
    final_tke = map((:StaticEnergy, :LiquidIcePotentialTemperature)) do formulation
        model = dry_column_model(formulation, TKEBasedTurbulenceClosure(), FT; Δθ=-20)
        ρ = model.dynamics.reference_state.density
        set!(model.tracers.ρe, 1)
        parent(model.tracers.ρe) .*= parent(ρ)
        for _ in 1:2000
            time_step!(model, 10)
        end
        maximum(interior_array(model.tracers.ρe) ./ interior_array(ρ))
    end
    @test final_tke[1] < 1
    @test final_tke[1] ≈ final_tke[2] rtol=1e-2
    Oceananigans.defaults.FloatType = Float64
end
