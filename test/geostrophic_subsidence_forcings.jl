using Breeze
using Breeze: ReferenceState, AnelasticDynamics, LiquidIcePotentialTemperatureFormulation, GeostrophicForcing
using Oceananigans: Oceananigans, prognostic_fields
using Oceananigans.Fields: interior
using Oceananigans.Grids: znodes, Center
using Statistics: mean
using Test

include("test_utils.jl")

@testset "GeostrophicForcing smoke test [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    # Simple geostrophic wind profiles
    uᵍ(z) = -10
    vᵍ(z) = 0
    geostrophic = geostrophic_forcings(uᵍ, vᵍ)
    coriolis = FPlane(f=1e-4)
    model = AtmosphereModel(grid; coriolis, forcing=geostrophic)

    # Check that forcing is materialized correctly
    @test haskey(model.forcing, :ρu)
    @test haskey(model.forcing, :ρv)

    # Check the forcing type
    @test model.forcing.ρu isa GeostrophicForcing
    @test model.forcing.ρv isa GeostrophicForcing

    # Time step should not error
    Δt = 1e-6
    time_step!(model, Δt)

    # With constant uᵍ = -10 and vᵍ = 0:
    # Fρu = -f * ρvᵍ = -f * ρᵣ * 0 = 0
    # Fρv = +f * ρuᵍ = +f * ρᵣ * (-10) < 0
    # So ρv should become NEGATIVE after one time step
    @test minimum(model.momentum.ρv) < 0
end

@testset "SubsidenceForcing smoke test [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    # Simple subsidence profile
    wˢ(z) = -0.01  # Constant downward velocity
    subsidence = SubsidenceForcing(wˢ)

    # Apply subsidence to energy (default formulation uses StaticEnergy)
    model = AtmosphereModel(grid; forcing=(; ρθ=subsidence))

    # Check that forcing is materialized correctly
    @test haskey(model.forcing, :ρθ)
    @test model.forcing.ρθ isa SubsidenceForcing

    # Check that the subsidence velocity field is set up correctly
    @test !isnothing(model.forcing.ρθ.subsidence_vertical_velocity)

    # Time step should not error
    Δt = 1e-6
    time_step!(model, Δt)
end

@testset "SubsidenceForcing with LiquidIcePotentialTemperature [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    
    Nz = 10
    Hz = 1000  # 1 km domain height
    grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100), y=(0, 100), z=(0, Hz))

    # Simple subsidence profile: constant downward velocity
    wˢ(z) = FT(-0.01)

    subsidence = SubsidenceForcing(wˢ)

    # Use LiquidIcePotentialTemperature thermodynamics
    reference_state = ReferenceState(grid)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature, forcing=(; ρqᵗ=subsidence))

    # Set potential temperature to reference state
    θ₀ = model.dynamics.reference_state.potential_temperature

    # Set up a linear moisture profile with known gradient
    q₀ = FT(0.015)  # 15 g/kg at surface
    Γq = FT(1e-5)   # moisture decreases with height
    qᵗ_profile(x, y, z) = q₀ - Γq * z
    set!(model, θ=θ₀, qᵗ=qᵗ_profile)

    # Check that forcing is materialized correctly
    @test haskey(model.forcing, :ρqᵗ)
    @test model.forcing.ρqᵗ isa SubsidenceForcing

    # Get initial moisture density for comparison
    ρqᵗ_initial = sum(model.moisture_density)

    # Time step (multiple iterations to see the effect)
    Δt = FT(0.1)
    for _ in 1:10
        time_step!(model, Δt)
    end

    ρqᵗ_final = sum(model.moisture_density)
    
    # Check simulation didn't produce NaN
    @test !isnan(ρqᵗ_final)
    
    # With downward subsidence (wˢ < 0) and moisture decreasing with height (∂qᵗ/∂z < 0),
    # the subsidence forcing is: F_ρqᵗ = -ρᵣ * wˢ * ∂qᵗ/∂z = -ρᵣ * (-0.01) * (-Γq) < 0
    # So moisture should DECREASE
    @test ρqᵗ_final < ρqᵗ_initial
end

@testset "θ → e conversion in StaticEnergy model [$(FT)]" for FT in test_float_types()
    # This test verifies that set!(model, θ=...) works correctly for StaticEnergy models
    # by converting potential temperature to energy density
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))
    reference_state = ReferenceState(grid)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, formulation=:StaticEnergy)

    # Get the reference potential temperature
    θ₀ = model.dynamics.reference_state.potential_temperature

    # This should work without error (tests the maybe_adjust_thermodynamic_state fix)
    set!(model, θ=θ₀)

    # Verify energy was set to a non-zero value
    @test sum(abs, model.formulation.energy_density) > 0

    # Time step should work
    Δt = 1e-6
    time_step!(model, Δt)
end

@testset "Combined GeostrophicForcing and SubsidenceForcing [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))
    coriolis = FPlane(f=1e-4)

    # Geostrophic wind profiles
    uᵍ(z) = -10
    vᵍ(z) = 0
    geostrophic = geostrophic_forcings(uᵍ, vᵍ)

    # Subsidence profile
    wˢ(z) = -0.01
    subsidence = SubsidenceForcing(wˢ)

    # Combine forcings: (subsidence, geostrophic) for momentum
    forcing = (;
        ρu = (subsidence, geostrophic.ρu),
        ρv = (subsidence, geostrophic.ρv)
    )

    coriolis = FPlane(f=1e-4)
    model = AtmosphereModel(grid; coriolis, forcing)

    # Check that forcings are materialized correctly
    # When tuples are passed, they get wrapped in MultipleForcings
    @test haskey(model.forcing, :ρu)
    @test haskey(model.forcing, :ρv)

    # Time step should not error
    Δt = 1e-6
    time_step!(model, Δt)

    # With constant uᵍ = -10 and vᵍ = 0:
    # The geostrophic forcing on ρv should make ρv negative
    @test maximum(model.momentum.ρv) < 0
end

#####
##### Analytical subsidence forcing tests
#####
#
# For a single-column model with constant subsidence wˢ and constant gradient Γ,
# the subsidence forcing is: F_{ρϕ} = -ρᵣ * wˢ * ∂z(ϕ) = -ρᵣ * wˢ * Γ
# After one time step Δt, the change in the specific field is: Δϕ = -Δt * wˢ * Γ

@testset "Subsidence forcing gradient [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 4), x=(0, 10), y=(0, 10), z=(0, 16))
    reference_state = ReferenceState(grid)
    dynamics = AnelasticDynamics(reference_state)

    wˢ = 1
    Γ = 1e-2
    ϕᵢ(x, y, z) = Γ * z
    Δt = 1e-2
    Δϕ = - Δt * wˢ * Γ |> FT
    subsidence = SubsidenceForcing(FT(wˢ))

    @testset "Subsidence forcing with constant gradient [$name, $FT]" for name in (:ρu, :ρv, :ρθ, :ρqᵗ, :ρc)
        for config in (:solo, :combined)
            @testset let config=config
                forcing = if config == :solo
                    forcing = (; name => subsidence)
                else
                    zero_forcing = Forcing(Returns(zero(FT)))
                    forcing = (; name => (subsidence, zero_forcing))
                end

                kw = (; advection=nothing, timestepper=:QuasiAdamsBashforth2, dynamics, formulation=:LiquidIcePotentialTemperature, forcing)
                model = AtmosphereModel(grid; tracers=:ρc, kw...)
                θ₀ = model.dynamics.reference_state.potential_temperature

                ρᵣ = model.dynamics.reference_state.density
                ρϕ = CenterField(grid)
                set!(ρϕ, ϕᵢ)
                set!(ρϕ, ρᵣ * ρϕ)

                kw = (; name => ρϕ)
                if name == :ρθ
                    set!(model; kw...)
                else
                    set!(model; θ=θ₀, kw...)
                end

                ρϕ = prognostic_fields(model)[name]
                ρϕ₀ = interior(ρϕ) |> Array
                time_step!(model, Δt)
                ρϕ₁ = interior(ρϕ) |> Array
                ρᵣ = interior(ρᵣ) |> Array

                @test ρϕ₁[1, 1, 1] - ρϕ₀[1, 1, 1] ≈ ρᵣ[1, 1, 1] * Δϕ rtol=1e-3
                @test ρϕ₁[1, 1, 2] - ρϕ₀[1, 1, 2] ≈ ρᵣ[1, 1, 2] * Δϕ rtol=1e-3
                @test ρϕ₁[1, 1, 3] - ρϕ₀[1, 1, 3] ≈ ρᵣ[1, 1, 3] * Δϕ rtol=1e-3
                @test ρϕ₁[1, 1, 4] - ρϕ₀[1, 1, 4] ≈ ρᵣ[1, 1, 4] * Δϕ rtol=1e-3
            end
        end
    end
end
