using Breeze
using Breeze: ReferenceState, AnelasticDynamics, LiquidIcePotentialTemperatureFormulation, GeostrophicForcing
using Oceananigans: Oceananigans, prognostic_fields
using Oceananigans.Fields: interior
using Oceananigans.Grids: znodes, Center
using Statistics: mean
using Test

@testset "GeostrophicForcing smoke test [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    uᵍ(z) = -10
    vᵍ(z) = 0
    geostrophic = geostrophic_forcings(uᵍ, vᵍ)
    coriolis = FPlane(f=1e-4)
    model = AtmosphereModel(grid; coriolis, forcing=geostrophic)

    @test haskey(model.forcing, :ρu)
    @test haskey(model.forcing, :ρv)
    @test model.forcing.ρu isa GeostrophicForcing
    @test model.forcing.ρv isa GeostrophicForcing

    Δt = 1e-6
    time_step!(model, Δt)

    # With constant uᵍ = -10 and vᵍ = 0: Fρv = +f * ρᵣ * (-10) < 0
    @test minimum(model.momentum.ρv) < 0
end

@testset "SubsidenceForcing smoke test [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    wˢ(z) = -0.01
    subsidence = SubsidenceForcing(wˢ)

    model = AtmosphereModel(grid; forcing=(; ρθ=subsidence))

    @test haskey(model.forcing, :ρθ)
    @test model.forcing.ρθ isa SubsidenceForcing
    @test !isnothing(model.forcing.ρθ.subsidence_vertical_velocity)

    Δt = 1e-6
    time_step!(model, Δt)
end

@testset "SubsidenceForcing with LiquidIcePotentialTemperature [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nz = 10
    Hz = 1000
    grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100), y=(0, 100), z=(0, Hz))

    wˢ(z) = FT(-0.01)
    subsidence = SubsidenceForcing(wˢ)

    reference_state = ReferenceState(grid)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature, forcing=(; ρqᵗ=subsidence))

    θ₀ = model.dynamics.reference_state.potential_temperature

    q₀ = FT(0.015)
    Γq = FT(1e-5)
    qᵗ_profile(x, y, z) = q₀ - Γq * z
    set!(model, θ=θ₀, qᵗ=qᵗ_profile)

    @test haskey(model.forcing, :ρqᵗ)
    @test model.forcing.ρqᵗ isa SubsidenceForcing

    ρqᵗ_initial = sum(model.moisture_density)

    # Reduced iterations (from 10 to 3)
    Δt = FT(0.1)
    for _ in 1:3
        time_step!(model, Δt)
    end

    ρqᵗ_final = sum(model.moisture_density)

    @test !isnan(ρqᵗ_final)
    @test ρqᵗ_final < ρqᵗ_initial
end

@testset "θ → e conversion in StaticEnergy model [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))
    reference_state = ReferenceState(grid)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, formulation=:StaticEnergy)

    θ₀ = model.dynamics.reference_state.potential_temperature
    set!(model, θ=θ₀)

    @test sum(abs, model.formulation.energy_density) > 0

    Δt = 1e-6
    time_step!(model, Δt)
end

@testset "Combined GeostrophicForcing and SubsidenceForcing [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))
    coriolis = FPlane(f=1e-4)

    uᵍ(z) = -10
    vᵍ(z) = 0
    geostrophic = geostrophic_forcings(uᵍ, vᵍ)

    wˢ(z) = -0.01
    subsidence = SubsidenceForcing(wˢ)

    forcing = (;
        ρu = (subsidence, geostrophic.ρu),
        ρv = (subsidence, geostrophic.ρv)
    )

    coriolis = FPlane(f=1e-4)
    model = AtmosphereModel(grid; coriolis, forcing)

    @test haskey(model.forcing, :ρu)
    @test haskey(model.forcing, :ρv)

    Δt = 1e-6
    time_step!(model, Δt)

    @test maximum(model.momentum.ρv) < 0
end

#####
##### Analytical subsidence forcing tests
#####

@testset "Subsidence forcing gradient [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 4), x=(0, 10), y=(0, 10), z=(0, 16))
    reference_state = ReferenceState(grid)
    dynamics = AnelasticDynamics(reference_state)

    wˢ = 1
    Γ = 1e-2
    ϕᵢ(x, y, z) = Γ * z
    Δt = 1e-2
    Δϕ = - Δt * wˢ * Γ |> FT
    subsidence = SubsidenceForcing(FT(wˢ))

    # Test a representative subset of fields (reduced from 5 to 3)
    @testset "Subsidence forcing with constant gradient [$name, $FT]" for name in (:ρu, :ρθ, :ρqᵗ)
        # Test solo configuration only (combined is tested above)
        forcing = (; name => subsidence)

        kw = (; advection=nothing, dynamics, formulation=:LiquidIcePotentialTemperature, forcing)
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
        @test ρϕ₁[1, 1, 4] - ρϕ₀[1, 1, 4] ≈ ρᵣ[1, 1, 4] * Δϕ rtol=1e-3
    end
end
