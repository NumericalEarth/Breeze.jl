using Breeze
using Oceananigans: Oceananigans
using Test

@testset "GeostrophicForcing [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    # Simple geostrophic wind profiles
    uᵍ(z) = -10
    vᵍ(z) = 0
    geostrophic = geostrophic_forcings(uᵍ, vᵍ)
    coriolis = FPlane(f=1e-4)
    model = AtmosphereModel(grid; coriolis, forcing=geostrophic)

    θ₀ = model.formulation.reference_state.potential_temperature
    set!(model, θ=θ₀)

    # Check that forcing is materialized correctly
    @test haskey(model.forcing, :ρu)
    @test haskey(model.forcing, :ρv)

    # Time step should not error
    Δt = 1e-6
    time_step!(model, Δt)

    # With constant uᵍ = -10 and vᵍ = 0:
    # Fρu = -f * ρᵣ * vᵍ = 0
    # Fρv = +f * ρᵣ * uᵍ = f * ρᵣ * (-10) < 0
    # So ρv should become negative after one time step
    @test maximum(model.momentum.ρv) < 0
end

@testset "SubsidenceForcing [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    # Simple subsidence profile
    wˢ(z) = -0.01  # Constant downward velocity
    subsidence = SubsidenceForcing(wˢ)

    # Apply subsidence to energy (default formulation uses StaticEnergy)
    model = AtmosphereModel(grid; forcing=(; ρθ=subsidence))

    θ₀ = model.formulation.reference_state.potential_temperature
    set!(model, θ=θ₀)

    # Check that forcing is materialized correctly
    @test haskey(model.forcing, :ρe)

    # Time step should not error
    Δt = 1e-6
    time_step!(model, Δt)
end

@testset "Combined GeostrophicForcing and SubsidenceForcing [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    # Geostrophic wind profiles
    uᵍ(z) = -10
    vᵍ(z) = 0

    # Subsidence profile
    wˢ(z) = -0.01

    coriolis = FPlane(f=1e-4)

    geostrophic = geostrophic_forcings(uᵍ, vᵍ)
    subsidence = SubsidenceForcing(wˢ)

    # Combine forcings: (subsidence, geostrophic) for momentum
    # Note: default formulation uses StaticEnergy, so use ρe not ρθ
    forcing = (;
        ρu = (subsidence, geostrophic.ρu),
        ρv = (subsidence, geostrophic.ρv),
        ρe = subsidence,
        ρqᵗ = subsidence
    )

    model = AtmosphereModel(grid; coriolis, forcing)

    θ₀ = model.formulation.reference_state.potential_temperature
    set!(model, θ=θ₀)

    # Time step should not error
    Δt = 1e-6
    time_step!(model, Δt)

    # With constant uᵍ = -10 and vᵍ = 0:
    # The geostrophic forcing on ρv should make ρv negative
    @test maximum(model.momentum.ρv) < 0
end
