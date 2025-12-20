using Breeze
using Oceananigans: Oceananigans, prognostic_fields
using Oceananigans.Fields: interior
using Oceananigans.Grids: znodes, Center
using Statistics: mean
using Test

@testset "GeostrophicForcing smoke test [$(FT)]" for FT in (Float32, Float64)
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

@testset "SubsidenceForcing smoke test [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    # Simple subsidence profile
    wˢ(z) = -0.01  # Constant downward velocity
    subsidence = SubsidenceForcing(wˢ)

    # Apply subsidence to energy (default formulation uses StaticEnergy)
    model = AtmosphereModel(grid; forcing=(; ρe=subsidence))

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
    coriolis = FPlane(f=1e-4)

    # Geostrophic wind profiles
    uᵍ(z) = -10
    vᵍ(z) = 0
    geostrophic = geostrophic_forcings(uᵍ, vᵍ)

    # Subsidence profile
    wˢ(z) = -0.01
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

#####
##### Analytical subsidence forcing tests
#####
#
# For a single-column model with constant subsidence wˢ and constant gradient Γ,
# the subsidence forcing is: F_{ρϕ} = -ρᵣ * wˢ * ∂z(ϕ) = -ρᵣ * wˢ * Γ
# After one time step Δt, the change in the specific field is: Δϕ = -Δt * wˢ * Γ

@testset "Subsidence forcing gradient [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 4), x=(0, 10), y=(0, 10), z=(0, 16))
    reference_state = ReferenceState(grid)
    formulation = AnelasticFormulation(reference_state, thermodynamics=:LiquidIcePotentialTemperature)

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

                kw = (; advection=nothing, timestepper=:QuasiAdamsBashforth2, formulation, forcing)
                model = AtmosphereModel(grid; tracers=:ρc, kw...)
                θ₀ = model.formulation.reference_state.potential_temperature

                ρᵣ = model.formulation.reference_state.density
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

                # Only test points that don't touch the boundary.
                @test ρϕ₁[1, 1, 2] - ρϕ₀[1, 1, 2] ≈ ρᵣ[1, 1, 2] * Δϕ rtol=1e-3
                @test ρϕ₁[1, 1, 3] - ρϕ₀[1, 1, 3] ≈ ρᵣ[1, 1, 3] * Δϕ rtol=1e-3
            end
        end
    end
end
