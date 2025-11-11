using Breeze
using Oceananigans
using Test
using GPUArraysCore: @allowscalar

"""
    thermal_bubble_model(grid; uᵢ=0, vᵢ=0, wᵢ=0)

Set up a thermal bubble initial condition for an `AtmosphereModel` on the provided grid.

The thermal bubble is a spherical potential temperature perturbation centered in the domain.
The background state has a stable stratification with Brunt-Väisälä frequency squared N² = 1e-6.

# Arguments
- `grid`: An `Oceananigans.AbstractGrid` or compatible grid for the model domain.
- `uᵢ`: Optional initial x-velocity (default: 0).
- `vᵢ`: Optional initial y-velocity (default: 0).
- `wᵢ`: Optional initial z-velocity (default: 0).
```
"""
function thermal_bubble_model(grid; uᵢ=0, vᵢ=0, wᵢ=0, qᵗ=0, microphysics=nothing)
    model = AtmosphereModel(grid; advection=WENO(), microphysics)
    N² = 1e-6
    r₀ = 2e3
    Δθ = 10  # K
    θ₀ = model.formulation.reference_state.potential_temperature
    g = model.thermodynamics.gravitational_acceleration
    dθdz = N² * θ₀ / g

    function θᵢ(x, y, z)
        θ̄ = θ₀ + dθdz * z
        r = sqrt(x^2 + y^2 + z^2)
        θ′ = Δθ * max(0, 1 - r / r₀)
        return θ̄ + θ′
    end
    
    set!(model, θ=θᵢ, u=uᵢ, v=vᵢ, w=wᵢ, qᵗ=qᵗ)

    return model
end

@testset "Energy conservation with thermal bubble [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; 
                           size = (32, 1, 32), 
                           x = (-10e3, 10e3), 
                           y = (-10e3, 10e3), 
                           z = (-3e3, 7e3),
                           topology = (Periodic, Periodic, Bounded),
                           halo = (5, 5, 5))
    
    for microphysics in (nothing, WarmPhaseMicrophysics())
        @testset let microphysics=microphysics
            # Set (moist) thermal bubble initial condition
            model = thermal_bubble_model(grid; qᵗ=1e-3, microphysics)
    
            # Compute initial total energy
            ∫ρe = Field(Integral(model.energy_density))
            compute!(∫ρe)
            E₀ = @allowscalar first(∫ρe)
    
            # Time step the model
            Nt = 10
    
            for step in 1:Nt
                time_step!(model, 1)
                compute!(∫ρe)
                E = @allowscalar first(∫ρe)
                @test E ≈ E₀
            end
        end
    end
end


@testset "Momentum conservation with spherical thermal bubble [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; 
                           size=(16, 16, 16), 
                           x=(-10e3, 10e3),
                           y=(-10e3, 10e3),
                           z=(-3e3, 7e3),
                           topology=(Periodic, Periodic, Bounded),
                           halo=(5, 5, 5))
    
    # Set spherical thermal bubble initial condition with u velocity only
    model = thermal_bubble_model(grid, uᵢ=5, vᵢ=3, wᵢ=1)
    
    # Compute initial total u-momentum
    ∫ρu = Field(Integral(model.momentum.ρu))
    ∫ρv = Field(Integral(model.momentum.ρv))
    ∫ρw = Field(Integral(model.momentum.ρw))
    Px₀ = @allowscalar first(∫ρu)
    Py₀ = @allowscalar first(∫ρv)
    Pz₀ = @allowscalar first(∫ρw)
    
    # Time step the model
    Nt = 10
    
    for step in 1:Nt
        time_step!(model, 1)
        compute!(∫ρu)
        compute!(∫ρv)
        compute!(∫ρw)
        Px = @allowscalar first(∫ρu)
        Py = @allowscalar first(∫ρv)
        Pz = @allowscalar first(∫ρw)
        @test Px ≈ Px₀
        @test Py ≈ Py₀
        @test Pz ≈ Pz₀
    end
end
