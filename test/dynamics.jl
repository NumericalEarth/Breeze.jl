using Breeze
using Breeze.Microphysics
using Oceananigans
using Test
using GPUArraysCore: @allowscalar

"""
    thermal_bubble_model(grid; Δθ=10, uᵢ=0, vᵢ=0, wᵢ=0)

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
function thermal_bubble_model(grid; Δθ=10, N²=1e-6, uᵢ=0, vᵢ=0, wᵢ=0, qᵗ=0, microphysics=nothing)
    model = AtmosphereModel(grid; advection=WENO(), microphysics)
    r₀ = 2e3
    θ₀ = model.dynamics.reference_state.potential_temperature
    g = model.thermodynamic_constants.gravitational_acceleration

    function θᵢ(x, y, z)
        θ̄ = θ₀ * exp(N² * z / g)
        r = sqrt(x^2 + y^2 + z^2)
        θ′ = Δθ * max(0, 1 - r / r₀)
        return θ̄ + θ′
    end

    set!(model, θ=θᵢ, u=uᵢ, v=vᵢ, w=wᵢ, qᵗ=qᵗ)

    return model
end

Δt = 1e-3
tol = 1e-6

@testset "Horizontal momentum conservation with spherical thermal bubble [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch;
                           size = (16, 16, 16),
                           x = (-10e3, 10e3),
                           y = (-10e3, 10e3),
                           z = (-3e3, 7e3),
                           topology = (Periodic, Periodic, Bounded),
                           halo = (5, 5, 5))

    # Set spherical thermal bubble initial condition with sheared horizontal velocities
    uᵢ(x, y, z) = 5 # * (z + 3e3) / 10e3
    vᵢ(x, y, z) = 3 # * (z + 3e3) / 10e3
    model = thermal_bubble_model(grid; uᵢ, vᵢ)

    # Compute initial total u-momentum
    ∫ρu = Field(Integral(model.momentum.ρu))
    ∫ρv = Field(Integral(model.momentum.ρv))
    Px₀ = @allowscalar first(∫ρu)
    Py₀ = @allowscalar first(∫ρv)

    # Time step the model
    Nt = 10

    for step in 1:Nt
        time_step!(model, Δt)
        compute!(∫ρu)
        compute!(∫ρv)
        Px = @allowscalar first(∫ρu)
        Py = @allowscalar first(∫ρv)
        @test Px ≈ Px₀
        @test Py ≈ Py₀
    end
end

# Float32 is excluded because the integrated momentum over the large domain (4e12 m³)
# accumulates roundoff errors to O(100), making the test meaningless for Float32.
@testset "Vertical momentum conservation for neutral initial condition [$(FT)]" for FT in (Float64,)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch;
                           size = (16, 5, 16),
                           x = (-10e3, 10e3),
                           y = (-10e3, 10e3),
                           z = (-5e3, 5e3),
                           topology = (Bounded, Periodic, Bounded),
                           halo = (5, 5, 5))

    # Set spherical thermal bubble initial condition with u velocity only
    wᵢ(x, y, z) = 0 # * x / 20e3 * exp(-z^2 / (2 * 1e3^2))
    model = thermal_bubble_model(grid; wᵢ, Δθ=0, N²=0)

    # Compute initial total u-momentum
    ∫ρu = Field(Integral(model.momentum.ρu))
    ∫ρw = Field(Integral(model.momentum.ρw))
    Px₀ = @allowscalar first(∫ρu)
    Pz₀ = @allowscalar first(∫ρw)

    # Time step the model
    Nt = 10

    # Use absolute tolerance for comparisons with zero.
    # Float32 has larger roundoff errors due to limited precision.
    atol = FT == Float32 ? FT(1e-2) : FT(1e-6)

    for step in 1:Nt
        time_step!(model, Δt)
        compute!(∫ρu)
        compute!(∫ρw)
        Px = @allowscalar first(∫ρu)
        Pz = @allowscalar first(∫ρw)
        @test Px ≈ Px₀
        @test isapprox(Pz, Pz₀, atol=atol)
    end
end
