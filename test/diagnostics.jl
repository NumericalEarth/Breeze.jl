using Test
using Breeze
using Oceananigans
using GPUArraysCore: @allowscalar

@testset "Hydrostatic pressure computation [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 16), x=(0, 1000), y=(0, 1000), z=(0, 1000))
    constants = ThermodynamicConstants(FT)
    
    p₀ = FT(101325)
    θ₀ = FT(288)
    reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation)
    
    # Set a simple initial condition with constant potential temperature
    # This should result in zero buoyancy perturbation (ρ = ρᵣ)
    set!(model; θ = θ₀)
    
    # Update model state to compute temperature and other diagnostics
    update_state!(model; compute_tendencies=false)
    
    # Create a pressure field for hydrostatic pressure
    ph = CenterField(grid)
    
    # Compute hydrostatic pressure
    Breeze.AtmosphereModels.compute_hydrostatic_pressure!(ph, model)
    
    # For a state in hydrostatic balance with the reference state,
    # the hydrostatic pressure perturbation should be approximately zero
    # (allowing for numerical errors)
    ph_max = @allowscalar maximum(abs, ph)
    @test ph_max < 100 * eps(FT)  # Very small tolerance for numerical errors
    
    # Test with a known buoyancy perturbation
    # Set a temperature perturbation that creates a known buoyancy
    ΔT = FT(10)  # 10 K temperature perturbation
    set!(model; T = (x, y, z) -> θ₀ + ΔT)
    
    # Update model state again
    update_state!(model; compute_tendencies=false)
    
    # Compute hydrostatic pressure again
    Breeze.AtmosphereModels.compute_hydrostatic_pressure!(ph, model)
    
    # The hydrostatic pressure should increase downward (more negative at bottom)
    # since warmer air is less dense (positive buoyancy)
    _, _, Nz = size(grid)
    k_top = Nz
    k_bottom = 1
    
    ph_top = @allowscalar ph[1, 1, k_top]
    ph_bottom = @allowscalar ph[1, 1, k_bottom]
    
    # Bottom should be more negative (or less positive) than top
    # because we integrate downward and accumulate negative pressure
    @test ph_bottom < ph_top
    
    # Test that pressure decreases monotonically downward
    # (since we're integrating buoyancy downward)
    for k in 1:Nz-1
        ph_k = @allowscalar ph[1, 1, k]
        ph_kp1 = @allowscalar ph[1, 1, k+1]
        # Pressure should decrease (become more negative) as we go down
        @test ph_k <= ph_kp1 + 100 * eps(FT)  # Allow small numerical errors
    end
    
    # Test that the pressure field is finite everywhere
    @test all(isfinite, ph)
end

