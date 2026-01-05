using Breeze
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.Fields: interior
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

using Breeze.Thermodynamics:
    MoistureMassFractions,
    LiquidIcePotentialTemperatureState,
    temperature,
    mixture_heat_capacity

#####
##### Helper functions for computing domain-integrated quantities
#####

"""
    total_moisture_mass(model)

Compute total moisture mass from ρqᵗ field: ∫ρqᵗ dV.
This is the prognostic total moisture that the model tracks.
"""
function total_moisture_mass(model)
    grid = model.grid
    ρᵣ = model.dynamics.reference_state.density
    qᵗ = model.specific_moisture
    
    Nx, Ny, Nz = size(grid)
    Δx = grid.Lx / Nx
    Δy = grid.Ly / Ny
    Δz = grid.Lz / Nz
    
    total_mass = zero(eltype(grid))
    
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ρ = @allowscalar ρᵣ[i, j, k]
        q = @allowscalar qᵗ[i, j, k]
        total_mass += ρ * q * Δx * Δy * Δz
    end
    
    return total_mass
end

"""
    total_rain_mass(model)

Compute total rain mass from ρqʳ field: ∫ρqʳ dV.
"""
function total_rain_mass(model)
    grid = model.grid
    ρqʳ = model.microphysical_fields.ρqʳ
    
    Nx, Ny, Nz = size(grid)
    Δx = grid.Lx / Nx
    Δy = grid.Ly / Ny
    Δz = grid.Lz / Nz
    
    total_mass = zero(eltype(grid))
    
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ρq = @allowscalar ρqʳ[i, j, k]
        total_mass += ρq * Δx * Δy * Δz
    end
    
    return total_mass
end

"""
    total_cloud_liquid_mass(model)

Compute total cloud liquid mass from ρqᶜˡ field: ∫ρqᶜˡ dV.
"""
function total_cloud_liquid_mass(model)
    grid = model.grid
    ρqᶜˡ = model.microphysical_fields.ρqᶜˡ
    
    Nx, Ny, Nz = size(grid)
    Δx = grid.Lx / Nx
    Δy = grid.Ly / Ny
    Δz = grid.Lz / Nz
    
    total_mass = zero(eltype(grid))
    
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ρq = @allowscalar ρqᶜˡ[i, j, k]
        total_mass += ρq * Δx * Δy * Δz
    end
    
    return total_mass
end

"""
    bottom_cell_theta(model, i=1, j=1)

Get liquid-ice potential temperature at the bottom cell.
"""
function bottom_cell_theta(model, i=1, j=1)
    return @allowscalar model.formulation.potential_temperature[i, j, 1]
end

"""
    bottom_cell_theta_density(model, i=1, j=1)

Get liquid-ice potential temperature density (ρθ) at the bottom cell.
"""
function bottom_cell_theta_density(model, i=1, j=1)
    return @allowscalar model.formulation.potential_temperature_density[i, j, 1]
end

#####
##### Conservation tests - These tests verify physics consistency
#####
##### IMPORTANT: The current 1M microphysics implementation does NOT implement
##### microphysical tendencies for Val(:ρqᵗ) or Val(:ρθ) when there is a 
##### precipitation flux. The tests below are designed to DETECT this issue.
#####

@testset "Total water conservation with ImpenetrableBoundaryCondition [$(FT)]" for FT in (Float32, Float64)
    # Test that total water is conserved when precipitation cannot exit the domain.
    # With ImpenetrableBoundaryCondition, rain collects at the bottom but doesn't leave,
    # so total water (vapor + cloud liquid + rain) should be conserved.
    
    Oceananigans.defaults.FloatType = FT
    
    Nz = 4
    Lz = FT(400)
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, Lz),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Use ImpenetrableBoundaryCondition to prevent rain from exiting
    microphysics = OneMomentCloudMicrophysics(FT; precipitation_boundary_condition=ImpenetrableBoundaryCondition())
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    # Set initial conditions with cloud liquid and some rain
    θ₀ = FT(300)
    qᵗ₀ = FT(0.020)
    qᶜˡ₀ = FT(0.002)
    qʳ₀ = FT(0.001)
    
    set!(model; θ=θ₀, qᵗ=qᵗ₀, qᶜˡ=qᶜˡ₀, qʳ=qʳ₀)
    
    # Compute initial total moisture mass (from ρqᵗ)
    total_moisture_initial = total_moisture_mass(model)
    
    # Time step
    τ = microphysics.categories.cloud_liquid.τ_relax
    Δt = τ / 10
    Nt = 50
    
    for _ in 1:Nt
        time_step!(model, Δt)
    end
    
    # Compute final total moisture mass
    total_moisture_final = total_moisture_mass(model)
    
    # With ImpenetrableBoundaryCondition, no water leaves, so ρqᵗ should be conserved
    rtol = FT == Float32 ? FT(1e-3) : FT(1e-6)
    @test isapprox(total_moisture_final, total_moisture_initial; rtol)
    
    # Verify terminal velocity is zero at bottom
    wʳ_bottom = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    @test wʳ_bottom == 0
end

@testset "Moisture budget consistency when rain sediments out [$(FT)]" for FT in (Float32, Float64)
    # This test checks for the PHYSICS BUG in the current implementation:
    # When rain sediments out of the domain, the prognostic ρqᵗ should decrease,
    # but the current implementation does NOT update ρqᵗ when rain exits.
    #
    # The test verifies that:
    # 1. Rain (ρqʳ) decreases as it sediments out
    # 2. Total moisture (ρqᵗ) SHOULD decrease by the same amount, but currently it DOESN'T
    # 3. This creates an inconsistency: qᵗ ≠ qᵛ + qᶜˡ + qʳ after sedimentation
    
    Oceananigans.defaults.FloatType = FT
    
    Nz = 4
    Lz = FT(400)
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, Lz),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Open boundary (default) - precipitation CAN exit
    microphysics = OneMomentCloudMicrophysics(FT)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    # Set initial conditions with significant rain that will sediment out
    θ₀ = FT(300)
    qᵗ₀ = FT(0.020)
    qᶜˡ₀ = FT(0.000)  # No cloud liquid to simplify
    qʳ₀ = FT(0.005)   # 5 g/kg rain
    
    set!(model; θ=θ₀, qᵗ=qᵗ₀, qᶜˡ=qᶜˡ₀, qʳ=qʳ₀)
    
    # Record initial values
    total_moisture_initial = total_moisture_mass(model)
    total_rain_initial = total_rain_mass(model)
    
    # Run for enough time to let rain sediment through and exit
    # Terminal velocity ~5-10 m/s, domain is 400m, so ~40-80s to fall through
    Δt = FT(2.0)
    Nt = 100  # 200 seconds
    
    for _ in 1:Nt
        time_step!(model, Δt)
    end
    
    # Compute final values
    total_moisture_final = total_moisture_mass(model)
    total_rain_final = total_rain_mass(model)
    
    # Rain should have decreased (some has sedimented out)
    rain_decrease = total_rain_initial - total_rain_final
    @test rain_decrease > FT(0)  # Rain has left the domain
    
    # BUG CHECK: The prognostic ρqᵗ SHOULD have decreased by the same amount,
    # but in the current implementation it does NOT because there's no
    # microphysical_tendency for Val(:ρqᵗ) to account for precipitation flux.
    moisture_decrease = total_moisture_initial - total_moisture_final
    
    # This test SHOULD pass if physics is correct: moisture should decrease when rain leaves
    # Currently this will FAIL because ρqᵗ is not properly updated when rain exits
    @test isapprox(moisture_decrease, rain_decrease; rtol=FT(0.1))
end

@testset "Local moisture consistency qᵗ = qᵛ + qᶜˡ + qʳ [$(FT)]" for FT in (Float32, Float64)
    # This test checks that at each grid cell, the moisture budget is consistent:
    # qᵗ = qᵛ + qᶜˡ + qʳ (or equivalently qᵛ = qᵗ - qᶜˡ - qʳ)
    #
    # If sedimentation is handled correctly, this should always hold.
    # If there's a bug (ρqᵗ not updated when rain exits), this will be violated.
    
    Oceananigans.defaults.FloatType = FT
    
    Nz = 4
    Lz = FT(400)
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, Lz),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Open boundary - precipitation can exit
    microphysics = OneMomentCloudMicrophysics(FT)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    # Set initial conditions
    θ₀ = FT(300)
    qᵗ₀ = FT(0.020)
    qᶜˡ₀ = FT(0.001)
    qʳ₀ = FT(0.005)
    
    set!(model; θ=θ₀, qᵗ=qᵗ₀, qᶜˡ=qᶜˡ₀, qʳ=qʳ₀)
    
    # Check initial consistency
    ρᵣ = reference_state.density
    for k in 1:Nz
        qᵗ = @allowscalar model.specific_moisture[1, 1, k]
        qᶜˡ = @allowscalar model.microphysical_fields.qᶜˡ[1, 1, k]
        qʳ = @allowscalar model.microphysical_fields.qʳ[1, 1, k]
        qᵛ = @allowscalar model.microphysical_fields.qᵛ[1, 1, k]
        
        # Initially qᵗ = qᵛ + qᶜˡ + qʳ should hold
        @test isapprox(qᵗ, qᵛ + qᶜˡ + qʳ; rtol=FT(1e-6))
    end
    
    # Time step to let rain sediment
    Δt = FT(2.0)
    Nt = 100
    
    for _ in 1:Nt
        time_step!(model, Δt)
    end
    
    # Check final consistency - THIS IS WHERE THE BUG SHOWS UP
    # After sedimentation, qᵗ should still equal qᵛ + qᶜˡ + qʳ at each cell
    max_inconsistency = zero(FT)
    for k in 1:Nz
        qᵗ = @allowscalar model.specific_moisture[1, 1, k]
        qᶜˡ = @allowscalar model.microphysical_fields.qᶜˡ[1, 1, k]
        qʳ = @allowscalar model.microphysical_fields.qʳ[1, 1, k]
        qᵛ = @allowscalar model.microphysical_fields.qᵛ[1, 1, k]
        
        # The sum of components should equal total moisture
        sum_components = qᵛ + qᶜˡ + qʳ
        inconsistency = abs(qᵗ - sum_components)
        max_inconsistency = max(max_inconsistency, inconsistency)
    end
    
    # This test SHOULD pass if physics is correct
    # It will FAIL if ρqᵗ is not properly updated when rain sediments
    @test max_inconsistency < FT(1e-6)
end

@testset "θˡⁱ consistency when rain sediments out [$(FT)]" for FT in (Float32, Float64)
    # This test checks the liquid-ice potential temperature budget.
    #
    # θˡⁱ is defined as: T/Π - (ℒˡᵣ qˡ + ℒⁱᵣ qⁱ)/(cᵖᵐ Π)
    #
    # When liquid water (rain) exits the domain via sedimentation, it carries enthalpy.
    # The microphysical_tendency for Val(:ρθ) should account for this.
    # Currently it returns 0, which means θˡⁱ doesn't change correctly.
    #
    # The physics: if we remove liquid water from a parcel without removing the
    # associated enthalpy (latent heat), the temperature should change.
    # But if microphysical_tendency(:ρθ) = 0, then ρθ only changes via advection,
    # not via the precipitation flux.
    
    Oceananigans.defaults.FloatType = FT
    
    Nz = 4
    Lz = FT(400)
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, Lz),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Open boundary - precipitation can exit
    microphysics = OneMomentCloudMicrophysics(FT)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    # Set up: rain in the bottom cell that will sediment out
    θ₀ = FT(300)
    qᵗ₀ = FT(0.020)
    qᶜˡ₀ = FT(0.000)
    qʳ₀ = FT(0.005)
    
    set!(model; θ=θ₀, qᵗ=qᵗ₀, qᶜˡ=qᶜˡ₀, qʳ=qʳ₀)
    
    # Record initial values at bottom cell
    θ_initial = bottom_cell_theta(model)
    ρθ_initial = bottom_cell_theta_density(model)
    qʳ_initial = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    
    # Time step to let rain sediment out
    Δt = FT(2.0)
    Nt = 50
    
    for _ in 1:Nt
        time_step!(model, Δt)
    end
    
    # Check final values
    θ_final = bottom_cell_theta(model)
    ρθ_final = bottom_cell_theta_density(model)
    qʳ_final = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    
    # Rain should have changed (either decreased from sedimentation or increased from above)
    Δqʳ = qʳ_final - qʳ_initial
    
    # If significant rain change occurred, θˡⁱ should also change to maintain energy consistency
    # The change in θˡⁱ due to liquid water change should be approximately:
    # Δθˡⁱ ≈ ℒˡᵣ Δqˡ / (cᵖᵐ Π) (with appropriate signs)
    #
    # If microphysical_tendency(:ρθ) = 0, then θˡⁱ won't change correctly
    # This test checks that θˡⁱ changes are consistent with moisture changes
    
    if abs(Δqʳ) > FT(1e-6)
        ℒˡᵣ = constants.liquid.reference_latent_heat
        cᵖᵈ = constants.dry_air.heat_capacity
        # Approximate expected θ change (simplified, ignoring Π variations)
        expected_Δθ_magnitude = abs(ℒˡᵣ * Δqʳ / cᵖᵈ)
        actual_Δθ = abs(θ_final - θ_initial)
        
        # The change in θˡⁱ should be non-negligible if rain changed significantly
        # This will FAIL if microphysical_tendency(:ρθ) returns 0
        @test actual_Δθ > expected_Δθ_magnitude * FT(0.1) || abs(Δqʳ) < FT(1e-4)
    end
    
    # Temperature should remain physically reasonable
    T_final = @allowscalar model.temperature[1, 1, 1]
    @test T_final > FT(250)
    @test T_final < FT(350)
end

@testset "θˡⁱ conservation with ImpenetrableBoundaryCondition [$(FT)]" for FT in (Float32, Float64)
    # With ImpenetrableBoundaryCondition, no precipitation exits.
    # In a uniform column with no advection, θˡⁱ should be conserved
    # during condensation/evaporation (which are reversible moist processes).
    
    Oceananigans.defaults.FloatType = FT
    
    Nz = 4
    Lz = FT(400)
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, Lz),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Use ImpenetrableBoundaryCondition
    microphysics = OneMomentCloudMicrophysics(FT; precipitation_boundary_condition=ImpenetrableBoundaryCondition())
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    # Set uniform initial conditions
    θ₀ = FT(300)
    qᵗ₀ = FT(0.015)
    qᶜˡ₀ = FT(0.001)
    qʳ₀ = FT(0.001)
    
    set!(model; θ=θ₀, qᵗ=qᵗ₀, qᶜˡ=qᶜˡ₀, qʳ=qʳ₀)
    
    # Get initial ρθ at bottom cell
    ρθ_initial = bottom_cell_theta_density(model)
    θ_initial = bottom_cell_theta(model)
    
    # Time step
    τ = microphysics.categories.cloud_liquid.τ_relax
    Δt = τ / 10
    Nt = 30
    
    for _ in 1:Nt
        time_step!(model, Δt)
    end
    
    # Get final values
    ρθ_final = bottom_cell_theta_density(model)
    θ_final = bottom_cell_theta(model)
    
    # With no precipitation flux and uniform conditions, θˡⁱ should be conserved
    rtol = FT == Float32 ? FT(1e-2) : FT(1e-4)
    @test isapprox(ρθ_final, ρθ_initial; rtol)
    @test isapprox(θ_final, θ_initial; rtol)
end
