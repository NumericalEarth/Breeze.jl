using Breeze
using CloudMicrophysics
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, surface_precipitation_flux

#####
##### Helper functions
#####

"""Compute total moisture mass: ∫ρqᵗ dV"""
total_moisture_mass(model) = @allowscalar Field(Integral(model.moisture_density))[]

"""Compute column-integrated potential temperature density: ∫ρθ dV"""
column_integrated_rho_theta(model) = @allowscalar Field(Integral(model.formulation.potential_temperature_density))[]

"""Get θˡⁱ at the bottom cell"""
bottom_cell_theta(model) = @allowscalar model.formulation.potential_temperature[1, 1, 1]



#####
##### Test setup helper
#####

function setup_test_model(FT; precipitation_boundary_condition=nothing)
    Nz = 4
    Lz = FT(400)
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, Lz),
                           topology=(Periodic, Periodic, Bounded))
    
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    
    if isnothing(precipitation_boundary_condition)
        microphysics = OneMomentCloudMicrophysics(FT)
    else
        microphysics = OneMomentCloudMicrophysics(FT; precipitation_boundary_condition)
    end
    
    return AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics), constants
end

#####
##### Conservation tests
#####

@testset "Total water conservation with ImpenetrableBoundaryCondition [$(FT)]" for FT in (Float32, Float64)
    # With ImpenetrableBoundaryCondition, rain collects at bottom but doesn't leave,
    # so total water should be conserved.
    
    Oceananigans.defaults.FloatType = FT
    model, _ = setup_test_model(FT; precipitation_boundary_condition=ImpenetrableBoundaryCondition())
    
    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0.002, qʳ=0.001)
    total_moisture_initial = total_moisture_mass(model)
    
    τ = model.microphysics.categories.cloud_liquid.τ_relax
    for _ in 1:50
        time_step!(model, τ / 10)
    end
    
    total_moisture_final = total_moisture_mass(model)
    
    rtol = FT == Float32 ? 1f-3 : 1e-6
    @test isapprox(total_moisture_final, total_moisture_initial; rtol)
    
    # Verify terminal velocity is zero at bottom (impenetrable)
    @test @allowscalar(model.microphysical_fields.wʳ[1, 1, 1]) == 0
end

@testset "Moisture decreases with open boundary [$(FT)]" for FT in (Float32, Float64)
    # With open boundary, total moisture decreases when rain sediments out.
    
    Oceananigans.defaults.FloatType = FT
    model, _ = setup_test_model(FT)
    
    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0.000, qʳ=0.005)
    total_moisture_initial = total_moisture_mass(model)
    
    for _ in 1:100
        time_step!(model, FT(2.0))
    end
    
    total_moisture_final = total_moisture_mass(model)
    
    # Moisture must decrease when precipitation exits the domain
    @test total_moisture_final < total_moisture_initial
end


@testset "θˡⁱ changes at bottom cell when precipitation exits [$(FT)]" for FT in (Float32, Float64)
    # When rain exits through the bottom boundary:
    # - Bottom cell loses liquid water (qˡ decreases)
    # - θˡⁱ = T/Π - ℒˡᵣ qˡ/(cᵖᵐ Π), so θˡⁱ INCREASES
    
    Oceananigans.defaults.FloatType = FT
    model, constants = setup_test_model(FT)
    
    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0.000, qʳ=0.005)
    
    θ_initial = bottom_cell_theta(model)
    qʳ_initial = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    
    for _ in 1:50
        time_step!(model, 2)
    end
    
    θ_final = bottom_cell_theta(model)
    qʳ_final = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    
    # Verify precipitation can exit (open BC)
    @test @allowscalar(model.microphysical_fields.wʳ[1, 1, 1]) < FT(0)
    
    # Check θˡⁱ response to rain change
    Δqʳ = qʳ_final - qʳ_initial
    if abs(Δqʳ) > FT(1e-6)
        ℒˡᵣ = constants.liquid.reference_latent_heat
        cᵖᵈ = constants.dry_air.heat_capacity
        expected_Δθ = abs(ℒˡᵣ * Δqʳ / cᵖᵈ)
        actual_Δθ = abs(θ_final - θ_initial)
        @test actual_Δθ > expected_Δθ * FT(0.1) || abs(Δqʳ) < FT(1e-4)
    end
    
    # Temperature should remain physical
    T_final = @allowscalar model.temperature[1, 1, 1]
    @test FT(250) < T_final < FT(350)
end

@testset "Column-integrated θˡⁱ conservation with ImpenetrableBoundaryCondition [$(FT)]" for FT in (Float32, Float64)
    # With ImpenetrableBoundaryCondition, column-integrated ρθˡⁱ should be conserved.
    # Individual cells can still change as rain redistributes internally.
    
    Oceananigans.defaults.FloatType = FT
    model, _ = setup_test_model(FT; precipitation_boundary_condition=ImpenetrableBoundaryCondition())
    
    set!(model; θ=300, qᵗ=0.015, qᶜˡ=0.001, qʳ=0.001)
    
    ρθ_column_initial = column_integrated_rho_theta(model)
    
    τ = model.microphysics.categories.cloud_liquid.τ_relax
    for _ in 1:30
        time_step!(model, τ / 10)
    end
    
    ρθ_column_final = column_integrated_rho_theta(model)
    
    rtol = FT == Float32 ? FT(1e-2) : FT(1e-4)
    @test isapprox(ρθ_column_final, ρθ_column_initial; rtol)
end
