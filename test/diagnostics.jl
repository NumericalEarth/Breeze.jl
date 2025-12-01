using Test
using Breeze
using Breeze.Thermodynamics: dry_air_gas_constant
using Oceananigans
using GPUArraysCore: @allowscalar

@testset "Hydrostatic pressure computation [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 16), x=(0, 1000), y=(0, 1000), z=(0, 1000))
    constants = ThermodynamicConstants()
    
    p₀, θ₀ = 101325, 288
    reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation)

    ρᵣ = reference_state.density
    pᵣ = reference_state.pressure
    set!(ρᵣ, 1)
    set!(pᵣ, 1)
    cᵖᵈ = constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration
    eᵢ(x, y, z) = cᵖᵈ * θ₀ + g * z
    set!(model; e=eᵢ)
    
    # Create a pressure field for hydrostatic pressure
    ph = CenterField(grid)
    Breeze.AtmosphereModels.compute_hydrostatic_pressure!(ph, model)
    
    Rᵈ = dry_air_gas_constant(constants)
    z = Field{Nothing, Nothing, Center}(grid)
    set!(z, z -> z)

    # ∂z p = b = g * (ρᵣ - ρ)
    dz_ph_expected = ZFaceField(grid)
    set!(dz_ph_expected, - g * (pᵣ / (Rᵈ * θ₀) - ρᵣ))
    dz_ph = Field(∂z(ph))
    @test interior(dz_ph, 1, 1, 2:grid.Nz) ≈ interior(dz_ph_expected, 1, 1, 2:grid.Nz)
end

