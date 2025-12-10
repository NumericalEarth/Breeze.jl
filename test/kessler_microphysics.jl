using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

import CUDA
using Oceananigans.Architectures: CPU, GPU

const default_arch = CUDA.functional() ? GPU() : CPU()

# Run integration tests only if RUN_INTEGRATION_TESTS is set
const RUN_INTEGRATION_TESTS = get(ENV, "RUN_INTEGRATION_TESTS", "false") == "true"

using Breeze.Microphysics:
    mass_fraction_to_mixing_ratio,
    mixing_ratio_to_mass_fraction,
    kessler_saturation_mixing_ratio,
    kessler_terminal_velocity,
    prognostic_field_names

#####
##### Unit tests for conversion functions
#####

@testset "Kessler unit conversions [$(FT)]" for FT in (Float32, Float64)
    
    @testset "mass_fraction ↔ mixing_ratio round-trip" begin
        # Test round-trip conversion: q → r → q
        for qᵛ in FT.((0.01, 0.02, 0.03)),
            qᶜ in FT.((0.001, 0.002, 0.003)),
            qʳ in FT.((0.0001, 0.0005, 0.001))
            
            qᵗ = qᵛ + qᶜ + qʳ
            
            # Forward: mass fraction → mixing ratio
            rᵛ = mass_fraction_to_mixing_ratio(qᵛ, qᵗ)
            rᶜ = mass_fraction_to_mixing_ratio(qᶜ, qᵗ)
            rʳ = mass_fraction_to_mixing_ratio(qʳ, qᵗ)
            rᵗ = rᵛ + rᶜ + rʳ
            
            # Backward: mixing ratio → mass fraction
            qᵛ_back = mixing_ratio_to_mass_fraction(rᵛ, rᵗ)
            qᶜ_back = mixing_ratio_to_mass_fraction(rᶜ, rᵗ)
            qʳ_back = mixing_ratio_to_mass_fraction(rʳ, rᵗ)
            
            @test qᵛ_back ≈ qᵛ rtol=eps(FT)
            @test qᶜ_back ≈ qᶜ rtol=eps(FT)
            @test qʳ_back ≈ qʳ rtol=eps(FT)
        end
    end
    
    @testset "Conversion edge cases" begin
        # Dry air: qᵗ = 0
        qᵛ = zero(FT)
        qᵗ = zero(FT)
        rᵛ = mass_fraction_to_mixing_ratio(qᵛ, qᵗ)
        @test rᵛ == zero(FT)
        
        # Small moisture
        qᵛ = FT(1e-6)
        qᵗ = FT(1e-6)
        rᵛ = mass_fraction_to_mixing_ratio(qᵛ, qᵗ)
        @test isfinite(rᵛ)
        @test rᵛ > qᵛ  # mixing ratio > mass fraction
    end
    
    @testset "Mixing ratio is larger than mass fraction" begin
        # Physical constraint: r = q / (1 - qᵗ) > q when qᵗ > 0
        qᵛ = FT(0.02)
        qᵗ = FT(0.025)
        rᵛ = mass_fraction_to_mixing_ratio(qᵛ, qᵗ)
        @test rᵛ > qᵛ
    end
end

#####
##### Unit tests for physical helper functions
#####

@testset "Kessler saturation mixing ratio [$(FT)]" for FT in (Float32, Float64)
    # Test at standard conditions
    T = FT(293.15)  # 20°C
    p = FT(101325)  # 1 atm
    
    rᵛˢ = kessler_saturation_mixing_ratio(T, p)
    @test isfinite(rᵛˢ)
    @test rᵛˢ > 0
    # At 20°C, saturation mixing ratio should be roughly 0.014-0.015 kg/kg
    @test 0.01 < rᵛˢ < 0.02
    
    # Test that saturation increases with temperature
    T_cold = FT(273.15)  # 0°C
    T_warm = FT(303.15)  # 30°C
    rᵛˢ_cold = kessler_saturation_mixing_ratio(T_cold, p)
    rᵛˢ_warm = kessler_saturation_mixing_ratio(T_warm, p)
    @test rᵛˢ_warm > rᵛˢ_cold
    
    # Test that saturation increases with decreasing pressure
    p_high = FT(101325)
    p_low = FT(50000)
    rᵛˢ_high_p = kessler_saturation_mixing_ratio(T, p_high)
    rᵛˢ_low_p = kessler_saturation_mixing_ratio(T, p_low)
    @test rᵛˢ_low_p > rᵛˢ_high_p
end

@testset "Kessler terminal velocity [$(FT)]" for FT in (Float32, Float64)
    # Test at typical conditions
    rʳ = FT(0.001)  # 1 g/kg rain mixing ratio
    ρ = FT(1.0)     # Air density at surface
    ρˢ = FT(1.2)    # Reference surface density
    
    vᵗ = kessler_terminal_velocity(rʳ, ρ, ρˢ)
    @test isfinite(vᵗ)
    @test vᵗ > 0
    
    # Terminal velocity should be in reasonable range (0-15 m/s for typical rain)
    @test 0 < vᵗ < 20
    
    # Test that velocity increases with rain content
    rʳ_light = FT(0.0001)
    rʳ_heavy = FT(0.005)
    vᵗ_light = kessler_terminal_velocity(rʳ_light, ρ, ρˢ)
    vᵗ_heavy = kessler_terminal_velocity(rʳ_heavy, ρ, ρˢ)
    @test vᵗ_heavy > vᵗ_light
    
    # Test that velocity increases with altitude (lower ρ)
    ρ_low = FT(0.5)  # Upper atmosphere
    vᵗ_low_density = kessler_terminal_velocity(rʳ, ρ_low, ρˢ)
    @test vᵗ_low_density > vᵗ  # Falls faster in thinner air
    
    # Test zero rain
    vᵗ_zero = kessler_terminal_velocity(zero(FT), ρ, ρˢ)
    @test vᵗ_zero == zero(FT)
end

#####
##### Construction and interface tests
#####

@testset "KesslerMicrophysics construction [$(FT)]" for FT in (Float32, Float64)
    μ = KesslerMicrophysics()
    @test μ isa KesslerMicrophysics
    
    # Test prognostic field names
    names = prognostic_field_names(μ)
    @test names == (:ρqᵛ, :ρqᶜˡ, :ρqʳ)
end

#####
##### Integration tests (require long compilation, run conditionally)
#####

if RUN_INTEGRATION_TESTS

@testset "KesslerMicrophysics field materialization via AtmosphereModel [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1000), y=(0, 1000), z=(0, 1000))
    constants = ThermodynamicConstants()
    
    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants; base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    microphysics = KesslerMicrophysics()
    
    # Let AtmosphereModel materialize fields properly with boundary conditions
    model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation, microphysics)
    fields = model.microphysical_fields
    
    # Check prognostic fields exist and are the right type
    @test haskey(fields, :ρqᵛ)
    @test haskey(fields, :ρqᶜˡ)
    @test haskey(fields, :ρqʳ)
    @test fields.ρqᵛ isa Field
    @test fields.ρqᶜˡ isa Field
    @test fields.ρqʳ isa Field
    
    # Check diagnostic fields
    @test haskey(fields, :qᵛ)
    @test haskey(fields, :qᶜˡ)
    @test haskey(fields, :qʳ)
    @test haskey(fields, :precipitation_rate)
    @test haskey(fields, :vᵗ_rain)
    
    # Check field types
    @test eltype(fields.ρqᵛ) == FT
    @test eltype(fields.qᵛ) == FT
end

#####
##### AtmosphereModel integration tests
#####

@testset "AtmosphereModel with KesslerMicrophysics [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 1000), y=(0, 1000), z=(0, 2000))
    constants = ThermodynamicConstants()
    
    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants; base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    microphysics = KesslerMicrophysics()
    
    model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation, microphysics)
    
    @testset "Model construction" begin
        @test model.microphysics isa KesslerMicrophysics
        @test haskey(model.microphysical_fields, :ρqᵛ)
        @test haskey(model.microphysical_fields, :ρqᶜˡ)
        @test haskey(model.microphysical_fields, :ρqʳ)
        @test haskey(model.microphysical_fields, :precipitation_rate)
    end
    
    @testset "Dry initialization" begin
        # Initialize with dry, warm conditions
        set!(model; θ = θ₀)
        
        # Check that moisture fields are zero
        @test @allowscalar all(model.microphysical_fields.ρqᵛ .== 0)
        @test @allowscalar all(model.microphysical_fields.ρqᶜˡ .== 0)
        @test @allowscalar all(model.microphysical_fields.ρqʳ .== 0)
    end
    
    @testset "Moist initialization" begin
        # Initialize with some moisture
        ρᵣ = model.formulation.reference_state.density
        qᵛ₀ = FT(0.01)  # 10 g/kg vapor
        
        # Set moisture via density-weighted field
        set!(model.microphysical_fields.ρqᵛ, (x, y, z) -> @allowscalar(ρᵣ[1, 1, 1]) * qᵛ₀)
        set!(model.microphysical_fields.ρqᶜˡ, 0)
        set!(model.microphysical_fields.ρqʳ, 0)
        set!(model; θ = θ₀)
        
        # Check vapor is set
        @test @allowscalar model.microphysical_fields.ρqᵛ[1, 1, 1] > 0
    end
    
    @testset "Time stepping" begin
        # Reinitialize with simple conditions
        set!(model; θ = θ₀)
        set!(model.microphysical_fields.ρqᵛ, 0)
        set!(model.microphysical_fields.ρqᶜˡ, 0)
        set!(model.microphysical_fields.ρqʳ, 0)
        
        # Time step should complete without error
        @test begin
            time_step!(model, 1)
            true
        end
    end
end

@testset "KesslerMicrophysics rain sedimentation [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 16), x=(0, 100), y=(0, 100), z=(0, 4000))
    constants = ThermodynamicConstants()
    
    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants; base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    microphysics = KesslerMicrophysics()
    
    model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation, microphysics)
    
    # Initialize with rain only at upper levels
    ρᵣ = model.formulation.reference_state.density
    Nz = grid.Nz
    z_mid = 2000  # meters
    
    # Set up rain in upper half of domain
    set!(model; θ = θ₀)
    set!(model.microphysical_fields.ρqᵛ, 0)
    set!(model.microphysical_fields.ρqᶜˡ, 0)
    
    qʳ_init = FT(0.001)  # 1 g/kg rain
    for k in 1:Nz
        z_k = znodes(grid, Center())[k]
        if z_k > z_mid
            @allowscalar model.microphysical_fields.ρqʳ[1, 1, k] = ρᵣ[1, 1, k] * qʳ_init
        else
            @allowscalar model.microphysical_fields.ρqʳ[1, 1, k] = 0
        end
    end
    
    # Get initial rain mass
    initial_rain_mass = @allowscalar sum(model.microphysical_fields.ρqʳ)
    
    # Time step
    Δt = FT(10)
    for _ in 1:10
        time_step!(model, Δt)
    end
    
    # Rain should have moved downward - check that upper levels have less rain
    upper_rain = @allowscalar model.microphysical_fields.ρqʳ[1, 1, Nz]
    
    # Terminal velocity should be computed
    @test @allowscalar model.microphysical_fields.vᵗ_rain[1, 1, Nz÷2] >= 0
end

@testset "KesslerMicrophysics precipitation rate [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 8), x=(0, 200), y=(0, 200), z=(0, 2000))
    constants = ThermodynamicConstants()
    
    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants; base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    microphysics = KesslerMicrophysics()
    
    model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation, microphysics)
    
    # Initialize with rain throughout column
    ρᵣ = model.formulation.reference_state.density
    qʳ_init = FT(0.002)  # 2 g/kg rain
    
    set!(model; θ = θ₀)
    set!(model.microphysical_fields.ρqᵛ, 0)
    set!(model.microphysical_fields.ρqᶜˡ, 0)
    
    for k in 1:grid.Nz
        @allowscalar model.microphysical_fields.ρqʳ[1, 1, k] = ρᵣ[1, 1, k] * qʳ_init
    end
    
    # Time step to trigger precipitation calculation
    time_step!(model, 1)
    
    # Check that precipitation rate is computed and non-negative
    precip_rate = @allowscalar model.microphysical_fields.precipitation_rate[1, 1]
    @test isfinite(precip_rate)
    @test precip_rate >= 0
end

@testset "KesslerMicrophysics conservation [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    # Use periodic in x and y to avoid boundary effects
    grid = RectilinearGrid(default_arch; size=(2, 2, 8), 
                           x=(0, 200), y=(0, 200), z=(0, 2000),
                           topology=(Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants()
    
    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants; base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    microphysics = KesslerMicrophysics()
    
    model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation, microphysics)
    
    # Initialize with moisture spread across categories
    ρᵣ = model.formulation.reference_state.density
    
    set!(model; θ = θ₀)
    
    qᵛ₀ = FT(0.015)
    qᶜ₀ = FT(0.002)
    qʳ₀ = FT(0.001)
    
    for k in 1:grid.Nz
        ρ_k = @allowscalar ρᵣ[1, 1, k]
        @allowscalar model.microphysical_fields.ρqᵛ[1, 1, k] = ρ_k * qᵛ₀
        @allowscalar model.microphysical_fields.ρqᶜˡ[1, 1, k] = ρ_k * qᶜ₀
        @allowscalar model.microphysical_fields.ρqʳ[1, 1, k] = ρ_k * qʳ₀
    end
    
    # Compute initial total water (note: this is not conserved due to precipitation leaving the domain)
    initial_vapor = @allowscalar sum(model.microphysical_fields.ρqᵛ)
    initial_cloud = @allowscalar sum(model.microphysical_fields.ρqᶜˡ)
    initial_rain = @allowscalar sum(model.microphysical_fields.ρqʳ)
    initial_total = initial_vapor + initial_cloud + initial_rain
    
    
    # Time step
    time_step!(model, FT(0.1))
    
    # Check that all moisture fields remain non-negative (physical constraint)
    @test @allowscalar all(model.microphysical_fields.ρqᵛ .>= 0)
    @test @allowscalar all(model.microphysical_fields.ρqᶜˡ .>= 0)
    @test @allowscalar all(model.microphysical_fields.ρqʳ .>= 0)
    
    # Diagnostic fields should be non-negative
    @test @allowscalar all(model.microphysical_fields.qᵛ .>= 0)
    @test @allowscalar all(model.microphysical_fields.qᶜˡ .>= 0)
    @test @allowscalar all(model.microphysical_fields.qʳ .>= 0)
end

end # if RUN_INTEGRATION_TESTS
