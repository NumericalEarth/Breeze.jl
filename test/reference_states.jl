#####
##### Tests for ReferenceState, compute_reference_state!, and related functions
#####

using Breeze
using Breeze.Thermodynamics:
    compute_reference_state!,
    compute_hydrostatic_reference!,
    ExnerReferenceState,
    dry_air_gas_constant,
    hydrostatic_pressure,
    vapor_gas_constant,
    saturation_specific_humidity,
    PlanarLiquidSurface

using Breeze.AtmosphereModels:
    set_to_mean!,
    specific_humidity,
    liquid_mass_fraction,
    ice_mass_fraction,
    total_density

using Oceananigans
using Oceananigans.Fields: ZeroField
using GPUArraysCore: @allowscalar
using Test

@testset "ReferenceState [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 16), x=(0, 100), y=(0, 100), z=(0, 10000),
                           topology=(Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(FT)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    g = constants.gravitational_acceleration

    #####
    ##### Constructor: default moisture fields are ZeroField
    #####

    @testset "Default constructor produces ZeroField moisture" begin
        ref = ReferenceState(grid, constants)
        @test ref.vapor_mass_fraction isa ZeroField
        @test ref.liquid_mass_fraction isa ZeroField
        @test ref.ice_mass_fraction isa ZeroField
        @test ref.pressure isa Field
        @test ref.density isa Field
        @test ref.temperature isa Field
    end

    @testset "Constructor with moisture = 0 allocates Field" begin
        ref = ReferenceState(grid, constants; vapor_mass_fraction=0)
        @test ref.vapor_mass_fraction isa Field
        @test ref.liquid_mass_fraction isa ZeroField
        @test ref.ice_mass_fraction isa ZeroField
    end

    @testset "Constructor with moisture function allocates Field" begin
        qᵛ(z) = 0.01 * exp(-z / 2500)
        ref = ReferenceState(grid, constants; vapor_mass_fraction=qᵛ)
        @test ref.vapor_mass_fraction isa Field
        # Check the profile was set
        qᵛ₁ = @allowscalar ref.vapor_mass_fraction[1, 1, 1]
        @test qᵛ₁ > 0
    end

    #####
    ##### surface_density(ref::ReferenceState)
    #####

    @testset "surface_density" begin
        ref = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
        ρ₀ = surface_density(ref)
        # Surface density should be close to p₀ / (Rᵈ T₀) where T₀ ≈ θ₀ Π₀
        # For θ₀=300 and p₀=101325, Π₀ ≈ 1, so T₀ ≈ 300 K
        ρ_expected = FT(101325) / (Rᵈ * FT(300))  # approximate
        @test ρ₀ isa FT
        @test isapprox(ρ₀, ρ_expected; rtol=0.01)
    end

    #####
    ##### compute_reference_state! — dry isothermal atmosphere
    #####
    #
    # For constant T and zero moisture, the hydrostatic equation gives:
    #   p(z) = p₀ exp(-g z / (Rᵈ T))
    #   ρ(z) = p(z) / (Rᵈ T)

    @testset "compute_reference_state! dry isothermal" begin
        T₀ = FT(250)
        p₀ = FT(101325)
        ref = ReferenceState(grid, constants; surface_pressure=p₀, vapor_mass_fraction=0)

        compute_reference_state!(ref, T₀, FT(0), constants)

        Nz = grid.Nz
        for k in 1:Nz
            z = @allowscalar Oceananigans.Grids.znode(1, 1, k, grid, Center(), Center(), Center())
            p_exact = p₀ * exp(-g * z / (Rᵈ * T₀))
            ρ_exact = p_exact / (Rᵈ * T₀)

            p_ref = @allowscalar ref.pressure[1, 1, k]
            ρ_ref = @allowscalar ref.density[1, 1, k]
            T_ref = @allowscalar ref.temperature[1, 1, k]

            @test T_ref ≈ T₀
            @test isapprox(p_ref, p_exact; rtol=FT(1e-4))
            @test isapprox(ρ_ref, ρ_exact; rtol=FT(1e-4))
        end
    end

    #####
    ##### compute_reference_state! — moist isothermal atmosphere
    #####
    #
    # For constant T and constant qᵛ (no condensate):
    #   Rᵐ = (1 - qᵛ) Rᵈ + qᵛ Rᵛ
    #   p(z) = p₀ exp(-g z / (Rᵐ T))
    #   ρ(z) = p(z) / (Rᵐ T)

    @testset "compute_reference_state! moist isothermal" begin
        T₀ = FT(280)
        qᵛ = FT(0.015)
        p₀ = FT(101325)
        Rᵐ = (1 - qᵛ) * Rᵈ + qᵛ * Rᵛ

        ref = ReferenceState(grid, constants; surface_pressure=p₀, vapor_mass_fraction=0)
        compute_reference_state!(ref, T₀, qᵛ, constants)

        Nz = grid.Nz
        for k in 1:Nz
            z = @allowscalar Oceananigans.Grids.znode(1, 1, k, grid, Center(), Center(), Center())
            p_exact = p₀ * exp(-g * z / (Rᵐ * T₀))
            ρ_exact = p_exact / (Rᵐ * T₀)

            p_ref = @allowscalar ref.pressure[1, 1, k]
            ρ_ref = @allowscalar ref.density[1, 1, k]

            @test isapprox(p_ref, p_exact; rtol=FT(1e-4))
            @test isapprox(ρ_ref, ρ_exact; rtol=FT(1e-4))
        end

        # Verify moisture was set
        qᵛ_ref = @allowscalar ref.vapor_mass_fraction[1, 1, 1]
        @test qᵛ_ref ≈ qᵛ
    end

    #####
    ##### compute_reference_state! — 5-argument form with individual mass fractions
    #####

    @testset "compute_reference_state! with individual mass fractions" begin
        T₀ = FT(260)
        qᵛ = FT(0.01)
        qˡ = FT(1e-4)
        qⁱ = FT(5e-5)
        p₀ = FT(101325)

        ref = ReferenceState(grid, constants;
                             surface_pressure=p₀,
                             vapor_mass_fraction=0,
                             liquid_mass_fraction=0,
                             ice_mass_fraction=0)

        compute_reference_state!(ref, T₀, qᵛ, qˡ, qⁱ, constants)

        # Verify moisture fields were set
        @test @allowscalar(ref.vapor_mass_fraction[1, 1, 1]) ≈ qᵛ
        @test @allowscalar(ref.liquid_mass_fraction[1, 1, 1]) ≈ qˡ
        @test @allowscalar(ref.ice_mass_fraction[1, 1, 1]) ≈ qⁱ

        # Verify pressure is physically reasonable
        p_top = @allowscalar ref.pressure[1, 1, grid.Nz]
        @test p_top < p₀  # pressure decreases with height
        @test p_top > 0    # still positive

        # Ideal gas consistency: ρ = p / (Rᵐ T)
        Rᵐ = (1 - qᵛ - qˡ - qⁱ) * Rᵈ + qᵛ * Rᵛ
        for k in 1:grid.Nz
            p_ref = @allowscalar ref.pressure[1, 1, k]
            ρ_ref = @allowscalar ref.density[1, 1, k]
            @test isapprox(ρ_ref, p_ref / (Rᵐ * T₀); rtol=FT(1e-5))
        end
    end

    #####
    ##### compute_reference_state! with function profiles
    #####

    @testset "compute_reference_state! with function profiles" begin
        p₀ = FT(101325)

        T_profile(z) = max(FT(210), FT(300) - FT(0.0065) * z)
        q_profile(z) = FT(0.015) * exp(-z / 3000)

        ref = ReferenceState(grid, constants; surface_pressure=p₀, vapor_mass_fraction=0)
        compute_reference_state!(ref, T_profile, q_profile, constants)

        # Temperature should follow the profile
        z₁ = @allowscalar Oceananigans.Grids.znode(1, 1, 1, grid, Center(), Center(), Center())
        T₁ = @allowscalar ref.temperature[1, 1, 1]
        @test isapprox(T₁, T_profile(z₁); rtol=FT(1e-5))

        # Moisture should follow the profile
        qᵛ₁ = @allowscalar ref.vapor_mass_fraction[1, 1, 1]
        @test isapprox(qᵛ₁, q_profile(z₁); rtol=FT(1e-5))

        # Pressure should decrease monotonically
        for k in 2:grid.Nz
            pᵏ = @allowscalar ref.pressure[1, 1, k]
            pᵏ⁻¹ = @allowscalar ref.pressure[1, 1, k-1]
            @test pᵏ < pᵏ⁻¹
        end

        # Density should decrease monotonically
        for k in 2:grid.Nz
            ρᵏ = @allowscalar ref.density[1, 1, k]
            ρᵏ⁻¹ = @allowscalar ref.density[1, 1, k-1]
            @test ρᵏ < ρᵏ⁻¹
        end
    end

    #####
    ##### compute_reference_state! overwrites previous state
    #####

    @testset "compute_reference_state! overwrites previous state" begin
        p₀ = FT(101325)
        ref = ReferenceState(grid, constants; surface_pressure=p₀, vapor_mass_fraction=0)

        # First adjustment: warm atmosphere
        compute_reference_state!(ref, FT(300), FT(0), constants)
        ρ_warm = @allowscalar ref.density[1, 1, 1]

        # Second adjustment: cold atmosphere → higher density
        compute_reference_state!(ref, FT(200), FT(0), constants)
        ρ_cold = @allowscalar ref.density[1, 1, 1]

        @test ρ_cold > ρ_warm
    end

    #####
    ##### ReferenceState with function-valued θ₀
    #####

    @testset "ReferenceState with function θ₀" begin
        p₀ = FT(100000)
        N² = FT(1e-4)
        θ_func(z) = FT(300) * exp(N² * z / g)

        ref = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ_func)

        # Pressure should decrease monotonically
        for k in 2:grid.Nz
            pᵏ = @allowscalar ref.pressure[1, 1, k]
            pᵏ⁻¹ = @allowscalar ref.pressure[1, 1, k-1]
            @test pᵏ < pᵏ⁻¹
        end

        # Density should decrease monotonically
        for k in 2:grid.Nz
            ρᵏ = @allowscalar ref.density[1, 1, k]
            ρᵏ⁻¹ = @allowscalar ref.density[1, 1, k-1]
            @test ρᵏ < ρᵏ⁻¹
        end

        # Surface density should be physical
        ρ₀ = surface_density(ref)
        @test ρ₀ > 0
        @test ρ₀ isa FT
    end

    @testset "Closed-form hydrostatic pressure respects standard pressure" begin
        p₀ = FT(101325)
        pˢᵗ = FT(100000)
        θ₀ = FT(288)
        cᵖᵈ = constants.dry_air.heat_capacity
        κ = Rᵈ / cᵖᵈ
        T₀ = θ₀ * (p₀ / pˢᵗ)^κ

        @test hydrostatic_pressure(FT(0), p₀, θ₀, pˢᵗ, constants) == p₀

        for z in (FT(1000), FT(5000), FT(20000))
            p_closed = hydrostatic_pressure(z, p₀, θ₀, pˢᵗ, constants)
            p_expected = p₀ * (1 - g * z / (cᵖᵈ * T₀))^(cᵖᵈ / Rᵈ)
            p_incorrect = p₀ * (1 - g * z / (cᵖᵈ * θ₀))^(cᵖᵈ / Rᵈ)

            @test isapprox(p_closed, p_expected; rtol=sqrt(eps(FT)))
            @test !isapprox(p_closed, p_incorrect; rtol=FT(1e-4))
        end
    end

    @testset "pressure_balanced_density accounts for condensate" begin
        pᵣ = FT(9e4)
        pˢᵗ = FT(1e5)
        ρ_background = FT(1.1)
        θ_background = FT(300)
        θ_initial = FT(303)
        q = Breeze.Thermodynamics.MoistureMassFractions(FT(0.01), FT(0.01), zero(FT))

        𝒰_background = Breeze.Thermodynamics.LiquidIcePotentialTemperatureState(θ_background, q, pˢᵗ, pᵣ)
        𝒰_initial = Breeze.Thermodynamics.LiquidIcePotentialTemperatureState(θ_initial, q, pˢᵗ, pᵣ)
        Rᵐ = mixture_gas_constant(q, constants)

        p_background = ρ_background * Rᵐ * temperature(𝒰_background, constants)

        ρ_shortcut = Breeze.Thermodynamics.pressure_balanced_density(ρ_background, θ_background, θ_initial)
        p_shortcut = ρ_shortcut * Rᵐ * temperature(𝒰_initial, constants)

        @test !isapprox(p_shortcut, p_background; rtol=FT(1e-5))

        ρ_condensate_aware = Breeze.Thermodynamics.pressure_balanced_density(ρ_background, θ_background, θ_initial, q, pᵣ, pˢᵗ, constants)
        p_condensate_aware = ρ_condensate_aware * Rᵐ * temperature(𝒰_initial, constants)

        @test isapprox(p_condensate_aware, p_background; rtol=sqrt(eps(FT)))
    end

    #####
    ##### ReferenceState with discrete_hydrostatic_balance
    #####

    @testset "ReferenceState with discrete_hydrostatic_balance" begin
        ref = ReferenceState(grid, constants; discrete_hydrostatic_balance=true)

        # Pressure should decrease monotonically
        for k in 2:grid.Nz
            pᵏ = @allowscalar ref.pressure[1, 1, k]
            pᵏ⁻¹ = @allowscalar ref.pressure[1, 1, k-1]
            @test pᵏ < pᵏ⁻¹
        end

        # Discrete hydrostatic balance: Δp + g * ℑρ * Δz ≈ 0
        for k in 2:grid.Nz
            pᵏ = @allowscalar ref.pressure[1, 1, k]
            pᵏ⁻¹ = @allowscalar ref.pressure[1, 1, k-1]
            ρᵏ = @allowscalar ref.density[1, 1, k]
            ρᵏ⁻¹ = @allowscalar ref.density[1, 1, k-1]
            zᶠ = @allowscalar Oceananigans.Grids.znode(1, 1, k, grid, Center(), Center(), Face())
            zᶠ⁻¹ = @allowscalar Oceananigans.Grids.znode(1, 1, k-1, grid, Center(), Center(), Face())
            Δz = zᶠ - zᶠ⁻¹
            residual = (pᵏ - pᵏ⁻¹) + g * (ρᵏ + ρᵏ⁻¹) / 2 * Δz
            @test abs(residual) < FT(1e-6)
        end
    end
end

#####
##### Mass fraction accessors and set_to_mean!
#####

@testset "Mass fraction accessors and set_to_mean! [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 100), y=(0, 100), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded), halo=(5, 5, 5))
    constants = ThermodynamicConstants(FT)

    #####
    ##### vapor/liquid/ice_mass_fraction for Nothing microphysics
    #####

    @testset "Mass fraction accessors (no microphysics)" begin
        model = AtmosphereModel(grid)
        set!(model, θ=FT(300), qᵗ=FT(0.01))

        qᵛ = specific_humidity(model)
        qˡ = liquid_mass_fraction(model)
        qⁱ = ice_mass_fraction(model)

        # With no microphysics: vapor = total moisture, liquid = ice = nothing
        @test qᵛ === specific_humidity(model)
        @test qˡ === nothing
        @test qⁱ === nothing

        # Check the field has the expected value
        qᵛ₁ = @allowscalar qᵛ[1, 1, 1]
        @test isapprox(qᵛ₁, FT(0.01); rtol=FT(1e-5))
    end

    #####
    ##### vapor/liquid/ice_mass_fraction for SaturationAdjustment
    #####

    @testset "Mass fraction accessors (SaturationAdjustment)" begin
        microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
        model = AtmosphereModel(grid; microphysics)
        set!(model, θ=FT(300), qᵗ=FT(0.01))
        time_step!(model, 1)  # triggers state update which populates microphysical fields

        qᵛ = specific_humidity(model)
        qˡ = liquid_mass_fraction(model)
        qⁱ = ice_mass_fraction(model)

        # SaturationAdjustment has prognostic qᵛ and qˡ fields
        @test qᵛ isa Field
        @test qˡ isa Field
        @test qⁱ === nothing  # WarmPhaseEquilibrium has no ice
    end

    #####
    ##### set_to_mean! with ZeroField moisture (default dry reference state)
    #####

    @testset "set_to_mean! with dry reference state" begin
        model = AtmosphereModel(grid)
        ref = model.dynamics.reference_state

        # Set a non-uniform temperature field
        set!(model, θ=FT(300))
        time_step!(model, 1)

        ρ_before = @allowscalar ref.density[1, 1, 1]

        set_to_mean!(ref, model)

        ρ_after = @allowscalar ref.density[1, 1, 1]

        # Reference state should be updated (density recomputed)
        @test ρ_after > 0
        @test ref.temperature isa Field
    end

    #####
    ##### set_to_mean! with allocated moisture fields
    #####

    @testset "set_to_mean! with moist reference state" begin
        reference_state = ReferenceState(grid, constants; vapor_mass_fraction=0)
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid; dynamics)

        set!(model, θ=FT(300), qᵗ=FT(0.01))
        time_step!(model, 1)

        set_to_mean!(reference_state, model)

        # Temperature should be set to the horizontal mean of model temperature
        T_ref = @allowscalar reference_state.temperature[1, 1, 1]
        @test T_ref > 0
        @test isfinite(T_ref)

        # Vapor mass fraction should be set to horizontal mean of model moisture
        qᵛ_ref = @allowscalar reference_state.vapor_mass_fraction[1, 1, 1]
        @test qᵛ_ref > 0
        @test isapprox(qᵛ_ref, FT(0.01); rtol=FT(0.1))

        # Pressure and density should be physically reasonable
        p_ref = @allowscalar reference_state.pressure[1, 1, 1]
        ρ_ref = @allowscalar reference_state.density[1, 1, 1]
        @test p_ref > 0
        @test ρ_ref > 0
    end

    #####
    ##### set_to_mean! preserves density-weighted prognostic fields
    #####
    #
    # Density-weighted prognostic fields (ρe, ρqᵗ, ρu) are left unchanged
    # by set_to_mean!. Only the reference state (ρᵣ, pᵣ, Tᵣ) is updated.

    @testset "set_to_mean! preserves density-weighted prognostics" begin
        reference_state = ReferenceState(grid, constants; vapor_mass_fraction=0)
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid; dynamics, formulation=:StaticEnergy)

        # Set initial conditions with non-trivial profiles
        T_prof(z) = max(FT(210), FT(300) - FT(6.5e-3) * z)
        q_prof(z) = FT(0.015) * exp(-z / FT(2500))

        # compute_reference_state! takes f(z); set!(model, ...) takes f(x, y, z)
        compute_reference_state!(reference_state, T_prof, q_prof, constants)
        set!(model, T=(x, y, z) -> T_prof(z), qᵗ=(x, y, z) -> q_prof(z), u=FT(5), w=FT(0))
        time_step!(model, 1)  # populates diagnostic fields

        # Record density-weighted prognostic fields before set_to_mean!
        ρe_before  = Array(interior(model.formulation.energy_density))
        ρqᵗ_before = Array(interior(model.moisture_density))
        ρu_before  = Array(interior(model.momentum.ρu))
        ρw_before  = Array(interior(model.momentum.ρw))

        # Call set_to_mean! — this changes ρᵣ but leaves prognostics unchanged
        set_to_mean!(reference_state, model)

        # Reference temperature should match model mean temperature
        T_before = Array(interior(model.temperature))
        Tᵣ_after = Array(interior(reference_state.temperature))
        T_mean = dropdims(sum(T_before, dims=(1, 2)) / (size(T_before, 1) * size(T_before, 2)), dims=(1, 2))
        for k in 1:grid.Nz
            @test isapprox(Tᵣ_after[1, 1, k], T_mean[k]; rtol=FT(1e-5))
        end

        # Density-weighted prognostic fields should be unchanged
        ρe_after  = Array(interior(model.formulation.energy_density))
        ρqᵗ_after = Array(interior(model.moisture_density))
        ρu_after  = Array(interior(model.momentum.ρu))
        ρw_after  = Array(interior(model.momentum.ρw))

        @test ρe_after  ≈ ρe_before
        @test ρqᵗ_after ≈ ρqᵗ_before
        @test ρu_after  ≈ ρu_before
        @test ρw_after  ≈ ρw_before
    end

    @testset "set!(; compute_reference_state) preserves anelastic specific quantities" begin
        θ_init = FT(290)
        θ_new  = FT(305)
        q_new  = FT(0.01)
        u_new  = FT(5)

        reference_state = ReferenceState(grid, constants; potential_temperature=θ_init,
                                         vapor_mass_fraction=0)
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics)

        set!(model; θ=θ_new, qᵗ=q_new, u=u_new, w=0,
             compute_reference_state=true, enforce_mass_conservation=false)

        @test all(Array(interior(model.formulation.potential_temperature)) .≈ θ_new)
        @test all(Array(interior(specific_humidity(model))) .≈ q_new)
        @test all(Array(interior(model.velocities.u)) .≈ u_new)
    end

    #####
    ##### set_to_mean! for the split-explicit ExnerReferenceState
    #####
    #
    # Recomputing the Exner reference from a uniform model state at θ_new must reproduce a fresh
    # ExnerReferenceState built directly at θ_new (the constructor and the recompute share the same
    # discrete column integration), to machine precision.

    @testset "set_to_mean! recomputes the Exner reference" begin
        θ_init = 290
        θ_new  = 305
        p₀     = 101325

        ref_truth = ExnerReferenceState(grid, constants; potential_temperature=θ_new, surface_pressure=p₀)

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=θ_init, surface_pressure=p₀)
        model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics)
        ref = model.dynamics.reference_state
        @test ref isa ExnerReferenceState

        set!(model; ρ=1, θˡⁱ=θ_new, qᵗ=0)
        set_to_mean!(ref, model)

        @test ref.pressure ≈ ref_truth.pressure rtol=1e-6
        @test ref.density  ≈ ref_truth.density  rtol=1e-6
    end

    @testset "set_to_mean! overwrites every column of a 3D Exner reference" begin
        θ_init(x, y, z) = FT(290) + FT(5) * sin(FT(2π) * x / FT(100))
        θ_new = FT(305)
        p₀    = FT(101325)

        ref_truth = ExnerReferenceState(grid, constants; potential_temperature=θ_new, surface_pressure=p₀)

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=θ_init, surface_pressure=p₀)
        model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics)
        ref = model.dynamics.reference_state
        @test ref isa ExnerReferenceState

        set!(model; ρ=1, θˡⁱ=θ_new, qᵗ=0)
        set_to_mean!(ref, model)

        @test ref.pressure ≈ ref_truth.pressure rtol=1e-6
        @test ref.density  ≈ ref_truth.density  rtol=1e-6
    end

    # The public path: `set!(model; compute_reference_state=true)` → `reset_reference_state!` →
    # `set_to_mean!`. Same outcome as calling `set_to_mean!` directly above.
    @testset "set!(; compute_reference_state) recomputes the reference" begin
        θ_init = 290
        θ_new  = 305
        p₀     = 101325

        ref_truth = ExnerReferenceState(grid, constants; potential_temperature=θ_new, surface_pressure=p₀)

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=θ_init, surface_pressure=p₀)
        model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics)

        set!(model; ρ=1, θˡⁱ=θ_new, qᵗ=0, compute_reference_state=true)

        @test model.dynamics.reference_state.pressure ≈ ref_truth.pressure rtol=1e-6
        @test model.dynamics.reference_state.density  ≈ ref_truth.density  rtol=1e-6
    end

    # `reset_reference_state!` is a no-op for dynamics without a reference state (here a
    # CompressibleDynamics built with no `reference_potential_temperature`).
    @testset "set!(; compute_reference_state) is a no-op without a reference state" begin
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization())
        model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics)
        @test model.dynamics.reference_state === nothing

        set!(model; ρ=1, θˡⁱ=300, qᵗ=0, compute_reference_state=true)
        @test model.dynamics.reference_state === nothing   # unchanged; no error
    end

    # `ρ = HydrostaticallyBalancedDensity()` integrates the hydrostatic column from the surface
    # pressure. For a uniform θ it must reproduce a fresh ExnerReferenceState built at that θ (same
    # per-column integration), to machine precision — and the balanced state survives the default
    # mass-conservation correction.
    @testset "HydrostaticallyBalancedDensity sets a balanced column" begin
        θ  = 300
        p₀ = 101325

        ref_truth = ExnerReferenceState(grid, constants; potential_temperature=θ, surface_pressure=p₀)

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=θ, surface_pressure=p₀)
        model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics)

        set!(model; θˡⁱ=θ, qᵗ=0, ρ=HydrostaticallyBalancedDensity(surface_pressure=p₀))

        @test model.dynamics.pressure        ≈ ref_truth.pressure rtol=1e-6
        @test total_density(model.dynamics)  ≈ ref_truth.density  rtol=1e-6
    end
end
