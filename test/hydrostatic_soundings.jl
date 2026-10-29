include(joinpath(@__DIR__, "setup.jl"))

#####
##### Tests for HydrostaticSounding and ReferenceState(grid, constants; sounding)
#####

using Breeze
using Breeze.Thermodynamics: MoistureMassFractions, sounding_pressure, sounding_virtual_temperature, virtual_temperature,
                             mixture_gas_constant, surface_pressure_value
using Oceananigans
using Oceananigans.Grids: ColumnEnsembleSize
using Test

@testset "HydrostaticSounding" begin
    constants = ThermodynamicConstants()
    Rᵈ = dry_air_gas_constant(constants)
    g = constants.gravitational_acceleration

    @testset "virtual_temperature" begin
        q = MoistureMassFractions(0.012, 0.002, 0.0005)
        T = 285.0
        @test virtual_temperature(T, q, constants) ≈ mixture_gas_constant(q, constants) * T / Rᵈ
        Rᵛ = vapor_gas_constant(constants)
        @test virtual_temperature(T, q, constants) ≈ T * (1 + (Rᵛ / Rᵈ - 1) * q.vapor - q.liquid - q.ice)
        @test virtual_temperature(T, MoistureMassFractions(0.0), constants) == T
    end

    @testset "Exact pressure of piecewise-linear profiles" begin
        # A constant lapse rate γ has p = p₁ (Tᵛ(z) / Tᵛ₁)^(g / (Rᵈ γ))
        γ = 0.0065
        z = [0, 2000, 11000]
        lapse(zₖ) = 300 - γ * zₖ
        sounding = HydrostaticSounding(z, lapse.(z); bottom_pressure=1e5)
        for zₖ in (0, 500, 3333.3, 11000)
            @test sounding_pressure(sounding, zₖ, constants) ≈ 1e5 * (lapse(zₖ) / 300)^(g / (Rᵈ * γ)) rtol=1e-13
            @test sounding_virtual_temperature(sounding, zₖ) ≈ lapse(zₖ)
        end

        # Isothermal beyond its ends, including below the anchor
        top = 1e5 * (lapse(11000) / 300)^(g / (Rᵈ * γ))
        @test sounding_pressure(sounding, 15000, constants) ≈ top * exp(-g * 4000 / (Rᵈ * lapse(11000))) rtol=1e-13
        anchored = HydrostaticSounding([10, 100], [290, 290]; bottom_pressure=1e5)
        @test sounding_pressure(anchored, 0, constants) ≈ 1e5 * exp(g * 10 / (Rᵈ * 290)) rtol=1e-14

        # Nearly isothermal layers are integrated accurately (log1p form)
        nearly = HydrostaticSounding([0, 1000], [250, 250 + 1e-10]; bottom_pressure=1e5)
        @test sounding_pressure(nearly, 1000, constants) ≈ 1e5 * exp(-g * 1000 / (Rᵈ * (250 + 0.5e-10))) rtol=1e-14

        # A repeated height is a jump: the value above applies at and above it
        jump = HydrostaticSounding([0, 1000, 1000, 2000], [300, 300, 250, 250]; bottom_pressure=1e5)
        @test sounding_virtual_temperature(jump, 1000) == 250
        @test sounding_virtual_temperature(jump, 999.999) ≈ 300
        p₁₀₀₀ = 1e5 * exp(-g * 1000 / (Rᵈ * 300))
        @test sounding_pressure(jump, 2000, constants) ≈ p₁₀₀₀ * exp(-g * 1000 / (Rᵈ * 250)) rtol=1e-13
    end

    @testset "From pressure and density" begin
        z = collect(10.0:20:3990)
        γ = 0.006
        Tᵛ = 300 .- γ .* z
        p = @. 1e5 * (Tᵛ / Tᵛ[1])^(g / (Rᵈ * γ))
        ρ = p ./ (Rᵈ .* Tᵛ)
        sounding = HydrostaticSounding(z, p, ρ, constants)
        @test sounding.bottom_pressure == p[1]
        @test maximum(abs, sounding_pressure.(Ref(sounding), z, Ref(constants)) .- p) < 1e-8

        # The same column under another gravitational acceleration is refused
        other = ThermodynamicConstants(gravitational_acceleration=9.80665)
        @test_throws ArgumentError HydrostaticSounding(z, p, ρ, other)
        @test_throws ArgumentError HydrostaticSounding(z, p, ρ[1:end-1], constants)
    end

    @testset "Validation" begin
        @test_throws ArgumentError HydrostaticSounding([0, 1000], [300]; bottom_pressure=1e5)
        @test_throws ArgumentError HydrostaticSounding([1000, 0], [300, 290]; bottom_pressure=1e5)
        @test_throws ArgumentError HydrostaticSounding([0, 1000], [300, -1]; bottom_pressure=1e5)
        @test_throws ArgumentError HydrostaticSounding([0, 1000], [300, 290]; bottom_pressure=0)
    end
end

@testset "ReferenceState from a sounding" begin
    arch = default_arch
    constants = ThermodynamicConstants()
    Rᵈ = dry_air_gas_constant(constants)
    g = constants.gravitational_acceleration
    γ = 0.0065
    lapse(z) = 300 - γ * z
    analytic(z) = 1e5 * (lapse(z) / 300)^(g / (Rᵈ * γ))
    sounding = HydrostaticSounding([0, 20000], lapse.([0, 20000]); bottom_pressure=1e5)

    @testset "One sounding" begin
        grid = RectilinearGrid(arch; size=40, z=(0, 12000), topology=(Flat, Flat, Bounded))
        reference = ReferenceState(grid, constants; sounding)
        zᶜ = znodes(grid, Center())
        @test Array(interior(reference.pressure))[1, 1, :] ≈ analytic.(zᶜ) rtol=1e-13
        @test Array(interior(reference.temperature))[1, 1, :] ≈ lapse.(zᶜ)
        @test Array(interior(reference.density))[1, 1, :] ≈ analytic.(zᶜ) ./ (Rᵈ .* lapse.(zᶜ)) rtol=1e-13
        @test reference.base_pressure ≈ 1e5
        @test surface_pressure_value(reference) ≈ 1e5
        @test reference.potential_temperature ≈ 300

        # The sounding's own column, whatever the grid: a raised domain anchors at its bottom face
        raised = RectilinearGrid(arch; size=20, z=(2000, 12000), topology=(Flat, Flat, Bounded))
        @test surface_pressure_value(ReferenceState(raised, constants; sounding)) ≈ analytic(2000) rtol=1e-13

        @test_throws ArgumentError ReferenceState(grid, constants; sounding, base_pressure=1e5)
        @test_throws ArgumentError ReferenceState(grid, constants; sounding, potential_temperature=300)
        @test_throws ArgumentError ReferenceState(grid, constants; sounding, vapor_mass_fraction=0.01)
        @test_throws ArgumentError ReferenceState(grid, constants; sounding, discrete_hydrostatic_balance=true)

        # It drives an anelastic model
        model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference), thermodynamic_constants=constants)
        set!(model; θ=z -> lapse(z) * (1e5 / analytic(z))^(Rᵈ / constants.dry_air.heat_capacity))
        time_step!(model, 1)
        @test all(isfinite, Array(interior(model.temperature)))
        @test Array(interior(model.temperature))[1, 1, :] ≈ lapse.(zᶜ) rtol=1e-4
    end

    @testset "One sounding per column" begin
        Nᶜ = 2
        grid = RectilinearGrid(arch; size=ColumnEnsembleSize(Nz=30, ensemble=(Nᶜ, 1)), z=(0, 9000),
                               topology=(Flat, Flat, Bounded))
        warm = HydrostaticSounding([0, 10000], [305, 240]; bottom_pressure=1.01e5)
        soundings = reshape([sounding, warm], Nᶜ, 1)
        reference = ReferenceState(grid, constants; sounding=soundings)
        zᶜ = znodes(grid, Center())
        pᵣ = Array(interior(reference.pressure))
        @test pᵣ[1, 1, :] ≈ analytic.(zᶜ) rtol=1e-13
        @test pᵣ[2, 1, :] ≈ sounding_pressure.(Ref(warm), zᶜ, Ref(constants)) rtol=1e-14
        @test Array(interior(reference.surface_pressure))[:, 1, 1] ≈ [1e5, 1.01e5]
        @test_throws ArgumentError ReferenceState(grid, constants; sounding=reshape([sounding], 1, 1))
    end
end
