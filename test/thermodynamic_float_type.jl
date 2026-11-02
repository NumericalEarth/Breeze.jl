include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: specific_humidity
using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState, MoistureMassFractions
using Oceananigans
using Statistics: mean
using Test

single_precision_grid(arch) = RectilinearGrid(arch, Float32; size=(4, 4, 4), extent=(100, 100, 100),
                                              topology=(Periodic, Periodic, Bounded))

@testset "thermodynamic_float_type" begin
    grid = single_precision_grid(default_arch)

    @testset "Storage precision" begin
        for formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
            model = AtmosphereModel(grid; formulation, thermodynamic_float_type=Float64)
            ρχ = only(Oceananigans.prognostic_fields(model.formulation))
            @test eltype(ρχ) == Float64
            @test eltype(model.moisture_density) == Float64
            @test eltype(specific_humidity(model)) == Float64
            @test eltype(model.momentum.ρu) == Float32
            @test eltype(model.temperature) == Float32

            # The initial-state storage of the time stepper keeps the float type of each prognostic field
            U⁰ = model.timestepper.U⁰
            @test eltype(U⁰.ρqᵛ) == Float64
            @test eltype(U⁰.ρu) == Float32
            time_step!(model, 1)
            @test eltype(model.moisture_density) == Float64
        end
    end

    @testset "Constructor warning" begin
        @test_logs (:warn, r"thermodynamic_float_type") AtmosphereModel(grid)
        @test_logs AtmosphereModel(grid; thermodynamic_float_type=Float32)
        @test_logs AtmosphereModel(grid; thermodynamic_float_type=Float64)
        grid64 = RectilinearGrid(default_arch, Float64; size=(4, 4, 4), extent=(100, 100, 100),
                                 topology=(Periodic, Periodic, Bounded))
        @test_logs AtmosphereModel(grid64)
    end

    @testset "Small increments are retained" begin
        # A heating of 10⁻⁵ K per step is below half the single-precision spacing of θ ≈ 300 K
        # (3 × 10⁻⁵ K), so it is rounded away when θ is stored in single precision.
        heating = 1e-5
        Nt = 10
        function warming(thermodynamic_float_type)
            model = AtmosphereModel(grid; thermodynamic_float_type, advection=nothing, closure=nothing,
                                    forcing=(; θ=Returns(heating)))
            set!(model; θ=300)
            θ₀ = mean(Array{Float64}(interior(model.formulation.potential_temperature)))
            for _ in 1:Nt
                time_step!(model, 1)
            end
            θ = mean(Array{Float64}(interior(model.formulation.potential_temperature)))
            return θ - θ₀
        end

        @test warming(Float64) ≈ Nt * heating rtol=1e-3
        @test !isapprox(warming(Float32), Nt * heating, rtol=0.1)
    end

    @testset "Promoting LiquidIcePotentialTemperatureState" begin
        q = MoistureMassFractions(0.01f0)
        𝒰 = LiquidIcePotentialTemperatureState(300.0, q, 1f5, 9f4)
        @test 𝒰.potential_temperature isa Float64
        @test 𝒰.moisture_mass_fractions.vapor isa Float64
        @test 𝒰.reference_pressure isa Float64
    end

    @testset "Double-precision moving frame on a single-precision grid" begin
        periodic = RectilinearGrid(default_arch, Float32; size=(2, 2, 2), extent=(1, 1, 1),
                                   topology=(Periodic, Periodic, Periodic))
        W = 2
        dynamics = BoussinesqDynamics(periodic; pressure=9e4, temperature=285, frame_vertical_velocity=W)
        model = AtmosphereModel(periodic; dynamics, thermodynamic_float_type=Float64)
        frame = model.dynamics.reference_state.frame
        @test frame.pressure isa Float64
        @test frame.time isa Float64
        set!(model; T=285)
        p₀ = frame.pressure

        # The per-step pressure decrement ρ̄ g W Δt ≈ 2 × 10⁻³ Pa is far below the single-precision
        # spacing of 9 × 10⁴ Pa (8 × 10⁻³ Pa)
        Δt = 1e-4
        Nt = 10
        for _ in 1:Nt
            time_step!(model, Δt)
        end
        ρ̄ = 9e4 / (Breeze.Thermodynamics.dry_air_gas_constant(model.thermodynamic_constants) * 285)
        g = model.thermodynamic_constants.gravitational_acceleration
        @test p₀ - frame.pressure ≈ ρ̄ * g * W * Nt * Δt rtol=1e-2
    end
end
