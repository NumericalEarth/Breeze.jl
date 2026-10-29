include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze: BulkSensibleHeatFlux, BulkVaporFlux, BulkDrag, PolynomialCoefficient
using Breeze.Thermodynamics: InternalEnergyDensityState, MoistureMassFractions, temperature, with_temperature,
                             mixture_gas_constant, mixture_heat_capacity, mixture_isochoric_heat_capacity,
                             liquid_referenced_latent_energy,
                             absolute_zero_latent_heat
using Breeze.CompressibleEquations: acoustic_linearization
using Breeze.TotalEnergyFormulations: kinetic_energyᶜᶜᶜ
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Models: BoundaryConditionOperation
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

function one_moment_microphysics(FT)
    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    return OneMomentCloudMicrophysics(FT; cloud_formation)
end

compressible_dynamics() = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300)

volume_integral(f) = sum(interior(f)) # uniform grids below: the cell volume factors out

@testset "InternalEnergyDensityState" begin
    constants = ThermodynamicConstants()
    q = MoistureMassFractions(0.012, 0.002, 0.0005)
    𝒰 = with_temperature(InternalEnergyDensityState(0.0, q, 1.1), 271.3, constants)
    @test temperature(𝒰, constants) ≈ 271.3 rtol=1e-14

    # Phase change at fixed internal energy and density: condensing 1 g/kg warms by about ℒᵥ / cᵛᵐ
    q₂ = MoistureMassFractions(0.011, 0.003, 0.0005)
    T₂ = temperature(Breeze.Thermodynamics.with_moisture(𝒰, q₂), constants)
    @test 3 < T₂ - 271.3 < 4

    # Origin: liquid water at zero kelvin
    @test liquid_referenced_latent_energy(MoistureMassFractions(0.0, 0.01, 0.0), constants) == 0
end

@testset "TotalEnergyFormulation construction" begin
    grid = RectilinearGrid(default_arch; size=(6, 6, 8), x=(0, 2e3), y=(0, 2e3), z=(0, 4e3), halo=(5, 5, 5))

    for name in (:TotalEnergy, :E, :ρE)
        model = AtmosphereModel(grid; dynamics=compressible_dynamics(), formulation=name)
        @test model.formulation isa TotalEnergyFormulation
        @test haskey(Oceananigans.prognostic_fields(model), :ρE)
        @test model.timestepper.substepper.linearization_density_weight !== nothing
    end

    # The anelastic equations have no total-energy formulation
    @test_throws ArgumentError AtmosphereModel(grid; formulation=:TotalEnergy)
end

@testset "Temperature round trip through ρE with condensate and wind [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=(6, 6, 16), x=(0, 2e3), y=(0, 2e3), z=(0, 8e3), halo=(5, 5, 5))
    microphysics = one_moment_microphysics(FT)
    model = AtmosphereModel(grid; dynamics=compressible_dynamics(), formulation=:TotalEnergy, microphysics)

    ref = model.dynamics.reference_state
    Tᵢ(x, y, z) = 300 - 6.5e-3 * z + sin(2π * x / 2e3)
    set!(model; T=Tᵢ, qᵗ=(x, y, z) -> 0.002 + 0.012 * exp(-z / 3e3), qᶜˡ=1e-3, qʳ=5e-4,
         u=(x, y, z) -> 10 + z / 1e3, v=-3, w=(x, y, z) -> 0.5 * sin(2π * y / 2e3) * sin(π * z / 8e3), ρ=ref.density)

    T_input = Field{Center, Center, Center}(grid)
    set!(T_input, Tᵢ)
    T = model.temperature
    rtol = FT == Float64 ? 1e-10 : 1e-5
    @test maximum(abs, interior(T) .- interior(T_input)) / 300 < rtol

    # Diagnostics are consistent: Ĥ = (ρE + p) / ρᵈ
    ρE = model.formulation.energy_density
    p = model.dynamics.pressure
    ρᵈ = model.dynamics.dry_density
    @test interior(model.formulation.total_enthalpy) ≈ (interior(ρE) .+ interior(p)) ./ interior(ρᵈ)
end

@testset "Acoustic linearization of the total energy [T₀ = $T₀]" for T₀ in (0, 273.16)
    FT = Float64
    grid = RectilinearGrid(CPU(); size=(6, 6, 8), x=(0, 2e3), y=(0, 2e3), z=(0, 4e3), halo=(5, 5, 5))
    model = AtmosphereModel(grid; dynamics=compressible_dynamics(),
                            formulation=TotalEnergyFormulation(reference_temperature=T₀),
                            microphysics=one_moment_microphysics(FT))
    ref = model.dynamics.reference_state
    set!(model; θ=300, qᵗ=0.012, qᶜˡ=1e-3, u=15, v=-7, w=2, ρ=ref.density)

    constants = model.thermodynamic_constants
    i, j, k = 3, 3, 5
    C, χ̂, E = acoustic_linearization(i, j, k, grid, model.formulation, model.dynamics, constants,
                                     model.microphysics, model.microphysical_fields,
                                     Breeze.specific_prognostic_moisture(model), model.velocities)

    # Pressure as a function of (ρᵈ, ρE) at fixed momentum and water per unit dry mass
    ρ₀ = model.dynamics.total_density[i, j, k]
    ρᵈ₀ = model.dynamics.dry_density[i, j, k]
    ρE₀ = model.formulation.energy_density[i, j, k]
    ρw = ρ₀ - ρᵈ₀
    q₀ = Breeze.AtmosphereModels.grid_moisture_fractions(i, j, k, grid, model.microphysics, ρ₀,
                                                         Breeze.specific_prognostic_moisture(model)[i, j, k],
                                                         model.microphysical_fields)
    ρq = (q₀.vapor * ρ₀, q₀.liquid * ρ₀, q₀.ice * ρ₀)
    K₀ = kinetic_energyᶜᶜᶜ(i, j, k, grid, model.velocities)
    m² = 2 * K₀ * ρᵈ₀^2 # |ρᵈ𝐮|²
    gz = constants.gravitational_acceleration * znode(i, j, k, grid, Center(), Center(), Center())

    cᵖᵈ = constants.dry_air.heat_capacity
    cˡ = constants.liquid.heat_capacity
    function pressure(ρᵈ, ρE)
        r = ρᵈ / ρᵈ₀ # the water moves with the dry air
        ρ = ρᵈ + r * ρw
        q = MoistureMassFractions(r * ρq[1] / ρ, r * ρq[2] / ρ, r * ρq[3] / ρ)
        𝓊 = (ρE + T₀ * (ρᵈ * cᵖᵈ + r * ρw * cˡ)) / ρ - m² / (2ρᵈ^2) - gz
        T = temperature(InternalEnergyDensityState(𝓊, q, ρ), constants)
        return ρ * mixture_gas_constant(q, constants) * T
    end

    δ = 1e-6
    ∂p∂ρE = (pressure(ρᵈ₀, ρE₀ + δ * ρE₀) - pressure(ρᵈ₀, ρE₀ - δ * ρE₀)) / (2δ * ρE₀)
    ∂p∂ρᵈ = (pressure(ρᵈ₀ + δ * ρᵈ₀, ρE₀) - pressure(ρᵈ₀ - δ * ρᵈ₀, ρE₀)) / (2δ * ρᵈ₀)

    @test pressure(ρᵈ₀, ρE₀) ≈ model.dynamics.pressure[i, j, k] rtol=1e-12
    @test C ≈ ∂p∂ρE rtol=1e-7
    @test C * E ≈ ∂p∂ρᵈ rtol=1e-6
    @test χ̂ ≈ (ρE₀ + model.dynamics.pressure[i, j, k]) / ρᵈ₀

    # C (Ĥ + E) is the squared sound speed per unit dry mass, whatever the origin
    p₀ = model.dynamics.pressure[i, j, k]
    T = model.temperature[i, j, k]
    γᵐRᵐT = C * mixture_heat_capacity(q₀, constants) * T # (Rᵐ / cᵛᵐ) cᵖᵐ T
    @test C * (χ̂ + E) ≈ (γᵐRᵐT + 2C * K₀) * ρ₀ / ρᵈ₀ rtol=1e-12
end

@testset "Closed box conserves energy, dry mass and water [$FT, T₀ = $T₀]" for FT in test_float_types(), T₀ in (0, 273.16)
    grid = RectilinearGrid(default_arch, FT; size=(16, 16, 24), x=(0, 4e3), y=(0, 4e3), z=(0, 6e3), halo=(5, 5, 5))
    closure = SmagorinskyLilly(VerticallyImplicitTimeDiscretization(); C=0.2)
    model = AtmosphereModel(grid; dynamics=compressible_dynamics(), closure,
                            formulation=TotalEnergyFormulation(reference_temperature=T₀),
                            microphysics=one_moment_microphysics(FT), advection=WENO(order=5))
    @test reference_temperature(model.formulation) == T₀

    ref = model.dynamics.reference_state
    θᵢ(x, y, z) = 300 + 2 * exp(-((x - 2e3)^2 + (y - 2e3)^2 + (z - 1.5e3)^2) / 600^2)
    qᵢ(x, y, z) = 0.016 * exp(-z / 2500)
    qʳᵢ(x, y, z) = 1e-3 * exp(-((x - 2e3)^2 + (y - 2e3)^2 + (z - 4.5e3)^2) / 500^2) # falls without landing
    set!(model; θ=θᵢ, qᵗ=qᵢ, qʳ=qʳᵢ, u=1, v=-0.5, ρ=ref.density)

    μ = model.microphysical_fields
    water() = volume_integral(model.moisture_density) +
              sum(volume_integral(μ[name]) for name in (:ρqᶜˡ, :ρqᶜⁱ, :ρqʳ, :ρqˢⁿ))

    ρE = model.formulation.energy_density
    E₀, M₀, W₀ = volume_integral(ρE), volume_integral(model.dynamics.dry_density), water()

    simulation = Simulation(model; Δt=2, stop_iteration=100, verbose=false)
    run!(simulation)

    rtol = FT == Float64 ? 1e-12 : 1e-5
    @test abs(volume_integral(ρE) - E₀) / E₀ < rtol
    @test abs(volume_integral(model.dynamics.dry_density) - M₀) / M₀ < rtol
    @test abs(water() - W₀) / W₀ < rtol

    # Something happened: the bubble rose, cloud formed and rain fell
    @test maximum(model.velocities.w) > 1
    @test maximum(μ.qᶜˡ) > 0
    @test !any(isnan, parent(ρE))
end

@testset "The energy origin does not change the solution" begin
    grid = RectilinearGrid(default_arch; size=(16, 16, 24), x=(0, 4e3), y=(0, 4e3), z=(0, 6e3), halo=(5, 5, 5))
    closure = SmagorinskyLilly(VerticallyImplicitTimeDiscretization(); C=0.2)
    θᵢ(x, y, z) = 300 + 2 * exp(-((x - 2e3)^2 + (y - 2e3)^2 + (z - 1.5e3)^2) / 600^2)
    qᵢ(x, y, z) = 0.016 * exp(-z / 2500)
    qʳᵢ(x, y, z) = 1e-3 * exp(-((x - 2e3)^2 + (y - 2e3)^2 + (z - 4.5e3)^2) / 500^2)

    temperatures = map((0, 273.16)) do T₀
        model = AtmosphereModel(grid; dynamics=compressible_dynamics(), closure,
                                formulation=TotalEnergyFormulation(reference_temperature=T₀),
                                microphysics=one_moment_microphysics(Float64), advection=WENO(order=5))
        set!(model; θ=θᵢ, qᵗ=qᵢ, qʳ=qʳᵢ, u=1, v=-0.5, ρ=model.dynamics.reference_state.density)
        T⁰ = Array(interior(model.temperature))
        simulation = Simulation(model; Δt=2, stop_iteration=20, verbose=false)
        run!(simulation)
        (T⁰, Array(interior(model.temperature)))
    end

    (T⁰₀, T₀), (T⁰₁, T₁) = temperatures
    @test maximum(abs, T⁰₀ .- T⁰₁) / 300 < 1e-12 # the initial state is identical
    # WENO reconstructs the shifted Ĥ, and the water moves with its own velocity and density, so the
    # solutions differ by the transport of the origin offset T₀ (qᵈ cᵖᵈ + qᵗ cˡ): small, not round-off
    difference = maximum(abs, T₀ .- T₁)
    @info "Energy origin: max |T(T₀ = 0) - T(T₀ = 273.16)| after 20 steps = $difference K"
    @test difference < 0.1
end

@testset "Wall fluxes carry the enthalpy of the evaporated water [T₀ = $T₀]" for T₀ in (0, 273.16)
    grid = RectilinearGrid(default_arch; size=(8, 8, 16), x=(0, 4e3), y=(0, 4e3), z=(0, 4e3), halo=(5, 5, 5))
    coefficient = PolynomialCoefficient(roughness_length=1.5e-4)
    Tˢ = 302
    boundary_conditions = (ρE = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(; coefficient, gustiness=1, surface_temperature=Tˢ)),
                           ρqᵗ = FieldBoundaryConditions(bottom=BulkVaporFlux(; coefficient, gustiness=1, surface_temperature=Tˢ)),
                           ρu = FieldBoundaryConditions(bottom=BulkDrag(; coefficient, gustiness=1, surface_temperature=Tˢ)))
    model = AtmosphereModel(grid; dynamics=compressible_dynamics(), boundary_conditions,
                            formulation=TotalEnergyFormulation(reference_temperature=T₀),
                            closure=SmagorinskyLilly(), advection=WENO(order=5))
    ref = model.dynamics.reference_state
    set!(model; θ=(x, y, z) -> 300 + 0.01 * sin(2π * x / 4e3), qᵗ=(x, y, z) -> 0.012 * exp(-z / 2e3), u=3, ρ=ref.density)

    constants = model.thermodynamic_constants
    cᵖᵛ = constants.vapor.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)

    𝒬ᴱ = Field(BoundaryConditionOperation(model.formulation.energy_density, :bottom, model))
    𝒬ᵀ = Field(BoundaryConditionOperation(static_energy_density(model), :bottom, model))
    Jᵛ = Field(BoundaryConditionOperation(model.moisture_density, :bottom, model))
    T₁ = Array(interior(model.temperature, :, :, 1:1))
    cˡ = constants.liquid.heat_capacity
    expected = Array(interior(𝒬ᵀ)) .+ Array(interior(Jᵛ)) .* (cᵖᵛ .* T₁ .+ ℒˡ₀ .- cˡ * T₀)
    @test Array(interior(𝒬ᴱ)) ≈ expected rtol=1e-14
    @test minimum(interior(Jᵛ)) > 0 # evaporating

    # The stored energy changes by exactly the wall input. Wicker–Skamarock RK3 advances
    # uⁿ⁺¹ = uⁿ + Δt R(U**), whose wall flux is built from the state that enters the last stage.
    Δt = 2
    A = 500^2 # horizontal cell area
    input = Ref(0.0)
    function accumulate_input!(model)
        model.clock.stage == 3 || return nothing
        compute!(𝒬ᴱ)
        input[] += Δt * sum(interior(𝒬ᴱ)) * A
        return nothing
    end

    V = 500^2 * 250
    E₀ = volume_integral(model.formulation.energy_density) * V
    simulation = Simulation(model; Δt, stop_iteration=50, verbose=false)
    simulation.callbacks[:input] = Callback(accumulate_input!, IterationInterval(1); callsite=Oceananigans.TendencyCallsite())
    run!(simulation)
    E₁ = volume_integral(model.formulation.energy_density) * V
    @test (E₁ - E₀) ≈ input[] rtol=1e-9
end

@testset "Dry rest state and warm bubble compared with θˡⁱ" begin
    grid = RectilinearGrid(default_arch; size=(8, 8, 32), x=(0, 8e3), y=(0, 8e3), z=(0, 16e3), halo=(5, 5, 5))
    for U in (0, 10)
        model = AtmosphereModel(grid; dynamics=compressible_dynamics(), formulation=:TotalEnergy, advection=WENO(order=5))
        set!(model; θ=300, u=U, qᵗ=0, ρ=model.dynamics.reference_state.density)
        simulation = Simulation(model; Δt=4, stop_iteration=300, verbose=false)
        run!(simulation)
        @test maximum(abs, model.velocities.w) < 1e-10
    end

    bubble_grid = RectilinearGrid(default_arch; size=(16, 16, 24), x=(0, 4e3), y=(0, 4e3), z=(0, 6e3), halo=(5, 5, 5))
    θᵢ(x, y, z) = 300 + 2 * exp(-((x - 2e3)^2 + (y - 2e3)^2 + (z - 1.5e3)^2) / 600^2)
    qᵢ(x, y, z) = 0.016 * exp(-z / 2500)

    maximum_w = map((:TotalEnergy, :LiquidIcePotentialTemperature)) do formulation
        model = AtmosphereModel(bubble_grid; dynamics=compressible_dynamics(), formulation, advection=WENO(order=5),
                                microphysics=one_moment_microphysics(Float64))
        set!(model; θ=θᵢ, qᵗ=qᵢ, ρ=model.dynamics.reference_state.density)
        simulation = Simulation(model; Δt=2, stop_iteration=60, verbose=false)
        run!(simulation)
        maximum(model.velocities.w)
    end

    w_energy, w_θ = maximum_w
    @test w_θ > 0.5
    @test 0.8 < w_energy / w_θ < 1.25
end

@testset "Explicit time stepping with implicit vertical diffusion conserves energy" begin
    grid = RectilinearGrid(default_arch; size=(8, 8, 16), x=(0, 2e3), y=(0, 2e3), z=(0, 4e3), halo=(5, 5, 5))
    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=300)
    closure = SmagorinskyLilly(VerticallyImplicitTimeDiscretization())
    model = AtmosphereModel(grid; dynamics, formulation=:TotalEnergy, closure, advection=WENO(order=5))
    set!(model; θ=(x, y, z) -> 300 + exp(-((x - 1e3)^2 + (z - 1e3)^2) / 300^2), u=2, qᵗ=0,
         ρ=model.dynamics.reference_state.density)
    ρE = model.formulation.energy_density
    E₀ = volume_integral(ρE)
    simulation = Simulation(model; Δt=0.5, stop_iteration=40, verbose=false)
    run!(simulation)
    @test abs(volume_integral(ρE) - E₀) / E₀ < 1e-12
    @test maximum(model.velocities.w) > 0
end
