include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.ParticleMicrophysics: droplet_water_mass, diffusional_growth_rate, ventilation_factor,
                                   potential_temperature_increment
using Breeze.Thermodynamics: saturation_vapor_pressure, vapor_gas_constant, dry_air_gas_constant, liquid_latent_heat,
                             mixture_heat_capacity, reference_value, MoistureMassFractions, PlanarLiquidSurface,
                             potential_temperature_from_temperature
using Oceananigans
using Statistics: mean
using Test

# Deterministic quasi-random numbers in [0, 1)
quasirandom(n, k=1) = mod(n * sqrt(2 + k), 1)

function droplet_box_model(arch, FT; Nd = 200, saturation_ratio = 1.005, frame_vertical_velocity = nothing, sorting = nothing,
                           thermodynamic_float_type = FT)
    L = FT(0.02)
    grid = RectilinearGrid(arch, FT; size=(6, 6, 6), x=(0, L), y=(0, L), z=(0, L),
                           topology=(Periodic, Periodic, Periodic))

    dynamics = BoussinesqDynamics(grid; pressure=90000, temperature=281, frame_vertical_velocity)
    n = 1:Nd
    droplets = (x = L .* quasirandom.(n, 1), y = L .* quasirandom.(n, 2), z = L .* quasirandom.(n, 3),
                r = fill(5e-6, Nd), rᵈ = fill(5e-8, Nd), κ = fill(0.6, Nd))
    microphysics = LagrangianMicrophysics(FT; droplets, sorting)
    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_float_type)

    # Vapor mass fraction with vapor pressure pᵛ = saturation_ratio * pᵛ⁺ at p = 90000 Pa
    constants = model.thermodynamic_constants
    T₀ = 281
    pᵛ = saturation_ratio * saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface())
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    qᵛ₀ = ε * pᵛ / (90000 - (1 - ε) * pᵛ)
    set!(model; T=T₀, qᵗ=qᵛ₀, u=(x, y, z) -> sin(2π * x / L) / 20, v=(x, y, z) -> -cos(2π * y / L) / 20)

    return model
end

function water_masses(model)
    grid = model.grid
    V = prod(size(grid)) * minimum_xspacing(grid) * minimum_yspacing(grid) * minimum_zspacing(grid)
    vapor = sum(Array(interior(model.moisture_density))) * V / prod(size(grid))
    droplets = model.microphysics.droplets.properties
    constants = model.thermodynamic_constants
    liquid = sum(droplet_water_mass.(Array(droplets.r), Array(droplets.rᵈ), Ref(constants)))
    return vapor, liquid
end

@testset "LagrangianMicrophysics [$FT]" for FT in test_float_types()
    model = droplet_box_model(default_arch, FT)
    @test model.microphysics isa LagrangianMicrophysics
    @test length(model.microphysics.droplets) == 200
    @test model.microphysics.droplets.properties isa AbstractVector{<:Droplet}
    @test occursin("200 droplets", sprint(show, model.microphysics))

    # Supersaturated air: droplets grow, vapor is depleted, and total water is conserved
    vapor₀, liquid₀ = water_masses(model)
    r₀ = Array(model.microphysics.droplets.properties.r)
    T₀ = mean(model.temperature)
    for _ in 1:10
        time_step!(model, 1e-2)
    end
    vapor₁, liquid₁ = water_masses(model)
    r₁ = Array(model.microphysics.droplets.properties.r)
    @test all(r₁ .> r₀)
    @test liquid₁ > liquid₀
    @test vapor₁ < vapor₀
    @test (vapor₁ + liquid₁) ≈ (vapor₀ + liquid₀) rtol = 100 * eps(FT)

    # The latent heat of the condensed water warms the air at constant pressure
    grid = model.grid
    constants = model.thermodynamic_constants
    ρ = reference_value(model.dynamics.reference_state.density)
    air_mass = ρ * prod(size(grid)) * minimum_xspacing(grid) * minimum_yspacing(grid) * minimum_zspacing(grid)
    q = MoistureMassFractions(mean(model.microphysical_fields.qᵛ))
    ΔT = liquid_latent_heat(T₀, constants) * (liquid₁ - liquid₀) / (mixture_heat_capacity(q, constants) * air_mass)
    @test mean(model.temperature) - T₀ ≈ ΔT rtol = FT == Float64 ? 1e-3 : 1e-2

    # The liquid mass fraction of the droplets is diagnosed on the grid
    @test sum(Array(interior(model.microphysical_fields.qˡ))) * air_mass / prod(size(grid)) ≈ liquid₁ rtol = sqrt(eps(FT))

    # Droplets stay in the periodic domain
    droplets = model.microphysics.droplets.properties
    L = FT(0.02)
    @test all(z -> 0 ≤ z ≤ L, Array(droplets.z))

    # Subsaturated air: droplets evaporate
    model = droplet_box_model(default_arch, FT; saturation_ratio = 0.95)
    r₀ = Array(model.microphysics.droplets.properties.r)
    vapor₀, liquid₀ = water_masses(model)
    for _ in 1:10
        time_step!(model, 1e-2)
    end
    vapor₁, liquid₁ = water_masses(model)
    @test all(Array(model.microphysics.droplets.properties.r) .< r₀)
    @test (vapor₁ + liquid₁) ≈ (vapor₀ + liquid₀) rtol = 100 * eps(FT)

    # A rising frame cools the air and raises the supersaturation, at a rate reduced by latent heating
    model = droplet_box_model(default_arch, FT; saturation_ratio = 1.001, frame_vertical_velocity = 2)
    T₀ = mean(model.temperature)
    vapor₀, liquid₀ = water_masses(model)
    for _ in 1:10
        time_step!(model, 1e-2)
    end
    vapor₁, liquid₁ = water_masses(model)
    @test liquid₁ > liquid₀
    q = MoistureMassFractions(mean(model.microphysical_fields.qᵛ))
    cᵖᵐ = mixture_heat_capacity(q, constants)
    g = constants.gravitational_acceleration
    ΔT_dry = - g * 2 * FT(0.1) / cᵖᵐ
    ΔT_latent = liquid_latent_heat(T₀, constants) * (liquid₁ - liquid₀) / (cᵖᵐ * air_mass)
    @test mean(model.temperature) - T₀ ≈ ΔT_dry + ΔT_latent rtol = FT == Float64 ? 1e-3 : 1e-2

    # Sorting droplets by cell reorders them but does not change their evolution
    unsorted = droplet_box_model(default_arch, FT)
    sorted = droplet_box_model(default_arch, FT; sorting = CountingSort())
    for _ in 1:5
        time_step!(unsorted, 1e-2)
        time_step!(sorted, 1e-2)
    end
    @test sort(Array(sorted.microphysics.droplets.properties.r)) ≈ sort(Array(unsorted.microphysics.droplets.properties.r))
    @test Array(interior(sorted.moisture_density)) ≈ Array(interior(unsorted.moisture_density))
end

@testset "LagrangianMicrophysics in single precision" begin
    # The per-step increments of θ and qᵛ in droplet DNS are comparable to the Float32 spacing of θ ≈ 300 K
    # and qᵛ ≈ 10⁻² (3 × 10⁻⁵ K and 5 × 10⁻¹⁰), and the per-step change of the frame pressure to Float32 epsilon.

    # The potential temperature increment of condensation is free of cancellation
    T, ΔT, p, qᵛ, Δqᵛ = 281.3, 2.1e-5, 9.02e4, 7.3e-3, -8.4e-10
    constants = ThermodynamicConstants(Float64)
    Δθ = potential_temperature_from_temperature(T + ΔT, p, 1e5, constants, qᵛ + Δqᵛ) -
         potential_temperature_from_temperature(T, p, 1e5, constants, qᵛ)
    @test potential_temperature_increment(T, ΔT, p, 1e5, qᵛ, Δqᵛ, constants) ≈ Δθ rtol = 1e-8
    Δθ³² = potential_temperature_increment(Float32.((T, ΔT, p, 1e5, qᵛ, Δqᵛ))..., ThermodynamicConstants(Float32))
    @test Δθ³² isa Float32
    @test Δθ³² ≈ Δθ rtol = 1e-5

    # The moving frame is kept in Float64 and read in the float type of the grid
    models = Dict(FT => droplet_box_model(default_arch, FT; saturation_ratio = 1.001, frame_vertical_velocity = 2)
                  for FT in (Float32, Float64))
    frame = models[Float32].dynamics.reference_state.frame
    @test frame.pressure isa Float64
    @test models[Float32].dynamics.reference_state.pressure[1, 1, 1] isa Float32
    p₀ = frame.pressure
    for _ in 1:100, FT in (Float32, Float64)
        time_step!(models[FT], 1e-3)
    end
    Δp(model) = model.dynamics.reference_state.frame.pressure - p₀
    @test Δp(models[Float32]) ≈ Δp(models[Float64]) rtol = 1e-4

    # Double-precision thermodynamic fields on a single-precision grid
    model = droplet_box_model(default_arch, Float32; thermodynamic_float_type = Float64)
    @test eltype(model.grid) == Float32
    @test eltype(model.formulation.potential_temperature_density) == Float64
    @test eltype(model.formulation.potential_temperature) == Float64
    @test eltype(model.moisture_density) == Float64
    @test eltype(model.microphysical_fields.qᵛ) == Float64
    @test eltype(model.temperature) == Float32
    @test eltype(model.velocities.u) == Float32

    reference = droplet_box_model(default_arch, Float64)
    θ₀ = mean(model.formulation.potential_temperature)
    θ₀ʳ = mean(reference.formulation.potential_temperature)
    for _ in 1:10
        time_step!(model, 1e-2)
        time_step!(reference, 1e-2)
    end
    # The warming of the air by condensation agrees with double precision to single-precision accuracy of the increment
    Δθ = mean(model.formulation.potential_temperature) - θ₀
    Δθʳ = mean(reference.formulation.potential_temperature) - θ₀ʳ
    @test Δθ ≈ Δθʳ rtol = 1e-3
end

@testset "Lattice Green's function of the second-order Laplacian" begin
    # The cell value is Watson's integral 0.25273 / Δ on an infinite cubic lattice, less the periodic images
    # 2.8373 / (4π N Δ) on N³ periodic cells (2.8373 is the Madelung-type constant of the simple cubic lattice)
    N = 64
    grid = RectilinearGrid(CPU(); size=(N, N, N), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
    @test Breeze.ParticleMicrophysics.periodic_lattice_greens_function(grid) / N ≈ 0.25273 - 2.8373 / (4π * N) rtol = 1e-3

    stretched = RectilinearGrid(CPU(); size=(4, 4, 4), x=(0, 1), y=(0, 1), z=[0, 0.1, 0.3, 0.6, 1],
                                topology=(Periodic, Periodic, Bounded))
    @test isnothing(Breeze.ParticleMicrophysics.periodic_lattice_greens_function(stretched))
end

# A single droplet in quiescent air with molecular diffusion: once its own vapor deficit and warming of its
# cell are quasi-steady, it must grow at the rate of the growth law evaluated at the box-mean (ambient) state.
function single_droplet_growth_error(self_interaction; N = 8, Δt = 1e-3, spin_up = 100, velocity = (0, 0, 0),
                                     averaging_steps = 1)
    FT = Float64
    L = N * 5e-4
    grid = RectilinearGrid(CPU(), FT; size=(N, N, N), x=(0, L), y=(0, L), z=(0, L),
                           topology=(Periodic, Periodic, Periodic))
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0) # no buoyancy-driven flow
    dynamics = BoussinesqDynamics(grid, constants; pressure=90000, temperature=281)
    # Mid-cell, so that the droplet settles (≈ 1 mm/s) within its cell during the test
    droplets = (x = [L / 2 + 2.5e-4], y = [L / 2 + 2.5e-4], z = [L / 2 + 3.5e-4], r = [3e-6], rᵈ = [2e-8], κ = [0.6])
    microphysics = LagrangianMicrophysics(FT; droplets, self_interaction)
    closure = ScalarDiffusivity(ν=1.6e-5, κ=(ρθ=2.2e-5, ρqᵛ=2.5e-5))
    model = AtmosphereModel(grid; dynamics, microphysics, closure, thermodynamic_constants=constants)

    T₀ = 281
    pᵛ = 1.005 * saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface())
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    set!(model; T=T₀, qᵗ=ε * pᵛ / (90000 - (1 - ε) * pᵛ), u=velocity[1], v=velocity[2], w=velocity[3])

    for _ in 1:spin_up
        time_step!(model, Δt)
    end

    properties = model.microphysics.droplets.properties
    r₀ = Array(properties.r)[1]
    for _ in 1:averaging_steps
        time_step!(model, Δt)
    end
    r₁ = Array(properties.r)[1]
    Φ = (r₁^2 - r₀^2) / (2 * averaging_steps * Δt)

    # The growth law at the box-mean state
    T = mean(model.temperature)
    q = MoistureMassFractions(mean(model.microphysical_fields.qᵛ))
    p = 90000
    ρᵗ = p / (Breeze.Thermodynamics.mixture_gas_constant(q, constants) * T)
    𝒮 = Breeze.Thermodynamics.supersaturation(T, ρᵗ, q, constants, PlanarLiquidSurface())
    physics = model.microphysics.physics
    𝕎 = Array(properties.𝕎)[1]
    fᵛ = ventilation_factor(r₁, 𝕎, T, p, constants, physics)
    Φ★ = diffusional_growth_rate(r₁, 2e-8, 0.6, 𝒮, fᵛ, T, p, constants, physics)

    return Φ / Φ★ - 1
end

@testset "Self-interaction correction of droplet growth" begin
    uncorrected = single_droplet_growth_error(nothing)
    corrected = single_droplet_growth_error(DiffusiveSelfInteraction())
    # Without the correction the droplet sees its own sink: error ≈ -4π g r ≈ -1.35% for r / Δ = 0.006,
    # with g reduced from 0.2527 / Δ by the periodic images and by diffusion during the time step
    @test -0.016 < uncorrected < -0.011
    @test abs(corrected) < 0.05 * abs(uncorrected)
end

@testset "Self-interaction correction of a droplet that moves through the grid" begin
    # A droplet carried by uniform flow across the grid at a cell Péclet number |U| Δ / κ ≈ 2 in each direction
    # samples about half of the perturbation it leaves in its cell at rest, so the correction must be reduced
    kwargs = (velocity = (0.1, 0.083, 0.117), averaging_steps = 100)
    uncorrected = single_droplet_growth_error(nothing; kwargs...)
    stationary = single_droplet_growth_error(DiffusiveSelfInteraction(grid_translation=false); kwargs...)
    corrected = single_droplet_growth_error(DiffusiveSelfInteraction(); kwargs...)
    @test uncorrected < 0
    @test stationary > 0.25 * abs(uncorrected) # the correction for droplets at rest overcorrects
    @test abs(corrected) < 0.1 * abs(uncorrected)
end
