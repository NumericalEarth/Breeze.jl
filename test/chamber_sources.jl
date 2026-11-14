include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.ParticleMicrophysics: droplet_water_mass
using Breeze.Thermodynamics: saturation_vapor_pressure, vapor_gas_constant, dry_air_gas_constant, PlanarLiquidSurface, reference_value
using Oceananigans
using Statistics: mean
using Test

# A quiescent, buoyancy-free box at 90 kPa and 281 K with saturation ratio `S`; gravity acts on the droplets only
function chamber_box(FT; L = 0.02, N = 4, S = 1.0, droplets = nothing, sources = nothing, physics = DropletPhysicsParameters(FT))
    grid = RectilinearGrid(default_arch, FT; size=(N, N, N), x=(0, L), y=(0, L), z=(0, L),
                           topology=(Periodic, Periodic, Periodic))
    dynamics = BoussinesqDynamics(grid; pressure=90000, temperature=281, buoyancy=false)
    microphysics = LagrangianMicrophysics(FT; droplets, sources, physics)
    model = AtmosphereModel(grid; dynamics, microphysics)
    constants = model.thermodynamic_constants
    pᵛ = S * saturation_vapor_pressure(281, constants, PlanarLiquidSurface())
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    set!(model; T=281, qᵗ=ε * pᵛ / (90000 - (1 - ε) * pᵛ))
    return model
end

quasirandom(n, k=1) = mod(n * sqrt(2 + k), 1)

# Water in the box: vapor on the grid, liquid on the droplets, and the liquid that the sources added and removed
function water_budget(model)
    grid = model.grid
    V = prod(size(grid)) * minimum_xspacing(grid) * minimum_yspacing(grid) * minimum_zspacing(grid)
    vapor = sum(Array(interior(model.moisture_density))) * V / prod(size(grid))
    d = model.microphysics.droplets.properties
    constants = model.thermodynamic_constants
    liquid = sum(droplet_water_mass.(Array(d.r), Array(d.rᵈ), Ref(constants)); init=zero(vapor))
    book = model.microphysics.sources.bookkeeping
    return (; vapor, liquid, removed=book.removed_liquid_mass, injected=book.injected_liquid_mass,
            total = vapor + liquid + book.removed_liquid_mass - book.injected_liquid_mass)
end

# The liquid mass diagnosed on the grid: Σ ρᵣ qˡ V over the cells (the Boussinesq reference density is uniform)
liquid_diagnostic(model) = sum(Array(interior(model.microphysical_fields.qˡ))) * reference_value(model.dynamics.reference_state.density) *
                           minimum_xspacing(model.grid) * minimum_yspacing(model.grid) * minimum_zspacing(model.grid)

@testset "ChamberSources: construction and empty cases [$FT]" for FT in test_float_types()
    injection = AerosolInjection(FT; rate=1e8, dry_radius=1e-7, hygroscopicity=1.2, interval=0.1)
    removal = VirtualChamberRemoval(FT; size=(2, 2, 1), interval=0.1)
    @test occursin("AerosolInjection", summary(injection))
    @test occursin("IntegratedVirtualPosition", summary(removal))
    @test_throws ArgumentError AerosolInjection(FT; rate=0, dry_radius=1e-7, hygroscopicity=1.2, interval=0.1)
    @test_throws ArgumentError VirtualChamberRemoval(FT; size=(2, 0, 1))

    # Sources require a triply periodic grid
    bounded = RectilinearGrid(default_arch, FT; size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
    @test_throws ArgumentError AtmosphereModel(bounded; dynamics=BoussinesqDynamics(bounded),
                                               microphysics=LagrangianMicrophysics(FT; sources=ChamberSources(; removal)))

    # No droplets and removal only: nothing happens, with either rule
    for rule in (IntegratedVirtualPosition(), ResampledVirtualPosition())
        model = chamber_box(FT; sources=ChamberSources(; removal=VirtualChamberRemoval(FT; interval=0.1, rule)))
        for _ in 1:5
            time_step!(model, FT(0.05))
        end
        @test length(model.microphysics.droplets) == 0
        @test model.microphysics.sources.bookkeeping.removed == 0
        @test occursin("VirtualChamberRemoval", sprint(show, model.microphysics))
    end

    # A time step longer than the resampling interval is rejected
    droplets = (x = [FT(0.01)], y = [FT(0.01)], z = [FT(0.01)], r = [FT(1e-6)], rᵈ = [FT(1e-7)], κ = [FT(0.6)])
    removal = VirtualChamberRemoval(FT; interval=0.01, rule=ResampledVirtualPosition())
    model = chamber_box(FT; droplets, sources=ChamberSources(; removal))
    @test_throws ArgumentError time_step!(model, FT(0.05))
end

@testset "ChamberSources: injection at a prescribed rate [$FT]" for FT in test_float_types()
    L = FT(0.02)
    rate = FT(2e7) # m⁻³ s⁻¹: 160 particles per second in the 8 cm³ box
    interval = FT(0.1)
    expected = rate * L^3 # per second
    counts = Int[]
    for Δt in (FT(0.02), FT(0.01)) # the injected count must not depend on the time step
        injection = AerosolInjection(FT; rate, dry_radius=1e-7, hygroscopicity=1.2, interval, seed=11)
        model = chamber_box(FT; L, S=0.9, sources=ChamberSources(; injection)) # subsaturated: haze stays haze
        constants = model.thermodynamic_constants
        t = 0
        while t < 1 - Δt / 2
            time_step!(model, Δt)
            t += Δt
        end
        push!(counts, length(model.microphysics.droplets))
        book = model.microphysics.sources.bookkeeping
        @test book.injected == length(model.microphysics.droplets)
        @test abs(book.injected - expected) ≤ 1 + 1e-5 * expected
        droplets = model.microphysics.droplets.properties
        @test all(x -> 0 ≤ x < L, Array(droplets.x))
        @test all(z -> 0 ≤ z < L, Array(droplets.z))
        @test all(Array(droplets.rᵈ) .== FT(1e-7))
        @test all(Array(droplets.κ) .== FT(1.2))
        @test all(isfinite, Array(droplets.r))
        @test book.injected_liquid_mass ≈ book.injected * droplet_water_mass(cbrt(FT(2)) * FT(1e-7), FT(1e-7), constants) rtol=1e-6
        # The haze is in the diagnosed liquid of the step in which it was injected
        @test liquid_diagnostic(model) ≈ water_budget(model).liquid rtol=1e-6
        @test all(Array(interior(model.moisture_density)) .> 0)
    end
    @test counts[1] == counts[2]

    # A time step spanning several injection events completes them all, and nothing is injected before the start time
    injection = AerosolInjection(FT; rate, dry_radius=1e-7, hygroscopicity=1.2, interval=0.05, start_time=0.5, seed=12)
    model = chamber_box(FT; L, S=0.9, sources=ChamberSources(; injection))
    time_step!(model, FT(0.3))
    @test length(model.microphysics.droplets) == 0
    time_step!(model, FT(0.3)) # t = 0.6: the batches of 0.55 and 0.60 are both due
    @test model.microphysics.sources.bookkeeping.injected ≈ 2 * expected * 0.05 atol=1
    time_step!(model, FT(0.5)) # t = 1.1: ten more batches
    @test model.microphysics.sources.bookkeeping.injected ≈ 12 * expected * 0.05 atol=1
end

@testset "ChamberSources: removal through a virtual chamber [$FT]" for FT in test_float_types()
    # Large droplets settle at the Stokes speed through a shallow virtual chamber while their computational
    # positions stay periodic; all leave within Lz / 𝕎 and the water budget closes
    L = FT(0.02)
    Nd = 64
    n = 1:Nd
    r₀ = FT(20e-6)
    droplets = (x = L .* quasirandom.(n, 1), y = L .* quasirandom.(n, 2), z = L .* quasirandom.(n, 3),
                r = fill(r₀, Nd), rᵈ = fill(FT(5e-8), Nd), κ = fill(FT(0.6), Nd))
    physics = DropletPhysicsParameters(FT; terminal_velocity=StokesTerminalVelocity())
    Lz = FT(0.02)
    removal = VirtualChamberRemoval(FT; size=(2, 2, Lz), seed=3)
    model = chamber_box(FT; L, droplets, physics, sources=ChamberSources(; removal))
    budget₀ = water_budget(model)
    virtual₀ = Array(model.microphysics.droplets.properties.zᵛ)
    @test all(z -> 0 ≤ z < Lz, virtual₀)
    @test virtual₀ ≈ Array(model.microphysics.droplets.properties.z) # congruent with the computational position: Lz = L

    𝕎 = 1.19e8 * r₀^2 # Rogers and Yau: about 4.8 cm/s
    Δt = FT(1e-3)
    nsteps = ceil(Int, 1.5 * Lz / 𝕎 / Δt)
    counts = Int[]
    for _ in 1:nsteps
        time_step!(model, Δt)
        push!(counts, length(model.microphysics.droplets))
        d = model.microphysics.droplets.properties
        @test all(z -> 0 ≤ z ≤ L, Array(d.z)) # computational positions remain periodic
        # Vapor + liquid + removed − injected is conserved to round-off: condensation moves water between
        # vapor and droplets exactly, and removal moves the droplets' water into the removed tally
        @test water_budget(model).total ≈ budget₀.total rtol = 100 * eps(FT)
        @test liquid_diagnostic(model) ≈ water_budget(model).liquid rtol = 1e-5
    end
    book = model.microphysics.sources.bookkeeping
    @test issorted(counts; rev=true)
    @test counts[end] == 0
    @test book.removed == Nd
    @test counts[nsteps ÷ 3] > 0 # not all gone at once: removal follows the virtual descent
    @test book.removed_liquid_mass > 0.99 * budget₀.liquid
    @test liquid_diagnostic(model) == 0 # the diagnosed liquid never shows vanished particles
    @test all(isfinite, Array(interior(model.moisture_density)))

    # The virtual position follows the trajectory of the time stepper exactly: in a uniform flow with no settling
    # the virtual displacement equals u × t to round-off, across several periodic wraps
    Nd = 16
    n = 1:Nd
    droplets = (x = L .* quasirandom.(n, 1), y = L .* quasirandom.(n, 2), z = L .* quasirandom.(n, 3),
                r = fill(FT(1e-6), Nd), rᵈ = fill(FT(1e-7), Nd), κ = fill(FT(0.6), Nd))
    no_settling = DropletPhysicsParameters(FT; terminal_velocity=RogersYauTerminalVelocity(FT; quadratic_coefficient=0,
                                                                                            linear_coefficient=0,
                                                                                            square_root_coefficient=0))
    removal = VirtualChamberRemoval(FT; size=(2, 2, 2), seed=4)
    model = chamber_box(FT; L, droplets, physics=no_settling, sources=ChamberSources(; removal), S=0.95)
    u₀ = FT(0.25)
    set!(model; u=u₀)
    xᵛ₀ = Array(model.microphysics.droplets.properties.xᵛ)
    t = 0
    for _ in 1:40
        time_step!(model, FT(0.01)) # 10 cm in total: five domain lengths
        t += FT(0.01)
    end
    xᵛ = Array(model.microphysics.droplets.properties.xᵛ)
    @test length(model.microphysics.droplets) == Nd
    @test xᵛ .- xᵛ₀ ≈ fill(u₀ * t, Nd) rtol = 1e-4
    @test all(x -> 0 ≤ x ≤ L, Array(model.microphysics.droplets.properties.x))

    # The resampled rule of the reference code removes at the expected mean rate 𝕎 τ / Lz per interval
    removal = VirtualChamberRemoval(FT; size=(2, 2, 1), interval=0.05, rule=ResampledVirtualPosition(), seed=5)
    Nd = 4000
    n = 1:Nd
    droplets = (x = L .* quasirandom.(n, 1), y = L .* quasirandom.(n, 2), z = L .* quasirandom.(n, 3),
                r = fill(r₀, Nd), rᵈ = fill(FT(5e-8), Nd), κ = fill(FT(0.6), Nd))
    model = chamber_box(FT; L, droplets, physics, sources=ChamberSources(; removal))
    budget₀ = water_budget(model)
    for _ in 1:20
        time_step!(model, FT(0.01)) # 4 resampling events
    end
    removed_fraction = model.microphysics.sources.bookkeeping.removed / Nd
    expected = 1 - (1 - 𝕎 * 0.05 / 1)^3 # three events with 𝕎 known (the first sees 𝕎 = 0 of the initial state)
    @test 0.5 * expected < removed_fraction < 1.5 * expected
    @test water_budget(model).total ≈ budget₀.total rtol = 100 * eps(FT)
end

@testset "ChamberSources: steady state of injection and removal [$FT]" for FT in test_float_types()
    # Haze injected into a subsaturated box and removed through a small virtual chamber reaches a finite population,
    # with a closed water budget
    L = FT(0.02)
    injection = AerosolInjection(FT; rate=1e8, dry_radius=2e-7, hygroscopicity=1.2, interval=0.05, seed=7)
    removal = VirtualChamberRemoval(FT; size=(0.05, 0.05, 0.05), interval=0.05, rule=ResampledVirtualPosition(), seed=8)
    model = chamber_box(FT; L, S=0.95, sources=ChamberSources(; injection, removal))
    set!(model; u=(x, y, z) -> 0.05 * sin(2π * z / L), w=(x, y, z) -> 0.05 * sin(2π * x / L))
    budget₀ = water_budget(model)
    counts = Int[]
    for _ in 1:200
        time_step!(model, FT(0.01))
        push!(counts, length(model.microphysics.droplets))
    end
    book = model.microphysics.sources.bookkeeping
    @test book.injected > 0
    @test book.removed > 0
    @test counts[end] == book.injected - book.removed
    @test counts[end] < book.injected
    @test all(isfinite, Array(model.microphysics.droplets.properties.r))
    @test all(Array(model.microphysics.droplets.properties.r) .> 0)
    @test all(isfinite, Array(interior(model.formulation.potential_temperature_density)))
    @test water_budget(model).total ≈ budget₀.total rtol = 100 * eps(FT)
    @test liquid_diagnostic(model) ≈ water_budget(model).liquid rtol = 1e-5
end
