include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.ParticleMicrophysics: droplet_water_mass, droplet_cell
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

# Deterministic quasi-random numbers in [0, 1) from the irrationals √2, √3, √5 (k = 0, 1, 3; √4 would be rational)
quasirandom(n, k=0) = mod(n * sqrt(2 + k), 1)
scattered(L, Nd) = (x = L .* quasirandom.(1:Nd, 0), y = L .* quasirandom.(1:Nd, 1), z = L .* quasirandom.(1:Nd, 3))

# Sums over the state in extended precision, so that the budget checks measure the model's rounding and not the test's:
# Julia's `sum` is naive within 1024-element blocks, and for 4000 identical 20 μm droplets its error reached 480 eps of the
# total water (audit jobs 16264, 16268, 16281 on 2026-11-14), while the model's vapor change equalled the kernel's condensed
# mass to 1e-6 of the exchanged mass.
exact_sum(x) = Float64(sum(big.(x)))

# Water in the box: vapor on the grid, liquid on the droplets, and the liquid that the sources added and removed
function water_budget(model)
    grid = model.grid
    cell_volume = Float64(minimum_xspacing(grid)) * Float64(minimum_yspacing(grid)) * Float64(minimum_zspacing(grid))
    vapor = exact_sum(Float64.(Array(interior(model.moisture_density)))) * cell_volume
    d = model.microphysics.droplets.properties
    constants = model.thermodynamic_constants
    masses = Float64.(Array(d.ξ)) .* droplet_water_mass.(Float64.(Array(d.r)), Float64.(Array(d.rᵈ)), Ref(constants))
    liquid = isempty(masses) ? 0.0 : exact_sum(masses)
    book = model.microphysics.sources.bookkeeping
    return (; vapor, liquid, removed=book.removed_liquid_mass, injected=book.injected_liquid_mass,
            total = vapor + liquid + book.removed_liquid_mass - book.injected_liquid_mass)
end

# Per-operation rounding budget of the water budget after `steps` time steps, relative to the total water `M` of `budget₀`
# (derivation in the status notes of 2026-11-14, audits 16264–16282). Operations on ρqᵛ in every cell and step:
#   - three SSP-RK3 substeps `u = (1 - α) u⁰ + α (u + Δt G)` (`_ssp_rk3_substep!`), each with four roundings of order eps(FT) u
#     (the Δt G term is far smaller than u), and one condensation update `ρqᵛ -= C`: 13 roundings of at most eps(FT) ρqᵛ per step,
#     so the vapor mass can drift by at most 13 × steps × eps(FT) × M_vapor if every rounding had the same sign (in practice they do
#     not; the measured drift is a few eps in total);
#   - the exchanged mass: each cell's C is a sum of n ≤ n_max atomic adds of Δm/V, each rounding at eps(FT) of the running sum, so
#     the water moved over the run is accounted to within n_max × eps(FT) × X, with X the total exchanged mass (accumulated by the
#     caller as Σ |Δvapor| over the steps); the kernel's Δm itself is accurate to a few eps of Δm (audit 16282);
#   - sources: the masses of removed droplets are evaluated in Float64 from the stored FT radii and accumulated in Float64, so the
#     only FT rounding is that of the stored radii, at most 3 eps(FT) of the removed mass; injected haze masses are exact;
#   - the two extended-precision state sums contribute nothing at this level; 4 eps(FT) covers the FT storage of the initial state.
# The Float32 saturated box of these tests does not condense at all (Δr is below single precision), so the condensing cases below
# use super- and subsaturated boxes; the measured Float32 residual without removals is exactly zero (audit 16268).
function budget_tolerance(model, budget₀, steps, exchanged)
    FT = eltype(model.grid)
    book = model.microphysics.sources.bookkeeping
    M = budget₀.total
    vapor = 13 * steps * eps(FT) * budget₀.vapor / M
    deposits = maximum_cell_occupancy(model) * eps(FT) * exchanged / M
    sources = 3 * eps(FT) * (book.removed_liquid_mass + book.injected_liquid_mass) / M
    return vapor + deposits + sources + 4 * eps(FT)
end

# The largest number of droplets in one cell, from the cell assignment of the current state (on the CPU)
function maximum_cell_occupancy(model)
    grid = model.grid
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    cpu_grid = RectilinearGrid(CPU(), FT; size=(Nx, Ny, Nz), x=(0, grid.Lx), y=(0, grid.Ly), z=(0, grid.Lz),
                               topology=(Periodic, Periodic, Periodic))
    d = model.microphysics.droplets.properties
    x, y, z = Array(d.x), Array(d.y), Array(d.z)
    counts = zeros(Int, Nx, Ny, Nz)
    for p in eachindex(x)
        i, j, k = droplet_cell((x[p], y[p], z[p]), cpu_grid)
        counts[i, j, k] += 1
    end
    return max(1, maximum(counts))
end

# The liquid mass diagnosed on the grid: Σ ρᵣ qˡ V over the cells (the Boussinesq reference density is uniform). The deposit is a
# sum of n atomic adds per cell in FT, each rounding at eps(FT) of the running cell sum, so it is within n_max eps(FT) of the exact
# cell liquid, with n_max the largest number of droplets in one cell (measured: 0.5 / 1.1 / 12.4 eps(Float32) for 64 / 400 / 4000
# droplets, audit 16268; 0–1.4 eps(Float64) against a same-assignment recomputation, audit 16278).
liquid_diagnostic(model) = exact_sum(Float64.(Array(interior(model.microphysical_fields.qˡ)))) *
                           Float64(reference_value(model.dynamics.reference_state.density)) *
                           Float64(minimum_xspacing(model.grid)) * Float64(minimum_yspacing(model.grid)) * Float64(minimum_zspacing(model.grid))

deposit_tolerance(model) = (maximum_cell_occupancy(model) + 2) * eps(eltype(model.grid))

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
        @test liquid_diagnostic(model) ≈ water_budget(model).liquid rtol=deposit_tolerance(model)
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

@testset "ChamberSources: injection with multiplicity [$FT]" for FT in test_float_types()
    # Each computational particle represents ξ physical particles: the computational batch is rate V Δt / ξ, the
    # physical count and the haze water are ξ-weighted, and the water budget closes with the weighted liquid
    L = FT(0.02)
    rate = FT(2e8) # 1600 physical particles per second in the box
    ξ = FT(10)
    injection = AerosolInjection(FT; rate, dry_radius=1e-7, hygroscopicity=1.2, interval=0.1, multiplicity=ξ, seed=13)
    removal = VirtualChamberRemoval(FT; size=(0.05, 0.05, 0.05), interval=0.05, rule=ResampledVirtualPosition(), seed=14)
    model = chamber_box(FT; L, S=1.002, sources=ChamberSources(; injection, removal)) # slightly supersaturated: haze grows
    set!(model; u=(x, y, z) -> 0.05 * sin(2π * z / L), w=(x, y, z) -> 0.05 * sin(2π * x / L))
    budget₀ = water_budget(model)
    t = 0
    exchanged = 0.0
    vapor_prev = budget₀.vapor
    for _ in 1:100
        time_step!(model, FT(0.01))
        t += 0.01
        vapor = water_budget(model).vapor
        exchanged += abs(vapor - vapor_prev)
        vapor_prev = vapor
    end
    book = model.microphysics.sources.bookkeeping
    @test abs(book.injected - rate * L^3 * t / ξ) ≤ 1 + 1e-5 * rate * L^3 * t / ξ
    @test book.injected_physical ≈ ξ * book.injected
    @test book.removed_physical ≈ ξ * book.removed
    @test all(Array(model.microphysics.droplets.properties.ξ) .== ξ)
    @test water_budget(model).total ≈ budget₀.total rtol = budget_tolerance(model, budget₀, 100, exchanged)
    @test liquid_diagnostic(model) ≈ water_budget(model).liquid rtol = deposit_tolerance(model)
    # The vapor lost equals ξ times the growth of the computational droplets (the budget above), and is nonzero
    @test budget₀.vapor - water_budget(model).vapor > 0
    @test_throws ArgumentError AerosolInjection(FT; rate, dry_radius=1e-7, hygroscopicity=1.2, interval=0.1, multiplicity=0)
end

@testset "ChamberSources: removal through a virtual chamber [$FT]" for FT in test_float_types()
    # Large droplets settle at the Stokes speed through a shallow virtual chamber while their computational
    # positions stay periodic; all leave within Lz / 𝕎 and the water budget closes
    L = FT(0.02)
    Nd = 64
    n = 1:Nd
    r₀ = FT(20e-6)
    droplets = merge(scattered(L, Nd), (r = fill(r₀, Nd), rᵈ = fill(FT(5e-8), Nd), κ = fill(FT(0.6), Nd)))
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
    exchanged = 0.0
    vapor_prev = budget₀.vapor
    for _ in 1:nsteps
        time_step!(model, Δt)
        push!(counts, length(model.microphysics.droplets))
        d = model.microphysics.droplets.properties
        @test all(z -> 0 ≤ z ≤ L, Array(d.z)) # computational positions remain periodic
        # Vapor + liquid + removed − injected is conserved to the rounding budget: condensation moves water between
        # vapor and droplets, and removal moves the droplets' water into the removed tally
        b = water_budget(model)
        exchanged += abs(b.vapor - vapor_prev)
        vapor_prev = b.vapor
        @test b.total ≈ budget₀.total rtol = budget_tolerance(model, budget₀, nsteps, exchanged)
        @test liquid_diagnostic(model) ≈ b.liquid rtol = deposit_tolerance(model)
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
    droplets = merge(scattered(L, Nd), (r = fill(FT(1e-6), Nd), rᵈ = fill(FT(1e-7), Nd), κ = fill(FT(0.6), Nd)))
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
    # Droplets whose virtual position reaches the 2 m wall are removed; compaction keeps the order of the others
    expected_removed = xᵛ₀ .+ u₀ * t .≥ 2
    @test model.microphysics.sources.bookkeeping.removed == count(expected_removed)
    @test length(model.microphysics.droplets) == Nd - count(expected_removed)
    @test xᵛ .- xᵛ₀[.!expected_removed] ≈ fill(u₀ * t, count(.!expected_removed)) rtol = 1e-4
    @test all(x -> 0 ≤ x ≤ L, Array(model.microphysics.droplets.properties.x))

    # The resampled rule of the reference code removes at the expected mean rate 𝕎 τ / Lz per interval
    removal = VirtualChamberRemoval(FT; size=(2, 2, 1), interval=0.05, rule=ResampledVirtualPosition(), seed=5)
    Nd = 4000
    n = 1:Nd
    droplets = merge(scattered(L, Nd), (r = fill(r₀, Nd), rᵈ = fill(FT(5e-8), Nd), κ = fill(FT(0.6), Nd)))
    model = chamber_box(FT; L, droplets, physics, sources=ChamberSources(; removal))
    budget₀ = water_budget(model)
    exchanged = 0.0
    vapor_prev = budget₀.vapor
    for _ in 1:20
        time_step!(model, FT(0.01)) # 4 resampling events
        vapor = water_budget(model).vapor
        exchanged += abs(vapor - vapor_prev)
        vapor_prev = vapor
    end
    removed_fraction = model.microphysics.sources.bookkeeping.removed / Nd
    expected = 1 - (1 - 𝕎 * 0.05 / 1)^3 # three events with 𝕎 known (the first sees 𝕎 = 0 of the initial state)
    @test 0.5 * expected < removed_fraction < 1.5 * expected
    @test water_budget(model).total ≈ budget₀.total rtol = budget_tolerance(model, budget₀, 20, exchanged)
end

@testset "ChamberSources: exact budget when nothing happens [$FT]" for FT in test_float_types()
    # Dry, insoluble aerosol (κ = 0, r = rᵈ) in a subsaturated box neither grows nor evaporates, injected particles are dry
    # (wet radius factor 1) and removal through a small chamber drops them: the vapor must be bit-identical, the liquid
    # identically zero, and the counts exact
    L = FT(0.02)
    Nd = 300
    droplets = merge(scattered(L, Nd), (r = fill(FT(1e-7), Nd), rᵈ = fill(FT(1e-7), Nd), κ = zeros(FT, Nd)))
    physics = DropletPhysicsParameters(FT; terminal_velocity=StokesTerminalVelocity())
    injection = AerosolInjection(FT; rate=1e8, dry_radius=1e-7, hygroscopicity=0, interval=0.05, wet_radius_factor=1, seed=21)
    removal = VirtualChamberRemoval(FT; size=(0.05, 0.05, 0.05), interval=0.05, rule=ResampledVirtualPosition(), seed=22)
    # Quiescent: with a flow, even a uniform vapor field changes by the round-off of the discrete divergence
    model = chamber_box(FT; L, S=0.9, droplets, physics, sources=ChamberSources(; injection, removal))
    vapor₀ = Array(interior(model.moisture_density))
    for _ in 1:100
        time_step!(model, FT(0.01))
    end
    book = model.microphysics.sources.bookkeeping
    @test Array(interior(model.moisture_density)) == vapor₀
    @test water_budget(model).liquid == 0
    @test book.injected_liquid_mass == 0
    @test book.removed_liquid_mass == 0
    @test liquid_diagnostic(model) == 0
    @test abs(book.injected - 1e8 * L^3 * 1.0) ≤ 1
    @test length(model.microphysics.droplets) == Nd + book.injected - book.removed
    @test all(Array(model.microphysics.droplets.properties.r) .== FT(1e-7))
end

@testset "ChamberSources: budget with condensation and evaporation [$FT]" for FT in test_float_types()
    # Radii change (supersaturated: growth; subsaturated: evaporation) while droplets settle out of a shallow chamber;
    # the budget closes within the per-operation rounding budget and the exchanged water is far above it
    L = FT(0.02)
    Nd = 500
    physics = DropletPhysicsParameters(FT; terminal_velocity=StokesTerminalVelocity())
    for (S, r₀) in ((FT(1.01), FT(5e-6)), (FT(0.90), FT(8e-6)))
        droplets = merge(scattered(L, Nd), (r = fill(r₀, Nd), rᵈ = fill(FT(5e-8), Nd), κ = fill(FT(0.6), Nd)))
        removal = VirtualChamberRemoval(FT; size=(2, 2, 0.002), seed=23) # integrated rule: a 2 mm fall to the floor
        model = chamber_box(FT; L, S, droplets, physics, sources=ChamberSources(; removal))
        budget₀ = water_budget(model)
        nsteps = 100
        exchanged = 0.0
        vapor_prev = budget₀.vapor
        for _ in 1:nsteps
            time_step!(model, FT(1e-2))
            b = water_budget(model)
            exchanged += abs(b.vapor - vapor_prev)
            vapor_prev = b.vapor
            @test b.total ≈ budget₀.total rtol = budget_tolerance(model, budget₀, nsteps, exchanged)
        end
        b = water_budget(model)
        @test b.vapor != budget₀.vapor # condensation or evaporation happened
        @test abs(b.vapor - budget₀.vapor) > 1e3 * eps(FT) * budget₀.total # and by much more than the rounding budget
        @test (S > 1) == (b.vapor < budget₀.vapor)
        @test model.microphysics.sources.bookkeeping.removed > 0
        @test liquid_diagnostic(model) ≈ b.liquid rtol = deposit_tolerance(model)
    end
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
    exchanged = 0.0
    vapor_prev = budget₀.vapor
    for _ in 1:200
        time_step!(model, FT(0.01))
        push!(counts, length(model.microphysics.droplets))
        vapor = water_budget(model).vapor
        exchanged += abs(vapor - vapor_prev)
        vapor_prev = vapor
    end
    book = model.microphysics.sources.bookkeeping
    @test book.injected > 0
    @test book.removed > 0
    @test counts[end] == book.injected - book.removed
    @test counts[end] < book.injected
    @test all(isfinite, Array(model.microphysics.droplets.properties.r))
    @test all(Array(model.microphysics.droplets.properties.r) .> 0)
    @test all(isfinite, Array(interior(model.formulation.potential_temperature_density)))
    @test water_budget(model).total ≈ budget₀.total rtol = budget_tolerance(model, budget₀, 200, exchanged)
    @test liquid_diagnostic(model) ≈ water_budget(model).liquid rtol = deposit_tolerance(model)
end
