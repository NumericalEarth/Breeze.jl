include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Operators: divᶜᶜᶜ, ∂xᶜᶜᶜ, ∂yᶠᶠᶜ, ∂zᶠᶜᶠ, ∂xᶠᶠᶜ, ∂yᶜᶜᶜ, ∂zᶜᶠᶠ, ∂xᶠᶜᶠ, ∂yᶜᶠᶠ, ∂zᶜᶜᶜ
using Statistics: mean
using Test

using Breeze.Forcings: advance_amplitudes!, number_of_modes, projected_amplitudes

const velocity_locations = ((Face, Center, Center), (Center, Face, Center), (Center, Center, Face))

triply_periodic_grid(FT, N, L=2π) =
    RectilinearGrid(default_arch, FT; size=(N, N, N), extent=(L, L, L), topology=(Periodic, Periodic, Periodic))

band_forced_model(grid, forcing; kw...) =
    AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid), forcing=(u=forcing, v=forcing, w=forcing), kw...)

# The forcing of each velocity component, evaluated by the kernel function of the band forcing
evaluate_forcing(i, j, k, grid, forcing, clock, fields) = forcing(i, j, k, grid, clock, fields)

function forcing_fields(model)
    names = (:ρu, :ρv, :ρw)
    return map(names, velocity_locations) do name, location
        forcing = model.forcing[name].forcing # unwrap the SpecificForcing
        operation = KernelFunctionOperation{location...}(evaluate_forcing, model.grid, forcing,
                                                         model.clock, fields(model))
        Field(operation)
    end
end

power_density(i, j, k, grid, φ, forcing, clock, fields) = @inbounds φ[i, j, k] * forcing(i, j, k, grid, clock, fields)

# The power input ⟨u ⋅ F⟩ is the sum of the means of these fields
function power_density_fields(model)
    names = (:ρu, :ρv, :ρw)
    return map(names, model.velocities, velocity_locations) do name, φ, location
        forcing = model.forcing[name].forcing
        Field(KernelFunctionOperation{location...}(power_density, model.grid, φ, forcing, model.clock, fields(model)))
    end
end

squared_derivative(i, j, k, grid, ∂, φ) = ∂(i, j, k, grid, φ)^2

# ν times the means of these fields is the viscous dissipation rate ν ⟨|∇u|²⟩, computed with the
# second-order differences of the viscous term, so that it balances the power input in a steady state
function squared_gradient_fields(model)
    u, v, w = model.velocities
    derivatives = (((Center, Center, Center), ∂xᶜᶜᶜ, u), ((Face, Face, Center), ∂yᶠᶠᶜ, u), ((Face, Center, Face), ∂zᶠᶜᶠ, u),
                   ((Face, Face, Center), ∂xᶠᶠᶜ, v), ((Center, Center, Center), ∂yᶜᶜᶜ, v), ((Center, Face, Face), ∂zᶜᶠᶠ, v),
                   ((Face, Center, Face), ∂xᶠᶜᶠ, w), ((Center, Face, Face), ∂yᶜᶠᶠ, w), ((Center, Center, Center), ∂zᶜᶜᶜ, w))

    return map(derivatives) do (location, ∂, φ)
        Field(KernelFunctionOperation{location...}(squared_derivative, model.grid, ∂, φ))
    end
end

function sum_of_means(fields)
    total = 0.0
    for field in fields
        compute!(field)
        total += mean(field)
    end
    return total
end

forcing_power(model) = sum_of_means(power_density_fields(model))

# The work done by a forcing held fixed over a time step is Δt ⟨u ⋅ F⟩ + Δt² ⟨|F|²⟩ / 2 to leading order,
# with u at the start of the step. For the stochastic forcing, which changes every time step,
# the second term is a large part of the power input.
self_work_rate(::ConstantPowerBandForcing, Δt) = 0
self_work_rate(forcing::StochasticBandForcing, Δt) = Δt * sum(abs2, Array(forcing.amplitudes)) # ⟨|F|²⟩ = 2 Σ |A|²

kinetic_energy(model) = (mean(model.velocities.u^2) + mean(model.velocities.v^2) + mean(model.velocities.w^2)) / 2

# An Arnold-Beltrami-Childress flow, which lives in the forced band and is discretely divergence-free
abc_flow(U) = (u = (x, y, z) -> U * (sin(z) + cos(y)),
               v = (x, y, z) -> U * (sin(x) + cos(z)),
               w = (x, y, z) -> U * (sin(y) + cos(x)))

# A flow with random amplitudes in the modes with |n|² = 1 or 2 of a 2π-periodic domain, plus noise,
# with velocity variance U² per component
function random_band_flow(U)
    wavevectors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1))
    amplitudes = map(wavevectors) do n
        a = randn(3) .+ im .* randn(3)
        a .- n .* (sum(n .* a) / sum(abs2, n)) # perpendicular to n
    end
    variance = sum(sum(abs2, a) for a in amplitudes) * 2 / 3 # 2 Re Σ ⇒ ⟨|u|²⟩ = 2 Σ |a|²
    scale = U / sqrt(variance)
    component(c) = (x, y, z) -> scale * 2 * sum(real(a[c] * exp(im * (n[1] * x + n[2] * y + n[3] * z)))
                                                for (n, a) in zip(wavevectors, amplitudes)) + U / 10 * randn()
    return (u = component(1), v = component(2), w = component(3))
end

@testset "Band forcing construction [$FT]" for FT in test_float_types()
    grid = triply_periodic_grid(FT, 8)

    forcing = ConstantPowerBandForcing(grid; energy_injection_rate=0.1)
    @test number_of_modes(forcing.band) == 9 # |n|² = 1 or 2, one of each conjugate pair
    @test forcing.energy_injection_rate isa FT
    @test sprint(show, forcing) == "ConstantPowerBandForcing(ε=0.1) with 9 modes with 0.0 ≤ |k| ≤ 1.5"

    stochastic = StochasticBandForcing(grid; energy_injection_rate=0.1, correlation_time=0.5, seed=3)
    @test number_of_modes(stochastic.band) == 9
    @test startswith(sprint(show, stochastic), "StochasticBandForcing(ε=0.1, T_f=0.5)")

    # 1 ≤ |n| ≤ √6: 6 + 12 + 8 + 6 + 24 + 24 = 80 wavevectors
    wide = ConstantPowerBandForcing(grid; energy_injection_rate=0.1, minimum_wavenumber=1, maximum_wavenumber=2.5)
    @test number_of_modes(wide.band) == 40

    model = band_forced_model(grid, forcing)
    @test summary(model.forcing.ρu.forcing) == "BandForcingComponent(u) of ConstantPowerBandForcing(ε=0.1)"

    # Band forcings act on velocity, not momentum
    @test_throws ArgumentError AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid), forcing=(; ρu=forcing))

    # Band forcings need a triply-periodic grid that resolves the band
    bounded_grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), extent=(2π, 2π, 2π),
                                   topology=(Periodic, Periodic, Bounded))
    @test_throws ArgumentError ConstantPowerBandForcing(bounded_grid; energy_injection_rate=0.1)
    @test_throws ArgumentError ConstantPowerBandForcing(grid; energy_injection_rate=0.1, maximum_wavenumber=4.5)
    @test_throws ArgumentError ConstantPowerBandForcing(grid; energy_injection_rate=0.1, minimum_wavenumber=1.6, maximum_wavenumber=1.7)
end

@testset "ConstantPowerBandForcing injects energy at the rate ε [$FT]" for FT in test_float_types()
    grid = triply_periodic_grid(FT, 16)
    ε = 0.1

    @testset "Power input and divergence of a random state" begin
        forcing = ConstantPowerBandForcing(grid; energy_injection_rate=ε)
        model = band_forced_model(grid, forcing)
        ξ(x, y, z) = randn()
        set!(model, u=ξ, v=ξ, w=ξ)
        time_step!(model, 1e-3)

        # The power input is exactly ε
        @test forcing_power(model) ≈ ε rtol=sqrt(eps(FT))

        # The forcing is discretely divergence-free...
        F = forcing_fields(model)
        δ = Field(KernelFunctionOperation{Center, Center, Center}(divᶜᶜᶜ, grid, F...))
        @test maximum(abs, δ) < 10 * eps(FT) * maximum(abs, F[1]) * size(grid, 1)

        # ... and the velocity stays divergence-free
        time_step!(model, 1e-3)
        δ = Field(KernelFunctionOperation{Center, Center, Center}(divᶜᶜᶜ, grid, model.momentum...))
        @test maximum(abs, δ) < sqrt(eps(FT)) * maximum(abs, model.momentum.ρu) * size(grid, 1)
    end

    @testset "Forced Arnold-Beltrami-Childress flow without advection or viscosity" begin
        forcing = ConstantPowerBandForcing(grid; energy_injection_rate=ε)
        model = band_forced_model(grid, forcing; advection=nothing)
        set!(model; abc_flow(1)...)

        u₀ = Array(interior(model.velocities.u))
        E₀ = kinetic_energy(model)
        @test E₀ ≈ 3/2 rtol=sqrt(eps(FT))

        Δt = 0.05
        for _ in 1:40
            time_step!(model, Δt)
        end

        # The energy grows linearly at the rate ε and the flow keeps its shape
        t = model.clock.time
        E = kinetic_energy(model)
        @test E ≈ E₀ + ε * t rtol=sqrt(eps(FT))
        u = Array(interior(model.velocities.u))
        @test u ≈ sqrt(E / E₀) * u₀ rtol=sqrt(eps(FT))
    end
end

@testset "StochasticBandForcing amplitudes [$FT]" for FT in test_float_types()
    grid = triply_periodic_grid(FT, 8)
    ε = 0.1
    T = 0.5

    # Reproducible with a seed
    amplitudes = map((1, 1, 2)) do seed
        forcing = StochasticBandForcing(grid; energy_injection_rate=ε, correlation_time=T, seed)
        advance_amplitudes!(forcing, 0.0, 0)
        advance_amplitudes!(forcing, 0.1, 1)
        copy(forcing.state.amplitudes)
    end
    @test amplitudes[1] == amplitudes[2]
    @test amplitudes[1] != amplitudes[3]

    # Statistics of the projected amplitudes: ⟨|F|²⟩ = 2 Σₘ |Aₘ|² = ε / T, and isotropy
    forcing = StochasticBandForcing(grid; energy_injection_rate=ε, correlation_time=T, seed=7)
    δt = T # nearly independent samples
    samples = 4000
    variances = zeros(3)
    lagged_covariance = 0.0
    variance = 0.0

    advance_amplitudes!(forcing, 0.0, 0)
    for n in 1:samples
        b⁻ = copy(forcing.state.amplitudes)
        advance_amplitudes!(forcing, n * δt, n)
        b = forcing.state.amplitudes
        A = projected_amplitudes(forcing)
        for c in 1:3
            variances[c] += 2 * sum(abs2, view(A, :, c)) / samples
        end
        lagged_covariance += real(sum(conj(b⁻) .* b)) / samples
        variance += sum(abs2, b) / samples
    end

    F² = sum(variances)
    @test F² ≈ ε / T rtol=0.05
    @test all(isapprox.(variances, F² / 3, rtol=0.08))
    @test lagged_covariance / variance ≈ exp(-δt / T) atol=0.03
end

@testset "StochasticBandForcing injects energy at the rate ε on average [$FT]" for FT in test_float_types()
    grid = triply_periodic_grid(FT, 8)
    ε = 0.1
    T = 0.05
    Δt = 0.01
    steps = 100

    # Without advection or viscosity, starting from rest, u = Δt Σₙ Fₙ after `steps` steps,
    # so the expected energy follows from the correlation exp(-Δt / T) of the piecewise-constant forcing.
    r = exp(-Δt / T)
    correlation_sum = sum(r^abs(n - n′) for n in 1:steps, n′ in 1:steps)
    expected_energy = Δt^2 / 2 * (ε / T) * correlation_sum

    seeds = 1:3
    energies = map(seeds) do seed
        forcing = StochasticBandForcing(grid; energy_injection_rate=ε, correlation_time=T, seed,
                                        minimum_wavenumber=1, maximum_wavenumber=2.5)
        model = band_forced_model(grid, forcing; advection=nothing)
        for _ in 1:steps
            time_step!(model, Δt)
        end

        # The forcing is divergence-free
        F = forcing_fields(model)
        δ = Field(KernelFunctionOperation{Center, Center, Center}(divᶜᶜᶜ, grid, F...))
        @test maximum(abs, δ) < 10 * eps(FT) * maximum(abs, F[1]) * size(grid, 1)

        kinetic_energy(model)
    end

    # 40 modes × 2 directions × 2 (real and imaginary parts) × 3 seeds = 480 independent
    # Gaussian processes, whose energies have a relative standard deviation √2: 6.5% for the mean.
    @test mean(energies) ≈ expected_energy rtol=0.25

    # The mean power is approximately ε for t ≫ T
    t = steps * Δt
    @test mean(energies) / t ≈ ε * (1 - T / t) rtol=0.3
end

@testset "Statistically stationary forced turbulence [$FT]" for FT in test_float_types()
    # 32³ on the GPU; 16³ on the CPU to keep the test short
    N = default_arch isa GPU ? 32 : 16
    grid = triply_periodic_grid(FT, N)
    ε = 0.1
    ν = 0.04 * 16 / N

    @testset "$(nameof(typeof(forcing)))" for forcing in (ConstantPowerBandForcing(grid; energy_injection_rate=ε),
                                                          StochasticBandForcing(grid; energy_injection_rate=ε,
                                                                                correlation_time=0.2, seed=1))
        model = band_forced_model(grid, forcing; closure=ScalarDiffusivity(ν=ν))
        set!(model; random_band_flow(0.5)...)

        Δt = 0.25 * 2π / N
        spin_up_time = 20
        averaging_time = 30
        simulation = Simulation(model; Δt, stop_time=spin_up_time, verbose=false)
        run!(simulation)

        E₁ = kinetic_energy(model)
        power_densities = power_density_fields(model)
        squared_gradients = squared_gradient_fields(model)
        statistics = []

        function record(sim)
            u, v, w = sim.model.velocities
            push!(statistics, (power = sum_of_means(power_densities) + self_work_rate(forcing, Δt),
                               dissipation = ν * sum_of_means(squared_gradients),
                               u² = mean(u^2), v² = mean(v^2), w² = mean(w^2)))
            return nothing
        end

        add_callback!(simulation, record, IterationInterval(1))
        simulation.stop_time = spin_up_time + averaging_time
        run!(simulation)
        E₂ = kinetic_energy(model)

        P = mean(s.power for s in statistics)
        D = mean(s.dissipation for s in statistics)
        u² = mean(s.u² for s in statistics)
        v² = mean(s.v² for s in statistics)
        w² = mean(s.w² for s in statistics)

        @info "Stationary turbulence" N FT forcing P D E₁ E₂ u² v² w²

        # The energy budget closes...
        @test P ≈ D + (E₂ - E₁) / averaging_time rtol=0.02

        # ... and the dissipation rate balances the target energy injection rate
        forcing isa ConstantPowerBandForcing && @test all(isapprox(s.power, ε, rtol=sqrt(eps(FT))) for s in statistics)
        @test D ≈ ε rtol=0.3

        # The velocity variance is nearly isotropic. The stochastic forcing is statistically isotropic;
        # the constant-power forcing amplifies the modes it finds, including their anisotropy,
        # which decays only through the nonlinear dynamics.
        q² = (u² + v² + w²) / 3
        isotropy_tolerance = forcing isa StochasticBandForcing ? 0.25 : 0.5
        @test all(isapprox.((u², v², w²), q², rtol=isotropy_tolerance))
    end
end
