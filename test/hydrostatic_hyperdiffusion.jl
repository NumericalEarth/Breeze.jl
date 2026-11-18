include(joinpath(@__DIR__, "setup.jl"))

#####
##### The grid-scaled ∇⁴ hyperdiffusion of HydrostaticDynamics on a latitude-longitude grid between
##### walls: damping rates, conservation of the Θ integral, dissipation, the closure at the walls
##### and the preservation of constants (G-B protocol v1.1 A4 and v1.3 C1)
#####

using Breeze
using Breeze.HydrostaticEquations: GridScaleHyperdiffusion, VerletSubsteps, hyperdiffusion_dissipation,
                                   potential_temperature_hyperdiffusion, x_velocity_hyperdiffusion,
                                   y_velocity_hyperdiffusion
using Oceananigans
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: Vᶜᶜᶜ, Vᶠᶜᶜ, Vᶜᶠᶜ
using Oceananigans.TimeSteppers: update_state!
using Printf
using Test

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

const τ₄ = 43200.0 # s: the G-B protocol value, 12 h
const ν₄ = 1 / (64 * τ₄)

# 4° cells between walls at 70°S and 70°N, as in the tests of the dry core
latitude_longitude_grid(; size=(90, 35, 2), latitude=(-70, 70)) =
    LatitudeLongitudeGrid(default_arch, FT; size, longitude=(-180, 180), latitude, z=(0, 1))

function hyperdiffusive_model(grid; damping_time=τ₄, kw...)
    A, B = hybrid_coefficients(size(grid, 3))
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B,
                                   hyperdiffusion=GridScaleHyperdiffusion(FT; damping_time))
    return hydrostatic_model(grid; dynamics, thermodynamic_constants=reference_model_constants(), kw...)
end

# The tendencies that the hyperdiffusion gives ρθ, u and v, as fields
function hyperdiffusion_tendencies(model)
    grid = model.grid
    hyperdiffusion = model.dynamics.hyperdiffusion
    ρ = model.dynamics.density
    θ = model.formulation.potential_temperature
    u, v, _ = model.velocities

    tendencies = (ρθ = Field(KernelFunctionOperation{Center, Center, Center}(potential_temperature_hyperdiffusion, grid, hyperdiffusion, ρ, θ)),
                  u = Field(KernelFunctionOperation{Face, Center, Center}(x_velocity_hyperdiffusion, grid, hyperdiffusion, ρ, u)),
                  v = Field(KernelFunctionOperation{Center, Face, Center}(y_velocity_hyperdiffusion, grid, hyperdiffusion, ρ, v)))

    foreach(compute!, tendencies)
    return tendencies
end

data(field) = Array(interior(field))

function cell_volumes(grid)
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(grid)
    return (c = [Vᶜᶜᶜ(i, j, k, cpu_grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz],
            u = [Vᶠᶜᶜ(i, j, k, cpu_grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz],
            v = [Vᶜᶠᶜ(i, j, k, cpu_grid) for i in 1:Nx, j in 1:Ny+1, k in 1:Nz])
end

# The patterns: the oblique checkerboard (two grid lengths in both directions), which defines
# τ₄, and the waves of two grid lengths along one axis
const patterns = (oblique = (i, j) -> (-1)^(i + j), zonal = (i, j) -> (-1)^i, meridional = (i, j) -> (-1)^j)
const expected_rate = (oblique = 1 / τ₄, zonal = 1 / (4 * τ₄), meridional = 1 / (4 * τ₄))

pattern_array(pattern, Nx, Ny, Nz) = [float(pattern(i, j)) for i in 1:Nx, j in 1:Ny, k in 1:Nz]

# Optional figure data: the directory named by BREEZE_P3B_FIGDATA
figdata_path(name) = (dir = get(ENV, "BREEZE_P3B_FIGDATA", ""); isempty(dir) ? nothing : joinpath(dir, name))

#####
##### Tolerances, written before these tests were run
#####
#
#   1. Damping rates read from the tendency of a 2Δ pattern of amplitude ε on a resting state of
#      uniform θ and surface pressure, rate = - tendency / (ε pattern), for θ, u and v:
#        a. oblique checkerboard: 1 / τ₄ (12 h) within 5 % in the rows next to the equator and
#           at 60° (the G-B protocol's gate; the departure is the meridional variation of the
#           cell mass, second order in the grid spacing);
#        b. zonal 2Δ wave: 1 / (4 τ₄) (48 h) within 1e-9, relative: the mass does not vary along
#           a row, so 𝓛 is the undivided second difference. For θ and u in every row, those at
#           the walls included. For v in the rows at least three faces from a wall: v vanishes
#           on the walls, so a pattern of v that is uniform in latitude does not exist next to
#           them, and 𝓛² reaches two faces.
#           Disclosure: this gate first asked the rows 3 to Ny - 2 of all three fields, and the
#           first run failed it. The failing field was v in the rows within reach of the walls,
#           where the gate was wrong for the reason above, not the operator, which was not
#           changed. (An intermediate edit blamed the rounding of θ and loosened its bound to
#           1e-7; the second run showed θ within 2.4e-11, and that edit was withdrawn.)
#        c. meridional 2Δ wave: 1 / (4 τ₄) within 5 % at the equator and at 60°.
#   2. e-folding times of the three patterns integrated in time under the operator alone
#      (third-order SSP Runge-Kutta, Δt = 600 s, 48 h), from a least-squares fit of the logarithm
#      of the amplitude at the equator: 12 h, 48 h and 48 h within 5 %. And the full model: a
#      resting atmosphere with an oblique checkerboard of 1e-3 K in θ loses the factor e of it
#      in 12 h within 5 % (the dynamics act on the pattern at second order in its amplitude),
#      while the velocities stay below 0.2 m s⁻¹.
#      Disclosure: the velocity bound first read 1e-2 m s⁻¹, from the speed of the gravity waves
#      that the pattern would radiate, and the first run failed it (0.053 m s⁻¹, at the wall row).
#      The state is isentropic, so nothing restores a sheared flow: the pressure-gradient force
#      of the pattern, cᵖᵈ θ′ ΔΠ / Δx ≈ 2e-6 m s⁻², accelerates the layers steadily, to 0.09 m s⁻¹
#      in 12 h without hyperdiffusion (scripts/probe_checkerboard_velocity.jl of the P3b
#      evidence: linear growth, 0.094 m s⁻¹ without the hyperdiffusion, 0.053 m s⁻¹ with it, and
#      exactly zero without the pattern). The bound was wrong, and is now twice that estimate.
#   3. Conservation: the volume integral of the tendency of ρθ vanishes within 1e-12 of the
#      integral of its magnitude, for random θ and a non-uniform surface pressure, and the Θ
#      integral of the full model changes by less than 1e-13, relative, over 20 time steps.
#   4. Dissipation: Σ m ψ ∂ₜψ = - Σ V ε with ε the dissipation rate of `hyperdiffusion_dissipation`,
#      within 1e-11, relative, for random θ, u and v; ε ≥ 0 everywhere and > 0 somewhere.
#   5. Walls: the tendencies agree with an independent implementation that never reads beyond a
#      wall (explicit loops, one-sided at the walls) within 1e-11 of the largest tendency.
#   6. Constants: a uniform θ and a uniform u have tendencies that are exactly zero between the
#      walls; on a periodic grid a uniform v as well.

@testset "Damping rates of 2Δ patterns [$FT]" begin
    grid = latitude_longitude_grid()
    Nx, Ny, Nz = size(grid)
    model = hyperdiffusive_model(grid)
    ε = 1e-3
    rows = (equator = 17:19, sixty = 32:33) # cell centers at 4°S-4°N and at 56°N-60°N; v faces at 58°N, 62°N
    lines = ["field,pattern,latitude_band,rate_per_s,efolding_h,expected_efolding_h"]

    for name in keys(patterns)
        s = pattern_array(patterns[name], Nx, Ny, Nz)
        sᵛ = cat(s, s[:, 1:1, :]; dims=2) # on the Ny + 1 faces: (-1)^j continues

        set!(model; θ = 300 .+ ε .* s, u = ε .* s, v = ε .* sᵛ)
        G = hyperdiffusion_tendencies(model)
        ρ = data(model.dynamics.density)

        rates = (θ = - data(G.ρθ) ./ (ρ .* ε .* s), u = - data(G.u) ./ (ε .* s), v = - data(G.v) ./ (ε .* sᵛ))

        for field in keys(rates), band in keys(rows)
            rate = rates[field][:, rows[band], :]
            tolerance = name == :zonal ? 1e-9 : 0.05
            @test maximum(abs, rate ./ expected_rate[name] .- 1) < tolerance        # gates 1a-1c
            mean_rate = sum(rate) / length(rate)
            push!(lines, @sprintf("%s,%s,%s,%.10e,%.6f,%.1f", field, name, band, mean_rate, 1 / mean_rate / 3600,
                                  1 / expected_rate[name] / 3600))
        end

        # The zonal wave damps at a quarter of the rate in every row: for θ and u at the walls
        # too, for v from the fourth face from a wall
        if name == :zonal
            departure(rate) = maximum(abs, rate ./ expected_rate.zonal .- 1)
            @test departure(rates.θ) < 1e-9
            @test departure(rates.u) < 1e-9
            @test departure(rates.v[:, 4:Ny-2, :]) < 1e-9
            @test departure(rates.v[:, 2:3, :]) > 1e-3 # the wall is felt two faces away
            @info "Zonal 2Δ wave: largest relative departure of the rate from 1 / (4 τ₄)" θ = departure(rates.θ) u = departure(rates.u) v_interior = departure(rates.v[:, 4:Ny-2, :]) v_by_row = [departure(rates.v[:, j:j, :]) for j in 2:6]
        end
    end

    foreach(line -> @info(line), lines)
    path = figdata_path("hyperdiffusion_damping_rates.csv")
    isnothing(path) || write(path, join(lines, "\n") * "\n")
end

# The weak arm of the G-B protocol (v1.3 C1): τ₄ = 48 h, for which the oblique checkerboard damps
# in 48 h and a 2Δ wave along one axis in 192 h. Tolerances, written before this test was run
# (it was added at the freeze of the operator, with no change of the operator):
#   a. the tendencies with 4 τ₄ are a quarter of those with τ₄ within 1e-12 of the largest, for
#      θ, u and v and the three patterns (the coefficient is the only place where τ₄ enters);
#   b. the zonal 2Δ wave of θ and of u damps at 1 / (16 τ₄) (192 h) within 1e-9 in every row.
@testset "The weak arm: rates scale with 1 / τ₄ [$FT]" begin
    grid = latitude_longitude_grid()
    Nx, Ny, Nz = size(grid)
    ε = 1e-3
    strong = hyperdiffusive_model(grid)
    weak = hyperdiffusive_model(grid; damping_time=4τ₄)
    @test strong.dynamics.hyperdiffusion.damping_time == τ₄
    @test weak.dynamics.hyperdiffusion.damping_time == 4τ₄

    for name in keys(patterns)
        s = pattern_array(patterns[name], Nx, Ny, Nz)
        sᵛ = cat(s, s[:, 1:1, :]; dims=2)
        set!(strong; θ = 300 .+ ε .* s, u = ε .* s, v = ε .* sᵛ)
        set!(weak; θ = 300 .+ ε .* s, u = ε .* s, v = ε .* sᵛ)
        G = hyperdiffusion_tendencies(strong)
        Gʷ = hyperdiffusion_tendencies(weak)

        for field in keys(G)
            @test maximum(abs, data(G[field])) > 0
            @test maximum(abs, data(G[field]) .- 4 .* data(Gʷ[field])) ≤ 1e-12 * maximum(abs, data(G[field])) # gate a
        end

        if name == :zonal
            ρ = data(weak.dynamics.density)
            weak_rates = (θ = - data(Gʷ.ρθ) ./ (ρ .* ε .* s), u = - data(Gʷ.u) ./ (ε .* s))
            for field in keys(weak_rates)
                departure = maximum(abs, weak_rates[field] .* (16 * τ₄) .- 1)
                @test departure < 1e-9                                                # gate b
                @info @sprintf("Weak arm (τ₄ = 48 h): zonal 2Δ wave of %s damps in %.6f h (expected 192 h), largest relative departure %.3e",
                               field, Nx * Ny * Nz / sum(weak_rates[field]) / 3600, departure)
            end
        end
    end
end

# Advance `field` under its hyperdiffusion tendency alone, with SSP RK3
function integrate_operator!(field, tendency, divide_by, Δt, steps, amplitude)
    history = [amplitude()]
    ψ⁰ = similar(data(field))

    stage!(α) = begin
        fill_halo_regions!(field)
        compute!(tendency)
        ψ = data(field) .+ Δt .* data(tendency) ./ divide_by
        interior(field) .= on_architecture(architecture(field), (1 - α) .* ψ⁰ .+ α .* ψ)
    end

    for _ in 1:steps
        ψ⁰ .= data(field)
        stage!(1)
        stage!(1 / 4)
        stage!(2 / 3)
        push!(history, amplitude())
    end

    fill_halo_regions!(field)
    return history
end

# e-folding time of a decaying series sampled every Δt, by least squares on its logarithm
function efolding_time(history, Δt)
    t = Δt .* (0:length(history)-1)
    y = log.(history)
    t̄, ȳ = sum(t) / length(t), sum(y) / length(y)
    return - sum(abs2, t .- t̄) / sum((t .- t̄) .* (y .- ȳ))
end

@testset "e-folding times under the operator [$FT]" begin
    grid = latitude_longitude_grid()
    Nx, Ny, Nz = size(grid)
    ε = 1e-3
    Δt, steps = 600, 288
    equator = 17:19
    series = Dict{String, Vector{Float64}}()

    for name in keys(patterns)
        s = pattern_array(patterns[name], Nx, Ny, Nz)
        sᵛ = cat(s, s[:, 1:1, :]; dims=2)

        model = hyperdiffusive_model(grid)
        set!(model; θ = 300 .+ ε .* s, u = ε .* s, v = ε .* sᵛ)
        G = hyperdiffusion_tendencies(model)
        ρ = data(model.dynamics.density)
        θ = model.formulation.potential_temperature
        u, v, _ = model.velocities

        projection(field, s, offset) = () -> sum((data(field)[:, equator, :] .- offset) .* s[:, equator, :]) / length(s[:, equator, :])

        histories = (θ = integrate_operator!(θ, G.ρθ, ρ, Δt, steps, projection(θ, s, 300)),
                     u = integrate_operator!(u, G.u, 1, Δt, steps, projection(u, s, 0)),
                     v = integrate_operator!(v, G.v, 1, Δt, steps, projection(v, sᵛ, 0)))

        for field in keys(histories)
            τ = efolding_time(histories[field], Δt)
            @test τ ≈ 1 / expected_rate[name] rtol=0.05                              # gate 2
            @test all(diff(histories[field]) .< 0)
            series["$(field)_$(name)"] = histories[field] ./ ε
            @info @sprintf("e-folding time of the %s 2Δ pattern of %s at the equator: %.3f h (expected %.0f h)",
                           name, field, τ / 3600, 1 / expected_rate[name] / 3600)
        end
    end

    path = figdata_path("hyperdiffusion_decay.csv")
    if !isnothing(path)
        names = sort(collect(keys(series)))
        open(path, "w") do io
            println(io, "time_h,", join(names, ","))
            for n in 0:steps
                println(io, @sprintf("%.6f", n * Δt / 3600), ",", join((@sprintf("%.8e", series[name][n+1]) for name in names), ","))
            end
        end
    end
end

@testset "Decay of a θ checkerboard in the full model [$FT, $name]" for (name, kw, Δt) in
        (("SSP RK3", (;), 300), ("MIS3C + Verlet", split_explicit(:MIS3C, VerletSubsteps()), 600))

    grid = latitude_longitude_grid()
    Nx, Ny, Nz = size(grid)
    ε = 1e-3
    s = pattern_array(patterns.oblique, Nx, Ny, Nz)
    equator = 17:19

    model = hyperdiffusive_model(grid; kw...)
    set!(model; θ = 300 .+ ε .* s)
    amplitude() = sum((data(model.formulation.potential_temperature)[:, equator, :] .- 300) .* s[:, equator, :]) /
                  (ε * length(s[:, equator, :]))

    for _ in 1:round(Int, τ₄ / Δt)
        time_step!(model, Δt)
    end

    @test amplitude() ≈ exp(-1) rtol=0.05                                            # gate 2
    @test maximum(abs, data(model.velocities.u)) < 0.2
    @info @sprintf("Full model [%s]: amplitude of the θ checkerboard after 12 h: %.4f (e⁻¹ = %.4f)", name, amplitude(), exp(-1))
end

# A deterministic pseudo-random array of numbers in [0, 1): a 64-bit linear congruential generator
# (Knuth's MMIX multiplier and increment), of which the high 53 bits are used. The test environment
# of the package does not have the standard library for random numbers, and these values need no
# statistical quality: they fill states with structure at every scale, uncorrelated from one point
# to the next. (The first final run of this phase failed to load this file, which then drew the
# same states from that library; the checks and their tolerances are unchanged, and only the
# numbers that fill the states differ.)
mutable struct Generator
    state :: UInt64
end

function next!(generator::Generator)
    generator.state = generator.state * 0x5851f42d4c957f2d + 0x14057b7ef767814f
    return (generator.state >> 11) / 2.0^53
end

function pseudo_random(seed::Integer, dims...)
    generator = Generator(seed)
    foreach(_ -> next!(generator), 1:16) # leave the seed behind
    return pseudo_random(generator, dims...)
end

pseudo_random(generator::Generator, dims...) = [next!(generator) for _ in CartesianIndices(dims)]

# A state with structure at every scale, a non-uniform surface pressure, and v = 0 on the walls
function set_random_state!(model; seed=1234)
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    rng = Generator(seed)
    foreach(_ -> next!(rng), 1:16)
    pˢ = 1e5 .+ 2000 .* pseudo_random(rng, Nx, Ny)
    θ = 300 .+ 10 .* pseudo_random(rng, Nx, Ny, Nz)
    u = 20 .* (pseudo_random(rng, Nx, Ny, Nz) .- 0.5)
    v = 20 .* (pseudo_random(rng, Nx, Ny + 1, Nz) .- 0.5)
    v[:, 1, :] .= 0
    v[:, Ny+1, :] .= 0
    set!(model; pˢ, θ, u, v)
    return nothing
end

@testset "Conservation of the Θ integral and dissipation [$FT]" begin
    grid = latitude_longitude_grid(size=(36, 16, 3), latitude=(-64, 64))
    model = hyperdiffusive_model(grid)
    set_random_state!(model)

    V = cell_volumes(grid)
    G = hyperdiffusion_tendencies(model)
    ρ = data(model.dynamics.density)
    θ = data(model.formulation.potential_temperature)
    u = data(model.velocities.u)
    v = data(model.velocities.v)

    # Gate 3: the integral of the tendency of ρθ
    Gρθ = data(G.ρθ)
    @test abs(sum(V.c .* Gρθ)) < 1e-12 * sum(V.c .* abs.(Gρθ))
    @test maximum(abs, Gρθ) > 0

    # Gate 4: the mass-weighted variance and kinetic energy budgets. The masses of the velocity
    # cells are the averages of the masses of the cells on either side.
    m = ρ .* V.c
    mᵘ = (m .+ circshift(m, (1, 0, 0))) ./ 2
    mᵛ = zeros(size(v))
    mᵛ[:, 2:end-1, :] .= (m[:, 1:end-1, :] .+ m[:, 2:end, :]) ./ 2

    ε = map(data ∘ Field, hyperdiffusion_dissipation(model))
    @test all(ε.θ .≥ 0) && all(ε.u .≥ 0) && all(ε.v .≥ 0)
    @test maximum(ε.θ) > 0 && maximum(ε.u) > 0 && maximum(ε.v) > 0

    variance_tendency = sum(m .* θ .* Gρθ ./ ρ)
    @test variance_tendency < 0
    @test variance_tendency ≈ - sum(V.c .* ε.θ) rtol=1e-11

    x_energy_tendency = sum(mᵘ .* u .* data(G.u))
    @test x_energy_tendency < 0
    @test x_energy_tendency ≈ - sum(V.u .* ε.u) rtol=1e-11

    y_energy_tendency = sum(mᵛ .* v .* data(G.v))
    @test y_energy_tendency < 0
    @test y_energy_tendency ≈ - sum(V.v[:, 2:end-1, :] .* ε.v[:, 2:end-1, :]) rtol=1e-11

    # The mass-weighted sum of u, the component along the walls, is conserved as well. That of v
    # is not: v vanishes on the walls, which take up its flux. (This file first asserted the
    # conservation of both; the assertion for v was wrong and failed, and was removed.)
    @test abs(sum(mᵘ .* data(G.u))) < 1e-12 * sum(mᵘ .* abs.(data(G.u)))

    @info "Hyperdiffusion budgets of a random state" variance_tendency x_energy_tendency y_energy_tendency
end

@testset "Θ integral of the full model with hyperdiffusion [$FT, $name]" for (name, kw, Δt) in
        (("SSP RK3", (;), 300), ("MIS3C + Verlet", split_explicit(:MIS3C, VerletSubsteps()), 600))

    grid = latitude_longitude_grid(size=(45, 17, 4), latitude=(-68, 68))
    model = hyperdiffusive_model(grid; coriolis=HydrostaticSphericalCoriolis(FT), kw...)
    set!(model; pˢ = (λ, φ) -> 1e5 + 300 * cosd(2λ) * cosd(φ)^2,
                θ = (λ, φ, z) -> 290 + 60 * z + 3 * sind(3λ) * cosd(φ),
                u = (λ, φ, z) -> 15 * cosd(φ) * (1 + z))

    Θ⁰ = total(model.formulation.potential_temperature_density)
    with_hyperdiffusion = data(model.formulation.potential_temperature_density)

    for _ in 1:20
        time_step!(model, Δt)
    end

    @test abs(total(model.formulation.potential_temperature_density) / Θ⁰ - 1) < 1e-13 # gate 3

    # The hyperdiffusion acted: the same run without it differs
    reference = hydrostatic_model(grid; dynamics=HydrostaticDynamics(pressure_coefficient=hybrid_coefficients(4)[1],
                                                                     surface_pressure_coefficient=hybrid_coefficients(4)[2]),
                                  thermodynamic_constants=reference_model_constants(),
                                  coriolis=HydrostaticSphericalCoriolis(FT), kw...)
    set!(reference; pˢ = (λ, φ) -> 1e5 + 300 * cosd(2λ) * cosd(φ)^2,
                    θ = (λ, φ, z) -> 290 + 60 * z + 3 * sind(3λ) * cosd(φ),
                    u = (λ, φ, z) -> 15 * cosd(φ) * (1 + z))
    for _ in 1:20
        time_step!(reference, Δt)
    end
    difference = maximum(abs, data(model.formulation.potential_temperature) .- data(reference.formulation.potential_temperature))
    @test 1e-8 < difference < 1
end

#####
##### The closure at the walls: an independent implementation with explicit loops
#####

# 𝓛 of a cell-centered array between walls in y, periodic in x: fluxes through the faces between
# cells only, weighted by the mean of the two masses
function reference_center_laplacian(c, m)
    Nx, Ny, Nz = size(c)
    L = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        e, w = mod1(i + 1, Nx), mod1(i - 1, Nx)
        flux = (m[i, j, k] + m[e, j, k]) / 2 * (c[e, j, k] - c[i, j, k]) -
               (m[i, j, k] + m[w, j, k]) / 2 * (c[i, j, k] - c[w, j, k])
        j < Ny && (flux += (m[i, j, k] + m[i, j+1, k]) / 2 * (c[i, j+1, k] - c[i, j, k]))
        j > 1 && (flux -= (m[i, j, k] + m[i, j-1, k]) / 2 * (c[i, j, k] - c[i, j-1, k]))
        L[i, j, k] = flux / m[i, j, k]
    end
    return L
end

# 𝓛 of u: the x-face `i` lies between the cells `i - 1` and `i`
function reference_x_velocity_laplacian(u, m)
    Nx, Ny, Nz = size(u)
    mᵘ = (m .+ circshift(m, (1, 0, 0))) ./ 2
    L = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        e, w = mod1(i + 1, Nx), mod1(i - 1, Nx)
        flux = m[i, j, k] * (u[e, j, k] - u[i, j, k]) - m[w, j, k] * (u[i, j, k] - u[w, j, k])
        j < Ny && (flux += (mᵘ[i, j, k] + mᵘ[i, j+1, k]) / 2 * (u[i, j+1, k] - u[i, j, k]))
        j > 1 && (flux -= (mᵘ[i, j, k] + mᵘ[i, j-1, k]) / 2 * (u[i, j, k] - u[i, j-1, k]))
        L[i, j, k] = flux / mᵘ[i, j, k]
    end
    return L
end

# 𝓛 of v: the y-face `j` lies between the cells `j - 1` and `j`; faces 1 and Ny + 1 are the walls,
# where v and 𝓛v vanish
function reference_y_velocity_laplacian(v, m)
    Nx, Nf, Nz = size(v)
    Ny = Nf - 1
    mᵛ = zeros(Nx, Nf, Nz)
    mᵛ[:, 2:Ny, :] .= (m[:, 1:Ny-1, :] .+ m[:, 2:Ny, :]) ./ 2
    L = zeros(Nx, Nf, Nz)
    for k in 1:Nz, j in 2:Ny, i in 1:Nx
        e, w = mod1(i + 1, Nx), mod1(i - 1, Nx)
        flux = m[i, j, k] * (v[i, j+1, k] - v[i, j, k]) - m[i, j-1, k] * (v[i, j, k] - v[i, j-1, k])
        flux += (mᵛ[i, j, k] + mᵛ[e, j, k]) / 2 * (v[e, j, k] - v[i, j, k])
        flux -= (mᵛ[i, j, k] + mᵛ[w, j, k]) / 2 * (v[i, j, k] - v[w, j, k])
        L[i, j, k] = flux / mᵛ[i, j, k]
    end
    return L
end

@testset "Closure at the walls against an independent implementation [$FT]" begin
    grid = latitude_longitude_grid(size=(12, 7, 2), latitude=(-63, 63))
    model = hyperdiffusive_model(grid)
    set_random_state!(model; seed=99)

    G = hyperdiffusion_tendencies(model)
    ρ = data(model.dynamics.density)
    m = ρ .* cell_volumes(grid).c
    θ = data(model.formulation.potential_temperature)
    u = data(model.velocities.u)
    v = data(model.velocities.v)

    reference = (ρθ = - ν₄ .* ρ .* reference_center_laplacian(reference_center_laplacian(θ, m), m),
                 u = - ν₄ .* reference_x_velocity_laplacian(reference_x_velocity_laplacian(u, m), m),
                 v = - ν₄ .* reference_y_velocity_laplacian(reference_y_velocity_laplacian(v, m), m))

    for name in keys(reference)
        tendency = data(G[name])
        @test maximum(abs, tendency .- reference[name]) < 1e-11 * maximum(abs, reference[name])   # gate 5
        @test maximum(abs, tendency[:, 1:2, :]) > 0 # the rows at the wall are not trivially zero
    end

    # v and its tendency vanish on the walls
    @test all(iszero, data(G.v)[:, 1, :]) && all(iszero, data(G.v)[:, end, :])

    # The halos beyond the walls are not read: overwriting them changes nothing
    θ_field = model.formulation.potential_temperature
    before = data(G.ρθ)
    view(parent(θ_field), :, 1:grid.Hy, :) .= 1e6
    view(parent(θ_field), :, size(parent(θ_field), 2)-grid.Hy+1:size(parent(θ_field), 2), :) .= -1e6
    compute!(G.ρθ)
    @test data(G.ρθ) == before
end

@testset "Constants are preserved [$FT]" begin
    grid = latitude_longitude_grid(size=(36, 16, 3), latitude=(-64, 64))
    model = hyperdiffusive_model(grid)
    Nx, Ny, Nz = size(grid)
    pˢ = 1e5 .+ 2000 .* pseudo_random(7, Nx, Ny)
    set!(model; pˢ, θ = 300, u = 12.5)
    G = hyperdiffusion_tendencies(model)
    @test all(iszero, data(G.ρθ))                                                      # gate 6
    @test all(iszero, data(G.u))

    periodic = RectilinearGrid(default_arch, FT; size=(8, 8, 3), x=(0, 8e5), y=(0, 8e5), z=(0, 1))
    model = hyperdiffusive_model(periodic)
    set!(model; pˢ = 1e5 .+ 2000 .* pseudo_random(8, 8, 8), θ = 300, u = 12.5, v = -7.25)
    G = hyperdiffusion_tendencies(model)
    @test all(iszero, data(G.ρθ)) && all(iszero, data(G.u)) && all(iszero, data(G.v))

    # And the model without hyperdiffusion has none
    @test_throws ArgumentError hyperdiffusion_dissipation(hydrostatic_model(periodic; dynamics=HydrostaticDynamics()))
    @test_throws ArgumentError hyperdiffusive_model(periodic; damping_time=0)
end

# The bare dry core stays selectable: without keywords the dynamics have no hyperdiffusion and no
# vertical diffusion and the model no radiation, before and after materialization, and the
# hyperdiffusion then adds exactly nothing to the tendencies. (Added at the freeze of the
# operator. That this core reproduces the accepted Phase 3a tree bit for bit is the controlled
# baseline of the evidence, scripts/baseline_state_hashes.jl.)
@testset "The bare dry core is the default [$FT]" begin
    dynamics = HydrostaticDynamics()
    @test isnothing(dynamics.hyperdiffusion)
    @test isnothing(dynamics.vertical_diffusion)

    grid = latitude_longitude_grid(size=(12, 7, 2), latitude=(-63, 63))
    A, B = hybrid_coefficients(size(grid, 3))
    model = hydrostatic_model(grid; dynamics=HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B),
                              thermodynamic_constants=reference_model_constants())
    @test isnothing(model.dynamics.hyperdiffusion)
    @test isnothing(model.dynamics.vertical_diffusion)
    @test isnothing(model.radiation)

    set_random_state!(model; seed=5)
    G = hyperdiffusion_tendencies(model)
    @test all(iszero, data(G.ρθ)) && all(iszero, data(G.u)) && all(iszero, data(G.v))
end
