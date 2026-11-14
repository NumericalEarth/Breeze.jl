include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.Forcings: ScalarVarianceBandForcing
using Oceananigans
using Statistics: mean, var
using Test

# Variance of the interior of a field, ⟨φ²⟩ - ⟨φ⟩², reduced in Float64: the moisture perturbations are 10⁻⁴ of the
# mean, so a single-precision reduction of (ρqᵛ)² would be limited to about eps(Float32) ⟨ρqᵛ⟩² ≈ 10⁻¹¹, the size of the
# variance increment being tested; the stored Float32 values themselves resolve the perturbations to 10⁻³
field_variance(f) = var(Array{Float64}(interior(f)); corrected=false)

@testset "ScalarVarianceBandForcing [$FT]" for FT in test_float_types()
    L = FT(0.2)
    N = 16
    grid = RectilinearGrid(default_arch, FT; size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), topology=(Periodic, Periodic, Periodic))

    r = FT(2e-3) # (kg m⁻³ K)² s⁻¹, on ρθ
    forcing = ScalarVarianceBandForcing(grid; variance_injection_rate=r)
    @test occursin("ScalarVarianceBandForcing(r=0.002)", summary(forcing))
    @test occursin("modes", sprint(show, forcing))

    # A quiescent, buoyancy-free box with a single in-band mode of θ: the band variance ½⟨(ρθ′)²⟩ grows at the
    # rate r, so the variance of ρθ grows at 2r, and the mean is unchanged. No diffusion, so nothing else acts.
    dynamics = BoussinesqDynamics(grid; temperature=290, buoyancy=false)
    model = AtmosphereModel(grid; dynamics, forcing=(; ρθ=forcing), closure=nothing)
    set!(model; θ=(x, y, z) -> 290 + 0.1 * sin(2π * x / L))
    ρθ = model.formulation.potential_temperature_density
    v₀ = field_variance(ρθ)
    m₀ = mean(ρθ)
    @test v₀ > 0
    Δt = FT(0.01)
    nsteps = 50
    for _ in 1:nsteps
        time_step!(model, Δt)
    end
    v₁ = field_variance(ρθ)
    @test v₁ - v₀ ≈ 2r * nsteps * Δt rtol = 0.02
    @test mean(ρθ) ≈ m₀ rtol = 1e-6
    @test maximum(abs, model.velocities.w) == 0

    # With the 1/k² weighting of the NCAR code, E = Σ |φ̂|² / k̃² for the single mode, so the variance grows at 2 r k̃²
    weighted = ScalarVarianceBandForcing(grid; variance_injection_rate=r, inverse_square_weighting=true)
    model = AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid; temperature=290, buoyancy=false),
                            forcing=(; ρθ=weighted), closure=nothing)
    set!(model; θ=(x, y, z) -> 290 + 0.1 * sin(2π * x / L))
    v₀ = field_variance(model.formulation.potential_temperature_density)
    for _ in 1:nsteps
        time_step!(model, Δt)
    end
    k̃ = 2N / L * sin(π / N) # modified wavenumber of the mode n = 1
    @test field_variance(model.formulation.potential_temperature_density) - v₀ ≈ 2r * k̃^2 * nsteps * Δt rtol = 0.02

    # A forcing belongs to one scalar; forcing ρqᵛ needs its own instance
    @test_throws ArgumentError AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid; buoyancy=false),
                                               forcing=(ρθ=forcing, ρqᵛ=forcing))
    moisture = ScalarVarianceBandForcing(grid; variance_injection_rate=1e-10)
    model = AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid; temperature=290, buoyancy=false),
                            forcing=(ρθ=ScalarVarianceBandForcing(grid; variance_injection_rate=r), ρqᵛ=moisture), closure=nothing)
    set!(model; θ=290, qᵗ=(x, y, z) -> 0.01 + 1e-4 * sin(2π * y / L))
    v₀ = field_variance(model.moisture_density)
    for _ in 1:nsteps
        time_step!(model, Δt)
    end
    @test field_variance(model.moisture_density) - v₀ ≈ 2e-10 * nsteps * Δt rtol = 0.02
end
