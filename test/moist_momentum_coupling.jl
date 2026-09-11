include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Test
using GPUArraysCore: @allowscalar

using Oceananigans.TimeSteppers: update_state!
using Breeze.AtmosphereModels: dynamics_mass_fractionᶠᶜᶜ, dynamics_mass_fractionᶜᶜᶠ
using Breeze.CompressibleEquations: assemble_slow_vertical_momentum_tendency!
using Breeze.TimeSteppers: compute_slow_momentum_tendencies!

#####
##### The mixture momentum balance carries -∇p and -g ρ with total ρ = ρᵈ + Σρˣ, but momentum is
##### dry-coupled (ρu = ρᵈ u), so both forces carry the face fraction qᵈ = ρᵈ/ρ. Without it a moist
##### parcel accelerates by 1/(1 - qᵗ) too much. qᵗ = 0 is the negative control.
#####

# The assertions below are exact, so one moist point suffices — any mis-weighting fails at any qᵗ.
const moisture_fractions = (0, 0.02)

uniform_grid() = RectilinearGrid(default_arch; size=(8, 8), halo=(5, 5),
                                 x=(0, 1000), z=(0, 1000),
                                 topology=(Periodic, Flat, Bounded))

# `reference_state = nothing` keeps the full-pressure form (tendency is exactly -∂z p - g ρ, with
# no reference cancellation); uniform ρ and θ make p uniform, isolating gravity.
function uniform_moist_model(time_discretization, qᵗ, grid)
    dynamics = CompressibleDynamics(time_discretization; reference_state=nothing)
    model = AtmosphereModel(grid; dynamics)
    set!(model; θ=300, ρ=1.1, qᵗ=qᵗ, u=0, w=0)
    update_state!(model)
    return model
end

# The tendency that actually reaches ρw: the ordinary one under explicit stepping, the assembled
# slow tendency (where split-explicit puts the PGF and buoyancy) under acoustic substepping.
vertical_momentum_tendency(model, ::ExplicitTimeStepping) =
    Array(interior(model.timestepper.Gⁿ.ρw))

function vertical_momentum_tendency(model, ::SplitExplicitTimeDiscretization)
    substepper = model.timestepper.substepper
    compute_slow_momentum_tendencies!(model)
    assemble_slow_vertical_momentum_tendency!(substepper, model)
    return Array(interior(substepper.slow_vertical_momentum_tendency))
end

@testset "Free fall of a uniform moist column [$FT]" for FT in test_float_types()
    old_FT = Oceananigans.defaults.FloatType
    Oceananigans.defaults.FloatType = FT
    grid = uniform_grid()

    time_discretizations = (ExplicitTimeStepping(), SplitExplicitTimeDiscretization(substeps=4))

    for time_discretization in time_discretizations, qᵗ in moisture_fractions
        model = uniform_moist_model(time_discretization, qᵗ, grid)
        g = FT(model.thermodynamic_constants.gravitational_acceleration)
        ρᵈ = Array(interior(model.dynamics.dry_density))
        ρ = Array(interior(model.dynamics.total_density))
        p = Array(interior(model.dynamics.pressure))
        Gρw = vertical_momentum_tendency(model, time_discretization)

        # Sanity: moist, uniform, uniform pressure.
        @test ρᵈ[2, 1, 2] ≈ (1 - qᵗ) * ρ[2, 1, 2]
        @test maximum(p) - minimum(p) ≈ 0 atol=eps(FT) * maximum(p)

        @test @allowscalar(dynamics_mass_fractionᶜᶜᶠ(2, 1, 3, grid, model.dynamics)) ≈ 1 - qᵗ
        @test @allowscalar(dynamics_mass_fractionᶠᶜᶜ(2, 1, 3, grid, model.dynamics)) ≈ 1 - qᵗ

        for k in 2:size(grid, 3)
            ρᵈᶜᶜᶠ = (ρᵈ[2, 1, k] + ρᵈ[2, 1, k - 1]) / 2
            # dw/dt = Gρw / ℑᶻ(ρᵈ) is -g whatever the moisture load.
            @test Gρw[2, 1, k] / ρᵈᶜᶜᶠ ≈ -g rtol=10eps(FT)
        end
    end

    # Anelastic carries a single density, so the weight is exactly one.
    anelastic_model = AtmosphereModel(grid)
    set!(anelastic_model; θ=300, qᵗ=0.02)
    @test @allowscalar(dynamics_mass_fractionᶜᶜᶠ(2, 1, 3, grid, anelastic_model.dynamics)) == 1
    @test @allowscalar(dynamics_mass_fractionᶠᶜᶜ(2, 1, 3, grid, anelastic_model.dynamics)) == 1

    Oceananigans.defaults.FloatType = old_FT
end

@testset "Horizontal pressure acceleration of moist air [$FT]" for FT in test_float_types()
    old_FT = Oceananigans.defaults.FloatType
    Oceananigans.defaults.FloatType = FT

    Lx = 1000
    Nx = 8
    grid = RectilinearGrid(default_arch; size=(Nx, 8), halo=(5, 5),
                           x=(0, Lx), z=(0, 1000),
                           topology=(Periodic, Flat, Bounded))

    for qᵗ in moisture_fractions
        dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_state=nothing)
        model = AtmosphereModel(grid; dynamics)
        # Varying ρ gives a horizontal pressure gradient; at rest, Gρu is that force alone.
        ρᵢ(x, z) = 1.1 * (1 + 0.05 * sinpi(2x / Lx))
        set!(model; θ=300, ρ=ρᵢ, qᵗ=qᵗ, u=0, w=0)
        update_state!(model)

        ρᵈ = Array(interior(model.dynamics.dry_density))
        ρ = Array(interior(model.dynamics.total_density))
        p = Array(interior(model.dynamics.pressure))
        Gρu = Array(interior(model.timestepper.Gⁿ.ρu))
        Δx = FT(Lx / Nx)

        @test maximum(p) - minimum(p) > 0

        for i in 2:Nx, k in 2:size(grid, 3)
            ρᵈᶠᶜᶜ = (ρᵈ[i, 1, k] + ρᵈ[i - 1, 1, k]) / 2
            ρᶠᶜᶜ = (ρ[i, 1, k] + ρ[i - 1, 1, k]) / 2
            ∂x_p = (p[i, 1, k] - p[i - 1, 1, k]) / Δx
            # du/dt = Gρu / ℑˣ(ρᵈ) is the mixture acceleration -∂x p / ℑˣ(ρ).
            @test Gρu[i, 1, k] / ρᵈᶠᶜᶜ ≈ -∂x_p / ρᶠᶜᶜ rtol=100eps(FT)
        end
    end

    Oceananigans.defaults.FloatType = old_FT
end

@testset "Moist split-explicit bubble runs with weighted forces [$FT]" for FT in test_float_types()
    old_FT = Oceananigans.defaults.FloatType
    Oceananigans.defaults.FloatType = FT

    # End-to-end: the slow assembly, horizontal step, and tridiagonal matrix/RHS pair must stay
    # consistent or the vertical solve diverges. Dry runs cannot see this.
    grid = RectilinearGrid(default_arch; size=(16, 16), halo=(5, 5),
                           x=(0, 8000), z=(0, 8000),
                           topology=(Periodic, Flat, Bounded))

    td = SplitExplicitTimeDiscretization(substeps=6)
    dynamics = CompressibleDynamics(td; reference_potential_temperature=300)
    model = AtmosphereModel(grid; advection=WENO(), dynamics)

    ref = model.dynamics.reference_state
    θᵢ(x, z) = 300 + 2 * max(0, 1 - sqrt((x - 4000)^2 + (z - 3000)^2) / 2000)
    set!(model; θ=θᵢ, qᵗ=0.02, ρ=ref.density)

    for _ in 1:5
        time_step!(model, 3)
    end

    w = Array(interior(model.velocities.w))
    @test !any(isnan, w)
    @test !any(isnan, parent(model.dynamics.dry_density))
    @test maximum(abs, w) < 10  # a 2 K bubble cannot produce anything like this in 15 s

    Oceananigans.defaults.FloatType = old_FT
end
