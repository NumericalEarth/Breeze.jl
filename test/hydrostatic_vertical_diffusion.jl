include(joinpath(@__DIR__, "setup.jl"))

#####
##### Implicit vertical diffusion of θ, u and v with HydrostaticDynamics, in mass coordinates:
##### conservation, an analytic solution, the tridiagonal solver against a dense solve, and a
##### convective column with the Richardson-number boundary-layer scheme
#####

using Breeze
using Breeze.HydrostaticEquations: VerticalDiffusion, RichardsonNumberBoundaryLayer, VerletSubsteps, eddy_diffusivity,
                                   validate_diffusivities, step_vertical_diffusion!
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Printf
using Test

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

column_grid(Nz) = RectilinearGrid(default_arch, FT; size=Nz, x=0, y=0, z=(0, 1), topology=(Flat, Flat, Bounded))

column(field) = vec(Array(interior(field)))

function column_model(Nz; vertical_diffusion, hybrid=false, surface_pressure=pˢᵗ, kw...)
    coefficients = hybrid ? hybrid_coefficients(Nz) : (nothing, nothing)
    dynamics = HydrostaticDynamics(; pressure_coefficient=coefficients[1], surface_pressure_coefficient=coefficients[2],
                                     top_pressure=200, surface_pressure, vertical_diffusion)
    return hydrostatic_model(column_grid(Nz); dynamics, thermodynamic_constants=reference_model_constants(), kw...)
end

# Mass of every layer per unit area, Δp / g = ρ Δz
layer_masses(model) = column(model.dynamics.density) ./ size(model.grid, 3)

# Column integrals per unit area: Θ = Σ ρθ Δz, the enthalpy Σ cᵖᵈ Π ρθ Δz, and the momentum Σ μ u
column_Θ(model) = sum(column(model.formulation.potential_temperature_density)) / size(model.grid, 3)
column_enthalpy(model) = sum(cᵖᵈ .* column(model.dynamics.exner_function) .*
                             column(model.formulation.potential_temperature_density)) / size(model.grid, 3)
column_momentum(model, velocity) = sum(layer_masses(model) .* column(velocity))

heights(model) = column(model.dynamics.geopotential) ./ g

figdata_path(name) = (dir = get(ENV, "BREEZE_P3B_FIGDATA", ""); isempty(dir) ? nothing : joinpath(dir, name))

# Solve the dense linear system `M x = b` by Gaussian elimination with partial pivoting: an
# independent solve of the tridiagonal systems of the vertical diffusion, which shares nothing
# with the Thomas algorithm of the model. (The test environment of the package does not have the
# standard linear algebra library. The first final run of this phase failed to load this file,
# which then solved the same system with that library's symmetric tridiagonal solver; the check
# and its tolerance are unchanged.)
function dense_solve(M, b)
    A = Float64.(M)
    x = Float64.(b)
    n = length(x)

    for p in 1:n-1
        pivot = p - 1 + argmax(abs.(A[p:n, p]))
        A[[p, pivot], :] = A[[pivot, p], :]
        x[[p, pivot]] = x[[pivot, p]]
        for r in p+1:n
            factor = A[r, p] / A[p, p]
            A[r, p:n] .-= factor .* A[p, p:n]
            x[r] -= factor * x[p]
        end
    end

    for r in n:-1:1
        x[r] = (x[r] - sum(A[r, c] * x[c] for c in r+1:n; init=0.0)) / A[r, r]
    end

    return x
end

# The matrix of one backward Euler step of the diffusion of a column, from its layer masses μ and
# the conductances C of its z-faces
function diffusion_matrix(μ, C, Δt)
    n = length(μ)
    M = zeros(n, n)
    for k in 1:n
        M[k, k] = μ[k] + Δt * (C[k] + C[k+1])
        k > 1 && (M[k, k-1] = - Δt * C[k])
        k < n && (M[k, k+1] = - Δt * C[k+1])
    end
    return M
end

const steppers = (("SSP RK3", (;)), ("MIS3C + Verlet", split_explicit(:MIS3C, VerletSubsteps())))

#####
##### Tolerances, written before these tests were run
#####
#
#   1. Conservation, column at rest in the horizontal, K = 10 m² s⁻¹, 20 steps of 600 s:
#        a. without boundary fluxes Θ = Σ ρθ Δz and the momenta Σ μ u, Σ μ v keep their initial
#           values within 1e-13, relative, while the profiles change by more than 0.1 K and
#           0.1 m s⁻¹;
#        b. with an energy flux 𝒬 through the ground and a flux J of u, Θ changes by
#           𝒬 t / (cᵖᵈ Π₁) and the momentum by ρ₁ J t, within 1e-12 of Θ and of Σ μ |u|: the flux
#           boundary conditions are the only sources.
#           Disclosure: this gate first asked the column enthalpy Σ cᵖᵈ Π ρθ Δz to change by 𝒬 t,
#           and the first run failed it. The gate was wrong: diffusing θ conserves Θ, and moving
#           θ between layers of different Π changes the enthalpy. Θ is what the flux boundary
#           condition supplies at the rate 𝒬 / (cᵖᵈ Π₁). The scheme was not changed; that the
#           enthalpy is not conserved by a diffusion of θ is reported with the results.
#   2. The solver: one step from a rough profile equals the solution of the same tridiagonal
#      system by a dense solve (built here from the model's conductances and layer masses)
#      within 1e-12, relative; and the conductances equal ρᵃ K / Δh, with ρᵃ = p / (Rᵈ T) on the
#      z-face and Δh from the geopotential of the dynamical core, within 1e-12.
#   3. Analytic solution. With uniform layer masses μ and K set so that the conductance C is the
#      same on every z-face, a cosine mode of the column decays by 1 / (1 + λ Δt) per step,
#      λ = (4 C / μ) sin²(π / 2Nz): after 200 steps the amplitude of the mode in θ and in u is
#      within 1e-4 of that factor (the conductance moves with θ by the 1e-5 that the mode is of
#      it) and within 1 % of the continuous solution exp(-D (π / (pˢ - pᵗ))² t), D = g C Δp
#      (backward Euler at λ Δt = 1 / 200 and 40 layers: 0.3 %).
#   4. The stability functions: K = 0 for a neutral layer at rest; K = ℓ² S for a neutral shear;
#      K = ℓ² (b / c) (-N²)^½ without shear; K decreases with Ri > 0; continuous at N² = 0.
#   5. Convective column: the standard atmosphere on 30 hybrid levels, 200 W m⁻² through the
#      ground for 6 h.
#        a. Without vertical diffusion the lowest layer is superadiabatic by more than 20 K
#           against the layer above (the heat stays where the boundary condition puts it).
#        b. With the boundary-layer scheme Θ still changes by 𝒬 t / (cᵖᵈ Π₁) within 1e-12, and
#           a mixed layer forms: θ of the lowest layer exceeds the coldest θ of the mixed layer
#           by less than 4 K, the spread of θ above the two lowest layers of the mixed layer is
#           less than 2 K, the lowest-layer temperature stays below 320 K, and the top of the
#           mixed layer is within two layers of the encroachment depth, the depth to which the
#           heat supplied mixes the initial profile to a uniform θ. (A local scheme carries the
#           heat flux down a gradient: about 6 K km⁻¹ at 100 m and 1 K km⁻¹ at 500 m for this
#           flux, from ℓ² (b / c) (g |∂θ/∂h| / θ)^½ ∂θ/∂h = 𝒬 / (ρ cᵖᵈ). The gates allow that.)

@testset "Conservation without boundary fluxes [$FT, $name]" for (name, kw) in steppers
    Nz = 20
    model = column_model(Nz; hybrid=true, vertical_diffusion=VerticalDiffusion(diffusivity=10, viscosity=10), kw...)
    set!(model; θ = z -> 290 + 60 * z + 3 * sinpi(8z), u = z -> 10 * z + 2 * cospi(6z), v = z -> 5 * sinpi(3z))

    Θ⁰, U⁰, V⁰ = column_Θ(model), column_momentum(model, model.velocities.u), column_momentum(model, model.velocities.v)
    θ⁰, u⁰ = column(model.formulation.potential_temperature), column(model.velocities.u)

    for _ in 1:20
        time_step!(model, 600)
    end

    @test abs(column_Θ(model) / Θ⁰ - 1) < 1e-13                                       # gate 1a
    @test abs(column_momentum(model, model.velocities.u) / U⁰ - 1) < 1e-13
    @test abs(column_momentum(model, model.velocities.v) / V⁰ - 1) < 1e-13
    @test maximum(abs, column(model.formulation.potential_temperature) .- θ⁰) > 0.1
    @test maximum(abs, column(model.velocities.u) .- u⁰) > 0.1
end

@testset "Flux boundary conditions are the only sources [$FT, $name]" for (name, kw) in steppers
    Nz = 20
    𝒬, J = 120.0, 2e-4 # W m⁻², and the flux of u in units of the grid's vertical coordinate times m s⁻²
    boundary_conditions = (ρE = FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬)),
                           u = FieldBoundaryConditions(bottom=FluxBoundaryCondition(J)))
    model = column_model(Nz; hybrid=true, boundary_conditions,
                         vertical_diffusion=VerticalDiffusion(diffusivity=10, viscosity=10), kw...)
    set!(model; θ = z -> 290 + 60 * z, u = z -> 10 * z)

    Θ⁰, H⁰, U⁰ = column_Θ(model), column_enthalpy(model), column_momentum(model, model.velocities.u)
    ρ₁ = column(model.dynamics.density)[1]
    Π₁ = column(model.dynamics.exner_function)[1]
    Δt, steps = 600, 20

    for _ in 1:steps
        time_step!(model, Δt)
    end

    ΔΘ = column_Θ(model) - Θ⁰
    ΔH = column_enthalpy(model) - H⁰
    ΔU = column_momentum(model, model.velocities.u) - U⁰
    @test abs(ΔΘ - 𝒬 * steps * Δt / (cᵖᵈ * Π₁)) < 1e-12 * Θ⁰                          # gate 1b
    @info "Diffusion of θ under a surface flux [$name]: enthalpy change over the heat supplied" ΔH / (𝒬 * steps * Δt)
    @test abs(ΔU - ρ₁ * J * steps * Δt) < 1e-12 * sum(layer_masses(model) .* abs.(column(model.velocities.u)))
    @test ΔU > 0.5 * ρ₁ * J * steps * Δt

    # The heat was carried up from the lowest layer
    @test column(model.formulation.potential_temperature)[3] > 290 + 60 * (2.5 / Nz) + 0.01
end

@testset "Tridiagonal solver against a dense solve [$FT]" begin
    Nz = 16
    Δt = 900
    model = column_model(Nz; hybrid=true, surface_pressure=97000,
                         vertical_diffusion=VerticalDiffusion(diffusivity = z -> 5 + 40 * z, viscosity = z -> 20 * (1 - z)))
    diffusion = model.dynamics.vertical_diffusion

    for (name, initial) in ((:θ, z -> 285 + 50 * z + 4 * sinpi(11z)), (:u, z -> 12 * z + 3 * cospi(9z)))
        set!(model; θ = z -> 285 + 50 * z, u = 0)
        set!(model; NamedTuple{(name,)}((initial,))...)

        field = name == :θ ? model.formulation.potential_temperature : model.velocities.u
        ψ⁰ = column(field)
        μ = layer_masses(model)

        # One step of the diffusion alone: nothing else acts on a column at rest in the horizontal
        # with these profiles, except that u has no tendency and θ none either
        Breeze.HydrostaticEquations.step_vertical_diffusion!(diffusion, model, Δt)
        update_state!(model)

        C = column(name == :θ ? diffusion.conductances.scalar : diffusion.conductances.momentum)
        @test C[1] == 0 && C[end] == 0
        @test all(C[2:end-1] .> 0)

        expected = dense_solve(diffusion_matrix(μ, C, Δt), μ .* ψ⁰)

        @test column(field) ≈ expected rtol=1e-12                                      # gate 2
        @test maximum(abs, column(field) .- ψ⁰) > 0.1
    end

    # The conductances: ρᵃ K / Δh, with the height between layers from the geopotential
    set!(model; θ = z -> 285 + 50 * z, u = 0)
    Breeze.HydrostaticEquations.step_vertical_diffusion!(diffusion, model, 0)
    A, B = hybrid_coefficients(Nz)
    θ = column(model.formulation.potential_temperature)
    Π = column(model.dynamics.exner_function)
    Πᶠ = column(model.dynamics.interface_exner_function)
    Φ = column(model.dynamics.geopotential)
    K = column(diffusion.diffusivity)

    for k in 2:Nz
        p = A[k] + B[k] * 97000
        θᶠ = ((Π[k-1] - Πᶠ[k]) * θ[k-1] + (Πᶠ[k] - Π[k]) * θ[k]) / (Π[k-1] - Π[k])
        ρᵃ = p / (Rᵈ * Πᶠ[k] * θᶠ)
        Δh = (Φ[k] - Φ[k-1]) / g
        @test column(diffusion.conductances.scalar)[k] ≈ ρᵃ * K[k] / Δh rtol=1e-12
    end
end

@testset "Analytic decay of a cosine mode [$FT]" begin
    Nz = 40
    Δt, steps = 300, 200
    ε = 1e-2
    model = column_model(Nz; vertical_diffusion=VerticalDiffusion(diffusivity=1, viscosity=1)) # pure sigma: uniform Δp
    diffusion = model.dynamics.vertical_diffusion
    mode = [cospi((k - 1/2) / Nz) for k in 1:Nz]

    set!(model; θ = 300)
    μ = layer_masses(model)
    @test maximum(abs, μ ./ μ[1] .- 1) < 1e-12

    # K on every z-face such that the conductance is C everywhere: K = C Δh / ρᵃ
    Δp = (pˢᵗ - 200) / Nz
    D = (pˢᵗ - 200)^2 / (π^2 * steps * Δt) # Pa² s⁻¹: the continuous mode decays by e⁻¹ over the run
    C = D / (g * Δp)
    Π = column(model.dynamics.exner_function)
    Πᶠ = column(model.dynamics.interface_exner_function)
    Φ = column(model.dynamics.geopotential)
    K = zeros(1, 1, Nz + 1)
    for k in 2:Nz
        ρᵃ = (pˢᵗ - (k - 1) * Δp) / (Rᵈ * Πᶠ[k] * 300)
        K[1, 1, k] = C * (Φ[k] - Φ[k-1]) / g / ρᵃ
    end
    set!(diffusion.diffusivity, K)
    set!(diffusion.viscosity, K)
    @info "Analytic mode: diffusivities that give a uniform conductance (m² s⁻¹)" extrema(K[1, 1, 2:Nz])

    set!(model; θ = reshape(300 .+ ε .* mode, 1, 1, Nz), u = reshape(mode, 1, 1, Nz))

    for _ in 1:steps
        time_step!(model, Δt)
    end

    amplitude(ψ) = 2 * sum(ψ .* mode) / Nz
    λ = 4 * C / μ[1] * sinpi(1 / 2Nz)^2
    backward_euler = (1 + λ * Δt)^(-steps)
    continuous = exp(-D * (π / (pˢᵗ - 200))^2 * steps * Δt)

    aᶿ = amplitude(column(model.formulation.potential_temperature) .- 300) / ε
    aᵘ = amplitude(column(model.velocities.u))
    @test continuous ≈ exp(-1)
    @test aᶿ ≈ backward_euler rtol=1e-4                                                # gate 3
    @test aᵘ ≈ backward_euler rtol=1e-4
    @test aᶿ ≈ continuous rtol=1e-2
    @test aᵘ ≈ continuous rtol=1e-2

    # Only the mode is there: the profile is the mode times its amplitude
    @test maximum(abs, column(model.velocities.u) .- aᵘ .* mode) < 1e-4

    @info @sprintf("Analytic mode after %d steps: θ %.6f, u %.6f, backward Euler %.6f, continuous %.6f",
                   steps, aᶿ, aᵘ, backward_euler, continuous)
end

# Gate 6 (written before the run): the diffusivity and the viscosity are an interface that a scheme
# outside the model may write, so each inadmissible value (negative, NaN, +Inf, -Inf), in either
# field, stops the next solve with a DomainError (the explicit stages of that time step have
# been taken by then), and stops the construction of a model with an ArgumentError. The cost of the check is measured, not gated.
@testset "Admissible diffusivities [$FT]" begin
    Nz = 12
    model = column_model(Nz; hybrid=true, vertical_diffusion=VerticalDiffusion(diffusivity=10, viscosity=10))
    diffusion = model.dynamics.vertical_diffusion
    set!(model; θ = z -> 290 + 60 * z, u = z -> 10 * z)
    time_step!(model, 600)
    @test isnothing(validate_diffusivities(diffusion))

    for field in (diffusion.diffusivity, diffusion.viscosity), bad in (-1e-3, NaN, Inf, -Inf)
        K = fill(10.0, 1, 1, Nz + 1)
        K[1, 1, 4] = bad
        set!(field, K)
        @test_throws DomainError time_step!(model, 600)
        @test_throws DomainError validate_diffusivities(diffusion)
        set!(field, 10)
        @test isnothing(validate_diffusivities(diffusion))
    end

    # The state is still usable after the values are repaired
    time_step!(model, 600)
    @test all(isfinite, column(model.formulation.potential_temperature))

    for bad in (-1, NaN, Inf)
        @test_throws ArgumentError column_model(Nz; vertical_diffusion=VerticalDiffusion(diffusivity=bad))
        @test_throws ArgumentError column_model(Nz; vertical_diffusion=VerticalDiffusion(viscosity=bad))
    end

    # Cost on the grid of the surface-heating smoke (90 x 35 x 30), relative to one solve
    grid = LatitudeLongitudeGrid(default_arch, FT; size=(90, 35, 30), longitude=(-180, 180), latitude=(-70, 70), z=(0, 1))
    A, B = hybrid_coefficients(30)
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B,
                                   vertical_diffusion=VerticalDiffusion(boundary_layer=RichardsonNumberBoundaryLayer(FT)))
    model = hydrostatic_model(grid; dynamics, thermodynamic_constants=reference_model_constants())
    set!(model; θ = (λ, φ, z) -> 290 + 60 * z)
    diffusion = model.dynamics.vertical_diffusion
    step_vertical_diffusion!(diffusion, model, 600); validate_diffusivities(diffusion) # compile
    check = minimum(@elapsed(validate_diffusivities(diffusion)) for _ in 1:20)
    solve = minimum(@elapsed(step_vertical_diffusion!(diffusion, model, 600)) for _ in 1:20)
    @test isfinite(check)
    @info @sprintf("Cost of the admissibility check on 90 x 35 x 30 (CPU, %d threads): %.3f ms, against %.3f ms for the whole implicit step that contains it (%.1f %%)",
                   Threads.nthreads(), 1e3 * check, 1e3 * solve, 100 * check / solve)
end

@testset "Stability functions of the boundary-layer scheme [$FT]" begin
    scheme = RichardsonNumberBoundaryLayer(FT)
    h = 200.0
    ℓ = 0.4 * h / (1 + 0.4 * h / 150)
    @test eddy_diffusivity(scheme, h, 0.0, 0.0) == 0                                   # gate 4
    @test eddy_diffusivity(scheme, h, 1e-4, 0.0) ≈ ℓ^2 * 1e-2
    @test eddy_diffusivity(scheme, h, 0.0, -4e-4) ≈ ℓ^2 * (9.4 / 5) * 2e-2
    @test eddy_diffusivity(scheme, h, 1e-4, 1e-4) ≈ ℓ^2 * 1e-2 / (1 + 4.7)^2
    @test eddy_diffusivity(scheme, h, 1e-4, 1e-3) < eddy_diffusivity(scheme, h, 1e-4, 1e-4) < eddy_diffusivity(scheme, h, 1e-4, 0.0)
    @test eddy_diffusivity(scheme, h, 1e-4, -1e-12) ≈ eddy_diffusivity(scheme, h, 1e-4, 1e-12) rtol=1e-6
    @test eddy_diffusivity(scheme, h, 0.0, 1e-4) == 0
    @test eddy_diffusivity(scheme, h, 1e-4, -1e-4) ≈ ℓ^2 * 1e-2 * (1 + 9.4 / (1 + 5))
    @test isfinite(eddy_diffusivity(scheme, 0.0, 0.0, 0.0))
    @test_throws ArgumentError RichardsonNumberBoundaryLayer(FT; asymptotic_mixing_length=0)
    @test_throws ArgumentError column_model(8; vertical_diffusion=VerticalDiffusion(diffusivity=-1))
end

# The layer up to which the supply `ΔΘ` (K kg m⁻²) mixes the profile θ to a uniform value: the
# last layer K to whose θ the layers below can be brought with it
function encroachment_layer(θ, μ, ΔΘ)
    for K in 2:length(θ)
        needed = sum(μ[k] * (θ[K] - θ[k]) for k in 1:K-1)
        needed > ΔΘ && return K - 1
    end
    return length(θ)
end

@testset "Convective column [$FT, $name]" for (name, kw) in steppers
    Nz = 30
    𝒬 = 200.0
    Δt, steps = 600, 36
    boundary_conditions = (; ρE = FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬)))

    function heated_column(vertical_diffusion)
        model = column_model(Nz; hybrid=true, boundary_conditions, vertical_diffusion, kw...)
        T = standard_temperature.(column(model.dynamics.pressure))
        set!(model; θ = reshape(T ./ column(model.dynamics.exner_function), 1, 1, Nz))
        return model
    end

    unmixed = heated_column(nothing)
    mixed = heated_column(VerticalDiffusion(boundary_layer=RichardsonNumberBoundaryLayer(FT)))

    θ⁰ = column(mixed.formulation.potential_temperature)
    h⁰ = heights(mixed)
    Π = column(mixed.dynamics.exner_function)
    μ = layer_masses(mixed)
    Θ⁰ = column_Θ(mixed)
    supplied = 𝒬 * steps * Δt / (cᵖᵈ * Π[1])

    for _ in 1:steps
        time_step!(unmixed, Δt)
        time_step!(mixed, Δt)
    end

    θᵘ = column(unmixed.formulation.potential_temperature)
    θᵐ = column(mixed.formulation.potential_temperature)
    Tᵐ = column(mixed.temperature)

    # Gate 5a: the heat stays in the lowest layer without mixing
    @test θᵘ[1] - θᵘ[2] > 20
    @test θᵘ[2:end] ≈ θ⁰[2:end] rtol=1e-12

    # Gate 5b
    @test abs(column_Θ(mixed) - Θ⁰ - supplied) < 1e-12 * Θ⁰
    top = findlast(k -> θᵐ[k] - θ⁰[k] > 0.1, 1:Nz)        # the highest layer that the heat reached
    expected_top = encroachment_layer(θ⁰, μ, supplied)
    @test top ≥ 4
    @test abs(top - expected_top) ≤ 2
    @test θᵐ[1] - minimum(θᵐ[1:top]) < 4
    @test maximum(θᵐ[3:top]) - minimum(θᵐ[3:top]) < 2
    @test Tᵐ[1] < 320
    @test all(isfinite, θᵐ)
    @test θᵐ[top+2:end] ≈ θ⁰[top+2:end] rtol=1e-6

    K = column(mixed.dynamics.vertical_diffusion.diffusivity)
    @info @sprintf("Convective column [%s]: mixed layer of %d layers (encroachment %d), top at %.0f m; θ₁ - min θ = %.2f K; spread above two layers %.2f K; unmixed θ₁ - θ₂ = %.1f K; largest K = %.0f m² s⁻¹",
                   name, top, expected_top, heights(mixed)[top], θᵐ[1] - minimum(θᵐ[1:top]),
                   maximum(θᵐ[3:top]) - minimum(θᵐ[3:top]), θᵘ[1] - θᵘ[2], maximum(K))

    path = figdata_path("mixed_layer_profiles_$(replace(lowercase(name), " " => "_", "+" => "and")).csv")
    if !isnothing(path)
        hᵐ, hᵘ = heights(mixed), heights(unmixed)
        open(path, "w") do io
            println(io, "layer,pressure_Pa,initial_height_m,initial_theta_K,unmixed_height_m,unmixed_theta_K,mixed_height_m,mixed_theta_K,mixed_temperature_K,diffusivity_at_lower_interface_m2_per_s")
            for k in 1:Nz
                println(io, @sprintf("%d,%.3f,%.3f,%.6f,%.3f,%.6f,%.3f,%.6f,%.6f,%.6f", k, column(mixed.dynamics.pressure)[k],
                                     h⁰[k], θ⁰[k], hᵘ[k], θᵘ[k], hᵐ[k], θᵐ[k], Tᵐ[k], K[k]))
            end
        end
    end
end

@testset "Vertical diffusion on a latitude-longitude grid [$FT, $name]" for (name, kw) in steppers
    grid = LatitudeLongitudeGrid(default_arch, FT; size=(24, 10, 6), longitude=(-180, 180), latitude=(-60, 60), z=(0, 1))
    A, B = hybrid_coefficients(6)
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B,
                                   vertical_diffusion=VerticalDiffusion(boundary_layer=RichardsonNumberBoundaryLayer(FT)))
    model = hydrostatic_model(grid; dynamics, coriolis=HydrostaticSphericalCoriolis(FT),
                              thermodynamic_constants=reference_model_constants(), kw...)
    set!(model; pˢ = (λ, φ) -> 1e5 + 300 * cosd(2λ) * cosd(φ)^2,
                θ = (λ, φ, z) -> 300 - 15 * z + 40 * z^2 + 2 * sind(3λ) * cosd(φ), # the lowest layer is warmer than the second
                u = (λ, φ, z) -> 15 * cosd(φ) * (1 + z))

    Θ⁰ = total(model.formulation.potential_temperature_density)
    time_step!(model, 300)
    largest_diffusivity = maximum(model.dynamics.vertical_diffusion.diffusivity) # of the unstable initial state

    for _ in 2:10
        time_step!(model, 300)
    end

    @test abs(total(model.formulation.potential_temperature_density) / Θ⁰ - 1) < 1e-13
    @test largest_diffusivity > 1
    @test all(isfinite, Array(interior(model.velocities.u)))
    @test all(isfinite, Array(interior(model.velocities.v)))
end
