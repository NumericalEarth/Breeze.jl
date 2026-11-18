include(joinpath(@__DIR__, "setup.jl"))

#####
##### The grid-scaled ∇⁴ hyperdiffusion of HydrostaticDynamics on a YinYangGrid: the owned Θ
##### integral with and without the conservation fixer, the seam budget of the operator, its
##### damping rate inside a panel, constants, and the sign of the dissipation
#####

using Breeze
using Breeze.HydrostaticEquations: GridScaleHyperdiffusion, VerticalDiffusion, RichardsonNumberBoundaryLayer,
                                   validate_velocity_conductances,
                                   hyperdiffusion_dissipation, potential_temperature_hyperdiffusion
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.MultiRegion: YinYangGrid
using Test

include(joinpath(@__DIR__, "hydrostatic_yin_yang_utils.jl"))

const τ₄ = 43200.0

function hyperdiffusive_crossing_flow(grid; kw...)
    A, B = hybrid_coefficients(size(grid, 3))
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B,
                                   hyperdiffusion=GridScaleHyperdiffusion(FT; damping_time=τ₄))
    model = hydrostatic_model(grid; dynamics, coriolis=HydrostaticSphericalCoriolis(FT; rotation_rate),
                              scalar_advection=(; ρθ=WENO(FT; order=5)),
                              thermodynamic_constants=reference_model_constants(), kw...)

    set!(model; pˢ = (λ, φ) -> 1e5 + 500 * cosd(2λ) * cosd(φ)^2,
                θ = (λ, φ, z) -> 290 + 60 * z + 2 * sind(λ) * sind(2φ),
                u = (λ, φ, z) -> 20 * (1 + z) * cosd(φ) + 5 * sind(λ),
                v = (λ, φ, z) -> 5 * z * sind(2λ) * cosd(φ))
    return model
end

# The tendency that the hyperdiffusion gives ρθ, as a field
function hyperdiffusion_tendency(model)
    tendency = Field(KernelFunctionOperation{Center, Center, Center}(potential_temperature_hyperdiffusion, model.grid,
                                                                     model.dynamics.hyperdiffusion, model.dynamics.density,
                                                                     model.formulation.potential_temperature))
    compute!(tendency)
    return tendency
end

#####
##### Tolerances, written before these runs
#####
#
# GATES (asserted):
#   1. With the conservation fixer, the owned Θ integral of a flow that crosses the seam, with
#      the hyperdiffusion acting, is within 1e-12 of its initial value, relative, after 20 time
#      steps, for SSP RK3 and both split-explicit candidates: the bound of the conservation tests
#      of the core without hyperdiffusion, since the fixer returns the seam budget of the
#      hyperdiffusive fluxes with that of the advective ones.
#   2. Inside a panel, at least six cells from its edges, the oblique checkerboard damps at
#      1 / τ₄ within 5 %.
#   3. A uniform θ has a tendency below 1e-12 θ ρ / τ₄: the interpolated halos of a constant are
#      that constant to rounding.
#   4. The owned integral of θ times the tendency of θ, the rate of change of the mass-weighted
#      variance by the hyperdiffusion, is negative for a state with structure at the grid scale.
#   5. With the implicit vertical diffusion and the boundary-layer scheme as well, on both
#      panels, and the fixer: the owned Θ integral and the mass after 10 steps of SSP RK3 are
#      within 1e-12 of their initial values (the vertical diffusion moves θ within a column and
#      changes neither), the state is finite, and the scheme mixed (K > 1 m² s⁻¹ somewhere in the
#      first step) the unstable lowest layers that the state is given. The conductances at the
#      velocity points, which next to the edge of a panel average a computed value with one
#      interpolated from the other panel, are not negative (the guard of the solve passes).
#   6. A prescribed viscosity that jumps from 0 to 100 m² s⁻¹ across a circle through the seam is a
#      case in which the cubic interpolation of the halos may undershoot zero. Either the
#      conductances at the velocity points stay admissible and the step completes, or the solve
#      refuses with a DomainError: no step is taken with a negative conductance. Which of the two
#      happens is recorded.
#
# MEASUREMENTS (recorded; finiteness asserted): the drift of the owned Θ integral without the
# fixer; the seam budget of the hyperdiffusion alone, as the owned integral of its tendency over
# the owned integral of the magnitude of its tendency; and the ratio of the variance tendency of
# gate 4 to minus the owned integral of the dissipation rate, which is one on a single grid and
# departs from it by what the panels exchange through their interpolated halos.

@testset "Owned Θ integral with hyperdiffusion [$(scheme.name), fixer $(fixer ? "on" : "off")]" for scheme in schemes, fixer in (false, true)
    resolution = first(resolutions)
    grid = yin_yang_grid(resolution; Nz=6)
    Δt = time_step(scheme, resolution)
    steps = 20

    model = hyperdiffusive_crossing_flow(grid; scheme.kw...)
    callbacks = fixer ? conservation_fixer_callbacks(model) : []
    Θ⁰ = total(model.formulation.potential_temperature_density)
    M⁰ = total(continuity_density(model.dynamics))

    for _ in 1:steps
        time_step!(model, Δt; callbacks)
    end

    ΔΘ = total(model.formulation.potential_temperature_density) / Θ⁰ - 1
    ΔM = total(continuity_density(model.dynamics)) / M⁰ - 1
    case = "$(resolution.name), $(scheme.name), Δt=$Δt, hyperdiffusion τ₄=$(τ₄) s, fixer $(fixer ? "on" : "off")"
    record("hyperdiffusion", case, "relative change of Θ over $steps steps", ΔΘ)
    record("hyperdiffusion", case, "relative change of mass over $steps steps", ΔM)

    @test isfinite(ΔΘ)
    fixer && @test abs(ΔΘ) < 1e-12                                # gate 1
    fixer && @test abs(ΔM) < 1e-12
end

@testset "The hyperdiffusion operator on the panels" begin
    resolution = first(resolutions)
    grid = yin_yang_grid(resolution; Nz=3)
    cells = sphere_cells(grid)
    Nx, Ny, Nz′ = size(grid)
    model = hyperdiffusive_crossing_flow(grid)
    ρ = panels(model.dynamics.density)

    # Gate 2: the oblique checkerboard, in the indices of each panel
    ε = 1e-3
    s = [float((-1)^(i + j)) for i in 1:Nx, j in 1:Ny, k in 1:Nz′]
    θ = model.formulation.potential_temperature
    for r in regions_of(grid)
        interior(region(θ, r)) .= on_architecture(default_arch, 300 .+ ε .* s)
    end
    fill_halo_regions!(θ)
    G = panels(hyperdiffusion_tendency(model))
    inner = (7:Nx-6, 7:Ny-6, 1:Nz′)
    for r in eachindex(G)
        rate = - G[r][inner...] ./ (ρ[r][inner...] .* ε .* s[inner...])
        @test maximum(abs, rate .* τ₄ .- 1) < 0.05
        record("hyperdiffusion", "$(resolution.name), panel $r", "e-folding time of the oblique checkerboard inside the panel, h",
               1 / (sum(rate) / length(rate)) / 3600)
    end

    # Gate 3: constants
    set!(model; pˢ = (λ, φ) -> 1e5 + 500 * cosd(2λ) * cosd(φ)^2, θ = 300)
    G = panels(hyperdiffusion_tendency(model))
    ρ = panels(model.dynamics.density)
    constancy = maximum(maximum(abs, G[r] ./ ρ[r]) for r in eachindex(G)) * τ₄ / 300
    record("hyperdiffusion", resolution.name, "tendency of a uniform θ, relative to θ / τ₄", constancy)
    @test constancy < 1e-12

    # Gate 4 and the dissipation measurement: a state with structure at the grid scale
    # (The state has no symmetry about a meridian or the equator: the integral of the tendency of
    # a state that has one vanishes by symmetry, whatever the seam does. A first version measured
    # the seam budget on the crossing flow, whose θ is odd in longitude, and recorded 1.5e-18.)
    set!(model; θ = (λ, φ, z) -> 300 + 3 * sind(11λ + 25) * cosd(9φ + 10) + 2 * cosd(17φ + 5) * z)
    G = hyperdiffusion_tendency(model)
    budget = total(G) / total(Field(abs(G)))
    record("hyperdiffusion", resolution.name, "seam budget of the hyperdiffusion of Θ over the integral of its magnitude", budget)
    @test isfinite(budget)
    variance_tendency = total(Field(θ * G)) # ∫ θ ∂ₜ(ρθ) dV = Σ m θ ∂ₜθ
    dissipation = total(Field(hyperdiffusion_dissipation(model).θ))
    @test variance_tendency < 0
    @test dissipation > 0
    record("hyperdiffusion", resolution.name, "variance tendency over minus the dissipation (one on a single grid)",
           - variance_tendency / dissipation)
end

@testset "Vertical diffusion on the panels, with hyperdiffusion and the fixer" begin
    resolution = first(resolutions)
    grid = yin_yang_grid(resolution; Nz=6)
    A, B = hybrid_coefficients(6)
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B,
                                   hyperdiffusion=GridScaleHyperdiffusion(FT; damping_time=τ₄),
                                   vertical_diffusion=VerticalDiffusion(boundary_layer=RichardsonNumberBoundaryLayer(FT)))
    model = hydrostatic_model(grid; dynamics, coriolis=HydrostaticSphericalCoriolis(FT; rotation_rate),
                              thermodynamic_constants=reference_model_constants())

    set!(model; pˢ = (λ, φ) -> 1e5 + 500 * cosd(2λ) * cosd(φ)^2,
                θ = (λ, φ, z) -> 300 - 15 * z + 40 * z^2 + 2 * sind(λ) * sind(2φ), # the lowest layer is warmer than the second
                u = (λ, φ, z) -> 20 * (1 + z) * cosd(φ) + 5 * sind(λ),
                v = (λ, φ, z) -> 5 * z * sind(2λ) * cosd(φ))

    callbacks = conservation_fixer_callbacks(model)
    Θ⁰ = total(model.formulation.potential_temperature_density)
    M⁰ = total(continuity_density(model.dynamics))

    time_step!(model, resolution.Δt; callbacks)
    largest_diffusivity = maximum(model.dynamics.vertical_diffusion.diffusivity)

    for _ in 2:10
        time_step!(model, resolution.Δt; callbacks)
    end

    ΔΘ = total(model.formulation.potential_temperature_density) / Θ⁰ - 1
    ΔM = total(continuity_density(model.dynamics)) / M⁰ - 1
    record("vertical diffusion", resolution.name, "relative change of Θ over 10 steps, fixer on", ΔΘ)
    record("vertical diffusion", resolution.name, "largest diffusivity of the first step, m² s⁻¹", largest_diffusivity)

    @test abs(ΔΘ) < 1e-12                                         # gate 5
    @test abs(ΔM) < 1e-12
    @test largest_diffusivity > 1
    @test all(all(isfinite, panel) for panel in panels(model.velocities.u))
    @test all(all(isfinite, panel) for panel in panels(model.formulation.potential_temperature))
    @test isnothing(validate_velocity_conductances(grid, model.dynamics.vertical_diffusion.conductances.momentum))

    # Gate 6: a viscosity with a jump across the seam
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B,
                                   vertical_diffusion=VerticalDiffusion(diffusivity = 10,
                                                                        viscosity = (λ, φ, z) -> great_circle_angle(λ, φ, 0, 45) < 20 ? 100 : 0))
    model = hydrostatic_model(grid; dynamics, thermodynamic_constants=reference_model_constants())
    set!(model; θ = (λ, φ, z) -> 290 + 60 * z, u = (λ, φ, z) -> 10 * z * cosd(φ))

    outcome = try
        time_step!(model, resolution.Δt)
        :completed
    catch err
        err isa DomainError || rethrow()
        :refused
    end

    C = model.dynamics.vertical_diffusion.conductances.momentum
    lowest_halo_value = minimum(minimum(parent(region(C, r))) for r in regions_of(grid))
    record("vertical diffusion", resolution.name, "jump of the viscosity across the seam: lowest conductance in a halo, kg m⁻² s⁻¹", lowest_halo_value)
    @info "Viscosity with a jump across the seam: the step was $outcome"
    @test outcome in (:completed, :refused)
    outcome == :completed && @test isnothing(validate_velocity_conductances(grid, C))
    outcome == :refused && @test lowest_halo_value < 0
end
