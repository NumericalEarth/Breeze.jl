include(joinpath(@__DIR__, "setup.jl"))

#####
##### AtmosphereModel with CompressibleDynamics on a YinYangGrid: the reference column, the reductions
##### behind the reference-state reset, the geographic components taken by `set!`, the paired halo fills
##### of the momentum, and balanced flows crossing the panel boundary.
#####

using Breeze
using Breeze.Thermodynamics: reference_column_grid
using Oceananigans
using Oceananigans: instantiated_location
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Coriolis: TriadScheme
using Oceananigans.Grids: λnode, φnode, halo_size
using Oceananigans.MultiRegion: YinYangGrid
using Oceananigans.OrthogonalSphericalShellGrids: rotated_basis_angle
using Oceananigans.Units
using Oceananigans.Utils: getregion
using Test

# The halo holds the WENO stencil, and `Nφ ≥ 2 (overlap + Hφ + 1)` keeps the panels off their poles
yin_yang_grid(arch, FT) = YinYangGrid(arch, FT; size=(48, 16, 8), z=(0, 10kilometers), halo=(3, 3, 3))

north_pole(panel) = panel.conformal_mapping.north_pole

# The `component` (1 for u, 2 for v) of the geographic vector `flow(λ, φ, z)` in the basis of the panel with `pole`
function panel_component(flow, component, λ, φ, z, pole)
    uᵍ, vᵍ = flow(λ, φ, z)
    ψ = rotated_basis_angle(λ, φ, pole...)
    return (uᵍ * cos(ψ) - vᵍ * sin(ψ), uᵍ * sin(ψ) + vᵍ * cos(ψ))[component]
end

# Test that `field` is the `component` of `flow`, to `rtol` of the largest magnitude over the points
# each panel owns, which are the points the reductions of a Yin-Yang grid consider
function test_panel_components(field, flow, component; rtol)
    exact = Field(instantiated_location(field), field.grid)
    for r in 1:2
        pole = north_pole(getregion(field.grid, r))
        set!(getregion(exact, r), (λ, φ, z) -> panel_component(flow, component, λ, φ, z, pole))
    end
    @test maximum(abs, field - exact) ≤ rtol * maximum(abs, exact)
end

# As `test_panel_components`, for the halo cells beside each edge of the panels at the lowest level,
# where the flow must not depend on `z`; the halo corners are left out
function test_halo_components(field, flow, component; rtol)
    ℓx, ℓy, ℓz = instantiated_location(field)
    Hx, Hy, Hz = halo_size(field.grid)

    for r in 1:2
        panel = on_architecture(CPU(), getregion(field.grid, r))
        data = on_architecture(CPU(), parent(getregion(field, r)))
        Nx, Ny, _ = size(getregion(field, r))
        pole = north_pole(panel)

        for (is, js) in ((1-Hx:0, 1:Ny), (Nx+1:Nx+Hx, 1:Ny), (1:Nx, 1-Hy:0), (1:Nx, Ny+1:Ny+Hy))
            λ = λnode.(is, permutedims(js), 1, Ref(panel), Ref(ℓx), Ref(ℓy), Ref(ℓz))
            φ = φnode.(is, permutedims(js), 1, Ref(panel), Ref(ℓx), Ref(ℓy), Ref(ℓz))
            exact = panel_component.(Ref(flow), component, λ, φ, 0, Ref(pole))
            @test maximum(abs, data[is .+ Hx, js .+ Hy, 1 + Hz] .- exact) ≤ rtol * maximum(abs, exact)
        end
    end
end

#####
##### Solid-body rotation about an axis tilted by α from the pole, with angular speed ω, in cyclostrophic
##### (α ≠ 0, non-rotating) or geostrophic-cyclostrophic (α = 0, rotating) balance with an isothermal
##### atmosphere. The flow has geographic components (Williamson et al. 1992)
#####
#####     u = ω a (cos φ cos α + sin φ cos λ sin α),    v = - ω a sin λ sin α,
#####
##### With sin φ′ = n ⋅ r̂ the latitude from the tilted axis n = (-sin α, 0, cos α), the flow is steady when
##### the pressure balances the centripetal (and, for α = 0, Coriolis) acceleration through the potential
#####
#####     Φ = g z - (ω² / 2 + Ω ω) a² cos² φ′,
#####
##### so that an isothermal atmosphere at T₀ has p = p₀ exp(-Φ / (Rᵈ T₀)): an exact steady solution of the
##### shallow-atmosphere equations with the traditional Coriolis force.
#####

function solid_body_rotation(grid, constants; α = π / 4, rotating = false, U = 40, T₀ = 300, p₀ = 1e5)
    a = getregion(grid, 1).radius
    Ω = rotating ? Oceananigans.defaults.planet_rotation_rate : 0
    Rᵈ = dry_air_gas_constant(constants)
    κ = Rᵈ / constants.dry_air.heat_capacity
    return (; α, ω = U / a, Ω, a, T₀, p₀, g = constants.gravitational_acceleration, Rᵈ, κ)
end

function geopotential(s, λ, φ, z)
    sinφ′ = cos(s.α) * sind(φ) - sin(s.α) * cosd(φ) * cosd(λ)
    return s.g * z - (s.ω^2 / 2 + s.Ω * s.ω) * s.a^2 * (1 - sinφ′^2)
end

pressure(s, λ, φ, z) = s.p₀ * exp(-geopotential(s, λ, φ, z) / (s.Rᵈ * s.T₀))
potential_temperature(s, λ, φ, z) = s.T₀ * (s.p₀ / pressure(s, λ, φ, z))^s.κ
velocity(s, λ, φ, z) = (s.ω * s.a * (cosd(φ) * cos(s.α) + sind(φ) * cosd(λ) * sin(s.α)), - s.ω * s.a * sind(λ) * sin(s.α))

# The balanced state, with the density integrated in discrete hydrostatic balance from the analytic
# surface pressure, so that the only imbalance is the horizontal discretization error
function balanced_flow_model(grid, s; coriolis = nothing)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = s.p₀, reference_temperature = s.T₀)
    model = AtmosphereModel(grid; dynamics, coriolis, advection = WENO(order=5))

    pˢ = Field{Center, Center, Nothing}(grid)
    set!(pˢ, (λ, φ) -> pressure(s, λ, φ, 0))
    fill_halo_regions!(pˢ)

    set!(model; θ = (λ, φ, z) -> potential_temperature(s, λ, φ, z),
                u = (λ, φ, z) -> velocity(s, λ, φ, z)[1],
                v = (λ, φ, z) -> velocity(s, λ, φ, z)[2],
                ρ = HydrostaticallyBalancedDensity(surface_pressure = pˢ))

    return model
end

@testset "Yin-Yang AtmosphereModel [$(FT)]" for FT in test_float_types()
    previous_float_type = Oceananigans.defaults.FloatType
    Oceananigans.defaults.FloatType = FT

    try
        grid = yin_yang_grid(default_arch, FT)
        constants = ThermodynamicConstants(FT)
        rounding = sqrt(eps(FT))
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5, reference_potential_temperature = 300)

        @testset "Construction and reference column" begin
            column_grid = reference_column_grid(grid)
            @test size(column_grid) == (1, 1, size(grid, 3))
            @test znodes(column_grid, Center()) == znodes(getregion(grid, 1), Center())

            model = AtmosphereModel(grid; dynamics, advection = WENO(order=5), forcing = held_suarez_forcings())
            @test size(model.dynamics.reference_state.pressure) == (1, 1, size(grid, 3))

            @test_throws ArgumentError AtmosphereModel(ImmersedBoundaryGrid(grid, GridFittedBottom((λ, φ) -> -1)))
            @test_throws ArgumentError AtmosphereModel(grid; dynamics = AnelasticDynamics(ReferenceState(grid)))
            @test_throws ArgumentError AtmosphereModel(grid; dynamics = PrescribedDynamics(ReferenceState(grid)))
        end

        @testset "Reference state from owned-cell means" begin
            # The same mean profile with and without a zero-mean perturbation antisymmetric about the equator
            θ̄ᵢ(z) = 300 + 3e-3 * z

            function reset_exner_function(θᵢ)
                model = AtmosphereModel(grid; dynamics, advection = WENO(order=5))
                set!(model; θ = θᵢ, ρ = 1, compute_reference_state = true)
                return model.dynamics.reference_state.exner_function
            end

            uniform = reset_exner_function((λ, φ, z) -> θ̄ᵢ(z))
            perturbed = reset_exner_function((λ, φ, z) -> θ̄ᵢ(z) + 5 * sind(φ) * cosd(φ)^2 * cosd(λ))
            @test interior(perturbed) ≈ interior(uniform) rtol=rounding

            # The reset reference is in hydrostatic balance with the mean profile, ∂z π = - g / (cᵖᵈ θ̄),
            # from the surface Exner function 1 of the base pressure, to the vertical discretization error
            g = constants.gravitational_acceleration
            cᵖᵈ = constants.dry_air.heat_capacity
            exner_function = Field{Nothing, Nothing, Center}(reference_column_grid(grid))
            set!(exner_function, z -> 1 - g / (cᵖᵈ * 3e-3) * log(θ̄ᵢ(z) / 300))
            @test interior(perturbed) ≈ interior(exner_function) rtol=1e-3
        end

        @testset "HydrostaticallyBalancedDensity anchors at the reference column" begin
            g = constants.gravitational_acceleration
            cᵖᵈ = constants.dry_air.heat_capacity
            θᵣ(z) = 250 * exp(g * z / (cᵖᵈ * 250))
            θᵢ(λ, φ, z) = (280 - 30 * sind(φ)^2) * exp(g * z / (cᵖᵈ * 300))

            function balanced_density(ρ)
                dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5, reference_potential_temperature = θᵣ)
                model = AtmosphereModel(grid; dynamics, thermodynamic_constants = constants, advection = WENO(order=5))
                set!(model; θ = θᵢ, ρ)
                return model.dynamics.dry_density
            end

            # The default anchor is the pressure of the reference column at z = 0, which is `base_pressure`
            default = balanced_density(HydrostaticallyBalancedDensity())
            explicit = balanced_density(HydrostaticallyBalancedDensity(surface_pressure = 1e5))
            @test maximum(abs, default - explicit) ≤ rounding * maximum(abs, explicit)
        end

        @testset "Geographic set! and paired halo fills" begin
            model = AtmosphereModel(grid; dynamics, advection = WENO(order=5))
            s = solid_body_rotation(grid, constants)
            flow(λ, φ, z) = velocity(s, λ, φ, z)
            set!(model; θ = 300, ρ = 1, u = (λ, φ, z) -> flow(λ, φ, z)[1], v = (λ, φ, z) -> flow(λ, φ, z)[2])
            test_panel_components(model.velocities.u, flow, 1; rtol=rounding)
            test_panel_components(model.velocities.v, flow, 2; rtol=rounding)

            # The halos of the momentum pair are filled from the other panel with the rotation: they agree
            # with the rotated analytic flow to the interpolation error
            test_halo_components(model.momentum.ρu, flow, 1; rtol=1e-4)
            test_halo_components(model.momentum.ρv, flow, 2; rtol=1e-4)

            # Momentum is set from geographic components too, and panel components are taken as they are
            flow²(λ, φ, z) = 2 .* flow(λ, φ, z)
            set!(model; ρu = (λ, φ, z) -> flow²(λ, φ, z)[1], ρv = (λ, φ, z) -> flow²(λ, φ, z)[2])
            test_panel_components(model.momentum.ρu, flow², 1; rtol=rounding)
            test_panel_components(model.momentum.ρv, flow², 2; rtol=rounding)

            set!(model; u = 1, v = 0, intrinsic_velocities = true)
            @test all(x -> x ≈ 1, interior(getregion(model.velocities.u, 2)))
        end

        # The flows stay balanced to a fraction of a percent after a few hours. The triads weight the Coriolis
        # parameter isotropically, so the geostrophic balance holds on the rotated Yang panel too.
        @testset "Balanced solid-body rotation [$(name)]" for (name, s, coriolis, Nsteps, rtol) in
                (("tilted axis", solid_body_rotation(grid, constants; α = π / 4), nothing, 18, 1e-3),
                 ("rotating zonal flow with the triad Coriolis scheme", solid_body_rotation(grid, constants; α = 0, rotating = true),
                  HydrostaticSphericalCoriolis(scheme = TriadScheme()), 12, 5e-3))

            model = balanced_flow_model(grid, s; coriolis)
            for _ in 1:Nsteps
                time_step!(model, 600)
            end

            flow(λ, φ, z) = velocity(s, λ, φ, z)
            test_panel_components(model.velocities.u, flow, 1; rtol)
            test_panel_components(model.velocities.v, flow, 2; rtol)
        end
    finally
        Oceananigans.defaults.FloatType = previous_float_type
    end
end
