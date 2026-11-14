include(joinpath(@__DIR__, "setup.jl"))

#####
##### The energy budget of TKEBasedTurbulenceClosure with the cloud buoyancy flux: the TKE equation
##### receives exactly the energy the closure's fluxes take from the mean state, and the energy it
##### dissipates heats the air. Checked step by step on a moist column with P3, ClosureCloud and
##### the subgrid distribution, with and without rain, from conversions computed here independently
##### of the closure (the derivatives of the buoyancy by central differences of the model's
##### temperature and equation of state).
#####
##### With u at centers and the flux ρᶠ K ∂z u at the faces, summation by parts gives
#####   -d/dt Σ ρᶜ Δzᶜ |u|²/2 = Σᶠ ρᶠ Δzᶠ Kᵘ |Δu / Δzᶠ|²                     (+ boundary terms)
#####   -d/dt Σ ρᶜ Δzᶜ h‡     = Σᶠ ρᶠ Δzᶠ (-Kᶜ) Σₛ β̄ₛ Δφₛ / Δzᶠ             (+ boundary terms)
##### with the dynamic enthalpy h‡ = -∫ b dz on the trapezoid rule, βₛ = ∂b/∂φₛ; and the heating
##### of the dissipation ε raises the enthalpy Σ ρ Δz cᵖᵐ T by Σ ρ Δz ε. The TKE equation must
##### receive Σ ρ Δz (P + B) and lose Σ ρ Δz ε with these.
#####

using Breeze
using Breeze.AtmosphereModels: ClosureCloud
using Breeze.Thermodynamics: MoistureMassFractions, temperature_from_potential_temperature,
                             mixture_gas_constant, mixture_heat_capacity
using Breeze.TurbulenceClosures: add_dissipative_heating!

using Oceananigans
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.Fields: ZeroField
using Oceananigans.Grids: znodes
using Test

budget_column(f) = Array(interior(f))[1, 1, :]
budget_column(f::ZeroField, Nz) = zeros(Nz)
budget_column(f, Nz) = budget_column(f)

function moist_tke_column(; rain = false, Nz = 40, H = 2400)
    grid = RectilinearGrid(CPU(); size=Nz, z=(0, H), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=299)
    microphysics = PredictedParticlePropertiesMicrophysics(; subgrid = ClosureCloud(),
                                                           negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    closure = TKEBasedTurbulenceClosure(; static_stability = DryStaticStability(), minimum_tke = 4e-4,
                                        subgrid_cloud_formulation = SubgridCloudFormulation())
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), microphysics, closure)

    # BOMEX-like: a mixed layer, a conditionally unstable cloud layer moistened to saturation, an inversion
    θ₀(z) = z < 520 ? 298.7 : z < 1480 ? 298.7 + (z - 520) * 3.7 / 960 : 302.4 + (z - 1480) * 5.8 / 520
    qᵗ₀(z) = 1e-3 * (z < 520 ? 17.0 - z * 0.7 / 520 : z < 1480 ? 16.3 - (z - 520) * 5.6 / 960 : max(4.2, 10.7 - (z - 1480) * 6.5 / 520))
    q₀(z) = qᵗ₀(z) + (500 < z < 1100 ? 2.5e-3 : 0.0)
    set!(model; θ=θ₀, qᵗ=q₀, u = z -> z < 700 ? -8.75 : -8.75 + (z - 700) * 1.8e-3, ρe = z -> 0.6 * exp(-z / 500))
    if rain
        set!(model.microphysical_fields.ρqʳ, z -> 300 < z < 1200 ? 2.2e-4 : 0.0)
        set!(model.microphysical_fields.ρnʳ, z -> 300 < z < 1200 ? 2.2e4 : 0.0)
    end
    return model
end

function closure_energy_conversions(model)
    grid = model.grid
    Nz = size(grid, 3)
    constants = model.thermodynamic_constants
    reference_state = model.dynamics.reference_state
    g = constants.gravitational_acceleration
    pˢᵗ = reference_state.standard_pressure
    ρ = budget_column(reference_state.density)
    p = budget_column(reference_state.pressure)
    Tᵣ = budget_column(reference_state.temperature)
    qᵣ = [MoistureMassFractions(budget_column(reference_state.vapor_mass_fraction, Nz)[k],
                                budget_column(reference_state.liquid_mass_fraction, Nz)[k],
                                budget_column(reference_state.ice_mass_fraction, Nz)[k]) for k in 1:Nz]
    zᶜ = znodes(grid, Center()); zᶠ = znodes(grid, Face())
    Δzᶜ = diff(zᶠ); Δzᶠ = diff(zᶜ)
    ρᶠ = [(ρ[k] + ρ[k-1]) / 2 for k in 2:Nz]

    buoyancy(θ, qᵛ, qˡ, qⁱ, k) = (q = MoistureMassFractions(qᵛ, qˡ, qⁱ);
        T = temperature_from_potential_temperature(θ, p[k], pˢᵗ, constants, q);
        g * (1 - mixture_gas_constant(qᵣ[k], constants) * Tᵣ[k] / (mixture_gas_constant(q, constants) * T)))

    cf = model.closure_fields
    cl = cf.subgrid_cloud.cloud
    μ = model.microphysical_fields
    θ = budget_column(model.formulation.potential_temperature_density) ./ ρ
    qᵛ = budget_column(model.moisture_density) ./ ρ
    qˡ = (budget_column(μ.ρqᶜˡ) .+ budget_column(μ.ρqʳ) .+ budget_column(μ.ρqʷⁱ)) ./ ρ
    qⁱ = budget_column(μ.ρqⁱ) ./ ρ
    u = budget_column(model.velocities.u); v = budget_column(model.velocities.v)
    Kᶜ = budget_column(cf.Kᶜ); Kᵘ = budget_column(cf.Kᵘ)
    e = budget_column(model.tracers.ρe) ./ ρ

    φ = hcat(θ, qᵛ, qˡ, qⁱ)
    h = (1e-4, 1e-7, 1e-7, 1e-7)
    β = zeros(Nz, 4)
    for k in 1:Nz, s in 1:4
        x⁺ = collect(φ[k, :]); x⁻ = collect(φ[k, :])
        x⁺[s] += h[s]; x⁻[s] -= h[s]
        β[k, s] = (buoyancy(x⁺..., k) - buoyancy(x⁻..., k)) / 2h[s]
    end
    Bᶠ = [-Kᶜ[f] * sum((β[f, s] + β[f-1, s]) / 2 * (φ[f, s] - φ[f-1, s]) for s in 1:4) / Δzᶠ[f-1] for f in 2:Nz]
    Pᶠ = [Kᵘ[f] * ((u[f] - u[f-1])^2 + (v[f] - v[f-1])^2) / Δzᶠ[f-1]^2 for f in 2:Nz]

    # What the TKE equation receives: P and B⁺ explicitly, the buoyancy sink ωᴮ e and the
    # dissipation implicitly (Lᵉ = -(ω + ωᴮ)), at the state these were formed from
    B = budget_column(cl.B)
    floor = sqrt(floatmin(Float64))
    ωᴮ = -min.(0, B) ./ max.(e, floor)
    ε = (-budget_column(cf.Lᵉ) .- ωᴮ) .* e

    G = (; ρθ = CenterField(grid))
    add_dissipative_heating!(G, cf.subgrid_cloud, cf, model.closure, model)
    ∂H∂θ = map(1:Nz) do k
        q = MoistureMassFractions(qᵛ[k], qˡ[k], qⁱ[k])
        T(θ) = temperature_from_potential_temperature(θ, p[k], pˢᵗ, constants, q)
        mixture_heat_capacity(q, constants) * (T(θ[k] + 1e-4) - T(θ[k] - 1e-4)) / 2e-4
    end

    return (mean_kinetic_energy_loss = sum(ρᶠ .* Δzᶠ .* Pᶠ),
            shear_production = sum(ρ .* Δzᶜ .* budget_column(cl.P)),
            buoyancy_conversion = sum(ρᶠ .* Δzᶠ .* Bᶠ),
            buoyancy_production = sum(ρ .* Δzᶜ .* (max.(B, 0) .- ωᴮ .* e)),
            dissipation = sum(ρ .* Δzᶜ .* ε),
            heating = sum(Δzᶜ .* budget_column(G.ρθ) .* ∂H∂θ))
end

@testset "TKE energy budget with the cloud buoyancy flux (rain = $rain)" for rain in (false, true)
    model = moist_tke_column(; rain)
    Oceananigans.TimeSteppers.update_state!(model)
    for n in 1:15
        time_step!(model, 10)
        c = closure_energy_conversions(model)
        scale = c.mean_kinetic_energy_loss + abs(c.buoyancy_conversion) + c.dissipation
        @info "step $n" c.mean_kinetic_energy_loss c.shear_production c.buoyancy_conversion c.buoyancy_production c.dissipation c.heating
        # Shear production is the mean kinetic energy the momentum fluxes remove (round-off)
        @test c.shear_production ≈ c.mean_kinetic_energy_loss rtol=1e-12
        # Buoyancy production is the dynamic enthalpy the scalar fluxes remove, to the accuracy
        # of the central differences of the buoyancy here (≈1e-9)
        @test c.buoyancy_production ≈ c.buoyancy_conversion rtol=1e-7 atol=1e-7 * scale
        # The dissipated energy heats the air
        @test c.heating ≈ c.dissipation rtol=1e-8
        total = (c.shear_production - c.mean_kinetic_energy_loss) +
                (c.buoyancy_production - c.buoyancy_conversion) +
                (c.heating - c.dissipation)
        @test abs(total) < 1e-7 * scale
    end
    @test maximum(budget_column(model.closure_fields.subgrid_cloud.cloud.qˡ)) > 0
end
