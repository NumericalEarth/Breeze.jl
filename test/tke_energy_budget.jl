include(joinpath(@__DIR__, "setup.jl"))

#####
##### The energy budget of TKEBasedTurbulenceClosure with the cloud buoyancy flux: the TKE equation
##### receives exactly the energy the closure's fluxes take from the mean state, and the energy it
##### dissipates heats the air. Checked step by step on a moist column with P3 (its own
##### condensation; no closure-owned cloud) and the subgrid distribution, with and without rain and surface fluxes, from conversions computed here independently
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
using Breeze.Thermodynamics: MoistureMassFractions, temperature_from_potential_temperature,
                             mixture_gas_constant, mixture_heat_capacity

using Oceananigans
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.Fields: ZeroField
using Oceananigans.Grids: znodes
using Test

budget_column(f) = Array(interior(f))[1, 1, :]
budget_column(f::ZeroField, Nz) = zeros(Nz)
budget_column(f, Nz) = budget_column(f)

function moist_tke_column(; rain = false, surface_fluxes = false, Nz = 40, H = 2400)
    grid = RectilinearGrid(CPU(); size=Nz, z=(0, H), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=299)
    microphysics = PredictedParticlePropertiesMicrophysics(;
                                                           negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    closure = TKEBasedTurbulenceClosure(; static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()), minimum_tke = 4e-4,
                                        subgrid_distribution = DoubleGaussian())
    boundary_conditions = surface_fluxes ?
        (ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 8e-3)),
         ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 5.2e-5)),
         ρu = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 0.08)),
         ρv = FieldBoundaryConditions(bottom = FluxBoundaryCondition(-1.17 * 0.02))) : NamedTuple()
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), microphysics, closure, boundary_conditions)

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

    return (mean_kinetic_energy_loss = sum(ρᶠ .* Δzᶠ .* Pᶠ),
            shear_production = sum(ρ .* Δzᶜ .* budget_column(cl.P)),
            buoyancy_conversion = sum(ρᶠ .* Δzᶠ .* Bᶠ),
            buoyancy_production = sum(ρ .* Δzᶜ .* (B .- budget_column(cl.Bᴸ))),
            latent_production = sum(ρ .* Δzᶜ .* budget_column(cl.Bᴸ)),
            latent_conversion = sum(ρᶠ .* Δzᶠ .* Kᶜ[2:Nz] .* budget_column(cl.ΔN²ᴸ)[2:Nz]),
            dissipation = sum(ρ .* Δzᶜ .* ε),
            β = β, stored_β = hcat(budget_column(cl.b_θ), budget_column(cl.b_qᵛ), budget_column(cl.b_qˡ), budget_column(cl.b_qⁱ)))
end

# The rates at the state of an update: the stage conversions equal the energy the fluxes take
@testset "TKE conversions of the applied fluxes (rain = $rain)" for rain in (false, true)
    model = moist_tke_column(; rain)
    Oceananigans.TimeSteppers.update_state!(model)
    for n in 1:5
        time_step!(model, 10)
        Oceananigans.TimeSteppers.update_state!(model)
        c = closure_energy_conversions(model)
        scale = c.mean_kinetic_energy_loss + abs(c.buoyancy_conversion) + c.dissipation
        @test c.shear_production ≈ c.mean_kinetic_energy_loss rtol=1e-12
        # to the accuracy of the central differences of the buoyancy here (≈1e-9)
        @test c.buoyancy_production ≈ c.buoyancy_conversion rtol=1e-7 atol=1e-7 * scale
        @test c.stored_β ≈ c.β rtol=1e-6
        # the latent part: Kᶜ aᴺ (N²ᵇ - N²ˢ) at the faces, drawn from the internal energy
        @test c.latent_production ≈ c.latent_conversion rtol=1e-12 atol=1e-14 * scale
    end
end

# The last Runge–Kutta stage of every step: across the vertically implicit solves the mean state
# loses exactly the energy the TKE gains, and the dissipation the TKE loses heats the air
function stage_energy_budget(model, Δt)
    grid = model.grid
    Nz = size(grid, 3)
    cf = model.closure_fields
    cl = cf.subgrid_cloud.cloud
    pre = cf.subgrid_cloud.presolve
    μ = model.microphysical_fields
    ρ = budget_column(model.dynamics.reference_state.density)
    zᶜ = znodes(grid, Center()); zᶠ = znodes(grid, Face())
    Δzᶜ = diff(zᶠ); Δzᶠ = diff(zᶜ)

    # mean kinetic energy across the momentum solves
    KE(ρu, ρv) = sum(Δzᶜ .* (ρu .^ 2 .+ ρv .^ 2) ./ (2ρ))
    ΔKE = KE(budget_column(model.momentum.ρu), budget_column(model.momentum.ρv)) -
          KE(budget_column(cl.ρu⁻), budget_column(cl.ρv⁻))

    # dynamic enthalpy, linearized with the stage's derivatives (trapezoid rule for Z = ∫ β dz)
    β = hcat(budget_column(cl.stage_b_θ), budget_column(cl.stage_b_qᵛ), budget_column(cl.stage_b_qˡ), budget_column(cl.stage_b_qⁱ))
    Z = zeros(Nz, 4)
    for s in 1:4
        Z[1, s] = β[1, s] * (zᶜ[1] - zᶠ[1])
        for k in 2:Nz
            Z[k, s] = Z[k-1, s] + (β[k, s] + β[k-1, s]) / 2 * Δzᶠ[k-1]
        end
    end
    ρφ(fields) = hcat(budget_column(fields.ρθ), budget_column(fields.ρqᵛ),
                      budget_column(fields.ρqᶜˡ) .+ budget_column(fields.ρqʳ) .+ budget_column(fields.ρqʷⁱ),
                      budget_column(fields.ρqⁱ))
    after = ρφ(merge(Oceananigans.prognostic_fields(model), (; ρθ = model.formulation.potential_temperature_density)))
    after[:, 1] .+= budget_column(cl.shortfall_ρθ) # the solves' change, without the cooling after them
    before = ρφ(pre)
    Δhᵈ = -sum(Δzᶜ .* sum(Z .* (after .- before), dims=2))

    # the TKE: explicit stage sources were added before its solve; the solve and the correction after
    Δρe = sum(Δzᶜ .* (budget_column(model.tracers.ρe) .- budget_column(pre.ρe)))
    given_explicitly = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.given))
    converted = Δt * sum(ρ .* Δzᶜ .* (budget_column(cl.Pᵖ) .+ budget_column(cl.Bᵖ) .+ budget_column(cl.stage_surface)))
    dissipated = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.ε))
    shortfall = sum(Δzᶜ .* budget_column(cl.shortfall)) # paid by the internal energy instead of the TKE
    return (; ΔKE, Δhᵈ, Δρe, given_explicitly, converted, dissipated, shortfall,
            shear = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.Pᵖ)), buoyancy = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.Bᵖ)))
end

@testset "TKE energy budget across the implicit solves (rain = $rain, surface fluxes = $surface_fluxes)" for rain in (false, true), surface_fluxes in (false, true)
    model = moist_tke_column(; rain, surface_fluxes)
    Δt = 10
    Δtₛ = model.timestepper.α³ * Δt # the last stage
    for n in 1:15
        time_step!(model, Δt)
        b = stage_energy_budget(model, Δtₛ)
        @info "step $n" b.ΔKE b.shear b.Δhᵈ b.buoyancy b.Δρe b.converted b.dissipated
        scale = abs(b.ΔKE) + abs(b.Δhᵈ) + abs(b.dissipated)
        # mean kinetic energy lost by the momentum solves = shear production (with backward Euler's own dissipation)
        @test -b.ΔKE ≈ b.shear rtol=1e-8 atol=1e-12 * scale # a difference of sums of ρ|u|²: round-off ≈1e-10
        # dynamic enthalpy lost by the scalar solves = buoyancy production (Σ Z Δ(ρθ) is a
        # difference of sums of O(Z ρθ) terms, so round-off here is ≈1e-8 relative)
        # With rain, the solve of ρqʳ also sediments it (implicit advection by its fall speed); the
        # loading change of that sedimentation (≈1e-6 of the conversion) is microphysics, not mixing
        @test -b.Δhᵈ ≈ b.buoyancy rtol=(rain ? 1e-5 : 1e-7) atol=1e-11 * scale
        # the TKE receives what the solves converted, less what it dissipates (which heats the air)
        @test b.Δρe + b.given_explicitly ≈ b.converted - b.dissipated + b.shortfall rtol=1e-9 atol=1e-11 * scale
    end
end
