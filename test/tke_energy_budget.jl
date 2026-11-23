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

# Without a distribution: a saturated fraction from outside the closure (as a microphysics subgrid
# model supplies it), which implies a condensate flux and its latent conversion
struct HalfSaturatedFraction end
@inline Breeze.TurbulenceClosures.saturated_stability_weightᶜᶜᶠ(i, j, k, grid, ::HalfSaturatedFraction, buoyancy, tracers, closure_fields) =
    one(eltype(grid)) / 2

shoc_components(c) = (mixing_length = c.mixing_length, stability_functions = c.stability_functions,
                      static_stability = c.static_stability, subgrid_distribution = c.subgrid_distribution)

# The two paths: the DoubleGaussian distribution, and none with the microphysics-like fraction
closure_paths = (distribution = (static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()),
                                 subgrid_distribution = DoubleGaussian()),
                 no_distribution = (static_stability = SaturationWeightedStaticStability(HalfSaturatedFraction()),
                                    subgrid_distribution = nothing),
                 # the third moment from the TKE budget, whose flux also transports TKE (ℂʷᵀ > 0)
                 budget_third_moment = (static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()),
                                        subgrid_distribution = DoubleGaussian(third_moment = TKEBudgetThirdMoment(ℂʷᴸ = 8, ℂʷᵀ = 0.3),
                                                                              skewness = TanhSaturatedSkewness())),
                 budget_third_moment_one_amplitude = (static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()),
                                                      subgrid_distribution = DoubleGaussian(third_moment = TKEBudgetThirdMoment(ℂʷᴸ = Inf, ℂʷᵀ = 0.3))),
                 # SimplifiedHigherOrderClosure: its stability functions give the TKE the share ℂᴾ = 0.1 of the shear
                 # production and heat the air by the rest
                 shoc = shoc_components(SimplifiedHigherOrderClosure()),
                 # and with ℂᴾ = 1, all of it
                 shoc_whole_shear = shoc_components(SimplifiedHigherOrderClosure(stability_functions = ColumnStabilityDamping(; ℂᴾ = 1))))
# A cooling of 5 K h⁻¹ over the top 100 m of the moist layer of `moist_tke_column` and a warming of 1 K h⁻¹ below
# 300 m, as the prescribed heating of a `HeatingBudgetFlux` (the tendency of θˡⁱ, K s⁻¹)
function cloud_top_cooling(grid)
    tendency = CenterField(grid)
    set!(tendency, z -> 1000 < z < 1100 ? -5 / 3600 : z < 300 ? 1 / 3600 : 0)
    return tendency
end

# The paths with the heating budget flux: built on the grid, which holds the prescribed heating. With the
# distribution and the budget third moment (whose local member the cooling signs), and without a distribution
heating_paths = (heating_budget_flux = grid -> (static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()),
                                               subgrid_distribution = DoubleGaussian(third_moment = TKEBudgetThirdMoment(ℂʷᵇ = 0.6, ℂʷᵀ = 0.3)),
                                               heating_budget_flux = HeatingBudgetFlux(ℂʷᑫ = 0.6, heating = PrescribedHeating(cloud_top_cooling(grid)))),
                 heating_budget_flux_no_distribution = grid -> (static_stability = SaturationWeightedStaticStability(HalfSaturatedFraction()),
                                                               subgrid_distribution = nothing,
                                                               heating_budget_flux = HeatingBudgetFlux(ℂᑫ = 0.7, ℂᑫᴸ = 4, heating = PrescribedHeating(cloud_top_cooling(grid)))))
all_paths = (keys(closure_paths)..., keys(heating_paths)...)

# The budgets of the whole step take the TKE to receive the whole shear production, which with ℂᴾ < 1 it receives
# in part (the rest heats the air in the stage tendency: tested below at the level of the tendencies)
whole_shear_paths = filter(!=(:shoc), all_paths)
path_keywords(path, grid) = haskey(closure_paths, path) ? closure_paths[path] : heating_paths[path](grid)

# The fields of the heating budget flux, zero without one
heating_column(cl, name, n) = hasproperty(cl, name) ? budget_column(getproperty(cl, name)) : zeros(n)

budget_column(f::ZeroField, Nz) = zeros(Nz)
budget_column(f, Nz) = budget_column(f)

function moist_tke_column(; rain = false, surface_fluxes = false, Nz = 40, H = 2400, path = :distribution, FT = Float64,
                          e₀ = z -> 0.6 * exp(-z / 500), timestepper = :SSPRungeKutta3, timestepper_kwargs = NamedTuple(),
                          closure_keywords = NamedTuple())
    grid = RectilinearGrid(CPU(), FT; size=Nz, z=(0, H), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=299)
    microphysics = PredictedParticlePropertiesMicrophysics(FT; thermodynamic_constants = constants,
                                                           negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    closure = TKEBasedTurbulenceClosure(FT; minimum_tke = 4e-4, path_keywords(path, grid)..., closure_keywords...)
    boundary_conditions = surface_fluxes ?
        (ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 8e-3)),
         ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 5.2e-5)),
         ρu = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 0.08)),
         ρv = FieldBoundaryConditions(bottom = FluxBoundaryCondition(-1.17 * 0.02))) : NamedTuple()
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), thermodynamic_constants=constants, microphysics, closure, boundary_conditions, timestepper,
                            timestepper_kwargs)

    # BOMEX-like: a mixed layer, a conditionally unstable cloud layer moistened to saturation, an inversion
    θ₀(z) = z < 520 ? 298.7 : z < 1480 ? 298.7 + (z - 520) * 3.7 / 960 : 302.4 + (z - 1480) * 5.8 / 520
    qᵗ₀(z) = 1e-3 * (z < 520 ? 17.0 - z * 0.7 / 520 : z < 1480 ? 16.3 - (z - 520) * 5.6 / 960 : max(4.2, 10.7 - (z - 1480) * 6.5 / 520))
    q₀(z) = qᵗ₀(z) + (500 < z < 1100 ? 2.5e-3 : 0.0)
        set!(model; θ=θ₀, qᵗ=q₀, u = z -> z < 700 ? -8.75 : -8.75 + (z - 700) * 1.8e-3, ρe = e₀)
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
    # the fluxes down the gradient and the heating budget flux of θˡⁱ (zero without one)
    Jᑫ = heating_column(cl, :Jᑫ, Nz + 1)
    Bᶠ = [-Kᶜ[f] * sum((β[f, s] + β[f-1, s]) / 2 * (φ[f, s] - φ[f-1, s]) for s in 1:4) / Δzᶠ[f-1] +
          (β[f, 1] + β[f-1, 1]) / 2 * Jᑫ[f] for f in 2:Nz]
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
            buoyancy_production = sum(ρ .* Δzᶜ .* (B .- budget_column(cl.Bᴸ) .- heating_column(cl, :Gᑫᴸ, Nz))),
            latent_production = sum(ρ .* Δzᶜ .* budget_column(cl.Bᴸ)),
            latent_conversion = sum(ρᶠ .* Δzᶠ .* Kᶜ[2:Nz] .* budget_column(cl.ΔN²ᴸ)[2:Nz]),
            dissipation = sum(ρ .* Δzᶜ .* ε),
            β = β, stored_β = hcat(budget_column(cl.b_θ), budget_column(cl.b_qᵛ), budget_column(cl.b_qˡ), budget_column(cl.b_qⁱ)))
end

# The rates at the state of an update: the stage conversions equal the energy the fluxes take
@testset "TKE conversions of the applied fluxes (rain = $rain, $path)" for rain in (false, true), path in all_paths
    model = moist_tke_column(; rain, path)
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
# `ρe` is the column of the TKE density (within `update_state!` the tracer holds its specific value)
function stage_energy_budget(model, Δt; ρe = budget_column(model.tracers.ρe))
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
    # The rounding of Δhᵈ: the stored densities after the solves carry a relative rounding of ϵ, which the sum
    # weights by Z. Where the conversions of θˡⁱ, vapor and condensate nearly cancel, Δhᵈ is small against the
    # terms it is the sum of and this rounding, not the scheme, sets how well it matches the conversion.
    Δhᵈ_rounding = eps(eltype(Z)) * sum(Δzᶜ .* sum(abs.(Z .* after) .+ abs.(Z .* before), dims=2))

    # the TKE: explicit stage sources were added before its solve; the solve and the correction after
    Δρe = sum(Δzᶜ .* (ρe .- budget_column(pre.ρe)))
    given_explicitly = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.given))
    converted = Δt * sum(ρ .* Δzᶜ .* (budget_column(cl.Pᵖ) .+ budget_column(cl.Bᵖ) .+ budget_column(cl.stage_surface)))
    dissipated = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.ε))
    shortfall = sum(Δzᶜ .* budget_column(cl.shortfall)) # paid by the internal energy instead of the TKE
    combination = sum(Δzᶜ .* budget_column(cl.stage_combination)) # the kinetic energy the stage combination removed
    # what the solves alone converted, and the rates of the stage state: given to the TKE by its
    # tendency, and converted by the surface fluxes and the latent heat
    solved = Δt * sum(ρ .* Δzᶜ .* (budget_column(cl.Pᵖ) .+ budget_column(cl.Bᵖ)))
    given_rate = sum(ρ .* Δzᶜ .* budget_column(cl.given))
    stage_state_rate = sum(ρ .* Δzᶜ .* budget_column(cl.stage_surface))
    return (; ΔKE, Δhᵈ, Δhᵈ_rounding, Δρe, given_explicitly, converted, dissipated, shortfall, combination, solved, given_rate, stage_state_rate,
            shear = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.Pᵖ)), buoyancy = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.Bᵖ)))
end

@testset "TKE energy budget across the implicit solves (rain = $rain, surface fluxes = $surface_fluxes, $path)" for rain in (false, true), surface_fluxes in (false, true), path in all_paths
    model = moist_tke_column(; rain, surface_fluxes, path)
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
        # the TKE receives what the solves converted and what the stage combination removed from the
        # mean flow, less what it dissipates (which heats the air)
        @test b.Δρe + b.given_explicitly ≈ b.converted - b.dissipated + b.shortfall + b.combination rtol=1e-9 atol=1e-11 * scale
    end
end

#####
##### The whole time step. A step is three stages, u⁽ᵐ⁾ = S[(1 - α) u⁰ + α (u⁽ᵐ⁻¹⁾ + Δt G)], α = 1, 1/4, 2/3, with S
##### the vertically implicit solves, so what a stage converts enters the step with the weights w = (1/6, 2/3, 1),
##### and the convex combination itself removes the kinetic energy D = α (1 - α) Σ ρ Δz |u⁽ᵐ⁻¹⁾ + Δt G - u⁰|² / 2
##### from the mean flow. With D given to the TKE, over the step
#####
#####   -[KE(uⁿ⁺¹) - KE(uⁿ)]     = Σₘ wₘ (shearₘ + Dₘ)
#####   Σ ρe Δz |ⁿ⁺¹ - Σ ρe Δz |ⁿ = Σₘ wₘ (convertedₘ - dissipatedₘ + shortfallₘ + Dₘ)
#####
##### so the shear production the TKE receives over the step is exactly the kinetic energy the mean flow loses.
##### Without surface fluxes the momentum has no explicit tendency, so u⁽ᵐ⁻¹⁾ + Δt G = u⁽ᵐ⁻¹⁾ and D is
##### recomputed here from the stage states, independently of the closure.
#####

# Records the stage's energy budget within the `update_state!` that follows each stage
struct StageRecorder{F}
    callsite :: Oceananigans.UpdateStateCallsite
    record :: F
end

StageRecorder(record) = StageRecorder(Oceananigans.UpdateStateCallsite(), record)
(recorder::StageRecorder)(model) = recorder.record(model)

@testset "TKE energy budget of the whole step ($path)" for path in whole_shear_paths
    model = moist_tke_column(; path)
    Δt = 10
    α = (model.timestepper.α¹, model.timestepper.α², model.timestepper.α³)
    w = (α[2] * α[3], α[3], 1)
    ρ = budget_column(model.dynamics.reference_state.density)
    Δzᶜ = diff(znodes(model.grid, Face()))
    KE(ρu, ρv) = sum(Δzᶜ .* (ρu .^ 2 .+ ρv .^ 2) ./ (2ρ))
    combination_deficit(ρu⁰, ρv⁰, ρu, ρv, α) = α * (1 - α) * sum(Δzᶜ .* ((ρu .- ρu⁰) .^ 2 .+ (ρv .- ρv⁰) .^ 2) ./ (2ρ))
    time_step!(model, Δt) # the first step's preparation runs `update_state!` once more
    for n in 1:10
        ρu⁰ = budget_column(model.momentum.ρu); ρv⁰ = budget_column(model.momentum.ρv)
        KE⁰ = KE(ρu⁰, ρv⁰)
        TKE⁰ = sum(Δzᶜ .* budget_column(model.tracers.ρe))
        stages = []
        recorder = StageRecorder(model -> begin
            m = length(stages) + 1
            # within `update_state!` the tracers hold their specific values
            budget = stage_energy_budget(model, α[m] * Δt; ρe = budget_column(model.tracers.ρe) .* ρ)
            push!(stages, merge(budget, (ρu = budget_column(model.momentum.ρu), ρv = budget_column(model.momentum.ρv))))
        end)
        time_step!(model, Δt; callbacks = [recorder])
        @test length(stages) == 3

        ΔKE = KE(budget_column(model.momentum.ρu), budget_column(model.momentum.ρv)) - KE⁰
        ΔTKE = sum(Δzᶜ .* budget_column(model.tracers.ρe)) - TKE⁰
        D = (0.0, combination_deficit(ρu⁰, ρv⁰, stages[1].ρu, stages[1].ρv, α[2]),
                  combination_deficit(ρu⁰, ρv⁰, stages[2].ρu, stages[2].ρv, α[3]))
        scale = abs(ΔKE) + sum(abs(s.Δhᵈ) + abs(s.dissipated) for s in stages)

        for (m, s) in enumerate(stages)
            sscale = abs(s.ΔKE) + abs(s.Δhᵈ) + abs(s.dissipated)
            @test -s.ΔKE ≈ s.shear rtol=1e-8 atol=1e-12 * sscale
            @test -s.Δhᵈ ≈ s.buoyancy rtol=1e-7 atol=max(1e-11 * sscale, s.Δhᵈ_rounding)
            @test s.Δρe + s.given_explicitly ≈ s.converted - s.dissipated + s.shortfall + s.combination rtol=1e-9 atol=1e-11 * sscale
            # what the closure gave the TKE is the kinetic energy the combination removed
            @test s.combination ≈ D[m] rtol=1e-10 atol=1e-14 * scale
        end
        @test D[3] > 0

        # the mean flow's loss over the step: what the solves converted and what the combinations removed
        shear = sum(w[m] * stages[m].shear for m in 1:3)
        deficit = sum(w[m] * D[m] for m in 1:3)
        @test -ΔKE ≈ shear + deficit rtol=1e-8 atol=1e-12 * scale
        # the TKE over the step
        received = sum(w[m] * (stages[m].converted - stages[m].dissipated + stages[m].shortfall + stages[m].combination) for m in 1:3)
        @test ΔTKE ≈ received rtol=1e-8 atol=1e-11 * scale
        # the shear production the TKE received over the step is the mean flow's loss
        @test sum(w[m] * (stages[m].shear + stages[m].combination) for m in 1:3) ≈ -ΔKE rtol=1e-8 atol=1e-12 * scale
    end
end

# A Float32 grid with the closure's parameters in Float64 (the default float type): where e = 0
# under a negative buoyancy conversion the buoyancy sink rate -B⁻ / max(e, floor) must stay finite
# in Float32, so its floor has to be representable there. Here e = 0 above 1500 m, in the inversion.
@testset "Float32 BOMEX-like column with the distribution stays finite ($path)" for path in all_paths
    model = moist_tke_column(; surface_fluxes = true, path, FT = Float32, e₀ = z -> z < 1500 ? 0.6 * exp(-z / 500) : 0.0)
    @test eltype(model.grid) == Float32
    Oceananigans.TimeSteppers.update_state!(model)
    @test all(isfinite, interior(model.closure_fields.Lᵉ))
    for n in 1:10
        time_step!(model, 10)
    end
    for field in (Oceananigans.prognostic_fields(model)..., model.closure_fields.Kᶜ, model.closure_fields.Lᵉ)
        @test all(isfinite, interior(field))
    end
end

#####
##### The whole step of a multirate infinitesimal step on a single column. Stage i of s forms
#####
#####   Vᵢ = Σⱼ wᵢⱼ Yⱼ + Δt Σⱼ βᵢⱼ Sⱼ,   wᵢⱼ = αᵢⱼ + γᵢⱼ (j ≥ 1),   wᵢ₀ = 1 - Σⱼ wᵢⱼ,   Y₀ = Uⁿ,   Sⱼ = S(Yⱼ₋₁),
#####
##### and Yᵢ follows from the vertically implicit solves over dᵢ Δt, dᵢ = Σⱼ βᵢⱼ. Every quantity X that is linear in
##### the prognostics obeys Xᵢ - Xⁿ = Σⱼ wᵢⱼ (Xⱼ - Xⁿ) + (what stage i adds), so what stage i adds reaches the end of the
##### step with the weight ωᵢ, ωₛ = 1, ωⱼ = Σ_{i>j} ωᵢ wᵢⱼ (Σᵢ ωᵢ dᵢ = 1). The kinetic energy is quadratic: the
##### recombination of the states removes Dᵢ = Σ_{p<q} wᵢₚ wᵢq Σ ρ Δz |uₚ - u_q|² / 2 from the mean flow (of either sign:
##### the weights are not all positive). With Dᵢ given to the TKE and the stage-state conversions recombined with
##### βᵢⱼ like the tendencies that carry them,
#####
#####   -[KE(uⁿ⁺¹) - KE(uⁿ)]      = Σᵢ ωᵢ (shearᵢ + Dᵢ)
#####   Σ ρe Δz |ⁿ⁺¹ - Σ ρe Δz |ⁿ = Σᵢ ωᵢ (solvedᵢ + Δt Σⱼ βᵢⱼ stage_stateⱼ - dissipatedᵢ + shortfallᵢ + Dᵢ)
#####
##### Dᵢ is recomputed here from the stage states, independently of the closure.
#####

multirate_state_weights(tableau, i) = (w = [tableau.α[i, j] + tableau.γ[i, j] for j in 1:i-1]; vcat(1 - sum(w), w))

function multirate_step_weights(tableau)
    s = length(tableau.d)
    ω = zeros(s)
    ω[s] = 1
    for j in s-1:-1:1
        ω[j] = sum(ω[i] * (tableau.α[i, j] + tableau.γ[i, j]) for i in j+1:s)
    end
    return ω
end

# The default coefficient set, which keeps two registers, and one that accumulates the stage forcings
@testset "TKE energy budget of the whole multirate infinitesimal step ($coefficients, $path, $FT, surface fluxes = $surface_fluxes)" for
        coefficients in (:ConvexLinearMIS6, :SSPMIS43), path in whole_shear_paths, FT in (Float64, Float32), surface_fluxes in (false, true)
    model = moist_tke_column(; path, FT, surface_fluxes, timestepper = :MultirateInfinitesimalStep,
                             timestepper_kwargs = (; coefficients))
    @test model.timestepper isa MultirateInfinitesimalStep
    @test isnothing(model.timestepper.substepper)
    Δt = 10
    tableau = Breeze.TimeSteppers.multirate_infinitesimal_step_tableau(coefficients) # in Float64
    (; β, d) = tableau
    s = length(d)
    ω = multirate_step_weights(tableau)
    @test sum(ω .* d) ≈ 1
    storage = model.timestepper.closure_storage
    ρ = Float64.(budget_column(model.dynamics.reference_state.density))
    Δzᶜ = diff(znodes(model.grid, Face()))
    KE(ρu, ρv) = sum(Δzᶜ .* (ρu .^ 2 .+ ρv .^ 2) ./ (2ρ))
    column(f) = Float64.(budget_column(f))
    # The sums are formed here in Float64 from the model's fields. In Float64 the budgets close to the tolerances
    # of the SSP-RK3 tests above; in Float32 to the round-off of the model's own arithmetic, a few ϵ of the summed
    # kinetic energy and TKE.
    ϵ = eps(FT)
    tight = FT == Float64
    time_step!(model, Δt) # the first step's preparation runs `update_state!` once more
    for n in 1:10
        states = [(ρu = column(model.momentum.ρu), ρv = column(model.momentum.ρv))]
        KE⁰ = KE(states[1].ρu, states[1].ρv)
        TKE⁰ = sum(Δzᶜ .* column(model.tracers.ρe))
        stages = []
        recorder = StageRecorder(model -> begin
            i = length(stages) + 1
            budget = stage_energy_budget(model, d[i] * Δt; ρe = budget_column(model.tracers.ρe) .* budget_column(model.dynamics.reference_state.density))
            exchange = d[i] * Δt * sum(ρ .* Δzᶜ .* column(storage.recombined_exchange))
            push!(stages, merge(budget, (; exchange)))
            push!(states, (ρu = column(model.momentum.ρu), ρv = column(model.momentum.ρv)))
        end)
        time_step!(model, Δt; callbacks = [recorder])
        @test length(stages) == s

        ΔKE = KE(column(model.momentum.ρu), column(model.momentum.ρv)) - KE⁰
        ΔTKE = sum(Δzᶜ .* column(model.tracers.ρe)) - TKE⁰
        D = map(1:s) do i
            w = multirate_state_weights(tableau, i)
            sum(w[p] * w[q] * sum(Δzᶜ .* ((states[p].ρu .- states[q].ρu) .^ 2 .+ (states[p].ρv .- states[q].ρv) .^ 2) ./ (2ρ))
                for p in 1:i for q in p+1:i; init = 0.0)
        end
        scale = abs(ΔKE) + sum(abs(b.Δhᵈ) + abs(b.dissipated) for b in stages)
        # In Float32 the residuals are those of rounding the prognostics: measured up to 1.5 ϵ KE⁰ in the kinetic
        # energy (0.014 J m⁻², where the mean flow loses 0.4 to 0.6 J m⁻² in a step) and 0.4 ϵ TKE⁰ in the TKE
        roundoff_KE = 8ϵ * KE⁰
        # The TKE also carries the rounding of each stage's update, cell by cell in the model's arithmetic: in every
        # stage the start state, the increment and the recovered state are rounded, and the tridiagonal solve of
        # the TKE rounds its elimination and back substitution (about seven operations per stage, each to ϵ of the
        # TKE of the cell and of the terms the stage adds). With a finite exchange length ℂʷᴸ the sink amplitude of
        # the budget third moment comes from one more tridiagonal solve over the column, whose elimination
        # accumulates up to Nz roundings. The same residual, measured in units of ϵ times the sum below, is seen in
        # Float64 and Float32 (up to 3 ϵ with one amplitude, 20 to 38 ϵ with the exchanged one), so it is round-off,
        # and the allowance is its bound, (7 (s + 1) + Nz) ϵ
        Nz = size(model.grid, 3)
        roundoff_TKE = (7 * (s + 1) + Nz) * ϵ * (TKE⁰ + sum(abs(b.Δρe) + abs(b.solved) + abs(b.exchange) + abs(b.dissipated) +
                                        abs(b.shortfall) + abs(b.combination) for b in stages))

        for (i, b) in enumerate(stages)
            sscale = abs(b.ΔKE) + abs(b.Δhᵈ) + abs(b.dissipated)
            @test -b.ΔKE ≈ b.shear rtol=1e-8 atol=(tight ? 1e-12 * sscale : roundoff_KE)
            tight && @test -b.Δhᵈ ≈ b.buoyancy rtol=1e-7 atol=max(1e-11 * sscale, b.Δhᵈ_rounding)
            # the TKE receives what the solves converted, the recombined exchange of the stage states, and
            # what the recombination removed from the mean flow, less what it dissipates
            @test b.Δρe ≈ b.solved + b.exchange - b.dissipated + b.shortfall + b.combination rtol=1e-9 atol=(tight ? 1e-11 * sscale : roundoff_TKE)
            # the recombined exchange is that of the stage states, with the weights of the slow tendencies
            recombined = Δt * sum(β[i, j] * (stages[j].stage_state_rate - stages[j].given_rate) for j in 1:i)
            @test b.exchange ≈ recombined rtol=(tight ? 1e-10 : 1e-4) atol=(tight ? 1e-13 * scale : roundoff_TKE)
            # without explicit momentum tendencies, what the closure gave the TKE is the kinetic energy the
            # recombination of the stage states removed
            surface_fluxes || @test b.combination ≈ D[i] rtol=(tight ? 1e-9 : 1e-3) atol=(tight ? 1e-14 * scale : roundoff_KE)
        end
        @test stages[1].combination == 0

        # the TKE over the step: every stage's solves and correction, and the sources its tendencies gave
        received = sum(ω[i] * (stages[i].Δρe + Δt * sum(β[i, j] * stages[j].given_rate for j in 1:i)) for i in 1:s)
        @test ΔTKE ≈ received rtol=1e-8 atol=(tight ? 1e-11 * scale : roundoff_TKE)
        accounted = sum(ω[i] * (stages[i].solved + Δt * sum(β[i, j] * stages[j].stage_state_rate for j in 1:i) -
                                stages[i].dissipated + stages[i].shortfall + stages[i].combination) for i in 1:s)
        @test ΔTKE ≈ accounted rtol=1e-8 atol=(tight ? 1e-11 * scale : roundoff_TKE)

        if !surface_fluxes
            # the mean flow's loss over the step: what the solves converted and what the recombinations removed
            @test -ΔKE ≈ sum(ω[i] * (stages[i].shear + D[i]) for i in 1:s) rtol=1e-8 atol=(tight ? 1e-12 * scale : roundoff_KE)
            # the shear production the TKE received over the step is the mean flow's loss
            @test sum(ω[i] * (stages[i].shear + stages[i].combination) for i in 1:s) ≈ -ΔKE rtol=1e-8 atol=(tight ? 1e-12 * scale : roundoff_KE)
            # stage 2 of SSPMIS43 extrapolates beyond the first stage's state, which adds kinetic energy
            coefficients === :SSPMIS43 && @test D[2] < 0
            @info "multirate step $n ($FT)" ΔKE shear=sum(ω[i] * stages[i].shear for i in 1:s) recombination=sum(ω .* D) D
        end
    end
end

#####
##### The heating budget flux: a flux of θˡⁱ in flux form, zero at the boundaries and in still air, whose conversion the
##### TKE equation receives. The dynamic enthalpy the flux takes from the mean state is recomputed here from the tendency
##### of ρθ, with the derivatives of the buoyancy by central differences.
#####

# The tendencies of ρθ and ρe that the closure adds at the start of a stage (the time steppers call
# `compute_closure_tendencies!` after the tendencies of `update_state!`), alone
function heating_tendencies(model)
    Gⁿ = model.timestepper.Gⁿ
    fill!(parent(Gⁿ.ρθ), 0)
    fill!(parent(Gⁿ.ρe), 0)
    Breeze.AtmosphereModels.compute_closure_tendencies!(Gⁿ, model.closure_fields, model.closure, model)
    return (ρθ = budget_column(Gⁿ.ρθ), ρe = budget_column(Gⁿ.ρe))
end

@testset "Heating budget flux: flux form, boundaries and still air ($path, $FT)" for path in keys(heating_paths), FT in (Float64, Float32)
    model = moist_tke_column(; path, FT)
    Oceananigans.TimeSteppers.update_state!(model)
    Nz = size(model.grid, 3)
    cl = model.closure_fields.subgrid_cloud.cloud
    hb = model.closure.heating_budget_flux
    ρ = Float64.(budget_column(model.dynamics.reference_state.density))
    Δzᶜ = diff(znodes(model.grid, Face()))
    𝒬 = Float64.(budget_column(cl.𝒬)); s = Float64.(budget_column(cl.sᑫ)); J = Float64.(budget_column(cl.Jᑫ)); Jᶜ = Float64.(budget_column(cl.Jᑫᶜ))
    ϵ = eps(FT)

    # the heating is the prescribed one; the redistribution conserves it, so the flux vanishes at the top
    @test 𝒬 ≈ Float64.(budget_column(hb.heating.tendency))
    @test minimum(𝒬) < 0 < maximum(𝒬)
    # (the sum of ρ Δz (𝒬 - s) rounds against both the heating and its redistribution, so the bound carries both)
    @test abs(sum(ρ .* Δzᶜ .* (𝒬 .- s))) ≤ 100ϵ * sum(ρ .* Δzᶜ .* (abs.(𝒬) .+ abs.(s)))
    @test J[1] == 0 && J[Nz+1] == 0 && Jᶜ[1] == 0 && Jᶜ[Nz+1] == 0
    @test all(≥(0), J) && all(≥(0), Jᶜ)

    # cooling above and warming below both drive an upward flux; it is the share ℂᑫ of the positive part of the
    # integral of ρ (𝒬 - s), and it is applied below the cooled cells, which export their cooling
    ρᶠ = [(ρ[k] + ρ[k-1]) / 2 for k in 2:Nz]
    ρJ = cumsum(ρ .* Δzᶜ .* (𝒬 .- s))[1:Nz-1]
    @test J[2:Nz] ≈ hb.ℂᑫ .* max.(ρJ, 0) ./ ρᶠ rtol=1e4ϵ atol=1e4ϵ * maximum(J)
    @test maximum(J) > 1e-3 # K m s⁻¹: a tenth of the 100 m × 5 K h⁻¹ of the cooling is 1.4e-2
    zᶠ = znodes(model.grid, Face())
    @test 300 < zᶠ[argmax(J)] ≤ 1100 # within the cooled cells or below them

    # the conversion the TKE equation receives is positive and reduced by its latent part
    Gᑫ = budget_column(cl.Gᑫ); Gᑫᴸ = budget_column(cl.Gᑫᴸ); Gᑫᶜ = budget_column(cl.Gᑫᶜ)
    @test all(≥(0), Gᑫ) && maximum(Gᑫ) > 0
    # (no latent part with the distribution here: the column has no cloud droplets yet, so nothing condenses in an eddy time)
    @test all(≤(0), Gᑫᴸ) && (path === :heating_budget_flux || minimum(Gᑫᴸ) < 0)
    @test all(≥(0), Gᑫᶜ) && maximum(Gᑫᶜ) > 0

    # still air: no turbulent kinetic energy where the heating acts and around it, and no flux at all
    still = moist_tke_column(; path, FT, e₀ = z -> z < 200 ? 0.5 : 0.0, closure_keywords = (; minimum_tke = 0))
    Oceananigans.TimeSteppers.update_state!(still)
    cl = still.closure_fields.subgrid_cloud.cloud
    Jˢ = budget_column(cl.Jᑫ); zᶠ = znodes(still.grid, Face())
    # (to the rounding of the balance of the turbulent layer below, which the integral carries upward)
    @test all(k -> abs(Jˢ[k]) ≤ 1e3ϵ * sum(ρ .* Δzᶜ .* abs.(𝒬)) / ρ[1], findall(>(300), zᶠ))
    @test all(k -> budget_column(cl.sᑫ)[k] == budget_column(cl.𝒬)[k], findall(>(300), znodes(still.grid, Center())))
end

# The flux applied to the mean state and its conversion, against the same column without the flux (ℂᑫ = 0). Without a
# distribution the saturated fraction does not read the flux, so the two columns differ by the flux alone.
@testset "Heating budget flux: the energy the flux takes from the mean state is what the TKE receives" begin
    path = :heating_budget_flux_no_distribution
    with_flux = moist_tke_column(; path, e₀ = z -> 0.6)
    hb = with_flux.closure.heating_budget_flux
    diagnostic = HeatingBudgetFlux(ℂᑫ = 0, ℂᑫᴸ = hb.ℂᑫᴸ, heating = PrescribedHeating(cloud_top_cooling(with_flux.grid)))
    without_flux = moist_tke_column(; path, e₀ = z -> 0.6, closure_keywords = (; heating_budget_flux = diagnostic))
    Oceananigans.TimeSteppers.update_state!(with_flux)
    Oceananigans.TimeSteppers.update_state!(without_flux)

    grid = with_flux.grid
    Nz = size(grid, 3)
    ρ = budget_column(with_flux.dynamics.reference_state.density)
    zᶜ = znodes(grid, Center()); zᶠ = znodes(grid, Face())
    Δzᶜ = diff(zᶠ); Δzᶠ = diff(zᶜ)
    cl = with_flux.closure_fields.subgrid_cloud.cloud
    cl⁰ = without_flux.closure_fields.subgrid_cloud.cloud
    G = heating_tendencies(with_flux); G⁰ = heating_tendencies(without_flux)

    # with ℂᑫ = 0 the flux is a diagnostic: nothing is applied or converted, and the flux of the cooling is still diagnosed
    @test all(==(0), budget_column(cl⁰.Jᑫ)) && all(==(0), budget_column(cl⁰.Gᑫ)) && all(==(0), budget_column(cl⁰.Gᑫᴸ))
    @test budget_column(cl⁰.Jᑫᶜ) == budget_column(cl.Jᑫᶜ) && maximum(budget_column(cl⁰.Gᑫᶜ)) > 0
    @test budget_column(cl⁰.B) .+ budget_column(cl.Gᑫ) ≈ budget_column(cl.B) rtol=1e-12

    # the tendency of ρθ: the divergence of ρᶠ Jᑫ, and the draw of the latent part of the conversion
    Jᑫ = budget_column(cl.Jᑫ); Gᑫ = budget_column(cl.Gᑫ); Gᑫᴸ = budget_column(cl.Gᑫᴸ)
    ρᶠ = vcat(0.0, [(ρ[k] + ρ[k-1]) / 2 for k in 2:Nz], 0.0)
    divergence = [(ρᶠ[k+1] * Jᑫ[k+1] - ρᶠ[k] * Jᑫ[k]) / Δzᶜ[k] for k in 1:Nz]
    latent_draw = ρ .* Gᑫᴸ ./ budget_column(cl.ĉᵖΠ)
    scale = maximum(abs, divergence)
    @test G.ρθ .- G⁰.ρθ ≈ -divergence .- latent_draw rtol=1e-9 atol=1e-9 * scale
    flux_tendency = G.ρθ .- G⁰.ρθ .+ latent_draw
    @test abs(sum(Δzᶜ .* flux_tendency)) ≤ 1e-10 * sum(Δzᶜ .* abs.(flux_tendency)) # flux form: θˡⁱ is conserved

    # the dynamic enthalpy the flux takes from the mean state, -d/dt Σ ρ Δz h‡ = Σ Δz Z ∂t(ρθ), with Z from the
    # derivatives of the buoyancy by central differences, is the dry part of the conversion
    c = closure_energy_conversions(with_flux)
    Z = zeros(Nz)
    Z[1] = c.β[1, 1] * (zᶜ[1] - zᶠ[1])
    for k in 2:Nz
        Z[k] = Z[k-1] + (c.β[k, 1] + c.β[k-1, 1]) / 2 * Δzᶠ[k-1]
    end
    dynamic_enthalpy_loss = sum(Δzᶜ .* Z .* flux_tendency)
    @test dynamic_enthalpy_loss > 0
    @test dynamic_enthalpy_loss ≈ sum(ρ .* Δzᶜ .* (Gᑫ .- Gᑫᴸ)) rtol=1e-7

    # the TKE equation receives the conversion (here B > 0 with and without it wherever the flux is)
    B = budget_column(cl.B); B⁰ = budget_column(cl⁰.B)
    @test G.ρe .- G⁰.ρe ≈ ρ .* (max.(B, 0) .- max.(B⁰, 0)) rtol=1e-9 atol=1e-12 * maximum(abs, G.ρe)
    # and the internal energy pays the latent part: Σ Δz ĉᵖΠ (draw) = -Σ ρ Δz Gᑫᴸ
    @test sum(Δzᶜ .* budget_column(cl.ĉᵖΠ) .* latent_draw) ≈ sum(ρ .* Δzᶜ .* Gᑫᴸ) rtol=1e-12
end

# The coefficients are parameters of the closure, within their bounds
@testset "Heating budget flux: coefficients" begin
    closure = TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian(), heating_budget_flux = HeatingBudgetFlux())
    parameters = closure_parameters(closure)
    bounds = heating_budget_flux_bounds()
    @test all(name -> bounds[name][1] ≤ parameters[name] ≤ bounds[name][2], keys(bounds))
    @test with_parameters(closure; ℂᑫ = 0.25, ℂᑫᴸ = 4).heating_budget_flux.ℂᑫ == 0.25
    @test with_parameters(closure; ℂᑫ = 0.25).heating_budget_flux.heating isa RadiativeHeating
    @test_throws ArgumentError HeatingBudgetFlux(ℂᑫ = 1.5)
    @test_throws ArgumentError HeatingBudgetFlux(ℂᑫᴸ = Inf)
    @test_throws ArgumentError TKEBasedTurbulenceClosure(; heating_budget_flux = HeatingBudgetFlux())
end

@testset "SimplifiedHigherOrderClosure: the shear production not given to the TKE heats the air" begin
    shoc = moist_tke_column(; path = :shoc, surface_fluxes = true)
    whole = moist_tke_column(; path = :shoc, surface_fluxes = true,
                             closure_keywords = (; stability_functions = ColumnStabilityDamping(; ℂᴾ = 1)))
    @test shoc.closure isa SimplifiedHigherOrderClosure
    @test shoc.closure.stability_functions.ℂᴾ == 0.1
    for model in (shoc, whole)
        Oceananigans.TimeSteppers.update_state!(model)
        map(G -> fill!(parent(G), 0), model.timestepper.Gⁿ)
        Breeze.AtmosphereModels.compute_closure_tendencies!(model.timestepper.Gⁿ, model.closure_fields, model.closure, model)
    end
    cl = shoc.closure_fields.subgrid_cloud.cloud
    ρ = budget_column(shoc.dynamics.reference_state.density)
    P = budget_column(cl.P)
    ĉᵖΠ = budget_column(cl.ĉᵖΠ)
    Gᵉ(model) = budget_column(model.timestepper.Gⁿ.ρe)
    Gᶿ(model) = budget_column(model.timestepper.Gⁿ.ρθ)
    # The same state: ℂᴾ changes only the partition of P between the TKE and the internal energy
    @test budget_column(shoc.tracers.ρe) == budget_column(whole.tracers.ρe)
    @test maximum(abs, P) > 0
    scale = maximum(abs, ρ .* P)
    # The stage tendency carries the shear production that the solves do not give before the TKE solve: with a time
    # stepper that solves momentum first, only the work of the surface stress Pˢ (the conversion of the solved
    # momentum reaches the TKE, ℂᴾ of it, and the air, the rest, before their solves)
    injected = Breeze.TurbulenceClosures.solved_shear_before_tke_solve(shoc.closure_fields.subgrid_cloud, shoc.closure, shoc)
    @test injected
    Pᵉ = injected ? budget_column(cl.Pˢ) : P
    scale = maximum(abs, ρ .* Pᵉ)
    @test Gᵉ(whole) .- Gᵉ(shoc) ≈ 0.9 .* ρ .* Pᵉ rtol=1e-10 atol=1e-12 * scale
    @test ĉᵖΠ .* (Gᶿ(shoc) .- Gᶿ(whole)) ≈ 0.9 .* ρ .* Pᵉ rtol=1e-10 atol=1e-12 * scale
end
