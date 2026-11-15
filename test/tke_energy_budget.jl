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

# The two paths: the DoubleGaussian distribution, and none with the microphysics-like fraction
closure_paths = (distribution = (static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()),
                                 subgrid_distribution = DoubleGaussian()),
                 no_distribution = (static_stability = SaturationWeightedStaticStability(HalfSaturatedFraction()),
                                    subgrid_distribution = nothing))
budget_column(f::ZeroField, Nz) = zeros(Nz)
budget_column(f, Nz) = budget_column(f)

function moist_tke_column(; rain = false, surface_fluxes = false, Nz = 40, H = 2400, path = :distribution, FT = Float64,
                          e₀ = z -> 0.6 * exp(-z / 500), timestepper = :SSPRungeKutta3)
    grid = RectilinearGrid(CPU(), FT; size=Nz, z=(0, H), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=299)
    microphysics = PredictedParticlePropertiesMicrophysics(FT; thermodynamic_constants = constants,
                                                           negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    closure = TKEBasedTurbulenceClosure(; minimum_tke = 4e-4, closure_paths[path]...)
    boundary_conditions = surface_fluxes ?
        (ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 8e-3)),
         ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 5.2e-5)),
         ρu = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 0.08)),
         ρv = FieldBoundaryConditions(bottom = FluxBoundaryCondition(-1.17 * 0.02))) : NamedTuple()
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), thermodynamic_constants=constants, microphysics, closure, boundary_conditions, timestepper)

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
@testset "TKE conversions of the applied fluxes (rain = $rain, $path)" for rain in (false, true), path in keys(closure_paths)
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
    return (; ΔKE, Δhᵈ, Δρe, given_explicitly, converted, dissipated, shortfall, combination, solved, given_rate, stage_state_rate,
            shear = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.Pᵖ)), buoyancy = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.Bᵖ)))
end

@testset "TKE energy budget across the implicit solves (rain = $rain, surface fluxes = $surface_fluxes, $path)" for rain in (false, true), surface_fluxes in (false, true), path in keys(closure_paths)
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

@testset "TKE energy budget of the whole step ($path)" for path in keys(closure_paths)
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
            @test -s.Δhᵈ ≈ s.buoyancy rtol=1e-7 atol=1e-11 * sscale
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
@testset "Float32 BOMEX-like column with the distribution stays finite ($path)" for path in keys(closure_paths)
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

@testset "TKE energy budget of the whole multirate infinitesimal step ($path, $FT, surface fluxes = $surface_fluxes)" for path in keys(closure_paths), FT in (Float64, Float32), surface_fluxes in (false, true)
    model = moist_tke_column(; path, FT, surface_fluxes, timestepper = :MultirateInfinitesimalStep)
    @test model.timestepper isa MultirateInfinitesimalStep
    @test isnothing(model.timestepper.substepper)
    Δt = 10
    tableau = Breeze.TimeSteppers.multirate_infinitesimal_step_tableau(:SSPMIS43) # the default, in Float64
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
        roundoff_TKE = 8ϵ * TKE⁰

        for (i, b) in enumerate(stages)
            sscale = abs(b.ΔKE) + abs(b.Δhᵈ) + abs(b.dissipated)
            @test -b.ΔKE ≈ b.shear rtol=1e-8 atol=(tight ? 1e-12 * sscale : roundoff_KE)
            tight && @test -b.Δhᵈ ≈ b.buoyancy rtol=1e-7 atol=1e-11 * sscale
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
            @test D[2] < 0
            @info "multirate step $n ($FT)" ΔKE shear=sum(ω[i] * stages[i].shear for i in 1:s) recombination=sum(ω .* D) D
        end
    end
end
