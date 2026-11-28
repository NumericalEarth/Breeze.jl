include(joinpath(@__DIR__, "setup.jl"))

#####
##### The energy budget of TKEBasedTurbulenceClosure: per stage of SSPRungeKutta3 and of MultirateInfinitesimalStep, the
##### TKE receives the energy the closure's fluxes take from the mean state and its dissipation heats the air
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
heating_column(cl, name, n) = hasproperty(cl, name) ? budget_column(getproperty(cl, name)) : zeros(n)

shoc_components(c; kw...) = (mixing_length = c.mixing_length, stability_functions = c.stability_functions,
                             static_stability = c.static_stability, subgrid_distribution = c.subgrid_distribution,
                             heating_budget_flux = nothing)

# The updated TKE closure (the defaults), SHOC, and SHOC with the whole shear production given to the TKE
closure_paths = (tke = NamedTuple(),
                 shoc = shoc_components(SimplifiedHigherOrderClosure()),
                 shoc_whole_shear = shoc_components(SimplifiedHigherOrderClosure(stability_functions = ColumnStabilityDamping(; ℂᴾ = 1))))

function moist_tke_column(; rain = false, surface_fluxes = false, Nz = 40, H = 2400, path = :distribution, FT = Float64,
                          e₀ = z -> 0.6 * exp(-z / 500), timestepper = :SSPRungeKutta3, timestepper_kwargs = NamedTuple(),
                          closure_keywords = NamedTuple())
    grid = RectilinearGrid(CPU(), FT; size=Nz, z=(0, H), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=299)
    microphysics = PredictedParticlePropertiesMicrophysics(FT; thermodynamic_constants = constants,
                                                           negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    closure = TKEBasedTurbulenceClosure(FT; minimum_tke = 4e-4, closure_paths[path]..., closure_keywords...)
    boundary_conditions = surface_fluxes ?
        (ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 8e-3)),
         ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 5.2e-5)),
         ρu = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 0.08)),
         ρv = FieldBoundaryConditions(bottom = FluxBoundaryCondition(-1.17 * 0.02))) : NamedTuple()
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), thermodynamic_constants=constants, microphysics, closure,
                            boundary_conditions, timestepper, timestepper_kwargs)

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
    Jᶿ = heating_column(cl, :Jᑫ, Nz + 1) # the heating budget flux of θˡⁱ, zero without one
    Bᶠ = [-Kᶜ[f] * sum((β[f, s] + β[f-1, s]) / 2 * (φ[f, s] - φ[f-1, s]) for s in 1:4) / Δzᶠ[f-1] +
          (β[f, 1] + β[f-1, 1]) / 2 * Jᶿ[f] for f in 2:Nz]
    Pᶠ = [Kᵘ[f] * ((u[f] - u[f-1])^2 + (v[f] - v[f-1])^2) / Δzᶠ[f-1]^2 for f in 2:Nz]

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

@testset "TKE conversions of the applied fluxes (rain = $rain, $path)" for rain in (false, true), path in (:tke, :shoc)
    model = moist_tke_column(; rain, path)
    Oceananigans.TimeSteppers.update_state!(model)
    for n in 1:5
        time_step!(model, 10)
        Oceananigans.TimeSteppers.update_state!(model)
        c = closure_energy_conversions(model)
        scale = c.mean_kinetic_energy_loss + abs(c.buoyancy_conversion) + c.dissipation
        @test c.shear_production ≈ c.mean_kinetic_energy_loss rtol=1e-12
        @test c.buoyancy_production ≈ c.buoyancy_conversion rtol=1e-7 atol=1e-7 * scale
        @test c.stored_β ≈ c.β rtol=1e-6
        @test c.latent_production ≈ c.latent_conversion rtol=1e-12 atol=1e-14 * scale
    end
end

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

    KE(ρu, ρv) = sum(Δzᶜ .* (ρu .^ 2 .+ ρv .^ 2) ./ (2ρ))
    ΔKE = KE(budget_column(model.momentum.ρu), budget_column(model.momentum.ρv)) -
          KE(budget_column(cl.ρu⁻), budget_column(cl.ρv⁻))

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
    after[:, 1] .-= budget_column(cl.after_solves_ρθ) # the solves' change, without what θˡⁱ receives after them
    before = ρφ(pre)
    Δhᵈ = -sum(Δzᶜ .* sum(Z .* (after .- before), dims=2))
    Δhᵈ_rounding = eps(eltype(Z)) * sum(Δzᶜ .* sum(abs.(Z .* after) .+ abs.(Z .* before), dims=2))

    Δρe = sum(Δzᶜ .* (ρe .- budget_column(pre.ρe)))
    given_explicitly = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.given))
    converted = Δt * sum(ρ .* Δzᶜ .* (budget_column(cl.Pᵖ) .+ budget_column(cl.Bᵖ) .+ budget_column(cl.Bᴸⁱ) .+ budget_column(cl.stage_surface)))
    dissipated = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.ε))
    shortfall = sum(Δzᶜ .* budget_column(cl.shortfall)) # paid by the internal energy instead of the TKE
    combination = sum(Δzᶜ .* budget_column(cl.stage_combination)) # the kinetic energy the stage combination removed
    solved = Δt * sum(ρ .* Δzᶜ .* (budget_column(cl.Pᵖ) .+ budget_column(cl.Bᵖ) .+ budget_column(cl.Bᴸⁱ)))
    given_rate = sum(ρ .* Δzᶜ .* budget_column(cl.given))
    stage_state_rate = sum(ρ .* Δzᶜ .* budget_column(cl.stage_surface))
    return (; ΔKE, Δhᵈ, Δhᵈ_rounding, Δρe, given_explicitly, converted, dissipated, shortfall, combination, solved, given_rate, stage_state_rate,
            shear = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.Pᵖ)), buoyancy = Δt * sum(ρ .* Δzᶜ .* budget_column(cl.Bᵖ)))
end

@testset "TKE energy budget across the implicit solves (rain = $rain, surface fluxes = $surface_fluxes, $path)" for
        rain in (false, true), surface_fluxes in (false, true), path in (:tke, :shoc)
    model = moist_tke_column(; rain, surface_fluxes, path)
    Δt = 10
    Δtₛ = model.timestepper.α³ * Δt
    for n in 1:15
        time_step!(model, Δt)
        b = stage_energy_budget(model, Δtₛ)
        scale = abs(b.ΔKE) + abs(b.Δhᵈ) + abs(b.dissipated)
        @test -b.ΔKE ≈ b.shear rtol=1e-8 atol=1e-12 * scale
        @test -b.Δhᵈ ≈ b.buoyancy rtol=(rain ? 1e-5 : 1e-7) atol=1e-11 * scale
        @test b.Δρe + b.given_explicitly ≈ b.converted - b.dissipated + b.shortfall + b.combination rtol=1e-9 atol=1e-11 * scale
    end
end

struct StageRecorder{F}
    callsite :: Oceananigans.UpdateStateCallsite
    record :: F
end

StageRecorder(record) = StageRecorder(Oceananigans.UpdateStateCallsite(), record)
(recorder::StageRecorder)(model) = recorder.record(model)

@testset "TKE energy budget of the whole step ($path)" for path in (:tke, :shoc_whole_shear)
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
            @test s.combination ≈ D[m] rtol=1e-10 atol=1e-14 * scale
        end
        @test D[3] > 0

        shear = sum(w[m] * stages[m].shear for m in 1:3)
        deficit = sum(w[m] * D[m] for m in 1:3)
        @test -ΔKE ≈ shear + deficit rtol=1e-8 atol=1e-12 * scale
        received = sum(w[m] * (stages[m].converted - stages[m].dissipated + stages[m].shortfall + stages[m].combination) for m in 1:3)
        @test ΔTKE ≈ received rtol=1e-8 atol=1e-11 * scale
        @test sum(w[m] * (stages[m].shear + stages[m].combination) for m in 1:3) ≈ -ΔKE rtol=1e-8 atol=1e-12 * scale
    end
end

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

@testset "TKE energy budget of the whole multirate infinitesimal step ($coefficients, $path, $FT, surface fluxes = $surface_fluxes)" for
        coefficients in (:ConvexLinearMIS6, :SSPMIS43), path in (:tke,), FT in (Float64, Float32), surface_fluxes in (false, true)
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
        roundoff_KE = 8ϵ * KE⁰
        Nz = size(model.grid, 3)
        roundoff_TKE = (7 * (s + 1) + Nz) * ϵ * (TKE⁰ + sum(abs(b.Δρe) + abs(b.solved) + abs(b.exchange) + abs(b.dissipated) +
                                        abs(b.shortfall) + abs(b.combination) for b in stages))

        for (i, b) in enumerate(stages)
            sscale = abs(b.ΔKE) + abs(b.Δhᵈ) + abs(b.dissipated)
            @test -b.ΔKE ≈ b.shear rtol=1e-8 atol=(tight ? 1e-12 * sscale : roundoff_KE)
            tight && @test -b.Δhᵈ ≈ b.buoyancy rtol=1e-7 atol=max(1e-11 * sscale, b.Δhᵈ_rounding)
            @test b.Δρe ≈ b.solved + b.exchange - b.dissipated + b.shortfall + b.combination rtol=1e-9 atol=(tight ? 1e-11 * sscale : roundoff_TKE)
            recombined = Δt * sum(β[i, j] * (stages[j].stage_state_rate - stages[j].given_rate) for j in 1:i)
            @test b.exchange ≈ recombined rtol=(tight ? 1e-10 : 1e-4) atol=(tight ? 1e-13 * scale : roundoff_TKE)
            surface_fluxes || @test b.combination ≈ D[i] rtol=(tight ? 1e-9 : 1e-3) atol=(tight ? 1e-14 * scale : roundoff_KE)
        end
        @test stages[1].combination == 0

        received = sum(ω[i] * (stages[i].Δρe + Δt * sum(β[i, j] * stages[j].given_rate for j in 1:i)) for i in 1:s)
        @test ΔTKE ≈ received rtol=1e-8 atol=(tight ? 1e-11 * scale : roundoff_TKE)
        accounted = sum(ω[i] * (stages[i].solved + Δt * sum(β[i, j] * stages[j].stage_state_rate for j in 1:i) -
                                stages[i].dissipated + stages[i].shortfall + stages[i].combination) for i in 1:s)
        @test ΔTKE ≈ accounted rtol=1e-8 atol=(tight ? 1e-11 * scale : roundoff_TKE)

        if !surface_fluxes
            @test -ΔKE ≈ sum(ω[i] * (stages[i].shear + D[i]) for i in 1:s) rtol=1e-8 atol=(tight ? 1e-12 * scale : roundoff_KE)
            @test sum(ω[i] * (stages[i].shear + stages[i].combination) for i in 1:s) ≈ -ΔKE rtol=1e-8 atol=(tight ? 1e-12 * scale : roundoff_KE)
            coefficients === :SSPMIS43 && @test D[2] < 0
        end
    end
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
    @test budget_column(shoc.tracers.ρe) == budget_column(whole.tracers.ρe)
    @test maximum(abs, P) > 0
    scale = maximum(abs, ρ .* P)
    injected = Breeze.TurbulenceClosures.solved_conversions_before_tke_solve(shoc.closure_fields.subgrid_cloud, shoc.closure, shoc)
    @test injected
    Pᵉ = injected ? budget_column(cl.Pˢ) : P
    scale = maximum(abs, ρ .* Pᵉ)
    @test Gᵉ(whole) .- Gᵉ(shoc) ≈ 0.9 .* ρ .* Pᵉ rtol=1e-10 atol=1e-12 * scale
    @test ĉᵖΠ .* (Gᶿ(shoc) .- Gᶿ(whole)) ≈ 0.9 .* ρ .* Pᵉ rtol=1e-10 atol=1e-12 * scale
end

