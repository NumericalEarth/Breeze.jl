using Breeze
using Oceananigans
using Test

include("hydrostatic_moist_utils.jl")

using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.TimeSteppers: update_state!

using Breeze.HydrostaticEquations: VerticalDiffusion, step_vertical_diffusion!
using Breeze.HydrostaticMoistTurbulence: MoistTurbulentMixing, moist_turbulent_mixing_step!, BulkSurfaceExchange
using Breeze.Thermodynamics: Thermodynamics, adjustment_saturation_specific_humidity, saturation_specific_humidity,
                             PlanarLiquidSurface, dry_air_gas_constant, vapor_gas_constant, absolute_zero_latent_heat,
                             specific_heat_difference, mixture_gas_constant

#####
##### Tests of `MoistTurbulentMixing`: the `TKEBasedTurbulenceClosure` with `MoistStaticStability` on
##### moist `HydrostaticDynamics`. Single columns, Float64, CPU, unless a test says otherwise.
#####
##### Tolerances. Those of the budgets, of the bitwise comparisons and of the re-evaluations follow
##### from what is asserted and were written before any run. The fixtures and the bounds of the
##### parcel oracle, of the refinement and of the latent response were chosen after exploratory
##### runs of the same operator (derivations/moist_closure/logs/explore0*.log), and all of them
##### were frozen before the first run of this file (derivations/moist_closure/FREEZE.txt).
#####

const columns_directory = joinpath(@__DIR__, "hydrostatic_moist_columns")
column_state(name) = read_column(joinpath(columns_directory, name * ".csv"))

const budget_relative = 1e-12              # closed-column water and energy, over O(10²) applications
const reevaluation_relative = 1e-9         # N² against the same formulas evaluated from the model's fields
const parcel_oracle_relative = 0.10        # N² against the reversible displacement of a parcel ...
const parcel_oracle_absolute = 2e-6        # ... plus this, in s⁻²
const dry_limit_residual = 1e-10           # residual of the dry enthalpy equation over its largest term
const energy_control = 1e-10               # the enthalpy alone must fail to close by more than this
const refinement_order = (0.6, 1.6)
const condensation_minimum_warming = 0.5   # K, where the cloud liquid exceeds 2.5e-4

#####
##### Helpers
#####

function closure_column_model(column; FT=Float64, Φˢ=0, microphysics=warm_rain_microphysics(FT))
    Nz = length(column.k)
    grid = RectilinearGrid(CPU(), FT; size=Nz, x=0, y=0, z=(0, 1), topology=(Flat, Flat, Bounded))
    A = vcat(column.A_lower, column.A_upper[end])
    B = vcat(column.B_lower, column.B_upper[end])
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B, surface_geopotential=Φˢ)
    return AtmosphereModel(grid; dynamics, microphysics, tracers = :ρe,
                           formulation = moist_formulation(),
                           momentum_advection = VectorInvariant())
end

shear_profile(k, Nz) = 10 * (k / Nz)^0.3
turning_profile(k, Nz) = 2 * sin(k)

# A column state with a sheared wind and turbulent kinetic energy
function turbulent_column(name; rain=true, water=true, e=0.1, kw...)
    column, header = column_state(name)
    model = closure_column_model(column; kw...)
    Nz = length(column.k)
    rᵉ = water ? column.r_e : zero(column.r_e)
    rʳ = (water && rain) ? column.r_r : zero(column.r_r)
    set_column_state!(model, header["dry_surface_pressure_Pa"], column.theta, rᵉ, rʳ)

    ρᵗ = Array(interior(model.dynamics.moisture.total_density))
    interior(model.velocities.u) .= reshape([shear_profile(k, Nz) for k in 1:Nz], 1, 1, Nz)
    interior(model.velocities.v) .= reshape([turning_profile(k, Nz) for k in 1:Nz], 1, 1, Nz)
    interior(model.tracers.ρe) .= ρᵗ .* e
    update_state!(model; compute_tendencies=false)

    return model, column, header
end

column_values(field) = vec(Array(interior(field)))

function column_kinetic_energy(model)
    ρᵗ = model.dynamics.moisture.total_density
    u, v, _ = model.velocities
    return sum(ρᵗ[1, 1, k] * Δzᶜᶜᶜ(1, 1, k, model.grid) * (u[1, 1, k]^2 + v[1, 1, k]^2) / 2 for k in 1:size(model.grid, 3))
end

column_turbulent_kinetic_energy(model) =
    sum(model.tracers.ρe[1, 1, k] * Δzᶜᶜᶜ(1, 1, k, model.grid) for k in 1:size(model.grid, 3))

total_column_energy(model) = column_energy(model)[1, 1] + column_kinetic_energy(model) + column_turbulent_kinetic_energy(model)

ledger_value(mixing, name) = getproperty(mixing.ledger, name)[1, 1, 1]

# The closure fields of a state: a step too short to change it
function diagnose_closure_fields(model)
    mixing = MoistTurbulentMixing(model)
    prognostic = map(field -> copy(parent(field)), values(Oceananigans.prognostic_fields(model)))
    moist_turbulent_mixing_step!(mixing, model, 1e-12)
    for (field, saved) in zip(values(Oceananigans.prognostic_fields(model)), prognostic)
        parent(field) .= saved
    end
    update_state!(model; compute_tendencies=false)
    return mixing
end

# N² on the face between the layers k - 1 and k from the model's fields and geopotential, by the
# formulas of `MoistStaticStability`, written out here: the dry branch g δ(ln θᵨ) / Δh, the
# saturated branch of Durran & Klemp (1982) and the rule that chooses between them.
function reevaluated_static_stability(model, k)
    constants = model.thermodynamic_constants
    dynamics = model.dynamics
    g = constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    μ = model.microphysical_fields
    θᵨ = dynamics.moisture.density_potential_temperature
    Δh = (dynamics.geopotential[1, 1, k] - dynamics.geopotential[1, 1, k-1]) / g

    function state(m)
        T = model.temperature[1, 1, m]
        p = dynamics.pressure[1, 1, m]
        q = MoistureMassFractions(μ.qᵛ[1, 1, m], μ.qˡ[1, 1, m])
        qᵗ = q.vapor + q.liquid
        ρ = Thermodynamics.density(T, p, q, constants)
        qˢ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        qʳ = μ.ρqʳ[1, 1, m] / dynamics.moisture.total_density[1, 1, m]
        return (; T, qˢ, rˢ = qˢ / (1 - qᵗ), rᵗ = qᵗ / (1 - qᵗ), qʷ = qᵗ - qʳ,
                  lnθ = log(T) + Rᵈ / cᵖᵈ * log(dynamics.standard_pressure / p))
    end

    below, above = state(k-1), state(k)
    dry = g * (log(θᵨ[1, 1, k]) - log(θᵨ[1, 1, k-1])) / Δh

    T = (below.T + above.T) / 2
    rˢ = (below.rˢ + above.rˢ) / 2
    ℒ = absolute_zero_latent_heat(constants, constants.liquid) + specific_heat_difference(constants, constants.liquid) * T
    A = (1 + ℒ * rˢ / (Rᵈ * T)) / (1 + Rᵈ / Rᵛ * ℒ^2 * rˢ / (cᵖᵈ * Rᵈ * T^2))
    saturated = g * (A * ((above.lnθ - below.lnθ) / Δh + ℒ / (cᵖᵈ * T) * (above.rˢ - below.rˢ) / Δh) -
                     (above.rᵗ - below.rᵗ) / Δh)

    is_saturated = (below.qʷ + above.qʷ) / 2 ≥ (below.qˢ + above.qˢ) / 2
    return (; N² = is_saturated ? saturated : dry, is_saturated, dry, saturated)
end

# Temperature and vapor of air with total water qᵗ and enthalpy H = cᵖᵐ T - ℒˡ₀ qˡ at the pressure
# p, in equilibrium over liquid, by bisection
function parcel_state(H, p, qᵗ, constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)

    # Above the boiling point at this pressure the saturation formula is negative: all water is vapor
    function vapor(T)
        qˢ = adjustment_saturation_specific_humidity(T, p, qᵗ, constants, PlanarLiquidSurface())
        return qˢ < 0 ? qᵗ : min(qᵗ, qˢ)
    end

    enthalpy(T) = ((1 - qᵗ) * cᵖᵈ + vapor(T) * cᵖᵛ + (qᵗ - vapor(T)) * cˡ) * T - ℒˡ₀ * (qᵗ - vapor(T))

    lower, upper = 150.0, 400.0
    for _ in 1:200
        middle = (lower + upper) / 2
        enthalpy(middle) < H ? (lower = middle) : (upper = middle)
    end

    T = (lower + upper) / 2
    return T, vapor(T)
end

parcel_gas_constant(qᵛ, qᵗ, constants) = (1 - qᵗ) * dry_air_gas_constant(constants) + qᵛ * vapor_gas_constant(constants)

# N² on the face between the layers k - 1 and k of a column without rain from the reversible
# displacement of a parcel, which is not how the closure computes it:
#
#   N² = - g [d(ln ρ)/dh of the column - d(ln ρ)/dh of the parcel] ,
#
# with the parcel of the mean state of the two layers moved in hydrostatic balance,
# dH = Rᵐ T d(ln p), its phases in equilibrium, over ± δ in ln p. The logarithm of the density is
# differenced, as the closure differences the logarithm of θᵨ: the density of two layers of the
# upper half of the column differs by a factor of two.
function parcel_static_stability(model, k; δ=1e-3)
    constants = model.thermodynamic_constants
    g = constants.gravitational_acceleration
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    μ = model.microphysical_fields
    p = model.dynamics.pressure
    T = model.temperature
    Φ = model.dynamics.geopotential

    layer_density(m) = p[1, 1, m] / (parcel_gas_constant(μ.qᵛ[1, 1, m], μ.qᵛ[1, 1, m] + μ.qᶜˡ[1, 1, m], constants) * T[1, 1, m])
    Δh = (Φ[1, 1, k] - Φ[1, 1, k-1]) / g

    Tᶠ = (T[1, 1, k] + T[1, 1, k-1]) / 2
    qᵛ = (μ.qᵛ[1, 1, k] + μ.qᵛ[1, 1, k-1]) / 2
    qᶜˡ = (μ.qᶜˡ[1, 1, k] + μ.qᶜˡ[1, 1, k-1]) / 2
    qᵗ = qᵛ + qᶜˡ
    pᶠ = sqrt(p[1, 1, k] * p[1, 1, k-1])
    H = ((1 - qᵗ) * cᵖᵈ + qᵛ * cᵖᵛ + qᶜˡ * cˡ) * Tᶠ - ℒˡ₀ * qᶜˡ
    T₀, qᵛ₀ = parcel_state(H, pᶠ, qᵗ, constants)

    function displaced_density(δ)
        H½ = H + parcel_gas_constant(qᵛ₀, qᵗ, constants) * T₀ * δ / 2
        T½, qᵛ½ = parcel_state(H½, pᶠ * exp(δ / 2), qᵗ, constants)
        T′, qᵛ′ = parcel_state(H + parcel_gas_constant(qᵛ½, qᵗ, constants) * T½ * δ, pᶠ * exp(δ), qᵗ, constants)
        return pᶠ * exp(δ) / (parcel_gas_constant(qᵛ′, qᵗ, constants) * T′)
    end

    parcel = log(displaced_density(δ) / displaced_density(-δ)) / 2δ * log(p[1, 1, k] / p[1, 1, k-1]) / Δh
    column = log(layer_density(k) / layer_density(k-1)) / Δh

    return - g * (column - parcel)
end

#####
##### Static stability
#####

@testset "Static stability in the heights of the hydrostatic column" begin
    @testset "No stratification, no water: N² = 0" begin
        column, header = column_state("dry_subsidence")
        model = closure_column_model(column)
        set_column_state!(model, header["dry_surface_pressure_Pa"], fill(300.0, length(column.k)), zero(column.r_e), zero(column.r_r))
        mixing = diagnose_closure_fields(model)
        N² = column_values(mixing.closure_fields.N²)
        @info "isentropic dry column" largest_N² = maximum(abs, N²)
        @test maximum(abs, N²) < 1e-10
    end

    for name in ("dry_subsidence", "moist_tropical", "midlatitude_frontal")
        @testset "Branches of MoistStaticStability [$name]" begin
            model, column, = turbulent_column(name; rain=false)
            mixing = diagnose_closure_fields(model)
            N² = mixing.closure_fields.N²
            Nz = length(column.k)
            qᶜˡ = column_values(model.microphysical_fields.qᶜˡ)

            reevaluated = [reevaluated_static_stability(model, k) for k in 2:Nz]
            difference = maximum(abs(N²[1, 1, k] - reevaluated[k-1].N²) / max(abs(reevaluated[k-1].N²), 1e-6) for k in 2:Nz)
            saturated_faces = [k for k in 2:Nz if reevaluated[k-1].is_saturated]
            interfaces = [k for k in 2:Nz if (qᶜˡ[k] > 0) != (qᶜˡ[k-1] > 0)]
            @info "N² against its re-evaluation [$name]" difference saturated_faces interfaces
            @test difference ≤ reevaluation_relative

            # Away from the interfaces a face is saturated where its two layers hold cloud
            @test all(reevaluated[k-1].is_saturated == (qᶜˡ[k] > 0 && qᶜˡ[k-1] > 0) for k in 2:Nz if !(k in interfaces))

            if name == "dry_subsidence"
                @test isempty(saturated_faces)
            else
                # Latent heating of the displaced parcel offsets part of the stratification
                @test !isempty(saturated_faces)
                @test all(reevaluated[k-1].saturated < reevaluated[k-1].dry for k in saturated_faces)

                # Cloudy / clear interfaces: one branch or the other by the rule of the face, and finite
                @test !isempty(interfaces)
                @test all(isfinite(N²[1, 1, k]) for k in interfaces)
                @test all(abs(N²[1, 1, k] - reevaluated[k-1].N²) ≤ reevaluation_relative * max(abs(reevaluated[k-1].N²), 1e-6) for k in interfaces)
            end

            # Away from the interfaces, the reversible displacement of a parcel gives the same N²
            oracle_faces = [k for k in 2:Nz if !(k in interfaces)]
            oracle = [parcel_static_stability(model, k) for k in oracle_faces]
            excess = maximum(abs(N²[1, 1, k] - oracle[n]) - parcel_oracle_relative * abs(oracle[n]) for (n, k) in enumerate(oracle_faces))
            worst_unsaturated, worst = findmax([reevaluated[k-1].is_saturated ? 0.0 : abs(N²[1, 1, k] - oracle[n]) for (n, k) in enumerate(oracle_faces)])
            @info "N² against the parcel oracle [$name]" excess_over_relative_bound = excess worst_unsaturated worst_face = oracle_faces[worst] closure = N²[1, 1, oracle_faces[worst]] parcel = oracle[worst]
            @test excess ≤ parcel_oracle_absolute

            if !isempty(saturated_faces)
                worst_saturated = maximum(abs(N²[1, 1, k] - oracle[n]) for (n, k) in enumerate(oracle_faces) if reevaluated[k-1].is_saturated)
                # The dry formula on the saturated faces is far from the parcel: the branch matters
                dry_formula_error = maximum(abs(reevaluated[k-1].dry - oracle[n]) for (n, k) in enumerate(oracle_faces) if reevaluated[k-1].is_saturated)
                @info "saturated faces [$name]" worst_saturated dry_formula_error
                @test dry_formula_error > 10 * parcel_oracle_absolute
            end
        end
    end
end

#####
##### Budgets, positivity
#####

@testset "Closed-column budgets of the mixing" begin
    for name in ("dry_subsidence", "moist_tropical", "moist_tropical_clear", "moist_tropical_heavy_rain", "midlatitude_frontal")
        @testset "Water, energy, dry mass, positivity [$name]" begin
            model, = turbulent_column(name)
            mixing = MoistTurbulentMixing(model)
            E₀ = total_column_energy(model)
            H₀ = column_energy(model)[1, 1]
            W₀ = column_water(model)[1, 1]
            M₀ = column_dry_mass(model)[1, 1]
            πˢ₀ = model.dynamics.surface_pressure[1, 1, 1]
            ρqʳ₀ = column_values(model.microphysical_fields.ρqʳ)
            lowest_water = Inf
            lowest_tke = Inf
            largest_diffusivity = 0.0

            for n in 1:50
                moist_turbulent_mixing_step!(mixing, model, 600)
                lowest_water = min(lowest_water, minimum(model.moisture_density))
                lowest_tke = min(lowest_tke, minimum(model.tracers.ρe))
                largest_diffusivity = max(largest_diffusivity, maximum(mixing.closure_fields.Kᶜ))
            end

            energy = (total_column_energy(model) - E₀) / E₀
            enthalpy_alone = (column_energy(model)[1, 1] - H₀) / H₀
            water = (column_water(model)[1, 1] - W₀) / W₀
            @info "50 steps of 600 s [$name]" energy water enthalpy_alone lowest_water lowest_tke largest_diffusivity

            @test abs(energy) ≤ budget_relative
            @test abs(water) ≤ budget_relative
            @test abs(enthalpy_alone) > energy_control
            @test column_dry_mass(model)[1, 1] == M₀
            @test model.dynamics.surface_pressure[1, 1, 1] == πˢ₀
            @test column_values(model.microphysical_fields.ρqʳ) == ρqʳ₀
            @test lowest_water ≥ 0
            @test lowest_tke ≥ 0
            @test all(isfinite, column_values(model.temperature))
            @test minimum(model.microphysical_fields.qᶜˡ) ≥ 0

            # The ledger holds the conversions, and nothing passed through the ground
            @test ledger_value(mixing, :shear_production) > 0
            @test ledger_value(mixing, :turbulent_kinetic_energy_sinks) > 0
            @test ledger_value(mixing, :surface_vapor) == 0
            @test ledger_value(mixing, :surface_energy) == 0
        end
    end

    @testset "With the dynamics, warm rain and precipitation" begin
        model, = turbulent_column("moist_tropical_heavy_rain")
        mixing = MoistTurbulentMixing(model)
        H₀ = column_energy(model)[1, 1]
        W₀ = column_water(model)[1, 1]

        for n in 1:12
            time_step!(model, 600)
            moist_turbulent_mixing_step!(mixing, model, 600)
        end

        precipitation = model.dynamics.moisture.ledger
        rain = precipitation.surface_rain[1, 1, 1]
        sources = (ledger_value(mixing, :turbulent_kinetic_energy_sinks) - ledger_value(mixing, :buoyancy_source) +
                   ledger_value(mixing, :surface_stress_work) - ledger_value(mixing, :water_kinetic_energy) -
                   precipitation.precipitation_energy[1, 1, 1])

        enthalpy = (column_energy(model)[1, 1] - H₀ - sources) / H₀
        water = (column_water(model)[1, 1] - W₀ + rain) / W₀
        @info "12 steps of dynamics, warm rain and mixing" enthalpy water rain
        @test rain > 1
        @test abs(enthalpy) ≤ budget_relative
        @test abs(water) ≤ budget_relative
        @test minimum(model.moisture_density) ≥ 0
        @test minimum(model.microphysical_fields.ρqʳ) ≥ 0
        @test minimum(model.tracers.ρe) ≥ 0
    end

    @testset "Water that vanishes in every other layer stays non-negative" begin
        column, header = column_state("moist_tropical")
        model = closure_column_model(column)
        Nz = length(column.k)
        rᵉ = [isodd(k) ? 0.0 : column.r_e[k] for k in 1:Nz]
        set_column_state!(model, header["dry_surface_pressure_Pa"], column.theta, rᵉ, zero(column.r_r))
        interior(model.velocities.u) .= reshape([30 * (-1)^k for k in 1:Nz], 1, 1, Nz)
        interior(model.tracers.ρe) .= interior(model.dynamics.moisture.total_density) .* reshape([iseven(k) ? 0.0 : 5.0 for k in 1:Nz], 1, 1, Nz)
        update_state!(model; compute_tendencies=false)
        mixing = MoistTurbulentMixing(model)
        E₀ = total_column_energy(model)
        W₀ = column_water(model)[1, 1]
        @test minimum(model.moisture_density) == 0
        @test minimum(model.tracers.ρe) == 0

        lowest_water = Inf
        lowest_tke = Inf
        largest_diffusivity = 0.0
        for n in 1:20
            moist_turbulent_mixing_step!(mixing, model, 1800)
            lowest_water = min(lowest_water, minimum(model.moisture_density))
            lowest_tke = min(lowest_tke, minimum(model.tracers.ρe))
            largest_diffusivity = max(largest_diffusivity, maximum(mixing.closure_fields.Kᶜ))
        end

        @info "alternating water and TKE, shear of 60 m/s between layers, Δt = 1800 s" lowest_water lowest_tke largest_diffusivity
        @test lowest_water ≥ 0
        @test lowest_tke ≥ 0
        @test largest_diffusivity > 10
        @test abs(total_column_energy(model) - E₀) / E₀ ≤ budget_relative
        @test abs(column_water(model)[1, 1] - W₀) / W₀ ≤ budget_relative
    end

    @testset "No turbulent kinetic energy, no shear: still positive" begin
        model, = turbulent_column("moist_tropical"; e=0)
        interior(model.velocities.u) .= 0
        interior(model.velocities.v) .= 0
        update_state!(model; compute_tendencies=false)
        mixing = MoistTurbulentMixing(model)
        E₀ = total_column_energy(model)

        lowest_tke = Inf
        for n in 1:20
            moist_turbulent_mixing_step!(mixing, model, 600)
            lowest_tke = min(lowest_tke, minimum(model.tracers.ρe))
        end

        @info "from rest" lowest_tke buoyancy_source = ledger_value(mixing, :buoyancy_source)
        @test lowest_tke ≥ 0
        @test minimum(model.moisture_density) ≥ 0
        @test abs(total_column_energy(model) - E₀) / E₀ ≤ budget_relative
    end
end

#####
##### Surface fluxes
#####

@testset "Surface fluxes of the mixing" begin
    @testset "Prescribed fluxes enter the budgets exactly" begin
        model, = turbulent_column("moist_tropical_clear")
        mixing = MoistTurbulentMixing(model)
        cᵖᵛ = model.thermodynamic_constants.vapor.heat_capacity
        set!(mixing.surface_fluxes.vapor_flux, 1e-4)
        set!(mixing.surface_fluxes.sensible_heat_flux, 25)
        set!(mixing.surface_fluxes.surface_temperature, 301)
        set!(mixing.surface_fluxes.x_stress, -0.08)
        set!(mixing.surface_fluxes.y_stress, 0.02)
        E₀ = total_column_energy(model)
        W₀ = column_water(model)[1, 1]

        for n in 1:30
            moist_turbulent_mixing_step!(mixing, model, 600)
        end

        ΔE = total_column_energy(model) - E₀
        ΔW = column_water(model)[1, 1] - W₀
        @test ΔW ≈ 30 * 600 * 1e-4 rtol=1e-12
        @test abs(ΔW - ledger_value(mixing, :surface_vapor)) ≤ budget_relative * W₀
        @test abs(ΔE - ledger_value(mixing, :surface_energy)) ≤ budget_relative * E₀
        @test ledger_value(mixing, :surface_energy) ≈ 30 * 600 * (25 + 1e-4 * cᵖᵛ * 301) rtol=1e-12
        @test ledger_value(mixing, :surface_stress_work) > 0
        @test minimum(model.tracers.ρe) ≥ 0
    end

    @testset "A vapor flux out of the air is limited to the water of the lowest layer" begin
        model, = turbulent_column("dry_subsidence")
        mixing = MoistTurbulentMixing(model)
        set!(mixing.surface_fluxes.vapor_flux, -1)
        set!(mixing.surface_fluxes.surface_temperature, 290)
        E₀ = total_column_energy(model)
        W₀ = column_water(model)[1, 1]

        lowest_water = Inf
        for n in 1:5
            moist_turbulent_mixing_step!(mixing, model, 600)
            lowest_water = min(lowest_water, minimum(model.moisture_density))
        end

        ΔW = column_water(model)[1, 1] - W₀
        @info "limited dew" lowest_water ΔW requested = -3000
        @test lowest_water ≥ 0
        @test -W₀ < ΔW < 0
        @test abs(ΔW - ledger_value(mixing, :surface_vapor)) ≤ budget_relative * W₀
        @test abs(total_column_energy(model) - E₀ - ledger_value(mixing, :surface_energy)) ≤ budget_relative * E₀
    end

    @testset "The bulk boundary conditions of Breeze, evaluated as they are" begin
        model, = turbulent_column("moist_tropical_clear")
        closure = TKEBasedTurbulenceClosure(Float64)
        Tˢ = 301
        drag = Breeze.BulkDrag(coefficient=1.2e-3, surface_temperature=Tˢ)
        sensible_heat = Breeze.BulkSensibleHeatFlux(coefficient=1.1e-3, surface_temperature=Tˢ)
        vapor = Breeze.BulkVaporFlux(coefficient=1.1e-3, surface_temperature=Tˢ)
        exchange = BulkSurfaceExchange(model, closure; drag, sensible_heat, vapor)
        mixing = MoistTurbulentMixing(model; closure, surface_exchange=exchange)

        # What the materialized conditions return on the state, before the step changes it
        constants = model.thermodynamic_constants
        arguments = (model.grid, model.clock, Oceananigans.fields(model),
                     Breeze.AtmosphereModels.dynamics_thermodynamic_fields(model.dynamics))
        evaluate(condition) = Oceananigans.BoundaryConditions.getbc(condition, 1, 1, arguments...)
        ρ = model.dynamics.density[1, 1, 1]
        ρᵗ = model.dynamics.moisture.total_density[1, 1, 1]
        ĉᵖ = (constants.dry_air.heat_capacity * ρ + constants.vapor.heat_capacity * (ρᵗ - ρ)) / ρᵗ
        expected = (x_stress = evaluate(exchange.x_drag), y_stress = evaluate(exchange.y_drag),
                    sensible_heat_flux = ĉᵖ * model.dynamics.exner_function[1, 1, 1] * evaluate(exchange.sensible_heat),
                    vapor_flux = evaluate(exchange.vapor))
        u₁, v₁ = model.velocities.u[1, 1, 1], model.velocities.v[1, 1, 1]
        T₁ = model.temperature[1, 1, 1]

        E₀ = total_column_energy(model)
        W₀ = column_water(model)[1, 1]
        moist_turbulent_mixing_step!(mixing, model, 600)

        fluxes = mixing.surface_fluxes
        @info "bulk fluxes" x_stress = fluxes.x_stress[1, 1, 1] y_stress = fluxes.y_stress[1, 1, 1] sensible = fluxes.sensible_heat_flux[1, 1, 1] vapor = fluxes.vapor_flux[1, 1, 1] T₁
        @test fluxes.x_stress[1, 1, 1] == expected.x_stress
        @test fluxes.y_stress[1, 1, 1] == expected.y_stress
        @test fluxes.sensible_heat_flux[1, 1, 1] == expected.sensible_heat_flux
        @test fluxes.vapor_flux[1, 1, 1] == expected.vapor_flux
        @test fluxes.surface_temperature[1, 1, 1] == Tˢ

        # Signs: the stress opposes the wind, a warmer and wet surface heats and moistens
        @test sign(fluxes.x_stress[1, 1, 1]) == -sign(u₁)
        @test sign(fluxes.y_stress[1, 1, 1]) == -sign(v₁)
        @test T₁ < Tˢ && fluxes.sensible_heat_flux[1, 1, 1] > 0
        @test fluxes.vapor_flux[1, 1, 1] > 0

        for n in 2:36
            moist_turbulent_mixing_step!(mixing, model, 600)
        end

        @test abs(total_column_energy(model) - E₀ - ledger_value(mixing, :surface_energy)) ≤ budget_relative * E₀
        @test abs(column_water(model)[1, 1] - W₀ - ledger_value(mixing, :surface_vapor)) ≤ budget_relative * W₀
        @test ledger_value(mixing, :surface_vapor) > 0
        @test minimum(model.tracers.ρe) ≥ 0

        polynomial = Breeze.BulkDrag(coefficient=Breeze.PolynomialCoefficient(), surface_temperature=Tˢ)
        @test_throws ArgumentError BulkSurfaceExchange(model, closure; drag=polynomial)
    end
end

#####
##### Latent response
#####

@testset "Latent response to a vapor flux" begin
    function moistened(vapor_flux)
        model, = turbulent_column("moist_tropical_clear")
        mixing = MoistTurbulentMixing(model)
        set!(mixing.surface_fluxes.vapor_flux, vapor_flux)
        set!(mixing.surface_fluxes.surface_temperature, 300)
        E₀ = total_column_energy(model)
        W₀ = column_water(model)[1, 1]
        for n in 1:36
            moist_turbulent_mixing_step!(mixing, model, 600)
        end
        return model, mixing, total_column_energy(model) - E₀, column_water(model)[1, 1] - W₀, E₀
    end

    wet, mixing, ΔE, ΔW, E₀ = moistened(3e-4)
    control, _, ΔE_control, ΔW_control, _ = moistened(0)
    constants = wet.thermodynamic_constants
    cᵖᵛ = constants.vapor.heat_capacity
    μ = wet.microphysical_fields

    # The vapor brings its enthalpy cᵖᵛ Tˢ and nothing else: condensation is not a source of energy
    @test abs(ΔE - cᵖᵛ * 300 * ΔW) ≤ budget_relative * E₀
    @test abs(ΔE_control) ≤ budget_relative * E₀
    @test ΔW_control == 0

    # The moistened column makes cloud that the control does not
    cloud = maximum(μ.qᶜˡ)
    control_cloud = maximum(control.microphysical_fields.qᶜˡ)
    @info "cloud after 6 h" cloud control_cloud
    @test cloud > control_cloud + 2.5e-4

    # Where it does, the air is warmer than the same enthalpy without condensation
    cloudy = [k for k in 1:size(wet.grid, 3) if μ.qᶜˡ[1, 1, k] > 2.5e-4]
    function warming(k)
        q = MoistureMassFractions(μ.qᵛ[1, 1, k], μ.qˡ[1, 1, k])
        qᵗ = q.vapor + q.liquid
        H = mixture_heat_capacity(q, constants) * wet.temperature[1, 1, k] - condensate_latent_energy(q, constants)
        return wet.temperature[1, 1, k] - H / ((1 - qᵗ) * constants.dry_air.heat_capacity + qᵗ * cᵖᵛ)
    end
    @info "latent warming of the cloudy layers" cloudy warming = warming.(cloudy)
    @test !isempty(cloudy)
    @test all(warming(k) > condensation_minimum_warming for k in cloudy)

    # and the closure sees a saturated face there
    fields = diagnose_closure_fields(wet).closure_fields
    saturated = [k for k in 2:size(wet.grid, 3) if reevaluated_static_stability(wet, k).is_saturated]
    @test all(abs(fields.N²[1, 1, k] - reevaluated_static_stability(wet, k).N²) ≤
              reevaluation_relative * max(abs(reevaluated_static_stability(wet, k).N²), 1e-6) for k in saturated)
end

#####
##### The dry limit
#####

@testset "Dry limit against the dry VerticalDiffusion" begin
    model, column, header = turbulent_column("dry_subsidence"; water=false)
    mixing = MoistTurbulentMixing(model)
    Nz = length(column.k)
    Δt = 600

    A = vcat(column.A_lower, column.A_upper[end])
    B = vcat(column.B_lower, column.B_upper[end])
    grid = RectilinearGrid(CPU(), Float64; size=Nz, x=0, y=0, z=(0, 1), topology=(Flat, Flat, Bounded))
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B, vertical_diffusion=VerticalDiffusion())
    dry = AtmosphereModel(grid; dynamics, momentum_advection=VectorInvariant(), formulation=moist_formulation())
    set!(dry.dynamics.surface_pressure, header["dry_surface_pressure_Pa"])
    set!(dry)
    ρ = Array(interior(dry.dynamics.density))
    set!(dry; ρθ = ρ .* reshape(column.theta, 1, 1, Nz))
    interior(dry.velocities.u) .= interior(model.velocities.u)
    interior(dry.velocities.v) .= interior(model.velocities.v)
    update_state!(dry; compute_tendencies=false)

    # Without water the moist diagnostics are those of the dry model, bit for bit
    @test column_values(dry.formulation.potential_temperature) == column_values(model.formulation.potential_temperature)
    @test column_values(dry.dynamics.exner_function) == column_values(model.dynamics.exner_function)
    @test column_values(model.dynamics.moisture.density_potential_temperature) == column_values(model.formulation.potential_temperature)

    θ₀ = column_values(model.formulation.potential_temperature)
    moist_turbulent_mixing_step!(mixing, model, Δt)

    # The dry path with the diffusivities that the closure diagnosed
    diffusion = dry.dynamics.vertical_diffusion
    interior(diffusion.diffusivity) .= interior(mixing.closure_fields.Kᶜ)
    interior(diffusion.viscosity) .= interior(mixing.closure_fields.Kᵘ)
    step_vertical_diffusion!(diffusion, dry, Δt)

    @test maximum(mixing.closure_fields.Kᶜ) > 0
    @test Array(interior(diffusion.conductances.momentum)) == Array(interior(mixing.scratch.conductances.momentum))
    @test Array(interior(diffusion.conductances.scalar)) == Array(interior(mixing.scratch.conductances.scalar))
    @test column_values(dry.velocities.u) == column_values(model.velocities.u)
    @test column_values(dry.velocities.v) == column_values(model.velocities.v)

    # The potential temperature solves the dry enthalpy equation with the conductances of the dry
    # path: its flux of θ through a face times cᵖᵈ Πᶠ, and the heating of the turbulent kinetic energy
    constants = model.thermodynamic_constants
    cᵖᵈ = constants.dry_air.heat_capacity
    θ = column_values(model.formulation.potential_temperature)
    Π = column_values(dry.dynamics.exner_function)
    Πᶠ = column_values(dry.dynamics.interface_exner_function)
    C = column_values(diffusion.conductances.scalar)
    S = column_values(mixing.scratch.heating)
    μ = [ρ[1, 1, k] * Δzᶜᶜᶜ(1, 1, k, grid) for k in 1:Nz]

    upward(k) = (2 ≤ k ≤ Nz) ? - C[k] * cᵖᵈ * Πᶠ[k] * (θ[k] - θ[k-1]) : 0.0
    change = [μ[k] * cᵖᵈ * Π[k] * (θ[k] - θ₀[k]) for k in 1:Nz]
    residual = [change[k] - Δt * (upward(k) - upward(k+1)) - S[k] for k in 1:Nz]
    relative_residual = maximum(abs, residual) / maximum(abs, change)

    θᵈʳʸ = column_values(dry.formulation.potential_temperature_density) ./ vec(ρ)
    @info "dry limit" relative_residual largest_change = maximum(abs, θ .- θ₀) difference_from_dry_path = maximum(abs, θ .- θᵈʳʸ)
    @test relative_residual ≤ dry_limit_residual

    # The dry path conserves Σ μ θ and not the energy; the adapter conserves the energy
    @test maximum(abs, θ .- θᵈʳʸ) > 0
end

#####
##### Time step refinement
#####

@testset "Time step refinement of the mixing" begin
    for name in ("dry_subsidence", "midlatitude_frontal")
        function mixed(Δt)
            model, = turbulent_column(name)
            mixing = MoistTurbulentMixing(model)
            for n in 1:round(Int, 3600 / Δt)
                moist_turbulent_mixing_step!(mixing, model, Δt)
            end
            return (θ = column_values(model.formulation.potential_temperature),
                    r = column_values(model.moisture_density) ./ column_values(model.dynamics.density),
                    u = column_values(model.velocities.u))
        end

        oracle = mixed(3600 / 512)
        errors = [map(name -> maximum(abs, mixed(Δt)[name] .- oracle[name]), (:θ, :r, :u)) for Δt in (600, 300, 150, 75)]
        orders = [log2.(errors[n] ./ errors[n+1]) for n in 1:3]
        @info "refinement over one hour [$name]" errors orders

        for variable in 1:3
            @test issorted([errors[n][variable] for n in 1:4]; rev=true)
            @test refinement_order[1] ≤ orders[3][variable] ≤ refinement_order[2]
        end
    end
end

#####
##### Restart
#####

@testset "Restart of the mixing from a checkpoint" begin
    function build()
        model, = turbulent_column("moist_tropical_heavy_rain")
        mixing = MoistTurbulentMixing(model)
        set!(mixing.surface_fluxes.vapor_flux, 1e-4)
        set!(mixing.surface_fluxes.surface_temperature, 301)
        set!(mixing.surface_fluxes.sensible_heat_flux, 20)
        set!(mixing.surface_fluxes.x_stress, -0.05)
        simulation = Simulation(model; Δt=600, stop_time=3600, verbose=false)
        add_callback!(simulation, Callback(mixing, IterationInterval(1)); name=:mixing)
        return simulation, model, mixing
    end

    state(model) = reduce(vcat, map(column_values, values(Oceananigans.prognostic_fields(model))))
    ledger(mixing) = [ledger_value(mixing, name) for name in propertynames(mixing.ledger)]
    precipitation(model) = [getproperty(model.dynamics.moisture.ledger, name)[1, 1, 1] for name in propertynames(model.dynamics.moisture.ledger)]

    continuous, model, mixing = build()
    run!(continuous)

    directory = mktempdir()
    first_half, first_model, first_mixing = build()
    first_half.stop_time = 1800
    first_half.output_writers[:checkpointer] = Checkpointer(first_model; schedule=TimeInterval(1800), dir=directory, prefix="mixing")
    run!(first_half)
    @test ledger_value(first_mixing, :surface_vapor) > 0

    restarted, restarted_model, restarted_mixing = build()
    restarted.output_writers[:checkpointer] = Checkpointer(restarted_model; schedule=TimeInterval(1e9), dir=directory, prefix="mixing")
    run!(restarted; pickup=true)

    @test restarted_model.clock.time == 3600
    @test restarted_model.clock.iteration == model.clock.iteration
    @test state(restarted_model) == state(model)
    @test ledger(restarted_mixing) == ledger(mixing)
    @test precipitation(restarted_model) == precipitation(model)
    @test ledger_value(mixing, :surface_vapor) ≈ 6 * 600 * 1e-4 rtol=1e-12
end

#####
##### Kernels on a staggered grid and in single precision
#####

@testset "Mixing of identical columns on a staggered grid" begin
    model = slice_model(; size=(4, 10), tracers=:ρe)
    set!(model; pˢ=1e5, θ=(x, z) -> 290 + 40z, qᵗ=(x, z) -> 0.015 * (1 - z)^3, u=(x, z) -> 10 * z^0.3)
    ρᵗ = model.dynamics.moisture.total_density
    interior(model.tracers.ρe) .= 0.1 .* interior(ρᵗ)
    update_state!(model; compute_tendencies=false)
    mixing = MoistTurbulentMixing(model)

    Δz = Δzᶜᶜᶜ(1, 1, 1, model.grid)
    kinetic(model) = sum(interior(model.dynamics.moisture.total_density) .* interior(model.velocities.u) .^ 2) * Δz / 2
    energy(model) = sum(column_energy(model)) + kinetic(model) + sum(interior(model.tracers.ρe)) * Δz
    E₀ = energy(model)
    W₀ = sum(column_water(model))

    for n in 1:10
        moist_turbulent_mixing_step!(mixing, model, 600)
    end

    for field in (model.formulation.potential_temperature, model.velocities.u, model.moisture_density, model.tracers.ρe)
        values = Array(interior(field))
        @test all(values[i, 1, :] == values[1, 1, :] for i in 2:4)
    end

    @test abs(energy(model) - E₀) / E₀ ≤ budget_relative
    @test abs(sum(column_water(model)) - W₀) / W₀ ≤ budget_relative
    @test maximum(mixing.closure_fields.Kᶜ) > 0
end

@testset "Mixing in single precision" begin
    column, header = column_state("moist_tropical")
    microphysics = OneMomentCloudMicrophysics(Float32; cloud_formation = SaturationAdjustment(Float32; equilibrium = WarmPhaseEquilibrium()))
    model = closure_column_model(column; FT=Float32, microphysics)
    Nz = length(column.k)
    set_column_state!(model, header["dry_surface_pressure_Pa"], column.theta, column.r_e, column.r_r)
    interior(model.velocities.u) .= reshape([shear_profile(k, Nz) for k in 1:Nz], 1, 1, Nz)
    interior(model.tracers.ρe) .= interior(model.dynamics.moisture.total_density) .* 0.1f0
    update_state!(model; compute_tendencies=false)
    mixing = MoistTurbulentMixing(model; closure = TKEBasedTurbulenceClosure(Float32))
    W₀ = column_water(model)[1, 1]

    for n in 1:10
        moist_turbulent_mixing_step!(mixing, model, 600)
    end

    closure_fields = mixing.closure_fields
    diagnosed = (closure_fields.Kᵘ, closure_fields.Kᶜ, closure_fields.Kᵉ, closure_fields.N², closure_fields.ℓᵐ, closure_fields.Lᵉ)

    for field in (values(Oceananigans.prognostic_fields(model))..., diagnosed..., values(mixing.ledger)...)
        @test eltype(field) == Float32
        @test all(isfinite, Array(interior(field)))
    end

    @test minimum(model.moisture_density) ≥ 0
    @test minimum(model.tracers.ρe) ≥ 0
    @test abs(column_water(model)[1, 1] - W₀) / W₀ ≤ 1e-5
end

@testset "Bulk surface exchange in single precision: no Float64 in the conditions or their fluxes" begin
    column, header = column_state("moist_tropical_clear")
    microphysics = OneMomentCloudMicrophysics(Float32; cloud_formation = SaturationAdjustment(Float32; equilibrium = WarmPhaseEquilibrium()))
    model = closure_column_model(column; FT=Float32, microphysics)
    Nz = length(column.k)
    set_column_state!(model, header["dry_surface_pressure_Pa"], column.theta, column.r_e, column.r_r)
    interior(model.velocities.u) .= reshape([shear_profile(k, Nz) for k in 1:Nz], 1, 1, Nz)
    interior(model.tracers.ρe) .= interior(model.dynamics.moisture.total_density) .* 0.1f0
    update_state!(model; compute_tendencies=false)

    # An integer surface temperature and Float64 coefficients, as a script would write them
    closure = TKEBasedTurbulenceClosure(Float32)
    exchange = BulkSurfaceExchange(model, closure; drag = Breeze.BulkDrag(coefficient=1.2e-3, surface_temperature=301),
                                   sensible_heat = Breeze.BulkSensibleHeatFlux(coefficient=1.1e-3, surface_temperature=301),
                                   vapor = Breeze.BulkVaporFlux(coefficient=1.1e-3, surface_temperature=301))

    arguments = (model.grid, model.clock, Oceananigans.fields(model),
                 Breeze.AtmosphereModels.dynamics_thermodynamic_fields(model.dynamics))

    for condition in (exchange.x_drag, exchange.y_drag, exchange.sensible_heat, exchange.vapor)
        numbers = [getfield(condition, name) for name in fieldnames(typeof(condition)) if getfield(condition, name) isa Number]
        @test !isempty(numbers)
        @test all(number -> number isa Float32, numbers)
        @test Oceananigans.BoundaryConditions.getbc(condition, 1, 1, arguments...) isa Float32
    end

    # The same conditions as the user built them do return a Float64: the conversion is what the test sees
    unconverted = Breeze.BulkVaporFlux(coefficient=1.1e-3, surface_temperature=301)
    materialized = Breeze.BoundaryConditions.materialize_atmosphere_boundary_condition(
        unconverted, Oceananigans.BoundaryConditions.Bottom(), (Center(), Center(), Center()), model.grid, model.dynamics,
        model.microphysics, model.thermodynamic_constants, closure)
    @test Oceananigans.BoundaryConditions.getbc(materialized.condition, 1, 1, arguments...) isa Float64

    mixing = MoistTurbulentMixing(model; closure, surface_exchange=exchange)
    for n in 1:5
        moist_turbulent_mixing_step!(mixing, model, 600)
    end
    @test mixing.surface_fluxes.vapor_flux[1, 1, 1] > 0
    @test all(isfinite, Array(interior(model.temperature)))
    @test minimum(model.moisture_density) ≥ 0
    @test minimum(model.tracers.ρe) ≥ 0
end

#####
##### What the mixing may be built with
#####

@testset "Constructor of the mixing" begin
    column, header = column_state("dry_subsidence")
    without_tracer = moist_column_model(column)
    @test_throws ArgumentError MoistTurbulentMixing(without_tracer)

    model = closure_column_model(column)
    explicit = TKEBasedTurbulenceClosure(Oceananigans.TurbulenceClosures.ExplicitTimeDiscretization(), Float64)
    @test_throws ArgumentError MoistTurbulentMixing(model; closure=explicit)
    @test_throws ArgumentError MoistTurbulentMixing(model; closure=ScalarDiffusivity(ν=1))
    @test MoistTurbulentMixing(model) isa MoistTurbulentMixing
end
