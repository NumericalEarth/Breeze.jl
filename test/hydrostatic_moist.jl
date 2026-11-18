using Breeze
using Oceananigans
using Test

include("hydrostatic_moist_utils.jl")

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Utils: launch!

using Breeze.HydrostaticEquations: HydrostaticEquations, warm_rain_column_physics!, hydrostatic_state_check,
                                   HydrostaticConservationFixer, GridScaleHyperdiffusion, VerticalDiffusion,
                                   VerletSubsteps
using Breeze.Thermodynamics: saturation_vapor_pressure, PlanarLiquidSurface, dry_air_gas_constant, vapor_gas_constant,
                             absolute_zero_latent_heat, TetensFormula
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics
using Breeze.AtmosphereModels: VerticalBorrowing

#####
##### Tests of moist `HydrostaticDynamics`: rows T1 to T10 of the frozen list of the moist derivation
##### (M3_TEST_PLAN.md), with its tolerances, for the first implementation step. Float64 on the CPU.
#####
##### No test uses a negative moisture correction, a clipping or a conservation fixer, and the
##### microphysics has no relaxation of negative rain on this core: the numerical entries of the
##### ledger are zero by construction.
#####

const columns_directory = joinpath(@__DIR__, "hydrostatic_moist_columns")
column_state(name) = read_column(joinpath(columns_directory, name * ".csv"))

const ledger_relative = 1e-12                    # ledgers over O(10²–10³) operator applications
const rest_terrain_relative = 1e-12              # |PGF| / (cᵖᵈ θᵨ)
const rest_dry_formula_control = 1e-6            # the dry formula on the same moist state must exceed this
const adjustment_minimum_warming = 0.5           # K
const precipitation_energy_control = 1e-9        # leaving the precipitation enthalpy out must exceed this
const entropy_drift = 1e-6                       # J kg⁻¹ K⁻¹ per e-folding of pressure
const entropy_drift_control = 1e-2               # without the expansion work the drift must exceed this
const refinement_order = (0.6, 1.6)


#####
##### T1b: a moist-capable model without water is the dry model, bit for bit
#####

ridge(x) = 9.81 * 1500 * exp(-((x - 8e5) / 2e5)^2)

@testset "T1b: dry limit of the moist model is bitwise [$(join(size, "×"))]" for size in ((16, 10), (8, 8, 6))
    function build(microphysics)
        if length(size) == 2
            model = slice_model(; size, Φˢ=ridge, microphysics, tracers=(:c,))
        else
            grid = RectilinearGrid(CPU(), Float64; size, x=(0, 1.6e6), y=(0, 1.6e6), z=(0, 1))
            dynamics = HydrostaticDynamics(surface_geopotential = (x, y) -> ridge(x) * exp(-((y - 8e5) / 3e5)^2),
                                           top_pressure = 2.5e4)
            model = AtmosphereModel(grid; dynamics, microphysics, tracers=(:c,), coriolis=FPlane(f=1e-4),
                                    formulation=moist_formulation(), momentum_advection=VectorInvariant())
        end
        θᵢ(x, z) = 290 + 40z + 2 * sin(2π * x / 1.6e6)
        θᵢ(x, y, z) = θᵢ(x, z) + cos(2π * y / 1.6e6)
        set!(model; pˢ=1e5, θ=θᵢ, u=10)
        set!(model; c = (x...) -> 1 + sin(2π * first(x) / 1.6e6)^2)
        return model
    end

    dry = build(nothing)
    moist = build(warm_rain_microphysics())
    @test moist.dynamics isa HydrostaticEquations.MoistHydrostaticDynamics
    @test isnothing(dry.dynamics.moisture)

    steps = length(size) == 2 ? 50 : 20
    for _ in 1:steps
        time_step!(dry, 60)
        time_step!(moist, 60)
    end

    @test max_abs(dry.velocities.u) > 1 # the flow is not at rest
    for name in (:pˢ, :u, :v, :ρθ, :c)
        a = Array(interior(Oceananigans.prognostic_fields(dry)[name]))
        b = Array(interior(Oceananigans.prognostic_fields(moist)[name]))
        @test a == b
    end
    @test all(iszero, Array(interior(moist.moisture_density)))
    @test all(iszero, Array(interior(moist.microphysical_fields.ρqʳ)))
    @test all(iszero, Array(interior(moist.dynamics.moisture.ledger.surface_rain)))
end

#####
##### T2a, T2b: a resting moist atmosphere of uniform density potential temperature
#####

# Unsaturated air of uniform θᵨ = θᵨ₀ whose humidity varies along x from none to `humidity` of
# saturation, under the total surface pressure for which Φˢ + cᵖᵈ θᵨ₀ Πˢ is uniform. Without
# condensate ϑ = Rᵐ / Rᵈ = 1 + qᵛ (Rᵛ / Rᵈ - 1), so θ = θᵨ₀ / ϑ.
function humid_resting_model(Φˢ; θᵨ₀=300, humidity=0.8)
    model = slice_model(; Φˢ)
    grid = model.grid
    constants = model.thermodynamic_constants
    Rᵈ, Rᵛ = dry_air_gas_constant(constants), vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ
    Nx, _, Nz = size(grid)

    surface_pressure(x) = 1e5 * (1 - Φˢ(x) / (cᵖᵈ * θᵨ₀))^(1 / κ)
    shape = [(1 - cos(2π * (i - 1/2) / Nx)) / 2 for i in 1:Nx]
    qᵛ = zeros(Nx, 1, Nz)
    set!(model; pˢ=surface_pressure, θ=θᵨ₀)

    for _ in 1:4
        T = Array(interior(model.temperature))
        p = Array(interior(model.dynamics.pressure))
        for i in 1:Nx, k in 1:Nz
            pᵛ⁺ = saturation_vapor_pressure(T[i, 1, k], constants, PlanarLiquidSurface())
            qᵛ[i, 1, k] = humidity * shape[i] * (Rᵈ / Rᵛ) * pᵛ⁺ / (p[i, 1, k] - (1 - Rᵈ / Rᵛ) * pᵛ⁺)
        end
        set!(model; pˢ=surface_pressure, θ = θᵨ₀ ./ (1 .+ qᵛ .* (Rᵛ / Rᵈ - 1)), qᵗ=qᵛ)
    end

    return model, θᵨ₀
end

# The largest pressure-gradient acceleration of the dry formulas on the state of `model`:
# θ in place of θᵨ in the hydrostatic integral and in the force
function dry_formula_acceleration(model)
    grid = model.grid
    dynamics = model.dynamics
    θ = model.formulation.potential_temperature
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    Φ = CenterField(grid)
    launch!(CPU(), grid, :xy, HydrostaticEquations._compute_geopotential!, Φ, grid, dynamics.surface_geopotential,
            θ, dynamics.exner_function, dynamics.interface_exner_function, cᵖᵈ)
    fill_halo_regions!(Φ)
    Nx, _, Nz = size(grid)
    return maximum(abs(HydrostaticEquations.x_pressure_gradient_acceleration(i, 1, k, grid, θ, dynamics.exner_function, Φ, cᵖᵈ))
                   for i in 1:Nx, k in 1:Nz)
end

@testset "T2: moist rest of uniform θᵨ [$name]" for (name, Φˢ) in (("T2a flat", x -> 0.0), ("T2b 1500 m ridge", ridge))
    model, θᵨ₀ = humid_resting_model(Φˢ)
    grid = model.grid
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
    Δx = 1.6e6 / size(grid, 1)
    scale = Δx / (cᵖᵈ * θᵨ₀)

    qᵛ = Array(interior(model.microphysical_fields.qᵛ))
    @test maximum(qᵛ) > 5e-3                                           # humid
    @test minimum(qᵛ[1, 1, :]) < 1e-3 * maximum(qᵛ)                    # and dry elsewhere
    @test all(iszero, Array(interior(model.microphysical_fields.qᶜˡ))) # unsaturated
    θᵨ = Array(interior(model.dynamics.moisture.density_potential_temperature))
    @test maximum(abs, θᵨ .- θᵨ₀) < 1e-10

    update_state!(model)
    acceleration = max_abs(model.timestepper.Gⁿ.u) * scale
    control = dry_formula_acceleration(model) * scale
    @info "$name: |∂ₜu| Δx / (cᵖᵈ θᵨ) = $acceleration; dry formulas on the same state = $control"
    @test acceleration ≤ rest_terrain_relative
    @test control > rest_dry_formula_control

    # The tolerance of the dry isentropic-rest test: 100 steps of 30 s, then to one day
    for _ in 1:100
        time_step!(model, 30)
    end
    @test max_abs(model.velocities.u) < 1e-10

    history = [max_abs(model.velocities.u)]
    for n in 1:(86400 - 3000) ÷ 120
        time_step!(model, 120)
        n % 175 == 0 && push!(history, max_abs(model.velocities.u))
    end
    @info "$name: max |u| at 50 min and every 5 h 50 min after: $history"
    # Recorded beside it, not gated: the dry model at rest with θ = θᵨ₀ on the same grid and steps
    dry = slice_model(; Φˢ, microphysics=nothing)
    set!(dry; pˢ = x -> 1e5 * (1 - Φˢ(x) / (cᵖᵈ * θᵨ₀))^(cᵖᵈ / dry_air_gas_constant(dry.thermodynamic_constants)), θ=θᵨ₀)
    for _ in 1:100
        time_step!(dry, 30)
    end
    for _ in 1:(86400 - 3000) ÷ 120
        time_step!(dry, 120)
    end

    @info "$name: max |u| after one day = $(max_abs(model.velocities.u)) m s⁻¹; dry model = $(max_abs(dry.velocities.u)) m s⁻¹"
    @test max_abs(model.velocities.u) < 1e-10
end

#####
##### T3: saturation adjustment and rain formation in a closed column
#####

@testset "T3: saturation adjustment ledger" begin
    column, header = column_state("midlatitude_frontal")
    model = moist_column_model(column)
    constants = model.thermodynamic_constants
    set_column_state!(model, header["dry_surface_pressure_Pa"], column.theta, column.r_e, column.r_r)

    water₀, energy₀ = column_water(model)[1], column_energy(model)[1]
    rain₀ = sum(Array(interior(model.microphysical_fields.ρqʳ)))

    for _ in 1:200
        warm_rain_column_physics!(model.microphysics, model; Δt=300, sedimentation=false)
    end

    water, energy = column_water(model)[1], column_energy(model)[1]
    @info "T3: water $((water - water₀) / water₀), energy $((energy - energy₀) / energy₀) relative, over 200 steps"
    @test abs(water - water₀) ≤ ledger_relative * water₀
    @test abs(energy - energy₀) ≤ ledger_relative * abs(energy₀)
    @test sum(Array(interior(model.microphysical_fields.ρqʳ))) != rain₀ # the sources acted
    @test minimum(model.microphysical_fields.ρqʳ) ≥ 0
    @test minimum(model.moisture_density) ≥ 0

    # Control: the latent heat is seen. Vapor 20% above saturation at the temperature T⁰ that the
    # layer would have if all its water were vapor (T⁰ = Π θ) must warm by more than 0.5 K.
    set_column_state!(model, header["dry_surface_pressure_Pa"], column.theta, column.r_e, 0 .* column.r_r)
    Π = Array(interior(model.dynamics.exner_function))[1, 1, :]
    p = Array(interior(model.dynamics.pressure))[1, 1, :]
    T⁰ = Π .* column.theta
    warm = findall(>(273.15), T⁰)
    ϵ = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    r⁺ = [ϵ * saturation_vapor_pressure(T⁰[k], constants, PlanarLiquidSurface()) /
          (p[k] - saturation_vapor_pressure(T⁰[k], constants, PlanarLiquidSurface())) for k in eachindex(T⁰)]
    rᵉ = copy(column.r_e)
    rᵉ[warm] .= 1.2 .* r⁺[warm]
    set_column_state!(model, header["dry_surface_pressure_Pa"], column.theta, rᵉ, 0 .* column.r_r)
    T = Array(interior(model.temperature))[1, 1, :]
    Π = Array(interior(model.dynamics.exner_function))[1, 1, :]
    warming = minimum(T[warm] .- Π[warm] .* column.theta[warm])
    @info "T3 control: least warming of a 20% supersaturated layer = $warming K"
    @test warming > adjustment_minimum_warming
end

#####
##### T4a: sources and sedimentation over terrain. Surface evaporation and sensible heat are not
##### part of this step (the boundary fluxes of water are still rejected), so E = 0 here.
#####

@testset "T4a: warm rain with sedimentation, ledgers" begin
    column, header = column_state("moist_tropical")
    Φˢ = 5000
    model = moist_column_model(column; Φˢ)
    g = model.thermodynamic_constants.gravitational_acceleration
    set_column_state!(model, header["dry_surface_pressure_Pa"], column.theta, column.r_e, column.r_r)
    ledger = model.dynamics.moisture.ledger

    water₀, energy₀, dry₀ = column_water(model)[1], column_energy(model)[1], column_dry_mass(model)[1]
    pˢ₀ = model.dynamics.moisture.total_surface_pressure[1, 1, 1]
    πˢ₀ = model.dynamics.surface_pressure[1, 1, 1]

    for _ in 1:144
        time_step!(model, 300)
    end

    water, energy = column_water(model)[1], column_energy(model)[1]
    pˢ = model.dynamics.moisture.total_surface_pressure[1, 1, 1]
    rain = ledger.surface_rain[1, 1, 1]
    rain_energy = ledger.precipitation_energy[1, 1, 1]

    water_residual = (water - water₀ + rain) / water₀
    pressure_residual = (pˢ - pˢ₀ + g * rain) / pˢ₀
    energy_residual = (energy - energy₀ + rain_energy) / abs(energy₀)
    control = (energy - energy₀) / abs(energy₀)
    @info "T4a: rain $rain kg m⁻²; residuals: water $water_residual, pˢ $pressure_residual, energy $energy_residual; " *
          "energy without the precipitation enthalpy $control; frictional heating " *
          "$(ledger.rain_frictional_heating[1, 1, 1]) J m⁻²; largest Courant number " *
          "$(ledger.maximum_sedimentation_courant[1, 1, 1])"

    @test rain > 1
    @test abs(water_residual) ≤ ledger_relative
    @test abs(pressure_residual) ≤ ledger_relative
    @test abs(energy_residual) ≤ ledger_relative
    @test abs(control) > precipitation_energy_control
    @test column_dry_mass(model)[1] == dry₀
    @test model.dynamics.surface_pressure[1, 1, 1] == πˢ₀
    @test minimum(model.microphysical_fields.ρqʳ) ≥ 0
    @test minimum(model.moisture_density) ≥ 0
    @test ledger.rain_frictional_heating[1, 1, 1] > 0
end

#####
##### T4d, T4e: implicit sedimentation alone, far beyond the explicit Courant limit
#####

function sediment_heavy_rain(Δt, steps)
    column, header = column_state("moist_tropical_heavy_rain")
    model = moist_column_model(column)
    set_column_state!(model, header["dry_surface_pressure_Pa"], column.theta, column.r_e, column.r_r)
    water₀, energy₀ = column_water(model)[1], column_energy(model)[1]
    lowest = (Inf, Inf)

    for _ in 1:steps
        warm_rain_column_physics!(model.microphysics, model; Δt, sources=false)
        lowest = min.(lowest, (minimum(model.microphysical_fields.ρqʳ), minimum(model.moisture_density)))
    end

    ledger = model.dynamics.moisture.ledger
    water_residual = (column_water(model)[1] - water₀ + ledger.surface_rain[1, 1, 1]) / water₀
    energy_residual = (column_energy(model)[1] - energy₀ + ledger.precipitation_energy[1, 1, 1]) / abs(energy₀)
    return (; model, column, lowest, water_residual, energy_residual,
              courant = ledger.maximum_sedimentation_courant[1, 1, 1])
end

@testset "T4d: implicit sedimentation, positivity and ledgers" begin
    result = sediment_heavy_rain(1800, 8)
    column = result.column

    # The Courant number of the label spacing is a different number from the physical one
    label_courant = maximum(column.rain_fall_speed .* 1800 .* length(column.k))
    physical_courant = maximum(column.rain_fall_speed .* 1800 ./ column.dz)
    @info "T4d: lowest ρqʳ, ρqᵉ = $(result.lowest); water $(result.water_residual), energy $(result.energy_residual); " *
          "largest Courant number V Δt / Δz: physical $(result.courant) (initial state $physical_courant), " *
          "with the label spacing $label_courant"

    @test all(result.lowest .≥ 0)
    @test abs(result.water_residual) ≤ ledger_relative
    @test abs(result.energy_residual) ≤ ledger_relative
    @test result.courant > 10                       # far beyond what an explicit step allows
    @test result.courant ≈ physical_courant rtol=1e-6
    @test abs(label_courant / physical_courant - 1) > 0.5
end

#####
##### T7d: expansion work. A column whose layers are material and whose surface pressure is
##### prescribed: the moist entropy of every layer is conserved by θ' = 𝒲 (ln p)'.
#####

function moist_entropy(T, p, qᵛ, qˡ, constants)
    Rᵈ, Rᵛ = dry_air_gas_constant(constants), vapor_gas_constant(constants)
    cᵖᵈ, cᵖᵛ, cˡ = constants.dry_air.heat_capacity, constants.vapor.heat_capacity, constants.liquid.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    Tᵗʳ, pᵗʳ = constants.triple_point_temperature, constants.triple_point_pressure
    sˡ₀ = - Rᵛ * log(pᵗʳ) + (cᵖᵛ - cˡ) * log(Tᵗʳ) - ℒˡ₀ / Tᵗʳ - (cᵖᵛ - cˡ)
    rᵛ, rˡ = qᵛ / (1 - qᵛ - qˡ), qˡ / (1 - qᵛ - qˡ)
    pᵈ = p / (1 + rᵛ * Rᵛ / Rᵈ)
    return cᵖᵈ * log(T) - Rᵈ * log(pᵈ) + rᵛ * (cᵖᵛ * log(T) - Rᵛ * log(p - pᵈ)) + rˡ * (cˡ * log(T) + sˡ₀)
end

function layer_entropies(model)
    μ = model.microphysical_fields
    return [moist_entropy(model.temperature[1, 1, k], model.dynamics.pressure[1, 1, k], μ.qᵛ[1, 1, k], μ.qˡ[1, 1, k],
                          model.thermodynamic_constants) for k in 1:size(model.grid, 3)]
end

# dθ/dπˢ of every layer from the expansion work of the model, for mixing ratios held fixed
function potential_temperature_response(model, πˢ, θ, rᵉ, rʳ)
    set_column_state!(model, πˢ, θ, rᵉ, rʳ)
    grid = model.grid
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    g = constants.gravitational_acceleration
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    μ = model.microphysical_fields
    Nz = size(grid, 3)
    B = dynamics.surface_pressure_coefficient

    Gρθ, Gρqᵉ, Gρqʳ = CenterField(grid), CenterField(grid), CenterField(grid)
    Gπˢ = Field{Center, Center, Nothing}(grid)
    set!(Gπˢ, 1)
    for k in 1:Nz # ρ = Δπ / (g Δz), so ∂ρ/∂πˢ = ΔB / (g Δz)
        ∂ρ = (B[1, 1, k] - B[1, 1, k+1]) / (g * Δzᶜᶜᶜ(1, 1, k, grid))
        Gρqᵉ[1, 1, k] = rᵉ[k] * ∂ρ
        Gρqʳ[1, 1, k] = rʳ[k] * ∂ρ
    end

    launch!(CPU(), grid, :xy, HydrostaticEquations._add_expansion_work!, Gρθ, grid, dynamics.density,
            model.velocities.u, model.velocities.v, dynamics.vertical_mass_flux, dynamics.exner_function,
            dynamics.interface_exner_function, dynamics.moisture.interface_pressure, B, Gπˢ, (Gρqᵉ, Gρqʳ),
            model.formulation.potential_temperature, model.temperature, μ.qᵛ, μ.qˡ, constants, κ, g)

    return [Gρθ[1, 1, k] / dynamics.density[1, 1, k] for k in 1:Nz]
end

@testset "T7d: expansion work conserves moist entropy" begin
    column, header = column_state("midlatitude_frontal")
    model = moist_column_model(column)
    rᵉ, rʳ = column.r_e, 0 .* column.r_r
    πˢ₀, πˢ₁ = header["dry_surface_pressure_Pa"], 0.7e5
    steps = 400 # the tolerance is that of the limit of small steps
    δ = (πˢ₁ - πˢ₀) / steps

    set_column_state!(model, πˢ₀, column.theta, rᵉ, rʳ)
    s₀ = layer_entropies(model)
    p₀ = Array(interior(model.dynamics.pressure))[1, 1, :]
    cloudy₀ = Array(interior(model.microphysical_fields.qᶜˡ))[1, 1, :] .> 0

    θ = copy(column.theta)
    for n in 0:steps-1 # fourth-order Runge-Kutta in πˢ
        πˢ = πˢ₀ + n * δ
        k₁ = potential_temperature_response(model, πˢ, θ, rᵉ, rʳ)
        k₂ = potential_temperature_response(model, πˢ + δ / 2, θ .+ δ / 2 .* k₁, rᵉ, rʳ)
        k₃ = potential_temperature_response(model, πˢ + δ / 2, θ .+ δ / 2 .* k₂, rᵉ, rʳ)
        k₄ = potential_temperature_response(model, πˢ + δ, θ .+ δ .* k₃, rᵉ, rʳ)
        θ .+= δ / 6 .* (k₁ .+ 2 .* k₂ .+ 2 .* k₃ .+ k₄)
    end

    set_column_state!(model, πˢ₁, θ, rᵉ, rʳ)
    s = layer_entropies(model)
    p = Array(interior(model.dynamics.pressure))[1, 1, :]
    cloudy₁ = Array(interior(model.microphysical_fields.qᶜˡ))[1, 1, :] .> 0
    set_column_state!(model, πˢ₁, column.theta, rᵉ, rʳ)
    s_without = layer_entropies(model)

    folds = abs.(log.(p ./ p₀))
    lifted = folds .> 0.1
    drift = maximum(abs.(s .- s₀)[lifted] ./ folds[lifted])
    clear_control = maximum((abs.(s_without .- s₀) ./ folds)[lifted .& .!cloudy₀])
    cloudy_control = maximum((abs.(s_without .- s₀) ./ folds)[lifted .& cloudy₀])
    # A layer that becomes cloudy on the way has a kink in its tendency, which the Runge-Kutta
    # steps resolve only at first order: its drift is the largest and falls with the step
    smooth = lifted .& (cloudy₀ .== cloudy₁)
    smooth_drift = maximum((abs.(s .- s₀) ./ folds)[smooth])
    @info "T7d: entropy drift per e-folding of pressure $drift J kg⁻¹ K⁻¹ over $(count(lifted)) layers " *
          "($(count(lifted .& cloudy₀)) cloudy at the start, $(count(lifted .& .!cloudy₀ .& cloudy₁)) becoming cloudy); " *
          "layers that do not change phase regime $smooth_drift; without the expansion work: clear " *
          "$clear_control, cloudy $cloudy_control"

    @test count(lifted .& cloudy₀) > 0 && count(lifted .& .!cloudy₀) > 0
    @test drift ≤ entropy_drift
    @test smooth_drift ≤ 1e-9
    @test clear_control > entropy_drift_control
    @test cloudy_control > entropy_drift_control
end

#####
##### T6a (without seams): constancy of a uniform mixing ratio of active water and of a tracer
#####

@testset "Constancy of uniform mixing ratios in a moving moist flow" begin
    # Warm and shallow enough to stay unsaturated: nothing condenses or falls, so the mixing
    # ratio of the water changes by transport alone
    model = slice_model(; Φˢ=ridge, tracers=(:c,), top_pressure=7e4)
    set!(model; pˢ=1e5, θ = (x, z) -> 300 + 10z, u=10)
    ρ = Array(interior(model.dynamics.density))
    r₀, χ₀ = 3e-3, 2.0
    set!(model; ρqᵉ = r₀ .* ρ)
    ρ = Array(interior(model.dynamics.density))
    interior(model.tracers.c) .= χ₀ .* ρ
    set!(model; ρqᵉ = r₀ .* ρ)

    θᵨ = Array(interior(model.dynamics.moisture.density_potential_temperature))
    θ = Array(interior(model.formulation.potential_temperature))
    @test maximum(abs, θᵨ ./ θ .- 1) > 1e-3 # the water is active

    for _ in 1:200
        time_step!(model, 60)
    end

    ρ = Array(interior(model.dynamics.density))
    water = Array(interior(model.moisture_density)) .+ Array(interior(model.microphysical_fields.ρqʳ))
    vapor_deviation = maximum(abs, water ./ ρ ./ r₀ .- 1)
    tracer_deviation = maximum(abs, Array(interior(model.tracers.c)) ./ ρ ./ χ₀ .- 1)
    @info "Constancy after 200 steps: water $vapor_deviation, tracer $tracer_deviation; max |u| = $(max_abs(model.velocities.u))"
    @test max_abs(model.velocities.u) > 5
    @test all(iszero, Array(interior(model.microphysical_fields.ρqʳ))) # no rain formed
    @test vapor_deviation ≤ 1e-12
    @test tracer_deviation ≤ 1e-12
end

#####
##### T7a (small): refinement of the time step against the same equations at a small step
#####

function moist_wave(Δt; stop_time=3600)
    model = slice_model(; size=(16, 12), Φˢ=ridge)
    set!(model; pˢ=1e5, θ = (x, z) -> 288 + 45z + 2 * exp(-((x - 4e5) / 2e5)^2), u=8,
                qᵗ = (x, z) -> 0.014 * exp(-5z), qʳ = (x, z) -> 5e-4 * exp(-((z - 0.3) / 0.15)^2))
    for _ in 1:round(Int, stop_time / Δt)
        time_step!(model, Δt)
    end
    return (T = Array(interior(model.temperature)), ρqᵉ = Array(interior(model.moisture_density)),
            πˢ = Array(interior(model.dynamics.surface_pressure)),
            rain = Array(interior(model.dynamics.moisture.ledger.surface_rain)),
            cloud = Array(interior(model.microphysical_fields.qᶜˡ)))
end

l2(a, b) = sqrt(sum(abs2, a .- b) / length(a))

@testset "T7a (small): Δt refinement against a small-step oracle" begin
    oracle = moist_wave(1.25)
    time_steps = (80, 40, 20, 10)
    runs = map(moist_wave, time_steps)
    @test maximum(oracle.rain) > 0.1   # it rains
    @test maximum(oracle.cloud) > 1e-5 # and there is cloud

    for name in (:T, :ρqᵉ, :πˢ)
        errors = [l2(run[name], oracle[name]) for run in runs]
        order = log2(errors[end-1] / errors[end])
        @info "T7a (small): L2 error of $name at Δt = $time_steps s: $errors; order of the finest pair $order"
        @test issorted(errors; rev=true)
        @test refinement_order[1] ≤ order ≤ refinement_order[2]
    end
end

#####
##### T10: unsupported combinations are rejected by name
#####

@testset "T10: guards of moist HydrostaticDynamics" begin
    grid = RectilinearGrid(CPU(), Float64; size=(8, 4), x=(0, 8e5), z=(0, 1), topology=(Periodic, Flat, Bounded))
    build(; dynamics=HydrostaticDynamics(top_pressure=2.5e4), formulation=moist_formulation(),
            microphysics=warm_rain_microphysics(), kw...) =
        AtmosphereModel(grid; dynamics, formulation, microphysics, momentum_advection=VectorInvariant(), kw...)

    @test build() isa AtmosphereModel

    # Microphysics that has not been adapted
    warm_adjustment = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
    @test_throws "has not been adapted" build(microphysics=warm_adjustment)
    @test_throws "has not been adapted" build(microphysics=OneMomentCloudMicrophysics())
    @test_throws "has not been adapted" build(microphysics=OneMomentCloudMicrophysics(cloud_formation=SaturationAdjustment()))
    @test_throws ArgumentError build(microphysics=DCMIP2016KesslerMicrophysics())

    # Thermodynamics
    @test_throws "dry Exner exponent" build(formulation=LiquidIcePotentialTemperatureFormulation())
    @test_throws ArgumentError build(formulation=:StaticEnergy)
    @test_throws "ClausiusClapeyron" build(thermodynamic_constants=ThermodynamicConstants(saturation_vapor_pressure=TetensFormula()))

    # Sources of water and of energy through forcings and boundaries
    @test_throws "forcing of ρqᵉ" build(forcing=(; ρqᵉ = (x, z, t) -> 1e-9))
    @test_throws "forcing of ρqʳ" build(forcing=(; ρqʳ = (x, z, t) -> 1e-9))
    @test_throws "boundary condition of the moisture density" build(
        boundary_conditions=(; ρqᵉ = FieldBoundaryConditions(bottom=FluxBoundaryCondition(1e-5))))
    @test_throws "source of energy" build(boundary_conditions=(; ρθ = FieldBoundaryConditions(bottom=FluxBoundaryCondition(1e-2))))
    @test_throws "source of energy" build(boundary_conditions=(; ρE = FieldBoundaryConditions(bottom=FluxBoundaryCondition(10))))

    # What is not derived or not verified with water
    @test_throws "negative_moisture_correction" build(microphysics=warm_rain_microphysics(negative_moisture_correction=VerticalBorrowing()))
    @test_throws "turbulence closures" build(closure=ScalarDiffusivity(ν=1))
    @test_throws "hyperdiffusion with moisture" build(dynamics=HydrostaticDynamics(top_pressure=2.5e4, hyperdiffusion=GridScaleHyperdiffusion()))
    @test_throws "vertical_diffusion with moisture" build(dynamics=HydrostaticDynamics(top_pressure=2.5e4, vertical_diffusion=VerticalDiffusion()))
    @test_throws "with moisture" build(timestepper=:GravityWaveMultirateInfinitesimalStep,
                                       timestepper_kwargs=(tableau=:MIS3C, inner_solver=VerletSubsteps()))

    model = build()
    @test_throws "hydrostatic_state_check with moisture" hydrostatic_state_check(model)
    @test_throws "HydrostaticConservationFixer with moisture" HydrostaticConservationFixer(model)
    @test_throws ArgumentError set!(model; ρu=1)

    # The dry model keeps its guard: water set in a model without microphysics is rejected
    dry = build(microphysics=nothing)
    @test_throws "moisture is not supported yet" set!(dry; θ=300, qᵗ=1e-3)
end
