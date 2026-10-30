include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.Thermodynamics: Thermodynamics as TD
using Breeze.PotentialTemperatureFormulations: PotentialTemperatureFormulations as PTF
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.TimeSteppers: update_state!
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

column_values(f) = Float64.(vec(Array(interior(f))))

# Internal energy per unit mass with Breeze's latent heats: 𝓊 = cᵛᵐ T + ℒˡ₀ qᵛ + (ℒˡ₀ − ℒⁱ₀) qⁱ
function internal_energy(T, q, constants)
    ℒˡ₀ = TD.absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = TD.absolute_zero_latent_heat(constants, constants.ice)
    cᵛᵐ = TD.mixture_heat_capacity(q, constants) - TD.mixture_gas_constant(q, constants)
    return cᵛᵐ * T + ℒˡ₀ * q.vapor + (ℒˡ₀ - ℒⁱ₀) * q.ice
end

# Heating per unit volume that a ρθ increment ΔΘ represents through the heating path
# ΔΘ = (ρᵈ / ρ) Q Δt / (ĉᵖ Π) = (ρᵈ / ρ) (θ / H) Q Δt
function heating_from_theta(model, ΔΘ)
    constants = model.thermodynamic_constants
    values(f) = Float64.(Array(interior(f)))  # reference fields are reduced in x and y; broadcast them
    ρ = values(AM.total_density(model.dynamics))
    ρᵈ = values(AM.dynamics_density(model.dynamics))
    θ = values(model.formulation.potential_temperature)
    T = values(model.temperature)
    qᵛ = values(AM.specific_humidity(model))
    qˡ_field = AM.liquid_mass_fraction(model)
    qˡ = isnothing(qˡ_field) ? zero(T) : values(qˡ_field)
    ℒˡ₀ = TD.absolute_zero_latent_heat(constants, constants.liquid)
    H = TD.mixture_heat_capacity.(TD.MoistureMassFractions.(qᵛ, qˡ, 0.0), Ref(constants)) .* T .- ℒˡ₀ .* qˡ
    return vec(ρ ./ ρᵈ .* H ./ θ) .* ΔΘ
end

@testset "Energy-closing source options" begin
    default = LiquidIcePotentialTemperatureFormulation()
    @test isnothing(default.precipitation_heating)
    @test isnothing(default.fixed_volume_phase_change)
    @test isnothing(default.dissipative_heating)

    all_on = LiquidIcePotentialTemperatureFormulation(precipitation_heating = true,
                                                      fixed_volume_phase_change = true,
                                                      dissipative_heating = true)
    @test all_on.precipitation_heating isa AM.PrecipitationDragHeating
    @test all_on.fixed_volume_phase_change isa AM.FixedVolumePhaseChange
    @test all_on.dissipative_heating isa AM.SubgridDissipativeHeating
    @test contains(sprint(show, all_on), "precipitation_heating: PrecipitationDragHeating()")
    @test contains(sprint(show, default), "dissipative_heating: false")
    @test LiquidIcePotentialTemperatureFormulation(precipitation_heating = AM.PrecipitationDragHeating()).precipitation_heating isa
          AM.PrecipitationDragHeating
    @test_throws ArgumentError LiquidIcePotentialTemperatureFormulation(precipitation_heating = :yes)
    @test_throws ArgumentError LiquidIcePotentialTemperatureFormulation(dissipative_heating = AM.PrecipitationDragHeating())

    # The fixed-volume source applies on the compressible core only
    grid = RectilinearGrid(default_arch; size=(1, 1, 4), x=(0, 100), y=(0, 100), z=(0, 1000))
    anelastic = AtmosphereModel(grid; formulation = all_on)
    @test isnothing(anelastic.formulation.fixed_volume_phase_change)
    @test anelastic.formulation.precipitation_heating isa AM.PrecipitationDragHeating
    @test anelastic.formulation.dissipative_heating isa AM.SubgridDissipativeHeating

    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=300)
    compressible = AtmosphereModel(grid; dynamics, formulation = all_on)
    @test compressible.formulation.fixed_volume_phase_change isa AM.FixedVolumePhaseChange

    # Saturation adjustment has no vapor-density tendency to correct
    equilibrium = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    model = AtmosphereModel(grid; dynamics, formulation = all_on, microphysics = equilibrium)
    @test_throws ArgumentError begin
        set!(model; ρ = model.dynamics.reference_state.density, θ = 300, qᵗ = 0.01)
        AM.compute_tendencies!(model)
    end
end

#####
##### Fixed-volume phase change: microphysics alone conserves ρ𝓊 in a closed box
#####

function closed_box_internal_energy_change(fixed_volume_phase_change; κ̂ = VaporEquivalentExponent())
    FT = Float64
    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 1e3), y=(0, 1e3), z=(0, 1e3))
    constants = ThermodynamicConstants(FT)
    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation,
                                              precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 7e4,
                                    reference_potential_temperature = z -> 290.0)
    formulation = LiquidIcePotentialTemperatureFormulation(; exner_exponent = κ̂, fixed_volume_phase_change)
    model = AtmosphereModel(grid; dynamics, formulation, microphysics, thermodynamic_constants = constants)

    # A supersaturated, mixed-phase state below freezing: condensation, deposition, autoconversion,
    # accretion, riming and the Wegener–Bergeron–Findeisen process all act.
    ρ₀ = 0.9
    T₀ = 265.0
    qᵛ⁺ = TD.saturation_vapor_pressure(T₀, constants, TD.PlanarLiquidSurface()) / (TD.vapor_gas_constant(constants) * T₀ * ρ₀)
    set!(model; T = T₀, qᵛ = 1.1qᵛ⁺, qᶜˡ = 1e-3, qʳ = 5e-4, qᶜⁱ = 2e-4, qˢⁿ = 3e-4, ρ = ρ₀)

    function state(model)
        ρ = first(column_values(AM.total_density(model.dynamics)))
        T = first(column_values(model.temperature))
        μ = model.microphysical_fields
        ρqᵛ = first(column_values(model.moisture_density))
        ρqˡ = first(column_values(μ.ρqᶜˡ)) + first(column_values(μ.ρqʳ))
        ρqⁱ = first(column_values(μ.ρqᶜⁱ)) + first(column_values(μ.ρqˢⁿ))
        q = TD.MoistureMassFractions(ρqᵛ / ρ, ρqˡ / ρ, ρqⁱ / ρ)
        return (; ρ𝓊 = ρ * internal_energy(T, q, constants), q)
    end

    s₀ = state(model)
    for _ in 1:300
        time_step!(model, 1)
    end
    s₁ = state(model)
    return (s₁.ρ𝓊 - s₀.ρ𝓊) / s₀.ρ𝓊, s₁.q.ice - s₀.q.ice
end

# The Jacobian test below covers both exponents; the closed box runs the default one only
@testset "Fixed-volume phase change conserves internal energy [$κ̂]" for κ̂ in (VaporEquivalentExponent(),)
    δ_off, Δqⁱ = closed_box_internal_energy_change(false; κ̂)
    δ_on, _ = closed_box_internal_energy_change(true; κ̂)
    @info "Closed box: relative change of ∫ρ𝓊 over 300 s" κ̂ δ_off δ_on
    @test Δqⁱ > 1e-3          # substantial phase change happened
    @test abs(δ_off) > 1e-5   # phase change at fixed θˡⁱ does not conserve ρ𝓊 at fixed volume
    @test abs(δ_on) < 1e-7
end

@testset "Fixed-volume weights match the Jacobians of ρ𝓊 [$κ̂]" for κ̂ in (VaporEquivalentExponent(), DryExponent())
    # ρ𝓊(ρᵈ, Θ, ρqᵛ, ρqˡ, ρqⁱ) at fixed volume, through the Newton inversion of θˡⁱ = Θ / ρᵈ at p = ρ Rᵐ T
    constants = ThermodynamicConstants(Float64)
    pˢᵗ = 1e5
    function temperature_and_fractions(ψ)
        ρᵈ, Θ, ρqᵛ, ρqˡ, ρqⁱ = ψ
        ρ = ρᵈ + ρqᵛ + ρqˡ + ρqⁱ
        q = TD.MoistureMassFractions(ρqᵛ / ρ, ρqˡ / ρ, ρqⁱ / ρ)
        solver = NewtonSolver(Float64; abstol = 1e-12, maxiter = 50)
        𝒰 = TD.LiquidIceDensityState(Θ / ρᵈ, q, pˢᵗ, ρ, solver, κ̂)
        return TD.temperature(𝒰, constants), q, ρ
    end
    function internal_energy_density(ψ)
        T, q, ρ = temperature_and_fractions(ψ)
        return ρ * internal_energy(T, q, constants)
    end

    ρᵈ, T₀ = 0.8, 262.0
    q₀ = TD.MoistureMassFractions(0.004, 0.001, 0.0006)
    ρ = ρᵈ / (1 - 0.004 - 0.001 - 0.0006)
    p = ρ * TD.mixture_gas_constant(q₀, constants) * T₀
    𝒰 = TD.with_temperature(TD.LiquidIcePotentialTemperatureState(0.0, q₀, pˢᵗ, p, κ̂), T₀, constants)
    ψ = [ρᵈ, ρᵈ * 𝒰.potential_temperature, ρ * q₀.vapor, ρ * q₀.liquid, ρ * q₀.ice]
    T, q, _ = temperature_and_fractions(ψ)
    @test T ≈ T₀ rtol=1e-10

    function jacobian(n)
        h = 1e-6 * ψ[n]
        ψ⁺ = copy(ψ); ψ⁺[n] += h
        ψ⁻ = copy(ψ); ψ⁻[n] -= h
        return (internal_energy_density(ψ⁺) - internal_energy_density(ψ⁻)) / 2h
    end
    𝒥 = [jacobian(n) for n in 1:5]

    W = PTF.fixed_volume_phase_change_weights(T, q, ρ, ψ[2] / ρ, pˢᵗ, κ̂, constants)
    @test W[1] ≈ - 𝒥[3] / 𝒥[2] rtol=1e-6
    @test W[2] ≈ - 𝒥[4] / 𝒥[2] rtol=1e-6
    @test W[3] ≈ - 𝒥[5] / 𝒥[2] rtol=1e-6
end

#####
##### Precipitation drag heating
#####

# A test-only scheme: rain with a prescribed fall speed and no phase changes.
struct FallingRain{FT}
    fall_speed :: FT
end

AM.prognostic_field_names(::FallingRain) = (:ρqʳ,)
AM.moisture_prognostic_name(::FallingRain) = :ρqᵛ
AM.condensate_liquid_fraction(::FallingRain, ::Val{:ρqʳ}) = 1
AM.sedimentation_velocity(::FallingRain, μ, ::Val{:ρqʳ}) = μ.wʳ
AM.compute_microphysical_tendencies!(::FallingRain, model) = nothing
AM.microphysics_model_update!(::FallingRain, model) = nothing
@inline AM.maybe_adjust_thermodynamic_state(state, ::FallingRain, vapor, constants) = state
@inline AM.microphysical_state(::FallingRain, ρ, μ::NamedTuple, state, velocities) = (; qʳ=μ.ρqʳ / ρ)
@inline AM.moisture_fractions(::FallingRain, state::NamedTuple, vapor) = TD.MoistureMassFractions(vapor, state.qʳ, oftype(vapor, 0))
AM.liquid_mass_fraction(::FallingRain, model) = model.microphysical_fields.qʳ
AM.ice_mass_fraction(::FallingRain, model) = nothing

function AM.materialize_microphysical_fields(::FallingRain, grid, bcs)
    return (; ρqʳ=CenterField(grid; boundary_conditions=bcs.ρqʳ),
              qʳ=CenterField(grid), qᵛ=CenterField(grid), wʳ=AM.sedimentation_velocity_field(grid))
end

@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, m::FallingRain, state, ρ, thermo, constants)
    @inbounds μ.qʳ[i, j, k] = state.qʳ
    @inbounds μ.qᵛ[i, j, k] = thermo.moisture_mass_fractions.vapor
    @inbounds μ.wʳ[i, j, k] = - m.fall_speed
    return nothing
end

rain_profile(z) = ifelse((z > 600) & (z < 1600), 2e-3, 1e-4)

function falling_rain_column(precipitation_heating; compressible = true, rain_advection = UpwindBiased(order=1),
                             time_stepping = ExplicitTimeStepping())
    grid = RectilinearGrid(default_arch; size=8, halo=3, z=(0, 2000), topology=(Flat, Flat, Bounded))
    formulation = LiquidIcePotentialTemperatureFormulation(; precipitation_heating)
    dynamics = compressible ? CompressibleDynamics(time_stepping; reference_potential_temperature = 290) :
                              AnelasticDynamics(ReferenceState(grid, ThermodynamicConstants(); potential_temperature = 290))
    model = AtmosphereModel(grid; dynamics, formulation, microphysics = FallingRain(8.0),
                            scalar_advection = (; ρqʳ = rain_advection))
    if compressible
        set!(model; ρ = model.dynamics.reference_state.density, θ = 290, qᵛ = 0.005,
             qʳ = rain_profile)
    else
        set!(model; θ = 290, qᵛ = 0.005, qʳ = rain_profile)
    end
    return model
end

# ∫ρ(𝓊 + g z) changes only by the enthalpy of the rain leaving through the bottom. At the
# tendency level, with the state held at rest and fixed volume, d(ρ𝓊)/dt = ρcᵛᵐ Ṫ + cˡ T ṁ and
# d(ρ g z)/dt = g z ṁ, where ṁ is the rain-density tendency and Ṫ is recovered from the ρθ and ρqʳ
# tendencies by inverting ρθ(T, ρqʳ) at fixed ρᵈ and ρqᵛ.
function column_energy_rates(model)
    update_state!(model)
    AM.compute_tendencies!(model)
    constants = model.thermodynamic_constants
    g = constants.gravitational_acceleration
    ρᵈ = column_values(AM.dynamics_density(model.dynamics))
    ρqᵛ = column_values(model.moisture_density)
    ρqʳ = column_values(model.microphysical_fields.ρqʳ)
    T = column_values(model.temperature)
    ṁ = column_values(model.timestepper.Gⁿ.ρqʳ)
    Θ̇ = column_values(model.timestepper.Gⁿ.ρθ)
    z = collect(znodes(model.grid, Center()))
    Δz = z[2] - z[1]
    pˢᵗ = AM.standard_pressure(model.dynamics)
    cˡ = constants.liquid.heat_capacity
    function theta_density(k, Tₖ, ρqʳₖ)
        ρ = ρᵈ[k] + ρqᵛ[k] + ρqʳₖ
        q = TD.MoistureMassFractions(ρqᵛ[k] / ρ, ρqʳₖ / ρ, 0.0)
        p = ρ * TD.mixture_gas_constant(q, constants) * Tₖ
        𝒰 = TD.with_temperature(TD.LiquidIcePotentialTemperatureState(0.0, q, pˢᵗ, p), Tₖ, constants)
        return ρᵈ[k] * 𝒰.potential_temperature
    end
    Nz = length(T)
    ∂Θ∂T = [(theta_density(k, T[k] + 1e-3, ρqʳ[k]) - theta_density(k, T[k] - 1e-3, ρqʳ[k])) / 2e-3 for k in 1:Nz]
    ∂Θ∂m = [(theta_density(k, T[k], ρqʳ[k] + 1e-7) - theta_density(k, T[k], ρqʳ[k] - 1e-7)) / 2e-7 for k in 1:Nz]
    Ṫ = (Θ̇ .- ∂Θ∂m .* ṁ) ./ ∂Θ∂T
    ρcᵛᵐ = [begin
                ρ = ρᵈ[k] + ρqᵛ[k] + ρqʳ[k]
                q = TD.MoistureMassFractions(ρqᵛ[k] / ρ, ρqʳ[k] / ρ, 0.0)
                ρ * (TD.mixture_heat_capacity(q, constants) - TD.mixture_gas_constant(q, constants))
            end for k in 1:Nz]
    energy_rate = sum(ρcᵛᵐ .* Ṫ .+ (cˡ .* T .+ g .* z) .* ṁ) * Δz    # W m⁻²
    precipitation_enthalpy = cˡ * T[1] * sum(ṁ) * Δz                   # what leaves through the bottom
    potential_energy_release = - g * sum(z .* ṁ) * Δz                   # W m⁻², ≥ 0 for falling rain
    return (; energy_rate, precipitation_enthalpy, potential_energy_release)
end

@testset "Falling rain: ∫ρ(𝓊 + g z) changes only by the precipitation enthalpy" begin
    off = column_energy_rates(falling_rain_column(false))
    on = column_energy_rates(falling_rain_column(true))
    @test off.potential_energy_release > 0
    @test sum(off.precipitation_enthalpy) < 0   # rain leaves through the bottom
    residual_off = off.energy_rate - off.precipitation_enthalpy
    residual_on = on.energy_rate - on.precipitation_enthalpy
    @info "Falling rain (W m⁻²)" off.potential_energy_release residual_off residual_on
    # Without the drag heating the potential energy the rain loses vanishes
    @test residual_off ≈ - off.potential_energy_release rtol=0.02
    # With it, the column closes to the accuracy of the heating path's 1 / (ĉᵖ Π)
    @test abs(residual_on) < 0.02 * off.potential_energy_release
end

# Over one time step, the heating the option adds to ρθ equals the potential energy that the
# rain loses, −g Σ z Δm, for explicit sedimentation and for the adaptive implicit split (where most
# of the fall is applied by `implicit_sedimentation_step!`), on both dynamical cores.
@testset "Drag heating equals the potential energy released [$label]" for (label, compressible, rain_advection) in (
        ("compressible, explicit", true, UpwindBiased(order=1)),
        ("compressible, adaptive implicit", true, WENO(time_discretization=AdaptiveVerticallyImplicitDiscretization(cfl=0.01))),
        ("anelastic, explicit", false, UpwindBiased(order=1)),
        ("anelastic, adaptive implicit", false, WENO(time_discretization=AdaptiveVerticallyImplicitDiscretization(cfl=0.01))))

    Δt = compressible ? 0.2 : 2.0
    models = map(heating -> falling_rain_column(heating; compressible, rain_advection), (false, true))
    Θ₀ = column_values(models[1].formulation.potential_temperature_density)
    m₀ = column_values(models[1].microphysical_fields.ρqʳ)
    for model in models
        time_step!(model, Δt)
    end
    off, on = models
    ΔΘ = column_values(on.formulation.potential_temperature_density) .- column_values(off.formulation.potential_temperature_density)
    Δm = column_values(off.microphysical_fields.ρqʳ) .- m₀
    z = collect(znodes(off.grid, Center()))
    g = off.thermodynamic_constants.gravitational_acceleration
    released = - g * sum(z .* Δm)
    heating = sum(heating_from_theta(on, ΔΘ))
    @info "Drag heating over one step [$label] (J m⁻² per cell area)" released heating
    @test released > 0
    @test heating ≈ released rtol=0.02
    @test all(isfinite, Θ₀)
end

#####
##### Subgrid dissipative heating
#####

# In a horizontally uniform shear flow u = U cos(π z / H), with no-stress walls, the only momentum
# tendency is the viscous one, and the heating should return the kinetic energy it removes.
function shear_flow_rates(closure; dissipative_heating = true)
    grid = RectilinearGrid(default_arch; size=(4, 4, 32), x=(0, 400), y=(0, 400), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded))
    formulation = LiquidIcePotentialTemperatureFormulation(; dissipative_heating)
    model = AtmosphereModel(grid; formulation, closure)
    set!(model; θ = 300, u = (x, y, z) -> 5 * cos(π * z / 1000))
    update_state!(model)
    AM.compute_tendencies!(model)
    return model
end

@testset "Dissipative heating balances the subgrid kinetic-energy loss" begin
    for closure in (ScalarDiffusivity(ν = 20), SmagorinskyLilly())
        on = shear_flow_rates(closure)
        off = shear_flow_rates(closure; dissipative_heating = false)
        ΔG = column_values(on.timestepper.Gⁿ.ρθ) .- column_values(off.timestepper.Gⁿ.ρθ)
        heating = sum(heating_from_theta(on, ΔG))   # W m⁻³ summed over the (uniform) cells
        u = column_values(on.velocities.u)
        Gρu = column_values(on.timestepper.Gⁿ.ρu)
        kinetic_energy_loss = - sum(u .* Gρu)
        @info "Shear flow [$(summary(closure))]" kinetic_energy_loss heating
        @test kinetic_energy_loss > 0
        @test heating > 0
        @test heating ≈ kinetic_energy_loss rtol=0.05
    end

    # A vertically implicit closure heats by the stresses it applies implicitly too
    explicit = shear_flow_rates(SmagorinskyLilly())
    implicit = shear_flow_rates(SmagorinskyLilly(VerticallyImplicitTimeDiscretization()))
    @test Array(interior(implicit.timestepper.Gⁿ.ρθ)) ≈ Array(interior(explicit.timestepper.Gⁿ.ρθ))

    # Without a closure the option adds nothing
    none = shear_flow_rates(nothing)
    reference = shear_flow_rates(nothing; dissipative_heating = false)
    @test Array(interior(none.timestepper.Gⁿ.ρθ)) == Array(interior(reference.timestepper.Gⁿ.ρθ))
end
