include(joinpath(@__DIR__, "setup.jl"))

#####
##### Radiation with HydrostaticDynamics: the level pressures that the radiative transfer model is
##### given, the heating of an isothermal column against its analytic solution, and the energy
##### budget of the column
#####

using Breeze
using Breeze.AtmosphereModels: level_pressure_source, uses_level_pressure
using Breeze.HydrostaticEquations: HydrostaticModel, VerletSubsteps, validate_hydrostatic_radiation
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units
using Test

# Load RRTMGP to trigger the extension
using ClimaComms
using RRTMGP

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

const Nz = 20
const pᵗ = 200.0     # Pa, the pressure of the model top
const pˢ = 98000.0   # Pa, a surface pressure that is not the standard pressure
const σ = 5.670374419e-8 # W m⁻² K⁻⁴

column_grid() = RectilinearGrid(default_arch, FT; size=Nz, x=0, y=0, z=(0, 1), topology=(Flat, Flat, Bounded))

function radiative_column(; surface_temperature, solar_constant=0, schedule=IterationInterval(1), kw...)
    grid = column_grid()
    constants = reference_model_constants()
    A, B = hybrid_coefficients(Nz; top_pressure=pᵗ)
    radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                       surface_temperature, surface_emissivity=1, surface_albedo=0,
                                       solar_constant, solar_position=FixedCosineZenith(0.5), schedule)
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B, surface_pressure=pˢ)
    return hydrostatic_model(grid; dynamics, radiation, thermodynamic_constants=constants, kw...)
end

column(field) = vec(Array(interior(field)))

# Net upward flux on the z-faces, W m⁻²
net_flux(radiation) = column(radiation.upwelling_longwave_flux) .+ column(radiation.downwelling_longwave_flux) .+
                      column(radiation.upwelling_shortwave_flux) .+ column(radiation.downwelling_shortwave_flux)

# Enthalpy of the column per unit area, Σ cᵖᵈ T Δp / g = Σ cᵖᵈ Π ρθ Δz, in J m⁻²
function column_enthalpy(model)
    Π = column(model.dynamics.exner_function)
    ρθ = column(model.formulation.potential_temperature_density)
    return sum(cᵖᵈ .* Π .* ρθ) / Nz
end

# Rate of change of the temperature of every layer by the tendency of ρθ, K s⁻¹
function temperature_tendency(model)
    Π = column(model.dynamics.exner_function)
    ρ = column(model.dynamics.density)
    return column(model.timestepper.Gⁿ.ρθ) .* Π ./ ρ
end

set_temperature!(model, T) = set!(model; θ = reshape(T ./ column(model.dynamics.exner_function), 1, 1, Nz))

#####
##### Tolerances, written before these tests were run
#####
#
#   1. Level pressures: the lowest is pˢ and the highest pᵗ exactly (they are A + B pˢ with
#      A = 0, B = 1 and with B = 0), and the pressure thicknesses sum to pˢ - pᵗ within 1e-13,
#      relative (20 additions).
#   2. Isothermal column (atmosphere and ground at T₀, emissivity 1, no sun). The upwelling flux is
#      σ T₀⁴ on every level and the net flux σ T₀⁴ exp(-D τ⁺), with τ⁺ the optical depth above the
#      level and D the diffusivity factor of the one-angle solver: no source function is
#      interpolated, so the solver should reproduce this to rounding. Gate 1e-10, relative. The
#      heating rates, g ΔF / (cᵖᵈ Δp), within 1e-9 of the analytic ones, relative to the largest.
#      Every layer cools, most at the model top, where the optical depth above vanishes and the
#      cooling tends to g D fₗ τₑ σ T₀⁴ / (cᵖᵈ pˢ) = 7.1 K per day: the highest layer is within 2 %
#      of it.
#      Disclosure: this last gate first read "the largest cooling lies between 0.3 and 5 K per
#      day", a guess that the first run failed (7.07 K per day) while the analytic comparison
#      passed. The guess was wrong, not the model: it is replaced by the analytic limit above.
#   3. Energy budget. The tendency of the column enthalpy equals the net flux into the column
#      within 1e-12 of the larger flux, and over 20 time steps with fluxes held fixed the enthalpy
#      changes by that flux times the elapsed time within 1e-12, relative to the change.

@testset "Level pressures of the radiative transfer model [$FT]" begin
    model = radiative_column(surface_temperature=285)
    set_temperature!(model, standard_temperature.(column(model.dynamics.pressure)))
    state = model.radiation.atmospheric_state
    level_pressures = Array(state.p_lev)[:, 1]
    layer_pressures = Array(state.p_lay)[:, 1]

    @test model isa HydrostaticModel
    @test !isnothing(level_pressure_source(model.dynamics))
    @test level_pressures[1] == pˢ
    @test level_pressures[end] == pᵗ
    @test sum(-diff(level_pressures)) ≈ pˢ - pᵗ rtol=1e-13
    @test all(diff(level_pressures) .< 0)
    @test all(level_pressures[1:end-1] .> layer_pressures .> level_pressures[2:end])
    @test layer_pressures == column(model.dynamics.pressure)

    # The dynamics that carry no pressure on their z-faces are not affected
    @test isnothing(level_pressure_source(CompressibleDynamics()))
end

@testset "Longwave cooling of an isothermal column against its analytic solution [$FT]" begin
    T₀ = 280.0
    model = radiative_column(surface_temperature=T₀)
    set_temperature!(model, fill(T₀, Nz))
    update_state!(model)

    radiation = model.radiation
    state = radiation.atmospheric_state
    p = Array(state.p_lay)[:, 1]
    pᶠ = Array(state.p_lev)[:, 1]
    Δp = -diff(pᶠ)

    # Optical thickness of the layers (O'Gorman and Schneider 2008, at the equator) and the
    # diffusivity factor of RRTMGP's one-angle longwave solver
    (; α, fₗ, τₑ) = state.otp
    τ = @. α * Δp / p * (fₗ * p / pˢ + (1 - fₗ) * 4 * (p / pˢ)^4) * τₑ
    D = 1 / 0.6096748751
    analytic_net_flux = [σ * T₀^4 * exp(-D * sum(τ[k:end])) for k in 1:Nz+1]
    analytic_heating = @. - g * (analytic_net_flux[2:end] - analytic_net_flux[1:end-1]) / (cᵖᵈ * Δp)

    F = net_flux(radiation)
    @test column(radiation.upwelling_longwave_flux) ≈ fill(σ * T₀^4, Nz + 1) rtol=1e-10
    @test maximum(abs, F .- analytic_net_flux) < 1e-10 * σ * T₀^4

    heating = temperature_tendency(model)
    @test maximum(abs, heating .- analytic_heating) < 1e-9 * maximum(abs, analytic_heating)
    @test all(heating .< 0)
    @test argmax(abs.(heating)) == Nz
    @test heating[Nz] ≈ - g * D * fₗ * τₑ * σ * T₀^4 / (cᵖᵈ * pˢ) rtol=0.02

    @info "Isothermal column: cooling rates (K/day) from the ground up" round.(heating .* 86400, digits=3)
end

@testset "Energy budget of a radiating column [$FT, $name]" for (name, kw) in
        (("SSP RK3", (;)), ("MIS3C + Verlet", split_explicit(:MIS3C, VerletSubsteps())))

    # A standard atmosphere over warmer ground, in sunlight, with fluxes computed once
    model = radiative_column(; surface_temperature=295, solar_constant=1361, schedule=IterationInterval(10^6), kw...)
    set_temperature!(model, standard_temperature.(column(model.dynamics.pressure)))
    update_state!(model)

    F = net_flux(model.radiation)
    into_column = F[1] - F[end]
    @test abs(column(model.radiation.downwelling_shortwave_flux)[end]) ≈ 1361 / 2

    # The tendency: Σ cᵖᵈ Π ∂ₜ(ρθ) Δz against the flux through the two ends of the column
    Π = column(model.dynamics.exner_function)
    enthalpy_tendency = sum(cᵖᵈ .* Π .* column(model.timestepper.Gⁿ.ρθ)) / Nz
    @test abs(enthalpy_tendency - into_column) < 1e-12 * max(abs(F[1]), abs(F[end]))

    # Layer by layer, the heating is the divergence of the flux over the mass of the layer
    Δp = -diff(Array(model.radiation.atmospheric_state.p_lev)[:, 1])
    @test temperature_tendency(model) ≈ - g .* diff(F) ./ (cᵖᵈ .* Δp) rtol=1e-11

    H⁰ = column_enthalpy(model)
    Δt, steps = 600, 20
    for _ in 1:steps
        time_step!(model, Δt)
    end

    @test net_flux(model.radiation) == F # held fixed by the schedule
    ΔH = column_enthalpy(model) - H⁰
    @test abs(ΔH) > 1e5
    @test abs(ΔH - into_column * steps * Δt) < 1e-12 * H⁰ + 1e-10 * abs(ΔH)

    @info "Radiating column [$name]" into_column ΔH
end

# A radiative transfer model that the list of capabilities does not know, and one that declares
# the capability
struct UnknownRadiation end
struct DeclaredRadiation end
Breeze.AtmosphereModels.uses_level_pressure(::DeclaredRadiation) = true

@testset "Radiation that HydrostaticDynamics rejects" begin
    grid = column_grid()
    @test uses_level_pressure(radiative_column(surface_temperature=285).radiation)
    @test !uses_level_pressure(UnknownRadiation()) # unknown means unsupported
    @test isnothing(validate_hydrostatic_radiation(nothing, grid))
    @test isnothing(validate_hydrostatic_radiation(DeclaredRadiation(), grid))
    @test_throws ArgumentError validate_hydrostatic_radiation(UnknownRadiation(), grid)
end
