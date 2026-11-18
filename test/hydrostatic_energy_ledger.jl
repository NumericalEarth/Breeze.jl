include(joinpath(@__DIR__, "setup.jl"))

#####
##### A signed energy ledger of HydrostaticDynamics with forcing and diffusion together
#####
##### For a hydrostatic atmosphere below a surface of constant pressure pᵗ, the internal and the
##### potential energy of a column sum to
#####
#####     I + P = ∫ cᵖᵈ T dp / g + (Φˢ pˢ - Φᵗ pᵗ) / g ,
#####
##### and the atmosphere does the work pᵗ ∂ₜzᵗ on what lies above its top, which is exactly the rate
##### of change of the last term. With 𝓗 = ∫ cᵖᵈ T dp / g + Φˢ pˢ / g (the enthalpy, plus the
##### potential energy of the mass at the height of the ground) and 𝓚 the kinetic energy of the
##### horizontal wind, the equations therefore give
#####
#####     Δ𝓗 + Δ𝓚 = boundary input ,
#####
##### the energy that the surface flux and the radiative fluxes bring through the ground and the
##### top. The ledger below reports each term with its sign, the top work pᵗ Δzᵗ (which is not part
##### of the balance: it is how much of the change of I + P the top term carries), and the residual
#####
#####     residual = Δ𝓗 + Δ𝓚 - boundary input .
#####
##### WHAT THIS IS, AND IS NOT. The tests of the vertical diffusion assert the conservation of Θ,
##### which is a property of the operator. That does NOT validate the heating: a diffusion of θ
##### moves θ between layers of different Π and so changes the enthalpy, and neither the vertical
##### diffusion nor the hyperdiffusion returns the kinetic energy it removes as heat. The residual
##### measured here is the size of those defects, with the discretization error of the core. It
##### is reported, not passed: a lowest layer that stays cool is no evidence of a faithful energy
##### budget.
#####

using Breeze
using Breeze.HydrostaticEquations: GridScaleHyperdiffusion, VerticalDiffusion, RichardsonNumberBoundaryLayer, VerletSubsteps
using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.Operators: Azᶜᶜᶜ, Vᶜᶜᶜ
using Oceananigans.Units
using Printf
using Test

# Load RRTMGP to trigger the extension
using ClimaComms
using RRTMGP

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

const pᵗ = 200.0
column(field) = vec(Array(interior(field)))
figdata_path(name) = (dir = get(ENV, "BREEZE_P3B_FIGDATA", ""); isempty(dir) ? nothing : joinpath(dir, name))

const ledger_header = "case,layers,dt_s,duration_h,enthalpy_change,kinetic_energy_change,surface_input,radiative_input,top_work,residual,residual_over_input,units"
const ledger_lines = String[ledger_header]

function report!(case, Nz, Δt, duration, ledger, units)
    line = @sprintf("%s,%d,%g,%g,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.6e,%s", case, Nz, Δt, duration / 3600,
                    ledger.ΔH, ledger.ΔK, ledger.surface, ledger.radiation, ledger.top_work, ledger.residual,
                    ledger.residual / (ledger.surface + ledger.radiation), units)
    push!(ledger_lines, line)
    @info "LEDGER " * line
    return nothing
end

#####
##### A single column: surface flux, fixed radiative fluxes, a sheared wind, vertical diffusion
#####

# Energies of a column per unit area, J m⁻², and the height of its top, m
function column_energies(model)
    Nz = size(model.grid, 3)
    Π = column(model.dynamics.exner_function)
    Πᶠ = column(model.dynamics.interface_exner_function)
    ρθ = column(model.formulation.potential_temperature_density)
    θ = column(model.formulation.potential_temperature)
    μ = column(model.dynamics.density) ./ Nz
    u, v = column(model.velocities.u), column(model.velocities.v)
    H = sum(cᵖᵈ .* Π .* ρθ) / Nz
    K = sum(μ .* (u .^ 2 .+ v .^ 2)) / 2
    zᵗ = sum(cᵖᵈ * θ[k] * (Πᶠ[k] - Πᶠ[k+1]) for k in 1:Nz) / g
    return (; H, K, zᵗ)
end

net_flux(radiation) = column(radiation.upwelling_longwave_flux) .+ column(radiation.downwelling_longwave_flux) .+
                      column(radiation.upwelling_shortwave_flux) .+ column(radiation.downwelling_shortwave_flux)

function column_ledger(Nz, Δt; diffusion, duration=6hours, 𝒬=200.0, kw...)
    grid = RectilinearGrid(default_arch, FT; size=Nz, x=0, y=0, z=(0, 1), topology=(Flat, Flat, Bounded))
    constants = reference_model_constants()
    A, B = hybrid_coefficients(Nz; top_pressure=pᵗ)

    # Fluxes computed once, from the initial state: the radiative input is then known exactly
    radiation = RadiativeTransferModel(grid, GrayOptics(), constants; surface_temperature=290, surface_emissivity=1,
                                       surface_albedo=0, solar_constant=1361, solar_position=FixedCosineZenith(0.5),
                                       schedule=IterationInterval(10^9))

    vertical_diffusion = diffusion ? VerticalDiffusion(boundary_layer=RichardsonNumberBoundaryLayer(FT)) : nothing
    dynamics = HydrostaticDynamics(; pressure_coefficient=A, surface_pressure_coefficient=B, vertical_diffusion)
    boundary_conditions = (; ρE = FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬)))
    model = hydrostatic_model(grid; dynamics, radiation, boundary_conditions, thermodynamic_constants=constants, kw...)

    T = standard_temperature.(column(model.dynamics.pressure))
    set!(model; θ = reshape(T ./ column(model.dynamics.exner_function), 1, 1, Nz), u = z -> 5 + 40 * z, v = z -> 10 * sinpi(z))

    F = net_flux(model.radiation)
    before = column_energies(model)
    steps = round(Int, duration / Δt)
    for _ in 1:steps
        time_step!(model, Δt)
    end
    after = column_energies(model)

    @assert net_flux(model.radiation) == F
    surface = 𝒬 * steps * Δt
    radiative = (F[1] - F[end]) * steps * Δt
    ΔH, ΔK = after.H - before.H, after.K - before.K

    return (; ΔH, ΔK, surface, radiation = radiative, top_work = pᵗ * (after.zᵗ - before.zᵗ),
              residual = ΔH + ΔK - surface - radiative, H = before.H)
end

#####
##### Gates, written before these runs; everything else is reported
#####
#
#   1. Forcing without diffusion in a column (surface flux and radiative fluxes, a wind that
#      nothing changes): the residual is rounding, below 1e-12 of the enthalpy of the column. This
#      is the statement that the forcing terms themselves put in exactly the energy they are
#      given, and the only one that is justified a priori.
#   2. With the vertical diffusion: the residual is finite. Its sign and size are reported for
#      three time steps and three vertical resolutions. Two defects are known to be in it, and
#      neither is a truncation error, so the residual is NOT expected to vanish under refinement:
#      the diffusion of θ changes the enthalpy (upward transport of θ to lower Π lowers it), and
#      the kinetic energy that the diffusion of the wind removes is not returned as heat.
#   3. On the latitude-longitude grid with a heated continent: finite. The residual of the bare
#      core (forcing, no dissipation) is the discretization error of its energy conversion, and
#      that of the run with vertical diffusion and hyperdiffusion adds the two defects above and
#      the kinetic energy that the hyperdiffusion removes; both are reported for two time steps.

@testset "Energy ledger of a forced column [$FT]" begin
    duration = 6hours

    for Nz in (15, 30, 60)
        ledger = column_ledger(Nz, 300; diffusion=false, duration)
        report!("column: forcing only", Nz, 300, duration, ledger, "J m-2")
        @test abs(ledger.residual) < 1e-12 * ledger.H                                  # gate 1
        @test ledger.ΔK == 0
    end

    refinements = ((30, 600), (30, 300), (30, 150), (15, 300), (60, 300))
    residuals = Dict{Tuple{Int, Int}, Float64}()

    for (Nz, Δt) in refinements
        ledger = column_ledger(Nz, Δt; diffusion=true, duration)
        report!("column: forcing + vertical diffusion", Nz, Δt, duration, ledger, "J m-2")
        @test isfinite(ledger.residual)                                                # gate 2
        @test ledger.ΔK < 0 # the diffusion of the wind removes kinetic energy
        residuals[(Nz, Δt)] = ledger.residual / (ledger.surface + ledger.radiation)
    end

    @info "Residual over the boundary input under refinement of the time step (30 layers)" Δt_600 = residuals[(30, 600)] Δt_300 = residuals[(30, 300)] Δt_150 = residuals[(30, 150)]
    @info "Residual over the boundary input under refinement of the layers (Δt = 300 s)" layers_15 = residuals[(15, 300)] layers_30 = residuals[(30, 300)] layers_60 = residuals[(60, 300)]

    ledger = column_ledger(30, 600; diffusion=true, duration, split_explicit(:MIS3C, VerletSubsteps())...)
    report!("column: forcing + vertical diffusion (MIS3C + Verlet)", 30, 600, duration, ledger, "J m-2")
    @test isfinite(ledger.residual)
end

#####
##### The latitude-longitude grid with a heated continent
#####

@inline over_land(λ, φ) = (-40 ≤ λ ≤ 60) & (-20 ≤ φ ≤ 40)
@inline surface_energy_flux(λ, φ, t) = ifelse(over_land(λ, φ), 220 * max(0, cos(2π * (t - 6 * 3600) / 86400 + λ * π / 180)) * cosd(φ), 0.0)

# Energies of the atmosphere, J, and the area integral of the height of its top, m³
function global_energies(model)
    grid = on_architecture(CPU(), model.grid)
    Nx, Ny, Nz = size(grid)
    V = [Vᶜᶜᶜ(i, j, k, grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    Az = [Azᶜᶜᶜ(i, j, 1, grid) for i in 1:Nx, j in 1:Ny]
    Π = Array(interior(model.dynamics.exner_function))
    Πᶠ = Array(interior(model.dynamics.interface_exner_function))
    ρθ = Array(interior(model.formulation.potential_temperature_density))
    θ = Array(interior(model.formulation.potential_temperature))
    m = Array(interior(model.dynamics.density)) .* V
    u = Array(interior(model.velocities.u))
    v = Array(interior(model.velocities.v))

    # The masses of the velocity cells are the averages of the masses of the cells on either side
    mᵘ = (m .+ circshift(m, (1, 0, 0))) ./ 2
    mᵛ = (m[:, 1:end-1, :] .+ m[:, 2:end, :]) ./ 2

    H = sum(cᵖᵈ .* Π .* ρθ .* V)
    K = sum(mᵘ .* u .^ 2) / 2 + sum(mᵛ .* v[:, 2:end-1, :] .^ 2) / 2
    zᵗ = sum(Az[i, j] * cᵖᵈ * θ[i, j, k] * (Πᶠ[i, j, k] - Πᶠ[i, j, k+1]) for i in 1:Nx, j in 1:Ny, k in 1:Nz) / g
    return (; H, K, zᵗ, Az)
end

function global_ledger(Δt; dissipation, duration=1day, Nz=12)
    grid = LatitudeLongitudeGrid(default_arch, FT; size=(90, 35, Nz), longitude=(-180, 180), latitude=(-70, 70), z=(0, 1))
    A, B = hybrid_coefficients(Nz; top_pressure=pᵗ)
    terms = dissipation ? (hyperdiffusion = GridScaleHyperdiffusion(FT; damping_time=43200),
                           vertical_diffusion = VerticalDiffusion(boundary_layer=RichardsonNumberBoundaryLayer(FT))) : (;)
    dynamics = HydrostaticDynamics(; pressure_coefficient=A, surface_pressure_coefficient=B, terms...)
    energy_flux = Field{Center, Center, Nothing}(grid)
    boundary_conditions = (; ρE = FieldBoundaryConditions(bottom=FluxBoundaryCondition(energy_flux)))
    model = hydrostatic_model(grid; dynamics, boundary_conditions, coriolis=HydrostaticSphericalCoriolis(FT; rotation_rate=7.29212e-5),
                              thermodynamic_constants=reference_model_constants())

    T = standard_temperature.(Array(interior(dynamics_pressure(model.dynamics))))
    set!(model; θ = T ./ Array(interior(model.dynamics.exner_function)))

    before = global_energies(model)
    surface = 0.0
    for _ in 1:round(Int, duration / Δt)
        t = model.clock.time
        set!(energy_flux, (λ, φ) -> surface_energy_flux(λ, φ, t))
        surface += Δt * sum(Array(interior(energy_flux))[:, :, 1] .* before.Az) # the flux is held over the step
        time_step!(model, Δt)
    end
    after = global_energies(model)

    ΔH, ΔK = after.H - before.H, after.K - before.K
    return (; ΔH, ΔK, surface, radiation = 0.0, top_work = pᵗ * (after.zᵗ - before.zᵗ),
              residual = ΔH + ΔK - surface, H = before.H)
end

@testset "Energy ledger of a heated continent [$FT]" begin
    duration = 1day
    for dissipation in (false, true), Δt in (300, 150)
        ledger = global_ledger(Δt; dissipation, duration)
        case = dissipation ? "latitude-longitude: forcing + vertical diffusion + hyperdiffusion" : "latitude-longitude: forcing only (bare core)"
        report!(case, 12, Δt, duration, ledger, "J")
        @test isfinite(ledger.residual)                                                # gate 3
        @test ledger.surface > 0
    end
end

path = figdata_path("energy_ledger.csv")
isnothing(path) || write(path, join(ledger_lines, "\n") * "\n")
