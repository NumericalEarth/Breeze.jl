include(joinpath(@__DIR__, "setup.jl"))

#####
##### Analytic gray column through the ecCKD kernel path
#####
##### A one-g-point `EcCKDGasOpticsModel` with a single absorbing gas (dry air) turns the column
##### kernel into a gray radiative transfer solver with closed-form solutions: an isothermal
##### column emits `σT⁴` upward everywhere and `σT⁴ (1 - e^{-D τ})` downward, a non-scattering
##### shortwave beam obeys Beer–Lambert, and the flux divergence follows from the face fluxes.
##### These pin the kernel's orientation, units (mol m⁻² gas amounts), boundary faces and the
##### diffusivity `D = 1.66` of the streaming longwave solver, at rounding accuracy.
#####

using Breeze
using Breeze.AtmosphereModels: total_density
using NumericalRadiation: EcCKDGasOpticsModel
using Oceananigans
using Oceananigans.Units
using Test

# The diffusivity of NumericalRadiation's no-scattering longwave solver, pinned here
const DIFFUSIVITY = 1.66

column_grid(FT, Nz, top) = RectilinearGrid(default_arch, FT; size = Nz, x = 0.0, y = 45.0, z = (0, top),
                                            topology = (Flat, Flat, Bounded))

# The gray gas optics: one longwave and one shortwave g point, absorption per mol m⁻² of dry air
gray_gas_model(FT, κ_lw, κ_sw) = EcCKDGasOpticsModel(names = (:composite,),
                                                     longwave_absorption = FT[κ_lw;;],
                                                     shortwave_absorption = FT[κ_sw;;])

function gray_radiation(grid, κ_lw, κ_sw; surface_temperature, cos_zenith, albedo = 0.3)
    FT = eltype(grid)
    return RadiativeTransferModel(grid, EcCKDOptics(gray_gas_model(FT, κ_lw, κ_sw)), ThermodynamicConstants();
                                  column_extension = nothing, surface_temperature, surface_emissivity = 1,
                                  surface_albedo = albedo, solar_constant = 1361,
                                  solar_position = FixedCosineZenith(cos_zenith))
end

# A dry anelastic column whose temperature is `T_target(z)` at every cell center: since the
# temperature is linear in θ at fixed reference pressure, one fixed-point step from the first
# guess `θ = T_target` lands on it to rounding
function set_temperature!(model, T_target)
    set!(model; θ = T_target)
    θ = CenterField(model.grid)
    set!(θ, T_target)
    interior(θ) .= interior(θ) .^ 2 ./ interior(model.temperature)
    set!(model; θ)
    return model
end

function gray_column_model(grid, radiation, T_target)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure = 101325, potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state),
                            formulation = :LiquidIcePotentialTemperature, radiation)
    return set_temperature!(model, T_target)
end

column(field) = Array(interior(field))[1, 1, :]

# Layer optical depths `κ n_dry` from the model's density (bottom-up, one per cell), and the
# cumulative optical depth above and below each face
function layer_optical_depths(model, κ)
    grid = model.grid
    Nz = size(grid, 3)
    mᵈ = model.thermodynamic_constants.dry_air.molar_mass
    ρ = column(total_density(model.dynamics))
    Δz = [Oceananigans.Operators.Δzᶜᶜᶜ(1, 1, k, grid) for k in 1:Nz]
    τ = κ .* ρ .* Δz ./ mᵈ
    τ_above = [sum(τ[k:Nz]) for k in 1:Nz+1]   # cells above face k
    τ_below = [sum(τ[1:k-1]) for k in 1:Nz+1]  # cells below face k
    return τ, τ_above, τ_below
end

# Both float types always: this is the test that pins the Float32 kernel path at rounding accuracy
@testset "Gray column through the ecCKD kernel [$(FT)]" for FT in all_float_types()
    Oceananigans.defaults.FloatType = FT
    rtol = FT == Float64 ? 1e-12 : 200 * eps(Float32)

    Nz = 16
    grid = column_grid(FT, Nz, 3kilometers)
    T₀ = 280
    κ_lw = 2e-5   # τ ≈ 2 over the 3 km column
    κ_sw = 1e-5   # τ ≈ 1
    μ₀ = FT(0.6)
    α = FT(0.3)
    S₀ = FT(1361)

    radiation = gray_radiation(grid, κ_lw, κ_sw; surface_temperature = T₀, cos_zenith = μ₀, albedo = α)
    model = gray_column_model(grid, radiation, z -> T₀)

    # The column is isothermal to rounding, and the grid is the whole atmosphere
    T = column(model.temperature)
    @test maximum(abs.(T .- T₀)) < 10 * eps(FT) * T₀
    @test isnothing(radiation.atmospheric_state.extension)

    ℐ_lw_up = column(radiation.upwelling_longwave_flux)
    ℐ_lw_dn = column(radiation.downwelling_longwave_flux)
    ℐ_sw_up = column(radiation.upwelling_shortwave_flux)
    ℐ_sw_dn = column(radiation.downwelling_shortwave_flux)

    # The gray source is `σ T⁴` with the Stefan–Boltzmann constant the gas model carries
    σ = radiation.longwave_solver.gas_model.stefan_boltzmann
    B = σ * T₀^4

    @testset "Isothermal longwave" begin
        τ, τ_above, _ = layer_optical_depths(model, κ_lw)
        @test all(τ .> 1e-3)   # every layer above the thin-layer branch of the solver

        # Upwelling: the surface (ε = 1) and every layer emit at the same temperature
        @test ℐ_lw_up ≈ fill(B, Nz + 1) rtol = rtol

        # Downwelling: emission of the atmosphere above each face with diffusivity D
        @test ℐ_lw_dn ≈ -B .* (1 .- exp.(-DIFFUSIVITY .* τ_above)) rtol = rtol
        @test ℐ_lw_dn[Nz+1] == 0
    end

    @testset "Beer–Lambert shortwave" begin
        _, τ_above, τ_below = layer_optical_depths(model, κ_sw)
        τ_s = τ_above[1]

        # Direct beam down the slant path, one Lambertian reflection, diffuse (two-stream,
        # ω = 0: transmittance e^{-2τ}) back up
        @test ℐ_sw_dn ≈ -S₀ * μ₀ .* exp.(-τ_above ./ μ₀) rtol = rtol
        @test ℐ_sw_up ≈ α * S₀ * μ₀ * exp(-τ_s / μ₀) .* exp.(-2 .* τ_below) rtol = rtol
        @test -ℐ_sw_dn[Nz+1] ≈ S₀ * μ₀ rtol = rtol
    end

    @testset "Flux divergence at every cell" begin
        _, τ_above_lw, _ = layer_optical_depths(model, κ_lw)
        _, τ_above_sw, τ_below_sw = layer_optical_depths(model, κ_sw)
        τ_s = τ_above_sw[1]

        # Analytic net flux (positive upward) at every face, boundary faces included
        F_net = B .* exp.(-DIFFUSIVITY .* τ_above_lw) .-
                S₀ * μ₀ .* exp.(-τ_above_sw ./ μ₀) .+
                α * S₀ * μ₀ * exp(-τ_s / μ₀) .* exp.(-2 .* τ_below_sw)

        Δz = [Oceananigans.Operators.Δzᶜᶜᶜ(1, 1, k, grid) for k in 1:Nz]
        divergence = -(F_net[2:Nz+1] .- F_net[1:Nz]) ./ Δz
        @test column(radiation.flux_divergence) ≈ divergence rtol = rtol
    end

    @testset "Night" begin
        night = gray_radiation(grid, κ_lw, κ_sw; surface_temperature = T₀, cos_zenith = 0, albedo = α)
        gray_column_model(grid, night, z -> T₀)
        @test all(iszero, column(night.upwelling_shortwave_flux))
        @test all(iszero, column(night.downwelling_shortwave_flux))
        @test column(night.upwelling_longwave_flux) == ℐ_lw_up
        @test column(night.downwelling_longwave_flux) == ℐ_lw_dn
    end
end

@testset "Gray column second-order convergence" begin
    Oceananigans.defaults.FloatType = Float64
    FT = Float64
    κ_lw = 2e-5
    T₀ = 290
    Γ = -6e-3   # K m⁻¹, so the Planck source is not linear in optical depth
    T_target(z) = T₀ + Γ * z

    # Outgoing and surface downwelling longwave of a linear temperature profile: the interface
    # sources are exact for a linear profile, and the half-level Planck path treats the source
    # as linear in optical depth within each layer, so the error is second order in Δz
    function longwave_boundary_fluxes(Nz)
        grid = column_grid(FT, Nz, 3kilometers)
        radiation = gray_radiation(grid, κ_lw, 0; surface_temperature = T₀, cos_zenith = 0)
        model = gray_column_model(grid, radiation, T_target)
        # Every layer, the reference's included, stays above the solver's thin-layer branch
        # (τ ≤ 1e-3), where the Planck path switches to its trapezoidal form and the error
        # being ratioed would turn first order
        @test all(layer_optical_depths(model, κ_lw)[1] .> 1e-3)
        return column(radiation.upwelling_longwave_flux)[Nz+1], -column(radiation.downwelling_longwave_flux)[1]
    end

    # A fine reference (its own error is 1/256 of the coarsest tested error)
    reference = longwave_boundary_fluxes(1024)
    errors = map((16, 32, 64)) do Nz
        fluxes = longwave_boundary_fluxes(Nz)
        return abs(fluxes[1] - reference[1]) + abs(fluxes[2] - reference[2])
    end

    @test all(errors .> 0)
    @test 3.3 < errors[1] / errors[2] < 4.7
    @test 3.3 < errors[2] / errors[3] < 4.7
end

Oceananigans.defaults.FloatType = Float64
