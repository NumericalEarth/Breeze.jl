include(joinpath(@__DIR__, "setup.jl"))

#####
##### Kernel path versus NumericalRadiation's array path on one staged column
#####
##### The column kernel calls NumericalRadiation's scalar layer API and streaming solvers, whose
##### array counterparts (`optical_properties!` and `radiative_fluxes!`) are loops over the same
##### functions. Running the array path on the host view of a staged column must therefore
##### reproduce the kernel's fluxes bit for bit.
#####

using Breeze
using GPUArraysCore: @allowscalar
using NCDatasets
using NumericalRadiation: LongwaveOptics, ShortwaveOptics, RadiativeFluxes,
                          CloudlessLongwave, CloudlessShortwave,
                          LongwaveBoundaryConditions, ShortwaveBoundaryConditions,
                          optical_properties!, radiative_fluxes!,
                          surface_longwave_emission, rayleigh_optical_depth
using Oceananigans
using Oceananigans.Units
using Test

const NumericalRadiationExt = Base.get_extension(Breeze, :BreezeNumericalRadiationExt)
using .NumericalRadiationExt: column_atmosphere, number_of_layers

# The array path of NumericalRadiation on the host view of staged column `(i, j)`, with the
# boundary conditions of the kernel: no downwelling longwave at the top, surface emission
# `ε B_g(Tₛ)` and reflection `1 - ε`; shortwave irradiance `S₀ max(μ₀, 0)` and the direct and
# diffuse surface albedos.
#
# The one documented difference is the Rayleigh air amount: the array `optical_properties!`
# forms the hydrostatic `Δp / (g Mᵈ)` of each layer from the interface pressures with its own
# constants, whereas the kernel uses the staged composite amount, the layer's mass over `Mᵈ`
# (`ρ Δz / Mᵈ` in the grid, `Δp / (g Mᵈ)` with Breeze's constants in the extension). The two
# agree to rounding and discretization; the reference is patched to the kernel's amount so
# the comparison stays bitwise.
function array_path_fluxes(rtm, i, j)
    columns = rtm.atmospheric_state
    gas_model = rtm.longwave_solver.gas_model
    FT = eltype(gas_model)
    N = number_of_layers(columns)
    ng_lw = length(gas_model.longwave_weights)
    ng_sw = length(gas_model.shortwave_weights)

    longwave = LongwaveOptics(zeros(FT, ng_lw, N), zeros(FT, ng_lw, N);
                              source_top = zeros(FT, ng_lw, N),
                              source_bottom = zeros(FT, ng_lw, N),
                              weights = zeros(FT, ng_lw))
    shortwave = ShortwaveOptics(zeros(FT, ng_sw, N); weights = zeros(FT, ng_sw))

    atmosphere = column_atmosphere(rtm, i, j)
    optical_properties!(longwave, shortwave, gas_model, atmosphere)

    air = atmosphere.gases.composite
    for k in 1:N, ig in 1:ng_sw
        shortwave.rayleigh_optical_depth[ig, k] = rayleigh_optical_depth(gas_model, ig, air[k])
    end

    Tₛ = atmosphere.surface.temperature
    ε = atmosphere.surface.emissivity
    μ₀ = atmosphere.geometry.cos_zenith
    S₀ = rtm.shortwave_solver.solar_constant
    α_direct = @allowscalar rtm.surface_radiation.direct_surface_albedo[i, j, 1]
    α_diffuse = @allowscalar rtm.surface_radiation.diffuse_surface_albedo[i, j, 1]

    longwave_bcs = LongwaveBoundaryConditions(surface_longwave_up = surface_longwave_emission(gas_model, Tₛ; emissivity = ε),
                                              surface_albedo = FT(1 - ε))
    shortwave_bcs = ShortwaveBoundaryConditions(toa_shortwave_down = FT(S₀ * max(μ₀, 0)),
                                                surface_albedo = FT(α_diffuse),
                                                surface_albedo_direct = FT(α_direct))

    fluxes = RadiativeFluxes(longwave_up = zeros(FT, N + 1), longwave_down = zeros(FT, N + 1),
                             shortwave_up = zeros(FT, N + 1), shortwave_down = zeros(FT, N + 1))
    radiative_fluxes!(fluxes, CloudlessLongwave(), longwave, atmosphere, longwave_bcs)
    radiative_fluxes!(fluxes, CloudlessShortwave(), shortwave, atmosphere, shortwave_bcs)

    return fluxes
end

function column_model(grid, radiation; humidity_factor = 1)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure = 101325, potential_temperature = 300)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, formulation = :LiquidIcePotentialTemperature, radiation)
    θ(x, y, z) = 300 + 0.01 * z / 1000
    qᵗ(x, y, z) = humidity_factor(x) * 0.015 * exp(-z / 2500)
    set!(model; θ, qᵗ)
    return model
end

# Both float types always: this is the test that pins the Float32 optics path bitwise
@testset "Kernel path versus array path [$(FT)]" for FT in all_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 12
    grid = RectilinearGrid(default_arch, FT; size = (2, 1, Nz), x = (0, 2), y = (0, 1), z = (0, 3kilometers),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants()

    for (extension, name) in ((ColumnExtension(FT; layers = 12), "extended"), (nothing, "grid only"))
        @testset "$name" begin
            radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants; column_extension = extension,
                                               surface_temperature = 300, surface_emissivity = 0.98,
                                               surface_albedo = 0.1, solar_position = FixedCosineZenith(0.5))
            # Column 2 is twice as humid as column 1
            model = column_model(grid, radiation; humidity_factor = x -> x < 1 ? 1 : 2)
            N = number_of_layers(radiation.atmospheric_state)
            faces = N + 2 .- (1:Nz+1)   # column interface of grid face k

            for i in 1:2
                reference = array_path_fluxes(radiation, i, 1)
                ℐ_lw_up = Array(interior(radiation.upwelling_longwave_flux))[i, 1, :]
                ℐ_lw_dn = Array(interior(radiation.downwelling_longwave_flux))[i, 1, :]
                ℐ_sw_up = Array(interior(radiation.upwelling_shortwave_flux))[i, 1, :]
                ℐ_sw_dn = Array(interior(radiation.downwelling_shortwave_flux))[i, 1, :]

                @test ℐ_lw_up == reference.longwave_up[faces]
                @test ℐ_lw_dn == -reference.longwave_down[faces]
                @test ℐ_sw_up == reference.shortwave_up[faces]
                @test ℐ_sw_dn == -reference.shortwave_down[faces]

                # The flux divergence is the face difference of the net flux
                F_net = ℐ_lw_up .+ ℐ_lw_dn .+ ℐ_sw_up .+ ℐ_sw_dn
                Δz = [Oceananigans.Operators.Δzᶜᶜᶜ(i, 1, k, grid) for k in 1:Nz]
                divergence = -(F_net[2:Nz+1] .- F_net[1:Nz]) ./ Δz
                @test Array(interior(radiation.flux_divergence))[i, 1, :] ≈ divergence rtol = 4 * eps(FT)
            end
        end
    end
end

Oceananigans.defaults.FloatType = Float64
