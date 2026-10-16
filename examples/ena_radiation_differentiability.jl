# A bounded, radiation-only prerequisite for the ENA inverse problem.
# This requires the Breeze branch with BreezeNumericalRadiationExt, currently
# available in ../Breeze-radiation. Run from the Breeze.jl root with
# julia --project=../Breeze-radiation/test examples/ena_radiation_differentiability.jl
# Use --metal for the Float32 CPU/Metal comparison instead of reverse AD.
# No LES is integrated here. Both gas interpolation and liquid/ice optical
# depths participate in the derivative; tables and effective radii are fixed.
# These checks validate the radiation component, not the coupled P3 adjoint
# or an improvement against observations.

using Breeze
using Oceananigans
using NCDatasets
using Metal
using NumericalRadiation: NumericalRadiation
using Enzyme
using Printf
using Test

const ENARadiationExt = Base.get_extension(Breeze, :BreezeNumericalRadiationExt)
isnothing(ENARadiationExt) && error("Use a Breeze environment with BreezeNumericalRadiationExt enabled.")

function ena_record_radiation_configuration(directory)
    mkpath(directory)
    write(joinpath(directory, "configuration.txt"),
          "experiment = synthetic cloudy radiation column; no LES integration\n" *
          "layers = 16\ncolumn_top_m = 16000\noptics = climate_32x32, Mie liquid, Baum ice\n" *
          "effective_radii_m = liquid 1e-5, ice 3e-5 (fixed)\n" *
          "gradient = Enzyme reverse mode on CPU Float64\nmetal = forward flux comparison, Float32\n" *
          "julia = $VERSION\nBreeze = $(pkgversion(Breeze)), $(pkgdir(Breeze))\n" *
          "NumericalRadiation = $(pkgversion(NumericalRadiation)), $(pkgdir(NumericalRadiation))\n" *
          "Enzyme = $(pkgversion(Enzyme))\nMetal = $(pkgversion(Metal))\n")
    return nothing
end

function ena_radiation_column(architecture, FT)
    grid = RectilinearGrid(architecture, FT; size=16, z=(0, 16000), topology=(Flat, Flat, Bounded))
    radiation = RadiativeTransferModel(grid, EcCKDOptics(clouds=CloudScatteringTables()),
                                      ThermodynamicConstants(FT);
                                      surface_temperature=FT(294.8), surface_emissivity=FT(0.98),
                                      surface_albedo=FT(0.06), solar_position=FixedCosineZenith(FT(0.6)),
                                      column_extension=nothing)
    return grid, radiation
end

# The layer sweep is also suitable for use inside a column kernel. Temperatures
# avoid interpolation knots. Pressure and layer masses are held fixed, as in an
# anelastic radiation update. This is a synthetic column, not an ENA sounding.
@inline function ena_stage_radiation_column!(columns, controls)
    FT = eltype(columns)
    N = size(columns.pressure_layers, 2)
    @inbounds for k in 1:N
        z = FT(1000) * (N - k + FT(0.5))
        temperature = max(FT(205.37), FT(294.37) - FT(0.006) * z)
        pressure = FT(101300) * exp(-z / FT(8000))
        mass = pressure / (FT(287) * temperature) * FT(1000)
        columns.pressure_layers[1, k] = pressure
        columns.temperature_layers[1, k] = temperature + controls[1]
        columns.dry_air[1, k] = mass / FT(0.02897)
        columns.water_vapor[1, k] = mass * FT(0.012) * exp(-z / FT(2000)) * exp(FT(0.1) * controls[2]) / FT(0.018015)
        columns.ozone[1, k] = columns.dry_air[1, k] * FT(5e-8)
        columns.liquid_water_path[1, k] = ifelse(k == N-1, FT(0.08) * exp(FT(0.2) * controls[3]), FT(0))
        columns.ice_water_path[1, k] = ifelse(k == N-7, FT(0.01) * exp(FT(0.2) * controls[4]), FT(0))
    end
    @inbounds for k in 1:N+1
        z = FT(1000) * (N + 1 - k)
        columns.pressure_interfaces[1, k] = FT(101300) * exp(-z / FT(8000))
        columns.temperature_interfaces[1, k] = max(FT(205.37), FT(294.37) - FT(0.006) * z) + controls[1]
    end
    columns.cos_zenith[1] = FT(0.6)
    return nothing
end

function ena_radiation_objective!(columns, controls, longwave, shortwave)
    FT = eltype(columns)
    N = size(columns.pressure_layers, 2)
    ena_stage_radiation_column!(columns, controls)
    gas = longwave.gas_model
    ENARadiationExt.stage_column_stencils!(columns, gas, 1, N)
    bracket(cloud, radius) = NumericalRadiation.effective_radius_bracket(cloud, FT(radius))
    liquid, ice = longwave.cloud.liquid, longwave.cloud.ice
    optics = ENARadiationExt.LongwaveLayerOptics(gas, columns, longwave.mole_fractions,
               liquid, ice, bracket(liquid, 10e-6), bracket(ice, 30e-6), 1)
    emission = NumericalRadiation.TabulatedSurfaceEmission(gas, FT(294.8) + controls[5]; emissivity=FT(0.98))
    NumericalRadiation.streaming_longwave_fluxes!(view(columns.longwave_up, 1, :), view(columns.longwave_down, 1, :),
        optics, emission, 1-FT(0.98), FT(0), gas.longwave_weights, length(gas.longwave_weights), N,
        view(columns.transmittance, 1, :), view(columns.source_up, 1, :))
    liquid, ice = shortwave.cloud.liquid, shortwave.cloud.ice
    optics = ENARadiationExt.ShortwaveLayerOptics(gas, columns, longwave.mole_fractions,
               liquid, ice, bracket(liquid, 10e-6), bracket(ice, 30e-6), 1)
    NumericalRadiation.streaming_shortwave_fluxes!(view(columns.shortwave_up, 1, :), view(columns.shortwave_down, 1, :),
        optics, FT(0.6), shortwave.solar_constant * FT(0.6), FT(0.06), FT(0.06),
        gas.shortwave_weights, length(gas.shortwave_weights), N, ENARadiationExt.column_scratch(columns.shortwave, 1))
    # Include local flux divergence, the quantity that heats the LES, as well
    # as boundary fluxes, so compensating layer errors cannot cancel entirely.
    loss = columns.longwave_up[1, 1] + FT(0.37) * columns.longwave_down[1, end] +
           FT(0.23) * columns.shortwave_up[1, 1] + FT(0.11) * columns.shortwave_down[1, end]
    @inbounds for k in 1:N
        net_top = columns.longwave_up[1,k] - columns.longwave_down[1,k] + columns.shortwave_up[1,k] - columns.shortwave_down[1,k]
        net_bottom = columns.longwave_up[1,k+1] - columns.longwave_down[1,k+1] + columns.shortwave_up[1,k+1] - columns.shortwave_down[1,k+1]
        loss += FT(0.001) * (net_bottom - net_top)^2
    end
    return loss
end

function ena_check_radiation_gradient(; directory=joinpath(@__DIR__, "ena_output", "optimization", "radiation"))
    @info "Constructing a 16-layer cloudy ecCKD column (32 LW + 32 SW g-points)"
    grid, radiation = ena_radiation_column(CPU(), Float64)
    columns = radiation.atmospheric_state
    longwave, shortwave = radiation.longwave_solver, radiation.shortwave_solver
    controls = zeros(5)
    objective(p) = ena_radiation_objective!(columns, p, longwave, shortwave)
    value = objective(controls)
    @info "Compiling reverse derivative" value
    dcolumns = Enzyme.make_zero(columns)
    gradient = zero(controls)
    elapsed = @elapsed Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.Reverse),
        ena_radiation_objective!, Enzyme.Active, Enzyme.Duplicated(columns, dcolumns),
        Enzyme.Duplicated(controls, gradient), Enzyme.Const(longwave), Enzyme.Const(shortwave))
    @info "Reverse derivative completed" elapsed gradient
    ena_record_radiation_configuration(directory)
    names = ("air_temperature", "water_vapor", "liquid_water", "ice_water", "surface_temperature")
    @testset "Cloudy ecCKD reverse derivative" begin
        @test all(isfinite, gradient)
        @test all(x -> abs(x) > 1e-6, gradient)
        open(joinpath(directory, "gradient_check.csv"), "w") do io
            println(io, "control,step,automatic_derivative,finite_difference,relative_error")
            for n in eachindex(controls), step in (1e-2, 1e-3, 1e-4)
                plus, minus = copy(controls), copy(controls)
                plus[n] += step
                minus[n] -= step
                finite_difference = (objective(plus) - objective(minus)) / (2step)
                relative_error = abs(gradient[n] - finite_difference) / max(abs(gradient[n]), 1e-10)
                @printf(io, "%s,%.8g,%.12g,%.12g,%.8g\n", names[n], step, gradient[n], finite_difference, relative_error)
                @test isapprox(gradient[n], finite_difference; rtol=1e-4, atol=1e-6)
            end
        end
    end
    return gradient
end

function ena_check_radiation_metal(; directory=joinpath(@__DIR__, "ena_output", "optimization", "radiation"))
    Metal.functional() || error("Metal is unavailable")
    grid, radiation = ena_radiation_column(CPU(), Float32)
    controls = zeros(Float32, 5)
    value = ena_radiation_objective!(radiation.atmospheric_state, controls, radiation.longwave_solver, radiation.shortwave_solver)
    flux_names = (:longwave_up, :longwave_down, :shortwave_up, :shortwave_down)
    reference = map(name -> copy(getproperty(radiation.atmospheric_state, name)), flux_names)
    ENARadiationExt.solve_spectral_columns!(radiation, grid)
    @testset "Streaming reference matches CPU column kernel" begin
        for (name, expected) in zip(flux_names, reference)
            @test getproperty(radiation.atmospheric_state, name) ≈ expected rtol=5e-6
        end
    end
    device_grid, device_radiation = ena_radiation_column(GPU(Metal.MetalBackend()), Float32)
    source, destination = radiation.atmospheric_state, device_radiation.atmospheric_state
    for name in (:pressure_layers, :temperature_layers, :pressure_interfaces, :temperature_interfaces,
                 :dry_air, :water_vapor, :ozone, :liquid_water_path, :ice_water_path, :cos_zenith)
        copyto!(getproperty(destination, name), getproperty(source, name))
    end
    @info "Compiling and running native Metal column kernel" value
    elapsed = @elapsed begin
        ENARadiationExt.solve_spectral_columns!(device_radiation, device_grid)
        Metal.synchronize()
    end
    ena_record_radiation_configuration(directory)
    @testset "Cloudy ecCKD Metal fluxes" begin
        open(joinpath(directory, "metal_flux_check.csv"), "w") do io
            println(io, "flux,max_absolute_error_W_m2,max_relative_error")
            for (name, expected) in zip(flux_names, reference)
                actual = Array(getproperty(destination, name))
                error = maximum(abs, actual - expected)
                relative_error = error / maximum(abs, expected)
                println(io, "$name,$error,$relative_error")
                @test all(isfinite, actual)
                @test isapprox(actual, expected; rtol=2e-5, atol=2e-3)
            end
        end
    end
    @info "Metal comparison completed" elapsed
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    "--metal" in ARGS ? ena_check_radiation_metal() : ena_check_radiation_gradient()
end
