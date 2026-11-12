include(joinpath(@__DIR__, "setup.jl"))

#####
##### The subgrid supersaturation hook of P3's cloud condensation and evaporation, with a dummy model
#####

using Breeze
using Breeze.AtmosphereModels: total_density, NoSubgridCorrelation, ClosureCloud
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties, AerosolActivation, AerosolMode,
                                                       P3MicrophysicalState, ProcessRate, compute_p3_process_rates,
                                                       p3_grid_process_inputs, limit_vapor_rates,
                                                       subgrid_supersaturation, subgrid_supersaturation_fields,
                                                       subgrid_supersaturation_statistics
using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions, LiquidIcePotentialTemperatureState,
                             with_temperature, saturation_specific_humidity, PlanarLiquidSurface, density

using Enzyme: Enzyme
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

const PPP = PredictedParticleProperties

# A dummy subgrid model: Γ = `Γ` times the statistics, which are a field value read on the grid
struct ScaledSupersaturation{FT}
    Γ :: FT
end

PPP.subgrid_supersaturation_fields(::ScaledSupersaturation, model) = (; scale = model.velocities.w)

@inline PPP.subgrid_supersaturation_statistics(i, j, k, grid, ::ScaledSupersaturation, fields) =
    @inbounds 1 + fields.scale[i, j, k]

@inline PPP.subgrid_supersaturation(model::ScaledSupersaturation, scale, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, P, ρ, constants) =
    model.Γ * scale

# One P3 cell at temperature T and 900 hPa, with vapor qᵛ, cloud liquid qᶜˡ and rain qʳ
function cell_state(qᵛ, qᶜˡ, constants; T = 285.0, qʳ = 0.0)
    q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ, 0.0)
    return with_temperature(LiquidIcePotentialTemperatureState(0.0, q, 1e5, 9e4), T, constants)
end

# The saturation vapor mass fraction P3 evaluates in that cell (at the density of the state)
function cell_saturation(qᵛ, qᶜˡ, constants; T = 285.0, qʳ = 0.0)
    𝒰 = cell_state(qᵛ, qᶜˡ, constants; T, qʳ)
    return saturation_specific_humidity(T, density(𝒰, constants), constants, PlanarLiquidSurface())
end

function cell_rates(p3, qᵛ, qᶜˡ, constants; T = 285.0, qʳ = 0.0, statistics = nothing)
    ρ = 1.0
    𝒰 = cell_state(qᵛ, qᶜˡ, constants; T, qʳ)
    ℳ = P3MicrophysicalState(qᶜˡ, 2e8, qʳ, ifelse(qʳ > 0, 1e4, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, nothing, T, 0.0, 0.0, statistics)
end

with_subgrid(Γ; options...) = P3Microphysics(Float64; subgrid = ScaledSupersaturation(Γ), options...)

@testset "P3 subgrid supersaturation hook: defaults" begin
    p3 = P3Microphysics(Float64)
    @test p3.subgrid === NoSubgridCorrelation()
    for none in (NoSubgridCorrelation(), ClosureCloud())
        @test subgrid_supersaturation(none, nothing, 1e-2, 1.1e-2, 0.0, 285.0, 9e4, 1.1, nothing) === nothing
        @test subgrid_supersaturation_statistics(1, 1, 1, nothing, none, nothing) === nothing
        @test subgrid_supersaturation_fields(none, nothing) === nothing
    end
    @test PPP.cloud_vapor_excess(nothing, -0.0) === -0.0
    @test PPP.cloud_vapor_excess(1e-4, -3e-4) ≈ -2e-4

    # The ten-argument constructor of the scheme without a subgrid model
    rebuilt = PredictedParticlePropertiesMicrophysics(ntuple(n -> getfield(p3, n), Val(10))...)
    @test rebuilt.subgrid === NoSubgridCorrelation()
    @test typeof(rebuilt) === typeof(p3)

    # Without a subgrid model the statistics are ignored: the rates are plain P3's, bitwise
    constants = ThermodynamicConstants(Float64)
    qᵛ⁺ = cell_saturation(9.5e-3, 0.0, constants)
    for (qᵛ, qᶜˡ, qʳ) in ((qᵛ⁺ - 5e-4, 1e-4, 0.0), (qᵛ⁺ + 2e-4, 1e-4, 1e-5), (qᵛ⁺ - 5e-4, 0.0, 1e-5), (qᵛ⁺ + 2e-4, 0.0, 0.0))
        plain = cell_rates(p3, qᵛ, qᶜˡ, constants; qʳ)
        @test cell_rates(p3, qᵛ, qᶜˡ, constants; qʳ, statistics = nothing) === plain
        @test cell_rates(p3, qᵛ, qᶜˡ, constants; qʳ, statistics = 2.0) === plain
    end
end

@testset "P3 subgrid supersaturation hook: cloud condensation and evaporation" begin
    constants = ThermodynamicConstants(Float64)
    T = 285.0
    qᵛ⁺ = cell_saturation(9.5e-3, 0.0, constants; T)
    p3 = P3Microphysics(Float64)

    # A subsaturated cell with cloud liquid and rain: plain P3 evaporates both
    qᵛ, qᶜˡ, qʳ = qᵛ⁺ - 3e-4, 2e-4, 5e-5
    plain = cell_rates(p3, qᵛ, qᶜˡ, constants; T, qʳ)
    @test plain.condensation < 0
    @test plain.rain_evaporation > 0

    # A correlation larger than the mean deficit: cloud liquid condenses in the subsaturated cell,
    # and every process that is not the vapor exchange of cloud liquid keeps its grid-mean rate
    subgrid = cell_rates(with_subgrid(5e-4), qᵛ, qᶜˡ, constants; T, qʳ, statistics = 1.0)
    @test subgrid.condensation > 0
    for name in (:rain_evaporation, :rain_condensation, :autoconversion, :accretion, :cloud_self_collection,
                 :rain_self_collection, :deposition)
        @test getproperty(subgrid, name) === getproperty(plain, name)
    end

    # The statistics scale the correlation; a negative one evaporates faster than the grid mean
    @test cell_rates(with_subgrid(5e-4), qᵛ, qᶜˡ, constants; T, qʳ, statistics = 2.0).condensation > subgrid.condensation
    weaker = cell_rates(with_subgrid(1e-4), qᵛ, qᶜˡ, constants; T, qʳ, statistics = 1.0)
    @test plain.condensation < weaker.condensation < 0

    # A zero correlation is plain P3 where cloud liquid does not share a cap with rain
    for (qᵛ₀, qᶜˡ₀) in ((qᵛ⁺ - 3e-4, 2e-4), (qᵛ⁺ + 2e-4, 2e-4), (qᵛ⁺ + 2e-4, 0.0))
        reference = cell_rates(p3, qᵛ₀, qᶜˡ₀, constants; T)
        zeroed = cell_rates(with_subgrid(0.0), qᵛ₀, qᶜˡ₀, constants; T, statistics = 1.0)
        @test zeroed.condensation ≈ reference.condensation
        @test zeroed.ccn_activation_mass ≈ reference.ccn_activation_mass
    end

    # Condensation is capped by the vapor excess that cloud liquid sees, (qᵛ - qᵛ⁺ˡ + Γ) / ξ
    τ = p3.process_rates.sink_limiting_timescale
    for Γ in (4e-4, 5e-4, 2e-3)
        rates = cell_rates(with_subgrid(Γ), qᵛ, qᶜˡ, constants; T, qʳ, statistics = 1.0)
        excess = qᵛ - cell_saturation(qᵛ, qᶜˡ, constants; T, qʳ) + Γ
        @test 0 < (rates.condensation + rates.ccn_activation_mass) * τ ≤ excess * (1 + 1e-12)
    end

    # No cloud liquid: the seed mass of prescribed-number droplets starts condensation in the
    # subsaturated cell; aerosol activation, which predicts the number, stays grid-mean
    @test cell_rates(p3, qᵛ, 0.0, constants; T).ccn_activation_mass == 0
    @test cell_rates(with_subgrid(5e-4), qᵛ, 0.0, constants; T, statistics = 1.0).ccn_activation_mass > 0
    aerosol = AerosolActivation(AerosolMode(Float64))
    activated = cell_rates(with_subgrid(5e-4; aerosol), qᵛ, 0.0, constants; T, statistics = 1.0)
    @test activated.ccn_activation_mass == 0
    @test activated.ccn_activation_number == 0
end

@testset "P3 subgrid supersaturation hook: vapor caps" begin
    constants = ThermodynamicConstants(Float64)
    T, P = 285.0, 9e4
    qᵛ⁺ = saturation_specific_humidity(T, 1.1, constants, PlanarLiquidSurface())
    qᵛ = qᵛ⁺ - 3e-4
    Δt = 10.0
    limit(cond, ccn, rain_cond, rain_evap, Γ...) =
        limit_vapor_rates(cond, ccn, 0.0, rain_cond, rain_evap, 0.0, 0.0, 0.0, 0.0, 0.0,
                          qᵛ, qᵛ⁺, T, P, qᵛ, constants, Δt, 273.15, 0.0, 0.0, Γ...)

    # `nothing` is the grid-mean limiter, bitwise
    @test limit(1e-5, 1e-6, 0.0, 2e-6, nothing) === limit(1e-5, 1e-6, 0.0, 2e-6)

    # Grid mean: no condensation in a subsaturated cell. With Γ: condensation up to the cloud's cap,
    # shared with the seed mass, and rain evaporation capped by the grid-mean deficit as before
    plain = limit(1e-3, 1e-4, 1e-4, 1e-3)
    @test plain.cond == 0 && plain.ccn_activation_mass == 0 && plain.rain_cond == 0
    subgrid = limit(1e-3, 1e-4, 1e-4, 1e-3, 5e-4)
    @test subgrid.cond > 0 && subgrid.ccn_activation_mass > 0
    @test subgrid.cond / subgrid.ccn_activation_mass ≈ 10
    @test subgrid.rain_cond == 0
    @test subgrid.rain_evap === plain.rain_evap

    # Cloud evaporation is capped by the deficit that cloud liquid sees
    @test limit(-1e-3, 0.0, 0.0, 0.0, 5e-4).cond == 0
    @test limit(-1e-3, 0.0, 0.0, 0.0, 1e-4).cond ≈ limit(-1e-3, 0.0, 0.0, 0.0).cond * 2 / 3
end

reverse_gradient(f, x) = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.Reverse), Enzyme.Const(f),
                                         Enzyme.Active, Enzyme.Active(x))[1][1]
central_difference(f, x, h) = (f(x * (1 + h)) - f(x * (1 - h))) / (2x * h)

@testset "P3 subgrid supersaturation hook: Enzyme against finite differences" begin
    constants = ThermodynamicConstants(Float64)
    T = 285.0
    qᵛ⁺ = cell_saturation(9.5e-3, 0.0, constants; T)
    template = P3Microphysics(Float64)
    with_Γ(Γ) = PredictedParticlePropertiesMicrophysics(ntuple(n -> getfield(template, n), Val(10))...,
                                                        ScaledSupersaturation(Γ))
    # The correlation is not capped in these cells, so the rates depend on it
    for (qᵛ, qᶜˡ, Γ₀) in ((qᵛ⁺ - 3e-4, 2e-4, 1e-4), (qᵛ⁺ - 3e-4, 2e-4, 4e-4), (qᵛ⁺ + 1e-4, 2e-4, 2e-5)),
        name in (:condensation,)

        in_Γ = Γ -> getproperty(cell_rates(with_Γ(Γ), qᵛ, qᶜˡ, constants; T, statistics = 1.0), name)
        in_statistics = s -> getproperty(cell_rates(with_Γ(Γ₀), qᵛ, qᶜˡ, constants; T, statistics = s), name)
        in_vapor = q -> getproperty(cell_rates(with_Γ(Γ₀), q, qᶜˡ, constants; T, statistics = 1.0), name)
        for (f, x) in ((in_Γ, Γ₀), (in_statistics, 1.0), (in_vapor, qᵛ))
            ad = reverse_gradient(f, x)
            fd = central_difference(f, x, 1e-6)
            @test isfinite(ad)
            @test ad != 0
            @test ad ≈ fd rtol=1e-4
        end
    end
end

@testset "P3 subgrid supersaturation hook: model [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size = (2, 2, 8), x = (0, 200), y = (0, 200), z = (0, 800))

    function hook_model(subgrid)
        model = AtmosphereModel(grid; microphysics = P3Microphysics(FT; subgrid))
        cloud(x, y, z) = ifelse(z > 400, FT(2e-4), zero(FT))
        updraft(x, y, z) = sinpi(z / 800) / 2
        set!(model; θ = 285, qᵛ = 0.005, qᶜˡ = cloud, qʳ = 1e-5, nʳ = 1e4, w = updraft,
             enforce_mass_conservation = false)
        return model
    end

    plain = hook_model(NoSubgridCorrelation())
    subgrid = hook_model(ScaledSupersaturation(FT(8e-3)))
    Nz = size(grid, 3)

    # The process-rate diagnostics carry the fields of the subgrid model, and evaluate the same
    # rates as the tendency kernel
    column(operation) = @allowscalar Array(interior(compute!(Field(operation))))[1, 1, :]
    plain_condensation = MicrophysicalProcessRate(plain, :condensation)
    subgrid_condensation = MicrophysicalProcessRate(subgrid, :condensation)
    @test plain_condensation.kernel_function.subgrid_fields === nothing
    @test subgrid_condensation.kernel_function.subgrid_fields.scale === subgrid.velocities.w

    Sᵖ = column(plain_condensation)
    Sˢ = column(subgrid_condensation)
    @test all(Sᵖ[Nz÷2+2:Nz] .< 0)  # the subsaturated cloud evaporates
    @test all(Sˢ[Nz÷2+2:Nz] .> 0)  # the correlation exceeds the mean deficit: it condenses
    @test column(MicrophysicalProcessRate(subgrid, :rain_evaporation)) == column(MicrophysicalProcessRate(plain, :rain_evaporation))

    μ = subgrid.microphysical_fields
    ρ = total_density(subgrid.dynamics)
    constants = subgrid.thermodynamic_constants
    p3 = subgrid.microphysics
    fields = subgrid_supersaturation_fields(p3.subgrid, subgrid)
    for k in (2, Nz - 1)
        rates = @allowscalar begin
            inputs = p3_grid_process_inputs(1, 1, k, grid, p3, μ, subgrid.formulation, subgrid.dynamics,
                                            constants, ρ, subgrid.velocities, nothing, nothing)
            statistics = subgrid_supersaturation_statistics(1, 1, k, grid, p3.subgrid, fields)
            compute_p3_process_rates(p3, inputs.ρ, inputs.ℳ, inputs.𝒰, constants, inputs.properties,
                                     inputs.surface_temperature, inputs.temperature_tendency,
                                     inputs.pressure_tendency, statistics)
        end
        @test Sˢ[k] ≈ rates.condensation
    end

    # Time stepping: the subgrid model holds cloud liquid that plain P3 evaporates
    for model in (plain, subgrid)
        for _ in 1:5
            time_step!(model, 1)
        end
    end
    liquid(model) = @allowscalar Array(interior(model.microphysical_fields.ρqᶜˡ))
    @test all(isfinite, liquid(subgrid))
    @test sum(liquid(subgrid)) > sum(liquid(plain))
end
