include(joinpath(@__DIR__, "setup.jl"))

#####
##### The subgrid supersaturation correlation Γ in P3's cloud condensation and evaporation
#####

using Breeze
using Breeze.AtmosphereModels: AtmosphereModels, total_density, PrescribedStatistics
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties, AerosolActivation, AerosolMode,
                                                       GridMeanCondensation, SubgridCondensation,
                                                       P3MicrophysicalState, compute_p3_process_rates,
                                                       p3_grid_process_inputs, limit_vapor_rates,
                                                       cloud_condensation_correlation,
                                                       subgrid_supersaturation_fields,
                                                       subgrid_supersaturation_statistics
using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions, LiquidIcePotentialTemperatureState,
                             with_temperature, saturation_specific_humidity, PlanarLiquidSurface, density

using Enzyme: Enzyme
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

const PPP = PredictedParticleProperties

# A test-only source of statistics: a Gaussian whose spread is `σ` times (1 + w), read on the grid
struct VelocityScaledStatistics{FT}
    σ :: FT
end

@inline function AtmosphereModels.supersaturation_statistics(statistics::VelocityScaledStatistics, i, j, k, grid,
                                                             closure, closure_fields, model_fields)
    FT = eltype(grid)
    σ = @inbounds statistics.σ * (1 + model_fields.velocities.w[i, j, k])
    return (; μ = zero(FT), σ = convert(FT, σ), skewness = zero(FT), f⁺ = one(FT) / 2)
end

gaussian(σ) = (; μ = zero(σ), σ, skewness = zero(σ), f⁺ = one(σ) / 2)

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

const subgrid_p3 = P3Microphysics(Float64; condensation = SubgridCondensation(PrescribedStatistics(0.0)))

subgrid_rates(σ, qᵛ, qᶜˡ, constants; p3 = subgrid_p3, kw...) =
    cell_rates(p3, qᵛ, qᶜˡ, constants; statistics = gaussian(σ), kw...)

@testset "P3 subgrid supersaturation: defaults" begin
    p3 = P3Microphysics(Float64)
    @test p3.condensation === GridMeanCondensation()
    @test subgrid_supersaturation_statistics(1, 1, 1, nothing, GridMeanCondensation(), nothing) === nothing
    @test subgrid_supersaturation_fields(GridMeanCondensation(), nothing) === nothing
    @test cloud_condensation_correlation(GridMeanCondensation(), gaussian(1e-3), 1e-2, 1.1e-2, 0.0,
                                         nothing, 285.0, 9e4, 1.1, nothing) === nothing
    @test PPP.cloud_vapor_excess(nothing, -0.0) === -0.0
    @test PPP.cloud_vapor_excess(1e-4, -3e-4) ≈ -2e-4

    # The ten-argument constructor of the scheme condenses at the grid mean
    rebuilt = PredictedParticlePropertiesMicrophysics(ntuple(n -> getfield(p3, n), Val(10))...)
    @test rebuilt.condensation === GridMeanCondensation()
    @test typeof(rebuilt) === typeof(p3)

    # With grid-mean condensation the statistics are ignored: the rates are plain P3's, bitwise
    constants = ThermodynamicConstants(Float64)
    qᵛ⁺ = cell_saturation(9.5e-3, 0.0, constants)
    for (qᵛ, qᶜˡ, qʳ) in ((qᵛ⁺ - 5e-4, 1e-4, 0.0), (qᵛ⁺ + 2e-4, 1e-4, 1e-5), (qᵛ⁺ - 5e-4, 0.0, 1e-5), (qᵛ⁺ + 2e-4, 0.0, 0.0))
        plain = cell_rates(p3, qᵛ, qᶜˡ, constants; qʳ)
        @test cell_rates(p3, qᵛ, qᶜˡ, constants; qʳ, statistics = nothing) === plain
        @test cell_rates(p3, qᵛ, qᶜˡ, constants; qʳ, statistics = gaussian(2e-3)) === plain
    end
end

@testset "P3 subgrid supersaturation: cloud condensation and evaporation" begin
    constants = ThermodynamicConstants(Float64)
    T = 285.0
    qᵛ⁺ = cell_saturation(9.5e-3, 0.0, constants; T)
    p3 = P3Microphysics(Float64)

    # A subsaturated cell with cloud liquid and rain: plain P3 evaporates both
    qᵛ, qᶜˡ, qʳ = qᵛ⁺ - 3e-4, 2e-4, 5e-5
    plain = cell_rates(p3, qᵛ, qᶜˡ, constants; T, qʳ)
    @test plain.condensation < 0
    @test plain.rain_evaporation > 0

    # A spread wide enough that the cell-mean positive saturation excess exceeds what the cloud
    # liquid holds: cloud liquid condenses in the subsaturated cell, and every process that is not
    # the vapor exchange of cloud liquid keeps its grid-mean rate
    subgrid = subgrid_rates(3e-3, qᵛ, qᶜˡ, constants; T, qʳ)
    @test subgrid.condensation > 0
    for name in (:rain_evaporation, :rain_condensation, :autoconversion, :accretion, :cloud_self_collection,
                 :rain_self_collection, :deposition)
        @test getproperty(subgrid, name) === getproperty(plain, name)
    end

    # A wider spread condenses faster; a narrow one still evaporates
    @test subgrid_rates(6e-3, qᵛ, qᶜˡ, constants; T, qʳ).condensation > subgrid.condensation
    @test subgrid_rates(1e-5, qᵛ, qᶜˡ, constants; T, qʳ).condensation < 0

    # Condensation is capped by the vapor excess that cloud liquid sees, (qᵛ - qᵛ⁺ˡ + Γ) / ξ
    τ = p3.process_rates.sink_limiting_timescale
    qᵛ⁺ˡ = cell_saturation(qᵛ, qᶜˡ, constants; T, qʳ)
    for σ in (3e-3, 6e-3, 2e-2)
        rates = subgrid_rates(σ, qᵛ, qᶜˡ, constants; T, qʳ)
        Γ = cloud_condensation_correlation(subgrid_p3.condensation, gaussian(σ), qᵛ, qᵛ⁺ˡ, qᶜˡ,
                                           MoistureMassFractions(qᵛ, qᶜˡ + qʳ, 0.0), T, 9e4, 1.0, constants)
        excess = qᵛ - qᵛ⁺ˡ + Γ
        @test 0 < (rates.condensation + rates.ccn_activation_mass) * τ ≤ excess * (1 + 1e-12)
    end

    # No cloud liquid: the seed mass of prescribed-number droplets starts condensation in the
    # subsaturated cell; aerosol activation, which predicts the number, stays grid-mean
    @test cell_rates(p3, qᵛ, 0.0, constants; T).ccn_activation_mass == 0
    @test subgrid_rates(3e-3, qᵛ, 0.0, constants; T).ccn_activation_mass > 0
    aerosol = AerosolActivation(AerosolMode(Float64))
    activated_p3 = P3Microphysics(Float64; aerosol, condensation = SubgridCondensation(PrescribedStatistics(0.0)))
    activated = subgrid_rates(3e-3, qᵛ, 0.0, constants; T, p3 = activated_p3)
    @test activated.ccn_activation_mass == 0
    @test activated.ccn_activation_number == 0
end

#####
##### Cloud starts from the subgrid distribution over the stage
#####

# A mean-subsaturated cell without cloud liquid whose subgrid distribution is saturated in part:
# the droplets' relaxation rate is zero, so the seed of activation is what starts cloud. The
# implicit relaxation over the stage must seed it as the explicit rates do, and the seeded droplets
# grow within the same stage.
@testset "P3 subgrid supersaturation: cloud starts in a subsaturated cell without cloud [$processes]" for
        processes in (PPP.CloudCondensationOnly(), PPP.AllProcesses())
    constants = ThermodynamicConstants(Float64)
    T = 285.0
    qᵛ⁺ = cell_saturation(9.5e-3, 0.0, constants; T)
    qᵛ = qᵛ⁺ - 3e-4
    qᵛ⁺ˡ = cell_saturation(qᵛ, 0.0, constants; T)
    @test qᵛ < qᵛ⁺ˡ
    σ = 3e-3
    p3 = P3Microphysics(Float64; condensation = SubgridCondensation(PrescribedStatistics(σ); processes))
    plain = P3Microphysics(Float64)
    seed = p3.cloud.number_concentration * PPP.activated_droplet_mass(p3.process_rates, Float64)
    ρ = 1.0
    𝒰 = cell_state(qᵛ, 0.0, constants; T)
    ℳ = P3MicrophysicalState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rates(p3, stage) = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, nothing, T, 0.0, 0.0, gaussian(σ), stage)

    @test rates(p3, nothing).ccn_activation_mass > 0  # the explicit rates
    for Δt in (1.0, 10.0, 1e3, 1e5)
        stage = PPP.LiquidRelaxationStage(Δt, 0.0, 0.0, true)
        @test rates(plain, stage).ccn_activation_mass == 0  # the grid mean is subsaturated
        implicit = rates(p3, stage)
        @test 0 < Δt * implicit.ccn_activation_mass ≤ seed * (1 + 1e-12)
        @test implicit.condensation > 0  # the seeded droplets grow within the stage
        @test Δt * (implicit.condensation + implicit.ccn_activation_mass) ≤ qᵛ
    end
end

@testset "P3 subgrid supersaturation: cloud forms in a subsaturated column without cloud [$processes]" for
        processes in (PPP.CloudCondensationOnly(), PPP.AllProcesses())
    grid = RectilinearGrid(default_arch, Float64; size = 32, z = (0, 3200), topology = (Flat, Flat, Bounded))
    reference_state = ReferenceState(grid; base_pressure = 101500, potential_temperature = 290)
    function liquid_after_steps(condensation)
        model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state),
                                microphysics = P3Microphysics(Float64; condensation))
        set!(model; θ = z -> 290 + 2e-3 * z + 2 * (1 + tanh((z - 1500) / 150)),
                    qᵗ = z -> 8e-3 - 1e-6 * z - 2e-3 * (1 + tanh((z - 1500) / 150)))
        liquid() = @allowscalar Array(interior(model.microphysical_fields.ρqᶜˡ))
        @test all(iszero, liquid())
        for _ in 1:20
            time_step!(model, 10)
        end
        return liquid()
    end
    # The grid mean stays subsaturated: grid-mean P3 forms no cloud, subgrid condensation does
    @test all(iszero, liquid_after_steps(GridMeanCondensation()))
    subgrid = liquid_after_steps(SubgridCondensation(PrescribedStatistics(2e-3); processes))
    @test all(isfinite, subgrid)
    @test all(≥(0), subgrid)
    @test maximum(subgrid) > 1e-5
end

@testset "P3 subgrid supersaturation: vapor caps" begin
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

# Runtime activity: on Julia 1.11 Enzyme cannot prove statically that the rebuilt scheme's constant storage is inactive
reverse_gradient(f, x) = Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.set_strong_zero(Enzyme.Reverse)), Enzyme.Const(f),
                                         Enzyme.Active, Enzyme.Active(x))[1][1]
central_difference(f, x, h) = (f(x * (1 + h)) - f(x * (1 - h))) / (2x * h)

@testset "P3 subgrid supersaturation: Enzyme against finite differences" begin
    constants = ThermodynamicConstants(Float64)
    T = 285.0
    qᵛ⁺ = cell_saturation(9.5e-3, 0.0, constants; T)
    # The correlation is not capped in these cells, so the rates depend on it
    for (qᵛ, qᶜˡ, σ₀) in ((qᵛ⁺ - 3e-4, 2e-4, 1e-4), (qᵛ⁺ - 3e-4, 2e-4, 4e-4), (qᵛ⁺ + 1e-4, 2e-4, 2e-4))
        in_spread = σ -> subgrid_rates(σ, qᵛ, qᶜˡ, constants; T).condensation
        in_vapor = q -> subgrid_rates(σ₀, q, qᶜˡ, constants; T).condensation
        for (f, x) in ((in_spread, σ₀), (in_vapor, qᵛ))
            ad = reverse_gradient(f, x)
            fd = central_difference(f, x, 1e-6)
            @test isfinite(ad)
            @test ad != 0
            @test ad ≈ fd rtol=1e-4
        end
    end
end

@testset "P3 subgrid supersaturation: model [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size = (2, 2, 8), x = (0, 200), y = (0, 200), z = (0, 800))

    function column_model(condensation)
        model = AtmosphereModel(grid; microphysics = P3Microphysics(FT; condensation))
        cloud(x, y, z) = ifelse(z > 400, FT(2e-4), zero(FT))
        updraft(x, y, z) = sinpi(z / 800) / 2
        set!(model; θ = 285, qᵛ = 0.005, qᶜˡ = cloud, qʳ = 1e-5, nʳ = 1e4, w = updraft,
             enforce_mass_conservation = false)
        return model
    end

    plain = column_model(GridMeanCondensation())
    subgrid = column_model(SubgridCondensation(VelocityScaledStatistics(FT(8e-3))))
    Nz = size(grid, 3)

    # The process-rate diagnostics carry the fields of the condensation option, and evaluate the
    # same rates as the tendency kernel
    column(operation) = @allowscalar Array(interior(compute!(Field(operation))))[1, 1, :]
    plain_condensation = MicrophysicalProcessRate(plain, :condensation)
    subgrid_condensation = MicrophysicalProcessRate(subgrid, :condensation)
    @test plain_condensation.kernel_function.condensation_fields === nothing
    @test subgrid_condensation.kernel_function.condensation_fields[3].velocities.w === subgrid.velocities.w

    Sᵖ = column(plain_condensation)
    Sˢ = column(subgrid_condensation)
    @test all(Sᵖ[Nz÷2+2:Nz] .< 0)  # the subsaturated cloud evaporates
    @test all(Sˢ[Nz÷2+2:Nz] .> 0)  # the spread exceeds the mean deficit: it condenses
    @test column(MicrophysicalProcessRate(subgrid, :rain_evaporation)) == column(MicrophysicalProcessRate(plain, :rain_evaporation))

    μ = subgrid.microphysical_fields
    ρ = total_density(subgrid.dynamics)
    constants = subgrid.thermodynamic_constants
    p3 = subgrid.microphysics
    fields = subgrid_supersaturation_fields(p3.condensation, subgrid)
    for k in (2, Nz - 1)
        rates = @allowscalar begin
            inputs = p3_grid_process_inputs(1, 1, k, grid, p3, μ, subgrid.formulation, subgrid.dynamics,
                                            constants, ρ, subgrid.velocities)
            statistics = subgrid_supersaturation_statistics(1, 1, k, grid, p3.condensation, fields)
            compute_p3_process_rates(p3, inputs.ρ, inputs.ℳ, inputs.𝒰, constants, inputs.properties,
                                     inputs.surface_temperature, inputs.temperature_tendency,
                                     inputs.pressure_tendency, statistics)
        end
        @test Sˢ[k] ≈ rates.condensation
    end

    # Time stepping: subgrid condensation holds cloud liquid that grid-mean P3 evaporates
    for model in (plain, subgrid)
        for _ in 1:5
            time_step!(model, 1)
        end
    end
    liquid(model) = @allowscalar Array(interior(model.microphysical_fields.ρqᶜˡ))
    @test all(isfinite, liquid(subgrid))
    @test sum(liquid(subgrid)) > sum(liquid(plain))
end

#####
##### A statistics source that supplies its own Γ: a uniform distribution of the saturation excess
#####

# The saturation excess s is uniform on [s̄ - h, s̄ + h]: ⟨s⁺⟩ = (s̄ + h)² / 4h inside the spread,
# so Γ = ⟨s⁺⟩ - s̄ = (h - s̄)² / 4h there, max(0, -s̄) outside.
struct UniformStatistics{FT} <: AtmosphereModels.AbstractSupersaturationStatistics
    h :: FT
end

@inline function AtmosphereModels.supersaturation_statistics(statistics::UniformStatistics, i, j, k, grid,
                                                             closure, closure_fields, model_fields)
    h = statistics.h
    return (; μ = zero(h), σ = h / sqrt(3 * one(h)), skewness = zero(h), f⁺ = one(h) / 2)
end

@inline function PPP.vapor_excess_correlation(statistics::UniformStatistics, distribution, cell)
    s̄ = cell.saturation_excess
    h = convert(typeof(s̄), statistics.h)
    inside = (h - s̄)^2 / (4h)
    return ifelse(abs(s̄) < h, inside, max(zero(s̄), -s̄))
end

uniform_p3(h) = P3Microphysics(Float64; condensation = SubgridCondensation(UniformStatistics(h)))

function uniform_rates(h, qᵛ, qᶜˡ, constants; kw...)
    statistics = AtmosphereModels.supersaturation_statistics(UniformStatistics(h), 1, 1, 1, nothing, nothing, nothing, nothing)
    return cell_rates(uniform_p3(h), qᵛ, qᶜˡ, constants; statistics, kw...)
end

@testset "P3 subgrid supersaturation: a statistics source supplies Γ" begin
    constants = ThermodynamicConstants(Float64)
    T, p, ρ = 285.0, 9e4, 1.0
    qᵛ⁺ = cell_saturation(9.5e-3, 0.0, constants; T)
    qᵛ, qᶜˡ = qᵛ⁺ - 3e-4, 2e-4
    q = MoistureMassFractions(qᵛ, qᶜˡ, 0.0)
    qᵛ⁺ˡ = cell_saturation(qᵛ, qᶜˡ, constants; T)
    ξˡ = PPP.p3_liquid_psychrometric_correction(qᵛ⁺ˡ, q, T, constants)
    s̄ = qᵛ - qᵛ⁺ˡ + ξˡ * qᶜˡ

    # The Gaussian default is the truncation about s̄, bitwise
    for σ in (0.0, 1e-4, 3e-3)
        g = gaussian(σ)
        @test cloud_condensation_correlation(SubgridCondensation(PrescribedStatistics(σ)), g, qᵛ, qᵛ⁺ˡ, qᶜˡ, q, T, p, ρ, constants) ===
              PPP.subgrid_vapor_excess_correlation(g, s̄)
    end

    # The supplied Γ reaches the condensation rate
    h = 3e-3
    distribution = AtmosphereModels.supersaturation_statistics(UniformStatistics(h), 1, 1, 1, nothing, nothing, nothing, nothing)
    Γᵘ = cloud_condensation_correlation(SubgridCondensation(UniformStatistics(h)), distribution, qᵛ, qᵛ⁺ˡ, qᶜˡ, q, T, p, ρ, constants)
    @test Γᵘ ≈ (h - s̄)^2 / 4h
    @test Γᵘ != PPP.subgrid_vapor_excess_correlation(distribution, s̄)
    uniform = uniform_rates(h, qᵛ, qᶜˡ, constants; T)
    same_spread = subgrid_rates(distribution.σ, qᵛ, qᶜˡ, constants; T)
    plain = cell_rates(P3Microphysics(Float64), qᵛ, qᶜˡ, constants; T)
    @test uniform.condensation != same_spread.condensation
    @test uniform.condensation > plain.condensation
    @test uniform.rain_evaporation === plain.rain_evaporation
    # A narrow spread is the grid mean, s̄ outside the spread
    @test uniform_rates(1e-6, qᵛ, qᶜˡ, constants; T).condensation < 0

    # Enzyme differentiates through the supplied Γ (the scheme is rebuilt around the statistics
    # without reading the lookup tables)
    base = uniform_p3(h)
    function in_width(x)
        p3 = PredictedParticlePropertiesMicrophysics(ntuple(n -> getfield(base, n), Val(10))...,
                                                     SubgridCondensation(UniformStatistics(x)))
        statistics = AtmosphereModels.supersaturation_statistics(UniformStatistics(x), 1, 1, 1, nothing, nothing, nothing, nothing)
        return cell_rates(p3, qᵛ, qᶜˡ, constants; T, statistics).condensation
    end
    ad = reverse_gradient(in_width, h)
    fd = central_difference(in_width, h, 1e-6)
    @test isfinite(ad) && ad != 0
    @test ad ≈ fd rtol = 1e-4
end

@testset "P3 subgrid supersaturation: the implicit relaxation targets the supplied Γ" begin
    constants = ThermodynamicConstants(Float64)
    T = 285.0
    qᵛ⁺ = cell_saturation(9.5e-3, 0.0, constants; T)
    qᵛ, qᶜˡ = qᵛ⁺ - 3e-4, 2e-4
    qᵛ⁺ˡ = cell_saturation(qᵛ, qᶜˡ, constants; T)
    h = 3e-3
    p3 = uniform_p3(h)
    distribution = AtmosphereModels.supersaturation_statistics(UniformStatistics(h), 1, 1, 1, nothing, nothing, nothing, nothing)
    ρ = 1.0
    𝒰 = cell_state(qᵛ, qᶜˡ, constants; T)
    ℳ = P3MicrophysicalState(qᶜˡ, 2e8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Δt = 1e4  # ωˡ Δt ≫ 1: one stage reaches the target
    stage = PPP.LiquidRelaxationStage(Δt, qᶜˡ, 2e8, true)
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, nothing, T, 0.0, 0.0, distribution, stage)
    q = 𝒰.moisture_mass_fractions
    Γ = cloud_condensation_correlation(p3.condensation, distribution, qᵛ, qᵛ⁺ˡ, qᶜˡ, q, T, 9e4, ρ, constants)
    # Γ is formed about the saturation excess with the psychrometric correction of the subgrid distribution, which
    # divides the latent heating by the mixture heat capacity: s̄ + Γ = ⟨s⁺⟩
    ξˡ = PPP.p3_liquid_psychrometric_correction(qᵛ⁺ˡ, q, T, constants)
    s̄ = qᵛ - qᵛ⁺ˡ + ξˡ * qᶜˡ
    positive_excess = (s̄ + h)^2 / (4h)
    @test s̄ + Γ ≈ positive_excess
    # The implicit relaxation forms its target with the psychrometric correction of the coupled rates, which divides
    # by the heat capacity of dry air
    ξᶜ = PPP.liquid_psychrometric_correction(constants, PPP.vaporization_latent_heat(constants, T), qᵛ⁺ˡ, T)
    target = PPP.liquid_equilibrium_condensate(Γ, qᵛ - qᵛ⁺ˡ + ξᶜ * qᶜˡ, ξᶜ)
    @test qᶜˡ + Δt * (rates.condensation + rates.ccn_activation_mass) ≈ target rtol = 1e-3
    # That target is (⟨s⁺⟩ + (ξᶜ - ξˡ) qᶜˡ) / ξᶜ: it departs from the equilibrium condensate ⟨s⁺⟩ / ξˡ of the
    # distribution by the relative difference of the two corrections times |ξˡ qᶜˡ / ⟨s⁺⟩ - 1|
    @test ξᶜ != ξˡ
    @test target ≈ positive_excess / ξˡ rtol = abs(ξᶜ - ξˡ) / ξᶜ * abs(ξˡ * qᶜˡ / positive_excess - 1) * (1 + 1e-6)
end

#####
##### The fixed condensation timescale
#####

@testset "P3 cloud relaxation: fixed condensation timescale" begin
    constants = ThermodynamicConstants(Float64)
    T = 285.0
    @test P3Microphysics(Float64).cloud_relaxation === PPP.DiffusionalGrowthRelaxation()
    @test_throws ArgumentError P3Microphysics(Float64; cloud_relaxation = 5.0)
    rebuilt = PredictedParticlePropertiesMicrophysics(ntuple(n -> getfield(subgrid_p3, n), Val(11))...)
    @test rebuilt.cloud_relaxation === PPP.DiffusionalGrowthRelaxation()

    τ = 7.0
    cloud = PPP.CloudDroplets(Float64; condensation_timescale = τ)
    fixed = P3Microphysics(Float64; cloud, cloud_relaxation = PPP.CondensationTimescaleRelaxation())
    physical = P3Microphysics(Float64; cloud)

    qᵛ⁺ = cell_saturation(9.5e-3, 0.0, constants; T)
    for (qᵛ, qᶜˡ) in ((qᵛ⁺ + 1e-4, 2e-4), (qᵛ⁺ - 1e-4, 2e-4), (qᵛ⁺ + 1e-4, 0.0))
        𝒰 = cell_state(qᵛ, qᶜˡ, constants; T)
        ℳ = P3MicrophysicalState(qᶜˡ, 2e8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ω = PPP.p3_phase_relaxation_rates(fixed, 1.0, ℳ, 𝒰, constants)
        @test ω.ωˡ === 1 / τ
        @test ω.ωᶜ === 1 / τ
        @test PPP.p3_phase_relaxation_rates(physical, 1.0, ℳ, 𝒰, constants).ωˡ != 1 / τ
    end

    # The condensation rate relaxes the vapor excess at 1/τ (no rain or ice; a short sink-limiting
    # time, over which the coupled solver's transient response is the relaxation rate itself)
    short = PPP.ProcessRate(Float64; sink_limiting_timescale = 1e-3)
    fixed = P3Microphysics(Float64; cloud, process_rates = short, cloud_relaxation = PPP.CondensationTimescaleRelaxation())
    qᵛ, qᶜˡ = qᵛ⁺ + 1e-5, 2e-4
    qᵛ⁺ˡ = cell_saturation(qᵛ, qᶜˡ, constants; T)
    rates = cell_rates(fixed, qᵛ, qᶜˡ, constants; T)
    @test rates.condensation ≈ (qᵛ - qᵛ⁺ˡ) / τ rtol = 1e-3
    # Without cloud liquid the fixed rate still condenses supersaturated vapor
    @test cell_rates(fixed, qᵛ, 0.0, constants; T).condensation > 0
end
