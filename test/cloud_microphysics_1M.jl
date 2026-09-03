include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: microphysical_velocities, sedimentation_velocity, condensate_phase,
                               total_density, dynamics_density, standard_pressure,
                               implicit_advection_velocities, density_weighted_advection_diagonal,
                               implicit_advection_density, implicit_step_scheme, closure_scalar_index,
                               ExplicitSedimentationFluxes
using Breeze.Thermodynamics: MoistureMassFractions, LiquidIcePotentialTemperatureState,
                             LiquidIceDensityState, mixture_gas_constant
using CloudMicrophysics
using CloudMicrophysics.Microphysics1M: conv_q_lcl_to_q_rai, accretion
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce, Microphysics1MParams
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics
using Breeze.Microphysics: ConstantRateCondensateFormation

using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.TimeSteppers: update_state!, implicit_step!
using Oceananigans: fields
using Oceananigans.Fields: ZeroField, ZFaceField
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.Advection: materialize_advection, adapt_advection_order

struct MockSurfaceFluxTransportModel{G, D, V, M, A, S, W}
    grid :: G
    dynamics :: D
    velocities :: V
    microphysical_fields :: M
    advection :: A
    sedimentation_constituents :: S
    transport_w :: W
end

Breeze.AtmosphereModels.transport_velocities(model::MockSurfaceFluxTransportModel) =
    (; u = ZeroField(), v = ZeroField(), w = model.transport_w)

#####
##### One-moment microphysics tests
#####

@testset "OneMomentCloudMicrophysics construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    # Default construction (non-equilibrium)
    μ1 = OneMomentCloudMicrophysics()
    @test μ1 isa BulkMicrophysics
    @test μ1.cloud_formation isa NonEquilibriumCloudFormation
    @test μ1.cloud_formation.liquid isa ConstantRateCondensateFormation
    @test μ1.cloud_formation.ice === nothing
    @test μ1.categories.parameters isa Microphysics1MParams
    @test μ1.categories.hydrometeor_velocities.blk1m === μ1.categories.parameters.terminal_velocity
    @test μ1.categories.freezing_temperature === FT(273.15)

    converted_categories = BreezeCloudMicrophysicsExt.one_moment_cloud_microphysics_categories(
        FT;
        freezing_temperature = 273,
    )
    @test converted_categories.freezing_temperature === FT(273)

    # Disabled formation options materialize as zero-rate Breeze models.
    disabled_parameters = Microphysics1MParams(
        FT;
        cloud_liquid_formation = nothing,
        cloud_ice_formation = nothing,
    )
    disabled_categories = BreezeCloudMicrophysicsExt.one_moment_cloud_microphysics_categories(
        FT;
        parameters = disabled_parameters,
    )
    μ1_disabled = OneMomentCloudMicrophysics(FT; categories = disabled_categories)
    @test iszero(μ1_disabled.cloud_formation.liquid.rate)
    @test μ1_disabled.cloud_formation.ice === nothing

    disabled_mixed_formation = NonEquilibriumCloudFormation(nothing, CloudIce(FT))
    μ1_disabled_mixed = OneMomentCloudMicrophysics(
        FT;
        categories = disabled_categories,
        cloud_formation = disabled_mixed_formation,
    )
    @test iszero(μ1_disabled_mixed.cloud_formation.ice.rate)

    μ1_vertical = OneMomentCloudMicrophysics(FT;
                                             negative_moisture_correction = Breeze.AtmosphereModels.VerticalBorrowing())
    @test μ1_vertical.negative_moisture_correction isa Breeze.AtmosphereModels.VerticalBorrowing

    # Mixed-phase non-equilibrium
    μ1_mixed = OneMomentCloudMicrophysics(cloud_formation = NonEquilibriumCloudFormation(nothing, ConstantRateCondensateFormation(FT(0))))
    @test μ1_mixed.cloud_formation.ice isa ConstantRateCondensateFormation

    # Check prognostic fields for non-equilibrium
    prog_fields = Breeze.AtmosphereModels.prognostic_field_names(μ1)
    @test :ρqᶜˡ in prog_fields
    @test :ρqʳ in prog_fields
end

# Precision of CloudMicrophysics parameters should match that of Breeze model fields
@testset "Accretion options stored at another precision [$(FT)]" for FT in all_float_types()
    other_FT = FT === Float32 ? Float64 : Float32

    parameters = Microphysics1MParams(FT)

    # Accretion parameter values stored at another precision than the model fields
    process_params = merge(parameters.process_params,
                           (cloud_liquid_rain_accretion = (; e = other_FT(0.8)),
                            cloud_ice_rain_accretion = (; e = other_FT(1)),
                            rain_snow_accretion = (; e = other_FT(1), coeff_disp = other_FT(0.2))))
    cloud_liquid = parameters.cloud.liquid
    cloud_ice = parameters.cloud.ice
    rain = parameters.precip.rain
    snow = parameters.precip.snow
    rain_velocity = parameters.terminal_velocity.rain
    snow_velocity = parameters.terminal_velocity.snow

    qᶜ = FT(1e-4)
    qᵖ = FT(1e-5)
    ρ = FT(1)

    Sᵃᶜᶜ = BreezeCloudMicrophysicsExt.cloud_precipitation_accretion(
        process_params.cloud_liquid_rain_accretion, cloud_liquid, rain, rain_velocity, qᶜ, qᵖ, ρ)
    @test Sᵃᶜᶜ isa FT

    Sᵃᶜᶜʳⁱ = BreezeCloudMicrophysicsExt.rain_sink_accretion(
        process_params.cloud_ice_rain_accretion, rain, cloud_ice, rain_velocity, qᶜ, qᵖ, ρ)
    @test Sᵃᶜᶜʳⁱ isa FT

    Sʳˢ = BreezeCloudMicrophysicsExt.rain_snow_accretion(
        process_params.rain_snow_accretion, snow, rain, snow_velocity, rain_velocity, qᶜ, qᵖ, ρ)
    @test Sʳˢ isa FT
end

@testset "OneMomentCloudMicrophysics with SaturationAdjustment [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    # Warm-phase saturation adjustment
    cloud_formation_warm = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    μ1_warm = OneMomentCloudMicrophysics(FT; cloud_formation=cloud_formation_warm)
    @test μ1_warm.cloud_formation isa SaturationAdjustment
    @test μ1_warm.cloud_formation.equilibrium isa WarmPhaseEquilibrium

    prog_fields_warm = Breeze.AtmosphereModels.prognostic_field_names(μ1_warm)
    @test :ρqʳ in prog_fields_warm
    @test :ρqᶜˡ ∉ prog_fields_warm

    # Mixed-phase saturation adjustment
    cloud_formation_mixed = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
    μ1_mixed = OneMomentCloudMicrophysics(FT; cloud_formation=cloud_formation_mixed)
    @test μ1_mixed.cloud_formation.equilibrium isa MixedPhaseEquilibrium

    prog_fields_mixed = Breeze.AtmosphereModels.prognostic_field_names(μ1_mixed)
    @test :ρqʳ in prog_fields_mixed
    @test :ρqˢ in prog_fields_mixed
end

@testset "OneMomentCloudMicrophysics non-equilibrium time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=300, qᵗ=0.015)

    @test haskey(model.microphysical_fields, :ρqᶜˡ)
    @test haskey(model.microphysical_fields, :ρqʳ)
    @test haskey(model.microphysical_fields, :qᶜˡ)
    @test haskey(model.microphysical_fields, :qʳ)

    # Single time step (reduced from 6 iterations)
    time_step!(model, 1)
    @test model.clock.time == 1
    @test model.clock.iteration == 1
end

@testset "OneMomentCloudMicrophysics saturation adjustment time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=300, qᵗ=0.015)

    @test haskey(model.microphysical_fields, :ρqʳ)
    @test haskey(model.microphysical_fields, :qᶜˡ)
    @test haskey(model.microphysical_fields, :qʳ)

    # Single time step (reduced from 6 iterations)
    time_step!(model, 1)
    @test model.clock.time == 1
end

@testset "OneMomentCloudMicrophysics mixed-phase time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    cloud_formation = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=300, qᵗ=0.015)

    @test haskey(model.microphysical_fields, :ρqʳ)
    @test haskey(model.microphysical_fields, :ρqˢ)
    @test haskey(model.microphysical_fields, :qᶜˡ)
    @test haskey(model.microphysical_fields, :qᶜⁱ)

    # Single time step (reduced from 6 iterations)
    time_step!(model, 1)
    @test model.clock.time == 1
end

@testset "OneMomentCloudMicrophysics precipitation rate diagnostic [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Test non-equilibrium scheme only (saturation adjustment is tested elsewhere)
    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)
    set!(model; θ=300, qᵗ=0.015)
    time_step!(model, 1)

    P = precipitation_rate(model, :liquid)
    @test P isa Field
    compute!(P)
    @test isfinite(maximum(P))

    P_ice = precipitation_rate(model, :ice)
    @test P_ice === nothing
end

@testset "NonEquilibriumCloudFormation construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    cloud_formation_default = NonEquilibriumCloudFormation(CloudLiquid(FT), nothing)
    @test cloud_formation_default.liquid isa CloudLiquid
    @test cloud_formation_default.ice === nothing

    cloud_formation_mixed = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    @test cloud_formation_mixed.liquid isa CloudLiquid
    @test cloud_formation_mixed.ice isa CloudIce

    μ1 = OneMomentCloudMicrophysics(FT; cloud_formation=cloud_formation_default)
    @test μ1.cloud_formation isa NonEquilibriumCloudFormation
    @test μ1.cloud_formation.liquid.rate == inv(FT(10.0))
end

@testset "Setting specific microphysical variables [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    ρᵣ = @allowscalar reference_state.density[1, 1, 1]

    qᶜˡ_value = FT(0.001)
    qʳ_value = FT(0.002)
    set!(model; θ=300, qᵗ=0.020, qᶜˡ=qᶜˡ_value, qʳ=qʳ_value)

    @test @allowscalar model.microphysical_fields.ρqᶜˡ[1, 1, 1] ≈ ρᵣ * qᶜˡ_value
    @test @allowscalar model.microphysical_fields.ρqʳ[1, 1, 1] ≈ ρᵣ * qʳ_value
    @test @allowscalar model.microphysical_fields.qᶜˡ[1, 1, 1] ≈ qᶜˡ_value
    @test @allowscalar model.microphysical_fields.qʳ[1, 1, 1] ≈ qʳ_value

    time_step!(model, 1)
    @test model.clock.iteration == 1
end

@testset "Surface precipitation flux diagnostic [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=300)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; ρ = 2, θ = 300, qᵗ = 0.020, qᶜˡ = 0.0001, qʳ = 0.005,
         enforce_mass_conservation = false)

    production = precipitation_rate(model, :liquid)
    compute!(production)

    parameters = microphysics.categories.parameters
    qᶜˡ = @allowscalar model.microphysical_fields.qᶜˡ[1, 1, 1]
    qʳ = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    ρ = @allowscalar total_density(model.dynamics)[1, 1, 1]
    accretion_params = parameters.process_params.cloud_liquid_rain_accretion
    expected_production =
        conv_q_lcl_to_q_rai(parameters.processes.rain_autoconversion, parameters, nothing, (; q_lcl = qᶜˡ), nothing) +
        accretion(parameters.cloud.liquid, parameters.precip.rain, parameters.terminal_velocity.rain,
                  accretion_params.e, qᶜˡ, qʳ, ρ)
    @test @allowscalar production[1, 1, 1] ≈ expected_production

    spf = surface_precipitation_flux(model)
    @test spf isa Field
    compute!(spf)

    # The surface precipitation flux uses the advection scheme's face reconstruction.
    # For uniform condensate fields with Centered(order=2) advection, each
    # face-reconstructed tracer equals its cell-center value. The density is
    # face-interpolated (ℑz) to match the advection operator.
    wᶜˡ = @allowscalar model.microphysical_fields.wᶜˡ[1, 1, 1]
    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    qᶜˡ = @allowscalar model.microphysical_fields.qᶜˡ[1, 1, 1]
    qʳ = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    ρ_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, total_density(model.dynamics))
    ρ_reference_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, model.dynamics.reference_state.density)
    expected_flux = -ρ_face * (wᶜˡ * qᶜˡ + wʳ * qʳ)

    # `set!` weights the specific condensate inputs by the supplied `ρ` and adds the
    # resulting partial densities to it, so the reconciled total density is
    # ρ (1 + qᶜˡ + qʳ) evaluated with the *input* specific values.
    @test ρ_face ≈ FT(2) * (1 + FT(0.0001) + FT(0.005))
    @test !isapprox(ρ_face, ρ_reference_face)
    @test @allowscalar spf[1, 1] ≈ expected_flux
    @test @allowscalar spf[1, 1] > 0
end

@testset "Bounds-preserving WENO surface precipitation flux [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(6, 6, 6), extent=(100, 100, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = OneMomentCloudMicrophysics()
    scalar_advection = (; ρqʳ = WENO(FT; order=5, bounds=(0, 1)))
    model = AtmosphereModel(grid; dynamics, microphysics, scalar_advection)

    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0, qʳ=0.001)
    flux = surface_precipitation_flux(model)
    compute!(flux)

    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    qʳ = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    ρ_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, total_density(model.dynamics))
    @test @allowscalar flux[1, 1] ≈ -ρ_face * wʳ * qʳ
end

@testset "Adaptive implicit sedimentation includes boundary outflow [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(6, 6, 6), extent=(100, 100, 1))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = OneMomentCloudMicrophysics()
    adaptive_discretization = AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.5)
    scalar_advection = (; ρqʳ = WENO(FT; order=5, time_discretization=adaptive_discretization))
    model = AtmosphereModel(grid; dynamics, microphysics, scalar_advection)

    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0, qʳ=0.001)
    advection = model.advection.ρqʳ
    td = Oceananigans.TimeSteppers.time_discretization(advection)
    Δt = FT(10)
    td.Δt[] = Δt

    velocities = implicit_advection_velocities(model.dynamics, model.velocities, :ρqʳ,
                                                model.microphysics, model.microphysical_fields)
    ρ = total_density(model.dynamics)

    # The Breeze-owned seam every scalar's implicit solve passes through (see
    # `DensityWeightedImplicitOperator`): with the bottom face unmasked, the outflow term makes
    # the bottom-cell diagonal positive, where the impermeable upstream coefficient vanishes.
    diagonal = @allowscalar density_weighted_advection_diagonal(1, 1, 1, grid, advection, velocities.w,
                                                                Δt, Center(), Center(), Center(), ρ)
    @test diagonal > 0

    flux = surface_precipitation_flux(model)
    compute!(flux)
    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    qʳ = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    ρ_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, ρ)
    @test @allowscalar flux[1, 1] ≈ -ρ_face * wʳ * qʳ
end

@testset "Adaptive implicit sedimentation heat follows the solved mass [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 8
    Δz = FT(20)
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 100), y=(0, 100), z=(0, Nz * Δz))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = OneMomentCloudMicrophysics()
    adaptive_discretization = AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.5)
    scalar_advection = (; ρqʳ = WENO(FT; order=5, time_discretization=adaptive_discretization))
    model = AtmosphereModel(grid; dynamics, microphysics, scalar_advection)
    μ = model.microphysical_fields

    # A rain shaft above a dry lower half, falling at 8 m/s through 20 m cells over 10 s: a fall
    # Courant number of 4, of which the explicit fraction carries 1/8 and the solve the rest.
    Δt = FT(10)
    w₀ = FT(-8)
    set!(model; θ=300, qᵗ=0.005, qᶜˡ=0, qʳ=(x, y, z) -> ifelse(z > Nz * Δz / 2, FT(1e-3), FT(0)))
    set!(μ.wʳ, (x, y, z) -> ifelse(z < Nz * Δz, w₀, FT(0)))
    set!(μ.wᶜˡ, 0)
    td = Oceananigans.TimeSteppers.time_discretization(model.advection.ρqʳ)
    td.Δt[] = Δt

    # The rain's implicit solve, exactly as the time steppers issue it
    ρqʳ = μ.ρqʳ
    ρqʳ⁰ = Array(interior(ρqʳ, 1, 1, :))
    implicit_step!(ρqʳ, model.timestepper.implicit_solver, model.closure, model.closure_fields,
                   closure_scalar_index(model, :ρqʳ), model.clock, fields(model), Δt,
                   implicit_step_scheme(model.advection.ρqʳ),
                   implicit_advection_velocities(model.dynamics, model.velocities, :ρqʳ, model.microphysics, μ),
                   implicit_advection_density(model.dynamics, model.formulation, :ρqʳ))
    Δρqʳ = Array(interior(ρqʳ, 1, 1, :)) .- ρqʳ⁰

    # Nothing enters through the top, so the top cell only drains: backward Euler leaves it
    # 1 / (1 + C) of its rain at the face-weighted implicit Courant number C, where an estimate at
    # the pre-solve state, C q, overstates the loss by the factor 1 + C.
    ρ = total_density(model.dynamics)
    ρᶠ = @allowscalar ℑzᵃᵃᶠ(1, 1, Nz, grid, ρ)
    ρᶜ = @allowscalar ρ[1, 1, Nz]
    α = -w₀ * Δt / Δz
    C = α * (1 - FT(0.5) / α) * ρᶠ / ρᶜ
    @test C > 3
    @test Δρqʳ[Nz] ≈ -C / (1 + C) * ρqʳ⁰[Nz]

    # With a uniform synthetic content, the heat the post-solve step moves is χ times the mass
    # the solve moved, cell by cell (the anelastic coupling ratio is one), outflow through the
    # bottom included, so the column loses deficit with the rain that leaves.
    χ = FT(-2500)
    uniform_content(i, j, k, grid) = (χ, zero(χ))
    ρθ = model.formulation.potential_temperature_density
    ρθ⁰ = Array(interior(ρθ, 1, 1, :))
    Breeze.AtmosphereModels.implicit_sedimentation_step!(model, Δt, model.velocities, uniform_content)
    Δρθ = Array(interior(ρθ, 1, 1, :)) .- ρθ⁰
    tolerance = sqrt(eps(FT)) * maximum(abs.(χ .* Δρqʳ))
    @test all(abs.(Δρθ .- χ .* Δρqʳ) .<= tolerance)
    @test sum(Δρqʳ) < 0
    @test sum(Δρθ) > 0

    # The step is wired into both time steppers: one step with the fall speed above the
    # explicit CFL leaves finite fields on the anelastic SSP path ...
    time_step!(model, Δt)
    @test all(isfinite, interior(ρθ))
    @test all(isfinite, interior(ρqʳ))

    # ... and on the compressible acoustic path.
    acoustic_grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 100), y=(0, 100), z=(0, 800))
    acoustic_dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300)
    acoustic_advection = (; ρqʳ = WENO(FT; order=5, time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.05)))
    acoustic_model = AtmosphereModel(acoustic_grid; dynamics=acoustic_dynamics, microphysics=OneMomentCloudMicrophysics(),
                                     scalar_advection=acoustic_advection)
    set!(acoustic_model; ρ=acoustic_model.dynamics.reference_state.density, θ=300, qᵗ=0.005, qᶜˡ=0,
         qʳ=(x, y, z) -> ifelse(z > 400, FT(1e-3), FT(0)))
    time_step!(acoustic_model, 1)
    @test all(isfinite, interior(acoustic_model.formulation.potential_temperature_density))
    @test all(isfinite, interior(acoustic_model.microphysical_fields.ρqʳ))
end

@testset "Surface precipitation flux uses transport velocities [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0, qʳ=0.001)

    transport_w = set!(ZFaceField(grid), FT(-1))
    mock_model = MockSurfaceFluxTransportModel(model.grid,
                                               model.dynamics,
                                               model.velocities,
                                               model.microphysical_fields,
                                               model.advection,
                                               model.sedimentation_constituents,
                                               transport_w)

    spf = surface_precipitation_flux(mock_model, microphysics)
    compute!(spf)

    wᵗ = @allowscalar transport_w[1, 1, 1]
    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    qʳ = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    ρ_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, total_density(model.dynamics))

    expected_flux = -ρ_face * (wᵗ + wʳ) * qʳ
    sedimentation_only_flux = -ρ_face * wʳ * qʳ

    @test !isapprox(expected_flux, sedimentation_only_flux)
    @test @allowscalar spf[1, 1] ≈ expected_flux
end

# Consolidated simulation-based tests (reduced simulation times)
@testset "Rain accumulation from autoconversion [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 10
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    set!(model; θ=300, qᵗ=FT(0.050))

    # Reduced simulation time (from 5τ + 30τ = 35τ to just 10τ total)
    τ = inv(microphysics.cloud_formation.liquid.rate)
    simulation = Simulation(model; Δt=τ/5, stop_time=10τ, verbose=false)
    run!(simulation)

    # Cloud liquid should have formed
    qᶜˡ_equilibrium = maximum(model.microphysical_fields.qᶜˡ)
    @test qᶜˡ_equilibrium > FT(0.001)

    # Rain should exist somewhere in the domain
    qʳ_max = maximum(model.microphysical_fields.qʳ)
    @test qʳ_max > FT(1e-10)
end

@testset "ImpenetrableBoundaryCondition prevents rain from exiting domain [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics(; precipitation_boundary_condition=ImpenetrableBoundaryCondition())
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    set!(model; θ=300, qᵗ=FT(0.050))

    # Reduced simulation time (from 10τ to 5τ)
    τ = inv(microphysics.cloud_formation.liquid.rate)
    simulation = Simulation(model; Δt=τ/10, stop_time=5τ, verbose=false)
    run!(simulation)

    # Terminal velocity should be zero at impenetrable bottom
    wʳ_bottom = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    @test wʳ_bottom == 0
end

@testset "Mixed-phase non-equilibrium time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=260)
    dynamics = AnelasticDynamics(reference_state)

    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    prog_fields = Breeze.AtmosphereModels.prognostic_field_names(microphysics)
    @test :ρqᶜˡ in prog_fields
    @test :ρqᶜⁱ in prog_fields
    @test :ρqʳ in prog_fields
    @test :ρqˢ in prog_fields

    set!(model; θ=260, qᵗ=0.010)
    @test haskey(model.microphysical_fields, :ρqᶜⁱ)
    @test haskey(model.microphysical_fields, :qᶜⁱ)

    time_step!(model, 1)
    @test model.clock.iteration == 1
end

@testset "OneMomentCloudMicrophysics show methods [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    μ_ne = OneMomentCloudMicrophysics()
    str_ne = sprint(show, μ_ne)
    @test contains(str_ne, "BulkMicrophysics")
    @test contains(str_ne, "cloud_formation")
end

@testset "sedimentation_velocity, condensate_phase, and microphysical_velocities [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)
    set!(model; θ=300, qᵗ=0.015, qʳ=0.001)

    μ = model.microphysical_fields

    # sedimentation_velocity returns vertical velocity component fields
    w_rain = sedimentation_velocity(microphysics, μ, Val(:ρqʳ))
    @test w_rain !== nothing
    @test w_rain === μ.wʳ

    # WPNE1M has cloud liquid sedimentation
    w_cloud = sedimentation_velocity(microphysics, μ, Val(:ρqᶜˡ))
    @test w_cloud !== nothing
    @test w_cloud === μ.wᶜˡ

    # Thermodynamic phase of each condensate mass
    @test condensate_phase(microphysics, Val(:ρqʳ)) === Val(:liquid)
    @test condensate_phase(microphysics, Val(:ρqᶜˡ)) === Val(:liquid)

    # microphysical_velocities wraps sedimentation_velocity in a velocity tuple
    vel_rain = microphysical_velocities(microphysics, μ, Val(:ρqʳ))
    @test vel_rain !== nothing
    @test haskey(vel_rain, :w)

    vel_cloud = microphysical_velocities(microphysics, μ, Val(:ρqᶜˡ))
    @test vel_cloud !== nothing
    @test haskey(vel_cloud, :w)

    # Sedimentation velocity values should be negative downward
    wʳ = @allowscalar μ.wʳ[1, 1, 2]
    @test wʳ <= 0

    # Both sedimenting masses are resolved as constituents, each with its own velocity field,
    # humidity field, phase, and the advection scheme that transports it
    constituents = model.sedimentation_constituents
    @test length(constituents) == 2
    rain = constituents[findfirst(c -> c.w === μ.wʳ, constituents)]
    @test rain.q === μ.qʳ
    @test rain.phase === Val(:liquid)
    @test rain.advection === model.advection.ρqʳ
    @test any(c -> c.w === μ.wᶜˡ && c.q === μ.qᶜˡ && c.phase === Val(:liquid), constituents)
end

@testset "Diagnosed cloud condensate is not a sedimentation constituent [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), extent=(100, 100, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=280)
    dynamics = AnelasticDynamics(reference_state)
    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=280, qᵗ=0.03, qʳ=0.001)
    μ = model.microphysical_fields
    @test @allowscalar(μ.qᶜˡ[1, 1, 1]) > 0

    # Under saturation adjustment only rain is prognostic and falls; the diagnosed cloud
    # liquid moves no mass and therefore appears in no constituent.
    constituents = model.sedimentation_constituents
    @test length(constituents) == 1
    @test constituents[1].w === μ.wʳ
    @test constituents[1].q === μ.qʳ
    @test constituents[1].phase === Val(:liquid)
end

@testset "Sedimentation transports the condensate part of ρθ and ρs [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 8
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 100), y=(0, 100), z=(0, 800))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())

    # Unsaturated column with a mid-column rain blob: no winds, no cloud, no closure, so
    # the only thermodynamic tendency is the sedimentation transport of the rain's content.
    rain_blob(x, y, z) = ifelse(300 < z < 500, FT(1e-3), FT(0))
    Δz = FT(100)

    for formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
        dynamics = AnelasticDynamics(reference_state)
        microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
        model = AtmosphereModel(grid; dynamics, microphysics, formulation)
        set!(model; θ=300, qᵗ=0.005, qʳ=rain_blob)
        update_state!(model)

        μ = model.microphysical_fields
        column(f) = Array(interior(f, 1, 1, :))
        qˡ = column(μ.qˡ)
        qᵛ = column(μ.qᵛ)
        pᵣ = column(model.dynamics.reference_state.pressure)
        pˢᵗ = reference_state.standard_pressure
        ρᵣ = model.dynamics.reference_state.density
        ρᵣᶠ = [(@allowscalar ℑzᵃᵃᶠ(1, 1, k, grid, ρᵣ)) for k in 1:Nz+1]

        # The sedimentation mass flux advection actually applies: with zero resolved
        # velocity and the default Centered(order=2) scheme, the flux difference at face k
        # is wʳ[k] times the face-interpolated rain humidity (only rain sediments here;
        # diagnosed cloud moves no mass and is not a constituent).
        wʳ = [(@allowscalar μ.wʳ[1, 1, k]) for k in 1:Nz+1]
        qʳᶠ = [(@allowscalar ℑzᵃᵃᶠ(1, 1, k, grid, μ.qʳ)) for k in 1:Nz+1]
        Φ = wʳ .* qʳᶠ

        # The content per unit falling liquid is ∂φ/∂qˡ at fixed temperature (see
        # `condensate_content` in setup.jl), with the temperature the tendency kernel works
        # with: the θ kernel diagnoses it from θ and q, the s kernel reads `model.temperature`.
        q = [MoistureMassFractions(qᵛ[k], qˡ[k], zero(FT)) for k in 1:Nz]
        if formulation == :LiquidIcePotentialTemperature
            θ = column(model.formulation.potential_temperature)
            T = [Breeze.Thermodynamics.temperature(LiquidIcePotentialTemperatureState(θ[k], q[k], pˢᵗ, pᵣ[k]), constants)
                 for k in 1:Nz]
            G = column(model.timestepper.Gⁿ.ρθ)
        else
            T = column(model.temperature)
            G = column(model.timestepper.Gⁿ.ρs)
        end
        χ = [condensate_content(formulation, :liquid, T[k], q[k], pᵣ[k], pˢᵗ) for k in 1:Nz]

        # With no transport velocity every flux is downward, so each face carries the content
        # of the cell above it (clamped at the top, where the flux vanishes)
        F = [ρᵣᶠ[k] * Φ[k] * χ[min(k, Nz)] for k in 1:Nz+1]
        G_expected = [-(F[k+1] - F[k]) / Δz for k in 1:Nz]

        scale = maximum(abs.(G))
        tolerance = scale * sqrt(eps(FT))
        @test scale > 0
        @test all(abs.(G .- G_expected) .<= tolerance)
        @test abs(sum(G)) <= tolerance # conservative: the blob is away from the boundaries
        @test G[3] < 0                 # rain arriving below the blob pre-cools
        @test G[5] > 0                 # rain leaving the blob top leaves latent warming behind
    end

    # Rain sedimenting out of the open bottom removes its deficit, so ∫ρθ rises: the latent
    # warming from forming the rain stays in the column.
    dynamics = AnelasticDynamics(reference_state)
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)
    low_rain(x, y, z) = ifelse(z < 200, FT(1e-3), FT(0))
    set!(model; θ=300, qᵗ=0.005, qʳ=low_rain)
    ρθ = model.formulation.potential_temperature_density
    Σρθ = sum(interior(ρθ))
    for _ in 1:5
        time_step!(model, 1)
    end
    @test sum(interior(ρθ)) > Σρθ
end

# Dynamics whose total density falls with the sedimenting condensate, so that the local mixture
# takes up the departed mass, wrapped around an anelastic model for the static-energy check below
struct MixtureReplacementDynamics{D}
    dynamics :: D
end
Breeze.AtmosphereModels.total_density(d::MixtureReplacementDynamics) = total_density(d.dynamics)
Breeze.AtmosphereModels.sedimentation_replacement(::MixtureReplacementDynamics, q) = q

@testset "Compressible sedimentation lets the mixture take up the departed mass [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 8
    Δz = FT(100)
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 100), y=(0, 100), z=(0, Nz * Δz))

    constants = ThermodynamicConstants()
    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    rain_blob(x, y, z) = ifelse(300 < z < 500, FT(2e-3), FT(0))

    # On the compressible core the prognostic dry density has no sedimentation source, so rain
    # leaving a cell lowers its total density and every mass fraction renormalizes: the content
    # is the derivative along q → q + ε (eˣ − q), the mixture rather than dry air taking up the
    # departed mass, and the divergence of the total-density-weighted content flux converts to
    # the prognostic ρᵈ φ through the cell's qᵈ = ρᵈ / ρ. Moist enough (unsaturated throughout)
    # that the dry-air derivative the anelastic core uses fails the check the kernel meets.
    # Only the θ formulation runs on the compressible core so far.
    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=300)
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)
    set!(model; ρ=model.dynamics.reference_state.density, θ=300, qᵗ=0.012, qʳ=rain_blob)
    update_state!(model)

    μ = model.microphysical_fields
    column(f) = Array(interior(f, 1, 1, :))
    qˡ = column(μ.qˡ)
    qᵛ = column(μ.qᵛ)
    ρ = total_density(model.dynamics)
    ρᶜ = column(ρ)
    qᵈ = column(dynamics_density(model.dynamics)) ./ ρᶜ
    ρᶠ = [(@allowscalar ℑzᵃᵃᶠ(1, 1, k, grid, ρ)) for k in 1:Nz+1]
    wʳ = [(@allowscalar μ.wʳ[1, 1, k]) for k in 1:Nz+1]
    qʳᶠ = [(@allowscalar ℑzᵃᵃᶠ(1, 1, k, grid, μ.qʳ)) for k in 1:Nz+1]
    Φ = wʳ .* qʳᶠ

    # The temperature the kernel works with is the inversion of θ and q at the total density.
    # The pressure p = ρ Rᵐ T is the gas-phase pressure (ρᵈ Rᵈ + ρᵛ Rᵛ) T, which condensate
    # leaving at fixed T does not change.
    q = [MoistureMassFractions(qᵛ[k], qˡ[k], zero(FT)) for k in 1:Nz]
    pˢᵗ = standard_pressure(model.dynamics)
    θ = column(model.formulation.potential_temperature)
    solver = model.formulation.temperature_solver
    T = [Breeze.Thermodynamics.temperature(LiquidIceDensityState(θ[k], q[k], pˢᵗ, ρᶜ[k], solver), constants)
         for k in 1:Nz]
    p = [ρᶜ[k] * mixture_gas_constant(q[k], constants) * T[k] for k in 1:Nz]
    G = column(model.timestepper.Gⁿ.ρθ)

    # Every flux is downward, so each face carries the content of the cell above it
    expected(χ) = [-qᵈ[k] * (ρᶠ[k+1] * Φ[k+1] * χ[min(k+1, Nz)] - ρᶠ[k] * Φ[k] * χ[k]) / Δz for k in 1:Nz]
    χ = [condensate_content(:LiquidIcePotentialTemperature, :liquid, T[k], q[k], p[k], pˢᵗ; replacement=:mixture) for k in 1:Nz]
    χ_dry = [condensate_content(:LiquidIcePotentialTemperature, :liquid, T[k], q[k], p[k], pˢᵗ) for k in 1:Nz]

    scale = maximum(abs.(G))
    tolerance = scale * sqrt(eps(FT))
    @test scale > 0
    @test all(abs.(G .- expected(χ)) .<= tolerance)
    @test any(abs.(G .- expected(χ_dry)) .> tolerance)
    @test G[3] < 0 # rain arriving below the blob pre-cools
    @test G[5] > 0 # rain leaving the blob top leaves latent warming behind

    # The static-energy content along the same composition change, hˣ − h_mixture, checked
    # against the central difference through dynamics that declare the mixture replacement
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    energy_model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state),
                                   microphysics = OneMomentCloudMicrophysics(FT; cloud_formation),
                                   formulation = :StaticEnergy)
    set!(energy_model; θ=300, qᵗ=0.012, qʳ=rain_blob)
    update_state!(energy_model)
    mixture_dynamics = MixtureReplacementDynamics(energy_model.dynamics)
    μₛ = energy_model.microphysical_fields
    qᵛₛ = column(μₛ.qᵛ)
    qˡₛ = column(μₛ.qˡ)
    Tₛ = column(energy_model.temperature)
    pᵣ = column(energy_model.dynamics.reference_state.pressure)
    for k in 1:Nz
        χˡ, _ = @allowscalar Breeze.StaticEnergyFormulations.energy_condensate_content(
            1, 1, k, grid, mixture_dynamics, constants, energy_model.microphysics, μₛ,
            Breeze.AtmosphereModels.specific_prognostic_moisture(energy_model), energy_model.temperature)
        qₖ = MoistureMassFractions(qᵛₛ[k], qˡₛ[k], zero(FT))
        χ_expected = condensate_content(:StaticEnergy, :liquid, Tₛ[k], qₖ, pᵣ[k], pˢᵗ; replacement=:mixture)
        χ_dry = condensate_content(:StaticEnergy, :liquid, Tₛ[k], qₖ, pᵣ[k], pˢᵗ)
        @test isapprox(χˡ, χ_expected; rtol=sqrt(eps(FT)))
        @test !isapprox(χˡ, χ_dry; rtol=sqrt(eps(FT)))
    end
end

@testset "Sedimentation content follows each flux's upwind cell [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 4
    Δz = FT(100)
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 100), y=(0, 100), z=(0, Nz * Δz))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)
    μ = model.microphysical_fields

    # Rain with a uniform fall speed and a height-dependent humidity; the cloud constituent is
    # emptied so only rain contributes. A synthetic content χ = k for cell k identifies the
    # cell each flux draws its content from.
    wʳ = FT(-2)
    set!(μ.wʳ, wʳ)
    set!(μ.qʳ, (x, y, z) -> FT(1e-3) * (1 + z / (Nz * Δz)))
    set!(μ.wᶜˡ, 0)
    set!(μ.qᶜˡ, 0)
    synthetic_content(i, j, k, grid) = (FT(k), FT(10k))

    ρᵣ = model.dynamics.reference_state.density
    ρᵣᶠ(k) = @allowscalar ℑzᵃᵃᶠ(1, 1, k, grid, ρᵣ)
    q̄(k) = @allowscalar ℑzᵃᵃᶠ(1, 1, k, grid, μ.qʳ) # Centered(order=2) face reconstruction
    Az = FT(100 * 100)

    # Each flux carries the content of the cell it drains: the cell above face k (content k)
    # for a downward velocity, the cell below (content k − 1) for an upward one. With no
    # transport both fluxes are downward; a transport velocity below the fall speed leaves the
    # total velocity downward while the transport flux it replaces drained the cell below; a
    # transport velocity above the fall speed makes both fluxes draw from the cell below.
    for wᵗ_value in (FT(0), FT(1), FT(5))
        wᵗ = set!(ZFaceField(grid), wᵗ_value)
        W = wᵗ_value + wʳ
        χ(w, k) = w > 0 ? FT(k - 1) : FT(k)
        F(k) = ρᵣᶠ(k) * Az * q̄(k) * (χ(W, k) * W - χ(wᵗ_value, k) * wᵗ_value)
        for k in 2:Nz-1
            expected = (F(k + 1) - F(k)) / (Az * Δz)
            computed = @allowscalar Breeze.AtmosphereModels.condensate_sedimentation_divergence(
                1, 1, k, grid, model.sedimentation_constituents, wᵗ, model.dynamics,
                ExplicitSedimentationFluxes(), synthetic_content)
            @test computed ≈ expected
        end
    end
end

@testset "Bounds-preserving WENO sedimentation heat follows the limited mass flux [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 8
    Δz = FT(100)
    grid = RectilinearGrid(default_arch; size=Nz, z=(0, Nz * Δz), topology=(Flat, Flat, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = OneMomentCloudMicrophysics()
    scalar_advection = (; ρqʳ = WENO(FT; order=5, bounds=(0, 1)))
    model = AtmosphereModel(grid; dynamics, microphysics, scalar_advection)
    μ = model.microphysical_fields

    # A rain shaft with a sharp base and a dry gap, falling at a uniform speed: WENO
    # reconstructs nonzero face values in the dry cells at the base and in the gap, where the
    # limiter then pins them to the cell's zero. A uniform content isolates the mass fluxes
    # from the donor rule.
    set!(μ.wʳ, FT(-2))
    set!(μ.qʳ, z -> ifelse(z > Nz * Δz / 2 && !(5Δz < z < 6Δz), FT(1e-3), FT(0)))
    set!(μ.wᶜˡ, 0)
    set!(μ.qᶜˡ, 0)
    χ = FT(-2.5e6)
    uniform_content(i, j, k, grid) = (χ, zero(χ))

    wᵗ = ZFaceField(grid)
    ρᵣ = model.dynamics.reference_state.density
    U = (; u = ZeroField(), v = ZeroField(), w = μ.wʳ)

    # The rain constituent carries the model's materialized bounds-preserving scheme; the
    # unlimited twin is the same WENO materialized the way the model does it.
    limited = model.sedimentation_constituents
    rain = only(filter(c -> c.q === μ.qʳ, limited))
    @test rain.advection isa WENO
    @test rain.advection.bounds == (0, 1)
    unlimited_scheme = adapt_advection_order(materialize_advection(WENO(FT; order=5), grid), grid)
    unlimited = ((; rain.w, rain.q, rain.ρq, rain.phase, advection = unlimited_scheme),)

    heat(constituents, k) = @allowscalar Breeze.AtmosphereModels.condensate_sedimentation_divergence(
        1, 1, k, grid, constituents, wᵗ, model.dynamics, ExplicitSedimentationFluxes(), uniform_content)
    mass(advection, k) = @allowscalar Breeze.AtmosphereModels.div_ρUc(1, 1, k, grid, advection, ρᵣ, U, μ.qʳ)
    atol = sqrt(eps(FT)) * abs(χ) * FT(1e-3) * 2 / Δz

    # Cell by cell, the heat divergence is χ times the mass divergence the tracer tendency
    # applies with its limited reconstructions...
    for k in 1:Nz
        @test isapprox(heat(limited, k), χ * mass(rain.advection, k); atol)
    end

    # ... and the limiter does act: the unlimited WENO fluxes, which the coupling used to
    # take, move heat somewhere the limited mass flux does not.
    @test any(k -> !isapprox(heat(limited, k), heat(unlimited, k); atol), 1:Nz)
end

@testset "Mixed-phase constituents and snow surface flux [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=250)
    dynamics = AnelasticDynamics(reference_state)
    cloud_formation = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=250, qᵗ=0.01, qʳ=0.0005, qˢ=0.0005)
    μ = model.microphysical_fields
    qᶜⁱ = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, μ.qᶜⁱ)
    qʳ = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, μ.qʳ)
    qˢ = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, μ.qˢ)
    wʳ = @allowscalar μ.wʳ[1, 1, 1]
    wˢ = @allowscalar μ.wˢ[1, 1, 1]

    # Rain and snow are the constituents, with liquid and ice phase; the diagnosed cloud
    # liquid and cloud ice move no mass and appear in neither
    constituents = model.sedimentation_constituents
    @test length(constituents) == 2
    @test any(c -> c.w === μ.wʳ && c.q === μ.qʳ && c.phase === Val(:liquid), constituents)
    @test any(c -> c.w === μ.wˢ && c.q === μ.qˢ && c.phase === Val(:ice), constituents)
    @test qᶜⁱ > 0
    @test wˢ < 0

    flux = surface_precipitation_flux(model)
    compute!(flux)
    ρ_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, total_density(model.dynamics))
    @test @allowscalar flux[1, 1] ≈ -ρ_face * (wʳ * qʳ + wˢ * qˢ)
end

@testset "Mixed-phase non-equilibrium snow field materialization [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=260)
    dynamics = AnelasticDynamics(reference_state)

    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    # Snow terminal velocity field should exist
    @test haskey(model.microphysical_fields, :wˢ)

    # Snow sedimentation velocity dispatch
    μ = model.microphysical_fields
    vel_snow = microphysical_velocities(microphysics, μ, Val(:ρqˢ))
    @test vel_snow !== nothing
    @test haskey(vel_snow, :w)

    # Other tracers still have correct dispatch
    vel_rain = microphysical_velocities(microphysics, μ, Val(:ρqʳ))
    @test vel_rain !== nothing

    # Cloud condensate velocity fields should exist
    @test haskey(model.microphysical_fields, :wᶜˡ)
    @test haskey(model.microphysical_fields, :wᶜⁱ)

    # Cloud liquid sedimentation velocity dispatch
    vel_cloud = microphysical_velocities(microphysics, μ, Val(:ρqᶜˡ))
    @test vel_cloud !== nothing
    @test haskey(vel_cloud, :w)

    # Cloud ice sedimentation velocity dispatch
    vel_ice = microphysical_velocities(microphysics, μ, Val(:ρqᶜⁱ))
    @test vel_ice !== nothing
    @test haskey(vel_ice, :w)
end

@testset "MPNE1M snow processes time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 10
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=260)
    dynamics = AnelasticDynamics(reference_state)

    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    # Cold, supersaturated conditions → cloud ice should form via deposition
    set!(model; θ=260, qᵗ=FT(0.010))

    # Run for a few relaxation timescales
    τ = FT(1) / microphysics.cloud_formation.ice.rate
    simulation = Simulation(model; Δt=τ/5, stop_time=10τ, verbose=false)
    run!(simulation)

    # Cloud ice should have formed from deposition
    qᶜⁱ_max = maximum(model.microphysical_fields.qᶜⁱ)
    @test qᶜⁱ_max > FT(1e-6)

    # Snow should have formed from ice autoconversion
    qˢ_max = maximum(model.microphysical_fields.qˢ)
    @test qˢ_max > FT(0)

    # Model should complete without errors (all tendencies computed)
    @test model.clock.iteration > 0
end
