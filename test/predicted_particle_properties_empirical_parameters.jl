include(joinpath(@__DIR__, "setup.jl"))

using Test

using Breeze
using Breeze.Microphysics.PredictedParticleProperties:
    CloudDropletProperties,
    CloudShapeParameters,
    PredictedParticlePropertiesMicrophysics,
    ProcessRateParameters,
    RainEvaporationVentilationEvaluator,
    RainFallSpeedParameters,
    RainMassWeightedVelocityEvaluator,
    RainNumberWeightedVelocityEvaluator,
    RainProperties,
    RainVentilationParameters,
    air_transport_properties,
    diagnose_cloud_dsd,
    immersion_freezing_cloud_rate,
    liu_daum_shape_parameter,
    rain_evaporation_rate,
    rain_fall_speed,
    rain_ventilation_integral,
    tabulate_rain_from_quadrature
using Breeze.Thermodynamics: ThermodynamicConstants

using Oceananigans: CPU, RectilinearGrid, set!, time_step!
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: interior

const PPP = Breeze.Microphysics.PredictedParticleProperties

#####
##### Verbatim copies of the pinned (pre-refactor) empirical relations
#####
##### `liu_daum_shape_parameter` and `rain_fall_speed` are pointwise, so default parity is
##### checked against these copies at every branch boundary rather than against a handful
##### of recorded numbers. Do not "fix" these to match a failing implementation: they are
##### the specification.
#####

@inline function reference_liu_daum_shape_parameter(Nᶜˡ)
    FT = typeof(float(Nᶜˡ))
    Nᶜˡ_cm³ = Nᶜˡ * FT(1e-6)              # m⁻³ → cm⁻³
    χ = FT(0.0005714) * Nᶜˡ_cm³ + FT(0.2714)
    μᶜˡ = FT(1) / χ^2 - FT(1)
    return clamp(μᶜˡ, FT(2), FT(15))
end

@inline function reference_rain_fall_speed(D, ρ_correction)
    FT = typeof(D)

    m_kg = (FT(π)/6) * FT(997) * D^3
    m_g = m_kg * 1000

    V_cm = ifelse(D <= FT(134.43e-6),  FT(4.5795e5) * cbrt(m_g)^2,
           ifelse(D <  FT(1511.64e-6), FT(4.962e3)  * cbrt(m_g),
           ifelse(D <  FT(3477.84e-6), FT(1.732e3)  * sqrt(cbrt(m_g)),
                                       FT(917))))

    return V_cm / 100 * ρ_correction
end

#####
##### Reference outputs recorded from the P3 implementation *before* the empirical
##### cloud-width / rain-ventilation / rain-fall-speed coefficients were promoted into
##### `CloudShapeParameters`, `RainVentilationParameters` and `RainFallSpeedParameters`.
#####
##### Recorded at Breeze bfd4edddccb655fe66eef57571e34ad1d82e5efc with the coefficients still hard-coded in
##### `rain_quadrature.jl` and `cloud_droplet_properties.jl`. They exist so the default
##### configuration of the parameterized implementation can be shown to reproduce the
##### pinned one. Do not regenerate them to make a failing test pass: a change here is a
##### change in the default physics.
#####
##### Only the quadrature-dependent quantities are recorded. The pointwise relations
##### (`liu_daum_shape_parameter`, `rain_fall_speed`) are instead checked against verbatim
##### copies of the pinned formulas in the parity testset, which covers every branch
##### boundary rather than a handful of sampled points.
#####

# log10(λʳ) nodes spanning the complete tabulated slope range
p3_reference_log_slopes(FT) = ntuple(i -> FT(2.5) + FT(0.25) * (i - 1), 13)

# (qʳ [kg/kg], nʳ [1/kg]) states for the full ventilation integral, evaluated at
# ν = 1.5e-5 m²/s and Dᵛ = 2.2e-5 m²/s.
const P3_REFERENCE_VENTILATION_VISCOSITY = 1.5e-5
const P3_REFERENCE_VENTILATION_DIFFUSIVITY = 2.2e-5

p3_reference_ventilation_states(FT) = ((FT(1e-5), FT(1e2)), (FT(1e-5), FT(1e4)),
                                       (FT(1e-4), FT(1e2)), (FT(1e-4), FT(1e4)),
                                       (FT(1e-3), FT(1e2)), (FT(1e-3), FT(1e4)))

const P3_REFERENCE = Dict{DataType, NamedTuple}()

P3_REFERENCE[Float64] = (
    velocity_mass = (9.138109770629223, 8.978035249205135, 8.335512834451823, 6.795669896030269,
         4.700793993368999, 2.8257540944965243, 1.594262096753617, 0.8794872318446458,
         0.4445906570590266, 0.1799716864121633, 0.05933741990233856, 0.01877776255387256,
         0.005938050856099867),
    velocity_number = (6.272913440186612, 4.9006802638405205, 3.424516127715861, 2.140223571379382,
         1.2267130378958857, 0.6631489028577771, 0.33515397886867143, 0.15006093389059794,
         0.05673494753839378, 0.018729028607490976, 0.005937238815908659, 0.0018775408732970404,
         0.000593730556293787),
    velocity_diameter_integral = (2.2130389996022022e-6, 5.016537995504443e-7, 1.0743571437000075e-7, 2.132970761046551e-8,
         3.959492610528624e-9, 7.059080771088357e-10, 1.2294359122321053e-10, 2.0428522433727886e-11,
         3.0987521343454978e-12, 4.28362057219744e-13, 5.726340527112435e-14, 7.636303249772484e-15,
         1.0183174054790342e-15),
    ventilation_integral = (3.6844916811732073e-7, 6.383096925818342e-9, 3.0633537655711103e-6, 4.5857933872193804e-8,
         2.3815348794250408e-5, 3.6844916811732073e-7),
)

P3_REFERENCE[Float32] = (
    velocity_mass = (9.138108f0, 8.978033f0, 8.33551f0, 6.795668f0,
         4.700793f0, 2.8257537f0, 1.5942619f0, 0.8794873f0,
         0.44459066f0, 0.17997168f0, 0.059337426f0, 0.018777765f0,
         0.0059380494f0),
    velocity_number = (6.2729144f0, 4.90068f0, 3.4245157f0, 2.1402237f0,
         1.2267132f0, 0.6631489f0, 0.335154f0, 0.15006097f0,
         0.05673495f0, 0.018729027f0, 0.005937241f0, 0.0018775404f0,
         0.0005937305f0),
    velocity_diameter_integral = (2.2130394f-6, 5.016539f-7, 1.0743572f-7, 2.132971f-8,
         3.959493f-9, 7.059079f-10, 1.2294361f-10, 2.0428523f-11,
         3.098752f-12, 4.2836202f-13, 5.7263405f-14, 7.636303f-15,
         1.0183173f-15),
    ventilation_integral = (3.6844915f-7, 6.3830927f-9, 3.0633512f-6, 4.5857924f-8,
         2.3815324f-5, 3.6844915f-7),
)

#####
##### Tolerances
#####
##### Exact bitwise equality is not expected, for two documented reasons:
#####
##### 1. The cloud-width coefficient is now stated in SI (5.714e-10 m³) instead of the
#####    published cm⁻³ form (0.0005714 cm³ applied to `Nᶜˡ × 1e-6`). Algebraically
#####    identical, but the two roundings differ in the last bits of χ, and
#####    `μ = 1/χ² - 1` amplifies a relative perturbation of χ by ≈ 2(μ+1)/μ.
#####
##### 2. Parameterizing the fall-speed exponents replaces the specialized `cbrt(m)^2`,
#####    `cbrt(m)` and `sqrt(cbrt(m))` with a generic `m^b`. `b` is not exactly
#####    representable, so `m^b` carries a relative error of order |b ln m| × ulp(b)/b —
#####    about 3e-15 in `Float64` and 2e-6 in `Float32` over the tabulated size range.
#####
##### The tolerances below are tight enough to catch a changed unit, a flipped branch
##### inequality, or a dropped parameter, all of which move results by percent or more.
#####

pointwise_tolerance(::Type{Float64}) = 1e-13
pointwise_tolerance(::Type{Float32}) = 2e-5

# The quadrature integrals accumulate the pointwise fall-speed differences over 128 nodes
# without cancellation, so they inherit the same relative tolerance.
integral_tolerance(::Type{Float64}) = 1e-12
integral_tolerance(::Type{Float32}) = 5e-5

# Cloud number concentrations [1/m³]: bound-active (both ends), marine, and continental.
cloud_number_concentrations(FT) = (FT(1e3), FT(1e6), FT(50e6), FT(100e6),
                                   FT(200e6), FT(300e6), FT(1e9))

# `@allocated` measures the code generated at its own call site, so the helpers under test
# are wrapped in `@noinline` one-liners: that way the count reflects the helper rather than
# the surrounding `@testset` body.
@noinline allocated_liu_daum(Nᶜˡ, shape) = @allocated liu_daum_shape_parameter(Nᶜˡ, shape)
@noinline allocated_rain_fall_speed(D, ρ_correction, fall_speed) =
    @allocated rain_fall_speed(D, ρ_correction, fall_speed)

# Diameters bracketing every transition of the piecewise fall-speed law [m].
function fall_speed_test_diameters(FT)
    edges = (FT(134.43e-6), FT(1511.64e-6), FT(3477.84e-6))
    Ds = FT[FT(1e-6), FT(1e-5), FT(5e-4), FT(2.5e-3), FT(5e-3), FT(1e-2)]
    for Dᵗ in edges
        push!(Ds, prevfloat(Dᵗ), Dᵗ, nextfloat(Dᵗ))
    end
    return sort!(Ds)
end

@testset "P3 empirical parameter containers" begin

    #####
    ##### 1. Container construction, typing, and host-side validation
    #####

    @testset "Container typing and validation [$FT]" for FT in all_float_types()
        shape = CloudShapeParameters(FT)
        fall_speed = RainFallSpeedParameters(FT)
        ventilation = RainVentilationParameters(FT)

        for parameters in (shape, fall_speed, ventilation)
            @test isbits(parameters)
            @test isbitstype(typeof(parameters))
            @test all(isconcretetype, fieldtypes(typeof(parameters)))
        end

        @test shape isa CloudShapeParameters{FT}
        @test fall_speed isa RainFallSpeedParameters{FT}
        @test ventilation isa RainVentilationParameters{FT}

        @test fall_speed.branch_velocity_scales isa NTuple{3, FT}
        @test fall_speed.branch_mass_exponents isa NTuple{3, FT}
        @test fall_speed.transition_diameters isa NTuple{3, FT}

        # Defaults are the pinned values, in SI.
        @test shape.relative_dispersion_number_coefficient == FT(5.714e-10)
        @test shape.relative_dispersion_intercept == FT(0.2714)
        @test shape.minimum_shape_parameter == FT(2)
        @test shape.maximum_shape_parameter == FT(15)

        @test fall_speed.branch_velocity_scales == FT.((4579.5, 49.62, 17.32))
        @test fall_speed.branch_mass_exponents == FT.((2/3, 1/3, 1/6))
        @test fall_speed.transition_diameters == FT.((134.43e-6, 1511.64e-6, 3477.84e-6))
        @test fall_speed.plateau_velocity == FT(9.17)

        @test ventilation.constant_coefficient == FT(0.78)
        @test ventilation.reynolds_coefficient == FT(0.32)

        # Keywords are converted, never stored at the keyword's own precision.
        widened = CloudShapeParameters(FT; relative_dispersion_intercept = 0.3)
        @test widened.relative_dispersion_intercept === FT(0.3)

        # Validation happens on the host, in the constructor.
        @test_throws ArgumentError CloudShapeParameters(FT;
            relative_dispersion_number_coefficient = -1e-10)
        @test_throws ArgumentError CloudShapeParameters(FT; relative_dispersion_intercept = 0)
        @test_throws ArgumentError CloudShapeParameters(FT; relative_dispersion_intercept = -0.1)
        @test_throws ArgumentError CloudShapeParameters(FT; minimum_shape_parameter = 20)
        @test_throws ArgumentError RainFallSpeedParameters(FT;
            branch_velocity_scales = (-1, 49.62, 17.32))
        @test_throws ArgumentError RainFallSpeedParameters(FT;
            branch_mass_exponents = (2/3, -1/3, 1/6))
        @test_throws ArgumentError RainFallSpeedParameters(FT;
            transition_diameters = (0, 1511.64e-6, 3477.84e-6))
        @test_throws ArgumentError RainFallSpeedParameters(FT;
            transition_diameters = (1511.64e-6, 134.43e-6, 3477.84e-6))
        @test_throws ArgumentError RainFallSpeedParameters(FT;
            transition_diameters = (134.43e-6, 134.43e-6, 3477.84e-6))
        @test_throws ArgumentError RainFallSpeedParameters(FT; plateau_velocity = -1)
        @test_throws ArgumentError RainVentilationParameters(FT; constant_coefficient = -0.1)
        @test_throws ArgumentError RainVentilationParameters(FT; reynolds_coefficient = -0.1)
    end

    #####
    ##### 2. Default parity with the pinned implementation
    #####

    @testset "Default parity: liu_daum_shape_parameter [$FT]" for FT in all_float_types()
        shape = CloudShapeParameters(FT)
        rtol = pointwise_tolerance(FT)

        for Nᶜˡ in cloud_number_concentrations(FT)
            @test liu_daum_shape_parameter(Nᶜˡ, shape) ≈
                  reference_liu_daum_shape_parameter(Nᶜˡ) rtol=rtol
        end

        # The lower bound is genuinely active at high droplet counts, so parity there is
        # a statement about the clamp and not only about the regression.
        @test liu_daum_shape_parameter(FT(1e9), shape) == FT(2)

        # The convenience wrapper must agree with the explicit call.
        @test liu_daum_shape_parameter(FT(200e6)) === liu_daum_shape_parameter(FT(200e6), shape)

        # Construction-time diagnosis and the pre-computed freezing correction.
        for Nᶜˡ in cloud_number_concentrations(FT)
            cloud = CloudDropletProperties(FT; number_concentration = Nᶜˡ)
            @test cloud.shape_parameter ≈ reference_liu_daum_shape_parameter(Nᶜˡ) rtol=rtol
            @test cloud.freezing_psd_correction ≈
                  PPP.psd_correction_spherical_volume(reference_liu_daum_shape_parameter(Nᶜˡ)) rtol=rtol
        end
    end

    @testset "Default parity: rain_fall_speed [$FT]" for FT in all_float_types()
        fall_speed = RainFallSpeedParameters(FT)
        rtol = pointwise_tolerance(FT)

        for D in fall_speed_test_diameters(FT), ρ_correction in (one(FT), FT(1.3))
            @test rain_fall_speed(D, ρ_correction, fall_speed) ≈
                  reference_rain_fall_speed(D, ρ_correction) rtol=rtol
        end

        # Above the largest boundary the law is exactly the plateau, with no power-law
        # residue: a test that would fail if the branch inequality were `<=` instead
        # of `<`, or if the plateau were still in cm/s.
        Dᵗ = fall_speed.transition_diameters[3]
        @test rain_fall_speed(Dᵗ, one(FT), fall_speed) == FT(9.17)
        @test rain_fall_speed(FT(8e-3), one(FT), fall_speed) == FT(9.17)
        @test rain_fall_speed(prevfloat(Dᵗ), one(FT), fall_speed) < FT(9.17)

        # Density correction is a plain multiplicative factor.
        @test rain_fall_speed(FT(1e-3), FT(2), fall_speed) ≈
              2 * rain_fall_speed(FT(1e-3), one(FT), fall_speed)
    end

    @testset "Default parity: startup quadrature [$FT]" for FT in all_float_types()
        reference = P3_REFERENCE[FT]
        rtol = integral_tolerance(FT)

        velocity_mass = RainMassWeightedVelocityEvaluator(FT)
        velocity_number = RainNumberWeightedVelocityEvaluator(FT)
        velocity_diameter = RainEvaporationVentilationEvaluator(FT)

        for (i, log_slope) in enumerate(p3_reference_log_slopes(FT))
            @test velocity_mass(log_slope) ≈ reference.velocity_mass[i] rtol=rtol
            @test velocity_number(log_slope) ≈ reference.velocity_number[i] rtol=rtol
            @test velocity_diameter(log_slope) ≈
                  reference.velocity_diameter_integral[i] rtol=rtol
        end
    end

    @testset "Default parity: full ventilation integral [$FT]" for FT in all_float_types()
        reference = P3_REFERENCE[FT]
        rtol = integral_tolerance(FT)

        p3 = PredictedParticlePropertiesMicrophysics(FT)
        ν = FT(P3_REFERENCE_VENTILATION_VISCOSITY)
        Dᵛ = FT(P3_REFERENCE_VENTILATION_DIFFUSIVITY)

        for (i, (qʳ, nʳ)) in enumerate(p3_reference_ventilation_states(FT))
            ventilation = rain_ventilation_integral(p3.rain.evaporation, p3.rain.ventilation,
                                                    qʳ, nʳ, ν, Dᵛ, p3.process_rates)
            @test ventilation.integral ≈ reference.ventilation_integral[i] rtol=rtol
        end
    end

    #####
    ##### 3. Parameter plumbing and sensitivity
    #####

    @testset "Cloud shape parameters reach every μᶜˡ path [$FT]" for FT in all_float_types()
        # A wider intercept narrows nothing physically; it simply moves χ, and with it
        # every μᶜˡ diagnosed from a local droplet number.
        custom_shape = CloudShapeParameters(FT; relative_dispersion_intercept = 0.35)

        default_cloud = CloudDropletProperties(FT; number_concentration = 100e6)
        custom_cloud = CloudDropletProperties(FT; number_concentration = 100e6,
                                              shape_parameters = custom_shape)

        # (1) construction-time shape_parameter
        @test custom_cloud.shape_parameters === custom_shape
        @test custom_cloud.shape_parameter != default_cloud.shape_parameter
        @test custom_cloud.shape_parameter ≈ liu_daum_shape_parameter(FT(100e6), custom_shape)
        @test custom_cloud.freezing_psd_correction != default_cloud.freezing_psd_correction

        default_p3 = PredictedParticlePropertiesMicrophysics(FT)
        custom_p3 = PredictedParticlePropertiesMicrophysics(FT; cloud = custom_cloud)
        @test custom_p3.cloud.shape_parameters === custom_shape

        ρ = FT(1.2)
        qᶜˡ = FT(5e-4)
        nᶜˡ = FT(100e6) / ρ         # specific number [1/kg] giving Nᶜˡ = 100e6 m⁻³

        # (2) prognostic diagnosis
        default_dsd = diagnose_cloud_dsd(default_p3, qᶜˡ, nᶜˡ, ρ)
        custom_dsd = diagnose_cloud_dsd(custom_p3, qᶜˡ, nᶜˡ, ρ)
        @test default_dsd.μᶜˡ ≈ liu_daum_shape_parameter(FT(100e6), CloudShapeParameters(FT))
        @test custom_dsd.μᶜˡ ≈ liu_daum_shape_parameter(FT(100e6), custom_shape)
        @test custom_dsd.μᶜˡ != default_dsd.μᶜˡ
        # The slope depends on μᶜˡ, so the whole diagnosed DSD moves with the fit.
        @test custom_dsd.λᶜˡ != default_dsd.λᶜˡ

        # An explicit construction-time `shape_parameter` override must NOT be what the
        # prognostic path uses: that path re-diagnoses from the local number.
        overridden_p3 = PredictedParticlePropertiesMicrophysics(FT;
            cloud = CloudDropletProperties(FT; number_concentration = 100e6,
                                           shape_parameter = 4))
        @test overridden_p3.cloud.shape_parameter == FT(4)
        @test diagnose_cloud_dsd(overridden_p3, qᶜˡ, nᶜˡ, ρ).μᶜˡ ≈ default_dsd.μᶜˡ

        # (3) immersion freezing reads the configured local relation
        T = FT(265)
        Nᶜˡ = FT(100e6)
        default_freezing = immersion_freezing_cloud_rate(default_p3, qᶜˡ, Nᶜˡ, T, ρ)
        custom_freezing = immersion_freezing_cloud_rate(custom_p3, qᶜˡ, Nᶜˡ, T, ρ)
        @test default_freezing[1] > 0
        @test custom_freezing[1] != default_freezing[1]
        # The number rate carries no PSD correction, so only the mass rate moves.
        @test custom_freezing[2] ≈ default_freezing[2]
    end

    @testset "Fall-speed parameters reach all three startup tables [$FT]" for FT in all_float_types()
        default_rain = RainProperties(FT)
        default_tables = tabulate_rain_from_quadrature(default_rain, CPU(), FT)

        # Doubling every velocity scale and the plateau doubles V(D) everywhere, so the
        # two velocity moment ratios double exactly and the √(V D) integral grows by √2.
        # A table built from a stale fall-speed law would not move at all.
        doubled = RainFallSpeedParameters(FT;
            branch_velocity_scales = 2 .* (4579.5, 49.62, 17.32),
            plateau_velocity = 2 * 9.17)
        doubled_tables = tabulate_rain_from_quadrature(RainProperties(FT; fall_speed = doubled),
                                                       CPU(), FT)

        rtol = integral_tolerance(FT)
        for log_slope in p3_reference_log_slopes(FT)
            @test doubled_tables.velocity_mass(log_slope) ≈
                  2 * default_tables.velocity_mass(log_slope) rtol=rtol
            @test doubled_tables.velocity_number(log_slope) ≈
                  2 * default_tables.velocity_number(log_slope) rtol=rtol
            @test doubled_tables.evaporation(log_slope) ≈
                  sqrt(FT(2)) * default_tables.evaporation(log_slope) rtol=rtol
        end

        # The materialized container keeps the physics parameters and replaces only the
        # lookup placeholders.
        @test doubled_tables.fall_speed == doubled
        @test doubled_tables.ventilation == RainVentilationParameters(FT)
        @test doubled_tables.maximum_mean_diameter == default_rain.maximum_mean_diameter
        @test isnothing(default_rain.velocity_mass)
        @test !isnothing(doubled_tables.velocity_mass)
    end

    @testset "Ventilation coefficients enter the expected terms [$FT]" for FT in all_float_types()
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        parameters = p3.process_rates
        table = p3.rain.evaporation

        qʳ = FT(1e-4)
        nʳ = FT(1e3)
        ν = FT(1.5e-5)
        Dᵛ = FT(2.2e-5)

        default_ventilation = RainVentilationParameters(FT)
        no_constant = RainVentilationParameters(FT; constant_coefficient = 0)
        no_reynolds = RainVentilationParameters(FT; reynolds_coefficient = 0)

        full = rain_ventilation_integral(table, default_ventilation, qʳ, nʳ, ν, Dᵛ, parameters)
        reynolds_only = rain_ventilation_integral(table, no_constant, qʳ, nʳ, ν, Dᵛ, parameters)
        constant_only = rain_ventilation_integral(table, no_reynolds, qʳ, nʳ, ν, Dᵛ, parameters)

        # The integral is exactly f₁ᵣ × (analytical term) + f₂ᵣ × (Reynolds term), so
        # zeroing one coefficient must remove precisely that term.
        @test full.integral ≈ reynolds_only.integral + constant_only.integral
        @test constant_only.integral ≈ FT(0.78) / full.λʳ^2
        @test reynolds_only.integral ≈ FT(0.32) * cbrt(ν / Dᵛ) / sqrt(ν) * table(log10(full.λʳ))
        # The slope and intercept do not depend on the ventilation coefficients.
        @test reynolds_only.λʳ == full.λʳ
        @test reynolds_only.Nʳ₀ == full.Nʳ₀

        # Both coefficients are linear, so doubling one doubles its term.
        doubled_constant = RainVentilationParameters(FT; constant_coefficient = 1.56)
        @test rain_ventilation_integral(table, doubled_constant, qʳ, nʳ, ν, Dᵛ, parameters).integral ≈
              full.integral + constant_only.integral

        # Direct evaporation carries them through.
        thermodynamic_factor = FT(1e8)
        S = FT(0.8)
        full_rate = rain_evaporation_rate(table, default_ventilation, qʳ, nʳ, S,
                                          thermodynamic_factor, parameters, ν, Dᵛ, FT)
        reduced_rate = rain_evaporation_rate(table, no_reynolds, qʳ, nʳ, S,
                                             thermodynamic_factor, parameters, ν, Dᵛ, FT)
        @test full_rate < 0                 # subsaturated: the internal helper is negative
        @test reduced_rate > full_rate      # dropping a positive term shrinks the magnitude

        # ... and so does the coupled saturation-adjustment relaxation coefficient.
        custom_p3 = PredictedParticlePropertiesMicrophysics(FT;
            rain = RainProperties(FT; ventilation = no_reynolds))
        constants = ThermodynamicConstants(FT)
        transport = air_transport_properties(FT(290), FT(90000), constants)
        ρ = FT(1.1)
        default_relaxation = PPP.rain_vapor_relaxation_coefficient(p3, qʳ, nʳ, ρ, transport)
        custom_relaxation = PPP.rain_vapor_relaxation_coefficient(custom_p3, qʳ, nʳ, ρ, transport)
        @test default_relaxation > 0
        @test custom_relaxation < default_relaxation
    end

    @testset "P3Microphysics preserves a custom rain configuration [$FT]" for FT in all_float_types()
        fall_speed = RainFallSpeedParameters(FT; plateau_velocity = 7.5,
                                             transition_diameters = (150e-6, 1400e-6, 3200e-6))
        ventilation = RainVentilationParameters(FT; constant_coefficient = 0.7,
                                                reynolds_coefficient = 0.4)
        rain = RainProperties(FT; fall_speed, ventilation)

        p3 = PredictedParticlePropertiesMicrophysics(FT; rain)

        # Survives `read_lookup_tables` and `tabulate_rain_from_quadrature`.
        @test p3.rain.fall_speed == fall_speed
        @test p3.rain.ventilation == ventilation
        @test p3.rain.velocity_mass isa PPP.TabulatedFunction1D
        @test p3.rain.velocity_number isa PPP.TabulatedFunction1D
        @test p3.rain.evaporation isa PPP.TabulatedFunction1D

        # The tables were actually built from the custom law, not the default one.
        default_p3 = PredictedParticlePropertiesMicrophysics(FT)
        @test p3.rain.velocity_mass(FT(2.5)) != default_p3.rain.velocity_mass(FT(2.5))
        @test p3.rain.evaporation(FT(2.5)) != default_p3.rain.evaporation(FT(2.5))

        # The materialized container is what the runtime rates read.
        @test p3.rain.ventilation.constant_coefficient == FT(0.7)
        @test p3.rain.ventilation.reynolds_coefficient == FT(0.4)
    end

    #####
    ##### 4. Every one of the sixteen promoted scalars is active somewhere
    #####

    @testset "Each promoted scalar changes a result [$FT]" for FT in all_float_types()
        # --- cloud width (4 scalars) --------------------------------------------------
        # Probed over concentrations spanning both clamps, so the bounds are exercised in
        # the regime where they bind rather than where they are inert.
        cloud_probe(shape) = map(N -> liu_daum_shape_parameter(N, shape),
                                 cloud_number_concentrations(FT))
        default_cloud_probe = cloud_probe(CloudShapeParameters(FT))

        cloud_perturbations = (
            ("relative_dispersion_number_coefficient",
             CloudShapeParameters(FT; relative_dispersion_number_coefficient = 6.5e-10)),
            ("relative_dispersion_intercept",
             CloudShapeParameters(FT; relative_dispersion_intercept = 0.3)),
            # The lower bound binds at Nᶜˡ = 10⁹ m⁻³, where the regression returns μᶜˡ < 2.
            ("minimum_shape_parameter",
             CloudShapeParameters(FT; minimum_shape_parameter = 3)),
            # The default upper bound never binds: with b = 0.2714 the regression caps
            # μᶜˡ at 1/b² - 1 ≈ 12.58 < 15. Lowering it into that range is the regime
            # where the parameter is active at all.
            ("maximum_shape_parameter",
             CloudShapeParameters(FT; maximum_shape_parameter = 8)),
        )

        @testset "cloud $name" for (name, shape) in cloud_perturbations
            @test cloud_probe(shape) != default_cloud_probe
        end

        # --- rain fall speed (10 scalars) ---------------------------------------------
        pointwise_probe(fall_speed) = map(D -> rain_fall_speed(D, one(FT), fall_speed),
                                          fall_speed_test_diameters(FT))

        function table_probe(fall_speed)
            tables = tabulate_rain_from_quadrature(RainProperties(FT; fall_speed), CPU(), FT)
            slopes = p3_reference_log_slopes(FT)
            return (map(tables.velocity_mass, slopes),
                    map(tables.velocity_number, slopes),
                    map(tables.evaporation, slopes))
        end

        default_fall_speed = RainFallSpeedParameters(FT)
        default_pointwise_probe = pointwise_probe(default_fall_speed)
        default_table_probe = table_probe(default_fall_speed)

        fall_speed_perturbations = (
            ("branch_velocity_scales[1]",
             RainFallSpeedParameters(FT; branch_velocity_scales = (4700.0, 49.62, 17.32))),
            ("branch_velocity_scales[2]",
             RainFallSpeedParameters(FT; branch_velocity_scales = (4579.5, 55.0, 17.32))),
            ("branch_velocity_scales[3]",
             RainFallSpeedParameters(FT; branch_velocity_scales = (4579.5, 49.62, 20.0))),
            ("branch_mass_exponents[1]",
             RainFallSpeedParameters(FT; branch_mass_exponents = (0.7, 1/3, 1/6))),
            ("branch_mass_exponents[2]",
             RainFallSpeedParameters(FT; branch_mass_exponents = (2/3, 0.4, 1/6))),
            ("branch_mass_exponents[3]",
             RainFallSpeedParameters(FT; branch_mass_exponents = (2/3, 1/3, 0.2))),
            ("transition_diameters[1]",
             RainFallSpeedParameters(FT; transition_diameters = (300e-6, 1511.64e-6, 3477.84e-6))),
            ("transition_diameters[2]",
             RainFallSpeedParameters(FT; transition_diameters = (134.43e-6, 1200e-6, 3477.84e-6))),
            ("transition_diameters[3]",
             RainFallSpeedParameters(FT; transition_diameters = (134.43e-6, 1511.64e-6, 3000e-6))),
            ("plateau_velocity",
             RainFallSpeedParameters(FT; plateau_velocity = 12)),
        )

        @testset "fall speed $name" for (name, fall_speed) in fall_speed_perturbations
            @test pointwise_probe(fall_speed) != default_pointwise_probe
            # ... and the change survives into the tables the model actually reads.
            @test table_probe(fall_speed) != default_table_probe
        end

        # --- rain ventilation (2 scalars) ---------------------------------------------
        evaporation_table = tabulate_rain_from_quadrature(RainProperties(FT), CPU(), FT).evaporation
        process_rates = ProcessRateParameters(FT)
        ventilation_probe(ventilation) =
            rain_ventilation_integral(evaporation_table, ventilation, FT(1e-4), FT(1e3),
                                      FT(1.5e-5), FT(2.2e-5), process_rates).integral
        default_ventilation_probe = ventilation_probe(RainVentilationParameters(FT))

        ventilation_perturbations = (
            ("constant_coefficient",
             RainVentilationParameters(FT; constant_coefficient = 0.9)),
            ("reynolds_coefficient",
             RainVentilationParameters(FT; reynolds_coefficient = 0.4)),
        )

        @testset "ventilation $name" for (name, ventilation) in ventilation_perturbations
            @test ventilation_probe(ventilation) != default_ventilation_probe
        end

        # 4 + 10 + 2 = 16 promoted scalars, each shown active above.
        @test length(cloud_perturbations) + length(fall_speed_perturbations) +
              length(ventilation_perturbations) == 16
    end

    #####
    ##### 5. Type stability, precision, and architecture adaptation
    #####

    @testset "Type stability and precision [$FT]" for FT in all_float_types()
        shape = CloudShapeParameters(FT)
        fall_speed = RainFallSpeedParameters(FT)
        ventilation = RainVentilationParameters(FT)

        @test fieldtypes(CloudShapeParameters{FT}) == (FT, FT, FT, FT)
        @test fieldtypes(RainVentilationParameters{FT}) == (FT, FT)
        @test fieldtypes(RainFallSpeedParameters{FT}) ==
              (NTuple{3, FT}, NTuple{3, FT}, NTuple{3, FT}, FT)

        @test @inferred(liu_daum_shape_parameter(FT(1e8), shape)) isa FT
        @test @inferred(rain_fall_speed(FT(1e-4), one(FT), fall_speed)) isa FT

        p3 = PredictedParticlePropertiesMicrophysics(FT)
        integral = @inferred rain_ventilation_integral(p3.rain.evaporation, ventilation,
                                                       FT(1e-4), FT(1e3), FT(1.5e-5),
                                                       FT(2.2e-5), p3.process_rates)
        @test integral.integral isa FT
        @test integral.λʳ isa FT
        @test integral.Nʳ₀ isa FT

        # No `Float64` promotion in a `Float32` scheme, including through the containers
        # and through the μᶜˡ diagnosis that a `Float64` keyword default would otherwise
        # widen.
        @test p3.cloud.shape_parameters isa CloudShapeParameters{FT}
        @test p3.rain.fall_speed isa RainFallSpeedParameters{FT}
        @test p3.rain.ventilation isa RainVentilationParameters{FT}
        @test p3.cloud.shape_parameter isa FT

        dsd = @inferred diagnose_cloud_dsd(p3, FT(5e-4), FT(1e8), FT(1.2))
        @test dsd.μᶜˡ isa FT
        @test dsd.λᶜˡ isa FT

        # Containers built at another precision are converted, not stored as-is.
        mixed = PredictedParticlePropertiesMicrophysics(FT;
            cloud = CloudDropletProperties(FT;
                        shape_parameters = CloudShapeParameters(Float64;
                                                                relative_dispersion_intercept = 0.3)),
            rain = RainProperties(FT;
                        fall_speed = RainFallSpeedParameters(Float64; plateau_velocity = 8.5)))
        @test mixed.cloud.shape_parameters isa CloudShapeParameters{FT}
        @test mixed.cloud.shape_parameters.relative_dispersion_intercept == FT(0.3)
        @test mixed.rain.fall_speed isa RainFallSpeedParameters{FT}
        @test mixed.rain.fall_speed.plateau_velocity == FT(8.5)
    end

    @testset "Kernel helpers are allocation-free [$FT]" for FT in all_float_types()
        shape = CloudShapeParameters(FT)
        fall_speed = RainFallSpeedParameters(FT)

        # Warm up first: `@allocated` counts compilation on the very first call.
        allocated_liu_daum(FT(1e8), shape)
        allocated_rain_fall_speed(FT(1e-4), one(FT), fall_speed)

        @test allocated_liu_daum(FT(1e8), shape) == 0
        @test allocated_rain_fall_speed(FT(1e-4), one(FT), fall_speed) == 0
    end

    @testset "Architecture adaptation preserves custom parameters" begin
        FT = Float64
        shape = CloudShapeParameters(FT; relative_dispersion_intercept = 0.3)
        fall_speed = RainFallSpeedParameters(FT; plateau_velocity = 8.5)
        ventilation = RainVentilationParameters(FT; reynolds_coefficient = 0.4)

        p3 = PredictedParticlePropertiesMicrophysics(FT;
            cloud = CloudDropletProperties(FT; shape_parameters = shape),
            rain = RainProperties(FT; fall_speed, ventilation))

        adapted = on_architecture(default_arch, p3)
        @test adapted.cloud.shape_parameters == shape
        @test adapted.rain.fall_speed == fall_speed
        @test adapted.rain.ventilation == ventilation

        # Step a model on `default_arch` so the custom values are exercised inside the
        # microphysics kernels after adaptation, not only in the host container.
        grid = RectilinearGrid(default_arch, FT; size = (2, 2, 2), extent = (100, 100, 100))
        constants = ThermodynamicConstants(FT)
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = FT(101325),
                                         potential_temperature = FT(285))
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid; dynamics, thermodynamic_constants = constants,
                                microphysics = p3)
        set!(model; θ = FT(283), qᵛ = FT(0.012), qᶜˡ = FT(0.002), qʳ = FT(5e-4),
             enforce_mass_conservation = false)
        time_step!(model, 1)

        for name in (:ρqᶜˡ, :ρqʳ, :ρqⁱ)
            @test all(isfinite, Array(interior(model.microphysical_fields[name])))
        end
    end
end
