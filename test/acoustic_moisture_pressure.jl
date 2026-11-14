include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Test

#####
##### The moisture's part of the pressure that the acoustic substep loop integrates.
#####
##### A sharp vapor jump under an inversion, advected by a uniform wind over small potential
##### temperature noise (the case of NumericalEarth/Breeze.jl#1093, after Xiao et al. 2015,
##### Fig. 7). The pressure does not feel the advection of the jump, so the vertical velocity in
##### the jump layer stays at the level the noise sets. A loop that reads the perturbation of the
##### thermodynamic density as its pressure, and so sees the vapor one stage late, amplifies it
##### there several-fold within ten minutes.
#####

interpolate_sounding(zs, vs, z) = (k = clamp(searchsortedlast(zs, z), 1, length(zs) - 1);
                                   vs[k] + (vs[k+1] - vs[k]) * clamp((z - zs[k]) / (zs[k+1] - zs[k]), 0, 1))

sounding_θ(z) = interpolate_sounding([0, 430, 1000, 1200, 1400, 1600, 2000, 3000, 5000],
                                     [295.1, 297.4, 303.1, 307.7, 312.6, 314.2, 314.6, 315.0, 315.4], z)

function sounding_q(z)
    r = 1e-3 * interpolate_sounding([0, 1150, 1600, 5000], [13, 13, 1, 1], z)
    return r / (1 + r)
end

@testset "Vapor jump advected under an inversion [$FT]" for FT in (Float64,)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (40, 250), halo = (5, 5), x = (0, 4e3), z = (0, 5e3),
                           topology = (Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps = 6); base_pressure = 97e3,
                                    reference_potential_temperature = sounding_θ,
                                    reference_vapor_mass_fraction = sounding_q)
    model = AtmosphereModel(grid; dynamics, advection = WENO(order = 5))

    # Deterministic noise of amplitude 0.1 K below 2 km.
    noise(x, z) = 2 * mod(43758.5453 * sin(0.129898x + 3.91165z), 1) - 1
    set!(model; θ = (x, z) -> sounding_θ(z) + 0.1 * (z < 2e3) * noise(x, z), qᵛ = (x, z) -> sounding_q(z),
         u = 12, ρ = Breeze.HydrostaticallyBalancedDensity())

    for _ in 1:1200
        time_step!(model, 0.5)
    end

    w = Array(interior(model.velocities.w))
    jump_layer = findall(z -> 1e3 <= z <= 1.8e3, znodes(grid, Face()))
    wʲ = w[:, 1, jump_layer]
    w′ = wʲ .- sum(wʲ; dims = 1) / size(wʲ, 1)

    @test maximum(abs, w′) < 0.08
end

#####
##### Under saturation adjustment the pressure the core re-diagnoses is that of the adjusted
##### partition. Its slopes (`adjusted_partition_response`) must be the derivatives of the pressure
##### after adjustment, in saturated air as in unsaturated air: finite differences in the
##### thermodynamic density, the moisture density and the dry density. The loop's coefficient Cᴸ
##### must be its slope along the loop's own flux, a compression of all three at fixed θ and
##### mixing ratios. The slopes of the saturation vapor density and of a mixed-phase liquid
##### fraction are centered differences themselves, which sets the tolerance.
#####

using Breeze.CompressibleEquations: refresh_linearization_basic_state!, moisture_pressure_sensitivities,
                                    adjusted_partition_response
using Breeze.Microphysics: SaturationAdjustment, WarmPhaseEquilibrium, MixedPhaseEquilibrium, adjust_thermodynamic_state
using Breeze.Thermodynamics: LiquidIceDensityState, MoistureMassFractions, air_pressure, temperature,
                             temperature_and_sensitivity, mixture_gas_constant
using GPUArraysCore: @allowscalar

@testset "Loop pressure slopes under saturation adjustment [$(nameof(typeof(equilibrium)))]" for
        equilibrium in (WarmPhaseEquilibrium(), MixedPhaseEquilibrium(Float64))

    Oceananigans.defaults.FloatType = Float64
    constants = ThermodynamicConstants(Float64)
    solver = Breeze.NewtonSolver(Float64; reltol=0.0, abstol=1e-11, maxiter=50)
    adjustment = SaturationAdjustment(Float64; equilibrium,
                                      solver=Breeze.SecantSolver(Float64; reltol=0.0, abstol=1e-12, maxiter=200))
    formulation = LiquidIcePotentialTemperatureFormulation(temperature_solver=solver)
    grid = RectilinearGrid(default_arch, Float64; size=4, z=(0, 400), topology=(Flat, Flat, Bounded))
    pˢᵗ = 1e5

    # The state after adjustment of the given densities, with all the moisture as vapor.
    function adjusted_state(ρᵈ, ρθ, ρʷ)
        ρ = ρᵈ + ρʷ
        𝒰 = LiquidIceDensityState(ρθ / ρᵈ, MoistureMassFractions(ρʷ / ρ), pˢᵗ, ρ, solver)
        return adjust_thermodynamic_state(𝒰, adjustment, constants)
    end

    adjusted_pressure(ρᵈ, ρθ, ρʷ) = air_pressure(adjusted_state(ρᵈ, ρθ, ρʷ), constants)

    # Warm cloud, unsaturated air, and cloud between and below the limits of the mixed phase.
    states = ((1.1, 285.0, 0.02), (1.1, 300.0, 0.005), (0.8, 300.0, 0.006), (0.5, 290.0, 0.002))
    saturation = (true, false, true, true)

    for ((ρᵈ, θ, ρʷ), saturated) in zip(states, saturation)
        ρθ = ρᵈ * θ
        ρ = ρᵈ + ρʷ
        𝒰 = adjusted_state(ρᵈ, ρθ, ρʷ)
        q = 𝒰.moisture_mass_fractions
        T = temperature(𝒰, constants)
        cloud = q.liquid + q.ice
        @test (cloud > 0) == saturated

        # The slopes at the partition the state holds, and the adjusted partition's response.
        _, θ∂T∂θ = temperature_and_sensitivity(𝒰, constants)
        C̄ = ρ * mixture_gas_constant(q, constants) * θ∂T∂θ / ρθ
        Λᵛ, Λˡ, Λⁱ = moisture_pressure_sensitivities(formulation, 𝒰, T, ρᵈ, constants)
        fᵛ, fˡ, fⁱ = saturated ? (0.0, q.liquid / cloud, q.ice / cloud) : (1.0, 0.0, 0.0)
        Λᵐ = fᵛ * Λᵛ + fˡ * Λˡ + fⁱ * Λⁱ
        σ, E = adjusted_partition_response(equilibrium, 𝒰, T, ρᵈ, ρʷ / ρ, fᵛ, fˡ, fⁱ, Λᵛ, Λˡ, Λⁱ, constants)
        ζ = 1 / (1 + C̄ * σ)

        model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation, microphysics=adjustment,
                                dynamics=CompressibleDynamics(SplitExplicitTimeDiscretization(); standard_pressure=pˢᵗ))
        set!(model; ρᵈ, ρqᵉ=ρʷ, ρθ)

        substepper = model.timestepper.substepper
        refresh_linearization_basic_state!(substepper, model)
        Cᴸ, θᴸ, Λᴴ = @allowscalar (substepper.linearization_pressure_coefficient[1, 1, 2],
                                   substepper.linearization_potential_temperature[1, 1, 2],
                                   substepper.fixed_composition_sensitivity[1, 1, 2])

        δθ, δ, ε = 1e-4, 1e-7, 1e-6
        ∂p_∂ρθ = (adjusted_pressure(ρᵈ, ρθ + δθ, ρʷ) - adjusted_pressure(ρᵈ, ρθ - δθ, ρʷ)) / 2δθ
        ∂p_∂ρʷ = (adjusted_pressure(ρᵈ, ρθ, ρʷ + δ) - adjusted_pressure(ρᵈ, ρθ, ρʷ - δ)) / 2δ
        ∂p_∂ρᵈ = (adjusted_pressure(ρᵈ + δ, ρθ, ρʷ) - adjusted_pressure(ρᵈ - δ, ρθ, ρʷ)) / 2δ
        compression = (adjusted_pressure((1 + ε) * ρᵈ, (1 + ε) * ρθ, (1 + ε) * ρʷ) -
                       adjusted_pressure((1 - ε) * ρᵈ, (1 - ε) * ρθ, (1 - ε) * ρʷ)) / 2ε

        rtol = 2e-4

        # The tangent of the adjusted pressure in each density.
        @test ζ * C̄ ≈ ∂p_∂ρθ rtol=rtol
        @test -ζ * C̄ * Λᵐ ≈ ∂p_∂ρʷ rtol=rtol
        @test ζ * C̄ * (Λᴴ + E + Λᵐ * ρʷ / ρᵈ) ≈ ∂p_∂ρᵈ rtol=10rtol

        # The loop's coefficient: the slope along its own flux, which carries θᴸ + Λᴴ.
        @test θᴸ ≈ θ + Λᴴ
        @test Cᴸ ≈ ζ * C̄ * (1 + E / θᴸ)
        @test Cᴸ * θᴸ * ρᵈ ≈ compression rtol=rtol

        if saturated
            # Warming evaporates condensate, which takes up heat: the adjusted pressure responds
            # to the thermodynamic density at less than the held partition's rate.
            @test ζ < 1
            @test E > 0
        else
            @test ζ == 1
            @test E == 0
            @test Λᴴ == 0
            @test Cᴸ ≈ C̄
        end

        if saturated && T > 280
            # In warm cloud the response at fixed densities is less than half. The response to a
            # compression is that along the moist adiabat, 10 to 15% below the held partition's
            # (NumericalEarth/Breeze.jl#1086), with the dry air adding the rest of its partial
            # pressure.
            @test ζ < 0.6
            @test 0.8 < compression / (C̄ * ρθ) < 0.92
        end
    end
end

#####
##### The pressure a stage's loop starts from, Cᴸ times the perturbation `initialize_stage_perturbations!`
##### gives it, must be the pressure the core re-diagnoses at the start state relative to the
##### stage-entry state: for a start state that differs in the thermodynamic density, in the moisture
##### density, in the dry density, in each condensate mass, and in all of them at once (a compression at
##### fixed θ and mixing ratios). The scheme decides which pressure that is. One that carries its
##### condensate holds the partition; saturation adjustment, alone or as the cloud formation of a bulk
##### scheme with precipitation, re-diagnoses the adjusted partition in cloud and the held one elsewhere.
##### At the held partition the loop would overestimate the response of saturated air to the
##### thermodynamic and moisture densities nearly threefold (NumericalEarth/Breeze.jl#1258).
#####

using Breeze.CompressibleEquations: prepare_acoustic_cache!, initialize_stage_perturbations!
using Breeze.AtmosphereModels: moisture_prognostic_name, specific_prognostic_moisture
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: update_state!
using CloudMicrophysics: CloudMicrophysics

const BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

# The ratio, in every cell, of the loop's start pressure to the re-diagnosed one, when the start state
# is the model's state with the densities `perturbed` scaled by 1 + ε.
function start_pressure_ratio(model, perturbed, ε)
    timestepper = model.timestepper
    substepper = timestepper.substepper
    U = prognostic_fields(model)
    U⁰ = timestepper.U⁰

    update_state!(model)
    prepare_acoustic_cache!(substepper, model)
    p = Array(interior(model.dynamics.pressure))

    for name in keys(U)
        parent(U⁰[name]) .= parent(U[name])
    end

    for name in perturbed
        parent(U⁰[name]) .*= 1 + ε
    end

    initialize_stage_perturbations!(substepper, model, U⁰)
    p′ = Array(interior(substepper.linearization_pressure_coefficient)) .*
         Array(interior(substepper.density_potential_temperature_perturbation))

    for name in keys(U)
        parent(U[name]) .= parent(U⁰[name])
    end

    update_state!(model)
    δp = Array(interior(model.dynamics.pressure)) .- p

    return p′ ./ δp
end

@testset "Loop start pressure is the re-diagnosed pressure [$label]" for (label, cloud_formation, condensates) in
        (("warm-phase saturation adjustment", :warm, (;)),
         ("mixed-phase saturation adjustment", :mixed, (;)),
         ("one-moment, warm-phase saturation adjustment", :warm, (; ρqʳ = 2e-3)),
         ("one-moment, mixed-phase saturation adjustment", :mixed, (; ρqʳ = 2e-3, ρqˢⁿ = 1e-3)),
         ("one-moment, non-equilibrium cloud", :non_equilibrium, (; ρqᶜˡ = 3e-3, ρqʳ = 2e-3)))

    FT = Float64
    Oceananigans.defaults.FloatType = FT

    # Solvers converged well below the perturbation, whose response is 10⁻⁶ of the pressure.
    temperature_solver = Breeze.NewtonSolver(FT; reltol=0, abstol=1e-11, maxiter=50)
    solver = Breeze.SecantSolver(FT; reltol=0, abstol=1e-11, maxiter=200)
    adjustment(equilibrium) = SaturationAdjustment(FT; equilibrium, solver)

    microphysics = if cloud_formation == :non_equilibrium
        OneMomentCloudMicrophysics(FT)
    else
        equilibrium = cloud_formation == :warm ? WarmPhaseEquilibrium() : MixedPhaseEquilibrium(FT)
        isempty(condensates) ? adjustment(equilibrium) : OneMomentCloudMicrophysics(FT; cloud_formation = adjustment(equilibrium))
    end

    grid = RectilinearGrid(default_arch, FT; size=6, z=(0, 600), topology=(Flat, Flat, Bounded))
    formulation = LiquidIcePotentialTemperatureFormulation(; temperature_solver)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); standard_pressure=1e5)

    # Warm cloud below 200 m, unsaturated air to 400 m, and above it cloud near 260 K, where a
    # mixed-phase equilibrium holds both liquid and ice.
    ρᵈ(z) = z < 400 ? 1.1 : 0.8
    θ(z) = z < 200 ? 285.0 : z < 400 ? 300.0 : 295.0
    ρqᵐ(z) = z < 200 ? 0.02 : z < 400 ? 0.005 : 0.004
    moisture_name = moisture_prognostic_name(microphysics)

    function initialized_model()
        model = AtmosphereModel(grid; formulation, dynamics, microphysics)
        set!(model; ρᵈ, ρθ = z -> ρᵈ(z) * θ(z), NamedTuple{(moisture_name,)}((ρqᵐ,))..., condensates...)
        return model
    end

    names = (:ρθ, moisture_name, :ρᵈ, keys(condensates)...)
    saturated = cloud_formation != :non_equilibrium

    for perturbed in (map(tuple, names)..., names)
        model = initialized_model()
        ratio = start_pressure_ratio(model, perturbed, 1e-6)
        @test all(r -> isapprox(r, 1; rtol=1e-4), ratio)

        if saturated
            # The adjustment diagnoses cloud in the two layers and none between.
            qᶜ = Array(interior(specific_prognostic_moisture(model))) .- Array(interior(model.microphysical_fields.qᵛ))
            @test all(qᶜ[1, 1, [1, 2, 5, 6]] .> 1e-4)
            @test all(abs.(qᶜ[1, 1, 3:4]) .< 1e-12)
        end
    end
end
