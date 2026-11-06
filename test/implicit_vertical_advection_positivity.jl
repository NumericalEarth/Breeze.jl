include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze: PrescribedDynamics
using Breeze.CompressibleEquations: CompressibleDynamics, SplitExplicitTimeDiscretization
using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.TimeSteppers: ExplicitTimeDiscretization
using Test

const AM = Breeze.AtmosphereModels
const TH = Breeze.Thermodynamics

# A test-only scheme: rain that falls at a prescribed speed, with no phase changes or sources,
# so the column's rain mass changes only by outflow through the bottom.
struct FallingRain{FT}
    fall_speed :: FT
    closed_bottom :: Bool
end

AM.prognostic_field_names(::FallingRain) = (:ρqʳ,)
AM.moisture_prognostic_name(::FallingRain) = :ρqᵛ
AM.condensate_liquid_fraction(::FallingRain, ::Val{:ρqʳ}) = 1
AM.sedimentation_velocity(::FallingRain, μ, ::Val{:ρqʳ}) = μ.wʳ
AM.compute_microphysical_tendencies!(::FallingRain, model) = nothing
AM.microphysics_model_update!(::FallingRain, model) = nothing
@inline AM.maybe_adjust_thermodynamic_state(state, ::FallingRain, vapor, constants) = state
@inline AM.microphysical_state(::FallingRain, ρ, μ::NamedTuple, state, velocities) = (; qʳ=μ.ρqʳ / ρ)
@inline AM.moisture_fractions(::FallingRain, state::NamedTuple, vapor) = TH.MoistureMassFractions(vapor, state.qʳ, zero(vapor))
AM.liquid_mass_fraction(::FallingRain, model) = model.microphysical_fields.qʳ
AM.ice_mass_fraction(::FallingRain, model) = nothing
AM.materialize_microphysical_fields(::FallingRain, grid, bcs) =
    (; ρqʳ=CenterField(grid; boundary_conditions=bcs.ρqʳ), qʳ=CenterField(grid), qᵛ=CenterField(grid),
       wʳ=AM.sedimentation_velocity_field(grid))

@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, m::FallingRain, state, ρ, thermo, constants)
    @inbounds μ.qʳ[i, j, k] = state.qʳ
    @inbounds μ.qᵛ[i, j, k] = thermo.moisture_mass_fractions.vapor
    @inbounds μ.wʳ[i, j, k] = ifelse(m.closed_bottom & (k == 1), zero(m.fall_speed), -m.fall_speed)
    return nothing
end

const Δz_column = 50
const fall_speed = 7

# 1 g/kg of rain in a 2 km layer over 50 m levels, falling at 7 m/s. `w₀` adds a prescribed
# air velocity w = w₀ sin(2π z / 4 km), so that the total transport w + wˢ changes sign.
function falling_rain_column(FT; courant, w₀=0, closed=false, cfl=1//2, bounds=(0, 1), Nt=20, compressible=false, implicit=true)
    Oceananigans.defaults.FloatType = FT
    Nz = 80
    Lz = Nz * Δz_column
    grid = RectilinearGrid(default_arch, FT; size=Nz, z=(0, Lz), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=300)

    # A column cannot carry a resolved anelastic w, so the moving-air cases prescribe it.
    dynamics = w₀ != 0    ? PrescribedDynamics(reference_state) :
               compressible ? CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300) :
                              AnelasticDynamics(reference_state)

    time_discretization = implicit ? AdaptiveVerticallyImplicitDiscretization(FT; cfl) : ExplicitTimeDiscretization()
    advection = WENO(FT; order=5, bounds, time_discretization)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants,
                            microphysics=FallingRain(FT(fall_speed), closed),
                            scalar_advection=(; ρqʳ=advection))

    qʳ₀(z) = ifelse(1500 < z < 3500, FT(1e-3), zero(FT))
    if compressible
        set!(model; ρ=reference_state.density, θ=300, qᵛ=FT(5e-3), qʳ=qʳ₀)
    else
        set!(model; θ=300, qᵛ=FT(5e-3), qʳ=qʳ₀)
    end
    w₀ == 0 || set!(model; w = z -> w₀ * sinpi(2z / Lz))

    ρqʳ = model.microphysical_fields.ρqʳ
    column_mass() = sum(Float64, Array(interior(ρqʳ))) * Δz_column
    M₀ = column_mass()
    Δt = FT(courant * Δz_column / fall_speed)

    minimum_ρqʳ = Inf
    increase = 0.0
    previous = M₀
    for _ in 1:Nt
        time_step!(model, Δt)
        minimum_ρqʳ = min(minimum_ρqʳ, Float64(minimum(interior(ρqʳ))))
        M = column_mass()
        increase = max(increase, M - previous)
        previous = M
    end

    return (; minimum_ρqʳ, M₀, M=column_mass(), increase, ρqʳ=Array(interior(ρqʳ)))
end

# Bounds-preserving WENO holds its bounds only while the donor cell's outflow Courant number is
# at most 5/18; before the bounded split, AIVA's explicit part ran at Courant numbers up to its
# `cfl` of 1/2 and this column reached ρqʳ ≈ -8e-7 (anelastic and acoustic alike).
@testset "Bounded AIVA keeps falling rain non-negative [$(FT)]" for FT in all_float_types()
    ε(M) = FT === Float64 ? 1e-12 : 1e-7 * 1.2e-3 # absolute, relative to ρqʳ ≈ 1.2e-3 kg m⁻³
    closure_tolerance = FT === Float64 ? 1e-6 : 1e-5

    cases = [(; w₀=0, compressible=false), (; w₀=5, compressible=false), (; w₀=0, compressible=true)]

    for case in cases, courant in (0.5, 2, 5, 20, 50)
        # Open bottom: rain leaves through the ground; nothing becomes negative.
        open = falling_rain_column(FT; courant, case...)
        @test open.minimum_ρqʳ ≥ -ε(open.M₀)
        @test open.increase ≤ 1e-6 * open.M₀ # outflow only removes rain

        # Closed bottom: the column keeps all its rain.
        closed = falling_rain_column(FT; courant, closed=true, case...)
        @test closed.minimum_ρqʳ ≥ -ε(closed.M₀)
        @test abs(closed.M - closed.M₀) ≤ closure_tolerance * closed.M₀
    end
end

# Where the vertical Courant number is within both the scheme's `cfl` and the limiter's bound,
# the split is inert and AIVA reproduces the explicit scheme. (The Wicker–Skamarock stages of the
# acoustic stepper are not strong-stability preserving, so there the explicit scheme itself may
# dip below zero by roundoff-sized amounts; positivity is asserted for the SSP stepper only.)
@testset "Bounded AIVA is explicit below the bound-preserving Courant number [$(FT)]" for FT in all_float_types()
    for compressible in (false, true)
        aiva = falling_rain_column(FT; courant=1//5, compressible, Nt=10)
        explicit = falling_rain_column(FT; courant=1//5, compressible, Nt=10, implicit=false)
        if compressible
            # The non-SSP explicit run dips below zero by roundoff; the outflow limiter keeps AIVA
            # non-negative there, so the two agree to that undershoot rather than bit for bit.
            tolerance = (FT === Float64 ? 1e-8 : 1e-5) * maximum(abs, explicit.ρqʳ)
            @test isapprox(aiva.ρqʳ, explicit.ρqʳ; rtol = 0, atol = tolerance)
            @test aiva.minimum_ρqʳ ≥ -eps(FT) * maximum(abs, explicit.ρqʳ)
        else
            # Agrees to roundoff, not bit for bit
            @test isapprox(aiva.ρqʳ, explicit.ρqʳ; rtol = 0, atol = eps(FT) * maximum(abs, explicit.ρqʳ))
            @test aiva.minimum_ρqʳ ≥ 0
        end
    end
end

# The limited tendency reads the face fluxes `update_outflow_limiter!` stored for the stage; it must
# equal the divergence of the outflow-limited fluxes rebuilt from the reconstructions.
const BA = Breeze.Advection

@inline function rebuilt_limited_flux_x(i, j, k, grid, advection, ρ, u, c)
    r = advection.bounds.limiter.outflow_fraction
    @inbounds rᵘ = ifelse(u[i, j, k] ≥ 0, r[i-1, j, k], r[i, j, k])
    return rᵘ * BA.bounded_face_flux_x(i, j, k, grid, advection, ρ, u, c)
end

@inline function rebuilt_limited_flux_y(i, j, k, grid, advection, ρ, v, c)
    r = advection.bounds.limiter.outflow_fraction
    @inbounds rᵘ = ifelse(v[i, j, k] ≥ 0, r[i, j-1, k], r[i, j, k])
    return rᵘ * BA.bounded_face_flux_y(i, j, k, grid, advection, ρ, v, c)
end

@inline function rebuilt_limited_divergence(i, j, k, grid, advection, ρ, U, c)
    wᵉ = BA.ExplicitVerticalVelocity(grid, advection, U.w, ρ)
    return Oceananigans.Operators.V⁻¹ᶜᶜᶜ(i, j, k, grid) *
        (Oceananigans.Operators.δxᶜᵃᵃ(i, j, k, grid, rebuilt_limited_flux_x, advection, ρ, U.u, c) +
         Oceananigans.Operators.δyᵃᶜᵃ(i, j, k, grid, rebuilt_limited_flux_y, advection, ρ, U.v, c) +
         Oceananigans.Operators.δzᵃᵃᶜ(i, j, k, grid, BA.limited_face_flux_z, advection, ρ, wᵉ, c))
end

@testset "Outflow-limited tendency reads the stage's stored face fluxes [$(FT)]" for FT in all_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=(8, 8, 16), halo=(6, 6, 6), x=(0, 800), y=(0, 800), z=(0, 800))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=300)
    advection = WENO(FT; order=5, bounds=(0, 1), time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=1//2))
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), thermodynamic_constants=constants,
                            microphysics=FallingRain(FT(fall_speed), false), scalar_advection=(; ρqʳ=advection))
    blob(x, y, z) = ifelse((200 < x < 400) & (300 < y < 500) & (300 < z < 600), FT(1e-3), zero(FT))
    set!(model; θ=300, qᵛ=FT(5e-3), qʳ=blob, u=(x, y, z) -> 20 * sinpi(2y / 800), v=(x, y, z) -> 15 * cospi(2x / 800))
    for _ in 1:3
        time_step!(model, 20)
    end
    Oceananigans.TimeSteppers.update_state!(model)

    scheme = model.advection.ρqʳ
    r = scheme.bounds.limiter.outflow_fraction
    @test minimum(interior(r)) < 1 # the limiter acts somewhere

    ρ = AM.total_density(model.dynamics)
    U = BA.scalar_transport_velocities(model, :ρqʳ)
    c = model.microphysical_fields.qʳ
    stored = Field(KernelFunctionOperation{Center, Center, Center}(AM.div_ρUc, grid, scheme, ρ, U, c))
    rebuilt = Field(KernelFunctionOperation{Center, Center, Center}(rebuilt_limited_divergence, grid, scheme, ρ, U, c))
    compute!(stored)
    compute!(rebuilt)
    # Roundoff only: on GPUs the two may contract multiply-adds differently.
    stored, rebuilt = Array(interior(stored)), Array(interior(rebuilt))
    @test maximum(abs, rebuilt) > 0
    @test isapprox(stored, rebuilt; rtol=0, atol=8 * eps(FT) * maximum(abs, rebuilt))
end
