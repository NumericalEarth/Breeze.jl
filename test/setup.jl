import Breeze
import CUDA
import Oceananigans
using GPUArraysCore: @allowscalar
using Oceananigans.Architectures: CPU, GPU
using Test: @inferred, @test

if get(ENV, "BREEZE_ENSURE_CUDA_FUNCTIONAL", "") == "true"
    CUDA.functional() || error("CUDA is not functional but we expect it to be, make sure it's set up correctly")
end

const default_arch = CUDA.functional() ? GPU() : CPU()

# Float type helpers for tests
# Default: Float64 only. Set BREEZE_TEST_FLOAT32=true to also test Float32.
function test_float_types()
    if get(ENV, "BREEZE_TEST_FLOAT32", "false") == "true"
        return (Float32, Float64)
    else
        return (Float64,)
    end
end

# Returns both Float32 and Float64 for tests that need both precision levels
all_float_types() = (Float32, Float64)

# Work around for <https://github.com/JuliaLang/julia/issues/54998>, often seen
# with Reactant code (and often only on the CI machine).
macro with_stack_size(stack_size, expr)
    return quote
        local _size = $(esc(stack_size))
        _size isa Integer || error("Stack size must be an integer")
        local task = Task(() -> $(esc(expr)), _size)
        schedule(task)
        wait(task)
        fetch(task)
    end
end

macro with_stack_size(expr)
    return :(@with_stack_size 16 << 20 $(esc(expr)))
end

# Content per unit falling condensate of `phase` (`:liquid` or `:ice`) for the thermodynamic
# variable of `formulation` (`:LiquidIcePotentialTemperature` or `:StaticEnergy`): the partial
# derivative of the specific variable with respect to that condensate mass fraction at fixed
# temperature `T` and pressure `p`, along the composition increment of the core, so that losing
# condensate at this content leaves the temperature unchanged: with `renormalize=false` the
# total density is fixed (anelastic) and the dry mass fraction makes up the departed mass; with
# `renormalize=true` the total density falls with the condensate (compressible) and every mass
# fraction renormalizes. A Float64 central difference of Breeze's own state functions, independent of the
# closed forms the tendencies use. The geopotential does not depend on the composition, so the
# height is immaterial and set to zero.
function condensate_content(formulation, phase, T, q, p, pˢᵗ; renormalize=false)
    Thermodynamics = Breeze.Thermodynamics
    constants = Thermodynamics.ThermodynamicConstants(Float64)
    δ = 1e-6
    q₀ = (Float64(q.vapor), Float64(q.liquid), Float64(q.ice))
    eˣ = phase === :liquid ? (0.0, 1.0, 0.0) : (0.0, 0.0, 1.0)
    function perturbed(ε)
        qε = q₀ .+ ε .* eˣ
        renormalize && (qε = qε ./ (1 + ε))
        return Thermodynamics.MoistureMassFractions(qε...)
    end
    function φ(qε)
        if formulation === :LiquidIcePotentialTemperature
            𝒰 = Thermodynamics.LiquidIcePotentialTemperatureState(0.0, qε, Float64(pˢᵗ), Float64(p))
            return Thermodynamics.with_temperature(𝒰, Float64(T), constants).potential_temperature
        else
            𝒰 = Thermodynamics.StaticEnergyState(0.0, qε, 0.0, Float64(p))
            return Thermodynamics.with_temperature(𝒰, Float64(T), constants).static_energy
        end
    end
    return (φ(perturbed(δ)) - φ(perturbed(-δ))) / 2δ
end

# Change of the specific variable of `formulation` per unit heating at fixed composition and
# pressure, ∂φ/∂h: a Float64 central difference in temperature of Breeze's own state functions,
# divided by the mixture heat capacity. One for static energy, 1 / (cᵖᵐ Π) for potential
# temperature. With fixed_volume=true, vary p proportionally to T at fixed gas partial
# densities and divide by cᵛᵐ instead. The height is immaterial and set to zero.
function heating_response(formulation, T, q, p, pˢᵗ; fixed_volume=false)
    Thermodynamics = Breeze.Thermodynamics
    constants = Thermodynamics.ThermodynamicConstants(Float64)
    δ = 1e-3
    q₀ = Thermodynamics.MoistureMassFractions(Float64(q.vapor), Float64(q.liquid), Float64(q.ice))
    function φ(Tε)
        if formulation === :LiquidIcePotentialTemperature
            pε = fixed_volume ? Float64(p) * Tε / Float64(T) : Float64(p)
            𝒰 = Thermodynamics.LiquidIcePotentialTemperatureState(0.0, q₀, Float64(pˢᵗ), pε)
            return Thermodynamics.with_temperature(𝒰, Tε, constants).potential_temperature
        else
            𝒰 = Thermodynamics.StaticEnergyState(0.0, q₀, 0.0, Float64(p))
            return Thermodynamics.with_temperature(𝒰, Tε, constants).static_energy
        end
    end
    cᵖᵐ = Thermodynamics.mixture_heat_capacity(q₀, constants)
    capacity = fixed_volume ? cᵖᵐ - Thermodynamics.mixture_gas_constant(q₀, constants) : cᵖᵐ
    return (φ(Float64(T) + δ) - φ(Float64(T) - δ)) / (2δ * capacity)
end

# Expected sedimentation tendency of the coupling-weighted thermodynamic prognostic in a column
# where every flux is downward (no transport velocity): `Φ[k]` is the sedimentation volume flux
# per unit density of one phase through face k, `ρᶠ[k]` the total density there, and `χ`, `h`
# and `β` the content, enthalpy and heating response of that phase per cell. The flux through
# the upper face of cell k drains cell k + 1 (clamped at the top, where the flux vanishes) and
# delivers χ[k] + β[k] (h[k+1] − h[k]); the flux through its lower face drains the cell itself
# and delivers χ[k]. `coupling` is the cell's coupling-to-total density ratio.
function expected_sedimentation_tendency(Nz, Δz, ρᶠ, Φ, χ, h, β; coupling = ones(Nz))
    above(k) = min(k, Nz)
    G = zeros(Nz)
    for k in 1:Nz
        delivered⁺ = χ[k] + β[k] * (h[above(k + 1)] - h[k])
        G[k] = -coupling[k] * (ρᶠ[k+1] * Φ[k+1] * delivered⁺ - ρᶠ[k] * Φ[k] * χ[k]) / Δz
    end
    return G
end

# Check that the pointwise functions called inside the `AtmosphereModel` kernels (auxiliary
# thermodynamic variables, microphysical tendencies, momentum and moisture tendencies) are
# inferred at cell `(i, j, k)`. This mirrors the argument assembly in
# `compute_auxiliary_variables!` and `compute_tendencies!`: keep them in sync.
function test_kernel_functions_inferred(model; i=2, j=2, k=2)
    AM = Breeze.AtmosphereModels
    FT = eltype(model.grid)
    grid = model.grid
    dynamics = model.dynamics
    microphysics = model.microphysics
    constants = model.thermodynamic_constants
    μ_fields = model.microphysical_fields
    qᵛᵉ_field = AM.specific_prognostic_moisture(model)
    velocities = AM.transport_velocities(model)
    model_fields = Oceananigans.fields(model)
    moist_name = AM.moisture_prognostic_name(microphysics)

    momentum_args = (AM.dynamics_density(dynamics), model.advection.momentum, model.velocities,
                     model.closure, model.closure_fields, AM.advecting_momentum(model),
                     model.coriolis, model.clock, model_fields)

    w_args = (momentum_args..., model.forcing.ρw, dynamics, model.formulation, model.temperature,
              qᵛᵉ_field, microphysics, μ_fields, constants)

    common_args = (dynamics, model.formulation, constants, qᵛᵉ_field, velocities, microphysics,
                   μ_fields, model.closure, model.closure_fields, model.clock, model_fields)

    ρq_args = (qᵛᵉ_field, Val(2), Val(moist_name), model.forcing[moist_name],
               model.advection[moist_name], common_args...)

    @allowscalar begin
        ρ = AM.total_density(dynamics)[i, j, k]
        qᵛᵉ = qᵛᵉ_field[i, j, k]
        q = @inferred AM.grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, μ_fields)
        𝒰₀ = @inferred AM.diagnose_thermodynamic_state(i, j, k, grid, model.formulation, dynamics, q)
        𝒰₁ = @inferred AM.maybe_adjust_thermodynamic_state(𝒰₀, microphysics, qᵛᵉ, constants)
        @test @inferred(Breeze.Thermodynamics.temperature(𝒰₁, constants)) isa FT

        ℳ = @inferred AM.grid_microphysical_state(i, j, k, grid, microphysics, μ_fields, ρ, 𝒰₁, velocities)
        for name in (moist_name, AM.prognostic_field_names(microphysics)...)
            @test @inferred(AM.microphysical_tendency(microphysics, Val(name), ρ, ℳ, 𝒰₁, constants)) isa FT
        end

        @test @inferred(AM.x_momentum_tendency(i, j, k, grid, momentum_args..., model.forcing.ρu, dynamics)) isa FT
        @test @inferred(AM.y_momentum_tendency(i, j, k, grid, momentum_args..., model.forcing.ρv, dynamics)) isa FT
        @test @inferred(AM.z_momentum_tendency(i, j, k, grid, w_args...)) isa FT
        @test @inferred(AM.scalar_tendency(i, j, k, grid, ρq_args...)) isa FT
    end

    return nothing
end
