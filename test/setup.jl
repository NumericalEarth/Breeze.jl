import CUDA
using Oceananigans.Architectures: CPU, GPU

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
# temperature `T` and pressure `p`, with `replacement` taking up the departed mass, so that
# sedimentation alone leaves the temperature unchanged: `:dry_air` on a core whose total
# density is fixed (anelastic), so the dry mass fraction absorbs the change, or `:mixture` on a
# core whose total density falls with the condensate (compressible), so every mass fraction
# renormalizes. A Float64 central difference of Breeze's own state functions, independent of the
# closed forms the tendencies use. The geopotential does not depend on the composition, so the
# height is immaterial and set to zero.
function condensate_content(formulation, phase, T, q, p, pˢᵗ; replacement=:dry_air)
    Thermodynamics = Breeze.Thermodynamics
    constants = Thermodynamics.ThermodynamicConstants(Float64)
    δ = 1e-6
    q₀ = (Float64(q.vapor), Float64(q.liquid), Float64(q.ice))
    eˣ = phase === :liquid ? (0.0, 1.0, 0.0) : (0.0, 0.0, 1.0)
    function perturbed(ε)
        qε = q₀ .+ ε .* eˣ
        replacement === :mixture && (qε = qε ./ (1 + ε))
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
