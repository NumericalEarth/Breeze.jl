using Reactant: @trace
using Oceananigans: ReactantState
using Breeze.CompressibleEquations: CompressibleEquations, apply_horizontal_pressure_gradient_substep

# `@trace` collects the loop-carried variables syntactically and skips names in call position,
# so `substep!(true)` would leave the closure's captured fields out of the while-loop operands.
# Passing `substep!` as an argument makes it a reference the macro carries through the loop.
call_substep!(substep!, apply_pressure_gradient) = substep!(apply_pressure_gradient)

# Trace the acoustic substep loop as a single while loop instead of unrolling `Nτ` copies of
# its body into the program. The first substep is peeled off so that `apply_pressure_gradient`
# stays a compile-time constant: it is the only quantity that depends on the substep index,
# and it is `true` for every substep after the first.
function CompressibleEquations.acoustic_substep_loop!(substep!, ::ReactantState, Nτ, apply_first_substep_pressure_gradient)
    substep!(apply_horizontal_pressure_gradient_substep(1, Nτ, apply_first_substep_pressure_gradient))
    @trace track_numbers=false for _ in 2:Nτ
        call_substep!(substep!, true)
    end
    return nothing
end
