#####
##### Switches for the P3 GPU optimizations (ported from Breeze PR #1092)
#####
##### Each switch is a zero-argument `@inline` function returning a literal `Bool`, so a kernel
##### that branches on it compiles only the chosen path. The values are read from the
##### environment at load (`__init__`), not frozen at precompile time, and
##### `set_p3_optimization_switches!` changes them in-process: it redefines the function, which
##### invalidates the code that inlined it, so the next launch compiles the other path.
#####
#####   BREEZE_P3_LINEAR_BRACKETS      (default true)  shared table brackets as a linear corner
#####                                                  offset plus per-axis steps
#####   BREEZE_P3_SPECIES_FALL_SPEEDS  (default true)  per-species fall-speed diagnosis
#####   BREEZE_P3_PROCESS_GUARDS       (default true)  process-group guards in the rates
#####   BREEZE_P3_CLEAR_AIR_SKIP       (default false, tier B) skip the rates in clear air
#####

@inline p3_linear_brackets() = true
@inline p3_species_fall_speeds() = true
@inline p3_process_guards() = true
@inline p3_clear_air_skip() = false

const P3_OPTIMIZATION_SWITCHES = (linear_brackets     = (:p3_linear_brackets,     "BREEZE_P3_LINEAR_BRACKETS",     true),
                                  species_fall_speeds = (:p3_species_fall_speeds, "BREEZE_P3_SPECIES_FALL_SPEEDS", true),
                                  process_guards      = (:p3_process_guards,      "BREEZE_P3_PROCESS_GUARDS",      true),
                                  clear_air_skip      = (:p3_clear_air_skip,      "BREEZE_P3_CLEAR_AIR_SKIP",      false))

"""
    p3_optimization_switches()

The current values of the P3 optimization switches, as a `NamedTuple`.
"""
p3_optimization_switches() = map(s -> Bool(getfield(@__MODULE__, s[1])()), P3_OPTIMIZATION_SWITCHES)

"""
    set_p3_optimization_switches!(; linear_brackets, species_fall_speeds, process_guards, clear_air_skip)

Set any of the P3 optimization switches in-process. Only switches whose value changes are
redefined, so unchanged kernels are not recompiled.
"""
function set_p3_optimization_switches!(; kw...)
    # The caller may run in an older world age than the last redefinition.
    current = Base.invokelatest(p3_optimization_switches)
    for (name, value) in pairs(kw)
        haskey(P3_OPTIMIZATION_SWITCHES, name) || throw(ArgumentError("unknown P3 switch $name"))
        value = Bool(value)
        if current[name] != value
            f = P3_OPTIMIZATION_SWITCHES[name][1]
            Core.eval(@__MODULE__, :(@inline $f() = $value))
        end
    end
    return Base.invokelatest(p3_optimization_switches)
end

p3_env_switch(name, default) = lowercase(get(ENV, name, string(default))) in ("1", "true")

function __init__()
    # Redefining methods while another package precompiles would break its cache.
    ccall(:jl_generating_output, Cint, ()) == 1 && return nothing
    values = map(s -> p3_env_switch(s[2], s[3]), P3_OPTIMIZATION_SWITCHES)
    set_p3_optimization_switches!(; values...)
    return nothing
end

# A process group guarded by `condition` runs when the guards are off or the condition holds.
@inline run_process_group(condition) = !p3_process_guards() | condition
