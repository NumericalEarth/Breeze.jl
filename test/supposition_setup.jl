# Shared configuration and generators for property-based tests (Supposition.jl).
# Test files that use `@breeze_check` include this after `setup.jl`.
# This is not a test file: it is removed from the suite in runtests.jl.

using Supposition
using Supposition: Data, assume!, event!
using Random: Xoshiro
using Dates: Dates

using Breeze.Thermodynamics: MoistureMassFractions

# Deterministic by default so a CI failure is reproducible from the log alone
# (Supposition does not print the seed). Override to fuzz locally, e.g.
#   BREEZE_CHECK_SEED=123 BREEZE_CHECK_MAX_EXAMPLES=20000
# An empty BREEZE_CHECK_SEED= selects a fresh random seed (`Xoshiro(nothing)`).
const BREEZE_CHECK_SEED = let seed = get(ENV, "BREEZE_CHECK_SEED", "20260922")
    isempty(seed) ? nothing : parse(UInt64, seed)
end
const BREEZE_CHECK_MAX_EXAMPLES = parse(Int, get(ENV, "BREEZE_CHECK_MAX_EXAMPLES", "1000"))

# Fresh RNG per property, so each property is independent of which ran before it.
# `Xoshiro(nothing)` seeds from system entropy, so with an empty seed each property
# gets its own random stream and a failure is reproducible only from the printed arguments.
spstn_rng() = Xoshiro(BREEZE_CHECK_SEED)

"""
    @breeze_check [option=value ...] function name(a = generator, ...) ... end

`Supposition.@check` with Breeze defaults `rng = spstn_rng()`,
`max_examples = BREEZE_CHECK_MAX_EXAMPLES`, `db = false`, `record = false`.
Explicit options win. `rng` has to be passed per check because
`SuppositionReport`'s `rng` keyword always overrides `CheckConfig.rng`.

The report is not recorded into the enclosing `@testset`: ParallelTestRunner
serializes test sets from its workers back to the main process, which does not
load Supposition and cannot deserialize a `SuppositionReport` (nor the property
closures it holds). Instead the check runs inside a plain `@testset` named after
the property and its outcome is asserted with `@test`. On a failure or error,
Supposition prints the shrunk counterexample, and `spstn_report_counterexample`
follows it with an exact `repr` of every argument, so that values whose `show`
rounds (such as `MoistureMassFractions`) can still be reproduced verbatim.
"""
macro breeze_check(exprs...)
    given = Set(e.args[1] for e in exprs if Meta.isexpr(e, :(=)) && e.args[1] isa Symbol)
    defaults = (:(rng = spstn_rng()), :(max_examples = BREEZE_CHECK_MAX_EXAMPLES), :(db = false), :(record = false))
    options = [d for d in defaults if !(d.args[1] in given)]
    name = spstn_property_name(exprs[end])
    return esc(quote
        Test.@testset $name begin
            local report = Supposition.@check $(options...) $(exprs...)
            spstn_report_counterexample(report)
            Test.@test spstn_passed(report)
        end
    end)
end

# Name of the property being checked, for the enclosing `@testset`
function spstn_property_name(expr)
    if Meta.isexpr(expr, :function) && Meta.isexpr(expr.args[1], :call)
        return string(expr.args[1].args[1])
    elseif Meta.isexpr(expr, :call)
        return string(expr.args[1])
    else
        return "property"
    end
end

# `Supposition.Pass` is the only passing outcome; `Fail` (a counterexample) and `Error`
# (an exception) have already been printed by Supposition when the check finished.
spstn_passed(report::Supposition.SuppositionReport) =
    !isnothing(report.result) && something(report.result) isa Supposition.Pass

# Exact, round-trippable text for a counterexample argument: `repr` for numbers and strings
# (the shortest round-trip form for floats), and the default constructor-style `show` for
# structs, since a pretty `show` method may round its values.
spstn_exact_repr(x::Union{Number, AbstractString, AbstractChar, Symbol, Dates.AbstractDateTime}) = repr(x)
spstn_exact_repr(x) = isstructtype(typeof(x)) ? sprint(Base.show_default, x) : repr(x)

function spstn_report_counterexample(report::Supposition.SuppositionReport)
    spstn_passed(report) && return nothing
    result = something(report.result)
    result isa Union{Supposition.Fail, Supposition.Error} || return nothing
    println(stderr, "  Exact arguments of the counterexample to `", report.description, "`:")
    for (name, value) in pairs(result.example)
        println(stderr, "      ", name, " = ", spstn_exact_repr(value))
    end
    return nothing
end

# Relative tolerance for closed-form identities, scaled by machine epsilon so one
# property serves Float32 and Float64.
spstn_rounding_rtol(FT, n=100) = n * eps(FT)

#####
##### Generators for physical ranges
#####

# Finite floats of type FT in [lo, hi]; finite bounds also disable Inf, NaN is off.
spstn_floats(FT; lo, hi) = Data.Floats{FT}(; minimum=FT(lo), maximum=FT(hi), nans=false, infs=false)

spstn_temperatures(FT; lo=200, hi=330) = spstn_floats(FT; lo, hi)   # K
spstn_pressures(FT; lo=3e4, hi=1.05e5) = spstn_floats(FT; lo, hi)  # Pa
spstn_heights(FT; lo=0, hi=2e4) = spstn_floats(FT; lo, hi)         # m
spstn_unit_interval(FT) = spstn_floats(FT; lo=0, hi=1)

# MoistureMassFractions with qᵛ + qˡ + qⁱ = qᵗ ≤ total_max by construction (no filtering):
# draw the total, the condensed fraction of it, and the ice fraction of the condensate.
function spstn_mass_fractions(FT; total_max=3e-2, condensate_fraction_max=0.5)
    qᵗ = spstn_floats(FT; lo=0, hi=total_max)
    fᶜ = spstn_floats(FT; lo=0, hi=condensate_fraction_max)
    fⁱ = spstn_unit_interval(FT)
    return map(qᵗ, fᶜ, fⁱ) do qᵗ, fᶜ, fⁱ
        qᶜ = fᶜ * qᵗ
        qⁱ = fⁱ * qᶜ
        return MoistureMassFractions(qᵗ - qᶜ, qᶜ - qⁱ, qⁱ)
    end
end

@info "Property-based tests" seed = BREEZE_CHECK_SEED max_examples = BREEZE_CHECK_MAX_EXAMPLES
