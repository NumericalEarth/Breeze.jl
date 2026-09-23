# Shared configuration and generators for property-based tests (Supposition.jl).
# Test files that use `@breeze_check` include this after `setup.jl`.
# This is not a test file: it is removed from the suite in runtests.jl.

using Supposition
using Supposition: Data, assume!, event!
using Random: Xoshiro

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
closures it holds). The outcome is asserted with a plain `@test` instead;
Supposition still prints a failing or erroring report, with the shrunk
counterexample, when the check finishes.
"""
macro breeze_check(exprs...)
    given = Set(e.args[1] for e in exprs if Meta.isexpr(e, :(=)) && e.args[1] isa Symbol)
    defaults = (:(rng = spstn_rng()), :(max_examples = BREEZE_CHECK_MAX_EXAMPLES), :(db = false), :(record = false))
    options = [d for d in defaults if !(d.args[1] in given)]
    return esc(quote
        local report = Supposition.@check $(options...) $(exprs...)
        Test.@test spstn_passed(report)
    end)
end

# `Supposition.Pass` is the only passing outcome; `Fail` (a counterexample) and `Error`
# (an exception) have already been printed by Supposition when the check finished.
spstn_passed(report::Supposition.SuppositionReport) =
    !isnothing(report.result) && something(report.result) isa Supposition.Pass

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
