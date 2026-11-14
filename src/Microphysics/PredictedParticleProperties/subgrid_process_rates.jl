#####
##### The warm-phase P3 rates under subgrid statistics
#####
#
# `compute_p3_process_rates` hands the grid-mean Phase 1 rates and the subgrid statistics 𝒮 of the
# cell to `p3_subgrid_phase1_rates`. P3 condenses its own cloud liquid (see the `condensation`
# option), so 𝒮 is `DiracStatistics()` and the grid-mean rates are returned unchanged. The former
# `ClosureCloud` path, under which a turbulence closure set the cloud liquid, is retired.

@inline p3_subgrid_phase1_rates(::DiracStatistics, phase1, p3, ρ, ℳ, constants, state) = phase1

# The alignment of the cloud liquid to the prognostic vapor excess (`PrognosticVaporExcess`).
@inline subgrid_vapor_excess_adjustment(𝒮, p3, qᶜˡ, qᵛ⁺ˡ, δqᵛ⁺ˡ, T, ρᵗʰ, 𝒰, constants) =
    prognostic_vapor_excess_adjustment(p3, qᶜˡ, qᵛ⁺ˡ, δqᵛ⁺ˡ, T, ρᵗʰ, 𝒰, constants)
