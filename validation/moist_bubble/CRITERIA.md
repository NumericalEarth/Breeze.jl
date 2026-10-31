# Pre-registered pass/fail criteria: moist Bryan & Fritsch (2002) bubble

Committed before any CM1 output was analyzed and before any Breeze bubble was run on this branch.
The only prior numbers known to the author are those in the earlier 2D RESULTS.md (the paper's θₑ′ range
[−0.306, 4.095] K at 100 m, and the contaminated Breeze runs). If a criterion turns out to be ill-posed it is
reported as such and not silently changed. Any later change is listed under "Amendments".

Case: 20 km × 10 km, closed free-slip box, rᵗ = 0.020, θₑ = 320 K at 1000 hPa, saturated neutral environment, BF02
bubble (θρ′/θ̄ρ = Δθ/300, Δθ = 2 cos²(πβ/2) K), no rain, no sedimentation, no explicit diffusion, WENO5, t = 1000 s.
All Breeze runs use BF02/CM1 constants (Rᵈ 287.04, Rᵛ 461.5, cᵖᵈ 1005.7, cᵖᵛ 1870, cˡ 4190, ℒ(273.15 K) 2501000, g 9.81).
θₑ is the reversible (wet) equivalent potential temperature with the same constants, using each model's own eₛ(T).
θₑ′ is relative to the model's undisturbed column at t = 0.

Reference: CM1 r21.1 eqtset = 2, same Δ, Δt = Δ/100. "Primary" grid: Δ = 100 m.

## Gate A: initial state (a formulation that fails this has an invalid test, not a physics result)

| id | quantity | tolerance |
|---|---|---|
| A1 | environment θₑ, independent of height, from the model's state after `set!` | max−min ≤ 0.01 K; mean within 0.01 K of 320 K |
| A2 | max\|T − T_intended\| after `set!` | ≤ 1e-3 K |
| A3 | max\|qˡ − qˡ_intended\| after `set!` | ≤ 1e-6 |
| A4 | compressible: max\|p − p_intended\|/p after `set!` | ≤ 1e-5 |
| A5 | environmental T(z) and qᵛ(z) against CM1 (different eₛ formula) | \|ΔT\| ≤ 0.5 K, \|Δqᵛ\| ≤ 3e-4 |
| A6 | null run (no bubble), 1000 s: max over time of max\|w\| | ≤ 0.05 m/s |
| A7 | null run, 1000 s: max\|θₑ′\| | ≤ 0.05 K |

## Gate B: bubble, Δ = 100 m, t = 1000 s, against CM1 at 100 m

| id | quantity | tolerance |
|---|---|---|
| B1 | max w at 1000 s | within 10 % of CM1 |
| B2 | θₑ′max at 1000 s | within 15 % of CM1 |
| B3 | θₑ′min at 1000 s | ≥ −0.5 K and within 0.25 K of CM1 |
| B4 | bubble top: highest z where a column has θₑ′ ≥ 1 K (the environment is saturated at every level, so a "cloud top" as qˡ edge does not exist) | within 200 m of CM1 |
| B5 | structure: relative L2 of θₑ′ against CM1 on the same grid | ≤ 0.25 |
| B6 | structure: relative L2 of w against CM1 on the same grid | ≤ 0.25 |
| B7 | θₑ conservation: \|Δ∫ρᵈθₑ′ dV\| / ∫ρᵈθₑ′(0) dV | ≤ 5 % |
| B8 | left-right symmetry: max\|θₑ′(x) − θₑ′(−x)\| | ≤ 0.1 K (a blown-up or noisy solution fails this) |

## Gate C: energy budget, Δ = 100 m

Total energy ℰ = ∫ρ(𝓊 + K + gz) dV on the compressible core, ∫ρᵣ(𝒽 + K + gz) dV on the anelastic core, and total water.

| id | quantity | tolerance |
|---|---|---|
| C1 | E: \|Δℰ\| / max KE | ≤ 1e-6 (structural) |
| C2 | θ, η: \|Δℰ\| / max KE | ≤ 5 % |
| C3 | anelastic θ: \|Δℰ\| / max KE | ≤ 5 % |
| C4 | \|Δ∫ρqᵗ\| / ∫ρqᵗ | ≤ 1e-10 |

## Verdict

A formulation passes if it passes A1–A7, B1–B8 and its C rows. η cannot use saturation adjustment (the model constructor
rejects it), so η is judged with non-equilibrium cloud formation with sedimentation off, and this is stated as such.
θ, E and anelastic θ are judged with saturation adjustment and also reported with the same non-equilibrium scheme.
A FAIL with a non-equilibrium lag diagnosed as the cause is reported with convergence in τ, not re-graded.

## Convergence (reported, not graded)

Δ = 250, 200, 100, and 50 m if cost allows: w max, θₑ′ range and bubble top from Breeze and CM1, and the relative L2 between
Breeze and CM1 at each Δ.

## Amendments

None yet.
