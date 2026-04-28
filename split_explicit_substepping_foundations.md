# Split-Explicit Substepping for the Compressible Euler Equations: Theoretical Foundations

*A working theoretical foundation for clean substepping implementations in atmospheric models, organized around modern multirate-integrator literature and informed by prior implementations (WRF, MPAS-A, ERF, COSMO, CM1, Breeze.jl).*

---

## STATUS

**This is a planning outline / working document, not a finished text.** It exists to organize what needs to be written, what content already exists in `_draft_first_pass.md` (the previous "Baldauf rewrite" draft), and which references each section needs before it can be made rigorous.

Sections are marked:
- ✅ — content drafted, needs migration from `_draft_first_pass.md` and likely revision
- 🟡 — partial content drafted, gaps flagged
- ❌ — not yet written; awaiting references
- 📚 — section blocked on tier-1/2 papers (see "References needed" below)

---

## PURPOSE AND SCOPE

This document aims to be the theoretical reference for split-explicit acoustic substepping in non-hydrostatic compressible atmospheric models. Its purpose is to:

1. **Lay out the mathematical framework** behind split-explicit time integration of the compressible Euler equations: slow/fast partitioning, linearization choices, substep discretization, and outer-integrator design.
2. **Place existing implementations within this framework**: how WRF, MPAS-A, ERF, COSMO, CM1, and Breeze.jl differ in their concrete choices, and what those differences imply for stability and accuracy.
3. **Connect to modern multirate-integrator theory**: the Wensch-Knoth-Galant MIS framework, the Knoth-Schlegel-Wensch atmospheric MIS methods, and the Sandu-Günther MRI-GARK unification.
4. **Recommend a specific design** for a clean Breeze.jl substepper that is rigorous, modular, and forward-compatible.

What it is *not*:
- A drop-in textbook chapter (it's specific to compressible atmospheric flow and to the implementations that exist now).
- A survey of all time-integration methods (no leapfrog, no semi-implicit IFS-style schemes, no SLAV semi-Lagrangian).
- An anelastic or Boussinesq treatment (Breeze has separate `AnelasticDynamics`; that's a different system).

Audience: Breeze developers, dynamical-core implementers familiar with split-explicit ideas, numerical-analysis-comfortable atmospheric modelers.

---

## NOTATION CONVENTIONS

Following Breeze.jl's [appendix/notation](https://numericalearth.github.io/BreezeDocumentation/dev/) and the substepper code conventions in PR #622.

- Conservative prognostics: $\rho$, $\rho\boldsymbol u = (\rho u, \rho v, \rho w)$, $\chi = \rho\theta$
- Hydrostatic reference (time-independent): $\rho_0(z), p_0(z), \theta_0(z), \pi_0(z)$
- MPAS-style perturbations from reference: $\rho''$, $(\rho u)''$, $(\rho v)''$, $(\rho w)''$, $(\rho\theta)''$
- Reference deviation pressure: $p' = p - p_0$ (single prime — distinguishes from substep perturbations)
- Time scales: $\Delta t$ (outer RK), $\Delta\tau$ (substep), $N_s$ (substeps per outer step), $N_\tau$ (substeps per stage)
- Sound speed: $\mathbb C^{ac} = \sqrt{\gamma^d R^d \pi_0 \theta_0}$
- Off-centering: $\beta_S$ (sound), $\beta_B$ (buoyancy), $\beta_d$ (damping)

Full table in Appendix A.

---

## OUTLINE

### Part I — Setting up the problem

**§1. Introduction and motivation.** ✅ (existing draft material reusable)

What split-explicit is, why we need it (multi-scale wave content), how it sits between fully-explicit (acoustic-CFL bound) and semi-implicit (Helmholtz solve every step). Brief survey of the families: explicit subcycling, semi-implicit, IMEX-RK / GARK, multirate-RK. Where this document lives in that landscape.

Material to migrate: §1 of `_draft_first_pass.md`.

---

**§2. Governing equations.** ✅ (existing draft material reusable)

The compressible Euler equations in conservative flux form following Breeze.jl's [governing equations](https://numericalearth.github.io/BreezeDocumentation/dev/) docs. Generic thermodynamic prognostic $\chi$; specialize to $\chi = \rho\theta$ for the dry case relevant to linear stability analysis. Equation of state $p = p^{st}(R^d\rho\theta/p^{st})^{\gamma^d}$.

Material to migrate: §2.1 of `_draft_first_pass.md`. Will need to expand with:

- Why $\chi = \rho\theta$ vs $\chi = \rho e$ (static energy) — Breeze supports both; the substep theory is largely independent but has implications for the linearized PGF form.
- Brief treatment of moisture and the moist EoS — flag the open question of moist acoustic PGF (PR #622's notes flag this as unresolved).

---

**§3. The hydrostatic reference state.** 🟡 (mostly drafted; needs more rigor)

Definition of $(\rho_0, p_0, \theta_0, \pi_0)$. Derived quantities $\mathbb C^{ac}, N^2, \omega_a$. Discussion of two reference-state choices used in practice:

- *Time-independent dry-adiabatic reference* (Breeze's `ExnerReferenceState`): $\theta_0 = $ constant, $\pi_0(z)$ from hydrostatic balance.
- *Time-independent stratified reference*: $\theta_0(z)$ chosen to match a target stratification (e.g., dry adiabatic + mean profile from analysis).

The choice matters because $\theta_0(z)$'s vertical structure feeds into the substep linearization (see §5).

Material to migrate: §2.2 of `_draft_first_pass.md`.

---

### Part II — Slow/fast partitioning and linearization

**§4. The slow/fast split.** 🟡 (drafted; needs rigorous justification)

The decomposition $\partial_t \mathbf U = \mathcal P_A \mathbf U + (\mathcal P_S + \mathcal P_B + \mathcal P_D)\mathbf U$ where:

- $\mathcal P_A$: advection + Coriolis + closure + microphysics (slow, evaluated once per RK stage)
- $\mathcal P_S$: sound (fast, every substep)
- $\mathcal P_B$: buoyancy (fast, every substep)
- $\mathcal P_D$: divergence damping (fast, every substep)

Why this split and not others — Skamarock & Klemp (1992) and Klemp-Skamarock-Dudhia (2007) lay this out. Need to cite their actual derivations, not paraphrase from Breeze docs.

📚 **Blocked on:** Skamarock-Klemp 1992; Klemp-Skamarock-Dudhia 2007.

Material to migrate: §2.5 of `_draft_first_pass.md` (will need rewriting once primary sources available).

---

**§5. Linearization of the fast operator.** ✅ (well-developed in current draft, keep)

The central conceptual chapter. Why we linearize at all, what state to linearize around, and where the linearization matters most.

5.1 Why linearize: stability analysis applicability + tridiagonal-coefficient efficiency.

5.2 The four candidate linearization states: time-independent reference, outer-step-frozen, stage-frozen, substep-evolving (= nonlinear). Trade-off table.

5.3 The key linearization is the thermodynamic flux: $\rho\theta\,\boldsymbol u \to \theta_0\,\boldsymbol M''$. Form 1 vs Form 2 derivations. Why this is the only equation where the reference state's vertical structure enters as a non-trivial multiplicative carrier.

5.4 The PGF linearization: $p' \approx \gamma^d R^d \pi_0\,(\rho\theta)''$. Where $\pi_0$ enters as a coefficient.

5.5 Buoyancy linearization: $-g\rho''$ after subtracting hydrostatic $\partial_z p_0$.

5.6 Mass equation: trivial (no reference appears).

5.7 Empirical evidence from Breeze PR #622: the $N_s$-consistency test as proof that linearization at the time-independent reference is the correct choice. The 60% → 0.1% spread fix.

Material to migrate: §2.3, §2.4, §A1 of `_draft_first_pass.md`. The σ/η confusion in earlier drafts is fixed; uses MPAS-style $\rho''$, etc.

---

### Part III — The discrete substep scheme

**§6. The HEVI Crank-Nicolson substep scheme.** ✅ (well-developed; needs minor cleanup)

The substep integrator: horizontally explicit (forward-Euler) + vertically implicit (off-centered Crank-Nicolson) for the linearized acoustic + buoyancy + damping system around the time-independent reference.

6.1 Three-phase substep structure (forward, predictor, vertical implicit).

6.2 Discrete forms of the substep equations. The MPAS-style perturbation variables; the tridiagonal coefficient names `cofwz, cofwr, cofwt, coftz`.

6.3 Off-centering: $\beta_S$ for sound, $\beta_B$ for buoyancy. Pure Crank-Nicolson at $\beta = \tfrac12$; off-centered for damping at $\beta > \tfrac12$. Why $\beta < \tfrac12$ is unconditionally unstable.

6.4 Why HEVI: the vertical CFL constraint $\Delta z \ll \Delta x$ at the surface forces vertical implicitness; horizontal explicit keeps the solve tridiagonal in $z$ rather than 3D Helmholtz.

6.5 Boundary semantics: how impenetrability and Bounded-topology halos enter the substep.

Material to migrate: §5 of `_draft_first_pass.md` (with the CN-not-forward-backward correction already applied).

---

**§7. Linear stability analysis methodology.** 🟡 (analysis sketches present; need rigorous derivations from primary sources)

Von Neumann analysis of the substep amplification matrix. Wavenumber-discretization conventions ($K_x, K_z, S_x, S_z$). Eigenvalue extraction; spectral radius vs spectral norm.

7.1 Sound substep alone: stability in $(C_S, \beta_S)$ plane. Trapezoidal $\beta_S = \tfrac12$ is neutral; $\beta_S > \tfrac12$ damping.

7.2 Buoyancy substep alone: oscillator equation; stability in $(C_B, \beta_B)$ plane.

7.3 Combined sound + buoyancy: Baldauf 2010's parameter scans. Why operator splitting between sound and buoyancy is bad; why addition-of-tendencies works.

7.4 Adding divergence damping: stability margin recovery.

📚 **Blocked on:** Skamarock-Klemp 1992 (the original framework); Klemp-Skamarock-Dudhia 2007 (the conservative-form version Baldauf builds on).

Material to migrate: §7, Appendix A2, A3 of `_draft_first_pass.md`. Stability proofs are partially worked out but should be re-verified against Skamarock-Klemp 1992 and Baldauf 2010 directly.

---

**§8. Damping strategies.** 🟡 (forms documented; derivations need primary sources)

Three damping strategies present in the literature and in Breeze:

8.1 Klemp-Skamarock 1992 / Klemp-Skamarock-Dudhia 2007 divergence damping: add $a_d \nabla(\nabla\!\cdot\!\boldsymbol u)$ to momentum equations. Damps the 2Δτ acoustic mode without affecting gravity-wave dispersion at leading order.

8.2 Klemp-Skamarock-Ha 2018 thermodynamic divergence damping: corrected form using $(\rho\theta)''$ tendency as a divergence proxy; preserves gravity-wave frequencies more cleanly. Effective horizontal diffusivity $\nu \approx 2\beta_d \Delta x^2/\Delta\tau$.

8.3 ERF/CM1 forward-extrapolation pressure-projection damping (Klemp-Skamarock-Dudhia 2007): $\widetilde{(\rho\theta)''} = (\rho\theta)''_\text{new} + \beta_d[(\rho\theta)''_\text{new} - (\rho\theta)''_\text{old}]$. Dimensionless; canonical $\beta_d = 0.1$ for synoptic, $\beta_d = 0.5$ for BCI-scale (Breeze default).

📚 **Blocked on:** Klemp-Skamarock-Dudhia 2007 (PPD derivation); Klemp-Skamarock-Ha 2018 (TDD derivation).

Empirical data from PR #622 NOTES.md: the two $\beta_d = 0.1$ choices give very different effective damping on stiff-IC LES problems (PPD too weak, TDD adequate). Worth flagging that the canonical coefficients are not on the same scale.

Material to migrate: §5.4 of `_draft_first_pass.md`.

---

### Part IV — The outer integrator landscape

**§9. The Wicker-Skamarock RK3 family.** ✅ (well-drafted)

9.1 Klemp-Wilhelmson 1978 leapfrog origin (briefly).

9.2 Wicker-Skamarock 1998 RK2 split-explicit.

9.3 Wicker-Skamarock 2002 RK3 (the WS-RK3 of Baldauf): stage fractions $1/3, 1/2, 1$, each stage resets to $\mathbf U^n$. The geometric-series substep accumulator.

9.4 The truncation-error expansion (Baldauf Eq. 14): why second-order in nonlinear case, third-order in linear; the $1/N_s$ splitting error.

9.5 SSP-RK3 (Shu-Osher) and TVD-RK3: monotonicity-preserving but $N_s$-degrading stability.

9.6 RK4MS (Baldauf 2008): four-stage second-order with 4th-order centered advection; 45% larger advective CFL.

9.7 The "Kyrill warning": why TVD-RK3 caused operational problems.

Material to migrate: §4, §8 of `_draft_first_pass.md`. The Baldauf truncation-error expansion (his Eq. 14) is well-established; should re-verify against the actual paper that I have.

---

**§10. The MIS framework (Wensch-Knoth-Galant 2009).** 📚 (currently sketched from ClimateMachine.jl docs; needs primary-source rewrite)

The general multirate-infinitesimal-step framework. Three lower-triangular coefficient matrices $(\alpha, \beta, \gamma)$.

10.1 Definition: each stage is an inner ODE with linear-in-$\tau$ slow forcing.

10.2 The "infinitesimal" terminology: order conditions derived in the $\Delta\tau \to 0$ limit, then substepped.

10.3 WS-RK3 as a special case ($\alpha = 0$, $\gamma = 0$, diagonal $\beta$). SSP-RK3 as another corner ($\alpha \ne 0$, $\gamma = 0$).

10.4 Why $\gamma \ne 0$ recovers third-order accuracy (cancels the $1/N_s$ splitting error).

10.5 Connection to commutator-free exponential integrators (Celledoni-Marthinsen-Owren 2003; Owren 2006; Hochbruck-Ostermann 2005).

10.6 Stability factor-of-two gain vs basic split-explicit.

📚 **Blocked on:** Wensch-Knoth-Galant 2009 (BIT) — need the actual derivation, not the ClimateMachine.jl docs' restatement. The $(\alpha, \beta, \gamma)$ structure I currently have may need correction.

Current §9 of `_draft_first_pass.md` is built on the ClimateMachine summary and **must be reworked** once Wensch-Knoth-Galant is in hand. Suspect issues:
- The exact form of the linear-in-$\tau$ slow forcing
- The relationship $c = (I - \alpha - \gamma)^{-1} d$ — physical interpretation
- Order conditions derivation

---

**§11. Knoth-Schlegel-Wensch 2014 atmospheric MIS methods.** 📚 (sketch only; needs primary-source detail)

The MIS family specifically tuned for atmospheric Euler. Specific tableaux. Stability optimization. Documented 4× macro-step claim.

11.1 The 3-stage and 5-stage variants; specific $(\alpha, \beta, \gamma)$ matrices.

11.2 Stability analysis for the atmospheric system; comparison to WS-RK3.

11.3 Per-step cost analysis: how the 5-stage variant ends up comparable to WS-RK3 because each stage's $d_i$ is smaller.

11.4 Convergence tests on standard atmospheric benchmarks (cold bubble, IGW, mountain wave).

📚 **Blocked on:** Knoth-Schlegel-Wensch 2014 (MWR 142) — need the explicit tableaux and the validation results.

---

**§12. The MRI-GARK framework (Sandu, Günther et al.).** 📚 (overview only)

Multirate Infinitesimal Generalized Additive Runge-Kutta — the unified framework that contains MIS, IMEX-RK, exponential RK, and split-explicit RK as special cases.

12.1 GARK foundation (Sandu-Günther 2015 Numer. Math.).

12.2 MRI-GARK extension (Sandu 2019 SINUM): allows implicit fast solvers, richer slow-fast coupling.

12.3 Order conditions theory; methods up to order 4.

12.4 Linearly-implicit MRI-GARK (Sandu et al. 2021 BIT): for stiff fast components.

12.5 What this framework offers Breeze beyond MIS: theoretical context for further extensions.

📚 **Blocked on:** Sandu 2019 (SINUM). Possibly Sandu-Günther 2015 (Numer. Math.).

---

**§13. Other Knoth-style alternatives.** 🟡 (briefly drafted; needs depth)

- Jebens, Knoth & Weiner 2009: explicit two-step peer methods. Cited in Baldauf 2010 as alternative to RK-based time splitting.
- Schlegel-Knoth-Arnold-Wolke (chemistry-transport multirate): different application; provenance for atmospheric MIS.

📚 **Blocked on:** Jebens-Knoth-Weiner 2009 (MWR 137) for tier 2.

---

### Part V — Implementation considerations

**§14. Conservative-form vs perturbation-form prognostics.** 🟡 (partially drafted)

The choice of substep prognostic variables matters for conservation, accuracy, and code structure.

14.1 Full conservative state $(\rho, \rho\boldsymbol u, \rho\theta)$ updated directly: the simplest mental model, used by ERF in some configurations.

14.2 MPAS-A perturbation form $(\rho'', (\rho u)'', \ldots, (\rho\theta)'')$: the variables Baldauf, MPAS-A, Breeze's PR #622, and most production implementations advance. Reference is subtracted before the substep loop and added back at recovery.

14.3 Velocity-Exner form $(u, v, w, \pi')$ (CM1; Breeze's published-but-superseded docs): a primitive-variable prognostic that simplifies the linearized PGF but complicates conservation.

14.4 Static-energy form $(\rho, \rho\boldsymbol u, \rho e)$: alternative thermodynamic prognostic; Breeze supports this.

Trade-offs in conservation, accuracy at the linearization boundary, and discrete operator complexity.

📚 **Blocked on:** Bryan-Fritsch 2002 (CM1 origin of velocity-Exner); Klemp-Skamarock-Dudhia 2007 (MPAS-A perturbation form).

---

**§15. Implementation comparison table.** ❌ (not drafted)

A side-by-side comparison of WRF, MPAS-A, ERF, COSMO, CM1, and Breeze.jl on:

- Outer RK family (WS-RK3 vs SSP-RK3)
- Substep variables (perturbation form vs primitive)
- Linearization point (reference state, outer-step frozen, stage frozen)
- Off-centering (single $\omega$ vs split $\beta_S, \beta_B$)
- Damping strategy (PPD, TDD, both, neither)
- Boundary handling (Periodic, Bounded, terrain-following)
- Acoustic substep variables (which fields actually advance)
- Recovery (how the substep result becomes the new RK stage state)

This table is the practical payoff of the theoretical framework. Once filled in it makes the design decisions for a new implementation transparent.

📚 **Blocked on:** Skamarock-Klemp 2008 (WRF dycore), Bryan-Fritsch 2002 (CM1), Skamarock et al. 2012 (MPAS-A — accessible). ERF docs are accessible. COSMO via Baldauf 2010 + cited derivatives.

---

**§16. Recommended Breeze design.** 🟡 (drafted from PR #622)

The synthesis. What Breeze should do, given the framework laid out in §1–§15.

16.1 Slow operator: advection + Coriolis + closure + microphysics only. No PGF, no buoyancy, no damping.

16.2 Fast operator: full PGF + full buoyancy + damping, linearized about the time-independent hydrostatic reference state.

16.3 Substep prognostic variables: MPAS-style $(\rho'', (\rho u)'', (\rho v)'', (\rho w)'', (\rho\theta)'')$.

16.4 Off-centering: $\beta_S = 0.5$ (neutral CN for sound), $\beta_B = 0.7$ (slight damping for buoyancy).

16.5 Damping default: `ThermodynamicDivergenceDamping(0.1)` for LES-scale; `PressureProjectionDamping(0.5)` for synoptic.

16.6 Outer integrator: SSP-RK3 default for monotonicity; `AcousticRungeKutta3` (WS-RK3) fallback for stability margin; `AcousticMIS3` / `AcousticMIS5` future for 4× macro-step gain.

16.7 Substep discretization: HEVI Crank-Nicolson, single tridiagonal solve per substep, addition-of-tendencies (no operator splitting between sound/buoyancy/damping).

16.8 Reference state: time-independent, dry-adiabatic with smooth $\theta_0(z)$ matching mean stratification.

Material to migrate: §10 of `_draft_first_pass.md`.

---

**§17. Verification and validation strategy.** ❌ (not drafted)

What tests should pass for a correct implementation. Drawn from PR #622's `validation/substepping/` suite and the canonical literature benchmarks.

17.1 $N_s$-consistency: vary $N_s$, expect convergence to a single answer (Baldauf Eq. 14). The PR #622 diagnostic.

17.2 Pure-acoustic plane-wave test: zero gravity, sinusoidal $(\rho\theta)''$ initial condition. Should propagate without growth.

17.3 Inertia-gravity wave (IGW) — Skamarock-Klemp 1994 standard benchmark.

17.4 Dry thermal bubble — Robert / Bryan-Fritsch sequence.

17.5 Cold density current — Straka et al. 1993.

17.6 DCMIP2016 baroclinic wave (where Breeze's PPD default $\beta_d = 0.5$ was tuned).

17.7 Linear hydrostatic / non-hydrostatic mountain wave (Schär-Durran-Stevens 2002).

17.8 Convergence rate tests against the "infinitesimal" limit (Knoth-Schlegel-Wensch 2014 use this for order verification).

📚 **Some tests blocked on:** primary-source derivations of the benchmarks.

---

### Appendices

**Appendix A. Notation and variable cross-reference.** ✅ (mostly drafted)

Comprehensive table mapping Breeze code symbols to mathematical notation to MPAS aliases to Baldauf 2010 symbols. Includes off-centering parameters, Courant numbers, tridiagonal coefficient names.

Material to migrate: §0 of `_draft_first_pass.md`.

---

**Appendix B. Linearization derivations.** ✅ (drafted)

Explicit step-by-step linearization of mass, momentum, thermodynamic, and EoS equations. Both Form 1 (conservative ratio) and Form 2 (potential-temperature × momentum) presentations of the thermodynamic flux.

Material to migrate: §A1 of `_draft_first_pass.md`. Already shown both forms; clean up and consolidate.

---

**Appendix C. Stability proofs.** 🟡 (sound + buoyancy proofs sketched; need verification against primary sources)

Eigenvalue analysis of the sound substep amplification matrix. Eigenvalue analysis of the buoyancy substep. Combined system. Divergence damping.

📚 **Cross-reference against:** Skamarock-Klemp 1992; Baldauf 2010 Appendix A.

Material to migrate: §A2, A3 of `_draft_first_pass.md`.

---

**Appendix D. Implementation provenance table.** ❌ (not drafted)

Detailed comparison across WRF, MPAS-A, ERF, COSMO, CM1, Breeze (this is what §15 references but here lives the actual full table with code links).

📚 **Blocked on:** Skamarock-Klemp 2008 (WRF), Bryan-Fritsch 2002 (CM1), Skamarock et al. 2012 (MPAS-A).

---

**Appendix E. Bretherton transformation: why it's not used here.** ✅ (drafted)

Standard caveat from Baldauf 2010 Appendix B: the Bretherton transformation interacts pathologically with divergence damping, producing spurious instabilities.

Material to migrate: §B of `_draft_first_pass.md`.

---

**Appendix F. The moist acoustic PGF: open question.** ❌ (not drafted; flagged in PR #622)

PR #622's `NOTES.md` documents that Breeze's `LiquidIcePotentialTemperatureFormulation` EoS uses moist mixture exponent $\gamma^m$, not dry $\gamma^d$; the standard MPAS-derived PGF is dry-correct only. For saturated air, latent-heat offset terms in the EoS contribute to $\partial\theta_{li}$ that the standard linearization misses. Open derivation.

This will need original work, not just a literature review.

---

## REFERENCES NEEDED

### Tier 1 (foundational, required for §10–§11 and §7 rigor)

| # | Citation | Purpose |
|---|---|---|
| 1 | Wensch, Knoth & Galant, 2009. *Multirate infinitesimal step methods for atmospheric flow simulation.* BIT 49, 449–473. | The foundational MIS paper; Sections §10 and §12 depend on it. |
| 2 | Knoth, Schlegel & Wensch, 2014. *Generalized split-explicit Runge–Kutta methods for the compressible Euler equations.* MWR 142, 2067–2086. | Atmospheric MIS tableaux and validation; §11. |
| 3 | Skamarock & Klemp, 1992. *The stability of time-split numerical methods for the hydrostatic and nonhydrostatic elastic equations.* MWR 120, 2109–2127. | Original stability framework; §7. |
| 4 | Klemp, Skamarock & Dudhia, 2007. *Conservative split-explicit time integration methods for the compressible non-hydrostatic equations.* MWR 135, 2897–2913. | Conservative-form linearization derivation; §4, §5, §8. |

### Tier 2 (important secondary)

| # | Citation | Purpose |
|---|---|---|
| 5 | Klemp, Skamarock & Ha, 2018. *Damping acoustic modes in compressible HEVI and split-explicit time integration schemes.* MWR 146, 1911–1929. | Thermodynamic divergence damping derivation; §8. |
| 6 | Sandu, 2019. *A class of multirate infinitesimal GARK methods.* SIAM J. Numer. Anal. 57, 2300–2327. | MRI-GARK framework; §12. |
| 7 | Jebens, Knoth & Weiner, 2009. *Explicit two-step peer methods for the compressible Euler equations.* MWR 137, 2380–2392. | Peer-method alternative; §13. |

### Tier 3 (useful for completeness)

| # | Citation | Purpose |
|---|---|---|
| 8 | Skamarock & Klemp, 2008. *A time-split nonhydrostatic atmospheric model for weather research and forecasting applications.* J. Comput. Phys. 227, 3465–3485. | WRF dycore; §15, Appendix D. |
| 9 | Bryan & Fritsch, 2002. *A benchmark simulation for moist nonhydrostatic numerical models.* MWR 130, 2917–2928. | CM1 velocity-Exner formulation; §14. |
| 10 | Sandu & Günther, 2015. *Multirate generalized additive Runge–Kutta methods.* Numer. Math. 133, 497–524. | GARK foundation; §12. |

### Already in hand

| Citation | Status |
|---|---|
| Baldauf, M., 2010. *Linear stability analysis of Runge–Kutta-based partial time-splitting schemes for the Euler equations.* MWR 138, 4475–4496. | Full PDF read. |
| Wicker, L. J. & Skamarock, W. C., 2002. *Time-splitting methods for elastic models using forward time schemes.* MWR 130, 2088–2097. | Full text read via UCAR. |
| Breeze.jl source + PR #622 (`acoustic_substepping.jl`, `REFACTOR_PLAN.md`, `NOTES.md`, `REPORT.md`). | Read. |
| Skamarock et al., 2012. *MPAS-A technical note* NCAR/TN-475+STR. | Citing directly via documentation. |
| ERF documentation (`erf.readthedocs.io`). | Citing via online docs. |

### Not blocking but worth having

- Hochbruck & Ostermann, 2005. *Exponential Runge–Kutta methods for parabolic problems.* Appl. Numer. Math. 53, 323–339.
- Celledoni, Marthinsen & Owren, 2003. *Commutator-free Lie group methods.* Future Gener. Comput. Syst. 19, 341–352.
- Owren, B., 2006. *Order conditions for commutator-free Lie group methods.* J. Phys. A 39, 5585–5599.
- Schlegel, Knoth, Arnold & Wolke, 2009/2012. Multirate methods for tropospheric chemistry-transport.
- Sandu et al., 2021. *Multirate linearly-implicit GARK schemes.* BIT.
- Wicker & Skamarock, 1998. *A time-splitting scheme for the elastic equations incorporating second-order Runge-Kutta time differencing.* MWR 126, 1992–1999.
- Klemp & Wilhelmson, 1978. *The simulation of three-dimensional convective storm dynamics.* J. Atmos. Sci. 35, 1070–1096.

---

## NEXT STEPS

1. **Greg supplies tier-1 papers** (Wensch-Knoth-Galant 2009, Knoth-Schlegel-Wensch 2014, Skamarock-Klemp 1992, Klemp-Skamarock-Dudhia 2007).

2. **Restructure** `_draft_first_pass.md` into the outline above. Most existing content migrates with light edits; the §9 (now §10–§11) MIS treatment needs full rewrite from primary sources.

3. **Verify** stability proofs in Appendix C against Skamarock-Klemp 1992 and Baldauf 2010 directly.

4. **Fill** §15 implementation comparison table from primary sources (WRF, CM1, MPAS-A papers).

5. **Draft** §17 verification suite by surveying PR #622's `validation/substepping/` and the literature benchmarks.

6. **Open derivations** flagged in Appendix F (moist acoustic PGF) — these may need original work beyond literature review.

---

## OPEN QUESTIONS THAT NEED RESOLUTION

1. **What is the right linearization point for $\theta_v$ in PR #622?** The merged code uses stage-frozen $\theta_v$ (computed each RK stage from current state); the PR's REFACTOR_PLAN flags moving this to time-independent reference as a TODO. Theoretical answer is "time-independent for Baldauf consistency" but practical accuracy implications need testing.

2. **What MIS tableau should Breeze adopt?** Knoth-Schlegel-Wensch 2014 presumably recommends a specific 5-stage method; need to read the paper to know which.

3. **Does the moist acoustic PGF derivation produce a linearly stable substep?** Open per PR #622 NOTES; may need original analysis.

4. **What is the stability of MIS schemes when combined with HEVI vertical implicit substep?** Wensch-Knoth-Galant analyze MIS with abstract $T_F$; need to verify their stability claims hold with the specific HEVI CN inner integrator.

5. **Does the $N_s$-consistency property of MIS schemes match Baldauf's $N_s \to \infty$ analysis for WS-RK3?** Conjecturally yes; needs verification.

6. **Is there a meaningful difference between the various damping strategies' effect on gravity-wave dispersion?** Klemp-Skamarock-Ha 2018's TDD is supposed to be better than KSD07's PPD on this; magnitude of the difference for atmospheric scales unclear without primary source.

7. **Should Breeze move toward conservative-form direct prognostics (no perturbation form) at all?** ERF supports both modes; pros/cons need explicit tradeoff analysis.

---

*Working draft: 2026-04-25. Companion file `_draft_first_pass.md` contains the previous "Baldauf rewrite" content that will migrate into the structure above as references arrive.*
