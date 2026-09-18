# Sedimentation energy campaign

Small CPU/CUDA/Metal falsification probes for PR #959, production source
`dae9e46d543720f4f1c4f3a57e8d7c7817e90f24`. No production methods are replaced.
The phase-enthalpy variant is a diagnostic callback, not a validated full correction.

## Run

From the repository root, with Julia 1.11.9 or newer:

```sh
julia --project=validation/sedimentation_energy -e 'using Pkg; Pkg.instantiate()'
julia --project=validation/sedimentation_energy validation/sedimentation_energy/run_campaign.jl cpu 64 cpu64.toml
julia --project=validation/sedimentation_energy validation/sedimentation_energy/run_campaign.jl cuda 64 cuda64.toml
julia --project=validation/sedimentation_energy validation/sedimentation_energy/run_campaign.jl cuda 32 cuda32.toml
julia --project=validation/sedimentation_energy validation/sedimentation_energy/run_campaign.jl metal 32 metal32.toml
```

Add `coupled` as the last argument for the separate, eight-case acoustic suite.
Metal does not support Float64. GPU scalar indexing is disabled. Model fields and
tendencies reside on the selected architecture; small arrays are copied to the
host for Float64 diagnostic reductions and analytic differentiation.

The dedicated manifest pins Oceananigans **0.111.0**, CUDA 6.4.0 and Metal 1.11.0.
It uses the enclosing Breeze checkout through a relative path. It does not
change the root project's dependencies. Outputs record the source revision,
dirty flag, script/environment SHA256 hashes, backend, device and precision.
Each completed case is saved immediately; `completed=true` marks a finished run.
Keep stdout/stderr alongside the TOML output to retain assertion failures and
backend information.

## What the probes establish

- `instantaneous-*`: two 50 or 100 m cells, 280 K, zero gravity, prescribed
  1 m/s fall speed, vapor mass fraction 0.005, condensate 0.001/0.01, initially
  uniform 100 kPa pressure. Dry/vapor partial densities stay fixed during the
  isolated sedimentation tendency. Closed boundaries; equal-composition and
  no-fall controls. Liquid and ice, ordinary upwind and WENO, and bounded WENO.
  **Two-cell WENO cases are routing coverage**, not interior high-order tests.
- `interior-32-*`: a 32-cell closed column, 20 m cells and a smooth cosine
  condensate profile in [0.001, 0.01]. Its maximum is at interior cell 17.
  The 4×4 horizontal grid avoids automatic directional scheme wrapping on
  singleton dimensions; the test asserts the bounded dispatch type. Diagnostics
  use horizontally averaged columns on a domain of 1 m² horizontal area.
  Bounded cases record actual limited-minus-raw reconstructions and
  raw-minus-donor-cell reconstructions, including interior cells 4:29. Column
  mass tendency and normalized mass residual are retained independently of
  thermodynamic consistency. A nonzero thermal error alone is not evidence of
  mass nonconservation.
- `donor-reversal-*`: bulk vertical velocities -1, 0.5 and 2 m/s against a
  1 m/s fall speed; tests the difference between combined and bulk fluxes with
  their separate donors. Isothermal, compressible, first-order reconstruction.
- `nonisothermal-*`: 280/275 K and 280/285 K, otherwise the two-cell setup. The reference
  assumes isolated transport of phase enthalpy and standard mixture internal
  energy, with fixed gas partial densities and no mechanical energy exchange.
  It tests a **conditional** heating contract, not the full multiphase model.
  Compares original, phase-enthalpy-only, and phase enthalpy with the proposed
  fixed-volume coefficient `beta_cv = [1-kappa*(1-A/T)]/[(cpm-Rm)*Pi]`.
- `euler-*`: diagnostic Euler updates at 0.1, 0.01 and 0.001 s check approach
  to the instantaneous rate. At Float32 the smaller temperature increments are
  below rounding resolution; the differentiated tendency is the primary test.
- `implicit-*`: anelastic, uniform reference density, two 20 m cells, fall
  speed 8 m/s, nominal fall CFL 0.6, 1 and 4, open/closed bottom. The actual
  implicit speed is `max(0, 8 - 0.5*20/dt)`. An independent backward-Euler mass
  balance and finite thermal reconstruction are compared with the production
  implicit sedimentation update. These supplementary tests are **not a
  compressible energy reference**. Static energy is the control formulation.
- `acoustic-*` (separate suite): actual split-explicit time stepping for 2 s
  in a closed eight-cell column; explicit dt=0.1/0.025 s and AIVA
  dt=0.1/2 s. Records temperature profiles/drift, condensate mass change and
  maximum vertical velocity; it does **not** report a total-energy residual.
  No-fall controls distinguish sedimentation-triggered changes, but do not
  uniquely attribute coupled errors to a single operator.

## Independent thermodynamic diagnostic

Let `a`, `v`, `r` be dry-air, vapor and condensate partial densities. With the
phase heat capacity `cx` and reference latent offset `L`, define

```
C = a*cpd + v*cpv + r*cx
G = a*Rd + v*Rv
p = G*T; kappa = G/C; A = r*L/C
Pi = (p/p_standard)^kappa
Theta = a*(T-A)/Pi
```

The script analytically differentiates this executable definition at fixed
`a,v`: `Tdot = (Theta_dot - Theta_r*r_dot)/Theta_T`. This includes pressure's
temperature dependence. It does not differentiate the PR's content callback
to construct its expected value. At equal donor/receiver temperatures the
isolated operator should give zero. The diagnostic callback changes only the
donor-difference quantity from mixture-relative enthalpy to `hx=cx*T-L`,
retaining the PR's composition derivative and heating coefficient.

For the cold-donor conditional reference, receiving-cell heating is
`cx*(T_donor-T_receiver)*r_dot/(C-G)`; the donor remains isothermal.
Here the bottom is closed, so `r_dot` is the incoming rate. With open outflow,
the net tracer tendency is not an incoming rate: use the signed face sum of
`(h_donor-h_receiver)*mass_flux`, or `Q_h-h_receiver*r_net`, instead.
For the anelastic implicit reference, `H=C*T-r*L`, capacity changes by
`(cx-cpd)*delta(r)`, and transported relative enthalpy is
`(cx-cpd)*T-L`. Residuals include the actual backward-Euler bottom mass flux
and its initial donor temperature. Units: K, K/s, kg/m², J/m², as appropriate.

Isothermal rate tolerances are 2e-11 K/s (Float64) and 2e-6 K/s (Float32).
Finite static-energy temperature tolerances are 2e-10 K and 2e-4 K.
The mass solve is checked at `rtol=100eps(FT), atol=0`. These tolerances
separate arithmetic noise from the observed defects; they are not accuracy
requirements for a general simulation. Harness assertions test controls and
diagnostic consistency. **An all-pass harness does not mean the PR conserves
energy**: `original_isothermal_pass` explicitly records the physical failures.

## Dependency scope and limits

Oceananigans 0.111's bounded operator limits only the current cell's two
reconstructions; the other two stay raw. PR `src/Advection.jl` mirrors that
behavior. In 0.112 and 0.113, all four states use their own donor's precomputed
limiter, producing shared face fluxes. Do not generalize a 0.111 failure to
those versions. This campaign does not execute the newer bounded operator.
Rebase hazard: PR `bounded_face_reconstructions` copies the 0.111 operator;
it must be reconciled with the stored, donor-indexed limiter when moving to
0.112. Retaining the old helper would break thermal/tracer flux consistency.

Unverified here: P3 mixed coating/rime and differential-speed species,
phase change, nonzero gravity/drag, a fully closed compressible energy law,
pressure-work and external-heating contracts, general nonisothermal finite
implicit reconstruction, and long-duration/converged coupled dynamics.
The two cell-width cases check local tendency scaling, not spatial convergence
of a fixed continuum initial-value problem. No unrelated acoustic/AIVA fix is
included. See the recorded results for actual tested configurations and failures.
Compressible StaticEnergy is not exercised: its constructor/diagnostic path is
unsupported in this baseline. No measured compressible static-energy ratio is
claimed. Coupled acoustic runs use the original callback only; the corrected
coupled path and gravity-on energy balance are NOT RUN.
