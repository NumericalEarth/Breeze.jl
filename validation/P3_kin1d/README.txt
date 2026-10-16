Outlier diagnosis (2026-10-15)
==============================

See outlier_results.txt for controlled attribution of the three matrix outliers:
Cooper nucleation adjustment, per-call ice/rime cleanup, and the saturation
gas-constant ratio shifting a fast freezing event. These are not interchangeable
with process-code bugs. One benchmark configuration error was fixed: upstream's
aerosol gas constant is 8.3145, not 8.314. Original matrix outputs/results remain
archived with the original setting; corrected outliers are saved separately.
Move old cached breeze_dt* directories aside before regenerating the full matrix
with the corrected constant (parcel_matrix.jl otherwise reuses existing files).

After generating the matrix and its finest outlier reference below, run:

  julia --project=validation/P3_kin1d validation/P3_kin1d/outlier_breeze.jl
  julia --project=validation/P3_kin1d validation/P3_kin1d/outlier_process_probes.jl
  julia --project=validation/P3_kin1d validation/P3_kin1d/outlier_reference.jl baseline
  julia --project=validation/P3_kin1d validation/P3_kin1d/outlier_reference.jl nucleation_relaxation
  julia --project=validation/P3_kin1d validation/P3_kin1d/outlier_reference.jl retain_trace_ice
  julia --project=validation/P3_kin1d validation/P3_kin1d/outlier_reference.jl retain_trace_rime
  julia --project=validation/P3_kin1d validation/P3_kin1d/outlier_reference.jl retain_trace_ice_and_rime
  julia --project=validation/P3_kin1d validation/P3_kin1d/outlier_reference.jl consistent_saturation
  julia --project=validation/P3_kin1d validation/P3_kin1d/evaluate_outliers.jl

The Breeze runs reuse run_production_parcel and its native Simulation/RK3, with
read-only output every 0.1 s. The reference script copies the existing generated
Fortran host/source into isolated output directories and saves each intervention
patch. Its baseline is checked bit-for-bit against the archived reference.
The interventions modify upstream numerics/constants only for causal diagnosis:
they are neither proposed upstream patches nor replacements for official data.
The process-probe script uses the reference adapter solely for identical-state
formula evaluation; it advances no Breeze parcel. All results/plots are under
validation_output/p3_matrix/outliers. No CM1 or docs build is required.

Adiabatic ParcelModel matrix (2026-10-15)
========================================

An 18-case factorial design varies initial temperature (288, 300 K), initial
liquid-relative humidity (60, 80, 100%), and prescribed ascent (0.5, 1, 2 m/s).
All cases start condensate-free at 101325 Pa and run for 5400 s. There is no
entrainment or fallout. theta_li is conserved. The same 300 K dry reference
pressure profile is used in every case. Thus the final heights are 2.7, 5.4,
and 10.8 km: the matrix includes both ascent-rate and ascent-distance effects.

Run after installing the validation environment as described below:

  julia --project=validation/P3_kin1d validation/P3_kin1d/process_reference.jl
  julia --project=validation/P3_kin1d validation/P3_kin1d/parcel_matrix.jl prepare
  julia --project=validation/P3_kin1d validation/P3_kin1d/trajectory_reference.jl 0.003125 liquid_ice validation_output/p3_matrix/cases.dat
  julia --project=validation/P3_kin1d validation/P3_kin1d/parcel_matrix.jl 0.01
  julia --project=validation/P3_kin1d validation/P3_kin1d/plot_parcel_matrix.jl

The matrix reuses run_production_parcel: actual ParcelModel dynamics and native
RK3 through Simulation/run!. The upstream uses compiled P3_MAIN in the existing
Fortran pressure/theta_li host, not CM1 or kin1d. It retains upstream process
updates and clipping. No Breeze comparison adapter or new advancement is used.
This adds validation cases and plots, with no production physics/numerics changes.

Initial vapor is derived from RH using Flatau saturation and epsilon=0.622;
the identical water amount is supplied to both models on their respective mass
bases. Fixed aerosol is 3e8 per kg dry air. As in the earlier benchmark, public
constructors select reference constants, while Breeze retains mixture
thermodynamics and prognostic supersaturation. Process timescale tau=0.025 s is
held fixed when refining the integration timestep; package defaults are unchanged.

Optional refinements (case IDs 3,9,14,18; then 3,9,18):

  julia --project=validation/P3_kin1d -e 'using DelimitedFiles; p="validation_output/p3_matrix"; a=readdlm(joinpath(p,"cases.dat")); writedlm(joinpath(p,"refinement_cases.dat"),a[[3,9,14,18],:]); writedlm(joinpath(p,"outlier_cases.dat"),a[[3,9,18],:])'
  julia --project=validation/P3_kin1d validation/P3_kin1d/parcel_matrix.jl 0.005 3,9,14,18
  julia --project=validation/P3_kin1d validation/P3_kin1d/trajectory_reference.jl 0.0015625 liquid_ice validation_output/p3_matrix/refinement_cases.dat
  julia --project=validation/P3_kin1d validation/P3_kin1d/parcel_matrix.jl 0.0025 3,9,18
  julia --project=validation/P3_kin1d validation/P3_kin1d/trajectory_reference.jl 0.00078125 liquid_ice validation_output/p3_matrix/outlier_cases.dat
  julia --project=validation/P3_kin1d validation/P3_kin1d/plot_parcel_matrix.jl

Outputs are in validation_output/p3_matrix. Baseline figures/reports are in
comparison_0.01_0.003125. Raw states are saved every 10 s, including prognostic
supersaturation, theta_li, and independently diagnosed total-water components.
Existing Breeze case files are reused; remove the relevant output directory
before rerunning after a physics/setup change. Do not plot while runs are pending.
The upstream raw states remain dry-air ratios; conversion for analysis is read-only.
The pinned upstream version and reference patch are documented below and retained
in validation_output/p3_processes.

Figures (PNG and SVG, large panels with cell values):
  final_{temperature,cloud,rain,ice,ice_number}: upstream, Breeze, signed difference;
  trajectory_errors: maximum temperature and mean cloud/rain/ice mass differences;
  ice_number_errors: maximum number difference / upstream peak;
  trajectory_envelopes: min-max design envelopes and medians, not confidence bands;
  outlier_refinement: temperature differences and ice number at available resolutions.

Mass-error heatmaps divide by initial total water, avoiding division by near-zero
species. Ice-number relative error is undefined (gray) when upstream peak number
is <=1e-3 per kg. Trajectory maxima refer to the saved 10 s samples, not every
internal step. Physical assertions check conservation, positivity, matching
forcing, and supersaturation consistency; no interimplementation error threshold
was chosen after seeing the results. See parcel_matrix_results.txt for findings.

Production ParcelModel comparison (2026-10-15)
==============================================

This is the production-trajectory validation. The older adapter-based comparisons
below are process tests and historical investigations, not ParcelModel validation.

Run from the repository root (Julia and gfortran required):

  julia --project=validation/P3_kin1d -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
  julia --project=validation/P3_kin1d validation/P3_kin1d/process_reference.jl
  julia --project=validation/P3_kin1d validation/P3_kin1d/trajectory_reference.jl 0.0015625 liquid_ice
  julia --project=validation/P3_kin1d validation/P3_kin1d/parcel_compare.jl 0.0025 0.025 reference_constants verified
  julia --project=validation/P3_kin1d validation/P3_kin1d/evaluate_parcel_model.jl

The fine runs take substantial time. parcel_compare.jl initializes an actual
AtmosphereModel with ParcelDynamics, then calls Simulation/run!. Its callbacks
only record output. It imports no comparison adapter, overrides no thermodynamic
methods, adds no custom advancement, and never resets supersaturation during the
simulation. Both cases start at T=300 K, p=101325 Pa, total water 0.015 kg/kg, with
no condensate; they ascend at 1 or 2 m/s for 5400 s. The latter reaches the ice
regime. The pressure profile is sampled on 1 m centers starting at z=0.

The upstream trajectory still comes from compiled P3_MAIN with a small Fortran
host that prescribes pressure and conserves theta_li. This does not run CM1 or
kin1d. The host's reconstruction, supersaturation resets, and P3's native internal
constant-cp approximations are retained in the reference, not copied into Breeze.

Configurations and numerical choices:
  reference_constants: Use public constructors to select upstream gas constants,
    gravity, liquid latent-heat reference, Flatau saturation, and aerosol constants.
    Breeze still uses native mixture thermodynamics and variable latent heats.
  native_defaults: Native constants, 1 s activation timescale; pass tau=10 for
    the other native process timescales. The fixed aerosol is a case choice.
  default: Native constants, but ALL listed process timescales, including
    activation, equal the passed tau. This is a controlled experiment, not the
    complete default configuration.

The comparison uses tau=0.025 s for activation, donor limiting, freezing,
refreezing, nucleation, rime densification, and rain evaporation. This is a
validation numerical setting; package defaults are unchanged. Hold tau fixed
while refining the actual RK timestep. Upstream's per-step limits have different
semantics. Matching a finite timestep is not evidence of a common continuum limit.
The aerosol reservoir is fixed at 3e8 per kg dry air, or 2.955e8 per kg moist air.
Both parcels are closed to water, with no sedimentation or entrainment.

Outputs: validation_output/p3_parcel_direct. Each run retains all prognostics,
theta_li, total water, storage density, thermodynamic density, and independently
diagnosed supersaturation. evaluate_parcel_model.jl converts only recorded upstream
outputs to the moist-air basis. Its supersaturation coordinate conversion is
s_q = q_d * epsilon * s_r / (epsilon + r_sat), not simply q_d*s_r.
It reports errors and makes large SVG/PNG plots. Its assertions check physical
consistency; they do not impose a post-hoc interimplementation agreement threshold.
See parcel_model_results.txt for measured differences, tests, and change categories.

Refinement (run each entry with the corresponding script above):
  Upstream dt: 0.00625, 0.003125, 0.0015625 s.
  ParcelModel dt: 0.005, 0.0025, 0.00125 s, always tau=0.025 s.
  julia --project=validation/P3_kin1d validation/P3_kin1d/parcel_refinement.jl

The following sections preserve earlier experiments and their original results.
Their statements about production supersaturation describe the earlier coupling.

Direct P3 validation (no CM1)
============================

Run the direct process and trajectory comparisons:

  julia --project=validation/P3_kin1d -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
  julia --project=validation/P3_kin1d validation/P3_kin1d/run_direct.jl

Julia and gfortran are required. Outputs, provenance, the complete reference patch,
raw states/rates, metrics, and large SVG/PNG figures go to
validation_output/p3_processes. The source is pinned and checked by SHA256;
no CM1 source is used. See direct_results.txt for quantitative results and scope.

The direct suite calls upstream P3_MAIN, not kin1d. It uses the same table artifact,
one ice category, two ice moments, liquid fraction, prognostic droplet number,
fixed aerosol distribution, and prognostic supersaturation. Sedimentation is
disabled in the reference because these are closed parcels. The only other
reference changes enable supersaturation, disable optional precipitation-type
diagnostics, expose the saturation function, and print process rates. No reference
microphysical process formula is changed.

process_reference.jl: twelve identical initial-state probes; 35 limited rates and
  full one-step prognostic updates from compiled Fortran P3.
process_compare.jl: compare Breeze's actual process-rate/tendency functions;
  assert rate agreement and projected state agreement, retaining both raw and
  projected Breeze states in output.
reference_conventions.jl: test-only thermodynamic and integration adapter.
trajectory_reference.jl / trajectory_compare.jl: twelve 600-second boxes and two
  5400-second ascents at 1 and 2 m/s; dt=0.1 s. Pass 1 or 0.5 to both scripts
  to inspect coarser runs. Those under-resolve the cold nucleation bursts and
  can fail the trajectory agreement assertions; their outputs are kept separate.

The adapter is essential: reference P3 uses dry-air mixing ratios, constant latent
heats and cp=1005, whereas production Breeze uses moist-air mass fractions and
mixture thermodynamics. Saturation, saturation-budget limits, and the melting
surface balance all use the reference dry-air convention here. Physical constants
are set to their documented upstream values, including aerosol molecular weights
and the rounded liquid-fraction shed-drop mass. These are not parameter fits.
Breeze normally relaxes PSD constraints through its RHS; the comparison driver
also applies upstream's final hard number/trace projection, using existing Breeze
diagnoses. Raw, unprojected one-step output is retained to expose that distinction.
The sink/freezing/nucleation and aerosol-activation timescales equal the reference
timestep in both implementations (1 s for the single-step probes; 0.1 s for the
accepted trajectories). All 14 trajectory cases are tested against the same
0.001 K temperature and 0.5%-of-peak state tolerances, plus a 1e-10 absolute floor
for trace species. Closed-parcel water conservation is checked independently.

For ascents the initial state is T=300 K, p=101325 Pa, vapor mass fraction 0.015
(converted to dry-air mixing ratio), and no condensate. Both implementations use
the same isentropic reference pressure profile and adiabatic pressure change.
After this external cooling, supersaturation is updated consistently with the
new temperature before calling P3. There is no entrainment or sedimentation.
These share the parcel example's launch conditions, with the 2 m/s case extending
into the ice regime. They deliberately remove native host thermodynamics as a
confounding variable; they are not claims of identical full-column integrations.

The older host-column study is retained below. Its differences motivated the
direct tests; it should not be interpreted as a failed microphysics rate test.

P3 kinematic-column intercomparison
==================================

This study runs the physical case from examples/kinematic_driver.jl with Breeze
and the upstream Fortran kin1d driver. It is a comparison of two complete host
integrations, not a process-by-process proof of equivalence. See results.txt for
the measured outcome and remaining differences.

Run from the Breeze repository root (Julia and gfortran required):

  julia --project=validation/P3_kin1d -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
  julia --project=validation/P3_kin1d validation/P3_kin1d/run_all.jl

Outputs go to validation_output/p3_kin1d, or P3_COMPARISON_OUTPUT if set. Expect
several minutes of Julia compilation. The Fortran simulation itself is short.
The root package environment and dependencies are not modified.

Scripts:
  run_breeze.jl        Export the example and its controlled variants.
  prepare_reference.jl Download pinned upstream sources, patch, compile, and run.
  compare.jl          Check alignment and finiteness, calculate errors, plot.
  run_all.jl          Reproduce the complete comparison sequentially.

Upstream:
  https://github.com/P3-microphysics/P3-microphysics
  commit f90df9613fc90360f846d2f12114b967ef3e53ce (P3 5.5.2)
  table 1 version 6.9-2momI; one ice category, two ice moments, liquid fraction on.
  Both implementations read Breeze's pinned table artifact. Its table 1 was
  checked byte-for-byte against the decompressed upstream file:
  SHA256 0992f75c5d534eeb5a2f37be7734612de8f87f8b9e0df759d3b1df18e53d373b

Case:
  100 cells, 200 m spacing, centers from 100 to 19900 m; 3600 s duration.
  Prescribed upward velocity 2 m/s; profiles saved every 30 s.
  Initial T, p, and vapor fraction exported directly from Breeze's example.
  Surface inflow T=300 K, vapor from the example's 80% liquid-relative humidity,
  no incoming condensate, open precipitation boundaries.
  Native host thermodynamics and sedimentation retained; these are not identical.

Reference changes (all recorded in kin1d_dt*/reference.patch):
  - Read the exact initial profiles and surface inflow; bypass native sounding.
  - Use the shared uniform grid, steady updraft, duration, and timestep.
  - Disable native low-level moisture replenishment and DIV/COMP terms.
    The latter makes transport of specific quantities consistent with the
    divergence-corrected Breeze setup.
  - Transport supersaturation and apply fresh-air inflow at the bottom cell.
  - Replace the rounded 0.286 Exner exponent with P3's actual rd/cp.
  - Enable log_predictSsat in microphy_p3.f90. The driver's prog_nc_ssat
    parameter does not control that hard-coded switch in this upstream version.
  - Expose qv_sat for diagnostics; do not change its formula.
  - Add common-format output. No microphysical process equations are changed.
  - Compile in double precision with bounds checking; compiler flags are in
    prepare_reference.jl. Compiler and source provenance are saved with outputs.

Variable mapping:
  Breeze uses fractions/numbers per moist-air mass. Reference P3 uses mixing
  ratios per dry-air mass. Initial vapor is converted with r=q/(1-q). Reference
  outputs are divided by 1+rv+rc+rr+ritot. Frozen ice is ritot-rili, because
  reference total ice includes coating whereas Breeze q^i excludes it.
  Supersaturation in reference output is also multiplied by the dry-air mass
  fraction; its native saturation convention still differs from Breeze's.

Variants:
  reservoir: Original example with prognostic aerosol depletion and WENO5.
  fixed: Fixed aerosol distribution, matching upstream's reservoir assumption.
  upwind: Fixed aerosol and first-order advection, closer to kin1d transport.
  reference_options: Upwind + Flatau saturation + 1 s sink/nucleation/freezing
    relaxation instead of Breeze's 10 s defaults. Initial T/p/vapor and vapor
    inflow are preserved. This is a bundled sensitivity test, not a claim that
    all thermodynamic or time-integration differences have been eliminated.
  Timestep refinement: fixed and reference at 1 s and 0.5 s.
  Dry control: Same sounding and updraft, zero water everywhere, fixed aerosol,
    first-order advection. Run in a separate output directory. P3 remains called
    but all water/particle fields must stay exactly zero. Temperature differences
    in this control are host thermodynamics/transport differences, not phase changes.

Interpretation:
  compare.jl asserts finite, complete output, positivity to roundoff, and matching
  initial T/p/vapor/grid/times. These are harness checks, not agreement tests.
  Mass-profile error is the density-weighted space-time L1 difference divided
  by the reference norm, excluding t=0. Temperature error is space-time RMS.
  Plotted column amounts use the same initial reference density for every run.
  They compare distributions; they are not a closed water or precipitation budget.
  No acceptance tolerance was chosen after seeing the results and no broad
  correctness claim is inferred from visually similar curves.

Remaining host differences:
  - Breeze integrates simultaneous RK3 tendencies with fixed relaxation times;
    reference P3 performs in-place, split updates with per-step limiters/clips.
  - Breeze's diffusional-growth forcing uses only -g*w/cp and zero external
    vapor tendency. Reference P3 uses resolved changes in T and vapor supplied
    by its driver (see aaa and equivalent velocity in microphy_p3.f90).
  - Reference P3 rediagnoses supersaturation at the end of its update; Breeze
    relaxes a transported prognostic toward a hypothetical post-process value.
  - Breeze uses moist heat capacities, variable latent heats, fixed reference
    density, and its saturation-specific-humidity convention. Reference P3
    uses dry thermodynamic approximations and diagnoses density from p/(Rd*T).
  - Sedimentation, boundary discretization, and time splitting remain different
    even in the first-order-advection variant.
  - The nominal fixed aerosol count is 3e8 per kg moist air in Breeze and per kg
    dry air upstream; that basis difference is not removed by a fixed scalar.

The next discriminating validation is an ensemble of identical-state,
single-process tendency/update comparisons, followed by matched transport-only
controls. A column mismatch alone cannot uniquely identify a process-code bug.

Liquid–ice potential-temperature rerun (2026-10-15)
=================================================

AtmosphereModel already defaults to :LiquidIcePotentialTemperature. ParcelDynamics
previously ignored that choice and always constructed StaticEnergyState. The parcel
materialization and initialization now honor the formulation; the examples explicitly
select liquid–ice potential temperature for clarity. Static energy remains opt-in.

Run the production parcel examples and conservation checks:

  julia --project=validation/P3_kin1d validation/P3_kin1d/validate_liquid_ice_parcels.jl

The trajectories, conservation report, and five large SVG/PNG figures are saved to
validation_output/p3_liquid_ice/parcels. Run the column example in the sibling
kinematic directory to regenerate its JLD2 profiles and two SVG/PNG figures. That
column was already using liquid–ice potential temperature before this rerun.

run_direct.jl now also runs the 14 direct-P3 trajectories with a conserved liquid–ice
potential-temperature host, at dt=0.05 and 0.025 s. To run just this comparison:

  julia --project=validation/P3_kin1d validation/P3_kin1d/trajectory_reference.jl 0.05 liquid_ice
  julia --project=validation/P3_kin1d validation/P3_kin1d/trajectory_compare.jl 0.05 liquid_ice

Outputs are in validation_output/p3_processes/trajectories_dt0.05_liquid_ice and
trajectories_dt0.025_liquid_ice. The ordinary reference-thermodynamics benchmark is
retained in its original directories. See liquid_ice_results.txt for results.

The Fortran driver independently computes

  cm = (cpd + cpv*rv + cpl*rl + cpi*ri) / (1 + rv + rl + ri)
  Rm = (Rd + Rv*rv) / (1 + rv + rl + ri)
  T = theta_li * (p/p_standard)^(Rm/cm) + (Lv*rl + Ls*ri)/(cm*(1+rv+rl+ri))

with constant theta_li. The Julia host uses Breeze's actual
LiquidIcePotentialTemperatureState instead of duplicating that equation. Its
DryBasisLiquidIceState adapter converts the reference dry mixing ratios to moist
mass fractions at the thermodynamic boundary, including hypothetical moisture
updates used by Breeze P3's supersaturation tendency. Both hosts reconstruct
T and supersaturation after each P3 call and after the imposed pressure change.
The initial theta_li is calculated from each case's initial T, pressure, and water.

This is a matched, operator-split host comparison. The upstream P3 source retains
its internal constant-cp heating and saturation-budget approximations; only the
external host reconstruction changes. Reference saturation, process constants,
per-step limiting, aerosol, and final PSD projection still match the original
benchmark. No production P3 formulas are changed. This does not make the direct
harness identical to production ParcelDynamics, which uses RK3, default relaxation
times, and the native moist-air P3 conventions. Production parcel conservation and
default selection are checked separately. Neither test asserts full kin1d column
agreement, since native kin1d retains different host thermodynamics and transport.
