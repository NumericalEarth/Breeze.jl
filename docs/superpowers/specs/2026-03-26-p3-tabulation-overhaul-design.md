# P3 Tabulation Overhaul Design

## Summary

This design replaces Breeze's current mixed analytical and table-backed P3 tabulation
implementation with a unified table system built around Oceananigans `TabulatedFunction{N}`
from Oceananigans `0.106.2`.

The overhaul has three goals:

1. Replace the runtime three-moment closure solve with table-backed lookup of `μᶦ`
   and related three-moment diagnostics, following the role of Fortran `lookupTable_3`
   while keeping Julia-native internal structure.
2. Fold the missing Fortran table families from `lookupTable_1` and `lookupTable_2`
   into Breeze so liquid-fraction-aware and multi-table process paths are available
   from a single tabulation subsystem.
3. Remove the bespoke Breeze `TabulatedFunction3D` and `TabulatedFunction4D`
   implementations in favor of the generic Oceananigans table type.

The result should make P3 table coverage auditable, keep runtime kernels allocation-free,
and provide a direct place to compare Julia table contents against the Fortran reference.

## Scope

### In scope

- Upgrade Breeze's Oceananigans compatibility to `0.106.2`.
- Replace custom P3 interpolation types with Oceananigans `TabulatedFunction{N}`.
- Introduce explicit P3 table families corresponding to Fortran `lookupTable_1`,
  `lookupTable_2`, and `lookupTable_3`.
- Replace the runtime three-moment closure solve for `μᶦ` with lookup-table access.
- Add the missing 5D `lookupTable_3`-style diagnostics used by three-moment ice.
- Add missing liquid-fraction-aware table families from Fortran `lookupTable_1`
  and `lookupTable_2`.
- Update tests and validation utilities so table content and runtime usage are checked
  against both internal invariants and selected Fortran comparisons.

### Out of scope

- Reproducing the Fortran file formats byte-for-byte.
- Refactoring unrelated microphysics processes outside the table-backed runtime paths.
- Reworking saturation adjustment or non-tabulated warm-rain physics.
- Changing root `Project.toml` dependencies beyond the required Oceananigans compatibility
  update and any minimal compatibility adjustments forced by that upgrade.

## Current Problems

### Fragmented table system

`src/Microphysics/PredictedParticleProperties/tabulation.jl` currently mixes:

- numerical evaluators,
- custom interpolation containers,
- process-specific normalization,
- partial Fortran table emulation,
- and runtime construction helpers.

This makes it difficult to reason about which Fortran tables are implemented,
which dimensions are missing, and which runtime paths still bypass tables.

### Missing three-moment lookup path

Breeze currently solves the three-moment closure iteratively at runtime in
`lambda_solver.jl`. That diverges from the intended design target for this work:
`μᶦ` and related diagnostics should come from a precomputed lookup table corresponding
to the role of Fortran `lookupTable_3`.

### Missing table families

The current implementation covers only part of the Fortran table surface.
Important missing pieces include:

- 5D three-moment diagnostic tables,
- additional liquid-fraction-aware `lookupTable_1` diagnostics,
- and the broader `lookupTable_2` family used for inter-category and rain-size-aware
  collection lookups.

### Wrong abstraction boundary

The current code treats each integral as if it independently owns its tabulation shape.
For the Fortran-inspired system, the correct abstraction is the table family:

- a family defines axes and shared semantics,
- multiple named diagnostics live on that family,
- runtime lookup code consumes physical quantities, not raw interpolation containers.

## Design Overview

The new subsystem is organized into three layers:

1. **Table family specification**
   Typed descriptors define axes, table names, and evaluator functions for
   `lookupTable_1`, `lookupTable_2`, and `lookupTable_3`.
2. **Table storage and interpolation**
   Each table entry is stored as an Oceananigans `TabulatedFunction{N}` or as a small
   Breeze wrapper around `TabulatedFunction{N}` where the family needs grouped access.
3. **Runtime lookup API**
   Process-rate code calls Julia-native accessors such as `shape_parameter_lookup`,
   `three_moment_diagnostics`, or `ice_rain_collection_lookup` instead of directly
   constructing closure solves or indexing raw arrays.

This keeps the interpolation mechanism generic while preserving domain-specific names
and physical meaning at the call sites.

## Table Families

### Family 1: Bulk and process integrals (`lookupTable_1`)

This family covers the existing integral-driven P3 tables plus the missing liquid-fraction
variants that Fortran carries in the same family.

Representative diagnostics include:

- fall speeds: `uns`, `ums`, `uzs`,
- deposition and melting ventilation terms: `vdep`, `vdep1`, `vdepm1` ... `vdepm4`,
- bulk properties: `eff`, `dmm`, `rhomm`, `refl`,
- bulk diagnostics currently treated specially: `lambda_i`, `mu_i_save`, `qshed`,
- collection and sixth-moment auxiliaries: `nagg`, `nrwat`, `m6*`,
- rain-size-dependent ice-rain tables presently handled separately in Breeze.

Not every entry must use the same dimensionality internally, but they should be grouped
under one family API so that the relationship to the Fortran family remains explicit.

### Family 2: Inter-category and rain-size-aware collection (`lookupTable_2`)

This family introduces the missing table set for inter-category ice interactions and the
liquid-fraction variants encoded in the Fortran generator.

The Julia design does not need to preserve the exact Fortran storage layout, but it must
preserve the physical lookup inputs and outputs. The family should expose typed accessors
for the collection quantities the runtime needs, rather than leaking raw table objects.

### Family 3: Three-moment diagnostics (`lookupTable_3`)

This family replaces the runtime three-moment closure solve.

Primary runtime outputs:

- `μᶦ` lookup,
- `λᶦ` lookup or sufficient auxiliary state to recover it cheaply,
- mean density and moment diagnostics needed by the three-moment path,
- any additional lookup quantities currently encoded in the Fortran family and required
  by Breeze's runtime evolution equations.

This family is 5D in the Julia implementation. The core axes are:

1. normalized reflectivity-like coordinate (`Znorm` role),
2. rime density,
3. normalized mass or mean-particle-mass coordinate (`Qnorm` role),
4. rime fraction,
5. liquid fraction.

The exact Julia-native parameterization may differ from the Fortran indexing formulas,
but it must preserve monotonicity, valid bounds, and physical interpretability.

## Axis Strategy

### General rule

Each table family owns its own physical axes. The runtime should not need to know
about table indices or Fortran loop order.

### Uniform axes

Where the current implementation already uses well-behaved uniform ranges, use
Oceananigans `TabulatedFunction{N}` directly.

Examples:

- `log10(mean_particle_mass)`,
- `rime_fraction`,
- `liquid_fraction`,
- `log10(rain_slope_parameter)`.

### Nonuniform or discrete axes

Some Fortran axes are intrinsically discrete or nonuniform, especially the five-point
rime-density set and any family whose native coordinates are defined by Fortran lookup
recipes rather than simple linear ranges.

For these cases Breeze should add a thin wrapper that:

- converts physical inputs into normalized continuous coordinates where possible, or
- performs a small explicit interpolation between a discrete outer axis and delegates the
  remaining interpolation to `TabulatedFunction{N}`.

This keeps the runtime code explicit and avoids overfitting the entire subsystem to the
most awkward axes.

## Runtime API

### New public or internal accessors

Introduce a small set of lookup functions that process-rate code can call directly:

- `shape_parameter_lookup(...)`
- `slope_parameter_lookup(...)`
- `three_moment_diagnostics(...)`
- `ice_integral_lookup(...)`
- `ice_rain_collection_lookup(...)`
- `inter_category_collection_lookup(...)`

These accessors should accept physical quantities already available at the runtime site,
not table indices.

### Removal of runtime three-moment solve

The runtime path that currently calls `solve_shape_parameter` for three-moment ice will be
replaced with table lookup. The old solver should remain available only as:

- a construction-time helper if needed while generating tables,
- a validation reference,
- or a fallback path used only in tests and diagnostics.

It should no longer be on the default simulation hot path for three-moment ice.

### Materialization pattern

To preserve type stability and GPU compatibility, table-bearing microphysics objects should
continue to use the Breeze materialization pattern:

- user-facing constructors can hold skeleton fields,
- `tabulate(...)` or a materialization helper installs concretely typed table families,
- runtime structs must not contain `Any`.

## Data Model Changes

### Ice properties

`IceProperties` currently stores concept containers such as `fall_speed`,
`deposition`, `bulk_properties`, `collection`, and `sixth_moment`.

This should evolve to distinguish between:

- conceptual physics groupings,
- and concrete lookup families.

Recommended direction:

- keep concept groupings for readability at call sites,
- add dedicated fields for table families where runtime access needs shared,
  multi-diagnostic lookup,
- keep the existing concept containers as wrappers around the new family entries where
  that minimizes churn in downstream code.

### Table parameter objects

`TabulationParameters` should be split or expanded so it can express family-specific
resolution cleanly. A single flat parameter bag is no longer enough once 5D and
family-specific axes are introduced.

Recommended direction:

- keep a top-level `P3TabulationParameters`,
- add nested family-specific parameter structs, for example
  `LookupTable1Parameters`, `LookupTable2Parameters`, and `LookupTable3Parameters`.

This makes default choices explicit and keeps future tuning localized.

## File Structure

The current single-file `tabulation.jl` is too broad for the new system.
Split responsibilities into focused files under
`src/Microphysics/PredictedParticleProperties/`.

Recommended structure:

- `tabulation.jl`
  top-level exports, orchestration, and high-level `tabulate` entry points.
- `tabulated_function_adapters.jl`
  Breeze glue around Oceananigans `TabulatedFunction{N}` and any discrete-axis helpers.
- `lookup_table_1.jl`
  family-1 descriptors, evaluators, and runtime accessors.
- `lookup_table_2.jl`
  family-2 descriptors, evaluators, and runtime accessors.
- `lookup_table_3.jl`
  family-3 descriptors, evaluators, and runtime accessors.
- `tabulation_parameters.jl`
  family-specific parameter structs and defaults.
- `table_generation_common.jl`
  shared coordinate transforms, normalization helpers, and construction utilities.

This split is intentionally by responsibility, not by interpolation dimension.

## Oceananigans Upgrade

The design assumes Oceananigans `0.106.2`, which ships the generic
`Oceananigans.Utils.TabulatedFunction` supporting 1D through 5D interpolation.

Compatibility work required:

- update `[compat]` for Oceananigans,
- confirm any API changes outside tabulation that affect Breeze compile or tests,
- replace direct usage of Breeze's custom table structs with the Oceananigans generic type,
- update imports and explicit-import QA expectations accordingly.

## Table Construction Strategy

### Construction-time only expensive work

All expensive quadrature, solver iteration, and diagnostic reconstruction should happen
at table-build time on CPU.

Runtime requirements:

- lookup access must remain allocation-free,
- runtime kernels must not branch into closure construction or CPU-side code,
- table-bearing structs must be transferable across architectures via the existing
  `on_architecture` and `Adapt` patterns.

### LookupTable3 generation

Because `lookupTable_3` replaces the runtime three-moment solve, its construction path may
reuse existing exact or approximate solver routines during table generation.

Construction algorithm:

1. map physical axes to a table coordinate system,
2. reconstruct the three-moment state represented by that point,
3. solve or diagnose `μᶦ`, `λᶦ`, and any companion fields offline,
4. store the resulting diagnostics in 5D tables,
5. validate monotonicity and boundedness of the stored fields.

This preserves the current solver code as a trusted construction-time reference while
moving the simulation hot path to table lookup.

## Testing Strategy

### Unit tests

Add targeted tests for:

- Oceananigans `TabulatedFunction{N}` integration in Breeze,
- family-specific axis transforms,
- interpolation correctness at interior points and clamped boundaries,
- monotonicity and boundedness of `μᶦ`, `λᶦ`, and related diagnostics,
- type stability of table-bearing structs and runtime accessors.

### Regression tests

Update the existing P3 tabulation tests to check:

- table creation still returns fully typed microphysics objects,
- three-moment runtime no longer invokes the iterative closure on the default path,
- table-backed and reference analytical or construction-time-solver results agree within
  specified tolerances over a representative state sample.

### Fortran comparison tests

Extend the validation utilities under `validation/p3/` to compare selected slices from:

- `lookupTable_1`,
- `lookupTable_2`,
- `lookupTable_3`.

These comparisons do not need bitwise parity. They should confirm:

- correct axes and ordering,
- correct limiting behavior,
- and acceptable quantitative agreement for representative points.

### QA and docs

Run at least:

- targeted P3 tests,
- `quality_assurance.jl`,
- any validation script directly affected by the new families.

Documentation should be updated where the old three-moment closure is described as the
runtime path.

## Migration Plan

Implementation should proceed in this order:

1. Upgrade Oceananigans and replace custom table containers with generic
   `TabulatedFunction{N}` wrappers.
2. Split `tabulation.jl` into family-focused files without changing behavior.
3. Introduce `lookupTable_3` family generation and lookup API.
4. Switch the three-moment runtime path from iterative closure solve to table lookup.
5. Add missing `lookupTable_1` and `lookupTable_2` family coverage.
6. Update tests, validation scripts, and documentation.
7. Remove obsolete custom interpolation types and dead closure-on-hot-path logic.

This staging keeps the risk concentrated and makes regressions easier to isolate.

## Risks

### Oceananigans upgrade risk

The version bump may surface unrelated compatibility failures. This should be handled
first so the table rewrite is not debugging against a moving dependency baseline.

### Axis-definition risk

The largest scientific risk is misdefining the Julia-native equivalents of the Fortran
axes, especially for `lookupTable_3`. The implementation should make the coordinate
transforms explicit and test them independently.

### Runtime-path drift

If old and new lookup paths coexist too long, the code will be difficult to trust.
The implementation should converge quickly on one default runtime path and keep the old
solver only as a construction and validation reference.

## Success Criteria

This overhaul is successful when all of the following are true:

- Breeze uses Oceananigans `TabulatedFunction{N}` for P3 tabulation.
- The runtime three-moment `μᶦ` path uses 5D table lookup, not iterative closure solve.
- Missing `lookupTable_1` and `lookupTable_2` family coverage is present in Breeze.
- The table subsystem is split into clear family-based modules rather than one large file.
- Targeted tests and validation scripts pass.
- The new structure makes it easy to answer which Fortran table family each Breeze
  diagnostic comes from and how it is generated.
