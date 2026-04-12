# Breeze.jl Reactant Correctness Tests — Implementation Plan

## Overview

Compare Reactant-compiled model output against vanilla CPU output, field-by-field,
to catch any divergence introduced by the Reactant compilation path.

Two test scripts:

1. **Serial correctness test** — single-device (CPU or GPU Reactant vs vanilla CPU)
2. **Sharded correctness test** — multi-device Reactant (distributed/sharded) vs vanilla CPU

## Target model config

- Grid: `LatitudeLongitudeGrid`, **16×8×8** (Nλ=16, Nφ=8, Nz=8), small for CI
- Halo: (8, 8, 8)
- Longitude: [0, 360], Latitude: [-80, 80], z: [0, 30 km]
- Advection: WENO(order=5), bounds-preserving WENO(order=5, bounds=(0,1)) for moisture tracers
- Microphysics: `OneMomentCloudMicrophysics` (1M, from `BreezeCloudMicrophysicsExt`), `NonEquilibriumCloudFormation(nothing, nothing)`
- Coriolis: `SphericalCoriolis()`
- Dynamics: `CompressibleDynamics(ExplicitTimeStepping())`
- ICs: analytic DCMIP-2016 moist baroclinic wave
- Δt: 1e-9 (tiny so FP differences stay small)
- No surface fluxes initially (add later)

### Sharded config

- Partition into 2 or 4 processes (e.g. `Rx=2, Ry=1` for 2 devices)
- Per-device grid-x=16 → `Tλ = 16 * Rx`, `Nλ = Tλ - 2H`
- Vanilla side: single-process CPU at the same total grid

## New files to create

| file | purpose |
|------|---------|
| `test/correctness_utils.jl` | Utility functions (not auto-discovered — no `@testset`) |
| `test/dcmip2016_baroclinic_wave_setup.jl` | DCMIP-2016 constants, IC functions, IC kernels, `build_correctness_model(arch; ...)` |
| `test/reactant_serial_correctness.jl` | Serial test: `ReactantState()` vs `CPU()`, 3 stages |
| `test/reactant_sharded_correctness.jl` | Sharded test: `Distributed(ReactantState())` vs `CPU()`, gated on device count |

## Comparison approach

- After construction: compare ICs
- `breeze_sync_states!` (copy vanilla → Reactant) to eliminate any IC drift
- `breeze_zero_tendencies!` on both models
- After `first_time_step!`: compare
- After N `time_step!` calls: compare
- Use `isapprox` with `rtol = 2√ε` (Float32) or similar
- Report per-field max|δ| and pass/fail

## `test/correctness_utils.jl`

Port from GordonBell25 `src/correctness.jl`. Key functions:

### `compare_interior(name, f1, f2; rtol, atol)`

- Convert both to `Array(interior(...))`
- Compute `δ = f1 .- f2`, `findmax(abs, δ)`
- `isapprox(f1, f2; rtol, atol)`
- Print diagnostic line: `(name) ψ₁ ≈ ψ₂: true/false, max|ψ₁|, max|ψ₂|, max|δ| at i j k`
- Return `Bool`

### `compare_parent(name, f1, f2; rtol, atol)`

- Same but on `parent(...)`, cropped to common overlap via `map(min, sz1, sz2)`
- Use the Oceananigans `reactant_correctness_utils.jl` version as base (slightly cleaner)

### `breeze_compare_states(m1, m2; rtol, atol, include_halos, throw_error)`

- Iterate over `Oceananigans.fields(model)` — compare each field
- Also compare `Gⁿ` tendencies (`model.timestepper.Gⁿ`)
- **No** `G⁻` (SSPRungeKutta3 has only `Gⁿ` and `U⁰` workspace)
- **No** `free_surface` or closure comparison (AtmosphereModel doesn't have these)
- Print summary: "consistent" or "discrepancy"
- Return `Bool` (for use with `@test`)

### `breeze_sync_states!(reactant_model, vanilla_model)`

- For each field: copy `parent(vanilla_field)` → `parent(reactant_field)`, cropped to common size
- Direction: CPU → Reactant

### `breeze_zero_tendencies!(model)`

- `parent(model.timestepper.Gⁿ[name]) .= 0` for each tendency field
- SSPRungeKutta3 only: no `G⁻`

## `test/dcmip2016_baroclinic_wave_setup.jl`

Port from GordonBell25 `src/moist_baroclinic_wave_model.jl`:

- Physical constants: `earth_radius`, `gravity`, `Rd_dry`, `cp_dry`, `p_ref`, etc.
- Balanced-state helpers: `vertical_structure`, `F_temperature`, `F_wind`, etc.
- IC functions: `initial_theta`, `initial_density`, `initial_zonal_wind`, `initial_moisture`, `theta_reference`
- IC kernels: `_set_moist_baroclinic_wave_kernel!`, `_set_zonal_wind_kernel!`, `set_moist_baroclinic_wave!`
- `build_correctness_model(arch; Nλ=16, Nφ=8, Nz=8, H=30e3, Δt=1e-9, halo=(8,8,8))`

## `test/reactant_serial_correctness.jl`

```julia
include("correctness_utils.jl")
include("dcmip2016_baroclinic_wave_setup.jl")

@testset "Reactant serial correctness — CompressibleDynamics" begin
    FT = Float64
    rtol = 2 * sqrt(eps(FT))
    atol = 0

    rmodel = build_correctness_model(ReactantState())
    vmodel = build_correctness_model(CPU())

    @testset "Initial conditions" begin
        @test breeze_compare_states(rmodel, vmodel; rtol, atol, include_halos=true)
    end

    breeze_sync_states!(rmodel, vmodel)
    breeze_zero_tendencies!(rmodel)
    breeze_zero_tendencies!(vmodel)

    # Compile and run first time step
    rfirst! = @compile sync=true raise=true first_time_step!(rmodel)
    rfirst!(rmodel)
    first_time_step!(vmodel)

    @testset "After first time step" begin
        @test breeze_compare_states(rmodel, vmodel; rtol, atol)
    end

    # Compile time_step!, run 10 steps on each side
    rstep! = @compile sync=true raise=true time_step!(rmodel)
    Nt = 10
    for _ in 1:Nt; rstep!(rmodel); end
    for _ in 1:Nt; time_step!(vmodel); end

    @testset "After $Nt time steps" begin
        @test breeze_compare_states(rmodel, vmodel; rtol, atol)
    end
end
```

## `test/reactant_sharded_correctness.jl`

Same structure but with:

- MPI initialization / `Reactant.Distributed.initialize()`
- `Rx, Ry` derived from `length(Reactant.devices())`
- `arch = Distributed(ReactantState(); partition=Partition(Rx, Ry, 1))`
- Grid: `Tλ = grid_x * Rx`, `Nλ = Tλ - 2H`
- Gated with env var or excluded from standard CI

## Changes to `test/runtests.jl`

Add to the `REACTANT_COMPAT` exclusion block:

```julia
if !REACTANT_COMPAT
    delete!(testsuite, "reactant_centered_compilation")
    delete!(testsuite, "reactant_weno_compilation")
    delete!(testsuite, "reactant_serial_correctness")       # NEW
end
# Always exclude sharded (requires multi-device)
delete!(testsuite, "reactant_sharded_correctness")          # NEW
```

## Gotchas

1. **`OneMomentCloudMicrophysics` is in an extension** — load via `Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)`, requires `using CloudMicrophysics` first
2. **Reactant parent arrays may be padded** — `compare_parent` must crop to common overlap
3. **`model.clock.last_Δt` must be set** before first step (SSP-RK3 reads it from traced path)
4. **`update_state!` must be called** before first step — `BreezeReactantExt.first_time_step!` handles this
5. **`Oceananigans.fields(model)` returns diagnostics too** — use `prognostic_fields` for primary check, diagnostics as secondary
6. **IC kernels use `@kernel`/`@index`** — work with Reactant when compiled via `@compile`
7. **Sharded grid size accounting**: user provides per-device total (interior + halo), global interior = `per_device * R - 2H`

## Implementation order

1. `test/correctness_utils.jl`
2. `test/dcmip2016_baroclinic_wave_setup.jl`
3. `test/reactant_serial_correctness.jl`
4. Update `test/runtests.jl`
5. `test/reactant_sharded_correctness.jl` (can be deferred)

## Source references

| component | source file |
|-----------|------------|
| compare functions | `GB-25/src/correctness.jl` |
| IC functions + model builder | `GB-25/src/moist_baroclinic_wave_model.jl` |
| time-stepping wrappers | `GB-25/src/timestepping_utils.jl` |
| serial correctness example | `GB-25/correctness/correctness_atmosphere_simulation_run.jl` |
| sharded correctness example | `GB-25/correctness/correctness_sharded_atmosphere_simulation_run.jl` |
| Oceananigans compare utils | `Oceananigans.jl/test/reactant_correctness_utils.jl` |
| Breeze Reactant tests (pattern) | `Breeze.jl/test/reactant_centered_compilation.jl` |
