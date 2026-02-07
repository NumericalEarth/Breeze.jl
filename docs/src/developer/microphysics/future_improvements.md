# Areas for Future Improvement

This page documents potential improvements to the microphysics interface, serving as a roadmap
for future development.

## 1. Consolidate Redundant State Types

**Issue**: `WarmRainState` (in `microphysics_interface.jl`) and `WarmPhaseOneMomentState`
(in the CloudMicrophysics extension) are nearly identical structs.

**Impact**: Code duplication, potential for inconsistency.

**Recommendation**: Use `WarmRainState` consistently across all warm-rain schemes, or merge
the two types into a single canonical representation.

## 2. Automate `materialize_microphysical_fields`

**Issue**: Each scheme implements `materialize_microphysical_fields` with similar boilerplate:
creating center fields for prognostics, center fields for auxiliaries, and face fields for
sedimentation velocities.

**Potential solution**: Add two new interface functions:

| Function | Returns | Example |
|----------|---------|---------|
| `auxiliary_field_names(microphysics)` | Tuple of diagnostic field names | `(:qᵛ, :qˡ, :qᶜˡ, :qʳ)` |
| `velocity_field_names(microphysics)` | Tuple of velocity field names | `(:wʳ,)` |

Then a generic implementation could handle most cases:

```julia
function materialize_microphysical_fields(microphysics, grid, bcs)
    # Prognostic center fields (with user BCs)
    prog_names = prognostic_field_names(microphysics)
    prog_fields = map(prog_names) do name
        bc = get(bcs, name, nothing)
        CenterField(grid; boundary_conditions=bc)
    end

    # Auxiliary center fields (no BCs needed)
    aux_names = auxiliary_field_names(microphysics)
    aux_fields = center_field_tuple(grid, aux_names...)

    # Velocity face fields (with bottom=nothing for sedimentation)
    vel_names = velocity_field_names(microphysics)
    w_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom=nothing)
    vel_fields = map(n -> ZFaceField(grid; boundary_conditions=w_bcs), vel_names)

    return (; zip(prog_names, prog_fields)...,
              zip(aux_names, aux_fields)...,
              zip(vel_names, vel_fields)...)
end
```

**Complications**:
- Some schemes have unusual fields (e.g., DCMIP2016Kessler's 2D `precipitation_rate`)
- May need an escape hatch for schemes with non-standard requirements

**Status**: Ready to implement (velocity field overhaul is complete, see item 5).

## 3. Reduce Number of Interface Functions

**Issue**: The interface has ~12 functions, some of which may be redundant or could be combined.

## 4. Document the Saturation Adjustment Exception

**Issue**: Saturation adjustment schemes have a fundamentally different structure:

| Aspect | Non-equilibrium schemes | Saturation adjustment schemes |
|--------|------------------------|------------------------------|
| Cloud condensate | Prognostic (evolved in time) | Diagnostic (computed from `𝒰`) |
| `grid_moisture_fractions` | Uses generic wrapper | Must override to read diagnostic fields |
| `maybe_adjust_thermodynamic_state` | Returns `𝒰` unchanged | Performs iterative adjustment |

**Recommendation**: Add clear documentation explaining:
1. Why SA schemes are structurally different
2. Which functions SA schemes must override
3. How moisture fraction computation differs

## 5. Overhaul `microphysical_velocities` — **Completed**

**Resolution**: The `sedimentation_speed` interface now provides a clean separation of concerns.

### What changed

- **`sedimentation_speed(microphysics, microphysical_fields, name)`** is the primary developer
  interface. Schemes return a positive sedimentation speed field (or `nothing`) for each tracer.
  This replaces the old `microphysical_velocities` as the function schemes must implement.
- **`microphysical_velocities`** is now a generic wrapper that calls `sedimentation_speed` and
  constructs a `(u=ZeroField(), v=ZeroField(), w=NegatedField(fs))` tuple for the advection
  operator. Scheme developers no longer override this function.
- **`total_water_sedimentation_speed_components(microphysics, microphysical_fields)`** returns
  `(speed_field, humidity_field)` pairs used to compute the aggregate total water sedimentation
  speed.
- **`model.bulk_sedimentation_velocities`** stores precomputed aggregate sedimentation velocities
  (currently just `ρqᵗ`), updated during `update_state!` via
  `update_bulk_sedimentation_velocities!`.

### Answers to previously-open questions

1. **Separation of concerns**: Yes — velocity *computation* happens in
   `update_microphysical_auxiliaries!` (which writes sedimentation speed values to `ZFaceField`s),
   while velocity *retrieval* happens via `sedimentation_speed` (which returns those fields).

2. **Naming conventions**: `microphysical_velocities` is retained as a generic wrapper, not
   eliminated. Schemes implement `sedimentation_speed` which returns the appropriate field by name.

3. **Multi-moment schemes**: Each tracer gets its own `sedimentation_speed` dispatch. For example,
   in the 2M scheme: `sedimentation_speed(bμp, μ, Val(:ρqʳ))` returns the mass-weighted rain
   sedimentation speed `μ.wʳ`, while `sedimentation_speed(bμp, μ, Val(:ρnʳ))` returns the
   number-weighted sedimentation speed `μ.wʳₙ`.

4. **Advection coupling**: For individual tracers, `microphysical_velocities` (wrapping
   `sedimentation_speed`) provides the velocity tuple added to bulk flow. For total moisture
   (`ρqᵗ`), the precomputed `model.bulk_sedimentation_velocities.ρqᵗ` is used directly.

5. **Parcel precipitation loss**: This remains an open question for future work. The insight that
   sedimentation is Eulerian-only is preserved — `sedimentation_speed` is Eulerian-only.

## Summary

| Priority | Item | Status |
|----------|------|--------|
| High | Consolidate state types | Ready to implement |
| Medium | Document SA exception | Ready to implement |
| ~~Medium~~ | ~~Overhaul velocities~~ | **Completed** (`sedimentation_speed` interface) |
| Low | Automate field materialization | Ready to implement |
| Low | Further function consolidation | Ongoing |

The interface is already well-structured around the gridless state abstraction. The main
remaining complexity is in:
1. Saturation adjustment special cases (needs documentation)
2. Redundant state types (straightforward to fix)
3. Automating field materialization (no longer blocked)
