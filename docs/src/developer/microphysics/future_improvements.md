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

**Status**: On hold pending velocity field overhaul (see item 5).

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

## 5. Overhaul `microphysical_velocities`

**Issue**: The relationship between `microphysical_velocities` and the velocity fields updated
in `update_microphysical_auxiliaries!` is unclear.

**Current state**:
- `update_microphysical_auxiliaries!` writes velocity values to fields (e.g., `μ.wʳ[i,j,k] = ...`)
- `microphysical_velocities(scheme, μ, name)` returns the velocity field for a given tracer

### Key Insight: Sedimentation is Eulerian-Only

Analysis of parcel models (`pyrcel`, `PySDM`) reveals that:

| Model | Type | Sedimentation handling |
|-------|------|----------------------|
| `pyrcel` | 0D parcel | **None** — droplets stay within parcel |
| `PySDM` (0D) | 0D parcel | Could be mass sink, but typically not used |
| `PySDM` (1D/2D) | Kinematic grid | Particle displacement through spatial mesh |
| Breeze `ParcelModel` | 0D parcel | **None** (currently) |
| Breeze `AtmosphereModel` | Eulerian LES | Tracer advection with terminal velocity |

**Implications**:
- `microphysical_velocities` is fundamentally an **Eulerian concept** — it provides velocities 
  for advecting tracer fields through a spatial grid
- In parcel models, sedimentation should be modeled as a **mass sink term** in 
  `microphysical_tendency`, not as spatial transport
- This means `microphysical_velocities` should remain Eulerian-only and not be part of the 
  minimal parcel interface

### Questions to resolve

1. **Separation of concerns**: Should velocity computation be separated from field writing?
   Currently, both happen in `update_microphysical_auxiliaries!`.

2. **Naming conventions**: Can `microphysical_velocities` be eliminated if velocity fields
   are stored with predictable names based on tracer names?

3. **Multi-moment schemes**: How should schemes handle different velocities for mass vs.
   number concentration (e.g., `wʳ` for rain mass, `wʳₙ` for rain number)?

4. **Advection coupling**: How does the velocity field connect to the advection machinery
   in `AtmosphereModel`?

5. **Parcel precipitation loss**: Should we add a standard pattern for precipitation removal
   in parcel models? This would be a sink term based on threshold size or collection efficiency,
   implemented in `microphysical_tendency`.

**Status**: Needs comprehensive review. This blocks automation of `materialize_microphysical_fields`.

## Summary

| Priority | Item | Status |
|----------|------|--------|
| High | Consolidate state types | Ready to implement |
| Medium | Document SA exception | Ready to implement |
| Medium | Overhaul velocities | Needs design work |
| Low | Automate field materialization | Blocked on velocity overhaul |
| Low | Further function consolidation | Ongoing |

The interface is already well-structured around the gridless state abstraction. The main
remaining complexity is in:
1. Velocity field handling (needs overhaul)
2. Saturation adjustment special cases (needs documentation)
3. Redundant state types (straightforward to fix)
