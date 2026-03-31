# VITS density fix: implicit vertical continuity flux

## Problem

The VITS solver zeros the vertical PGF and buoyancy from the explicit ρw tendency
and restores them via the Helmholtz solve. But the acoustic mode is a **three-way
coupling**: w ↔ p ↔ ρ. Without implicit treatment of the vertical density flux
in the continuity equation, the density responds explicitly to vertical mass flux,
leaving the 3D acoustic mode unstable.

Gardner et al. (2018, GMD) identify the **minimum** implicit terms for HEVI:
1. Vertical PGF + buoyancy in ρw equation ← already done
2. Vertical density flux in continuity equation ← **missing**

The current VITS is worse than explicit at every Δt because the Helmholtz solve
corrects ρw and ρθ but ρ doesn't know about it, so the equation of state
sees inconsistent (ρ, ρθ) and the pressure field is wrong.

## What needs to change

### 1. Split the density tendency

The current density tendency kernel computes the full 3D divergence:

```julia
# compressible_density_tendency.jl
@kernel function _compute_density_tendency!(Gρ, grid, momentum)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρ[i, j, k] = - divᶜᶜᶜ(i, j, k, grid, momentum.ρu, momentum.ρv, momentum.ρw)
end
```

For VITS, the vertical part must be excluded from the explicit tendency:

```julia
@kernel function _compute_density_tendency!(Gρ, grid, momentum,
                                             ::VerticallyImplicitTimeStepping)
    i, j, k = @index(Global, NTuple)
    # Only horizontal divergence — vertical is handled implicitly
    @inbounds Gρ[i, j, k] = - div_xyᶜᶜᶜ(i, j, k, grid, momentum.ρu, momentum.ρv)
end
```

where `div_xyᶜᶜᶜ` is the horizontal-only divergence (already exists in Oceananigans).

This requires passing the time discretization to the density tendency kernel.
The cleanest way: dispatch `compute_dynamics_tendency!` on the dynamics type.

### 2. Add ρ update after the Helmholtz solve

After the implicit solve gives corrected ρw⁺, update ρ from the vertical
divergence of ρw⁺:

```julia
## In _vertical_acoustic_implicit_step!:

## 5. Update ρ from vertical divergence of ρw⁺
launch!(arch, grid, :xyz, _update_density_from_ρw!,
        ρ, grid, αΔt, ρw)
```

where:

```julia
@kernel function _update_density_from_ρw!(ρ, grid, αΔt, ρw)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)
    @inbounds begin
        ## Vertical divergence of ρw at cell center: (1/V) δz(Az ρw)
        ## For uniform Δz: (ρw[k+1] - ρw[k]) / Δz
        ρw_top = ρw[i, j, k + 1] * (k < Nz)
        ρw_bot = ρw[i, j, k] * (k > 1)
        Δzᶜ = Δzᶜᶜᶜ(i, j, k, grid)

        ρ[i, j, k] = ρ[i, j, k] - αΔt * (ρw_top - ρw_bot) / Δzᶜ
    end
end
```

Note: this is the vertical divergence of the **full** ρw⁺, not just δρw.
This works because the explicit step computed only the horizontal divergence
(step 1 above), so the vertical part was not yet applied. The implicit step
applies the full vertical divergence of the corrected ρw⁺.

### 3. Consistency check

After the implicit step, the state should satisfy:
- ρw⁺: corrected by back-solve from δρθ
- ρθ⁺: corrected by Helmholtz solve
- ρ⁺: corrected by vertical divergence of ρw⁺
- θ⁺ = ρθ⁺ / ρ⁺: consistent

The pressure is then diagnosed from (ρ⁺, θ⁺) via the equation of state.
If ρ and ρθ are both updated consistently, the pressure will be physical.

## Implementation checklist

- [ ] Add time discretization dispatch to `compute_dynamics_tendency!`
- [ ] For VITS: use `div_xyᶜᶜᶜ` (horizontal only) instead of `divᶜᶜᶜ` (full 3D)
- [ ] Add `_update_density_from_ρw!` kernel to `vertical_implicit_solver.jl`
- [ ] Call it as step 5 in `_vertical_acoustic_implicit_step!`
- [ ] Test on CPU: VITS at Δt=2s should match explicit
- [ ] Test on CPU: VITS at Δt=40s should be stable
- [ ] Test on GPU: same
- [ ] Run stability sweep: compare VITS vs explicit at Δt = 2, 5, 10, 20, 40s

## Lesson from first attempt

Simply adding the ρ update without also splitting the ρθ vertical advection
causes ρ to go negative immediately. The problem:

- Explicit step: ρ sees only horizontal divergence, ρθ sees **full** 3D advection
- After the explicit substep, ρ and ρθ are **inconsistent** (ρθ was advected
  vertically but ρ was not)
- The implicit solve tries to fix the ρw-ρθ-ρ coupling but overcorrects
  because ρθ already moved vertically while ρ didn't

The fix requires splitting ALL vertically-advected quantities consistently.
Following Gardner et al. (2018), splitting C (their recommended minimum):

**Explicit:**
- Horizontal advection of all quantities (ρu, ρv, ρθ, ρ)
- Horizontal PGF
- Coriolis, metric terms
- Vertical advection of ρθ (splitting C keeps this explicit)
- Vertical advection of horizontal momentum

**Implicit:**
- Vertical PGF + buoyancy in ρw
- Vertical density flux in continuity: `∂(ρw)/∂z`

With splitting C, the ρθ equation stays fully explicit (no vertical splitting
needed). Only ρ and ρw are modified:

- ρ tendency: horizontal divergence only (exclude vertical `∂(ρw)/∂z`)
- ρw tendency: exclude vertical PGF + buoyancy (already done)
- After implicit solve: update ρ += -αΔt ∂(ρw⁺)/∂z

But this means ρ and ρθ evolve on different "clocks" for the vertical —
ρ gets the implicit ρw correction but ρθ doesn't. This is OK because
the Helmholtz solve accounts for the ρθ response through the linearized
coupling, and the back-solve corrects ρw for the ρθ change.

The key is: update ρ AFTER ρw is corrected, using only the vertical
divergence of the corrected ρw. Don't also correct ρθ — the Helmholtz
already handles that.

## Expected outcome

With both ρw and ρ treated implicitly, the vertical acoustic CFL should be
removed. The stability limit becomes the horizontal acoustic CFL:
- At 2° / 85°: Δx ≈ 19 km → Δt < 56s
- At 2° / equator: Δx ≈ 222 km → Δt < 653s
- With polar filter at 60°: Δx_eff ≈ 111 km → Δt < 326s

Target: Δt ≈ 40-50s at 2° without polar filter, ~200s with polar filter.
