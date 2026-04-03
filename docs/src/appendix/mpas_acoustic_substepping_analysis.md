# MPAS acoustic substepping analysis

Notes from studying `/tmp/hevi-codes/MPAS-Model/src/core_atmosphere/dynamics/mpas_atm_time_integration.F`
to guide development of Breeze's split-explicit and HEVI time stepping.

## MPAS algorithm structure

MPAS uses a 3-stage Runge-Kutta (RK3) outer loop with acoustic substeps inside each stage.
This is NOT an IMEX-ARK method — the outer RK3 evaluates the FULL tendency (no fᴱ/fᴵ split).

```
for rk_step = 1:3
    1. Compute slow tendencies (full PGF + buoyancy + advection + Coriolis)
    2. Convert momentum tendencies to velocity tendencies (divide by ρ)
    3. Initialize acoustic perturbations: rtheta_pp = 0, rho_pp = 0, rw_p = 0
    4. for small_step = 1:N_substeps
         Forward-backward acoustic update of (u, v, w, Π', ρ', ρθ')
       end
    5. Recover large-step variables from base state + accumulated perturbations
end
```

## Base state

- **Fixed 1D profile**: `rb(k)`, `rtb(k)`, `pb(k)` (rho_base, rtheta_base, exner_base)
- Set once at initialization, NEVER updated during time integration
- Stored as 3D arrays (nVertLevels × nCells) but same profile everywhere horizontally

## Slow w tendency (line 5905-5907)

```fortran
tend_w_euler(k) = tend_w_euler(k) - cqw(k) * (
    rdzu(k) * (pp(k) - pp(k-1))                              ! vertical PGF (perturbation Exner)
  - (fzm(k) * dpdz(k) + fzp(k) * dpdz(k-1))                ! buoyancy
)
```

where:
- `pp` = full Exner function (not perturbation!)
- `dpdz` = buoyancy = `-gravity * (rb*qtot + rr_save*(1+qtot))` (line 5357)
- `rr_save` = density perturbation from base state

The slow tendency uses the FULL Exner pressure gradient (not perturbation).
The buoyancy is relative to the FIXED base state density `rb`.

## Acoustic perturbation variables

Reset to zero at the start of each RK stage (line 2850-2860):
```fortran
if (small_step == 1) then
    rho_pp(k) = 0
    rtheta_pp(k) = 0
    rw_p(k) = 0
end if
```

These measure the acoustic response WITHIN one RK stage. They're relative to the
current RK stage state, NOT the fixed base state. Two levels of perturbation:
1. Slow: ρ' = ρ - ρ_base (relative to fixed 1D base) — used for buoyancy
2. Fast: ρ_pp (acoustic) — reset each RK stage, always small

## Acoustic w update (lines 2907-2915)

```fortran
rw_p(k) = rw_p(k) + dts * tend_rw(k)
         - cofwz(k) * [(zz(k)*ts(k) - zz(k-1)*ts(k-1))
                      + resm*(zz(k)*rtheta_pp(k) - zz(k-1)*rtheta_pp(k-1))]
         - cofwr(k) * [(rs(k) + rs(k-1)) + resm*(rho_pp(k) + rho_pp(k-1))]
         + cofwt(k) * (ts(k) + resm*rtheta_pp(k))
         + cofwt(k-1) * (ts(k-1) + resm*rtheta_pp(k-1))
```

where:
- `cofwz` = perturbation Exner PGF coefficient (acoustic)
- `cofwr` = gravity on perturbation density (gravity-density coupling)
- `cofwt` = vertical buoyancy coupling (temperature advection feedback)
- `resm = (1-epssm)/(1+epssm)` with `epssm ≈ 0.1` (off-centering)
- `ts`, `rs` = accumulated θ' and ρ' horizontal flux contributions

## Key differences from IMEX-ARK

| Aspect | MPAS (split-explicit) | Breeze IMEX-ARK |
|--------|----------------------|-----------------|
| Outer step | Full tendency, uniform Butcher weights | fᴱ + fᴵ with DIFFERENT weights |
| Inner solve | Forward-backward substeps (many small) | Single Helmholtz (linearized) |
| Perturbation base | Current RK stage state (reset each stage) | Fixed reference state |
| Density update | Full 3D divergence in outer RK | Split: div_xy (explicit) + div_z (implicit) |
| Buoyancy | In slow tendency, uniform weight | In fᴱ, different weight from fᴵ |

## The fundamental IMEX-ARK problem

When fᴱ and fᴵ are large and nearly cancel (e.g., buoyancy ≈ -PGF in hydrostatic balance),
the IMEX predictor amplifies the residual through different Butcher coefficients:

```
z₂* = yₙ + h·[aᴱ₂₁·fᴱ₁ + aᴵ₂₁·fᴵ₁]

Error ∝ |aᴱ - aᴵ| × |fᴱ| ≈ 0.586 × |buoyancy|
```

MPAS avoids this because the outer RK3 applies buoyancy + PGF as a SINGLE slow tendency
with uniform weights. There is no differential weighting of cancelling terms.

## Implications for Breeze

### For split-explicit substepping (SplitExplicitTimeDiscretization)

Breeze's acoustic substepping should follow MPAS more closely:
1. The slow tendency should include FULL PGF + buoyancy (currently does this)
2. The acoustic loop should use perturbation variables relative to the CURRENT RK stage
   (currently uses perturbation relative to a fixed 1D ExnerReferenceState — this is
   the source of the baroclinic wave instability)
3. A latitude-dependent reference state or per-stage base state could fix the issue

### For HEVI (VerticallyImplicitTimeStepping)

The IMEX-ARK approach has a structural limitation: the Butcher coefficient mismatch
amplifies errors proportional to |fᴱ| when fᴱ and fᴵ nearly cancel. The error is
small when perturbations from the reference are small (IGW test works). For baroclinic
waves with large meridional gradients, the error grows unless the reference state
captures the horizontal temperature structure.

Options:
1. Use a latitude-dependent ExnerReferenceState matching the IC's θ(φ,z)
2. Reformulate to avoid the fᴱ-fᴵ cancellation (not straightforward with IMEX-ARK)
3. Use split-explicit substepping instead of IMEX-ARK (different algorithm, avoids the issue)

## MPAS implicit coefficients in the acoustic step

For reference, the vertically implicit coefficients (lines 2302-2324):
- `cofwr(k) = 0.5*dtseps*gravity*(fzm(k)*zz(k) + fzp(k)*zz(k-1))` — gravity on ρ'
- `cofwz(k) = dtseps*c2*(...)` — Exner PGF on ρθ'
- `cofwt(k) = 0.5*dtseps*rcv*zz(k)*gravity*rb(k)/(1+qtotal)*p(k)/((rtb(k)+rt(k))*pb(k))` — buoyancy feedback

where `c2 = cp*rcv`, `rcv = rgas/(cp-rgas)`, `dtseps = 0.5*dts*(1+epssm)`.

The `cofwt` term (buoyancy feedback = gravity wave coupling) appears in MPAS but NOT in
our Helmholtz. This term stabilizes the vertical gravity wave mode in the implicit solve.
It could be important for HEVI methods that treat gravity waves semi-implicitly.
