# MPAS acoustic substepping: complete algorithm for implementation

Self-contained reference for implementing the MPAS split-explicit algorithm in Breeze.
All equations are for **dry air, no terrain, shallow atmosphere** (ζ_z = 1, z_H = 0,
q = 0, cqw = 1). This eliminates all terrain Jacobian and moisture terms.

Sources: MPAS-A tech note v8 (Skamarock et al. 2025), MPAS source code
(`mpas_atm_time_integration.F`), Klemp et al. (2007, 2018).

---

## 1. Constants

```
R_d = 287.0       dry air gas constant [J/(kg·K)]
c_p = 1004.5      heat capacity at constant pressure [J/(kg·K)]
c_v = c_p - R_d   heat capacity at constant volume [J/(kg·K)]
γ = c_p / c_v     ratio of heat capacities ≈ 1.4
κ = R_d / c_p     Poisson constant ≈ 0.2857
rcv = R_d / c_v   ≈ 0.4
c2 = c_p * rcv    = c_p² / c_v = γ R_d ≈ 401.9 [J/(kg·K)]
g = 9.80616       gravitational acceleration [m/s²]
p₀ = 1e5          reference surface pressure [Pa]
pˢᵗ = 1e5         standard pressure for potential temperature [Pa]
```

## 2. Grid (Arakawa C-grid, height coordinate)

Cell centers: k = 1, 2, ..., Nz (k=1 at bottom, k=Nz at top)
Cell faces (w-levels): k = 1, 2, ..., Nz+1 (k=1 = bottom boundary, k=Nz+1 = top)

```
Δz_w(k) = z_face(k+1) - z_face(k)       cell thickness (center-to-center: Δz_u below)
Δz_u(k) = z_center(k) - z_center(k-1)   face spacing (face k between centers k-1 and k)
rdzw(k) = 1 / Δz_w(k)
rdzu(k) = 1 / Δz_u(k)
```

For uniform grid: Δz_w = Δz_u = Δz = H / Nz.

Horizontal: structured lat-lon with spacings Δx(i,j), Δy(j), areas Ax, Ay, V.
Velocities u at x-faces (ᶠᶜᶜ), v at y-faces (ᶜᶠᶜ), w at z-faces (ᶜᶜᶠ).
Scalars (ρ, ρθ, p, Π) at cell centers (ᶜᶜᶜ).

## 3. Base state (1D, function of z only)

Isothermal atmosphere at T₀ = 250K (MPAS default for baroclinic wave):

```
p_base(k) = p₀ exp(-g z(k) / (R_d T₀))
ρ_base(k) = p_base(k) / (R_d T₀)
θ_base(k) = T₀ / Π_base(k)     where Π_base = (p_base / pˢᵗ)^κ
ρθ_base(k) = ρ_base(k) * θ_base(k)
```

**Critically**: Π_base must be computed by **discrete upward integration** to ensure
exact discrete hydrostatic balance:

```
Π_base(1) = (p₀ / pˢᵗ)^κ
Π_base(k) = Π_base(k-1) - g * Δz_u(k) / (c_p * θ₀_face(k))     for k = 2, ..., Nz
```

where `θ₀_face(k) = (θ_base(k-1) + θ_base(k)) / 2`.

This ensures `c_p θ₀_face (Π_base(k) - Π_base(k-1)) / Δz_u = -g` EXACTLY.

Then derive: `p_base(k) = pˢᵗ * Π_base(k)^(1/κ)`, `ρ_base(k) = p_base(k) / (R_d θ_base(k) Π_base(k))`.

Store: `ρ_base(k)`, `ρθ_base(k)`, `Π_base(k)`, `p_base(k)` for all k.

## 4. Perturbation variables

State decomposition:
```
ρ(k)  = ρ_base(k) + ρ_p(k)
ρθ(k) = ρθ_base(k) + ρθ_p(k)
Π(k)  = (p(k) / pˢᵗ)^κ             (full Exner from EOS, NOT decomposed)
p(k)  = pˢᵗ (R_d ρθ(k) / pˢᵗ)^γ    (full EOS pressure)
```

## 5. Linearized perturbation pressure (MPAS `pp`, line 6985)

**This is the most important formula for discrete consistency.**

```
pp(k) = R_d * (Π(k) * ρθ_p(k) + ρθ_base(k) * (Π(k) - Π_base(k)))
```

This is NOT `p(k) - p_base(k)`. It is a specific linearization of the EOS that
ensures exact discrete hydrostatic cancellation with the buoyancy `dpdz`.

**Verification**: For a state in exact discrete hydrostatic balance (matching the
base state), `ρθ_p = 0` and `Π = Π_base`, so `pp = 0` everywhere. The vertical
gradient `(pp(k) - pp(k-1)) / Δz_u = 0`, and the buoyancy `dpdz = -g * ρ_p = 0`.
Total forcing = 0 exactly. No O(Δz²) residual.

## 6. Buoyancy (MPAS `dpdz`, line 5357)

For dry air (q = 0):
```
dpdz(k) = -g * ρ_p(k) = -g * (ρ(k) - ρ_base(k))
```

Note: MPAS uses `rr_save` (the density perturbation SAVED at the beginning of
each RK stage), not the current density. This freezes the buoyancy during substeps.

## 7. Slow w tendency (MPAS `tend_w_euler`, line 5905)

```
tend_w_euler(k) = -rdzu(k) * (pp(k) - pp(k-1)) + (fzm(k)*dpdz(k) + fzp(k)*dpdz(k-1))
```

where `fzm(k) = Δz_w(k-1) / (Δz_w(k-1) + Δz_w(k))` and `fzp(k) = 1 - fzm(k)`
are vertical interpolation weights. For uniform grid: `fzm = fzp = 0.5`.

**Physical meaning**: The first term is the vertical gradient of the linearized
perturbation pressure (PGF). The second term is the face-interpolated buoyancy.
These nearly cancel for a hydrostatically balanced state.

The FULL slow w tendency is:
```
tend_rw(k) = tend_w_advection(k) + tend_w_coriolis(k) + tend_w_euler(k) + tend_w_diffusion(k)
```

For the baroclinic wave test, `tend_w_advection ≈ 0` (weak vertical motion initially),
`tend_w_coriolis ≈ 0` (small for w), and `tend_w_diffusion ≈ 0` (no explicit diffusion
in standard test). So `tend_rw ≈ tend_w_euler`.

## 8. Slow horizontal tendency (MPAS `tend_u_euler`, line 5383)

```
tend_u_euler(k, edge) = -(pp(k,cell2) - pp(k,cell1)) / dcEdge
                        + 0.5 * (dpdz(k,cell1) + dpdz(k,cell2)) * zxu(k,edge)
```

For no terrain: `zxu = 0`, so the second term vanishes:
```
tend_u_euler(k) = -(pp(k,cell2) - pp(k,cell1)) / dcEdge
```

In Breeze structured grid:
```
tend_u_euler(i,j,k) = -(pp(i,j,k) - pp(i-1,j,k)) / Δx(i,j)    [at u-face]
tend_v_euler(i,j,k) = -(pp(i,j,k) - pp(i,j-1,k)) / Δy(j)      [at v-face]
```

**This uses the SAME linearized `pp` as the vertical PGF. Discrete consistency.**

## 9. Off-centering parameters

```
ε = 0.1                           (config_epssm, MPAS default)
α = 0.5 * (1 + ε) = 0.55         (implicit weight, called dtseps/Δτ)
β = 0.5 * (1 - ε) = 0.45         (explicit weight)
resm = (1 - ε) / (1 + ε) = β/α   (old-time ratio ≈ 0.818)
dtseps = α * Δτ = 0.55 * Δτ      (implicit time scale)
```

Time-averaged quantity: `φ̄ = α φ_new + β φ_old = α (φ_new + resm φ_old)`

## 10. Implicit coefficients (MPAS Appendix A, Eqs. A.4-A.8)

All per-unit-dtseps (multiply by `dtseps` or `dtseps²` at runtime):

```
C_Ωz(k) = c2 * Π_face(k) * rdzu(k)                            [at face k]
C_Ωθ(k) = (rcv / 2) * g * ρ_base(k) * Π(k) / (ρθ(k) * Π_base(k))  [at center k]
C_Ωρ(k) = g / 2                                                 [at face k]
C_Θ(k)  = θ_m_face(k) = (θ_m(k-1) + θ_m(k)) / 2              [at face k]
C_ρ(k)  = rdzw(k) = 1 / Δz_w(k)                               [at center k]
```

where:
- `Π_face(k) = (Π(k-1) + Π(k)) / 2` (face-interpolated FULL Exner)
- `Π(k) = (p(k) / pˢᵗ)^κ` (from EOS)
- `ρθ(k)` = current (stage-frozen) potential temperature density
- `θ_m(k) = ρθ(k) / ρ(k)` = potential temperature

MPAS Fortran names: `cofwz(k) = dtseps * C_Ωz(k)`, `cofwt(k) = dtseps * C_Ωθ(k)`,
`cofwr(k) = dtseps * C_Ωρ(k)`, `coftz(k) = dtseps * C_Θ(k)`, `cofrz(k) = dtseps * C_ρ(k)`.

**These are computed ONCE per RK stage** (not per substep) because they depend on
the stage-frozen state.

## 11. Double-prime acoustic perturbation variables

Reset to zero at the start of each RK stage:
```
rw_p(k) = 0         vertical momentum perturbation [at faces]
rtheta_pp(k) = 0    ρθ double-perturbation [at centers]
rho_pp(k) = 0       ρ double-perturbation [at centers]
ru_p(i,j,k) = 0     horizontal momentum perturbation [at u-faces]
rv_p(i,j,k) = 0     horizontal momentum perturbation [at v-faces]
```

## 12. RK3 outer loop

Wicker-Skamarock RK3 with N acoustic substeps per full time step:

```
Stage 1: 1 substep,   Δτ = Δt/3     (total interval = Δt/3)
Stage 2: N/2 substeps, Δτ = Δt/N    (total interval = Δt/2)
Stage 3: N substeps,   Δτ = Δt/N    (total interval = Δt)
```

N must be even. N is chosen so `Δτ < Δx / c_s` (horizontal acoustic CFL).
Rule of thumb: Δt ≈ 5-6 × Δx(km) seconds, N = 6.

At each stage:
1. Reset state to saved values from time level t
2. Compute slow tendencies from current state
3. Run acoustic substep loop
4. Recover full state from base + perturbations

## 13. Acoustic substep algorithm (one substep)

Given: current state (u, v, w, ρθ, ρ, p) and perturbations (rw_p, rtheta_pp, rho_pp, ru_p, rv_p),
slow tendencies (Gˢu, Gˢv, Gˢw, Gˢρ, Gˢρθ), frozen coefficients (cofwz, cofwr, cofwt, coftz, cofrz).

### Step 1: Horizontal forward step (update u, v, ru_p, rv_p)

For substep > 1 (on substep 1, rtheta_pp = 0 so pgrad = 0):
```
pgrad_u(i,j,k) = c2 * Π_u_face * (rtheta_pp(i,j,k) - rtheta_pp(i-1,j,k)) / Δx(i,j)
Π_u_face = (Π(i,j,k) + Π(i-1,j,k)) / 2     [stage-frozen FULL Exner]

du = Δτ * (Gˢu(i,j,k) - pgrad_u / ρ_u_face)     [velocity increment]
u(i,j,k) += du
ru_p(i,j,k) += ρ_u_face * du                      [momentum perturbation]
```

Same for v with y-operators. The sign: `pgrad_u` is positive for a positive ρθ
gradient, and we SUBTRACT it from the tendency (pressure gradient opposes motion).

### Step 2: Save old rtheta_pp for divergence damping
```
rtheta_pp_old(k) = rtheta_pp(k)     [copy before update]
```

### Step 3: Initialize perturbations on first substep
```
if substep == 1:
    rw_p(k) = 0,  rho_pp(k) = 0,  rtheta_pp(k) = 0
```

### Step 4: Compute ts, rs (provisional ρθ and ρ updates)

Horizontal flux divergence from ru_p (area-weighted):
```
mass_div(k) = (Ax_e ru_p(i+1) - Ax_w ru_p(i) + Ay_n rv_p(j+1) - Ay_s rv_p(j)) / V
theta_div(k) = (Ax_e ru_p(i+1) θ_e - Ax_w ru_p(i) θ_w + ...) / V

where θ_e = (θ_m(i,j,k) + θ_m(i+1,j,k)) / 2  [face-interpolated stage-frozen θ]
```

Combine with old perturbation, slow tendency, and off-centered old vertical flux:
```
rs(k) = rho_pp(k) + Δτ * Gˢρ(k) - Δτ * mass_div(k)
        - resm * cofrz(k) * (rw_p(k+1) - rw_p(k))

ts(k) = rtheta_pp(k) + Δτ * Gˢρθ(k) - Δτ * theta_div(k)
        - resm * rdzw(k) * (coftz(k+1) * rw_p(k+1) - coftz(k) * rw_p(k))
```

where `coftz(k) = dtseps * θ_m_face(k)` and `cofrz(k) = dtseps / Δz_w(k)`.

Note: the `rw_p` terms use the OLD (pre-solve) values with weight `resm`.
The NEW values will come from the tridiagonal solve.

### Step 5: Accumulate old-time wwAvg
```
wwAvg(k) += 0.5 * (1 - ε) * rw_p(k)     [before tridiagonal solve]
```

### Step 6: Build RHS for w tridiagonal

```
rw_p(k) = rw_p(k) + Δτ * ρ_face(k) * Gˢw(k)
         - cofwz(k) * ((ts(k) - ts(k-1)) + resm * (rtheta_pp(k) - rtheta_pp(k-1)))
         - cofwr(k) * ((rs(k) + rs(k-1)) + resm * (rho_pp(k) + rho_pp(k-1)))
         + cofwt(k) * (ts(k) + resm * rtheta_pp(k))
         + cofwt(k-1) * (ts(k-1) + resm * rtheta_pp(k-1))
```

Physical meaning of each line:
- Line 1: slow tendency (advection + PGF + buoyancy from Section 7)
- Line 2: vertical PGF from ρθ perturbation (acoustic, uses cofwz)
- Line 3: buoyancy from ρ perturbation (gravity-density coupling, uses cofwr)
- Line 4-5: EOS buoyancy correction from θ perturbation (uses cofwt)

Each term has both:
- NEW-time contribution from `ts(k)`, `rs(k)` (implicit, coefficient 1)
- OLD-time contribution from `rtheta_pp(k)`, `rho_pp(k)` (explicit, coefficient `resm`)

### Step 7: Tridiagonal solve for rw_p

The tridiagonal matrix [a(k), b(k), c(k)] at face k (k=2,...,Nz):

```
a(k) = -cofwz(k) * coftz(k-1) * rdzw(k-1)         [acoustic: PGF × θ flux below]
      + cofwr(k) * cofrz(k-1)                       [gravity × ρ divergence below]
      - cofwt(k-1) * coftz(k-1) * rdzw(k-1)        [EOS correction × θ flux below]

b(k) = 1
      + cofwz(k) * coftz(k) * (rdzw(k) + rdzw(k-1))   [acoustic self-coupling]
      - coftz(k) * (cofwt(k)*rdzw(k) - cofwt(k-1)*rdzw(k-1))  [EOS correction]
      + cofwr(k) * (cofrz(k) - cofrz(k-1))              [gravity self-coupling]

c(k) = -cofwz(k) * coftz(k+1) * rdzw(k)            [acoustic: PGF × θ flux above]
      - cofwr(k) * cofrz(k)                          [gravity × ρ divergence above]
      + cofwt(k) * coftz(k+1) * rdzw(k)             [EOS correction × θ flux above]
```

Boundary conditions: rw_p(1) = 0 (bottom), rw_p(Nz+1) = 0 (top, rigid lid).

Solve by Thomas algorithm:
```
Forward: for k = 2 to Nz:
    m = a(k) / b(k-1)          [but b(1) never used; start at k=2]
    b(k) -= m * c(k-1)
    rw_p(k) -= m * rw_p(k-1)

Backward: rw_p(Nz) = rw_p(Nz) / b(Nz)
    for k = Nz-1 down to 2:
        rw_p(k) = (rw_p(k) - c(k) * rw_p(k+1)) / b(k)
```

MPAS precomputes LU factors `alpha_tri(k) = 1/b(k)` and `gamma_tri(k) = c(k) * alpha_tri(k)`.

**Note on tridiagonal convention**: The Oceananigans `BatchedTridiagonalSolver` uses
a SHIFTED convention where `lower[k]` is the sub-diagonal entry at row k+1 (the
coefficient of x[k] in equation k+1). Verify this when translating.

### Step 8: Rayleigh damping (absorbing layer at model top)
```
rw_p(k) = (rw_p(k) - Δτ R_Ω Ω_save(k)) / (1 + Δτ R_Ω)
```
where `R_Ω = ν sin²(π/2 (z-z_d)/(z_t-z_d))` for z ≥ z_d, zero below.
For tests without a sponge layer: skip this step.

### Step 9: Accumulate new-time wwAvg
```
wwAvg(k) += 0.5 * (1 + ε) * rw_p(k)     [after tridiagonal solve]
```

### Step 10: Update rho_pp and rtheta_pp from new rw_p
```
rho_pp(k) = rs(k) - cofrz(k) * (rw_p(k+1) - rw_p(k))
rtheta_pp(k) = ts(k) - rdzw(k) * (coftz(k+1) * rw_p(k+1) - coftz(k) * rw_p(k))
```

### Step 11: Divergence damping (Klemp et al. 2018)

Applied to horizontal momentum perturbation AFTER each substep:
```
coef = 2 * smdiv * Δx_min / Δτ       [smdiv = 0.1, Δx_min = minimum grid spacing]

divΘ(k) = -(rtheta_pp(k) - rtheta_pp_old(k))    [NEGATIVE of the change]

ru_p(i,j,k) += coef * (divΘ(i,j,k) - divΘ(i-1,j,k)) / (θ_m(i,j,k) + θ_m(i-1,j,k))
rv_p(i,j,k) += coef * (divΘ(i,j,k) - divΘ(i,j-1,k)) / (θ_m(i,j,k) + θ_m(i,j-1,k))
```

Note: denominator is the SUM (θ₁ + θ₂), NOT the average.

### Step 12: Accumulate ruAvg
```
ruAvg(i,j,k) += ru_p(i,j,k)     [simple sum, divided by Nτ at end]
```

## 14. State recovery after acoustic loop

After all Nτ substeps complete:

```
ρθ_p(k) = ρθ_p_save(k) + rtheta_pp(k)
ρ_p(k) = ρ_p_save(k) + rho_pp(k)
ρθ(k) = ρθ_base(k) + ρθ_p(k)
ρ(k) = ρ_base(k) + ρ_p(k)
θ(k) = ρθ(k) / ρ(k)
p(k) = pˢᵗ (R_d ρθ(k) / pˢᵗ)^γ     [full EOS]
```

Time-averaged velocities:
```
u_avg = u_save + ruAvg / (Nτ * ρ_u_face)     [or accumulate velocity directly]
w_avg = w_save + wwAvg / Nτ
```

## 15. Test problem: Jablonowski-Williamson baroclinic wave (DCMIP2016)

### Constants
```
T_E = 310 K,  T_P = 240 K,  T_0 = (T_E + T_P)/2 = 275 K
Γ = 0.005 K/m (lapse rate parameter)
K = 3 (zonal wavenumber)
b = 2 (half-width parameter)
p₀ = 1e5 Pa
a = 6371220 m (Earth radius)
Ω = 7.29212e-5 rad/s (rotation rate)
```

### Temperature and pressure profiles
```
τ₁(z) = exp(Γz/T_m)/T_m + A(1-2η²)exp(-η²)
τ₂(z) = C(1-2η²)exp(-η²)
where η = z/(b H_s), H_s = R_d T_m/g, A = (T_m-T_P)/(T_m T_P), C = (K+2)/2 (T_E-T_P)/(T_E T_P)

F(φ) = cos(φ)^K - K/(K+2) cos(φ)^(K+2)

T(φ,z) = 1 / (τ₁ - τ₂ F(φ))
p(φ,z) = p₀ exp(-g/R_d (∫τ₁ - ∫τ₂ F(φ)))
ρ(φ,z) = p/(R_d T)
θ(φ,z) = T (p₀/p)^κ
```

### Wind
```
U(φ,z) = g/(a K) ∫τ₂ dF/dφ T      (thermal wind)
u_bal(φ,z) = -Ωa cos(φ) + sqrt((Ωa cos(φ))² + a cos(φ) U)

Perturbation: u_p = u_p0 exp(-gc²) taper(z)   at (λ_p, φ_p)
where gc = great_circle_distance / r_p, taper = 1 - 3(z/z_p)² + 2(z/z_p)³
u_p0 = 1 m/s, r_p = 0.1, z_p = 15000 m
```

### Base state for MPAS
```
T₀ = 250 K (isothermal reference, NOT T_m = 275)
p_base(k) = p₀ exp(-g z(k)/(R_d T₀))
ρ_base(k) = p_base(k)/(R_d T₀)
Π_base(k) = discrete integration (Section 3 above)
```

### Grid
```
4° lat-lon: Nλ=90, Nφ=42 (latitude -85° to 85°), Nz=15 or 30, H=30 km
Δt ≈ 720s for 4° (rule: Δt(s) ≈ 5-6 × Δx(km))
N = 6 acoustic substeps → Δτ ≈ 120s
Polar filter at 60° latitude
```

### Expected behavior
- Days 0-5: quiet, initial adjustment only
- Days 7-9: baroclinic instability onset
- Days 10-15: clear Rossby wave pattern, max|θ'| ~ 10-20 K
- max|w| ~ 0.5-3 m/s during instability growth

## 16. Verification checklist

1. **Hydrostatic fixed point**: Initialize with base state (ρ = ρ_base, θ = θ_base,
   u = v = w = 0). After N steps, max|w| should be < 1e-10. If not, the linearized
   pp formula or buoyancy dpdz is discretely inconsistent.

2. **IGW test**: Skamarock-Klemp (1994). 300km × 10km, Δθ = 0.01K perturbation.
   Acoustic substepping at Δt >> Δz/cs should match explicit at Δt = Δz/cs.

3. **Baroclinic wave**: At Δt = 720s (4°), should run stably for 15+ days.
   Compare with MPAS at same resolution.
