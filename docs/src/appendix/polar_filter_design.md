# Polar filter design for `LatitudeLongitudeGrid`

## Problem

On a latitude-longitude grid, the zonal grid spacing shrinks with latitude:

```
Δx(φ) = a cos(φ) Δλ
```

At 85° latitude, Δx is ~11× smaller than at the equator. This causes:

1. **CFL bottleneck**: The acoustic time step is limited by the smallest Δx anywhere on the grid. At 2° resolution, Δx ≈ 17 km at 85° forces Δt ≈ 50s (acoustic), while at the equator Δx ≈ 220 km would allow Δt ≈ 650s.

2. **Computational modes**: The zonal grid can represent features with wavelength 2Δx(φ), but the meridional grid can only resolve features with wavelength 2Δy. Near the poles, 2Δx ≪ 2Δy, so the zonal direction over-resolves — features that look fine zonally are unresolvable meridionally. These are computational noise, not physical.

## How WRF's polar filter works

WRF applies an FFT-based filter at each time step for latitudes poleward of a threshold φ_c (typically ~60°):

```
For each latitude row φ where cos(φ) < cos(φ_c):
  1. FFT the field in the zonal direction
  2. Zero out wavenumbers k > k_max(φ)
  3. Inverse FFT back to physical space
```

where the cutoff wavenumber is:

```
k_max(φ) = floor(N_λ × cos(φ) / cos(φ_c))
```

This ensures the effective zonal resolution matches the meridional resolution everywhere. At the threshold latitude, no filtering occurs; at the pole, almost everything is filtered out.

WRF applies this to all prognostic fields (u, v, w, θ, moisture) after each Runge-Kutta stage.

## Proposed Breeze implementation

### User API

```julia
simulation = Simulation(model; Δt, stop_time)
add_polar_filter!(simulation;
                  threshold_latitude = 60,  # degrees, default
                  rolloff = :smooth,        # or :sharp
                  fields = :prognostic)     # or specify which fields
```

This adds a callback that runs after each time step (or after each RK stage if we want tighter control).

### Implementation as a callback

```julia
struct PolarFilter{P, G}
    threshold_latitude :: Float64
    fft_plans :: P       # Pre-allocated 1D FFT plans (one per filtered row)
    grid :: G
end

function PolarFilter(grid; threshold_latitude=60)
    # Identify which j-indices need filtering
    # Pre-allocate FFT plans for those rows
    # On GPU: use batched CUFFT for all filtered rows at once
end

function (filter::PolarFilter)(simulation)
    model = simulation.model
    for field in prognostic_fields(model)
        apply_polar_filter!(field, filter)
    end
end

function apply_polar_filter!(field, filter)
    grid = filter.grid
    Nλ = size(grid, 1)

    for j in filtered_indices(filter)
        φ = latitude(grid, j)
        k_max = floor(Int, Nλ * cosd(φ) / cosd(filter.threshold_latitude))

        for k in 1:size(grid, 3)
            row = view(interior(field), :, j, k)
            # FFT → truncate → IFFT
            f̂ = filter.fft_plans[j] * row
            f̂[k_max+1:end] .= 0
            row .= filter.fft_plans[j] \ f̂
        end
    end
end
```

### GPU considerations

On GPU, the key optimization is batching:

1. Collect all rows that need filtering into a 2D array (N_λ × N_filtered_rows)
2. Apply a single batched 1D FFT (CUFFT handles this efficiently)
3. Zero out high wavenumbers with a kernel
4. Inverse FFT
5. Scatter back to the field

This avoids launching many small FFTs and keeps the GPU busy.

```julia
# Pseudocode for GPU version
filtered_rows = gather_filtered_rows(field, filter)  # Nλ × N_filtered
fft!(filtered_rows, 1)                                # batched FFT along dim 1
zero_high_wavenumbers!(filtered_rows, k_max_per_row)  # single kernel
ifft!(filtered_rows, 1)
scatter_filtered_rows!(field, filtered_rows, filter)
```

### Sharp vs smooth rolloff

Sharp truncation (zero modes above k_max) causes Gibbs ringing. A smooth rolloff is better:

```
filter(k) = exp(-(k / k_max)^p)  for some power p (e.g., p=8 for steep rolloff)
```

or an exponential filter:

```
filter(k) = exp(-α (k / k_max)^2s)  where s is the filter order
```

WRF uses sharp truncation; NCAR's spectral models use smooth filters.

### What fields to filter

Options:
- **All prognostics** (u, v, w, ρ, ρθ, ρqᵗ): safest, what WRF does
- **Tendencies only**: less dissipative, but may not suppress existing noise
- **Velocity only**: might be enough if density/θ noise is driven by velocity

### When to apply

Options:
- **After each RK stage**: tightest control, but 3× more FFTs per step
- **After each full time step**: simpler, less FFT work, may allow 1-step growth of noise
- **WRF approach**: after each RK stage

For the callback API, "after each time step" is the natural choice. For "after each RK stage", we'd need to hook into the time stepper internals, which is more invasive.

### Effect on CFL

The polar filter does NOT directly change the CFL constraint — the acoustic waves still propagate at the physical grid spacing. However:

1. By removing the high-wavenumber modes that actually trigger the CFL violation, the simulation becomes *effectively* stable at larger Δt
2. WRF combines the polar filter with a reduced effective grid spacing for CFL computation
3. A more aggressive approach: compute CFL using the *filtered* Δx rather than the physical Δx

### Estimated effort

- **Minimal CPU version**: 1 day (FFT filtering in a callback, sharp truncation)
- **GPU version with batched FFTs**: 2-3 days (gather/scatter, CUFFT batched plans)
- **Smooth rolloff + tests**: +1 day
- **Integration with CFL computation**: +1 day
- **Total**: ~1 week for a production-quality implementation

### Dependencies

- `FFTW.jl` (CPU) — already a transitive dependency via Oceananigans
- `CUDA.CUFFT` (GPU) — already available via CUDA.jl

### References

- Skamarock, W.C., J.B. Klemp, M.G. Duda, L.D. Fowler, S.-H. Park, and T.D. Ringler, 2012: A multiscale nonhydrostatic atmospheric model using centroidal Voronoi tesselations and C-grid staggering. MWR, 140, 3090-3105.
- WRF Technical Note v4, Section 2.5: "Polar filtering"
- Park, S.-H., W.C. Skamarock, J.B. Klemp, L.D. Fowler, and M.G. Duda, 2013: Evaluation of global atmospheric solvers. MWR, 141, 3116-3129.
