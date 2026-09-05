# Single column mode

Breeze can run an [`AtmosphereModel`](@ref) as a **single column** — or as a *forest* of many
independent columns advanced concurrently — on a grid with `topology = (Flat, Flat, Bounded)`.
This mirrors the single-column / column-ensemble mode of
`Oceananigans.HydrostaticFreeSurfaceModel`, and is designed for ensembles: parameter sweeps,
turbulence-closure calibration, and boundary-layer scheme development.

## What single column mode does

On a `(Flat, Flat, Bounded)` grid every horizontal finite-difference operator returns zero, so
horizontal advection, diffusion, and the horizontal pressure gradient all drop out and no halo
information is exchanged between columns. For anelastic dynamics, Breeze additionally:

- **omits the pressure solve.** The anelastic mass constraint ``∂_z(ρ_r w) = 0`` together with rigid
  top and bottom boundaries forces ``w ≡ 0`` in a single column, so there is no elliptic problem to
  solve and no pressure solver is allocated.
- **omits the vertical-velocity stepping.** The vertical momentum ``ρw`` stays at zero, so ``w ≡ 0``.
  Vertical transport is carried entirely by the turbulence closure and by prescribed large-scale
  forcing (for example subsidence).

The horizontal momentum ``(u, v)`` and all scalars (potential temperature, moisture, tracers) still
evolve under vertical mixing, Coriolis, microphysics, radiation, and forcing — exactly a
single-column model.

## A single column

Build a `(Flat, Flat, Bounded)` grid with a scalar `size` (the number of vertical levels):

```jldoctest scm
using Breeze
using Oceananigans

grid = RectilinearGrid(size=32, z=(0, 3000), topology=(Flat, Flat, Bounded))
summary(grid)

# output
"1×1×32 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo"
```

Any `AtmosphereModel` built on such a grid runs in single column mode automatically — and no
pressure solver is constructed (`model.pressure_solver` is `nothing`):

```jldoctest scm
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants)
dynamics = AnelasticDynamics(reference_state)

model = AtmosphereModel(grid; dynamics, closure=VerticalScalarDiffusivity(κ=1))
summary(model)

# output
"AtmosphereModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)"
```

## A forest of columns

`Oceananigans.Grids.ColumnEnsembleSize(Nz, ensemble=(N₁, N₂), Hz)` lays out `N₁ × N₂` independent
columns in the (Flat) horizontal dimensions, forcing the horizontal halos to zero so the columns
never interact:

```jldoctest scm
using Oceananigans.Grids: ColumnEnsembleSize

ensemble_grid = RectilinearGrid(size = ColumnEnsembleSize(Nz=32, ensemble=(10, 1), Hz=3),
                                z = (0, 3000), topology = (Flat, Flat, Bounded))
summary(ensemble_grid)

# output
"10×1×32 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo"
```

Every column of the ensemble evolves exactly as if it were a standalone single-column model — the
columns are provably independent. This is what makes ensembles cheap: `N₁·N₂` columns are stepped in
a single kernel launch, on CPU or GPU.

## Per-column parameters

For ensembles to be useful, each column can carry its own parameters, supplied as *arrays* indexed by
column. (On a `(Flat, Flat, Bounded)` grid there are no meaningful horizontal coordinates, so
per-column values are given as arrays, not coordinate functions.)

- **Closure.** Pass an `(N₁, N₂)` array of closures — one per column — to vary the mixing
  parameters that a calibration would tune:

  ```julia
  κs = [1, 3, 10, 30, 100]  # one vertical diffusivity per column
  closures = [VerticalScalarDiffusivity(κ=κ) for κ in κs, j in 1:1]
  model = AtmosphereModel(ensemble_grid; dynamics, closure=closures)
  ```

- **Coriolis.** Pass an `(N₁, N₂)` array of rotations (for example one `FPlane` per column) to place
  columns at different latitudes.

- **Forcing.** A discrete forcing whose `parameters` are a per-column array applies a different
  large-scale tendency to each column:

  ```julia
  @inline column_heating(i, j, k, grid, clock, fields, Q) = @inbounds Q[i, j]
  θ_forcing = Forcing(column_heating, discrete_form=true, parameters=Q)  # Q is an (N₁, N₂) array
  ```

- **Reference state.** Array-valued `surface_pressure` and/or `potential_temperature` give each
  column its own adiabatic background profile:

  ```julia
  θ₀ = [285 + 5i for i in 1:N₁, j in 1:N₂]
  reference_state = ReferenceState(ensemble_grid, constants; potential_temperature=θ₀)
  ```

See the [Single-column ensemble](@ref "Single-column ensemble") example for an end-to-end demonstration
of a column ensemble with per-column vertical diffusivity.
