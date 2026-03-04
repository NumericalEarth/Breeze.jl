---
name: new-simulation
description: Set up, run, and visualize a new Breeze atmospheric simulation
user_invocable: true
---

# New Simulation

Set up, run, and visualize a new atmospheric simulation with Breeze.

## Step 1: Understand the Case

**If reproducing a paper:**
- Read the paper carefully and extract ALL parameters: domain size, resolution, physical constants,
  boundary conditions, initial conditions, forcing, closure parameters
- Check parameter tables, figure captions, and coordinate conventions
- Identify the paper's prognostic variables and how forcing is applied
- Identify Breeze's prognostics (`ρθ` or `ρe`) and derive any transformations needed

**If designing a new case:**
- Ask the user for the science goal or phenomenon to simulate
- Clarify: domain geometry, resolution, physics (buoyancy, Coriolis, microphysics, radiation), run duration
- Study working examples first: BOMEX, RICO, prescribed_SST, thermal_bubble

## Step 2: Set Up Geometry

- Create the grid and verify domain extents
- If reproducing a paper, compare geometry to paper figures

## Step 3: Set Initial Conditions

- Apply initial conditions, then verify:
  - `minimum(field)` and `maximum(field)` make physical sense
  - Stratification, moisture profiles are correct
- Use `set!` ideally once (it calls `update_state!` internally)

## Step 4: Short Test Run

- Run a few timesteps on CPU at low resolution
- Check for NaNs: `maximum(abs, u)`, `maximum(abs, w)`, etc.
- Verify flow is developing (velocities changing from initial state)

## Step 5: Progressive Validation

- Run a short simulation (~1 hour sim time) and visualize results
- If reproducing a paper, compare to early-time figures

## Step 6: Production Run and Comparison

- Run at full resolution / full duration
- Create diagnostic visualizations matching the science goal
- If reproducing a paper, match figure format, colormaps, axis ranges, time snapshots

## Visualization Guide

**CRITICAL**: Plot `Field` objects directly — avoid `interior()`. Use `view(field, i, j, k)` to slice.

```julia
using CairoMakie
using Oceananigans, Breeze

# 2D field — just pass the field
heatmap!(ax, field)

# Slice 3D field
b_section = view(b, :, grid.Ny÷2, :)
heatmap!(ax, b_section)

# Animations with Observables
n = Observable(1)
field_n = @lift field_ts[$n]
heatmap!(ax, field_n)
```

- Always add axis labels and colorbars
- **Color palette**: `:dodgerblue` (vapor), `:lime` (cloud), `:orangered` (rain), `:magenta` (temperature)
- **Do not convert units** except spatial coordinates to km for axis labels

## Common Issues

- **NaN blowups**: timestep too large, unstable ICs, `if`/`else` on GPU (use `ifelse`)
- **Nothing happening**: wrong buoyancy sign, ICs not applied, forcing inactive
- **Wrong flow direction**: check coordinate conventions
- **Thermodynamic variable mismatch**: paper uses T but Breeze uses θ — don't forget Exner function!

## Output

- Place example scripts in `examples/`
- Follow existing conventions and Literate.jl style
