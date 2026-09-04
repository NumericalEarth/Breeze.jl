# Lagrangian droplets

Breeze can carry cloud droplets as Lagrangian particles: each particle is a κ-Köhler
droplet that is advected by the resolved flow (see [Lagrangian particles](@ref)) and
grows or evaporates by condensation with the supersaturation interpolated to its
position. The droplets do not feed back on the flow.

## Droplet physics

A wet aerosol particle of diameter ``D``, dry diameter ``D^d``, and hygroscopicity
``κ`` is in equilibrium with air at the supersaturation

```math
𝒮^e(D) = \exp(A / D) \frac{D^3 - (D^d)^3}{D^3 - (D^d)^3 (1 - κ)} - 1 ,
\qquad A = \frac{4 σ M^w}{R T ρ^w} ,
```

the κ-Köhler curve of Petters and Kreidenweis (2007). Its maximum, at the critical
diameter ``D^c``, is the critical supersaturation ``𝒮^c``; a droplet with ``D ≥ D^c`` is
activated. Away from equilibrium the squared diameter obeys the Maxwell–Mason equation

```math
\frac{d D^2}{dt} = 8 G(T, p, D) \left[ 𝒮 - 𝒮^e(D) \right] ,
```

with a growth coefficient ``G`` that includes the kinetic corrections of the vapor
diffusivity and thermal conductivity near a small droplet. Breeze integrates this stiff
equation with a backward-Euler step in ``D^2``, solved by a fixed number of Newton
iterations with the diameter floored at ``D^d``, so that sub-micron haze is stable at the
time step of the flow.

```@docs
Breeze.LagrangianMicrophysics.equilibrium_supersaturation
Breeze.LagrangianMicrophysics.critical_diameter
Breeze.LagrangianMicrophysics.critical_supersaturation
Breeze.LagrangianMicrophysics.equilibrium_diameter
Breeze.LagrangianMicrophysics.growth_coefficient
Breeze.LagrangianMicrophysics.implicit_growth_step
Breeze.LagrangianMicrophysics.ambient_supersaturation
```

## Droplets in a model

A set of droplets is a `StructArray` of [`Droplet`](@ref Breeze.LagrangianMicrophysics.Droplet)s
attached to the model through `LagrangianParticles` with [`DropletDynamics`](@ref Breeze.LagrangianMicrophysics.DropletDynamics)
as their `dynamics`:

```jldoctest droplets
using Breeze, Oceananigans
using StructArrays: StructArray

grid = RectilinearGrid(size=(4, 4, 4), x=(0, 400), y=(0, 400), z=(0, 400))
constants = ThermodynamicConstants()

Dᵈ, κ, T₀ = 130e-9, 1.0, 290.0
Dᶜ = critical_diameter(Dᵈ, κ, T₀, constants)
D₀ = equilibrium_diameter(-0.2, Dᵈ, κ, T₀, constants)

N = 100
column(v) = fill(v, N)
droplets = StructArray{Droplet{Float64}}((400rand(N), 400rand(N), 400rand(N),
                                          column(Dᵈ), column(κ), column(D₀^2), column(Dᶜ),
                                          column(0.0), column(0.0), column(0.0), column(0.0)))

particles = LagrangianParticles(droplets; dynamics=DropletDynamics())
model = AtmosphereModel(grid; particles)
model.particles.dynamics

# output
DropletDynamics{Float64}(accommodation=0.3, thermal_accommodation=0.96, newton_iterations=8, substeps=1)
```

Once per time step, `DropletDynamics` interpolates the temperature, vapor mass fraction,
and pressure of the model to every droplet and advances its diameter; the droplets are
then advected. The supersaturation each droplet saw is stored in its `𝒮` property, and
[`activated_fraction`](@ref Breeze.LagrangianMicrophysics.activated_fraction) counts the
droplets beyond their critical diameter.

```@docs
Breeze.LagrangianMicrophysics.Droplet
Breeze.LagrangianMicrophysics.DropletDynamics
Breeze.LagrangianMicrophysics.droplet_tracked_fields
Breeze.LagrangianMicrophysics.activated
Breeze.LagrangianMicrophysics.activated_fraction
```
