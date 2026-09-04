#####
##### Lagrangian droplets
#####
##### A `Droplet` is the element type of a `StructArray` of Lagrangian particles carrying
##### κ-Köhler droplet state alongside their position. Oceananigans' `LagrangianParticles`
##### advects such particles with the resolved flow. `DropletDynamics`, the particles'
##### `dynamics`, first interpolates the model's temperature, vapor mass fraction, and
##### pressure to every droplet (into the `T`, `qᵛ`, and `p` properties, like Oceananigans'
##### `tracked_fields` but clamped away from unfilled halo corners) and then grows or evaporates it by one
##### implicit step of the Maxwell–Mason equation, once per time step, before advection.
##### The interpolation happens inside the dynamics because the model fields do not exist
##### yet when the particles are constructed.
#####

"""
    Droplet{FT}

The state of one Lagrangian droplet: its position, its dry diameter `Dᵈ` and hygroscopicity
`κ`, its squared wet diameter `D²`, the critical diameter `Dᶜ` of its Köhler curve at the
reference temperature at which it was initialized, the temperature `T`, vapor mass fraction
`qᵛ`, and pressure `p` interpolated from the flow, and the resulting supersaturation `𝒮`
seen by the droplet, computed from `T`, `qᵛ`, and `p` exactly as the model's own
thermodynamics computes it (the gas-phase density is diagnosed at the reference pressure). The wet diameter is carried continuously through activation and
deactivation; a droplet is activated when `D² ≥ Dᶜ²`.
"""
struct Droplet{FT}
    x :: FT
    y :: FT
    z :: FT
    Dᵈ :: FT
    κ :: FT
    D² :: FT
    Dᶜ :: FT
    T :: FT
    qᵛ :: FT
    p :: FT
    𝒮 :: FT
end

"""
$(TYPEDSIGNATURES)

The model fields that [`DropletDynamics`](@ref) interpolates to every droplet, named after
the droplet properties that receive them: the temperature `T`, the vapor mass fraction
`qᵛ`, and the pressure `p` (the reference pressure under anelastic dynamics).
"""
droplet_tracked_fields(model) = (; T = model.temperature,
                                   qᵛ = specific_prognostic_moisture(model),
                                   p = dynamics_pressure(model.dynamics))

#####
##### Interpolation to the droplets
#####
##### Oceananigans interpolates `tracked_fields` to particles trilinearly from the cell
##### centres, which within half a cell of a wall reaches into the halo. On a fully bounded
##### grid the halo fills cover only the interior extent of the other two directions, so the
##### edge and corner halo cells of a field are never written and a particle near an edge
##### blends the field with garbage. The finite-volume stencils never touch those cells, so
##### this only matters for interpolation. Breeze therefore interpolates at a position clamped
##### to the range of cell centres in every `Bounded` direction: within the wall half-cell the
##### droplet sees the value of the wall-adjacent cell.
#####
##### TODO (upstream Oceananigans): `update_property!` and `advect_particle` share this
##### exposure (`fill_halo_regions!` with bounded conditions uses `size(grid, loc)` in the
##### tangential directions, `fill_halo_regions.jl`); either fill the edge halos of tracked
##### fields and velocities or clamp the interpolation node in `flattened_node`.
#####

@inline clamp_to_centers(x, ::Bounded, x₁, xₙ) = clamp(x, x₁, xₙ)
@inline clamp_to_centers(x, topology, x₁, xₙ) = x

# The flattened node (with `z` mapped to the reference coordinate `r` on terrain-following
# grids) clamped to the cell centres in each `Bounded` direction. Flat directions are absent
# from the flattened node, as in Oceananigans.
@inline function clamped_node((x, y, z), grid)
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)
    c = Center()
    x′, y′, r′ = flattened_node((x, y, z), grid)
    x′ = clamp_to_centers(x′, TX(), xnode(1, 1, 1, grid, c, c, c), xnode(Nx, 1, 1, grid, c, c, c))
    y′ = clamp_to_centers(y′, TY(), ynode(1, 1, 1, grid, c, c, c), ynode(1, Ny, 1, grid, c, c, c))
    r′ = clamp_to_centers(r′, TZ(), rnode(1, 1, 1, grid, c, c, c), rnode(1, 1, Nz, grid, c, c, c))
    return (x′, y′, r′)
end

@inline function clamped_node((x, y, z), grid::XFlatGrid)
    _, TY, TZ = topology(grid)
    _, Ny, Nz = size(grid)
    c = Center()
    y′, r′ = flattened_node((x, y, z), grid)
    y′ = clamp_to_centers(y′, TY(), ynode(1, 1, 1, grid, c, c, c), ynode(1, Ny, 1, grid, c, c, c))
    r′ = clamp_to_centers(r′, TZ(), rnode(1, 1, 1, grid, c, c, c), rnode(1, 1, Nz, grid, c, c, c))
    return (y′, r′)
end

@inline function clamped_node((x, y, z), grid::YFlatGrid)
    TX, _, TZ = topology(grid)
    Nx, _, Nz = size(grid)
    c = Center()
    x′, r′ = flattened_node((x, y, z), grid)
    x′ = clamp_to_centers(x′, TX(), xnode(1, 1, 1, grid, c, c, c), xnode(Nx, 1, 1, grid, c, c, c))
    r′ = clamp_to_centers(r′, TZ(), rnode(1, 1, 1, grid, c, c, c), rnode(1, 1, Nz, grid, c, c, c))
    return (x′, r′)
end

# A single column: only the vertical coordinate remains
@inline function clamped_node((x, y, z), grid::XYFlatGrid)
    _, _, TZ = topology(grid)
    _, _, Nz = size(grid)
    c = Center()
    r′, = flattened_node((x, y, z), grid)
    r′ = clamp_to_centers(r′, TZ(), rnode(1, 1, 1, grid, c, c, c), rnode(1, 1, Nz, grid, c, c, c))
    return (r′,)
end

@kernel function _interpolate_to_droplets!(droplets, grid, T, ℓT, qᵛ, ℓq, p, ℓp)
    n = @index(Global)
    @inbounds begin
        X = clamped_node((droplets.x[n], droplets.y[n], droplets.z[n]), grid)
        droplets.T[n] = interpolate(X, T, ℓT, grid)
        droplets.qᵛ[n] = interpolate(X, qᵛ, ℓq, grid)
        droplets.p[n] = interpolate(X, p, ℓp, grid)
    end
end

"""
$(TYPEDSIGNATURES)

Interpolate the model's temperature, vapor mass fraction, and pressure to the `droplets`
(the `properties` of `LagrangianParticles`), into their `T`, `qᵛ`, and `p` properties.
"""
function interpolate_to_droplets!(droplets, model)
    grid = model.grid
    arch = architecture(grid)
    fields = droplet_tracked_fields(model)
    ℓ(field) = map(instantiate, location(field))
    launch!(arch, grid, KernelParameters(1:length(droplets)), _interpolate_to_droplets!, droplets, grid,
            fields.T, ℓ(fields.T), fields.qᵛ, ℓ(fields.qᵛ), fields.p, ℓ(fields.p))
    return nothing
end

#####
##### Dynamics
#####

struct DropletDynamics{FT, TC}
    accommodation :: FT
    thermal_accommodation :: FT
    newton_iterations :: Int
    substeps :: Int
    thermodynamic_constants :: TC
end

"""
$(TYPEDSIGNATURES)

The Lagrangian particle `dynamics` that grows and evaporates [`Droplet`](@ref)s by
condensation, `d(D²)/dt = 8 G (𝒮 − 𝒮ᵉ)`, with the ambient supersaturation `𝒮` computed
from the temperature, vapor mass fraction, and pressure interpolated to each droplet
(see [`droplet_tracked_fields`](@ref)). Each time step is integrated by `substeps`
backward-Euler steps of [`implicit_growth_step`](@ref), each solved with
`newton_iterations` Newton iterations.

Keyword arguments
=================

- `accommodation`: the condensation (mass accommodation) coefficient in the kinetic
  correction of the vapor diffusivity (default: `0.3`, as in `pyrcel`).
- `thermal_accommodation`: the thermal accommodation coefficient in the kinetic
  correction of the thermal conductivity (default: `0.96`).
- `newton_iterations`: Newton iterations per implicit step (default: `8`).
- `substeps`: implicit steps per model time step (default: `1`).
- `thermodynamic_constants`: the [`ThermodynamicConstants`](@ref) used for the
  saturation vapor pressure, latent heat, gas constants, and liquid density.

The droplets are advected and grown by attaching the dynamics to the particles of an
`AtmosphereModel`, `LagrangianParticles(droplets; dynamics=DropletDynamics())`, where
`droplets` is a `StructArray` of [`Droplet`](@ref)s.

Example
=======

```jldoctest
using Breeze

dynamics = DropletDynamics(substeps=4)

# output
DropletDynamics{Float64}(accommodation=0.3, thermal_accommodation=0.96, newton_iterations=8, substeps=4)
```
"""
function DropletDynamics(FT = Oceananigans.defaults.FloatType;
                         accommodation = 0.3,
                         thermal_accommodation = 0.96,
                         newton_iterations = 8,
                         substeps = 1,
                         thermodynamic_constants = ThermodynamicConstants(FT))

    return DropletDynamics(convert(FT, accommodation),
                           convert(FT, thermal_accommodation),
                           Int(newton_iterations),
                           Int(substeps),
                           thermodynamic_constants)
end

Base.summary(dynamics::DropletDynamics{FT}) where FT =
    string("DropletDynamics{", FT, "}(accommodation=", dynamics.accommodation,
           ", thermal_accommodation=", dynamics.thermal_accommodation,
           ", newton_iterations=", dynamics.newton_iterations,
           ", substeps=", dynamics.substeps, ")")

Base.show(io::IO, dynamics::DropletDynamics) = print(io, summary(dynamics))

# The `dynamics` hook of `LagrangianParticles`: called once per time step with the whole
# model, before advection. Interpolate the ambient state to the droplets, then grow them.
function (dynamics::DropletDynamics)(particles, model, Δt)
    droplets = particles.properties
    interpolate_to_droplets!(droplets, model)

    grid = model.grid
    arch = architecture(grid)
    launch!(arch, grid, KernelParameters(1:length(droplets)), _grow_droplets!, droplets, dynamics, Δt)
    return nothing
end

@kernel function _grow_droplets!(droplets, dynamics, Δt)
    n = @index(Global)
    constants = dynamics.thermodynamic_constants

    @inbounds begin
        T = droplets.T[n]
        qᵛ = droplets.qᵛ[n]
        p = droplets.p[n]
        Dᵈ = droplets.Dᵈ[n]
        κ = droplets.κ[n]
        D² = droplets.D²[n]
    end

    𝒮 = ambient_supersaturation(T, qᵛ, p, constants)
    δt = Δt / dynamics.substeps
    for _ in 1:dynamics.substeps
        D² = implicit_growth_step(D², 𝒮, T, p, Dᵈ, κ, δt, dynamics, constants)
    end

    @inbounds begin
        droplets.D²[n] = D²
        droplets.𝒮[n] = 𝒮
    end
end

#####
##### Diagnostics on a set of droplets
#####

"""
$(TYPEDSIGNATURES)

Whether each droplet in `droplets` (the `properties` of `LagrangianParticles`, or any
struct-of-arrays with `D²` and `Dᶜ`) is activated, `D² ≥ Dᶜ²`.
"""
activated(droplets) = droplets.D² .≥ droplets.Dᶜ .^ 2

"""
$(TYPEDSIGNATURES)

The fraction of activated droplets in `droplets`.
"""
activated_fraction(droplets) = count(activated(droplets)) / length(droplets.D²)
