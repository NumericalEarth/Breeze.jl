#####
##### The implicit step of the vertical diffusion of HydrostaticDynamics
#####

AtmosphereModels.implicit_dynamics_step!(dynamics::HydrostaticDynamics, model, Δt) =
    step_vertical_diffusion!(dynamics.vertical_diffusion, model, Δt)

step_vertical_diffusion!(::Nothing, model, Δt) = nothing

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Advance the potential temperature and the velocities of `model` by one backward Euler step `Δt`
of the vertical `diffusion` of its [`HydrostaticDynamics`](@ref), with one tridiagonal solve per
column and field.

The time steppers call this after the explicit stages of a time step. The layer masses, the
Exner functions and the potential temperature are first diagnosed from the state that the stages
produced, then the diffusivities by the boundary-layer scheme, if there is one, and the
conductances `ρᵃ K / Δh` of the coordinate surfaces from them.
"""
function step_vertical_diffusion!(diffusion::VerticalDiffusion, model, Δt)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    ρ = dynamics.density
    ρθ = model.formulation.potential_temperature_density
    u, v, _ = model.velocities
    C = diffusion.conductances
    γ = diffusion.scratch
    Δt = convert(eltype(grid), Δt)

    update_state!(model; compute_tendencies=false)

    compute_boundary_layer_diffusivities!(diffusion.boundary_layer, diffusion, model)
    validate_diffusivities(diffusion)

    @apply_regionally launch!(arch, grid, :xy, _compute_conductances!,
                              C.scalar, C.momentum, grid, diffusion.diffusivity, diffusion.viscosity,
                              model.formulation.potential_temperature,
                              dynamics.exner_function, dynamics.interface_exner_function,
                              dynamics.pressure_coefficient, dynamics.surface_pressure_coefficient,
                              dynamics.surface_pressure,
                              dry_air_gas_constant(constants), constants.dry_air.heat_capacity,
                              constants.gravitational_acceleration)

    fill_halo_regions!(C.momentum)
    validate_velocity_conductances(grid, C.momentum)

    @apply_regionally launch!(arch, grid, :xy, _diffuse_potential_temperature_density!, ρθ, γ, grid, ρ, C.scalar, Δt)
    @apply_regionally launch!(arch, grid, :xy, _diffuse_velocity!, u, γ, grid, ρ, C.momentum, Δt, ℑxᶠᵃᵃ)
    @apply_regionally launch!(arch, grid, :xy, _diffuse_velocity!, v, γ, grid, ρ, C.momentum, Δt, ℑyᵃᶠᵃ)

    return nothing
end

# The conductance of a velocity column is the average of those of the columns of cells on either
# side of it. On a single grid both are computed, and the average of two admissible values is
# admissible. On a multi-region grid one of the two may lie in a halo, where the conductance is
# interpolated from the other region: on a `YinYangGrid` with a cubic interpolation, which can
# overshoot below zero next to a sharp change of K. A negative conductance would make the
# implicit solve anti-diffusive, so it is refused rather than clipped.
validate_velocity_conductances(grid, C) = nothing

function validate_velocity_conductances(grid::MultiRegion.MultiRegionGrids, C)
    Cᵘ = KernelFunctionOperation{Face, Center, Face}(ℑxᶠᵃᵃ, grid, C)
    Cᵛ = KernelFunctionOperation{Center, Face, Face}(ℑyᵃᶠᵃ, grid, C)

    for (name, conductance) in ((:u, Cᵘ), (:v, Cᵛ))
        is_admissible_diffusivity(conductance) ||
            throw(DomainError(name, "The vertical diffusion of $name has a conductance that is negative, infinite " *
                                    "or not a number at a velocity point next to the edge of a region, where it " *
                                    "is interpolated from the other region (between $(minimum(conductance)) and " *
                                    "$(maximum(conductance))). The viscosity changes too sharply across the edge " *
                                    "for the interpolation of the halos."))
    end

    return nothing
end

#####
##### The column between two layers
#####

# Height between the layers `k - 1` and `k`, from the hydrostatic integral of a layer of uniform
# potential temperature: g Δh = cᵖᵈ [θ⁻ (Π⁻ - Πᶠ) + θ⁺ (Πᶠ - Π⁺)]
@inline function height_between_layers(i, j, k, grid, θ, Π, Πᶠ, cᵖᵈ, g)
    @inbounds begin
        ΔΦ⁻ = cᵖᵈ * θ[i, j, k-1] * (Π[i, j, k-1] - Πᶠ[i, j, k])
        ΔΦ⁺ = cᵖᵈ * θ[i, j, k] * (Πᶠ[i, j, k] - Π[i, j, k])
    end
    return (ΔΦ⁻ + ΔΦ⁺) / g
end

# Mass of layer `k` per unit area, Δp / g, at cell centers
@inline layer_mass_per_area(i, j, k, grid, ρ) = @inbounds ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
@inline layer_mass_per_area(i, j, k, grid::PressureLevelGrid, ρ) = layer_mass(i, j, k, grid)

#####
##### Conductances
#####

# C = ρᵃ K / Δh on the z-faces between two layers, with ρᵃ = p / (Rᵈ T) the density of the air
# on the face, at the pressure A + B pˢ and the temperature Πᶠ θᶠ. Nothing passes through the
# ground or the model top.
@kernel function _compute_conductances!(Cᶜ, Cᵐ, grid, Kᶜ, Kᵐ, θ, Π, Πᶠ, A, B, pˢ, Rᵈ, cᵖᵈ, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        Cᶜ[i, j, 1] = zero(grid)
        Cᵐ[i, j, 1] = zero(grid)

        for k in 2:Nz
            p = interface_pressure(i, j, k, A, B, pˢ)
            T = Πᶠ[i, j, k] * interface_potential_temperature(i, j, k, grid, θ, Π, Πᶠ)
            ρᵃ = p / (Rᵈ * T)
            Δh = height_between_layers(i, j, k, grid, θ, Π, Πᶠ, cᵖᵈ, g)

            Cᶜ[i, j, k] = ρᵃ * Kᶜ[i, j, k] / Δh
            Cᵐ[i, j, k] = ρᵃ * Kᵐ[i, j, k] / Δh
        end

        Cᶜ[i, j, Nz+1] = zero(grid)
        Cᵐ[i, j, Nz+1] = zero(grid)
    end
end

#####
##### Boundary-layer diffusivities
#####

compute_boundary_layer_diffusivities!(::Nothing, diffusion, model) = nothing

function compute_boundary_layer_diffusivities!(scheme::RichardsonNumberBoundaryLayer, diffusion, model)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    u, v, _ = model.velocities

    @apply_regionally launch!(arch, grid, :xy, _compute_richardson_number_diffusivities!,
                              diffusion.diffusivity, diffusion.viscosity, grid, scheme,
                              model.formulation.potential_temperature, u, v,
                              dynamics.exner_function, dynamics.interface_exner_function,
                              constants.dry_air.heat_capacity, constants.gravitational_acceleration)

    return nothing
end

@kernel function _compute_richardson_number_diffusivities!(Kᶜ, Kᵐ, grid, scheme, θ, u, v, Π, Πᶠ, cᵖᵈ, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        Kᶜ[i, j, 1] = zero(grid)
        Kᵐ[i, j, 1] = zero(grid)

        # Height of the z-face above the ground, accumulated over the layers below it
        h = zero(grid)

        for k in 2:Nz
            h += cᵖᵈ * θ[i, j, k-1] * (Πᶠ[i, j, k-1] - Πᶠ[i, j, k]) / g
            Δh = height_between_layers(i, j, k, grid, θ, Π, Πᶠ, cᵖᵈ, g)

            δu = ℑxᶜᵃᵃ(i, j, k, grid, u) - ℑxᶜᵃᵃ(i, j, k-1, grid, u)
            δv = ℑyᵃᶜᵃ(i, j, k, grid, v) - ℑyᵃᶜᵃ(i, j, k-1, grid, v)
            S² = (δu^2 + δv^2) / Δh^2
            N² = 2g * (θ[i, j, k] - θ[i, j, k-1]) / ((θ[i, j, k] + θ[i, j, k-1]) * Δh)

            K = eddy_diffusivity(scheme, h, S², N²)
            Kᶜ[i, j, k] = K
            Kᵐ[i, j, k] = K
        end

        Kᶜ[i, j, Nz+1] = zero(grid)
        Kᵐ[i, j, Nz+1] = zero(grid)
    end
end

#####
##### Tridiagonal solves
#####

# One backward Euler step of μ ∂ₜψ = C⁺ (ψ[k+1] - ψ[k]) - C⁻ (ψ[k] - ψ[k-1]) in the column (i, j),
# by the Thomas algorithm: `ψ` is overwritten with the solution and `γ` holds the eliminated
# upper diagonal. `mass(i, j, k, grid, ρ)` is the layer mass per unit area at the location of ψ
# and `conductance(i, j, k, grid, C)` the conductance of the z-face `k` there. Row `k` reads
#
#   - Δt C[k] ψ[k-1] + (μ[k] + Δt C[k] + Δt C[k+1]) ψ[k] - Δt C[k+1] ψ[k+1] = μ[k] ψⁿ[k] ,
#
# and the sum of the rows is Σ μ ψ = Σ μ ψⁿ, since C vanishes on the first and the last z-face.
@inline function diffuse_column!(ψ, γ, i, j, grid, Δt, ρ, C, mass, conductance)
    Nz = size(grid, 3)

    @inbounds begin
        μ = mass(i, j, 1, grid, ρ)
        C⁺ = conductance(i, j, 2, grid, C)
        β = μ + Δt * C⁺
        ψ[i, j, 1] = μ * ψ[i, j, 1] / β

        for k in 2:Nz
            C⁻ = C⁺
            C⁺ = conductance(i, j, k+1, grid, C)
            μ = mass(i, j, k, grid, ρ)

            γ[i, j, k] = - Δt * C⁻ / β
            β = μ + Δt * (C⁻ + C⁺) + Δt * C⁻ * γ[i, j, k]
            ψ[i, j, k] = (μ * ψ[i, j, k] + Δt * C⁻ * ψ[i, j, k-1]) / β
        end

        for k in Nz-1:-1:1
            ψ[i, j, k] -= γ[i, j, k+1] * ψ[i, j, k+1]
        end
    end

    return nothing
end

@inline face_conductance(i, j, k, grid, C) = @inbounds C[i, j, k]

# θ = ρθ / ρ is diffused, and ρθ = ρ θ returned: the layer mass does not change
@kernel function _diffuse_potential_temperature_density!(ρθ, γ, grid, ρ, C, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds for k in 1:Nz
        ρθ[i, j, k] /= ρ[i, j, k]
    end

    diffuse_column!(ρθ, γ, i, j, grid, Δt, ρ, C, layer_mass_per_area, face_conductance)

    @inbounds for k in 1:Nz
        ρθ[i, j, k] *= ρ[i, j, k]
    end
end

# The mass and the conductance of a velocity column are those of the two columns of cells on
# either side of it, brought to the velocity point by `interpolate` (ℑxᶠᵃᵃ for u, ℑyᵃᶠᵃ for v)
@inline interpolated_layer_mass(i, j, k, grid, (ρ, interpolate)) = interpolate(i, j, k, grid, layer_mass_per_area, ρ)
@inline interpolated_conductance(i, j, k, grid, (C, interpolate)) = interpolate(i, j, k, grid, C)

@kernel function _diffuse_velocity!(velocity, γ, grid, ρ, C, Δt, interpolate)
    i, j = @index(Global, NTuple)
    diffuse_column!(velocity, γ, i, j, grid, Δt, (ρ, interpolate), (C, interpolate),
                    interpolated_layer_mass, interpolated_conductance)
end
