#####
##### Gravity-wave substepping for HydrostaticDynamics
#####
##### The fast system is the linear gravity-wave operator of the hydrostatic equations, linearized
##### about the state Uᴸ of each column at the start of the time step, with the velocity at rest.
##### Writing primes for departures from Uᴸ, (pˢ′, u′, v′, ρθ′) = U − Uᴸ, it reads
#####
#####   ∂τ pˢ′ = - g Σ D(ρᴸ, u′, v′)                                      column continuity
#####   ∂τ ρθ′ = - ∇·(ρᴸ 𝐮′ θᴸ) - ∂z(ρw′ θᶠᴸ)                            flux of θᴸ by the perturbation mass flux
#####   ∂τ 𝐮′  = - cᵖᵈ θ′ ∇Πᴸ - cᵖᵈ θᴸ ∇Π′ - ∇Φ′                        linearized pressure-gradient force
#####
##### with ρw′ the vertical mass flux that continuity gives for (ρᴸ, u′, v′), and
#####
#####   θ′ = (ρθ′ - θᴸ ∂ρ/∂pˢ pˢ′) / ρᴸ ,   Π′ = ∂Π/∂pˢ pˢ′ ,   Φ′ = Φ[θ′; Πᴸ] + ∂Φ/∂pˢ pˢ′ .
#####
##### Every term is built from the operators of the full tendencies (`hydrostatic_tendencies.jl`,
##### `hydrostatic_column_diagnostics.jl`) applied to the perturbation or to the frozen state:
##### the divergence, the continuity integral, the flux of θ, the hydrostatic integral and the
##### pressure-gradient force are not written a second time. Only the derivatives of the Exner
##### functions with respect to pˢ are new.
#####
##### A time stepper integrates ∂τ U′ = L U′ + F over each of its stages, with F constant, and
##### obtains the slow tendency as the full tendency less L U′ (`subtract_fast_tendencies!`), so
##### that the split scheme and the unsplit one discretize the same equations.
#####
##### Two inner solvers advance (pˢ′, ρθ′) and (u′, v′) in turn:
#####
#####   - `ForwardBackwardSubsteps`: (pˢ′, ρθ′) with the old velocity, then the velocity with the
#####     new (pˢ′, ρθ′) extrapolated forward by the divergence damping βᵈ. First order.
#####   - `VerletSubsteps`: half a step of (pˢ′, ρθ′), a step of the velocity, half a step of
#####     (pˢ′, ρθ′). Time centered, second order, undamped.
#####
##### The velocity perturbation that moves mass in each substep is accumulated, weighted by the
##### substep, so that scalars can be transported with the same mass fluxes as the layer mass.
#####

#####
##### Inner solvers
#####

"""
$(TYPEDEF)

Forward-backward substeps of the gravity-wave system of
[`HydrostaticDynamics`](@ref): the surface pressure and the potential temperature density are
advanced with the old velocity, then the velocity with the pressure-gradient force of their new
values ``Y``, extrapolated forward to ``Y + βᵈ (Y - Y⁻)`` with the `divergence_damping` ``βᵈ``.

The substeps are stable for a stage Courant number ``ω Δτ ≤ 2 / \\sqrt{1 + 2 βᵈ}``, with ``ω``
the largest gravity-wave frequency, and are first order accurate in time for any ``βᵈ``.
"""
struct ForwardBackwardSubsteps{FT}
    divergence_damping :: FT # βᵈ
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return [`ForwardBackwardSubsteps`](@ref) with `divergence_damping`, which may not be negative.
"""
function ForwardBackwardSubsteps(; divergence_damping=0.1)
    divergence_damping ≥ 0 ||
        throw(ArgumentError("divergence_damping may not be negative, got $divergence_damping."))
    return ForwardBackwardSubsteps(divergence_damping)
end

"""
$(TYPEDEF)

Time-centered forward-backward (Störmer-Verlet) substeps of the gravity-wave system of
[`HydrostaticDynamics`](@ref): half a substep of the surface pressure and the potential
temperature density, a substep of the velocity, and half a substep of the former again.
Consecutive half substeps are merged.

The substeps are stable for a stage Courant number ``ω Δτ ≤ 2`` and second order accurate in
time. They carry no divergence damping, which would reduce them to first order.
"""
struct VerletSubsteps end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the largest stage Courant number ``ω Δτ`` for which the substeps of `solver` are stable,
with ``ω`` the largest frequency of the gravity-wave system and ``Δτ`` the substep.
"""
stage_cfl_bound(solver::ForwardBackwardSubsteps) = 2 / sqrt(1 + 2 * solver.divergence_damping)
stage_cfl_bound(::VerletSubsteps) = 2

Base.summary(solver::ForwardBackwardSubsteps) =
    string("ForwardBackwardSubsteps(divergence_damping=", prettysummary(solver.divergence_damping), ")")

Base.summary(::VerletSubsteps) = "VerletSubsteps()"

#####
##### Substepper
#####

# The prognostic variables that the gravity-wave substeps advance
const gravity_wave_prognostic_names = (:pˢ, :u, :v, :ρθ)

struct GravityWaveSubstepper{S, N, FT, A, L, C, P, W, M, I}
    inner_solver :: S                # ForwardBackwardSubsteps or VerletSubsteps
    substeps :: N                    # substeps per time step, or nothing for the Courant number rule
    cfl :: FT                        # stage Courant number ω Δτ aimed at by the rule
    maximum_wavenumber :: FT         # largest horizontal wavenumber of the grid, 2 √(Δx⁻² + Δy⁻²)
    advection :: A                   # centered scheme that puts θᴸ and cᴸ on the cell faces
    linearization :: L               # (ρ, θ, Π, Πᶠ, dΠdpˢ, dΠᶠdpˢ, dΦdpˢ) of the state at the start of the step
    specific_scalars :: C            # cᴸ = ρc / ρ of every other prognostic scalar at the start of the step
    perturbations :: P               # (pˢ, u, v, ρθ): departures from the state at the start of the step
    forcing :: P                     # (pˢ, u, v, ρθ): tendencies held constant over a stage
    weighted_perturbations :: W      # (pˢ, θ, Φ): forward-weighted pˢ′ and θ′, and the geopotential of θ′
    mass_flux_perturbation :: M      # (ρw, w): vertical mass flux of the velocity perturbation
    integrated_velocities :: I       # (u, v, w, ρw): time integral of the velocity perturbation that moved mass
end

@regional GravityWaveSubstepper

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a `GravityWaveSubstepper` for [`HydrostaticDynamics`](@ref) on `grid`, which integrates
the linear gravity-wave system with `inner_solver`.

`prognostic_fields` are those of the model. The perturbations of the surface pressure `pˢ`, the
velocities `u` and `v` and the potential temperature density `ρθ` are fields like them, so that
the velocity perturbations vanish on the same walls as the velocities. The other prognostic
fields are the scalars that the substeps do not advance, and that a time stepper transports
with the accumulated mass fluxes.

Keyword arguments
=================

- `substeps`: number of substeps in a time step ``Δt``; a stage that covers ``d Δt`` takes
  ``⌈d N⌉`` of them. With `nothing`, the default, each stage takes the smallest number for which
  its Courant number ``ω Δτ`` does not exceed `cfl`.
- `moist`: whether the linearization holds the density factor `ϑ` and the density potential
  temperature `θᵨ` of moist `HydrostaticDynamics`. Default: `false`.
- `cfl`: the stage Courant number aimed at when `substeps` is `nothing`. Default: 0.7 times the
  stability bound of `inner_solver` (see [`stage_cfl_bound`](@ref)), which may not be exceeded.
"""
function GravityWaveSubstepper(grid, inner_solver, prognostic_fields;
                               substeps = nothing,
                               cfl = nothing,
                               moist = false)

    FT = eltype(grid)
    bound = stage_cfl_bound(inner_solver)
    cfl = isnothing(cfl) ? 7 * bound / 10 : cfl

    0 < cfl ≤ bound ||
        throw(ArgumentError("The stage Courant number cfl = $cfl must be positive and may not exceed " *
                            "$bound, the stability bound of $(summary(inner_solver))."))

    (isnothing(substeps) || (substeps isa Integer && substeps ≥ 1)) ||
        throw(ArgumentError("substeps must be a positive integer or nothing, got $substeps."))

    center_field() = CenterField(grid)
    surface_field() = Field{Center, Center, Nothing}(grid)

    linearization = (ρ = center_field(),
                     θ = center_field(),
                     Π = center_field(),
                     Πᶠ = ZFaceField(grid),
                     dΠdpˢ = center_field(),
                     dΠᶠdpˢ = ZFaceField(grid),
                     dΦdpˢ = center_field())

    # With water the pressure-gradient force reads the density potential temperature θᵨ = ϑ θ: the
    # linearization also freezes ϑ and θᵨ (see `moist_hydrostatic_dynamics.jl`). Without it the
    # linearization, and every kernel that reads it, is that of dry air.
    moist && (linearization = merge(linearization, (ϑ = center_field(), θᵨ = center_field())))

    scalar_names = filter(name -> name ∉ gravity_wave_prognostic_names, keys(prognostic_fields))
    specific_scalars = NamedTuple{scalar_names}(map(name -> center_field(), scalar_names))

    # `similar` would drop the boundary conditions that hold the velocities at rest on walls
    like(field) = Field(instantiated_location(field), grid; boundary_conditions=field.boundary_conditions)
    u, v = prognostic_fields.u, prognostic_fields.v
    prognostic_like() = (pˢ = surface_field(), u = like(u), v = like(v), ρθ = center_field())
    perturbations = prognostic_like()
    forcing = prognostic_like()

    weighted_perturbations = (pˢ = surface_field(), θ = center_field(), Φ = center_field())
    mass_flux_perturbation = (ρw = ZFaceField(grid), w = ZFaceField(grid))
    integrated_velocities = (u = like(u), v = like(v), w = ZFaceField(grid), ρw = ZFaceField(grid))

    return GravityWaveSubstepper(inner_solver,
                                 substeps,
                                 convert(FT, cfl),
                                 convert(FT, maximum_horizontal_wavenumber(grid)),
                                 Centered(FT; order=2),
                                 linearization,
                                 specific_scalars,
                                 perturbations,
                                 forcing,
                                 weighted_perturbations,
                                 mass_flux_perturbation,
                                 integrated_velocities)
end

# Largest wavenumber of the second-order C-grid gradient and divergence: 2 / Δ in each
# horizontal direction that the grid has
function maximum_horizontal_wavenumber(grid)
    TX, TY, _ = topology(grid)
    kˣ = TX === Flat ? 0 : 2 / minimum_xspacing(grid)
    kʸ = TY === Flat ? 0 : 2 / minimum_yspacing(grid)
    return sqrt(kˣ^2 + kʸ^2)
end

#####
##### Substep count
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return an upper bound of the speed of the fastest gravity wave of `model`: the speed
``\\sqrt{γ Rᵈ T}`` of the Lamb wave of an isothermal atmosphere at the highest temperature of
the model. The external mode of the discrete hydrostatic column is slower than this.
"""
function maximum_gravity_wave_speed(model)
    constants = model.thermodynamic_constants
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
    T = maximum(model.temperature)
    return sqrt(γ * Rᵈ * T)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the number of substeps `Nτ` and the substep `Δτ` with which `substepper` covers a stage
of length `Δt_stage`, a fraction `stage_fraction` of the time step, when the fastest gravity
wave travels at `wave_speed`.

The stage Courant number is ``ω Δτ``, with ``ω`` the product of `wave_speed` and the largest
horizontal wavenumber of the grid. Without a prescribed number of substeps, `Nτ` is the
smallest number that keeps it below `substepper.cfl`. With one, `Nτ = ⌈stage_fraction × substeps⌉`,
and a stage Courant number above the stability bound of the inner solver is an error: the
substeps would amplify the fastest waves.
"""
function stage_substeps(substepper::GravityWaveSubstepper, Δt_stage, stage_fraction, wave_speed)
    ω = wave_speed * substepper.maximum_wavenumber
    Nτ = stage_substep_count(substepper.substeps, ω * Δt_stage, stage_fraction, substepper.cfl)
    Δτ = Δt_stage / Nτ

    bound = stage_cfl_bound(substepper.inner_solver)
    ω * Δτ ≤ bound ||
        throw(ArgumentError("The stage Courant number ω Δτ = $(ω * Δτ) of the gravity-wave substeps exceeds " *
                            "$bound, the stability bound of $(summary(substepper.inner_solver)), with $Nτ " *
                            "substeps in a stage of $Δt_stage s. Increase substeps or reduce the time step."))

    return Nτ, Δτ
end

stage_substep_count(::Nothing, stage_cfl, stage_fraction, cfl) = max(1, ceil(Int, stage_cfl / cfl))
stage_substep_count(substeps::Integer, stage_cfl, stage_fraction, cfl) = max(1, ceil(Int, stage_fraction * substeps - 1e-9))

#####
##### Linearization about the state at the start of the time step
#####

# Derivatives of the Exner functions of `_compute_layer_mass_and_exner_functions!` with respect
# to the surface pressure. On the z-faces p = A + B pˢ, so ∂Πᶠ/∂pˢ = κ B Πᶠ / p, which vanishes
# where B does, including at a model top of zero pressure. For the layer, with
# (1 + κ) Δp Π = p⁻ Πᶠ⁻ - p⁺ Πᶠ⁺ and ∂(p Πᶠ)/∂pˢ = (1 + κ) B Πᶠ,
#
#   ∂Π/∂pˢ = [B⁻ Πᶠ⁻ - B⁺ Πᶠ⁺ - ΔB Π] / Δp .
@kernel function _compute_exner_function_derivatives!(dΠdpˢ, dΠᶠdpˢ, grid, A, B, pˢ, Π, Πᶠ, κ)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        for k in 1:Nz+1
            p = interface_pressure(i, j, k, A, B, pˢ)
            Bᵏ = B[i, j, k]
            constant_pressure = Bᵏ == 0
            safe_p = ifelse(constant_pressure, one(p), p)
            dΠᶠdpˢ[i, j, k] = ifelse(constant_pressure, zero(p), κ * Bᵏ * Πᶠ[i, j, k] / safe_p)
        end

        for k in 1:Nz
            Δp = layer_pressure_thickness(i, j, k, A, B, pˢ)
            ΔB = B[i, j, k] - B[i, j, k+1]
            dΠdpˢ[i, j, k] = (B[i, j, k] * Πᶠ[i, j, k] - B[i, j, k+1] * Πᶠ[i, j, k+1] - ΔB * Π[i, j, k]) / Δp
        end
    end
end

@kernel function _compute_specific_scalar!(c, ρc, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds c[i, j, k] = ρc[i, j, k] / ρ[i, j, k]
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Freeze the state of `model` about which `substepper` linearizes the gravity-wave system: the
layer mass, the potential temperature and the Exner functions of every column, with the
derivatives of the Exner functions and of the geopotential with respect to the surface
pressure, and the specific value of every prognostic scalar that the substeps do not advance.
Called at the start of a time step, when the diagnostics of `model` are those of its state.
"""
function freeze_gravity_wave_linearization!(substepper::GravityWaveSubstepper, model)
    dynamics = model.dynamics
    linearization = substepper.linearization

    copy_parent!(linearization.ρ, dynamics.density)
    copy_parent!(linearization.θ, model.formulation.potential_temperature)
    copy_parent!(linearization.Π, dynamics.exner_function)
    copy_parent!(linearization.Πᶠ, dynamics.interface_exner_function)
    freeze_moist_linearization!(substepper, model)

    @apply_regionally launch_linearization_derivatives!(substepper, model)

    fill_halo_regions!(linearization.dΠdpˢ)
    fill_halo_regions!(linearization.dΦdpˢ)

    for (c, ρc) in zip(substepper.specific_scalars, substepped_scalars(substepper, model))
        @apply_regionally launch!(model.grid.architecture, model.grid, :xyz, _compute_specific_scalar!, c, ρc, linearization.ρ)
        fill_halo_regions!(c)
    end

    return nothing
end

# Dry air has no density factor to freeze. The method of a moist model is in `moist_hydrostatic_dynamics.jl`.
freeze_moist_linearization!(substepper, model) = nothing

# The potential temperature that the pressure-gradient force and the hydrostatic integral read
# from the frozen state: θᴸ in dry air, the density potential temperature θᵨᴸ with water
pressure_gradient_potential_temperature(linearization) =
    hasproperty(linearization, :θᵨ) ? linearization.θᵨ : linearization.θ

# ϑᴸ = θᵨᴸ / θᴸ, or nothing in dry air
density_factor(linearization) = hasproperty(linearization, :ϑ) ? linearization.ϑ : nothing

# The perturbation ϑᴸ θ′ of the density potential temperature at frozen ϑ; θ′ itself in dry air
@inline density_potential_temperature_perturbation(i, j, k, ::Nothing, θ′) = θ′
@inline density_potential_temperature_perturbation(i, j, k, ϑᴸ, θ′) = @inbounds ϑᴸ[i, j, k] * θ′

# The prognostic scalars that the substeps do not advance, in the order of `specific_scalars`
function substepped_scalars(substepper, model)
    names = keys(substepper.specific_scalars)
    return NamedTuple{names}(map(name -> Oceananigans.prognostic_fields(model)[name], names))
end

launch_linearization_derivatives!(substepper, model) =
    launch_linearization_derivatives!(substepper, model, model.dynamics.moisture)

# Dry air. The method of a moist model is in `moist_hydrostatic_dynamics.jl`.
function launch_linearization_derivatives!(substepper, model, ::Nothing)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = dry_air_gas_constant(constants) / cᵖᵈ
    (; θ, Π, Πᶠ, dΠdpˢ, dΠᶠdpˢ, dΦdpˢ) = substepper.linearization

    launch!(arch, grid, :xy, _compute_exner_function_derivatives!, dΠdpˢ, dΠᶠdpˢ, grid,
            dynamics.pressure_coefficient, dynamics.surface_pressure_coefficient,
            dynamics.surface_pressure, Π, Πᶠ, κ)

    # ∂Φ/∂pˢ at fixed θ: the hydrostatic integral is linear in the Exner functions
    launch!(arch, grid, :xy, _compute_geopotential!, dΦdpˢ, grid, ZeroField(eltype(grid)), θ, dΠdpˢ, dΠᶠdpˢ, cᵖᵈ)

    return nothing
end

#####
##### The linear gravity-wave operator
#####

# ∂ρ/∂pˢ of layer `k`: the layer mass ρ = (ΔA + ΔB pˢ) / (g Δz) is linear in pˢ
@inline layer_mass_derivative(i, j, k, grid, B, g) = @inbounds (B[i, j, k] - B[i, j, k+1]) / (g * Δzᶜᶜᶜ(i, j, k, grid))

# θ′ of the perturbation (pˢ′, ρθ′) of θ = ρθ / ρ, given the value `ρθ′` at (i, j, k)
@inline potential_temperature_perturbation(i, j, k, grid, ρθ′, pˢ′, ρᴸ, θᴸ, B, g) =
    @inbounds (ρθ′ - θᴸ[i, j, k] * layer_mass_derivative(i, j, k, grid, B, g) * pˢ′[i, j, 1]) / ρᴸ[i, j, k]

# Three-dimensional fields that are a profile of the frozen column times the surface pressure
# perturbation, evaluated where they are read rather than stored.
struct ExnerFunctionPerturbation{D, P}
    dΠdpˢ :: D
    pˢ′ :: P
end

struct GeopotentialPerturbation{F, D, P}
    Φ′ :: F     # hydrostatic integral of θ′ over the frozen Exner functions
    dΦdpˢ :: D
    pˢ′ :: P
end

Base.@propagate_inbounds Base.getindex(Π′::ExnerFunctionPerturbation, i, j, k) = Π′.dΠdpˢ[i, j, k] * Π′.pˢ′[i, j, 1]
Base.@propagate_inbounds Base.getindex(Φ′::GeopotentialPerturbation, i, j, k) = Φ′.Φ′[i, j, k] + Φ′.dΦdpˢ[i, j, k] * Φ′.pˢ′[i, j, 1]

@inline fast_surface_pressure_tendency(i, j, grid, ρᴸ, u′, v′, g) = surface_pressure_tendency(i, j, grid, ρᴸ, u′, v′, g)

@inline fast_potential_temperature_tendency(i, j, k, grid, advection, ρᴸ, 𝐮′, ρw′, θᴸ, Πᴸ, Πᶠᴸ) =
    potential_temperature_flux_convergence(i, j, k, grid, advection, ρᴸ, 𝐮′, ρw′, θᴸ, Πᴸ, Πᶠᴸ)

# The pressure-gradient force is bilinear in θ and (Π, Φ), so its linearization is the force of
# θ′ over the frozen Exner function with the geopotential perturbation, plus the force of the
# frozen θ over the Exner function perturbation.
@inline function fast_x_velocity_tendency(i, j, k, grid, θ′, pˢ′, Φ′, θᴸ, Πᴸ, dΠdpˢ, dΦdpˢ, cᵖᵈ)
    Π′ = ExnerFunctionPerturbation(dΠdpˢ, pˢ′)
    Φ′ = GeopotentialPerturbation(Φ′, dΦdpˢ, pˢ′)
    return ( - x_pressure_gradient_acceleration(i, j, k, grid, θ′, Πᴸ, Φ′, cᵖᵈ)
             - x_pressure_gradient_acceleration(i, j, k, grid, θᴸ, Π′, ZeroField(eltype(grid)), cᵖᵈ))
end

@inline function fast_y_velocity_tendency(i, j, k, grid, θ′, pˢ′, Φ′, θᴸ, Πᴸ, dΠdpˢ, dΦdpˢ, cᵖᵈ)
    Π′ = ExnerFunctionPerturbation(dΠdpˢ, pˢ′)
    Φ′ = GeopotentialPerturbation(Φ′, dΦdpˢ, pˢ′)
    return ( - y_pressure_gradient_acceleration(i, j, k, grid, θ′, Πᴸ, Φ′, cᵖᵈ)
             - y_pressure_gradient_acceleration(i, j, k, grid, θᴸ, Π′, ZeroField(eltype(grid)), cᵖᵈ))
end

#####
##### Substep kernels
#####

# Surface pressure perturbation over Δτ, and its forward-weighted value pˢ★ = pˢ′ + βᵈ (pˢ′ - pˢ′⁻)
@kernel function _step_surface_pressure_perturbation!(pˢ′, pˢ★, grid, ρᴸ, u′, v′, Fpˢ, g, Δτ, βᵈ)
    i, j = @index(Global, NTuple)

    @inbounds begin
        Gpˢ = fast_surface_pressure_tendency(i, j, grid, ρᴸ, u′, v′, g) + Fpˢ[i, j, 1]
        pˢ′[i, j, 1] += Δτ * Gpˢ
        pˢ★[i, j, 1] = pˢ′[i, j, 1] + βᵈ * Δτ * Gpˢ
    end
end

# Potential temperature density perturbation over Δτ, the forward-weighted θ★ that the
# pressure-gradient force reads, and the velocity perturbation that moved the mass.
@kernel function _step_potential_temperature_perturbation!(ρθ′, θ★, ∫u′, ∫v′, grid, advection, ρᴸ, 𝐮′, ρw′,
                                                           θᴸ, ϑᴸ, Πᴸ, Πᶠᴸ, pˢ★, B, Fρθ, g, Δτ, βᵈ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Gρθ = fast_potential_temperature_tendency(i, j, k, grid, advection, ρᴸ, 𝐮′, ρw′, θᴸ, Πᴸ, Πᶠᴸ) + Fρθ[i, j, k]
        ρθ′[i, j, k] += Δτ * Gρθ
        ρθ★ = ρθ′[i, j, k] + βᵈ * Δτ * Gρθ
        θ′ = potential_temperature_perturbation(i, j, k, grid, ρθ★, pˢ★, ρᴸ, θᴸ, B, g)
        θ★[i, j, k] = density_potential_temperature_perturbation(i, j, k, ϑᴸ, θ′)

        ∫u′[i, j, k] += Δτ * 𝐮′.u[i, j, k]
        ∫v′[i, j, k] += Δτ * 𝐮′.v[i, j, k]
    end
end

@kernel function _step_velocity_perturbations!(u′, v′, grid, pressure_args, Fu, Fv, Δτ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        u′[i, j, k] += Δτ * (fast_x_velocity_tendency(i, j, k, grid, pressure_args...) + Fu[i, j, k])
        v′[i, j, k] += Δτ * (fast_y_velocity_tendency(i, j, k, grid, pressure_args...) + Fv[i, j, k])
    end
end

#####
##### Substeps
#####

pressure_gradient_arguments(substepper, model) =
    (substepper.weighted_perturbations.θ,
     substepper.weighted_perturbations.pˢ,
     substepper.weighted_perturbations.Φ,
     pressure_gradient_potential_temperature(substepper.linearization),
     substepper.linearization.Π,
     substepper.linearization.dΠdpˢ,
     substepper.linearization.dΦdpˢ,
     model.thermodynamic_constants.dry_air.heat_capacity)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Advance the surface pressure and potential temperature density perturbations of `substepper`
over `Δτ` with the mass fluxes of its velocity perturbation, and diagnose what the
pressure-gradient force reads: the perturbations weighted forward by `βᵈ`, and the geopotential
of the potential temperature perturbation. With `Δτ = 0` this diagnoses them for the
perturbations as they are.
"""
function step_mass_and_potential_temperature!(substepper::GravityWaveSubstepper, model, Δτ, βᵈ)
    @apply_regionally launch_mass_and_potential_temperature_step!(substepper, model, Δτ, βᵈ)

    weighted = substepper.weighted_perturbations
    fill_halo_regions!(weighted.pˢ)
    fill_halo_regions!(weighted.θ)

    @apply_regionally launch_geopotential_perturbation!(substepper, model)
    fill_halo_regions!(weighted.Φ)

    return nothing
end

function launch_mass_and_potential_temperature_step!(substepper, model, Δτ, βᵈ)
    grid = model.grid
    arch = grid.architecture
    B = model.dynamics.surface_pressure_coefficient
    g = model.thermodynamic_constants.gravitational_acceleration
    FT = eltype(grid)
    Δτ = kernel_time_step(arch, grid, Δτ)
    βᵈ = convert(FT, βᵈ)

    (; ρ, θ, Π, Πᶠ) = substepper.linearization
    pˢ′, u′, v′, ρθ′ = substepper.perturbations
    F = substepper.forcing
    weighted = substepper.weighted_perturbations
    (; ρw, w) = substepper.mass_flux_perturbation
    ∫𝐮′ = substepper.integrated_velocities

    launch!(arch, grid, :xy, _compute_vertical_mass_flux!, ρw, w, grid, B, ρ, u′, v′)

    launch!(arch, grid, :xy, _step_surface_pressure_perturbation!,
            pˢ′, weighted.pˢ, grid, ρ, u′, v′, F.pˢ, g, Δτ, βᵈ)

    launch!(arch, grid, :xyz, _step_potential_temperature_perturbation!,
            ρθ′, weighted.θ, ∫𝐮′.u, ∫𝐮′.v, grid, substepper.advection, ρ, (; u=u′, v=v′), ρw,
            θ, density_factor(substepper.linearization), Π, Πᶠ, weighted.pˢ, B, F.ρθ, g, Δτ, βᵈ)

    return nothing
end

# The hydrostatic integral of the full state, applied to θ′ over the frozen Exner functions
function launch_geopotential_perturbation!(substepper, model)
    grid = model.grid
    weighted = substepper.weighted_perturbations
    (; Π, Πᶠ) = substepper.linearization
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity

    launch!(grid.architecture, grid, :xy, _compute_geopotential!,
            weighted.Φ, grid, ZeroField(eltype(grid)), weighted.θ, Π, Πᶠ, cᵖᵈ)

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Advance the velocity perturbations of `substepper` over `Δτ` with the linearized
pressure-gradient force of the perturbations diagnosed by
[`step_mass_and_potential_temperature!`](@ref).
"""
function step_velocities!(substepper::GravityWaveSubstepper, model, Δτ)
    @apply_regionally launch_velocity_step!(substepper, model, Δτ)
    fill_halo_regions!((substepper.perturbations.u, substepper.perturbations.v))
    return nothing
end

function launch_velocity_step!(substepper, model, Δτ)
    grid = model.grid
    arch = grid.architecture
    (; u, v) = substepper.perturbations
    F = substepper.forcing

    launch!(arch, grid, :xyz, _step_velocity_perturbations!, u, v, grid,
            pressure_gradient_arguments(substepper, model), F.u, F.v, kernel_time_step(arch, grid, Δτ))

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Integrate the gravity-wave system of `substepper` over `Nτ` substeps of `Δτ` with its inner
solver, from the perturbations and with the forcing that it holds. The halos of the velocity
perturbations must be filled. The velocity perturbation that moves mass is added, weighted by
the substep, to the integrated velocities.
"""
gravity_wave_substep_loop!(substepper::GravityWaveSubstepper, model, Nτ, Δτ) =
    gravity_wave_substep_loop!(substepper.inner_solver, substepper, model, Nτ, Δτ)

function gravity_wave_substep_loop!(solver::ForwardBackwardSubsteps, substepper, model, Nτ, Δτ)
    βᵈ = solver.divergence_damping

    for _ in 1:Nτ
        step_mass_and_potential_temperature!(substepper, model, Δτ, βᵈ)
        step_velocities!(substepper, model, Δτ)
    end

    return nothing
end

# Half a substep of (pˢ′, ρθ′) on either side of each velocity substep; the two half substeps
# between consecutive velocity substeps read the same velocity and are taken as one.
function gravity_wave_substep_loop!(::VerletSubsteps, substepper, model, Nτ, Δτ)
    step_mass_and_potential_temperature!(substepper, model, Δτ / 2, 0)

    for n in 1:Nτ
        step_velocities!(substepper, model, Δτ)
        step_mass_and_potential_temperature!(substepper, model, n == Nτ ? Δτ / 2 : Δτ, 0)
    end

    return nothing
end

#####
##### The fast part of a tendency
#####

@kernel function _subtract_fast_surface_pressure_tendency!(Gpˢ, grid, ρᴸ, u′, v′, g)
    i, j = @index(Global, NTuple)
    @inbounds Gpˢ[i, j, 1] -= fast_surface_pressure_tendency(i, j, grid, ρᴸ, u′, v′, g)
end

@kernel function _subtract_fast_tendencies!(Gu, Gv, Gρθ, grid, pressure_args, advection, ρᴸ, 𝐮′, ρw′, θᴸ, Πᴸ, Πᶠᴸ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Gu[i, j, k] -= fast_x_velocity_tendency(i, j, k, grid, pressure_args...)
        Gv[i, j, k] -= fast_y_velocity_tendency(i, j, k, grid, pressure_args...)
        Gρθ[i, j, k] -= fast_potential_temperature_tendency(i, j, k, grid, advection, ρᴸ, 𝐮′, ρw′, θᴸ, Πᴸ, Πᶠᴸ)
    end
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Subtract from the tendencies `G` of the surface pressure, the velocities and the potential
temperature density the linear gravity-wave operator applied to the perturbations that
`substepper` holds, with the kernel functions that the substeps integrate. Given the full
tendencies of the state these perturbations belong to, this leaves the slow tendencies.
"""
function subtract_fast_tendencies!(G, substepper::GravityWaveSubstepper, model)
    # The mass flux and the pressure perturbations of the perturbations as they are
    step_mass_and_potential_temperature!(substepper, model, 0, 0)
    @apply_regionally launch_fast_tendency_subtraction!(G, substepper, model)
    return nothing
end

function launch_fast_tendency_subtraction!(G, substepper, model)
    grid = model.grid
    arch = grid.architecture
    g = model.thermodynamic_constants.gravitational_acceleration
    (; ρ, θ, Π, Πᶠ) = substepper.linearization
    (; u, v) = substepper.perturbations
    ρw = substepper.mass_flux_perturbation.ρw

    launch!(arch, grid, :xy, _subtract_fast_surface_pressure_tendency!, G.pˢ, grid, ρ, u, v, g)

    launch!(arch, grid, :xyz, _subtract_fast_tendencies!, G.u, G.v, G.ρθ, grid,
            pressure_gradient_arguments(substepper, model), substepper.advection, ρ, (; u, v), ρw, θ, Π, Πᶠ)

    return nothing
end

#####
##### Transport of the other scalars with the accumulated mass fluxes
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Diagnose, from continuity, the vertical mass flux and velocity that belong to the integrated
horizontal velocity perturbations of `substepper` on the frozen layer mass. The three
components then carry the mass that the surface pressure perturbation accounts for.
"""
function compute_integrated_vertical_velocity!(substepper::GravityWaveSubstepper, model)
    ∫𝐮′ = substepper.integrated_velocities
    fill_halo_regions!((∫𝐮′.u, ∫𝐮′.v))

    grid = model.grid
    B = model.dynamics.surface_pressure_coefficient
    ρᴸ = substepper.linearization.ρ

    @apply_regionally launch!(grid.architecture, grid, :xy, _compute_vertical_mass_flux!,
                              ∫𝐮′.ρw, ∫𝐮′.w, grid, B, ρᴸ, ∫𝐮′.u, ∫𝐮′.v)

    fill_halo_regions!(∫𝐮′.w)
    fill_coordinate_mass_flux_halos!(grid, ∫𝐮′.ρw)

    return nothing
end

# The transport tendency of a scalar in the units in which a time stepper advances it: `div_ρUc`,
# per unit volume, on most grids, and the rate of change of the mass per unit area of the scalar,
# exactly, on a pressure-level grid (methods in Breeze.Advection)
function layer_mass_flux_divergence end

# The increment is in the units of the stage: per unit volume, or the mass per unit area of a
# pressure-level grid (see `PressureLevelDiscretization.scalar_layer_mass`)
@kernel function _add_perturbation_transport!(δρc, grid, advection, ρᴸ, ∫𝐮′, cᴸ, sign)
    i, j, k = @index(Global, NTuple)
    @inbounds δρc[i, j, k] -= sign * layer_mass_flux_divergence(i, j, k, grid, advection, ρᴸ, ∫𝐮′, cᴸ)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Add to the increment `δρc` of the prognostic scalar `name` its transport by the integrated
velocity perturbations of `substepper`: the convergence of the flux of the scalar's frozen
specific value carried by the mass fluxes that changed the layer mass. For the potential
temperature the substeps apply the same fluxes one at a time; for a scalar that does not act on
the gravity waves, applying their sum once is the same.
"""
function add_perturbation_transport!(δρc, substepper::GravityWaveSubstepper, model, name, sign=1)
    grid = model.grid
    (; u, v, w) = substepper.integrated_velocities

    @apply_regionally launch!(grid.architecture, grid, :xyz, _add_perturbation_transport!,
                              δρc, grid, substepper.advection, substepper.linearization.ρ,
                              coordinate_transport_velocities(grid, u, v, w, substepper.integrated_velocities.ρw),
                              substepper.specific_scalars[name], convert(eltype(grid), sign))

    return nothing
end
