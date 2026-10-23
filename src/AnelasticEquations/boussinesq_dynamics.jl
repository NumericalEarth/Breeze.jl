#####
##### BoussinesqDynamics: anelastic dynamics about a uniform reference state
#####

"""
    BoussinesqDynamics

[`AnelasticDynamics`](@ref) expanded about a [`BoussinesqReferenceState`](@ref).

With a uniform reference density ``ρᵣ``, the anelastic constraint ``\\boldsymbol{\\nabla} ⋅ (ρᵣ \\boldsymbol{u}) = 0``
reduces to ``\\boldsymbol{\\nabla} ⋅ \\boldsymbol{u} = 0``, and the anelastic buoyancy reduces to the Boussinesq
buoyancy of virtual temperature anomalies. Because the reference pressure is uniform, the Exner function is
also uniform, and temperature does not decrease adiabatically with height within the domain.
These approximations are accurate for domains much shallower than the atmospheric scale height,
such as direct numerical simulations of cloud turbulence or laboratory convection.

On grids with a `Periodic` vertical direction, the pressure is computed with an FFT-based Poisson solver.

!!! warning "Mean buoyancy in triply-periodic domains"
    A periodic pressure field cannot balance a domain-mean buoyancy force, so a nonzero domain-mean
    buoyancy accelerates the domain-mean vertical momentum in a vertically-periodic domain.
    Setting `frame_vertical_velocity` (to zero, for a fixed frame) measures buoyancy relative to
    the domain mean, which removes the domain-mean buoyancy; see [`BoussinesqReferenceState`](@ref).
"""
const BoussinesqDynamics = AnelasticDynamics{<:BoussinesqReferenceState}

"""
$(TYPEDSIGNATURES)

Return [`BoussinesqDynamics`](@ref) on `grid`, expanded about a [`BoussinesqReferenceState`](@ref)
constructed from `constants` and `reference_state_kwargs`, such as `pressure` and `temperature`.

Examples
========

```jldoctest
julia> using Breeze

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic));

julia> dynamics = BoussinesqDynamics(grid, temperature=290)
BoussinesqDynamics(pᵣ=101325.0, Tᵣ=290.0, ρᵣ=1.2174)
└── pressure_anomaly: not materialized
```
"""
BoussinesqDynamics(grid, constants=ThermodynamicConstants(eltype(grid)); reference_state_kwargs...) =
    AnelasticDynamics(BoussinesqReferenceState(grid, constants; reference_state_kwargs...))

function Base.summary(dynamics::BoussinesqDynamics)
    ref = dynamics.reference_state
    return string("BoussinesqDynamics(pᵣ=", prettysummary(reference_value(ref.pressure)),
                  ", Tᵣ=", prettysummary(reference_value(ref.temperature)),
                  ", ρᵣ=", prettysummary(reference_value(ref.density)),
                  frame_summary(ref.frame), ")")
end

const BoussinesqModel = AtmosphereModel{<:BoussinesqDynamics}

#####
##### Pressure
#####

AtmosphereModels.pressure_anomaly(dynamics::BoussinesqDynamics) =
    reference_value(dynamics.reference_state.density) * dynamics.pressure_anomaly

AtmosphereModels.total_pressure(dynamics::BoussinesqDynamics) =
    reference_value(dynamics.reference_state.pressure) + pressure_anomaly(dynamics)

#####
##### Moving reference frames
#####

AtmosphereModels.reference_frame(dynamics::BoussinesqDynamics) = dynamics.reference_state.frame

function AtmosphereModels.advance_dynamics_state!(dynamics::BoussinesqDynamics, model)
    frame = dynamics.reference_state.frame
    validate_reference_frame(frame, model.formulation)
    advance_frame!(frame, model.clock.time, model.thermodynamic_constants)
    return nothing
end

validate_reference_frame(frame, formulation) = nothing
validate_reference_frame(::MovingFrame, ::LiquidIcePotentialTemperatureFormulation) = nothing

function validate_reference_frame(::MovingFrame, formulation)
    throw(ArgumentError("A moving reference frame requires the liquid-ice potential temperature formulation, " *
                        "but the model uses $(summary(formulation))."))
end

# The reference temperature and moisture of a moving frame follow the domain mean,
# so that buoyancy is measured relative to the domain mean.
AtmosphereModels.compute_auxiliary_dynamics_variables!(model::BoussinesqModel) =
    update_frame_means!(model.dynamics.reference_state.frame, model)

update_frame_means!(::Nothing, model) = nothing

function update_frame_means!(frame::MovingFrame, model)
    frame.temperature = mean(model.temperature)
    frame.vapor_mass_fraction = domain_mean(specific_humidity(model))
    frame.liquid_mass_fraction = domain_mean(liquid_mass_fraction(model))
    frame.ice_mass_fraction = domain_mean(ice_mass_fraction(model))
    return nothing
end

domain_mean(::Nothing) = 0
domain_mean(field) = mean(field)

#####
##### Pressure solver for vertically-periodic domains
#####

const ZPeriodicGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Periodic}

fft_poisson_solver_grid(grid) = grid
fft_poisson_solver_grid(grid::ImmersedBoundaryGrid) = grid.underlying_grid

# With a uniform reference density the pressure equation is a Poisson equation with constant coefficients,
# which is solved with FFTs in vertically-periodic domains.
AtmosphereModels.dynamics_pressure_solver(::BoussinesqDynamics, grid::ZPeriodicGrid) =
    FFTBasedPoissonSolver(fft_poisson_solver_grid(grid))

# The kinematic pressure ϕ = p′ / ρᵣ satisfies ∇²ϕ = ∇⋅(ρŨ) / (ρᵣ Δt) when ρᵣ is uniform.
# The FFT-based solver stores the source term in `solver.storage`.
function solve_for_anelastic_pressure!(ϕ, solver::FFTBasedPoissonSolver, ρŨ, Δt, reference_density)
    rhs = solver.storage
    arch = architecture(solver)
    grid = solver.grid
    kernel_Δt = kernel_time_step(arch, grid, Δt)
    launch!(arch, grid, :xyz, _compute_boussinesq_source_term!, rhs, grid, ρŨ, kernel_Δt, reference_density)
    solve!(ϕ, solver)
    return ϕ
end

@kernel function _compute_boussinesq_source_term!(rhs, grid, ρŨ, Δt, ρᵣ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    ρu, ρv, ρw = ρŨ
    δ = divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)
    @inbounds rhs[i, j, k] = active * δ / (ρᵣ[i, j, k] * Δt)
end
