# # Tropical cyclone world (Cronin & Chavas, 2019)
#
# This example follows the "tropical cyclone world" radiative–convective equilibrium
# experiment described by [Cronin & Chavas (2019)](https://doi.org/10.1175/JAS-D-18-0357.1),
# but implemented with `Breeze.AtmosphereModel`.  The goal is to couple a simple
# surface-layer bulk scheme (wind stress, sensible, and latent heat fluxes) to an
# anelastic atmosphere, apply a horizontally-uniform radiative cooling, and allow the
# model to organize convection and tropical cyclones on a doubly-periodic domain.
#
# The script is written in `Literate.jl` style: prose cells introduce each block of
# code, the physics it represents, and how it connects to the Cronin & Chavas setup.
# The numerical values below are close to those used in the paper, but whenever
# possible we pull thermodynamic constants directly from `Breeze.ThermodynamicConstants`
# rather than hard-coding numbers.

# ## Packages and utilities
#
# We rely on `Breeze` for the model, `Oceananigans` utilities for grids, `Random`
# for small perturbations, and `CairoMakie` for quick-look visualizations.

using Breeze
using Breeze.Thermodynamics:
    ThermodynamicConstants,
    ReferenceState,
    MoistureMassFractions,
    mixture_heat_capacity,
    saturation_specific_humidity

using Breeze.AtmosphereModels: AtmosphereModel
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Printf
using Random
using CairoMakie

# ## Domain, grid, and reference state
#
# Cronin & Chavas integrate over a square, doubly-periodic domain that spans a few
# thousand kilometers and reaches the lower stratosphere.  We adopt a 4 096 × 4 096 km
# horizontal box with a 25 km model top.  The "modest" resolution requested corresponds
# to 128 × 128 × 32 cells (≈32 km horizontally, ≈780 m vertically).  For CI or
# documentation builds we automatically reduce the resolution to keep runtimes low.

arch = CPU()                 # swap to GPU() when accelerators are available
Nx = Ny = 128
Nz = 32

if get(ENV, "CI", "false") == "true"
    Nx = Ny = 16
    Nz = 8
end

Lx = 4_096_000.0             # meters
Ly = 4_096_000.0
H  = 25_000.0
halo = (5, 5, 5)

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, H),
                       halo = halo,
                       topology = (Periodic, Periodic, Bounded))

reference_state = ReferenceState(grid; base_pressure=101325, potential_temperature=300)
formulation = AnelasticFormulation(reference_state)

# ## Surface exchange physics: drag, sensible, and latent heat/moisture fluxes
#
# The Cronin & Chavas case uses a bulk aerodynamic surface scheme.  The exchange
# coefficients below (drag `Cd`, sensible `Ch`, and moisture `Ce`) are close to their
# recommended values for a TC-permitting LES.  The functions return boundary fluxes for
# the prognostic variables (`ρu`, `ρv`, `ρe`, `ρqᵗ`) and follow Oceananigans' sign
# convention: positive fluxes at the bottom boundary add that quantity to the interior.

# Compute surface specific humidity and density
# TODO: move this to source code
thermo = reference_state.thermodynamics
T₀ = 302
p₀ = reference_state.base_pressure
pᵛ⁺₀ = saturation_vapor_pressure(T₀, thermo, thermo.liquid)
Rᵛ = vapor_gas_constant(thermo)
Rᵈ = dry_air_gas_constant(thermo)
ϵᵛᵈ = Rᵛ / Rᵈ - 1
qᵛ⁺₀ = pᵛ⁺₀ / (p₀ - ϵᵛᵈ * pᵛ⁺₀)
q₀ = MoistureMassFractions(qᵛ⁺₀, zero(qᵛ⁺₀), zero(qᵛ⁺₀))
ρ₀ = density(p₀, T₀, q₀, thermo)

surface_exchange = (; ρ₀, T₀, qᵛ⁺₀,
                    Cd = 1.5e-3,
                    Ch = 1.1e-3,
                    Ce = 1.1e-3,
                    𝕌₀ = 0.5,
                    ℒˡᵣ = thermo.liquid.reference_latent_heat,
                    cᵖᵈ = thermo.dry_air.heat_capacity,
                    thermo)

@inline function surface_speed(ρu, ρv, params)
    u = ρu / params.ρₛ
    v = ρv / params.ρₛ
    s = sqrt(u^2 + v^2 + params.𝕌₀^2)
    return s, u, v
end

@inline function drag_flux_x(x, y, t, ρu, ρv, params)
    s, u, _ = surface_speed(ρu, ρv, params)
    τx = params.ρ₀ * params.Cd * s * u
    return -τx
end

@inline function drag_flux_y(x, y, t, ρu, ρv, params)
    s, _, v = surface_speed(ρu, ρv, params)
    τy = params.ρ₀ * params.Cd * s * v
    return -τy
end

@inline function surface_moisture_flux(x, y, t, ρqᵗ, ρu, ρv, params)
    s, _, _ = surface_speed(ρu, ρv, params)
    qᵗ = ρqᵗ / params.ρ₀
    Δq = max(0, params.qᵛ⁺₀ - qᵗ)
    return params.ρ₀ * params.Ce * s * (params.qᵛ⁺₀ - qᵗ)
end

@inline function surface_energy_flux(x, y, t, ρe, ρqᵗ, ρu, ρv, params)
    s, _, _ = surface_speed(ρu, ρv, params)
    qᵗ = ρqᵗ / params.ρ₀
    q = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))
    cᵖᵐ = mixture_heat_capacity(q, params.thermo)
    # ρe = ρᵣ (cᵖᵐ T + qᵗ Lᵥ) at z ≈ 0 ⇒ T = (ρe/ρᵣ - qᵗ Lᵥ) / cᵖᵐ
    Tₐ = (ρe / params.ρ₀ - qᵗ * params.ℒˡᵣ) / cᵖᵐ
    sensible = params.ρ₀ * params.cᵖᵈ * params.Ch * s * (params.T₀ - Tₐ)
    moisture_flux = surface_moisture_flux(x, y, t, ρqᵗ, ρu, ρv, params)
    latent = params.ℒˡᵣ * moisture_flux
    return sensible + latent
end

ρu_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(drag_flux_x,
                                                                field_dependencies = (:ρu, :ρv),
                                                                parameters = surface_exchange))
ρv_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(drag_flux_y,
                                                                field_dependencies = (:ρu, :ρv),
                                                                parameters = surface_exchange))
ρq_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(surface_moisture_flux,
                                                                field_dependencies = (:ρqᵗ, :ρu, :ρv),
                                                                parameters = surface_exchange))
ρe_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(surface_energy_flux,
                                                                field_dependencies = (:ρe, :ρqᵗ, :ρu, :ρv),
                                                                parameters = surface_exchange))

boundary_conditions = (; ρu = ρu_bcs,
                        ρv = ρv_bcs,
                        ρqᵗ = ρq_bcs,
                        ρe = ρe_bcs)

# ## Radiative forcing
#
# Cronin & Chavas impose a vertically varying radiative cooling (≈ -1.5 K day⁻¹ near
# cloud base that tapers with height).  We encode this forcing as an energy tendency
# that will be added directly to the `ρe` equation after the model is constructed.

using Oceananigans.Units: day

dt_Tˢ = -1.5 / day
λᴿ = 12_000
ρᵣ = reference_state.density
cᵖᵈ = thermo.dry_air.heat_capacity
dt_Tᴿ = CenterField(grid)
dt_Tᴿ_func = ρᵣ * cᵖᵈ * dt_Tˢ * exp(-z / λᴿ)

set!(dt_Tᴿ, dt_Tᴿ_func)
radiative_forcing = Forcing(dt_Tᴿ_func)

# ## Build the AtmosphereModel
#
# We use 9th-order WENO advection, an f-plane with mid-latitude Coriolis parameter,
# and no microphysics for now (warm-rain processes would be the next upgrade).

coriolis = FPlane(f = 5e-5)
advection = WENO(order=9)

model = AtmosphereModel(grid;
                        thermodynamics = thermo,
                        formulation,
                        boundary_conditions,
                        coriolis,
                        advection)

model.forcing = merge(model.forcing, (; ρe = radiative_forcing))

# ## Initial conditions
#
# We start from a weakly stratified reference potential temperature profile, a moist
# boundary layer that decays exponentially with height, and small random velocity and
# moisture perturbations to seed convection.  The helper below keeps the perturbations
# reproducible.

Random.seed!(42)
N² = 1e-4
g = thermo.gravitational_acceleration
dθdz = N² * θ_ref / g
q_surface = 0.018
q_scale_height = 3_000.0

θ_profile(z) = θ_ref + dθdz * z
q_profile(z) = q_surface * exp(-z / q_scale_height)

δθ = 0.5
δq = 1e-4
δu = 0.5

θᵢ(x, y, z) = θ_profile(z) + δθ * randn()
qᵢ(x, y, z) = max(q_profile(z) + δq * randn(), 0)
Ξᵢ(x, y, z) = δu * randn()

set!(model, θ = θᵢ, qᵗ = qᵢ, u = Ξᵢ, v = Ξᵢ, w = (x, y, z) -> 0.0)

# ## Simulation control and diagnostics
#
# We advance the system for ~100 time steps on CPU to validate the setup quickly.
# The CFL wizard keeps the model stable, and a simple callback prints thermodynamic
# and kinematic extrema so long runs can be monitored.

Δt = 1.0
stop_iteration = 100

if get(ENV, "CI", "false") == "true"
    stop_iteration = 20
end

simulation = Simulation(model; Δt, stop_iteration)
conjure_time_step_wizard!(simulation, cfl = 0.6)

ρe = model.energy
u, v, w = model.velocities

function progress(sim)
    ρemin = minimum(ρe)
    ρemax = maximum(ρe)
    u_max = maximum(abs, u)
    v_max = maximum(abs, v)
    w_max = maximum(abs, w)

    msg = @sprintf("Iter: %5d, t: %8.2fs, extrema(ρe) = (%.3e, %.3e) J/m³, max|u,v,w| = (%.2f, %.2f, %.2f) m/s",
                   iteration(sim), sim.model.clock.time,
                   ρemin, ρemax, u_max, v_max, w_max)
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# ## Output writers
#
# We capture surface slices and mid-plane sections of vertical velocity, energy, and
# vertical vorticity for documentation plots.  The file lives inside `examples/` so it
# can be inspected manually after the script finishes.

ζ = ∂x(v) - ∂y(u)
outputs = (; u, v, w, ζ, ρe)
output_path = joinpath(@__DIR__, "tropical_cyclone_world_$(Nx)x$(Ny)x$(Nz).jld2")
schedule = IterationInterval(max(1, stop_iteration ÷ 10))

simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                              filename = output_path,
                                              schedule,
                                              overwrite_existing = true)

@info "Starting tropical cyclone world spin-up on $(summary(grid))"
run!(simulation)
@info "Finished spin-up, output saved to $(output_path)"

# ## Quick-look visualization
#
# We make horizontal (surface) and vertical (mid-plane) slices of the vorticity,
# vertical velocity, and moist static energy.  The energy slice is a placeholder until
# a dedicated potential-temperature diagnostic is exposed.

if get(ENV, "CI", "false") == "false" && isfile(output_path)
    ζt = FieldTimeSeries(output_path, "ζ")
    wt = FieldTimeSeries(output_path, "w")
    ρet = FieldTimeSeries(output_path, "ρe")

    n = length(ζt)
    ζ_xy = Array(interior(ζt[n], :, :, 1))
    w_xy = Array(interior(wt[n], :, :, 1))
    y_index = size(w_xy, 2) ÷ 2
    w_xz = Array(interior(wt[n], :, y_index, :))
    ρe_xz = Array(interior(ρet[n], :, y_index, :))

    x = xnodes(ζt) ./ 1000  # km
    y = ynodes(ζt) ./ 1000
    z = znodes(ρet) ./ 1000

    fig = Figure(size = (1100, 900), fontsize = 12)
    ax1 = Axis(fig[1, 1], title = "Surface ζ", xlabel = "x (km)", ylabel = "y (km)")
    ax2 = Axis(fig[1, 2], title = "Surface w", xlabel = "x (km)", ylabel = "y (km)")
    ax3 = Axis(fig[2, 1], title = "Mid-plane w", xlabel = "x (km)", ylabel = "z (km)")
    ax4 = Axis(fig[2, 2], title = "Mid-plane ρe", xlabel = "x (km)", ylabel = "z (km)")

    ζ_lim = maximum(abs, ζ_xy)
    w_lim_surface = maximum(abs, w_xy)
    w_lim_mid = maximum(abs, w_xz)
    ζ_map = heatmap!(ax1, x, y, ζ_xy';
                     colormap = :balance, colorrange = (-ζ_lim, ζ_lim))
    w_surface = heatmap!(ax2, x, y, w_xy';
                         colormap = :balance, colorrange = (-w_lim_surface, w_lim_surface))

    w_mid_map = heatmap!(ax3, x, z, w_xz';
                         colormap = :balance, colorrange = (-w_lim_mid, w_lim_mid))
    ρe_map = heatmap!(ax4, x, z, ρe_xz';
                      colormap = :thermal)

    Colorbar(fig[1, 3], ζ_map, label = "ζ (s⁻¹)")
    Colorbar(fig[1, 4], w_surface, label = "w (m s⁻¹)")
    Colorbar(fig[2, 3], w_mid_map, label = "w (m s⁻¹)")
    Colorbar(fig[2, 4], ρe_map, label = "ρe (J m⁻³)")

    png_path = replace(output_path, ".jld2" => "_slices.png")
    save(png_path, fig)
    @info "Saved slice visualization to $(png_path)"
end
