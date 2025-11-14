# # Prescribed sea surface temperature convection
#
# This example simulates moist convection driven by a prescribed sea surface temperature (SST).
# The simulation uses bulk aerodynamic formulas to compute surface fluxes of momentum,
# sensible heat, and latent heat based on bulk transfer coefficients and friction velocity.
# The model uses zero-moment bulk microphysics from CloudMicrophysics, which instantly
# removes precipitable condensate above a threshold, providing a simple representation
# of precipitation processes.

using Breeze
using Oceananigans.Units
using Printf
using CloudMicrophysics

# Load CloudMicrophysics extension
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: ZeroMomentCloudMicrophysics

# ## Grid setup
#
# We use a 2D domain with periodic horizontal boundaries and a bounded vertical domain.
# The domain extends 20 km horizontally and 10 km vertically.

Nx = 128
Nz = 128
Lz = 10 * 1024  # 10 km domain height
grid = RectilinearGrid(size = (Nx, Nz),
                       x = (-10e3, 10e3),
                       z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

# ## Model setup
#
# We create an AtmosphereModel with zero-moment bulk microphysics. The reference state
# uses a base pressure p₀ = 101325 Pa and reference potential temperature θ₀ = 285 K,
# typical of the lower atmosphere.

p₀ = 101325 # Pa
θ₀ = 285 # K

advection = WENO()
microphysics = ZeroMomentCloudMicrophysics()
reference_state = ReferenceState(grid, base_pressure=p₀, potential_temperature=θ₀)
formulation = AnelasticFormulation(reference_state)
thermodynamics = ThermodynamicConstants(eltype(grid))

# Get reference density and heat capacity for flux calculations
ρ₀ = reference_state.density[1, 1, 1]  # Reference density at surface
cᵖᵈ = thermodynamics.dry_air.heat_capacity

# ## Surface flux parameters
#
# We define bulk transfer coefficients (Cᴰ for drag, Cᴴ for heat, Cᵛ for vapor)
# and a prescribed sea surface temperature θˢ = 295 K (10 K warmer than θ₀).
# The fluxes are computed using bulk aerodynamic formulas with friction velocity u★.

parameters = (; 
    drag_coefficient = 1e-3,      # Cᴰ: drag coefficient
    heat_transfer_coefficient = 1e-3,  # Cᴴ: heat transfer coefficient
    vapor_transfer_coefficient = 1e-3,  # Cᵛ: vapor transfer coefficient
    sea_surface_temperature = θ₀ + 10,  # θˢ = 295 K
    gust_speed = 1e-2,  # Minimum friction velocity u★ (m/s)
    ρ₀,  # Reference density at surface
    cᵖᵈ,  # Heat capacity of dry air
    thermodynamics,
    reference_state
)

# ## Boundary condition functions
#
# The boundary conditions compute surface fluxes using bulk aerodynamic formulas.
# We need to convert temperature and moisture fluxes to energy density and moisture
# density fluxes for AtmosphereModel.

# Utility for computing saturation specific humidity at sea surface
@inline function surface_saturation_specific_humidity(T, ρ, thermo)
    return Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, thermo, Breeze.Thermodynamics.PlanarLiquidSurface())
end

# Friction velocity based on wind speed and drag coefficient
# The friction velocity u★ is computed from the bulk drag law: u★² = Cᴰ U²
@inline function friction_velocity(i, j, grid, clock, fields, parameters)
    Cᴰ = parameters.drag_coefficient
    u = fields.u[i, j, 1]
    v = fields.v[i, j, 1]
    # Stationary ocean: wind speed relative to ocean surface
    U = sqrt(u^2 + v^2)
    u★ = sqrt(Cᴰ * U^2) + parameters.gust_speed
    return u★
end

# Momentum flux (x-direction)
# Returns kinematic momentum flux τₓ/ρ₀ = -u★² u/U for the ρu boundary condition
@inline function x_momentum_flux(i, j, grid, clock, fields, parameters)
    u = fields.u[i, j, 1]
    v = fields.v[i, j, 1]
    u★ = friction_velocity(i, j, grid, clock, fields, parameters)
    U = sqrt(u^2 + v^2)
    # Kinematic momentum flux (m²/s²)
    return - u★^2 * u / U * (U > 0)
end

# Momentum flux (y-direction)
# Returns kinematic momentum flux τᵧ/ρ₀ = -u★² v/U for the ρv boundary condition
@inline function y_momentum_flux(i, j, grid, clock, fields, parameters)
    u = fields.u[i, j, 1]
    v = fields.v[i, j, 1]
    u★ = friction_velocity(i, j, grid, clock, fields, parameters)
    U = sqrt(u^2 + v^2)
    # Kinematic momentum flux (m²/s²)
    return - u★^2 * v / U * (U > 0)
end

# Energy density flux (converted from potential temperature flux)
# The sensible heat flux is computed using bulk transfer: Jθ = -u★ θ★
# where θ★ is the temperature scale computed from the bulk transfer coefficient
@inline function energy_density_flux(i, j, grid, clock, fields, parameters)
    u★ = friction_velocity(i, j, grid, clock, fields, parameters)
    θˢ = parameters.sea_surface_temperature
    Cᴰ = parameters.drag_coefficient
    Cᴴ = parameters.heat_transfer_coefficient
    
    # Get temperature from fields (approximate θ ≈ T near surface)
    T = fields.T[i, j, 1]
    θ = T  # Approximation valid near surface
    Δθ = θ - θˢ
    
    # Temperature scale from bulk transfer: u★ θ★ = Cᴴ U Δθ
    # Using U ≈ u★ / √Cᴰ for neutral conditions
    θ★ = Cᴴ / sqrt(Cᴰ) * Δθ
    
    # Convert to energy density flux: Jρe = ρ₀ cᵖᵈ Jθ
    Jθ = - u★ * θ★  # Potential temperature flux (K m/s)
    return parameters.ρ₀ * parameters.cᵖᵈ * Jθ  # Energy density flux (J/(m² s))
end

# Moisture density flux (converted from specific humidity flux)
# The latent heat flux is computed using bulk transfer: Jq = -u★ q★
# where q★ is the moisture scale computed from the bulk transfer coefficient
@inline function moisture_density_flux(i, j, grid, clock, fields, parameters)
    u★ = friction_velocity(i, j, grid, clock, fields, parameters)
    θˢ = parameters.sea_surface_temperature
    Cᴰ = parameters.drag_coefficient
    Cᵛ = parameters.vapor_transfer_coefficient
    
    # Compute saturation specific humidity at sea surface
    T_sst = θˢ  # Approximate SST as potential temperature
    ρ_sst = parameters.ρ₀  # Use surface reference density
    qˢ = surface_saturation_specific_humidity(T_sst, ρ_sst, parameters.thermodynamics)
    
    # Get total moisture mass fraction from fields
    qᵗ = fields.qᵗ[i, j, 1]
    Δq = qᵗ - qˢ
    
    # Moisture scale from bulk transfer: u★ q★ = Cᵛ U Δq
    # Using U ≈ u★ / √Cᴰ for neutral conditions
    q★ = Cᵛ / sqrt(Cᴰ) * Δq
    
    # Convert to moisture density flux: Jρqᵗ = ρ₀ Jq
    Jq = - u★ * q★  # Specific humidity flux (m/s)
    return parameters.ρ₀ * Jq  # Moisture density flux (kg/(m² s))
end

# Create boundary conditions
ρu_surface_flux = FluxBoundaryCondition(x_momentum_flux; discrete_form=true, parameters)
ρv_surface_flux = FluxBoundaryCondition(y_momentum_flux; discrete_form=true, parameters)
ρe_surface_flux = FluxBoundaryCondition(energy_density_flux; discrete_form=true, parameters)
ρqᵗ_surface_flux = FluxBoundaryCondition(moisture_density_flux; discrete_form=true, parameters)

ρu_bcs = FieldBoundaryConditions(bottom=ρu_surface_flux)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_surface_flux)
ρe_bcs = FieldBoundaryConditions(bottom=ρe_surface_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_surface_flux)

# Create model with boundary conditions
model = AtmosphereModel(grid; advection, microphysics, formulation,
                        boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs))

# ## Initial conditions
#
# We initialize the model with a constant potential temperature θ = θ₀ throughout the domain.
# Then we compute the saturation specific humidity field qᵛ⁺ and use it to construct a
# moisture profile qᵗ that:
# - Starts at 80% of saturation at the surface (z = 0)
# - Ramps linearly to 101% saturation at 1 km (supersaturated to promote cloud formation)
# - Remains constant at 101% from 1-2 km
# - Ramps linearly down to zero by 10 km (top of domain)

Lz = grid.Lz
θ₀ = reference_state.potential_temperature

# Set constant potential temperature
function θᵢ(x, z)
    return θ₀
end

set!(model, θ=θᵢ)

# Update model state to compute temperature T from potential temperature θ
Breeze.AtmosphereModels.update_state!(model)

# Moisture profile qᵗ based on saturation specific humidity qᵛ⁺
# We compute qᵛ⁺ directly using the reference state and thermodynamics
function qᵢ(x, z)
    z_km = z / 1000
    
    # Compute saturation specific humidity at this height
    # Get reference pressure and density at this height
    # For a constant θ profile, T ≈ θ near surface, but decreases with height due to pressure
    # We'll compute qᵛ⁺ using the reference state
    pᵣ_z = reference_state.pressure[1, 1, min(Int(round(z / (Lz / Nz))) + 1, Nz)]
    ρᵣ_z = reference_state.density[1, 1, min(Int(round(z / (Lz / Nz))) + 1, Nz)]
    
    # Approximate temperature from potential temperature
    # For constant θ, T decreases with height due to pressure
    # Using ideal gas: T = θ * (pᵣ/p₀)^(Rᵈ/cᵖᵈ)
    Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(thermodynamics)
    cᵖᵈ = thermodynamics.dry_air.heat_capacity
    T_z = θ₀ * (pᵣ_z / p₀)^(Rᵈ / cᵖᵈ)
    
    # Compute density for dry air at this T and p
    q_dry = Breeze.Thermodynamics.MoistureMassFractions(zero(Float64))
    ρ_z = Breeze.Thermodynamics.density(pᵣ_z, T_z, q_dry, thermodynamics)
    
    # Compute saturation specific humidity
    qᵛ★ = Breeze.Thermodynamics.saturation_specific_humidity(T_z, ρ_z, thermodynamics, 
                                                              Breeze.Thermodynamics.PlanarLiquidSurface())
    
    if z_km <= 1.0
        # Ramp from 80% to 101% between 0 and 1 km
        fraction = z_km / 1.0
        q_rel = 0.80 + fraction * (1.01 - 0.80)
        return q_rel * qᵛ★
    elseif z_km <= 2.0
        # Stay at 101% from 1-2 km (supersaturated layer)
        return 1.01 * qᵛ★
    else
        # Ramp down from 101% to 0% between 2 km and 10 km
        fraction = (z_km - 2.0) / (10.0 - 2.0)
        q_rel = 1.01 * (1.0 - fraction)
        return q_rel * qᵛ★
    end
end

set!(model, qᵗ=qᵢ)

# ## Simulation setup

simulation = Simulation(model, Δt=10, stop_time=4hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Diagnostic fields
#
# We compute diagnostic fields for temperature T, saturation specific humidity qᵛ⁺,
# liquid condensate mass fraction qˡ, and supersaturation δ = qᵗ - qᵛ⁺.

T = model.temperature
qᵛ★ = Breeze.AtmosphereModels.SaturationSpecificHumidityField(model)

# Liquid condensate mass fraction from microphysical fields
# For zero-moment microphysics with saturation adjustment, qˡ is stored in microphysical_fields
qˡ = model.microphysical_fields.qˡ

# Supersaturation: positive values indicate supersaturated conditions
δ = Field(model.moisture_mass_fraction - qᵛ★)

# ## Progress callback

function progress(sim)
    compute!(T)
    compute!(qᵛ★)
    compute!(qˡ)
    compute!(δ)
    
    qᵗ = sim.model.moisture_mass_fraction
    u, v, w = sim.model.velocities

    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    qᵗmin = minimum(qᵗ)
    qᵗmax = maximum(qᵗ)
    qˡmax = maximum(qˡ)
    δmax = maximum(δ)

    Tmin = minimum(T)
    Tmax = maximum(T)

    msg = @sprintf("Iter: %d, t = %s, max|u|: (%.2e, %.2e, %.2e)",
                    iteration(sim), prettytime(sim), umax, vmax, wmax)

    msg *= @sprintf(", extrema(qᵗ): (%.2e, %.2e), max(qˡ): %.2e, max(δ): %.2e, extrema(T): (%.2e, %.2e)",
                     qᵗmin, qᵗmax, qˡmax, δmax, Tmin, Tmax)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# ## Output
#
# We write diagnostic fields to a JLD2 file for later analysis.

outputs = merge(model.velocities, (; T, qˡ, qᵛ★, qᵗ=model.moisture_mass_fraction))

ow = JLD2Writer(model, outputs,
                filename = "prescribed_sst_convection.jld2",
                schedule = TimeInterval(2minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

run!(simulation)

#=
# ## Visualization
#
# The plotting code below can be uncommented to visualize the results.
# It creates animations of the temperature, moisture, and condensate fields.

using GLMakie

θt = FieldTimeSeries("prescribed_sst_convection.jld2", "θ")
Tt = FieldTimeSeries("prescribed_sst_convection.jld2", "T")
qt = FieldTimeSeries("prescribed_sst_convection.jld2", "q")
qˡt = FieldTimeSeries("prescribed_sst_convection.jld2", "qˡ")
times = qt.times
Nt = length(θt)

n = Observable(1)

θn = @lift θt[$n]
qn = @lift qt[$n]
Tn = @lift Tt[$n]
qˡn = @lift qˡt[$n]
title = @lift "t = $(prettytime(times[$n]))"

fig = Figure(size=(800, 400), fontsize=12)
axθ = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)")
axT = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)")
axqˡ = Axis(fig[2, 2], xlabel="x (m)", ylabel="z (m)")

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

Tmin = minimum(Tt)
Tmax = maximum(Tt)

hmθ = heatmap!(axθ, θn, colorrange=(Tₛ, Tₛ+Δθ))
hmq = heatmap!(axq, qn, colorrange=(0.97e-2, 1.05e-2), colormap=:magma)
hmT = heatmap!(axT, Tn, colorrange=(Tmin, Tmax))
hmqˡ = heatmap!(axqˡ, qˡn, colorrange=(0, 1.5e-3), colormap=:magma)

Colorbar(fig[1, 0], hmθ, label = "θ [K]", vertical=true)
Colorbar(fig[1, 3], hmq, label = "q", vertical=true)
Colorbar(fig[2, 0], hmT, label = "T [K]", vertical=true)
Colorbar(fig[2, 3], hmqˡ, label = "qˡ", vertical=true)

fig

record(fig, "prescribed_sst.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end

# Surface flux timeseries
QSH = FieldTimeSeries("prescribed_sst_convection.jld2", "JSH")
QLH = FieldTimeSeries("prescribed_sst_convection.jld2", "JLH")
Qu = FieldTimeSeries("prescribed_sst_convection.jld2", "Ju")
Qv = FieldTimeSeries("prescribed_sst_convection.jld2", "Jv")

QSH_av = zeros(length(QSH.times))
QLH_av = zeros(length(QSH.times))
ΔT_av = zeros(length(QSH.times))
Qv_av = zeros(length(QSH.times))
Qu_av = zeros(length(QSH.times))

for n in 1:length(QSH.times)
    QSHn = Field(Average(QSH[n], dims=1))
    compute!(QSHn)
    QSH_av[n] = QSHn[1, 1, 1]

    QLHn = Field(Average(QLH[n], dims=1))
    compute!(QLHn)
    QLH_av[n] = QLHn[1, 1, 1]

    Qun = Field(Average(Qu[n], dims=1))
    compute!(Qun)
    Qu_av[n] = Qun[1, 1, 1]

    Qvn = Field(Average(Qv[n], dims=1))
    compute!(Qvn)
    Qv_av[n] = Qvn[1, 1, 1]

    ΔT_av[n] = -mean(θt[:,1,1,n])+parameters.sea_surface_temperature
end

ΔTtitle = string(znodes(grid, Center())[1], "m air-sea temperature difference")
fig = Figure(size=(600, 700), fontsize=12)
axtau = Axis(fig[1, 1], ylabel=L"\tau/\rho_0 ~(\mathrm{m}^2/\mathrm{s}^2)", xlabel = "time [h]")
axΔT = Axis(fig[3, 1], ylabel=L"\Delta T ~(\mathrm{K})", xlabel = "time [h]", title = ΔTtitle)
axSH = Axis(fig[2, 1], ylabel=L"J~(\mathrm{W}/\mathrm{m}^2)", xlabel = "time [h]")

lines!(axtau, QSH.times/3600, Qu_av, label = L"\tau_x/\rho_0")
lines!(axtau, QSH.times/3600, Qv_av, label = L"\tau_y/\rho_0")
lines!(axΔT, QSH.times/3600, ΔT_av)
lines!(axSH, QSH.times/3600, QSH_av, label = L"J^{SH}")
lines!(axSH, QLH.times/3600, QLH_av, label = L"J^{LH}")

fig[1, 2] = Legend(fig, axtau, framevisible = false)
fig[2, 2] = Legend(fig, axSH, framevisible = false)
save("Surface_fluxes.png", fig, px_per_unit = 4)
=#

nothing #hide
