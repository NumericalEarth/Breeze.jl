# # Dry and Semidry Tropical Cyclones (Cronin & Chavas, 2019)
#
# This example implements the rotating radiative-convective equilibrium (RCE) experiment
# from [Cronin and Chavas (2019)](@cite Cronin2019), demonstrating that tropical cyclones
# can form and persist even in completely dry atmospheres -- challenging the conventional
# wisdom that moisture is essential for TC dynamics.
#
# The key innovation is the **surface wetness parameter β** (0-1):
# - β = 0: Completely dry surface (no evaporation) — TCs still form!
# - β = 1: Fully moist surface (standard TC behavior)
# - Intermediate values: "semidry" TCs
#
# This script defaults to the **dry case (β = 0)** to demonstrate the paper's novel finding.
# Change β to explore moist and semidry regimes.

# ## Packages

using Breeze
using Oceananigans
using Oceananigans.Grids: xnode, ynode, znode, xnodes, ynodes, znodes
using Oceananigans.Units

using Printf
using Random
using CairoMakie

Random.seed!(2019)  # For reproducibility (paper year!)

# ## Experiment Parameters
#
# These parameters follow Cronin & Chavas (2019), Section 2a.
# The primary control parameter is surface wetness β.

# ### Primary control parameters (user-configurable)

β  = 0.0    # Surface wetness: 0 = dry, 1 = moist (paper's novel finding: β=0 works!)
Tₛ = 300.0  # Surface temperature (K)
Tₜ = 210.0  # Tropopause temperature (K) — for Set 2, use Tₜ = 0.7 * Tₛ

# ### Physical parameters from the paper

Cᴰ = 1.5e-3      # Drag coefficient
Cₖ = 1.5e-3      # Enthalpy exchange coefficient (= Cᴰ in paper)
v★ = 1.0         # Gustiness / minimum wind speed (m/s)
Q̇  = 1.0 / day   # Radiative cooling rate (K/s) for T > Tₜ
τᵣ = 20days      # Newtonian relaxation timescale for T ≤ Tₜ
f₀ = 3e-4        # Coriolis parameter (s⁻¹)

# ## Domain and Grid
#
# The paper uses a 1152 km × 1152 km doubly-periodic domain with 2 km horizontal
# resolution and a 28 km model top. We use reduced resolution for faster runs.

arch = CPU()

# Domain size (paper values)
Lx = Ly = 1152e3  # 1152 km
H  = 28e3         # 28 km model top

# Resolution (reduced for CI/documentation — increase for science runs)
Nx = Ny = 64      # Paper uses 576 (2 km spacing)
Nz = 32           # Paper uses ~80 with vertical stretching

if get(ENV, "CI", "false") == "true"
    Nx = Ny = 32
    Nz = 16
end

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, H),
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference State and Dynamics
#
# We use the anelastic formulation with a reference state based on the surface
# potential temperature. For the dry case, we use static energy formulation
# since there's no moisture to track.

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = 101325,
                                 potential_temperature = Tₛ)

dynamics = AnelasticDynamics(reference_state)

coriolis = FPlane(f = f₀)

# ## Surface Fluxes
#
# Following paper Eqs. (2)-(4), the surface fluxes use bulk aerodynamic formulas
# with equal drag and enthalpy exchange coefficients, and a gustiness parameter
# to prevent singularities at low wind speeds.
#
# For the dry case (β = 0), we only need momentum drag and sensible heat flux.
# The latent heat flux is zero when β = 0.

# ### Momentum drag (Eq. 4 with quadratic drag)
#
# The paper uses: τ = ρ Cᴰ Vₛ u, where Vₛ = √(u² + v² + v★²)

FT = eltype(grid)
p₀ = reference_state.surface_pressure
θ₀ = reference_state.potential_temperature
ρ₀ = Breeze.Thermodynamics.density(θ₀, p₀,
        zero(Breeze.Thermodynamics.MoistureMassFractions{FT}), constants)

drag_params = (; Cᴰ = FT(Cᴰ), v★ = FT(v★), ρ₀ = FT(ρ₀))

@inline function ρu_drag_flux(x, y, t, ρu, ρv, p)
    u = ρu / p.ρ₀
    v = ρv / p.ρ₀
    U² = u^2 + v^2
    U = sqrt(U²)
    Vₛ² = U² + p.v★^2
    # τˣ = -ρ₀ Cᴰ Vₛ u = -Cᴰ Vₛ ρu
    # But we want flux of ρu, so: Jρu = -Cᴰ Vₛ² ρu / U (for U > 0)
    return ifelse(U > 0, -p.Cᴰ * Vₛ² * ρu / U, zero(ρu))
end

@inline function ρv_drag_flux(x, y, t, ρu, ρv, p)
    u = ρu / p.ρ₀
    v = ρv / p.ρ₀
    U² = u^2 + v^2
    U = sqrt(U²)
    Vₛ² = U² + p.v★^2
    return ifelse(U > 0, -p.Cᴰ * Vₛ² * ρv / U, zero(ρv))
end

ρu_drag_bc = FluxBoundaryCondition(ρu_drag_flux;
                                   field_dependencies = (:ρu, :ρv),
                                   parameters = drag_params)
ρv_drag_bc = FluxBoundaryCondition(ρv_drag_flux;
                                   field_dependencies = (:ρu, :ρv),
                                   parameters = drag_params)

ρu_bcs = FieldBoundaryConditions(bottom = ρu_drag_bc)
ρv_bcs = FieldBoundaryConditions(bottom = ρv_drag_bc)

# ### Sensible heat flux (Eq. 2)
#
# H = ρ Cₖ cₚ Vₛ (Tₛ - θ₁)
#
# For anelastic models, we apply this as a flux of ρθ:
# Jρθ = ρ₀ Cₖ Vₛ (Tₛ - θ₁)

cᵖᵈ = constants.dry_air.heat_capacity
sensible_params = (; Cₖ = FT(Cₖ), v★ = FT(v★), Tₛ = FT(Tₛ), ρ₀ = FT(ρ₀))

@inline function sensible_heat_flux(x, y, t, ρθ, ρu, ρv, p)
    θ = ρθ / p.ρ₀
    u = ρu / p.ρ₀
    v = ρv / p.ρ₀
    Vₛ = sqrt(u^2 + v^2 + p.v★^2)
    # Flux of θ (multiplied by ρ₀ for ρθ flux)
    return p.ρ₀ * p.Cₖ * Vₛ * (p.Tₛ - θ)
end

ρθ_sensible_bc = FluxBoundaryCondition(sensible_heat_flux;
                                       field_dependencies = (:ρθ, :ρu, :ρv),
                                       parameters = sensible_params)

ρθ_bcs = FieldBoundaryConditions(bottom = ρθ_sensible_bc)

# ### Moisture flux (Eq. 3) — only for β > 0
#
# E = ρ Cₖ Lᵥ Vₛ (β q★ₛ - q₁)
#
# where q★ₛ is saturation specific humidity at surface temperature.
# For β = 0, there is no moisture flux (completely dry surface).

if β > 0
    # Compute saturation specific humidity at surface
    q★ₛ = Breeze.Thermodynamics.saturation_specific_humidity(Tₛ, ρ₀, constants, constants.liquid)

    moisture_params = (; Cₖ = FT(Cₖ), v★ = FT(v★), β = FT(β), q★ₛ = FT(q★ₛ), ρ₀ = FT(ρ₀))

    @inline function moisture_flux(x, y, t, ρqᵗ, ρu, ρv, p)
        qᵗ = ρqᵗ / p.ρ₀
        u = ρu / p.ρ₀
        v = ρv / p.ρ₀
        Vₛ = sqrt(u^2 + v^2 + p.v★^2)
        # Flux of qᵗ (multiplied by ρ₀ for ρqᵗ flux)
        # Paper Eq. 3: E = ρ Cₖ Lᵥ Vₛ (β q★ₛ - q₁)
        # We return moisture mass flux (without Lᵥ, which is handled by energy equation)
        return p.ρ₀ * p.Cₖ * Vₛ * (p.β * p.q★ₛ - qᵗ)
    end

    ρqᵗ_moisture_bc = FluxBoundaryCondition(moisture_flux;
                                            field_dependencies = (:ρqᵗ, :ρu, :ρv),
                                            parameters = moisture_params)
    ρqᵗ_bcs = FieldBoundaryConditions(bottom = ρqᵗ_moisture_bc)

    # Collect all boundary conditions (with moisture)
    boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs, ρθ = ρθ_bcs, ρqᵗ = ρqᵗ_bcs)
else
    # Collect all boundary conditions (dry case, no moisture)
    boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs, ρθ = ρθ_bcs)
end

# ## Radiative Forcing
#
# The paper (Eq. 1) prescribes a piecewise radiative tendency:
# - For T > Tₜ: constant cooling at rate -Q̇ = -1 K/day
# - For T ≤ Tₜ: Newtonian relaxation toward Tₜ with timescale τ = 20 days
#
# We implement this as a forcing on ρθ. Since we're in the dry limit and
# the atmosphere is nearly dry-adiabatic, we approximate T ≈ θ × Π where
# Π is the Exner function. For simplicity in the dry case, we use θ directly
# as a proxy for the temperature structure.

ρᵣ = reference_state.density

# For the dry case, we use a height-based approximation:
# The tropopause is roughly where θ(z) corresponds to Tₜ on a reference adiabat.
# A simpler approach: apply constant cooling throughout the troposphere.

# We create a static forcing field based on height
∂t_ρθ_rad = Field{Nothing, Nothing, Center}(grid)

# Approximate: constant cooling of -1 K/day throughout troposphere
# This is a simplification; a full implementation would use discrete forcing
# that depends on the actual temperature field.
set!(∂t_ρθ_rad, ρᵣ * (-Q̇))

radiation_forcing = Forcing(∂t_ρθ_rad)

forcing = (; ρθ = radiation_forcing)

# ## Model Construction
#
# For the dry case (β = 0), we don't need microphysics since there's no moisture.
# For moist cases (β > 0), we use saturation adjustment for cloud formation.

advection = WENO(order = 9)

# Microphysics: only needed when there's moisture
microphysics = β > 0 ? SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()) : nothing

model = AtmosphereModel(grid;
                        dynamics,
                        coriolis,
                        advection,
                        microphysics,
                        forcing,
                        boundary_conditions)

# ## Initial Conditions
#
# We initialize with a dry adiabatic temperature profile and add small random
# perturbations to the potential temperature in the lowest levels to trigger
# convection. For moist cases (β > 0), we also add a moisture profile.

g = constants.gravitational_acceleration
N² = 1e-5  # Weak stable stratification (nearly neutral for dry convection)

# θ profile: nearly neutral with weak stratification
θ_profile(z) = θ₀ * exp(N² * z / g)

# Random perturbation parameters
δθ = 0.5   # K (perturbation amplitude)
zδ = 1000  # m (depth of perturbation layer)

θᵢ(x, y, z) = θ_profile(z) + δθ * (2 * rand() - 1) * (z < zδ)

if β > 0
    # Moisture profile for moist cases
    # Surface specific humidity scaled by β
    q_surface = 0.015  # ~15 g/kg near tropical surface
    q_scale_height = 3000.0  # m

    qᵗ_profile(z) = β * q_surface * exp(-z / q_scale_height)

    δq = 1e-4  # kg/kg (perturbation amplitude)
    qᵢ(x, y, z) = max(0, qᵗ_profile(z) + δq * (2 * rand() - 1) * (z < zδ))

    set!(model, θ = θᵢ, qᵗ = qᵢ)
else
    set!(model, θ = θᵢ)
end

# ## Simulation Setup

Δt = 10.0  # Initial timestep (seconds)
stop_time = 6hours

if get(ENV, "CI", "false") == "true"
    stop_time = 30minutes
end

simulation = Simulation(model; Δt, stop_time)
conjure_time_step_wizard!(simulation, cfl = 0.7)

# ### Progress callback

u, v, w = model.velocities
θ = liquid_ice_potential_temperature(model)

function progress(sim)
    wmax = maximum(abs, w)
    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    θmin, θmax = extrema(θ)

    msg = @sprintf("Iter %d, t = %s, Δt = %s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt))
    msg *= @sprintf(", max|u,v,w| = (%.2f, %.2f, %.2f) m/s", umax, vmax, wmax)
    msg *= @sprintf(", θ ∈ [%.1f, %.1f] K", θmin, θmax)

    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# ### Output writers

# Full 3D fields at lower frequency for detailed analysis
outputs_3d = (; u, v, w, θ)
simulation.output_writers[:fields] = JLD2Writer(model, outputs_3d;
                                                filename = "tropical_cyclone_world_fields.jld2",
                                                schedule = TimeInterval(1hour),
                                                overwrite_existing = true)

# Surface fields at higher frequency for TC tracking
surface_k = 1
surface_outputs = (u_surface = view(u, :, :, surface_k),
                   v_surface = view(v, :, :, surface_k),
                   θ_surface = view(θ, :, :, surface_k))

simulation.output_writers[:surface] = JLD2Writer(model, surface_outputs;
                                                 filename = "tropical_cyclone_world_surface.jld2",
                                                 schedule = TimeInterval(10minutes),
                                                 overwrite_existing = true)

# Horizontally-averaged profiles for comparing with paper Figure 3
avg_outputs = (θ_avg = Average(θ, dims=(1, 2)),
               u_avg = Average(u, dims=(1, 2)),
               v_avg = Average(v, dims=(1, 2)))

simulation.output_writers[:profiles] = JLD2Writer(model, avg_outputs;
                                                  filename = "tropical_cyclone_world_averages.jld2",
                                                  schedule = TimeInterval(30minutes),
                                                  overwrite_existing = true)

# ## Run the simulation

case_name = β == 0 ? "dry" : (β == 1 ? "moist" : "semidry (β=$β)")
@info "Running $case_name tropical cyclone world on $(summary(grid))"
@info "Surface wetness β = $β, surface temperature Tₛ = $Tₛ K, tropopause Tₜ = $Tₜ K"
@info "Stop time: $(prettytime(stop_time))"

run!(simulation)

@info "Simulation complete!"

# ## Visualization
#
# We create several figures for comparison with Cronin & Chavas (2019):
# 1. Surface wind speed evolution (snapshots) — compare to their Fig. 1
# 2. Time series of maximum wind speed — compare to their Fig. 5
# 3. Mean profile evolution — compare to their Fig. 3

if get(ENV, "CI", "false") != "true"
    @info "Creating visualizations..."

    surface_file = "tropical_cyclone_world_surface.jld2"
    profile_file = "tropical_cyclone_world_averages.jld2"

    title_case = β == 0 ? "Dry" : (β == 1 ? "Moist" : "Semidry")

    # ### Figure 1: Surface wind speed snapshots
    #
    # Shows the horizontal structure of the developing convection/TC at several times.
    # Compare to Cronin & Chavas (2019), Figure 1.

    if isfile(surface_file)
        u_ts = FieldTimeSeries(surface_file, "u_surface")
        v_ts = FieldTimeSeries(surface_file, "v_surface")
        times = u_ts.times
        Nt = length(times)

        x = xnodes(grid, Center()) ./ 1e3  # km
        y = ynodes(grid, Center()) ./ 1e3  # km

        # Compute wind speed for all times
        function wind_speed(n)
            u_data = interior(u_ts[n], :, :, 1)
            v_data = interior(v_ts[n], :, :, 1)
            return sqrt.(u_data.^2 .+ v_data.^2)
        end

        # Determine color range from all data
        U_max = maximum(maximum(wind_speed(n)) for n in 1:Nt)

        fig = Figure(size = (1200, 400), fontsize = 12)

        # Plot first, middle, and last snapshots
        indices = [1, max(1, Nt ÷ 2), Nt]

        local hm_last  # for colorbar reference
        for (i, n) in enumerate(indices)
            local ax_snap = Axis(fig[1, i];
                                 xlabel = "x (km)",
                                 ylabel = i == 1 ? "y (km)" : "",
                                 title = "t = $(prettytime(times[n]))",
                                 aspect = 1)

            U_data = wind_speed(n)
            hm_last = heatmap!(ax_snap, x, y, U_data; colormap = :speed, colorrange = (0, max(1, U_max)))
        end
        Colorbar(fig[1, length(indices)+1], hm_last; label = "Surface wind speed (m/s)")

        Label(fig[0, :], "$title_case Tropical Cyclone World (β = $β) — Surface Wind Speed",
              fontsize = 16, tellwidth = false)

        save("tropical_cyclone_world_surface_winds.png", fig) #src
        fig

        # ### Figure 2: Time series of maximum wind speed
        #
        # Shows TC intensification over time. Compare to Cronin & Chavas (2019), Figure 5.

        max_wind = [maximum(wind_speed(n)) for n in 1:Nt]
        times_hours = times ./ 3600

        fig2 = Figure(size = (600, 400), fontsize = 14)
        ax_ts = Axis(fig2[1, 1];
                     xlabel = "Time (hours)",
                     ylabel = "Maximum surface wind speed (m/s)",
                     title = "$title_case TC (β = $β) — Intensity Evolution")

        lines!(ax_ts, times_hours, max_wind; linewidth = 2, color = :dodgerblue)
        scatter!(ax_ts, times_hours, max_wind; markersize = 8, color = :dodgerblue)

        save("tropical_cyclone_world_intensity.png", fig2) #src
        fig2

        # ### Figure 3: Animation of surface wind speed
        #
        # Animated evolution to see convective organization and TC formation.

        fig3 = Figure(size = (600, 550), fontsize = 14)
        ax_anim = Axis(fig3[1, 1];
                       xlabel = "x (km)",
                       ylabel = "y (km)",
                       aspect = 1)

        n_obs = Observable(1)
        U_n = @lift wind_speed($n_obs)
        title3 = @lift "$title_case TC (β = $β) — t = $(prettytime(times[$n_obs]))"

        hm_anim = heatmap!(ax_anim, x, y, U_n; colormap = :speed, colorrange = (0, max(1, U_max)))
        Colorbar(fig3[1, 2], hm_anim; label = "Wind speed (m/s)")
        Label(fig3[0, :], title3, fontsize = 16, tellwidth = false)

        CairoMakie.record(fig3, "tropical_cyclone_world.mp4", 1:Nt, framerate = 8) do nn
            n_obs[] = nn
        end
        nothing #hide

        # ![](tropical_cyclone_world.mp4)
    end

    # ### Figure 4: Mean profile evolution
    #
    # Shows how the horizontally-averaged thermodynamic structure evolves.
    # Compare to Cronin & Chavas (2019), Figure 3.

    if isfile(profile_file)
        θ_avg_ts = FieldTimeSeries(profile_file, "θ_avg")
        u_avg_ts = FieldTimeSeries(profile_file, "u_avg")
        v_avg_ts = FieldTimeSeries(profile_file, "v_avg")

        times = θ_avg_ts.times
        Nt = length(times)
        z_km = znodes(grid, Center()) ./ 1e3  # km

        fig4 = Figure(size = (900, 400), fontsize = 14)

        axθ = Axis(fig4[1, 1]; xlabel = "θ (K)", ylabel = "z (km)", title = "Potential temperature")
        axu = Axis(fig4[1, 2]; xlabel = "u (m/s)", title = "Zonal wind")
        axv = Axis(fig4[1, 3]; xlabel = "v (m/s)", title = "Meridional wind")

        default_colors = Makie.wong_colors()
        colors = [default_colors[mod1(i, length(default_colors))] for i in 1:Nt]

        for n in 1:Nt
            label = n == 1 ? "initial" : "t = $(prettytime(times[n]))"
            θ_data = interior(θ_avg_ts[n], 1, 1, :)
            u_data = interior(u_avg_ts[n], 1, 1, :)
            v_data = interior(v_avg_ts[n], 1, 1, :)

            lines!(axθ, θ_data, z_km; color = colors[n], label = label)
            lines!(axu, u_data, z_km; color = colors[n])
            lines!(axv, v_data, z_km; color = colors[n])
        end

        for ax in (axu, axv)
            ax.yticksvisible = false
            ax.yticklabelsvisible = false
        end

        axislegend(axθ, position = :rb, labelsize = 10)

        Label(fig4[0, :], "$title_case TC (β = $β) — Mean Profile Evolution",
              fontsize = 16, tellwidth = false)

        save("tropical_cyclone_world_profiles.png", fig4) #src
        fig4
    end
end

# ## Exploring Different Regimes
#
# This script supports the full range of surface wetness values:
#
# - **β = 0**: Completely dry (no evaporation) — the paper's novel finding!
# - **β = 1**: Fully moist (standard tropical cyclone)
# - **0 < β < 1**: Semidry (intermediate behavior)
#
# To change the regime, modify the `β` parameter at the top of the script.
#
# ## Key Findings from Cronin & Chavas (2019)
#
# 1. TCs form and persist in both dry (β=0) and moist (β=1) limits
# 2. A "no-storms-land" exists at intermediate values (β ≈ 0.01-0.3)
#    where spontaneous TC genesis does not occur
# 3. Dry TCs have smaller outer radii but similar-sized convective cores
# 4. TC intensity decreases as the surface is dried (lower β)
#
# ## Simplifications in this Implementation
#
# - Uses `SaturationAdjustment` microphysics (no precipitation fallout)
# - Paper uses full single-moment bulk microphysics with rain/ice
# - Radiative cooling is constant throughout troposphere (simplified)
# - Paper uses temperature-dependent piecewise cooling
# - Grid is uniform (paper uses stretched vertical grid)
#
# For more faithful reproduction, consider using `OneMomentCloudMicrophysics`
# and implementing the full temperature-dependent radiative scheme.
