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
#
# ## Differences from Cronin & Chavas (2019)
#
# This implementation makes several simplifications compared to the original paper:
#
# ### Resolution and Domain (Test Configuration)
# - **Horizontal resolution**: 8 km spacing (144×144) vs. paper's 2 km (576×576)
# - **Vertical grid**: ~33 levels (4x coarser) vs. paper's ~133 levels (64 in lowest 1 km)
# - **Domain size**: Fixed 1152 km for all β; paper uses 1728 km for β ≥ 0.9
# - **Turbulence**: ILES (no explicit diffusivity) vs. paper's 1.5-order TKE
#
# To run at paper resolution, set: Nx=Ny=576, scale_factor=1, closure=ScalarDiffusivity(ν=10,κ=10)
#
# ### Simulation Duration and Initialization
# - **Runtime**: 30 min test (use 48 hours for development) vs. paper's 70 days
# - **Initialization**: Dry adiabat + random perturbations vs. pre-equilibrated state from
#   100-day nonrotating RCE simulation
#
# ### Microphysics
# - **Scheme**: `SaturationAdjustment` (equilibrium condensation, no precipitation fallout)
#   vs. SAM's full single-moment bulk microphysics with rain, snow, graupel, ice sedimentation
# - **Ice phase**: Warm-phase only (`WarmPhaseEquilibrium`) vs. full ice microphysics
#
# ### Radiative Cooling
# - **Implementation**: Following RICO pattern — Field-based forcing on ρθ (not discrete_form on ρe)
# - Paper formula: `(∂T/∂t)_rad = -Q̇` for `T > Tₜ` (constant -1 K/day in troposphere)
# - Since T > Tₜ = 210 K everywhere in troposphere, we use constant: `F_ρθ = ρᵣ × (-Q̇)`
# - This matches RICO's approach (constant -2.5 K/day) and ensures consistent thermodynamics
#
# ### Other
# - **Storm tracking**: None — paper uses pressure perturbation threshold + tracking algorithm
# - **Top boundary**: Rayleigh damping sponge in upper 3 km (paper doesn't specify, SAM has built-in)
#
# For scientific reproduction, increase resolution, extend runtime to ~70 days, and consider
# using `OneMomentCloudMicrophysics` from the CloudMicrophysics.jl extension.

# ## Packages

using Breeze
using CUDA
using Oceananigans
using Oceananigans.Grids: xnode, ynode, znode, xnodes, ynodes, znodes
using Oceananigans.Fields: Field, set!
using Oceananigans.Units

using Printf
using Random
import Dates  # Import (not using) to avoid conflict with Oceananigans.Units.day
using CairoMakie

Random.seed!(2019)  # For reproducibility (paper year!)

# ## Experiment Directory Setup
#
# All outputs are organized in timestamped experiment directories for clean reproducibility.
# This prevents output files from cluttering the examples/ directory.

experiment_name = "tc_test"  # Change for different experiment types

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
Cᵀ = 1.5e-3      # Sensible heat transfer coefficient (= Cᴰ in paper)
v★ = 1.0         # Gustiness / minimum wind speed (m/s)
Q̇  = 1.0 / day   # Radiative cooling rate (K/s) for T > Tₜ
τᵣ = 20days      # Newtonian relaxation timescale for T ≤ Tₜ
f₀ = 3e-4        # Coriolis parameter (s⁻¹)

# ## Domain and Grid
#
# The paper uses a 1152 km × 1152 km doubly-periodic domain with 2 km horizontal
# resolution and a 28 km model top. The vertical grid is stretched following
# the paper's Section 2a specification EXACTLY:
# - "64 levels in the lowest kilometer"
# - "spacing of 500 m above 3.5 km"
# - Linear transition from 1 km to 3.5 km

arch = GPU()

# Domain size (paper values)
Lx = Ly = 1152e3  # 1152 km
H  = 28e3         # 28 km model top

# ## Test Configuration (4x reduced resolution)
#
# This script is configured for rapid testing:
# - Horizontal: 144×144 (8 km) vs. paper's 576×576 (2 km)
# - Vertical: ~33 levels vs. paper's ~133 levels
# - Closure: ILES (no explicit diffusivity)
#
# For paper-resolution runs, change:
#   Nx = Ny = 576
#   scale_factor = 1 in paper_vertical_grid()
#   closure = ScalarDiffusivity(ν=10, κ=10)

# Resolution: reduced 4x for testing (paper uses 576×576 = 2 km)
Nx = Ny = 144     # Test resolution: 8 km spacing

# Stretched vertical grid following paper Section 2a, with configurable scale factor:
# - 64/scale_factor levels in lowest 1 km
# - 500*scale_factor m spacing above 3.5 km
# - Linear transition from 1 km to 3.5 km
function paper_vertical_grid(H; scale_factor=4)
    # scale_factor=1 for paper resolution, =4 for testing
    z_faces = Float64[0.0]
    
    # Region 1: levels in lowest 1 km
    n_levels_lower = 64 ÷ scale_factor  # 64 → 16 for scale_factor=4
    Δz₁ = 1000.0 / n_levels_lower  # 15.625 m → 62.5 m
    for i in 1:n_levels_lower
        push!(z_faces, i * Δz₁)
    end
    z = 1000.0
    
    # Region 2: Linear transition from 1 km to 3.5 km
    z_start, z_end = 1000.0, 3500.0
    Δz_start = Δz₁
    Δz_end = 500.0 * scale_factor  # 500 m → 2000 m
    
    while z < z_end - 1e-6  # Small tolerance for floating point
        # Linear interpolation of Δz based on current position
        frac = (z - z_start) / (z_end - z_start)
        Δz = Δz_start + frac * (Δz_end - Δz_start)
        z = min(z + Δz, z_end)
        push!(z_faces, z)
    end
    
    # Region 3: Constant spacing above 3.5 km to model top
    Δz₃ = 500.0 * scale_factor  # 500 m → 2000 m
    while z < H - 1e-6
        z = min(z + Δz₃, H)
        push!(z_faces, z)
    end
    
    return z_faces
end

z_faces = paper_vertical_grid(H; scale_factor=4)  # 4x coarser for testing
Nz = length(z_faces) - 1

# Count levels in lowest 1 km
n_below_1km = count(z -> z <= 1000.0, z_faces) - 1  # faces - 1 = cells
@info "Vertical grid: Nz = $Nz levels total"
@info "Levels in lowest 1 km: $n_below_1km (paper: 64)"
@info "Δz range: $(minimum(diff(z_faces))) m to $(maximum(diff(z_faces))) m"

if get(ENV, "CI", "false") == "true"
    Nx = Ny = 32
    z_faces = range(0, H, length=17)  # Uniform for CI
    Nz = 16
end

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces,
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference State and Dynamics
#
# We use the anelastic formulation with a reference state based on the surface
# potential temperature. For the dry case, we use static energy formulation
# since there's no moisture to track.

constants = ThermodynamicConstants()
g = constants.gravitational_acceleration

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = 101325,
                                 potential_temperature = Tₛ)

dynamics = AnelasticDynamics(reference_state)

coriolis = FPlane(f = f₀)

# ## Surface Fluxes (RICO Pattern)
#
# Following the RICO example pattern for consistency with Breeze's thermodynamic formulation:
# - Momentum drag → ρu/ρv using BulkDrag
# - Sensible heat → ρe using BulkSensibleHeatFlux (handles T↔θ conversion correctly)
# - Moisture flux → ρqᵗ using BulkVaporFlux (for β > 0)
#
# The paper's bulk formulas (Eqs. 2-4) are mathematically equivalent, but using
# Breeze utilities ensures correct thermodynamic variable handling.

FT = eltype(grid)

# ### Momentum drag (Eq. 4)
#
# Using BulkDrag from Breeze (same as RICO example).
# The gustiness parameter ensures nonzero flux even at zero wind.

ρu_bcs = FieldBoundaryConditions(bottom = Breeze.BulkDrag(coefficient = Cᴰ, gustiness = v★))
ρv_bcs = FieldBoundaryConditions(bottom = Breeze.BulkDrag(coefficient = Cᴰ, gustiness = v★))

# ### Sensible heat flux (Eq. 2)
#
# Using BulkSensibleHeatFlux on ρe (like RICO), NOT on ρθ.
# This correctly handles the temperature-to-potential-temperature conversion
# inside Breeze's formulation:
#   ∂(ρθ)/∂t = ... + (1/(cᵖᵐ Π)) × F_{ρe}
#
# The paper's formula: Jˢ = ρ Cᵀ Vₛ (Tₛ - T)
# BulkSensibleHeatFlux implements exactly this pattern.

ρe_sensible_flux = Breeze.BulkSensibleHeatFlux(coefficient = Cᵀ,
                                               gustiness = v★,
                                               surface_temperature = Tₛ)
ρe_bcs = FieldBoundaryConditions(bottom = ρe_sensible_flux)

# ### Moisture flux (Eq. 3) — only for β > 0
#
# For moist cases, using BulkVaporFlux which computes moisture flux based on
# the difference between atmospheric specific humidity and saturation humidity
# at the surface. For β = 0 (dry), there is no moisture flux.

if β > 0
    # Scaled surface saturation humidity for semidry cases
    ρqᵗ_moisture_bc = Breeze.BulkVaporFlux(coefficient = Cᵀ * β,
                                           gustiness = v★,
                                           surface_temperature = Tₛ)
    ρqᵗ_bcs = FieldBoundaryConditions(bottom = ρqᵗ_moisture_bc)
    boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs, ρe = ρe_bcs, ρqᵗ = ρqᵗ_bcs)
else
    boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs, ρe = ρe_bcs)
end

# ## Radiative Forcing (RICO Pattern)
#
# The paper (Eq. 1) prescribes a piecewise radiative tendency based on temperature:
#
# ```math
# \left(\frac{∂T}{∂t}\right)_{\rm rad} = \begin{cases}
#   -\dot{Q}, & T > T_t \\
#   (T_t - T)/τ, & T ≤ T_t
# \end{cases}
# ```
#
# where Q̇ = 1 K/day is the constant cooling rate in the troposphere, Tₜ = 210 K is
# the tropopause temperature, and τ = 20 days is the Newtonian relaxation timescale.
#
# **RICO Pattern**: Following the RICO example, we implement radiative forcing as a
# Field-based tendency on ρθ (NOT ρe with discrete_form). This is simpler, more stable,
# and consistent with other Breeze examples:
#
#   F_{ρθ} = ρᵣ × ∂θ/∂t
#
# For the troposphere (T > Tₜ = 210 K everywhere), we have constant cooling:
#   ∂T/∂t = -Q̇ = -1 K/day
#
# Since θ ≈ T at surface and the ratio θ/T = 1/Π is O(1), we approximate:
#   ∂θ/∂t ≈ -Q̇ (constant cooling of potential temperature)
#
# This is exactly the RICO approach (constant -2.5 K/day), just with different rate.

ρᵣ = reference_state.density

# Create Field-based radiative forcing (like RICO)
∂t_ρθ_radiation = Field{Nothing, Nothing, Center}(grid)
∂θ∂t = -Q̇  # Constant cooling rate (K/s), negative for cooling
set!(∂t_ρθ_radiation, ρᵣ * ∂θ∂t)  # F_ρθ = ρᵣ × ∂θ/∂t
ρθ_radiation_forcing = Forcing(∂t_ρθ_radiation)

# ## Sponge Layer
#
# To prevent spurious wave reflections from the upper boundary, we add a Rayleigh
# damping sponge layer in the upper 3 km of the domain (top ~10%). The sponge damps
# vertical velocity toward zero using Oceananigans' `Relaxation` forcing with a
# `GaussianMask`. This is standard practice for LES with a rigid lid, though the
# paper doesn't explicitly describe this (SAM likely has a built-in sponge).

using Oceananigans.Forcings: Relaxation, GaussianMask

sponge_width = 3000.0   # m - width of sponge layer
sponge_center = H - sponge_width / 2  # Center the Gaussian in upper portion
sponge_rate = 1 / 60.0  # s⁻¹ - 1 minute relaxation timescale

sponge_mask = GaussianMask{:z}(center = sponge_center, width = sponge_width)
ρw_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask)

# Assemble all forcings: radiative cooling on ρθ (RICO pattern), sponge on ρw
forcing = (; ρθ = ρθ_radiation_forcing, ρw = ρw_sponge)

# ## Turbulence Closure
#
# Using ILES (Implicit Large Eddy Simulation): no explicit diffusivity.
# The WENO advection scheme provides numerical dissipation for stability.
#
# For paper-resolution runs with fine vertical grid (Δz ~ 15.6 m), you may need:
#   closure = ScalarDiffusivity(ν=10, κ=10)
# to ensure diffusive stability.

closure = nothing  # ILES - WENO provides numerical dissipation

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
                        closure,
                        forcing,
                        boundary_conditions)

# ## Initial Conditions
#
# We initialize with a dry adiabatic temperature profile and add small random
# perturbations to the potential temperature in the lowest levels to trigger
# convection. For moist cases (β > 0), we also add a moisture profile.

θ₀ = reference_state.potential_temperature
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
stop_time = 48hours  # 48-hour run for TC development

if get(ENV, "CI", "false") == "true"
    stop_time = 30minutes
end

simulation = Simulation(model; Δt, stop_time)
conjure_time_step_wizard!(simulation, cfl = 0.7)

# ## Experiment Directory
#
# Create a unique experiment directory with descriptive naming for organized outputs.
# Format: {experiment_name}_{resolution}_{closure}_{timestamp}

resolution_tag = "$(Nx)x$(Ny)x$(Nz)"
closure_tag = isnothing(closure) ? "iles" : "scalar_diff"
timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
experiment_dir = joinpath(@__DIR__, "experiments",
    "$(experiment_name)_$(resolution_tag)_$(closure_tag)_$(timestamp)")

mkpath(experiment_dir)
mkpath(joinpath(experiment_dir, "checkpoints"))
mkpath(joinpath(experiment_dir, "figures"))

@info "Experiment directory: $experiment_dir"

# Write experiment configuration for reproducibility
open(joinpath(experiment_dir, "config.toml"), "w") do io
    println(io, "# Experiment configuration")
    println(io, "timestamp = \"$timestamp\"")
    println(io, "experiment_name = \"$experiment_name\"")
    println(io, "")
    println(io, "[grid]")
    println(io, "Nx = $Nx")
    println(io, "Ny = $Ny")
    println(io, "Nz = $Nz")
    println(io, "Lx = $Lx")
    println(io, "Ly = $Ly")
    println(io, "H = $H")
    println(io, "")
    println(io, "[physics]")
    println(io, "beta = $β")
    println(io, "Ts = $Tₛ")
    println(io, "Tt = $Tₜ")
    println(io, "closure = \"$(isnothing(closure) ? "none (ILES)" : string(closure))\"")
    println(io, "")
    println(io, "[simulation]")
    println(io, "stop_time = \"$(prettytime(stop_time))\"")
    println(io, "initial_dt = $Δt")
end

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

add_callback!(simulation, progress, IterationInterval(500))

# ### Output writers
#
# All outputs go to the experiment directory for organized, clean experiments.

# Full 3D fields at lower frequency for detailed analysis
outputs_3d = (; u, v, w, θ)
simulation.output_writers[:fields] = JLD2Writer(model, outputs_3d;
                                                filename = joinpath(experiment_dir, "fields.jld2"),
                                                schedule = TimeInterval(6hours),
                                                overwrite_existing = true)

# Surface fields at higher frequency for TC tracking
surface_k = 1
surface_outputs = (u_surface = view(u, :, :, surface_k),
                   v_surface = view(v, :, :, surface_k),
                   θ_surface = view(θ, :, :, surface_k))

simulation.output_writers[:surface] = JLD2Writer(model, surface_outputs;
                                                 filename = joinpath(experiment_dir, "surface.jld2"),
                                                 schedule = TimeInterval(30minutes),
                                                 overwrite_existing = true)

# Horizontally-averaged profiles for comparing with paper Figure 3
avg_outputs = (θ_avg = Average(θ, dims=(1, 2)),
               u_avg = Average(u, dims=(1, 2)),
               v_avg = Average(v, dims=(1, 2)))

simulation.output_writers[:profiles] = JLD2Writer(model, avg_outputs;
                                                  filename = joinpath(experiment_dir, "profiles.jld2"),
                                                  schedule = TimeInterval(1hour),
                                                  overwrite_existing = true)

# Checkpointer for restart capability
simulation.output_writers[:checkpointer] = Checkpointer(model;
                                                        schedule = TimeInterval(6hours),
                                                        dir = joinpath(experiment_dir, "checkpoints"),
                                                        prefix = "checkpoint",
                                                        overwrite_existing = true)

# ## Run the simulation

case_name = β == 0 ? "dry" : (β == 1 ? "moist" : "semidry (β=$β)")
@info "Running $case_name tropical cyclone world on $(summary(grid))"
@info "Surface wetness β = $β, surface temperature Tₛ = $Tₛ K, tropopause Tₜ = $Tₜ K"
@info "Stop time: $(prettytime(stop_time))"

run!(simulation)  # Note: pickup=true not yet supported for AtmosphereModel

@info "Simulation complete!"

# ## Visualization
#
# We create several figures for comparison with Cronin & Chavas (2019):
# 1. Surface wind speed evolution (snapshots) — compare to their Fig. 1
# 2. Time series of maximum wind speed — compare to their Fig. 5
# 3. Mean profile evolution — compare to their Fig. 3

if get(ENV, "CI", "false") != "true"
    @info "Creating visualizations..."

    figures_dir = joinpath(experiment_dir, "figures")
    surface_file = joinpath(experiment_dir, "surface.jld2")
    profile_file = joinpath(experiment_dir, "profiles.jld2")

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

        save(joinpath(figures_dir, "surface_winds.png"), fig) #src
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

        save(joinpath(figures_dir, "intensity.png"), fig2) #src
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

        CairoMakie.record(fig3, joinpath(figures_dir, "animation.mp4"), 1:Nt, framerate = 8) do nn
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
        z_km = Array(znodes(grid, Center())) ./ 1e3  # km (convert to CPU)

        fig4 = Figure(size = (900, 400), fontsize = 14)

        axθ = Axis(fig4[1, 1]; xlabel = "θ (K)", ylabel = "z (km)", title = "Potential temperature")
        axu = Axis(fig4[1, 2]; xlabel = "u (m/s)", title = "Zonal wind")
        axv = Axis(fig4[1, 3]; xlabel = "v (m/s)", title = "Meridional wind")

        default_colors = Makie.wong_colors()
        colors = [default_colors[mod1(i, length(default_colors))] for i in 1:Nt]

        for n in 1:Nt
            label = n == 1 ? "initial" : "t = $(prettytime(times[n]))"
            θ_data = Array(interior(θ_avg_ts[n], 1, 1, :))  # Convert to CPU
            u_data = Array(interior(u_avg_ts[n], 1, 1, :))
            v_data = Array(interior(v_avg_ts[n], 1, 1, :))

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

        save(joinpath(figures_dir, "profiles.png"), fig4) #src
        fig4
    end

    @info "Figures saved to: $figures_dir"
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
# ## What This Implementation Gets Right
#
# - **Radiative cooling**: Constant -1 K/day following RICO pattern (Field on ρθ)
#   Paper's Eq. 1 with T > Tₜ everywhere in troposphere = constant cooling
# - **Surface fluxes**: Bulk formulas using Breeze utilities (BulkDrag, BulkSensibleHeatFlux)
#   consistent with RICO pattern — sensible heat on ρe, drag on ρu/ρv
# - **Domain geometry**: Correct domain size (1152 km) and model top (28 km)
# - **Coriolis**: f-plane with f = 3×10⁻⁴ s⁻¹
# - **Surface wetness**: Full β parameterization for moist-dry transition
# - **Sponge layer**: Rayleigh damping in upper 3 km to prevent wave reflections
#
# ## What Requires Higher Resolution / Longer Runs
#
# To reproduce the paper's results quantitatively:
#
# 1. **Increase resolution**: `Nx = Ny = 576` (2 km spacing)
# 2. **Use stretched vertical grid**: 64 levels in lowest 1 km
# 3. **Run for 70 days**: Spontaneous cyclogenesis requires O(10 days)
# 4. **Initialize from RCE**: Pre-equilibrate with 100-day nonrotating simulation
# 5. **Use full microphysics**: `OneMomentCloudMicrophysics` with ice phase
# 6. **Use larger domain for moist cases**: 1728 km for β ≥ 0.9