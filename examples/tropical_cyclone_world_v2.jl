# # Tropical Cyclone World (Cronin & Chavas, 2019)
#
# This example implements the rotating radiative-convective equilibrium (RCE) experiment
# from [Cronin and Chavas (2019)](@cite Cronin2019), which demonstrates that tropical cyclones
# can form and persist even in completely dry atmospheres — challenging the conventional
# wisdom that moisture is essential for TC dynamics.
#
# ## Surface Wetness Parameter β
#
# The key innovation is the **surface wetness parameter β** (0-1):
# - β = 0: Completely dry surface (no evaporation) — dry TCs!
# - β = 1: Fully moist surface (standard TC behavior)
# - Intermediate values: "semidry" TCs
#
# **This script defaults to β = 1 (moist)**, which produces robust spontaneous TC genesis
# at moderate resolution. See "Lessons Learned" below for why.
#
# ## Implementation Notes
#
# ### Resolution and Domain (Test Configuration)
# - **Horizontal resolution**: 8 km spacing (144×144) vs. paper's 2 km (576×576)
# - **Vertical grid**: ~36 levels (4x coarser) vs. paper's ~133 levels (64 in lowest 1 km)
# - **Domain size**: 1152 km × 1152 km doubly-periodic
# - **Turbulence**: ILES (no explicit diffusivity) vs. paper's 1.5-order TKE
#
# To run at paper resolution: Nx=Ny=576, scale_factor=1
#
# ### Initialization
# - **Equilibrated profile**: Dry adiabat in troposphere, isothermal at Tₜ in stratosphere
# - This avoids the paper's 100-day nonrotating RCE spinup by starting near equilibrium
#
# ### Microphysics
# - **Scheme**: `SaturationAdjustment` (equilibrium condensation, no precipitation fallout)
# - **Ice phase**: Warm-phase only (`WarmPhaseEquilibrium`)
#
# ### Radiative Cooling (Paper Eq. 1)
# - Troposphere (T > Tₜ): constant cooling at -Q̇ = -1 K/day
# - Stratosphere (T ≤ Tₜ): Newtonian relaxation toward Tₜ with τ = 20 days
# - Applied to ρe (energy density) following BOMEX pattern
#
# ## Lessons Learned (for LLM Agents)
#
# 1. **Moist physics enables spontaneous TC genesis**: β=1 produces organized TCs within
#    ~5 days at 8km resolution. Dry (β=0) requires paper resolution (2km), seeding, or
#    extreme forcing to produce TCs.
#
# 2. **Domain size constrains TC formation**: The domain must be large enough to contain
#    multiple vortices for the upscale cascade (vortex merger → organized TC). The 1152 km
#    domain works well; smaller domains produce "lattice" equilibria.
#
# 3. **Initial conditions matter**: Starting from an equilibrated temperature profile
#    (near RCE) prevents transient numerical instability. Don't use uniform θ.
#
# 4. **Monitor long simulations**: Periodic figure updates during the run are essential
#    for catching issues early. See the `update_figures` callback.
#
# 5. **WISHE feedback**: Wind-induced surface heat exchange is the primary TC intensification
#    mechanism. Stronger surface-air temperature disequilibrium drives stronger heat fluxes.

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

experiment_name = "tc_world"

# ## Experiment Parameters
#
# These parameters follow Cronin & Chavas (2019), Section 2a.
# The primary control parameter is surface wetness β.

# ### Primary control parameters (user-configurable)

β  = 1.0    # Surface wetness: 0 = dry, 1 = moist (default: moist for robust TC genesis)
Tₛ = 300.0  # Surface temperature (K) — paper value
Tₜ = 210.0  # Tropopause temperature (K) — paper value
θ_init = 300.0  # Initial atmospheric θ (K) — paper value (near-equilibrium)

# Warm-core vortex seed parameters (to help TC genesis at coarse resolution)
seed_vortex = false       # Disable seeding — test spontaneous genesis
seed_θ_anomaly = 0.0      # K — no warm core anomaly
seed_radius = 100e3       # m — horizontal radius of warm core
seed_height = 10e3        # m — vertical extent of warm core
# Note: seed_x and seed_y are set after Lx/Ly are defined (see below)

# ### Physical parameters from the paper

Cᴰ = 1.5e-3      # Drag coefficient (paper value)
# Heat exchange efficiency factor (1.0 = paper value, increase for coarse vertical grids)
heat_exchange_efficiency = 1.0
Cᵀ = Cᴰ * heat_exchange_efficiency  # 1.5e-3 (paper value)
v★ = 1.0         # Gustiness / minimum wind speed (m/s)
Q̇  = 1.0 / day   # Radiative cooling rate (K/s) — paper value
τᵣ = 20days      # Newtonian relaxation timescale for T ≤ Tₜ
# Coriolis parameter: paper value
f₀ = 3e-4        # Coriolis parameter (s⁻¹) — paper value

# ## Domain and Grid
#
# The paper uses a 1152 km × 1152 km doubly-periodic domain with 2 km horizontal
# resolution and a 28 km model top. The vertical grid is stretched following
# the paper's Section 2a specification EXACTLY:
# - "64 levels in the lowest kilometer"
# - "spacing of 500 m above 3.5 km"
# - Linear transition from 1 km to 3.5 km

arch = GPU()

# Domain size (full paper domain)
Lx = Ly = 1152e3  # 1152 km (paper domain size)
H  = 28e3         # 28 km model top

# Seed position (center of domain)
seed_x = 0.5 * Lx  # m — x-position
seed_y = 0.5 * Ly  # m — y-position

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

# Resolution: 8 km spacing (coarse resolution for testing)
Nx = Ny = 144     # 8 km spacing in 1152 km domain

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

z_faces = paper_vertical_grid(H; scale_factor=4)  # Coarse vertical resolution (36 levels)
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

# ## Radiative Forcing (Paper-Equivalent)
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
# **Thermodynamic Conversion**: Since Breeze uses potential temperature (θ), we need
# to convert the paper's T-based forcing to θ-based forcing. Using T = θ × Π:
#
#   ∂T/∂t = Π × ∂θ/∂t  →  ∂θ/∂t = (∂T/∂t) / Π
#
# where Π = (p/pˢᵗ)^(R/cₚ) is the Exner function. This means:
# - At surface (Π ≈ 1): ∂θ/∂t ≈ ∂T/∂t
# - At altitude (Π < 1): ∂θ/∂t > ∂T/∂t (stronger θ cooling to achieve same T cooling)
#
# The forcing on ρθ is then: F_ρθ = ρ × ∂θ/∂t = ρ × (∂T/∂t) / Π

# Extract reference state fields for discrete forcing
ρᵣ = reference_state.density
pᵣ = reference_state.pressure
pˢᵗ = reference_state.standard_pressure

# Thermodynamic constants for Exner function
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ = Rᵈ / cᵖᵈ  # R/cₚ ≈ 0.286 for dry air

# Relaxation timescale for stratosphere (paper uses 20 days)
τᵣ = 20days

# ## Piecewise Radiative Forcing (Paper's Eq. 1)
#
# Following BOMEX pattern: apply T-based forcing to ρe, letting Breeze handle
# the conversion to ρθ tendency via division by (cᵖᵐ × Π).
#
# F_ρe = ρ × cₚ × ∂T/∂t
#
# Breeze computes: G_ρθ += F_ρe / (cᵖᵐ × Π) = ρ × ∂T/∂t / Π = ρ × ∂θ/∂t
#
# With equilibrated initial conditions (T ≈ Tₜ in stratosphere), this is stable.

forcing_params = (; Tₜ = FT(Tₜ), Q̇ = FT(Q̇), τ = FT(τᵣ), ρᵣ, cₚ = FT(cᵖᵈ))

# Discrete form forcing applied to ρe (energy density)
@inline function piecewise_T_forcing(i, j, k, grid, clock, model_fields, p)
    # Get temperature from model diagnostics
    @inbounds T = model_fields.T[i, j, k]
    @inbounds ρ = p.ρᵣ[i, j, k]

    # Paper's piecewise T tendency (Eq. 1):
    # - T > Tₜ: constant cooling at -Q̇
    # - T ≤ Tₜ: Newtonian relaxation toward Tₜ
    ∂T∂t = ifelse(T > p.Tₜ, -p.Q̇, (p.Tₜ - T) / p.τ)

    # Return energy forcing: F_ρe = ρ × cₚ × ∂T/∂t
    # Breeze will divide by (cᵖᵐ × Π) to get ρθ tendency
    return ρ * p.cₚ * ∂T∂t
end

ρe_radiation_forcing = Forcing(piecewise_T_forcing;
                               discrete_form = true,
                               parameters = forcing_params)

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

# Assemble all forcings: piecewise radiative forcing on ρe (BOMEX pattern), sponge on ρw
forcing = (; ρe = ρe_radiation_forcing, ρw = ρw_sponge)

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
# We initialize with an equilibrated temperature profile following the paper's RCE:
# - Troposphere: dry adiabatic (θ ≈ θ₀), T decreases with height
# - Stratosphere: isothermal at T = Tₜ = 210 K (θ = Tₜ/Π increases with height)
#
# This ensures the piecewise radiative forcing starts near equilibrium:
# - Troposphere (T > Tₜ): moderate cooling at 1 K/day
# - Stratosphere (T ≈ Tₜ): minimal forcing (already at target temperature)
#
# The paper initialized from 100-day nonrotating RCE spinup; we approximate that
# equilibrium profile directly.

# Use θ_init for initial condition (NOT reference_state.potential_temperature = Tₛ)
# This creates surface-air disequilibrium: Tₛ - θ_init ≈ 14 K at startup (near equilibrium)
θ₀ = θ_init  # Use the separate initial θ parameter
pᵣ_field = reference_state.pressure
pˢᵗ_val = reference_state.standard_pressure

# Convert pressure field to array for initialization function
pᵣ_array = Array(interior(pᵣ_field))

# Equilibrated θ profile: dry adiabat in troposphere, isothermal stratosphere
function equilibrated_θ_profile(x, y, z, k, pᵣ_arr, θ_surface, T_tropopause, p_standard, kappa)
    # Get pressure at this level (use k index)
    p_local = pᵣ_arr[1, 1, k]

    # Exner function
    Π = (p_local / p_standard)^kappa

    # Temperature on dry adiabat
    T_adiabat = θ_surface * Π

    if T_adiabat > T_tropopause
        # Troposphere: follow dry adiabat
        return θ_surface
    else
        # Stratosphere: isothermal at Tₜ, so θ = Tₜ/Π
        return T_tropopause / Π
    end
end

# Random perturbation parameters
δθ = 0.5   # K (perturbation amplitude)
zδ = 1000  # m (depth of perturbation layer)

# Build 3D θ initial condition array
# Convert GPU arrays to CPU for initialization loop
x_nodes = Array(Oceananigans.Grids.xnodes(grid, Center()))
y_nodes = Array(Oceananigans.Grids.ynodes(grid, Center()))
z_nodes = Array(Oceananigans.Grids.znodes(grid, Center()))
θᵢ_array = zeros(FT, Nx, Ny, Nz)

for k in 1:Nz
    z = z_nodes[k]
    θ_eq = equilibrated_θ_profile(0, 0, z, k, pᵣ_array, θ₀, Tₜ, pˢᵗ_val, κ)
    for j in 1:Ny
        for i in 1:Nx
            # Add random perturbation in lowest 1 km
            perturbation = z < zδ ? δθ * (2 * rand() - 1) : 0.0

            # Add warm-core vortex seed if enabled
            if seed_vortex && z < seed_height
                x = x_nodes[i]
                y = y_nodes[j]
                r = sqrt((x - seed_x)^2 + (y - seed_y)^2)
                # Gaussian warm core: max at center, decays with radius and height
                warm_core = seed_θ_anomaly * exp(-r^2 / (2 * seed_radius^2)) * (1 - z / seed_height)
                perturbation += warm_core
            end

            θᵢ_array[i, j, k] = θ_eq + perturbation
        end
    end
end

if seed_vortex
    @info "Added warm-core vortex seed: Δθ = $seed_θ_anomaly K, radius = $(seed_radius/1e3) km, height = $(seed_height/1e3) km"
end

if β > 0
    # Moisture profile for moist cases
    q_surface = 0.015  # ~15 g/kg near tropical surface
    q_scale_height = 3000.0  # m
    δq = 1e-4  # kg/kg (perturbation amplitude)

    qᵢ_array = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz
        z = z_nodes[k]
        q_mean = β * q_surface * exp(-z / q_scale_height)
        for j in 1:Ny
            for i in 1:Nx
                perturbation = z < zδ ? δq * (2 * rand() - 1) : 0.0
                qᵢ_array[i, j, k] = max(0, q_mean + perturbation)
            end
        end
    end

    set!(model, θ = θᵢ_array, qᵗ = qᵢ_array)
else
    set!(model, θ = θᵢ_array)
end

# Log initial θ profile info
θ_min, θ_max = extrema(θᵢ_array)
@info "Initial θ profile: θ ∈ [$θ_min, $θ_max] K"
@info "Surface-air disequilibrium: Tₛ - θ_init = $(Tₛ - θ_init) K (drives strong sensible heat flux)"

# ## Simulation Setup

Δt = 10.0  # Initial timestep (seconds) — 8 km resolution allows larger dt
stop_time = 10days   # Moist TCs form within ~5 days; 10 days shows full intensification

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

# Create figures directory early for periodic monitoring
figures_dir = joinpath(experiment_dir, "figures")
mkpath(figures_dir)

# ### Periodic figure updates for monitoring
#
# Generate all plots periodically so we can monitor TC development during the run.

function update_figures(sim)
    # Only update if we have enough data
    t_days = sim.model.clock.time / 86400
    if t_days < 0.5
        return nothing  # Skip early timesteps
    end

    try
        surface_file = joinpath(experiment_dir, "surface.jld2")
        profile_file = joinpath(experiment_dir, "profiles.jld2")

        if !isfile(surface_file)
            return nothing
        end

        u_ts = FieldTimeSeries(surface_file, "u_surface")
        v_ts = FieldTimeSeries(surface_file, "v_surface")
        times = u_ts.times
        Nt = length(times)

        if Nt < 2
            return nothing
        end

        # Compute wind speed for all timesteps
        wind_speed(n) = sqrt.(Array(interior(u_ts[n])).^2 .+ Array(interior(v_ts[n])).^2)
        max_wind = [maximum(wind_speed(n)) for n in 1:Nt]
        times_hours = times ./ 3600

        # --- Figure 1: Intensity plot ---
        title_case = β == 0 ? "Dry" : (β == 1 ? "Moist" : "Semidry")
        fig1 = Figure(size = (600, 400), fontsize = 14)
        ax1 = Axis(fig1[1, 1];
                   xlabel = "Time (hours)",
                   ylabel = "Maximum surface wind speed (m/s)",
                   title = "$title_case TC (β = $β) — Intensity Evolution")
        lines!(ax1, times_hours, max_wind; linewidth = 2, color = :dodgerblue)
        scatter!(ax1, times_hours, max_wind; markersize = 4, color = :dodgerblue)
        save(joinpath(figures_dir, "intensity.png"), fig1)

        # --- Figure 2: Surface wind snapshots ---
        x_km = Array(xnodes(grid, Center())) ./ 1e3
        y_km = Array(ynodes(grid, Center())) ./ 1e3
        U_max = max(1.0, maximum(max_wind))

        fig2 = Figure(size = (1200, 400), fontsize = 12)
        indices = [1, max(1, Nt ÷ 2), Nt]
        for (i, n) in enumerate(indices)
            ax = Axis(fig2[1, i];
                      xlabel = "x (km)", ylabel = i == 1 ? "y (km)" : "",
                      title = "t = $(prettytime(times[n]))",
                      aspect = 1)
            hm = heatmap!(ax, x_km, y_km, wind_speed(n);
                          colormap = :viridis, colorrange = (0, U_max))
            if i == length(indices)
                Colorbar(fig2[1, i+1], hm; label = "Surface wind speed (m/s)")
            end
        end
        Label(fig2[0, :], "$title_case Tropical Cyclone World (β = $β) — Surface Wind Speed",
              fontsize = 16, tellwidth = false)
        save(joinpath(figures_dir, "surface_winds.png"), fig2)

        # --- Figure 3: Profile evolution (if available) ---
        if isfile(profile_file)
            θ_avg_ts = FieldTimeSeries(profile_file, "θ_avg")
            u_avg_ts = FieldTimeSeries(profile_file, "u_avg")
            v_avg_ts = FieldTimeSeries(profile_file, "v_avg")
            prof_times = θ_avg_ts.times
            Nt_prof = length(prof_times)
            z_km = Array(znodes(grid, Center())) ./ 1e3

            fig3 = Figure(size = (900, 400), fontsize = 14)
            axθ = Axis(fig3[1, 1]; xlabel = "θ (K)", ylabel = "z (km)", title = "Potential temperature")
            axu = Axis(fig3[1, 2]; xlabel = "u (m/s)", title = "Zonal wind")
            axv = Axis(fig3[1, 3]; xlabel = "v (m/s)", title = "Meridional wind")

            prof_indices = [1, max(1, Nt_prof ÷ 2), Nt_prof]
            colors = [:blue, :green, :red]
            for (i, n) in enumerate(prof_indices)
                label = "t = $(prettytime(prof_times[n]))"
                lines!(axθ, Array(vec(interior(θ_avg_ts[n]))), z_km; color = colors[i], label = label)
                lines!(axu, Array(vec(interior(u_avg_ts[n]))), z_km; color = colors[i])
                lines!(axv, Array(vec(interior(v_avg_ts[n]))), z_km; color = colors[i])
            end
            axislegend(axθ; position = :rt)
            Label(fig3[0, :], "$title_case TC (β = $β) — Mean Profile Evolution", fontsize = 16, tellwidth = false)
            save(joinpath(figures_dir, "profiles.png"), fig3)
        end

        @info "Updated all figures at t = $(prettytime(sim.model.clock.time))"
    catch e
        @warn "Figure update failed: $e"
        @warn "Stacktrace: $(catch_backtrace())"
    end

    return nothing
end

add_callback!(simulation, update_figures, TimeInterval(12hours))

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
            u_data = Array(interior(u_ts[n], :, :, 1))
            v_data = Array(interior(v_ts[n], :, :, 1))
            return sqrt.(u_data.^2 .+ v_data.^2)
        end

        # Determine color range from all data (ensure U_max ≥ 1 to avoid colormap issues)
        U_max = max(1.0, maximum(maximum(wind_speed(n)) for n in 1:Nt))

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
# - **β = 1** (default): Moist — robust spontaneous TC genesis at 8 km resolution
# - **β = 0**: Dry — requires paper resolution (2 km), seeding, or extreme forcing
# - **0 < β < 1**: Semidry — intermediate behavior, "no-storms-land" at β ≈ 0.01-0.3
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
# ## Implementation Details
#
# - **Radiative cooling**: Piecewise (Eq. 1) — constant -1 K/day in troposphere (T > Tₜ),
#   Newtonian relaxation in stratosphere (T ≤ Tₜ)
# - **Surface fluxes**: Bulk formulas (BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux)
# - **Domain geometry**: 1152 km × 1152 km doubly-periodic, 28 km model top
# - **Coriolis**: f-plane with f = 3×10⁻⁴ s⁻¹
# - **Sponge layer**: Rayleigh damping in upper 3 km
#
# ## Reproducing Paper Results Quantitatively
#
# For full reproduction of Cronin & Chavas (2019):
#
# 1. **Resolution**: `Nx = Ny = 576` (2 km), `scale_factor = 1` (133 vertical levels)
# 2. **Runtime**: 70 days (spontaneous dry TC genesis requires O(10-30 days))
# 3. **Domain**: 1728 km for β ≥ 0.9
# 4. **Microphysics**: Full bulk scheme with precipitation (CloudMicrophysics.jl extension)
# 5. **Turbulence**: 1.5-order TKE closure instead of ILES
