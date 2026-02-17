# # Inertia-gravity waves
#
# This example simulates the propagation of inertia-gravity waves in a stably stratified
# atmosphere, following the classical benchmark test case described by [SkamarockKlemp1994](@citet).
# This test evaluates the accuracy of numerical pressure solvers by introducing a small-amplitude
# temperature perturbation into a stratified environment with constant Brunt-Väisälä frequency,
# triggering propagating inertia-gravity waves.
#
# The test case is particularly useful for validating anelastic and compressible solvers,
# as discussed at the [CM1 inertia-gravity wave test page](https://www2.mmm.ucar.edu/people/bryan/cm1/test_inertia_gravity_waves/).
#
# ## Physical setup
#
# The background state is a stably stratified atmosphere with constant Brunt-Väisälä frequency ``N``,
# which gives a potential temperature profile
#
# ```math
# θ^{\rm bg}(z) = θ_0 \exp\left( \frac{N^2 z}{g} \right)
# ```
#
# where ``θ_0 = 300 \, {\rm K}`` is the surface potential temperature and ``g`` is
# the gravitational acceleration.
#
# The initial perturbation is a localized temperature anomaly centered at ``x = x_0``:
#
# ```math
# θ'(x, z) = Δθ \frac{\sin(π z / L_z)}{1 + (x - x_0)^2 / a^2}
# ```
#
# with amplitude ``Δθ = 0.01 \, {\rm K}``, half-width parameter ``a = 5000 \, {\rm m}``,
# and perturbation center ``x_0 = L_x / 3``. A uniform mean wind ``U = 20 \, {\rm m \, s^{-1}}``
# advects the waves.
#
# ## Comparison of dynamical formulations
#
# We compare six dynamical formulations:
#
# 1. **Anelastic**: Filters acoustic waves via the anelastic approximation
# 2. **Compressible (explicit)**: Fully compressible with explicit time stepping (small Δt)
# 3. **Boussinesq**: Anelastic with constant reference density
# 4. **Split-explicit (adaptive SSP-RK3)**: Acoustic substepping with adaptive substep count at advective Δt
# 5. **Split-explicit SSP-RK3**: Acoustic substepping with SSP-RK3 outer loop at Δt = 12 s
# 6. **Split-explicit WS-RK3**: Acoustic substepping with Wicker-Skamarock RK3 outer loop at Δt = 12 s

using Breeze
using Oceananigans.Units
using Printf
using CairoMakie

# ## Problem parameters
#
# We define the thermodynamic base state and mean wind following [SkamarockKlemp1994](@citet):

p₀ = 100000  # Pa - surface pressure
θ₀ = 300     # K - reference potential temperature
U  = 20      # m s⁻¹ - mean wind
N² = 0.01^2  # s⁻² - Brunt-Väisälä frequency squared

# ## Grid configuration
#
# The domain is 300 km × 10 km with 300 × 10 grid points, matching the nonhydrostatic case
# configuration in the paper by [SkamarockKlemp1994](@citet).

Nx, Nz = 300, 10
Lx, Lz = 300kilometers, 10kilometers

grid = RectilinearGrid(CPU(), size=(Nx, Nz), halo=(5, 5),
                       x=(0, Lx), z=(0, Lz),
                       topology=(Periodic, Flat, Bounded))

# ## Initial conditions
#
# The perturbation parameters from the paper by [SkamarockKlemp1994](@citet):

Δθ = 0.01    # K - perturbation amplitude
a  = 5000    # m - perturbation half-width parameter
x₀ = Lx / 3 # m - perturbation center in x

constants = ThermodynamicConstants()
g = constants.gravitational_acceleration
pˢᵗ = 1e5

θᵇᵍ(z) = θ₀ * exp(N² * z / g)
θᵢ(x, z) = θᵇᵍ(z) + Δθ * sin(π * z / Lz) / (1 + (x - x₀)^2 / a^2)
ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)

# ## Build all six models

advection = WENO()
Ns = 8 # acoustic substeps for fixed-substep cases
surface_pressure = p₀
potential_temperature = θ₀

# Case 1: Anelastic

reference_state = ReferenceState(grid, constants; surface_pressure, potential_temperature)
anelastic_dynamics = AnelasticDynamics(reference_state)

# Case 2: Compressible (fully explicit, no substepping)
compressible_dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                              surface_pressure,
                                              reference_potential_temperature=θᵇᵍ)

# Case 3: Boussinesq (constant reference density)
constant_density_reference_state = ReferenceState(grid, constants; surface_pressure, potential_temperature)

ρ₀ = adiabatic_hydrostatic_density(0, p₀, θ₀, pˢᵗ, constants)
set!(constant_density_reference_state.density, ρ₀)
boussinesq_dynamics = AnelasticDynamics(constant_density_reference_state)

# Case 4: Split-explicit with adaptive substeps at the advective time step.
# The number of acoustic substeps is computed automatically from the CFL condition
# each time step: `N = ceil(safety_factor · Δt · cₛ / Δx_min)`.
# We use SSP-RK3 because it is stable at larger advective CFL than WS-RK3.
adaptive_dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                          surface_pressure,
                                          reference_potential_temperature=θᵇᵍ)

# Case 5: Split-explicit with SSP-RK3 outer loop
# Uses acoustic substepping with Exner pressure variables (velocity + π') and
# vertically implicit w-π' coupling. The reference potential temperature enables
# base-state subtraction for accurate perturbation pressure.
ssp_time_discretization = SplitExplicitTimeDiscretization(substeps=Ns, divergence_damping_coefficient=0.05)
ssp_dynamics = CompressibleDynamics(ssp_time_discretization;
                                      surface_pressure,
                                      reference_potential_temperature=θᵇᵍ)

# Case 6: Split-explicit with Wicker-Skamarock RK3 outer loop
# Same acoustic substepping as SSP-RK3, but with WS-RK3 stage fractions (Δt/3, Δt/2, Δt)
# instead of SSP convex combinations. This is the default for CompressibleDynamics with
# SplitExplicitTimeDiscretization.
ws_time_discretization = SplitExplicitTimeDiscretization(substeps=Ns, divergence_damping_coefficient=0.10)
ws_dynamics = CompressibleDynamics(ws_time_discretization;
                                     surface_pressure,
                                     reference_potential_temperature=θᵇᵍ)

# Build all models:
models = Dict(
    :anelastic      => AtmosphereModel(grid; advection, dynamics=anelastic_dynamics),
    :compressible   => AtmosphereModel(grid; advection, dynamics=compressible_dynamics),
    :boussinesq     => AtmosphereModel(grid; advection, dynamics=boussinesq_dynamics),
    :adaptive       => AtmosphereModel(grid; advection, dynamics=adaptive_dynamics,
                                       timestepper=:AcousticSSPRungeKutta3),
    :ssp_rk3        => AtmosphereModel(grid; advection, dynamics=ssp_dynamics,
                                       timestepper=:AcousticSSPRungeKutta3),
    :ws_rk3         => AtmosphereModel(grid; advection, dynamics=ws_dynamics),
)

# Set initial conditions:

for name in (:anelastic, :boussinesq)
    set!(models[name]; θ=θᵢ, u=U)
end

for name in (:compressible, :adaptive, :ssp_rk3, :ws_rk3)
    ref = models[name].dynamics.reference_state
    set!(models[name]; θ=θᵢ, u=U, qᵗ=0, ρ=ref.density)
end

# ## Time stepping constraints
#
# The anelastic and split-explicit models use the advective CFL,
# while the fully explicit compressible model is limited by the sound speed.

Δx, Δz = Lx / Nx, Lz / Nz
Rᵈ = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
cₛ = sqrt(cᵖᵈ / (cᵖᵈ - Rᵈ) * Rᵈ * θ₀)

cfl = 0.5
Δt_advective    = cfl * min(Δx, Δz) / U
Δt_compressible = cfl * min(Δx, Δz) / (cₛ + U)
Δt_split        = 12.0

time_steps = Dict(
    :anelastic      => Δt_advective,
    :compressible   => Δt_compressible,
    :boussinesq     => Δt_advective,
    :adaptive       => Δt_advective,
    :ssp_rk3        => Δt_split,
    :ws_rk3         => Δt_split,
)

@info "Time steps" Δt_advective Δt_compressible Δt_split

# ## Run all simulations

stop_time = 3000 # seconds

case_names = Dict(
    :anelastic      => "Anelastic",
    :compressible   => "Compressible (explicit)",
    :boussinesq     => "Boussinesq",
    :adaptive       => "Split-explicit (adaptive SSP-RK3)",
    :ssp_rk3        => "Split-explicit (SSP-RK3)",
    :ws_rk3         => "Split-explicit (WS-RK3)",
)

# Background θ field for computing perturbation
θᵇᵍ_field = CenterField(grid)
set!(θᵇᵍ_field, (x, z) -> θᵇᵍ(z))

simulations = Dict{Symbol, Simulation}()

for (key, model) in models
    Δt = time_steps[key]
    sim = Simulation(model; Δt, stop_time)

    ## Progress callback
    θ′ = PotentialTemperature(model) - θᵇᵍ_field
    name = case_names[key]

    function progress(sim)
        w = sim.model.velocities.w
        @info @sprintf("%s - Iter: %4d, t: %s, max(θ′): %.4e, max|w|: %.4f",
                       name, iteration(sim), prettytime(sim), maximum(θ′), maximum(abs, w))
        return nothing
    end

    callback_interval = key ∈ (:compressible,) ? IterationInterval(500) : IterationInterval(100)
    add_callback!(sim, progress, callback_interval)

    ## Output
    outputs = merge(model.velocities, (; θ′))
    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = "igw_$(key).jld2",
                                           schedule = TimeInterval(100),
                                           overwrite_existing = true)
    simulations[key] = sim
end

for (key, sim) in simulations
    @info "Running $(case_names[key])..."
    run!(sim)
end

# ## Results
#
# We compare the potential temperature perturbation at the final time for all five
# formulations. This comparison reveals how well each formulation captures
# inertia-gravity wave propagation.

cases = [:anelastic, :compressible, :boussinesq, :adaptive, :ssp_rk3, :ws_rk3]
titles = [case_names[k] for k in cases]

θ′ts = Dict(k => FieldTimeSeries("igw_$(k).jld2", "θ′") for k in cases)
times = θ′ts[:anelastic].times
Nt = length(times)

x_km = range(0, Lx / 1000, length=Nx)
z_km = range(0, Lz / 1000, length=Nz)
levels = range(-Δθ / 2, stop=Δθ / 2, length=21)
θ′_final = Dict(k => interior(θ′ts[k][Nt], :, 1, :) for k in cases)

# ## Contour comparison

fig = Figure(size=(1400, 900))

# 2×3 layout: top row [anelastic, compressible], middle row [boussinesq, adaptive],
#             bottom row [ssp_rk3, ws_rk3]
axes_layout = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
axes = [Axis(fig[r, c]; title=titles[i],
             ylabel = c == 1 ? "z (km)" : "",
             xlabel = r == 2 ? "x (km)" : "")
        for (i, (r, c)) in enumerate(axes_layout)]

for ax in axes; if ax.xlabel[] == ""; hidexdecorations!(ax, grid=false); end; end
for ax in axes; if ax.ylabel[] == ""; hideydecorations!(ax, grid=false); end; end

cfs = [contourf!(axes[i], x_km, z_km, θ′_final[k]; colormap=:balance, levels)
       for (i, k) in enumerate(cases)]

Colorbar(fig[1:2, 4], first(cfs); label="θ′ (K)")
fig[0, :] = Label(fig, "Inertia-gravity waves: θ′ at t = $(prettytime(times[Nt]))", fontsize=20)

save("inertia_gravity_wave_comparison.png", fig)

fig

# ## Animation

n = Observable(1)

fig_anim = Figure(size=(1400, 900))
anim_axes = [Axis(fig_anim[r, c]; title=titles[i],
                   ylabel = c == 1 ? "z (km)" : "",
                   xlabel = r == 2 ? "x (km)" : "")
             for (i, (r, c)) in enumerate(axes_layout)]

for ax in anim_axes; if ax.xlabel[] == ""; hidexdecorations!(ax, grid=false); end; end
for ax in anim_axes; if ax.ylabel[] == ""; hideydecorations!(ax, grid=false); end; end

hm_anims = [contourf!(anim_axes[i], @lift(θ′ts[k][$n]);
                      colormap=:balance, levels, extendhigh=:auto, extendlow=:auto)
            for (i, k) in enumerate(cases)]

Colorbar(fig_anim[1:2, 4], last(hm_anims); label="θ′ (K)")
anim_title = @lift "Inertia-gravity waves: θ′ at t = $(prettytime(times[$n]))"
fig_anim[0, :] = Label(fig_anim, anim_title, fontsize=20, tellwidth=false)

record(fig_anim, "inertia_gravity_wave.mp4", 1:Nt, framerate=8) do nn
    n[] = nn
end
nothing #hide

# ![](inertia_gravity_wave.mp4)

# ## Cross-section comparison
#
# A vertical cross-section at mid-height shows the wave phase and amplitude differences:

z_mid = Nz ÷ 2
linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot, :solid]
colors = Makie.wong_colors()

fig_cross = Figure(size=(900, 400))
ax = Axis(fig_cross[1, 1], xlabel="x (km)", ylabel="θ′ (K)",
          title="θ′ at z = $(round(z_km[z_mid], digits=1)) km, t = $(prettytime(times[Nt]))")

for (i, k) in enumerate(cases)
    lines!(ax, x_km, θ′_final[k][:, z_mid];
           label=titles[i], linewidth=2, linestyle=linestyles[i], color=colors[i])
end

axislegend(ax, position=:rt)
save("inertia_gravity_wave_cross_section.png", fig_cross)

fig_cross
