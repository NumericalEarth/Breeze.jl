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
# We compare five dynamical formulations:
#
# 1. **Anelastic**: Filters acoustic waves via the anelastic approximation
# 2. **Boussinesq**: Anelastic with constant reference density
# 3. **Compressible (explicit)**: Fully compressible with explicit time stepping
# 4. **Split-explicit (explicit vertical)**: Acoustic substepping with explicit vertical
# 5. **Split-explicit (implicit vertical)**: Acoustic substepping with implicit vertical solve

using Breeze
using Breeze.CompressibleEquations: ExplicitTimeStepping
using Breeze.Thermodynamics: adiabatic_hydrostatic_density
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

# ## Build all five models

advection = WENO()
Ns = 6 # acoustic substeps
surface_pressure = p₀
potential_temperature = θ₀

# Case 1: Anelastic

reference_state = ReferenceState(grid, constants; surface_pressure, potential_temperature)
anelastic_dynamics = AnelasticDynamics(reference_state)

# Case 2: Boussinesq (constant reference density)
constant_density_reference_state = ReferenceState(grid, constants; surface_pressure, potential_temperature)

ρ₀ = adiabatic_hydrostatic_density(0, p₀, θ₀, pˢᵗ, constants)
set!(constant_density_reference_state.density, ρ₀)
boussinesq_dynamics = AnelasticDynamics(constant_density_reference_state)

# Case 3: Compressible (fully explicit, no substepping)
compressible_dynamics = CompressibleDynamics(; surface_pressure, time_discretization=ExplicitTimeStepping())

# Case 4: Split-explicit with explicit vertical substepping
time_discretization = SplitExplicitTimeDiscretization(substeps=Ns)
explicit_split_dynamics = CompressibleDynamics(; surface_pressure, time_discretization)

# Case 5: Split-explicit with vertically implicit substepping
time_discretization = SplitExplicitTimeDiscretization(VerticallyImplicit(0.5), substeps=Ns)
implicit_split_dynamics = CompressibleDynamics(; surface_pressure, time_discretization)

# Build all models:
models = Dict(
    :anelastic      => AtmosphereModel(grid; advection, dynamics=anelastic_dynamics),
    :boussinesq     => AtmosphereModel(grid; advection, dynamics=boussinesq_dynamics),
    :compressible   => AtmosphereModel(grid; advection, dynamics=compressible_dynamics),
    :explicit_split => AtmosphereModel(grid; advection, dynamics=explicit_split_dynamics),
    :implicit_split => AtmosphereModel(grid; advection, dynamics=implicit_split_dynamics),
)

# Set initial conditions:

for name in (:anelastic, :boussinesq)
    set!(models[name]; θ=θᵢ, u=U)
end

for name in (:compressible, :explicit_split, :implicit_split)
    set!(models[name]; θ=θᵢ, u=U, qᵗ=0, ρ=ρᵢ)
end
    
# ## Time stepping constraints
#
# The anelastic and split-explicit models use the advective CFL,
# while the fully explicit compressible model is limited by the sound speed.

Δx, Δz = Lx / Nx, Lz / Nz
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
cₛ = sqrt(cᵖᵈ / (cᵖᵈ - Rᵈ) * Rᵈ * θ₀)

cfl = 0.5
Δt_advective    = cfl * min(Δx, Δz) / U
Δt_compressible = cfl * min(Δx, Δz) / (cₛ + U)
Δt_split        = 2.0

time_steps = Dict(
    :anelastic      => Δt_advective,
    :boussinesq     => Δt_advective,
    :compressible   => Δt_compressible,
    :explicit_split => Δt_split,
    :implicit_split => Δt_split,
)

@info "Time steps" Δt_advective Δt_compressible Δt_split

# ## Run all simulations

stop_time = 3000 # seconds

case_names = Dict(
    :anelastic      => "Anelastic",
    :boussinesq     => "Boussinesq",
    :compressible   => "Compressible",
    :explicit_split => "Split (explicit vert)",
    :implicit_split => "Split (implicit vert)",
)

# Background θ field for computing perturbation
θᵇᵍ_field = CenterField(grid)
set!(θᵇᵍ_field, (x, z) -> θᵇᵍ(z))

simulations = Dict{Symbol, Simulation}()

for (key, model) in models
    Δt = time_steps[key]
    sim = Simulation(model; Δt, stop_time)

    # Progress callback
    θ′ = PotentialTemperature(model) - θᵇᵍ_field
    name = case_names[key]

    function progress(sim)
        w = sim.model.velocities.w
        @info @sprintf("%s - Iter: %4d, t: %s, max(θ′): %.4e, max|w|: %.4f",
                       name, iteration(sim), prettytime(sim), maximum(θ′), maximum(abs, w))
        return nothing
    end

    callback_interval = key == :compressible ? IterationInterval(500) : IterationInterval(50)
    add_callback!(sim, progress, IterationInterval(50))

    # Output
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

cases = [:anelastic, :boussinesq, :compressible, :explicit_split, :implicit_split]
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

axes_layout = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)]
axes = [Axis(fig[r, c]; title=titles[i],
             ylabel = c == 1 ? "z (km)" : "",
             xlabel = r == 2 ? "x (km)" : "")
        for (i, (r, c)) in enumerate(axes_layout)]

for ax in axes; if ax.xlabel[] == ""; hidexdecorations!(ax, grid=false); end; end
for ax in axes; if ax.ylabel[] == ""; hideydecorations!(ax, grid=false); end; end

hm = nothing
for (i, k) in enumerate(cases)
    hm = contourf!(axes[i], x_km, z_km, θ′_final[k]; colormap=:balance, levels)
end

Colorbar(fig[1:2, 4], hm; label="θ′ (K)")
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

hm_anim = nothing
for (i, k) in enumerate(cases)
    data = @lift interior(θ′ts[k][$n], :, 1, :)
    hm_anim = contourf!(anim_axes[i], x_km, z_km, data;
                         colormap=:balance, levels, extendhigh=:auto, extendlow=:auto)
end

Colorbar(fig_anim[1:2, 4], hm_anim; label="θ′ (K)")
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
linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]

fig_cross = Figure(size=(900, 400))
ax = Axis(fig_cross[1, 1], xlabel="x (km)", ylabel="θ′ (K)",
          title="θ′ at z = $(round(z_km[z_mid], digits=1)) km, t = $(prettytime(times[Nt]))")

for (i, k) in enumerate(cases)
    lines!(ax, x_km, θ′_final[k][:, z_mid];
           label=titles[i], linewidth=2, linestyle=linestyles[i])
end

axislegend(ax, position=:rt)
save("inertia_gravity_wave_cross_section.png", fig_cross)

fig_cross
