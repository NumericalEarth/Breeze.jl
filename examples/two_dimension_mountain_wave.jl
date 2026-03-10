# # Schär mountain wave with terrain-following coordinates
#
# This example simulates the classic Schär mountain wave test case using the fully
# compressible equations on a terrain-following grid. The test case features a bell-shaped
# mountain with superimposed small-scale corrugations, generating both propagating and
# evanescent wave modes.
#
# The Schär mountain wave ([Schär et al. (2002)](@cite Schar2002)) is a stringent benchmark
# for terrain-following coordinates because the fine-scale terrain corrugations create steep
# coordinate-surface slopes that challenge the horizontal pressure gradient computation.
# Instead of using an `ImmersedBoundaryGrid` with partial cells, we deform the
# computational grid itself using [`follow_terrain!`](@ref), which applies the
# [Gal-Chen and Somerville (1975)](@cite GalChen1975) coordinate transformation.
# The terrain-following approach exactly captures the mountain profile at any resolution,
# whereas the immersed boundary method requires high resolution to resolve the low-amplitude
# corrugations.
#
# ## References
#
# ```@bibliography
# Pages = ["two_dimension_mountain_wave.md"]
# Canonical = false
# ```
#
# ## Physical setup
#
# The simulation initializes an atmosphere with constant Brunt–Väisälä frequency ``N`` and
# uniform background wind ``U``. A bell-shaped mountain with superimposed oscillations
# triggers both propagating gravity waves (for wavenumbers below the critical value ``k^*``)
# and evanescent waves (above ``k^*``).
#
# ### Mountain profile
#
# The terrain follows the Schär mountain profile (Equation 46 by [Schar2002](@citet)):
#
# ```math
# h(x) = h_0 \exp\left(-\frac{x^2}{a^2}\right) \cos^2\left(\frac{\pi x}{\lambda}\right)
# ```
#
# where ``h_0`` is the peak height, ``a`` is the Gaussian half-width, and ``\lambda`` is the
# wavelength of the terrain corrugations.
#
# ### Constant stratification base state
#
# For a reference temperature ``T_0``, the background atmosphere corresponds to a constant
# Brunt–Väisälä frequency ``N``:
#
# ```math
# N^2 = \frac{g^2}{c_p^d T_0}
# ```
#
# The density-scale height parameter is:
#
# ```math
# \beta = \frac{g}{R^d T_0}
# ```
#
# The potential temperature profile that maintains constant ``N`` is:
#
# ```math
# \theta(z) = \theta_0 \exp\left(\frac{N^2 z}{g}\right)
# ```
#
# ### Linear wave theory
#
# For the linearized mountain wave problem, vertical wavenumber ``m`` satisfies the
# dispersion relation (Appendix A of [KlempEtAl2015](@citet)):
#
# ```math
# m^2 = \frac{N^2}{U^2} - \frac{\beta^2}{4} - k^2
# ```
#
# Waves propagate vertically when ``m^2 > 0`` (i.e., ``k < k^*``), and decay exponentially
# when ``m^2 < 0`` (i.e., ``k > k^*``), where the critical wavenumber is:
#
# ```math
# k^* = \sqrt{\frac{N^2}{U^2} - \frac{\beta^2}{4}}
# ```
#
# !!! note "Current limitations"
#     Open lateral boundary conditions have not been implemented; periodic boundaries
#     are used instead, which is not ideal for this test case.

using Breeze
using Breeze.TerrainFollowingDiscretization: SlopeInsideInterpolation
using Oceananigans.Grids: MutableVerticalDiscretization, xnodes, znode
using Oceananigans.Operators: Δzᶜᶜᶠ
using Oceananigans.Units
using Oceananigans: Face, Center
using Breeze.Thermodynamics: dry_air_gas_constant, hydrostatic_pressure
using Printf
using CairoMakie
using CUDA

# ## Thermodynamic parameters
#
# We define the base state with surface pressure ``p_0 = 1000 \, {\rm hPa}``
# and reference temperature ``T_0 = 300 \, {\rm K}``. We also set the background wind
# at ``U = 20 \, {\rm m/s}``:

constants = ThermodynamicConstants(Float64)
g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity
Rᵈ = dry_air_gas_constant(constants)

p₀ = 100000                 # Pa - surface pressure
T₀ = 300                    # K - reference temperature
θ₀ = T₀                     # K - reference potential temperature
U  = 20                     # m s⁻¹ - uniform background wind

# Derived atmospheric parameters for an isothermal base state at ``T_0``.
# Note: the standard [Schar2002](@citet) parameters use ``N = 0.01 \, {\rm s}^{-1}``
# (set `N² = 1e-4` to match that paper exactly).

N² = g^2 / (cᵖᵈ * T₀)       # s⁻² - Brunt–Väisälä frequency squared (≈ 3.2e-4)
N  = sqrt(N²)               # s⁻¹ - Brunt–Väisälä frequency
β  = g / (Rᵈ * T₀)          # m⁻¹ - density scale parameter

# ## Schär mountain parameters
#
# The mountain profile with the parameters used by [Schar2002](@citet) is:

h₀ = 250                    # m - peak mountain height (use 25 m for strict linearity)
a  = 5000                   # m - Gaussian half-width parameter
λ  = 4000                   # m - terrain corrugation wavelength
K  = 2π / λ                 # rad m⁻¹ - terrain wavenumber

hill(x, y) = h₀ * exp(-(x / a)^2) * cos(π * x / λ)^2

# ## Grid setup
#
# The domain spans 200 km horizontally (±100 km) and 20 km vertically. We use a
# `MutableVerticalDiscretization` with uniform spacing in the computational coordinate;
# the terrain-following transformation deforms these surfaces to follow the mountain,
# so no vertical grid stretching near the surface is needed.
#
# The horizontal resolution must adequately resolve the terrain corrugation wavelength
# ``\lambda``. With the 2nd-order terrain pressure gradient stencil, at least 8 grid
# points per corrugation wavelength are needed (``\Delta x \leq \lambda / 8``).
# With ``\lambda = 4000 \, {\rm m}``, this requires ``\Delta x \leq 500 \, {\rm m}``.

arch = CUDA.functional() ? GPU() : CPU()

Nx, Nz = 201, 50
L = 100kilometers
const H = 20kilometers

z_faces = MutableVerticalDiscretization(collect(range(0, H, length=Nz+1)))
grid = RectilinearGrid(arch, size = (Nx, Nz),
                       x = (-L/2, L/2), z = z_faces,
                       topology = (Periodic, Flat, Bounded))

# ## Terrain
#
# Apply the Schär mountain profile to the grid via [`follow_terrain!`](@ref). This sets
# the column scaling factors ``\sigma`` and surface displacement ``\eta`` on the grid's
# `MutableVerticalDiscretization` and returns a [`TerrainMetrics`](@ref) object with
# precomputed terrain slopes.

metrics = follow_terrain!(grid, hill; pressure_gradient_stencil = SlopeInsideInterpolation())

# ## Plot: Terrain-following coordinate surfaces
#
# Visualize how the computational grid deforms to follow the corrugated terrain.
# Unlike an immersed boundary representation, the terrain-following coordinate exactly
# captures the mountain profile regardless of vertical resolution.

x_grid = xnodes(grid, Center())

fig = Figure(size=(900, 400))
ax = Axis(fig[1, 1],
          xlabel = "x (m)",
          ylabel = "Height (m)",
          title = "Terrain-following grid near the Schär mountain")

## Plot coordinate surfaces at selected vertical levels
CUDA.@allowscalar for k in [1, 3, 5, 8, 12, 17, 25]
    z_surface = [znode(i, 1, k, grid, Center(), Center(), Face()) for i in 1:Nx]
    lines!(ax, x_grid, z_surface, color = :gray, linewidth = 0.5)
end

## The bottom coordinate surface (terrain)
z_bottom = CUDA.@allowscalar [znode(i, 1, 1, grid, Center(), Center(), Face()) for i in 1:Nx]
lines!(ax, x_grid, z_bottom, color = :brown, linewidth = 2, label = "Terrain surface")
band!(ax, x_grid, zero(z_bottom), z_bottom, color = (:brown, 0.2))

## Analytical terrain for comparison
x_fine = range(-L/6, L/6, length=2000)
lines!(ax, collect(x_fine), [hill(x, 0) for x in x_fine],
       color = :black, linestyle = :dash, linewidth = 1, label = "Analytical h(x)")

axislegend(ax, position = :rt)
xlims!(ax, -L/6, L/6)
ylims!(ax, -100, 3500)
fig

# ## Potential temperature profile
#
# The potential temperature profile that maintains constant Brunt–Väisälä frequency is:
#
# ```math
# \theta(z) = \theta_0 \exp\left(\frac{N^2 z}{g}\right)
# ```

θ_of_z(z) = θ₀ * exp(N² * z / g)

# ## Rayleigh damping layer
#
# A sponge layer near the domain top absorbs upward-propagating waves and prevents
# spurious reflections from the rigid lid. The Gaussian mask is concentrated in the
# upper quarter of the domain and decays to effectively zero below mid-level.

const sponge_width = H / 4
sponge_mask(x, z) = exp(-(z - H)^2 / sponge_width^2)
ρw_sponge = Relaxation(rate=1/10, mask=sponge_mask)

# ## Model construction
#
# Build a compressible model with explicit time-stepping, 5th-order WENO advection, and
# terrain corrections. Passing `terrain_metrics` to [`CompressibleDynamics`](@ref)
# activates the terrain-following physics: contravariant vertical velocity, corrected
# pressure gradient, and terrain-aware divergence. The `reference_potential_temperature`
# enables a perturbation pressure approach for the horizontal pressure gradient that
# reduces the truncation error inherent in terrain-following coordinates
# ([Klemp (2011)](@cite Klemp2011)).

pˢᵗ = 1e5                   # Pa - standard pressure
κ = Rᵈ / cᵖᵈ

dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                terrain_metrics = metrics,
                                surface_pressure = p₀,
                                reference_potential_temperature = θ_of_z)

model = AtmosphereModel(grid; dynamics, advection=WENO(order=5), forcing=(; ρw=ρw_sponge))

# ## Initial conditions
#
# We initialize the atmosphere in discrete hydrostatic balance using Exner function
# integration. This is essential for compressible models on terrain-following grids:
# the equation of state alone does not produce a pressure field in discrete hydrostatic
# balance. Because we passed `reference_potential_temperature` to `CompressibleDynamics`,
# the model has already computed this discrete reference state for us in `terrain_reference_density`!

ρ_ref = model.dynamics.terrain_reference_density
ρ_field = model.dynamics.density
ρθ_field = model.formulation.potential_temperature_density

CUDA.@allowscalar for i in 1:Nx
    for k in 1:Nz
        z_phys = znode(i, 1, k, grid, Center(), Center(), Center())
        θₖ = θ_of_z(z_phys)

        ρₖ = ρ_ref[i, 1, k]

        ρ_field[i, 1, k] = ρₖ
        ρθ_field[i, 1, k] = ρₖ * θₖ
    end
end

set!(model, u=U)

# ## Simulation
#
# Acoustic waves require a CFL condition based on the sound speed, which gives a much
# smaller time step than the anelastic equations. We use the potential temperature at
# the domain top for a conservative estimate, since θ (and thus the sound speed)
# increases with height.

γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
θ_top = θ_of_z(H)
ℂᵃᶜ = sqrt(γ * Rᵈ * θ_top)

Δx = L / Nx
Δz = H / Nz
Δt = 0.4 * min(Δx, Δz) / (ℂᵃᶜ + U)

stop_time = 2hours

simulation = Simulation(model; Δt, stop_time)

# Progress callback to monitor simulation health:

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    u, v, w = sim.model.velocities
    msg = @sprintf("Iter: %d, time: %s, wall time: %s, max|u|: %.2f, max|w|: %.4f m/s",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, u), maximum(abs, w))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

add_callback!(simulation, progress, name=:progress, IterationInterval(500))

# ## Output
#
# Save vertical velocity and contravariant vertical velocity for post-processing:

w = model.velocities.w
Ω̃ = model.dynamics.Ω̃

filename = "mountain_waves"
outputs = (; w, Ω̃)
simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                              filename,
                                              schedule = TimeInterval(2minutes),
                                              overwrite_existing = true)

run!(simulation)

# ## Analytical solution
#
# The linear analytical solution for mountain waves provides a validation benchmark.
# Following Appendix A of [KlempEtAl2015](@citet), the vertical velocity field is computed
# via Fourier integration over wavenumber space.
#
# ### Fourier transform of terrain
#
# The Fourier transform of the Schär mountain profile (Equation A8):
#
# ```math
# \hat{h}(k) = \frac{\sqrt{\pi} h_0 a}{4} \left[
#     e^{-a^2(K+k)^2/4} + e^{-a^2(K-k)^2/4} + 2e^{-a^2 k^2/4}
# \right]
# ```

ĥ(k) = sqrt(π) * h₀ * a / 4 * (exp(-a^2 * (K + k)^2 / 4) +
                                exp(-a^2 * (K - k)^2 / 4) +
                                2exp(-a^2 * k^2 / 4))

# ### Dispersion relation
#
# Vertical wavenumber squared (Equation A5) and critical wavenumber (Equation A11)
# by [KlempEtAl2015](@citet):

m²(k) = (N² / U^2 - β^2 / 4) - k^2
k★ = sqrt(N² / U^2 - β^2 / 4)

# ### Linear vertical velocity
#
# Compute the analytical vertical velocity ``w(x, z)`` via Equation A10
# by [KlempEtAl2015](@citet):
#
# ```math
# w(x, z) = -\frac{U}{\pi} e^{\beta z/2} \left[
#     \int_0^{k^*} k \hat{h}(k) \sin(m z + k x) \, \mathrm{d}k +
#     \int_{k^*}^{\infty} k \hat{h}(k) e^{-|m| z} \sin(k x) \, \mathrm{d}k
# \right]
# ```
#
# Above, the first integral represents propagating waves and the second represents
# evanescent waves.

"""
    w_linear(x, z; nk=100)

Compute the 2-D linear vertical velocity `w(x,z)` from the analytical solution
(Appendix A, Equation A10 of Klemp et al., 2015).
"""
function w_linear(x, z; nk=100)
    k = range(0, 10k★; length=nk)
    m_abs = @. sqrt(abs(m²(k)))
    integrand = @. k * ĥ(k) * ifelse(m²(k) ≥ 0,
                                   sin(m_abs * z + k * x),
                                   exp(-m_abs * z) * sin(k * x))

    ## Numerical integration using trapezoidal rule:
    Δk = step(k)
    integral = Δk * (sum(integrand) - (first(integrand) + last(integrand)) / 2)
    return -(U / π) * exp(β * z / 2) * integral
end
nothing #hide

# ## Results: Comparison with analytical solution
#
# We compare the simulated vertical velocity field at 10 minutes with the linear
# analytical solution. The terrain-following grid exactly represents the corrugated
# mountain profile, avoiding the staircase artifacts of the immersed boundary method.
# The heatmaps display fields in physical ``(x, z)`` coordinates, with the deformed
# grid automatically mapped to the true terrain geometry.
#
# First, we compute analytical vertical velocity ``w`` on the same grid as the simulation.

w_analytical = Field{Center, Nothing, Face}(grid)
set!(w_analytical, w_linear)

fig = Figure(size=(900, 800), fontsize=14)

ax1 = Axis(fig[1, 1], ylabel = "z (m)", title = "Simulated w at t = 2 hours")
ax2 = Axis(fig[2, 1], xlabel = "x (m)", ylabel = "z (m)", title = "Linear Analytical w")
hidexdecorations!(ax1, grid = false)

hm1 = heatmap!(ax1, w, colormap = :balance, colorrange = (-1, 1))
hm2 = heatmap!(ax2, w_analytical, colormap = :balance, colorrange = (-1, 1))

Colorbar(fig[1:2, 2], hm1, label = "w (m s⁻¹)")

for ax in (ax1, ax2)
    ax.limits = ((-30e3, 30e3), (0, 10e3))
end

fig
