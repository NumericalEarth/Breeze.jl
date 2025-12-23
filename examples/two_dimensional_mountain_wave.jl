# # Schär mountain wave
#
# This example simulates the classic Schär mountain wave test case, which evaluates the model's
# ability to capture terrain-induced gravity waves in a stably stratified atmosphere. The test
# case features a bell-shaped mountain with superimposed small-scale corrugations, generating
# both propagating and evanescent wave modes.
#
# ## References
#
# - [Schar2002](@cite) Schär, C., Leuenberger, D., Fuhrer, O., Lüthi, D., & Girard, C. (2002).
#   "A new terrain-following vertical coordinate formulation for atmospheric prediction models."
#   Monthly Weather Review, 130(10), 2459–2480.
# - [KlempEtAl2015](@cite) Klemp, J. B., Skamarock, W. C., & Park, S.-H. (2015).
#   "Idealized global nonhydrostatic atmospheric test cases on a reduced-radius sphere."
#   Journal of Advances in Modeling Earth Systems, 7(3), 1155–1177.
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
# The terrain follows the Schär mountain profile (Equation 46 in [Schar2002](@cite)):
#
# ```math
# h(x) = h_0 \exp\left(-\frac{x^2}{a^2}\right) \cos^2\left(\frac{\pi x}{\lambda}\right)
# ```
#
# where ``h_0 = 250 \, {\rm m}`` is the peak height, ``a = 5 \, {\rm km}`` is the
# Gaussian half-width, and ``\lambda = 4 \, {\rm km}`` is the wavelength of the
# terrain corrugations.
#
# ### Constant stratification base state
#
# The background atmosphere has constant Brunt–Väisälä frequency ``N = 0.01 \, {\rm s}^{-1}``.
# Using a reference temperature ``T_0 = 300 \, {\rm K}``, this gives:
#
# ```math
# N^2 = \frac{g^2}{c_p^d T_0}
# ```
#
# The density scale height parameter is:
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
#     This validation case requires high resolution to properly resolve the low-elevation
#     terrain corrugations with the immersed boundary method. Additionally, open lateral
#     boundary conditions have not been implemented; periodic boundaries are used instead,
#     which is not ideal for this test case.

using Breeze
using Oceananigans.Units
using Printf
using CairoMakie

using Oceananigans.Grids: znode, xnodes
using Oceananigans: Face, Center
using Breeze.Thermodynamics: dry_air_gas_constant
using CUDA

# ## Thermodynamic parameters
#
# We define the base state with surface pressure ``p_0 = 1000 \, {\rm hPa}``
# and reference temperature ``T_0 = 300 \, {\rm K}``:

constants = ThermodynamicConstants()
g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity
Rᵈ = dry_air_gas_constant(constants)

p₀ = 100000                 # Pa - surface pressure
T₀ = 300                    # K - reference temperature
θ₀ = T₀                     # K - reference potential temperature
U  = 20                     # m s⁻¹ - uniform background wind

# Derived atmospheric parameters:

N² = g^2 / (cᵖᵈ * T₀)       # s⁻² - Brunt–Väisälä frequency squared
N  = sqrt(N²)               # s⁻¹ - Brunt–Väisälä frequency
β  = g / (Rᵈ * T₀)          # m⁻¹ - density scale parameter

# ## Schär mountain parameters
#
# The mountain profile parameters following [Schar2002](@cite):

h₀ = 250                    # m - peak mountain height (use 25 m for strict linearity)
a  = 5000                   # m - Gaussian half-width parameter
λ  = 4000                   # m - terrain corrugation wavelength
K  = 2π / λ                 # rad m⁻¹ - terrain wavenumber

# ## Grid configuration
#
# The domain spans 200 km horizontally (±100 km) and 20 km vertically. We use a non-uniform
# vertical grid with exponential refinement near the surface to resolve the terrain,
# transitioning to uniform 500 m spacing above 1 km altitude.

Nx, Nz = 200, 100
L, H = 100kilometers, 20kilometers

# Vertical grid stretching parameters:

z_transition = 1000         # m - transition height to uniform spacing
dz_top = 500                # m - constant spacing above transition

# Calculate grid distribution:

Nz_top = ceil(Int, (H - z_transition) / dz_top)     # cells in uniform region
Nz_bottom = Nz - Nz_top                             # cells in stretched region

# Construct hybrid vertical grid:

z_stretched = ExponentialDiscretization(Nz_bottom, 0, z_transition, scale = z_transition / 8, bias=:left)
z_uniform = range(z_transition + dz_top, H; length=Nz_top)
z_faces = vcat(z_stretched.faces, collect(z_uniform))
nothing #hide

# Create the underlying rectilinear grid:

underlying_grid = RectilinearGrid(GPU(),
                                  size = (Nx, Nz),
                                  halo = (4, 4),
                                  x = (-L, L),
                                  z = z_faces,
                                  topology = (Periodic, Flat, Bounded))

# ## Mountain profile and immersed boundary
#
# Define the Schär mountain profile and create an immersed boundary grid using the
# partial cell method for terrain representation:

hill(x) = h₀ * exp(-(x / a)^2) * cos(π * x / λ)^2
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(hill))

# ## Plot: Mountain profile and vertical grid
#
# Visualize the terrain comparing the analytical profile with the model's discretized
# representation:

# Analytical profile on a high-resolution grid:
analytical_grid = RectilinearGrid(CPU(), size=500, x=(-30e3, 30e3), topology=(Periodic, Flat, Flat))
h_analytical = Field{Center, Nothing, Nothing}(analytical_grid)
set!(h_analytical, hill)

# Discretized profile as represented in the model:
h_model = grid.immersed_boundary.bottom_height

fig_terrain = Figure(size=(900, 400))
ax_terrain = Axis(fig_terrain[1, 1],
                  xlabel = "x (m)",
                  ylabel = "Height (m)",
                  title = "Schär Mountain Profile")
lines!(ax_terrain, h_analytical, linewidth = 1, color = :black, 
       label = "Analytical")
lines!(ax_terrain, h_model, linewidth = 2, color = :brown, linestyle = :dash,
       label = "Model")

# band! requires arrays, not Fields
x_ana = xnodes(h_analytical)
h_ana = interior(h_analytical, :, 1, 1)
band!(ax_terrain, x_ana, zeros(length(x_ana)), h_ana, color = (:brown, 0.2))
xlims!(ax_terrain, -30e3, 30e3)
axislegend(ax_terrain, position = :rt)

save("mountain_wave_terrain.png", fig_terrain)
fig_terrain

# ## Rayleigh damping layer
#
# A sponge layer at the top of the domain prevents spurious wave reflections from
# the upper boundary. The damping follows a Gaussian profile centered at the domain top:
#
# ```math
# \mathcal{S}(\rho w) = -\omega \, \exp\left(-\frac{(z - z_0)^2}{2 \Delta z^2}\right) \rho w
# ```
#
# where ``\omega = 1/60 \, {\rm s}^{-1}`` is the relaxation rate and ``\Delta z = L_z/2``.

@inline gaussian_mask(z, p) = exp(-(z - p.z0)^2 / 2p.dz^2)

@inline function sponge(i, j, k, grid, clock, params, ρc, lz)
    z = znode(k, grid, lz)
    ω = params.ω
    m = gaussian_mask(z, params)
    return @inbounds -ω * m * ρc[i, j, k]
end

@inline ρw_sponge(i, j, k, grid, clock, model_fields, params) =
    sponge(i, j, k, grid, clock, params, model_fields.ρw, Face())

sponge_params = (z0 = grid.Lz, dz = grid.Lz / 2, ω = 1/60)
ρw_forcing = Forcing(ρw_sponge, discrete_form=true, parameters=sponge_params)

# ## Model initialization
#
# Create the atmosphere model with the anelastic formulation, 5th-order WENO advection,
# and the Rayleigh damping layer:
reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
formulation = AnelasticFormulation(reference_state)

advection = WENO(order=5)
model = AtmosphereModel(grid; formulation, advection, forcing=(; ρw=ρw_forcing))

# ## Initial conditions
#
# Initialize with the constant-``N`` stratification and uniform background wind.
# The potential temperature profile that maintains constant Brunt–Väisälä frequency is:
#
# ```math
# \theta(z) = \theta_0 \exp\left(\frac{N^2 z}{g}\right)
# ```

θᵢ(x, z) = θ₀ * exp(N² * z / g)
set!(model, θ = θᵢ, u = U)

# ## Simulation
#
# Run for 2 hours with a fixed time step. A small time step is required for numerical stability with this test case:

Δt = 2.0            # s - time step (reduced for stability)
stop_time = 2hours  # total simulation time

simulation = Simulation(model; Δt, stop_time, align_time_step=false)

# Progress callback to monitor simulation health:

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, time: %s, wall time: %s, max|w|: %.3e m s⁻¹",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

add_callback!(simulation, progress, name=:progress, IterationInterval(200))

# ## Output
#
# Save velocity fields for post-processing and validation:

filename = "mountain_waves"
simulation.output_writers[:fields] = JLD2Writer(model, model.velocities;
                                                filename,
                                                schedule = TimeInterval(100),
                                                overwrite_existing = true)

run!(simulation)

# ## Analytical solution
#
# The linear analytical solution for mountain waves provides a validation benchmark.
# Following Appendix A of [KlempEtAl2015](@cite), the vertical velocity field is computed
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

hhat(k) = sqrt(π) * h₀ * a / 4 * (exp(-a^2 * (K + k)^2 / 4) +
                                   exp(-a^2 * (K - k)^2 / 4) +
                                   2exp(-a^2 * k^2 / 4))

# ### Dispersion relation
#
# Vertical wavenumber squared (Equation A5) and critical wavenumber (Equation A11):

m²(k) = (N² / U^2 - β^2 / 4) - k^2
k★ = sqrt(N² / U^2 - β^2 / 4)

# ### Linear vertical velocity
#
# Compute the analytical vertical velocity ``w(x, z)`` from Equation A10:
#
# ```math
# w(x, z) = -\frac{U}{\pi} e^{\beta z/2} \left[
#     \int_0^{k^*} k \hat{h}(k) \sin(m z + k x) \, \mathrm{d}k +
#     \int_{k^*}^{\infty} k \hat{h}(k) e^{-|m| z} \sin(k x) \, \mathrm{d}k
# \right]
# ```
#
# where the first integral represents propagating waves and the second represents
# evanescent waves.

"""
    w_linear(x, z; nk=100)

Compute the 2-D linear vertical velocity `w(x,z)` from the analytical solution
(Appendix A, Equation A10 of Klemp et al., 2015).
"""
function w_linear(x, z; nk=100)
    k = range(0, 10k★; length=nk)
    m2 = m².(k)
    ĥ = hhat.(k)

    m_abs = sqrt.(abs.(m2))
    integrand = @. k * ĥ * ifelse(m2 ≥ 0,
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
# We compare the simulated vertical velocity field at 2 hours with the linear
# analytical solution. While the simulation reproduces the general gravity wave
# pattern, noticeable discrepancies in wavenumber appear. The immersed boundary
# method struggles to resolve the low-amplitude, fine-scale terrain corrugations
# at this resolution.

# Create comparison figure with simulated and analytical vertical velocity:

fig = Figure(size=(900, 800), fontsize=14)
nothing #hide

# Plot simulated field:

w_simulated = model.velocities.w

ax1 = Axis(fig[1, 1],
           xlabel = "x (m)",
           ylabel = "z (m)",
           title = "Simulated w at t = 2 hours")
hm1 = heatmap!(ax1, w_simulated, colormap = :balance, colorrange = (-1, 1))
ax1.limits = ((-30000, 30000), (0, 10000))

# Compute analytical solution on the same grid as the simulation:

w_analytical = Field{Center, Nothing, Face}(grid)
set!(w_analytical, w_linear)

ax2 = Axis(fig[2, 1],
           xlabel = "x (m)",
           ylabel = "z (m)",
           title = "Linear Analytical w")
hm2 = heatmap!(ax2, w_analytical, colormap = :balance, colorrange = (-1, 1))
ax2.limits = ((-30000, 30000), (0, 10000))

# Shared colorbar:

Colorbar(fig[1:2, 2], hm1, label = "w (m s⁻¹)")

save("mountain_wave_w_comparison.png", fig)
fig
