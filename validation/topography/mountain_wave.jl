using Breeze
using Oceananigans.Units
using CUDA

# Schär mountain wave test case
# References: 
# Klemp et al. (2015) "Idealized global nonhydrostatic atmospheric test cases on a reduced-radius sphere"
# Schär et al. (2002) "A New Terrain-Following Vertical Coordinate Formulation for Atmospheric Prediction Models"

# This validation case is currently not performing well (speed and accuracy) for the following reasons:
# 1) It requires very high resolution to properly resolve the short mountain due to the immersed boundary method.
# 2) Open lateral boundary conditions have not been implemented yet; periodic boundaries are used instead, which is not ideal for this test case.


# Problem parameters 
thermo = ThermodynamicConstants()
g = thermo.gravitational_acceleration
cᵖᵈ = thermo.dry_air.heat_capacity
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(thermo)

# Isothermal base state and mean wind
T₀ = 300                    # K
θ₀ = T₀                     # K - reference potential temperature
U  = 20                     # m s^-1 (mean wind)
N² = g^2 / (cᵖᵈ * T₀)       # Brunt–Väisälä frequency squared
β  = g / (Rᵈ * T₀)          # density scale parameter
N  = sqrt(N²)

# Schär mountain parameters
h₀ = 250                    # m (use 25 m for strict linearity; 250 m for full case)
a  = 5000                   # m (Gaussian half-width parameter)
λ  = 4000                   # m (terrain envelope wavelength)
K  = 2*π/λ                  # rad m^-1

#  grid configuration
Nx, Nz = 256, 256
L, H = 100kilometers, 29kilometers

# Vertical grid stretching parameters, constant spacing (500 m) above 3 km
z_transition = 3000  # m - transition height
dz_top = 250  # m - constant spacing above 3 km

# Calculate number of cells needed above transition
Nz_top = ceil(Int, (H - z_transition) / dz_top)

# Remaining cells for stretched region
Nz_bottom = Nz - Nz_top

# Exponential refinement up to z_transition
z_stretched = ExponentialDiscretization(Nz_bottom, 0, z_transition, scale = z_transition / 8, bias=:left)
z_uniform = range(z_transition+dz_top, H; length=Nz_top)
z_faces = vcat(z_stretched.faces, collect(z_uniform))

# Set up the simulation doamin
underlying_grid = RectilinearGrid(GPU(), size = (Nx, Nz), halo = (4, 4),
                                  x = (-L, L), z = z_faces,
                                  topology = (Periodic, Flat, Bounded))

# Define the mountain profile and immersed boundary grid
hill(x) = h₀ * exp(-(x / a)^2) * cos(π * x / λ)^2
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(hill))

# Rayleigh damping layer at the top of the domain
# damping_rate = 1/60 # relax fields on a 60 second time-scale
# top_mask = GaussianMask{:z}(center=grid.Lz, width=grid.Lz/2)
# sponge = Relaxation(rate=damping_rate, mask=top_mask)

@inline gaussian_mask(z, p) = exp(-(z - p.z0)^2 / 2p.dz^2)

@inline function sponge(i, j, k, grid, clock, params, ρc, lz)
    z = znode(k, grid, lz)
    ω = params.ω
    m = gaussian_mask(z, params)
    return @inbounds - ω * m * ρc[i, j, k]
end

@inline ρw_sponge(i, j, k, grid, clock, model_fields, params) =
    sponge(i, j, k, grid, clock, params, model_fields.ρw, Face())

params = (z0=grid.Lz, dz=grid.Lz/2, ω=1/60)
ρw_forcing = Forcing(ρw_sponge, discrete_form=true, parameters=params)

# Atmosphere model setup
model = AtmosphereModel(grid, advection = WENO(), forcing=(; ρw=ρw_forcing))

# Initial conditions and initialization
θᵢ(x, z) = θ₀ * exp(N² * z / g) # background stratification for isothermal atmosphere
set!(model, θ=θᵢ, u=U)

# Time-stepping and simulation setup
Δt = 6.0 # seconds
stop_iteration = 1800
simulation = Simulation(model; Δt, stop_iteration, align_time_step=false)

using Printf

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

add_callback!(simulation, progress, name=:progress, IterationInterval(200))

filename = "mountain_waves"
simulation.output_writers[:fields] = JLD2Writer(model, model.velocities; filename,
                                                schedule = TimeInterval(100),
                                                overwrite_existing = true)

run!(simulation)


# Analytical solution for linear mountain wave problem (Appendix A of Klemp et al, 2015)
# Analytic Fourier transform (A8)
hhat(k) = @. sqrt(π) * h₀ * a/4 * (exp(-a^2 * (K + k)^2 / 4) + exp(-a^2 * (K - k)^2 / 4) + 2exp(-a^2 * k^2 / 4))

# m² and k* (A5, A11)
m²(k) = (N²/U^2 - β^2/4) - k^2
kstar = sqrt(N²/U^2 - β^2/4)

# Integral using trapezoidal rule
trapz(x, f) = sum((@view(f[1:end-1]) .+ @view(f[2:end])) .* diff(x)) / 2

"""
    w_linear(x, z; nk=10000)

Compute the 2-D linear vertical velocity w(x,z) from Appendix A, eq. (A10).

Arguments:
- `x`: horizontal position in meters
- `z`: vertical position in meters
- `nk`: controls resolution of the wavenumber space (default: 10000)
"""
function w_linear(x, z; nk=10000)
    # Discretize wavenumber space
    k = range(0, 100kstar; length=nk)
    m2 = m².(k)
    ĥ = hhat.(k)

    # Oscillatory part: 0 ≤ k ≤ k*
    idx = findall(ki -> ki ≤ kstar, k)
    Iosc = 0.0
    if !isempty(idx)
        ko = k[idx]
        mo = sqrt.(clamp.(m2[idx], 0, Inf))
        ĥo = ĥ[idx]
        integrand = @. ko * ĥo * sin(mo*z + ko*x)
        Iosc = trapz(ko, integrand)
    end

    # Evanescent part: k* ≤ k < ∞
    idx = findall(ki -> ki ≥ kstar, k)
    Iev = 0.0
    if !isempty(idx)
        ke = k[idx]
        me = sqrt.(clamp.(-m2[idx], 0, Inf))  # |m| when m^2<0
        ĥe = ĥ[idx]
        integrand = @. ke * ĥe * exp(-me * z) * sin(ke*x)
        Iev = trapz(ke, integrand)
    end

    # Assemble (A10)
    return - U / π * exp(β*z/2) * (Iosc + Iev)
end

# Visualization
using CairoMakie

fig = Figure()
gb = fig[1, 1]

xs = LinRange(-L, L, Nx)
zs = z_faces # LinRange(0, H, Nz+1)

ax1, hm = heatmap(gb[1, 1], xs, zs, Array(interior(model.velocities.w, :, 1, :)), colormap = :bwr, colorrange = (-1, 1))
ax1.xlabel = "x [m]"
ax1.ylabel = "z [m]"
ax1.title = "Simulated w at 3-hr"
ax1.limits = ((-30000, 30000), (0, 10000))

xs = range(-30e3, 30e3; length=60) # x ∈ [-30, 30] km
zs = range(0, 10e3; length=40)     # z ∈ [0, 10] km
w_analytical = [w_linear(x, z) for z in zs, x in xs]

ax2, hm2 = heatmap(gb[2, 1], xs, zs, w_analytical', colormap = :bwr, colorrange = (-1, 1))
ax2.xlabel = "x [m]"
ax2.ylabel = "z [m]"
ax2.title = "Linear Analytical w"
ax2.limits = ((-30000, 30000), (0, 10000))

cb = Colorbar(gb[1:2, 2], hm, label = "w [m s⁻¹]")

#fig
save("mountain_wave_w_comparison.png", fig)
