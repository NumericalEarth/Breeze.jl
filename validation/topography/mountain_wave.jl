using Breeze
using Oceananigans.Units
using Oceananigans.Forcings
using LinearAlgebra

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
N² = g^2/(cᵖᵈ*T₀)           # Brunt–Väisälä frequency squared
N  = sqrt(N²)
β  = g/(Rᵈ*T₀)              # density scale parameter


# Schär mountain parameters
h₀ = 250                    # m   (use 25 m for strict linearity; 250 m for full case)
a  = 5000                   # m   (Gaussian half-width parameter)
λ  = 4000                   # m   (terrain envelope wavelength)
K  = 2*π/λ                  # rad m^-1

#  grid configuration
Nx, Nz = 200, 800
L, H = 100kilometers, 20kilometers

underlying_grid = RectilinearGrid(CPU(), size = (Nx, Nz), halo = (4, 4),
                                  x = (-L, L), z = (0, H),
                                  topology = (Periodic, Flat, Bounded))

# Define the mountain profile and immersed boundary grid
hill(x) = h₀ * exp(-(x / a)^2) * cos(π * x / λ)^2
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(hill))


# Rayleigh damping layer at the top of the domain
damping_rate = 1/10 # relax fields on a 10 second time-scale
top_mask = GaussianMask{:z}(center=grid.Lz, width=grid.Lz/2)
sponge = Relaxation(rate=damping_rate, mask=top_mask)

# Atmosphere model setup
model = AtmosphereModel(grid, advection = WENO(), forcing=(u=sponge, w=sponge, θ=sponge))

# Initial conditions and initialization
θᵢ(x, z) = θ₀ * exp(N² * z / g) # background stratification for isothermal atmosphere
set!(model, θ=θᵢ, u=U)

# Time-stepping and simulation setup
Δt = 6.0 # seconds
stop_iteration = 600
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
hhat(k) = sqrt(π)*h₀*a/4 * ( exp.(-0.25*a^2*(K .+ k).^2) .+ exp.(-0.25*a^2*(K .- k).^2) .+ 2 * exp.(-0.25*a^2*k.^2) )

# m² and k* (A5, A11)
m²(k) = (N²/U^2 - β^2/4) .- k.^2
kstar = sqrt(N²/U^2 - β^2/4)

# Integral using trapezoidal rule
trapz(x, f) = sum((@view(f[1:end-1]) .+ @view(f[2:end])).*diff(x)) / 2

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
    k = range(0.0, kstar*100; length=nk)
    m2 = m²(k)
    ĥ   = hhat(k)

    # Oscillatory part: 0 ≤ k ≤ k*
    idx = findall(ki -> ki ≤ kstar, k)
    Iosc = 0.0
    if !isempty(idx)
        ko   = k[idx]
        mo   = sqrt.(clamp.(m2[idx], 0, Inf))
        ĥo   = ĥ[idx]
        integrand = ko .* ĥo .* sin.(mo*z .+ ko*x)
        Iosc = trapz(ko, integrand)
    end

    # Evanescent part: k* ≤ k < ∞
    idx = findall(ki -> ki ≥ kstar, k)
    Iev = 0.0
    if !isempty(idx)
        ke   = k[idx]
        me   = sqrt.(clamp.(-m2[idx], 0, Inf))  # |m| when m^2<0
        ĥe   = ĥ[idx]
        integrand = ke .* ĥe .* exp.(-me*z) .* sin.(ke*x)
        Iev = trapz(ke, integrand)
    end

    # Assemble (A10)
    return -(U/π) * exp(β*z/2) * (Iosc + Iev)
end

# Visualization
using CairoMakie
fig = Figure(size = (1600, 1200))
gb = fig[1, 1]

xs = LinRange(-L, L, Nx)
zs = LinRange(0, H, Nz+1)

ax1, hm = heatmap(gb[1,1], xs, zs, interior(model.velocities.w, :,1,:), colormap = :bwr, colorrange = (-1.0, 1.0))
ax1.xlabel = "x [m]"
ax1.ylabel = "z [m]"
ax1.title = "Simulated w [m s⁻¹]"
ax1.limits = ((-30000., 30000.), (0, 10000.))


xs = range(-30e3, 30e3; length=60)   # x ∈ [-30, 60] km
zs = range(0.0, 10e3; length=100)     # z ∈ [0, 12] km
w_analytical   = [w_linear(x, z) for z in zs, x in xs]
ax2, hm2 = heatmap(gb[2,1], xs, zs, w_analytical', colormap = :bwr, colorrange = (-1.0, 1.0))
ax2.xlabel = "x [m]"
ax2.ylabel = "z [m]"
ax2.title = "Linear Analytical w [m s⁻¹]"
ax2.limits = ((-30000., 30000.), (0, 10000.))

cb = Colorbar(gb[1:2, 2], hm, label = "w [m s⁻¹]")

fig
