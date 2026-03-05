# # Differentiable Baroclinic Wave on the Sphere
#
# The Jablonowski–Williamson (2006) baroclinic instability test case is the
# canonical dynamical-core benchmark.  A zonally symmetric midlatitude jet in
# thermal-wind balance is seeded with a small localised perturbation.  Over
# 5–10 days the perturbation amplifies via baroclinic instability, eventually
# producing realistic-looking extratropical cyclones with fronts and wave
# breaking.
#
# This example sets up the test case on a `LatitudeLongitudeGrid` using Breeze's
# compressible dynamics, then demonstrates how **Reactant + Enzyme** can
# differentiate a scalar loss through the full simulation:
#
# ```math
# J = \langle v^2 \rangle \quad \text{over a midlatitude band}
# ```
#
# Since ``v = 0`` initially (the balanced state is zonally symmetric), any
# meridional velocity that develops is purely from the growing instability.
# The sensitivity ``\partial J / \partial \theta_0`` reveals which initial
# temperature perturbations most efficiently amplify baroclinic growth.
#
# Three distinct regimes emerge as integration time ``T`` increases:
#
# | Regime | ``T`` | Behaviour |
# |--------|-------|-----------|
# | Linear | ≲ 5 days | Exponential growth; adjoint is well-behaved |
# | Nonlinear saturation | 7–10 days | Wave breaking; sensitivity develops frontal features |
# | Chaotic | ≳ 12 days | Filamentation; adjoint blows up |
#
# Plotting ``\|\nabla_A J\|`` versus ``T`` shows the transition from
# "AD works perfectly" to "AD gives you garbage" — the central motivating
# figure for understanding when tangent-linear methods break down.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using Printf
using CairoMakie
using CUDA

# ## Jablonowski–Williamson parameters
#
# All values follow the standard intercomparison protocol (JW06).  The
# background atmosphere has a constant tropospheric lapse rate ``\Gamma`` with
# an isothermal stratosphere above ``\eta_t = 0.2``.  The jet peaks at
# ``\eta_0 = 0.252``, approximately 250 hPa (≈ 10.5 km).

a_earth = 6.371229e6   # Earth radius [m]
Ω_earth = 7.29212e-5   # rotation rate [s⁻¹]
g       = 9.80616      # gravitational acceleration [m/s²]

T_sfc   = 288.0        # equatorial surface temperature [K]
Γ_lapse = 0.005        # tropospheric lapse rate [K/m]
p_sfc   = 1e5          # surface pressure [Pa]
pˢᵗ     = 1e5          # standard (reference) pressure [Pa]

u_jet   = 35.0         # maximum jet speed [m/s]
η₀      = 0.252        # jet peak in η coordinates
η_t     = 0.2          # tropopause η level

# Perturbation parameters — a localised Gaussian bump in potential temperature
# at a single longitude, seeding the most unstable baroclinic mode.
A_pert  = 1.0          # perturbation amplitude [K]
λ_c     = 20.0         # perturbation centre longitude [°]
φ_c     = 40.0         # perturbation centre latitude [°]
σ_pert  = 10.0         # perturbation angular width [°]

# ## Grid and model
#
# A coarse grid for fast compilation; increase resolution for physical runs.

Nλ = 32
Nφ = 32
Nz = 12
z_top = 30000.0         # model top [m]

lat_south = -80.0
lat_north =  80.0

grid_kwargs = (size = (Nλ, Nφ, Nz),
               longitude = (0, 360),
               latitude = (lat_south, lat_north),
               z = (0, z_top),
               topology = (Periodic, Bounded, Bounded))

@info "Building grid…"
@time grid = LatitudeLongitudeGrid(ReactantState(); grid_kwargs...)

FT = eltype(grid)

@info "Building model…"
@time model = AtmosphereModel(grid;
    dynamics = CompressibleDynamics(),
    coriolis = HydrostaticSphericalCoriolis(FT))

constants = model.thermodynamic_constants
Rᵈ  = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ

# ## Jablonowski–Williamson initial condition
#
# The balanced state is specified analytically in the pressure coordinate
# ``\eta = p / p_s``.  We map from geometric height ``z`` to ``\eta`` using the
# barometric formula for a constant-lapse-rate troposphere with an isothermal
# stratosphere above.  This mapping is approximate (ignores the latitude-
# dependent pressure field) but sufficiently accurate for initialization.

z_trop  = T_sfc / Γ_lapse * (1 - η_t ^ (Rᵈ * Γ_lapse / g))   # tropopause height [m]
T_strat = T_sfc * η_t ^ (Rᵈ * Γ_lapse / g)                     # stratospheric T [K]

@info "Derived parameters" z_trop T_strat

function η_from_z(z)
    if z ≤ z_trop
        return (1 - Γ_lapse * z / T_sfc) ^ (g / (Rᵈ * Γ_lapse))
    else
        return η_t * exp(-g * (z - z_trop) / (Rᵈ * T_strat))
    end
end

# ### Zonal wind
#
# The jet lives in the troposphere (``\eta \ge \eta_0``) and is zero in the
# stratosphere.  It peaks at ``\eta = \eta_0 \approx 252\,\text{hPa}`` and
# decays toward the surface.

function u_balanced(φ_deg, z)
    η = η_from_z(z)
    η < η₀ && return 0.0
    φ = deg2rad(φ_deg)
    ηᵥ = (η - η₀) * π / 2
    return u_jet * cos(ηᵥ)^(3/2) * sin(2φ)^2
end

# ### Temperature
#
# Horizontal-mean profile ``\bar{T}(\eta)`` plus a thermal-wind correction
# that ensures gradient-wind balance with the zonal jet.

function T_balanced(φ_deg, z)
    η = η_from_z(z)
    φ = deg2rad(φ_deg)

    T_mean = η > η_t ? T_sfc * η ^ (Rᵈ * Γ_lapse / g) : T_strat

    ηᵥ = η ≥ η₀ ? (η - η₀) * π / 2 : 0.0
    sinηᵥ = sin(ηᵥ)
    cosηᵥ = cos(ηᵥ)

    (η < 1e-12 || abs(sinηᵥ) < 1e-14) && return T_mean

    sinφ = sin(φ)
    cosφ = cos(φ)

    A = (3 / (4η)) * (u_jet / Rᵈ) * sinηᵥ * sqrt(cosηᵥ)
    B = (-2sinφ^6 * (cosφ^2 + 1/3) + 10/63) * 2u_jet * cosηᵥ^(3/2)
    C = (8/5 * cosφ^3 * (sinφ^2 + 2/3) - π/4) * a_earth * Ω_earth

    return T_mean + A * (B + C)
end

# ### Potential temperature, density, perturbation

function θ_balanced(φ_deg, z)
    T = T_balanced(φ_deg, z)
    p = p_sfc * η_from_z(z)
    return T * (pˢᵗ / p) ^ κ
end

function ρ_balanced(φ_deg, z)
    T = T_balanced(φ_deg, z)
    p = p_sfc * η_from_z(z)
    return p / (Rᵈ * T)
end

function θ_perturbation(λ, φ, z)
    d² = ((λ - λ_c) * cosd(φ_c))^2 + (φ - φ_c)^2
    z_env = z < z_trop ? sin(π * z / z_trop) : FT(0)
    return FT(A_pert) * exp(-d² / σ_pert^2) * z_env
end

# ### Full initial-condition functions
#
# `set!` on a `LatitudeLongitudeGrid` passes ``(\lambda, \varphi, z)`` with
# longitude and latitude in **degrees**.

u_initial(λ, φ, z) = FT(u_balanced(φ, z))
θ_initial(λ, φ, z) = FT(θ_balanced(φ, z)) + θ_perturbation(λ, φ, z)
ρ_initial(λ, φ, z) = FT(ρ_balanced(φ, z))

# ### Set model state
#
# Density must be set before velocity so that momentum ``\rho u`` is
# computed correctly inside `set_velocity!`.

@info "Setting initial model state…"
@time begin
    set!(model; ρ = ρ_initial)
    set!(model; u = u_initial, θ = θ_initial)
end

θ_init_arr = Array(interior(model.formulation.potential_temperature))
u_init_arr = Array(interior(model.velocities.u))
@info "Initial θ diagnostics" θ_min=minimum(θ_init_arr) θ_max=maximum(θ_init_arr)
@info "Initial u diagnostics" u_min=minimum(u_init_arr) u_max=maximum(u_init_arr)

# ## Time step
#
# CFL on the acoustic sound speed.  The vertical spacing is the binding
# constraint on a coarse grid.

γ  = cᵖᵈ / (cᵖᵈ - Rᵈ)
cₛ = sqrt(γ * Rᵈ * T_sfc)
Δz_cell = z_top / Nz
Δt = FT(0.4 * Δz_cell / cₛ)

@info "Time step" Δt cₛ

# ## Cell-centre coordinates
#
# Used for index-range computation and plotting.

Δλ = 360.0 / Nλ
Δφ = (lat_north - lat_south) / Nφ

λ = range(Δλ / 2, 360 - Δλ / 2, length = Nλ)
φ = range(lat_south + Δφ / 2, lat_north - Δφ / 2, length = Nφ)
z = range(Δz_cell / 2, z_top - Δz_cell / 2, length = Nz)

# ## Loss function and adjoint
#
# The loss is the mean squared meridional velocity inside a midlatitude target
# band.  Because ``v = 0`` initially, ``\langle v^2 \rangle`` is exactly the
# meridional component of eddy kinetic energy — a direct measure of baroclinic
# growth rate.
#
# Index ranges are precomputed outside the loss so that Enzyme only sees
# fixed-size array slicing (no integer arithmetic inside the AD tape).

nsteps = 4

# Precompute initial-condition fields for the loss function.
@info "Initializing θ fields for loss function…"
@time begin
    θ_init  = CenterField(grid); set!(θ_init,  θ_initial)
    dθ_init = CenterField(grid); set!(dθ_init, FT(0))
end

# Target region in latitude (midlatitude storm track) and height (troposphere).
φ_bounds = (25.0, 65.0)
z_bounds = (0.0, 15000.0)

@inline function bounded_index_range(bounds, coords)
    lo, hi = bounds
    vals = collect(coords)
    i_lo = findfirst(v -> v ≥ lo, vals)
    i_hi = findlast(v -> v ≤ hi, vals)
    (isnothing(i_lo) || isnothing(i_hi) || i_hi < i_lo) &&
        error("Invalid bounds $bounds for coordinates.")
    return i_lo:i_hi
end

λR = 1:Nλ
φR = bounded_index_range(φ_bounds, φ)
zR = bounded_index_range(z_bounds, z)

@info "Loss target region" λR φR zR

function loss(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = FT(1))
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    v = model.velocities.v
    v_int = interior(v)
    target = @view v_int[λR, φR, zR]
    return mean(target .^ 2)
end

function grad_loss(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    parent(dθ_init) .= 0
    _, loss_val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return dθ_init, loss_val
end

# ## Reactant compilation
#
# `Reactant.@compile` traces both the forward model and the Enzyme-generated
# adjoint into XLA/StableHLO, producing fused, optimized kernels.

@info "Compiling forward pass…"
@time compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true loss(
    model, θ_init, Δt, nsteps)

@info "Compiling backward pass (Enzyme reverse mode)…"
dmodel = Enzyme.make_zero(model)
@time compiled_bwd = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model, dmodel, θ_init, dθ_init, Δt, nsteps)

# ## Forward state and sensitivity

@info "Running compiled forward pass…"
@time compiled_fwd(model, θ_init, Δt, nsteps)

v_evolved = Array(interior(model.velocities.v))
θ_evolved = Array(interior(model.formulation.potential_temperature))

@info "Computing sensitivity (Enzyme reverse mode)…"
@time dθ, J = compiled_bwd(model, dmodel, θ_init, dθ_init, Δt, nsteps)
sensitivity = Array(interior(dθ))

@info "Loss value" J
@info "Max |∂J/∂θ₀|" maximum_sensitivity = maximum(abs, sensitivity)

# ## Visualisation
#
# Top row: evolved meridional velocity (map view + latitude–height cross-section).
# Bottom row: adjoint sensitivity ``\partial J / \partial \theta_0``.

k_tropo = argmin(abs.(collect(z) .- 5000.0))
i_pert  = argmin(abs.(collect(λ) .- λ_c))

fig = Figure(size = (1400, 900), fontsize = 14)

Label(fig[0, :],
    @sprintf("Baroclinic wave (JW06) — %d steps, Δt = %.2f s, J = %.6e", nsteps, Δt, J),
    fontsize = 16, tellwidth = false)

# Top row: evolved v.

vlim = max(maximum(abs, v_evolved), eps(FT))

ax1 = Axis(fig[1, 1]; xlabel = "λ (°)", ylabel = "φ (°)",
           title = "v  — z ≈ $(Int(round(z[k_tropo]))) m",
           aspect = DataAspect())
hm1 = heatmap!(ax1, collect(λ), collect(φ), v_evolved[:, :, k_tropo];
               colormap = :balance, colorrange = (-vlim, vlim))
Colorbar(fig[1, 2], hm1; label = "v (m/s)")

ax2 = Axis(fig[1, 3]; xlabel = "φ (°)", ylabel = "z (m)",
           title = "v  — λ ≈ $(Int(round(λ[i_pert])))°")
hm2 = heatmap!(ax2, collect(φ), collect(z), v_evolved[i_pert, :, :];
               colormap = :balance, colorrange = (-vlim, vlim))
Colorbar(fig[1, 4], hm2; label = "v (m/s)")

# Bottom row: sensitivity ∂J/∂θ₀.

slimit = max(maximum(abs, sensitivity), eps(FT))

ax3 = Axis(fig[2, 1]; xlabel = "λ (°)", ylabel = "φ (°)",
           title = "∂J/∂θ₀  — z ≈ $(Int(round(z[k_tropo]))) m",
           aspect = DataAspect())
hm3 = heatmap!(ax3, collect(λ), collect(φ), sensitivity[:, :, k_tropo];
               colormap = :balance, colorrange = (-slimit, slimit))
Colorbar(fig[2, 2], hm3; label = "∂J/∂θ₀")

ax4 = Axis(fig[2, 3]; xlabel = "φ (°)", ylabel = "z (m)",
           title = "∂J/∂θ₀  — λ ≈ $(Int(round(λ[i_pert])))°")
hm4 = heatmap!(ax4, collect(φ), collect(z), sensitivity[i_pert, :, :];
               colormap = :balance, colorrange = (-slimit, slimit))
Colorbar(fig[2, 4], hm4; label = "∂J/∂θ₀")

@time save("baroclinic_wave.png", fig; px_per_unit = 2)
@info "Saved baroclinic_wave.png"

nothing #hide
