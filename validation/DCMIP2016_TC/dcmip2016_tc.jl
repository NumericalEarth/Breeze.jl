# DCMIP2016 tropical cyclone (Reed–Jablonowski) validation case — Breeze.jl
#
# Analytic balanced vortex (Reed & Jablonowski 2011; Ullrich et al. 2016, DCMIP2016;
# Willson et al. 2024, GMD 17:2493) in a quiescent moist tropical environment, on a
# LatitudeLongitudeGrid with compressible dynamics + acoustic substepping. A weak
# warm-core vortex at (λc, φc) = (180°, 10°N) intensifies into a tropical cyclone over
# ~10 days, driven by bulk surface enthalpy fluxes over a fixed SST = 302.15 K.
#
# This file exposes a single generator, `dcmip2016_tropical_cyclone_simulation`, that
# builds a fully configured `Simulation` for a given horizontal resolution and advection
# order. Running the file directly executes the "best" configuration (0.25° + WENO9 +
# complete Reed–Jablonowski simple physics); `dcmip2016_tc_intercomparison.jl` reuses the
# generator to sweep resolution × advection order. Everything below the generator (the RJ
# constants and analytic initial state) is the fixed test definition; only the generator's
# keyword arguments vary between runs.
#
# Reed–Jablonowski "simple physics" (all three components):
#   1. wind-dependent bulk surface drag  Cᴰ = min(a + b|v|, cmax)  — defined below (`WindDependentDrag`)
#   2. wind-dependent boundary-layer eddy diffusivity              — defined below (`rj_Km`/`rj_Ke`)
#   3. large-scale condensation with instantaneous rain-out that retains latent warming —
#      Breeze's built-in `InstantaneousPrecipitation` scheme (shipped with Breeze), selected here
#
# Usage:
#   julia --project dcmip2016_tc.jl                         # runs the best configuration
#   include("dcmip2016_tc.jl")                              # just defines the generator
#   sim = dcmip2016_tropical_cyclone_simulation(; resolution=0.25, advection_order=9)

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using Oceananigans.TurbulenceClosures: VerticalScalarDiffusivity
using Oceananigans.AbstractOperations: @at, KernelFunctionOperation
using Oceananigans.Grids: znode
using Oceananigans: UpdateStateCallsite
## Bulk-flux types collide with Oceananigans' own exports; take Breeze's.
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Breeze.BoundaryConditions: wind_speed²ᶜᶜᶜ

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

# ---------------------------------------------------------------- physics types
# RJ wind-dependent surface drag: Cᴰ = min(a + b|v|, cmax). Implemented as a coefficient
# type dispatching Breeze's `bulk_coefficient` (which `BulkDrag` calls with the
# cell-centered surface wind). Heat/moisture coefficients stay constant.
struct WindDependentDrag{T}
    a::T
    b::T
    cmax::T
end
@inline (cd::WindDependentDrag)(i, j, grid, U, T₀) = min(cd.a + cd.b * U, cd.cmax)
@inline Breeze.BoundaryConditions.bulk_coefficient(i, j, grid, cd::WindDependentDrag, fields, T₀, ::Nothing) =
    cd(i, j, grid, sqrt(wind_speed²ᶜᶜᶜ(i, j, grid, fields)), T₀)
# Wind-only drag: no stability correction, so no filtered virtual-potential-temperature is
# needed (mirrors Breeze's constant-coefficient path `filtered_θᵥ_source(::Number) = nothing`).
Breeze.BoundaryConditions.filtered_θᵥ_source(::WindDependentDrag) = nothing

# Reed–Jablonowski (2012) wind-dependent boundary-layer eddy diffusivity, per the
# DCMIP2016 simple_physics reference (TC_PBL_mod = false):
#   Kₘ = Cᴰ(|v_sfc|)·|v_sfc|·zₐ · taper(p),   Kₑ = C·|v_sfc|·zₐ · taper(p),
# with C = 1.1e-3, zₐ = lowest-level center height, taper = 1 for p ≥ 850 hPa else
# exp(−(p_top−p)²/pblconst²) (p_top = 85000 Pa, pblconst = 10000 Pa). Cᴰ is the same
# wind-dependent drag used at the surface. Kₘ, Kₑ are recomputed each step into stored
# Center fields by a callback (so the diffusion operator does a plain array read,
# sidestepping in-kernel dynamic dispatch); the generator wires that up below.
@inline function pbl_wind_taper(i, j, k, grid, u, v, pressure)
    FT = eltype(grid)
    wind = sqrt(wind_speed²ᶜᶜᶜ(i, j, grid, (u = u, v = v)))                # |v| at lowest level
    @inbounds p = pressure[i, j, k]
    taper = ifelse(p ≥ FT(85000), one(FT), exp(-(FT(85000) - p)^2 / FT(1e8)))
    return wind, taper
end
@inline function rj_Km(i, j, k, grid, u, v, pressure, zₐ)
    FT = eltype(grid)
    wind, taper = pbl_wind_taper(i, j, k, grid, u, v, pressure)
    Cᴰ = ifelse(wind < FT(20), FT(7e-4) + FT(6.5e-5) * wind, FT(2e-3))     # = surface drag
    return Cᴰ * wind * FT(zₐ) * taper
end
@inline function rj_Ke(i, j, k, grid, u, v, pressure, zₐ)
    FT = eltype(grid)
    wind, taper = pbl_wind_taper(i, j, k, grid, u, v, pressure)
    return FT(1.1e-3) * wind * FT(zₐ) * taper
end

# ----------------------------------------------------- fixed DCMIP2016 constants
# DCMIP2016 constants (Tables 2–3)
constants = ThermodynamicConstants(;
    gravitational_acceleration = 9.80616,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287)   # => Rᵈ = 287.0

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

# RJ test constants (Table 3)
zt  = 15000.0      # tropopause height (m)
q0  = 0.021        # max specific humidity (kg/kg)
qtᵘ = 1e-11        # upper-atmosphere specific humidity
T0  = 302.15       # surface air temperature (K)
Ts  = 302.15       # SST (K)
zq1 = 3000.0
zq2 = 8000.0
Γ   = 0.007        # virtual-temperature lapse rate (K/m)
pb  = 101500.0     # background surface pressure (Pa, 1015 hPa)
φc  = 10.0         # vortex-center latitude (deg, π/18)
λc  = 180.0        # vortex-center longitude (deg, π)
Δp  = 1115.0       # central pressure deficit (Pa, 11.15 hPa)
rp  = 282000.0     # horizontal half-width of p perturbation (m)
zp  = 7000.0       # vertical decay scale of p perturbation (m)
ϵ0  = 1e-25        # divide-by-zero guard
Mv  = 0.608        # virtual-temperature coefficient
p00 = 1e5          # reference pressure for θ (Pa, 1000 hPa)

Tv0 = T0 * (1 + Mv * q0)          # surface virtual temperature
Tvt = Tv0 - Γ * zt                # tropopause virtual temperature
pt  = pb * (Tvt / Tv0)^(g / (Rᵈ * Γ))   # tropopause pressure (Eq 5)
fc  = 2Ω * sind(φc)               # Coriolis at vortex center

# ----------------------------------------------------- analytic initial state
# Background sounding (Eqs 1, 2, 4)
@inline q̄(z)  = ifelse(z ≤ zt, q0 * exp(-z / zq1) * exp(-(z / zq2)^2), qtᵘ)
@inline T̄ᵥ(z) = ifelse(z ≤ zt, Tv0 - Γ * z, Tvt)
@inline p̄(z)  = ifelse(z ≤ zt,
                       pb * ((Tv0 - Γ * z) / Tv0)^(g / (Rᵈ * Γ)),
                       pt * exp(g * (zt - z) / (Rᵈ * Tvt)))

# Great-circle radius from vortex center (Eq 7), argument clamped to [-1, 1]
@inline function radius(λ, φ)
    arg = sind(φc) * sind(φ) + cosd(φc) * cosd(φ) * cosd(λ - λc)
    return a * acos(min(1.0, max(-1.0, arg)))
end

# Pressure perturbation (Eq 8) and full pressure (Eq 6)
@inline function pressure(λ, φ, z)
    r  = radius(λ, φ)
    A  = (r / rp)^1.5
    B  = (z / zp)^2
    p′ = ifelse(z ≤ zt, -Δp * exp(-A - B) * ((Tv0 - Γ * z) / Tv0)^(g / (Rᵈ * Γ)), 0.0)
    return p̄(z) + p′
end

# Virtual temperature (Eqs 11–12) and density (Eq 16)
@inline function virtual_temperature(λ, φ, z)
    r = radius(λ, φ)
    A = (r / rp)^1.5
    B = (z / zp)^2
    E = exp(A + B)
    inner = 1 + (2Rᵈ * (Tv0 - Γ * z) * z) / (g * zp^2 * (1 - (pb / Δp) * E))
    Tᵥ′ = ifelse(z ≤ zt, (Tv0 - Γ * z) * (1 / inner - 1), 0.0)
    return T̄ᵥ(z) + Tᵥ′
end

density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))

# Temperature and potential temperature (Eqs 3, 13–15)
temperature(λ, φ, z) = virtual_temperature(λ, φ, z) / (1 + Mv * q̄(z))
potential_temperature(λ, φ, z) = temperature(λ, φ, z) * (p00 / pressure(λ, φ, z))^κ

total_specific_humidity(λ, φ, z) = q̄(z)

# Gradient-wind tangential velocity (Eq 18) projected to (u, v) (Eqs 19–23)
@inline function tangential_velocity(λ, φ, z)
    r = radius(λ, φ)
    A = (r / rp)^1.5
    B = (z / zp)^2
    E = exp(A + B)
    denom = 1 + (2Rᵈ * (Tv0 - Γ * z) * z) / (g * zp^2) - (pb / Δp) * E
    under = (fc^2 * r^2) / 4 - (1.5 * A * (Tv0 - Γ * z) * Rᵈ) / denom
    return ifelse(z ≤ zt, -fc * r / 2 + sqrt(max(0.0, under)), 0.0)
end

@inline function projection(λ, φ)
    d1 = sind(φc) * cosd(φ) - cosd(φc) * sind(φ) * cosd(λ - λc)
    d2 = cosd(φc) * sind(λ - λc)
    d  = max(ϵ0, sqrt(d1^2 + d2^2))
    return d1 / d, d2 / d
end

function zonal_velocity(λ, φ, z)
    p1, _ = projection(λ, φ)
    return tangential_velocity(λ, φ, z) * p1
end

function meridional_velocity(λ, φ, z)
    _, p2 = projection(λ, φ)
    return tangential_velocity(λ, φ, z) * p2
end

# ---------------------------------------------------------------- vertical grids
# All vertical grids share the 30 km rigid lid. `stretched_z_faces` is the standard
# exponentially-stretched, surface-refined grid; the 32-level instance (s = 4.2,
# Δz₁ ≈ 64 m, Δz_top ≈ 3.7 km) is the DCMIP2016 baseline. The 64-level variants below feed
# the vertical-resolution study in `dcmip2016_tc_intercomparison.jl`.
const TC_LID = 30kilometers

stretched_z_faces(Nz, s; lid = TC_LID) = [lid * (exp(s * k / Nz) - 1) / (exp(s) - 1) for k in 0:Nz]

# 64-level grid that re-places levels into the 5–14 km eyewall-updraft layer instead of the
# boundary layer (motivated by FV3's near-dissipation-free Lagrangian vertical coordinate):
# Δz₁ = 64 m (matches the baseline lowest level, so surface fluxes are unchanged), ramped to a
# uniform 400 m spacing through 2–16 km (22 levels in 5–14 km vs 7 in the baseline s = 4.2
# grid), then stretched to the 30 km lid. The explicit faces are recorded here so the grid is
# reproducible without the generating tool.
updraft_refined_z_faces() = [
       0.0,    64.0,   147.15,   255.19,   395.56,   577.94,   814.89,  1122.76,
    1522.76,  1922.76,  2322.76,  2722.76,  3122.76,  3522.76,  3922.76,  4322.76,
    4722.76,  5122.76,  5522.76,  5922.76,  6322.76,  6722.76,  7122.76,  7522.76,
    7922.76,  8322.76,  8722.76,  9122.76,  9522.76,  9922.76, 10322.76, 10722.76,
   11122.76, 11522.76, 11922.76, 12322.76, 12722.76, 13122.76, 13522.76, 13922.76,
   14322.76, 14722.76, 15122.76, 15522.76, 15922.76, 16322.76, 16745.11, 17191.05,
   17661.91, 18159.07, 18684.01, 19238.28, 19823.52, 20441.45, 21093.90, 21782.81,
   22510.21, 23278.24, 24089.18, 24945.43, 25849.52, 26804.12, 27812.05, 28876.30,
   30000.0]

# ------------------------------------------------------------------- generator
"""
    dcmip2016_tropical_cyclone_simulation(; resolution = 0.25, advection_order = 9, kwargs...)

Build a `Simulation` for the DCMIP2016 Reed–Jablonowski tropical cyclone at the given
horizontal `resolution` (degrees) and WENO `advection_order`, with the complete RJ simple
physics (wind-dependent surface drag, wind-dependent boundary-layer mixing, and
`InstantaneousPrecipitation` rain-out). The vortex, sounding, SST, lid (30 km), and
vertical grid (32 surface-refined levels) are the fixed DCMIP2016 test definition.

Keyword arguments
=================
  - `resolution`: horizontal grid spacing in degrees (0.5 and 0.25 are the validated values).
  - `advection_order`: WENO order (5 or 9).
  - `z_faces`: vector of vertical cell-face heights (default: 32 surface-refined levels to a
    30 km lid). `Nz` is inferred as `length(z_faces) - 1`.
  - `architecture`: `GPU()` (default) or `CPU()`.
  - `stop_time`: simulated duration (default 10 days).
  - `initial_Δt`, `max_Δt`, `cfl`: time-step wizard controls.
  - `output_prefix`: prefix for the JLD2 output files (`<prefix>_psfc.jld2`, `<prefix>_speed.jld2`).
  - `save_fields`: also write the 3D wind-speed + vertical-velocity series (large; default `true`).
  - `output_interval`: output cadence (default 2 hours).

Returns the `Simulation`; call `run!` on it.
"""
function dcmip2016_tropical_cyclone_simulation(; resolution = 0.25,
                                                 advection_order = 9,
                                                 z_faces = stretched_z_faces(32, 4.2),
                                                 architecture = GPU(),
                                                 stop_time = 10days,
                                                 initial_Δt = 30,
                                                 max_Δt = 180,
                                                 cfl = 0.8,
                                                 output_prefix = "dcmip_tc",
                                                 save_fields = true,
                                                 output_interval = 2hours)

    # Global longitude (periodic); latitude band around the vortex; the vertical grid is
    # supplied via `z_faces` (default: 32 surface-refined levels, lowest ≈ 64 m, 30 km lid).
    Nλ = round(Int, 360 / resolution)
    φ_south, φ_north = -40.0, 60.0
    Nφ = round(Int, (φ_north - φ_south) / resolution)
    Nz = length(z_faces) - 1

    grid = LatitudeLongitudeGrid(architecture;
                                 size = (Nλ, Nφ, Nz),
                                 halo = (5, 5, 5),
                                 longitude = (0, 360),
                                 latitude = (φ_south, φ_north),
                                 z = z_faces)

    # Boundary-layer mixing: Center fields hold the wind-dependent diffusivities; a callback
    # refreshes them each step. Passing the fields directly lets the closure's own adapt handle
    # the device transfer and infer the (Center, Center, Center) location.
    Kₘ_field = CenterField(grid)
    Kₑ_field = CenterField(grid)
    # lowest-level center height (≈ 32 m); @allowscalar for the single host read of the z-array
    zₐ = CUDA.@allowscalar znode(1, 1, 1, grid, Center(), Center(), Center())
    closure = VerticalScalarDiffusivity(ν = Kₘ_field, κ = Kₑ_field)

    coriolis = SphericalCoriolis(rotation_rate = Ω)

    T₀ᵣ = 250.0
    θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    surface_pressure = pb,
                                    reference_potential_temperature = θᵣ)

    microphysics = InstantaneousPrecipitation(equilibrium = WarmPhaseEquilibrium())

    # Bulk surface fluxes (Reed–Jablonowski simple physics) over fixed SST.
    # Wind-dependent drag Cᴰ = min(7e-4 + 6.5e-5|v|, 2e-3); constant heat/moisture Cᵀ = 1.1e-3.
    # `surface_temperature = Ts` sets the ideal-gas surface density ρ₀ in the drag stress
    # τ = ρ₀ Cᴰ |v| v (required under CompressibleDynamics, which has no reference profile),
    # consistent with the SST used by the enthalpy/moisture fluxes.
    FT = Oceananigans.defaults.FloatType
    Cᴰ = WindDependentDrag(FT(7e-4), FT(6.5e-5), FT(2e-3))
    Cᵀ = 1.1e-3
    Uᵍ = 1.0
    ρu_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ, gustiness = Uᵍ, surface_temperature = Ts))
    ρv_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ, gustiness = Uᵍ, surface_temperature = Ts))
    ρe_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient = Cᵀ, gustiness = Uᵍ,
                                                                    surface_temperature = Ts))
    ρqᵛ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient = Cᵀ, gustiness = Uᵍ,
                                                             surface_temperature = Ts))
    boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs, ρe = ρe_bcs, ρqᵛ = ρqᵛ_bcs)

    model = AtmosphereModel(grid; dynamics, coriolis, microphysics, closure, boundary_conditions,
                            thermodynamic_constants = constants,
                            advection = WENO(order = advection_order))

    set!(model;
         ρ  = density,
         θ  = potential_temperature,
         u  = zonal_velocity,
         v  = meridional_velocity,
         qᵗ = total_specific_humidity)

    @info @sprintf("Post-init: max|u|=%.1f max|v|=%.1f m/s | max T=%.1f min T=%.1f K | max qᵗ=%.4f | ρ(sfc)≈%.3f",
                   maximum(abs, model.velocities.u), maximum(abs, model.velocities.v),
                   maximum(model.temperature), minimum(model.temperature),
                   maximum(model.moisture_density) / maximum(model.dynamics.density),
                   maximum(model.dynamics.density))

    # Build the diffusivity operations now that the model fields exist. The captured
    # velocities/pressure are updated in place each step, so recomputing refreshes Kₘ, Kₑ.
    Km_op = KernelFunctionOperation{Center, Center, Center}(rj_Km, grid, model.velocities.u, model.velocities.v, model.dynamics.pressure, zₐ)
    Ke_op = KernelFunctionOperation{Center, Center, Center}(rj_Ke, grid, model.velocities.u, model.velocities.v, model.dynamics.pressure, zₐ)
    update_pbl_diffusivities!(sim) = (Kₘ_field .= Km_op; Kₑ_field .= Ke_op; nothing)

    simulation = Simulation(model; Δt = initial_Δt, stop_time)
    # max_Δt capped below the explicit vertical-diffusion limit Δz_min²/(2K). The RJ
    # wind-dependent K peaks ≈ Cᴰ·|v|·zₐ ≈ 2e-3·80·32 ≈ 5 m²/s, so 180 s keeps a margin.
    conjure_time_step_wizard!(simulation; cfl, max_Δt)
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

    # Seed the PBL diffusivity fields (velocities+pressure are valid after set!), then
    # refresh them every step inside update_state! (before the tendencies use them).
    update_pbl_diffusivities!(simulation)
    add_callback!(simulation, update_pbl_diffusivities!, IterationInterval(1); callsite = UpdateStateCallsite())

    # Key TC intensity metrics: minimum surface pressure (MSP) and max surface wind.
    pᵈ = model.dynamics.pressure
    function progress(sim)
        u, v, w = sim.model.velocities
        MSP = minimum(view(pᵈ, :, :, 1)) / 100   # hPa
        @info @sprintf("iter %5d | t=%s | Δt=%5.1fs | MSP=%.1f hPa | max|u|=%.1f | max|v|=%.1f | max|w|=%.3f",
                       iteration(sim), prettytime(sim), sim.Δt, MSP,
                       maximum(abs, u), maximum(abs, v), maximum(abs, w))
        return nothing
    end
    add_callback!(simulation, progress, IterationInterval(20))

    # Surface pressure (eye tracking + radial pressure profile) is always written; the 3D
    # wind speed + vertical velocity (cross-sections, animations) are large and optional.
    simulation.output_writers[:psfc] = JLD2Writer(model, (; p = pᵈ);
        filename = "$(output_prefix)_psfc.jld2",
        indices = (:, :, 1),
        schedule = TimeInterval(output_interval), overwrite_existing = true)
    if save_fields
        u, v, w = model.velocities
        speed = @at (Center, Center, Center) sqrt(u^2 + v^2)
        w_c   = @at (Center, Center, Center) w
        simulation.output_writers[:speed] = JLD2Writer(model, (; speed, w = w_c);
            filename = "$(output_prefix)_speed.jld2",
            schedule = TimeInterval(output_interval), overwrite_existing = true)
    end

    @info "Configured DCMIP2016 TC: $(Nλ)×$(Nφ)×$(Nz) ($(resolution)° band $(φ_south)–$(φ_north)°), WENO$(advection_order)"
    return simulation
end

# Running this file directly executes the best configuration: 0.25° + WENO9 + complete RJ
# physics — the deepest storm (≈ 921 hPa), ≈ 37 min on an H100. For a quicker check pass
# `resolution = 0.5` (≈ 8 min, min MSP ≈ 963 hPa); it is only ~5× faster despite 16× fewer
# cells, since the H100 is under-utilized at 0.5°.
if abspath(PROGRAM_FILE) == @__FILE__
    simulation = dcmip2016_tropical_cyclone_simulation(; resolution = 0.25, advection_order = 9)
    run!(simulation)
    @info "Completed without NaN. Final Δt=$(simulation.Δt) s, t=$(prettytime(simulation))"
end
