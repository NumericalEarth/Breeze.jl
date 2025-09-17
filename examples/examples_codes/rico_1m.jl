# using Pkg; Pkg.activate(".")

################################################################################
# RICO (precipitating shallow cumulus) — 1‑moment warm‑rain microphysics
# - Domain: 12.8 km × 12.8 km × 4 km  (change for CI below)
# - 1M microphysics: autoconversion + accretion + evaporation + sedimentation
# - Interactive bulk surface fluxes (SST 299.8 K; CM/CH/CE from RICO spec)
# - No top sponge active (optional block included but commented out)
################################################################################

@info "Loading packages..."
using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
@info "Packages loaded"

using AtmosphericProfilesLibrary
using CloudMicrophysics
using Printf
using CairoMakie

using Oceananigans.Operators: ∂zᶜᶜᶠ, ℑzᵃᵃᶜ

# Aliases for CloudMicrophysics
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.Microphysics1M as CM1
using Breeze.Thermodynamics: ReferenceState, reference_density, AtmosphereThermodynamics
using Breeze.Thermodynamics.CloudMicrophysicsCompat: evaporation_sublimation_adapter_fast, accretion_adapter

# ------------------------------------------------------------------------------
# Domain & resolution
# ------------------------------------------------------------------------------

# Nx = Ny = 128
# Nz = 100

# Lx = 12_800.0  # m
# Ly = 12_800.0  # m
# Lz = 4_000.0   # m

arch = CPU()  
stop_time = 24hours        # RICO standard: 24 hours

# Lightweight settings for CI
stop_time = 10minutes
Nx = Ny = 8
Nz = 30
Lx = 800.0
Ly = 800.0
Lz = 1_000.0

grid = RectilinearGrid(arch;
    size = (Nx, Ny, Nz),
    x = (0, Lx),
    y = (0, Ly),
    z = (0, Lz),
    halo = (5, 5, 5),
    topology = (Periodic, Periodic, Bounded)
)

FT = eltype(grid)

# ------------------------------------------------------------------------------
# RICO sounding and large-scale tendencies (from AtmosphericProfilesLibrary)
# ------------------------------------------------------------------------------

θ_rico = AtmosphericProfilesLibrary.Rico_θ_liq_ice(FT)
q_rico = AtmosphericProfilesLibrary.Rico_q_tot(FT)
u_rico = AtmosphericProfilesLibrary.Rico_u(FT)
v_rico = AtmosphericProfilesLibrary.Rico_v(FT)

# Reference state for buoyancy (use near-surface θ_l)
p₀ = FT(101_500.0)        # Pa
θ₀ = θ_rico(FT(0))        # K
reference_constants = Breeze.Thermodynamics.ReferenceConstants(base_pressure = p₀,
                                                               potential_temperature = θ₀)
buoyancy = Breeze.MoistAirBuoyancy(; reference_constants)

# ------------------------------------------------------------------------------
# 1‑moment warm‑rain microphysics (CliMA-style)
#   - add rain tracer :qr
#   - tendencies: autoconversion, accretion, evaporation (qr↔q), sedimentation
# ------------------------------------------------------------------------------

# Parameter bundles (defaults align with CliMA's 1M setup)
rain    = CMP.Rain(FT)               # autoconversion + rain properties

liquid  = CMP.CloudLiquid(FT)        # cloud-liquid properties
ce      = CMP.CollisionEff(FT)       # collision efficiency
velpars = CMP.Blk1MVelType(FT)       # bulk 1M terminal-velocity law (can switch to Chen2022)
aps     = CMP.AirProperties(FT)      # air transport properties
tps     = Breeze.Thermodynamics.AtmosphereThermodynamics(FT)

# Reference density profile for microphysics (from Breeze thermodynamics)
ref_state = ReferenceState(FT; base_pressure = FT(101_500.0), potential_temperature = θ_rico(FT(0)))
ρ_ref = CenterField(grid)
set!(ρ_ref, (x, y, z) -> reference_density(z, ref_state, tps))

# Face-centered rain fall speed field
wʳ = ZFaceField(grid)
# Initialize a simple constant fall speed; can be upgraded to ρ-dependent later
set!(wʳ, (x, y, z) -> FT(5.0))

# Helper for discrete vertical flux terms (you used this pattern already)
@inline w_dz_ϕ(i, j, k, grid, w, ϕ) = @inbounds w[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, ϕ)

# We'll diagnose temperature and cloud liquid each time step into Fields
# so the microphysics kernels can read them on GPU.
T_f  = CenterField(grid)
ql_f = CenterField(grid)

# 1) Rain formation + evaporation (tendencies for qr and q) via CloudMicrophysics 1M
@inline function tend_qr_1M(i, j, k, grid, clock, fields, p)
    @inbounds begin
        ql   = p.ql_f[i, j, k]     # diagnosed cloud liquid
        qr   = fields.qr[i, j, k]
        qtot = fields.q[i, j, k]   # non-precipitating total water (vapor+cloud)
        Tloc = p.T_f[i, j, k]
        ρ    = p.ρ_ref[i, j, k]
    end

    # Autoconversion (liq -> rain)
    acnv = CM1.conv_q_liq_to_q_rai(p.rain.acnv1M, ql, true)

    # Accretion (rain collects cloud)
    accr = accretion_adapter(p.cloud, p.rain, p.velpars.rain, p.ce, ql, qr, ρ)

    # Evaporation (rain -> vapor), negative rate implies qr loss
    evap = -evaporation_sublimation_adapter_fast(p.rain, p.velpars.rain, p.aps, qtot, ql, qr, ρ, Tloc)

    # Limit evaporation by available rain mass in this step
    Δt = clock.Δt
    evap_limited = evap < 0 ? -min(qr, -evap * Δt) / Δt : evap

    return acnv + accr + evap_limited
end

@inline function tend_q_from_1M(i, j, k, grid, clock, fields, p)
    @inbounds begin
        ql   = p.ql_f[i, j, k]
        qr   = fields.qr[i, j, k]
        qtot = fields.q[i, j, k]
        Tloc = p.T_f[i, j, k]
        ρ    = p.ρ_ref[i, j, k]
    end

    acnv = CM1.conv_q_liq_to_q_rai(p.rain.acnv1M, ql, true)
    accr = accretion_adapter(p.cloud, p.rain, p.velpars.rain, p.ce, ql, qr, ρ)
    evap = -evaporation_sublimation_adapter_fast(p.rain, p.velpars.rain, p.aps, qtot, ql, qr, ρ, Tloc)

    Δt = clock.Δt
    evap_to_q = evap < 0 ? min(qr, -evap * Δt) / Δt : 0

    # q loses autoconversion+accretion (mass to precip), gains from rain evaporation
    return -(acnv + accr) + evap_to_q
end

qr_mp_forcing = Forcing(tend_qr_1M, discrete_form = true, field_dependencies = (:θ, :q, :qr),
                        parameters = (; ρ_ref = ρ_ref, ql_f = ql_f, T_f = T_f,
                                      rain = rain, cloud = liquid, velpars = velpars,
                                      ce = ce, aps = aps))

q_mp_forcing  = Forcing(tend_q_from_1M, discrete_form = true, field_dependencies = (:θ, :q, :qr),
                        parameters = (; ρ_ref = ρ_ref, ql_f = ql_f, T_f = T_f,
                                      rain = rain, cloud = liquid, velpars = velpars,
                                      ce = ce, aps = aps))

# 2) Rain sedimentation:  -∂z ( wʳ * qʳ )
@inline function Fqr_sedimentation(i, j, k, grid, clock, fields, p)
    return - ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wʳ, fields.qr)
end
qr_sed_forcing = Forcing(Fqr_sedimentation, discrete_form = true, parameters = (; wʳ))

# ------------------------------------------------------------------------------
# Surface fluxes: interactive bulk formulas (RICO)
#   SST = 299.8 K; C_M/C_H/C_E from intercomparison; pressure pₛ ~ 1015 hPa
# ------------------------------------------------------------------------------

const SST = FT(299.8)              # K
const C_M = FT(0.001229)
const C_H = FT(0.001094)
const C_E = FT(0.001133)

const ε   = FT(0.622)
const pₛ  = FT(101_500.0)

# Saturation mixing ratio at SST (Buck equation for es)
@inline function e_sat_buck(T::FT)
    Tc = T - FT(273.15)
    return FT(611.21) * exp(FT(17.502) * Tc / (FT(240.97) + Tc))  # Pa
end
const qv_star_SST = let e = e_sat_buck(SST); ε * e / (pₛ - (FT(1) - ε) * e) end

@inline speed(u, v) = sqrt(u*u + v*v + 1e-12)  # avoid 0/0

@inline function θ_flux_bulk(x, y, t, u, v, θ, p)
    U = sqrt(u*u + v*v + 1e-12)
    return - p.C_H * U * (θ - p.θs)            # θ is close to T near sea level
end

@inline function q_flux_bulk(x, y, t, u, v, q, p)
    U = sqrt(u*u + v*v + 1e-12)
    return - p.C_E * U * (q - p.qs)            # moistening if air is drier than surface
end

θ_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(θ_flux_bulk,
                    field_dependencies = (:u, :v, :θ), parameters = (; C_H = C_H, θs = SST)))
q_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(q_flux_bulk,
                    field_dependencies = (:u, :v, :q), parameters = (; C_E = C_E, qs = qv_star_SST)))

# Momentum drag using the same bulk |U| form
@inline τu_bulk(x, y, t, u, v, p) = - p.C_M * sqrt(u*u + v*v + 1e-12) * u
@inline τv_bulk(x, y, t, u, v, p) = - p.C_M * sqrt(u*u + v*v + 1e-12) * v
u_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(τu_bulk, field_dependencies = (:u, :v),
                                                               parameters = (; C_M = C_M)))
v_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(τv_bulk, field_dependencies = (:u, :v),
                                                               parameters = (; C_M = C_M)))

# ------------------------------------------------------------------------------
# Large-scale subsidence (discrete form, same pattern as your BOMEX)
# ------------------------------------------------------------------------------

@inline function Fu_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    u_avg = parameters.u_avg
    w_dz_U = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, u_avg)
    return -w_dz_U
end

@inline function Fv_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    v_avg = parameters.v_avg
    w_dz_V = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, v_avg)
    return -w_dz_V
end

@inline function Fθ_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    θ_avg = parameters.θ_avg
    w_dz_T = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, θ_avg)
    return -w_dz_T
end

@inline function Fq_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    q_avg = parameters.q_avg
    w_dz_Q = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, q_avg)
    return -w_dz_Q
end

# Allocate mean profiles used in discrete forcings
u_avg_f = CenterField(grid)
v_avg_f = CenterField(grid)
θ_avg_f = CenterField(grid)
q_avg_f = CenterField(grid)

# Subsidence profile from APL
wˢ = ZFaceField(grid)
w_rico = AtmosphericProfilesLibrary.Rico_subsidence(FT)
set!(wˢ, (x, y, z) -> w_rico(z))

u_subsidence_forcing = Forcing(Fu_subsidence, discrete_form = true, parameters = (; u_avg = u_avg_f, wˢ))
v_subsidence_forcing = Forcing(Fv_subsidence, discrete_form = true, parameters = (; v_avg = v_avg_f, wˢ))
θ_subsidence_forcing = Forcing(Fθ_subsidence, discrete_form = true, parameters = (; θ_avg = θ_avg_f, wˢ))
q_subsidence_forcing = Forcing(Fq_subsidence, discrete_form = true, parameters = (; q_avg = q_avg_f, wˢ))

# ------------------------------------------------------------------------------
# Geostrophic wind relaxation on an f-plane (RICO ~ 18°N ⇒ f ≈ 4.51e-5 s⁻¹)
# ------------------------------------------------------------------------------

coriolis = FPlane(f = FT(4.5068e-5)) # 2Ω sin(18°)

uᵍ = CenterField(grid)
vᵍ = CenterField(grid)
uᵍ_rico = AtmosphericProfilesLibrary.Rico_geostrophic_ug(FT)
vᵍ_rico = AtmosphericProfilesLibrary.Rico_geostrophic_vg(FT)
set!(uᵍ, (x, y, z) -> uᵍ_rico(z))
set!(vᵍ, (x, y, z) -> vᵍ_rico(z))

@inline function Fu_geostrophic(i, j, k, grid, clock, fields, parameters)
    f = parameters.f
    @inbounds begin
        v_avg = parameters.v_avg[1, 1, k]
        v = fields.v[i, j, k]
        vᵍ = parameters.vᵍ[1, 1, k]
    end
    return + f * (v_avg - vᵍ)
end

@inline function Fv_geostrophic(i, j, k, grid, clock, fields, parameters)
    f = parameters.f
    @inbounds begin
        u_avg = parameters.u_avg[1, 1, k]
        u = fields.u[i, j, k]
        uᵍ = parameters.uᵍ[1, 1, k]
    end
    return - f * (u_avg - uᵍ)
end

u_geostrophic_forcing = Forcing(Fu_geostrophic, discrete_form = true,
                                parameters = (; v_avg = v_avg_f, f = coriolis.f, vᵍ))
v_geostrophic_forcing = Forcing(Fv_geostrophic, discrete_form = true,
                                parameters = (; u_avg = u_avg_f, f = coriolis.f, uᵍ))

u_forcing = (u_subsidence_forcing, u_geostrophic_forcing)
v_forcing = (v_subsidence_forcing, v_geostrophic_forcing)

# ------------------------------------------------------------------------------
# Moisture & temperature large-scale tendencies
#   - RICO provides dq_t/dt; radiation usually folded into a θ tendency.
#   - We'll use dq_t/dt from APL and subsidence for θ (you can add a θ LS tendency if desired).
# ------------------------------------------------------------------------------

drying = CenterField(grid)
dqdt_rico = AtmosphericProfilesLibrary.Rico_dqtdt(FT)
set!(drying, (x, y, z) -> dqdt_rico(z))
q_drying_forcing = Forcing(drying)

# θ large-scale tendency (optional; commented — add if you have a profile)
# Fθ_field = Field{Nothing, Nothing, Center}(grid)
# set!(Fθ_field, z -> z < 740  ? (-2.0/86400) :
#                   z < 2260 ? (-1.5/86400) :
#                               (-1.0/86400))
# θ_ls_forcing = Forcing(Fθ_field)

# ------------------------------------------------------------------------------
# (Optional) top sponge (commented out per your request)
# ------------------------------------------------------------------------------

# z_sponge = 0.8Lz
# τ_sponge = FT(300.0)
# α = Field{Nothing, Nothing, Center}(grid)
# set!(α, z -> z > z_sponge ? (z - z_sponge) / (Lz - z_sponge) / τ_sponge : FT(0))
# θ_ref = Field{Nothing, Nothing, Center}(grid); set!(θ_ref, z -> θ_rico(z))
# @inline function sponge_θ(i, j, k, grid, clock, fields, p)
#     return - p.α[i, j, k] * (fields.θ[i, j, k] - p.θ_ref[1, 1, k])
# end
# θ_sponge_forcing = Forcing(sponge_θ, field_dependencies = :θ, parameters = (; α, θ_ref))

# ------------------------------------------------------------------------------
# Numerics, model, and boundary conditions
# ------------------------------------------------------------------------------

advection = WENO(order = 9)
closure  = SmagorinskyLilly()   # common for RICO/LES; swap if you prefer

# q (non‑precip) gets: LS drying + subsidence + 1M source/sink (−acnv−accr+evap)
q_forcing  = (q_drying_forcing, q_subsidence_forcing, q_mp_forcing)

# qr (precip) gets: 1M formation/evap + sedimentation
qr_forcing = (qr_mp_forcing, qr_sed_forcing)

# θ gets: subsidence (+ optional LS tendency and sponge if you enable)
θ_forcing = (θ_subsidence_forcing,)
# θ_forcing = (θ_subsidence_forcing, θ_ls_forcing, θ_sponge_forcing)  # example if enabling options

@info "Creating model..."
@info "Setting up grid..."
model = NonhydrostaticModel(; grid, advection, buoyancy, coriolis, closure,
    tracers = (:θ, :q, :qr),
    forcing = (; q = q_forcing, qr = qr_forcing, u = u_forcing, v = v_forcing, θ = θ_forcing),
    boundary_conditions = (θ = θ_bcs, q = q_bcs, u = u_bcs, v = v_bcs)
)

# ------------------------------------------------------------------------------
# Initial conditions (θ_l, q_t from RICO; u, v from RICO; random θ/q noise)
# ------------------------------------------------------------------------------

θϵ = FT(0.1)
qϵ = FT(2.5e-5)
θᵢ(x, y, z) = θ_rico(z) + θϵ * randn()
qᵢ(x, y, z) = q_rico(z) + qϵ * randn()
uᵢ(x, y, z) = u_rico(z)
vᵢ(x, y, z) = v_rico(z)
set!(model, θ = θᵢ, q = qᵢ, u = uᵢ, v = vᵢ, qr = (x, y, z) -> FT(0))  # start with no rain

# ------------------------------------------------------------------------------
# Time stepping & callbacks
# ------------------------------------------------------------------------------

simulation = Simulation(model; Δt = 10.0, stop_time)
conjure_time_step_wizard!(simulation, cfl = 0.7)

# Column means used by the discrete forcings
u_avg = Field(Average(model.velocities.u, dims = (1, 2)))
v_avg = Field(Average(model.velocities.v, dims = (1, 2)))
θ_avg = Field(Average(model.tracers.θ,    dims = (1, 2)))
q_avg = Field(Average(model.tracers.q,    dims = (1, 2)))

function compute_averages!(sim)
    compute!(u_avg); parent(u_avg_f) .= parent(u_avg)
    compute!(v_avg); parent(v_avg_f) .= parent(v_avg)
    compute!(θ_avg); parent(θ_avg_f) .= parent(θ_avg)
    compute!(q_avg); parent(q_avg_f) .= parent(q_avg)
    return nothing
end
add_callback!(simulation, compute_averages!)

# Diagnose T and q_l each step into plain Fields used by microphysics kernels
T    = Breeze.TemperatureField(model)
qˡ   = Breeze.CondensateField(model, T)
qᵛ★  = Breeze.SaturationField(model, T)  # for RH diagnostic

function update_thermo!(sim)
    compute!(T); compute!(qˡ)
    # Use interior to avoid halo regions
    interior(T_f)  .= interior(T)
    interior(ql_f) .= interior(qˡ)
    return nothing
end
add_callback!(simulation, update_thermo!)  # run every iteration

# ------------------------------------------------------------------------------
# Diagnostics (same as your BOMEX/RICO scaffold)
# ------------------------------------------------------------------------------

rh   = Field(model.tracers.q / qᵛ★)  # relative humidity

fig = Figure()
axu = Axis(fig[1, 1]); axv = Axis(fig[1, 2])
axθ = Axis(fig[1, 3]); axq = Axis(fig[1, 4])

function progress(sim)
    compute!(T); compute!(qˡ); compute!(rh)

    q  = sim.model.tracers.q
    qr = sim.model.tracers.qr
    θ  = sim.model.tracers.θ
    u, v, w = sim.model.velocities

    umax = maximum(abs, u_avg)
    vmax = maximum(abs, v_avg)
    qmax = maximum(q); qˡmax = maximum(qˡ); qrmax = maximum(qr); rhmax = maximum(rh)
    θmin = minimum(θ); θmax = maximum(θ)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, max|ū|: (%.2e, %.2e), max(rh): %.2f",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), umax, vmax, rhmax)
    msg *= @sprintf(", max(q): %.2e, max(qˡ): %.2e, max(qr): %.2e, extrema(θ): (%.3e, %.3e)",
                    qmax, qˡmax, qrmax, θmin, θmax)
    @info msg
    return nothing
end
add_callback!(simulation, progress, IterationInterval(10))

# ------------------------------------------------------------------------------
# Outputs (full fields + x–y averages, every minute)
# ------------------------------------------------------------------------------

outputs = merge(model.velocities, model.tracers, (; T, qˡ, qᵛ★))
averaged_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

filename          = string("rico_", Nx, "_", Ny, "_", Nz, ".jld2")
averages_filename = string("rico_averages_", Nx, "_", Ny, "_", Nz, ".jld2")

ow = JLD2Writer(model, outputs; filename,
                schedule = TimeInterval(1minutes),
                overwrite_existing = true)
simulation.output_writers[:jld2] = ow

avg_ow = JLD2Writer(model, averaged_outputs;
                    filename = averages_filename,
                    schedule = TimeInterval(1minutes),
                    overwrite_existing = true)
simulation.output_writers[:avg] = avg_ow

@info "Running RICO 1M on grid: \n $grid \n and using model: \n $model"
run!(simulation)

# ------------------------------------------------------------------------------
# Quick-look vertical-mean time series (unchanged)
# ------------------------------------------------------------------------------

if get(ENV, "CI", "false") == "false"
    θt  = FieldTimeSeries(averages_filename, "θ")
    Tt  = FieldTimeSeries(averages_filename, "T")
    qt  = FieldTimeSeries(averages_filename, "q")
    qrt = FieldTimeSeries(averages_filename, "qr")
    qˡt = FieldTimeSeries(averages_filename, "qˡ")
    times = qt.times
    Nt = length(θt)

    fig = Figure(size=(1200, 800), fontsize=12)
    axθ  = Axis(fig[1, 1], xlabel="θ (K)",       ylabel="z (m)")
    axq  = Axis(fig[1, 2], xlabel="q (kg/kg)",   ylabel="z (m)")
    axT  = Axis(fig[2, 1], xlabel="T (K)",       ylabel="z (m)")
    axqˡ = Axis(fig[2, 2], xlabel="qˡ (kg/kg)",  ylabel="z (m)")

    slider = Slider(fig[3, 1:2], range=1:Nt, startvalue=1)
    n  = slider.value
    θn  = @lift interior(θt[$n], 1, 1, :)
    qn  = @lift interior(qt[$n], 1, 1, :)
    Tn  = @lift interior(Tt[$n], 1, 1, :)
    qrn = @lift interior(qrt[$n], 1, 1, :)
    qˡn = @lift interior(qˡt[$n], 1, 1, :)
    z   = znodes(θt)
    title = @lift "t = $(prettytime(times[$n]))"

    fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)
    lines!(axθ,  θn, z)
    lines!(axq,  qn, z); lines!(axq, qrn, z)   # overlay qr on q panel
    lines!(axT,  Tn, z)
    lines!(axqˡ, qˡn, z)
    xlims!(axqˡ, -1e-4, 1.5e-3)
    fig

    CairoMakie.record(fig, "rico.mp4", 1:Nt, framerate=12) do nn
        @info "Drawing frame $nn of $Nt..."
        n[] = nn
    end
end
