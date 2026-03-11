# Quick test of tropical_cyclone_with_rainband.jl
# Uses CPU, small domain, short run

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Breeze: WENO, DCMIP2016KesslerMicrophysics, TetensFormula
using CairoMakie
using Printf
using Random
using CSV
using DataFrames

Random.seed!(42)

const use_yu_didlake_2019 = false

Oceananigans.defaults.FloatType = Float32

arch = CPU()
Nx = Ny = 20
Nz = 32

x = y = (0, 200_000.0)
z = (0, 20500.0)

grid = RectilinearGrid(arch; x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

println("\n=== Loading sounding ===")
sounding   = CSV.read("dunion2011_moist_tropical_MT.csv", DataFrame)
Tˢ_data    = 273.15 .+ Float64.(sounding[:, :Temperature_C])
pˢ_data    = 100.0  .* Float64.(sounding[:, :Pressure_hPa])
zˢ_data    = Float64.(sounding[:, :GPH_m])
θˢ_data    = Float64.(sounding[:, :Theta_K])
qᵗˢ_gkg    = Float64.(sounding[:, :Mixing_ratio_g_kg])
qᵗˢ_data   = qᵗˢ_gkg ./ 1000.0

Tˢ_data    = reverse(Tˢ_data)
pˢ_data    = reverse(pˢ_data)
zˢ_data    = reverse(zˢ_data)
θˢ_data    = reverse(θˢ_data)
qᵗˢ_data   = reverse(qᵗˢ_data)
zˢ_data[1] = 0.0


function make_column_interp(zs::AbstractVector, vs::AbstractVector)
    function interp(z)
        z = clamp(z, zs[1], zs[end])
        i = clamp(searchsortedfirst(zs, z), 2, length(zs))
        t = (z - zs[i-1]) / (zs[i] - zs[i-1])
        return vs[i-1] + t * (vs[i] - vs[i-1])
    end
    return interp
end

θ_sounding_interp  = make_column_interp(zˢ_data, θˢ_data)
qᵗ_sounding_interp = make_column_interp(zˢ_data, qᵗˢ_data)
T_sounding_interp  = make_column_interp(zˢ_data, Tˢ_data)
p_sounding_interp  = make_column_interp(zˢ_data, pˢ_data)

reference_state = ReferenceState(grid, constants,
                                 surface_pressure     = pˢ_data[1],
                                 potential_temperature = θˢ_data[1])
dynamics = AnelasticDynamics(reference_state)

Cᴰ = 1.229e-3; Cᵀ = 1.094e-3; Cᵛ = 1.133e-3; T₀ = 300.0
ρe_bcs  = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=T₀))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(coefficient=Cᵛ, surface_temperature=T₀))
ρu_bcs  = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs  = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
boundary_conditions = (ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)

sponge = Relaxation(rate=1/300.0, mask=GaussianMask{:z}(center=19_000.0, width=1_500.0))
coriolis = FPlane(f=5e-5)

x_center = Float64(x[1] + x[2]) / 2
y_center = Float64(y[1] + y[2]) / 2

Rᵈ    = constants.molar_gas_constant / constants.dry_air.molar_mass
T_avg = sum(Tˢ_data) / length(Tˢ_data)
ρ_sfc = pˢ_data[1] / (Rᵈ * Tˢ_data[1])
H_ρ   = Rᵈ * T_avg / constants.gravitational_acceleration
r_rb  = 70_000.0

Q_con_max = 3.0/3600.0; σ_rc=2_000.0; z_bc=0.0; σ_zc=7_000.0
Q_str_max = 1.5/3600.0; σ_rs=6_000.0; z_bs=4_000.0; σ_zs=2_000.0
ϕ_rb = Float32(-π/4); σ_ϕ = Float32(π/4)

con_params = (xc=Float32(x_center), yc=Float32(y_center), Q_max=Float32(Q_con_max),
              r_rb=Float32(r_rb), σ_r=Float32(σ_rc), z_bot=Float32(z_bc),
              z_top=Float32(z_bc+σ_zc), ϕ_rb, σ_ϕ, ρ_sfc=Float32(ρ_sfc), H_ρ=Float32(H_ρ))
str_params = (xc=Float32(x_center), yc=Float32(y_center), Q_max=Float32(Q_str_max),
              r_rb=Float32(r_rb), σ_r=Float32(σ_rs), z_bs=Float32(z_bs),
              σ_zs=Float32(σ_zs), ϕ_rb, σ_ϕ, ρ_sfc=Float32(ρ_sfc), H_ρ=Float32(H_ρ))

@inline function convective_rainband_heating(x, y, z, t, p)
    r = sqrt((x - p.xc)^2 + (y - p.yc)^2)
    ϕ = atan(y - p.yc, x - p.xc)
    ρ = p.ρ_sfc * exp(-z / p.H_ρ)
    azimuthal  = exp(-((ϕ - p.ϕ_rb) / p.σ_ϕ)^8)
    radial     = exp(-((r - p.r_rb)  / p.σ_r)^2)
    in_layer   = (z > p.z_bot) & (z < p.z_top)
    sinusoidal = sin(oftype(z, π) * (z - p.z_bot) / (p.z_top - p.z_bot))
    vertical   = ifelse(in_layer, sinusoidal, zero(z))
    return ρ * p.Q_max * radial * azimuthal * vertical
end

@inline function stratiform_rainband_heating(x, y, z, t, p)
    r = sqrt((x - p.xc)^2 + (y - p.yc)^2)
    ϕ = atan(y - p.yc, x - p.xc)
    ρ = p.ρ_sfc * exp(-z / p.H_ρ)
    azimuthal  = exp(-((ϕ - p.ϕ_rb) / p.σ_ϕ)^8)
    radial     = exp(-((r - p.r_rb)  / p.σ_r)^2)
    in_layer   = (z > p.z_bs - p.σ_zs) & (z < p.z_bs + p.σ_zs)
    sinusoidal = sin(oftype(z, π) * (z - p.z_bs) / p.σ_zs)
    vertical   = ifelse(in_layer, sinusoidal, zero(z))
    return ρ * p.Q_max * radial * azimuthal * vertical
end

convective_forcing = Forcing(convective_rainband_heating, parameters=con_params)
stratiform_forcing = Forcing(stratiform_rainband_heating, parameters=str_params)
forcing = (ρθ = (convective_forcing, stratiform_forcing), ρw = sponge)

println("=== Creating model ===")
microphysics = DCMIP2016KesslerMicrophysics()
weno = WENO(order=5); bpweno = WENO(order=5, bounds=(0,1))
scalar_advection = (ρθ=weno, ρqᵗ=bpweno, ρqᶜˡ=bpweno, ρqʳ=bpweno)
model = AtmosphereModel(grid; dynamics, coriolis, microphysics,
                        momentum_advection=weno, scalar_advection, forcing, boundary_conditions,
                        thermodynamic_constants=constants)
println("  Model created OK")

# Vortex ICs
RMW=31_000.0; V_RMW=43.0; a=0.5
z_nodes_cpu = Array(znodes(grid, Center()))
Δz_step = z_nodes_cpu[2] - z_nodes_cpu[1]
T_out_K = T_sounding_interp(16_000.0)
rmw_profile = zeros(Float64, Nz)
rmw_profile[1] = RMW
for k in 2:Nz
    z_k = z_nodes_cpu[k]
    z_lo = clamp(z_k-Δz_step/2, zˢ_data[1], zˢ_data[end])
    z_hi = clamp(z_k+Δz_step/2, zˢ_data[1], zˢ_data[end])
    dTdZ = (T_sounding_interp(z_hi) - T_sounding_interp(z_lo)) / Δz_step
    T_k  = T_sounding_interp(clamp(z_k, zˢ_data[1], zˢ_data[end]))
    denom = 2.0*(T_k - T_out_K)
    drdZ  = abs(denom) > 1.0 ? -rmw_profile[k-1]/denom*dTdZ : 0.0
    rmw_profile[k] = max(rmw_profile[k-1]+drdZ*Δz_step, RMW)
end
function rmw_at_height(z)
    k = clamp(searchsortedfirst(z_nodes_cpu, z), 1, Nz)
    return rmw_profile[k]
end
function tangential_wind(x, y, z)
    r = sqrt((x-x_center)^2 + (y-y_center)^2)
    rmw_z = rmw_at_height(z)
    v_adj = RMW/rmw_z
    z >= 16_000.0 && return zero(typeof(r))
    r <= rmw_z ? V_RMW*v_adj*r/rmw_z : V_RMW*v_adj*(rmw_z/r)^a
end

∂r_int=1_000.0; max_radius=500_000.0  # Reduced for test
rrange = collect(0.0:∂r_int:max_radius)
Nr = length(rrange)
p_vortex = zeros(Float64, Nz, Nr)
println("  Computing pressure field...")
for k in 1:Nz
    z_k = z_nodes_cpu[k]
    z_c = clamp(z_k, zˢ_data[1], zˢ_data[end])
    T_k = T_sounding_interp(z_c)
    p_bg = pˢ_data[1]*exp(-constants.gravitational_acceleration*z_k/(Rᵈ*T_k))
    ρ_k  = p_bg/(Rᵈ*T_k)
    p_vortex[k, Nr] = p_bg
    for r_idx in (Nr-1):-1:1
        r = rrange[r_idx]
        v_tang = tangential_wind(x_center+r, y_center, z_k)
        dp_dr  = ρ_k*(v_tang*coriolis.f + v_tang^2/max(r,1.0))
        p_vortex[k, r_idx] = p_vortex[k, r_idx+1] - dp_dr*∂r_int
    end
end
p_outer = p_vortex[:, Nr]
function p_at(x, y, z)
    r = sqrt((x-x_center)^2 + (y-y_center)^2)
    k = clamp(searchsortedfirst(z_nodes_cpu, z), 1, Nz)
    r_idx = clamp(searchsortedfirst(rrange, r), 1, Nr)
    return p_vortex[k, r_idx]
end

function u_init(x, y, z)
    ϕ = atan(y-y_center, x-x_center)
    return -sin(ϕ)*tangential_wind(x,y,z)
end
function v_init(x, y, z)
    ϕ = atan(y-y_center, x-x_center)
    return cos(ϕ)*tangential_wind(x,y,z)
end
function θ_init(x, y, z)
    z_c = clamp(z, zˢ_data[1], zˢ_data[end])
    θ_bg = θ_sounding_interp(z_c)
    k = clamp(searchsortedfirst(z_nodes_cpu, z_c), 1, Nz)
    p_ref = p_outer[k]
    p_loc = p_at(x, y, z)
    return θ_bg*(p_ref/p_loc)
end
function qᵗ_init(x, y, z)
    z_c = clamp(z, zˢ_data[1], zˢ_data[end])
    return qᵗ_sounding_interp(z_c)
end

println("  Setting ICs...")
set!(model, θ=θ_init, qᵗ=qᵗ_init, u=u_init, v=v_init)
println("  ICs set OK")

# Quick checks
u_arr = Array(interior(model.velocities.u))
println("  max|u| = $(maximum(abs.(u_arr))) m/s  (expected ~43 m/s)")
qᵗ_arr = Array(interior(model.specific_moisture))
println("  max qᵗ = $(maximum(qᵗ_arr)) kg/kg  (expected ~0.019)")
println("  min qᵗ = $(minimum(qᵗ_arr)) kg/kg  (expected ~0.0)")

# Run for 1 time step to check no errors
println("\n=== Running 1 time step ===")
time_step!(model, 1.0)
println("  Time step OK; model time = $(model.clock.time) s")

# Run for 3 minutes
println("\n=== Running 3-minute test ===")
simulation = Simulation(model; Δt=2, stop_time=3minutes)
conjure_time_step_wizard!(simulation, cfl=0.7)

u, v, w = model.velocities
function prog(sim)
    @info @sprintf("Iter: %d, t: %s, Δt: %s, max|u|: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, u))
end
add_callback!(simulation, prog, IterationInterval(10))
run!(simulation)
println("\n=== TEST PASSED ===")
