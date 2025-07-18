# using Pkg; Pkg.activate(".")
using Oceananigans
using Oceananigans.Units
using Printf
using Breeze

arch = CPU()

# Siebesma et al (2003) resolution!
# DOI: https://doi.org/10.1175/1520-0469(2003)60<1201:ALESIS>2.0.CO;2
# Nx = Ny = 64
# Nz = 75

Nx = 128
Ny = 128
Nz = 150

Lx = 6400
Ly = 6400
Lz = 3000

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       topology = (Periodic, Periodic, Bounded))

using AtmosphericProfilesLibrary                       

FT = eltype(grid)
θ_bomex = AtmosphericProfilesLibrary.Bomex_θ_liq_ice(FT)
q_bomex = AtmosphericProfilesLibrary.Bomex_q_tot(FT)
u_bomex = AtmosphericProfilesLibrary.Bomex_u(FT)

p₀ = 101325 # Pa
θ₀ = θ_bomex(0) # K
reference_constants = Breeze.Thermodynamics.ReferenceConstants(base_pressure=p₀, potential_temperature=θ₀)
buoyancy = Breeze.MoistAirBuoyancy(; reference_constants) #, microphysics)

# Simple precipitation scheme from CloudMicrophysics    
using CloudMicrophysics 
using CloudMicrophysics.Microphysics0M: remove_precipitation

FT = eltype(grid)
microphysics = CloudMicrophysics.Parameters.Parameters0M{FT}(τ_precip=600, S_0=0, qc_0=0.02)
@inline precipitation(x, y, z, t, q, params) = remove_precipitation(params, q, 0)
Fq_precip = Forcing(precipitation, field_dependencies=:q, parameters=microphysics)

θ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(8e-3))
q_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(5.2e-5))

u★ = 0.28 # m/s
@inline u_drag(x, y, t, u, v, u★) = - u★^2 * u / sqrt(u^2 + v^2)
@inline v_drag(x, y, t, u, v, u★) = - u★^2 * v / sqrt(u^2 + v^2)
u_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(u_drag, field_dependencies=(:u, :v), parameters=u★))
v_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(v_drag, field_dependencies=(:u, :v), parameters=u★))

coriolis = FPlane(f=3.76e-5)
uᵍ = Field{Nothing, Nothing, Center}(grid)
vᵍ = Field{Nothing, Nothing, Center}(grid)
uᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_u(FT)
vᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_v(FT)
set!(uᵍ, z -> uᵍ_bomex(z))
set!(vᵍ, z -> vᵍ_bomex(z))

Fu = Field{Nothing, Nothing, Center}(grid)
Fv = Field{Nothing, Nothing, Center}(grid)

using Oceananigans.Operators: ∂zᶜᶜᶠ, ℑzᵃᵃᶜ
@inline w_dz_ϕ(i, j, k, grid, w, ϕ) = @inbounds w[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, ϕ)

@inline function u_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    u_avg = parameters.u_avg
    w_dz_U = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, u_avg)
    return - w_dz_U
end

@inline function v_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    v_avg = parameters.v_avg
    w_dz_V = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, v_avg)
    return - w_dz_V
end

@inline function θ_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    θ_avg = parameters.θ_avg
    w_dz_T = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, θ_avg)
    return - w_dz_T
end

@inline function q_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    q_avg = parameters.q_avg
    w_dz_Q = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, q_avg)
    return - w_dz_Q
end

# f for "forcing"
u_avg_f = Field{Nothing, Nothing, Center}(grid)
v_avg_f = Field{Nothing, Nothing, Center}(grid)
θ_avg_f = Field{Nothing, Nothing, Center}(grid)
q_avg_f = Field{Nothing, Nothing, Center}(grid)

wˢ = Field{Nothing, Nothing, Face}(grid)
w_bomex = AtmosphericProfilesLibrary.Bomex_subsidence(FT)
set!(wˢ, z -> w_bomex(z))

Fu_subsidence = Forcing(u_subsidence, discrete_form=true, parameters=(; u_avg=u_avg_f, wˢ))
Fv_subsidence = Forcing(v_subsidence, discrete_form=true, parameters=(; v_avg=v_avg_f, wˢ))
Fθ_subsidence = Forcing(θ_subsidence, discrete_form=true, parameters=(; θ_avg=θ_avg_f, wˢ))
Fq_subsidence = Forcing(q_subsidence, discrete_form=true, parameters=(; q_avg=q_avg_f, wˢ))

set!(Fu, z -> - coriolis.f * vᵍ_bomex(z))
set!(Fv, z -> + coriolis.f * uᵍ_bomex(z))
Fu_geostrophic = Forcing(Fu)
Fv_geostrophic = Forcing(Fv)

u_forcing = (Fu_subsidence, Fu_geostrophic)
v_forcing = (Fv_subsidence, Fv_geostrophic)

drying = Field{Nothing, Nothing, Center}(grid)
dqdt_bomex = AtmosphericProfilesLibrary.Bomex_dqtdt(FT)
set!(drying, z -> dqdt_bomex(z))
Fq_drying = Forcing(drying)
q_forcing = (Fq_precip, Fq_drying, Fq_subsidence)

Fθ_field = Field{Nothing, Nothing, Center}(grid)
dTdt_bomex = AtmosphericProfilesLibrary.Bomex_dTdt(FT)
set!(Fθ_field, z -> dTdt_bomex(1, z))
Fθ_radiation = Forcing(Fθ_field)
θ_forcing = (Fθ_radiation, Fθ_subsidence)

advection = WENO() #(momentum=WENO(), θ=WENO(), q=WENO(bounds=(0, 1)))
tracers = (:θ, :q)
model = NonhydrostaticModel(; grid, advection, buoyancy, coriolis,
                            tracers = (:θ, :q),
                            # tracers = (:θ, :q, :qˡ, :qⁱ, :qʳ, :qˢ),
                            forcing = (; q=q_forcing, u=u_forcing, v=v_forcing, θ=θ_forcing),
                            boundary_conditions = (θ=θ_bcs, q=q_bcs, u=u_bcs))

θϵ = 20
qϵ = 1e-2
θᵢ(x, y, z) = θ_bomex(z) + 1e-2 * θϵ * rand()
qᵢ(x, y, z) = q_bomex(z) + 1e-2 * qϵ * rand()
uᵢ(x, y, z) = u_bomex(z)
set!(model, θ=θᵢ, q=qᵢ, u=uᵢ)

simulation = Simulation(model, Δt=10, stop_time=6hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

# Write a callback to compute *_avg_f
u_avg = Field(Average(model.velocities.u, dims=(1, 2)))
v_avg = Field(Average(model.velocities.v, dims=(1, 2)))
θ_avg = Field(Average(model.tracers.θ, dims=(1, 2)))
q_avg = Field(Average(model.tracers.q, dims=(1, 2)))

function compute_averages!(sim)
    compute!(u_avg)
    compute!(v_avg)
    compute!(θ_avg)
    compute!(q_avg)
    parent(u_avg_f) .= parent(u_avg)
    parent(v_avg_f) .= parent(v_avg)
    parent(θ_avg_f) .= parent(θ_avg)
    parent(q_avg_f) .= parent(q_avg)
    return nothing
end

add_callback!(simulation, compute_averages!)

T = Breeze.TemperatureField(model)
qˡ = Breeze.CondensateField(model, T)
qᵛ★ = Breeze.SaturationField(model, T)
δ = Field(model.tracers.q - qᵛ★)

function progress(sim)
    compute!(T)
    compute!(qˡ)
    compute!(δ)
    q = sim.model.tracers.q
    θ = sim.model.tracers.θ
    u, v, w = sim.model.velocities

    umax = maximum(u)
    vmax = maximum(v)
    wmax = maximum(w)

    qmin = minimum(q)
    qmax = maximum(q)
    qˡmax = maximum(qˡ)
    δmax = maximum(δ)

    θmin = minimum(θ)
    θmax = maximum(θ)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, max|u|: (%.2e, %.2e, %.2e)",
                    iteration(sim), prettytime(sim), prettytime(sim.Δt), umax, vmax, wmax)

    msg *= @sprintf(", extrema(q): (%.2e, %.2e), max(qˡ): %.2e, max(δ): %.2e, extrema(θ): (%.2e, %.2e)",
                     qmin, qmax, qˡmax, δmax, θmin, θmax)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# using Oceananigans.Models: ForcingOperation
# Sʳ = ForcingOperation(:q, model)
# outputs = merge(model.velocities, model.tracers, (; T, qˡ, qᵛ★, Sʳ))
outputs = merge(model.velocities, model.tracers, (; T, qˡ, qᵛ★))

ow = JLD2Writer(model, outputs,
                filename = "bomex.jld2",
                schedule = TimeInterval(1minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

run!(simulation)

wt = FieldTimeSeries("bomex.jld2", "w")
θt = FieldTimeSeries("bomex.jld2", "θ")
Tt = FieldTimeSeries("bomex.jld2", "T")
qt = FieldTimeSeries("bomex.jld2", "q")
qˡt = FieldTimeSeries("bomex.jld2", "qˡ")
times = qt.times
Nt = length(θt)

using GLMakie, Printf

fig = Figure(size=(1200, 800), fontsize=12)
axθ = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)")
axT = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)")
axqˡ = Axis(fig[2, 2], xlabel="x (m)", ylabel="z (m)")
axw = Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)")

Nt = length(θt)
slider = Slider(fig[4, 1:2], range=1:Nt, startvalue=1)

n = slider.value #Observable(length(θt))
wn = @lift view(wt[$n], :, 1, :)
θn = @lift view(θt[$n], :, 1, :)
qn = @lift view(qt[$n], :, 1, :)
Tn = @lift view(Tt[$n], :, 1, :)
qˡn = @lift view(qˡt[$n], :, 1, :)
title = @lift "t = $(prettytime(times[$n]))"


fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

Tmin = minimum(Tt)
Tmax = maximum(Tt)
wlim = maximum(abs, wt) / 2
qlim = maximum(abs, qt)
qˡlim = maximum(abs, qˡt) / 2

Tₛ = θ_bomex(0)
Δθ = θ_bomex(Lz) - θ_bomex(0)
hmθ = heatmap!(axθ, θn, colorrange=(Tₛ, Tₛ+Δθ))
hmq = heatmap!(axq, qn, colorrange=(0, qlim), colormap=:magma)
hmT = heatmap!(axT, Tn, colorrange=(Tmin, Tmax))
hmqˡ = heatmap!(axqˡ, qˡn, colorrange=(0, qˡlim), colormap=:magma)
hmw = heatmap!(axw, wn, colorrange=(-wlim, wlim), colormap=:balance)

# Label(fig[0, 1], "θ", tellwidth=false)
# Label(fig[0, 2], "q", tellwidth=false)
# Label(fig[0, 1], "θ", tellwidth=false)
# Label(fig[0, 2], "q", tellwidth=false)

Colorbar(fig[1, 0], hmθ, label = "θ [K]", vertical=true)
Colorbar(fig[1, 3], hmq, label = "q", vertical=true)
Colorbar(fig[2, 0], hmT, label = "T [K]", vertical=true)
Colorbar(fig[2, 3], hmqˡ, label = "qˡ", vertical=true)
Colorbar(fig[3, 0], hmw, label = "w", vertical=true)

fig

record(fig, "bomex.mp4", 1:Nt, framerate=12) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end
