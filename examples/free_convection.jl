using Oceananigans
using Oceananigans.Units
using Printf
using AquaSkyLES

arch = GPU()

Nx = Nz = 128
Lz = 4 * 1024
grid = RectilinearGrid(arch, size=(Nx, Nz), x=(0, 2Lz), z=(0, Lz), topology=(Periodic, Flat, Bounded))

p₀ = 101325 # Pa
θ₀ = 288 # K
reference_state = AquaSkyLES.ReferenceState(base_pressure=p₀, potential_temperature=θ₀)
buoyancy = AquaSkyLES.MoistAirBuoyancy(; reference_state) #, microphysics)

# Simple precipitation scheme from CloudMicrophysics
using CloudMicrophysics
using CloudMicrophysics.Microphysics0M: remove_precipitation

FT = eltype(grid)
microphysics = CloudMicrophysics.Parameters.Parameters0M{FT}(τ_precip=600, S_0=0, qc_0=0.02)
@inline precipitation(x, z, t, q, params) = remove_precipitation(params, q, 0)
q_forcing = Forcing(precipitation, field_dependencies=:q, parameters=microphysics)

ρ₀ = AquaSkyLES.base_density(buoyancy) # air density at z=0
cₚ = buoyancy.thermodynamics.dry_air.heat_capacity
Q₀ = 1000 # heat flux in W / m²
Jθ = Q₀ / (ρ₀ * cₚ) # temperature flux
θ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(Jθ))

vapor_flux = FluxBoundaryCondition(1e-2)
q_bcs = FieldBoundaryConditions(bottom=vapor_flux)

advection = WENO() #(momentum=WENO(), θ=WENO(), q=WENO(bounds=(0, 1)))
tracers = (:θ, :q)
model = NonhydrostaticModel(; grid, buoyancy,
                            tracers = (:θ, :q))
                            ,
                            forcing = (; q=q_forcing),
                            boundary_conditions = (θ=θ_bcs, q=q_bcs))

Lz = grid.Lz
Δθ = 5 # K
Tₛ = reference_state.θ # K
θᵢ(x, z) = Tₛ + Δθ * z / Lz + 1e-2 * Δθ * randn()
qᵢ(x, z) = 0 # 1e-2 + 1e-5 * rand()
set!(model, θ=θᵢ, q=qᵢ)

simulation = Simulation(model, Δt=10, stop_time=4hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

T = AquaSkyLES.TemperatureField(model)
qˡ = AquaSkyLES.CondensateField(model, T)
qᵛ★ = AquaSkyLES.SaturationField(model, T)
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

    msg *= @sprintf(", extrema(q): (%.2e, %.2e), max(qˡ): %.2e, min(δ): %.2e, extrema(θ): (%.2e, %.2e)",
                     qmin, qmax, qˡmax, δmax, θmin, θmax)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

using Oceananigans.Models: ForcingOperation
Sʳ = ForcingOperation(:q, model)
outputs = merge(model.velocities, model.tracers, (; T, qˡ, qᵛ★, Sʳ))

ow = JLD2Writer(model, outputs,
                filename = "free_convection.jld2",
                schedule = TimeInterval(1minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

run!(simulation)

wt = FieldTimeSeries("free_convection.jld2", "θ")
θt = FieldTimeSeries("free_convection.jld2", "θ")
Tt = FieldTimeSeries("free_convection.jld2", "T")
qt = FieldTimeSeries("free_convection.jld2", "q")
qˡt = FieldTimeSeries("free_convection.jld2", "qˡ")
Sʳt = FieldTimeSeries("free_convection.jld2", "Sʳ")
times = qt.times
Nt = length(θt)

using GLMakie, Printf

n = Observable(length(θt))

wn = @lift θt[$n]
θn = @lift θt[$n]
qn = @lift qt[$n]
Tn = @lift Tt[$n]
qˡn = @lift qˡt[$n]
Sʳn = @lift Sʳt[$n]
title = @lift "t = $(prettytime(times[$n]))"

fig = Figure(size=(1200, 800), fontsize=12)
axθ = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)")
axT = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)")
axqˡ = Axis(fig[2, 2], xlabel="x (m)", ylabel="z (m)")
axw = Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)")
axS = Axis(fig[3, 2], xlabel="x (m)", ylabel="z (m)")

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

Tmin = minimum(Tt)
Tmax = maximum(Tt)
wlim = maximum(abs, wt) / 2
qlim = maximum(abs, qt)
qˡlim = maximum(abs, qˡt) / 2

hmθ = heatmap!(axθ, θn, colorrange=(Tₛ, Tₛ+Δθ))
hmq = heatmap!(axq, qn, colorrange=(0, qlim), colormap=:magma)
hmT = heatmap!(axT, Tn, colorrange=(Tmin, Tmax))
hmqˡ = heatmap!(axqˡ, qˡn, colorrange=(0, qˡlim), colormap=:magma)
hmw = heatmap!(axw, wn, colorrange=(-wlim, wlim), colormap=:balance)
hmS = heatmap!(axS, Sʳn, colorrange=(-1e-4, 0), colormap=:grays)

# Label(fig[0, 1], "θ", tellwidth=false)
# Label(fig[0, 2], "q", tellwidth=false)
# Label(fig[0, 1], "θ", tellwidth=false)
# Label(fig[0, 2], "q", tellwidth=false)

Colorbar(fig[1, 0], hmθ, label = "θ [K]", vertical=true)
Colorbar(fig[1, 3], hmq, label = "q", vertical=true)
Colorbar(fig[2, 0], hmT, label = "T [K]", vertical=true)
Colorbar(fig[2, 3], hmqˡ, label = "qˡ", vertical=true)
Colorbar(fig[3, 0], hmw, label = "w", vertical=true)
Colorbar(fig[3, 3], hmS, label = "Sʳ", vertical=true)

fig

record(fig, "free_convection.mp4", 1:Nt, framerate=12) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end
