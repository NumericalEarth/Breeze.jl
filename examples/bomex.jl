# using Pkg; Pkg.activate(".")
using Breeze
using Oceananigans
using Oceananigans.Units

using AtmosphericProfilesLibrary                       
using CloudMicrophysics 
using Printf
using CairoMakie

using Oceananigans.Operators: ∂zᶜᶜᶠ, ℑzᵃᵃᶜ
using CloudMicrophysics.Microphysics0M: remove_precipitation

# Siebesma et al (2003) resolution!
# DOI: https://doi.org/10.1175/1520-0469(2003)60<1201:ALESIS>2.0.CO;2
Nx = Ny = 32 #64
Nz = 35 #75

Lx = 6400
Ly = 6400
Lz = 3000

arch = CPU() # try changing to GPU()
stop_time = 6hours

if get(ENV, "CI", "false") == "true" # change values for CI
    stop_time = 10minutes # 6hours
    Nx = Ny = 8
end

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

FT = eltype(grid)
θ_bomex = AtmosphericProfilesLibrary.Bomex_θ_liq_ice(FT)
q_bomex = AtmosphericProfilesLibrary.Bomex_q_tot(FT)
u_bomex = AtmosphericProfilesLibrary.Bomex_u(FT)

p₀ = 101325 # Pa
θ₀ = 299.1 # K
reference_constants = Breeze.Thermodynamics.ReferenceConstants(base_pressure=p₀, potential_temperature=θ₀)
buoyancy = Breeze.MoistAirBuoyancy(; reference_constants) #, microphysics)

# Simple precipitation scheme from CloudMicrophysics    
FT = eltype(grid)
microphysics = CloudMicrophysics.Parameters.Parameters0M{FT}(τ_precip=600, S_0=0, qc_0=0.02)
@inline precipitation(x, y, z, t, q, params) = remove_precipitation(params, q, 0)
q_precip_forcing = Forcing(precipitation, field_dependencies=:q, parameters=microphysics)

θ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(8e-3))
q_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(5.2e-5))

u★ = 0.28 # m/s
@inline u_drag(x, y, t, u, v, u★) = - u★^2 * u / sqrt(u^2 + v^2)
@inline v_drag(x, y, t, u, v, u★) = - u★^2 * v / sqrt(u^2 + v^2)
u_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(u_drag, field_dependencies=(:u, :v), parameters=u★))
v_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(v_drag, field_dependencies=(:u, :v), parameters=u★))

@inline w_dz_ϕ(i, j, k, grid, w, ϕ) = @inbounds w[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, ϕ)

@inline function Fu_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    u_avg = parameters.u_avg
    w_dz_U = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, u_avg)
    return - w_dz_U
end

@inline function Fv_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    v_avg = parameters.v_avg
    w_dz_V = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, v_avg)
    return - w_dz_V
end

@inline function Fθ_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    θ_avg = parameters.θ_avg
    w_dz_T = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, θ_avg)
    return - w_dz_T
end

@inline function Fq_subsidence(i, j, k, grid, clock, fields, parameters)
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

u_subsidence_forcing = Forcing(Fu_subsidence, discrete_form=true, parameters=(; u_avg=u_avg_f, wˢ))
v_subsidence_forcing = Forcing(Fv_subsidence, discrete_form=true, parameters=(; v_avg=v_avg_f, wˢ))
θ_subsidence_forcing = Forcing(Fθ_subsidence, discrete_form=true, parameters=(; θ_avg=θ_avg_f, wˢ))
q_subsidence_forcing = Forcing(Fq_subsidence, discrete_form=true, parameters=(; q_avg=q_avg_f, wˢ))

coriolis = FPlane(f=3.76e-5)
uᵍ = Field{Nothing, Nothing, Center}(grid)
vᵍ = Field{Nothing, Nothing, Center}(grid)
uᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_u(FT)
vᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_v(FT)
set!(uᵍ, z -> uᵍ_bomex(z))
set!(vᵍ, z -> vᵍ_bomex(z))

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

u_geostrophic_forcing = Forcing(Fu_geostrophic, discrete_form=true, parameters=(; v_avg=v_avg_f, f=coriolis.f, vᵍ))
v_geostrophic_forcing = Forcing(Fv_geostrophic, discrete_form=true, parameters=(; u_avg=u_avg_f, f=coriolis.f, uᵍ))

u_forcing = (u_subsidence_forcing, u_geostrophic_forcing)
v_forcing = (v_subsidence_forcing, v_geostrophic_forcing)

drying = Field{Nothing, Nothing, Center}(grid)
dqdt_bomex = AtmosphericProfilesLibrary.Bomex_dqtdt(FT)
set!(drying, z -> dqdt_bomex(z))
q_drying_forcing = Forcing(drying)
q_forcing = (q_precip_forcing, q_drying_forcing, q_subsidence_forcing)

Fθ_field = Field{Nothing, Nothing, Center}(grid)
dTdt_bomex = AtmosphericProfilesLibrary.Bomex_dTdt(FT)
set!(Fθ_field, z -> dTdt_bomex(1, z))
θ_radiation_forcing = Forcing(Fθ_field)
θ_forcing = (θ_radiation_forcing, θ_subsidence_forcing)

# Note that most of the models that participated in Siebesma et al 2003
# use 2nd order advection together with either TKE or Smag-Lilly closure.
advection = WENO(order=9)
closure = nothing
# advection = Centered(order=2)
# closure = SmagorinskyLilly()

model = NonhydrostaticModel(; grid, advection, buoyancy, coriolis, closure,
                            tracers = (:θ, :q),
                            forcing = (; q=q_forcing, u=u_forcing, v=v_forcing, θ=θ_forcing),
                            boundary_conditions = (θ=θ_bcs, q=q_bcs, u=u_bcs, v=v_bcs))

# Values for the initial perturbations can be found in Appendix B
# of Siebesma et al 2003, 3rd paragraph
θϵ = 0.1
qϵ = 2.5e-5
θᵢ(x, y, z) = θ_bomex(z) + θϵ * randn()
qᵢ(x, y, z) = q_bomex(z) + qϵ * randn()
uᵢ(x, y, z) = u_bomex(z)
set!(model, θ=θᵢ, q=qᵢ, u=uᵢ)

#=
fig = Figure()
axθ = Axis(fig[1, 1])
axq = Axis(fig[1, 2])
axu = Axis(fig[1, 3])
θavg = Field(Average(model.tracers.θ, dims=(1, 2)))
qavg = Field(Average(model.tracers.q, dims=(1, 2)))
uavg = Field(Average(model.velocities.u, dims=(1, 2)))
lines!(axθ, θavg)
lines!(axq, qavg)
lines!(axu, uavg)
xlims!(axθ, 298, 310)
xlims!(axq, 2e-3, 18e-3)
ylims!(axθ, 0, 2500)
ylims!(axq, 0, 2500)
ylims!(axu, 0, 2500)
=#

simulation = Simulation(model; Δt=10, stop_time)
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
rh = Field(model.tracers.q / qᵛ★) # relative humidity

fig = Figure()
axu = Axis(fig[1, 1])
axv = Axis(fig[1, 2])
axθ = Axis(fig[1, 3])
axq = Axis(fig[1, 4])

function progress(sim)
    compute!(T)
    compute!(qˡ)
    compute!(rh)

    q = sim.model.tracers.q
    θ = sim.model.tracers.θ
    u, v, w = sim.model.velocities

    umax = maximum(abs, u_avg)
    vmax = maximum(abs, v_avg)

    qmin = minimum(q)
    qmax = maximum(q)
    qˡmax = maximum(qˡ)
    rhmax = maximum(rh)

    θmin = minimum(θ)
    θmax = maximum(θ)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, max|ū|: (%.2e, %.2e), max(rh): %.2f",
                    iteration(sim), prettytime(sim), prettytime(sim.Δt), umax, vmax, rhmax)

    msg *= @sprintf(", max(q): %.2e, max(qˡ): %.2e, extrema(θ): (%.3e, %.3e)",
                     qmax, qˡmax, θmin, θmax)

    @info msg

    lines!(axu, u_avg)
    lines!(axv, v_avg)
    lines!(axθ, θ_avg)
    lines!(axq, q_avg)
    display(fig)

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# The commented out lines below diagnose the forcing applied to model.tracers.q
# using Oceananigans.Models: ForcingOperation
# Sʳ = ForcingOperation(:q, model)
# outputs = merge(model.velocities, model.tracers, (; T, qˡ, qᵛ★, Sʳ))
outputs = merge(model.velocities, model.tracers, (; T, qˡ, qᵛ★))

filename = string("bomex_", Nx, "_", Ny, "_", Nz, ".jld2")

ow = JLD2Writer(model, outputs; filename,
                schedule = TimeInterval(1minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

@info "Running BOMEX on grid: \n $grid \n and using model: \n $model"
run!(simulation)

if get(ENV, "CI", "false") == "false" # change values for CI
    wt = FieldTimeSeries("bomex.jld2", "w")
    θt = FieldTimeSeries("bomex.jld2", "θ")
    Tt = FieldTimeSeries("bomex.jld2", "T")
    qt = FieldTimeSeries("bomex.jld2", "q")
    qˡt = FieldTimeSeries("bomex.jld2", "qˡ")
    times = qt.times
    Nt = length(θt)

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

    CairoMakie.record(fig, "bomex.mp4", 1:Nt, framerate=12) do nn
        @info "Drawing frame $nn of $Nt..."
        n[] = nn
    end
end

