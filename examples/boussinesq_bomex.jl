# using Pkg; Pkg.activate(".")
using Breeze
using Oceananigans
using Oceananigans.Units

using AtmosphericProfilesLibrary
using CloudMicrophysics
using Printf

using Oceananigans.Operators: ∂zᶜᶜᶠ, ℑzᵃᵃᶜ
using CloudMicrophysics.Microphysics0M: remove_precipitation

# Siebesma et al (2003) resolution!
# DOI: https://doi.org/10.1175/1520-0469(2003)60<1201:ALESIS>2.0.CO;2
Nx = Ny = 64
Nz = 75

Lx = 6400
Ly = 6400
Lz = 3000

arch = CPU() # if changing to GPU() add `using CUDA` above
stop_time = 6hours

if get(ENV, "CI", "false") == "true" # change values for CI
    stop_time = 1minutes # 6hours
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
qᵗ_bomex = AtmosphericProfilesLibrary.Bomex_q_tot(FT)
u_bomex = AtmosphericProfilesLibrary.Bomex_u(FT)

p₀ = 101500 # Pa
θ₀ = 299.1 # K
buoyancy = Breeze.MoistAirBuoyancy(grid, base_pressure=p₀, reference_potential_temperature=θ₀)

# Simple precipitation scheme from CloudMicrophysics
FT = eltype(grid)
microphysics = CloudMicrophysics.Parameters.Parameters0M{FT}(τ_precip=600, S_0=0, qc_0=0.02)
@inline precipitation(x, y, z, t, qᵗ, params) = remove_precipitation(params, qᵗ, 0)
qᵗ_precip_forcing = Forcing(precipitation, field_dependencies=:qᵗ, parameters=microphysics)

θ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(8e-3))
qᵗ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(5.2e-5))

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

@inline function Fqᵗ_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    q_avg = parameters.q_avg
    w_dz_Q = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, q_avg)
    return - w_dz_Q
end

# f for "forcing"
u_avg_f = Field{Nothing, Nothing, Center}(grid)
v_avg_f = Field{Nothing, Nothing, Center}(grid)
θ_avg_f = Field{Nothing, Nothing, Center}(grid)
qᵗ_avg_f = Field{Nothing, Nothing, Center}(grid)

wˢ = Field{Nothing, Nothing, Face}(grid)
w_bomex = AtmosphericProfilesLibrary.Bomex_subsidence(FT)
set!(wˢ, z -> w_bomex(z))

u_subsidence_forcing = Forcing(Fu_subsidence, discrete_form=true, parameters=(; u_avg=u_avg_f, wˢ))
v_subsidence_forcing = Forcing(Fv_subsidence, discrete_form=true, parameters=(; v_avg=v_avg_f, wˢ))
θ_subsidence_forcing = Forcing(Fθ_subsidence, discrete_form=true, parameters=(; θ_avg=θ_avg_f, wˢ))
qᵗ_subsidence_forcing = Forcing(Fqᵗ_subsidence, discrete_form=true, parameters=(; qᵗ_avg=qᵗ_avg_f, wˢ))

coriolis = FPlane(f=3.76e-5)
uᵍ = Field{Nothing, Nothing, Center}(grid)
vᵍ = Field{Nothing, Nothing, Center}(grid)
uᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_u(FT)
vᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_v(FT)
set!(uᵍ, z -> uᵍ_bomex(z))
set!(vᵍ, z -> vᵍ_bomex(z))

@inline function Fu_geostrophic(i, j, k, grid, clock, fields, parameters)
    f = parameters.f
    @inbounds vᵍ = parameters.vᵍ[1, 1, k]
    return - f * vᵍ
end

@inline function Fv_geostrophic(i, j, k, grid, clock, fields, parameters)
    f = parameters.f
    @inbounds uᵍ = parameters.uᵍ[1, 1, k]
    return + f * uᵍ
end

u_geostrophic_forcing = Forcing(Fu_geostrophic, discrete_form=true, parameters=(; v_avg=v_avg_f, f=coriolis.f, vᵍ))
v_geostrophic_forcing = Forcing(Fv_geostrophic, discrete_form=true, parameters=(; u_avg=u_avg_f, f=coriolis.f, uᵍ))

u_forcing = (u_subsidence_forcing, u_geostrophic_forcing)
v_forcing = (v_subsidence_forcing, v_geostrophic_forcing)

drying = Field{Nothing, Nothing, Center}(grid)
dqᵗdt_bomex = AtmosphericProfilesLibrary.Bomex_dqtdt(FT)
set!(drying, z -> dqᵗdt_bomex(z))
qᵗ_drying_forcing = Forcing(drying)
qᵗ_forcing = (qᵗ_precip_forcing, qᵗ_drying_forcing, qᵗ_subsidence_forcing)

Fθ_field = Field{Nothing, Nothing, Center}(grid)
dTdt_bomex = AtmosphericProfilesLibrary.Bomex_dTdt(FT)
set!(Fθ_field, z -> dTdt_bomex(1, z))
θ_radiation_forcing = Forcing(Fθ_field)
θ_forcing = (θ_radiation_forcing, θ_subsidence_forcing)

model = NonhydrostaticModel(; grid, buoyancy, coriolis,
                            advection = WENO(order=5), 
                            tracers = (:θ, :qᵗ),
                            forcing = (; qᵗ=qᵗ_forcing, u=u_forcing, v=v_forcing, θ=θ_forcing),
                            boundary_conditions = (θ=θ_bcs, qᵗ=qᵗ_bcs, u=u_bcs, v=v_bcs))

# Values for the initial perturbations can be found in Appendix B
# of Siebesma et al 2003, 3rd paragraph
θϵ = 0.1
qᵗϵ = 2.5e-5
θᵢ(x, y, z) = θ_bomex(z) + θϵ * randn()
qᵢ(x, y, z) = qᵗ_bomex(z) + qᵗϵ * randn()
uᵢ(x, y, z) = u_bomex(z)
set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ)

simulation = Simulation(model; Δt=1, stop_time)
conjure_time_step_wizard!(simulation, cfl=0.7)

# Write a callback to compute *_avg_f
u_avg = Field(Average(model.velocities.u, dims=(1, 2)))
v_avg = Field(Average(model.velocities.v, dims=(1, 2)))
θ_avg = Field(Average(model.tracers.θ, dims=(1, 2)))
qᵗ_avg = Field(Average(model.tracers.qᵗ, dims=(1, 2)))

function compute_averages!(sim)
    compute!(u_avg)
    compute!(v_avg)
    compute!(θ_avg)
    compute!(q_avg)
    parent(u_avg_f) .= parent(u_avg)
    parent(v_avg_f) .= parent(v_avg)
    parent(θ_avg_f) .= parent(θ_avg)
    parent(qᵗ_avg_f) .= parent(qᵗ_avg)
    return nothing
end

add_callback!(simulation, compute_averages!)

T = Breeze.TemperatureField(model)
qˡ = Breeze.CondensateField(model, T)
qᵛ⁺ = Breeze.SaturationField(model, T)
qᵛ = model.tracers.qᵗ - qˡ
rh = Field(qᵛ / qᵛ⁺) # relative humidity

T_avg = Field(Average(T, dims=(1, 2)))
qˡ_avg = Field(Average(qˡ, dims=(1, 2)))
qᵛ⁺_avg = Field(Average(qᵛ⁺, dims=(1, 2)))
rh_avg = Field(Average(rh, dims=(1, 2)))

# Uncomment to make plots
#=
using GLMakie

fig = Figure(size=(1200, 800), fontsize=12)
axT = Axis(fig[1, 1], xlabel="T (ᵒK)", ylabel="z (m)")
axqˡ = Axis(fig[1, 2], xlabel="qˡ (kg/kg)", ylabel="z (m)")
axrh = Axis(fig[1, 3], xlabel="rh (%)", ylabel="z (m)")
axu = Axis(fig[2, 1], xlabel="u, v (m/s)", ylabel="z (m)")
axq = Axis(fig[2, 2], xlabel="q (kg/kg)", ylabel="z (m)")
axθ = Axis(fig[2, 3], xlabel="θ (ᵒK)", ylabel="z (m)")

function update_plots!(sim)
    compute!(T_avg)
    compute!(qˡ_avg)
    compute!(qᵛ⁺_avg)
    compute!(rh_avg)
    compute!(qᵗ_avg)
    compute!(θ_avg)
    compute!(u_avg)
    compute!(v_avg)

    lines!(axT, T_avg)
    lines!(axqˡ, qˡ_avg)
    lines!(axrh, rh_avg)
    lines!(axu, u_avg)
    lines!(axu, v_avg)
    lines!(axq, qᵗ_avg)
    lines!(axq, qᵛ⁺_avg)
    lines!(axθ, θ_avg)
    display(fig)
    return nothing
end

add_callback!(simulation, update_plots!, TimeInterval(20minutes))
=#

function progress(sim)
    compute!(T)
    compute!(qˡ)
    qˡmax = maximum(qˡ)

    compute!(rh)
    rhmax = maximum(rh)

    umax = maximum(abs, u_avg)
    vmax = maximum(abs, v_avg)

    qᵗ = sim.model.tracers.qᵗ
    qᵗmax = maximum(qᵗ)

    θ = sim.model.tracers.θ
    θmin = minimum(θ)
    θmax = maximum(θ)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, max|ū|: (%.2e, %.2e), max(rh): %.2f",
                    iteration(sim), prettytime(sim), prettytime(sim.Δt), umax, vmax, rhmax)

    msg *= @sprintf(", max(qᵗ): %.2e, max(qˡ): %.2e, extrema(θ): (%.3e, %.3e)",
                     qᵗmax, qˡmax, θmin, θmax)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# The commented out lines below diagnose the forcing applied to model.tracers.q
# using Oceananigans.Models: ForcingOperation
# Sʳ = ForcingOperation(:q, model)
outputs = merge(model.velocities, model.tracers, (; T, qˡ, qᵛ⁺))
averaged_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

filename = string("bomex_", Nx, "_", Ny, "_", Nz, ".jld2")
averages_filename = string("bomex_averages_", Nx, "_", Ny, "_", Nz, ".jld2")

ow = JLD2Writer(model, outputs; filename,
                schedule = TimeInterval(1minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

averages_ow = JLD2Writer(model, averaged_outputs;
                         filename = averages_filename,
                         schedule = TimeInterval(1minutes),
                         overwrite_existing = true)

simulation.output_writers[:avg] = averages_ow

@info "Running BOMEX on grid: \n $grid \n and using model: \n $model"
run!(simulation)

using CairoMakie

if get(ENV, "CI", "false") == "false"
    θt  = FieldTimeSeries(averages_filename, "θ")
    Tt  = FieldTimeSeries(averages_filename, "T")
    qᵗt  = FieldTimeSeries(averages_filename, "qᵗ")
    qˡt = FieldTimeSeries(averages_filename, "qˡ")
    times = qᵗt.times
    Nt = length(θt)

    fig = Figure(size=(1200, 800), fontsize=12)
    axθ  = Axis(fig[1, 1], xlabel="θ (ᵒK)", ylabel="z (m)")
    axqᵗ = Axis(fig[1, 2], xlabel="qᵗ (kg/kg)", ylabel="z (m)")
    axT  = Axis(fig[2, 1], xlabel="T (ᵒK)", ylabel="z (m)")
    axqˡ = Axis(fig[2, 2], xlabel="qˡ (kg/kg)", ylabel="z (m)")

    Nt = length(θt)
    slider = Slider(fig[3, 1:2], range=1:Nt, startvalue=1)

    n = slider.value #Observable(length(θt))
    θn  = @lift interior(θt[$n], 1, 1, :)
    Tn  = @lift interior(Tt[$n], 1, 1, :)
    qᵗn = @lift interior(qᵗt[$n], 1, 1, :)
    qˡn = @lift interior(qˡt[$n], 1, 1, :)
    z = znodes(θt)
    title = @lift "t = $(prettytime(times[$n]))"

    fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

    hmθ  = lines!(axθ, θn, z)
    hmT  = lines!(axT, Tn, z)
    hmqᵗ = lines!(axqᵗ, qᵗn, z)
    hmqˡ = lines!(axqˡ, qˡn, z)
    xlims!(axqˡ, -1e-4, 1.5e-3)

    fig

    CairoMakie.record(fig, "bomex.mp4", 1:Nt, framerate=12) do nn
        @info "Drawing frame $nn of $Nt..."
        n[] = nn
    end
end
