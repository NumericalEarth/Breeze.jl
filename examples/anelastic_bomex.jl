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

p₀ = 101500 # Pa
θ₀ = 299.1 # K
thermo = ThermodynamicConstants()
reference_state = ReferenceState(grid, thermo, base_pressure=p₀, potential_temperature=θ₀)
formulation = AnelasticFormulation(reference_state)

# Simple precipitation scheme from CloudMicrophysics
FT = eltype(grid)
microphysics = CloudMicrophysics.Parameters.Parameters0M{FT}(τ_precip=600, S_0=0, qc_0=0.02)
@inline precipitation(x, y, z, t, qᵗ, params) = remove_precipitation(params, qᵗ, 0)
q_precip_forcing = Forcing(precipitation, field_dependencies=:qᵗ, parameters=microphysics)

FT = eltype(grid)
q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
ρ₀ = Breeze.Thermodynamics.density(p₀, θ₀, q₀, thermo)
cᵖᵈ = thermo.dry_air.heat_capacity
ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * cᵖᵈ * 8e-3))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * 5.2e-5))

u★ = 0.28 # m/s
@inline ρu_drag(x, y, t, ρu, ρv, u★) = - u★^2 * ρu / sqrt(ρu^2 + ρv^2)
@inline ρv_drag(x, y, t, ρu, ρv, u★) = - u★^2 * ρv / sqrt(ρu^2 + ρv^2)
ρu_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=u★))
ρv_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=u★))

@inline w_dz_ϕ(i, j, k, grid, w, ϕ) = @inbounds w[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, ϕ)

@inline function Fρu_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    u_avg = parameters.u_avg
    w_dz_U = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, u_avg)
    ρᵣ = @inbounds parameters.ρᵣ[i, j, k]
    return - ρᵣ * w_dz_U
end

@inline function Fρv_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    v_avg = parameters.v_avg
    w_dz_V = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, v_avg)
    ρᵣ = @inbounds parameters.ρᵣ[i, j, k]
    return - ρᵣ * w_dz_V
end

@inline function Fρe_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    e_avg = parameters.e_avg
    w_dz_T = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, e_avg)
    ρᵣ = @inbounds parameters.ρᵣ[i, j, k]
    return - ρᵣ * w_dz_T
end

@inline function Fρqᵗ_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    q_avg = parameters.qᵗ_avg
    w_dz_Q = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, q_avg)
    ρᵣ = @inbounds parameters.ρᵣ[i, j, k]
    return - ρᵣ * w_dz_Q
end

# f for "forcing"
u_avg_f = Field{Nothing, Nothing, Center}(grid)
v_avg_f = Field{Nothing, Nothing, Center}(grid)
e_avg_f = Field{Nothing, Nothing, Center}(grid)
qᵗ_avg_f = Field{Nothing, Nothing, Center}(grid)

wˢ = Field{Nothing, Nothing, Face}(grid)
w_bomex = AtmosphericProfilesLibrary.Bomex_subsidence(FT)
set!(wˢ, z -> w_bomex(z))

ρᵣ = formulation.reference_state.density
ρu_subsidence_forcing = Forcing(Fρu_subsidence, discrete_form=true, parameters=(; u_avg=u_avg_f, wˢ, ρᵣ))
ρv_subsidence_forcing = Forcing(Fρv_subsidence, discrete_form=true, parameters=(; v_avg=v_avg_f, wˢ, ρᵣ))
ρe_subsidence_forcing = Forcing(Fρe_subsidence, discrete_form=true, parameters=(; e_avg=e_avg_f, wˢ, ρᵣ))
ρqᵗ_subsidence_forcing = Forcing(Fρqᵗ_subsidence, discrete_form=true, parameters=(; qᵗ_avg=qᵗ_avg_f, wˢ, ρᵣ))

coriolis = FPlane(f=3.76e-5)
ρuᵍ = Field{Nothing, Nothing, Center}(grid)
ρvᵍ = Field{Nothing, Nothing, Center}(grid)
uᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_u(FT)
vᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_v(FT)
set!(ρuᵍ, z -> uᵍ_bomex(z))
set!(ρvᵍ, z -> vᵍ_bomex(z))
set!(ρuᵍ, ρᵣ * ρuᵍ)
set!(ρvᵍ, ρᵣ * ρvᵍ)

@inline function Fρu_geostrophic(i, j, k, grid, clock, fields, parameters)
    f = parameters.f
    @inbounds ρvᵍᵢ = parameters.ρvᵍ[1, 1, k]
    return - f * ρvᵍᵢ
end

@inline function Fρv_geostrophic(i, j, k, grid, clock, fields, parameters)
    f = parameters.f
    @inbounds ρuᵍᵢ = parameters.ρuᵍ[1, 1, k]
    return + f * ρuᵍᵢ
end

ρu_geostrophic_forcing = Forcing(Fρu_geostrophic, discrete_form=true, parameters=(; f=coriolis.f, ρvᵍ))
ρv_geostrophic_forcing = Forcing(Fρv_geostrophic, discrete_form=true, parameters=(; f=coriolis.f, ρuᵍ))

ρu_forcing = (ρu_subsidence_forcing, ρu_geostrophic_forcing)
ρv_forcing = (ρv_subsidence_forcing, ρv_geostrophic_forcing)

drying = Field{Nothing, Nothing, Center}(grid)
dqdt_bomex = AtmosphericProfilesLibrary.Bomex_dqtdt(FT)
set!(drying, z -> dqdt_bomex(z))
ρᵣ = formulation.reference_state.density
set!(drying, ρᵣ * drying)
ρqᵗ_drying_forcing = Forcing(drying)
ρqᵗ_forcing = (ρqᵗ_drying_forcing, ρqᵗ_subsidence_forcing)
# q_forcing = (q_precip_forcing, q_drying_forcing, q_subsidence_forcing)

Fρe_field = Field{Nothing, Nothing, Center}(grid)
dTdt_bomex = AtmosphericProfilesLibrary.Bomex_dTdt(FT)
set!(Fρe_field, z -> dTdt_bomex(1, z))
set!(Fρe_field, ρᵣ * Fρe_field)
ρe_radiation_forcing = Forcing(Fρe_field)
ρe_forcing = (ρe_radiation_forcing, ρe_subsidence_forcing)

microphysics = Breeze.Microphysics.WarmPhaseSaturationAdjustment()

model = AtmosphereModel(grid; coriolis, microphysics,
                        advection = WENO(order=5),
                        forcing = (; ρqᵗ=ρqᵗ_forcing, ρu=ρu_forcing, ρv=ρv_forcing, ρe=ρe_forcing),
                        boundary_conditions = (ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs))

# Values for the initial perturbations can be found in Appendix B
# of Siebesma et al 2003, 3rd paragraph
θϵ = 0.1
qϵ = 2.5e-5
θᵢ(x, y, z) = θ_bomex(z) + θϵ * randn()
qᵢ(x, y, z) = q_bomex(z) + qϵ * randn()
uᵢ(x, y, z) = u_bomex(z)
set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ)

simulation = Simulation(model; Δt=1, stop_time)
conjure_time_step_wizard!(simulation, cfl=0.7)

# Write a callback to compute *_avg_f
u_avg = Field(Average(model.velocities.u, dims=(1, 2)))
v_avg = Field(Average(model.velocities.v, dims=(1, 2)))
e_avg = Field(Average(model.energy_density / ρᵣ, dims=(1, 2)))
qᵗ_avg = Field(Average(model.moisture_mass_fraction, dims=(1, 2)))

function compute_averages!(sim)
    compute!(u_avg)
    compute!(v_avg)
    compute!(e_avg)
    compute!(qᵗ_avg)
    parent(u_avg_f) .= parent(u_avg)
    parent(v_avg_f) .= parent(v_avg)
    parent(e_avg_f) .= parent(e_avg)
    parent(qᵗ_avg_f) .= parent(qᵗ_avg)
    return nothing
end

add_callback!(simulation, compute_averages!)

T = model.temperature
qˡ = model.microphysical_fields.liquid_mass_fraction
qᵛ = model.microphysical_fields.specific_humidity

qᵛ⁺ = Breeze.AtmosphereModels.SaturationSpecificHumidityField(model)
rh = Field(qᵛ / qᵛ⁺) # relative humidity

T_avg = Field(Average(T, dims=(1, 2)))
qˡ_avg = Field(Average(qˡ, dims=(1, 2)))
qᵛ⁺_avg = Field(Average(qᵛ⁺, dims=(1, 2)))
rh_avg = Field(Average(rh, dims=(1, 2)))

# Uncomment to make plots
using WGLMakie

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
    compute!(e_avg)
    compute!(u_avg)
    compute!(v_avg)

    lines!(axT, T_avg)
    lines!(axqˡ, qˡ_avg)
    lines!(axrh, rh_avg)
    lines!(axu, u_avg)
    lines!(axu, v_avg)
    lines!(axq, qᵗ_avg)
    # lines!(axq, qᵛ⁺_avg)
    lines!(axθ, e_avg)
    display(fig)
    return nothing
end

# update_plots!(simulation)
# display(fig)

add_callback!(simulation, update_plots!, TimeInterval(20minutes))

function progress(sim)
    compute!(T)
    qˡmax = maximum(qˡ)
    qᵛmax = maximum(qᵛ)

    compute!(rh)
    rhmax = maximum(rh)

    umax = maximum(abs, u_avg)
    vmax = maximum(abs, v_avg)

    qᵗ = sim.model.moisture_mass_fraction
    qᵗmax = maximum(qᵗ)

    ρe = sim.model.energy_density
    ρemin = minimum(ρe)
    ρemax = maximum(ρe)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, max|ū|: (%.2e, %.2e), max(rh): %.2f",
                    iteration(sim), prettytime(sim), prettytime(sim.Δt), umax, vmax, rhmax)

    msg *= @sprintf(", max(qᵗ): %.2e, max(qᵛ): %.2e, max(qˡ): %.2e, extrema(ρe): (%.3e, %.3e)",
                     qᵗmax, qᵛmax, qˡmax, ρemin, ρemax)

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

#=
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
=#
