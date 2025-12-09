# using Pkg; Pkg.activate(".")
using Breeze
using Oceananigans.Units

using AtmosphericProfilesLibrary
using Printf

using Oceananigans.Operators: ∂zᶜᶜᶠ, ℑzᵃᵃᶜ
using CUDA
using CairoMakie

# Siebesma et al (2003) resolution!
# DOI: https://doi.org/10.1175/1520-0469(2003)60<1201:ALESIS>2.0.CO;2
Nx = Ny = 64
Nz = 75

x = y = (0, 6400)
z = (0, 3000)

arch = GPU() # if changing to CPU() remove the `using CUDA` line above
stop_time = 6hours

grid = RectilinearGrid(arch; x, y, z, 
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

FT = eltype(grid)
θ_bomex = AtmosphericProfilesLibrary.Bomex_θ_liq_ice(FT)
q_bomex = AtmosphericProfilesLibrary.Bomex_q_tot(FT)
u_bomex = AtmosphericProfilesLibrary.Bomex_u(FT)

p₀, θ₀ = 101500, 299.1
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
formulation = AnelasticFormulation(reference_state, thermodynamics=:LiquidIcePotentialTemperature)

q₀ = Breeze.Thermodynamics.MoistureMassFractions{eltype(grid)} |> zero
ρ₀ = Breeze.Thermodynamics.density(p₀, θ₀, q₀, constants)
cᵖᵈ = constants.dry_air.heat_capacity
Lˡ = constants.liquid.reference_latent_heat
w′T′, w′q′ = 8e-3, 5.2e-5
Q = ρ₀ * cᵖᵈ * w′T′ 
F = ρ₀ * w′q′

ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(Q))
ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′T′))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(F))

u★ = 0.28 # m/s
@inline ρu_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / sqrt(ρu^2 + ρv^2)
@inline ρv_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / sqrt(ρu^2 + ρv^2)
ρu_drag_bc = FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρv_drag_bc = FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρu_bcs = FieldBoundaryConditions(bottom=ρu_drag_bc)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_drag_bc)

@inline w_dz_ϕ(i, j, k, grid, w, ϕ) = @inbounds w[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, ϕ)

@inline function Fρu_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_U = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.u_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_U
end

@inline function Fρv_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_V = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.v_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_V
end

@inline function Fρe_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_E = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.e_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_E
end

@inline function Fρθ_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_T = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.θ_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_T
end

@inline function Fρqᵗ_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_Qᵗ = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.qᵗ_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_Qᵗ
end

# f for "forcing"
u_avg_f = Field{Nothing, Nothing, Center}(grid)
v_avg_f = Field{Nothing, Nothing, Center}(grid)
e_avg_f = Field{Nothing, Nothing, Center}(grid)
θ_avg_f = Field{Nothing, Nothing, Center}(grid)
qᵗ_avg_f = Field{Nothing, Nothing, Center}(grid)

wˢ = Field{Nothing, Nothing, Face}(grid)
w_bomex = AtmosphericProfilesLibrary.Bomex_subsidence(FT)
set!(wˢ, z -> w_bomex(z))

ρᵣ = formulation.reference_state.density
ρu_subsidence_forcing  = Forcing(Fρu_subsidence,  discrete_form=true, parameters=(; u_avg=u_avg_f, wˢ, ρᵣ))
ρv_subsidence_forcing  = Forcing(Fρv_subsidence,  discrete_form=true, parameters=(; v_avg=v_avg_f, wˢ, ρᵣ))
ρe_subsidence_forcing  = Forcing(Fρe_subsidence,  discrete_form=true, parameters=(; e_avg=e_avg_f, wˢ, ρᵣ))
ρθ_subsidence_forcing  = Forcing(Fρθ_subsidence,  discrete_form=true, parameters=(; θ_avg=θ_avg_f, wˢ, ρᵣ))
ρqᵗ_subsidence_forcing = Forcing(Fρqᵗ_subsidence, discrete_form=true, parameters=(; qᵗ_avg=qᵗ_avg_f, wˢ, ρᵣ))

coriolis = FPlane(f=3.76e-5)
uᵍ = Field{Nothing, Nothing, Center}(grid)
vᵍ = Field{Nothing, Nothing, Center}(grid)
uᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_u(FT)
vᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_v(FT)
set!(uᵍ, z -> uᵍ_bomex(z))
set!(vᵍ, z -> vᵍ_bomex(z))
ρuᵍ = Field(ρᵣ * uᵍ)
ρvᵍ = Field(ρᵣ * vᵍ)

@inline Fρu_geostrophic(i, j, k, grid, clock, fields, p) = @inbounds - p.f * p.ρvᵍ[i, j, k]
@inline Fρv_geostrophic(i, j, k, grid, clock, fields, p) = @inbounds p.f * p.ρuᵍ[i, j, k]

ρu_geostrophic_forcing = Forcing(Fρu_geostrophic, discrete_form=true, parameters=(; f=coriolis.f, ρvᵍ))
ρv_geostrophic_forcing = Forcing(Fρv_geostrophic, discrete_form=true, parameters=(; f=coriolis.f, ρuᵍ))

ρu_forcing = (ρu_subsidence_forcing, ρu_geostrophic_forcing)
ρv_forcing = (ρv_subsidence_forcing, ρv_geostrophic_forcing)

drying = Field{Nothing, Nothing, Center}(grid)
dqdt_bomex = AtmosphericProfilesLibrary.Bomex_dqtdt(FT)
set!(drying, z -> dqdt_bomex(z))
set!(drying, ρᵣ * drying)
ρqᵗ_drying_forcing = Forcing(drying)
ρqᵗ_forcing = (ρqᵗ_drying_forcing, ρqᵗ_subsidence_forcing)

Fρe_field = Field{Nothing, Nothing, Center}(grid)
Fρθ_field = Field{Nothing, Nothing, Center}(grid)
dTdt_bomex = AtmosphericProfilesLibrary.Bomex_dTdt(FT)
set!(Fρe_field, z -> dTdt_bomex(1, z))
set!(Fρe_field, ρᵣ * cᵖᵈ * Fρe_field)
set!(Fρθ_field, z -> dTdt_bomex(1, z))
set!(Fρθ_field, ρᵣ * Fρe_field)

ρe_radiation_forcing = Forcing(Fρe_field)
ρe_forcing = (ρe_radiation_forcing, ρe_subsidence_forcing)

ρθ_radiation_forcing = Forcing(Fρθ_field)
ρθ_forcing = (ρθ_radiation_forcing, ρθ_subsidence_forcing)

fig = Figure()
axe = Axis(fig[1, 1], xlabel="z (m)", ylabel="Fρe (K/s)")
axq = Axis(fig[1, 2], xlabel="z (m)", ylabel="Fρqᵗ (1/s)")
lines!(axe, Fρe_field)
lines!(axq, drying)
save("forcings.png", fig)

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
advection = WENO(order=9)

model = AtmosphereModel(grid; formulation, coriolis, microphysics, advection,
                        forcing = (ρqᵗ=ρqᵗ_forcing, ρu=ρu_forcing, ρv=ρv_forcing, ρθ=ρθ_subsidence_forcing, ρe=ρe_radiation_forcing),
                        boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs))

# Values for the initial perturbations can be found in Appendix B
# of Siebesma et al 2003, 3rd paragraph
θϵ, qϵ, zϵ = 0.1, 2.5e-5, 1600
θᵢ(x, y, z) = θ_bomex(z) + θϵ * rand() * (z < zϵ)
qᵢ(x, y, z) = q_bomex(z) + qϵ * rand() * (z < zϵ)
uᵢ(x, y, z) = u_bomex(z)
set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ)

simulation = Simulation(model; Δt=10, stop_time)
conjure_time_step_wizard!(simulation, cfl=0.7)

# Write a callback to compute *_avg_f
e = static_energy(model)
θ = liquid_ice_potential_temperature(model)
u_avg = Field(Average(model.velocities.u, dims=(1, 2)))
v_avg = Field(Average(model.velocities.v, dims=(1, 2)))
e_avg = Field(Average(e, dims=(1, 2)))
θ_avg = Field(Average(θ, dims=(1, 2)))
qᵗ_avg = Field(Average(model.specific_moisture, dims=(1, 2)))

function compute_averages!(sim)
    compute!(u_avg)
    compute!(v_avg)
    compute!(e_avg)
    compute!(θ_avg)
    compute!(qᵗ_avg)
    parent(u_avg_f) .= parent(u_avg)
    parent(v_avg_f) .= parent(v_avg)
    parent(e_avg_f) .= parent(e_avg)
    parent(θ_avg_f) .= parent(θ_avg)
    parent(qᵗ_avg_f) .= parent(qᵗ_avg)
    return nothing
end

add_callback!(simulation, compute_averages!)

qˡ = model.microphysical_fields.qˡ
qᵛ = model.microphysical_fields.qᵛ
qᵗ = model.specific_moisture
qᵛ⁺ = SaturationSpecificHumidityField(model)
qˡ_avg = Average(qˡ, dims=(1, 2)) |> Field

fig = Figure()
axθ =  Axis(fig[1, 1], xlabel="Potential temperature (K)", ylabel="z (m)")
axu =  Axis(fig[1, 2], xlabel="Velocity (m/s)", ylabel="z (m)")
axt =  Axis(fig[2, 1], xlabel="Specific moisture", ylabel="z (m)")
axl =  Axis(fig[2, 2], xlabel="Liquid mass fraction", ylabel="z (m)")

xlims!(axl, -1e-6, 1e-5)

function plot_averages(sim)
    lines!(axθ, θ_avg)
    lines!(axu, u_avg)
    lines!(axu, v_avg)
    lines!(axt, qᵗ_avg)
    lines!(axl, qˡ_avg)
    save("averages.png", fig)
    return nothing
end

add_callback!(simulation, plot_averages, TimeInterval(1hour))

fig_lower = Figure(size=(900, 400))
ax_ρe = Axis(fig_lower[1, 1], xlabel="Energy density", ylabel="z (m)")
ax_e  = Axis(fig_lower[1, 2], xlabel="Specific energy", ylabel="z (m)")
ax_θ  = Axis(fig_lower[1, 3], xlabel="Potential temperature (K)", ylabel="z (m)")

e = static_energy(model)
ρe = ρᵣ * e
ρe_avg = Average(ρe, dims=(1, 2)) |> Field
e_avg = Average(e, dims=(1, 2)) |> Field

ylims!(ax_ρe, 0, 400)
ylims!(ax_e, 0, 400)
ylims!(ax_θ, 0, 400)

xlims!(ax_ρe, 3.45e5, 3.60e5)
xlims!(ax_e, 304200, 304600)
xlims!(ax_θ, 298.0, 299.5)

function plot_lower_averages(sim)
    compute!(ρe_avg)
    compute!(e_avg)
    compute!(θ_avg)
    scatterlines!(ax_ρe, ρe_avg)
    scatterlines!(ax_e, e_avg)
    scatterlines!(ax_θ, θ_avg)
    save("lower_averages.png", fig_lower)
    return nothing
end

add_callback!(simulation, plot_lower_averages, TimeInterval(1hour))

function progress(sim)
    qˡmax = maximum(qˡ)
    qᵛmax = maximum(qᵛ)

    umax = maximum(abs, u_avg)
    vmax = maximum(abs, v_avg)

    qᵗ = sim.model.specific_moisture
    qᵗmax = maximum(qᵗ)

    ρe = static_energy_density(sim.model)
    ρemin = minimum(ρe)
    ρemax = maximum(ρe)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, max|ū|: (%.2e, %.2e)",
                    iteration(sim), prettytime(sim), prettytime(sim.Δt), umax, vmax)

    msg *= @sprintf(", max(qᵗ): %.2e, max(qᵛ): %.2e, max(qˡ): %.2e, extrema(ρe): (%.3e, %.3e)",
                     qᵗmax, qᵛmax, qˡmax, ρemin, ρemax)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

outputs = merge(model.velocities, model.tracers, (; θ, qˡ, qᵛ))
averaged_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

filename = string("bomex_", Nx, "_", Ny, "_", Nz, ".jld2")
averages_filename = string("bomex_avg_", Nx, "_", Ny, "_", Nz, ".jld2")

ow = JLD2Writer(model, outputs; filename,
                schedule = TimeInterval(1minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

averages_ow = JLD2Writer(model, averaged_outputs;
                         filename = averages_filename,
                         schedule = AveragedTimeInterval(60minutes, window=60minutes, stride=1),
                         overwrite_existing = true)

simulation.output_writers[:avg] = averages_ow

@info "Running BOMEX on grid: \n $grid \n and using model: \n $model"
run!(simulation)

#####
##### Post-processing
#####

θt  = FieldTimeSeries(averages_filename, "θ")
qᵛt  = FieldTimeSeries(averages_filename, "qᵛ")
qˡt = FieldTimeSeries(averages_filename, "qˡ")
ut = FieldTimeSeries(averages_filename, "u")
vt = FieldTimeSeries(averages_filename, "v")

times = qᵛt.times
Nt = length(θt)

fig = Figure(size=(800, 800), fontsize=12)
axθ  = Axis(fig[1, 1], xlabel="θ [K]", ylabel="z (m)")
axqᵛ = Axis(fig[1, 2], xlabel="qᵛ [g/kg]", ylabel="z (m)")
axuv = Axis(fig[2, 1], xlabel="u, v [m/s]", ylabel="z (m)")
axqˡ = Axis(fig[2, 2], xlabel="qˡ [g/kg]", ylabel="z (m)")


n = Nt
θn  = @lift interior(θt[$n], 1, 1, :)
qᵛn = @lift interior(qᵛt[$n], 1, 1, :)
qˡn = @lift interior(qˡt[$n], 1, 1, :)
un = @lift interior(ut[$n], 1, 1, :)
vn = @lift interior(vt[$n], 1, 1, :)
z = znodes(θt)
title = "Mean profile averaged over the last hour ($(Int(stop_time - 1hours)/3600) - $(Int(stop_time)/3600) hours)"

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

hmθ  = lines!(axθ, θn, z)
hmuv_u = lines!(axuv, un, z)
hmuv_v = lines!(axuv, vn, z)
hmqᵛ = lines!(axqᵛ, @lift($qᵛn .* 1000), z)
hmqˡ = lines!(axqˡ, @lift($qˡn .* 1000), z)
xlims!(axθ, (298, 310))
ylims!(axθ, (0, 2500))
xlims!(axuv, (-10, 2))
ylims!(axuv, (0, 2500))
xlims!(axqᵛ, (4, 18))
ylims!(axqᵛ, (0, 2500))
ylims!(axqˡ, (0, 2500))

save("bomex_avg_profiles.png", fig)
