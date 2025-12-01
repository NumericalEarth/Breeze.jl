using Breeze
using Oceananigans.Units
using Printf
using GLMakie

Nx = Nz = 64
Δx, Δz = 100, 50
Lx, Lz = Δx * Nx, Δz * Nz
grid = RectilinearGrid(size=(Nx, Nz), x=(0, Lx), z=(0, Lz), topology=(Periodic, Flat, Bounded))

ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(1000))
model = AtmosphereModel(grid, advection=WENO(), boundary_conditions=(; ρe=ρe_bcs))

Tₛ = model.formulation.reference_state.potential_temperature
g = model.thermodynamic_constants.gravitational_acceleration
N² = 1e-6
dθdz = N² * Tₛ / g  # Background potential temperature gradient
θᵢ(x, z) = Tₛ + dθdz * z # background stratification
Ξᵢ(x, z) = 1e-2 * randn()
set!(model, θ=θᵢ, u=Ξᵢ, v=Ξᵢ)

simulation = Simulation(model, Δt=10, stop_time=20minutes)
conjure_time_step_wizard!(simulation, cfl=0.7)
∫ρe = Field(Integral(model.energy_density))

function progress(sim)
    compute!(∫ρe)
    max_w = maximum(abs, model.velocities.w)

    @info @sprintf("%s: t = %s, ∫ρe dV = %.9e J/kg, max|w| = %.2e m/s",
                   iteration(sim), prettytime(sim), first(∫ρe), max_w)

    return nothing
end

add_callback!(simulation, progress, IterationInterval(5))

∫ρe_series = Float64[]
w_series = []
t = []

function record_data!(sim)
    compute!(∫ρe)
    push!(∫ρe_series, first(∫ρe))
    push!(w_series, deepcopy(model.velocities.w))
    push!(t, sim.model.clock.time)
    return nothing
end
add_callback!(simulation, record_data!)

run!(simulation)

fig = Figure(size=(800, 550), fontsize=12)

Nt = length(∫ρe_series)
axw = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)", aspect=DataAspect())
axe = Axis(fig[2, 1], xlabel="Time (s)", ylabel="∫ρe dV / ∫ρe dV |t=0", height=Relative(0.2))
slider = Slider(fig[3, 1], range=1:Nt, startvalue=1)
n = slider.value

∫ρe_n = @lift ∫ρe_series[$n] / ∫ρe_series[1]
t_n = @lift t[$n]
w_n = @lift w_series[$n]
w_lim = maximum(abs, model.velocities.w)

lines!(axe, t, ∫ρe_series / ∫ρe_series[1])
scatter!(axe, t_n, ∫ρe_n, markersize=10)

hm = heatmap!(axw, w_n, colorrange=(-w_lim, w_lim), colormap=:balance)
Colorbar(fig[1, 2], hm, label="w (m/s)")

rowgap!(fig.layout, 1, Relative(-0.2))
rowgap!(fig.layout, 2, Relative(-0.2))

display(fig)
