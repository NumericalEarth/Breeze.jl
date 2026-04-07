# # Neutral atmospheric boundary layer (ABL)
#
# This canonical setup is based on [Moeng1994](@citet) and is also a demonstration
# case for the NCAR LES subgrid-scale model development [Sullivan1994](@cite).
# Sometimes, this model configuration is called "conventionally" neutral [Pedersen2014](@cite)
# or "conditionally" neutral [Berg2020](@cite), which indicate idealized dry
# shear-driven ABL, capped by a stable inversion layer, with no surface heating.
# Forcings come from a specified geostrophic wind (i.e., a specified background
# pressure gradient) and Coriolis forces; the temperature lapse rate in the free
# atmosphere is maintained with a sponge layer.
#
# In lieu of more sophisticated surface layer modeling in this example, we
# impose a fixed friction velocity at the bottom boundary.

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using CUDA
using Printf
using Random

Random.seed!(42)
if CUDA.functional()
    CUDA.seed!(42)
end

# ## Domain and grid
#
# For this documentation example, we reduce the numerical precision to Float32.
# This yields a 10x speed up on an NVidia T4 (which is used to build the docs).

arch = GPU()
Oceananigans.defaults.FloatType = Float32

# Simulation S [Moeng1994](@citet)
Nx = Ny = Nz = 96
x = y = (0, 3000)
z = (0, 1000)

grid = RectilinearGrid(arch; x, y, z,
                       size = (Nx, Ny, Nz), halo = (3, 3, 3),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state and formulation

p₀ = 1e5
θ₀ = 300

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants,
                                 surface_pressure = p₀,
                                 potential_temperature = θ₀)

dynamics = AnelasticDynamics(reference_state)

# capping inversion [Moeng1994](cite)
Δz = grid.z.Δᵃᵃᶜ
zi  = 468  # m
zi2 = zi + 6Δz  # m
Δθi = 8 / (6Δz)  # K
Γᵗᵒᵖ = 0.003 # K / m  == dθ/dz
θᵣ(z) = z < zi  ? θ₀ :
        z < zi2 ? θ₀ + Δθi * (z - zi) :
        θ₀ + Δθi * (zi2 - zi) + Γᵗᵒᵖ * (z - zi2)

# ## Surface momentum flux (drag)
#
# A bulk drag parameterization is applied with friction velocity

q₀ = Breeze.Thermodynamics.MoistureMassFractions{Float32} |> zero
ρ₀ = Breeze.Thermodynamics.density(θ₀, p₀, q₀, constants)

u★ = 0.5  # m/s, simulation "S" in [Moeng1994](@citet)
@inline ρu_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / max(sqrt(ρu^2 + ρv^2), p.ρ₀ * 1e-6)
@inline ρv_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / max(sqrt(ρu^2 + ρv^2), p.ρ₀ * 1e-6)

ρu_drag_bc = FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρv_drag_bc = FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρu_bcs = FieldBoundaryConditions(bottom=ρu_drag_bc)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_drag_bc)

# ## Sponge layer
#
# effective depth ≈ 500 m
# at |z - zᵗᵒᵖ| = 500, exp(-0.5*(500/sponge_width)^2) = 0.04 ~ 0
sponge_rate = 0.01  # 1/s -- ad hoc value, stronger (shorter damping timescale) makes no difference; weaker may be OK
sponge_width = 200  # m
sponge_mask = GaussianMask{:z}(center = last(z), width = sponge_width)

ρw_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask) # relax to 0 by default

# relax to initial temperature profile
ρᵣ = reference_state.density
ρθˡⁱᵣ = Field{Nothing, Nothing, Center}(grid)
set!(ρθˡⁱᵣ, z -> θᵣ(z))
set!(ρθˡⁱᵣ, ρᵣ * ρθˡⁱᵣ)
ρθˡⁱᵣ_data = interior(ρθˡⁱᵣ, 1, 1, :)
@inline function ρθ_sponge_fun(i, j, k, grid, clock, model_fields, p)
    zc = znode(k, grid, Center())
    return p.rate * p.mask(0, 0, zc) * (@inbounds p.target[k] - model_fields.ρθ[i, j, k])
end

ρθ_sponge = Forcing(
    ρθ_sponge_fun; discrete_form = true,
    parameters = (rate = sponge_rate, mask = sponge_mask, target = ρθˡⁱᵣ_data)
)

# ## Background forcings

coriolis = FPlane(f=1e-4)

uᵍ, vᵍ = 15, 0  # m/s, case "S" in [Moeng1994](@citet)
geostrophic = geostrophic_forcings(uᵍ, vᵍ)

# ## Assembling all the forcings

ρu_forcing = geostrophic.ρu
ρv_forcing = geostrophic.ρv
ρw_forcing = ρw_sponge
ρθ_forcing = ρθ_sponge

forcing = (; ρu=ρu_forcing, ρv=ρv_forcing, ρw=ρw_forcing, ρθ=ρθ_forcing)
nothing #hide

# ## Model setup

#advection = WENO(order=5) # too dissipative
advection = Centered(order=6)

closure = SmagorinskyLilly(C=0.18)  # [Sullivan1994](@citet)

model = AtmosphereModel(grid; dynamics, coriolis, advection, forcing, closure,
                        boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs))

# ## Initial conditions

# add velocity and temperature perturbations
δu = δv = 0.01  # m/s
δθ = 0.1  # K
zδ = 400  # m < zi

ϵ() = rand() - 1/2
uᵢ(x, y, z) = uᵍ + δu * ϵ() * (z < zδ)
vᵢ(x, y, z) = vᵍ + δv * ϵ() * (z < zδ)
θᵢ(x, y, z) = θᵣ(z) + δθ * ϵ() * (z < zδ)

set!(model, θ=θᵢ, u=uᵢ, v=vᵢ)

# ## Simulation
#
# We run the simulation for 6 hours with adaptive time-stepping.

simulation = Simulation(model; Δt=0.5, stop_time=5hour)
conjure_time_step_wizard!(simulation, cfl=0.7)

Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

# ## Output and progress
#
# We add a progress callback and output the hourly time-averages of the horizontally-averaged
# profiles for post-processing.

u, v, w = model.velocities
θ = liquid_ice_potential_temperature(model)
νₑ = model.closure_fields.νₑ

## For keeping track of the computational expense
wall_clock = Ref(time_ns())

function progress(sim)
    wmax = maximum(abs, sim.model.velocities.w)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    msg = @sprintf("Iter: %d, t: % 12s, Δt: %s, elapsed: %s; max|w|: %.2e m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed), wmax)
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# Output averaged products for calculating variances
outputs = merge(model.velocities, model.tracers, (; θ, νₑ,
                                                    uu = u*u, vv = v*v, ww = w*w,
                                                    uw = u*w, vw = v*w, θw = θ*w))
avg_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

filename = "abl_averages.jld2"
simulation.output_writers[:averages] = JLD2Writer(model, avg_outputs; filename,
                                                  schedule = AveragedTimeInterval(1hour),
                                                  overwrite_existing = true)

# Output horizontal slices for animation
# Find the k-index closest to z = 100 m
z = Oceananigans.Grids.znodes(grid, Center())
k = searchsortedfirst(z, 100.0)
@info "Saving slices at z = $(z[k]) m (k = $k)"

# Find the j-index closest to the domain center
y = Oceananigans.Grids.ynodes(grid, Center())
jmid = Ny ÷ 2
@info "Saving slices at y = $(y[jmid]) m (j = $jmid)"

slice_fields = (; u, v, w, θ)
slice_outputs = (
    u_xy = view(u, :, :, k),
    v_xy = view(v, :, :, k),
    w_xy = view(w, :, :, k),
    θ_xy = view(θ, :, :, k),
    u_xz = view(u, :, jmid, :),
    θ_xz = view(θ, :, jmid, :),
)

simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = "abl_slices.jld2",
                                                schedule = TimeInterval(300seconds),
                                                overwrite_existing = true)

@info "Running ABL simulation..."
run!(simulation)

