# # [Differentiating a stratocumulus forecast](@id ena_differentiability_example)
#
# How does the cloud in a stratocumulus-topped boundary layer respond to small changes
# in its temperature and humidity profiles? This example answers that question for a
# short forecast of the marine boundary layer at the Atmospheric Radiation Measurement (ARM)
# Eastern North Atlantic (ENA) site on Graciosa Island, Azores, on 18 July 2017.
# The large eddy simulation uses the anelastic equations and mixed-phase, one-moment
# microphysics in which cloud liquid and cloud ice condense and evaporate at finite rates,
# rather than through instantaneous saturation adjustment.
# We differentiate a short forecast with [Reactant](https://github.com/EnzymeAD/Reactant.jl)
# and [Enzyme](https://github.com/EnzymeAD/Enzyme.jl). One reverse-mode sweep gives the
# sensitivity of the final liquid water path to the initial potential temperature
# and water vapor at every height.
#
# The sounding, large-scale forcing, and surface fluxes are those of the case study
# by [Covert2022](@citet), who simulated this day with SAM.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState, on_architecture
using Oceananigans.TimeSteppers: reset!, update_state!
using Oceananigans: initialize!
using Oceananigans.Operators: ℑxyᶜᶠᵃ, ℑxyᶠᶜᵃ
using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState, MoistureMassFractions
using Breeze.Microphysics: adjust_thermodynamic_state
using Reactant
using Reactant: @trace
using Enzyme
using CUDA
using CairoMakie
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using Downloads
using Printf
using Random
using Statistics: mean

Random.seed!(718)

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

# Differentiation needs a tight comparison against finite differences, so we use
# double precision. The forward model runs on the GPU when one is available, and
# Reactant compiles the derivative for the same device.

Oceananigans.defaults.FloatType = Float64
architecture = CUDA.functional() ? GPU() : CPU()
Reactant.set_default_backend(CUDA.functional() ? "gpu" : "cpu")

# ## The ENA case
#
# [Covert2022](@citet) published their SAM input files. We download the sounding,
# the time-varying large-scale forcing, and the surface fluxes at a fixed revision.

revision = "a403498ff44045b531af5dbfe2ee461a9c878dd9"
source = "https://raw.githubusercontent.com/dmechem/ENA_variability_LES_bulk_paper/$revision"
sounding_path = Downloads.download("$source/snd")
forcing_path = Downloads.download("$source/lsf_time_varying")
surface_path = Downloads.download("$source/sfc_time_varying")

# SAM stores profiles in blocks: a header with the day, the number of levels and the
# surface pressure (hPa), followed by one row per level. We read the first block,
# which applies at 06 UTC.

function first_sam_profile(path, columns)
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    header = split(replace(lines[2], ',' => ' '))
    levels = parse(Int, header[2])
    surface_pressure = 100 * parse(Float64, header[3])
    rows = [parse.(Float64, split(line)[1:columns]) for line in lines[3:levels+2]]
    return permutedims(reduce(hcat, rows)), surface_pressure
end

sounding, surface_pressure = first_sam_profile(sounding_path, 6)
forcing, _ = first_sam_profile(forcing_path, 7)

# The sounding is given on pressure levels, with potential temperature `θ` and total
# water in g/kg. A level is repeated at the inversion, which we drop. We integrate the
# hypsometric equation to find the height of each level.

sounding = sounding[unique(i -> sounding[i, 2], axes(sounding, 1)), :]
pressure = 100 .* sounding[:, 2]
θ_sounding = sounding[:, 3]
q_sounding = sounding[:, 4] ./ 1000
u_sounding = sounding[:, 5]
v_sounding = sounding[:, 6]

constants = ThermodynamicConstants()
Rᵈ = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
g = constants.gravitational_acceleration
κ = Rᵈ / cᵖᵈ

Π_sounding = (pressure ./ 1e5) .^ κ
Tᵛ = θ_sounding .* Π_sounding .* (1 .+ 0.608 .* q_sounding)
z_sounding = zeros(length(pressure))
for n in 2:length(pressure)
    z_sounding[n] = z_sounding[n-1] + Rᵈ / g * (Tᵛ[n-1] + Tᵛ[n]) / 2 * log(pressure[n-1] / pressure[n])
end

# The sounding begins below sea level, at 1040 hPa. We shift the heights so that
# ``z = 0`` falls at the surface pressure.

function interpolate(coordinates, values, x)
    n = clamp(searchsortedlast(coordinates, x), 1, length(coordinates) - 1)
    ζ = clamp((x - coordinates[n]) / (coordinates[n+1] - coordinates[n]), 0, 1)
    return (1 - ζ) * values[n] + ζ * values[n+1]
end

z_sounding .-= interpolate(reverse(pressure), reverse(z_sounding), surface_pressure)

# The forcing file holds, on its own pressure levels, the large-scale temperature and
# moisture tendencies, the geostrophic wind, and the subsidence velocity.
# We interpolate them to the sounding heights.

forcing_pressure = reverse(100 .* forcing[:, 2])
forcing_profile(column) = [interpolate(forcing_pressure, reverse(forcing[:, column]), p) for p in pressure]
temperature_tendency = forcing_profile(3)
moisture_tendency = forcing_profile(4)
uᵍ_sounding = forcing_profile(5)
vᵍ_sounding = forcing_profile(6)
wˢ_sounding = forcing_profile(7)

# The surface sensible and latent heat fluxes and the kinematic surface stress are
# prescribed. We use their 06 UTC values, which change little over the forecast.

## Columns: day, sea surface temperature, sensible heat flux, latent heat flux, stress.
surface = parse.(Float64, split(readlines(surface_path)[2]))
sensible_heat_flux = surface[3]
latent_heat_flux = surface[4]
kinematic_stress = surface[5]

# ## Grid and model
#
# The boundary layer is capped by a 6 K inversion near 1.15 km. Our domain is
# 3.2 km × 3.2 km × 2 km with 100 m horizontal and 50 m vertical spacing: small
# for a stratocumulus large eddy simulation, but quick to differentiate.
# We build the same model on two architectures: one to spin up the turbulence and
# check derivatives, and one on `ReactantState` to compile the derivative.

Nx = Ny = 32
Nz = 40
Lx = Ly = 3200
Lz = 2000

z_centers = collect(range(Lz / 2Nz, Lz - Lz / 2Nz, length=Nz))
z_faces = collect(range(0, Lz, length=Nz+1))
on_centers(values) = [interpolate(z_sounding, values, z) for z in z_centers]

# Large-scale forcing converts SAM's temperature tendency to a potential temperature
# tendency. Radiative cooling is idealized as 70 W/m² of flux divergence concentrated
# in a 50 m layer at the inversion, where longwave cooling drives stratocumulus turbulence.

lower = findall(<(3000), z_sounding)
z_inversion = z_sounding[lower][argmax(diff(θ_sounding[lower]))]
θ_tendency = on_centers(temperature_tendency ./ Π_sounding)
qᵗ_tendency = on_centers(moisture_tendency)
cooling = @. -70 * exp(-(z_centers - z_inversion)^2 / 50^2) / sqrt(π) / 50
@info @sprintf("Inversion height: %.0f m", z_inversion)

@inline function surface_stress(i, j, grid, clock, fields, parameters)
    ## Align the stress with the near-surface wind, interpolated to the stress location.
    @inbounds begin
        u = ifelse(parameters.component == 1, fields.u[i, j, 1], ℑxyᶜᶠᵃ(i, j, 1, grid, fields.u))
        v = ifelse(parameters.component == 1, ℑxyᶠᶜᵃ(i, j, 1, grid, fields.v), fields.v[i, j, 1])
    end
    speed = max(sqrt(u^2 + v^2), 1e-6)
    velocity = ifelse(parameters.component == 1, u, v)
    return -parameters.density * parameters.stress * velocity / speed
end

function ena_model(arch)
    grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(0, Lz),
                           halo=(5, 5, 5), topology=(Periodic, Periodic, Bounded))

    reference_state = ReferenceState(grid, constants; base_pressure=surface_pressure, potential_temperature=θ_sounding[1])
    dynamics = AnelasticDynamics(reference_state)

    column(location, values) = set!(Field{Nothing, Nothing, location}(grid), reshape(values, 1, 1, :))
    wˢ = column(Face, [interpolate(z_sounding, wˢ_sounding, z) for z in z_faces])
    subsidence = SubsidenceForcing(wˢ)
    geostrophic = geostrophic_forcings(column(Center, on_centers(uᵍ_sounding)),
                                       column(Center, on_centers(vᵍ_sounding)))

    ## Subsidence advects every scalar, including the cloud and precipitation condensates.
    forcing = (θ  = (subsidence, Forcing(column(Center, θ_tendency))),
               qᵗ = (subsidence, Forcing(column(Center, qᵗ_tendency))),
               qᶜˡ = subsidence, qᶜⁱ = subsidence, qʳ = subsidence, qˢⁿ = subsidence,
               u  = (subsidence, geostrophic.u),
               v  = (subsidence, geostrophic.v),
               ρE = Forcing(column(Center, cooling)),
               w  = Relaxation(rate=1/30, mask=GaussianMask{:z}(center=Lz, width=300)))

    ℒˡ = constants.liquid.reference_latent_heat
    ρ₀ = surface_density(reference_state)
    stress(component) = FluxBoundaryCondition(surface_stress; discrete_form=true,
                                              parameters=(; density=ρ₀, stress=kinematic_stress, component))
    boundary_conditions = (ρE  = FieldBoundaryConditions(bottom=FluxBoundaryCondition(sensible_heat_flux)),
                           ρqᵗ = FieldBoundaryConditions(bottom=FluxBoundaryCondition(latent_heat_flux / ℒˡ)),
                           ρu  = FieldBoundaryConditions(bottom=stress(1)),
                           ρv  = FieldBoundaryConditions(bottom=stress(2)))

    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(Float64), CloudIce(Float64))
    microphysics = OneMomentCloudMicrophysics(; cloud_formation)
    weno = WENO(order=5)
    positive = WENO(order=5, bounds=(0, 1))
    scalar_advection = (ρθ=weno, ρqᵛ=positive, ρqᶜˡ=positive, ρqᶜⁱ=positive, ρqʳ=positive, ρqˢⁿ=positive)

    return AtmosphereModel(grid; dynamics, microphysics, forcing, boundary_conditions,
                           coriolis=FPlane(latitude=39.09), closure=SmagorinskyLilly(),
                           momentum_advection=weno, scalar_advection)
end

model = ena_model(architecture)

# ## Initial conditions
#
# The initial total water is partitioned into vapor and cloud liquid by saturation
# adjustment at the reference pressure. The microphysics then evolves the cloud prognostically.
# Small random temperature perturbations below the inversion seed the turbulence.

p_centers = vec(Array(interior(model.dynamics.reference_state.pressure)))
p₀ = model.dynamics.reference_state.standard_pressure
adjustment = SaturationAdjustment(; equilibrium=WarmPhaseEquilibrium())

θ_column = on_centers(θ_sounding)
qᵗ_column = on_centers(q_sounding)
qᶜˡ_column = map(θ_column, qᵗ_column, p_centers) do θ, qᵗ, p
    state = LiquidIcePotentialTemperatureState(θ, MoistureMassFractions(qᵗ), p₀, p)
    adjust_thermodynamic_state(state, adjustment, constants).moisture_mass_fractions.liquid
end

horizontally_uniform(column) = repeat(reshape(column, 1, 1, Nz), Nx, Ny, 1)
δθ = 0.1 .* (rand(Nx, Ny, Nz) .- 0.5) .* reshape(z_centers .< z_inversion, 1, 1, Nz)

set!(model; θ = horizontally_uniform(θ_column) .+ δθ,
            qᵗ = horizontally_uniform(qᵗ_column),
            qᶜˡ = horizontally_uniform(qᶜˡ_column),
            u = horizontally_uniform(on_centers(u_sounding)),
            v = horizontally_uniform(on_centers(v_sounding)))

# The liquid water path, in kg/m², is the column-integrated cloud and rain water.
# We compute it from the prognostic densities so that the same function can be traced.

function liquid_water_path(model)
    ρqˡ = interior(model.microphysical_fields.ρqᶜˡ) .+ interior(model.microphysical_fields.ρqʳ)
    return sum(ρqˡ) * (Lz / Nz) / (Nx * Ny)
end

@info @sprintf("Initial liquid water path: %.3f kg/m²", liquid_water_path(model))

# ## Spin up
#
# Before differentiating, we spin up turbulence for one hour.

simulation = Simulation(model; Δt=1, stop_time=1hour)
conjure_time_step_wizard!(simulation, cfl=0.7, max_Δt=2)

function progress(simulation)
    model = simulation.model
    @info @sprintf("Spin-up: t = %s, Δt = %s, max|w| = %.2f m/s, LWP = %.3f kg/m²",
                   prettytime(simulation), prettytime(simulation.Δt),
                   maximum(abs, model.velocities.w), liquid_water_path(model))
    return nothing
end

add_callback!(simulation, progress, TimeInterval(10minutes))
run!(simulation)

# ## The forecast and its derivative
#
# Starting from the spun-up state, we forecast for 4 minutes with a fixed 2 s step.
# The controls are horizontally uniform corrections ``δθ(z)`` and ``δqᵛ(z)`` to the
# initial potential temperature and water vapor. Every other prognostic variable starts
# from the spun-up state, which we copy once onto the device that will run the forecast.

Δt = 2.0
steps = 120

function forecast_state(model, arch)
    θ = liquid_ice_potential_temperature(model)
    compute!(θ)
    copy_to(field) = on_architecture(arch, Array(interior(field)))
    fixed = (; (name => copy_to(field) for (name, field) in pairs(Oceananigans.prognostic_fields(model))
                if name ∉ (:ρθ, :ρqᵛ))...)
    return (; θ=copy_to(θ), qᵛ=copy_to(model.microphysical_fields.qᵛ),
              temperature=copy_to(model.temperature), fixed)
end

# Each forecast restores every prognostic variable and the clock, so the same model
# (and the same compiled derivative) can be reused for any number of forecasts.

function forecast_liquid_water_path(model, δθ, δqᵛ, state, Δt, steps)
    reset!(model.clock)
    set!(model.temperature, state.temperature)
    set!(model; θ = state.θ .+ reshape(δθ, 1, 1, :),
                qᵛ = state.qᵛ .+ reshape(δqᵛ, 1, 1, :),
                state.fixed...)
    ## Prepare the first step outside the traced loop and take every step inside it,
    ## so the compiled program contains a single time step.
    initialize!(model)
    update_state!(model)
    @trace mincut=true checkpointing=true track_numbers=false for step in 1:steps
        time_step!(model, Δt)
    end
    return liquid_water_path(model)
end

# Reverse-mode differentiation returns the gradient with respect to both profiles at
# once. `set_strong_zero` keeps zero sensitivities exactly zero through inactive
# microphysical branches, whose fractional powers have singular derivatives at zero.

function liquid_water_path_gradient!(model, dmodel, δθ, dθ, δqᵛ, dqᵛ, state, Δt, steps)
    Enzyme.remake_zero!(dmodel)
    dθ .= 0
    dqᵛ .= 0
    _, path = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                              forecast_liquid_water_path, Enzyme.Active,
                              Enzyme.Duplicated(model, dmodel),
                              Enzyme.Duplicated(δθ, dθ), Enzyme.Duplicated(δqᵛ, dqᵛ),
                              Enzyme.Const(state), Enzyme.Const(Δt), Enzyme.Const(steps))
    return path, dθ, dqᵛ
end

# Now we build the model on `ReactantState`, copy in the spun-up state, and compile.
# Compilation runs inside a task with a large stack, which Reactant's compiler needs
# for a model of this size.

function with_large_stack(f)
    task = Task(f, 256 * 1024^2)
    schedule(task)
    return fetch(task)
end

reactant_model = with_large_stack(() -> ena_model(ReactantState()))
reactant_state = forecast_state(model, ReactantState())
dmodel = Enzyme.make_zero(reactant_model)
δθ = Reactant.to_rarray(zeros(Nz))
δqᵛ = Reactant.to_rarray(zeros(Nz))
dθ = Reactant.to_rarray(zeros(Nz))
dqᵛ = Reactant.to_rarray(zeros(Nz))

# We exclude one of Reactant's optimization patterns, `concat_insert_dim_elementwise`,
# which batches concatenated elementwise operations. On the differentiated time-step loop
# it more than triples the time spent optimizing the derivative, and excluding it leaves
# the derivative unchanged.

compile_options = Reactant.CompileOptions(raise=true, raise_first=true, sync=true,
                                          excluded_passes=["concat_insert_dim_elementwise"])

compile_time = @elapsed compiled_gradient! = with_large_stack() do
    Reactant.@compile compile_options=compile_options liquid_water_path_gradient!(
        reactant_model, dmodel, δθ, dθ, δqᵛ, dqᵛ, reactant_state, Δt, steps)
end

path, ∂θ, ∂qᵛ = compiled_gradient!(reactant_model, dmodel, δθ, dθ, δqᵛ, dqᵛ, reactant_state, Δt, steps)
path = Float64(path)
∂θ, ∂qᵛ = Array(∂θ), Array(∂qᵛ)
@info @sprintf("Compiled the derivative in %.0f s. Final liquid water path: %.4f kg/m²", compile_time, path)

# ## Checking the derivative
#
# We check the reverse-mode derivative against central differences of the forward
# model, along two directions: a 0.1 K warming of the cloud layer, and a
# 0.1 g/kg moistening of the subcloud layer.

state = forecast_state(model, architecture)
zero_profile = zeros(Nz)
forecast(δθ, δqᵛ) = forecast_liquid_water_path(model, on_architecture(architecture, δθ),
                                               on_architecture(architecture, δqᵛ), state, Δt, steps)

reference_path = forecast(zero_profile, zero_profile)

cloud_layer = @. 0.1 * exp(-((z_centers - z_inversion + 200) / 150)^2)
subcloud_layer = @. 1e-4 * (z_centers < 500)

checks = map((("warming the cloud layer", cloud_layer, zero_profile, ∂θ),
              ("moistening the subcloud layer", zero_profile, subcloud_layer, ∂qᵛ))) do (label, θ_direction, q_direction, gradient)
    ε = 1e-2
    plus = forecast(ε .* θ_direction, ε .* q_direction)
    minus = forecast(-ε .* θ_direction, -ε .* q_direction)
    finite_difference = (plus - minus) / 2ε
    automatic = sum(gradient .* (θ_direction .+ q_direction))
    @info @sprintf("%s: automatic %.6e, finite difference %.6e kg/m²", label, automatic, finite_difference)
    (; label, automatic, finite_difference)
end

@info @sprintf("Forward and Reactant forecasts agree to %.1e", abs(path - reference_path) / reference_path)

# ## Sensitivity profiles
#
# Each sensitivity panel shows how the final liquid water path responds to a change of
# the initial profile *in one 50 m layer*, beside the mean profiles at the start of the forecast.

θ̄ = vec(mean(Array(state.θ); dims=(1, 2)))
q̄ᵛ = vec(mean(Array(state.qᵛ); dims=(1, 2)))
ρ̄ = vec(Array(interior(model.dynamics.reference_state.density)))
q̄ᶜˡ = vec(mean(Array(state.fixed.ρqᶜˡ); dims=(1, 2))) ./ ρ̄

fig = Figure(size=(1200, 500), fontsize=14)
axθ = Axis(fig[1, 1]; xlabel="θ (K)", ylabel="z (m)", title="Initial state")
axq = Axis(fig[1, 2]; xlabel="qᵛ (kg/kg)", yticklabelsvisible=false)
axc = Axis(fig[1, 3]; xlabel="qᶜˡ (kg/kg)", yticklabelsvisible=false)
axdθ = Axis(fig[1, 4]; xlabel="∂LWP/∂θ (kg m⁻² K⁻¹)", yticklabelsvisible=false, title="Sensitivity")
axdq = Axis(fig[1, 5]; xlabel="∂LWP/∂qᵛ (kg m⁻²)", yticklabelsvisible=false)

lines!(axθ, θ̄, z_centers; color=:magenta)
lines!(axq, q̄ᵛ, z_centers; color=:dodgerblue)
lines!(axc, q̄ᶜˡ, z_centers; color=:lime)
lines!(axdθ, ∂θ, z_centers; color=:magenta)
lines!(axdq, ∂qᵛ, z_centers; color=:dodgerblue)

for ax in (axθ, axq, axc, axdθ, axdq)
    hlines!(ax, z_inversion; color=:gray, linestyle=:dash)
    ylims!(ax, 0, Lz)
end

for ax in (axdθ, axdq)
    vlines!(ax, 0; color=:gray)
end

Label(fig[0, :], @sprintf("ENA, 18 July 2017: sensitivity of the liquid water path after %s", prettytime(steps * Δt)),
      fontsize=16, tellwidth=false)

save("ena_differentiability.png", fig) #src
fig

# The derivative is exact for this discrete, finite-time forecast, including the
# active microphysical and advection branches. It is not the sensitivity of
# long-time turbulent statistics: over longer windows, chaotic growth makes
# trajectory derivatives large and noisy.
