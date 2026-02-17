# # Radiative Shallow Convection (2D)
#
# A 2D (x-z) shallow convection case with all-sky RRTMGP radiation and
# two-moment cloud microphysics. This example validates the coupling between
# radiation, dynamics, and precipitation in a computationally cheap configuration.
#
# The setup resembles a trade-cumulus regime: a warm, moist boundary layer beneath
# a capping inversion at ~2 km, with shallow clouds forming and precipitating.
# All-sky radiation provides cloud-radiation feedback via RRTMGP spectral transfer.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf, Random, Statistics

using RRTMGP
using CloudMicrophysics

Random.seed!(2025)

# ## Parameters

SST = 300                    # Sea surface temperature [K]
solar_constant = 551.58      # RCEMIP reduced solar constant [W/m²]
cos_zenith = cosd(42.05)     # Fixed zenith angle (RCEMIP perpetual insolation)
surface_albedo = 0.07        # Ocean surface albedo

# ## Grid
#
# 2D vertical slice: periodic in x, flat in y, bounded in z.
# Shallow 4 km domain captures the trade-cumulus layer.

Nx = 128
Nz = 80
Lx = 12800   # 12.8 km (100 m horizontal spacing)
zᵗ = 4000    # 4 km domain top

arch = CPU()
FT = Float64

grid = RectilinearGrid(arch, FT;
                        size = (Nx, Nz),
                        x = (0, Lx),
                        z = (0, zᵗ),
                        halo = (5, 5),
                        topology = (Periodic, Flat, Bounded))

# ## Reference state

p₀ = 101325  # Surface pressure [Pa]
θ₀ = 300     # Reference potential temperature [K]

constants = ThermodynamicConstants(FT)

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = p₀,
                                 potential_temperature = θ₀,
                                 vapor_mass_fraction = 0)

dynamics = AnelasticDynamics(reference_state)

# ## Background atmosphere
#
# Trace gas concentrations for RRTMGP. Ozone profile is a simple tropical approximation.

@inline function tropical_ozone(z)
    troposphere_O₃ = 30e-9 * (1 + 0.5 * z / 10_000)
    zˢᵗ = 25e3
    Hˢᵗ = 5e3
    stratosphere_O₃ = 8e-6 * exp(-((z - zˢᵗ) / Hˢᵗ)^2)
    χˢᵗ = 1 / (1 + exp(-(z - 15e3) / 2))
    return troposphere_O₃ * (1 - χˢᵗ) + stratosphere_O₃ * χˢᵗ
end

background_atmosphere = BackgroundAtmosphere(
    CO₂ = 348e-6,
    CH₄ = 1650e-9,
    N₂O = 306e-9,
    O₃ = tropical_ozone
)

# ## Radiation
#
# All-sky RRTMGP with cloud-radiation interaction. Scheduled every 5 minutes
# for tight coupling with the shallow cloud layer.

radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                   surface_albedo,
                                   solar_constant,
                                   background_atmosphere,
                                   surface_temperature = SST,
                                   surface_emissivity = 0.98,
                                   schedule = TimeInterval(5minutes),
                                   coordinate = cos_zenith,
                                   liquid_effective_radius = ConstantRadiusParticles(10e-6),
                                   ice_effective_radius = ConstantRadiusParticles(30e-6))

# ## Surface fluxes

Cᴰ = 1.0e-3
Cᵀ = 1.0e-3
Cᵛ = 1.2e-3

ρθ_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=SST)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=SST)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)
ρu_bcs = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=Cᴰ))

# ## Sponge layer
#
# Rayleigh damping in the top 1 km to absorb gravity waves and damp
# spurious thermal perturbations from radiation at the domain top.
# The sponge relaxes momentum toward zero and ρθ toward the reference state.

zˢ = 3000  # Sponge starts at 3 km
λ = 1/20   # Maximum damping rate [1/s] (20 s e-folding timescale)

@inline function sponge_mask(i, j, k, grid, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    return clamp((z - p.zˢ) / (p.zᵗ - p.zˢ), 0, 1)
end

@inline function w_sponge(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Face())
    mask = clamp((z - p.zˢ) / (p.zᵗ - p.zˢ), 0, 1)
    @inbounds ρw = fields.ρw[i, j, k]
    return -p.λ * mask * ρw
end

@inline function u_sponge(i, j, k, grid, clock, fields, p)
    mask = sponge_mask(i, j, k, grid, p)
    @inbounds ρu = fields.ρu[i, j, k]
    return -p.λ * mask * ρu
end

# Theta sponge: relax ρθ toward the reference state to counteract
# the unrealistic radiative cooling at the domain top (where RRTMGP
# sees no atmosphere above the model domain).
@inline function θ_sponge(i, j, k, grid, clock, fields, p)
    mask = sponge_mask(i, j, k, grid, p)
    @inbounds begin
        ρθ = fields.ρθ[i, j, k]
        Tᵣ = p.Tᵣ[i, j, k]
        pᵣ = p.pᵣ[i, j, k]
        ρᵣ = p.ρᵣ[i, j, k]
    end
    Π = (pᵣ / p.pˢᵗ) ^ p.κ
    θᵣ = Tᵣ / Π
    ρθᵣ = ρᵣ * θᵣ
    return -p.λ * mask * (ρθ - ρθᵣ)
end

cᵖᵈ = constants.dry_air.heat_capacity
Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
κ = Rᵈ / cᵖᵈ
pˢᵗ = reference_state.standard_pressure

sponge_params = (; λ, zˢ, zᵗ)
ρw_sponge = Forcing(w_sponge, discrete_form=true, parameters=sponge_params)
ρu_sponge = Forcing(u_sponge, discrete_form=true, parameters=sponge_params)

θ_sponge_params = (; λ, zˢ, zᵗ, κ, pˢᵗ,
                     Tᵣ=reference_state.temperature,
                     pᵣ=reference_state.pressure,
                     ρᵣ=reference_state.density)
ρθ_sponge = Forcing(θ_sponge, discrete_form=true, parameters=θ_sponge_params)

# ## Microphysics
#
# Two-moment warm-rain microphysics: prognostic cloud liquid mass and number,
# rain mass and number, and aerosol number.

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
TwoMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.TwoMomentCloudMicrophysics

microphysics = TwoMomentCloudMicrophysics()

# ## Model assembly

boundary_conditions = (; ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs)

weno_order = 5
momentum_advection = WENO(order=weno_order)

scalar_advection = (ρθ   = WENO(order=weno_order),
                    ρqᵗ  = WENO(order=weno_order, bounds=(0, 1)),
                    ρqᶜˡ = WENO(order=weno_order, bounds=(0, 1)),
                    ρnᶜˡ = WENO(order=weno_order),
                    ρqʳ  = WENO(order=weno_order, bounds=(0, 1)),
                    ρnʳ  = WENO(order=weno_order),
                    ρnᵃ  = WENO(order=weno_order))

forcing = (; ρw=ρw_sponge, ρu=ρu_sponge, ρθ=ρθ_sponge)

model = AtmosphereModel(grid; dynamics, microphysics, radiation,
                        momentum_advection, scalar_advection,
                        boundary_conditions, forcing)

# ## Initial conditions
#
# RICO-like tropical trade-cumulus sounding: moist boundary layer with a
# weak inversion near 1.5 km and drier free troposphere above.

function Tᵢ(z)
    Tˢᶠᶜ = 299.2  # Surface air temperature [K]
    if z ≤ 740
        return Tˢᶠᶜ - 0.004 * z                              # Well-mixed boundary layer
    elseif z ≤ 2000
        return Tˢᶠᶜ - 0.004 * 740 - 0.003 * (z - 740)       # Cloud layer
    else
        return Tˢᶠᶜ - 0.004 * 740 - 0.003 * 1260 - 0.002 * (z - 2000)  # Free troposphere
    end
end

function qᵗᵢ(z)
    q₀ = 0.020    # Surface specific humidity [kg/kg] (~90% RH at SST)
    Hq = 3000     # Moisture scale height [m]
    q_min = 1e-6  # Minimum humidity
    return max(q₀ * exp(-z / Hq), q_min)
end

# Trade wind profile: ~5 m/s at surface, decreasing with height
uᵢ(x, z) = -5.0 * max(1 - z / 3000, 0)

# Random perturbations in the lowest 500 m to trigger convection
δT = 0.5
δq = 5e-4
zδ = 500

ϵ() = rand() - 0.5
Tᵢ_pert(x, z) = Tᵢ(z) + δT * ϵ() * (z < zδ)
qᵢ_pert(x, z) = qᵗᵢ(z) + δq * ϵ() * (z < zδ)

compute_reference_state!(reference_state, Tᵢ, qᵗᵢ, constants)
set!(model; T=Tᵢ_pert, qᵗ=qᵢ_pert, u=uᵢ)

T = model.temperature
qᵗ = model.specific_moisture
u, w = model.velocities.u, model.velocities.w
qˡ = model.microphysical_fields.qˡ

@info "Radiative Shallow Convection (2D)"
@info "Grid: $(Nx) × $(Nz), domain: $(Lx/1000) km × $(zᵗ/1000) km"
@info "Initial T range: $(minimum(T)) - $(maximum(T)) K"
@info "Initial qᵗ range: $(minimum(qᵗ)*1000) - $(maximum(qᵗ)*1000) g/kg"

# ## Simulation

simulation = Simulation(model; Δt=1, stop_time=4hours)
conjure_time_step_wizard!(simulation, cfl=0.5, max_Δt=5)

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    wmax = maximum(abs, w)
    Tmin, Tmax = extrema(T)
    qˡmax = maximum(qˡ)

    OLR = mean(view(radiation.upwelling_longwave_flux, :, 1, Nz+1))
    SW_in = -mean(view(radiation.downwelling_shortwave_flux, :, 1, Nz+1))

    msg = @sprintf("Iter: %5d, t: %8s, Δt: %5.1fs, wall: %8s",
                   iteration(sim), prettytime(sim), sim.Δt, prettytime(elapsed))
    msg *= @sprintf(", max|w|: %5.2f m/s, T: [%5.1f, %5.1f] K, max(qˡ): %.2e",
                   wmax, Tmin, Tmax, qˡmax)
    msg *= @sprintf(", OLR: %.1f W/m², SW_in: %.1f W/m²", OLR, SW_in)
    @info msg

    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# ## Output

qᵛ = model.microphysical_fields.qᵛ

outputs = (; u, w, T, qˡ, qᵛ)
avg_outputs = NamedTuple(name => Average(outputs[name], dims=1) for name in keys(outputs))

filename = "radiative_shallow_convection"
simulation.output_writers[:averages] = JLD2Writer(model, avg_outputs;
                                                  filename = filename * "_averages.jld2",
                                                  schedule = AveragedTimeInterval(30minutes),
                                                  overwrite_existing = true)

slice_outputs = (; w, qˡ, T)
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = filename * "_slices.jld2",
                                                schedule = TimeInterval(15minutes),
                                                overwrite_existing = true)

@info "Starting simulation..."
run!(simulation)
@info "Simulation completed!"
