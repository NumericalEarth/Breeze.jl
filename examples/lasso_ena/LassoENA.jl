# LASSO operators adapted from BreezeLab; see NOTICE and LICENSE in this directory.
module LassoENA
using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znodes
using Oceananigans.Fields: interior
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticlePropertiesMicrophysics,
    CloudDroplets, AerosolActivation, AerosolMode, has_prognostic_aerosol, activated_number
using Dates: DateTime
using SHA: sha256
using NumericalRadiation
using NCDatasets

include("sam_input_files.jl")
include("initial_conditions.jl")
include("forcing_profiles.jl")
include("large_scale_forcings.jl")
include("sam_ocean_surface_fluxes.jl")
include("microphysics.jl")
include("advection.jl")

const MEMBER = "20170718era5d25x100_sbmwrm-aer2-flxsst"
const INPUT_HASHES = (
    snd = "5b8338069f8cc8205364179b162bb1bbf91f20a3cd543b19e91786944784db08",
    lsf = "32cae2de038c8007e23b6a177a4a4c437b0644b30ec8e7f1556dfe9d843e9567",
    sfc = "6b9f5ee0fdd7b0a4e52222acd889dbdd4b2a52d281e9c6743d9d5b6a15869b7c",
    prm = "738be3202e199e7b5fa17c5075291d5aa1f67082b5f6c35b76e9bfebb9fd7505",
    grd = "fb217f6cc3768d0ac90ddab031751a94df0ca617567709a65df2838ee4737ef1",
)

# These hashes identify the staged ARM samin files, not the unrelated Covert inputs.
function validate_inputs(directory)
    for (name, expected) in pairs(INPUT_HASHES)
        path = joinpath(directory, string(name))
        isfile(path) || throw(ArgumentError("Missing $path. Stage ARM LASSO member $MEMBER (DOI 10.5439/2572661) and set ENA_LASSO_BUNDLE."))
        bytes2hex(sha256(read(path))) == expected || throw(ArgumentError("LASSO checksum mismatch: $path"))
    end
    return nothing
end

# In SAM's grd, cell faces follow recursively from the supplied cell centers.
function grid_faces(centers)
    faces = zeros(length(centers) + 1)
    for k in eachindex(centers)
        faces[k+1] = 2centers[k] - faces[k]
    end
    return faces
end

# Refresh callback-owned state in both Simulation and the differentiated loop.
# The SST field and aerosol fields belong to the model, so their mutations are traced.
function refresh_inputs!(model, sst)
    time = model.clock.time
    set!(model.radiation.surface_radiation.surface_temperature,
         (x, y) -> sst[1, 1, 1, Time(time)])
    fill_halo_regions!(model.radiation.surface_radiation.surface_temperature)
    reservoir = sum(mode.number_mixing_ratio for mode in model.microphysics.aerosol.modes)
    DiagnosticCCNProjection(model, reservoir)((; model))
    return nothing
end

function build_case(directory, arch; Nx=32, Ny=32)
    validate_inputs(directory)
    namelist = read_sam_namelist(joinpath(directory, "prm"))
    day0 = namelist["day0"]
    soundings = read_sam_sounding(joinpath(directory, "snd"))
    lsf = read_sam_large_scale_forcing(joinpath(directory, "lsf"))
    sfc = read_sam_surface_forcing(joinpath(directory, "sfc"))
    # Retain all 260 vertical cells and the 100 m horizontal spacing. Only the
    # horizontal extent is reduced for this differentiation experiment.
    faces = grid_faces(read_sam_grd(joinpath(directory, "grd"))[1:260])
    grid = RectilinearGrid(arch; size=(Nx, Ny, 260), x=(0, 100Nx), y=(0, 100Ny),
                          z=faces, halo=(5, 5, 5), topology=(Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants()
    profiles = SoundingProfiles(initial_sounding(soundings, day0); moisture_basis=:mixing_ratio)
    reference = sounding_reference_state(grid, profiles, constants)
    z = Array(znodes(grid, Center()))
    p = Array(interior(reference.pressure, 1, 1, :))
    density = Array(interior(reference.density, 1, 1, :))
    aerosol = lasso_aerosol(; reference_density=density[1], maximum_supersaturation=0.003)
    microphysics = P3Microphysics(; cloud=CloudDroplets(; number_concentration=75e6), aerosol)
    series = LargeScaleForcingProfiles(grid, lsf, z, p; day0)
    geostrophic = time_varying_geostrophic_forcings(series.ug, series.vg)
    subsidence = LargeScaleVerticalAdvection(series.wls)
    thermo = large_scale_thermodynamic_forcings(series.tls, series.qls;
        microphysics, thermodynamic_constants=constants, moisture_name=:qᵛ, moisture_basis=:mixing_ratio)
    targets = SoundingTargetProfiles(grid, soundings, z, p; day0, moisture_basis=:mixing_ratio)
    upper = upper_boundary_relaxation_forcings(targets.T, targets.q;
        microphysics, thermodynamic_constants=constants, moisture_name=:qᵛ)
    sponge = SAMSponge()
    forcing = Dict{Symbol, Tuple}(
        :u => (geostrophic.u, MeanProfileNudging(series.uls; timescale=7200), subsidence, sponge),
        :v => (geostrophic.v, MeanProfileNudging(series.vls; timescale=7200), subsidence, sponge),
        :w => (sponge,), :E => (thermo.E, upper.E), :θ => (subsidence,))
    for name in specific_prognostic_names(microphysics)
        forcing[name] = (subsidence,)
    end
    forcing[:qᵛ] = (thermo.qᵛ, subsidence, upper.qᵛ)
    sst = profile_time_series(grid, (sfc.day .- day0) .* 86400,
                              [fill(T, length(z)) for T in sfc.sst])
    surface_temperature = Field{Center, Center, Nothing}(grid)
    set!(surface_temperature, interpolate_profile((sfc.day .- day0) .* 86400, sfc.sst, 0))
    fill_halo_regions!(surface_temperature)
    surface_density = sam_surface_interface_density(p, z, constants.gravitational_acceleration)
    boundary_conditions = sam_ocean_surface_flux_boundary_conditions(grid, surface_temperature;
        moisture_name=:qᵛ, microphysics, thermodynamic_constants=constants, surface_density)
    radiation = RadiativeTransferModel(grid, EcCKDOptics(; clouds=CloudScatteringTables()), constants;
        surface_temperature, surface_albedo=0.07, surface_emissivity=0.95,
        background_atmosphere=BackgroundAtmosphere(CO₂=405e-6, CH₄=1.85e-6, N₂O=330e-9),
        solar_position=ApparentSolarPosition(; coordinate=(-28.0257, 39.0916), epoch=DateTime(2017, 7, 18)),
        schedule=IterationInterval(30), liquid_effective_radius=ConstantRadiusParticles(14e-6),
        ice_effective_radius=ConstantRadiusParticles(30e-6))
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference),
        formulation=:LiquidIcePotentialTemperature, microphysics, radiation,
        forcing=compact_forcing(forcing), boundary_conditions, thermodynamic_constants=constants,
        coriolis=FPlane(f=9.19626e-5), closure=SmagorinskyLilly(), momentum_advection=WENO(order=5),
        scalar_advection=scalar_advection_schemes(5, microphysics, :qᵛ))
    columns = initial_state_columns(profiles, z, p; constants=ThermodynamicConstants(Float64))
    set_sounding_initial_state!(model, columns; perturbation=InitialPerturbation(; sam_perturb_type=5),
        moisture_basis=:mixing_ratio, initialization=:condensate_free)
    return (; model, sst, faces, z)
end
end
