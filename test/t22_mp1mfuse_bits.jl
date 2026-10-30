# T22-mp1mfuse: tendency and state hashes for MP1M models, to compare the switches
# BREEZE_FUSED_MICROPHYSICS_SEDIMENTATION and BREEZE_STABLE_SCALAR_LAUNCH on/off in SEPARATE processes
# (the switches are read in `__init__`). Terrain-following compressible (acoustic RK3, TKE) and anelastic
# rectilinear, Float32 and Float64, with zero, negative zero and negative precipitation cells.
# Usage: BREEZE_...=<bool> julia --project=<env> test/t22_mp1mfuse_bits.jl [CPU|GPU]
using Breeze, Oceananigans
using UUIDs: UUID
Base.require(Base.PkgId(UUID("6a9e3e04-43cd-43ba-94b9-e8782df3c71b"), "CloudMicrophysics"))
using CRC32c: crc32c
using Oceananigans.TimeSteppers: update_state!, time_step!
get(ARGS, 1, "CPU") == "GPU" && @eval using CUDA
const arch = get(ARGS, 1, "CPU") == "GPU" ? Base.invokelatest(GPU) : CPU()
const OneMomentCloudMicrophysics = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt).OneMomentCloudMicrophysics

function terrain_model(FT)
    Oceananigans.defaults.FloatType = FT
    z = TerrainFollowingVerticalDiscretization(collect(range(0, 12e3, length = 12)))
    grid = LatitudeLongitudeGrid(arch, FT; size = (6, 5, 11), halo = (5, 5, 5), z,
                                 longitude = (-100, -99.4), latitude = (35, 35.5),
                                 topology = (Bounded, Bounded, Bounded))
    materialize_terrain!(grid, (λ, φ) -> 600 * exp(-((λ + 99.7)^2 + (φ - 35.25)^2) / 0.15^2))
    cloud_formation = SaturationAdjustment(FT; equilibrium = MixedPhaseEquilibrium(FT))
    bounded = WENO(order = 5, bounds = (0, 1))
    model = AtmosphereModel(grid;
                            dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                                            reference_potential_temperature = 300),
                            microphysics = OneMomentCloudMicrophysics(FT; cloud_formation),
                            closure = TKEBasedTurbulenceClosure(),
                            momentum_advection = WENO(order = 5),
                            scalar_advection = (ρθ = WENO(order = 5), ρqᵉ = bounded, ρqʳ = bounded, ρqˢⁿ = bounded,
                                                ρe = Centered(order = 2)),
                            timestepper = :AcousticRungeKutta3)
    set!(model; ρ = model.dynamics.reference_state.density,
         θ = (λ, φ, z) -> 300 + 4e-3 * z + 0.5 * sin(3λ) * exp(-z / 3000),
         qᵗ = (λ, φ, z) -> 0.018 * exp(-z / 2500) * (1 + 0.3 * cos(5λ + 2φ)),
         u = (λ, φ, z) -> 8 + 1e-3 * z, v = (λ, φ, z) -> -3 + cos(3λ),
         ρe = (λ, φ, z) -> 0.5 * exp(-z / 1000),
         qʳ = (λ, φ, z) -> 1e-3 * exp(-((z - 3000) / 1500)^2) * (1 + sin(7λ)),
         qˢⁿ = (λ, φ, z) -> 5e-4 * exp(-(z - 6000)^2 / 2000^2))
    return model
end

function anelastic_model(FT)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(arch, FT; size = (6, 5, 16), halo = (5, 5, 5), x = (0, 6e3), y = (0, 5e3), z = (0, 8e3))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure = FT(101325), potential_temperature = FT(290))
    cloud_formation = SaturationAdjustment(FT; equilibrium = MixedPhaseEquilibrium(FT))
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), thermodynamic_constants = constants,
                            microphysics = OneMomentCloudMicrophysics(FT; cloud_formation))
    set!(model; θ = (x, y, z) -> 290 + 4e-3 * z + 0.3 * sin(x / 1e3),
         qᵗ = (x, y, z) -> 0.014 * exp(-z / 2500) * (1 + 0.2 * cos(y / 900)),
         qʳ = (x, y, z) -> 1e-3 * exp(-((z - 3000) / 1500)^2) * (1 + sin(x / 700)),
         qˢⁿ = (x, y, z) -> 8e-4 * exp(-((z - 5500) / 1200)^2), u = 3)
    return model
end

function special_values!(model, FT)
    ρqʳ = model.microphysical_fields.ρqʳ; ρqˢ = model.microphysical_fields.ρqˢⁿ
    a = Array(interior(ρqʳ)); b = Array(interior(ρqˢ))
    a[1, 1, 3] = 0; a[2, 1, 3] = -zero(FT); a[3, 1, 3] = FT(-1e-9); a[4, 1, 3] = FT(-1e-5)
    b[1, 2, 9] = 0; b[2, 2, 9] = -zero(FT); b[3, 2, 9] = FT(-1e-7)
    copyto!(interior(ρqʳ), a); copyto!(interior(ρqˢ), b)
    return nothing
end

crc(f) = string(crc32c(reinterpret(UInt8, vec(Array(interior(f))))), base = 16)

AMs = Breeze.AtmosphereModels
println("SWITCHES fused=$(AMs.FUSED_MICROPHYSICS_SEDIMENTATION[]) stable=$(AMs.STABLE_SCALAR_LAUNCH[])")
for (label, build, Δt) in (("terrain", terrain_model, 2), ("anelastic", anelastic_model, 2)), FT in (Float32, Float64)
    model = build(FT)
    special_values!(model, FT)
    update_state!(model)
    G = join([string(n, "=", crc(g)) for (n, g) in pairs(model.timestepper.Gⁿ)], " ")
    for _ in 1:3; time_step!(model, FT(Δt)); end
    S = join([string(n, "=", crc(f)) for (n, f) in pairs(Oceananigans.prognostic_fields(model))], " ")
    println("BITS $arch $label $FT G: $G | S: $S")
end
