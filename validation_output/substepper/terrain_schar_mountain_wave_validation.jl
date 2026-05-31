# Dry Schär mountain-wave validation for the terrain-following acoustic substepper.
#
# Usage:
#   julia --project=. validation_output/substepper/terrain_schar_mountain_wave_validation.jl
#
# Optional environment:
#   SCHAR_OUTPUT_DIR=validation_output/substepper/terrain_schar_mountain_wave
#   SCHAR_NX=128
#   SCHAR_NZ=48
#   SCHAR_STOP_SECONDS=600
#   SCHAR_DT=2
#   SCHAR_H0=250
#   SCHAR_A=5000
#   SCHAR_LAMBDA=4000
#   SCHAR_HILL_X_SHIFT=0
#   SCHAR_U=10
#   SCHAR_N=0.01
#   SCHAR_SPONGE_DEPTH=10000
#   SCHAR_SPONGE_RATE=0.1
#   SCHAR_ACOUSTIC_CFL=0.5
#   SCHAR_FORWARD_WEIGHT=0.65
#   SCHAR_APPLY_FIRST_SUBSTEP_PRESSURE_GRADIENT=false
#   SCHAR_SUBSTEP_DISTRIBUTION=proportional
#     options: proportional, monolithic_first_stage
#   SCHAR_ACOUSTIC_THETA_TENDENCY_FACTOR=1
#   SCHAR_ACOUSTIC_VERTICAL_MOMENTUM_TENDENCY_FACTOR=1
#   SCHAR_ACOUSTIC_VERTICAL_PRESSURE_TENDENCY_FACTOR=1
#   SCHAR_ACOUSTIC_FINAL_STAGE_VERTICAL_PRESSURE_TENDENCY_FACTOR=1
#   SCHAR_DIVERGENCE_DAMPING=thermal
#   SCHAR_DIVERGENCE_DAMPING_COEFFICIENT=0.1
#   SCHAR_DIVERGENCE_DAMPING_LENGTH_SCALE=auto
#   SCHAR_DIVERGENCE_DAMPING_VERTICAL=false
#   SCHAR_PROGNOSTIC_SPONGE=false
#   SCHAR_PROGNOSTIC_SPONGE_RATE=0.0033333333333333335
#   SCHAR_PRESSURE_GRADIENT_STENCIL=inside
#     options: inside, outside
#   SCHAR_MAKE_MOVIE=false
#   SCHAR_MOVIE_INTERVAL_SECONDS=60
#   SCHAR_WRITE_W_SNAPSHOT_CSVS=false
#   SCHAR_WRITE_FIELD_SNAPSHOT_CSVS=false
#   SCHAR_WRITE_ACOUSTIC_INCREMENT_BUDGET=false
#   SCHAR_WRITE_ENERGY_TIMESERIES=true
#   SCHAR_SKIP_RUN=false
#   SCHAR_ENERGY_INTERVAL_SECONDS=60
#
# Output:
#   terrain_schar_mountain_wave_metrics.csv
#   terrain_schar_mountain_wave_summary.txt
#   terrain_schar_mountain_wave_w_slice.csv
#   terrain_schar_mountain_wave_state_slice.csv
#   terrain_schar_mountain_wave_energy_timeseries.csv if SCHAR_WRITE_ENERGY_TIMESERIES=true
#   terrain_schar_mountain_wave_w_comparison.mp4 if SCHAR_MAKE_MOVIE=true
#
# This script provides a Breeze-side Schär diagnostic artifact. Cross-model
# comparisons can consume its CSV/summary outputs alongside WRF/MPAS/FV3 data.

using Breeze
using Breeze.CompressibleEquations: compute_contravariant_velocity!

# Allow GPU runs via `SCHAR_ARCH=gpu` (loads CUDA on demand). Defaults to CPU.
const SCHAR_ARCH = lowercase(get(ENV, "SCHAR_ARCH", "cpu"))
if SCHAR_ARCH == "gpu"
    @eval using CUDA
end
schar_arch() = SCHAR_ARCH == "gpu" ? GPU() : CPU()
# @allowscalar: when on GPU, allow scalar indexing of GPU arrays inside a
# block; when on CPU, a no-op pass-through. The macro lives in
# GPUArraysCore but we re-export it from CUDA in the GPU branch and define
# a no-op fallback for CPU runs.
if SCHAR_ARCH == "gpu"
    using CUDA: @allowscalar
else
    macro allowscalar(expr) esc(expr) end
end
using Breeze.TerrainFollowingDiscretization: SlopeInsideInterpolation, SlopeOutsideInterpolation,
                                              TerrainFollowingVerticalDiscretization,
                                              LinearDecay, SLEVE, materialize_terrain!,
                                              build_terrain_metrics
using Oceananigans
using Oceananigans.Grids: xnode, znode, rnode
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑzᵃᵃᶠ
using Oceananigans.TimeSteppers: update_state!
using KernelAbstractions: @kernel, @index
using Breeze.TerrainFollowingDiscretization: ∂z∂x
using Printf

# Initialize w to match CM1's IC: w is exactly zero everywhere except at
# the bottom face, where the kinematic terrain BC sets w(face k=1) = u · ∂h/∂x
# (so the flow follows the terrain surface). The interior REST IC (w=0 above
# the terrain face) is the same one CM1 uses for Schär — the wave develops
# from the kinematic forcing at the bottom rather than being pre-loaded
# through the column. Setting w throughout the column with a `b(ζ)` decay
# (as the old `set!(w = U·∂x_h·(1-z/Lz))` and the previous kernel did) makes
# Breeze launch with a column of physical w that CM1 doesn't have, which
# excites a non-physical acoustic mode.
@kernel function _init_terrain_bottom_face_w!(ρw, w, ρ, ρu, grid)
    i, j = @index(Global, NTuple)
    k = 1
    slope_x = ℑxᶜᵃᵃ(i, j, k, grid, ∂z∂x, Oceananigans.Face())
    ρu_ccf  = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, ρu)
    @inbounds begin
        ρw_target = slope_x * ρu_ccf
        ρ_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρw[i, j, k] = ρw_target
        w[i, j, k]  = ρw_target / ρ_ccf
    end
end

parse_env(::Type{T}, name, default) where T = parse(T, get(ENV, name, string(default)))
parse_env(::Type{Bool}, name, default) = lowercase(get(ENV, name, string(default))) in ("true", "1", "yes")

const OUTPUT_DIR = get(ENV, "SCHAR_OUTPUT_DIR",
                       "validation_output/substepper/terrain_schar_mountain_wave")

const Nx = parse_env(Int, "SCHAR_NX", 128)
const Nz = parse_env(Int, "SCHAR_NZ", 48)
const Lx = parse_env(Float64, "SCHAR_LX", 200e3)
const Lz = parse_env(Float64, "SCHAR_LZ", 30e3)
const stop_seconds = parse_env(Float64, "SCHAR_STOP_SECONDS", 600.0)
const Δt = parse_env(Float64, "SCHAR_DT", 2.0)

const h₀ = parse_env(Float64, "SCHAR_H0", 250.0)
const a = parse_env(Float64, "SCHAR_A", 5e3)
const λ = parse_env(Float64, "SCHAR_LAMBDA", 4e3)
const hill_x_shift = parse_env(Float64, "SCHAR_HILL_X_SHIFT", 0.0)
const thermodynamic_constants_name = lowercase(get(ENV, "SCHAR_THERMODYNAMIC_CONSTANTS", "breeze"))
const advection_name = lowercase(get(ENV, "SCHAR_ADVECTION", "centered2"))
const U = parse_env(Float64, "SCHAR_U", 10.0)
const N = parse_env(Float64, "SCHAR_N", 0.01)
const sponge_depth = parse_env(Float64, "SCHAR_SPONGE_DEPTH", 10e3)
const sponge_rate = parse_env(Float64, "SCHAR_SPONGE_RATE", 0.1)
const acoustic_cfl = parse_env(Float64, "SCHAR_ACOUSTIC_CFL", 0.5)
const forward_weight = parse_env(Float64, "SCHAR_FORWARD_WEIGHT", 0.65)
const apply_first_substep_pressure_gradient =
    parse_env(Bool, "SCHAR_APPLY_FIRST_SUBSTEP_PRESSURE_GRADIENT", false)
const substep_distribution_name = lowercase(get(ENV, "SCHAR_SUBSTEP_DISTRIBUTION", "proportional"))
const acoustic_theta_tendency_factor = parse_env(Float64, "SCHAR_ACOUSTIC_THETA_TENDENCY_FACTOR", 1.0)
const acoustic_vertical_momentum_tendency_factor =
    parse_env(Float64, "SCHAR_ACOUSTIC_VERTICAL_MOMENTUM_TENDENCY_FACTOR", 1.0)
const acoustic_vertical_pressure_tendency_factor =
    parse_env(Float64, "SCHAR_ACOUSTIC_VERTICAL_PRESSURE_TENDENCY_FACTOR", 1.0)
const acoustic_final_stage_vertical_pressure_tendency_factor =
    parse_env(Float64, "SCHAR_ACOUSTIC_FINAL_STAGE_VERTICAL_PRESSURE_TENDENCY_FACTOR", 1.0)
const divergence_damping_name = lowercase(get(ENV, "SCHAR_DIVERGENCE_DAMPING", "thermal"))
const divergence_damping_coefficient = parse_env(Float64, "SCHAR_DIVERGENCE_DAMPING_COEFFICIENT", 0.1)
const divergence_damping_length_scale_name = lowercase(get(ENV, "SCHAR_DIVERGENCE_DAMPING_LENGTH_SCALE", "auto"))
const divergence_damping_vertical = parse_env(Bool, "SCHAR_DIVERGENCE_DAMPING_VERTICAL", false)
const prognostic_sponge = parse_env(Bool, "SCHAR_PROGNOSTIC_SPONGE", false)
const prognostic_sponge_rate = parse_env(Float64, "SCHAR_PROGNOSTIC_SPONGE_RATE", 1 / 300)
const pressure_gradient_stencil_name = lowercase(get(ENV, "SCHAR_PRESSURE_GRADIENT_STENCIL", "inside"))
const make_movie = parse_env(Bool, "SCHAR_MAKE_MOVIE", false)
const movie_interval_seconds = parse_env(Float64, "SCHAR_MOVIE_INTERVAL_SECONDS", 60.0)
const write_w_snapshot_csvs = parse_env(Bool, "SCHAR_WRITE_W_SNAPSHOT_CSVS", false)
const write_field_snapshot_csvs = parse_env(Bool, "SCHAR_WRITE_FIELD_SNAPSHOT_CSVS", false)
const write_acoustic_increment_budget = parse_env(Bool, "SCHAR_WRITE_ACOUSTIC_INCREMENT_BUDGET", false)
const write_energy_timeseries = parse_env(Bool, "SCHAR_WRITE_ENERGY_TIMESERIES", true)
const skip_run = parse_env(Bool, "SCHAR_SKIP_RUN", false)
const energy_interval_seconds = parse_env(Float64, "SCHAR_ENERGY_INTERVAL_SECONDS", 60.0)

const p₀ = 100000.0
const θ₀ = 300.0
const pˢᵗ = 1e5

mkpath(OUTPUT_DIR)

function schar_thermodynamic_constants(name)
    name == "breeze" && return ThermodynamicConstants(Float64)

    if name == "cm1"
        molar_gas_constant = 8.314462618
        return ThermodynamicConstants(Float64;
                                      molar_gas_constant,
                                      dry_air_molar_mass = molar_gas_constant / 287.04,
                                      dry_air_heat_capacity = 1005.7,
                                      gravitational_acceleration = 9.81)
    end

    error("SCHAR_THERMODYNAMIC_CONSTANTS must be `breeze` or `cm1`, got `$name`")
end

constants = schar_thermodynamic_constants(thermodynamic_constants_name)
g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity
Rᵈ = Breeze.dry_air_gas_constant(constants)
β = g / (Rᵈ * θ₀)
N² = N^2
K = 2π / λ

@inline θ_of_z(z) = θ₀ * exp(N² * z / g)
function schar_advection(name)
    name in ("centered2", "centered") && return Centered(order = 2)
    name in ("weno9", "cm1") && return WENO(order = 9)
    error("SCHAR_ADVECTION must be `centered2`, `weno9`, or `cm1`, got `$name`")
end

function pressure_gradient_stencil(name)
    name == "inside" && return SlopeInsideInterpolation()
    name == "outside" && return SlopeOutsideInterpolation()
    error("SCHAR_PRESSURE_GRADIENT_STENCIL must be `inside` or `outside`, got `$name`")
end

const diagnostic_hill_x_shift = hill_x_shift

@inline terrain_shifted_x(x) = x + hill_x_shift
@inline diagnostic_shifted_x(x) = x + diagnostic_hill_x_shift
# SCHAR_MODULATION=cos2 (default) → classic Schär envelope·cos² modulation;
#                  =none           → plain Gaussian envelope (no cos² ripple)
const _schar_modulation = lowercase(get(ENV, "SCHAR_MODULATION", "cos2"))
@inline _modulation(x) = ifelse(_schar_modulation == "none", 1.0, cos(π * x / λ)^2)
@inline _modulation_deriv(x) = ifelse(_schar_modulation == "none", 0.0,
                                       -(π / λ) * sin(2π * x / λ))
@inline hill(x, y) = h₀ * exp(-(terrain_shifted_x(x) / a)^2) *
                     _modulation(terrain_shifted_x(x))
∂x_hill(x) = h₀ * exp(-(diagnostic_shifted_x(x) / a)^2) *
             (-2diagnostic_shifted_x(x) / a^2 *
              _modulation(diagnostic_shifted_x(x)) +
              _modulation_deriv(diagnostic_shifted_x(x)))

@inline function upper_sponge_weight(z)
    z_start = Lz - sponge_depth
    s = clamp((z - z_start) / sponge_depth, 0.0, 1.0)
    return s^2 * (3 - 2s)
end

function prognostic_sponge_forcing()
    rate = prognostic_sponge_rate
    z_start = Lz - sponge_depth
    inverse_sponge_depth = 1 / sponge_depth
    background_wind = U
    surface_potential_temperature = θ₀
    stratification_over_gravity = N² / g

    @inline sponge_weight(z) = begin
        s = clamp((z - z_start) * inverse_sponge_depth, 0, 1)
        s^2 * (3 - 2s)
    end

    Fρu = Forcing((x, z, t, ρu, ρ) -> -rate * sponge_weight(z) *
                                             (ρu - ρ * background_wind),
                  field_dependencies = (:ρu, :ρ))
    Fρw = Forcing((x, z, t, ρw) -> -rate * sponge_weight(z) * ρw,
                  field_dependencies = :ρw)
    Fρθ = Forcing((x, z, t, ρθ, ρ) -> -rate * sponge_weight(z) *
                                             (ρθ - ρ * surface_potential_temperature *
                                                    exp(stratification_over_gravity * z)),
                  field_dependencies = (:ρθ, :ρ))
    return (; ρu = Fρu, ρw = Fρw, ρθ = Fρθ)
end

function divergence_damping(name)
    name == "none" && return NoDivergenceDamping()

    if name == "thermal"
        length_scale = divergence_damping_length_scale_name in ("auto", "nothing") ?
                       nothing :
                       parse(Float64, divergence_damping_length_scale_name)
        return ThermalDivergenceDamping(; coefficient = divergence_damping_coefficient,
                                          length_scale,
                                          damp_vertical = divergence_damping_vertical)
    end

    error("SCHAR_DIVERGENCE_DAMPING must be `thermal` or `none`, got `$name`")
end

function substep_distribution(name)
    name in ("proportional", "default") && return ProportionalSubsteps()
    name in ("monolithic_first_stage", "monolithic", "first_stage") &&
        return MonolithicFirstStage()
    error("SCHAR_SUBSTEP_DISTRIBUTION must be `proportional` or `monolithic_first_stage`, got `$name`")
end

# Vertical coordinate: "sleve" uses the Schär et al. (2002) SLEVE formulation;
# "linear_tfvd" uses the Gal-Chen linear-decay formulation on the same
# TerrainFollowingVerticalDiscretization grid type.
const SCHAR_TERRAIN_COORDINATE = lowercase(get(ENV, "SCHAR_TERRAIN_COORDINATE", "sleve"))
const stencil = pressure_gradient_stencil(pressure_gradient_stencil_name)

SCHAR_TERRAIN_COORDINATE in ("sleve", "linear_tfvd") ||
    error("SCHAR_TERRAIN_COORDINATE must be `sleve` or `linear_tfvd`, got `$SCHAR_TERRAIN_COORDINATE`")

s₁ = parse_env(Float64, "SCHAR_SLEVE_S1", Lz / 2)
s₂ = parse_env(Float64, "SCHAR_SLEVE_S2", 2.5e3)
formulation = SCHAR_TERRAIN_COORDINATE == "sleve" ?
    SLEVE(large_scale_height = s₁, small_scale_height = s₂) : LinearDecay()
z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length = Nz + 1));
                                                 formulation = formulation)
grid = RectilinearGrid(schar_arch(); size = (Nx, Nz), halo = (5, 5),
                       x = (-Lx / 2, Lx / 2), z = z_faces,
                       topology = (Periodic, Flat, Bounded))
materialize_terrain!(grid, hill)
metrics = build_terrain_metrics(grid, stencil)

time_discretization = SplitExplicitTimeDiscretization(acoustic_cfl = acoustic_cfl,
                                                      forward_weight = forward_weight,
                                                      thermodynamic_tendency_factor = acoustic_theta_tendency_factor,
                                                      vertical_momentum_tendency_factor = acoustic_vertical_momentum_tendency_factor,
                                                      vertical_pressure_tendency_factor = acoustic_vertical_pressure_tendency_factor,
                                                      final_stage_vertical_pressure_tendency_factor = acoustic_final_stage_vertical_pressure_tendency_factor,
                                                      apply_first_substep_pressure_gradient = apply_first_substep_pressure_gradient,
                                                      substep_distribution = substep_distribution(substep_distribution_name),
                                                      damping = divergence_damping(divergence_damping_name),
                                                      sponge = UpperSponge(damping_rate = sponge_rate,
                                                                           depth = sponge_depth))

dynamics = CompressibleDynamics(time_discretization;
                                terrain_metrics = metrics,
                                reference_potential_temperature = θ_of_z,
                                surface_pressure = p₀,
                                standard_pressure = pˢᵗ)

model = AtmosphereModel(grid; dynamics,
                        thermodynamic_constants = constants,
                        timestepper = :AcousticRungeKutta3,
                        advection = schar_advection(advection_name),
                        forcing = prognostic_sponge ? prognostic_sponge_forcing() : NamedTuple())

set!(model,
     ρ = model.dynamics.terrain_reference_density,
     θ = (x, z) -> θ_of_z(z),
     u = U,
     v = 0,
     w = 0)
ρ = model.dynamics.density
# Initialize only the bottom-face w to match CM1's terrain kinematic condition.
# Interior w remains zero after `set!`, so the column is not pre-loaded with a
# formulation-decayed balanced vertical velocity. The kernel still uses the
# grid's own bottom-face slope so the bottom boundary condition matches the
# active terrain formulation.
let ρu = model.momentum.ρu, ρw = model.momentum.ρw,
    w  = model.velocities.w, ρ  = model.dynamics.density,
    grd = model.grid, arch = Oceananigans.Architectures.architecture(grd)

    Oceananigans.Utils.launch!(arch, grd, :xy, _init_terrain_bottom_face_w!,
                               ρw, w, ρ, ρu, grd)
end
update_state!(model)
compute_contravariant_velocity!(model)
initial_mass = sum(Array(interior(ρ)))

ĥ(k) = sqrt(π) * h₀ * a / 4 *
       (exp(-a^2 * (K + k)^2 / 4) +
        exp(-a^2 * (K - k)^2 / 4) +
        2exp(-a^2 * k^2 / 4))

m²(k) = N² / U^2 - β^2 / 4 - k^2
k★ = sqrt(max(0, N² / U^2 - β^2 / 4))

function w_linear(x, z; nk = 512)
    k_max = max(10 / a, 10 * K, 10 * k★)
    k = range(0, k_max, length = nk)
    Δk = step(k)

    integral = zero(Float64)
    for n in eachindex(k)
        kⁿ = k[n]
        weight = (n == firstindex(k) || n == lastindex(k)) ? 0.5 : 1.0
        m²ⁿ = m²(kⁿ)
        m_abs = sqrt(abs(m²ⁿ))
        phase = ifelse(m²ⁿ >= 0,
                       sin(m_abs * z + kⁿ * x),
                       exp(-m_abs * z) * sin(kⁿ * x))
        integral += weight * kⁿ * ĥ(kⁿ) * phase
    end

    return -(U / π) * exp(β * z / 2) * Δk * integral
end

function w_tilde_linear(x, z, k)
    ζ = rnode(k, grid, Face())
    slope_x = ∂x_hill(x) * (1 - ζ / Lz)
    return w_linear(x, z) - U * slope_x
end

function field_matrix(field)
    array = Array(interior(field))
    return ndims(array) == 3 ? dropdims(array; dims = 2) : array
end

function analytical_w_matrix()
    analytical = Matrix{Float64}(undef, Nx, Nz + 1)
    @allowscalar for i in 1:Nx, k in 1:Nz+1
        x = xnode(i, grid, Center())
        z = znode(i, 1, k, grid, Center(), Center(), Face())
        analytical[i, k] = w_linear(x + diagnostic_hill_x_shift, z)
    end
    return analytical
end

function pressure_perturbation_matrix(model)
    pressure = Array(interior(model.dynamics.pressure))
    reference_pressure = Array(interior(model.dynamics.terrain_reference_pressure))
    pressure_perturbation = pressure .- reference_pressure
    return ndims(pressure_perturbation) == 3 ?
           dropdims(pressure_perturbation; dims = 2) :
           pressure_perturbation
end

@inline function x_face_density_matrix(density, i, k)
    left = i == 1 ? Nx : i - 1
    return 0.5 * (density[left, k] + density[i, k])
end

function acoustic_increment_reference!(model)
    Breeze.TimeSteppers.compute_slow_momentum_tendencies!(model)
    initial_u = field_matrix(model.velocities.u)
    initial_density = field_matrix(model.dynamics.density)
    slow_momentum_tendency = field_matrix(model.timestepper.Gⁿ.ρu)
    return (; initial_u, initial_density, slow_momentum_tendency)
end

function write_acoustic_increment_budget!(filename, model, reference)
    final_u = field_matrix(model.velocities.u)

    open(filename, "w") do io
        println(io, "term,i,k,x,z,acceleration")
        @allowscalar for i in 1:Nx, k in 1:Nz
            x = xnode(i, grid, Face())
            z = znode(i, 1, k, grid, Face(), Center(), Center())
            ρ_face = x_face_density_matrix(reference.initial_density, i, k)
            slow_acceleration = reference.slow_momentum_tendency[i, k] / ρ_face
            outer_increment = (final_u[i, k] - reference.initial_u[i, k]) / Δt
            acoustic_increment = outer_increment - slow_acceleration
            println(io, join(("ub_acoustic_increment_velocity", i, k, x, z,
                              acoustic_increment), ","))
            println(io, join(("ub_outer_increment_velocity", i, k, x, z,
                              outer_increment), ","))
            println(io, join(("ub_slow_acceleration", i, k, x, z,
                              slow_acceleration), ","))
        end
    end

    return filename
end

snapshot_times = Float64[]
w_snapshots = Matrix{Float64}[]
pressure_snapshots = Matrix{Float64}[]
energy_rows = NamedTuple[]
acoustic_increment_file = joinpath(OUTPUT_DIR, "terrain_schar_mountain_wave_acoustic_increment_budget.csv")

simulation = Simulation(model; Δt, stop_time = stop_seconds)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

# Optional JLD2 snapshot writer for animations (`SCHAR_SNAPSHOT_INTERVAL=600`
# enables snapshots every 600 simulation-seconds). Outputs `w`, `u`, θ', p' as
# Center-staggered diagnostic fields so they pair directly with CM1's
# `winterp`, `uinterp`, `thpert`, `prspert` arrays.
const SCHAR_SNAPSHOT_INTERVAL = parse_env(Float64, "SCHAR_SNAPSHOT_INTERVAL", 0.0)
if SCHAR_SNAPSHOT_INTERVAL > 0
    using Oceananigans.OutputWriters: JLD2Writer, TimeInterval
    using Oceananigans.Fields: Field, @at
    snapshot_outputs = (
        u = model.velocities.u,
        w = model.velocities.w,
    )
    simulation.output_writers[:snapshots] = JLD2Writer(
        model, snapshot_outputs;
        filename     = joinpath(OUTPUT_DIR, "terrain_schar_snapshots.jld2"),
        schedule     = TimeInterval(SCHAR_SNAPSHOT_INTERVAL),
        overwrite_existing = true,
    )
end
walltime_start_ns = Ref(time_ns())
acoustic_increment_reference =
    write_acoustic_increment_budget ? acoustic_increment_reference!(model) : nothing

function spectral_high_k_fraction(values)
    count = length(values)
    count < 4 && return NaN
    mean_value = sum(values) / count
    total_power = 0.0
    high_power = 0.0

    for mode in 1:(count ÷ 2)
        real_part = 0.0
        imaginary_part = 0.0
        for index in eachindex(values)
            phase = 2π * mode * (index - 1) / count
            anomaly = values[index] - mean_value
            real_part += anomaly * cos(phase)
            imaginary_part -= anomaly * sin(phase)
        end

        power = real_part^2 + imaginary_part^2
        total_power += power
        mode > count / 4 && (high_power += power)
    end

    return high_power / max(eps(), total_power)
end

function collect_energy_row(simulation)
    compute_contravariant_velocity!(simulation.model)

    w = simulation.model.velocities.w
    u = simulation.model.velocities.u
    density = simulation.model.dynamics.density
    pressure = simulation.model.dynamics.pressure
    reference_pressure = simulation.model.dynamics.terrain_reference_pressure
    thermodynamic_density =
        Breeze.AtmosphereModels.thermodynamic_density(simulation.model.formulation)
    contravariant_vertical_velocity =
        simulation.model.dynamics.contravariant_vertical_velocity

    sponge_base = Lz - sponge_depth
    lower_limit = 0.5 * sponge_base

    lower_energy = 0.0
    below_sponge_energy = 0.0
    sponge_energy = 0.0
    lower_points = 0
    below_sponge_points = 0
    sponge_points = 0
    maximum_w = 0.0
    maximum_u = 0.0
    maximum_pressure_perturbation = 0.0
    maximum_cfl = 0.0
    maximum_acoustic_cfl = acoustic_cfl
    nan_count = 0
    inf_count = 0
    current_mass = 0.0
    domain_kinetic_energy = 0.0
    bottom_normal_velocity_max_abs = 0.0
    high_k_signal = Float64[]
    mountain_drag = 0.0
    Δx = Lx / Nx
    Δz = Lz / Nz

    @allowscalar for i in 1:Nx, k in 2:Nz
        z = znode(i, 1, k, grid, Center(), Center(), Face())
        w² = w[i, 1, k]^2

        if z <= lower_limit
            lower_energy += w²
            lower_points += 1
        elseif z <= sponge_base
            below_sponge_energy += w²
            below_sponge_points += 1
        else
            sponge_energy += w²
            sponge_points += 1
        end
    end

    @allowscalar for i in 1:Nx, k in 1:Nz
        w_center = 0.5 * (w[i, 1, k] + w[i, 1, k + 1])
        pressure_perturbation = pressure[i, 1, k] - reference_pressure[i, 1, k]
        maximum_w = max(maximum_w, abs(w_center))
        maximum_u = max(maximum_u, abs(u[i, 1, k]))
        maximum_pressure_perturbation =
            max(maximum_pressure_perturbation, abs(pressure_perturbation))
        maximum_cfl = max(maximum_cfl,
                          Δt * (abs(u[i, 1, k]) / Δx + abs(w_center) / Δz))
        current_mass += density[i, 1, k]
        domain_kinetic_energy += 0.5 * density[i, 1, k] *
                                 (u[i, 1, k]^2 + w_center^2)
        values = (density[i, 1, k], u[i, 1, k], w_center,
                  thermodynamic_density[i, 1, k], pressure[i, 1, k])
        nan_count += count(isnan, values)
        inf_count += count(isinf, values)
        k == 1 && push!(high_k_signal, w_center)
    end

    @allowscalar for i in 1:Nx
        x = xnode(i, grid, Center())
        pressure_perturbation = pressure[i, 1, 1] - reference_pressure[i, 1, 1]
        mountain_drag += pressure_perturbation * ∂x_hill(x) * (Lx / Nx)
        bottom_normal_velocity_max_abs =
            max(bottom_normal_velocity_max_abs,
                abs(contravariant_vertical_velocity[i, 1, 1]))
    end

    lower_energy /= max(1, lower_points)
    below_sponge_energy /= max(1, below_sponge_points)
    sponge_energy /= max(1, sponge_points)
    domain_kinetic_energy /= Nx * Nz

    return (; time = time(simulation),
              maximum_w,
              maximum_u,
              maximum_pressure_perturbation,
              domain_kinetic_energy,
              mass_relative_drift =
                  (current_mass - initial_mass) / max(eps(), abs(initial_mass)),
              maximum_cfl,
              maximum_acoustic_cfl,
              nan_count,
              inf_count,
              bottom_normal_velocity_max_abs,
              high_k_energy_fraction_near_terrain =
                  spectral_high_k_fraction(high_k_signal),
              reflection_energy_fraction_above_sponge_start =
                  sponge_energy / max(eps(), below_sponge_energy),
              walltime_per_step =
                  1e-9 * (time_ns() - walltime_start_ns[]) /
                  max(1, simulation.model.clock.iteration),
              mountain_drag,
              mountain_drag_p_dhdx = mountain_drag,
              mountain_drag_negative_p_dhdx = -mountain_drag,
              lower_energy,
              below_sponge_energy,
              sponge_energy,
              below_sponge_to_lower_energy =
                  below_sponge_energy / max(eps(), lower_energy),
              sponge_to_lower_energy =
                  sponge_energy / max(eps(), lower_energy),
              lower_points,
              below_sponge_points,
              sponge_points)
end

function collect_energy_row!(simulation)
    current_time = time(simulation)
    if !isempty(energy_rows) && last(energy_rows).time == current_time
        return nothing
    end

    push!(energy_rows, collect_energy_row(simulation))
    return nothing
end

if write_energy_timeseries
    energy_interval = max(1, round(Int, energy_interval_seconds / Δt))
    add_callback!(simulation, collect_energy_row!, name = :schar_energy_timeseries,
                  IterationInterval(energy_interval))
    collect_energy_row!(simulation)
end

if make_movie
    snapshot_interval = max(1, round(Int, movie_interval_seconds / Δt))

    function collect_w_snapshot!(simulation)
        current_time = time(simulation)
        if !isempty(snapshot_times) && last(snapshot_times) == current_time
            return nothing
        end

        push!(snapshot_times, current_time)
        push!(w_snapshots, copy(field_matrix(simulation.model.velocities.w)))
        push!(pressure_snapshots, copy(pressure_perturbation_matrix(simulation.model)))
        return nothing
    end

    add_callback!(simulation, collect_w_snapshot!, name = :schar_w_snapshot,
                  IterationInterval(snapshot_interval))
    collect_w_snapshot!(simulation)
end

if get(ENV, "SCHAR_TRACE_MAX", "false") == "true"
    function _trace_max!(sim)
        m = sim.model
        ρ  = m.dynamics.density
        u  = m.velocities.u
        w  = m.velocities.w
        ρθ = Breeze.AtmosphereModels.thermodynamic_density(m.formulation)
        p  = m.dynamics.pressure
        @info @sprintf("trace iter=%4d  t=%9.3f  max|ρ|=%.6e  max|u|=%.6e  max|w|=%.6e  max|ρθ|=%.6e  max|p|=%.6e",
                       m.clock.iteration, m.clock.time,
                       maximum(abs, interior(ρ)),
                       maximum(abs, interior(u)),
                       maximum(abs, interior(w)),
                       maximum(abs, interior(ρθ)),
                       maximum(abs, interior(p)))
        return nothing
    end
    add_callback!(simulation, _trace_max!, name = :_trace_max,
                  IterationInterval(1))
end

walltime_start_ns[] = time_ns()
run_start_ns = walltime_start_ns[]
skip_run || run!(simulation)
run_wall_clock_seconds = 1e-9 * (time_ns() - run_start_ns)
wall_clock_seconds_per_simulated_hour =
    run_wall_clock_seconds / max(eps(), stop_seconds / 3600)

if write_energy_timeseries && (isempty(energy_rows) || last(energy_rows).time < time(simulation))
    collect_energy_row!(simulation)
end

if make_movie && (isempty(snapshot_times) || last(snapshot_times) < time(simulation))
    push!(snapshot_times, time(simulation))
    push!(w_snapshots, copy(field_matrix(model.velocities.w)))
    push!(pressure_snapshots, copy(pressure_perturbation_matrix(model)))
end

if write_acoustic_increment_budget
    write_acoustic_increment_budget!(acoustic_increment_file, model,
                                     acoustic_increment_reference)
end

function collect_wave_diagnostics(model)
    compute_contravariant_velocity!(model)

    w = model.velocities.w
    u = model.velocities.u
    density = model.dynamics.density
    pressure = model.dynamics.pressure
    reference_pressure = model.dynamics.terrain_reference_pressure
    thermodynamic_density = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
    contravariant_vertical_velocity = model.dynamics.contravariant_vertical_velocity

    squared_error = 0.0
    squared_reference = 0.0
    maximum_w = 0.0
    maximum_reference_w = 0.0
    maximum_u = 0.0
    maximum_cfl = 0.0
    maximum_acoustic_cfl = acoustic_cfl
    nan_count = 0
    inf_count = 0
    current_mass = 0.0
    bottom_normal_velocity_max_abs = 0.0
    high_k_signal = Float64[]
    momentum_flux = 0.0
    mountain_drag = 0.0
    n = 0
    Δx = Lx / Nx
    Δz = Lz / Nz

    @allowscalar for i in 1:Nx, k in 2:Nz
        z = znode(i, 1, k, grid, Center(), Center(), Face())
        z > Lz - sponge_depth && continue

        x = xnode(i, grid, Center())
        simulated_w = w[i, 1, k]
        reference_w = w_linear(x + diagnostic_hill_x_shift, z)
        ρᶠ = 0.5 * (density[i, 1, k - 1] + density[i, 1, k])
        uᶠ = 0.5 * (u[i, 1, k - 1] + u[i, 1, k])

        squared_error += (simulated_w - reference_w)^2
        squared_reference += reference_w^2
        maximum_w = max(maximum_w, abs(simulated_w))
        maximum_reference_w = max(maximum_reference_w, abs(reference_w))
        momentum_flux += ρᶠ * (uᶠ - U) * simulated_w
        n += 1
    end

    @allowscalar for i in 1:Nx, k in 1:Nz
        w_center = 0.5 * (w[i, 1, k] + w[i, 1, k + 1])
        maximum_u = max(maximum_u, abs(u[i, 1, k]))
        maximum_cfl = max(maximum_cfl,
                          Δt * (abs(u[i, 1, k]) / Δx + abs(w_center) / Δz))
        current_mass += density[i, 1, k]
        values = (density[i, 1, k], u[i, 1, k], w_center,
                  thermodynamic_density[i, 1, k], pressure[i, 1, k])
        nan_count += count(isnan, values)
        inf_count += count(isinf, values)
        k == 1 && push!(high_k_signal, w_center)
    end

    @allowscalar for i in 1:Nx
        x = xnode(i, grid, Center())
        pressure_perturbation = pressure[i, 1, 1] - reference_pressure[i, 1, 1]
        mountain_drag += pressure_perturbation * ∂x_hill(x) * (Lx / Nx)
        bottom_normal_velocity_max_abs =
            max(bottom_normal_velocity_max_abs,
                abs(contravariant_vertical_velocity[i, 1, 1]))
    end

    normalized_rmse = sqrt(squared_error / max(eps(), squared_reference))
    amplitude_error = abs(maximum_w - maximum_reference_w) / max(eps(), maximum_reference_w)
    mean_momentum_flux = momentum_flux / n
    mountain_drag_p_dhdx = mountain_drag
    mountain_drag_negative_p_dhdx = -mountain_drag

    return (; normalized_rmse,
              amplitude_error,
              maximum_w,
              maximum_reference_w,
              maximum_u,
              maximum_cfl,
              maximum_acoustic_cfl,
              nan_count,
              inf_count,
              mass_relative_drift =
                  (current_mass - initial_mass) / max(eps(), abs(initial_mass)),
              bottom_normal_velocity_max_abs,
              high_k_energy_fraction_near_terrain =
                  spectral_high_k_fraction(high_k_signal),
              mean_momentum_flux,
              mountain_drag,
              mountain_drag_p_dhdx,
              mountain_drag_negative_p_dhdx,
              mountain_drag_atmosphere_on_mountain = mountain_drag_p_dhdx,
              mountain_drag_mountain_on_atmosphere = mountain_drag_negative_p_dhdx,
              wall_clock_seconds = run_wall_clock_seconds,
              wall_clock_seconds_per_simulated_hour,
              walltime_per_step =
                  run_wall_clock_seconds / max(1, model.clock.iteration),
              stable_dt = Δt,
              samples = n)
end

diagnostics = collect_wave_diagnostics(model)
plot_file = joinpath(OUTPUT_DIR, "terrain_schar_mountain_wave_w_comparison.ppm")
plot_frames_dir = joinpath(OUTPUT_DIR, "terrain_schar_mountain_wave_w_comparison_frames")
snapshot_csv_dir = joinpath(OUTPUT_DIR, "terrain_schar_mountain_wave_w_snapshot_csvs")
field_snapshot_csv_dir = joinpath(OUTPUT_DIR, "terrain_schar_mountain_wave_field_snapshot_csvs")

slice_file = joinpath(OUTPUT_DIR, "terrain_schar_mountain_wave_w_slice.csv")
open(slice_file, "w") do io
    println(io, "i,k,x,z,w,w_linear,w_error,w_tilde,w_tilde_linear,w_tilde_error")
    w = model.velocities.w
    w̃ = model.dynamics.contravariant_vertical_velocity
    @allowscalar for i in 1:Nx, k in 1:Nz+1
        x = xnode(i, grid, Center())
        z = znode(i, 1, k, grid, Center(), Center(), Face())
        simulated_w = w[i, 1, k]
        reference_w = w_linear(x + diagnostic_hill_x_shift, z)
        simulated_w_tilde = w̃[i, 1, k]
        reference_w_tilde = w_tilde_linear(x + diagnostic_hill_x_shift, z, k)
        println(io, join((i, k, x, z, simulated_w, reference_w,
                          simulated_w - reference_w,
                          simulated_w_tilde, reference_w_tilde,
                          simulated_w_tilde - reference_w_tilde), ","))
    end
end

state_slice_file = joinpath(OUTPUT_DIR, "terrain_schar_mountain_wave_state_slice.csv")
open(state_slice_file, "w") do io
    println(io, "i,k,x,z,u,w_center,theta_perturbation,pressure_perturbation")
    w = model.velocities.w
    u = model.velocities.u
    pressure = model.dynamics.pressure
    reference_pressure = model.dynamics.terrain_reference_pressure
    density = model.dynamics.density
    thermodynamic_density = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)

    @allowscalar for i in 1:Nx, k in 1:Nz
        x = xnode(i, grid, Center())
        z = znode(i, 1, k, grid, Center(), Center(), Center())
        w_center = 0.5 * (w[i, 1, k] + w[i, 1, k + 1])
        θ = thermodynamic_density[i, 1, k] / density[i, 1, k]
        θ′ = θ - θ_of_z(z)
        p′ = pressure[i, 1, k] - reference_pressure[i, 1, k]
        println(io, join((i, k, x, z, u[i, 1, k], w_center, θ′, p′), ","))
    end
end

energy_timeseries_file = joinpath(OUTPUT_DIR, "terrain_schar_mountain_wave_energy_timeseries.csv")
if write_energy_timeseries
    open(energy_timeseries_file, "w") do io
        println(io, join(propertynames(first(energy_rows)), ","))
        for row in energy_rows
            println(io, join((getproperty(row, name) for name in propertynames(row)), ","))
        end
    end
end

function color_triplet(value, limit)
    scaled = clamp(value / max(eps(), limit), -1, 1)
    if scaled >= 0
        red = 255
        green_blue = round(UInt8, 255 * (1 - scaled))
        return UInt8(red), green_blue, green_blue
    else
        blue = 255
        red_green = round(UInt8, 255 * (1 + scaled))
        return red_green, red_green, UInt8(blue)
    end
end

function write_heatmap_panel!(io, matrix, limit)
    Nx_panel, Nz_panel = size(matrix)
    for k in Nz_panel:-1:1, i in 1:Nx_panel
        red, green, blue = color_triplet(matrix[i, k], limit)
        write(io, red)
        write(io, green)
        write(io, blue)
    end
    return nothing
end

function write_comparison_ppm!(filename, simulated, analytical, color_limit)
    difference = simulated .- analytical
    width = size(simulated, 1)
    panel_height = size(simulated, 2)
    height = 3 * panel_height

    open(filename, "w") do io
        write(io, "P6\n$width $height\n255\n")
        write_heatmap_panel!(io, simulated, color_limit)
        write_heatmap_panel!(io, analytical, color_limit)
        write_heatmap_panel!(io, difference, color_limit)
    end

    return filename
end

function write_validation_plots!(filename, frames_dir, snapshot_times, w_snapshots)
    analytical = analytical_w_matrix()
    color_limit = maximum(abs, analytical)
    for snapshot in w_snapshots
        color_limit = max(color_limit, maximum(abs, snapshot))
    end
    color_limit = max(color_limit, eps())

    mkpath(frames_dir)
    for n in eachindex(w_snapshots)
        frame_file = joinpath(frames_dir, @sprintf("frame_%04d.ppm", n))
        write_comparison_ppm!(frame_file, w_snapshots[n], analytical, color_limit)
    end

    write_comparison_ppm!(filename, last(w_snapshots), analytical, color_limit)
    return filename, frames_dir
end

function write_snapshot_csvs!(snapshot_csv_dir, snapshot_times, w_snapshots)
    mkpath(snapshot_csv_dir)

    time_file = joinpath(snapshot_csv_dir, "snapshot_times.csv")
    open(time_file, "w") do io
        println(io, "frame,time_seconds")
        for n in eachindex(snapshot_times)
            @printf(io, "%d,%.7e\n", n, snapshot_times[n])
        end
    end

    for n in eachindex(w_snapshots)
        snapshot_file = joinpath(snapshot_csv_dir, @sprintf("w_snapshot_%04d.csv", n))
        snapshot = w_snapshots[n]

        open(snapshot_file, "w") do io
            println(io, "i,k,x,z,w")
            @allowscalar for i in 1:Nx, k in 1:Nz+1
                x = xnode(i, grid, Center())
                z = znode(i, 1, k, grid, Center(), Center(), Face())
                println(io, join((i, k, x, z, snapshot[i, k]), ","))
            end
        end
    end

    return snapshot_csv_dir
end

function write_field_snapshot_csvs!(field_snapshot_csv_dir, snapshot_times,
                                    w_snapshots, pressure_snapshots)
    length(snapshot_times) == length(w_snapshots) == length(pressure_snapshots) ||
        error("snapshot count mismatch: times=$(length(snapshot_times)), w=$(length(w_snapshots)), pressure=$(length(pressure_snapshots))")

    mkpath(field_snapshot_csv_dir)

    time_file = joinpath(field_snapshot_csv_dir, "snapshot_times.csv")
    open(time_file, "w") do io
        println(io, "frame,time_seconds")
        for n in eachindex(snapshot_times)
            @printf(io, "%d,%.7e\n", n, snapshot_times[n])
        end
    end

    for n in eachindex(snapshot_times)
        snapshot_file = joinpath(field_snapshot_csv_dir, @sprintf("field_snapshot_%04d.csv", n))
        w_snapshot = w_snapshots[n]
        pressure_snapshot = pressure_snapshots[n]

        open(snapshot_file, "w") do io
            println(io, "i,k,x,z,w_center,pressure_perturbation")
            @allowscalar for i in 1:Nx, k in 1:Nz
                x = xnode(i, grid, Center())
                z = znode(i, 1, k, grid, Center(), Center(), Center())
                w_center = 0.5 * (w_snapshot[i, k] + w_snapshot[i, k + 1])
                println(io, join((i, k, x, z, w_center, pressure_snapshot[i, k]), ","))
            end
        end
    end

    return field_snapshot_csv_dir
end

if make_movie
    write_validation_plots!(plot_file, plot_frames_dir, snapshot_times, w_snapshots)
    write_w_snapshot_csvs && write_snapshot_csvs!(snapshot_csv_dir, snapshot_times, w_snapshots)
    write_field_snapshot_csvs &&
        write_field_snapshot_csvs!(field_snapshot_csv_dir, snapshot_times,
                                   w_snapshots, pressure_snapshots)
end

metrics_file = joinpath(OUTPUT_DIR, "terrain_schar_mountain_wave_metrics.csv")
open(metrics_file, "w") do io
    println(io, "metric,value")
    for name in propertynames(diagnostics)
        println(io, "$(name),$(getproperty(diagnostics, name))")
    end
end

summary_file = joinpath(OUTPUT_DIR, "terrain_schar_mountain_wave_summary.txt")
open(summary_file, "w") do io
    println(io, "Terrain-following Schär mountain-wave validation")
    println(io)
    println(io, "Configuration")
    @printf(io, "  Nx, Nz = %d, %d\n", Nx, Nz)
    @printf(io, "  Lx, Lz = %.6e, %.6e m\n", Lx, Lz)
    @printf(io, "  stop time = %.6e s, Δt = %.6e s\n", stop_seconds, Δt)
    @printf(io, "  U = %.6e m/s, N = %.6e 1/s\n", U, N)
    @printf(io, "  h0 = %.6e m, a = %.6e m, lambda = %.6e m\n", h₀, a, λ)
    @printf(io, "  hill x shift = %.6e m\n", hill_x_shift)
    println(io, "  terrain coordinate = $SCHAR_TERRAIN_COORDINATE")
    @printf(io, "  diagnostic hill x shift = %.6e m\n", diagnostic_hill_x_shift)
    println(io, "  substep distribution = $substep_distribution_name")
    @printf(io, "  acoustic theta tendency factor = %.6e\n", acoustic_theta_tendency_factor)
    @printf(io, "  sponge depth = %.6e m, sponge rate = %.6e 1/s\n", sponge_depth, sponge_rate)
    println(io, "  prognostic sponge = $prognostic_sponge")
    @printf(io, "  prognostic sponge rate = %.6e 1/s\n", prognostic_sponge_rate)
    @printf(io, "  k_star = %.6e 1/m\n", k★)
    @printf(io, "  wall clock = %.6e s\n", run_wall_clock_seconds)
    @printf(io, "  wall-clock seconds per simulated hour = %.6e\n",
            wall_clock_seconds_per_simulated_hour)
    println(io)
    println(io, "Diagnostics")
    for name in propertynames(diagnostics)
        value = getproperty(diagnostics, name)
        value isa Integer ? println(io, "  $name = $value") :
            @printf(io, "  %s = %.9e\n", string(name), value)
    end
    println(io)
    println(io, "Acceptance targets from terrain_following_substepper_plan.md")
    println(io, "  normalized RMSE of w versus reference or fine-grid Breeze < 10%")
    println(io, "  vertical velocity amplitude error < 5% to 10%")
    println(io, "  mountain drag error < 10% to 20%")
    println(io, "  monotone refinement should reduce w, θ', and drag errors")
    println(io, "  this script reports diagnostics but does not hard-fail by default")
    println(io)
    println(io, "Final-time w slice")
    println(io, "  $slice_file")
    println(io)
    println(io, "Final-time state slice")
    println(io, "  $state_slice_file")
    if write_energy_timeseries
        println(io)
        println(io, "Wave-energy time series")
        println(io, "  $energy_timeseries_file")
    end
    if write_acoustic_increment_budget
        println(io)
        println(io, "Acoustic increment budget")
        println(io, "  $acoustic_increment_file")
    end
    if make_movie
        println(io)
        println(io, "Validation plot")
        println(io, "  $plot_file")
        println(io, "Validation plot frames")
        println(io, "  $plot_frames_dir")
        if write_w_snapshot_csvs
            println(io, "Raw w snapshot CSVs")
            println(io, "  $snapshot_csv_dir")
        end
        if write_field_snapshot_csvs
            println(io, "Raw w/pressure snapshot CSVs")
            println(io, "  $field_snapshot_csv_dir")
        end
    end
end

@info "wrote $metrics_file"
@info "wrote $summary_file"
@info "wrote $slice_file"
@info "wrote $state_slice_file"
write_energy_timeseries && @info "wrote $energy_timeseries_file"
write_acoustic_increment_budget && @info "wrote $acoustic_increment_file"
make_movie && @info "wrote $plot_file"
