## Benchmark: cost of one P3 microphysics update on a small grid (CPU vs GPU).
##
## Builds an AtmosphereModel with P3 over a 32×32×32 mixed-phase column,
## runs `microphysics_model_update!` repeatedly, and reports per-cell time.
## Compares against the same setup with DCMIP2016 Kessler microphysics so we
## have a calibration point for "fast" mixed-phase-relevant behavior.
##
## Run:
##   julia --project=benchmarking benchmarking/p3_microphysics_step.jl
## or, with CUDA disabled (CPU only):
##   CUDA_VISIBLE_DEVICES=-1 julia --project=benchmarking benchmarking/p3_microphysics_step.jl

using Breeze
using Breeze: PredictedParticlePropertiesMicrophysics, DCMIP2016KesslerMicrophysics, TetensFormula
using Breeze.AtmosphereModels: microphysics_model_update!
using Oceananigans: Oceananigans, RectilinearGrid, CPU, GPU, Periodic, Bounded
using Oceananigans.Architectures: architecture
using Printf
using Statistics

const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

# ----------------------------------------------------------------------
# Background atmosphere helpers (mirrors examples/splitting_supercell_p3.jl)
# ----------------------------------------------------------------------

const θ₀ = 300.0
const θᵖ = 343.0
const zᵖ = 12000.0
const Tᵖ = 213.0

θ_background(z) = (z ≤ zᵖ) * (θ₀ + (θᵖ - θ₀) * (z / zᵖ)^(5/4)) +
                  (z >  zᵖ) * (θᵖ * exp(9.81 / (1004 * Tᵖ) * (z - zᵖ)))

ℋ_background(z) = (1 - 3/4 * (z / zᵖ)^(5/4)) * (z ≤ zᵖ) + 1/4 * (z > zᵖ)

# Cosine-squared warm bubble (filled mid-domain so we exercise mixed-phase rates)
function θᵢ(x, y, z, Lx, Ly)
    rᵇʰ = 8000.0; rᵇᵛ = 1500.0; zᵇ = 4000.0
    xᵇ = Lx / 2;  yᵇ = Ly / 2
    r = sqrt((x - xᵇ)^2 + (y - yᵇ)^2)
    R = sqrt((r / rᵇʰ)^2 + ((z - zᵇ) / rᵇᵛ)^2)
    θ̄ = θ_background(z)
    θ′ = ifelse(R < 1, 4 * cos(π * R / 2)^2, 0.0)
    return θ̄ + θ′
end

# ----------------------------------------------------------------------
# Build a model and time microphysics_model_update! on it
# ----------------------------------------------------------------------

function build_model(arch, microphysics; Nx=32, Ny=32, Nz=32)
    Lx = Ly = 32_000.0
    Lz = 16_000.0
    grid = RectilinearGrid(arch;
                           size=(Nx, Ny, Nz),
                           x=(0, Lx), y=(0, Ly), z=(0, Lz),
                           halo=(5, 5, 5),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants(saturation_vapor_pressure=TetensFormula())
    reference_state = ReferenceState(grid, constants;
                                     surface_pressure=100000,
                                     potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    advection = WENO(order=5)  # 5 is plenty for a microphysics-only timing
    model = AtmosphereModel(grid; dynamics, microphysics, advection,
                            thermodynamic_constants=constants)

    set!(model;
         θ = (x, y, z) -> θᵢ(x, y, z, Lx, Ly),
         ℋ = (x, y, z) -> ℋ_background(z))

    # Seed a mid-level mixed-phase region so process rates aren't all early-exit zero.
    # We only need the prognostic fields populated; specific-humidity diagnostics get
    # refreshed by the next update_state! when microphysics_model_update! runs.
    seed_mixed_phase!(model.microphysical_fields; Lx, Ly, Lz=16_000.0)

    return model
end

"""
Set non-zero cloud, rain, and ice in a vertical band around the bubble.
Uses GPU-compatible coordinate-based seeding via Oceananigans `set!`.
"""
function seed_mixed_phase!(μ; Lx, Ly, Lz)
    # Mid-level (~3-6 km) mixed-phase region overlapping the warm bubble
    seed_q(x, y, z, qmax) = begin
        rh = sqrt((x - Lx/2)^2 + (y - Ly/2)^2)
        zc = 4500.0; rh_max = 6000.0; dz = 1500.0
        envelope = exp(-((z - zc)/dz)^2 - (rh/rh_max)^2)
        qmax * envelope
    end
    if hasproperty(μ, :ρqᶜˡ)
        set!(μ.ρqᶜˡ, (x, y, z) -> seed_q(x, y, z, 5e-4))   # cloud
    end
    if hasproperty(μ, :ρqʳ)
        set!(μ.ρqʳ,  (x, y, z) -> seed_q(x, y, z, 1e-4))   # rain
    end
    if hasproperty(μ, :ρnʳ)
        set!(μ.ρnʳ,  (x, y, z) -> seed_q(x, y, z, 1e4))    # rain number
    end
    if hasproperty(μ, :ρqⁱ)
        set!(μ.ρqⁱ,  (x, y, z) -> seed_q(x, y, z, 1e-4))   # ice
    end
    if hasproperty(μ, :ρnⁱ)
        set!(μ.ρnⁱ,  (x, y, z) -> seed_q(x, y, z, 1e5))    # ice number
    end
    if hasproperty(μ, :ρnᶜˡ)
        set!(μ.ρnᶜˡ, (x, y, z) -> seed_q(x, y, z, 200e6))  # cloud number
    end
    if hasproperty(μ, :ρqᶠ)
        set!(μ.ρqᶠ,  (x, y, z) -> seed_q(x, y, z, 1e-5))   # rime mass
    end
    if hasproperty(μ, :ρbᶠ)
        set!(μ.ρbᶠ,  (x, y, z) -> seed_q(x, y, z, 2.5e-8)) # rime volume
    end
    return nothing
end

function time_microphysics(model; warmup=3, samples=50)
    arch = architecture(model.grid)
    is_gpu = !(arch isa CPU)

    # Warm up (compile + first table-lookup pass)
    for _ in 1:warmup
        microphysics_model_update!(model.microphysics, model)
        is_gpu && CUDA.synchronize()
    end

    times = Float64[]
    for _ in 1:samples
        is_gpu && CUDA.synchronize()
        t0 = time_ns()
        microphysics_model_update!(model.microphysics, model)
        is_gpu && CUDA.synchronize()
        push!(times, (time_ns() - t0) * 1e-9)
    end

    return times
end

function report(label, times, ncells)
    tmin, tmed, tmax = minimum(times), median(times), maximum(times)
    @printf "%-30s  min %.3f ms   median %.3f ms   max %.3f ms   per-cell %.3f μs\n" label  tmin*1e3 tmed*1e3 tmax*1e3 (tmed/ncells*1e6)
end

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

function main()
    Nx, Ny, Nz = 32, 32, 32
    ncells = Nx * Ny * Nz

    println("Grid: $Nx × $Ny × $Nz ($(ncells) cells)\n")

    archs = Tuple{Any, String}[(CPU(), "CPU")]
    if HAS_CUDA
        push!(archs, (GPU(), "GPU"))
    else
        @info "CUDA not functional — skipping GPU benchmark"
    end

    for (arch, archlabel) in archs
        println("== $archlabel ==")

        # P3
        p3 = PredictedParticlePropertiesMicrophysics(Float64)
        model = build_model(arch, p3; Nx, Ny, Nz)
        times = time_microphysics(model)
        report("$archlabel  P3", times, ncells)

        # Kessler (calibration baseline)
        kessler = DCMIP2016KesslerMicrophysics()
        model_k = build_model(arch, kessler; Nx, Ny, Nz)
        times_k = time_microphysics(model_k)
        report("$archlabel  Kessler", times_k, ncells)

        println()
    end
end

main()
