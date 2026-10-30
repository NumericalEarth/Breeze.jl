# T21-sedfuse: state hashes after a few acoustic RK3 steps with operator-split sedimentation, for
# 1M and P3, implicit and substep, Float32 and Float64, on CPU or GPU. Run the same script with the
# environment of the reference branch and of the fused branch (separate processes) and compare the
# printed lines: the fused single-kernel step must be bit-identical.
# Usage: julia --project=<env> test/t21_split_sedimentation_bits.jl [CPU|GPU]
using Breeze, Oceananigans
using Oceananigans.TimeSteppers: time_step!
using UUIDs: UUID
using CRC32c: crc32c
Base.require(Base.PkgId(UUID("6a9e3e04-43cd-43ba-94b9-e8782df3c71b"), "CloudMicrophysics"))
const OneMomentCloudMicrophysics = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt).OneMomentCloudMicrophysics
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticlePropertiesMicrophysics
const AMs = Breeze.AtmosphereModels

arch = get(ARGS, 1, "CPU") == "GPU" ? (Base.require(Base.PkgId(UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")); GPU()) : CPU()

function build(FT, scheme)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(arch, FT; size=(6, 5, 16), halo=(5, 5, 5), x=(0, 6e3), y=(0, 5e3), z=(0, 8e3))
    constants = ThermodynamicConstants(FT)
    microphysics = scheme == :p3 ? PredictedParticlePropertiesMicrophysics(FT) :
        OneMomentCloudMicrophysics(FT; cloud_formation=SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT)))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=290)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics,
                            timestepper=:AcousticRungeKutta3)
    ref = model.dynamics.reference_state
    set!(model; θ=(x, y, z) -> FT(290) + FT(0.004) * z + FT(0.3) * sin(x / 1e3), ρ=ref.density,
         qᵗ=(x, y, z) -> FT(0.012) * exp(-z / 2500) * (1 + FT(0.2) * cos(y / 900)), u=FT(3))
    μ = model.microphysical_fields
    bump(z, z₀, a) = a * exp(-((z - z₀) / 1500)^2)
    for (name, amp, z₀) in ((:ρqʳ, 1e-3, 3000), (:ρqˢⁿ, 8e-4, 5500), (:ρqⁱ, 6e-4, 5500), (:ρnʳ, 2e4, 3000),
                            (:ρnⁱ, 1e5, 5500), (:ρqᶠ, 2e-4, 5500), (:ρbᶠ, 4e-7, 5500))
        haskey(μ, name) || continue
        set!(μ[name], (x, y, z) -> FT(bump(z, z₀, amp) * (1 + 0.5 * sin(x / 700))))
    end
    return model
end

for scheme in (:onemoment, :p3), FT in (Float32, Float64), mode in (:implicit, :substep)
    AMs.SEDIMENTATION_IMPLICIT[] = mode == :implicit
    AMs.SEDIMENTATION_SUBSTEP[] = mode == :substep
    model = build(FT, scheme)
    for _ in 1:3; time_step!(model, FT(10)); end
    fields = Oceananigans.prognostic_fields(model)
    h = [string(n, "=", string(crc32c(reinterpret(UInt8, vec(Array(interior(f))))), base=16)) for (n, f) in pairs(fields)]
    finite = all(f -> all(isfinite, Array(interior(f))), values(fields))
    println("BITS $(arch) $scheme $FT $mode finite=$finite ", join(h, " "))
end
AMs.SEDIMENTATION_IMPLICIT[] = false; AMs.SEDIMENTATION_SUBSTEP[] = false
