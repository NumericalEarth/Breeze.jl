# Mixed-precision droplet box: what besides condensation changes ρθ in a step?
using Breeze, Oceananigans, Statistics, Printf, CUDA
include(joinpath(@__DIR__, "box_precision_setup.jl"))

for (FT, TF) in ((Float64, Float64), (Float32, Float64))
    m = box(FT; thermodynamic_float_type=TF, velocity=false)
    ts = m.timestepper
    @info "types" FT TF eltype(ts.U⁰.ρθ) eltype(ts.Gⁿ.ρθ) eltype(ts.U⁰.ρqᵛ) eltype(m.formulation.potential_temperature)
    for n in 1:2
        ρθ₀ = Float64.(Array(interior(m.formulation.potential_temperature_density)))
        time_step!(m, 1e-2)
        ρθ₁ = Float64.(Array(interior(m.formulation.potential_temperature_density)))
        G = Float64.(Array(interior(ts.Gⁿ.ρθ)))
        Gq = Float64.(Array(interior(ts.Gⁿ.ρqᵛ)))
        w = Float64.(Array(interior(m.velocities.w)))
        u = Float64.(Array(interior(m.velocities.u)))
        @info @sprintf("%s/%s step %d: max|Gρθ| = %.3e mean Gρθ = %.3e, max|Gρqᵛ| = %.3e, mean w = %.3e, max|w - mean w| = %.3e, max|u| = %.3e, Δρθ range [%.4e, %.4e]",
                       FT, TF, n, maximum(abs, G), mean(G), maximum(abs, Gq), mean(w), maximum(abs, w .- mean(w)),
                       maximum(abs, u), minimum(ρθ₁ .- ρθ₀), maximum(ρθ₁ .- ρθ₀))
    end
end
