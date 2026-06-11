## GATE vertical grid (after PR #397): a smoothly stretched column with spacing that
## ramps 50 m → 100 m → 300 m across three transition heights. Returns the z-face
## coordinates (length Nz+1). Shared by the distributed TC driver and the movie scripts
## so post-processing uses exactly the grid the simulation ran on.
function gate_vertical_grid(zᵗ; Δz⁰ = 50, Δzᵖ = 100, Δzᵗ = 300)
    z₁, z₂, z₃ = 1275, 5100, 18000   # transition heights
    z_faces = [0.0]
    z = 0.0
    while z < zᵗ
        α = clamp((z - z₁) / (z₂ - z₁), 0, 1)
        β = clamp((z - z₂) / (z₃ - z₂), 0, 1)
        Δz = Δz⁰ + α * (Δzᵖ - Δz⁰) + β * (Δzᵗ - Δzᵖ)
        z = min(z + Δz, zᵗ)
        push!(z_faces, z)
    end
    return z_faces
end
