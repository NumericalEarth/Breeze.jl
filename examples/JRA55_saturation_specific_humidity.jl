using JLD2
using Breeze
using Breeze: saturation_specific_humidity, AtmosphereThermodynamics
using GLMakie

@load "JRA55_atmospheric_state_Jan_1_1991.jld2" q T p

thermo = AtmosphereThermodynamics()

ρ = 1.2
qᵛ★ = saturation_specific_humidity.(T, 1.2, Ref(thermo))
qˡ = @. max(0, q - q★)

fig = Figure(size=(1200, 600))
ax1 = Axis(fig[1, 2], title="Temperature")
ax2 = Axis(fig[2, 1], title="Saturation specific humidity")
ax3 = Axis(fig[1, 1], title="Total specific humidity")
ax4 = Axis(fig[2, 2], title="Liquid specific humidity")
# heatmap!(ax1, T, colormap=:magma)
# heatmap!(ax2, qᵛ★)
# heatmap!(ax3, q, colormap=:grays)
# heatmap!(ax4, qˡ, colormap=:grays, colorrange=(0, 0.001))

# Compute cloudiness for instantaneous drop
θ⁻ = T .- 10
Ψ = Breeze.ThermodynamicState{Float64}.(θ⁻, q, 0)
ℛ = Breeze.ReferenceStateConstants{Float64}(101325, 20)
T⁻ = Breeze.temperature.(Ψ, Ref(ℛ), Ref(thermo))
qᵛ★ = saturation_specific_humidity.(T⁻, 1.2, Ref(thermo))
qˡ = @. max(0, q - qᵛ★)
Π = Breeze.exner_function.(Ψ, Ref(ℛ), Ref(thermo))
Tu = @. Π * θ⁻
ΔT = T⁻ - Tu

heatmap!(ax1, ΔT, colormap=:magma)
heatmap!(ax2, qᵛ★)
heatmap!(ax3, q, colormap=:grays)
heatmap!(ax4, qˡ, colormap=:grays, colorrange=(0, 0.01))
display(fig)