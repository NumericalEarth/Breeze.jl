using GLMakie
using Breeze

thermo = ThermodynamicConstants()

saturation_specific_humidity_large_yeager(T, ρ) = 640380 * exp(-5107.4 / T) / ρ

ρ = 1.2
T₀ = 273.15
T = collect(T₀:0.01:T₀+50)
q★_large_yeager = saturation_specific_humidity_large_yeager.(T, ρ)
q★_aqua_sky = Breeze.Thermodynamics.saturation_specific_humidity.(T, ρ, Ref(thermo), Ref(thermo.liquid))

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Temperature (K)", ylabel = "Saturation Specific Humidity (g/kg)")
lines!(ax, T, q★_large_yeager, label = "Large and Yeager (2009)")
lines!(ax, T, q★_aqua_sky, label = "Breeze")
Legend(fig[1, 2], ax)
display(fig)