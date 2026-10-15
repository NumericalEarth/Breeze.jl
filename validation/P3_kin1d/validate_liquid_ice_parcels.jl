# Run the production ParcelDynamics examples, check their conserved quantities,
# and retain full scalar trajectories alongside the figures.
using Test
using DelimitedFiles
using Printf
root = normpath(joinpath(@__DIR__, "../.."))
directory = joinpath(root, "validation_output/p3_liquid_ice/parcels")
mkpath(directory)
cd(directory) do
    include(joinpath(root, "examples/rising_parcels.jl"))
end
open(joinpath(directory, "conservation.txt"), "w") do io
    @testset "Production θ_li parcels: $name" for (name, parcel) in (("dry", dry), ("one_moment", one_moment),
        ("kessler", kessler), ("two_moment", two_moment), ("buoyant_two_moment", buoyant_two_moment),
        ("prescribed", cloudy), ("buoyant", buoyant))
        constants = parcel.model.thermodynamic_constants
        snapshots = parcel.snapshots
        @test all(s.𝒰 isa Breeze.Thermodynamics.LiquidIcePotentialTemperatureState for s in snapshots)
        theta = [s.𝒰.potential_temperature for s in snapshots]
        water = [sum((s.𝒰.moisture_mass_fractions.vapor, s.𝒰.moisture_mass_fractions.liquid,
                      s.𝒰.moisture_mass_fractions.ice)) for s in snapshots]
        theta_error = maximum(abs.(theta .- first(theta)))
        water_error = maximum(abs.(water .- first(water)))
        @test theta_error < 1e-8
        @test water_error < 1e-10
        @printf(io, "%s: max θ_li drift = %.6g K; max total-water drift = %.6g kg/kg\n", name, theta_error, water_error)
        data = [[s.t, s.z, s.w, s.ρ, temperature(s.𝒰, constants), s.𝒰.potential_temperature,
                 s.𝒰.moisture_mass_fractions.vapor, s.𝒰.moisture_mass_fractions.liquid,
                 s.𝒰.moisture_mass_fractions.ice, (isnothing(s.μ) ? () : values(s.μ))...] for s in snapshots]
        columns = ["t", "z", "w", "rho", "T", "theta_li", "qv", "ql", "qi", (isnothing(first(snapshots).μ) ? () : string.(keys(first(snapshots).μ)))...]
        open(joinpath(directory, name * ".dat"), "w") do output
            println(output, "# ", join(columns, ' '))
            writedlm(output, permutedims(hcat(data...)))
        end
        @test all(isfinite, reduce(vcat, data))
    end
end
print(read(joinpath(directory, "conservation.txt"), String))
