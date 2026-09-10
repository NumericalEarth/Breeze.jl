include("setup.jl")

using Test, Breeze, Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior
using OffsetArrays: OffsetArray
using Breeze.Microphysics.PredictedParticleProperties: AerosolActivation, AerosolMode,
    fill_p3_specific_boundary_halos!

boundary_data(field) = OffsetArray(Array(parent(field)), axes(field.data))

@testset "Prescribed P3 face values" for FT in (Float32, Float64), activation in (false, true), supersaturation in (false, true)
    grid = RectilinearGrid(default_arch, FT; size=(3, 3, 3), extent=(3, 3, 3),
                           halo=(3, 3, 3), topology=(Bounded, Bounded, Bounded))
    aerosol = activation ? AerosolActivation(AerosolMode(FT)) : nothing
    p3 = PredictedParticlePropertiesMicrophysics(FT; aerosol, predict_supersaturation=supersaturation)
    names = Breeze.AtmosphereModels.prognostic_field_names(p3)
    sides = (:west, :east, :south, :north, :bottom, :top)
    bcs = NamedTuple(name => FieldBoundaryConditions(;
        NamedTuple(side => ValueBoundaryCondition(FT(n)) for side in sides)...)
        for (n, name) in enumerate(names))
    microphysical_fields = Breeze.AtmosphereModels.materialize_microphysical_fields(p3, grid, bcs)
    density = CenterField(grid)
    data = boundary_data(density)
    for k in axes(data, 3), j in axes(data, 2), i in axes(data, 1)
        data[i, j, k] = FT(2 + i/32 + j/64 + k/128)
    end
    copyto!(parent(density), parent(data))
    pairs = (((0,2,2),(1,2,2)), ((4,2,2),(3,2,2)),
             ((2,0,2),(2,1,2)), ((2,4,2),(2,3,2)),
             ((2,2,0),(2,2,1)), ((2,2,4),(2,2,3)))
    for (n, name) in enumerate(names)
        raw = microphysical_fields[name]
        set!(raw, FT(2n))
        fill_halo_regions!(raw)
        raw_values = boundary_data(raw)
        specific_name = Breeze.AtmosphereModels.specific_field_name(name)
        specific = microphysical_fields[specific_name]
        set!(specific, FT(n/32))
        initial = Array(interior(specific))
        fill_p3_specific_boundary_halos!(specific, raw, density)
        values = boundary_data(specific)
        for (ghost, inside) in pairs
            @test raw_values[ghost...] == 0
            face_density = (data[ghost...] + data[inside...]) / 2
            face_value = (values[ghost...] + values[inside...]) / 2
            @test face_density * face_value ≈ FT(n) rtol=16eps(FT)
        end
        @test Array(interior(specific)) == initial
    end
end
