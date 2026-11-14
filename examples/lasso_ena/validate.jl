# Run with: ENA_LASSO_BUNDLE=/path/to/member julia --project=examples examples/lasso_ena/validate.jl
using Test
isdefined(Main, :LassoENA) || include("LassoENA.jl")
using .LassoENA
using Breeze, Oceananigans
using Oceananigans.Units: Time
using Oceananigans.TimeSteppers: time_step!

@testset "LASSO input and forcing adapter" begin
    bundle = ENV["ENA_LASSO_BUNDLE"]
    @test isnothing(LassoENA.validate_inputs(bundle))
    mktempdir() do directory
        @test_throws ArgumentError LassoENA.validate_inputs(directory)
    end
    lsf = LassoENA.read_sam_large_scale_forcing(joinpath(bundle, "lsf"))
    sfc = LassoENA.read_sam_surface_forcing(joinpath(bundle, "sfc"))
    @test length(lsf) == length(sfc.day) == 37
    @test first(sfc.day) == 198.75
    @test last(sfc.day) == 200.25
    faces = LassoENA.grid_faces(LassoENA.read_sam_grd(joinpath(bundle, "grd"))[1:260])
    @test all(diff(faces) .> 0)
    @test (faces[1:end-1] .+ faces[2:end]) ./ 2 ≈ LassoENA.read_sam_grd(joinpath(bundle, "grd"))[1:260]
    @test LassoENA.sam_interpolate_column([0., 100.], [0., 2.], [50.]; pressure_grid=false) == [1.]
    case = LassoENA.build_case(bundle, CPU(); Nx=8, Ny=8)
    model = case.model
    @test LassoENA.has_aerosol_reservoir(model.microphysics)
    @test hasproperty(model.microphysical_fields, :ρnᶜˡ)
    series = LassoENA.LargeScaleForcingProfiles(model.grid, lsf, case.z,
        Array(interior(model.dynamics.reference_state.pressure, 1, 1, :)); day0=199.)
    at(t) = [series.tls[1, 1, k, Time(t)] for k in eachindex(case.z)]
    @test at(1800.) ≈ (at(0.) .+ at(3600.)) ./ 2
    @test any(at(0.) .!= at(3600.))
    @test model.radiation !== nothing
    @test model.radiation.schedule.interval == 30
    # A nonzero UTC time must select the appropriate SST, not the first file record.
    model.clock.time = 4500.0
    LassoENA.refresh_inputs!(model, case.sst)
    expected = LassoENA.interpolate_profile((sfc.day .- 199) .* 86400, sfc.sst, 4500.)
    @test all(interior(model.radiation.surface_radiation.surface_temperature) .≈ expected)
    for n in 1:2
        LassoENA.refresh_inputs!(model, case.sst)
        time_step!(model, 1.0)
    end
    @test model.clock.time == 4502.0
    @test all(all(isfinite, interior(f)) for f in values(Oceananigans.prognostic_fields(model)))
end
