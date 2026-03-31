using Breeze
using Breeze: apply_polar_filter!
using Oceananigans
using Oceananigans.Grids: φnodes
using Test

# Note: When run through the test runner, test_float_types and default_arch
# are defined in the init_code. When run directly, we need fallbacks.
if !@isdefined(test_float_types)
    test_float_types() = (Float64,)
end
if !@isdefined(default_arch)
    const default_arch = CPU()
end

function build_polar_filter_test_grid(arch; FT=Float64, Nx=36, Ny=34, Nz=4)
    Oceananigans.defaults.FloatType = FT
    LatitudeLongitudeGrid(arch;
                          size = (Nx, Ny, Nz),
                          halo = (3, 3, 3),
                          longitude = (0, 360),
                          latitude = (-85, 85),
                          z = (0, 30000))
end

#####
##### Constructor tests
#####

@testset "PolarFilter constructor [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(CPU(); FT)
    φ = φnodes(grid, Center())

    pf = PolarFilter(grid; threshold_latitude=60)

    ## filtered_indices should cover both hemispheres
    @test length(pf.filtered_indices) > 0
    @test all(abs(φ[j]) > 60 for j in pf.filtered_indices)

    ## Both hemispheres represented
    @test any(φ[j] < 0 for j in pf.filtered_indices)
    @test any(φ[j] > 0 for j in pf.filtered_indices)

    ## Symmetric: same number of filtered rows per hemisphere
    n_south = count(φ[j] < 0 for j in pf.filtered_indices)
    n_north = count(φ[j] > 0 for j in pf.filtered_indices)
    @test n_south == n_north

    ## Spectral mask has correct shape
    Nk = grid.Nx ÷ 2 + 1
    @test size(pf.spectral_mask) == (Nk, length(pf.filtered_indices))

    ## Buffer sizes
    N_batch = length(pf.filtered_indices) * grid.Nz
    @test size(pf.buffer_real) == (grid.Nx, N_batch)
    @test size(pf.buffer_complex) == (Nk, N_batch)

    ## threshold_latitude stored correctly
    @test pf.threshold_latitude == FT(60)
end

#####
##### Spectral mask tests
#####

@testset "Spectral mask: k_max formula [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(CPU(); FT)
    φ = φnodes(grid, Center())
    Nk = grid.Nx ÷ 2 + 1

    pf_sharp = PolarFilter(grid; threshold_latitude=60, filter_mode=SharpTruncation())
    cos60 = cosd(60)

    for (row, j) in enumerate(pf_sharp.filtered_indices)
        expected_kmax = max(1, floor(Int, grid.Nx * cosd(abs(φ[j])) / cos60))
        ## Clamp to Nk since rfft only produces Nk coefficients
        expected_pass = min(expected_kmax, Nk)
        ## Count how many wavenumbers pass the sharp filter
        n_pass = count(pf_sharp.spectral_mask[:, row] .> 0.5)
        @test n_pass == expected_pass
    end
end

@testset "Spectral mask: sharp truncation [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(CPU(); FT)
    pf = PolarFilter(grid; threshold_latitude=60, filter_mode=SharpTruncation())
    Nk = grid.Nx ÷ 2 + 1

    for row in 1:length(pf.filtered_indices)
        mask = pf.spectral_mask[:, row]
        ## All values should be exactly 0 or 1
        @test all(m -> m == 0 || m == 1, mask)
        ## First element (mean) should always pass
        @test mask[1] == 1
    end
end

@testset "Spectral mask: exponential rolloff [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(CPU(); FT)
    φ = φnodes(grid, Center())
    pf = PolarFilter(grid; threshold_latitude=60, filter_mode=ExponentialRolloff(8))
    Nk = grid.Nx ÷ 2 + 1
    cos60 = cosd(60)

    for (row, j) in enumerate(pf.filtered_indices)
        mask = pf.spectral_mask[:, row]
        k_max = max(1, floor(Int, grid.Nx * cosd(abs(φ[j])) / cos60))
        ## Mean (k=1) should always be 1
        @test mask[1] == 1
        ## Nyquist should be near machine precision (only when filtering occurs)
        if k_max < Nk
            @test mask[end] < 1e-10
        end
        ## Mask should be monotonically non-increasing
        @test all(mask[k] >= mask[k+1] - eps(FT) for k in 1:Nk-1)
    end
end

#####
##### Filtering correctness tests
#####

@testset "Low wavenumber preserved [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(CPU(); FT)
    pf = PolarFilter(grid; threshold_latitude=60)

    f = CenterField(grid)
    set!(f, (λ, φ, z) -> cosd(λ))
    f_orig = deepcopy(interior(f))

    apply_polar_filter!(pf, f)

    @test maximum(abs, interior(f) .- f_orig) < 100 * eps(FT)
end

@testset "High wavenumber removed at high latitudes [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(CPU(); FT)
    φ = φnodes(grid, Center())

    ## Sharp truncation should completely zero out k=17 at φ=-82.5° (k_max=9)
    pf_sharp = PolarFilter(grid; threshold_latitude=60, filter_mode=SharpTruncation())
    f = CenterField(grid)
    set!(f, (λ, φ, z) -> cosd(17λ))
    apply_polar_filter!(pf_sharp, f)

    j_high = pf_sharp.filtered_indices[1]
    rms_high = sqrt(sum(interior(f)[:, j_high, 1] .^ 2) / grid.Nx)
    @test rms_high < 100 * eps(FT)

    ## Exponential rolloff should strongly attenuate
    pf_exp = PolarFilter(grid; threshold_latitude=60, filter_mode=ExponentialRolloff(8))
    set!(f, (λ, φ, z) -> cosd(17λ))
    apply_polar_filter!(pf_exp, f)

    rms_exp = sqrt(sum(interior(f)[:, j_high, 1] .^ 2) / grid.Nx)
    @test rms_exp < 1e-4
end

@testset "Equatorial latitudes unchanged [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(CPU(); FT)
    pf = PolarFilter(grid; threshold_latitude=60)

    f = CenterField(grid)
    set!(f, (λ, φ, z) -> cosd(17λ) * cosd(φ))
    f_orig = deepcopy(interior(f))

    apply_polar_filter!(pf, f)

    ## Rows below the threshold should be completely untouched
    φ = φnodes(grid, Center())
    for j in 1:grid.Ny
        if abs(φ[j]) <= 60
            @test maximum(abs, interior(f)[:, j, :] .- f_orig[:, j, :]) < eps(FT)
        end
    end
end

#####
##### Multiple fields (NamedTuple) test
#####

@testset "Filter applied to multiple fields [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(CPU(); FT)
    pf = PolarFilter(grid; threshold_latitude=60, filter_mode=SharpTruncation())

    f1 = CenterField(grid)
    f2 = CenterField(grid)
    set!(f1, (λ, φ, z) -> cosd(17λ))
    set!(f2, (λ, φ, z) -> sind(17λ))

    apply_polar_filter!(pf, (f1, f2))

    j_high = pf.filtered_indices[1]
    @test sqrt(sum(interior(f1)[:, j_high, 1] .^ 2) / grid.Nx) < 100 * eps(FT)
    @test sqrt(sum(interior(f2)[:, j_high, 1] .^ 2) / grid.Nx) < 100 * eps(FT)
end

#####
##### Callback integration test
#####

@testset "add_polar_filter! callback [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = build_polar_filter_test_grid(CPU(); FT, Nx=36, Ny=34, Nz=4)

    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                    surface_pressure = 100000,
                                    reference_potential_temperature = 300)
    coriolis = HydrostaticSphericalCoriolis()
    model = AtmosphereModel(grid; dynamics, coriolis, advection=WENO())
    set!(model; θ=300, ρ=1.2)

    simulation = Simulation(model; Δt=0.1, stop_iteration=3, verbose=false)
    filter = add_polar_filter!(simulation; threshold_latitude=60)

    @test filter isa PolarFilter
    @test length(filter.filtered_indices) > 0

    run!(simulation)

    @test model.clock.iteration == 3
    @test !any(isnan, parent(model.dynamics.density))
    @test !any(isnan, parent(model.momentum.ρu))
    @test !any(isnan, parent(model.momentum.ρv))
end
