include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans.Units
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: mixing_length_column!, mixing_length
using Test

#####
##### The mixing length of SHOC is a component of the closure, `closure.mixing_length`: the column
##### functions dispatch on its type. `BlendedMixingLength` is the blend of κ z, l∞ and √e / N of
##### `shoc.F90`; `GradientLimitedMixingLength` is the two-sweep stratification envelope
##### ℓ = min(ℂˢ z, min over z′ of [ℓᵇ(z′) + ℂⁱ |z - z′|]), computed by the functions of
##### `TKEBasedTurbulenceClosure`.
#####

# The pair of coefficients published with `TKEBasedTurbulenceClosure`
calibrated_gradient_limited_length(FT) = GradientLimitedMixingLength(FT(0.7877315198717979), FT(1.777871783491512))

# The gradient-limited length with the components it was calibrated with in the TKE closure: the
# stability functions of the Richardson number and a shear production coefficient of 1
eddy_diffusivity_core(FT) = (mixing_length = calibrated_gradient_limited_length(FT),
                             turbulence_time_scale = RichardsonStabilityFunctions(),
                             tke_update = ImplicitSinkTKE(ℂᴾ = 1))

# `compute_shoc_mix_shoc_length` followed by `check_length_scale_shoc_length` of E3SM `shoc.F90`,
# written out with the literals of the Fortran
function fortran_mixing_length(tke, brunt, zt, l_inf, host_scale)
    FT = typeof(tke)
    length_fac, tscale, vk, maxlen, minlen = FT(0.5), FT(400), FT(0.4), FT(20000), FT(20)
    tkes = sqrt(tke)
    brunt2 = brunt >= 0 ? brunt : zero(FT)
    mix = min(maxlen, (FT(2.8284) * sqrt(1 / ((1 / (tscale * tkes * vk * zt))
              + (1 / (tscale * tkes * l_inf)) + FT(0.01) * (brunt2 / tke)))) / length_fac)
    mix = min(maxlen, mix)
    mix = max(minlen, mix)
    mix = min(host_scale, mix)
    return mix
end

piecewise_linear(z, nodes) = begin
    z ≤ nodes[1][1] && return nodes[1][2]
    for n in 2:length(nodes)
        (z₁, v₁), (z₂, v₂) = nodes[n-1], nodes[n]
        z ≤ z₂ && return v₁ + (v₂ - v₁) * (z - z₁) / (z₂ - z₁)
    end
    return nodes[end][2]
end

# The dry convective boundary layer and BOMEX (Siebesma et al. 2003) of the SHOC column
# comparisons: 75 levels of 40 m, surface fluxes applied by SHOC
function single_column_model(case, arch, FT, components)
    Nz, Lz = 75, 3000
    grid = RectilinearGrid(arch, FT; size = Nz, z = (0, Lz), topology = (Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    host_grid_spacing = 50_000 / 15

    if case == :dry_cbl
        θᵢ(z) = 300 + 3e-3 * z
        reference_state = ReferenceState(grid, constants; base_pressure = 1e5, potential_temperature = θᵢ)
        closure = SimplifiedHigherOrderClosure(FT; surface_temperature_flux = 0.1, host_grid_spacing, components...)
        model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), closure)
        set!(model, θ = (args...) -> θᵢ(last(args)))
        set!(model.tracers.ρe, reference_state.density * FT(4e-4))
        return model
    end

    θˡᵢ(z) = piecewise_linear(z, ((0.0, 298.7), (520.0, 298.7), (1480.0, 302.4), (2000.0, 308.2), (3000.0, 311.85)))
    qᵗᵢ(z) = piecewise_linear(z, ((0.0, 17.0e-3), (520.0, 16.3e-3), (1480.0, 10.7e-3), (2000.0, 4.2e-3), (3000.0, 3.0e-3)))
    uᵢ(z) = piecewise_linear(z, ((0.0, -8.75), (700.0, -8.75), (3000.0, -4.61)))
    eᵢ(z) = max(1 - z / 3000, 4e-4)
    wˢ(z) = piecewise_linear(z, ((0.0, 0.0), (1500.0, -0.65e-2), (2100.0, 0.0)))
    ∂t_θˡ(z) = piecewise_linear(z, ((0.0, -2.0 / 86400), (1500.0, -2.0 / 86400), (2500.0, 0.0)))
    ∂t_qᵗ(z) = piecewise_linear(z, ((0.0, -1.2e-8), (300.0, -1.2e-8), (500.0, 0.0)))
    uᵍ(z) = -10 + 1.8e-3 * z
    vᵍ(z) = 0

    reference_state = ReferenceState(grid, constants; base_pressure = 1.015e5, potential_temperature = θˡᵢ,
                                     vapor_mass_fraction = qᵗᵢ)
    subsidence = SubsidenceForcing(wˢ)
    geostrophic = geostrophic_forcings(uᵍ, vᵍ)
    cooling = Field{Nothing, Nothing, Center}(grid)
    drying = Field{Nothing, Nothing, Center}(grid)
    set!(cooling, ∂t_θˡ)
    set!(drying, ∂t_qᵗ)
    forcing = (u = (subsidence, geostrophic.u), v = (subsidence, geostrophic.v),
               θ = (subsidence, Forcing(cooling)), qᵗ = (subsidence, Forcing(drying)))

    closure = SimplifiedHigherOrderClosure(FT; surface_temperature_flux = 8e-3, surface_moisture_flux = 5.2e-5,
                   friction_velocity = 0.28, host_grid_spacing, components...)
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), forcing, closure,
                            coriolis = FPlane(f = 0.376e-4))
    set!(model, θ = (args...) -> θˡᵢ(last(args)), qᵗ = (args...) -> qᵗᵢ(last(args)), u = (args...) -> uᵢ(last(args)))
    e₀ = CenterField(grid)
    set!(e₀, (args...) -> eᵢ(last(args)))
    set!(model.tracers.ρe, reference_state.density * e₀)
    return model
end

column(field) = vec(Array(interior(field)))

function run_single_column(case, arch, FT; steps, Δt = 100, components...)
    model = single_column_model(case, arch, FT, components)
    for _ in 1:steps
        time_step!(model, Δt)
    end
    ρ = column(model.dynamics.reference_state.density)
    d = model.closure_fields.diagnostics
    return (z = collect(znodes(model.grid, Center())),
            θ = column(model.formulation.potential_temperature_density) ./ ρ,
            q = column(model.moisture_density) ./ ρ,
            e = column(model.tracers.ρe) ./ ρ,
            ℓ = column(d.shoc_mix),
            tkh = column(d.tkh),
            cloud_fraction = column(d.shoc_cldfrac),
            closure = model.closure,
            scratch = model.closure_fields.scratch)
end

profiles_are_finite(run) = all(name -> all(isfinite, getproperty(run, name)), (:θ, :q, :e, :ℓ, :tkh, :cloud_fraction))

@testset "SHOC mixing length components [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    hard = HardClipping()
    constants = SHOCConstants(FT)

    @testset "The mixing length is a field of the closure" begin
        original = SimplifiedHigherOrderClosure(FT)
        @test original.mixing_length === BlendedMixingLength(FT)
        @test original.mixing_length isa BlendedMixingLength{FT}
        @test SimplifiedHigherOrderClosure(FT).mixing_length === BlendedMixingLength(FT)

        # The numbers of shoc.F90
        ml = original.mixing_length
        @test (ml.ℂˡ₁, ml.ℂˡ₂, ml.ℂˡ₃, ml.ℂˡ₄, ml.ℂˡ₅, ml.maxlen, ml.minlen) ===
              (FT(0.5), FT(2.8284), FT(0.01), FT(0.1), FT(400), FT(20000), FT(20))

        # Giving the default component changes nothing; giving another one changes only that slot
        @test SimplifiedHigherOrderClosure(FT; mixing_length = BlendedMixingLength(FT)) === original
        @test SimplifiedHigherOrderClosure(FT; mixing_length = BlendedMixingLength()) === original
        gl = calibrated_gradient_limited_length(FT)
        swapped = SimplifiedHigherOrderClosure(FT; mixing_length = gl)
        @test swapped.mixing_length === gl
        @test swapped.mixing_length isa GradientLimitedMixingLength{FT}
        @test typeof(swapped) !== typeof(original)
        for name in fieldnames(typeof(original))
            name == :mixing_length || @test getfield(swapped, name) === getfield(original, name)
        end

        # A component of another float type is converted to that of the closure
        @test SimplifiedHigherOrderClosure(FT; mixing_length = GradientLimitedMixingLength(0.8f0, 1.8f0)).mixing_length isa
              GradientLimitedMixingLength{FT}
        @test SimplifiedHigherOrderClosure(FT; mixing_length = BlendedMixingLength(Float32)).mixing_length isa BlendedMixingLength{FT}

        # The coefficients published with the TKE closure
        @test TKEBasedTurbulenceClosure().mixing_length == calibrated_gradient_limited_length(FT)

        # Plain data, for kernels and for a closure rebuilt from traced scalars
        @test isbits(original) && isbits(swapped)
        @test isbitstype(BlendedMixingLength{FT}) && isbitstype(typeof(GradientLimitedMixingLength(FT(0.8), FT(1.8))))

        # The coefficients of the length are those of the closure, by name
        @test closure_parameters(swapped).ℂˢ === gl.ℂˢ
        @test closure_parameters(swapped).ℂⁱ === gl.ℂⁱ
        @test !haskey(closure_parameters(swapped), :ℂˡ₁) && !haskey(closure_parameters(original), :ℂˢ)
        changed = with_parameters(swapped; ℂˢ = FT(0.5), ℂⁱ = FT(1.2))
        @test changed.mixing_length === GradientLimitedMixingLength(FT(0.5), FT(1.2))
        @test with_parameters(original; ℂˡ₃ = FT(0.02)).mixing_length.ℂˡ₃ === FT(0.02)
        @test_throws ArgumentError with_parameters(original; ℂˢ = FT(0.5))
    end

    @testset "BlendedMixingLength is the expression of shoc.F90, bit for bit" begin
        ml = BlendedMixingLength(FT)
        nothing_else = (nothing, nothing, nothing, nothing)
        @test mixing_length_column!(ml, hard, constants, 1, 1, nothing, 10, FT(731), nothing_else...) === ml.ℂˡ₄ * FT(731)

        compared = 0
        for tke in FT.((4e-4, 1.3e-3, 0.02, 0.37, 1.1, 6.5, 50)), brunt in FT.((-3e-4, -1e-7, 0, 1e-7, 1.1e-5, 1.3e-4, 2e-3)),
            zt in FT.((20, 61.5, 420, 1777, 2980, 15000)), l_inf in FT.((3.7, 41, 123.4, 950)), host_scale in FT.((150, 3333.3, 1e5))

            mix = mixing_length(ml, hard, constants, l_inf, 1, 1, 1, nothing, tke, brunt, zt, host_scale, nothing, nothing)
            @test mix === fortran_mixing_length(tke, brunt, zt, l_inf, host_scale)
            compared += 1
        end
        @test compared == 7 * 7 * 6 * 4 * 3

        # The three bounds are reached
        @test mixing_length(ml, hard, constants, FT(1e5), 1, 1, 1, nothing, FT(50), FT(0), FT(1e5), FT(1e6), nothing, nothing) === ml.maxlen
        @test mixing_length(ml, hard, constants, FT(10), 1, 1, 1, nothing, FT(4e-4), FT(1e-2), FT(20), FT(1e6), nothing, nothing) === ml.minlen
        @test mixing_length(ml, hard, constants, FT(1e4), 1, 1, 1, nothing, FT(50), FT(0), FT(1e4), FT(150), nothing, nothing) === FT(150)
    end

    @testset "The envelope of GradientLimitedMixingLength in SHOC's column" begin
        ml = GradientLimitedMixingLength(FT(0.8), FT(1.8))
        model = single_column_model(:dry_cbl, default_arch, FT, (; mixing_length = ml))
        grid = model.grid
        Nz = size(grid, 3)
        fields = model.closure_fields
        state, host, scratch = fields.state, fields.host, fields.scratch

        # Turbulence that decays upward under two stable layers with neutral air between and above
        e(z) = 0.02 + 0.6 * exp(-z / 700)
        θᵥ(z) = 300 + 4e-3 * clamp(z - 900, 0, 400) + 8e-3 * clamp(z - 2000, 0, 300)
        set!(state.tke, e)
        set!(host.thv, θᵥ)
        set!(host.zt_grid, z -> z)

        context = @allowscalar mixing_length_column!(ml, hard, constants, 1, 1, grid, Nz, zero(FT), state, host,
                                                     fields.diagnostics, scratch)
        ℓ = @allowscalar [mixing_length(ml, hard, constants, context, 1, 1, k, grid, zero(FT), zero(FT), zero(FT),
                                        FT(1e5), state, scratch) for k in 1:Nz]

        # Level 1 is the lowest level of SHOC's arrays in Breeze, as in the grid (E3SM counts from
        # the top): the surface and the top of the interface stratification are faces 1 and Nz+1
        zᶜ = collect(znodes(grid, Center()))
        zᶠ = collect(znodes(grid, Face()))
        N² = column(scratch.interface_stratification)
        @test N²[1] == 0 && N²[Nz+1] == 0
        θ = θᵥ.(zᶜ)
        g = constants.ggr
        expected_N² = [2g / (θ[k] + θ[k-1]) * (θ[k] - θ[k-1]) / (zᶜ[k] - zᶜ[k-1]) for k in 2:Nz]
        # (in single precision the differences of θᵥ between levels carry four digits)
        tolerance = FT == Float64 ? 1e-10 : 2e-3
        @test isapprox(N²[2:Nz], expected_N²; rtol = tolerance, atol = tolerance * g / 300 * 8e-3)
        @test all(N²[2:Nz][zᶠ[2:Nz] .< 880] .== 0)
        @test all(N²[2:Nz][940 .< zᶠ[2:Nz] .< 1260] .> 0)

        # The envelope of the two sweeps is min over z′ of ℓᵇ(z′) + ℂⁱ |z - z′|, with the buoyancy
        # penetration depth ℓᵇ = √e / N of the stable faces: here by brute force
        velocity = sqrt.(max.(e.(zᶜ), constants.mintke))
        velocityᶠ = [k == 1 ? velocity[1] : k == Nz+1 ? velocity[Nz] : (velocity[k-1] + velocity[k]) / 2 for k in 1:Nz+1]
        ℓᵇ = [k > 1 && N²[k] > 0 ? velocityᶠ[k] / sqrt(N²[k]) : FT(Inf) for k in 1:Nz+1]
        brute_force = [minimum(ℓᵇ[m] + ml.ℂⁱ * abs(zᶠ[k] - zᶠ[m]) for m in 1:Nz+1) for k in 1:Nz+1]
        ℓˢ = column(scratch.stratification_envelope)
        @test all(isfinite, ℓˢ)
        @test isapprox(ℓˢ, brute_force; rtol = 200eps(FT))
        @test all(abs.(diff(ℓˢ)) .≤ ml.ℂⁱ .* diff(zᶠ) .* (1 + 100eps(FT)))   # the gradient limit
        @test column(scratch.gradient_limited_length) == min.(ℓˢ, ml.ℂˢ .* zᶠ)

        # The length of a level: the wall length near the surface, never above the wall length,
        # the envelope carried half a level, nor the local penetration depth
        @test ℓ[1] ≈ ml.ℂˢ * zᶜ[1]
        @test all(ℓ .> 0) && all(isfinite, ℓ)
        @test all(ℓ .≤ ml.ℂˢ .* zᶜ .* (1 + 10eps(FT)))
        carried = [min(ℓˢ[k], ℓˢ[k+1]) + ml.ℂⁱ * (zᶠ[k+1] - zᶠ[k]) / 2 for k in 1:Nz]
        @test all(ℓ .≤ carried .* (1 + 10eps(FT)))
        @test any(ℓ .< FT(0.99) .* ml.ℂˢ .* zᶜ)      # the stratification limits it somewhere
        inside = findfirst(>(1000), zᶜ)
        @test ℓ[inside] < ml.ℂˢ * zᶜ[inside] / 5        # inside the first stable layer

        # Each coefficient acts where it should: ℂˢ on the surface layer only, ℂⁱ away from it
        function lengths(ml)
            context = @allowscalar mixing_length_column!(ml, hard, constants, 1, 1, grid, Nz, zero(FT), state, host,
                                                         fields.diagnostics, scratch)
            return @allowscalar [mixing_length(ml, hard, constants, context, 1, 1, k, grid, zero(FT), zero(FT), zero(FT),
                                               FT(1e5), state, scratch) for k in 1:Nz]
        end
        wider = lengths(GradientLimitedMixingLength(FT(0.9), FT(1.8)))
        steeper = lengths(GradientLimitedMixingLength(FT(0.8), FT(2.4)))
        @test wider[1] ≈ FT(0.9) * zᶜ[1] && wider[inside] == ℓ[inside]
        @test steeper[1] == ℓ[1] && all(steeper .≥ ℓ) && any(steeper .> ℓ)

        # The host mesh caps the length
        capped = @allowscalar mixing_length(ml, hard, constants, context, 1, 1, 3, grid, zero(FT), zero(FT), zero(FT),
                                            FT(7), state, scratch)
        @test capped === FT(7)
    end

    steps = 36 # one hour
    @testset "Single column $case" for case in (:dry_cbl, :bomex)
        original = run_single_column(case, default_arch, FT; steps)
        @test original.closure.mixing_length isa BlendedMixingLength{FT}
        @test profiles_are_finite(original)

        # Naming the default component is the same run, value for value
        named = run_single_column(case, default_arch, FT; steps, mixing_length = BlendedMixingLength(FT))
        for name in (:θ, :q, :e, :ℓ, :tkh, :cloud_fraction)
            @test getproperty(named, name) == getproperty(original, name)
        end
        ml = original.closure.mixing_length
        @test all(ml.minlen .≤ original.ℓ .≤ original.closure.host_grid_spacing)

        gl = calibrated_gradient_limited_length(FT)
        for (label, kw) in ("SHOC with the gradient-limited length" => (; mixing_length = gl),
                            "... and the stability functions and TKE sink of the TKE closure" => eddy_diffusivity_core(FT))
            @testset "$label" begin
                run = run_single_column(case, default_arch, FT; steps, kw...)
                @test run.closure.mixing_length === gl
                @test profiles_are_finite(run)
                @test all(run.e .> 0)
                @test all(0 .≤ run.cloud_fraction .≤ 1)

                # The length is that of the envelope: positive, not above the wall length or the host mesh
                @test all(run.ℓ .> 0)
                @test all(run.ℓ .≤ gl.ℂˢ .* run.z .* (1 + 10eps(FT)))
                @test all(run.ℓ .≤ run.closure.host_grid_spacing)
                @test run.ℓ[1] ≈ gl.ℂˢ * run.z[1]
                @test all(isfinite, column(run.scratch.stratification_envelope))
                @test run.ℓ != original.ℓ

                # Turbulence develops and mixes the surface flux upward
                @test maximum(run.e) > FT(0.05)
                @test maximum(run.tkh) > 1
                if case == :dry_cbl
                    θ₀ = 300 .+ FT(3e-3) .* run.z
                    warmed = run.θ .- θ₀
                    @test warmed[1] > FT(0.3)
                    @test count(>(FT(0.05)), warmed) ≥ 4                     # a layer, not one level
                    @test all(abs.(warmed[run.z .> 2000]) .< FT(1e-2))     # the free troposphere is untouched
                    @test isapprox(run.θ, original.θ; rtol = 3e-3)
                end
            end
        end
    end

    # On a GPU, the same columns on the CPU. In single precision round-off grows with the
    # turbulence of the first steps, so only the mean profiles are compared
    if default_arch isa GPU
        @testset "GPU against CPU, $case" for case in (:dry_cbl, :bomex)
            gl = calibrated_gradient_limited_length(FT)
            tolerance = FT == Float64 ? 1e-9 : 1e-4
            names = FT == Float64 ? (:θ, :q, :e, :ℓ, :tkh) : (:θ, :q)
            for kw in ((; mixing_length = gl), eddy_diffusivity_core(FT))
                gpu = run_single_column(case, GPU(), FT; steps = 12, kw...)
                cpu = run_single_column(case, CPU(), FT; steps = 12, kw...)
                for name in names
                    a, b = getproperty(gpu, name), getproperty(cpu, name)
                    @test isapprox(a, b; rtol = tolerance)
                end
            end
        end
    end
end
