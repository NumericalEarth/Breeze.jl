# Usage: julia --project=validation/sedimentation_energy validation/sedimentation_energy/run_campaign.jl cpu 64 results.toml
using TOML
using SHA
include("experiments.jl")

backend = get(ARGS, 1, "cpu")
precision = parse(Int, get(ARGS, 2, "64"))
output = get(ARGS, 3, "sedimentation-results.toml")
suite = get(ARGS, 4, "isolated")
suite in ("isolated", "coupled") || error("Suite must be isolated or coupled")
FT = precision == 32 ? Float32 : Float64
precision in (32, 64) || error("Precision must be 32 or 64")
if backend == "cuda"
    @eval using CUDA
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
    arch = GPU()
    device = string(CUDA.device())
    CUDA.versioninfo()
elseif backend == "metal"
    precision == 32 || error("Metal requires Float32")
    @eval using Metal
    Metal.functional() || error("Metal is not functional")
    Metal.allowscalar(false)
    arch = GPU(Metal.MetalBackend())
    device = string(Metal.device())
    Metal.versioninfo()
elseif backend == "cpu"
    arch = CPU()
    device = Sys.CPU_NAME
else
    error("Backend must be cpu, cuda, or metal")
end

as_dictionary(x::NamedTuple) = Dict(string(k) => as_dictionary(v) for (k, v) in pairs(x))
as_dictionary(x::AbstractArray) = map(as_dictionary, x)
as_dictionary(x) = x

# Differentiate the executable theta definition at fixed dry/vapor partial densities.
# Host Float64 analysis of tendencies computed entirely on the selected architecture.
function compressible_temperature_rate(model, condensate_rate, thermal_rate)
    s = state_arrays(model)
    a = s.ρ .- s.ρv .- s.ρr
    constants = model.thermodynamic_constants
    gas = a .* Float64(TH.dry_air_gas_constant(constants)) .+
          s.ρv .* Float64(TH.vapor_gas_constant(constants))
    pressure = gas .* s.T
    κ = gas ./ s.C
    latent = s.ρr .* s.L ./ s.C
    pressure_log = log.(pressure ./ Float64(AM.standard_pressure(model.dynamics)))
    exner = exp.(κ .* pressure_log)
    derivative_condensate = a ./ exner .* (-(s.L .* s.C .- s.ρr .* s.L .* s.cx) ./ s.C.^2 .+
                            (s.T .- latent) .* pressure_log .* gas .* s.cx ./ s.C.^2)
    derivative_temperature = a ./ exner .* (1 .- κ .* (s.T .- latent) ./ s.T)
    return (thermal_rate .- derivative_condensate .* condensate_rate) ./ derivative_temperature
end

function instantaneous_case(; volume_test=false, kwargs...)
    model = make_column(; core=:compressible, FT, arch, kwargs...)
    mass_rate = Float64.(host_column(rain_tendency(model)))
    original_rate = compressible_temperature_rate(model, mass_rate, Float64.(column(thermal_tendency(model))))
    phase_rate = compressible_temperature_rate(model, mass_rate, Float64.(column(thermal_tendency(model; corrected=true))))
    s = state_arrays(model)
    a = s.ρ .- s.ρv .- s.ρr
    constants = model.thermodynamic_constants
    gas = a .* Float64(TH.dry_air_gas_constant(constants)) .+
          s.ρv .* Float64(TH.vapor_gas_constant(constants))
    # Conditional first law: isolated upwind fall, fixed gas partial densities,
    # no kinetic/gravitational energy or drag; only cell 1 receives colder mass.
    internal_energy_reference = [s.cx * (s.T[2] - s.T[1]) * mass_rate[1] / (s.C[1] - gas[1]), 0.0]
    @test all(isfinite, original_rate)
    @test all(isfinite, phase_rate)
    extra = if volume_test
        volume_rate = compressible_temperature_rate(model, mass_rate,
                      Float64.(column(thermal_tendency(model; volume_corrected=true))))
        (; volume_rate)
    else
        (;)
    end
    return (; mass_rate, original_rate, phase_rate, internal_energy_reference, extra...)
end

root = normpath(joinpath(@__DIR__, "../.."))
result = Dict{String, Any}(
    "source_commit" => strip(read(`git -C $root rev-parse HEAD`, String)),
    "production_base" => "dae9e46d543720f4f1c4f3a57e8d7c7817e90f24",
    "source_dirty" => !isempty(read(`git -C $root status --porcelain`, String)),
    "julia" => string(VERSION), "oceananigans" => string(pkgversion(Oceananigans)),
    "backend" => backend, "device" => device, "precision" => precision,
    "suite" => suite,
    "campaign_sha256" => Dict(f => bytes2hex(sha256(read(joinpath(@__DIR__, f))))
                              for f in ("experiments.jl", "run_campaign.jl", "Project.toml", "Manifest.toml")),
    "analysis_precision" => 64, "isothermal_rate_tolerance" => (precision == 32 ? 2e-6 : 2e-11),
    "cases" => Any[])
function save_case(name, data)
    push!(result["cases"], merge(Dict("name" => name), as_dictionary(data)))
    open(output, "w") do io
        TOML.print(io, result)
    end
    println("COMPLETED ", name, " ", data)
    flush(stdout)
end

@testset "Sedimentation campaign ($backend, Float$precision)" begin
    if suite == "isolated"
    for phase in (:liquid, :ice), scheme_kind in (:upwind, :weno, :bounded), dz in (50.0, 100.0)
        for control in (:contrast, :equal, :nofall)
            data = instantaneous_case(; phase, scheme_kind, dz, speed=control === :nofall ? 0.0 : 1.0,
                                      equal_composition=control === :equal, closed=true)
            tolerance = result["isothermal_rate_tolerance"]
            save_case("instantaneous-$phase-$scheme_kind-dz$dz-$control",
                      (; phase=string(phase), scheme=string(scheme_kind), dz, control=string(control), data...,
                         original_isothermal_pass=maximum(abs, data.original_rate) < tolerance,
                         phase_isothermal_pass=maximum(abs, data.phase_rate) < tolerance,
                         closed_column_mass_rate=sum(data.mass_rate) * dz))
            @test maximum(abs, data.phase_rate) < tolerance
            if control !== :contrast
                @test maximum(abs, data.original_rate) < tolerance
            end
        end
    end
    for scheme_kind in (:upwind, :weno, :bounded)
        model = make_column(; core=:compressible, Nz=32, horizontal_size=4, dz=20.0, speed=1.0, closed=true,
                              smooth_profile=true, scheme_kind, bounds=(0.001, 0.01), FT, arch)
        mass_rate = Float64.(host_column(rain_tendency(model)))
        original_rate = compressible_temperature_rate(model, mass_rate, Float64.(column(thermal_tendency(model))))
        phase_rate = compressible_temperature_rate(model, mass_rate, Float64.(column(thermal_tendency(model; corrected=true))))
        changes = scheme_kind === :bounded ? reconstruction_changes(model) : (;)
        save_case("interior-32-$scheme_kind", (; mass_rate, original_rate, phase_rate, changes...,
                  advection_type=string(typeof(model.advection.ρqʳ)), horizontal_size=4,
                  closed_column_mass_rate=sum(mass_rate) * 20,
                  relative_mass_rate_residual=sum(mass_rate) / sum(abs, mass_rate)))
        if scheme_kind === :bounded
            @test model.advection.ρqʳ isa Oceananigans.Advection.BoundsPreservingWENO
            @test maximum(changes.limiter_change[4:29]) > 1e-8
            @test maximum(changes.high_order_change[4:29]) > 1e-6
        end
    end
    for phase in (:liquid, :ice)
        for upper_temperature in (275.0, 285.0)
            data = instantaneous_case(; phase, upper_temperature, dz=100.0, speed=1.0, closed=true, volume_test=true)
            save_case("nonisothermal-$phase-T$upper_temperature-conditional-internal-energy",
                      (; phase=string(phase), upper_temperature, data...))
            @test maximum(abs.(data.volume_rate .- data.internal_energy_reference)) < result["isothermal_rate_tolerance"]
        end
        for resolved_velocity in (-1.0, 0.5, 2.0)
            data = instantaneous_case(; phase, resolved_velocity, dz=100.0, speed=1.0, closed=true)
            save_case("donor-reversal-$phase-w$resolved_velocity", (; phase=string(phase), resolved_velocity, data...))
            @test maximum(abs, data.phase_rate) < result["isothermal_rate_tolerance"]
        end
    end
    # Diagnostic Euler refinement checks the independently differentiated T response.
    for phase in (:liquid, :ice), corrected in (false, true), Δt in (0.1, 0.01, 0.001)
        data = explicit_case(; core=:compressible, phase, corrected, Δt, dz=100.0, speed=1.0, FT, arch)
        save_case("euler-$phase-phaseh$corrected-dt$Δt", (; phase=string(phase), phase_enthalpy=corrected, Δt,
                  temperature_rate=Float64.(data.temperature_rate), mass_rate=Float64.(data.mass_rate)))
    end
    # Conditional anelastic finite thermal balance, NOT compressible total energy.
    for phase in (:liquid, :ice), formulation in (:LiquidIcePotentialTemperature, :StaticEnergy),
        Δt in (1.5, 2.5, 10.0), closed in (false, true)
        data = implicit_case(; Δt, formulation, phase, closed, FT, arch)
        save_case("implicit-$phase-$formulation-dt$Δt-closed$closed", (; closed, data...))
        if formulation === :StaticEnergy
            @test maximum(abs, data.temperature_error) < (precision == 32 ? 2e-4 : 2e-10)
        end
    end
    else
        for implicit in (false, true), Δt in (implicit ? (0.1, 2.0) : (0.1, 0.025)), speed in (0.0, 8.0)
            data = coupled_case(; Δt, steps=round(Int, 2.0 / Δt), implicit, speed, FT, arch)
            save_case("acoustic-implicit$implicit-dt$Δt-speed$speed", data)
        end
    end
end
result["completed"] = true
open(output, "w") do io
    TOML.print(io, result)
end
