# T17-mp1m: stored temperature (MP1M kernel) and stored condensate content (explicit sedimentation)
# must give bit-identical tendencies and states to the per-cell rediagnosis, including cells with
# zero, negative zero, negative and NaN precipitation. Usage: julia --project=<env> test/t17_mp1m_stored_bits.jl
using Breeze, Oceananigans
using Oceananigans.TimeSteppers: update_state!, time_step!
using UUIDs: UUID
Base.require(Base.PkgId(UUID("6a9e3e04-43cd-43ba-94b9-e8782df3c71b"), "CloudMicrophysics"))
const nfail = Ref(0)
macro test(ex) :( $(esc(ex)) || (nfail[] += 1; println("FAIL: ", $(string(ex)))) ) end
const CE = Breeze.CompressibleEquations
const OneMomentCloudMicrophysics = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt).OneMomentCloudMicrophysics

function build(FT)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(CPU(), FT; size=(6, 5, 16), x=(0, 6e3), y=(0, 5e3), z=(0, 8e3))
    constants = ThermodynamicConstants(FT)
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation=SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT)))
    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=290)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)
    Tᵢ(x, y, z) = FT(295) - FT(0.0065) * z + FT(0.3) * sin(x / 1e3)
    ρᵢ(x, y, z) = FT(1.2) * exp(-z / 8000)
    qᵗ(x, y, z) = FT(0.014) * exp(-z / 2500) * (1 + FT(0.2) * cos(y / 900))
    qʳ(x, y, z) = FT(1e-3) * exp(-((z - 3000) / 1500)^2) * (1 + sin(x / 700))
    qˢ(x, y, z) = FT(8e-4) * exp(-((z - 5500) / 1200)^2)
    set!(model; ρ=ρᵢ, T=Tᵢ, qᵗ=qᵗ, qʳ=qʳ, qˢⁿ=qˢ, u=FT(3), w=(x, y, z) -> FT(0.5) * sin(x / 1e3) * sin(π * z / 8e3))
    # special values in a few cells (precipitation densities)
    ρqʳ = interior(model.microphysical_fields.ρqʳ); ρqˢ = interior(model.microphysical_fields.ρqˢⁿ)
    ρqʳ[1, 1, 3] = 0; ρqʳ[2, 1, 3] = -zero(FT); ρqʳ[3, 1, 3] = FT(-1e-9); ρqʳ[4, 1, 3] = FT(-1e-5)
    ρqˢ[1, 2, 9] = 0; ρqˢ[2, 2, 9] = -zero(FT); ρqˢ[3, 2, 9] = FT(-1e-7); ρqʳ[5, 5, 12] = FT(NaN)
    return model
end

bitsof(f) = (a = vec(Array(interior(f))); reinterpret(eltype(a) == Float32 ? UInt32 : UInt64, a))

function arm(FT, on; steps=3, temperature=on, content=on)
    CE.STORED_MICROPHYSICS_TEMPERATURE[] = temperature
    CE.STORED_CONDENSATE_CONTENT[] = content
    model = build(FT)
    @test (Breeze.AtmosphereModels.stored_microphysics_temperature(model) !== nothing) == temperature
    @test (Breeze.AtmosphereModels.stored_condensate_content(model) !== nothing) == content
    update_state!(model); Breeze.AtmosphereModels.compute_tendencies!(model)
    G = Dict(n => bitsof(g) for (n, g) in pairs(model.timestepper.Gⁿ))
    for _ in 1:steps; time_step!(model, FT(2)); end
    S = Dict(n => bitsof(f) for (n, f) in pairs(Oceananigans.prognostic_fields(model)))
    return G, S
end

for FT in (Float32, Float64)
    Gon, Son = arm(FT, true; temperature = get(ENV, "ARM", "both") != "content", content = get(ENV, "ARM", "both") != "temperature"); Goff, Soff = arm(FT, false)
    for n in keys(Goff); @test Gon[n] == Goff[n]; end
    for n in keys(Soff); @test Son[n] == Soff[n]; d = findall(Son[n] .!= Soff[n]); isempty(d) || println(n, " differ at ", length(d), " cells; first ", d[1:min(3,end)], " on=", reinterpret(FT, Son[n][d[1:min(3,end)]]), " off=", reinterpret(FT, Soff[n][d[1:min(3,end)]]), " on bits=", string.(Son[n][d[1:min(3,end)]], base=16), " off bits=", string.(Soff[n][d[1:min(3,end)]], base=16), " nonNaN diffs=", count(i -> !(isnan(reinterpret(FT, Son[n][i:i])[1]) && isnan(reinterpret(FT, Soff[n][i:i])[1])), d)); end
    nnan = count(isnan, reinterpret(FT, Goff[:ρqʳ]))
    println("$FT: tendencies $(length(Goff)) fields, states $(length(Soff)) fields bit-identical; NaN tendency cells ρqʳ = $nnan")
end
CE.STORED_MICROPHYSICS_TEMPERATURE[] = true; CE.STORED_CONDENSATE_CONTENT[] = true
println(nfail[] == 0 ? "ALL BIT-IDENTICAL" : "FAILURES: $(nfail[])")
