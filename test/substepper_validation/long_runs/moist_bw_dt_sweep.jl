#####
##### Moist BW Δt sweep — isolation test for Q1.
#####
##### Run the same moist BW lat-lon setup at multiple fixed Δt values.
##### See whether the NaN onset at day 3.3 (Δt=20s) shifts cleanly with
##### Δt or whether it's invariant — distinguishes substepper instability
##### (Δt-shifted onset) from moist-physics runaway (Δt-invariant onset).
#####
##### Each run is capped at 5 simulated days. Purpose is failure-mode
##### diagnosis, not full validation.
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using JLD2
using CloudMicrophysics

const arch = CUDA.functional() ? GPU() : CPU()
const OUTDIR = @__DIR__
const RUN_LABEL = "moist_bw_dt_sweep"

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220.0
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

g   = constants.gravitational_acceleration
Rᵈ  = Breeze.dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
p₀  = 1e5
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

function τ_and_integrals(z; Tᴱ=310.0, Tᴾ=240.0, Γ=0.005, K=3, b=2)
    Tₘ=(Tᴱ+Tᴾ)/2
    Hₛ = Rᵈ * Tₘ / g; η = z/(b*Hₛ); e = exp(-η^2)
    A = (Tₘ-Tᴾ)/(Tₘ*Tᴾ); C = (K+2)/2 * (Tᴱ-Tᴾ)/(Tᴱ*Tᴾ)
    return (exp(Γ*z/Tₘ)/Tₘ + A*(1-2η^2)*e,
            C*(1-2η^2)*e,
            (exp(Γ*z/Tₘ)-1)/Γ + A*z*e,
            C*z*e)
end
F(φ; K=3) = cosd(φ)^K - K/(K+2)*cosd(φ)^(K+2)
dF(φ; K=3) = cosd(φ)^(K-1) - cosd(φ)^(K+1)
ε_v = 0.608

function vT(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end
function pres(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g/Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end
dens(λ, φ, z) = pres(λ, φ, z) / (Rᵈ * vT(λ, φ, z))
function qv(λ, φ, z)
    p = pres(λ, φ, z); η = p/p₀; φʳ = deg2rad(φ)
    qt = 0.018 * exp(-(φʳ/(2π/9))^4) * exp(-((η-1)*p₀/34000)^2)
    return ifelse(η > 0.1, qt, 1e-12)
end
T_phys(λ, φ, z) = vT(λ, φ, z) / (1 + ε_v * qv(λ, φ, z))
θ_pot(λ, φ, z) = T_phys(λ, φ, z) * (p₀/pres(λ, φ, z))^κ
function uvel(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    T = T_phys(λ, φ, z)
    U = g/a * 3 * ∫τ₂ * dF(φ) * T
    rcosφ = a*cosd(φ); Ωrcosφ = Ω*rcosφ
    u_b = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ*U)
    φʳ=deg2rad(φ); λʳ=deg2rad(λ); zₚ=15000.0
    gc = acos(sin(2π/9)*sin(φʳ) + cos(2π/9)*cos(φʳ)*cos(λʳ-π/9))/0.1
    taper = ifelse(z < zₚ, 1 - 3*(z/zₚ)^2 + 2*(z/zₚ)^3, 0.0)
    return u_b + ifelse(gc < 1, taper * exp(-gc^2), 0.0)
end

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)

T_surface_func(λ, φ) = vT(λ, φ, 0.0)
function build_model()
    grid = LatitudeLongitudeGrid(arch; size=(360,160,64), halo=(5,5,5),
                                 longitude=(0,360), latitude=(-80,80), z=(0,30kilometers))
    coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)
    θ_ref(z) = 250 * exp(g*z/(cᵖᵈ*250))
    dyn = CompressibleDynamics(SplitExplicitTimeDiscretization();
                               surface_pressure=p₀, reference_potential_temperature=θ_ref)
    τ_relax = 200.0
    cloud_formation = NonEquilibriumCloudFormation(
        ConstantRateCondensateFormation(1/τ_relax),
        ConstantRateCondensateFormation(1/τ_relax))
    microphysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics(; cloud_formation)
    Cᴰ=1e-3; Uᵍ=1e-2
    bcs = (
        ρu  = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface_func)),
        ρv  = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface_func)),
        ρθ  = FieldBoundaryConditions(bottom=Breeze.BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface_func)),
        ρqᵛ = FieldBoundaryConditions(bottom=Breeze.BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface_func)),
    )
    weno = WENO(); bp_weno = WENO(order=5, bounds=(0,1))
    model = AtmosphereModel(grid; dynamics=dyn, coriolis, microphysics,
                            boundary_conditions=bcs,
                            thermodynamic_constants=constants,
                            momentum_advection=weno,
                            scalar_advection=(ρθ=weno,ρqᵛ=bp_weno,ρqᶜˡ=bp_weno,ρqᶜⁱ=bp_weno,ρqʳ=bp_weno,ρqˢ=bp_weno),
                            timestepper=:AcousticRungeKutta3)
    set!(model, θ=θ_pot, u=uvel, ρ=dens, qᵛ=qv)
    return model
end

# Sweep
sweep_results = NamedTuple[]
for Δt_test in (10.0, 5.0, 2.0)
    @info "============== Δt = $(Δt_test) s =============="
    model = build_model()
    sim = Simulation(model; Δt = Δt_test, stop_time = 5days)

    iters=Int[]; ts=Float64[]; wmax=Float64[]; umax=Float64[]
    qcl=Float64[]; psurf=Float64[]; walls=Float64[]
    wall0 = Ref(time_ns())
    function diag_cb(s)
        m = s.model
        wm = Float64(maximum(abs, interior(m.velocities.w)))
        um = Float64(maximum(abs, interior(m.velocities.u)))
        qm = Float64(maximum(interior(m.microphysical_fields.ρqᶜˡ)))
        pm = Float64(minimum(view(interior(m.dynamics.pressure), :, :, 1)))
        push!(iters, iteration(s)); push!(ts, time(s))
        push!(wmax, wm); push!(umax, um); push!(qcl, qm); push!(psurf, pm)
        push!(walls, (time_ns()-wall0[])/1e9)
        @info @sprintf("[Δt=%g] iter=%6d t=%6.3fd max|u|=%.2f max|w|=%.3e max(ρqcl)=%.2e p_surf=%.0f wall=%.0fs",
                       Δt_test, iteration(s), time(s)/86400, um, wm, qm, pm, walls[end])
        flush(stdout)
    end
    # diagnostics every fixed wall-time interval to keep log size manageable
    add_callback!(sim, diag_cb, IterationInterval(max(1, Int(round(2000/Δt_test)))))

    wall0[] = time_ns()
    crashed = false
    try
        run!(sim)
    catch e
        crashed = true
        @error "[Δt=$(Δt_test)] CRASHED" e
    end
    push!(sweep_results, (Δt = Δt_test, iters=iters, ts=ts, wmax=wmax, umax=umax,
                          qcl=qcl, psurf=psurf, walls=walls, crashed=crashed,
                          final_t=time(sim), final_iter=iteration(sim)))
end

jldsave(joinpath(OUTDIR, RUN_LABEL * "_results.jld2"); sweep_results)
@info "Done."
for r in sweep_results
    @info @sprintf("Δt=%g s: completed %g days (%d iters), crashed=%s",
                   r.Δt, r.final_t/86400, r.final_iter, r.crashed)
end
