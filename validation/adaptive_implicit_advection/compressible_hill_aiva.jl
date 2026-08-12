# Reproducer for issue #897: adaptive implicit vertical advection in the compressible
# split-explicit core. Hydrostatic adiabatic initialization, 15 m/s flow over a 1200 m
# hill, Δz = 100 m. Explicit WENO(5) integrates stably past its nominal vertical CFL;
# AIVA — whose purpose is exactly that regime — destabilizes within iterations because
# the implicit remainder is applied once per RK stage outside the acoustic loop
# (see the issue for the phase-resolved mechanism).
using Breeze
using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Breeze.CompressibleEquations: CompressibleDynamics, SplitExplicitTimeDiscretization
using Breeze.Thermodynamics: adiabatic_hydrostatic_density
using Oceananigans: prognostic_fields
using Printf

Oceananigans.defaults.FloatType = Float32

const Nx, Nz = 64, 80
const Lx, Lz = 32e3, 8e3

function build(; scalar_aiva, momentum_aiva=false)
    z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1));
                                                     formulation = LinearDecay())
    grid = RectilinearGrid(CPU(); size=(Nx, Nz), halo=(5, 5),
                           x=(-Lx/2, Lx/2), z=z_faces,
                           topology=(Periodic, Flat, Bounded))
    materialize_terrain!(grid, x -> 1200 * exp(-x^2 / (2 * (2.5e3)^2)))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    reference_potential_temperature = 300)
    aiva() = WENO(order=5, time_discretization=AdaptiveVerticallyImplicitDiscretization(Float32; cfl=0.5))
    model = AtmosphereModel(grid; dynamics,
                            momentum_advection = momentum_aiva ? aiva() : WENO(order=5),
                            scalar_advection = (; ρθ = scalar_aiva ? aiva() : WENO(order=5)))
    ρᵢ(x, z) = adiabatic_hydrostatic_density(z, 101325.0, 300.0, 1e5, model.thermodynamic_constants)
    set!(model, ρ=ρᵢ, θ=300, u=15)
    return model
end

function run_case(label; Δt=6.0, N=400, kw...)
    println(label, ":")
    model = build(; kw...)
    ρθ = prognostic_fields(model).ρθ
    ρᵈ = prognostic_fields(model).ρᵈ
    for n in 1:N
        try
            time_step!(model, Δt)
        catch err
            @printf("  THROWN at iter %d: min(ρθ)=%.4g min(ρᵈ)=%.4g — %s\n", n,
                    minimum(interior(ρθ)), minimum(interior(ρᵈ)),
                    first(sprint(showerror, err), 60))
            return
        end
        ρθmin = minimum(interior(ρθ)); ρᵈmin = minimum(interior(ρᵈ))
        w = maximum(abs, interior(model.velocities.w))
        if !isfinite(w) || ρθmin < 0 || ρᵈmin < 0
            @printf("  BAD at iter %d: |w|=%g min(ρθ)=%.4g min(ρᵈ)=%.4g\n", n, w, ρθmin, ρᵈmin)
            return
        end
        n % 100 == 0 && @printf("  iter %3d: |w|max=%.3f min(ρθ)=%.3f min(ρᵈ)=%.4f\n", n, w, ρθmin, ρᵈmin)
    end
    println("  survived $(N) iterations")
end

# Δz = 100 m; w over the hill ≈ 7 m/s ⇒ α ≈ 0.07·Δt. Explicit should fail past its
# vertical CFL; AIVA should carry through — that is its entire purpose.
run_case("Δt=10 explicit (α≈0.7)";  Δt=10.0, scalar_aiva=false)
run_case("Δt=10 AIVA both";         Δt=10.0, scalar_aiva=true, momentum_aiva=true)
run_case("Δt=25 explicit (α≈1.8)";  Δt=25.0, scalar_aiva=false)
run_case("Δt=25 AIVA both";         Δt=25.0, scalar_aiva=true, momentum_aiva=true)
run_case("Δt=50 AIVA both (α≈3.5)"; Δt=50.0, scalar_aiva=true, momentum_aiva=true)
