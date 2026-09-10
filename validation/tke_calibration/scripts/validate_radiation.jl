# Interactive radiation against the LES: RRTMGP on the LES's initial profiles (its own cloud water) in the tall
# column, compared with the LES's RRTM heating over its first hour. Prints, per member, the cloud-top cooling and
# the column-integrated radiative cooling of both, and the RMS difference of the heating profiles on 100 m cells.
#     julia --project scripts/validate_radiation.jl
using BreezeCalibration, Statistics, Printf
using Oceananigans: interior
using Oceananigans.Units
cᵖ = 1004.7
for (s, m) in ((22, "07"), (17, "07"), (3, "01"), (8, "01"), (11, "01"))
    member = load_member(s, m)
    problem = ColumnEnsembleProblem([member]; top = 25_000)
    params = reshape(collect(Float64, default_parameters()), :, 1)
    means, model = run_ensemble(problem, params; stop_time = 60.0, averaging_window = (0.0, 60.0), radiation = :interactive, radiation_interval = 1minute)
    zc = problem.zc; ρ = Array(interior(model.dynamics.reference_state.density))[1, 1, :]
    H = Array(interior(model.radiation.flux_divergence))[1, 1, :]                    # W m⁻³
    rate = 86400 .* H ./ (ρ .* cᵖ)                                                   # K/day
    les_rate = 86400 .* member.heating[:, 1] ./ cᵖ                                  # first hour, K/day, on the LES grid
    # Both on 100 m cells to 3 km
    to = collect(0.0:100.0:4000.0)
    r_scm = regrid_column(rate, problem.zf, to); r_les = regrid_column(les_rate, problem.les_zf, to)
    Δz = diff(problem.zf); kles = problem.zf[2:end] .≤ 4000
    ∫scm = sum(H[kles] .* Δz[kles]); ∫les = sum(member.heating[:, 1] .* ρ[1:200] .* 20)   # W m⁻² over the LES depth
    @printf "site %2d %s: cloud-top cooling SCM %7.1f K/day at %4.0f m (100 m cells: %6.2f) | LES %6.2f K/day at %4.0f m | column ∫ρcₚ dT/dt below 4 km: SCM %6.1f, LES %6.1f W/m² | RMS diff on 100 m cells %.2f K/day\n" s m minimum(rate[1:200]) zc[argmin(rate[1:200])] minimum(r_scm) minimum(les_rate) member.z[argmin(les_rate)] ∫scm ∫les sqrt(mean((r_scm .- r_les) .^ 2))
end
