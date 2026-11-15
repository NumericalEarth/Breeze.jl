#####
##### Coefficient tables for MultirateInfinitesimalStep
#####
##### Convention ([Knoth and Wensch 2014](@cite KnothWensch2014), eqs. 1a–1b and 2): for stage
##### i = 1, …, s, with Y₀ = Uⁿ, ΔYⱼ = Yⱼ − Uⁿ and Fⱼ = f_slow(Yⱼ₋₁), stage i integrates
#####
#####     V(0) = Uⁿ + Σⱼ₌₁^{i-1} αᵢⱼ ΔYⱼ
#####     ∂τ V = f_fast(V) + (1 / dᵢ) [Σⱼ₌₁^{i-1} γᵢⱼ ΔYⱼ / Δt + Σⱼ₌₁^{i} βᵢⱼ Fⱼ],   0 ≤ τ ≤ dᵢ Δt
#####
##### with dᵢ = Σⱼ βᵢⱼ, and sets Yᵢ = V(dᵢ Δt); Uⁿ⁺¹ = Yₛ. The matrices are lower triangular
##### (β including its diagonal), indexed as in ClimaTimeSteppers.jl's `mis.jl`, from which
##### the MIS4a and TVDMISB entries below are transcribed (they in turn transcribe
##### Wensch, Knoth and Galant 2009 and Knoth and Wensch 2014; the MIS4a β₄₁ entry is the
##### authors' corrected value, not the published one).
##### Wicker–Skamarock RK3 is the special case α = γ = 0, β = diag(1/3, 1/2, 1).
#####
##### Order and stability of these sets with *this* fast integrator (linearized,
##### off-centered Crank–Nicolson vertically; horizontally, symmetric substeps for the sets that
##### recombine stage increments and forward–backward ones otherwise, see `SymmetricSubsteps`) are
##### not established by their published slow tableaux; see the MIS analysis scripts.
#####

"""
$(TYPEDEF)

Coefficients of a multirate infinitesimal step (MIS) method in the convention of
[Knoth and Wensch (2014)](@cite KnothWensch2014): lower-triangular `α`, `β` (with diagonal)
and `γ`, the stage interval fractions `d` (row sums of `β`), and the stage abscissae `c`
(stage ``i`` ends at ``tⁿ + cᵢ Δt``).
"""
struct MultirateInfinitesimalStepTableau{FT}
    α :: Matrix{FT}
    β :: Matrix{FT}
    γ :: Matrix{FT}
    d :: Vector{FT}
    c :: Vector{FT}
end

Base.summary(::MultirateInfinitesimalStepTableau{FT}) where FT = "MultirateInfinitesimalStepTableau{$FT}"

function MultirateInfinitesimalStepTableau(FT, α, β, γ)
    α = convert(Matrix{FT}, α)
    β = convert(Matrix{FT}, β)
    γ = convert(Matrix{FT}, γ)
    s = size(β, 1)
    d = [sum(β[i, :]) for i in 1:s]

    # cᵢ = dᵢ + Σⱼ (αᵢⱼ + γᵢⱼ) cⱼ: the time reached by stage i's fast integration.
    c = zeros(FT, s)
    for i in 1:s
        c[i] = d[i]
        for j in 1:i-1
            c[i] += (α[i, j] + γ[i, j]) * c[j]
        end
    end

    return MultirateInfinitesimalStepTableau{FT}(α, β, γ, d, c)
end

number_of_stages(tableau::MultirateInfinitesimalStepTableau) = length(tableau.d)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the [`MultirateInfinitesimalStepTableau`](@ref) named `name` in float type `FT`.
Available:

- `:SSPMIS43`: four stages, second order with the fast and slow parts coupled, with a
  strong-stability-preserving slow method. The default of [`MultirateInfinitesimalStep`](@ref).
- `:MIS4a`: four stages, third order.
- `:ConvexMIS52`, `:ConvexMIS52r20`, `:ConvexMIS42`: five, five and four stages, second order, with convex stage start states and
  non-negative slow-tendency weights (experimental).
- `:TVDMISB`: three stages, second order.
- `:WickerSkamarock3`: the diagonal tableau ``β = \text{diag}(1/3, 1/2, 1)``, ``α = γ = 0``, which
  reproduces [`AcousticRungeKutta3`](@ref) when the linearization is refreshed every stage
  (`relinearize_stages = true`).
"""
multirate_infinitesimal_step_tableau(name::Symbol, FT=Float64) =
    multirate_infinitesimal_step_tableau(Val(name), FT)

multirate_infinitesimal_step_tableau(tableau::MultirateInfinitesimalStepTableau, FT=Float64) =
    MultirateInfinitesimalStepTableau(FT, tableau.α, tableau.β, tableau.γ)

function multirate_infinitesimal_step_tableau(::Val{:WickerSkamarock3}, FT)
    α = zeros(3, 3)
    γ = zeros(3, 3)
    β = [1//3 0    0
         0    1//2 0
         0    0    1]
    return MultirateInfinitesimalStepTableau(FT, α, β, γ)
end

function multirate_infinitesimal_step_tableau(::Val{:MIS4a}, FT)
    α = [ 0                      0                     0                    0
          0.52349249922385610    0                     0                    0
          1.1683374366893629    -0.75762080241712637   0                    0
         -0.036477233846797109   0.56936148730740477   0.47746263002599681  0]
    β = [ 0.38758444641450318    0                     0                    0
         -0.025318448354142823   0.38668943087310403   0                    0
          0.20899983523553325   -0.45856648476371231   0.43423187573425748  0
         -0.10048822195663100   -0.46186171956333327   0.83045062122462809  0.27014914900250392]
    γ = [ 0                      0                     0                    0
          0.13145089796226542    0                     0                    0
         -0.36855857648747881    0.33159232636600550   0                    0
         -0.065767130537473045   0.040591093109036858  0.064902111640806712 0]
    return MultirateInfinitesimalStepTableau(FT, α, β, γ)
end

function multirate_infinitesimal_step_tableau(::Val{:TVDMISB}, FT)
    α = [0                     0                     0
         0.42668232863311001   0                     0
         0.26570779016173801   0.41489966891866698   0]
    β = [ 2//3                  0                     0
         -0.25492859100078202   4//9                  0
         -0.26452517179288798   0.11424084424766399   9//16]
    γ = [0                     0                     0
         0.28904389120139701   0                     0
         0.45113560071334202  -0.25006656847591002   0]
    return MultirateInfinitesimalStepTableau(FT, α, β, γ)
end

# Four stages, second order coupled and third order in the slow part alone, with a strong-stability-
# preserving slow method (SSP coefficient 1.79) and Σdᵢ = 1.15 to 1.18. Found by optimization: the
# order conditions as equality constraints, and CMA-ES over the remaining freedom for the cost per
# simulated time at a fixed acoustic substep. Checked against the analytic order conditions, in a
# 2D linear model with Breeze's own fast integrator and in nonlinear ODE tests, and validated in a
# 3D supercell simulation, where it had the largest stable advective Courant number of the sets tried.
function multirate_infinitesimal_step_tableau(::Val{:SSPMIS43}, FT)
    α = [0 0 0 0
         0.62724492162547518 0 0 0
         0.23430164294884198 1.0654114265586287 0 0
         0.0044647991429055409 0.47676213827145197 0.34061317941416769 0]
    β = [0.55978704832753967 0 0 0
         -0.13703374696667955 0.4455138059089358 0 0
         -0.08356056647074829 0.011356587798664766 0.21157449047031091 0
         -0.07717839985836826 -0.28295021995546038 0.054034557649107431 0.43893059297785891]
    γ = [0 0 0 0
         0.42598752307186377 0 0 0
         -0.033886544164628218 -0.7094153778115867 0 0
         -0.33319589552418832 0.35581112503939216 0.19062097801779304 0]
    return MultirateInfinitesimalStepTableau(FT, α, β, γ)
end

# Five stages, second order coupled. Every stage is a convex combination of the step-start state and earlier
# stage states plus non-negative multiples of the slow tendencies (implemented Shu–Osher coefficient 1.5), and the
# acoustic start states are convex as well. Found by constrained optimization of the linear stable step per cost
# (see the tableau-search scripts): stable to the upwind-5 advective limit of its slow polynomial (Courant 2.58) with
# symmetric substeps, with or without divergence damping.
function multirate_infinitesimal_step_tableau(::Val{:ConvexMIS52}, FT)
    α = [0 0 0 0 0
         0.027106156402681904 0 0 0 0
         0.067188878295825205 0.4736160730706519 0 0 0
         0.10311015215841544 0.64300400308661487 0.089851869962699837 0 0
         0.11663370286563961 0.23755057105835853 0.39110290048156215 0.079322189256211553 0]
    β = [0.60607365591531326 0 0 0 0
         0.0011868576990268209 0.23448522942442709 0 0 0
         0.0026173419314852804 0.0010473060378833245 0.29326962675565887 0 0
         0.0077634771856887668 0.031197390806889932 0.0016270376445233087 0.41175884844844118 0
         0.00053430848625262263 0.0037687546924390033 0.0020199480954774299 0.0010348064840297487 0.22679464582518061]
    γ = [0 0 0 0 0
         0.32783161147279133 0 0 0 0
         0.16384341387368889 -0.026147391782588597 0 0 0
         0.091716600603941725 -0.48551267875773957 0.53306774740651985 0 0
         -0.051396848878430532 -0.20540193543937066 0.16529167970049155 0.2636743706302378 0]
    return MultirateInfinitesimalStepTableau(FT, α, β, γ)
end

# Five stages (N = 5), second order coupled (M = 2); as `:ConvexMIS52`, with a larger implemented Shu–Osher
# coefficient: r = 2.01, stage-scaled r_d = 0.49, Σdᵢ = 1.671, linear advective Courant limit 2.36 with or
# without divergence damping. From the same constrained search (the tableau-search scripts).
function multirate_infinitesimal_step_tableau(::Val{:ConvexMIS52r20}, FT)
    α = [0 0 0 0 0
         0.047988146509683409 0 0 0 0
         0.13369993983380479 0.53721386623251022 0 0 0
         0.3493364187910199 0.31211339542705741 0.1996477869781465 0 0
         0.14979036652619998 0.24545838486793631 0.43405237801609098 0.038902421614832164 0]
    β = [0.48611226255839973 0 0 0 0
         0.0025905961767336075 0.23993248738022652 0 0 0
         0.0043468748180844471 0.0021636972143530427 0.31421106667277793 0 0
         0.0037052042456514832 0.030009085520365322 0.0023610806857223216 0.33513153656185607 0
         0.00032250633695096791 0.0027917652677077187 0.0021270072293294519 0.0012672479402294492 0.24428009326671793]
    γ = [0 0 0 0 0
         0.43489186114856199 0 0 0 0
         -0.11832286498947768 0.096064055497295103 0 0 0
         -0.11983704105423366 -0.25237341584340556 0.47613648418612381 0 0
         -0.036045052866620381 -0.15166826926182883 -0.1355224350239439 0.45287690506363104 0]
    return MultirateInfinitesimalStepTableau(FT, α, β, γ)
end

# Four stages, second order coupled; as `:ConvexMIS52` (implemented Shu–Osher coefficient 1.5, linear advective
# Courant limit 1.87 with or without divergence damping).
function multirate_infinitesimal_step_tableau(::Val{:ConvexMIS42}, FT)
    α = [0 0 0 0
         0.0071760873686080922 0 0 0
         0.30422746030087555 0.4613122625663546 0 0
         0.33017426269827777 0.3386730509152 0.13158351927546355 0]
    β = [0.57869390801943088 0 0 0
         0.0051603819152294377 0.32856028743996452 0 0
         0.0018549407443330876 0.0016866408008596651 0.4386866481107613 0
         0.0011885024303392199 0.0016901415107267457 0.0013995714078297583 0.28211143350099843]
    γ = [0 0 0 0
         0.48732955960831836 0 0 0
         -0.26878121392634668 0.19955071648410094 0 0
         -0.20321594150197009 0.095588515604954211 0.29361821951576456 0]
    return MultirateInfinitesimalStepTableau(FT, α, β, γ)
end
