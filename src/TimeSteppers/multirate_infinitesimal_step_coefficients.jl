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
and `γ`, the stage interval fractions `d` (row sums of `β`), the stage abscissae `c`
(stage ``i`` ends at ``tⁿ + cᵢ Δt``), and the strong-stability coefficient of the slow
method (see [`strong_stability_coefficient`](@ref)).
"""
struct MultirateInfinitesimalStepTableau{FT}
    α :: Matrix{FT}
    β :: Matrix{FT}
    γ :: Matrix{FT}
    d :: Vector{FT}
    c :: Vector{FT}
    strong_stability_coefficient :: FT
end

Base.summary(::MultirateInfinitesimalStepTableau{FT}) where FT = "MultirateInfinitesimalStepTableau{$FT}"

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a [`MultirateInfinitesimalStepTableau`](@ref) in float type `FT` from the square,
lower-triangular coefficient matrices `α`, `β` (with diagonal) and `γ` of
[Knoth and Wensch (2014)](@cite KnothWensch2014). The stage interval fractions ``dᵢ = Σⱼ βᵢⱼ``,
the stage abscissae ``cᵢ = dᵢ + Σⱼ (αᵢⱼ + γᵢⱼ) cⱼ`` and the strong-stability coefficient of the
slow method are computed from them.

```jldoctest
using Breeze.TimeSteppers: MultirateInfinitesimalStepTableau

α = zeros(3, 3)
γ = zeros(3, 3)
β = [1//3 0 0; 0 1//2 0; 0 0 1]
tableau = MultirateInfinitesimalStepTableau(Float64, α, β, γ)
tableau.d

# output
3-element Vector{Float64}:
 0.3333333333333333
 0.5
 1.0
```
"""
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

    return MultirateInfinitesimalStepTableau{FT}(α, β, γ, d, c, strong_stability_coefficient(α, β, γ))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The strong-stability (SSP) coefficient ``C`` of the slow method of an MIS coefficient set: the
radius of absolute monotonicity ([Kraaijevanger (1991)](@cite Kraaijevanger1991)) of the
Runge–Kutta method that the set reduces to for a variable with no fast tendency,

```math
Yᵢ = Uⁿ + Σⱼ (αᵢⱼ + γᵢⱼ) (Yⱼ - Uⁿ) + Δt Σⱼ βᵢⱼ S(Yⱼ₋₁) .
```

For ``C > 0`` every stage is a convex combination of forward-Euler steps of length at most
``Δt / C``, each taken from the state its tendency was evaluated at. A tendency that keeps a
variable non-negative over a forward-Euler step of length ``Δt / C`` therefore keeps it
non-negative over the whole step. ``C = 0`` (Wicker–Skamarock RK3, for example) means there is
no such decomposition, and no forward-Euler bound carries over.
"""
function strong_stability_coefficient(α, β, γ)
    s = size(β, 1)

    # ΔYᵢ = Δt Σⱼ K[i, j] S(Yⱼ₋₁)
    K = zeros(s, s)
    for i in 1:s
        K[i, :] .= β[i, :]
        for j in 1:i-1
            K[i, :] .+= (α[i, j] + γ[i, j]) .* K[j, :]
        end
    end

    # Rows are the states Y₀, …, Yₛ and columns the tendencies evaluated at them, so `M` is
    # strictly lower triangular and `I + r M` is inverted by forward substitution.
    n = s + 1
    M = zeros(n, n)
    M[2:n, 1:s] .= K

    # Absolutely monotonic at r: M (I + r M)⁻¹ ≥ 0 and (I + r M)⁻¹ 𝟏 ≥ 0.
    function absolutely_monotonic(r)
        inverse = zeros(n, n)
        for column in 1:n, row in 1:n
            identity = row == column ? 1.0 : 0.0
            inverse[row, column] = identity - r * sum(M[row, m] * inverse[m, column] for m in 1:row-1; init=0.0)
        end
        tolerance = 1e-12
        return all(M * inverse .≥ -tolerance) && all(sum(inverse, dims=2) .≥ -tolerance)
    end

    absolutely_monotonic(1e-9) || return 0.0

    lower, upper = 0.0, 16.0
    for _ in 1:60
        r = (lower + upper) / 2
        absolutely_monotonic(r) ? (lower = r) : (upper = r)
    end

    return lower
end

number_of_stages(tableau::MultirateInfinitesimalStepTableau) = length(tableau.d)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the [`MultirateInfinitesimalStepTableau`](@ref) named `name` in float type `FT`.
Available:

- `:SSPMIS43`: four stages, second order with the fast and slow parts coupled, with a
  strong-stability-preserving slow method. The default of [`MultirateInfinitesimalStep`](@ref).
- `:MIS4a`: four stages, third order.
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
