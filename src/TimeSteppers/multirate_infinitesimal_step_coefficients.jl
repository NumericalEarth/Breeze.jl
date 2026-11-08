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
##### the MIS2, MIS3C, MIS4, MIS4a, TVDMISA and TVDMISB entries below are transcribed
##### (they in turn transcribe Wensch, Knoth and Galant 2009 and Knoth and Wensch 2014;
##### the MIS4a β₄₁ entry is the authors' corrected value, not the published one).
##### Wicker–Skamarock RK3 is the special case α = γ = 0, β = diag(1/3, 1/2, 1).
#####
##### Order and stability of these sets with *this* fast integrator (linearized,
##### off-centered Crank–Nicolson vertically, forward–backward horizontally) are not
##### established by their published slow tableaux; see the MIS analysis scripts.
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
$(TYPEDSIGNATURES)

Return the [`MultirateInfinitesimalStepTableau`](@ref) named `name` in float type `FT`.
Available: `:WickerSkamarock3`, `:MIS2`, `:MIS3C`, `:MIS4`, `:MIS4a`, `:TVDMISA`, `:TVDMISB`, and the
experimental optimized sets `:MIS3opt`, `:SSPMIS43`, `:SSPMIS42`.
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

function multirate_infinitesimal_step_tableau(::Val{:MIS2}, FT)
    α = [0                 0                0
         0.536946566710    0                0
         0.480892968551    0.500561163566   0]
    β = [ 0.126848494553    0                 0
         -0.784838278826    1.37442675268     0
         -0.0456727081749  -0.00875082271190  0.524775788629]
    γ = [ 0                 0                0
          0.652465126004    0                0
         -0.0732769849457   0.144902430420   0]
    return MultirateInfinitesimalStepTableau(FT, α, β, γ)
end

function multirate_infinitesimal_step_tableau(::Val{:MIS3C}, FT)
    α = [0                0                0
         0.589557277145   0                0
         0.544036601551   0.565511042564   0]
    β = [ 0.397525189225    0                0
         -0.227036463644    0.624528794618   0
         -0.00295238076840 -0.270971764284   0.671323159437]
    γ = [ 0                 0                 0
          0.142798786398    0                 0
         -0.0428918957402   0.0202720980282   0]
    return MultirateInfinitesimalStepTableau(FT, α, β, γ)
end

function multirate_infinitesimal_step_tableau(::Val{:MIS4}, FT)
    α = [0                0                 0                0
         0.914092810304   0                 0                0
         1.14274417397   -0.295211246188    0                0
         0.112965282231   0.337369411296    0.503747183119   0]
    β = [0.136296478423    0                 0                0
         0.280462398979   -0.0160351333596   0                0
         0.904713355208   -1.04011183154     0.652337563489   0
         0.0671969845546  -0.365621862610   -0.154861470835   0.970362444469]
    γ = [ 0                0                 0                 0
          0.678951983291   0                 0                 0
         -1.38974164070    0.503864576302    0                 0
         -0.375328608282   0.320925021109   -0.158259688945    0]
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

function multirate_infinitesimal_step_tableau(::Val{:TVDMISA}, FT)
    α = [0                    0                    0
         0.1946360605647457   0                    0
         0.3971200136786614   0.2609434606211801   0]
    β = [ 2//3                 0                    0
         -0.28247174703488398  4//9                 0
         -0.31198081960042401  0.18082737579913699  9//16]
    γ = [0                    0                    0
         0.5624048933209129   0                    0
         0.4408467475713277  -0.2459300561692391   0]
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

#####
##### Experimental optimized sets (breeze-mis optimizer v2: order conditions as equality constraints, CMA-ES over
##### the remaining freedom; objective = cost per simulated time with the acoustic substep fixed, i.e.
##### w_slow·s/C + w_fast·Σdᵢ). Verified in a 2D linear model with Breeze's actual fast integrator, by analytic
##### order conditions and nonlinear ODE tests — NOT yet validated in nonlinear 3D simulations. Labeled experimental.
#####

# Experimental: 3-stage, coupled 2nd / slow 3rd order, minimized Σdᵢ (1.02) at the RK3 advective limit C_max = 1.43; pure-fast amplification 1.10 (relies on divergence damping).
function multirate_infinitesimal_step_tableau(::Val{:MIS3opt}, FT)
    α = [0 0 0
         0.81787672620735752 0 0
         -0.032272457774980737 1.0424429521043383 0]
    β = [0.62058614776978571 0 0
         -0.25432774773148764 0.3588923369108708 0
         -0.039490893678879285 -0.41234937297395752 0.74831161613593478]
    γ = [0 0 0
         0.088039904277198405 0 0
         -0.073597787227516837 0.11123416923677598 0]
    return MultirateInfinitesimalStepTableau(FT, α, β, γ)
end

# Experimental: 4-stage, coupled 2nd / slow 3rd order, strict SSP (C_SSP = 1.79), Σdᵢ = 1.15–1.18; pure-fast amplification 1.00.
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

# Experimental: 4-stage, coupled 2nd / slow 2nd order, strict SSP (C_SSP = 1.78), Σdᵢ = 1.03.
function multirate_infinitesimal_step_tableau(::Val{:SSPMIS42}, FT)
    α = [0 0 0 0
         0.93090262262783308 0 0 0
         0.055075823891310327 0.90859594461706827 0 0
         0.12876158840650917 0.010471028874744041 0.94687250883678153 0]
    β = [0.51441875850385166 0 0 0
         -0.17895800913077078 0.3609653018802319 0 0
         -0.085537006406372754 -0.11757058350895166 0.32683431177008909 0
         0.015048779441518256 -0.044308807401374599 -0.12312350260141104 0.35839537060819626]
    γ = [0 0 0 0
         0.0642695535248788 0 0 0
         -0.04057498955669165 0.010035624487130182 0 0
         -0.034722055023560496 -0.060694567163454026 0.068480072531976369 0]
    return MultirateInfinitesimalStepTableau(FT, α, β, γ)
end
