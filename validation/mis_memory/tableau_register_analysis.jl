# Minimum full-state registers of a multirate infinitesimal step tableau, and whether its slow
# method admits a low-storage form.
#
#   julia validation/mis_memory/tableau_register_analysis.jl [coefficients file]
#
# The coefficients file defaults to src/TimeSteppers/multirate_infinitesimal_step_coefficients.jl;
# pass the one of another branch to analyse its sets. Needs only LinearAlgebra.
#
# Conventions. Stage i = 1, …, s starts from Uⁿ + zᵢ, zᵢ = Σⱼ αᵢⱼ ΔYⱼ, and is forced with
# fᵢ + (βᵢᵢ/dᵢ) Sᵢ, fᵢ = (1/dᵢ) Σⱼ (γᵢⱼ ΔYⱼ/Δt + βᵢⱼ Sⱼ) (j < i). A register is one array per
# prognostic field in the time stepper; the model's own prognostic field is not counted, the
# tendency storage G is. Wicker–Skamarock RK3 needs 2 (Uⁿ and G).
#
# Counts, each "2 + max over stages of a rank" or "1 + …", with ΔYⱼ and Sⱼ independent symbols
# unless stated:
#
#   acoustic     start state and forcing separate (the acoustic loop integrates from one with the
#                other). ΔYᵢ₋₁ is read from the state when stage i is assembled.
#   stored       as acoustic, but the state is changed between stages (user tracers), so ΔYᵢ is
#                captured when stage i ends.
#   merged       slow-only scalar that the loop does not read: Yᵢ = c₀ᵢ Uⁿ + Σⱼ Rᵢⱼ Yⱼ + Δt Σⱼ βᵢⱼ Sⱼ,
#                R = α + γ, c₀ᵢ = 1 − Σⱼ Rᵢⱼ. Round-off-level change; valid with implicit solves.
#   explicit     as merged, with ΔYᵢ = Δt Σⱼ Aᵢⱼ Sⱼ, A = (I − R)⁻¹ β: valid only if nothing but the
#                explicit update changes the scalar (no implicit solve).

using LinearAlgebra

module Coefficients
    macro doc_str(s) nothing end
    const TYPEDEF = ""
    const TYPEDSIGNATURES_WITH_DEFAULTS = ""
    include(isempty(ARGS) ? joinpath(@__DIR__, "..", "..", "src", "TimeSteppers",
                                     "multirate_infinitesimal_step_coefficients.jl") : ARGS[1])
end

tableau(name) = Coefficients.multirate_infinitesimal_step_tableau(Val(name), Float64)
names = [m.sig.parameters[2].parameters[1] for m in methods(Coefficients.multirate_infinitesimal_step_tableau)
         if m.sig.parameters[2] isa DataType && m.sig.parameters[2] <: Val]

rank_of(M) = isempty(M) ? 0 : rank(M; rtol = 1e-9)
rows(vectors) = isempty(vectors) ? zeros(0, 0) : permutedims(reduce(hcat, vectors))

# Columns: ΔY₁ … ΔY_n, then S₁ … S_m.
z_row(α, k, n, m) = vcat(α[k, 1:n], zeros(m))
f_row(β, γ, d, k, n, m) = vcat(γ[k, 1:n], β[k, 1:m]) ./ d[k]

function acoustic_registers(t; stored = false)
    (; α, β, γ, d) = t
    s = length(d)
    needed = 0
    per_stage = Int[]
    for i in 1:s
        # During the loop of stage i: ΔY₁ … ΔYᵢ₋₁ and S₁ … Sᵢ are known.
        n, m = i - 1, i
        future = vcat([z_row(α, k, n, m) for k in i+1:s], [f_row(β, γ, d, k, n, m) for k in i+1:s])
        zᵢ = z_row(α, i, n, m)
        if stored || i == 1
            during = rank_of(rows(vcat(future, [zᵢ])))
        else
            e = zeros(n + m); e[n] = 1
            with_state = vcat(future, [e])
            during = rank_of(rows(future)) + (rank_of(rows(vcat(with_state, [zᵢ]))) > rank_of(rows(with_state)))
        end
        # After stage i, with ΔYᵢ captured.
        after = stored ? rank_of(rows(vcat([z_row(α, k, i, i) for k in i+1:s],
                                           [f_row(β, γ, d, k, i, i) for k in i+1:s]))) : 0
        push!(per_stage, max(during, after))
    end
    return 2 + maximum(per_stage), per_stage
end

function merged_registers(t; stored)
    (; α, β, γ, d) = t
    s = length(d)
    R = α + γ
    c₀ = [1 - sum(R[k, :]) for k in 1:s]
    ranks = Int[]
    for i in 1:s-1
        n = stored ? i : i - 1 # stage states held when stage i has been assembled (and, stored, has ended)
        push!(ranks, rank_of(hcat(c₀[i+1:s], R[i+1:s, 1:n], β[i+1:s, 1:i])))
    end
    return 1 + maximum(ranks), ranks
end

effective_runge_kutta(t) = (I - (t.α + t.γ)) \ t.β

function explicit_registers(t)
    A = effective_runge_kutta(t)
    s = size(A, 1)
    ranks = [rank_of(A[i+1:s, 1:i] .- A[i:i, 1:i]) for i in 1:s-1]
    return 1 + maximum(ranks), ranks
end

# Williamson 2N: Yᵢ − Yᵢ₋₁ = Bᵢ qᵢ, qᵢ = Aᵢ qᵢ₋₁ + Δt Sᵢ, so each row of the differences
# Dᵢⱼ = Aᵢⱼ − Aᵢ₋₁,ⱼ (j < i) is parallel to the row before it, diagonal included. Returns the largest
# normalized 2×2 minor that must vanish.
function williamson_defect(t)
    A = effective_runge_kutta(t)
    s = size(A, 1)
    D = A .- vcat(zeros(1, s), A[1:s-1, :])
    defect = 0.0
    for i in 3:s
        a, b = D[i, 1:i-1], D[i-1, 1:i-1]
        defect = max(defect, sqrt(max(0, 1 - (dot(a, b) / (norm(a) * norm(b)))^2)))
    end
    return defect
end

# The smallest singular value, relative to the largest, of each Dᵢ that must drop rank for one
# register besides G (the storage of Wicker–Skamarock RK3): rank Dᵢ ≤ 1 for every i.
function two_register_defect(t)
    A = effective_runge_kutta(t)
    s = size(A, 1)
    defect = 0.0
    for i in 2:s-2
        σ = svdvals(A[i+1:s, 1:i] .- A[i:i, 1:i])
        defect = max(defect, σ[2] / σ[1])
    end
    return defect
end

current_registers(s) = 3 + 2 * (s - 2)                       # U⁰, Gⁿ, Uⁿ and 2(s − 2) accumulators
rotated_registers(s) = 2 + (s - 2) + 1 + max(s - 4, 0)       # this branch, increments read from the state
stored_registers(s) = 2 + 2 * (s - 2)                        # this branch, increments stored (user tracers)

println(rpad("set", 18), "s  current  branch  branch(tracer) | acoustic  stored | merged  merged(stored)  explicit | 2N defect  2-register defect")
for name in names
    t = tableau(name)
    s = length(t.d)
    a, a_stage = acoustic_registers(t)
    st, st_stage = acoustic_registers(t; stored = true)
    m, m_rank = merged_registers(t; stored = false)
    ms, ms_rank = merged_registers(t; stored = true)
    e, e_rank = explicit_registers(t)
    println(rpad(string(name), 18), s, "  ", lpad(current_registers(s), 7), lpad(rotated_registers(s), 8),
            lpad(stored_registers(s), 16), " | ", lpad(a, 8), lpad(st, 8), " | ", lpad(m, 6), lpad(ms, 16), lpad(e, 10), " | ",
            lpad(round(williamson_defect(t), sigdigits = 2), 9), lpad(round(two_register_defect(t), sigdigits = 2), 18))
    println("    per stage: acoustic ", a_stage, ", stored ", st_stage, "; ranks: merged ", m_rank, ", merged(stored) ", ms_rank,
            ", explicit ", e_rank)
    A = effective_runge_kutta(t)
    println("    effective slow Runge–Kutta matrix A = (I − α − γ)⁻¹ β:")
    foreach(i -> println("      ", round.(A[i, 1:i], sigdigits = 6)), 1:s)
end

#####
##### Numerical check of the explicit count: integrate y′ = L y with one step of each form
#####

# The implemented form: start state Uⁿ + Σ α ΔY, forcing (1/d)[Σ γ ΔY / Δt + Σ β S], update over d Δt.
function implemented_step(t, f, Uⁿ::Vector{T}, Δt) where T
    α, β, γ, d = T.(t.α), T.(t.β), T.(t.γ), T.(t.d)
    s = length(d)
    ΔY = Vector{Vector{T}}()
    S = Vector{Vector{T}}()
    Y = Uⁿ
    for i in 1:s
        push!(S, f(Y))
        U⁰ = copy(Uⁿ)
        G = zero(Uⁿ)
        for j in 1:i-1
            U⁰ .+= α[i, j] .* ΔY[j]
            G .+= (γ[i, j] / (d[i] * Δt)) .* ΔY[j]
        end
        for j in 1:i
            G .+= (β[i, j] / d[i]) .* S[j]
        end
        Y = U⁰ .+ (d[i] * Δt) .* G
        push!(ΔY, Y .- Uⁿ)
    end
    return Y
end

# The same step with the state, the tendency G and `m` more registers, each a linear combination
# of Uⁿ and the tendencies so far. Rows are coefficients on (Uⁿ, Δt S₁, …, Δt S_s). Returns the
# state and the largest residual of the recombinations, which vanishes if `m` registers suffice.
function low_storage_step(t, f, Uⁿ::Vector{T}, Δt, m) where T
    A = effective_runge_kutta(t)
    s = size(A, 1)
    target(k) = vcat(1.0, A[k, :])
    known(row, i) = vcat(row[1:i+1], zeros(s - i))

    state_row = vcat(1.0, zeros(s))
    register_rows = [zeros(s + 1) for _ in 1:m]
    state = copy(Uⁿ)
    registers = [zero(Uⁿ) for _ in 1:m]
    defect = 0.0

    for i in 1:s
        G = Δt .* f(state)
        G_row = zeros(s + 1); G_row[i + 1] = 1
        basis = hcat(state_row, register_rows..., G_row) # columns: what is in memory

        new_state_row = target(i)
        # What later stages need of the symbols known now, beyond the new state.
        later = [known(target(k), i) for k in i+1:s]
        new_register_rows = [zeros(s + 1) for _ in 1:m]
        if !isempty(later)
            M = rows(later)
            M = M .- (M * new_state_row) * new_state_row' ./ dot(new_state_row, new_state_row)
            F = svd(M)
            for q in 1:min(m, length(F.S))
                new_register_rows[q] = F.Vt[q, :]
            end
            defect = max(defect, length(F.S) > m ? F.S[m + 1] / max(F.S[1], eps()) : 0.0)
        end

        coefficients(row) = pinv(basis) * row
        c_state = coefficients(new_state_row)
        defect = max(defect, norm(basis * c_state - new_state_row))
        c_registers = map(coefficients, new_register_rows)
        foreach(q -> (defect = max(defect, norm(basis * c_registers[q] - new_register_rows[q]))), 1:m)

        memory = (state, registers..., G)
        combine(c) = sum(T(c[q]) .* memory[q] for q in eachindex(c))
        new_state = combine(c_state)
        registers = map(combine, c_registers)
        state = new_state
        state_row, register_rows = new_state_row, new_register_rows
    end

    return state, defect
end

println("\nOne step of y′ = L y (40 unknowns, |Δt L| ≈ 1): the low-storage form against the implemented one")
println(rpad("set", 18), "registers besides G | recombination defect | Float64 difference | Float32 difference (units of eps)")
let
    n = 40
    L = [((i * 7 + j * 13) % 11 - 5) / 20 for i in 1:n, j in 1:n] - 1.5I
    y₀ = [1 + ((3i) % 7) / 7 for i in 1:n]
    for name in names
        t = tableau(name)
        for m in 0:2
            results = map((Float64, Float32)) do T
                f(y) = T.(L) * y
                reference = implemented_step(t, f, T.(y₀), T(0.3))
                y, defect = low_storage_step(t, f, T.(y₀), T(0.3), m)
                maximum(abs, y .- reference) / (eps(T) * maximum(abs, reference)), defect
            end
            println(rpad(string(name), 18), lpad(m, 19), " | ", lpad(round(results[1][2], sigdigits = 2), 20), " | ",
                    lpad(round(results[1][1], sigdigits = 3), 18), " | ", lpad(round(results[2][1], sigdigits = 3), 18))
        end
    end
end
