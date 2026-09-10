#####
##### Several grids at once
#####

"""
`ColumnEnsembleProblem`s for the same members on different vertical grids, calibrated together: the
forward map is the concatenation of the grids' forward maps and the observations the concatenation of
their targets — identical, since all are means over the same observation cells — with the noise of
grid `r` divided by `weights[r]`, equal by default.
"""
struct MultiResolutionProblem
    problems :: Vector{ColumnEnsembleProblem}
    weights :: Vector{Float64}
end

MultiResolutionProblem(problems; weights = ones(length(problems))) =
    MultiResolutionProblem(collect(problems), collect(Float64, weights))

const AnyProblem = Union{ColumnEnsembleProblem, MultiResolutionProblem}

problems(p::ColumnEnsembleProblem) = [p]
problems(m::MultiResolutionProblem) = m.problems
members(p::AnyProblem) = first(problems(p)).members

function forward_map(m::MultiResolutionProblem, params::AbstractMatrix; kw...)
    results = [forward_map(p, params; kw...) for p in m.problems]
    return vcat(first.(results)...), last.(results)
end

function observations(m::MultiResolutionProblem; kw...)
    ys = Vector{Float64}[]
    σ²s = Vector{Float64}[]
    for (p, w) in zip(m.problems, m.weights)
        y, Γ = observations(p; kw...)
        push!(ys, y)
        push!(σ²s, diag(Γ) ./ w^2)
    end
    return vcat(ys...), Diagonal(vcat(σ²s...))
end
