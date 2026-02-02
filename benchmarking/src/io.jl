#####
##### I/O
#####

"""
    save_benchmark(filename, results::BenchmarkResult)

Save benchmark results to a JLD2 file.
`results` can be a single `BenchmarkResult` or a vector of results.
"""
function save_benchmark(filename, results::BenchmarkResult)
    jldopen(filename, "w") do file
        file["results"] = results
    end
    return nothing
end

"""
    load_benchmark(filename)

Load benchmark results from a JLD2 file.
Returns either a single `BenchmarkResult` or a vector of results.
"""
function load_benchmark(filename)
    return jldopen(filename, "r") do file
        file["results"]
    end
end
