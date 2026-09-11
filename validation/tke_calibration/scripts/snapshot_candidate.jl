# Freeze a running calibration, so that every check made on a candidate is made on the same file.
#
# A production checkpoint is rewritten every iteration. Two validation evaluations queued against the
# live file — Δt = 7.5 s and Δt = 3.75 s, say — can start minutes apart, read different `selected_mean`
# entries, and produce a comparison of two parameter sets while appearing to compare two numerical
# settings. Nothing in either output would reveal it.
#
# The snapshot is a **byte copy of the whole checkpoint**, not a re-serialization. Three things that
# rules out, each of which a rewrite gets wrong:
#
#   - Hashing the source and then loading it opens the file twice, so an atomic replacement in between
#     records the hash of a different generation than the one that was read. Here a single descriptor
#     is opened once and copied; a replacement afterwards leaves that descriptor on the old inode.
#   - Keeping one history entry while `selected_mean.iteration` still names its original index breaks
#     any reader that indexes `history[iteration]`. The copy keeps the whole history, so every index
#     means what it meant.
#   - A load-and-resave in a process without Breeze persists JLD2's reconstructed wrappers in place of
#     the native types, and a later reader *with* Breeze does not get them back. Copying bytes leaves
#     the serialized schema exactly as written.
#
# Metadata goes in a sidecar rather than inside the copy, for the same reason. Snapshots are never
# overwritten: a frozen candidate that can change is not frozen.
#
#     julia --project scripts/snapshot_candidate.jl source=results/final/ri/n400_seed1.jld2 \
#           output=results/final/ri/candidate_ri.jld2
using JLD2, SHA, Printf, Dates

options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
source = String(get(() -> error("Pass source=<checkpoint>"), options, "source"))
output = String(get(options, "output", replace(source, r"\.jld2$" => "") * "_candidate.jld2"))
sidecar = replace(output, r"\.jld2$" => "") * "_snapshot.toml"

ispath(output) && error("$output exists. Snapshots are immutable; write a new path rather than " *
                        "overwriting one that other results may already refer to.")
isfile(source) || error("$source does not exist")
isempty(dirname(output)) || mkpath(dirname(output))

# One descriptor, opened once. If the live checkpoint is atomically replaced while this runs, the
# open handle still refers to the generation we started reading, so the copy is internally consistent.
temporary = output * ".partial"
bytes = open(source, "r") do io
    open(temporary, "w") do out
        write(out, io)
    end
end
digest = open(sha256, temporary)

# Validate the copy by reading it back before it is given its final name, so a truncated or torn copy
# never appears under a path something else may pick up.
selected, iterations = try
    saved = load(temporary)
    haskey(saved, "selected_mean") && !isnothing(saved["selected_mean"]) ||
        error("no directly evaluated mean: the run has not completed an iteration, or predates the diagnostic")
    saved["selected_mean"], length(saved["history"])
catch err
    rm(temporary; force = true)
    rethrow(err)
end

mv(temporary, output)   # atomic within a filesystem: the snapshot appears complete or not at all

open(sidecar, "w") do io
    println(io, "# Provenance of an immutable candidate snapshot. The hash is of the snapshot itself,")
    println(io, "# so a reader can verify the bytes it is scoring.")
    println(io, "source = \"", source, "\"")
    println(io, "snapshot = \"", basename(output), "\"")
    println(io, "sha256 = \"", bytes2hex(digest), "\"")
    println(io, "bytes = ", bytes)
    println(io, "selected_iteration = ", selected.iteration)
    println(io, "iterations_at_snapshot = ", iterations)
    println(io, "objective = ", selected.objective)
    println(io, "taken = \"", Dates.now(), "\"")
end

@printf "froze %s at iteration %d of %d\n" source selected.iteration iterations
@printf "  objective %.6f, %.1f MB, sha256 %s\n" selected.objective bytes/2^20 bytes2hex(digest)[1:16]
@printf "  wrote %s and %s\n" output sidecar
println("\nPoint every check at the snapshot, never at the live checkpoint.")
