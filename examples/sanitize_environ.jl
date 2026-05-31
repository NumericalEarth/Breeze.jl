## Workaround for a Cray MPICH bug on Perlmutter: on multi-node jobs launched with
## srun, a malformed environment entry (no `=`) gets inserted after MPI_Init.
## CUDA.jl is sensitive to it and multi-node GPU runs hang/break. Call
## SanitizeEnviron.sanitize_environ!() immediately after MPI.Init().
## Source: CliMA/Oceananigans.jl discussion #5513 (romanlee), "Using NERSC's Perlmutter HPC".
module SanitizeEnviron

export sanitize_environ!, raw_entries, malformed_indices

function environ_ptr()
    unsafe_load(cglobal(:environ, Ptr{Ptr{UInt8}}))
end

function raw_entries(; limit::Int=4096)
    entries = String[]
    envp = environ_ptr()
    for i in 0:(limit - 1)
        ptr = unsafe_load(envp, i + 1)
        ptr == C_NULL && break
        push!(entries, unsafe_string(Base.Cstring(ptr)))
    end
    return entries
end

function malformed_indices(entries::AbstractVector{<:AbstractString})
    bad = Int[]
    for (i, entry) in pairs(entries)
        occursin('=', entry) || push!(bad, i)
    end
    return bad
end

function sanitize_environ!()
    entries = raw_entries()
    valid_entries = String[]
    removed = String[]
    for entry in entries
        if occursin('=', entry)
            push!(valid_entries, entry)
        else
            push!(removed, entry)
        end
    end

    ccall(:clearenv, Cint, ())
    for entry in valid_entries
        key, value = split(entry, '='; limit=2)
        rc = ccall(:setenv, Cint, (Cstring, Cstring, Cint), key, value, 1)
        rc == 0 || error("setenv failed for $key with rc=$rc")
    end
    return removed
end

end
