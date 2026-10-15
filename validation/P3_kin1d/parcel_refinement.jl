using DelimitedFiles
using Printf
root = normpath(joinpath(@__DIR__, "../../validation_output"))
function compare_refinement(io, first, second, label)
    @printf(io, "%s: max ΔT %.7g K\n", label, maximum(abs.(first[:,4]-second[:,4])))
    for (column, name) in ((6,"cloud"),(7,"cloud number"),(8,"rain"),(10,"ice"),(11,"ice number"))
        a,b=first[:,column],second[:,column]
        @printf(io, "  %-12s max/peak %.6f%%; L1 %.6f%%\n", name,
                100maximum(abs.(a-b))/max(maximum(abs.(b)),1e-30),100sum(abs.(a-b))/max(sum(abs.(b)),1e-30))
    end
end
open(joinpath(root,"p3_parcel_direct/refinement.txt"),"w") do io
    for (coarse,fine) in ((0.00625,0.003125),(0.003125,0.0015625))
        a=readdlm(joinpath(root,"p3_processes/trajectories_dt$(coarse)_liquid_ice/reference_final.dat"))
        b=readdlm(joinpath(root,"p3_processes/trajectories_dt$(fine)_liquid_ice/reference_final.dat"))
        compare_refinement(io,a[a[:,1].==14,:],b[b[:,1].==14,:],"Upstream, 2 m/s, $coarse → $fine s")
    end
    for speed in (1,2), (coarse,fine) in ((0.005,0.0025),(0.0025,0.00125))
        a=readdlm(joinpath(root,"p3_parcel_direct/verified_reference_constants_dt$(coarse)_tau0.025/parcel_$(speed).dat"))
        b=readdlm(joinpath(root,"p3_parcel_direct/verified_reference_constants_dt$(fine)_tau0.025/parcel_$(speed).dat"))
        compare_refinement(io,a,b,"ParcelModel, $speed m/s, fixed τ=0.025 s, $coarse → $fine s")
    end
end
print(read(joinpath(root,"p3_parcel_direct/refinement.txt"),String))
