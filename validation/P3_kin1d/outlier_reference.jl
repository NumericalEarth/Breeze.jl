# Controlled upstream-only interventions. These are causal diagnostics, NOT
# replacement reference data and NOT proposed changes to upstream or Breeze.
using DelimitedFiles

variant = get(ARGS, 1, "baseline")
@assert variant in ("baseline", "nucleation_relaxation", "retain_trace_ice", "retain_trace_rime", "retain_trace_ice_and_rime", "consistent_saturation")
root = normpath(joinpath(@__DIR__, "../../validation_output/p3_matrix"))
template = joinpath(root, "reference_dt0.00078125")
directory = joinpath(root, "outliers", variant)
mkpath(directory)
source = read(joinpath(template, "microphy_p3.f90"), String)
driver = read(joinpath(template, "trajectory_probe.f90"), String)

function replace_checked(source, old, new; count=1)
    @assert length(findall(old,source)) == count old
    return replace(source,old=>new)
end

if variant == "nucleation_relaxation"
    # Replace only the Cooper target's per-step adjustment by a fixed 0.025 s
    # relaxation. All other process limits and cleanup remain native upstream.
    source = replace_checked(source,"(dum-sum(nitot(i,k,:)))*i_dt",
                                    "(dum-sum(nitot(i,k,:)))/0.025")
    source = replace_checked(source,"(dum-sum(nitot(i,k,:)))*mi0*i_dt",
                                    "(dum-sum(nitot(i,k,:)))*mi0/0.025")
elseif variant == "consistent_saturation"
    # Isolate the rounded molecular-weight ratio in qv_sat. The native value
    # 0.622 differs from the ratio of upstream's own gas constants Rd/Rv.
    source = replace_checked(source,"ep_2   = 0.622", "ep_2   = rd/rv")
elseif variant in ("retain_trace_ice", "retain_trace_ice_and_rime")
    # Allow newly formed ice to accumulate below qsmall. Keep qsmall for all
    # process/lookup activation gates. Change only the three trace-removal sites
    # reached by this one-category, sedimentation-free setup.
    source = replace_checked(source,"qitot(i,k,iice).lt.qsmall .or. (qitot(i,k,iice).lt.qsmall_dry",
                                    "qitot(i,k,iice).lt.0. .or. (qitot(i,k,iice).lt.qsmall_dry")
    source = replace_checked(source,"if (qitot(i,k,iice).lt.qsmall) then",
                                    "if (qitot(i,k,iice).lt.0.) then")
    source = replace_checked(source,"else  !from 'if qi_not_small'",
                                    "elseif (qitot(i,k,iice).le.0.) then  !retain positive trace ice")
end
if variant in ("retain_trace_rime", "retain_trace_ice_and_rime")
    # Preserve positive rime volume/mass in calc_bulkRhoRime. Keep the density
    # bounds, ice trace removal, and all process activation gates unchanged.
    source = replace_checked(source,"if (bi_rim.ge.1.e-15) then", "if (bi_rim.gt.0.) then")
    source = replace_checked(source,"if (qi_rim.lt.qsmall) then", "if (qi_rim.lt.0.) then")
end

# Read-only instrumentation: host time/case and the state used for process rates.
source = replace_checked(source," public :: p3_main, polysvp1, p3_init",
    " public :: p3_main, polysvp1, p3_init\n real, public :: probe_time, probe_case\n logical, public :: probe_record")
source = replace_checked(source,"if (.false.) write(80,'(*(es24.15,1x))') &",
    "if (i.eq.1.and.k.eq.3.and.probe_record) write(80,'(*(es24.15,1x))') &\n" *
    "probe_case,probe_time,t(i,k),pres(i,k),qv(i,k),qc(i,k),qr(i,k),qitot(i,k,1),nitot(i,k,1),supi_cld, &")
driver = replace_checked(driver,"call p3_main", """
   probe_time=step*dt; probe_case=case_identifier
   probe_record=mod(step,nint(0.1/dt)).eq.0.and. &
     ((case_identifier.eq.3.and.probe_time.ge.1250..and.probe_time.le.1600.).or. &
      (case_identifier.eq.9.and.probe_time.ge.4700..and.probe_time.le.4800.).or. &
      (case_identifier.eq.18.and.probe_time.ge.3560..and.probe_time.le.3600.))
   call p3_main""")

write(joinpath(directory,"microphy_p3.f90"),source)
write(joinpath(directory,"trajectory_probe.f90"),driver)
cp(joinpath(template,"cases.dat"),joinpath(directory,"cases.dat");force=true)
ispath(joinpath(directory,"lookup_tables")) ||
    symlink(realpath(joinpath(template,"lookup_tables")),joinpath(directory,"lookup_tables"))
write(joinpath(directory,"intervention.patch"),read(ignorestatus(`diff -u $(joinpath(template,"microphy_p3.f90")) $(joinpath(directory,"microphy_p3.f90"))`),String))
write(joinpath(directory,"rate_columns.txt"),"case time T p rv rc rr ritotal ni ice_relative_supersaturation\n" *
      read(joinpath(root,"../p3_processes/process_names.txt"),String))
cd(directory) do
    run(`gfortran -O2 -cpp -ffree-line-length-none -fdefault-real-8 -fdefault-double-8 -fcheck=bounds -fbacktrace microphy_p3.f90 trajectory_probe.f90 -o trajectory_probe`)
    run(`./trajectory_probe`)
end
