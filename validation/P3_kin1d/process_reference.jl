# Instrument the pinned upstream P3 for identical-state process-rate comparisons.
using Pkg.Artifacts
using DelimitedFiles
using Downloads
using SHA

directory = normpath(joinpath(@__DIR__, "../../validation_output/p3_processes"))
mkpath(directory)
upstream = normpath(joinpath(@__DIR__, "../../validation_output/p3_kin1d/upstream/microphy_p3.f90"))
mkpath(dirname(upstream))
commit = "f90df9613fc90360f846d2f12114b967ef3e53ce"
if !isfile(upstream)
    Downloads.download("https://raw.githubusercontent.com/P3-microphysics/P3-microphysics/$commit/src/microphy_p3.f90", upstream)
end
@assert bytes2hex(sha256(read(upstream))) == "6e354a9da133db13cd628938d1bb6f836dc00360c80c2ce48d8e7e3785eb20e2"
source = read(upstream, String)
function replace_once(source, old, new)
    length(findall(old, source)) == 1 || error("Expected exactly one occurrence of $old")
    return replace(source, old => new)
end
source = replace(source, " log_predictSsat = .false." => " log_predictSsat = .true.",
                         " log_typeDiags  = .true." => " log_typeDiags  = .false.",
                         " public :: mp_p3_wrapper_wrfcm1" => " public :: mp_p3_wrapper_wrfcm1, qv_sat")
const process_mapping = (
    :condensation => "qccon-qcevp",
    :ccn_activation_mass => "qcnuc",
    :ccn_activation_number => "ncnuc",
    :autoconversion => "qcaut",
    :accretion => "qcacc",
    :rain_evaporation => "qrevp",
    :rain_condensation => "qrcon",
    :rain_self_collection => "nrslf",
    :deposition => "qidep(1)-qisub(1)",
    :sublimation_number => "nisub(1)+nlevp(1)",
    :aggregation => "nislf(1)",
    :cloud_riming => "qccol(1)",
    :rain_riming => "qrcol(1)",
    :rain_riming_number => "nrcol(1)",
    :partial_melting => "qimlt(1)",
    :complete_melting => "qrmlt(1)",
    :melting_number => "nimlt(1)",
    :coating_condensation => "qlcon(1)",
    :coating_evaporation => "qlevp(1)",
    :cloud_riming_number => "nccol(1)",
    :shedding => "qlshd(1)",
    :shedding_number => "nlshd(1)",
    :refreezing => "qifrz(1)",
    :nucleation_mass => "qinuc(1)",
    :nucleation_number => "ninuc(1)",
    :cloud_freezing_mass => "qcheti(1)",
    :cloud_freezing_number => "ncheti(1)",
    :rain_freezing_mass => "qrheti(1)",
    :rain_freezing_number => "nrheti(1)",
    :splintering_mass => "qcmul(1)+qrmul(1)",
    :splintering_number => "nimul(1)",
    :cloud_warm_collection => "qccoll(1)",
    :rain_warm_collection => "qrcoll(1)",
    :wet_growth_cloud => "qwgrth1c(1)",
    :wet_growth_rain => "qwgrth1r(1)")
write(joinpath(directory, "process_names.txt"), join(first.(process_mapping), '\n') * "\n")
diagnostics = "\n       if (i.eq.1.and.k.eq.3) write(80,'(*(es24.15,1x))') &\n           " *
              join(last.(process_mapping), ", &\n           ") * "\n"
source = replace_once(source, "! update prognostic microphysics and thermodynamics variables",
                          diagnostics * "! update prognostic microphysics and thermodynamics variables")
source = replace_once(source, "! Sedimentation:\n\n do i = its,ite",
                         "! Sedimentation:\n if (.false.) then\n do i = its,ite")
source = replace_once(source, "! End of sedimentation section", " endif\n! End of sedimentation section")
write(joinpath(directory, "microphy_p3.f90"), source)
patch = read(ignorestatus(`diff -u $upstream $(joinpath(directory, "microphy_p3.f90"))`), String)
write(joinpath(directory, "reference.patch"), patch)
write(joinpath(directory, "provenance.txt"), "Upstream commit: $commit\n" *
      "Source SHA256: $(bytes2hex(sha256(read(upstream))))\n" * read(`gfortran --version`, String))

# T, p, liquid RH, cloud mass/number, rain mass/number, frozen ice mass/number,
# rime mass, rime density, coating. All q and n use upstream's dry-air convention.
cases = [290 85000 1.01 1e-3 1e8 0 0 0 0 0 400 0;
         290 85000 0.80 1e-3 1e8 0 0 0 0 0 400 0;
         280 85000 0.80 0 0 1e-3 1e5 0 0 0 400 0;
         280 85000 1.01 1e-3 1e8 1e-3 1e5 0 0 0 400 0;
         258 70000 1.02 0 0 0 0 1e-3 1e4 0 400 0;
         268 80000 0.80 0 0 0 0 1e-3 1e4 2e-4 400 0;
         268 80000 1.00 1e-3 1e8 1e-3 1e5 1e-3 1e4 2e-4 400 0;
         275 85000 1.00 1e-3 1e8 1e-3 1e5 1e-3 1e4 2e-4 400 2e-4;
         270 80000 1.01 1e-3 1e8 1e-3 1e5 1e-3 1e4 5e-4 500 1e-4;
         232 50000 1.00 1e-4 1e7 1e-4 1e4 1e-4 1e3 0 400 0;
         253 65000 1.01 0 0 0 0 0 0 0 400 0;
         288 85000 1.01 0 0 0 0 0 0 0 400 0]
writedlm(joinpath(directory, "cases.dat"), cases)

driver = raw"""
program process_probe
 use microphy_p3
 implicit none
 integer, parameter :: nk=5, ncat=1
 integer :: stat, test, k
 real :: v(12), dt, T0, p0, qv0
 real :: qc(1,nk),nc(1,nk),qr(1,nk),nr(1,nk),th0(1,nk),th(1,nk),qvold(1,nk),qv(1,nk)
 real :: qi(1,nk,1),qf(1,nk,1),qli(1,nk,1),ni(1,nk,1),bf(1,nk,1),zi(1,nk,1)
 real :: ssat(1,nk),w(1,nk),p(1,nk),dz(1,nk),pl(1),ps(1)
 real :: ze(1,nk),ec(1,nk),ei(1,nk,1),vm(1,nk,1),di(1,nk,1),rhoi(1,nk,1)
 real :: d2(1,20),d3(1,nk,20),scf(1,nk)
 call p3_init('lookup_tables',1,.false.,.true.,'KIN1D',stat,.true.,.true.)
 open(10,file='cases.dat')
 open(80,file='reference_rates.dat')
 open(81,file='initial.dat')
 open(82,file='reference_final.dat')
 dt=1.
 do test=1,12
   read(10,*) v
   T0=v(1); p0=v(2); qv0=v(3)*qv_sat(T0,p0,0)
   qc=v(4); nc=v(5); qr=v(6); nr=v(7)
   qi=v(8)+v(12); ni=v(9); qf=v(10); bf=v(10)/v(11); qli=v(12); zi=0.
   p=p0; th=T0*(1.e5/p0)**(287.15/1005.); th0=th
   qv=qv0; qvold=qv; ssat=qv0-qv_sat(T0,p0,0); w=0.; dz=200.
   d2=0.; d3=0.; scf=1.
   write(81,'(*(es24.15,1x))') T0,p0,p0/(287.15*T0),qv0,qv_sat(T0,p0,0),qv_sat(T0,p0,1), &
      v(4:12)
   call p3_main(qc,nc,qr,nr,th0,th,qvold,qv,dt,qi,qf,qli,ni,bf,zi,ssat,w,p,dz,2,pl,ps, &
       1,1,1,nk,1,ze,ec,ei,vm,di,rhoi,20,d2,20,d3,.true.,'KIN1D',1.,1.,.false., &
       .false.,1.,1.,scf,.false.,.true.)
   write(82,'(*(es24.15,1x))') th(1,3)*(p0/1.e5)**(287.15/1005.),qv(1,3),qc(1,3),nc(1,3), &
       qr(1,3),nr(1,3),qi(1,3,1)-qli(1,3,1),ni(1,3,1),qf(1,3,1),bf(1,3,1),qli(1,3,1),ssat(1,3)
 enddo
end program
"""
write(joinpath(directory, "process_probe.f90"), driver)
artifacts = normpath(joinpath(@__DIR__, "../../Artifacts.toml"))
ensure_artifact_installed("P3_lookup_tables", artifacts)
tables = artifact_path(artifact_hash("P3_lookup_tables", artifacts))
ispath(joinpath(directory, "lookup_tables")) || symlink(tables, joinpath(directory, "lookup_tables"))
cd(directory) do
    run(`gfortran -O2 -cpp -ffree-line-length-none -fdefault-real-8 -fdefault-double-8 -fcheck=bounds -fbacktrace microphy_p3.f90 process_probe.f90 -o process_probe`)
    run(`./process_probe`)
end
