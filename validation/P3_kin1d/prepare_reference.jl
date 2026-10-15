# Fetch a pinned upstream kin1d/P3 and adapt only its case/IO plus the P3 Ssat switch.
# No CM1 code is used. All reference modifications are retained as a unified diff.
using Downloads
using SHA
using Pkg.Artifacts

const upstream_commit = "f90df9613fc90360f846d2f12114b967ef3e53ce"
output_directory = get(ENV, "P3_COMPARISON_OUTPUT",
                       normpath(joinpath(@__DIR__, "../../validation_output/p3_kin1d")))
dt = parse(Float64, get(ARGS, 1, "1"))
build = joinpath(output_directory, "kin1d_dt$(dt)")
upstream = joinpath(output_directory, "upstream")
mkpath(build)
mkpath(upstream)

for (name, path) in (("microphy_p3.f90", "src/microphy_p3.f90"),
                     ("cld1d.f90", "kin1d/src/cld1d.f90"),
                     ("routinesCld1d.f90", "kin1d/src/routinesCld1d.f90"),
                     ("execld.f90", "kin1d/src/execld.f90"))
    destination = joinpath(upstream, name)
    if !isfile(destination)
        Downloads.download("https://raw.githubusercontent.com/P3-microphysics/P3-microphysics/$upstream_commit/$path", destination)
    end
    cp(destination, joinpath(build, name); force=true)
end

function replace_once(source, old, new)
    length(findall(old, source)) == 1 || error("Expected exactly one occurrence of $old")
    return replace(source, old => new)
end

source = read(joinpath(upstream, "cld1d.f90"), String)
source = replace_once(source, "TDMIN        = .true.", "TDMIN        = .false.")
source = replace_once(source, "EVOLVING     = .true.", "EVOLVING     = .false.")
source = replace_once(source, "nk           =  41", "nk           = 100")
source = replace_once(source, "dt           = 10.", "dt           = $(dt)")
source = replace_once(source, "ttotmin      = 90", "ttotmin      = 60")
source = replace_once(source, "abort_on_err = .false.", "abort_on_err = .true.")
# Use the same Exner exponent as p3_main (rd/cp), avoiding a diagnostic mismatch
# between the driver's rounded 0.286 and the temperature actually used by P3.
source = replace(source, "**0.286" => "**(287.15/1005.)")
source = replace_once(source, "      integer      :: iice", "      integer      :: iice\n      real :: inflow_t, inflow_qv, inflow_ssat, inflow_w, courant, air_mass_factor")

# Retain kin1d's top-to-bottom indexing. Import Breeze's physical cell centers and
# initial T/p/qv without a second sounding interpolation or saturation formula.
first_initial = findfirst("      H= 12000.", source).start
last_initial = findfirst("!  END OF INITIAL SET-UP", source).stop
initialization = """
      open(unit=12,file='initial_state.dat')
      do k=1,nk
         read(12,*) z(k),p(k),tt(k),rho(k),Qv(1,k),ssat(1,k)
         ! Breeze stores fractions per moist-air mass; P3 uses dry-air mixing ratios.
         Qv(1,k) = Qv(1,k)/(1.-Qv(1,k))
         ssat(1,k) = Qv(1,k)-qv_sat(tt(k),p(k),0)
      enddo
      close(12)
      open(unit=12,file='inflow.dat')
      read(12,*) inflow_t,inflow_qv,inflow_ssat,inflow_w
      close(12)
      inflow_qv = inflow_qv/(1.-inflow_qv)
      inflow_ssat = inflow_qv-qv_sat(inflow_t,100000.,0)
      H = z(1)
      H0 = z(nk)
      Hcld = H
      nkcld = nk
      dz2d = (H-H0)/(nk-1)
      w = inflow_w
      w1 = inflow_w
      DIV = 0.
      COMP = 0.
      Qv_m = Qv
      Qvinit = Qv
      tt0(1,:) = tt
      tt1 = tt0
      p2d(1,:) = p
      Qc = 0.; Nc = 0.; Qr = 0.; Nr = 0.
      Qc_m = 0.; Nc_m = 0.; Qr_m = 0.; Nr_m = 0.
      Qitot = 0.; Qirim = 0.; Qiliq = 0.; Nitot = 0.; Birim = 0.; Zitot = 0.
      Qitot_m = 0.; Qirim_m = 0.; Qiliq_m = 0.
      Nitot_m = 0.; Birim_m = 0.; Zitot_m = 0.
      p0 = 100000.; p0m = p0
      diag_2d = 0.; diag_3d = 0.; scpf_cldfrac = 1.
      prt_liq = 0.; prt_sol = 0.
      open(unit=31,file='profiles.dat')
      write(31,'(A)') '# time z T p rho qv qc qr qi qli nc nr ni ssat diagnosed_ssat'
      do k=1,nk
         air_mass_factor = 1./(1.+Qv(1,k)+Qc(1,k)+Qr(1,k)+sum(Qitot(1,k,:)))
         write(31,'(15es24.15)') 0.,z(k),tt1(1,k),p(k),rho(k),Qv(1,k), &
              Qc(1,k),Qr(1,k),Qitot(1,k,1)-Qiliq(1,k,1),Qiliq(1,k,1),Nc(1,k),Nr(1,k), &
              Nitot(1,k,1),ssat(1,k),Qv(1,k)-qv_sat(tt1(1,k),p(k),0)
      enddo
!  END OF INITIAL SET-UP"""
source = source[1:first_initial-1] * initialization * source[last_initial+1:end]
source = replace_once(source, "!       call advec(ssat,w,H,H0,Hcld,nk,nkcld,dt)",
                               "        call advec(ssat,w,H,H0,Hcld,nk,nkcld,dt)")

# The native driver leaves the bottom cell unadvected. Supply the same incoming
# air as the Breeze example, with zero incoming condensate and particle number.
boundary = """
        courant = inflow_w*dt/dz2d(1,nk)
        tt1(1,nk) = tt1(1,nk) + courant*(inflow_t-tt1(1,nk))
        Qv(1,nk) = Qv(1,nk) + courant*(inflow_qv-Qv(1,nk))
        ssat(1,nk) = ssat(1,nk) + courant*(inflow_ssat-ssat(1,nk))
        Qc(1,nk) = (1.-courant)*Qc(1,nk)
        Qr(1,nk) = (1.-courant)*Qr(1,nk)
        Nc(1,nk) = (1.-courant)*Nc(1,nk)
        Nr(1,nk) = (1.-courant)*Nr(1,nk)
        Qitot(1,nk,:) = (1.-courant)*Qitot(1,nk,:)
        Qirim(1,nk,:) = (1.-courant)*Qirim(1,nk,:)
        Qiliq(1,nk,:) = (1.-courant)*Qiliq(1,nk,:)
        Nitot(1,nk,:) = (1.-courant)*Nitot(1,nk,:)
        Birim(1,nk,:) = (1.-courant)*Birim(1,nk,:)
        Zitot(1,nk,:) = (1.-courant)*Zitot(1,nk,:)

"""
source = replace_once(source, "        do k=1,nk\n          Qv(1,k)", boundary * "        do k=1,nk\n          Qv(1,k)")
output = """
         if (mod(step,nint(30./dt)).eq.0) then
            do k=1,nk
               air_mass_factor = 1./(1.+Qv(1,k)+Qc(1,k)+Qr(1,k)+sum(Qitot(1,k,:)))
               write(31,'(15es24.15)') tsec,z(k),tt1(1,k),p(k),rho(k),Qv(1,k), &
                    Qc(1,k),Qr(1,k),Qitot(1,k,1)-Qiliq(1,k,1),Qiliq(1,k,1),Nc(1,k),Nr(1,k), &
                    Nitot(1,k,1),ssat(1,k),Qv(1,k)-qv_sat(tt1(1,k),p(k),0)
            enddo
         endif
"""
source = replace_once(source, "!  Output to files:", "!  Output to files:\n" * output)
# Convert all mass and number outputs to the Breeze moist-air convention.
# Frozen ice is Qitot-Qiliq; Qitot alone includes the liquid coating in reference P3.
source = replace(source,
    "p(k),rho(k),Qv(1,k), &" => "p(k),rho(k),air_mass_factor*Qv(1,k), &",
    "Qc(1,k),Qr(1,k),Qitot(1,k,1)-Qiliq(1,k,1),Qiliq(1,k,1),Nc(1,k),Nr(1,k), &" =>
    "air_mass_factor*Qc(1,k),air_mass_factor*Qr(1,k),air_mass_factor*(Qitot(1,k,1)-Qiliq(1,k,1)), &\n" *
    "              air_mass_factor*Qiliq(1,k,1),air_mass_factor*Nc(1,k),air_mass_factor*Nr(1,k), &",
    "Nitot(1,k,1),ssat(1,k),Qv(1,k)-qv_sat(tt1(1,k),p(k),0)" =>
    "air_mass_factor*Nitot(1,k,1),air_mass_factor*ssat(1,k), &\n" *
    "              air_mass_factor*(Qv(1,k)-qv_sat(tt1(1,k),p(k),0))")
write(joinpath(build, "cld1d.f90"), source)

microphysics = read(joinpath(upstream, "microphy_p3.f90"), String)
microphysics = replace_once(microphysics, " log_predictSsat = .false.", " log_predictSsat = .true.")
microphysics = replace_once(microphysics, " public :: mp_p3_wrapper_wrfcm1", " public :: mp_p3_wrapper_wrfcm1, qv_sat")
write(joinpath(build, "microphy_p3.f90"), microphysics)
write(joinpath(build, "p3_config.txt"), "# P3 comparison\n# nCat, triple-moment ice, liquid fraction\n1\nF\nT\n")
for filename in ("initial_state.dat", "inflow.dat")
    cp(joinpath(output_directory, filename), joinpath(build, filename); force=true)
end

artifact_file = normpath(joinpath(@__DIR__, "../../Artifacts.toml"))
tables = artifact_path(artifact_hash("P3_lookup_tables", artifact_file))
isdir(tables) || error("Run the Breeze case first to install its lookup tables")
ispath(joinpath(build, "lookup_tables")) || symlink(tables, joinpath(build, "lookup_tables"))
open(joinpath(build, "provenance.txt"), "w") do io
    println(io, "Upstream commit: ", upstream_commit)
    println(io, "P3 source version: 5.5.2; table 1: 6.9-2momI; table 3: 1.4")
    println(io, "Breeze table artifact: ", basename(tables))
    for name in ("cld1d.f90", "microphy_p3.f90", "routinesCld1d.f90", "execld.f90")
        println(io, name, " original SHA256 ", bytes2hex(open(sha256, joinpath(upstream, name))))
    end
end
open(joinpath(build, "reference.patch"), "w") do io
    for name in ("cld1d.f90", "microphy_p3.f90")
        run(pipeline(ignorestatus(`diff -u $(joinpath(upstream, name)) $(joinpath(build, name))`); stdout=io))
    end
end
cd(build) do
    run(`gfortran -O2 -cpp -ffree-line-length-none -fdefault-real-8 -fdefault-double-8 -fcheck=bounds -fbacktrace -c microphy_p3.f90 routinesCld1d.f90 cld1d.f90 execld.f90`)
    run(`gfortran -o execld microphy_p3.o routinesCld1d.o cld1d.o execld.o`)
    run(`./execld`)
end
@info "kin1d comparison complete" dt build
