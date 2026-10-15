# Run the same direct P3 routine repeatedly, without a dynamical host model.
using DelimitedFiles
include("reference_conventions.jl")
root = normpath(joinpath(@__DIR__, "../../validation_output/p3_processes"))
dt = parse(Float64, get(ARGS, 1, "0.1"))
liquid_ice = get(ARGS, 2, "reference") == "liquid_ice"
name = dt == 1 ? "trajectories" : "trajectories_dt$(dt)"
directory = joinpath(root, liquid_ice ? name * "_liquid_ice" : name)
mkpath(directory)
source = read(joinpath(root, "microphy_p3.f90"), String)
# Single-step probes save rates; trajectories need only states.
source = replace(source, "if (i.eq.1.and.k.eq.3) write(80," => "if (.false.) write(80,")
write(joinpath(directory, "microphy_p3.f90"), source)
ispath(joinpath(directory, "lookup_tables")) || symlink(realpath(joinpath(root, "lookup_tables")), joinpath(directory, "lookup_tables"))
cases = readdlm(joinpath(root, "cases.dat"))
cases = vcat(cases, [300 101325 1 0 0 0 0 0 0 0 400 0; 300 101325 1 0 0 0 0 0 0 0 400 0])
writedlm(joinpath(directory, "cases.dat"), cases)
driver = read(joinpath(root, "process_probe.f90"), String)
driver = replace(driver,
    "dt=1." => "dt=$(dt)",
    "integer :: stat, test, k" => "integer :: stat, test, k, step, steps",
    "real :: v(12), dt, T0, p0, qv0" => "real :: v(12), dt, T0, p0, qv0, pnew, Tnew",
    "do test=1,12" => "do test=1,14",
    "qc=v(4)" => "if (test.ge.13) qv0=0.015/(1.-0.015)\n   qc=v(4)",
    "   call p3_main" => """
   steps=nint(600./dt)
   if (test.ge.13) steps=nint(5400./dt)
   do step=1,steps
     if (test.ge.13) then
       pnew=101325.*(1.-9.816*(test-12)*step*dt/(1005.*300.))**(1005./287.15)
       Tnew=th(1,3)*(pnew/1.e5)**(287.15/1005.)
       p=pnew
       ssat=qv-qv_sat(Tnew,pnew,0)
     endif
     th0=th; qvold=qv
   call p3_main""",
    "write(82,'(*(es24.15,1x))') th(1,3)*(p0/1.e5)**(287.15/1005.)," =>
        "if (mod(step,nint(10./dt)).eq.0) write(82,'(*(es24.15,1x))') real(test),step*dt,p(1,3),th(1,3)*(p(1,3)/1.e5)**(287.15/1005.),",
    " enddo\nend program" => "   enddo\n enddo\nend program")
if liquid_ice
    # Only the host driver changes. P3_MAIN retains its native internal heating.
    # Dry mixing ratios become moist mass fractions in the host θ_li equation.
    constants = reference_constants(1e5)
    Lv = constants.liquid.reference_latent_heat
    Ls = constants.ice.reference_latent_heat
    driver = replace(driver,
        "pnew, Tnew" => "pnew, Tnew, theta_li, cm, rm, latent, total, rv, rl, ri",
        "steps=nint(600./dt)" => """
call mixture(qv(1,3),qc(1,3)+qr(1,3)+qli(1,3,1),qi(1,3,1)-qli(1,3,1),cm,rm,latent)
theta_li=(T0-latent/cm)/(p0/1.e5)**(rm/cm)
steps=nint(600./dt)""",
        "Tnew=th(1,3)*(pnew/1.e5)**(287.15/1005.)" => """
call mixture(qv(1,3),qc(1,3)+qr(1,3)+qli(1,3,1),qi(1,3,1)-qli(1,3,1),cm,rm,latent)
    Tnew=theta_li*(pnew/1.e5)**(rm/cm)+latent/cm
    th=Tnew*(1.e5/pnew)**(287.15/1005.)""",
        "   if (mod(step" => """
   call mixture(qv(1,3),qc(1,3)+qr(1,3)+qli(1,3,1),qi(1,3,1)-qli(1,3,1),cm,rm,latent)
   Tnew=theta_li*(p(1,3)/1.e5)**(rm/cm)+latent/cm
   th=Tnew*(1.e5/p(1,3))**(287.15/1005.)
   ssat=qv-qv_sat(Tnew,p(1,3),0)
   if (mod(step""",
        "end program" => """
contains
subroutine mixture(rv,rl,ri,cm,rm,latent)
 real,intent(in)::rv,rl,ri
 real,intent(out)::cm,rm,latent
 real::total
 total=1.+rv+rl+ri
 cm=(1005.+1850.*rv+4218.*rl+2108.*ri)/total
 rm=(287.15+461.51*rv)/total
 latent=($(Lv)*rl+$(Ls)*ri)/total
end subroutine
end program""")
end
write(joinpath(directory, "trajectory_probe.f90"), driver)
cd(directory) do
    run(`gfortran -O2 -cpp -ffree-line-length-none -fdefault-real-8 -fdefault-double-8 -fcheck=bounds -fbacktrace microphy_p3.f90 trajectory_probe.f90 -o trajectory_probe`)
    run(`./trajectory_probe`)
end
