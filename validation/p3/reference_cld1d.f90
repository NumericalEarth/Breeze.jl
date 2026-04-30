subroutine columnmodel

!----------------------------------------------------------------------------!
!  1-D kinematic cloud model driver for testing microphysics scheme.
!  Reads in temperature and humidity from a sounding, prescribes an updraft
!  profile, and begins a time integration for a simple cloud simulation.
!
!  The following procedures are performed at each time step:
!
!  - update vertical velocity
!  - compute column-integrated mass (and number) for the beginning of step
!  - update model variables due to advection, compressibility, and divergence
!  - force global concervation of mass (and number)
!  - compute total integrated mass before call to microphysics
!  - call microphysics
!  - compute total integrated mass after call to microphysics
!  - add low-level moisture (to prevent depletion)
!  - write output files
!
!--------------------------------------------------------------------------!
!  Author:         Jason Milbrandt
!  Last modified:  2025 May
!--------------------------------------------------------------------------!

      use subs_cld1d
      use microphy_p3

      implicit none

      character(len=20), parameter :: version_p3 = 'v5.4.7+'

      !for use when reading in from config file:
      integer            :: nCat
      logical            :: trplMomIce
      logical            :: liqFrac

      integer, parameter :: max_nCat     = 5         ! this allows arrays to be declared (below) with nCat read in
      logical, parameter :: scpf_on      = .false.   ! switch for cloud fraction parameterization (SCPF)
      real,    parameter :: scpf_pfrac   = 1.        ! precipitation fraction factor (SCPF)
      real,    parameter :: scpf_resfact = 1.        ! model resolution factor (SCPF)
      logical, parameter :: prog_nc_ssat = .true.
      logical, parameter :: debug_on     = .false.    ! switch for run-time check-values in p3_main
      real,    parameter :: clbfact_dep  = 1.0       ! calibration factor for deposition
      real,    parameter :: clbfact_sub  = 1.0       ! calibration factor for sublimation
      logical, parameter :: log_predictNc = .true.

      logical, parameter :: microON      = .true.    ! call microphysics
      logical, parameter :: TDMIN        = .true.    ! prevent low-level moisture depletion
      logical, parameter :: EVOLVING     = .true.    ! switch for evolving updraft
      real,    parameter :: AMPA         = 2.        ! initial central updraft speed [m s-1] (evolving only)
      real,    parameter :: AMPB         = 5.        ! maximum central updraft speed [m s-1]
      real,    parameter :: Htop0        = 5000.     ! initial height of cloud top   [m]
      real,    parameter :: tscale1      = 5400.     ! period for evolving AMPB      [s]
      real,    parameter :: tscale2      = 5400.     ! period for evolving Hcld      [s]
      integer, parameter :: nk           =  41       ! number of vertical levels
!     integer, parameter :: nk           =  62       ! number of vertical levels
!     integer, parameter :: nk           =  86       ! number of vertical levels
      integer, parameter :: outfreq      =  1        ! output every 'OUTFREQ'        [min]
      real,    parameter :: dt           = 10.       ! time step                     [s]
      integer, parameter :: ttotmin      = 90        ! total integration time	       [min]
      logical, parameter :: abort_on_err = .false.
      logical, parameter :: dowr         = .true.
      integer, parameter :: ni           = 1
      integer, parameter :: lnmax        = 100
      integer, parameter :: nt           = nint(ttotmin*60/dt)

      character(len=10),   parameter :: sndcase = 'ALBERTA'
      character(len=16),   parameter :: model   = 'KIN1D'
!     character(len=16),   parameter :: model   = 'WRF'  !for level tests
      character(len=1024), parameter :: LT_path = './lookup_tables'
!     character(len=1024), parameter :: LT_path = '/users/milbrand/mp_p3/lookupTables/tables'  ! override default

      integer      :: iice

!---------------------------------------------------------------------------------!
!#include "consphy.cdk"  (necessary parameters only)
      real, parameter :: TRPL     =.27316e+3          !K; triple point of water
      real, parameter :: EPS1     =.62194800221014    ! ; RGASD/RGASV
      real, parameter :: EPS2     =.3780199778986     !; 1 - EPS1
      real, parameter :: PI       =.314159265359e+1   ! PI constant = ACOS(-1)
      real, parameter :: GRAV     =.980616e+1         ! M s-2; gravitational acceleration
      real            :: TTT,PRS,QQQ
      real*8          :: FOEW,FOQST

!#include "fintern.cdk"
!   DEFINITION DES FONCTIONS THERMODYNAMIQUES DE BASE
!   POUR LES CONSTANTES, UTILISER LE COMMON /CONSPHY/
!     NOTE: TOUTES LES FONCTIONS TRAVAILLENT AVEC LES UNITES S.I.
!     FONCTION DE TENSION DE VAPEUR SATURANTE (TETENS) - EW OU EI SELON TT
      FOEW(TTT) = 610.78D0*DEXP( DMIN1(DSIGN(17.269D0,                     &
       DBLE(TTT)-DBLE(TRPL)),DSIGN                                         &
       (21.875D0,DBLE(TTT)-DBLE(TRPL)))*DABS(DBLE(TTT)-DBLE(TRPL))/        &
       (DBLE(TTT)-35.86D0+DMAX1(0.D0,DSIGN                                 &
       (28.2D0,DBLE(TRPL)-DBLE(TTT)))))

!     FONCTION CALCULANT L'HUMIDITE SPECIFIQUE SATURANTE (QSAT)
      FOQST(TTT,PRS) = DBLE(EPS1)/(DMAX1(1.D0,DBLE(PRS)/FOEW(TTT))-DBLE(EPS2))
!------------------------------------------------------------------------------!

      integer :: i,j,k,k1,k2,nkcld,n,step,lv,nlvs,tmin
      integer :: its,ite,kts,kte

      real PMv0,PMc0,PMr0,PMi0,PMg0,PMs0,PMh0,PMv1,PMc1,PMr1,PMi1,PMg1,PMs1,PMh1,     &
           PMv3,PMc3,PMr3,PMi3,PMg3,PMs3,PMh3,rhoQmax,depth,gam1,gam3,lamda,No,cnt,   &
           dm,alpha,cxh,cmx,Nfact1,Dmx(nk),ref(nk),Zplt,H,H0,Hcld,ZePlt,HcldTop,      &
           wmaxH,c1,c2,esat,alfa,alfa0,walfa,Kdiff,dz,dzsq,INTthr,dum,HcldBase,       &
           LAMr,ZZ,AMPL,AMPL0,Dc,Dr,Di,Ds,Dg,Dh,cmr,cmi,cms,cmg,cmh,thrd,sig,Ltot,    &
           M1,M2,Aii,BASE,RATIOc,RATIOv,RATIOr,RATIOi,RATIOg,RATIOs,RATIOh,tsec,      &
           tminr,af,bf,cm,Cx,Nox,rhox,PR

      real, parameter :: eps    = 0.622
      real, parameter :: g_cld  = 9.81
      real, parameter :: Rd_cld = 287.0
      real, parameter :: T0     = 273.15
      real, parameter :: cp_cld = 1005.

      real, dimension(nk)    :: z,p,tt,td,tt2,w1,DIV,Tdry,zcld,w1cld,rho,alfa2
      real, dimension(ni,nk) :: tt0,tt1,Qv_m,Qv,w,SIGMA,Qsat,womega,th2d0,th2d1,p2d,dz2d
      real, dimension(lnmax) :: Pin,Zin,TTin,TDin

    ! Prognostic hydrometeor variables:
    !   [x]_m are the the values of [x] at the previous time step
      real, dimension(ni,nk)      :: Qc_m,Qc,Nc_m,Nc,Qr_m,Qr,Nr_m,Nr,ssat        !,ssat_m
      real, dimension(ni,nk,max_nCat) :: Qitot_m,Qirim_m,Qiliq_m,Nitot_m,Birim_m,Zitot_m
      real, dimension(ni,nk,max_nCat) :: Qitot  ,Qirim,  Qiliq,  Nitot,  Birim,  Zitot

    ! Source-Sink term arrays:
      integer, parameter               :: n_diag_2d = 20
      integer, parameter               :: n_diag_3d = 20
      real, dimension(ni,n_diag_2d)    :: diag_2d     !user-defined 2D diagnostic arrays (for output)
      real, dimension(ni,nk,n_diag_3d) :: diag_3d     !user-defined 3D diagnostic arrays (for output)
      real, dimension(ni,nk,max_nCat)      :: diag_reffi,diag_vmi,diag_di,diag_rhoi,diag_dhmax, &
                                          diag_vni,diag_vzi
      real, dimension(ni,nk)           :: diag_reffc

    ! Precipitation rates:
      real, dimension(ni) :: prt_liq       ! precip rate, total liquid     m s-1
      real, dimension(ni) :: prt_sol       ! precip rate, total solid      m s-1
      real, dimension(ni) :: prt_drzl      ! precip rate, drizzle          m s-1
      real, dimension(ni) :: prt_rain      ! precip rate, rain             m s-1
      real, dimension(ni) :: prt_crys      ! precip rate, ice cystals      m s-1
      real, dimension(ni) :: prt_snow      ! precip rate, snow             m s-1
      real, dimension(ni) :: prt_grpl      ! precip rate, graupel          m s-1
      real, dimension(ni) :: prt_pell      ! precip rate, ice pellets      m s-1
      real, dimension(ni) :: prt_hail      ! precip rate, hail             m s-1
      real, dimension(ni) :: prt_sndp      ! precip rate, unmelted snow    m s-1
      real, dimension(ni) :: prt_wsnow     ! precip rate, very wet snow    m s-1

    ! Diagnostics, etc.
      real, dimension(ni,nk)   :: diag_ZET,Qvinit,GZ,scpf_cldfrac
      real, dimension(nk)      :: COMP
      real, dimension(ni)      :: p0,p0m,LR,SR,diag_ZEC
      real, dimension(ni,nk,6) :: qi_type
      real                     :: delz,dum1,dum2,dum3,dum4,dum5,dum6,dum7,dum8
      real                     :: time1,time2,time3,time4,time5  ! for timing tests
      integer                  :: nk_read,kk,stat
      integer, dimension(200)  :: kskiplevs

      real, dimension(20)      :: timer,timer_accum
      character(len=20), dimension(20) :: timer_txt
      integer :: ind

      print*
      print*, '** Remember to use compiler debug options for testing new code (modify Makefile appropriately) **'

!---------------------------------------------------------------------------------------!

      ! get P3 configs from config file (in lieu of specifying them as parameters)
      open (unit=10, file='p3_config.txt')
      read(10,*)
      read(10,*)
      read(10,*) nCat
      read(10,*) trplMomIce
      read(10,*) liqFrac
      close(10)

      call P3_INIT(LT_path,nCat,trplMomIce,liqFrac,model,stat,abort_on_err,dowr)

!---------------------------------------------------------------------------------------!

      diag_dhmax = -1.

      open (unit=30, file='out_p3.dat')

      its = 1
      ite = 1
      kts = 1
      kte = nk

      PR    = 0. ! accumulated precipitation
      time3 = 0. ! for total timing test
      timer_accum = 0.
      timer_txt = ''

!------- INITIALIZE w, tt, td, p AND Q[x]  PROFILES: -------

      H= 12000.
!     open(unit=12, file='./soundings/snd_input.Alta_3.data')   !grnd=  0m
!     open(unit=12, file='./Tsfc2.txt')
      open(unit=12, file='./soundings/snd_input.KOUN_00z1june2008.data')

!  Input uninterpolated sounding data:
      read(12,*) nlvs
      read(12,*)
      read(12,*)
      read(12,*)
      read(12,*)
      read(12,*)
      do lv=1,nlvs
         !read(12,*) dum,Pin(lv),Zin(lv),TTin(lv),TDin(lv)
         read(12,*) Pin(lv),Zin(lv),TTin(lv),TDin(lv)   !new
         TTin(lv) = TTin(lv) + T0		        !convert from C to Kelvins
	      TDin(lv) = TDin(lv) + T0
         !if(sndcase=='ALBERTA') TDin(lv) = TDin(lv) + T0 !        " "
         !if(sndcase=='HARP')    TDin(lv) = TDin(lv)      !'TD' is actually Qv
         Pin(lv)  = Pin(lv)*100                          !convert from mb to Pa
      enddo
      close(12)

!-- Set up vertical level (evenly-spaced z-levels)
!       RE-CODE if necessary

!!-- Read in levels from file:
!!    open(unit=20, file='./levels/levs_86.dat')
!!    open(unit=20, file='./levels/levs_62.dat')
      open(unit=20, file='./levels/levs_41.dat')

      read(20,*) nk_read
      if (nk /= nk_read) then
         print*, '*** Abort in CLD1D ***'
         print*, 'Mismatch in nk specified for arrays and nk from table of levels: ',nk,nk_read
         stop
      endif
      read(20,*)
      read(20,*)
      do kk = 1,nk_read
         read(20,*) k,dum1,z(k),dum1,dum2
      enddo
      close(20)

!-- Compute dz2d(1,k) for upwind sedimentation.
!    note: this overrides the "DELTA_Z" in the file, which is not appropriate for P3
      if (trim(model)=='WRF') then
         do k = 1,nk-1
            dz2d(1,k) = z(k+1) - z(k)
         enddo
         dz2d(1,nk) = dz2d(1,nk-1)  !copy second highest DZ to highest
      else
         do k = 2,nk
            dz2d(1,k) = z(k-1) - z(k)
         enddo
         dz2d(1,1) = dz2d(1,2)  !copy second highest DZ to highest
      endif

!  Set up vertical level (evenly-spaced z-levels)
!      z(nk) = Zin(1)
!      delz = (H-Zin(1))/(nk-1)
!      do k=1,nk
!         z(k) = Zin(1) + delz*(nk-k)
!         dz2d(1,k) = delz
!      enddo
!      dz = z(1)-z(2)
!      dzsq = dz**2.
!      H0= z(nk)
!      GZ(1,:)= z*GRAV

!  Interpolate p,tt,td from sounding data and initialize Q[x]:
      do k=1,nk
         call vertint2b( p(k),z(k),Pin, Zin,nk,nlvs)
         call vertint2b(tt(k),z(k),TTin,Zin,nk,nlvs)
         call vertint2b(td(k),z(k),TDin,Zin,nk,nlvs)

         rho(k)= p(k)/(Rd_cld*tt(k))
!        rho(k)= 1.
         p2d(1,k) = p(k)

         Qsat(1,k)= FOQST(tt(k),p(k))
         if(sndcase=='ALBERTA') Qv_m(1,k) = FOQST(td(k),p(k))  !Alberta (Td read in)
         Qvinit(1,k) = Qv_m(1,k)

         Qc_m(1,k) = 0.
         Nc_m(1,k) = 0.
         Qr_m(1,k) = 0.
         Nr_m(1,k) = 0.
!        ssat_m(1,k) = 0.   !supersaturation mixing ratio (qv-qv_sat)

         Qitot_m(1,k,:) = 0.   !qitot (total ice mass mixing ratio)
         Qirim_m(1,k,:) = 0.   !qirim (riming mass mixing ratio)
         Qiliq_m(1,k,:) = 0.   !qiliq (liquid on ice mass mixing ratio)
         Nitot_m(1,k,:) = 0.   !nitot (number mixing ratio)
         Birim_m(1,k,:) = 0.   !birim (rime volume mixing ratio)
         Zitot_m(1,k,:) = 0.   !zitot (reflectivity mixing ratio)

         tt0(1,k) = tt(k)
         tt1(1,k) = tt(k)

         Dmx(k) = 0.
      enddo

      p0(1)  = 100000.    !surface air pressure [Pa]
      p0m(1) = p0(1)
      do k = 1,nk
        SIGMA(1,k) = p(k)/p(nk)
      enddo

!  Define initial w-profiles (and profiles related to w):
      H0    = 0.
      Hcld  = H
      nkcld = nk
      if (.not.EVOLVING) then
        call NewWprof2(w,w1,AMPB,H,DIV,COMP,rho,dt,z,nk)
        do k=1,nk
          womega(:,k) = -w(:,k)*rho(k)*GRAV
        enddo
      endif
!**  z, w, p, tt, td and Qx0 are now in arrays from 1 to nk  **


! !---
! ! For sedimentation testing::
! !   set-up for prescribed initial values
! do k = 20,22
!   Qitot_m(1,k,1) = 6.E-03
!   Qirim_m(1,k,1) = 5.E-03
!   Qiliq_m(1,k,1) = 0.00000000
!   Nitot_m(1,k,1) = 38238.1250
!   Birim_m(1,k,1) = 7.54279154E-06
!   Zitot_m(1,k,1) = 1.45307844E-04   !note: zitot is the advected variable, not mom6
! enddo
! !---


      do k = 1,nk
         Qsat(1,k) = 0.
         Qv(1,k) = Qv_m(1,k)
         Qc(1,k) = Qc_m(1,k)
         Nc(1,k) = Nc_m(1,k)
         Qr(1,k) = Qr_m(1,k)
         Nr(1,k) = Nr_m(1,k)
!        ssat(1,k) = ssat_m(1,k)

         Qitot(1,k,:) = Qitot_m(1,k,:)
         Qirim(1,k,:) = Qirim_m(1,k,:)
         Qiliq(1,k,:) = Qiliq_m(1,k,:)
         Nitot(1,k,:) = Nitot_m(1,k,:)
         Birim(1,k,:) = Birim_m(1,k,:)
         Zitot(1,k,:) = Zitot_m(1,k,:)
      enddo



!-------------------------------------------------------------------------!
!  END OF INITIAL SET-UP

! !     call P3_INIT(LT_path,nCat,trplMomIce,model,stat,abort_on_err)                   !v4.5.1
!       call P3_INIT(LT_path,nCat,trplMomIce,liqFrac,model,stat,abort_on_err,dowr)      !v5


!       do k=1,nk
!          write(100,'(1f10.0, 1f8.1, 8e16.6, 1f10.2, 1f10.1, 1f10.3)') z(k),diag_ZET(1,k),          &
!                     Qc(1,k), Nc(1,k), Qr(1,k), Qitot(1,k,1), Qirim(1,k,1), Birim(1,k,1), Nitot(1,k,1),  &
!                     Qiliq(1,k,1), w(1,k)  , tt1(1,k), scpf_cldfrac(1,k)
! !        write(200,'(1i6,1f10.0,1f8.1,6e16.6,2f10.2,2e16.6)') k,z(k),diag_ZET(1,k),Qr(1,k), Nr(1,k)
! !        write(300,'(1f10.0,1f8.1,20e16.6)') z(k),diag_ZET(1,k),Qitot(1,k,1),Zitot(1,k,1)
!        enddo


!=========================================================================!
!  MAIN TIME LOOP:
      print*, 'Starting main loop...'
      print*

      DO step = 1,nt

        tsec  = step*dt         !integration time [s]    (real)
        tminr = tsec/60.        !integration time [min]  (real)
        tmin  = nint(tminr)     !integration time [min]  (integer)


!  tsec-dependent profiles: (evolving updraft only)
        if (EVOLVING) then
          wmaxH = AMPA + (AMPB-AMPA)*0.5*(cos((tsec/tscale1)*2.*pi+pi)+1.)
          if (tsec .gt. 3600.) wmaxH= 0.

          Hcld = Htop0+ (H-Htop0)*0.5*(cos((tsec/tscale2)*2.*pi+pi)+1.)
          k = 1  !Find nkcld:
          do while(Hcld<z(k))
            nkcld = k
            k = k+1
          enddo
          nkcld = min(nk,nk+1-nkcld)
!         call NewWprof2(w,w1,wmaxH,H,DIV,COMP,rho,dt,z,nk)     !t-dep W only
          call NewWprof2(w,w1,wmaxH,Hcld,DIV,COMP,rho,dt,z,nk)  !t-dep W & top
          do k=1,nk
            womega(:,k) = -w(:,k)*rho(k)*GRAV
          enddo
        endif

!       goto 300  !skip advection (testing only)

        call advec(tt1,w,H,H0,Hcld,nk,nkcld,dt)
        call advec(Qv,w,H,H0,Hcld,nk,nkcld,dt)
        call advec(Qc,w,H,H0,Hcld,nk,nkcld,dt)
        call advec(Qr,w,H,H0,Hcld,nk,nkcld,dt)
        call advec(Nc,w,H,H0,Hcld,nk,nkcld,dt)
        call advec(Nr,w,H,H0,Hcld,nk,nkcld,dt)
!       call advec(ssat,w,H,H0,Hcld,nk,nkcld,dt)

        do iice = 1,nCat
           call advec(Qitot(:,:,iice),w,H,H0,Hcld,nk,nkcld,dt)
           call advec(Qirim(:,:,iice),w,H,H0,Hcld,nk,nkcld,dt)
           call advec(Qiliq(:,:,iice),w,H,H0,Hcld,nk,nkcld,dt)
           call advec(Nitot(:,:,iice),w,H,H0,Hcld,nk,nkcld,dt)
           call advec(Birim(:,:,iice),w,H,H0,Hcld,nk,nkcld,dt)
           call advec(Zitot(:,:,iice),w,H,H0,Hcld,nk,nkcld,dt)
        enddo


        do k=1,nk
          Qv(1,k) = amax1(0., Qv(1,k)-Qv_m(1,k)*(DIV(k)+COMP(k))*dt)
          Qc(1,k) = amax1(0., Qc(1,k)-Qc_m(1,k)*(DIV(k)+COMP(k))*dt)
          Qr(1,k) = amax1(0., Qr(1,k)-Qr_m(1,k)*(DIV(k)+COMP(k))*dt)
          Nc(1,k) = amax1(0., Nc(1,k)-Nc_m(1,k)*(DIV(k)+COMP(k))*dt)
          Nr(1,k) = amax1(0., Nr(1,k)-Nr_m(1,k)*(DIV(k)+COMP(k))*dt)
!         ssat(1,k)= amax1(0., ssat(1,k)-ssat_m(1,k)*(DIV(k)+COMP(k))*dt)
!         ssat(1,k)= ssat(1,k)-ssat_m(1,k)*(DIV(k)+COMP(k))*dt


          do iice = 1,nCat
             Qitot(1,k,iice) = amax1(0., Qitot(1,k,iice)-Qitot_m(1,k,iice)*(DIV(k)+COMP(k))*dt)
             Qirim(1,k,iice) = amax1(0., Qirim(1,k,iice)-Qirim_m(1,k,iice)*(DIV(k)+COMP(k))*dt)
             Qiliq(1,k,iice) = amax1(0., Qiliq(1,k,iice)-Qiliq_m(1,k,iice)*(DIV(k)+COMP(k))*dt)
             Birim(1,k,iice) = amax1(0., Birim(1,k,iice)-Birim_m(1,k,iice)*(DIV(k)+COMP(k))*dt)
             Nitot(1,k,iice) = amax1(0., Nitot(1,k,iice)-Nitot_m(1,k,iice)*(DIV(k)+COMP(k))*dt)
             Zitot(1,k,iice) = amax1(0., Zitot(1,k,iice)-Zitot_m(1,k,iice)*(DIV(k)+COMP(k))*dt)
          enddo

!       Adiabatic (cooling due to ascent):
          tt1(1,k) = tt1(1,k) + dt*(-g_cld/cp_cld*w(1,k))
        enddo

 300    continue

!------------------------------------------------------------------------!
       IF (microON) THEN

          th2d0(1,:) = tt0(1,:)*(1.e+5/p(:))**0.286
          th2d1(1,:) = tt1(1,:)*(1.e+5/p(:))**0.286

          if (.not. trplMomIce) Zitot(:,:,:) = 0.
          if (.not. liqFrac)    Qiliq(:,:,:) = 0.

          call cpu_time(time1)

#ifdef v4
!v4.5.1:
          if (.not. trplMomIce) then

             CALL P3_MAIN(Qc,Nc,Qr,Nr,th2d0,th2d1,Qv_m,Qv,dt,Qitot,Qirim,Nitot,Birim,ssat, &
                          w,p2d,dz2d,step,prt_liq,prt_sol,its,ite,kts,kte,nCat,         &
                          diag_ZET,diag_reffc,diag_reffi,diag_vmi,diag_di,diag_rhoi,    &
                          n_diag_2d,diag_2d,n_diag_3d,diag_3d,log_predictNc,            &
!                           typeDiags_ON,trim(model),clbfact_dep,clbfact_sub,debug_on,    &
                          .false.,trim(model),clbfact_dep,clbfact_sub,debug_on,         &
                          scpf_on,scpf_pfrac,scpf_resfact,scpf_cldfrac,                 &
                          prt_drzl   = prt_drzl,   &
                          prt_rain   = prt_rain,   &
                          prt_crys   = prt_crys,   &
                          prt_snow   = prt_snow,   &
                          prt_grpl   = prt_grpl,   &
                          prt_pell   = prt_pell,   &
                          prt_hail   = prt_hail,   &
                          prt_sndp   = prt_sndp,   &
                          qi_type    = qi_type,    &  !) !,    &
!                           diag_dhmax = diag_dhmax, &
!                           diag_lami  = diag_lami,  &
!                           diag_mui   = diag_mui)
                          timer = timer,                                                 &
                          timer_description = timer_txt)

          else

             CALL P3_MAIN(Qc,Nc,Qr,Nr,th2d0,th2d1,Qv_m,Qv,dt,Qitot,Qirim,Nitot,Birim,ssat, &
                          w,p2d,dz2d,step,prt_liq,prt_sol,its,ite,kts,kte,nCat,         &
                          diag_ZET,diag_reffc,diag_reffi,diag_vmi,diag_di,diag_rhoi,    &
                          n_diag_2d,diag_2d,n_diag_3d,diag_3d,log_predictNc,            &
!                         typeDiags_ON,trim(model),clbfact_dep,clbfact_sub,debug_on,    &
                          .false.,trim(model),clbfact_dep,clbfact_sub,debug_on,         &
                          scpf_on,scpf_pfrac,scpf_resfact,scpf_cldfrac,                 &
                          prt_drzl   = prt_drzl,   &
                          prt_rain   = prt_rain,   &
                          prt_crys   = prt_crys,   &
                          prt_snow   = prt_snow,   &
                          prt_grpl   = prt_grpl,   &
                          prt_pell   = prt_pell,   &
                          prt_hail   = prt_hail,   &
                          prt_sndp   = prt_sndp,   &
                          qi_type    = qi_type,    &
                          zitot      = Zitot,      &  !) !,        &
!                           diag_dhmax = diag_dhmax, &
!                           diag_lami  = diag_lami,  &
!                           diag_mui   = diag_mui)
                          timer = timer,                                                  &
                          timer_description = timer_txt)
          endif
#else

!v5.3.14:
          CALL P3_MAIN(Qc,Nc,Qr,Nr,th2d0,th2d1,Qv_m,Qv,dt,Qitot,Qirim,Qiliq,Nitot,Birim,Zitot,  &
                             ssat,w,p2d,dz2d,step,prt_liq,prt_sol,its,ite,kts,kte,nCat,         &
                             diag_ZET,diag_reffc,diag_reffi,diag_vmi,diag_di,diag_rhoi,         &
                             n_diag_2d,diag_2d,n_diag_3d,diag_3d,log_predictNc,                 &
                             trim(model),clbfact_dep,clbfact_sub,debug_on,                      &
                             scpf_on,scpf_pfrac,scpf_resfact,scpf_cldfrac,trplMomIce,liqFrac,   &
                             prt_drzl = prt_drzl,                                               &
                             prt_rain = prt_rain,                                               &
                             prt_crys = prt_crys,                                               &
                             prt_snow = prt_snow,                                               &
                             prt_grpl = prt_grpl,                                               &
                             prt_pell = prt_pell,                                               &
                             prt_hail = prt_hail,                                               &
                             prt_sndp = prt_sndp,                                               &
                             prt_wsnow = prt_wsnow,                                             &
                             qi_type  = qi_type,                                                &
!                            diag_dhmax = diag_dhmax)
                             diag_dhmax = diag_dhmax,                                           &
                             timer = timer,                                                     &
                             timer_description = timer_txt)

#endif

          call cpu_time(time2)

          tt1(1,:) = th2d1(1,:)*(p(:)*1.e-5)**0.286

         !convert precipitation rates to units mm h-1 (from m s-1)
         prt_liq  = prt_liq *3.6e+6  !total liquid
         prt_sol  = prt_sol *3.6e+6  !total solid
         prt_drzl = prt_drzl*3.6e+6
         prt_rain = prt_rain*3.6e+6
         prt_crys = prt_crys*3.6e+6
         prt_snow = prt_snow*3.6e+6
         prt_grpl = prt_grpl*3.6e+6
         prt_pell = prt_pell*3.6e+6
         prt_hail = prt_hail*3.6e+6
         prt_sndp = prt_sndp*3.6e+6
         prt_wsnow = prt_wsnow*3.6e+6

         timer_accum(:) = timer_accum(:) + timer(:)
!        timer_accum(1) = timer_accum(1) + timer(1)    ! full p3_main call

!        timer_accum(1)   ! full p3_main call
!        timer_accum(2)   ! ice sedimentation
!        timer_accum(3)   ! update full reflectivity

!*-----------------------------------------------------------------------*!

      ENDIF   !(microON)

!  Rearrange arrays:
         DO k = 1,nk
            Qv_m(1,k) = Qv(1,k)
            tt0(1,k) = tt1(1,k)
            Qc_m(1,k) = Qc(1,k)
            Qr_m(1,k) = Qr(1,k)
            Nc_m(1,k) = Nc(1,k)
            Nr_m(1,k) = Nr(1,k)
            Qitot_m(1,k,:) = Qitot(1,k,:)
            Qirim_m(1,k,:) = Qirim(1,k,:)
            Qiliq_m(1,k,:) = Qiliq(1,k,:)
            Nitot_m(1,k,:) = Nitot(1,k,:)
            Birim_m(1,k,:) = Birim(1,k,:)
            Zitot_m(1,k,:) = Zitot(1,k,:)
!           ssat_m(1,k) = ssat(1,k)
         ENDDO
         p0m(1) = p0(1)

!--  Prevent low-level moisture depletion:     ----------------!
!    (Impose Td >= Td_initial after  xx min. [..< xx)] )
!        if (TDMIN .and. step<70) then
         if (TDMIN) then
           do k = 1,nk    ! or, e.g., k=nk-10,nk
             if (z(k)<1000.) then
!              Qv_m(1,k)= amax1(Qv(1,k),Qvinit(1,k))
               Qv_m(1,k)= amax1(Qv(1,k),Qvinit(1,k)*0.4) !test
               Qv(1,k)= Qv_m(1,k)
             endif
           enddo
         endif

         PR = PR + (prt_liq(1) + prt_sol(1))*dt*1.e-3  !accumulated precipitation, mm

         time3 = time3 + (time2-time1)

!----------------------------------------------------------------!

!  Output to files:

         if (step.eq.1) then
            write(*,'(A)') '                PR_liq      PR_sol      Precip      max Ze       Time '
            write(*,'(A)') '   Time         mm h-1      mm h-1        mm         dBZ           s  '
            write(*,'(A)') '  ------        ------      ------      ------      ------       -----'

!            print*
         endif

         if (mod(tminr,float(5)) < 1.e-5)  &
           write(*,'(I4,A, 6F12.3)') int(tminr),' min: ',prt_liq(1),prt_sol(1),PR, maxval(diag_ZET(1,:)), time3

         IF (mod(tminr,float(outfreq)) < 1.e-5) THEN  !output only every OUTFREQ min   !for TESING

	         do k=1,nk

              if (nCat.eq.1) then

                 dum1 = 0.
                 dum2 = 0.
                 dum3 = 0.
                 if (Qitot(1,k,1).gt.0.) dum1 = Qirim(1,k,1)/(Qitot(1,k,1)-Qiliq(1,k,1))      ! rime fraction
                 if (Qitot(1,k,1).gt.0.) dum2 = Qiliq(1,k,1)/Qitot(1,k,1)                     ! liquid fraction
                 if (Nr(1,k).gt.0.) dum3 = (6.*Qr(1,k)/(pi*1000.*Nr(1,k)))**(1./3.)           ! Drm
                 write(30,'(26e13.4)')   &
                      z(k),              &  !col 0 (for python plot script)
                      w(1,k),            &  !col 1
                      prt_liq(1),        &  !col 2
                      prt_sol(1),        &  !col 3
                      diag_ZET(1,k),     &  !col 4
                      tt1(1,k)-T0,       &  !col 5
                      Qc(1,k),           &  !col 6
                      Qr(1,k),           &  !col 7
                      Nc(1,k),           &  !col 8
                      Nr(1,k),           &  !col 9
                      sum(Qitot(1,k,:)), &  !col 10
                      sum(Nitot(1,k,:)), &  !col 11
                      dum1,              &  !col 12
                      dum2,              &  !col 13
                      dum3,              &  !col 14
                      Qitot(1,k,1),      &  !col 15
                      Qirim(1,k,1),      &  !col 16
                      Qiliq(1,k,1),      &  !col 17
                      Nitot(1,k,1),      &  !col 18
                      Birim(1,k,1),      &  !col 19
                      Zitot(1,k,1),      &  !col 20
                      diag_rhoi(1,k,1),  &  !col 21
                      diag_di(1,k,1)        !col 22

              elseif (nCat.eq.2) then

                 dum1 = 0.
                 dum2 = 0.
                 dum3 = 0.
                 write(30,'(31e13.4)')   &
                      z(k),              &  !col 0 (for python plot script)
                      w(1,k),            &  !col 1
                      prt_liq(1),        &  !col 2
                      prt_sol(1),        &  !col 3
                      diag_ZET(1,k),     &  !col 4
                      tt1(1,k)-T0,       &  !col 5
                      Qc(1,k),           &  !col 6
                      Qr(1,k),           &  !col 7
                      Nc(1,k),           &  !col 8
                      Nr(1,k),           &  !col 9
                      sum(Qitot(1,k,:)), &  !col 10
                      sum(Nitot(1,k,:)), &  !col 11
                      dum1,              &  !col 12
                      dum2,              &  !col 13
                      dum3,              &  !col 14
                      Qitot(1,k,1),      &  !col 15
                      Qirim(1,k,1),      &  !col 16
                      Qiliq(1,k,1),      &  !col 17
                      Nitot(1,k,1),      &  !col 18
                      Birim(1,k,1),      &  !col 19
                      Zitot(1,k,1),      &  !col 20
                      diag_rhoi(1,k,1),  &  !col 21
                      diag_di(1,k,1),    &  !col 22
                      Qitot(1,k,2),      &  !col 23
                      Qirim(1,k,2),      &  !col 24
                      Qiliq(1,k,2),      &  !col 25
                      Nitot(1,k,2),      &  !col 26
                      Birim(1,k,2),      &  !col 27
                      Zitot(1,k,2),      &  !col 28
                      diag_rhoi(1,k,2),  &  !col 29
                      diag_di(1,k,2)        !col 30

              elseif (nCat.eq.3) then

                 dum1 = 0.
                 dum2 = 0.
                 dum3 = 0.
                 write(30,'(39e13.4)')   &
                      z(k),              &  !col 0 (for python plot script)
                      w(1,k),            &  !col 1
                      prt_liq(1),        &  !col 2
                      prt_sol(1),        &  !col 3
                      diag_ZET(1,k),     &  !col 4
                      tt1(1,k)-T0,       &  !col 5
                      Qc(1,k),           &  !col 6
                      Qr(1,k),           &  !col 7
                      Nc(1,k),           &  !col 8
                      Nr(1,k),           &  !col 9
                      sum(Qitot(1,k,:)), &  !col 10
                      sum(Nitot(1,k,:)), &  !col 11
                      dum1,              &  !col 12
                      dum2,              &  !col 13
                      dum3,              &  !col 14
                      Qitot(1,k,1),      &  !col 15
                      Qirim(1,k,1),      &  !col 16
                      Qiliq(1,k,1),      &  !col 17
                      Nitot(1,k,1),      &  !col 18
                      Birim(1,k,1),      &  !col 19
                      Zitot(1,k,1),      &  !col 20
                      diag_rhoi(1,k,1),  &  !col 21
                      diag_di(1,k,1),    &  !col 22
                      Qitot(1,k,2),      &  !col 23
                      Qirim(1,k,2),      &  !col 24
                      Qiliq(1,k,2),      &  !col 25
                      Nitot(1,k,2),      &  !col 26
                      Birim(1,k,2),      &  !col 27
                      Zitot(1,k,2),      &  !col 28
                      diag_rhoi(1,k,2),  &  !col 29
                      diag_di(1,k,2),    &  !col 30
                      Qitot(1,k,3),      &  !col 31
                      Qirim(1,k,3),      &  !col 32
                      Qiliq(1,k,3),      &  !col 33
                      Nitot(1,k,3),      &  !col 34
                      Birim(1,k,3),      &  !col 35
                      Zitot(1,k,3),      &  !col 36
                      diag_rhoi(1,k,3),  &  !col 37
                      diag_di(1,k,3)        !col 38

              endif

            enddo  ! k-loop

         ENDIF    ! outfreq

      ENDDO  !main time loop


      print*
      print*, 'P3 CONFIGURATION:'
      print*
      write(*,'(a12,a20)')  'version:  ', version_p3
      write(*,'(1a12,1i1)') 'nCat   :  ', nCat
      write(*,'(1a12,1L)')  'trlMom :  ', trplMomIce
      write(*,'(1a12,1L)')  'liqFrac:  ', liqFrac
      print*
      print*, 'ACCUMULATED CPU TIMINGS:'
      print*
      print*, 'Block Description              Time (ms)     % of p3_main'
      print*, '------------------             ---------     ------------'
!      write(*,'(1a20,10f16.6)') timer_txt(2)//': ',timer_accum(1),timer_accum(2),timer_accum(3)
      do ind = 1,20
         if (timer_txt(ind) /= '')  &
           write(*,'(i4,1a20,2f16.6)') ind,' '//timer_txt(ind)//': ',timer_accum(ind)*10.,timer_accum(ind)/timer_accum(1)*100.
      enddo

      open (unit=55, file='timing.txt')

      write(55,*)
      write(55,*) 'P3 CONFIGURATION:'
      write(55,*)
      write(55,'(a12,a20)')  'version:  ', version_p3
      write(55,'(1a12,1i1)') 'nCat   :  ', nCat
      write(55,'(1a12,1L)')  'trlMom :  ', trplMomIce
      write(55,'(1a12,1L)')  'liqFrac:  ', liqFrac
      write(55,*)
      write(55,*) 'ACCUMULATED CPU TIMINGS:'
      write(55,*)
      write(55,*) 'Block Description              Time (ms)     % of p3_main'
      write(55,*) '------------------             ---------     ------------'
      do ind = 1,20
         if (timer_txt(ind) /= '')  &
           write(55,'(i4,1a20,2f16.6)') ind,' '//timer_txt(ind)//': ',timer_accum(ind)*10.,timer_accum(ind)/timer_accum(1)*100.
      enddo
      write(55,*)

      close(55)

      close(30)

!----------------------------------------------------------!
!            ---  End of time integreation  ---            !
!----------------------------------------------------------!
      print*
      print*, 'DONE'

end subroutine columnmodel
!=============================================================================!
