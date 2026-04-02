! Standalone MPAS acoustic substep reference implementation
! Implements Sections 1-8 of mpas_atm_time_integration.F for a single column.
! Used to validate Breeze's Julia implementation coefficient-by-coefficient.
!
! Compile: gfortran -o mpas_acoustic_reference mpas_acoustic_reference.f90
! Run:     ./mpas_acoustic_reference

program mpas_acoustic_reference
  implicit none

  integer, parameter :: nVertLevels = 10
  real(8), parameter :: gravity = 9.80616d0
  real(8), parameter :: rgas = 287.0d0
  real(8), parameter :: cp = 1004.5d0
  real(8), parameter :: p0 = 1.0d5
  real(8), parameter :: t0b = 300.0d0  ! reference temperature for base state (θ₀=300)

  real(8) :: rcv, c2, kappa, dts, dtseps, epssm, resm
  real(8) :: dz, rdzu, rdzw

  ! Base state (1D isothermal at T₀=300K, matching Breeze θ₀=300)
  real(8) :: rb(nVertLevels), rtb(nVertLevels), pb(nVertLevels)
  real(8) :: theta_b(nVertLevels), ppb(nVertLevels)

  ! Current state (same as base state + small perturbation)
  real(8) :: p_exner(nVertLevels), theta_m(nVertLevels)
  real(8) :: rho(nVertLevels), rtheta(nVertLevels)

  ! Coefficients
  real(8) :: cofwz(nVertLevels), cofwr(nVertLevels), cofwt(nVertLevels)
  real(8) :: coftz(nVertLevels+1), cofrz(nVertLevels)

  ! Tridiagonal
  real(8) :: a_tri(nVertLevels), b_tri(nVertLevels), c_tri(nVertLevels)
  real(8) :: alpha_tri(nVertLevels), gamma_tri(nVertLevels)

  ! Acoustic perturbation variables
  real(8) :: rw_p(nVertLevels+1), rho_pp(nVertLevels), rtheta_pp(nVertLevels)
  real(8) :: ts(nVertLevels), rs(nVertLevels)

  ! Slow tendencies (set to small test values)
  real(8) :: tend_rw(nVertLevels+1), tend_rho(nVertLevels), tend_rt(nVertLevels)

  real(8) :: Hs, z, fzm, fzp
  integer :: k

  ! Parameters
  kappa = rgas / cp
  rcv = rgas / (cp - rgas)
  c2 = cp * rcv
  dz = 20000.0d0 / nVertLevels   ! 2km per level (matching 20km domain)
  rdzu = 1.0d0 / dz
  rdzw = 1.0d0 / dz
  dts = 2.0d0  ! acoustic substep size
  epssm = 0.2d0  ! off-centering (ω = 0.6 → epssm = 2*0.6 - 1 = 0.2)
  dtseps = 0.5d0 * dts * (1.0d0 + epssm)
  resm = (1.0d0 - epssm) / (1.0d0 + epssm)

  ! --- Build base state (isothermal at T₀=300K) ---
  Hs = rgas * t0b / gravity
  do k = 1, nVertLevels
    z = (k - 0.5d0) * dz  ! cell center height
    ppb(k) = p0 * exp(-gravity * z / (rgas * t0b))
    pb(k) = (ppb(k) / p0) ** kappa
    rb(k) = ppb(k) / (rgas * t0b)
    theta_b(k) = t0b / pb(k)
    rtb(k) = rb(k) * theta_b(k)
  end do

  ! --- Current state = base state + meridional θ perturbation ---
  ! (Mimicking θ = 300 + 30*y/L at the midpoint of the domain)
  do k = 1, nVertLevels
    theta_m(k) = theta_b(k) + 15.0d0  ! θ perturbation of +15K
    p_exner(k) = pb(k)  ! approximate (would need rebalancing for exact)
    rho(k) = rb(k)
    rtheta(k) = rho(k) * theta_m(k)
  end do

  ! --- Slow tendencies (small, from buoyancy of θ perturbation) ---
  do k = 1, nVertLevels
    tend_rho(k) = 0.0d0
    tend_rt(k) = 0.0d0
  end do
  do k = 2, nVertLevels
    ! Slow w tendency ≈ buoyancy from θ perturbation: g Δθ/θ₀
    tend_rw(k) = rho(k) * gravity * 15.0d0 / 300.0d0  ! ρ * g * Δθ/θ₀
  end do
  tend_rw(1) = 0.0d0
  tend_rw(nVertLevels + 1) = 0.0d0

  ! --- Section 1: Compute coefficients ---
  do k = 1, nVertLevels
    cofrz(k) = dtseps * rdzw
  end do

  coftz(1) = 0.0d0
  coftz(nVertLevels + 1) = 0.0d0
  do k = 2, nVertLevels
    fzm = 0.5d0  ! uniform grid: equal weights
    fzp = 0.5d0
    cofwr(k) = 0.5d0 * dtseps * gravity  ! zz = 1
    cofwz(k) = dtseps * c2 * rdzu * (fzm * p_exner(k) + fzp * p_exner(k-1))
    coftz(k) = dtseps * (fzm * theta_m(k) + fzp * theta_m(k-1))
  end do

  do k = 1, nVertLevels
    cofwt(k) = 0.5d0 * dtseps * rcv * gravity * rb(k) &
               * p_exner(k) / (rtheta(k) * pb(k))
  end do

  ! --- Section 2: Build tridiagonal and LU decompose ---
  a_tri(1) = 0.0d0
  b_tri(1) = 1.0d0
  c_tri(1) = 0.0d0
  gamma_tri(1) = 0.0d0

  do k = 2, nVertLevels
    a_tri(k) = -cofwz(k) * coftz(k-1) * rdzw &
               + cofwr(k) * cofrz(k-1) &
               - cofwt(k-1) * coftz(k-1) * rdzw

    b_tri(k) = 1.0d0 &
               + cofwz(k) * (coftz(k) * rdzw + coftz(k) * rdzw) &
               - coftz(k) * (cofwt(k) * rdzw - cofwt(k-1) * rdzw) &
               + cofwr(k) * (cofrz(k) - cofrz(k-1))

    c_tri(k) = -cofwz(k) * coftz(k+1) * rdzw &
               - cofwr(k) * cofrz(k) &
               + cofwt(k) * coftz(k+1) * rdzw
  end do

  do k = 2, nVertLevels
    alpha_tri(k) = 1.0d0 / (b_tri(k) - a_tri(k) * gamma_tri(k-1))
    gamma_tri(k) = c_tri(k) * alpha_tri(k)
  end do

  ! Print coefficients
  write(*,*) '=== MPAS Reference Coefficients ==='
  write(*,*) 'dtseps =', dtseps, 'resm =', resm
  do k = 2, nVertLevels
    write(*,'(A,I2,6(A,E12.5))') 'k=', k, &
      ' cofwz=', cofwz(k), ' cofwr=', cofwr(k), ' cofwt_k=', cofwt(k), &
      ' cofwt_km1=', cofwt(k-1), ' coftz=', coftz(k), ' cofrz=', cofrz(k)
  end do
  do k = 2, nVertLevels
    write(*,'(A,I2,4(A,E12.5))') 'k=', k, &
      ' a_tri=', a_tri(k), ' b_tri=', b_tri(k), &
      ' alpha=', alpha_tri(k), ' gamma=', gamma_tri(k)
  end do

  ! --- Section 3: Initialize perturbation variables ---
  do k = 1, nVertLevels
    rho_pp(k) = 0.0d0
    rtheta_pp(k) = 0.0d0
    rw_p(k) = 0.0d0
  end do
  rw_p(nVertLevels + 1) = 0.0d0

  ! --- Section 4: Compute ts, rs (no horizontal flux for this test) ---
  do k = 1, nVertLevels
    ts(k) = 0.0d0
    rs(k) = 0.0d0
  end do
  do k = 1, nVertLevels
    rs(k) = rho_pp(k) + dts * tend_rho(k) + rs(k) &
            - cofrz(k) * resm * (rw_p(k+1) - rw_p(k))
    ts(k) = rtheta_pp(k) + dts * tend_rt(k) + ts(k) &
            - resm * rdzw * (coftz(k+1) * rw_p(k+1) - coftz(k) * rw_p(k))
  end do

  ! --- Section 5: Explicit w update ---
  do k = 2, nVertLevels
    rw_p(k) = rw_p(k) + dts * tend_rw(k) &
              - cofwz(k) * ((ts(k) - ts(k-1)) &
                + resm * (rtheta_pp(k) - rtheta_pp(k-1))) &
              - cofwr(k) * ((rs(k) + rs(k-1)) &
                + resm * (rho_pp(k) + rho_pp(k-1))) &
              + cofwt(k) * (ts(k) + resm * rtheta_pp(k)) &
              + cofwt(k-1) * (ts(k-1) + resm * rtheta_pp(k-1))
  end do

  write(*,*) ''
  write(*,*) '=== After explicit w update (before tridiag) ==='
  do k = 2, nVertLevels
    write(*,'(A,I2,3(A,E12.5))') 'k=', k, &
      ' rw_p=', rw_p(k), ' ts=', ts(k), ' rs=', rs(k)
  end do

  ! --- Section 6: Tridiagonal solve ---
  do k = 2, nVertLevels
    rw_p(k) = (rw_p(k) - a_tri(k) * rw_p(k-1)) * alpha_tri(k)
  end do
  do k = nVertLevels - 1, 2, -1
    rw_p(k) = rw_p(k) - gamma_tri(k) * rw_p(k+1)
  end do

  write(*,*) ''
  write(*,*) '=== After tridiagonal solve ==='
  do k = 2, nVertLevels
    write(*,'(A,I2,A,E12.5)') 'k=', k, ' rw_p=', rw_p(k)
  end do

  ! --- Section 8: Update rho_pp, rtheta_pp ---
  do k = 1, nVertLevels
    rho_pp(k) = rs(k) - cofrz(k) * (rw_p(k+1) - rw_p(k))
    rtheta_pp(k) = ts(k) - rdzw * (coftz(k+1) * rw_p(k+1) - coftz(k) * rw_p(k))
  end do

  write(*,*) ''
  write(*,*) '=== After rho_pp/rtheta_pp update ==='
  do k = 1, nVertLevels
    write(*,'(A,I2,2(A,E12.5))') 'k=', k, &
      ' rho_pp=', rho_pp(k), ' rtheta_pp=', rtheta_pp(k)
  end do

end program mpas_acoustic_reference
