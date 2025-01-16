! ******************************************************************************!
! Important: UMAT and USDFLD is called before UMATHT for each integration point !
! ******************************************************************************!

! This is the purely elastic model

module precision
    use iso_fortran_env
    implicit none
    integer, parameter :: dp = real64
end module precision

subroutine UMAT(stress,statev,ddsdde,sse,spd,scd, &
    rpl,ddsddt,drplde,drpldt, &
    stran,dstran,time,dtime,temp,dtemp,predef,dpred,cmname, &
    ndi,nshr,ntens,nstatv,props,nprops,coords,drot,pnewdt, &
    celent,dfgrd0,dfgrd1,noel,npt,layer,kspt,kstep,kinc) 

! This subroutine requires us to update stress, ddsdde, statev, and possibly sse, spd, scd

    use precision
    include 'aba_param.inc'

    character*80 cmname
    dimension stress(ntens),statev(nstatv), &
       ddsdde(ntens,ntens), &
       ddsddt(ntens),drplde(ntens), &
       stran(ntens),dstran(ntens),time(2),predef(1),dpred(1), &
       props(nprops),coords(3),drot(3,3),dfgrd0(3,3),dfgrd1(3,3)

    ddsdde = 0.0d0
    stress = 0.0d0
    
return
end

!***********************************************************************

subroutine UMATHT(u,dudt,dudg,flux,dfdt,dfdg, &
    statev,temp,dtemp,dtemdx,time,dtime,predef,dpred, &
    cmname,ntgrd,nstatv,props,nprops,coords,pnewdt, &
    noel,npt,layer,kspt,kstep,kinc)

    use precision
    include 'aba_param.inc'

    character(len=80) :: cmname
    dimension dudg(ntgrd),flux(ntgrd),dfdt(ntgrd), &
      dfdg(ntgrd,ntgrd),statev(nstatv),dtemdx(ntgrd), &
      time(2),predef(1),dpred(1),props(nprops),coords(3)

    ! This subroutine requires us to update u, dudt, dudg, flux, dfdt, dfdg, and possibly statev, pnewdt
    
    real(kind=dp) :: R, T, VH, DL, CL, volume, molar_mass_H, molar_mass_Fe, mass_H, density_metal, mass_metal

    R        = props(1) ! Universal gas constant (8.31446 J/(mol K))
    T        = props(2) ! Temperature (300 K)
    VH       = props(3) ! Partial molar volume of hydrogen (2e-06 m^3/mol), 
    DL       = props(4) ! Diffusion coefficient for hydrogen 

    ! print *, 'R = ', R
    ! print *, 'T = ', T
    ! print *, 'VH = ', VH
    ! print *, 'DL = ', DL

    
    ! Since dudt is dCbar_total/dCL, current temperature is the current CL
    ! and the current dtemp is the current dCL 
    
    ! volume of the Thermal Desorption Spectroscopy (TDS) sample is 
    volume = 6075.0d-11 ! 0.015 m * 0.0045 m * 0.0009 m = 6075e-11 m^3
    density_metal = 7900.0d0 ! kg/m^3
    molar_mass_H = 1.00784 ! g/mol
    molar_mass_Fe = 55.845 ! g/mol
    
    mass_metal = volume * density_metal ! kg

    ! Conversion formula, if CL is in wtppm and Cbar_L is in mol/m^3
    ! Cbar_L (mol/m^3) = [ CL (wtppm) * 1e-06 (1g/1000kg) * density_metal (kg/m^3) * 1000 (g/kg) ] / molar_mass_H (g/mol)
    ! We use this case

    ! Conversion formula, if CL is in mol/m^3 and Cbar_L is in wtppm
    ! CL (wtppm) = [ Cbar_L (mol/m^3) * molar_mass_H (g/mol) ] / [ density_metal (kg/m^3) * 1e-06 (1g/1000kg) * 1000 (g/kg) ]

    CL_mol = temp + dtemp ! Unit is mol/m^3

    ! Now we need to calculate CL in wtppm
    CL_wtppm = (CL_mol * molar_mass_H) / (density_metal * 1e-06 * 1000)
    
    
    ! Equation (23) The 1st equation to update dudt, since there is no trapped hydrogen
    ! u is also t => partial (Cbar_total) / partial (Nbar_L) = 1    
    ! partial_Cbar_total_partial_CL = 1

    dudt = 1
    
    ! Equation (22) The total hydrogen diffusion equation to update u 
    ! Cbar_total_t is u
    ! Cbar_total_t_plus_1 = Cbar_total_t + partial_Cbar_total_partial_CL * dCL
    
    ! temp_u = u + dudt * dtemp
    ! if (temp_u < 0) then
    !     u = 0
    ! else
    !     u = temp_u
    ! end if

    u = u + dudt * dtemp
    
    
    
    ! Since the problem is 2-dimensional, ntgrd = 2
    ! ntgrd: Number of spatial gradients of temperature

    do i = 1, ntgrd
        ! Equation (10) to update the flux
        ! J_m = DL * CL * grad sigma_H / (R * T) - DL * grad CL
        
        ! Pure diffusion
        flux(i) = - DL * dtemdx(i)

        ! dudg is not partial (Cbar_total) / partial (Nbar_trap), be careful
        ! dudg is partial (Cbar_total) / partial (grad Cbar L), which is supposed to be 0
        dudg(i) = 0

        ! Equation (23) The 3rd equation to update dfdt
        ! partial J_m / partial (CL) = (DL * VH) / (R * T) * grad_sigma_H(i)
        !  dfdt(i) = (DL * VH * grad_sigma_hydrostatic(noel, npt, i)) / (R * T)  ! from common block
        
        ! pure diffusion
        dfdt(i) = 0  ! from common block

        ! Equation (23) The 4th equation to update dfdg
        ! partial J_m / partial (grad CL) = - DL * I 
        dfdg(i,i) = - DL

    end do
    
    ! Enforce positive concentration

    ! if (CL_mol < 0) then
    !     CL_mol = 0
    ! end if

    ! if (CL_wtppm < 0) then
    !     CL_wtppm = 0
    ! end if

    statev(1) = CL_mol
    statev(2) = CL_wtppm
    
return
end