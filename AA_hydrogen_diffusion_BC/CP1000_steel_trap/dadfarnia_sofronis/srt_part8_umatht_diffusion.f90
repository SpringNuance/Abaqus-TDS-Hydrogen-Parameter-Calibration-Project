!***********************************************************************

subroutine UMATHT_H2_diffusion(u,dudt,dudg,flux,dfdt,dfdg, &
    statev,temp,dtemp,dtemdx,time,dtime,predef,dpred, &
    cmname,ntgrd,nstatv,props,nprops,coords,pnewdt, &
    noel,npt,layer,kspt,kstep,kinc)

    use precision
    use common_block
    inCLude 'aba_param.inc'

    character(len=80) :: cmname
    dimension dudg(ntgrd),flux(ntgrd),dfdt(ntgrd), &
      dfdg(ntgrd,ntgrd),statev(nstatv),dtemdx(ntgrd), &
      time(2),predef(1),dpred(1),props(nprops),coords(3)
    
    ! This subroutine requires us to update u, dudt, dudg, flux, dfdt, dfdg, and possibly statev, pnewdt
    
    ! Define all real for all variables used 
    real(kind=dp) :: mode, mode_tol, R, T, VH, DL
    ! Define all variables related to traps
    ! There are three trap types: dislocations, grain boundaries, and carbides
    real(kind=dp) :: CT_dis, CT_gb, CT_carb, alpha_dis, alpha_gb, alpha_carb, WB_dis, WB_gb, WB_carb
    real(kind=dp) :: temp_dis, temp_gb, temp_carb, thetaT_dis, thetaT_gb, thetaT_carb, K_dis, K_gb, K_carb

    real(kind=dp) :: rho_d, N_trap, Nbar_trap, dNT_ds_deqplas, NL
    real(kind=dp) :: C_trap, Cbar_trap, Cbar_total

    ! UMATHT props indices are from 17 to 40
    mode     = props(17) ! Mode of the traps
    mode_tol = 1.0e-6
    ! mode_tol is only for checking the equality since mode is a real instead of an integer
    ! mode = 1: CT = 0
    ! mode = 2: CT_dis = 0, CT_gb and CT_carb nonzero
    ! mode = 3: Sofronis and McMeeking formulation (1989) Numerical analysis of hydrogen transport near a blunting crack tip
    ! mode = 4: Krom et al. (1999) Hydrogen transport near a blunting crack tip
    ! mode = 5: Dadfarnia et al. (2011) Hydrogen interaction with multiple traps: Can it be used to mitigate embrittlement
    
    crystal_structure = props(18) ! Crystal structure (1 - BCC, 2 - FCC)
    crystal_tol = 1.0e-6 ! same for above reason

    R           = props(19) ! Universal gas constant (8.31446 J/(mol K))
    T           = props(20) ! Temperature (300 K)
    VH          = props(21) ! Partial molar volume of hydrogen (2e-06 m^3/mol), 
    DL          = props(22) ! Diffusion coefficient for hydrogen in Nickel (3.8e-11 m^2/s) 
    avogadro    = props(23) ! Avogadro's constant (6.022e23 1/mol)
    NL          = props(24) ! Number of solvent atoms (Ni) per unit volume (9.24e28 1/m^3)
    alpha_dis   = props(25) ! Number of interstitial sites per trap site (dislocations) (1.0 dimless)
    alpha_gb    = props(26) ! Number of interstitial sites per trap site (grain boundaries) (1.0 dimless)
    alpha_carb  = props(27) ! Number of interstitial sites per trap site (carbides) (1.0 dimless)
    NT_dis      = props(28) ! Number of trap type of dislocations (1/m^2) (0 - instead defined by Dadfarnia et al. 2016)
    NT_gb       = props(29) ! Number of trap type of grain boundaries (1/m^2) (8.464e22)
    NT_carb     = props(30) ! Number of trap type of carbides NT_carb (1/m^2) (8.464e26)
    WB_dis      = props(31) ! Binding energy of hydrogen to dislocations (
    WB_gb       = props(32) ! Binding energy of hydrogen to grain boundaries (
    WB_carb     = props(33) ! Binding energy of hydrogen to carbides (
    beta_BCC    = props(34) ! Number of number of hydrogen atoms that can reside in each lattice site (6.0 dimless)
    beta_FCC    = props(35) ! Number of number of hydrogen atoms that can reside in each lattice site (1.0 dimless)
    a_lattice   = props(36) ! Lattice parameter (2.86e-10 m)
    gamma       = props(37) ! gamma fitting parameter in Dadfarnia et al. (2.0e16 1/m^2)
    rho_d0      = props(38) ! Dislocation density for the annealed material in Dadfarnia et al. (10.0e10 1/m^2)

    ! Hydrogen concentration in this subroutine is in unit atoms/m^3
    ! However, the same logic also applies to mol/m^3. So you should either uses consistent wppm or mol/m^3
    
    CLprev = temp
    dCL = dtemp
    CL = CLprev + dCL ! (1/m^3)

    ! Arhenius reaction rate constant for trap types 
    K_dis = exp( -WB_dis / (R * T)) ! constant
    K_gb = exp( -WB_gb / (R * T)) ! constant
    K_carb = exp( -WB_carb / (R * T)) ! constant


    ! slip occurs along the plane of the shortest Burgers vector
    if (abs(crystal_structure - 1.0d0) <= crystal_tol) then ! BCC crystal structure
        beta = beta_BCC ! beta is taken to be 6 for BCC as indirect
                        ! evidence indicates tetrahedral site occupancy rather than 
                        ! octahedral site occupancy at room temperature in alpha-iron
        ! slip is assumed to occur along the {110} plane and ⟨111⟩ direction
        inverse_burgers_vector = 2/(sqrt(3.0d0) * a_lattice)
    elseif (abs(crystal_structure - 2.0d0) <= crystal_tol) then ! FCC crystal structure
        beta = beta_FCC ! beta is taken to be 1 for FCC, resulting from the more favourable 
                        ! octahedral site occupancy (beta = 2 for tetrahedral)
        ! slip occurs along the closed packed plane {111} and slip direction ⟨110⟩
        inverse_burgers_vector = sqrt(2.0d0)/a_lattice
    end if

    if (abs(mode - 1) <= mode_tol) then ! Lattice H only
        NT_dis = 0.d0
        NT_gb = 0.d0
        NT_carb = 0.d0
        partial_C_partial_NT_dis = 0.d0
        dNT_dis_deqplas = 0.d0
        
    elseif (abs(mode - 2) <= mode_tol) then ! No dislocations: WB_dis and NT_dis irrelevant
        NT_dis = 0.d0
        partial_C_partial_NT_dis = 0.d0
        dNT_dis_deqplas = 0.d0
        
    elseif (abs(mode - 3) <= mode_tol) then ! Sofronis & McMeeking (in sites/m^3), developed from Kumnick & Johnson
        NT_dis = 10.d0 ** (23.26d0 - 2.33d0 * dexp(-5.5d0 * eqplas)) ! (1/m^3)
        partial_C_partial_NT_dis = 0.0d0
        dNT_dis_deqplas = 0.d0
        
    elseif (abs(mode - 4) <= mode_tol) then ! Krom et al. (in sites/m^3), developed from Kumnick & Johnson 
        NT_dis = 10.d0 ** (23.26d0 - 2.33d0 * dexp(-5.5d0 * eqplas))
        partial_C_partial_NT_dis = (K_dis * CL)/(K_dis * CL + beta * NL)
        dNT_dis_deqplas = 29.5d0 * dexp(-5.5d0 * eqplas) * NT_dis
    
    elseif (abs(mode - 5) <= mode_tol) then ! Dadfarnia et al.
        if (eqplas < 0.5) then
            NT_dis = inverse_burgers_vector * (rho_d0 + eqplas * gamma)
            partial_C_partial_NT_dis = (K_dis * CL)/(K_dis * CL + beta * NL)
            dNT_dis_deqplas = inverse_burgers_vector * 2e16
        elseif (eqplas >= 0.5) then
            NT_dis = 1e16/(b*1e6)
            partial_C_partial_NT_dis = 0.d0
            dNT_dis_deqplas = 0.d0
        endif
    endif

    
    partial_CT_dis_partial_CL = (NT_dis * K_dis * NL * beta)/((K_dis * CL + NL * beta)**2.d0)
    partial_CT_gb_partial_CL = (NT_gb * K_gb * NL * beta)/((K_gb * CL + NL * beta)**2.d0)
    partial_CT_carb_partial_CL = (NT_carb * K_carb * NL * beta)/((K_carb * CL + NL * beta)**2.d0)

    total_partial_CT_partial_CL = partial_CT_dis_partial_CL + &
                                  partial_CT_gb_partial_CL + &
                                  partial_CT_carb_partial_CL
    
    ! Finally, we update all the variables in UMATHT
    ! partial_C_partial_CL = partial_CL_partial_CL + partial_CT_partial_CL
    !                      = 1 + total_partial_CT_partial_CL
    partial_C_partial_CL = 1.d0 + total_partial_CT_partial_CL	
    dudt = partial_C_partial_CL
    
    deqplas = statev(...) 

    currentC = u
    newC = currentC + partial_C_partial_CL * dCL + partial_C_partial_NT_dis * dNT_dis_deqplas * deqplas    
    u = newC

    dfdg = 0.0

    do i = 1, ntgrd
        ! Update the flux
        grad_CL_i = dtemdx(i) ! = (mol/m^3) / m = (mol/m^4)
        flux(i) = DL * Cbar_L * VH * common_grad_sig_H(noel, npt, i) / (R * T) - DL * grad_CL_i 
        ! The flux gradient w.r.t Hydrogen is assumed to be 0
        dudg(i) = 0
        ! Update dfdt
        dfdt(i) = (DL * VH * common_grad_sig_H(noel, npt, i)) / (R * T) 
        ! Update dudg
        dfdg(i,i) = -DL ! = - m^2/s
    end do


    ! Finding thetaL 
    thetaL = CL / (beta * NL) ! = (1/m^3) / (dimless * 1/m^3) = dimless

    ! Finding theta_trap based on Oriani equilibrium theory
    ! which results in a Fermi-Dirac relation

    ! thetaT / (1 - thetaT) = K * thetaL / (1 - thetaL)
    ! However if thetaL << 1  then 
    ! thetaT / (1 - thetaT) = K * thetaL
    
    temp_dis = K_dis * thetaL / (1 - thetaL) ! (dimless)
    thetaT_dis = temp_dis / (1 + temp_dis) ! (dimless)
    temp_gb = K_gb * thetaL / (1 - thetaL) ! (dimless)
    thetaT_gb = temp_gb / (1 + temp_gb) ! (dimless)
    temp_carb = K_carb * thetaL / (1 - thetaL) ! (dimless)
    thetaT_carb = temp_carb / (1 + temp_carb) ! (dimless)

    ! store the concentration in each trap, in all traps and in traps and lattice
    
    CT = 0.0d0
    C = 0.0d0
      
    CT_dis = thetaT_dis * alpha_dis * NT_dis 
    ! That is also equivalent to this (assuming alpha_dis = 1)
    ! CT_dis = (NT_dis * K_dis * CL)/(K_dis * CL + beta * NL)
    CT_gb = thetaT_gb * alpha_gb * NT_gb
    ! That is also equivalent to this (assuming alpha_gb = 1)
    ! CT_gb = (NT_gb * K_gb * CL)/(K_gb * CL + beta * NL)
    CT_carb = thetaT_carb * alpha_carb * NT_carb
    ! That is also equivalent to this (assuming alpha_carb = 1)
    ! CT_carb = (NT_carb * K_carb * CL)/(K_carb * CL + beta * NL)    
    CT = CT_dis + CT_gb + CT_carb
    
    C = CL + CT

    statev(...) = rho_d
    statev(...) = CL
    statev(...) = CT_dis
    statev(...) = CT_gb
    statev(...) = CT_carb
    statev(...) = C
    statev(...) = thetaL
    statev(...) = thetaT_dis
    statev(...) = thetaT_gb
    statev(...) = thetaT_carb


    return
    end