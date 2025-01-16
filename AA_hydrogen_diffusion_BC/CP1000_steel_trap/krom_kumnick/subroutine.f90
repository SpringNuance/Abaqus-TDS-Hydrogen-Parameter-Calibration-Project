! This subroutine is trying to replicate the elastic plastic plate 3rd study case results from the paper
! Modelling the coupling between hydrogen diffusion and the mechanical behavior of metals

!***********************************************************************

module common_block
    implicit none
    ! 50000 simply refers to the number of elements, but it is a very big number 
    ! to accomodate varying number of elements in the future if we remesh
    ! 100 is the maximum number of integration points in an element, such as 4, 9, 20, etc

    real(kind=8) :: common_coords(50000, 100, 2)
    real(kind=8) :: grad_sigma_hydrostatic(50000, 100, 2)
    real(kind=8) :: sigma_hydrostatic(50000, 100)

    save 
    ! The save command is very important. 
    ! It allows the values to be stored and shared between subroutines 
    ! without resetting them to zero every time the subroutine is called
end module   

!***********************************************************************

subroutine UEXTERNALDB(lop,lrestart,time,dtime,kstep,kinc)

    use common_block
    include 'aba_param.inc' 
    dimension time(2)
    
    ! LOP=0 indicates that the subroutine is being called at the start of the analysis.
    if (lop == 0) then 
        common_coords = 0.0
        grad_sigma_hydrostatic = 0.0
        sigma_hydrostatic = 0.0
    end if

return
end

! CPE8RT: 8-node biquadratic displacement, bilinear temperature, reduced integration

subroutine calculate_grad_sigma_hydrostatic_CPE8RT(noel)

    use common_block

    real(kind=8) shape_func_deriv(2,4), Jacobian(2,2), inverse_Jacobian(2,2)

    ! This stores the integration point coordinate in isoparametric spaces

    integer, parameter :: isoparametric_coord_X(4) = (/ -1, +1, -1, +1 /)
    integer, parameter :: isoparametric_coord_Y(4) = (/ -1, -1, +1, +1 /)
    
    do integration_point_id = 1,4 

        xi = isoparametric_coord_X(integration_point_id)
        eta = isoparametric_coord_Y(integration_point_id)
        
        shape_func_deriv(1,1) = -(1.d0/4.0) * (1 - eta) 
        shape_func_deriv(1,2) =  (1.d0/4.0) * (1 - eta)
        shape_func_deriv(1,3) = -(1.d0/4.0) * (1 + eta)
        shape_func_deriv(1,4) =  (1.d0/4.0) * (1 + eta)
        shape_func_deriv(2,1) = -(1.d0/4.0) * (1 - xi)
        shape_func_deriv(2,2) = -(1.d0/4.0) * (1 + xi)
        shape_func_deriv(2,3) =  (1.d0/4.0) * (1 - xi)
        shape_func_deriv(2,4) =  (1.d0/4.0) * (1 + xi)

        Jacobian(1,1) = shape_func_deriv(1,1) * common_coords(noel,1,1) + &
                        shape_func_deriv(1,2) * common_coords(noel,2,1) + &
                        shape_func_deriv(1,3) * common_coords(noel,3,1) + &
                        shape_func_deriv(1,4) * common_coords(noel,4,1)
    
        Jacobian(1,2) = shape_func_deriv(1,1) * common_coords(noel,1,2) + &
                        shape_func_deriv(1,2) * common_coords(noel,2,2) + &
                        shape_func_deriv(1,3) * common_coords(noel,3,2) + &
                        shape_func_deriv(1,4) * common_coords(noel,4,2)
    
        Jacobian(2,1) = shape_func_deriv(2,1) * common_coords(noel,1,1) + &
                        shape_func_deriv(2,2) * common_coords(noel,2,1) + &
                        shape_func_deriv(2,3) * common_coords(noel,3,1) + &
                        shape_func_deriv(2,4) * common_coords(noel,4,1)
    
        Jacobian(2,2) = shape_func_deriv(2,1) * common_coords(noel,1,2) + &
                        shape_func_deriv(2,2) * common_coords(noel,2,2) + &
                        shape_func_deriv(2,3) * common_coords(noel,3,2) + &
                        shape_func_deriv(2,4) * common_coords(noel,4,2) 

        determinant_Jacobian = Jacobian(1,1) * Jacobian(2,2) - Jacobian(1,2) * Jacobian(2,1) 
    
        inverse_Jacobian(1,1) =  Jacobian(2,2)/determinant_Jacobian
        inverse_Jacobian(1,2) = -Jacobian(1,2)/determinant_Jacobian  
        inverse_Jacobian(2,1) = -Jacobian(2,1)/determinant_Jacobian   
        inverse_Jacobian(2,2) =  Jacobian(1,1)/determinant_Jacobian

        a1 = inverse_Jacobian(1,1) * shape_func_deriv(1,1) + inverse_Jacobian(1,2) * shape_func_deriv(2,1) 
        a2 = inverse_Jacobian(1,1) * shape_func_deriv(1,2) + inverse_Jacobian(1,2) * shape_func_deriv(2,2) 
        a3 = inverse_Jacobian(1,1) * shape_func_deriv(1,3) + inverse_Jacobian(1,2) * shape_func_deriv(2,3) 
        a4 = inverse_Jacobian(1,1) * shape_func_deriv(1,4) + inverse_Jacobian(1,2) * shape_func_deriv(2,4) 
        b1 = inverse_Jacobian(2,1) * shape_func_deriv(1,1) + inverse_Jacobian(2,2) * shape_func_deriv(2,1) 
        b2 = inverse_Jacobian(2,1) * shape_func_deriv(1,2) + inverse_Jacobian(2,2) * shape_func_deriv(2,2) 
        b3 = inverse_Jacobian(2,1) * shape_func_deriv(1,3) + inverse_Jacobian(2,2) * shape_func_deriv(2,3) 
        b4 = inverse_Jacobian(2,1) * shape_func_deriv(1,4) + inverse_Jacobian(2,2) * shape_func_deriv(2,4)  
    
        grad_sigma_hydrostatic(noel, integration_point_id, 1) = a1 * sigma_hydrostatic(noel,1) + &
                                                                a2 * sigma_hydrostatic(noel,2) + &
                                                                a3 * sigma_hydrostatic(noel,3) + &
                                                                a4 * sigma_hydrostatic(noel,4)
        grad_sigma_hydrostatic(noel, integration_point_id, 2) = b1 * sigma_hydrostatic(noel,1) + &
                                                                b2 * sigma_hydrostatic(noel,2) + &
                                                                b3 * sigma_hydrostatic(noel,3) + &
                                                                b4 * sigma_hydrostatic(noel,4)
    end do 
return
end

! 8-node biquadratic displacement, bilinear temperature, hybrid with linear pressure

subroutine calculate_grad_sigma_hydrostatic_CPE8HT(noel)

    use common_block
    
    real(kind=8) shape_func_deriv(2,4), Jacobian(2,2), inverse_Jacobian(2,2)
    
    ! This stores the integration point coordinate in isoparametric spaces
    real(kind=8), parameter :: sqrt_3_over_5 = sqrt(3.0d0 / 5.0d0)
    real(kind=8), parameter :: isoparametric_coord_X(9) = (/ -sqrt_3_over_5, 0.0d0, sqrt_3_over_5, -sqrt_3_over_5, 0.0d0, sqrt_3_over_5, -sqrt_3_over_5, 0.0d0, sqrt_3_over_5 /)
    real(kind=8), parameter :: isoparametric_coord_Y(9) = (/ -sqrt_3_over_5, -sqrt_3_over_5, -sqrt_3_over_5, 0.0d0, 0.0d0, 0.0d0, sqrt_3_over_5, sqrt_3_over_5, sqrt_3_over_5 /)
    
    do integration_point_id = 1, 9

        xi = isoparametric_coord_X(integration_point_id)
        eta = isoparametric_coord_Y(integration_point_id)

        ! Derivatives of the shape functions with respect to xi
        shape_func_deriv(1,1) = (eta**2 - eta)*(0.5*xi - 0.25)
        shape_func_deriv(1,2) = -1.0*xi*(eta**2 - eta)
        shape_func_deriv(1,3) = (eta**2 - eta)*(0.5*xi + 0.25)
        shape_func_deriv(1,4) = (1 - eta**2)*(1.0*xi - 0.5)
        shape_func_deriv(1,5) = -2*xi*(1 - eta**2)
        shape_func_deriv(1,6) = (1 - eta**2)*(1.0*xi + 0.5)
        shape_func_deriv(1,7) = (eta**2 + eta)*(0.5*xi - 0.25)
        shape_func_deriv(1,8) = -1.0*xi*(eta**2 + eta)
        shape_func_deriv(1,9) = (eta**2 + eta)*(0.5*xi + 0.25)

        ! Derivatives of the shape functions with respect to eta
        shape_func_deriv(2,1) = (2*eta - 1)*(0.25*xi**2 - 0.25*xi)
        shape_func_deriv(2,2) = (0.5 - 0.5*xi**2)*(2*eta - 1)
        shape_func_deriv(2,3) = (2*eta - 1)*(0.25*xi**2 + 0.25*xi)
        shape_func_deriv(2,4) = -2*eta*(0.5*xi**2 - 0.5*xi)
        shape_func_deriv(2,5) = -2*eta*(1 - xi**2)
        shape_func_deriv(2,6) = -2*eta*(0.5*xi**2 + 0.5*xi)
        shape_func_deriv(2,7) = (2*eta + 1)*(0.25*xi**2 - 0.25*xi)
        shape_func_deriv(2,8) = (0.5 - 0.5*xi**2)*(2*eta + 1)
        shape_func_deriv(2,9) = (2*eta + 1)*(0.25*xi**2 + 0.25*xi)

        ! partial x partial xi
        Jacobian(1,1) = shape_func_deriv(1,1) * common_coords(noel,1,1) + shape_func_deriv(1,2) * common_coords(noel,2,1) &
                      + shape_func_deriv(1,3) * common_coords(noel,3,1) + shape_func_deriv(1,4) * common_coords(noel,4,1) &
                      + shape_func_deriv(1,5) * common_coords(noel,5,1) + shape_func_deriv(1,6) * common_coords(noel,6,1) &
                      + shape_func_deriv(1,7) * common_coords(noel,7,1) + shape_func_deriv(1,8) * common_coords(noel,8,1) &
                      + shape_func_deriv(1,9) * common_coords(noel,9,1)
        
        ! partial x partial eta
        Jacobian(1,2) = shape_func_deriv(1,1) * common_coords(noel,1,2) + shape_func_deriv(1,2) * common_coords(noel,2,2) &
                      + shape_func_deriv(1,3) * common_coords(noel,3,2) + shape_func_deriv(1,4) * common_coords(noel,4,2) &
                      + shape_func_deriv(1,5) * common_coords(noel,5,2) + shape_func_deriv(1,6) * common_coords(noel,6,2) &
                      + shape_func_deriv(1,7) * common_coords(noel,7,2) + shape_func_deriv(1,8) * common_coords(noel,8,2) &
                      + shape_func_deriv(1,9) * common_coords(noel,9,2)
       
        ! partial y partial xi
        Jacobian(2,1) = shape_func_deriv(2,1) * common_coords(noel,1,1) + shape_func_deriv(2,2) * common_coords(noel,2,1) &
                      + shape_func_deriv(2,3) * common_coords(noel,3,1) + shape_func_deriv(2,4) * common_coords(noel,4,1) &
                      + shape_func_deriv(2,5) * common_coords(noel,5,1) + shape_func_deriv(2,6) * common_coords(noel,6,1) &
                      + shape_func_deriv(2,7) * common_coords(noel,7,1) + shape_func_deriv(2,8) * common_coords(noel,8,1) &
                      + shape_func_deriv(2,9) * common_coords(noel,9,1)
        
        ! partial y partial eta
        Jacobian(2,2) = shape_func_deriv(2,1) * common_coords(noel,1,2) + shape_func_deriv(2,2) * common_coords(noel,2,2) &
                      + shape_func_deriv(2,3) * common_coords(noel,3,2) + shape_func_deriv(2,4) * common_coords(noel,4,2) &
                      + shape_func_deriv(2,5) * common_coords(noel,5,2) + shape_func_deriv(2,6) * common_coords(noel,6,2) &
                      + shape_func_deriv(2,7) * common_coords(noel,7,2) + shape_func_deriv(2,8) * common_coords(noel,8,2) &
                      + shape_func_deriv(2,9) * common_coords(noel,9,2)
    
        determinant_Jacobian = Jacobian(1,1) * Jacobian(2,2) - Jacobian(1,2) * Jacobian(2,1)
        
        ! Inverse Jacobian matrix
        inverse_Jacobian(1,1) =  Jacobian(2,2) / determinant_Jacobian
        inverse_Jacobian(1,2) = -Jacobian(1,2) / determinant_Jacobian
        inverse_Jacobian(2,1) = -Jacobian(2,1) / determinant_Jacobian
        inverse_Jacobian(2,2) =  Jacobian(1,1) / determinant_Jacobian

    
        part_sigma_hydrostatic_part_xi = 0 
        part_sigma_hydrostatic_part_eta = 0

        do id = 1, 9
            part_sigma_hydrostatic_part_xi = part_sigma_hydrostatic_part_xi &
                                           + shape_func_deriv(1,id) * sigma_hydrostatic(noel, id)

            part_sigma_hydrostatic_part_eta = part_sigma_hydrostatic_part_eta &
                                           + shape_func_deriv(2,id) * sigma_hydrostatic(noel, id)
        end do

        grad_sigma_hydrostatic(noel, integration_point_id, 1) = part_sigma_hydrostatic_part_xi * inverse_Jacobian(1,1) + &
                                                  part_sigma_hydrostatic_part_eta * inverse_Jacobian(1,2)
                                    
        grad_sigma_hydrostatic(noel, integration_point_id, 2) = part_sigma_hydrostatic_part_xi * inverse_Jacobian(2,1) + &
                                                  part_sigma_hydrostatic_part_eta * inverse_Jacobian(2,2)
        
    end do

return
end

! ******************************************************************************!
! Important: UMAT and USDFLD is called before UMATHT for each integration point !
! ******************************************************************************!

! This is isotropic von Mises plasticity model

subroutine UMAT(stress,statev,ddsdde,sse,spd,scd, &
    rpl,ddsddt,drplde,drpldt, &
    stran,dstran,time,dtime,temp,dtemp,predef,dpred,cmname, &
    ndi,nshr,ntens,nstatv,props,nprops,coords,drot,pnewdt, &
    celent,dfgrd0,dfgrd1,noel,npt,layer,kspt,kstep,kinc) 

! This subroutine requires us to update stress, ddsdde, statev, and possibly sse, spd, scd

    use common_block
    include 'aba_param.inc'

    character*80 cmname
    dimension stress(ntens),statev(nstatv), &
       ddsdde(ntens,ntens), &
       ddsddt(ntens),drplde(ntens), &
       stran(ntens),dstran(ntens),time(2),predef(1),dpred(1), &
       props(nprops),coords(3),drot(3,3),dfgrd0(3,3),dfgrd1(3,3)

    real(kind=8) :: E, nu, sigma_0, n_hardening, lambda, mu, PEEQ, von_Mises_stress, hydrostatic_stress 
    real(kind=8) :: sigma_flow, sigma_H0, dPEEQ, E_tangent, effective_mu, effective_lambda, effective_hard
    integer :: PSI_Cbar_L_flag, element_flag

    dimension eps_elastic(ntens),eps_plastic(ntens),flow_stress(ntens), &
         old_stress(ntens), old_eps_platic(ntens)

    real(kind=8), parameter :: toler = 1e-12
    real(kind=8), parameter :: newton = 100

    ! UMAT and UMATHT are integration point level, 
    ! which loops over all the integration points over all elements

    ! noel: The current element number
    ! npt: The current integration point number

    ! TIME(1)
    ! Value of step time at the beginning of the current increment or frequency.

    ! TIME(2)
    ! Value of total time at the beginning of the current increment.

    ! material properties

    E = props(1)           ! Young's modulus (200e9 Pa)
    nu = props(2)          ! Poisson's ratio (0.3)
    sigma_0 = props(3)     ! Initial yield strength in the absence of hydrogen (MPa)
    n_hardening = props(4) ! Strain hardening exponent not affected by Hydrogen
                           ! in paper it is 5, but it is already inversed here, so it is 0.2
    PSI_Cbar_L_flag = props(5) ! 1 for PSI_Cbar_L = 1.0, 
                               ! 2 for PSI_Cbar_L = 0.2, 
                               ! 3 for gradually decreasing function of Cbar_L
    element_flag = props(6) ! 1 for CPE8RT, 2 for CPE8HT

    dPEEQ = 0.d0           ! Equivalent plastic strain increment
    PEEQ = statev(13)      ! Equivalent plastic strain

    old_stress = stress
    old_eps_platic = eps_plastic
    
    if (noel == 1 .and. npt == 1 .and. time(1) == 0) then
        print *, 'element_flag: ', element_flag
    end if

    !  Compute the gradient of the hydrostatic stress  
    if (npt == 1 .and. time(1) > 0) then
        if (element_flag == 1) then
            call calculate_grad_sigma_hydrostatic_CPE8RT(noel)
        else if (element_flag == 2) then
            call calculate_grad_sigma_hydrostatic_CPE8HT(noel)
        end if
    end if  

    call rotsig(statev(1), drot, eps_elastic, 2, ndi, nshr)
    call rotsig(statev(ntens+1), drot, eps_plastic, 2, ndi, nshr)
    
    ! Lame's parameters
    mu = E/(2.0d0 * (1.0 + nu))  ! Shear modulus, 7.6923e10 Pa (76.92 GPa)
    lambda = E*nu/((1.0 + nu) * (1.0 - 2.0 * nu)) ! Lame's first constant
    ! = 200e9 * 0.3 / ((1 + 0.3) * (1 - 2 * 0.3)) = 1.15384e11 Pa (115.384 GPa)

    ! initialize as 0
    ddsdde = 0.0 ! Their unit is Pa
    
    do i = 1, ndi
        do j = 1, ndi
            ddsdde(j, i) = lambda
        end do 
        ddsdde(i,i) = lambda + 2.0 * mu
    end do 

    ! Shear contribution
    do i = ndi + 1, ntens
        ddsdde(i,i) = mu
    end do 

    ! Update the elastic stress

    do i =1, ntens
        do j=1, ntens
            stress(j)=stress(j)+ddsdde(j, i)*dstran(i)
        end do
    end do
    
    ! calculate elastic strain

    eps_elastic = eps_elastic + dstran

    
    ! Calculate equivalent von Mises stress
    
    von_Mises_stress = (stress(1) - stress(2))**2 + &
                       (stress(2) - stress(3))**2 + &
                       (stress(3) - stress(1))**2
    
    do i = ndi + 1, ntens
        von_Mises_stress = von_Mises_stress + 6.d0 * stress(i)**2
    end do
    von_Mises_stress = sqrt(von_Mises_stress/2.d0) ! Unit is Pa

    if (PSI_Cbar_L_flag == 1) then
        ! First extreme case
        PSI_Cbar_L = 1.0
    else if (PSI_Cbar_L_flag == 2) then
        ! Second extreme case
        PSI_Cbar_L = 0.2
    else if (PSI_Cbar_L_flag == 3) then
        ! Third case: gradually decreasing function of Cbar_L
        PSI_Cbar_L = statev(17)
    end if
    
    ! From now, sigma_H0 is the yield strength in the presence of hydrogen

    ! Equation 33: We assume that the current yield strength, sigma_flow, 
    ! is a function of the equivalent plastic strain PEEQ
    ! and Cbar_L according to the relationship
    
    sigma_H0 = PSI_Cbar_L * sigma_0 ! Pa

    ! Initial yield strain in the absence of hydrogen
    ! PEEQ_0 = sigma_H0 / E, but we expand it later instead of using it directly
    
    sigma_flow = sigma_H0 * (1.d0 + E * PEEQ/sigma_H0) ** n_hardening 

    ! This is equivalent to 
    ! sigma_flow = sigma_H0 * (1 + PEEQ / PEEQ_0) ** n_hardening
    
    ! Determine if active yielding

    if (von_Mises_stress > (1.d0 + toler) * sigma_flow) then

        hydrostatic_stress = (stress(1) + stress(2) + stress(3))/3.d0
        flow_stress(1:ndi) = (stress(1:ndi) - hydrostatic_stress)/von_Mises_stress
        flow_stress(ndi+1:ntens) = stress(ndi+1:ntens)/von_Mises_stress
        
        ! Newton-Raphson iterative solution
        ! the Newton-Raphson method aims to solve the following nonlinear equation:
        ! von_Mises_stress - (3.d0 * mu * dPEEQ) - sigma_flow = 0
        ! Newton-Raphson is an iterative root-finding method used to solve equations of the form 
        ! f(x)=0. It uses the derivative of the function f(x) to iteratively converge to a root. 
        ! The method requires an initial guess and proceeds as follows:

        ! x_{n+1} = x_n - f(x_n)/f'(x_n)
        ! where x_n is the current guess and x_{n+1} is the next guess.
        ! f(x_n) is the function evaluated at the current estimate.
        ! f'(x_n) is the derivative of the function evaluated at the current estimate.

        ! initial guess for dPEEQ
        dPEEQ = 0.d0
    
        ! Tangent Modulus Derivation
        ! It represents the slope of the stress-strain curve at any point, particularly in the plastic region of the curve.
        ! It is the derivative of the yield stress with respect to the plastic strain (derivative of sigma_flow w.r.t PEEQ)
            
        E_tangent = E * n_hardening * (1.d0 + E * PEEQ/sigma_H0) ** (n_hardening - 1)
        
        do k_newton = 1, newton
            rhs = von_Mises_stress - (3.d0 * mu * dPEEQ) - sigma_flow
            dPEEQ = dPEEQ + rhs / ((3.d0 * mu) + E_tangent)

            sigma_flow = sigma_H0 * (1.d0 + E * (PEEQ + dPEEQ)/sigma_H0) ** n_hardening
            E_tangent = E * n_hardening * (1.d0 + E * (PEEQ + dPEEQ)/sigma_H0) ** (n_hardening-1)

            if (abs(rhs) < toler * sigma_H0) exit
        end do

        if (k_newton == newton) write(7,*)'WARNING: plasticity loop failed'

        ! Update stresses and strains
        stress(1:ndi) = flow_stress(1:ndi) * sigma_flow + hydrostatic_stress

        ! Update the elastic and plastic strains
        eps_plastic(1:ndi) = eps_plastic(1:ndi) + 3.d0/2.d0 * flow_stress(1:ndi) * dPEEQ
        eps_elastic(1:ndi) = eps_elastic(1:ndi) - 3.d0/2.d0 * flow_stress(1:ndi) * dPEEQ
        
        stress(ndi+1:ntens) = flow_stress(ndi+1:ntens) * sigma_flow

        eps_plastic(ndi+1:ntens) = eps_plastic(ndi+1:ntens) + 3.d0 * flow_stress(ndi+1:ntens) * dPEEQ
        eps_elastic(ndi+1:ntens) = eps_elastic(ndi+1:ntens) - 3.d0 * flow_stress(ndi+1:ntens) * dPEEQ

        ! Finally, we update the equivalent plastic strain
        PEEQ = PEEQ + dPEEQ

        ! Calculate the plastic strain energy density
        do i = 1, ntens
            spd = spd+(stress(i) + old_stress(i)) * (eps_plastic(i) - old_eps_platic(i))/2.d0
        end do

        ! Formulate the jacobian (material tangent)   

        ! effective shear modulus
        effective_mu = mu * sigma_flow / von_Mises_stress 

        ! effective Lame's constant
        effective_lambda = (E/(1.d0 - 2.d0 * nu) - 2.d0 * effective_mu)/3.d0 

        ! effective hardening modulus
        effective_hard = 3.d0 * mu * E_tangent/(3.d0 * mu + E_tangent) - 3.d0 * effective_mu 

        do i = 1, ndi
            do j = 1, ndi
                ddsdde(j,i) = effective_lambda
            end do
            ddsdde(i,i) = 2.d0 * effective_mu + effective_lambda
        end do

        do i = ndi + 1, ntens
            ddsdde(i,i) = effective_mu
        end do

        do i = 1, ntens
            do j = 1, ntens
                ddsdde(j,i) = ddsdde(j,i) + effective_hard * flow_stress(j) * flow_stress(i)
            end do
        end do
    endif

    ! Storing the hydrostatic stress and the common_coords of the current integration point   

    sigma_hydrostatic(noel, npt) = (stress(1) + stress(2) + stress(3)) / 3.0    
    common_coords(noel, npt, 1) = coords(1)
    common_coords(noel, npt, 2) = coords(2)

    ! Updating the state dependent variables
    ! Store strains in state variable array (ntens = 4 in our case)
    statev(1:4) = eps_elastic           
    statev(5:8) = eps_plastic       
    statev(9:12) = eps_elastic + eps_plastic   
    statev(13) = PEEQ                    
    statev(14) = dPEEQ                               
    statev(15) = (stress(1) + stress(2) + stress(3))/3.d0 ! Hydrostatic stress
    
    ! You can find the element id from querying in Abaqus
    
    logging = 1 
    
    if (logging == 1) then

        noel_notch_root = 3306
        
        npt_index = 1 

        if (noel == noel_notch_root .and. npt == npt_index) then
            print *, 'time(1): ', time(1)
            print *, 'eps_22: ', eps_elastic(2) + eps_plastic(2)
            print *, 'sigma_22: ', stress(2)
            print *, 'von_Mises_stress: ', von_Mises_stress
            print *, 'sigma_flow: ', sigma_flow
            print *, 'PEEQ: ', PEEQ
            print *, ''
        end if
    end if
    
return
end

!***********************************************************************

subroutine UMATHT(u,dudt,dudg,flux,dfdt,dfdg, &
    statev,temp,dtemp,dtemdx,time,dtime,predef,dpred, &
    cmname,ntgrd,nstatv,props,nprops,coords,pnewdt, &
    noel,npt,layer,kspt,kstep,kinc)

    use common_block
    include 'aba_param.inc'

    character(len=80) :: cmname
    dimension dudg(ntgrd),flux(ntgrd),dfdt(ntgrd), &
      dfdg(ntgrd,ntgrd),statev(nstatv),dtemdx(ntgrd), &
      time(2),predef(1),dpred(1),props(nprops),coords(3)
    
    ! This subroutine requires us to update u, dudt, dudg, flux, dfdt, dfdg, and possibly statev, pnewdt

    ! Define all real for all variables used 
    real(kind=8) :: R, T, VH, DL, Cbar_L, K, C_L, theta_L, theta_trap, temp_result
    real(kind=8) :: rho_d, N_trap, Nbar_trap, dNbar_trap_dPEEQ, Nbar_L
    real(kind=8) :: C_trap, Cbar_trap, Cbar_total

    real(kind=8) :: NA = 6.023D23 ! Avogadro constant (1/mol)
    real(kind=8) :: N_L = 9.24E28 ! Number of solvent atoms (Ni) per unit volume (1/m^3)
    real(kind=8) :: beta = 6.0D0 ! Number of interstitial sites per solvent (Ni) atom
    real(kind=8) :: alpha = 1.0D0 ! Number of interstitial sites per trap site
    
    ! Iron crystallizes in a bcc system with a lattice parameter of 2.861 Angstrom
    ! 1 Angstrom = 1e-10 m
    real(kind=8) :: a_lattice = 2.86D-10 ! Lattice parameter (m)
    real(kind=8) :: WB = -18.0D3 ! Binding energy of hydrogen to dislocations (J/mol)

    real(kind=8) :: xi = 0.2D0 ! xi fitting parameter in Eq 37 

    real(kind=8) :: Cbar_max = 35.0D0 ! (mol/m^3) in Eq 37
    real(kind=8) :: Cbar_min = 0D0 ! (mol/m^3) in Eq 37 
    ! Cbar min is not 15, but 0 as they mention they change Cbar_min near the end of the paper

    real(kind=8) :: gamma = 2.0D16 ! gamma fitting parameter in Eq 35 (1/m^2)
    real(kind=8) :: rho_d0 = 10.0D10 ! Dislocation density for the annealed material (1/m^2)

    R        = props(1) ! Universal gas constant (8.31446 J/(mol K))
    T        = props(2) ! Temperature ( 293 K)
    VH       = props(3) ! Partial molar volume of hydrogen (2e-06 m^3/mol), 
    DL       = props(4) ! Diffusion coefficient for hydrogen in Nickel (3.8e-11 m^2/s)   
     
    ! Since dudt is dCbar_total/dCbar_L, current temperature is the current Cbar_L
    ! and the current dtemp is the current dCbar_L 
    
    ! prev_Cbar_L is temp
    ! dCbar_L is dtemp
    Cbar_L = temp + dtemp ! (mol/m^3)

    ! Equation 15: Arhenius reaction rate constant 
    K = exp( -WB / (R * T)) ! constant, around 1617.679 (dimless)

    ! Finding theta_L based on equations (1) and (5)
    C_L = Cbar_L * NA ! = (mol/m^3) * (1/mol) = (1/m^3)
    theta_L = C_L / (beta * N_L) ! = (1/m^3) / (dimless * 1/m^3) = dimless

    ! Finding theta_trap based on Oriani equilibrium theory
    ! which results in a Fermi-Dirac relation

    ! theta_trap / (1 - theta_trap) = K * theta_L / (1 - theta_L)
    ! However if theta_L << 1  then 
    ! theta_trap / (1 - theta_trap) = K * theta_L (Equation 14)
    
    temp_result = K * theta_L / (1 - theta_L) ! (dimless)
    ! temp_result = K * theta_L
    theta_trap = temp_result / (1 + temp_result) ! = (dimless)
    
    ! Extract PEEQ and calculate rho_d and dNbar_trap_dPEEQ
    PEEQ = statev(13)
    
    if (PEEQ < 0.5) then

        ! Equation 35: Hydrogen trapping in dislocations
        rho_d = rho_d0 + gamma * PEEQ ! = (1/m^2) + (1/m^2) * (dimless) = (1/m^2)
        ! Equation 4: We assume that N_trap is proportional to the dislocation density rho_d
        ! and that the dislocation density is a function of the accumulated plastic strain PEEQ
        N_trap = (sqrt(2.0)/a_lattice) * rho_d  ! = (sqrt2/ m) * (1/m^2) = (1/m^3)
        Nbar_trap = N_trap / NA ! = (1/m^3) / (1/mol) = (mol/m^3)
        
        ! Equation 4: obtain C_trap
        C_trap = alpha * theta_trap * N_trap ! = (dimless) * (1/m^3) = (1/m^3)
        !print *, 'C_trap: ', C_trap

        ! Equation 6: obtain Cbar_trap
        Cbar_trap = C_trap / NA ! = (1/m^3) / (1/mol) = (mol/m^3)
        ! Cbar_trap = (N_trap / NA) * (1 / (1 + (1 - theta_trap)/theta_trap))
        !print *, 'Cbar_trap: ', Cbar_trap

        ! Equation 17: relationship between Nbar and PEEQ
        ! = 1.414 / 2.86 * 10^(-10) * 2 * 10^16 / 6.023 * 10^23 = 164.197
        dNbar_trap_dPEEQ = gamma * (sqrt(2.0) / a_lattice) / NA 
        ! = (1/m^2) * (sqrt2/m) / (1/mol) = (mol/m^3)
    else
        rho_d = 10D16 ! Equation 36
        N_trap = (sqrt(2.0)/a_lattice) * rho_d
        Nbar_trap = N_trap / NA
        C_trap = 1 * theta_trap * N_trap
        Cbar_trap = C_trap / NA
        dNbar_trap_dPEEQ = 0
    end if

    ! Equation 17: obtain Nbar_L (constant)

    Nbar_L = N_L / NA ! = (1/m^3) / (1/mol) = (mol/m^3)

    ! Equation (23) The 1st equation to update dudt (partial_Cbar_total_partial_Cbar_L)

    dudt = 1 + (beta * Nbar_trap * K * Nbar_L) / ((beta * Nbar_L + K * Cbar_L) ** 2.d0)

    ! = 1 + (dimless * mol/m^3 * dimless * mol/m^3) / ((dimless * mol/m^3 + dimless * mol/m^3) ^ 2)
    ! = 1 + (mol^2/m^6) / (mol^2/m^6) = dimless

    ! Equation (23) The 2nd equation to update dudtrap (partial_Cbar_total_partial_Nbar_trap)
    
    dudtrap = (K * Cbar_L) / (K * Cbar_L + beta * Nbar_L)
    ! = (dimless * mol/m^3) / (dimless * mol/m^3 + dimless * mol/m^3) = dimless

    ! Equation (22) The total hydrogen diffusion equation to update u 
    ! Cbar_total_t+1 = Cbar_total_t + partial_Cbar_total_partial_Cbar_L * dCbar_L 
    !                               + partial_Cbar_total_partial_Nbar_trap * dNbar_trap_dPEEQ * dPEEQ
    
    dPEEQ = statev(14) 

    u = u + dudt * dtemp + dudtrap * dNbar_trap_dPEEQ * dPEEQ
    ! = (mol/m^3) + dimless * (mol/m^3) + dimless * (mol/m^3) * (dimless) = (mol/m^3)
    
    ! Since the problem is 2-dimensional, ntgrd = 2
    ! ntgrd: Number of spatial gradients of temperature
    
    dfdg = 0.0

    do i = 1, ntgrd

        ! Equation (10) to update the flux
        grad_Cbar_L_i = dtemdx(i) ! = (mol/m^3) / m = (mol/m^4)
        
        flux(i) = DL * Cbar_L * VH * grad_sigma_hydrostatic(noel, npt, i) / (R * T) - DL * grad_Cbar_L_i 
        ! = ((m^2/s * m^3/mol * mol/m^3) * (N / m^3) / (Nm/(mol K) * K))  - (m^2/s * mol/m^4) = (mol/m^2) / s
        
        ! Assumed to be 0
        dudg(i) = 0

        ! Equation (23) The 3rd equation to update dfdt
        ! part_Jm_part_Cbar_L = (DL * VH) / (R * T) * grad_sigma_hydrostatic(noel, npt, i) ! from common block
        dfdt(i) = (DL * VH * grad_sigma_hydrostatic(noel, npt, i)) / (R * T) 
        ! = (m^2/s * m^3/mol) / (Nm/mol K * K) * (N / m^3) = (m/s)

        ! Equation (23) The 4th equation to update dfdg
        ! part_Jm_part_grad_Cbar_L = - DL
        dfdg(i,i) = -DL ! = - m^2/s

    end do
    
    ! Equation 37: Update monotonically decreasing function of Cbar_L
    ! which will then be passed to UMAT

    PSI_Cbar_L = 1 - (1 - xi) * ((Cbar_L - Cbar_min) / (Cbar_max - Cbar_min))
    Cbar_total = Cbar_L + Cbar_trap
    
    ! Updating the state variables

    statev(16) = rho_d
    statev(17) = PSI_Cbar_L
    statev(18) = Cbar_L
    statev(19) = Cbar_trap
    statev(20) = Cbar_total
    statev(21) = theta_L
    statev(22) = theta_trap

    logging = 1 
    
    if (logging == 1) then

        noel_notch_root = 3306
        npt_index = 1
        
        if (noel == noel_notch_root .and. element_flag == 0 .and. npt == npt_index) then
            print *, 'time(1): ', time(1)
            print *, 'Cbar_L: ', Cbar_L
            print *, 'Cbar_trap: ', Cbar_trap
            print *, 'theta_L: ', theta_L
            print *, 'theta_trap: ', theta_trap
            print *, 'PSI_Cbar_L: ', PSI_Cbar_L
            print *, ''
        end if
    
    end if

    return
    end