!====================================================================
!          Program for mechanical loading and hydrogen diffusion 
!          Mechanical model: standard Hooke's law elasticity 
!                            isotropic Von Mises plasticity
!          Hydrogen diffusion model: Fick's second law
!          Damage model: None. 
!          Deformation affects hydrogen diffusion but not vice versa
!          This is the basis model that users can use to extend to more complex damage model
!          by Nguyen Xuan Binh
!          binh.nguyen@aalto.fi
!          July 2024, Abaqus 2023
!          DO NOT DISTRIBUTE WITHOUT AUTHOR'S PERMISSION
!====================================================================

    ! State variables  
    ! UMAT statv
    ! 1 to 6: sig11, sig22, sig33, sig12, sig13, sig23
    ! 7 to 12: stran11, stran22, stran33, stran12, stran13, stran23
    ! 13 to 18: eelas11, eelas22, eelas33, eelas12, eelas13, eelas23
    ! 19 to 24: eplas11, eplas22, eplas33, eplas12, eplas13, eplas23
    ! "25, AR25_eqplas, AR25_eqplas   ", 
    ! "26, AR26_deqplas, AR26_deqplas   ",
    ! "27, AR27_sig_H, AR27_sig_H   ",
    ! "28, AR28_sig_vonMises, AR28_sig_vonMises   ",
    ! "29, AR29_sig_Tresca, AR29_sig_Tresca   ",
    ! "30, AR30_sig_P1, AR30_sig_P1   ",
    ! "31, AR31_sig_P2, AR31_sig_P2   ",
    ! "32, AR32_sig_P3, AR32_sig_P3   ",
    ! "33, AR33_triax, AR33_triax   ",
    ! "34, AR34_lode, AR34_lode   ",


    ! UMATHT statv
    ! "35, AR35_C_mol, AR35_C_mol   ",
    ! "36, AR36_CL_mol, AR36_CL_mol   ",
    ! "37, AR37_CT_mol, AR37_CT_mol   ",
    ! "38, AR38_CT_dis_mol, AR38_CT_dis_mol   ",
    ! "39, AR39_CT_gb_mol, AR39_CT_gb_mol   ",
    ! "40, AR40_CT_carb_mol, AR30_CT_carb_mol   ",
    ! "41, AR41_C_wtppm, AR41_C_wtppm   ",
    ! "42, AR42_CL_wtppm, AR42_CL_wtppm   ",
    ! "43, AR43_CT_wtppm, AR43_CT_wtppm   ",
    ! "44, AR44_thetaL, AR44_thetaL   ",
    ! "45, AR45_thetaT_dis, AR45_thetaT_dis   ",
    ! "46, AR46_thetaT_gb, AR46_thetaT_gb   ",
    ! "47, AR47_thetaT_carb, AR47_thetaT_carb   ",
    ! "48, AR48_rho_d, AR48_rho_d   ",
    ! "49, AR49_theta_coverage, AR49_theta_coverage   ",
    ! "50, AR50_k_HEHE, AR50_k_HEHE   ",


!***********************************************************************

subroutine pause(seconds)
    ! Pauses the program execution for the specified number of seconds.
    integer, intent(in) :: seconds
    integer :: start_time, end_time, rate

    ! Get the system clock rate (ticks per second)
    call system_clock(count_rate = rate)

    ! Get the current time in clock ticks
    call system_clock(start_time)

    ! Loop until the required time has passed
    do
        call system_clock(end_time)
        if ((end_time - start_time) >= seconds * rate) exit
    end do
return
end

! To ensure that the code are highly optimized, we should avoid
! using division and exponentiation as much as possible
! Specifically, replace division by multiplying with its constant inverse
! Replace squares by multiplication with itself

module precision
    use iso_fortran_env
    integer, parameter :: dp = real64
end module precision

!***********************************************************************


module common_block
    use precision
    implicit none

    ! THESE TWO VALUES ARE HARD-CODED
    ! YOU MUST CHANGE IT TO THE ACTUAL NUMBER OF ELEMENTS AND NODES IN .INP FILE

    ! CP1000 mesh
    ! **************************************
    ! GEOMETRY | total_elems | total_nodes !
    ! CHD2     | 121260      | 136770      !
    ! CHD4     | 121695      | 138132      !
    ! NDBR2p5  | 121375      | 136548      !
    ! NDBR6    | 121695      | 136853      !
    ! NDBR15   | 121285      | 136786      !
    ! NDBR40   | 121705      | 136935      !
    ! SH115    | 121510      | 137643      !
    ! **************************************

    integer, parameter :: total_elems = 121260 ! Storing the actual number of elements
    integer, parameter :: total_nodes = 136770 ! Storing the actual number of nodes

    real(kind=dp), parameter :: molar_mass_H = 1.00784d0 ! g/mol
    real(kind=dp), parameter :: molar_mass_Fe = 55.845d0 ! g/mol
    real(kind=dp), parameter :: ratio_molar_mass_Fe_H = 55.415d0
    real(kind=dp), parameter :: density_metal = 7900.0d0 ! kg/m^3
    ! CL_wtppm = (CL_mol * molar_mass_H) / (density_metal * 1.d-03)
    real(kind=dp), parameter :: conversion_mol_to_wtppm = 0.127574683544d0 ! wtppm
    ! CL_molfrac = (CL_wtppm * 1.d-6) * (ratio_molar_mass_Fe_H)
    real(kind=dp), parameter :: conversion_wtppm_to_molfrac = 55.4105d-6 ! molfrac 
    ! Inverse of conversion_wtppm_to_mol
    real(kind=dp), parameter :: conversion_wtppm_to_mol = 7.838545801d0 ! mol
    real(kind=dp), parameter :: conversion_mol_to_molfrac = 7.068977d-6 ! molfrac
    real(kind=dp), parameter :: pi = 3.14159d0 ! dimless
    real(kind=dp), parameter :: inv_pi = 1.0d0 / 3.14159d0 ! dimless
    real(kind=dp), parameter :: half = 1.0d0 / 2.0d0 ! dimless
    real(kind=dp), parameter :: third = 1.0d0 / 3.0d0 ! dimless
    real(kind=dp), parameter :: fourth = 1.0d0 / 4.0d0 ! dimless
    real(kind=dp), parameter :: sixth = 1.0d0 / 6.0d0 ! dimless
    real(kind=dp), parameter :: three_half = 3.0d0 / 2.0d0 ! dimless
    real(kind=dp), parameter :: sqrt_three_half = dsqrt(3.0d0 / 2.0d0) ! dimless
    real(kind=dp), parameter :: two_third = 2.0d0 / 3.0d0 ! dimless
    real(kind=dp), parameter :: sqrt_two_third = dsqrt(2.0d0 / 3.0d0) ! dimless
    real(kind=dp), parameter :: nine_half = 9.0d0 / 2.0d0 ! dimless

    integer, parameter :: before_flow_props_idx = 8 ! Index of the first flow curve data in props in UMAT
    
    ! Index of statev in UMAT
    integer, parameter :: sig_start_idx = 1 ! Starting index of the stress component in statev
    integer, parameter :: sig_end_idx = 6 ! Ending index of the strain component in statev
    integer, parameter :: stran_start_idx = 7 ! Starting index of the total strain component in statev
    integer, parameter :: stran_end_idx = 12 ! Ending index of the total strain component in statev
    integer, parameter :: eelas_start_idx = 13 ! Starting index of the elastic strain component in statev
    integer, parameter :: eelas_end_idx = 18 ! Ending index of the elastic strain component in statev
    integer, parameter :: eplas_start_idx = 19 ! Starting index of the plastic strain component in statev
    integer, parameter :: eplas_end_idx = 24 ! Ending index of the plastic strain component in statev
    integer, parameter :: eqplas_idx = 25 ! Index of the equivalent plastic strain in statev
    integer, parameter :: deqplas_idx = 26 ! Index of the increment of the equivalent plastic strain in statev
    integer, parameter :: sig_H_idx = 27 ! Index of the hydrogen concentration in statev
    integer, parameter :: sig_vonMises_idx = 28 ! Index of the equivalent von Mises stress in statev
    integer, parameter :: sig_Tresca_idx = 29 ! Index of the equivalent Tresca stress in statev
    integer, parameter :: sig_P1_idx = 30 ! Index of the first principal stress in statev
    integer, parameter :: sig_P2_idx = 31 ! Index of the second principal stress in statev
    integer, parameter :: sig_P3_idx = 32 ! Index of the third principal stress in statev
    integer, parameter :: triax_idx = 33 ! Index of the triaxiality in statev
    integer, parameter :: lode_idx = 34 ! Index of the Lode parameter in statev
    
    ! Index of statev in UMATHT
    integer, parameter :: C_mol_idx = 35 ! Index of the hydrogen concentration in mol in statev
    integer, parameter :: CL_mol_idx = 36 ! Index of the hydrogen concentration in lattice in mol in statev
    integer, parameter :: CT_mol_idx = 37 ! Index of the hydrogen concentration in trap in mol in statev
    integer, parameter :: CT_dis_mol_idx = 38 ! Index of the hydrogen concentration in dislocation trap in mol in statev
    integer, parameter :: CT_gb_mol_idx = 39 ! Index of the hydrogen concentration in grain boundary trap in mol in statev
    integer, parameter :: CT_carb_mol_idx = 40 ! Index of the hydrogen concentration in carbide trap in mol in statev
    integer, parameter :: C_wtppm_idx = 41 ! Index of the hydrogen concentration in wtppm in statev
    integer, parameter :: CL_wtppm_idx = 42 ! Index of the hydrogen concentration in lattice in wtppm in statev
    integer, parameter :: CT_wtppm_idx = 43 ! Index of the hydrogen concentration in trap in wtppm in statev
    integer, parameter :: thetaL_idx = 44 ! Index of the lattice hydrogen concentration in statev
    integer, parameter :: thetaT_dis_idx = 45 ! Index of the dislocation trap hydrogen concentration in statev
    integer, parameter :: thetaT_gb_idx = 46 ! Index of the grain boundary trap hydrogen concentration in statev
    integer, parameter :: thetaT_carb_idx = 47 ! Index of the carbide trap hydrogen concentration in statev
    integer, parameter :: rho_d_idx = 48 ! Index of the dislocation density in statev
    integer, parameter :: theta_coverage_idx = 49 ! Index of the hydrogen surface coverage in statev
    integer, parameter :: k_HEHE_idx = 50 ! Index of the factor decreasing cohesive strength (HEDE)
    
    integer, parameter :: ndim = 3 ! Number of spatial dimensions
    integer, parameter :: ninpt = 8 ! Number of integration points (IP) in the element
    integer, parameter :: nnode = 8 ! Number of nodes in the element
    integer, parameter :: nmax_elems = 20 ! Maximum number of elements that a node can have
    integer, parameter :: nsvars = 50 ! Number of state dependent variables in UMAT and UMATHT

    ! Technically, nmax_elems is 10 for C3D8 mesh
    ! However we are uncertain what could be maximum number of elements that a node can have
    ! So we just set it to large number to be safe
    
    ! First dim: number of nodes on the mesh
    ! Second dim: maximum number of elements that contains the node in the first dim
    ! In meshing algorithm in FEA softwares lie Abaqus that used hexahedral, nmax_elems is proven to be 8
    ! Third dim: first value tells the element ID that contains this node
    !            second value tells the ith position of the node in this element ID
    ! When this node does not have nmax_elems containing it, all others are padded with 0

    ! Example: Lets say element of ID 10, 20, 30, 40 contains node of ID 7
    !          In element 10, node 7 is at position 1
    !          In element 20, node 7 is at position 6
    !          In element 30, node 7 is at position 3
    !          In element 40, node 7 is at position 5

    ! Then nodes_to_elems_matrix(7, 1, 1) = 10
    !      nodes_to_elems_matrix(7, 1, 2) = 1
    !      nodes_to_elems_matrix(7, 2, 1) = 20
    !      nodes_to_elems_matrix(7, 2, 2) = 6
    !      nodes_to_elems_matrix(7, 3, 1) = 30
    !      nodes_to_elems_matrix(7, 3, 2) = 3
    !      nodes_to_elems_matrix(7, 4, 1) = 40
    !      nodes_to_elems_matrix(7, 4, 2) = 5
    !      nodes_to_elems_matrix(7, 5:8, 1:2) = 0
    
    ! This global array will store all SDVs for all elements in the mesh
    ! Naturally, it will include hydrostatic stress value at sig_H_idx

    real(kind=dp) :: kuser_vars(total_elems, nsvars, ninpt)

    integer :: nodes_to_elems_matrix(total_nodes, nmax_elems, 2) !

    integer :: num_elems_of_nodes_matrix(total_nodes) ! Number of elements that contain the node
    
    ! First dim: number of elements in the mesh
    ! Second dim: number of nodes in the element
    ! Since all elements must have 8 nodes, it does not need to be padded
    ! The nodes are also in their correct order as well from knode to 1 to 8
    
    integer :: elems_to_nodes_matrix(total_elems, nnode)

    ! Shape function for all IP in the local isoparametric coordinate
    ! Given values at nodal points, this matrix will give the values at IP (interpolation)
    ! This is helpful for calculating values at IPs from the DoF values defined at nodal points 
    real(kind=dp) :: all_N_inpt_to_local_knode(nnode, ninpt)
    
    ! Shape function for all nodes in the local isoparametric coordinate
    ! Given values at IPs, this matrix will give the values at nodal points (extrapolation)
    ! This is helpful for calculating values at nodal points from values defined at IPs
    ! Such as hydrostatic stress gradient or for visualization
    real(kind=dp) :: all_N_node_to_local_kinpt(ninpt, nnode)
    
    ! Gradient of all_N_node_to_local_kinpt
    real(kind=dp) :: all_N_grad_node_to_local_kinpt(ninpt, ndim, nnode)
    
    ! This matrix keeps track of extrapolated sig_H from IP onto nodal points for all elements
    real(kind=dp) :: sig_H_all_elems_at_nodes(total_elems, nnode)

    ! This matrix keeps track of the gradient of sig_H from IP onto nodal points for all elements
    real(kind=dp) :: sig_H_grad_all_elems_at_inpts(total_elems, ninpt, ndim)
    ! Finally, this matrix keeps track of the average sig_H at each node from the elements
    ! that contain the node. The average is weighted based on change of volume 
    ! (determinant of Jacobian matrix)

    real(kind=dp) :: sig_H_at_nodes(total_nodes) ! (total_nodes)

    ! This stores the IP coordinates at previous time step
    real(kind=dp) :: coords_all_inpts(total_elems, ninpt, ndim)

    ! This stores the nodal coordinates at previous time step
    real(kind=dp) :: coords_all_nodes(total_nodes, ndim)

    ! This stores the determinant of Jacobian matrix of all nodes based on coordinates of the previous time step
    real(kind=dp) :: djac_all_elems_at_nodes(total_elems, nnode)


    save
    ! The save command is very important. 
    ! It allows the values to be stored and shared between subroutines 
    ! without resetting them to zero every time the subroutine is called

end module


! C3D8 element

!*****************************************************************
!  8-node     8---------------7
!  brick     /|              /|       zeta (positive)
!           / |  x 7   x 8  / |       
!          5---------------6  |       |     eta (positive)
!          |  | x 5   x 6  |  |       |   /
!          |  |    x 0     |  |       |  /
!          |  4------------|--3       | /
!          | /   x 3   x 4 | /        |/
!          |/   x 1   x 2  |/         O--------- xi (positive)
!          1---------------2           origin at cube center
!          
!         Outer number is nodal points
!         Inner number marked with x is integration points
!
!*****************************************************************

include "helper_functions.f90"

module iso_module

    use precision
    use common_block
    real(kind=dp), parameter :: coord_inter = 1.0d0
    real(kind=dp), parameter :: int_inter = 1.0d0 / sqrt(3.0d0)
    real(kind=dp), parameter :: coord_extra = sqrt(3.0d0)
    real(kind=dp), parameter :: int_extra = 1.0d0
    
    ! weight is the IP weight for their shape function contribution
    real(kind=dp), parameter :: weight(ninpt) = (/1.d0, 1.d0, 1.d0, 1.d0, 1.d0, 1.d0, 1.d0, 1.d0/)
    
    ! Interpolating coordinates (nodal to int)
    
    ! Isoparametric coordinates for nodal points in hexahedral 3D element
    real(kind=dp), parameter :: xi_nodal_inter(nnode)   = (/ -coord_inter,  coord_inter,  coord_inter, -coord_inter, &
                                                             -coord_inter,  coord_inter,  coord_inter, -coord_inter /)
    real(kind=dp), parameter :: eta_nodal_inter(nnode)  = (/ -coord_inter, -coord_inter,  coord_inter,  coord_inter, &
                                                             -coord_inter, -coord_inter,  coord_inter,  coord_inter /)
    real(kind=dp), parameter :: zeta_nodal_inter(nnode) = (/ -coord_inter, -coord_inter, -coord_inter, -coord_inter, &
                                                              coord_inter,  coord_inter,  coord_inter,  coord_inter /)

    ! Isoparametric coordinates for integration points in hexahedral 3D element
    real(kind=dp), parameter :: xi_int_inter(ninpt)     = (/ -int_inter,  int_inter, -int_inter,  int_inter, &
                                                             -int_inter,  int_inter, -int_inter,  int_inter /)
    real(kind=dp), parameter :: eta_int_inter(ninpt)    = (/ -int_inter, -int_inter,  int_inter,  int_inter, &
                                                             -int_inter, -int_inter,  int_inter,  int_inter /)
    real(kind=dp), parameter :: zeta_int_inter(ninpt)   = (/ -int_inter, -int_inter, -int_inter, -int_inter, &
                                                              int_inter,  int_inter,  int_inter,  int_inter /)

    ! Extrapolating coordinates (int to nodal)

    ! Isoparametric coordinates for nodal points in hexahedral 3D element
    real(kind=dp), parameter :: xi_nodal_extra(nnode)   = (/ -coord_extra,  coord_extra,  coord_extra, -coord_extra, &
                                                             -coord_extra,  coord_extra,  coord_extra, -coord_extra /)
    real(kind=dp), parameter :: eta_nodal_extra(nnode)  = (/ -coord_extra, -coord_extra,  coord_extra,  coord_extra, &
                                                             -coord_extra, -coord_extra,  coord_extra,  coord_extra /)
    real(kind=dp), parameter :: zeta_nodal_extra(nnode) = (/ -coord_extra, -coord_extra, -coord_extra, -coord_extra, &
                                                              coord_extra,  coord_extra,  coord_extra,  coord_extra /)

    ! Isoparametric coordinates for integration points in hexahedral 3D element
    real(kind=dp), parameter :: xi_int_extra(ninpt)   = (/ -int_extra,  int_extra, -int_extra,  int_extra, &
                                                           -int_extra,  int_extra, -int_extra,  int_extra /)
    real(kind=dp), parameter :: eta_int_extra(ninpt)  = (/ -int_extra, -int_extra,  int_extra,  int_extra, &
                                                           -int_extra, -int_extra,  int_extra,  int_extra /)
    real(kind=dp), parameter :: zeta_int_extra(ninpt) = (/ -int_extra, -int_extra, -int_extra, -int_extra, &
                                                            int_extra,  int_extra,  int_extra,  int_extra /)

end module iso_module

subroutine calc_N_inpt_to_local_coords(xi_coord, eta_coord, zeta_coord, N_inpt_to_local_coords)
    ! Calculate the shape function at the nodal points
    ! xi_coord, eta_coord, zeta_coord: Isoparametric coordinates of the nodal points

    use precision
    use common_block
    real(kind=dp), dimension(ninpt) :: N_inpt_to_local_coords
    real(kind=dp) :: xi_coord, eta_coord, zeta_coord

    !   shape functions
    N_inpt_to_local_coords(1) = 0.125d0 * (1.d0 - xi_coord) * (1.d0 - eta_coord) * (1.d0 - zeta_coord)
    N_inpt_to_local_coords(2) = 0.125d0 * (1.d0 + xi_coord) * (1.d0 - eta_coord) * (1.d0 - zeta_coord)
    N_inpt_to_local_coords(3) = 0.125d0 * (1.d0 - xi_coord) * (1.d0 + eta_coord) * (1.d0 - zeta_coord)
    N_inpt_to_local_coords(4) = 0.125d0 * (1.d0 + xi_coord) * (1.d0 + eta_coord) * (1.d0 - zeta_coord)
    N_inpt_to_local_coords(5) = 0.125d0 * (1.d0 - xi_coord) * (1.d0 - eta_coord) * (1.d0 + zeta_coord)
    N_inpt_to_local_coords(6) = 0.125d0 * (1.d0 + xi_coord) * (1.d0 - eta_coord) * (1.d0 + zeta_coord)
    N_inpt_to_local_coords(7) = 0.125d0 * (1.d0 - xi_coord) * (1.d0 + eta_coord) * (1.d0 + zeta_coord)
    N_inpt_to_local_coords(8) = 0.125d0 * (1.d0 + xi_coord) * (1.d0 + eta_coord) * (1.d0 + zeta_coord)

return
end

subroutine calc_N_node_to_local_coords(xi_coord, eta_coord, zeta_coord, &
                                       N_node_to_local_coords)
    ! Calculate the shape function at the integration points
    ! xi_coord, eta_coord, zeta_coord: Isoparametric coordinates of the integration points

    use precision
    use common_block    
    real(kind=dp), dimension(nnode) :: N_node_to_local_coords 
    real(kind=dp) :: xi_coord, eta_coord, zeta_coord

    !   shape functions
    N_node_to_local_coords(1)=0.125d0 * (1.d0 - xi_coord) * (1.d0 - eta_coord) * (1.d0 - zeta_coord)
    N_node_to_local_coords(2)=0.125d0 * (1.d0 + xi_coord) * (1.d0 - eta_coord) * (1.d0 - zeta_coord)
    N_node_to_local_coords(3)=0.125d0 * (1.d0 + xi_coord) * (1.d0 + eta_coord) * (1.d0 - zeta_coord)
    N_node_to_local_coords(4)=0.125d0 * (1.d0 - xi_coord) * (1.d0 + eta_coord) * (1.d0 - zeta_coord)
    N_node_to_local_coords(5)=0.125d0 * (1.d0 - xi_coord) * (1.d0 - eta_coord) * (1.d0 + zeta_coord)
    N_node_to_local_coords(6)=0.125d0 * (1.d0 + xi_coord) * (1.d0 - eta_coord) * (1.d0 + zeta_coord)
    N_node_to_local_coords(7)=0.125d0 * (1.d0 + xi_coord) * (1.d0 + eta_coord) * (1.d0 + zeta_coord)
    N_node_to_local_coords(8)=0.125d0 * (1.d0 - xi_coord) * (1.d0 + eta_coord) * (1.d0 + zeta_coord)

return
end

subroutine calc_N_grad_node_to_local_coords(xi_coord, eta_coord, zeta_coord, &
                                            N_grad_node_to_local_coords)
    !
    ! Calculate the shape function derivative at the integration points
    ! Basically derivatives of calc_N_node_to_local_coords
    ! N_grad_node_to_local_coords: (ndim, nnode)
    ! xi_coord, eta_coord, zeta_coord: Isoparametric coordinates of the integration points

    use precision
    use common_block
    real(kind=dp), dimension(ndim, nnode) :: N_grad_node_to_local_coords
    real(kind=dp) :: xi_coord, eta_coord, zeta_coord

    !   derivative d(Ni)/d(xi_coord)
    N_grad_node_to_local_coords(1, 1) = -0.125d0 * (1.d0 - eta_coord) * (1.d0 - zeta_coord)
    N_grad_node_to_local_coords(1, 2) =  0.125d0 * (1.d0 - eta_coord) * (1.d0 - zeta_coord)
    N_grad_node_to_local_coords(1, 3) =  0.125d0 * (1.d0 + eta_coord) * (1.d0 - zeta_coord)
    N_grad_node_to_local_coords(1, 4) = -0.125d0 * (1.d0 + eta_coord) * (1.d0 - zeta_coord)
    N_grad_node_to_local_coords(1, 5) = -0.125d0 * (1.d0 - eta_coord) * (1.d0 + zeta_coord)
    N_grad_node_to_local_coords(1, 6) =  0.125d0 * (1.d0 - eta_coord) * (1.d0 + zeta_coord)
    N_grad_node_to_local_coords(1, 7) =  0.125d0 * (1.d0 + eta_coord) * (1.d0 + zeta_coord)
    N_grad_node_to_local_coords(1, 8) = -0.125d0 * (1.d0 + eta_coord) * (1.d0 + zeta_coord)

    !     derivative d(Ni)/d(eta_coord)
    N_grad_node_to_local_coords(2, 1) = -0.125d0 * (1.d0 - xi_coord) * (1.d0 - zeta_coord)
    N_grad_node_to_local_coords(2, 2) = -0.125d0 * (1.d0 + xi_coord) * (1.d0 - zeta_coord)
    N_grad_node_to_local_coords(2, 3) =  0.125d0 * (1.d0 + xi_coord) * (1.d0 - zeta_coord)
    N_grad_node_to_local_coords(2, 4) =  0.125d0 * (1.d0 - xi_coord) * (1.d0 - zeta_coord)
    N_grad_node_to_local_coords(2, 5) = -0.125d0 * (1.d0 - xi_coord) * (1.d0 + zeta_coord)
    N_grad_node_to_local_coords(2, 6) = -0.125d0 * (1.d0 + xi_coord) * (1.d0 + zeta_coord)
    N_grad_node_to_local_coords(2, 7) =  0.125d0 * (1.d0 + xi_coord) * (1.d0 + zeta_coord)
    N_grad_node_to_local_coords(2, 8) =  0.125d0 * (1.d0 - xi_coord) * (1.d0 + zeta_coord)

    !     derivative d(Ni)/d(zeta_coord)
    N_grad_node_to_local_coords(3, 1) = -0.125d0 * (1.d0 - xi_coord) * (1.d0 - eta_coord)
    N_grad_node_to_local_coords(3, 2) = -0.125d0 * (1.d0 + xi_coord) * (1.d0 - eta_coord)
    N_grad_node_to_local_coords(3, 3) = -0.125d0 * (1.d0 + xi_coord) * (1.d0 + eta_coord)
    N_grad_node_to_local_coords(3, 4) = -0.125d0 * (1.d0 - xi_coord) * (1.d0 + eta_coord)
    N_grad_node_to_local_coords(3, 5) =  0.125d0 * (1.d0 - xi_coord) * (1.d0 - eta_coord)
    N_grad_node_to_local_coords(3, 6) =  0.125d0 * (1.d0 + xi_coord) * (1.d0 - eta_coord)
    N_grad_node_to_local_coords(3, 7) =  0.125d0 * (1.d0 + xi_coord) * (1.d0 + eta_coord)
    N_grad_node_to_local_coords(3, 8) =  0.125d0 * (1.d0 - xi_coord) * (1.d0 + eta_coord)

return
end


!***********************************************************************

subroutine UEXTERNALDB(lop,lrestart,time,dtime,kstep,kinc)
    use precision
    use common_block
    use iso_module
    use iso_c_binding

    include 'aba_param.inc' 

    dimension time(2)
    
    integer :: iostat, element_ID, node_id, current_element_idx
    integer, dimension(nnode + 1) :: values

    character(len=256) :: line, outdir, aelread, aelwrite, andread, andwrite, anelwrite
    
    real(kind=dp) :: xi_node, eta_node, zeta_node, xi_int, eta_int, zeta_int, &
                     xi_int_corner, eta_int_corner, zeta_int_corner
    real(kind=dp), dimension(ninpt) :: N_inpt_to_local_knode
    real(kind=dp), dimension(nnode) :: N_node_to_local_kinpt
    real(kind=dp), dimension(ndim, nnode) :: N_grad_node_to_local_kinpt
    real(kind=dp), dimension(ndim, nnode) :: N_grad_node_to_local_knode
    
    real(kind=dp), dimension(ndim,ndim) :: xjac
    real(kind=dp), dimension(nnode) :: x_nodes_current_elem, y_nodes_current_elem, z_nodes_current_elem
    real(kind=dp) :: djac

    ! lop = 0 indicates that UEXTERNALDB is called at the start of the analysis
    
    if (lop == 0) then 

        ! Ensuring that only one thread is accessing the shared memory
        
        do klock = 1, 40
            call mutexInit(klock)
        end do
        
        call mutexLock(1)

        ! ========================================================
        ! Initialize all common matrixes as zeros
        ! ========================================================

        kuser_vars = 0.0d0
        sig_H_all_elems_at_nodes = 0.0d0
        sig_H_at_nodes = 0.0d0
        sig_H_grad_all_elems = 0.0d0
        coords_all_inpts = 0.0d0
        coords_all_nodes = 0.0d0
        djac_all_elems_at_nodes = 0.0d0

        ! ========================================================
        ! BUILDING THE CONNECTIVITY MATRIX
        ! elems_to_nodes_matrix: (total_elems, nnode)
        ! nodes_to_elems_matrix: (total_nodes, nmax_elems, 2)
        ! ========================================================

        ! BEWARE: Make sure that the mesh txt file must not have any empty lines at the end

        call getoutdir(outdir, lenoutdir)
        aelread = trim(outdir) // '/processing_input/original_mesh.txt'
        
        open(unit=10, file=aelread, status="old", action="read")
        
        ! Read the file line by line and populate the nodes_to_elems_matrix
        do line_idx = 1, total_elems
            read(10, '(A)', iostat=iostat) line

            ! Convert the line into integers
            read(line, *) values  ! Read the 9 values (element ID and 8 nodes)

            ! values(1) is element ID of C3D8 element
            ! values(2:nnode+1) are the node ID of the C3D8 element

            ! Populate the elems_to_nodes_matrix

            do knode = 2, nnode + 1 ! Looping over the 8 nodes
                element_ID = values(1)
                node_ID = values(knode)
                elems_to_nodes_matrix(element_ID, knode-1) = node_ID
            end do
        end do

        ! Close the file
        close(10)

        ! ! Optional: print part of the matrix to verify
        ! do i = 10000, 10050  ! Print first 10 nodes for checking
        !     print *, 'Element', i, ': ', elems_to_nodes_matrix(i, 1:8)
        ! end do

        ! call pause(180)

        open(unit=10, file=aelread, status="old", action="read")

        ! Initialize nodes_to_elems_matrix with 0 to indicate unused slots
        nodes_to_elems_matrix = 0
        num_elems_of_nodes_matrix = 0

        ! Read the file line by line and populate the nodes_to_elems_matrix
        do line_idx = 1, total_elems
            read(10, '(A)', iostat=iostat) line

            ! Convert the line into integers
            read(line, *) values  ! Read the 9 values (element ID and 8 nodes)

            ! values(1) is element ID of C3D8 element
            ! values(2:nnode+1) are the node ID of the C3D8 element
            
            ! Populating the nodes_to_elems_matrix

            element_ID = values(1)
                
            ! Loop over each node in the current element
            do knode = 1, nnode  ! nnode is the number of nodes per element
                node_ID = elems_to_nodes_matrix(element_ID, knode)  ! Get the global node number

                num_elems_of_nodes_matrix(node_ID) = num_elems_of_nodes_matrix(node_ID) + 1
                kelem = num_elems_of_nodes_matrix(node_ID)  ! Get the current index for element containing this node

                ! Store the element ID and local node number in nodes_to_elems_matrix
                nodes_to_elems_matrix(node_ID, kelem, 1) = element_ID  ! Store the element ID
                nodes_to_elems_matrix(node_ID, kelem, 2) = knode  ! Store the local node number
            end do  ! End loop over nodes in the element

        end do

        ! Close the file
        close(10)

        ! ! Optional: print part of the matrix to verify
        ! do i = 10000, 10050  ! Print first 10 nodes for checking
        !     print *, 'Node', i, ': ', nodes_to_elems_matrix(i, 1:8, 1:2)
        !     print *, 'Number of elements: ', num_elems_of_nodes_matrix(i)
        ! end do

        ! call pause(180)


        ! ========================================================
        ! CALCULATING SHAPE FUNCTIONS AND THE SHAPE FUNCTION GRADIENT
        ! W.R.T LOCAL COORDINATES. THESE VALUES NEVER CHANGE DURING THE ANALYSIS
        ! all_N_inpt_to_local_knode: (nnode, ninpt)
        ! all_N_node_to_local_kinpt: (ninpt, nnode)
        ! all_N_grad_node_to_local_kinpt: (ninpt, ndim, nnode)
        ! ========================================================

        do knode = 1, nnode
            xi_node = xi_nodal_extra(knode)
            eta_node = eta_nodal_extra(knode)
            zeta_node = zeta_nodal_extra(knode)

            call calc_N_inpt_to_local_coords(xi_node, eta_node, zeta_node, N_inpt_to_local_knode)
            all_N_inpt_to_local_knode(knode, 1:ninpt) = N_inpt_to_local_knode
            !print *, 'N_inpt_to_local_knode: ', N_inpt_to_local_knode
        end do      

        ! print *, 'all_N_inpt_to_local_knode'
        ! print *, all_N_inpt_to_local_knode(1:8, 1:8)
        

        do kinpt = 1, ninpt

            xi_int = xi_int_inter(kinpt)
            eta_int = eta_int_inter(kinpt)
            zeta_int = zeta_int_inter(kinpt)

            xi_int_corner = xi_int_extra(kinpt)
            eta_int_corner = eta_int_extra(kinpt)
            zeta_int_corner = zeta_int_extra(kinpt)

            call calc_N_node_to_local_coords(xi_int, eta_int, zeta_int, N_node_to_local_kinpt)
            all_N_node_to_local_kinpt(kinpt, 1:nnode) = N_node_to_local_kinpt(1:nnode)

            call calc_N_grad_node_to_local_coords(xi_int, eta_int, zeta_int, N_grad_node_to_local_kinpt)
            all_N_grad_node_to_local_kinpt(kinpt,1:ndim,1:nnode) = N_grad_node_to_local_kinpt(1:ndim,1:nnode)
            ! print *, 'N_node_to_local_kinpt: '
            ! print *, N_node_to_local_kinpt
            ! print *, 'N_grad_node_to_local_kinpt: '
            ! print *, N_grad_node_to_local_kinpt
        end do

        ! call pause(180)

        ! Releasing the lock
        call mutexUnlock(1)

        ! print *, 'kuser_vars'
        ! print *, kuser_vars(1:5, sig_H_idx, 1:8)

    end if

    ! lop = 2 indicates that UEXTERNALDB is called at the end of the current analysis increment
    
    if (lop == 2) then 
        ! call mutexInit(2)
        call mutexLock(2)
        ! print *, 'kuser_vars'
        ! print *, kuser_vars(1:5, sig_H_idx, 1:ninpt)
        ! call pause(180)
        
        ! ========================================================
        ! Populating sig_H_all_elems_at_nodes(total_elems, nnode)
        ! by using N_inpt_to_local_knode to extrapolate from IPs to nodes
        ! ========================================================

        !print *, 'Populating sig_H_all_elems_at_nodes'

        do element_ID = 1, total_elems
            ! Loop over each node in the current element
            do knode = 1, nnode
                ! Initialize hydrostatic stress for the current node to zero
                sig_H_all_elems_at_nodes(element_ID, knode) = 0.0d0

                ! Compute the hydrostatic stress at the nodal point by summing the products
                do kinpt = 1, ninpt
                    sig_H_all_elems_at_nodes(element_ID, knode) = sig_H_all_elems_at_nodes(element_ID, knode) &
                        + all_N_inpt_to_local_knode(knode, kinpt) * kuser_vars(element_ID, sig_H_idx, kinpt)
                end do
            end do
        end do


        ! print *, 'all_N_inpt_to_local_knode'
        ! print *, all_N_inpt_to_local_knode(1:8, 1:8)
        ! print *, 'kuser_vars'
        ! print *, kuser_vars(1:5, sig_H_idx, 1:8)
        ! print *, 'sig_H_all_elems_at_nodes'
        ! print *, sig_H_all_elems_at_nodes(1:5, 1:8)

        !print *, 'Populating sig_H_all_elems_at_nodes done'

        ! ========================================================
        ! Populating djac_all_elems_at_nodes(total_elems, nnode)
        ! ========================================================

        !print *, 'Populating djac_all_elems_at_nodes'

        do element_ID = 1, total_elems
            
            do knode = 1,nnode
                !   Retrieve the node_ID of the current node in this element
                node_ID = elems_to_nodes_matrix(element_ID,knode)
                x_nodes_current_elem(knode) = coords_all_nodes(node_ID,1)
                y_nodes_current_elem(knode) = coords_all_nodes(node_ID,2)
                z_nodes_current_elem(knode) = coords_all_nodes(node_ID,3)
            end do

            ! Loop over each node in the current element
            do knode = 1, nnode

                ! Natural coordinates at the current node
                xi_node = xi_nodal_inter(knode)
                eta_node = eta_nodal_inter(knode)
                zeta_node = zeta_nodal_inter(knode)

                ! Calculate derivatives of shape function of nodal points 
                ! with respect to natural coordinates of themselves
                call calc_N_grad_node_to_local_coords(xi_node, eta_node, zeta_node, &
                                                      N_grad_node_to_local_knode)



                ! Initialize Jacobian matrix xjac to zero
                xjac = 0.0d0

                ! Calculate Jacobian matrix xjac at the current node
                do knode_inner = 1, nnode
                    xjac(1, 1) = xjac(1, 1) + N_grad_node_to_local_knode(1, knode_inner) * x_nodes_current_elem(knode_inner)
                    xjac(1, 2) = xjac(1, 2) + N_grad_node_to_local_knode(1, knode_inner) * y_nodes_current_elem(knode_inner)
                    xjac(1, 3) = xjac(1, 3) + N_grad_node_to_local_knode(1, knode_inner) * z_nodes_current_elem(knode_inner)
                    xjac(2, 1) = xjac(2, 1) + N_grad_node_to_local_knode(2, knode_inner) * x_nodes_current_elem(knode_inner)
                    xjac(2, 2) = xjac(2, 2) + N_grad_node_to_local_knode(2, knode_inner) * y_nodes_current_elem(knode_inner)
                    xjac(2, 3) = xjac(2, 3) + N_grad_node_to_local_knode(2, knode_inner) * z_nodes_current_elem(knode_inner)
                    xjac(3, 1) = xjac(3, 1) + N_grad_node_to_local_knode(3, knode_inner) * x_nodes_current_elem(knode_inner)
                    xjac(3, 2) = xjac(3, 2) + N_grad_node_to_local_knode(3, knode_inner) * y_nodes_current_elem(knode_inner)
                    xjac(3, 3) = xjac(3, 3) + N_grad_node_to_local_knode(3, knode_inner) * z_nodes_current_elem(knode_inner)
                end do

                ! Compute determinant of Jacobian matrix
                djac = xjac(1, 1) * (xjac(2, 2) * xjac(3, 3) - xjac(3, 2) * xjac(2, 3)) &
                     - xjac(1, 2) * (xjac(2, 1) * xjac(3, 3) - xjac(3, 1) * xjac(2, 3)) &
                     + xjac(1, 3) * (xjac(2, 1) * xjac(3, 2) - xjac(3, 1) * xjac(2, 2))

                ! Store djac for the current node in the element
                djac_all_elems_at_nodes(element_ID, knode) = djac
            end do
        end do

        !print *, 'Populating djac_all_elems_at_nodes done'
        !call pause(180)

        ! ========================================================
        ! Populating sig_H_at_nodes(total_nodes)
        ! Weighted average based on determinant of Jacobian
        ! ========================================================

        ! print *, 'Populating sig_H_at_nodes'

        ! Initialize sig_H_at_nodes to zero
        sig_H_at_nodes = 0.0d0

        ! Loop over all nodes in the mesh
        do node_ID = 1, total_nodes
            num_elems_containing_node = num_elems_of_nodes_matrix(node_ID)

            !print *, 'Node ID: ', node_ID, 'Number of elements: ', num_elems_containing_node
            ! Initialize temporary variables for summing sig_H and djac
            sum_sig_H_djac = 0.0d0
            sum_djac = 0.0d0

            ! Loop over all elements that contain the current node
            do kelem = 1, num_elems_containing_node
                ! Get the element ID and local node number for this node
                element_ID = nodes_to_elems_matrix(node_ID, kelem, 1)
                local_knode = nodes_to_elems_matrix(node_ID, kelem, 2)

                ! Retrieve sig_H for the current element and node
                sig_H_knode = sig_H_all_elems_at_nodes(element_ID, local_knode)

                ! Retrieve djac for the current element and node
                djac_knode = djac_all_elems_at_nodes(element_ID, local_knode)

                ! Accumulate the weighted sum of sig_H and the sum of djac
                sum_sig_H_djac = sum_sig_H_djac + sig_H_knode * djac_knode
                sum_djac = sum_djac + djac_knode
            end do

            ! Compute the weighted average of sig_H for the current node
            if (sum_djac > 0.d0) then
                sig_H_at_nodes(node_ID) = sum_sig_H_djac / sum_djac
            else
                sig_H_at_nodes(node_ID) = 0.d0
            end if

        end do
        
        
        !print *, 'Populating sig_H_at_nodes done'
        
        call mutexUnlock(2)
        ! if (kinc == 5) then
        !     print *, 'sig_H_all_elems_at_nodes'
        !     print *, sig_H_all_elems_at_nodes(1:10, 1:8)
        !     print *, 'djac_all_elems_at_nodes'
        !     print *, djac_all_elems_at_nodes(1:10, 1:8)
        !     print *, 'sig_H_at_nodes'
        !     print *, sig_H_at_nodes(1:100)
        !     call pause(180)
        ! end if

        
    end if

return
end


! *****************************************************************
! USER SUBROUTINE used just to read current nodal coordinates

subroutine UFIELD(field, kfield, nsecpt, kstep, kinc, time, node, &
                  coords, temp, dtemp, nfield)
    use precision
    use common_block
    include 'aba_param.inc'
   
    dimension field(nsecpt,nfield), time(2), coords(3), &
              temp(nsecpt), dtemp(nsecpt)

    ! print *, 'UFIELD: node = ', node, 'coords = ', coords

    ! IMPORTANT: coords in this subroutine is nodal coordinates, not IP coordinates
    ! like the one in UMAT and UMATHT

    ! Lock Mutex #5 to ensure safe writing to shared memory
    !call mutexInit(5)
    call mutexLock(5)

    ! Assign the current nodal coordinates to coords_all_nodes
    coords_all_nodes(node, 1) = coords(1)
    coords_all_nodes(node, 2) = coords(2)
    coords_all_nodes(node, 3) = coords(3)

    ! Unlock Mutex #5
    call mutexUnlock(5)

return
end



subroutine calc_sig_H_grad_all_elems_at_inpts(noel, kinpt, kinc)

    use precision
    use iso_module
    use common_block

    integer :: noel, kinc, jelem, knode, kinpt, node_ID, idim, jdim
    real(kind=dp), dimension(nnode) :: N_node_to_local_kinpt, sig_H_noel_at_nodes
    real(kind=dp), dimension(ndim,nnode) :: N_grad_node_to_local_kinpt, N_grad_node_to_global_kinpt
    real(kind=dp), dimension(ndim,ndim) :: xjac, xjac_inv
    real(kind=dp), dimension(ndim) :: sig_H_grad_noel_at_kinpt


    ! Extract the hydrostatic stress at nodal points
    ! which is extrapolated from the integration points in UEXTERNALDB

    ! Professor Aravas approach
    do knode = 1, nnode
        node_ID = elems_to_nodes_matrix(noel, knode)
        sig_H_noel_at_nodes(knode) = sig_H_at_nodes(node_ID)
    end do

    ! Binh's original approach

    ! do knode = 1, nnode
    !     sig_H_noel_at_nodes(knode) = sig_H_all_elems_at_nodes(noel, knode)
    ! end do

    ! if (kinc == 3) then
    !     print *, 'sig_H_noel_at_nodes: ', sig_H_noel_at_nodes
    ! end if

    ! if (kinc == 3) then
    !     do node_ID = 100, 110
    !         print *, 'node_ID: ', node_ID
    !         print *, 'sig_H_at_nodes: ', sig_H_at_nodes(node_ID)
    !         ! Now we would want to compare it with sig_H_all_elems_at_nodes
    !         do element_ID = 1, num_elems_of_nodes_matrix(node_ID)
    !             jelem = nodes_to_elems_matrix(node_ID, element_ID, 1)
    !             local_knode = nodes_to_elems_matrix(node_ID, element_ID, 2)
    !             print *, 'sig_H_all_elems_at_nodes: ', sig_H_all_elems_at_nodes(jelem, local_knode)
    !         end do
    !     end do
        
    !     call pause(180)
    ! end if

            
    N_node_to_local_kinpt = all_N_node_to_local_kinpt(kinpt, 1:nnode)

    N_grad_node_to_local_kinpt = all_N_grad_node_to_local_kinpt(kinpt, 1:ndim, 1:nnode)

    !print *, 'N_grad_node_to_local_kinpt: ', N_grad_node_to_local_kinpt
    !print *, 'N_node_to_local_kinpt: ', N_node_to_local_kinpt
    xjac = 0.d0

!   Compute the Jacobian matrix (xjac)
    do knode = 1, nnode
        do idim = 1, ndim
            do jdim = 1, ndim
                node_ID = elems_to_nodes_matrix(noel, knode)
                xjac(jdim, idim) = xjac(jdim, idim) + &
                    N_grad_node_to_local_kinpt(jdim, knode) * &
                        coords_all_nodes(node_ID, idim)
            end do  
        end do
    end do

    !   Calculate the determinant of the Jacobian matrix (djac)
    djac =  xjac(1,1)*xjac(2,2)*xjac(3,3)+xjac(2,1)*xjac(3,2)*xjac(1,3) &
            + xjac(3,1)*xjac(2,3)*xjac(1,2)-xjac(3,1)*xjac(2,2)*xjac(1,3) &
            - xjac(2,1)*xjac(1,2)*xjac(3,3)-xjac(1,1)*xjac(2,3)*xjac(3,2)
    
    xjac_inv(1,1)=(xjac(2,2)*xjac(3,3)-xjac(2,3)*xjac(3,2))/djac
    xjac_inv(1,2)=(xjac(1,3)*xjac(3,2)-xjac(1,2)*xjac(3,3))/djac
    xjac_inv(1,3)=(xjac(1,2)*xjac(2,3)-xjac(1,3)*xjac(2,2))/djac
    xjac_inv(2,1)=(xjac(2,3)*xjac(3,1)-xjac(2,1)*xjac(3,3))/djac
    xjac_inv(2,2)=(xjac(1,1)*xjac(3,3)-xjac(1,3)*xjac(3,1))/djac
    xjac_inv(2,3)=(xjac(1,3)*xjac(2,1)-xjac(1,1)*xjac(2,3))/djac
    xjac_inv(3,1)=(xjac(2,1)*xjac(3,2)-xjac(2,2)*xjac(3,1))/djac
    xjac_inv(3,2)=(xjac(1,2)*xjac(3,1)-xjac(1,1)*xjac(3,2))/djac
    xjac_inv(3,3)=(xjac(1,1)*xjac(2,2)-xjac(1,2)*xjac(2,1))/djac
    
    if (djac < 0.d0) then ! negative or zero jacobian
        write(7,*) 'WARNING: element', jelem, 'has neg. Jacobian'
    endif

!   Compute the derivatives of shape functions with respect to global coordinates (B_deriv_global)
    N_grad_node_to_global_kinpt = matmul(xjac_inv, N_grad_node_to_local_kinpt)

    sig_H_grad_noel_at_kinpt = matmul(N_grad_node_to_global_kinpt, sig_H_noel_at_nodes) ! shape (3, 8) * shape (8) = shape (3)

    sig_H_grad_all_elems_at_inpts(noel, kinpt, :) = sig_H_grad_noel_at_kinpt

    !print *, 'sig_H_grad_noel_at_kinpt: ', sig_H_grad_noel_at_kinpt
    

return
end

! This is isotropic von Mises plasticity model
! Note: flow curve in the props should appear at the end of props. 
! If there is any non flow curve props behind the flow curve, you must move
! it before_flow_props_idx index in the props array.

!***********************************************************************

subroutine UMAT(stress,statev,ddsdde,sse,spd,scd,rpl,ddsddt, &
    drplde,drpldt,stran,dstran,time,dtime,temp2,dtemp,predef,dpred, &
    cmname,ndi,nshr,ntens,nstatv,props,nprops,coords,drot,pnewdt, &
    celent,dfgrd0,dfgrd1,noel,npt,layer,kspt,jstep,kinc)

    use precision
    use common_block
    !use ieee_arithmetic
    include 'aba_param.inc' 

    character*8 cmname
    dimension stress(ntens),statev(nstatv),ddsdde(ntens,ntens), &
        ddsddt(ntens),drplde(ntens),stran(ntens),dstran(ntens), &
        time(2),predef(1),dpred(1),props(nprops),coords(3),drot(3,3), &
        dfgrd0(3,3),dfgrd1(3,3),jstep(4)
    
    real(kind=dp) :: E, nu, lambda, mu, eqplas, deqplas, rhs 
    real(kind=dp) :: syield, syiel0, sig_vonMises, sig_H, sig_P1, sig_P2, sig_P3, sig_Tresca
    real(kind=dp) :: effective_mu, effective_lambda, effective_hard    

    real(kind=dp) :: eelas(ntens), eplas(ntens), flow(ntens), stress_copy(ntens)
    real(kind=dp) :: hard(3), old_stress(ntens), old_eplas(ntens)
    real(kind=dp) :: sig_principal_unsorted(ndim), sig_principal_sorted(ndim)
    real(kind=dp) :: sig_principal_dir(ndim, ndim)
    real(kind=dp) :: invariant_p, invariant_q, invariant_r, triaxiality, lode_norm
    real(kind=dp) :: sig_principal_1, sig_principal_2, sig_principal_3
    real(kind=dp), parameter :: toler = 1e-12
    real(kind=dp), parameter :: newton = 100
    integer :: k_newton

    ! LOCAL ARRAYS
    ! ----------------------------------------------------------------
    ! EELAS - ELASTIC STRAINS
    ! EPLAS - PLASTIC STRAINS
    ! FLOW - DIRECTION OF PLASTIC FLOW
    ! ----------------------------------------------------------------
    
    ! ----------------------------------------------------------------
    ! UMAT FOR ISOTROPIC ELASTICITY AND ISOTROPIC MISES PLASTICITY
    ! CANNOT BE USED FOR PLANE STRESS
    ! ----------------------------------------------------------------
    ! PROPS(before_mech_props_idx+1) - E
    ! PROPS(before_mech_props_idx+2) - NU
    ! PROPS(before_flow_props_idx+1:nprops) - SYIELD AN HARDENING DATA
    ! props(before_flow_props_idx+1) - syiel0, 
    ! props(before_flow_props_idx+2) - eqpl0, 
    ! props(before_flow_props_idx+3) - syiel1, 
    ! props(before_flow_props_idx+4) - eqpl1, ...
    ! and props(nprops-1) - syield_N, props(nprops) - eqplas_N
    ! CALLS UHARD FOR CURVE OF YIELD STRESS VS. PLASTIC STRAIN
    ! ----------------------------------------------------------------

    ! material properties

    E = props(1)           ! Young's modulus 
    nu = props(2)          ! Poisson's ratio 
    
    ! print *, 'E = ', E
    ! print *, 'nu = ', nu
    !eelas(1:ntens) = statev(eelas_start_idx:eelas_end_idx)
    !eplas(1:ntens) = statev(eplas_start_idx:eplas_end_idx)
    eqplas = statev(eqplas_idx)
    deqplas = 0.0d0
    old_stress = stress
    old_eplas = eplas

    !  Compute the gradient of the hydrostatic stress  

    if (time(1) > 0) then
        call calc_sig_H_grad_all_elems_at_inpts(noel, npt, kinc)
    end if  


    call rotsig(statev(eelas_start_idx), drot, eelas, 2, ndi, nshr)
    call rotsig(statev(eplas_start_idx), drot, eplas, 2, ndi, nshr)

    ! Lame's parameters
    mu = E/(2.0d0 * (1.0d0 + nu))  ! Shear modulus
    lambda = E*nu/((1.0d0 + nu) * (1.0d0 - 2.0d0 * nu)) ! Lame's first constant


    ! initialize as 0
    ddsdde = 0.0d0 ! Their unit is Pa
    
    do i = 1, ndi
        do j = 1, ndi
            ddsdde(j, i) = lambda
        end do 
        ddsdde(i,i) = lambda + 2.0d0 * mu
    end do 

    ! Shear contribution
    do i = ndi + 1, ntens
        ddsdde(i,i) = mu
    end do 

    ! do i = 1, ntens
    !     if (stress(i) /= stress(i)) then
    !         write(7,*) 'WARNING: stress is NaN'
    !     end if
    ! end do

    !    Calculate predictor stress and elastic strain
    stress = stress + matmul(ddsdde,dstran)



    eelas = eelas + dstran

    ! Calculate equivalent von Mises stress
    
    sig_vonMises = (stress(1) - stress(2))**2.0d0 + &
                   (stress(2) - stress(3))**2.0d0 + &
                   (stress(3) - stress(1))**2.0d0 + &
                    6.0d0 * (stress(4)**2.0d0 + stress(5)**2.0d0 + stress(6)**2.0d0)

    sig_vonMises = sqrt(sig_vonMises/2.0d0)
    
    ! get yield stress from the specified hardening curve
    ! nvalue equal to number of points on the hardening curve
    
    nvalue = (nprops - before_flow_props_idx) / 2

    ! print *, 'nvalue = ', nvalue ! 100
    ! print *, 'before_flow_props_idx = ', before_flow_props_idx ! 40
    
    call UHARD(syiel0, hard, eqplas, &
                                statev, nvalue, props(before_flow_props_idx + 1))
    

    ! Determine if active yielding

    if (sig_vonMises > (1.0d0 + toler) * syiel0) then

        ! print *, 'active yielding'
        ! actively yielding
        ! separate the hydrostatic from the deviatoric stress
        ! calculate the flow direction

        sig_H = (stress(1) + stress(2) + stress(3))/3.0d0
        flow(1:ndi) = (stress(1:ndi) - sig_H)/sig_vonMises
        flow(ndi+1:ntens) = stress(ndi+1:ntens)/sig_vonMises
        
        ! solve for equivalent von Mises stress and equivalent plastic strain increment 
        ! using Newton-Raphson iteration

        syield = syiel0
        deqplas = 0.0d0
        do k_newton = 1, newton
            rhs = sig_vonMises - (3.0d0 * mu * deqplas) - syield
            deqplas = deqplas + rhs / ((3.0d0 * mu) + hard(1))

            call UHARD(syield, hard, eqplas + deqplas, &
                        statev, nvalue, props(before_flow_props_idx + 1))
                                
            if (abs(rhs) < toler * syiel0) exit
        end do

        if (k_newton == newton) write(7,*) 'WARNING: plasticity loop failed'

        ! Update stresses, elastic and plastic strains
 
        stress(1:ndi) = flow(1:ndi) * syield + sig_H
        eplas(1:ndi) = eplas(1:ndi) + 3.0d0/2.0d0 * flow(1:ndi) * deqplas
        eelas(1:ndi) = eelas(1:ndi) - 3.0d0/2.0d0 * flow(1:ndi) * deqplas
        
        stress(ndi + 1:ntens) = flow(ndi + 1:ntens) * syield
        eplas(ndi + 1:ntens) = eplas(ndi + 1:ntens) + 3.0d0 * flow(ndi + 1:ntens) * deqplas
        eelas(ndi + 1:ntens) = eelas(ndi + 1:ntens) - 3.0d0 * flow(ndi + 1:ntens) * deqplas

        ! Finally, we update the equivalent plastic strain
        eqplas = eqplas + deqplas

        ! Calculate the plastic strain energy density
        ! psi_plas = deqplas * (syiel0 + syield) / 2.d0

        do i=1,ntens
            spd = spd + (stress(i)+old_stress(i)) * (eplas(i) - old_eplas(i))/2.0d0
        end do

        ! Formulate the jacobian (material tangent)   

        ! effective shear modulus
        effective_mu = mu * syield / sig_vonMises 

        ! effective Lame's constant
        effective_lambda = (E/(1.0d0 - 2.0d0 * nu) - 2.0d0 * effective_mu)/3.0d0 

        ! effective hardening modulus
        effective_hard = 3.0d0 * mu * hard(1)/(3.0d0 * mu + hard(1)) - 3.0d0 * effective_mu 

        do i = 1, ndi
            do j = 1, ndi
                ddsdde(j,i) = effective_lambda
            end do
            ddsdde(i,i) = 2.0d0 * effective_mu + effective_lambda
        end do

        do i = ndi + 1, ntens
            ddsdde(i,i) = effective_mu
        end do

        do i = 1, ntens
            do j = 1, ntens
                ddsdde(j,i) = ddsdde(j,i) + effective_hard * flow(j) * flow(i)
            end do
        end do
    endif

    ! Recalculate the stress
    sig_vonMises = (stress(1) - stress(2))**2.0d0 + &
                   (stress(2) - stress(3))**2.0d0 + &
                   (stress(3) - stress(1))**2.0d0 + &
                    6.0d0 * (stress(4)**2.0d0 + stress(5)**2.0d0 + stress(6)**2.0d0)

    sig_vonMises = sqrt(sig_vonMises/2.0d0)

    sig_H = (stress(1) + stress(2) + stress(3))/3.0d0

    call calc_stress_invariants(stress, ntens, invariant_p, invariant_q, invariant_r)
    call calc_triaxiality(invariant_p, invariant_q, triaxiality)
    call calc_normalized_lode(invariant_r, invariant_q, lode_norm)
    ! call calc_max_principal_stress(stress, sig_principal_sorted, ntens, ndim)
    ! Abaqus library function to calculate principal stresses
    do i = 1, ntens
        stress_copy(i) = stress(i)
    end do

    call sprind(stress_copy,sig_principal_unsorted,sig_principal_dir,1,ndi,nshr)
    call sort_descending(sig_principal_unsorted, sig_principal_sorted, ndim)

    sig_P1 = sig_principal_sorted(1)
    sig_P2 = sig_principal_sorted(2)
    sig_P3 = sig_principal_sorted(3)

    sig_Tresca = (sig_P1 - sig_P3)/2.0d0

    ! if (kinc == 5) then
    !     print *, 'stress = ', stress
    !     print *, 'invariant_p = ', invariant_p
    !     print *, 'invariant_q = ', invariant_q
    !     print *, 'invariant_r = ', invariant_r
    !     print *, 'triaxiality = ', triaxiality
    !     print *, 'lode_norm = ', lode_norm
    ! end if

    ! Update coords at integration points

    coords_all_inpts(noel, npt, 1) = coords(1)
    coords_all_inpts(noel, npt, 2) = coords(2)
    coords_all_inpts(noel, npt, 3) = coords(3)

    ! update state variables
    
    statev(sig_start_idx:sig_end_idx) = stress(1:ntens)
    statev(stran_start_idx:stran_end_idx) = stran(1:ntens)
    statev(eelas_start_idx:eelas_end_idx) = eelas(1:ntens)
    statev(eplas_start_idx:eplas_end_idx) = eplas(1:ntens)
    statev(eqplas_idx) = eqplas
    statev(deqplas_idx) = deqplas
    statev(sig_H_idx) = sig_H
    statev(sig_vonMises_idx) = sig_vonMises
    statev(sig_Tresca_idx) = sig_Tresca
    statev(sig_P1_idx) = sig_P1
    statev(sig_P2_idx) = sig_P2
    statev(sig_P3_idx) = sig_P3
    statev(triax_idx) = triaxiality
    statev(lode_idx) = lode_norm

    ! Update the kuser_vars
    kuser_vars(noel, sig_start_idx:sig_end_idx, npt) = stress(1:ntens)
    kuser_vars(noel, stran_start_idx:stran_end_idx, npt) = stran(1:ntens)
    kuser_vars(noel, eelas_start_idx:eelas_end_idx, npt) = eelas(1:ntens)
    kuser_vars(noel, eplas_start_idx:eplas_end_idx, npt) = eplas(1:ntens)
    kuser_vars(noel, eqplas_idx, npt) = eqplas
    kuser_vars(noel, deqplas_idx, npt) = deqplas
    kuser_vars(noel, sig_H_idx, npt) = sig_H
    kuser_vars(noel, sig_vonMises_idx, npt) = sig_vonMises
    kuser_vars(noel, sig_Tresca_idx, npt) = sig_Tresca
    kuser_vars(noel, sig_P1_idx, npt) = sig_P1
    kuser_vars(noel, sig_P2_idx, npt) = sig_P2
    kuser_vars(noel, sig_P3_idx, npt) = sig_P3
    kuser_vars(noel, triax_idx, npt) = triaxiality
    kuser_vars(noel, lode_idx, npt) = lode_norm
    
return
end

!***********************************************************************

subroutine UHARD(syield, hard, eqplas, statev, nvalue, table)

    use precision
    include 'aba_param.inc'

    character*80 cmname
    dimension hard(3),statev(*),table(2, nvalue)
    
    ! set yield stress to last value of table, hardening to zero
    
    syield = table(1, nvalue)
    hard(1) = 0.d0

    ! if more than one entry, search table
    
    if (nvalue > 1) then
        do k1 = 1, nvalue - 1
            eqpl1 = table(2, k1 + 1)
            if (eqplas < eqpl1) then
                eqpl0 = table(2, k1)
                if (eqpl1 <= eqpl0) then
                    write(7,*) 'error - plastic strain must be entered in ascending order'
                end if

                ! current yield stress and hardening

                deqpl = eqpl1 - eqpl0
                syiel0 = table(1, k1)
                syiel1 = table(1, k1 + 1)
                dsyiel = syiel1 - syiel0
                hard(1) = dsyiel/deqpl
                syield = syiel0 + (eqplas - eqpl0) * hard(1)
                exit
            endif
        end do
    endif

return
end


!***********************************************************************

subroutine UMATHT(u,dudt,dudg,flux,dfdt,dfdg, &
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
    real(kind=dp) :: mode, mode_tol, crystal_structure, crystal_tol, R, T, VH, DL 
    real(kind=dp) :: avogadro, NL, alpha_dis, alpha_gb, alpha_carb, NT_dis, NT_gb, NT_carb
    real(kind=dp) :: WB_dis, WB_gb, WB_carb, beta_BCC, beta_FCC, a_lattice_BCC, a_lattice_FCC
    real(kind=dp) :: gamma, rho_d0, theta_coverage, k_HEDE
    real(kind=dp) :: CL_mol_old, dCL_mol, CL_mol, CL, K_dis, K_gb, K_carb
    real(kind=dp) :: burgers_vector, inverse_burgers_vector, beta, NL_mol
    real(kind=dp) :: thetaL, temp_dis, thetaT_dis, temp_gb, thetaT_gb, temp_carb, thetaT_carb
    real(kind=dp) :: eqplas, rho_d, NT_dis_mol, CT_dis, CT_dis_mol
    real(kind=dp) :: part_C_mol_part_NT_dis_mol, dNT_dis_deqplas
    real(kind=dp) :: part_CT_dis_mol_part_CL_mol, part_CT_gb_mol_part_CL_mol, part_CT_carb_mol_part_CL_mol
    real(kind=dp) :: total_part_CT_mol_part_CL_mol, part_C_mol_part_CL_mol, deqplas
    real(kind=dp) :: C_mol, CT_mol, CT_gb, CT_gb_mol, CT_carb, CT_carb_mol

    mode            = props(1) ! Mode of the traps
    mode_tol        = 1.0e-6
    ! mode_tol is only for checking the equality since mode is a real instead of an integer
    ! mode = 1: CT = 0 (no trap hydrogen)
    ! mode = 2: Kumnick, Krom et al. (1999) Hydrogen transport near a blunting crack tip
    ! mode = 3: Sofronis, Dadfarnia et al. (2011) Hydrogen interaction with multiple traps: Can it be used to mitigate embrittlement
    
    crystal_structure = props(2) ! Crystal structure (1 - BCC, 2 - FCC)
    crystal_tol     = 1.0e-3 ! same for above reason

    R               = props(3) ! Universal gas constant (8.31446 J/(mol K))
    T               = props(4) ! Temperature (300 K)
    VH              = props(5) ! Partial molar volume of hydrogen (2e-06 m^3/mol), 
    DL              = props(6) ! Diffusion coefficient for lattice hydrogen in the steel (m^2/s), from e-10 to e-15
                               ! This parameter varies widely for different steels and different conditions
    avogadro        = props(7) ! Avogadro's constant (6.022e23 1/mol)
    ! NL = Avogadro (NA) * (density of metal rho_M) / (atomic weight MM) 
    ! NL = 6.02214 × 10e23 [atom/mol] * 7900 [kg/m**3] / (55.85 * 10e-3 [kg/mol]) = 8.518e28 [atom/m**3]
    NL              = props(8) ! Number of solvent metal atoms per unit volume (8.518e28 1/m^3)
    alpha_dis       = props(9) ! Number of interstitial sites per trap site (dislocations) (1.0 dimless)
    alpha_gb        = props(10) ! Number of interstitial sites per trap site (grain boundaries) (1.0 dimless)
    alpha_carb      = props(11) ! Number of interstitial sites per trap site (carbides) (1.0 dimless)
    NT_dis          = props(12) ! Number of trap type of dislocations (1/m^3) (0 - instead defined by Dadfarnia et al. 2016)
    NT_gb           = props(13) ! Number of trap type of grain boundaries (1/m^3) (8.464e22)
    NT_carb         = props(14) ! Number of trap type of carbides (1/m^3) (8.464e26)
    WB_dis          = props(15) ! Binding energy of hydrogen to dislocations (- 20.2e3) (J/mol)
    WB_gb           = props(16) ! Binding energy of hydrogen to grain boundaries (- 58.6e3) (J/mol)
    WB_carb         = props(17) ! Binding energy of hydrogen to carbides (- 11.5e3) (J/mol)
    beta_BCC        = props(18) ! Number of number of hydrogen atoms that can reside in each lattice site (6.0 dimless)
    beta_FCC        = props(19) ! Number of number of hydrogen atoms that can reside in each lattice site (1.0 dimless)
    ! References for lattice parameter
    a_lattice_BCC   = props(20) ! Lattice parameter (2.866 Angstrom or 2.866e-10 m)
    a_lattice_FCC   = props(21) ! Lattice parameter (3.571 Angstrom or 3.571e-10 m)
    gamma           = props(22) ! gamma fitting parameter in Dadfarnia et al. (2.0e16 1/m^2)
    rho_d0          = props(23) ! Dislocation density for the annealed material in Dadfarnia et al. (1.0e10 1/m^2)
    delta_g_b0      = props(24) ! Variation of Gibbs free energy (30e3 J/mol)

    ! THE DEGREE OF FREEDOM FOR HYDROGEN CONCENTRATION IS mol/m^3
    ! It is marked by the suffix _mol in the variable name
    ! We can convert it to 1/m^3 by multiplying with Avogadro's number, which does not have any suffix
    ! Example: CL_mol = CL / avogadro or CL (1/m^3) = CL_mol (mol/m^3) * avogadro (1/mol)

    CL_mol_old = temp ! (mol/m^3)
    dCL_mol = dtemp ! (mol/m^3)
    CL_mol = CL_mol_old + dCL_mol ! (mol/m^3)

    CL = CL_mol * avogadro ! (1/m^3)

    ! Arrhenius reaction rate constant for trap types 
    ! (dimless) = exp( - (J/mol) / (J/(mol K) * K))
    K_dis = dexp( -WB_dis / (R * T)) ! constant, dimless
    K_gb = dexp( -WB_gb / (R * T)) ! constant, dimless
    K_carb = dexp( -WB_carb / (R * T)) ! constant, dimless

    ! Finding theta_trap based on Oriani equilibrium theory
    ! which results in a Fermi-Dirac relation

    ! slip occurs along the plane of the shortest Burgers vector
    if (abs(crystal_structure - 1.0d0) <= crystal_tol) then ! BCC crystal structure
        beta = beta_BCC ! beta is taken to be 6 for BCC as indirect
                        ! evidence indicates tetrahedral site occupancy rather than 
                        ! octahedral site occupancy at room temperature in alpha-iron
        ! slip is assumed to occur along the {110} plane and ⟨111⟩ direction
        burgers_vector = (dsqrt(3.0d0)/2.0d0) * a_lattice_BCC ! (m) 
        inverse_burgers_vector = 1.0d0/burgers_vector ! (1/m)
    elseif (abs(crystal_structure - 2.0d0) <= crystal_tol) then ! FCC crystal structure
        beta = beta_FCC ! beta is taken to be 1 for FCC, resulting from the more favourable 
                        ! octahedral site occupancy (beta = 2 for tetrahedral)
        ! slip occurs along the closed packed plane {111} and slip direction ⟨110⟩
        burgers_vector = (dsqrt(2.0d0)/2.0d0) * a_lattice_FCC ! (m)
        inverse_burgers_vector = 1.0d0/burgers_vector ! (1/m)
    end if
    
    ! Finding NL_mol
    NL_mol = NL / avogadro ! (mol/m^3) = (1/m^3) / (1/mol)

    ! Finding thetaL 
    thetaL = CL / (beta * NL) ! dimless = (1/m^3) / (dimless * 1/m^3)

    ! thetaT / (1 - thetaT) = K * thetaL / (1 - thetaL)
    ! However if thetaL << 1  then 
    ! thetaT / (1 - thetaT) = K * thetaL

    temp_dis = K_dis * thetaL / (1.0d0 - thetaL) ! (dimless)
    thetaT_dis = temp_dis / (1.0d0 + temp_dis) ! (dimless)
    temp_gb = K_gb * thetaL / (1.0d0 - thetaL) ! (dimless)
    thetaT_gb = temp_gb / (1.0d0 + temp_gb) ! (dimless)
    temp_carb = K_carb * thetaL / (1.0d0 - thetaL) ! (dimless)
    thetaT_carb = temp_carb / (1.0d0 + temp_carb) ! (dimless)

    eqplas = statev(eqplas_idx) ! (dimless) equivalent plastic strain

    if (abs(mode - 1) <= mode_tol) then ! Lattice H only
        rho_d = 0.d0 ! (1/m^2)
        NT_dis = 0.d0 ! (1/m^3)
        NT_dis_mol = 0.d0 ! (mol/m^3)
        CT_dis = 0.d0 ! (1/m^3)
        CT_dis_mol = 0.d0 ! (mol/m^3)
        NT_gb = 0.d0 ! (1/m^3)
        NT_carb = 0.d0 ! (1/m^3)
        part_C_mol_part_NT_dis_mol = 0.d0 ! (1/m^3) / (1/m^3) = (dimless)
        dNT_dis_deqplas = 0.d0 ! (1/m^3) / (dimless) = (1/m^3)	
        
    elseif (abs(mode - 2) <= mode_tol) then ! Krom et al. (in sites/m^3), developed from Kumnick & Johnson 
        rho_d = 0.d0
        NT_dis = 10.d0 ** (23.26d0 - 2.33d0 * dexp(-5.5d0 * eqplas))
        NT_dis_mol = NT_dis / avogadro
        CT_dis = alpha_dis * thetaT_dis * NT_dis
        CT_dis_mol = CT_dis / avogadro
        part_C_mol_part_NT_dis_mol = (K_dis * CL_mol)/(K_dis * CL_mol + beta * NL_mol)
        dNT_dis_mol_deqplas = (29.5d0 * dexp(-5.5d0 * eqplas) * NT_dis ) / avogadro
    
    elseif (abs(mode - 3) <= mode_tol) then ! Dadfarnia et al.
        
        if (eqplas < 0.5) then
            rho_d = rho_d0 + eqplas * gamma ! rho_d unit is 1/m^2 = 1/m^2 + dimless * 1/m^2
            NT_dis = inverse_burgers_vector * rho_d ! NT_dis unit is 1/m^3 = 1/m * 1/m^2
            NT_dis_mol = NT_dis / avogadro ! NT_dis_mol unit is mol/m^3 = 1/m^3 / 1/mol
            CT_dis = alpha_dis * thetaT_dis * NT_dis ! CT_dis unit is 1/m^3 = dimless * dimless * 1/m^3
            CT_dis_mol = CT_dis / avogadro ! = (1/m^3) / (1/mol) = (mol/m^3)

            ! part_C_part_NT_dis = (K_dis * CL)/(K_dis * CL + beta * NL) ! dimless = dimless * 1/m^3 / (dimless * 1/m^3 + dimless * 1/m^3)
            part_C_mol_part_NT_dis_mol = (K_dis * CL_mol)/(K_dis * CL_mol + beta * NL_mol) ! dimless = dimless * mol/m^3 / (dimless * mol/m^3 + dimless * mol/m^3)
            ! dNT_dis_deqplas = inverse_burgers_vector * gamma ! 1/m^3 = 1/m * 1/m^2
            dNT_dis_mol_deqplas = (inverse_burgers_vector * gamma) / avogadro ! mol/m^3 = (1/m * 1/m^2) / (1/mol)
            ! du2 in emilio is part_C_mol_part_NT_dis_mol * dNT_dis_mol_deqplas * deqplas     
        
        elseif (eqplas >= 0.5) then
            rho_d = 1.0d16 ! (1/m^2)
            NT_dis = inverse_burgers_vector * rho_d ! (1/m^3)
            NT_dis_mol = NT_dis / avogadro ! (mol/m^3)
            CT_dis = alpha_dis * thetaT_dis * NT_dis ! (1/m^3)
            CT_dis_mol = CT_dis / avogadro ! (mol/m^3)
            ! part_C_part_NT_dis = 0.d0
            part_C_mol_part_NT_dis_mol = (K_dis * CL_mol)/(K_dis * CL_mol + beta * NL_mol) ! dimless = dimless * mol/m^3 / (dimless * mol/m^3 + dimless * mol/m^3)
            ! dNT_dis_deqplas = 0.d0
            dNT_dis_mol_deqplas = 0.d0
        endif
    end if
    
    ! part_CT_dis_part_CL = (NT_dis * K_dis * NL * beta)/((K_dis * CL + NL * beta)**2.d0) 
    ! ! (dimless) = (1/m^3 * dimless * 1/m^3 * dimless) / ((dimless * 1/m^3 + 1/m^3 * dimless)**2)
    ! part_CT_gb_part_CL = (NT_gb * K_gb * NL * beta)/((K_gb * CL + NL * beta)**2.d0)
    ! part_CT_carb_part_CL = (NT_carb * K_carb * NL * beta)/((K_carb * CL + NL * beta)**2.d0)

    NT_dis_mol = NT_dis / avogadro ! (mol/m^3)
    NT_gb_mol = NT_gb / avogadro ! (mol/m^3)
    NT_carb_mol = NT_carb / avogadro ! (mol/m^3)

    ! (dimless) = (mol/m^3 * dimless * mol/m^3 * dimless) / ((dimless * mol/m^3 + mol/m^3 * dimless)**2)

    part_CT_dis_mol_part_CL_mol = (NT_dis_mol * K_dis * NL_mol * beta)/ &
                                    ((K_dis * CL_mol + NL_mol * beta)**2.d0)
    part_CT_gb_mol_part_CL_mol = (NT_gb_mol * K_gb * NL_mol * beta)/ &
                                    ((K_gb * CL_mol + NL_mol * beta)**2.d0)
    part_CT_carb_mol_part_CL_mol = (NT_carb_mol * K_carb * NL_mol * beta)/ &
                                    ((K_carb * CL_mol + NL_mol * beta)**2.d0)

    total_part_CT_mol_part_CL_mol = part_CT_dis_mol_part_CL_mol + &
                                  part_CT_gb_mol_part_CL_mol + &
                                  part_CT_carb_mol_part_CL_mol
    
    ! Finally, we update all the variables in UMATHT
    ! part_C_mol_part_CL_mol = part_CL_mol_part_CL_mol + part_CT_mol_part_CL_mol
    !                              = 1 + total_part_CT_mol_part_CL_mol

    part_C_mol_part_CL_mol = 1.d0 + total_part_CT_mol_part_CL_mol	
    dudt = part_C_mol_part_CL_mol
    
    deqplas = statev(deqplas_idx) ! (dimless) equivalent plastic strain increment
 
    ! (mol/m^3) = (mol/m^3) + (dimless * mol/m^3) + (dimless * mol/m^3 * dimless)
    u = u + part_C_mol_part_CL_mol * dCL_mol &
          + part_C_mol_part_NT_dis_mol * dNT_dis_mol_deqplas * deqplas

    dfdg = 0.0d0

    do kdim = 1, ntgrd
        ! Update the flux
        ! J_m = DL * Cbar_L * grad sigma_H / (R * T) - DL * grad Cbar_L
        grad_CL_mol_kdim = dtemdx(kdim) ! = (mol/m^3) / m = (mol/m^4)
        
        ! if (kinc == 4) then
        !    print *, 'sig_H_grad_all_elems_at_inpts = ', sig_H_grad_all_elems_at_inpts(noel, npt, kdim)
        ! end if

        flux(kdim) = DL * CL_mol * VH * sig_H_grad_all_elems_at_inpts(noel, npt, kdim) / (R * T) &
                   - DL * grad_CL_mol_kdim
        
        ! dudg is partial (Cbar_total) / partial (grad Cbar L), which is supposed to be 0
        dudg(kdim) = 0.0d0

        ! partial J_m / partial (Cbar_L) = (DL * VH) / (R * T) * grad_sigma_H(i)
        dfdt(kdim) = (DL * VH * sig_H_grad_all_elems_at_inpts(noel, npt, kdim)) / (R * T) 
        
        ! Update dudg
        dfdg(kdim,kdim) = - DL ! = - m^2/s
    end do

    ! store the concentration in each trap, in all traps and in traps and lattice
    
    CT_mol = 0.0d0
    
    ! CT_dis and CT_dis_mol is already calculated above
    ! CT_dis = alpha_dis * thetaT_dis * NT_dis ! (1/m^3)
    ! CT_dis_mol = CT_dis / avogadro ! (mol/m^3)
    ! CT_dis_mol = (NT_dis_mol * K_dis * CL_mol)/(K_dis * CL_mol + beta * NL_mol)
    
    CT_gb = alpha_gb * thetaT_gb * NT_gb ! (1/m^3)
    CT_gb_mol = CT_gb / avogadro ! (mol/m^3)
    ! That is also equivalent to this (assuming alpha_gb = 1)
    ! CT_gb_mol = (NT_gb_mol * K_gb * CL_mol)/(K_gb * CL_mol + beta * NL_mol)
    
    CT_carb = alpha_carb * thetaT_carb * NT_carb ! (1/m^3)
    CT_carb_mol = CT_carb / avogadro ! (mol/m^3)
    ! That is also equivalent to this (assuming alpha_carb = 1)
    ! CT_carb_mol = (NT_carb_mol * K_carb * CL_mol)/(K_carb * CL_mol + beta * NL_mol)
    
    CT_mol = CT_dis_mol + CT_gb_mol + CT_carb_mol
    
    C_mol = CL_mol + CT_mol

    C_molfrac = C_mol * conversion_mol_to_molfrac

    C_wtppm = C_mol * conversion_mol_to_wtppm
    CL_wtppm = CL_mol * conversion_mol_to_wtppm
    CT_wtppm = CT_mol * conversion_mol_to_wtppm

    ! Hydrogen coverage factor
    theta_coverage = C_molfrac / (C_molfrac + exp(-delta_g_b0 /(R * T))) 

    ! Factor decreasing cohesive strength, based on HEDE mechanism
    k_HEDE = 1.0d0 - 1.0467d0 * theta_coverage + 0.1687d0 * theta_coverage ** 2.0d0

    statev(C_mol_idx) = C_mol
    statev(CL_mol_idx) = CL_mol
    statev(CT_mol_idx) = CT_mol
    statev(CT_dis_mol_idx) = CT_dis_mol
    statev(CT_gb_mol_idx) = CT_gb_mol
    statev(CT_carb_mol_idx) = CT_carb_mol
    statev(C_wtppm_idx) = C_wtppm
    statev(CL_wtppm_idx) = CL_wtppm
    statev(CT_wtppm_idx) = CT_wtppm
    statev(thetaL_idx) = thetaL
    statev(thetaT_dis_idx) = thetaT_dis
    statev(thetaT_gb_idx) = thetaT_gb
    statev(thetaT_carb_idx) = thetaT_carb
    statev(rho_d_idx) = rho_d
    statev(theta_coverage_idx) = theta_coverage
    statev(k_HEHE_idx) = k_HEDE

    ! Update the kuser_vars
    kuser_vars(noel, C_mol_idx, npt) = C_mol
    kuser_vars(noel, CL_mol_idx, npt) = CL_mol
    kuser_vars(noel, CT_mol_idx, npt) = CT_mol
    kuser_vars(noel, CT_dis_mol_idx, npt) = CT_dis_mol
    kuser_vars(noel, CT_gb_mol_idx, npt) = CT_gb_mol
    kuser_vars(noel, CT_carb_mol_idx, npt) = CT_carb_mol
    kuser_vars(noel, C_wtppm_idx, npt) = C_wtppm
    kuser_vars(noel, CL_wtppm_idx, npt) = CL_wtppm
    kuser_vars(noel, CT_wtppm_idx, npt) = CT_wtppm
    kuser_vars(noel, thetaL_idx, npt) = thetaL
    kuser_vars(noel, thetaT_dis_idx, npt) = thetaT_dis
    kuser_vars(noel, thetaT_gb_idx, npt) = thetaT_gb
    kuser_vars(noel, thetaT_carb_idx, npt) = thetaT_carb
    kuser_vars(noel, rho_d_idx, npt) = rho_d
    kuser_vars(noel, theta_coverage_idx, npt) = theta_coverage
    kuser_vars(noel, k_HEHE_idx, npt) = k_HEDE

return
end


subroutine UVARM(uvar,direct,t,time,dtime,cmname,orname, &
    nuvarm,noel,npt,layer,kspt,kstep,kinc,ndi,nshr,coord, &
    jmac,jmatyp,matlayo,laccfla)
    
    use precision
    use common_block
    include 'aba_param.inc'
!
    character*80 cmname,orname
    character*3 flgray(100)
    dimension uvar(nuvarm),direct(3,3),t(3,3),time(2)
    dimension array(100),jarray(100),jmac(*),jmatyp(*),coord(*)

    integer :: iostat, element_ID, node_id, current_element_idx
    integer, dimension(11) :: values

    character(len=512) :: line, outdir, aelread, aelwrite, andread, andwrite, anelwrite

    !     the dimensions of the variables flgray, array and jarray
    !     must be set equal to or greater than 15.
    !     Number 100 is arbitrary, which can accomodate lots of SDV

    ! Variables to Be Defined
    ! uvar(nuvarm)
    ! An array containing the user-defined output variables. 
    ! These are passed in as the values at the beginning of the increment 
    ! and must be returned as the values at the end of the increment.
    
    call GETVRM('SDV',array,jarray,flgray,jcrd,jmac,jmatyp,matlayo,laccfla)
    
    ! Choose only a subset of SDV to output
    ! If we output all SDV the odb file would be extremely heavy

    ! Manually define which sdv to be output
    ! Please refer to processing_input/depvar.xlsx for the list of chosen SDV
    
    uvar(1) = array(eqplas_idx) 
    uvar(2) = array(sig_H_idx)
    uvar(3) = array(sig_vonMises_idx)
    uvar(4) = array(sig_Tresca_idx)
    uvar(5) = array(sig_P1_idx)
    uvar(6) = array(triax_idx)
    uvar(7) = array(lode_idx)
    uvar(8) = array(C_mol_idx)
    uvar(9) = array(CL_mol_idx)
    uvar(10) = array(CT_mol_idx)
    !uvar(10) = array(C_wtppm_idx)
    !uvar(11) = array(CL_wtppm_idx)
    !uvar(12) = array(CT_wtppm_idx)
    !uvar(13) = array(rho_d_idx)
    !uvar(14) = array(theta_coverage_idx)
    !uvar(15) = array(k_HEHE_idx)
    

    ! call getoutdir(outdir, lenoutdir)
    ! aelread = trim(outdir) // '/processing_input/output_uvarm.txt'

    ! print *, 'aelread = ', aelread
    
    ! open(unit=15, file=aelread, status="old", action="read", iostat=iostat)
    ! if (iostat /= 0) then
    !     print *, "Error opening file: ", iostat
    !     return
    ! end if

    ! read(15, '(A)', iostat=iostat) line
    ! if (iostat /= 0) then
    !     print *, "Error reading file: ", iostat
    !     close(15)
    !     return
    ! end if

    ! ! Convert the line into integers
    ! read(line, *) values
    ! print *, 'values = ', values
    ! close(15)

    ! ! The first index in the file is the number of output SDV
    ! num_chosen_SDV = values(1)
    
    ! do uvar_idx = 1, num_chosen_SDV
    !     sdv_idx = values(uvar_idx + 1)
    !     uvar(uvar_idx) = array(sdv_idx)
    ! end do

    

return
end

