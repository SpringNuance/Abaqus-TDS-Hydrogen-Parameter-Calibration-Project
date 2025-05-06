C***********************************************************************
C    
      SUBROUTINE KDETJEL(total_elems)
C
C*** For each element in the mesh, calculates detJ at each node of the element
C    and stores it in array ADETJEL. These values are then used to calculate
C    nodal pressures using weighted element pressures based on "element area"
C    converging to the specific node
C
      use ktransfer      
      include 'aba_param.inc'
      DIMENSION X(nnode),Y(nnode),Z(nnode)
      DIMENSION AJ(kxdim,kxdim),DN(kxdim,nnode)
      DIMENSION XINODE(nnode),ETANODE(nnode),ZETANODE(nnode)
C
C*** Element nodal coordinates (�i,�i) (or (�i,�i,�i) in 3D elements) in natural space
C

        XINODE(1) = -1.D0;  ETANODE(1) = -1.D0;  ZETANODE(1) = -1.D0;
        XINODE(2) =  1.D0;  ETANODE(2) = -1.D0;  ZETANODE(2) = -1.D0; 
        XINODE(3) =  1.D0;  ETANODE(3) =  1.D0;  ZETANODE(3) = -1.D0; 
        XINODE(4) = -1.D0;  ETANODE(4) =  1.D0;  ZETANODE(4) = -1.D0; 
        XINODE(5) = -1.D0;  ETANODE(5) = -1.D0;  ZETANODE(5) = 1.D0; 
        XINODE(6) =  1.D0;  ETANODE(6) = -1.D0;  ZETANODE(6) = 1.D0; 
        XINODE(7) =  1.D0;  ETANODE(7) =  1.D0;  ZETANODE(7) = 1.D0; 
        XINODE(8) = -1.D0;  ETANODE(8) =  1.D0;  ZETANODE(8) = 1.D0; 
    
C
      DO element_ID=1, total_elems    ! Do loop over all element labels
C
C*** Global coordinates (x,y) (or (x,y,z) in 3D elements) of point (node) under consideration
C
        X=0.D0; Y=0.D0; Z=0.D0
        DO knode=1,nnode
          node_ID = elems_to_nodes_matrix(element_ID,knode)
          X(knode) = common_coords_all_nodes(node_ID,1)
          Y(knode) = common_coords_all_nodes(node_ID,2)
          Z(knode) = common_coords_all_nodes(node_ID,3)
        END DO
C
        DO knode=1,nnode  ! Do loop over all nodes of element element_ID 
C
C*** Natural Coordinates (�,�) (or (�,�,�) in 3D elements) of point (node) under consideration
C
          XI=XINODE(knode); ETA=ETANODE(knode); ZETA=ZETANODE(knode)  
C
C*** Calculate [N'] at (�,�) (or (�,�,�) in 3D elements)
C

         ! 3D Element
            
            DO knode=1,8
              DN(1,knode) = 1.D0/8.D0*XINODE(knode)*(1.D0 + ETANODE(knode)*ETA)*(1.D0 + ZETANODE(knode)*ZETA)
              DN(2,knode) = 1.D0/8.D0*ETANODE(knode)*(1.D0 + XINODE(knode)*XI)*(1.D0 + ZETANODE(knode)*ZETA)
              DN(3,knode) = 1.D0/8.D0*ZETANODE(knode)*(1.D0 + XINODE(knode)*XI)*(1.D0 + ETANODE(knode)*ETA)
            END DO
C
C*** Calculate [J]=[dx_j/d�_i] and det[J] at (�,�) (or (�,�,�) in 3D elements)
C
          
          ! 3D Element
            AJ = 0.D0
            DO I=1,8
              AJ(1,1) = AJ(1,1) + DN(1,I)*X(I)
              AJ(1,2) = AJ(1,2) + DN(1,I)*Y(I)
              AJ(1,3) = AJ(1,3) + DN(1,I)*Z(I)
C
              AJ(2,1) = AJ(2,1) + DN(2,I)*X(I)
              AJ(2,2) = AJ(2,2) + DN(2,I)*Y(I)
              AJ(2,3) = AJ(2,3) + DN(2,I)*Z(I)
C
              AJ(3,1) = AJ(3,1) + DN(3,I)*X(I)
              AJ(3,2) = AJ(3,2) + DN(3,I)*Y(I)
              AJ(3,3) = AJ(3,3) + DN(3,I)*Z(I)
            END DO
            ADETJ = AJ(1,1)*(AJ(2,2)*AJ(3,3) - AJ(3,2)*AJ(2,3))
     +        - AJ(1,2)*(AJ(2,1)*AJ(3,3) - AJ(3,1)*AJ(2,3))
     +        + AJ(1,3)*(AJ(2,1)*AJ(3,2) - AJ(3,1)*AJ(2,2))

C
C*** Store detJ for node knode of element element_ID in global array ADETJEL          
C
          ADETJEL(element_ID,knode) = ADETJ 
C
        END DO  ! Do loop over all nodes in element_ID Ends
      END DO    ! Do loop over all element labels Ends
C
      END
C 