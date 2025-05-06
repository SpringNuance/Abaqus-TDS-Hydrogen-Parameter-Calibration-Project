      subroutine umatht(u,dudt,dudg,flux,dfdt,dfdg,statev,temp,dtemp,
     1 dtemdx,time,dtime,predef,dpred,cmname,ntgrd,nstatv,props,nprops,
     2 coords,pnewdt,noel,npt,layer,kspt,kstep,kinc)
C
      include 'aba_param.inc'
C
      character*80 cmname
      INTEGER ntraps,kflag,Tflag,ntgrd,nstatv,nprops,noel,npt
      INTEGER kspt,kstep,kinc	  
      REAL*8 dudg(ntgrd),flux(ntgrd),dfdt(ntgrd),dfdg(ntgrd,ntgrd),
     1 statev(nstatv),dtemdx(ntgrd),time(2),predef(1),dpred(1),
     2 props(nprops),coords(3),u,dudt,temp,dtemp,pnewdt
C
      REAL*8 Wb(3),xK(3),xNt(3),du2,dudt2,cL
      REAL*8 D,D0,T,R,Vh,xNl,Ea,xNFe,AtMFe,AtMH
      	  
C
C Step-1: Read input data & Initialize
C	  
      cL=temp+dtemp
C
      T=props(1)
C 	  
      D0=props(2)
      Ea=props(3)  
C Lattice trap density ([sites/mm^3])	  
      xNl=props(4)	  
      kflag=props(5)
      Tflag=props(6)
      IF (Tflag.eq.1) THEN
        T = predef(1)+dpred(1)	  
      ENDIF	  
      ntraps = props(7)	  
C
C Partial molar volume of hydrogen ([mm^3/mol])
      Vh=2000.d0 
C Universal gas constant ([N*mm/(mol*K)])
      R=8314.5d0
C Fe atoms per unit of volume ([Fe atoms/mm^3])
      xNFe=8.4113d19 
C Molar weight of Fe ([kg/mol])
      AtMFe = 55.845d-3
C Molar weight of H ([kg/mol])	  
      AtMH = 1.00784d-3
C Lattice diffusion coefficient	  
      D=D0*exp(-Ea/(R*T))
C      
      do k1=1,ntraps
       Wb(k1)=props(7+2*k1-1)
       xNt(k1)=props(7+2*k1)
       xK(k1)=exp(Wb(k1)/(R*T))
      end do
C
C if (kflag = 0) -> traps not considered
      if (kflag.eq.0) then 
        do k1=1,ntraps		
          xNt(k1)=0.d0
        end do
      endif		
C
      du2=0.d0
      do k1=1,ntraps
        du2 = du2 - xNt(k1)*xK(k1)*(Wb(k1)/(R*T**2.d0))*xNl*cL*
     1    (1.d0 - cL/xNl)/(xNl + (xK(k1)-1.D0)*cL)**2.d0
      end do	  
      du2=du2*dpred(1)
C
      dudt2=0.d0
      do k1=1,ntraps
       dudt2=dudt2+xNt(k1)*xK(k1)*xNl/(((xK(k1)-1.D0)*cL+xNl)**2.d0)
      end do 
      dudt=1.d0+dudt2	   
      u=u+dudt*dtemp+du2
      do i=1,ntgrd
       dudg(i)=0.d0
       flux(i)=-D*dtemdx(i)
       dfdt(i)=0.d0
       dfdg(i,i)=-D
      end do
C     store the concentration in each trap, in all traps and in traps and lattice
      statev(ntraps+1)=0
      do k1=1,ntraps      
C      SDV1-3       	  
       statev(k1)=xNt(k1)*xK(k1)*cL/((xK(k1)-1.D0)*cL+xNl)
C      SDV4	   
       statev(ntraps+1)=statev(ntraps+1)+statev(k1)
      end do
C     SDV 5	  
      statev(ntraps+2)=cL+statev(ntraps+1)
C	  
C     store the concentration (expressed in wt ppm) in each trap, in all traps and in traps and lattice
      id=ntraps+2
      statev(ntraps+1+id)=0
      do k1=1,ntraps      
C      SDV6-8	  
       statev(k1+id) = statev(k1)*AtMH*1.d6/(xNFe*AtMFe)
C      SDV9	   
       statev(ntraps+1+id)=statev(ntraps+1+id)+statev(k1+id)
      end do
C     SDV10	  
      statev(ntraps+2+id)=cL*AtMH*1.d6/(xNFe*AtMFe)
     1	  +statev(ntraps+1+id)
C     SDV11	 
      statev(ntraps+3+id)=cL*AtMH*1.d6/(xNFe*AtMFe)
C     SDV12	 
      statev(ntraps+4+id)=u
      return
      end