1

   Abaqus 2023                                  Date 08-Dec-2024   Time 19:23:44
   For use by CSC_ IT CENTER FOR SCIENCE LTD under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     C O U P L E D - T E M P E R A T U R E - D I S P L A C E M E N T   A N A L Y S I S


                                                                                          

          FIXED TIME INCREMENTS
          TIME INCREMENT IS                                     360.    
          TIME PERIOD IS                                       1.800E+03

          CREEP AND SWELLING EFFECTS ARE OMITTED IN THIS STEP
          THE RATE DEPENDENCY WILL BE CONSIDERED IN THIS STEP

          CREEP AND SWELLING EFFECTS WILL BE IGNORED.

 CONVERGENCE TOLERANCE PARAMETERS FOR FORCE    
     CRITERION FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR DISP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE FORCE                               1.000E-02
     AVERAGE FORCE     IS TIME AVERAGE FORCE    
     ALTERNATE CRIT. FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO FORCE     RELATIVE TO TIME AVRG. FORCE         1.000E-05
     CRITERION FOR RESIDUAL FORCE     WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR DISP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL FORCE     FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO FORCE     REL. TO TIME AVRG. MAX. FORCE        1.000E-05
     CRITERION FOR ZERO DISP.    RELATIVE TO CHARACTERISTIC LENGTH     1.000E-08

 CONVERGENCE TOLERANCE PARAMETERS FOR HEAT FLUX
     CRITERION FOR RESIDUAL HEAT FLUX FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR TEMP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE HEAT FLUX                           1.000E-02
     AVERAGE HEAT FLUX IS TIME AVERAGE HEAT FLUX
     ALTERNATE CRIT. FOR RESIDUAL HEAT FLUX FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO HEAT FLUX RELATIVE TO TIME AVRG. HEAT FLUX     1.000E-05
     CRITERION FOR RESIDUAL HEAT FLUX WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR TEMP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL HEAT FLUX FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO HEAT FLUX REL. TO TIME AVRG. MAX. HEAT FLUX    1.000E-05

     VOLUMETRIC STRAIN COMPATIBILITY TOLERANCE FOR HYBRID SOLIDS       1.000E-05
     AXIAL STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS             1.000E-05
     TRANS. SHEAR STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS      1.000E-05
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P>P0          5.000E-03
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P=0.0         0.100    
     CONTACT FORCE ERROR TOLERANCE FOR CONVERT SDI=YES                 1.00    
     DISPLACEMENT COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS           1.000E-05
     ROTATION COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS               1.000E-05

 EQUILIBRIUM WILL BE CHECKED FOR SEVERE DISCONTINUITY ITERATIONS

 ITERATION CONTROL PARAMETERS:
     FIRST EQUILIBRIUM ITERATION FOR CONSECUTIVE DIVERGENCE CHECK              4
     EQUILIBRIUM ITERATION AFTER WHICH ALTERNATE RESIDUAL IS USED              9
     MAXIMUM EQUILIBRIUM ITERATIONS ALLOWED                                   16
     MAXIMUM ITERATIONS FOR SEVERE DISCONTINUITIES                            12
     GLOBAL STABILIZATION CONTROL IS NOT USED

          PRINT OF INCREMENT NUMBER, TIME, ETC., EVERY    1  INCREMENTS

     THE MAXIMUM NUMBER OF INCREMENTS IN THIS STEP IS                 1000000

          LARGE DISPLACEMENT THEORY WILL BE USED

          UNSYMMETRIC MATRIX STORAGE AND SOLUTION WILL BE USED

     LINEAR EXTRAPOLATION WILL BE USED

     CHARACTERISTIC ELEMENT LENGTH     1.329E-04

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     ELEMENT OPERATIONS WILL BE CARRIED OUT IN PARALLEL USING   8 THREADS ON 1 DOMAIN

     PARALLEL CONTACT TRACKING ENABLED


     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   360.    
 
     INITIAL ELEMENT LOAD BALANCE
            DOMAIN    1     THREAD    NUMBER OF ELEMENTS
                              1            15344
                              2            15472
                              3            15512
                              4            15464
                              5            15308
                              6            15544
                              7            15552
                              8            13064
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   138.39    
       ELAPSED SYSTEM TIME (SEC)    =   11.670    
       ELAPSED TOTAL CPU TIME (SEC) =   150.06    
       ELAPSED WALLCLOCK TIME (SEC) =           20

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  5.872E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         5.977E-16   AT NODE       4461   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.          8.66       AT NODE          1   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.         8.66       AT NODE          1   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   360.    ,  FRACTION OF STEP COMPLETED  0.200    
 STEP TIME COMPLETED        360.    ,  TOTAL TIME COMPLETED         360.    


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   360.    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   119.84    
       ELAPSED SYSTEM TIME (SEC)    =  0.53000    
       ELAPSED TOTAL CPU TIME (SEC) =   120.37    
       ELAPSED WALLCLOCK TIME (SEC) =           15

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  3.905E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.334E-16   AT NODE       4833   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.          2.15       AT NODE     119662   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        -8.66       AT NODE          1   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   360.    ,  FRACTION OF STEP COMPLETED  0.400    
 STEP TIME COMPLETED        720.    ,  TOTAL TIME COMPLETED         720.    


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   360.    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   124.14    
       ELAPSED SYSTEM TIME (SEC)    =  0.61000    
       ELAPSED TOTAL CPU TIME (SEC) =   124.75    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  2.969E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.934E-17   AT NODE       4462   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.          1.54       AT NODE      18642   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        -1.13       AT NODE     106795   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   360.    ,  FRACTION OF STEP COMPLETED  0.600    
 STEP TIME COMPLETED       1.080E+03,  TOTAL TIME COMPLETED        1.080E+03


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   360.    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   122.08    
       ELAPSED SYSTEM TIME (SEC)    =  0.53000    
       ELAPSED TOTAL CPU TIME (SEC) =   122.61    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  2.234E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -9.233E-18   AT NODE       4463   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.          1.17       AT NODE      79219   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -0.513       AT NODE     124912   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   360.    ,  FRACTION OF STEP COMPLETED  0.800    
 STEP TIME COMPLETED       1.440E+03,  TOTAL TIME COMPLETED        1.440E+03


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   360.    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   126.05    
       ELAPSED SYSTEM TIME (SEC)    =  0.51000    
       ELAPSED TOTAL CPU TIME (SEC) =   126.56    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  1.674E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.785E-18   AT NODE       4835   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         0.873       AT NODE      18955   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -0.346       AT NODE      80125   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   360.    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED       1.800E+03,  TOTAL TIME COMPLETED        1.800E+03
1

   Abaqus 2023                                  Date 08-Dec-2024   Time 19:26:04
   For use by CSC_ IT CENTER FOR SCIENCE LTD under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    2     INCREMENT     1     STEP TIME    0.00    


                        S T E P       2     C O U P L E D - T E M P E R A T U R E - D I S P L A C E M E N T   A N A L Y S I S


                                                                                          

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                 7.16    
          AND A TOTAL TIME PERIOD OF                            716.    
          THE MINIMUM TIME INCREMENT ALLOWED IS                1.000E-07
          THE MAXIMUM TIME INCREMENT ALLOWED IS                 7.16    

          CREEP AND SWELLING EFFECTS ARE OMITTED IN THIS STEP

     THE SIZE OF THE TIME INCREMENT IS CONTROLLED BY -
          THE TEMPERATURE CHANGE PER INCREMENT NOT EXCEEDING           1.000E+03
          THE RATE DEPENDENCY WILL BE CONSIDERED IN THIS STEP

          CREEP AND SWELLING EFFECTS WILL BE IGNORED.

 CONVERGENCE TOLERANCE PARAMETERS FOR FORCE    
     CRITERION FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR DISP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INIT. VALUE OF TIME AVG. FORCE     IS TIME AVG. FORCE     IN PREVIOUS STEP
     AVERAGE FORCE     IS TIME AVERAGE FORCE    
     ALTERNATE CRIT. FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO FORCE     RELATIVE TO TIME AVRG. FORCE         1.000E-05
     CRITERION FOR RESIDUAL FORCE     WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR DISP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL FORCE     FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO FORCE     REL. TO TIME AVRG. MAX. FORCE        1.000E-05
     CRITERION FOR ZERO DISP.    RELATIVE TO CHARACTERISTIC LENGTH     1.000E-08

 CONVERGENCE TOLERANCE PARAMETERS FOR HEAT FLUX
     CRITERION FOR RESIDUAL HEAT FLUX FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR TEMP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INIT. VALUE OF TIME AVG. HEAT FLUX IS TIME AVG. HEAT FLUX IN PREVIOUS STEP
     AVERAGE HEAT FLUX IS TIME AVERAGE HEAT FLUX
     ALTERNATE CRIT. FOR RESIDUAL HEAT FLUX FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO HEAT FLUX RELATIVE TO TIME AVRG. HEAT FLUX     1.000E-05
     CRITERION FOR RESIDUAL HEAT FLUX WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR TEMP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL HEAT FLUX FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO HEAT FLUX REL. TO TIME AVRG. MAX. HEAT FLUX    1.000E-05

     VOLUMETRIC STRAIN COMPATIBILITY TOLERANCE FOR HYBRID SOLIDS       1.000E-05
     AXIAL STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS             1.000E-05
     TRANS. SHEAR STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS      1.000E-05
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P>P0          5.000E-03
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P=0.0         0.100    
     CONTACT FORCE ERROR TOLERANCE FOR CONVERT SDI=YES                 1.00    
     DISPLACEMENT COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS           1.000E-05
     ROTATION COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS               1.000E-05

 EQUILIBRIUM WILL BE CHECKED FOR SEVERE DISCONTINUITY ITERATIONS

 TIME INCREMENTATION CONTROL PARAMETERS:
     FIRST EQUILIBRIUM ITERATION FOR CONSECUTIVE DIVERGENCE CHECK              4
     EQUILIBRIUM ITERATION AT WHICH LOG. CONVERGENCE RATE CHECK BEGINS         8
     EQUILIBRIUM ITERATION AFTER WHICH ALTERNATE RESIDUAL IS USED              9
     MAXIMUM EQUILIBRIUM ITERATIONS ALLOWED                                   16
     EQUILIBRIUM ITERATION COUNT FOR CUT-BACK IN NEXT INCREMENT               10
     MAXIMUM EQUILIB. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      4
     MAXIMUM ITERATIONS FOR SEVERE DISCONTINUITIES                            50
     MAXIMUM ATTEMPTS ALLOWED IN AN INCREMENT                                  5
     MAXIMUM DISCON. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      50
     MAXIMUM CONTACT AUGMENTATIONS FOR *SURFACE BEHAVIOR,AUGMENTED LAGRANGE   50
     CONSECUTIVE INCS MEETING TIME INTEG. TOL. FOR TIME INC. INCREASE          3
     CUT-BACK FACTOR AFTER DIVERGENCE                                 0.2500    
     CUT-BACK FACTOR FOR TOO SLOW CONVERGENCE                         0.5000    
     CUT-BACK FACTOR AFTER TOO MANY EQUILIBRIUM ITERATIONS            0.7500    
     CUT-BACK FACTOR AFTER TOO MANY SEVERE DISCONTINUITY ITERATIONS   0.2500    
     CUT-BACK FACTOR AFTER PROBLEMS IN ELEMENT ASSEMBLY               0.2500    
     INCREASE FACTOR AFTER TWO INCREMENTS THAT CONVERGE QUICKLY        1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED                       1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DYNAMICS)            1.250    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DIFFUSION)           2.000    
     MINIMUM TIME INCREMENT RATIO FOR EXTRAPOLATION TO OCCUR          0.1000    
     CUT-BACK FACTOR WHEN TIME INTEGRATION ACCURACY TOL. EXCEEDED     0.8500    
     MAX. RATIO OF TIME INTEG. MEASURE TO TOL. FOR TIME INC. INCREASE 0.7500    
     INCREASE FACTOR FOR TIME INCREMENT WHEN POSSIBLE                 0.8000    
     MIN. TIME INC. INCREASE FACTOR IN LINEAR TRANSIENT PROBLEMS      0.9500    
     MAX. RATIO OF TIME INCREMENT TO STABILITY LIMIT                   1.000    
     FRACTION OF STABILITY LIMIT FOR NEW TIME INCREMENT               0.9500    
     TIME INCREMENT INCREASE FACTOR BEFORE A TIME POINT                1.000    
     GLOBAL STABILIZATION CONTROL IS NOT USED

          PRINT OF INCREMENT NUMBER, TIME, ETC., EVERY    1  INCREMENTS

     THE MAXIMUM NUMBER OF INCREMENTS IN THIS STEP IS                 1000000

          LARGE DISPLACEMENT THEORY WILL BE USED

          UNSYMMETRIC MATRIX STORAGE AND SOLUTION WILL BE USED

     LINEAR EXTRAPOLATION WILL BE USED

     CHARACTERISTIC ELEMENT LENGTH     1.329E-04

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     ELEMENT OPERATIONS WILL BE CARRIED OUT IN PARALLEL USING   8 THREADS ON 1 DOMAIN

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
     INITIAL ELEMENT LOAD BALANCE
            DOMAIN    1     THREAD    NUMBER OF ELEMENTS
                              1            15040
                              2            15192
                              3            15144
                              4            15224
                              5            15204
                              6            15072
                              7            15160
                              8            15224
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   126.52    
       ELAPSED SYSTEM TIME (SEC)    =   5.4400    
       ELAPSED TOTAL CPU TIME (SEC) =   131.96    
       ELAPSED WALLCLOCK TIME (SEC) =           17

 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116326 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116330 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116525 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116529 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116340 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116336 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116539 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116535 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116531 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116334 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116897 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116900 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116910 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116906 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116902 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116730 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116718 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116722 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116724 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 116728 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 88177 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 88181 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 88376 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 88380 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 88191 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 88187 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 88390 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 88386 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 88382 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 88185 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 126840 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 126836 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 127039 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 127035 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 132289 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 132086 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 132090 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 132285 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 126826 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 126830 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 127025 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 127029 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 132279 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 126834 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 127031 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 132076 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 132080 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 132084 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 132275 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: Solver problem. Zero pivot when processing D.O.F. 11 of 49 nodes. 
             The nodes have been identified in node set 
             WarnNodeSolvProbZeroPiv_11_2_1_1_1.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.619E-02   TIME AVG. FORCE       2.619E-02
 LARGEST RESIDUAL FORCE             1.138E-03   AT NODE       1341   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.663E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         6.083E-17   AT NODE       4649   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.727E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.727E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.08    
       ELAPSED SYSTEM TIME (SEC)    =  0.69000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.77    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.618E-02   TIME AVG. FORCE       2.618E-02
 LARGEST RESIDUAL FORCE            -1.603E-08   AT NODE       6765   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        9.966E-11   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.663E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.512E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.727E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.421E-06   AT NODE      88478   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.727E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  1.000E-02
 STEP TIME COMPLETED        7.16    ,  TOTAL TIME COMPLETED        1.807E+03


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.49    
       ELAPSED SYSTEM TIME (SEC)    =  0.64000    
       ELAPSED TOTAL CPU TIME (SEC) =   129.13    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.238E-02   TIME AVG. FORCE       4.428E-02
 LARGEST RESIDUAL FORCE            -2.711E-08   AT NODE       6996   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.893E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.652E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.905E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.718E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.094E-03   AT NODE      28846   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.718E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  2.000E-02
 STEP TIME COMPLETED        14.3    ,  TOTAL TIME COMPLETED        1.814E+03


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.51    
       ELAPSED SYSTEM TIME (SEC)    =  0.57000    
       ELAPSED TOTAL CPU TIME (SEC) =   129.08    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.356E-02   TIME AVG. FORCE       6.071E-02
 LARGEST RESIDUAL FORCE            -3.892E-08   AT NODE       5456   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.892E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.641E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.857E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.708E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.798E-04   AT NODE      88227   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.708E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  3.000E-02
 STEP TIME COMPLETED        21.5    ,  TOTAL TIME COMPLETED        1.821E+03


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   129.97    
       ELAPSED SYSTEM TIME (SEC)    =  0.61000    
       ELAPSED TOTAL CPU TIME (SEC) =   130.58    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.125       TIME AVG. FORCE       7.672E-02
 LARGEST RESIDUAL FORCE            -5.577E-08   AT NODE       5456   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.892E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.630E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.809E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.699E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.550E-04   AT NODE      88227   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.699E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  4.000E-02
 STEP TIME COMPLETED        28.6    ,  TOTAL TIME COMPLETED        1.829E+03


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   129.74    
       ELAPSED SYSTEM TIME (SEC)    =  0.56000    
       ELAPSED TOTAL CPU TIME (SEC) =   130.30    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.156       TIME AVG. FORCE       9.256E-02
 LARGEST RESIDUAL FORCE            -7.261E-08   AT NODE       5417   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.892E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.619E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.760E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.690E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.642E-04   AT NODE      88227   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.690E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  5.000E-02
 STEP TIME COMPLETED        35.8    ,  TOTAL TIME COMPLETED        1.836E+03


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   130.14    
       ELAPSED SYSTEM TIME (SEC)    =  0.61000    
       ELAPSED TOTAL CPU TIME (SEC) =   130.75    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.187       TIME AVG. FORCE       0.108    
 LARGEST RESIDUAL FORCE            -8.944E-08   AT NODE       5417   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.892E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.609E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.712E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.681E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.019E-04   AT NODE      28826   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.681E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  6.000E-02
 STEP TIME COMPLETED        43.0    ,  TOTAL TIME COMPLETED        1.843E+03


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.97    
       ELAPSED SYSTEM TIME (SEC)    =  0.56000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.53    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.218       TIME AVG. FORCE       0.124    
 LARGEST RESIDUAL FORCE            -1.063E-07   AT NODE       5417   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.891E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.598E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.664E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.671E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.561E-04   AT NODE      19147   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.671E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  7.000E-02
 STEP TIME COMPLETED        50.1    ,  TOTAL TIME COMPLETED        1.850E+03


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.15    
       ELAPSED SYSTEM TIME (SEC)    =  0.59000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.74    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.249       TIME AVG. FORCE       0.140    
 LARGEST RESIDUAL FORCE            -1.231E-07   AT NODE       5417   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.891E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.588E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.616E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.662E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.213E-04   AT NODE      19147   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.662E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  8.000E-02
 STEP TIME COMPLETED        57.3    ,  TOTAL TIME COMPLETED        1.857E+03


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.77    
       ELAPSED SYSTEM TIME (SEC)    =  0.61000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.38    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.281       TIME AVG. FORCE       0.155    
 LARGEST RESIDUAL FORCE            -1.399E-07   AT NODE       5456   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.891E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.577E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.568E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.653E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.944E-04   AT NODE      28820   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.653E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  9.000E-02
 STEP TIME COMPLETED        64.4    ,  TOTAL TIME COMPLETED        1.864E+03


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.44    
       ELAPSED SYSTEM TIME (SEC)    =  0.56000    
       ELAPSED TOTAL CPU TIME (SEC) =   129.00    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.312       TIME AVG. FORCE       0.171    
 LARGEST RESIDUAL FORCE            -1.567E-07   AT NODE       5417   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.891E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.567E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.521E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.644E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.821E-04   AT NODE      88403   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.644E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.100    
 STEP TIME COMPLETED        71.6    ,  TOTAL TIME COMPLETED        1.872E+03


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   125.47    
       ELAPSED SYSTEM TIME (SEC)    =  0.55000    
       ELAPSED TOTAL CPU TIME (SEC) =   126.02    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.343       TIME AVG. FORCE       0.187    
 LARGEST RESIDUAL FORCE            -1.735E-07   AT NODE       5417   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.890E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.557E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.473E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.635E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.735E-04   AT NODE      88403   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.635E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.110    
 STEP TIME COMPLETED        78.8    ,  TOTAL TIME COMPLETED        1.879E+03


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   119.94    
       ELAPSED SYSTEM TIME (SEC)    =  0.44000    
       ELAPSED TOTAL CPU TIME (SEC) =   120.38    
       ELAPSED WALLCLOCK TIME (SEC) =           15

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.374       TIME AVG. FORCE       0.202    
 LARGEST RESIDUAL FORCE            -1.902E-07   AT NODE       5417   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.890E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.546E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.426E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.625E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.656E-04   AT NODE      88424   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.625E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.120    
 STEP TIME COMPLETED        85.9    ,  TOTAL TIME COMPLETED        1.886E+03


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   123.73    
       ELAPSED SYSTEM TIME (SEC)    =  0.56000    
       ELAPSED TOTAL CPU TIME (SEC) =   124.29    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.405       TIME AVG. FORCE       0.218    
 LARGEST RESIDUAL FORCE            -2.070E-07   AT NODE       5417   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.890E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.536E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.378E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.616E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.597E-04   AT NODE      88424   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.616E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.130    
 STEP TIME COMPLETED        93.1    ,  TOTAL TIME COMPLETED        1.893E+03


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   125.32    
       ELAPSED SYSTEM TIME (SEC)    =  0.50000    
       ELAPSED TOTAL CPU TIME (SEC) =   125.82    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.436       TIME AVG. FORCE       0.233    
 LARGEST RESIDUAL FORCE            -2.238E-07   AT NODE       5456   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.890E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.526E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.331E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.607E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.540E-04   AT NODE      88424   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.607E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.140    
 STEP TIME COMPLETED        100.    ,  TOTAL TIME COMPLETED        1.900E+03


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   125.62    
       ELAPSED SYSTEM TIME (SEC)    =  0.52000    
       ELAPSED TOTAL CPU TIME (SEC) =   126.14    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.467       TIME AVG. FORCE       0.249    
 LARGEST RESIDUAL FORCE            -2.405E-07   AT NODE       5417   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.890E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.516E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.285E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.598E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.486E-04   AT NODE      29023   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.598E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.150    
 STEP TIME COMPLETED        107.    ,  TOTAL TIME COMPLETED        1.907E+03


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.05    
       ELAPSED SYSTEM TIME (SEC)    =  0.57000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.62    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.499       TIME AVG. FORCE       0.265    
 LARGEST RESIDUAL FORCE            -2.573E-07   AT NODE       5456   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.889E-10   AT NODE       6943   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.506E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.238E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.589E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.444E-04   AT NODE      29023   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.589E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.160    
 STEP TIME COMPLETED        115.    ,  TOTAL TIME COMPLETED        1.915E+03


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.86    
       ELAPSED SYSTEM TIME (SEC)    =  0.60000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.46    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.530       TIME AVG. FORCE       0.280    
 LARGEST RESIDUAL FORCE             1.908E-04   AT NODE      21487   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.346E-10   AT NODE      23899   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.496E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.227E-21   AT NODE       8225   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.580E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.403E-04   AT NODE      29023   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.580E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.170    
 STEP TIME COMPLETED        122.    ,  TOTAL TIME COMPLETED        1.922E+03


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.78    
       ELAPSED SYSTEM TIME (SEC)    =  0.60000    
       ELAPSED TOTAL CPU TIME (SEC) =   129.38    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.561       TIME AVG. FORCE       0.296    
 LARGEST RESIDUAL FORCE            -1.383E-03   AT NODE      81515   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.794E-09   AT NODE      23918   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.486E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.464E-19   AT NODE      20441   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.571E-02   AT NODE       9937   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.288E-03   AT NODE      88443   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.571E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.180    
 STEP TIME COMPLETED        129.    ,  TOTAL TIME COMPLETED        1.929E+03


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.18    
       ELAPSED SYSTEM TIME (SEC)    =  0.57000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.75    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.592       TIME AVG. FORCE       0.311    
 LARGEST RESIDUAL FORCE            -2.375E-03   AT NODE      81663   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.003E-08   AT NODE      23559   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.477E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.056E-19   AT NODE     132180   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.932E-02   AT NODE      29233   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.427E-02   AT NODE      29233   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.79    
       ELAPSED SYSTEM TIME (SEC)    =  0.56000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.35    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.592       TIME AVG. FORCE       0.311    
 LARGEST RESIDUAL FORCE            -5.430E-05   AT NODE      81119   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.290E-10   AT NODE      81518   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.477E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.748E-20   AT NODE      21897   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.924E-02   AT NODE      29233   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -9.086E-05   AT NODE      67970   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.924E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.190    
 STEP TIME COMPLETED        136.    ,  TOTAL TIME COMPLETED        1.936E+03


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.86    
       ELAPSED SYSTEM TIME (SEC)    =  0.57000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.43    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.623       TIME AVG. FORCE       0.327    
 LARGEST RESIDUAL FORCE            -2.644E-03   AT NODE      81793   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        9.656E-09   AT NODE      23636   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.467E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.874E-19   AT NODE      36403   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.361E-02   AT NODE      29254   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.590E-02   AT NODE     116597   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.45    
       ELAPSED SYSTEM TIME (SEC)    =  0.52000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.97    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.623       TIME AVG. FORCE       0.327    
 LARGEST RESIDUAL FORCE             2.056E-05   AT NODE      20750   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.671E-10   AT NODE      21426   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.467E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.970E-20   AT NODE      20443   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.351E-02   AT NODE      29254   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.116E-04   AT NODE     111530   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.351E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.200    
 STEP TIME COMPLETED        143.    ,  TOTAL TIME COMPLETED        1.943E+03


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.87    
       ELAPSED SYSTEM TIME (SEC)    =  0.56000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.43    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.654       TIME AVG. FORCE       0.343    
 LARGEST RESIDUAL FORCE             2.312E-03   AT NODE      20972   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        9.922E-09   AT NODE      21385   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.457E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.086E-18   AT NODE     110987   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.371E-02   AT NODE      19181   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.584E-02   AT NODE      90532   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.15    
       ELAPSED SYSTEM TIME (SEC)    =  0.62000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.77    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.654       TIME AVG. FORCE       0.343    
 LARGEST RESIDUAL FORCE            -3.478E-05   AT NODE      81751   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.582E-10   AT NODE      81339   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.457E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.188E-20   AT NODE     105737   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.360E-02   AT NODE      19181   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.217E-04   AT NODE      50809   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.360E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.210    
 STEP TIME COMPLETED        150.    ,  TOTAL TIME COMPLETED        1.950E+03


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.78    
       ELAPSED SYSTEM TIME (SEC)    =  0.57000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.35    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.685       TIME AVG. FORCE       0.358    
 LARGEST RESIDUAL FORCE            -1.958E-03   AT NODE      73052   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.018E-08   AT NODE      23574   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.448E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.067E-18   AT NODE     105737   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.440E-02   AT NODE      88985   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.025E-02   AT NODE      94134   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.84    
       ELAPSED SYSTEM TIME (SEC)    =  0.54000    
       ELAPSED TOTAL CPU TIME (SEC) =   129.38    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.685       TIME AVG. FORCE       0.358    
 LARGEST RESIDUAL FORCE            -4.821E-05   AT NODE      20130   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.914E-10   AT NODE      20973   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.448E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.315E-20   AT NODE      22373   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.430E-02   AT NODE      88985   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.247E-04   AT NODE     111559   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.430E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.220    
 STEP TIME COMPLETED        158.    ,  TOTAL TIME COMPLETED        1.958E+03


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   124.18    
       ELAPSED SYSTEM TIME (SEC)    =  0.46000    
       ELAPSED TOTAL CPU TIME (SEC) =   124.64    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.715       TIME AVG. FORCE       0.374    
 LARGEST RESIDUAL FORCE             2.256E-03   AT NODE      62588   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.006E-08   AT NODE      21337   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.438E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -8.836E-19   AT NODE     110993   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.338E-02   AT NODE      80220   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.351E-02   AT NODE      31853   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   125.16    
       ELAPSED SYSTEM TIME (SEC)    =  0.51000    
       ELAPSED TOTAL CPU TIME (SEC) =   125.67    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.715       TIME AVG. FORCE       0.374    
 LARGEST RESIDUAL FORCE            -5.591E-05   AT NODE     105747   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.410E-10   AT NODE      21338   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.438E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.149E-20   AT NODE      20289   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.326E-02   AT NODE      80220   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.315E-04   AT NODE      52795   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.326E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.230    
 STEP TIME COMPLETED        165.    ,  TOTAL TIME COMPLETED        1.965E+03


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   118.91    
       ELAPSED SYSTEM TIME (SEC)    =  0.52000    
       ELAPSED TOTAL CPU TIME (SEC) =   119.43    
       ELAPSED WALLCLOCK TIME (SEC) =           15

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.746       TIME AVG. FORCE       0.389    
 LARGEST RESIDUAL FORCE            -1.870E-03   AT NODE     121698   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.013E-08   AT NODE      84646   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.429E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -9.133E-19   AT NODE      73580   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.243E-02   AT NODE      80229   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.176E-02   AT NODE      35154   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.243E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.240    
 STEP TIME COMPLETED        172.    ,  TOTAL TIME COMPLETED        1.972E+03


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   124.82    
       ELAPSED SYSTEM TIME (SEC)    =  0.53000    
       ELAPSED TOTAL CPU TIME (SEC) =   125.35    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.775       TIME AVG. FORCE       0.405    
 LARGEST RESIDUAL FORCE            -2.605E-03   AT NODE      62398   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.091E-08   AT NODE      83043   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.420E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.420E-18   AT NODE      68667   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.312E-02   AT NODE      23883   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.511E-02   AT NODE      35160   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   124.81    
       ELAPSED SYSTEM TIME (SEC)    =  0.53000    
       ELAPSED TOTAL CPU TIME (SEC) =   125.34    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.775       TIME AVG. FORCE       0.405    
 LARGEST RESIDUAL FORCE            -3.232E-05   AT NODE      21969   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.940E-10   AT NODE     122238   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.420E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.013E-20   AT NODE     126952   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.301E-02   AT NODE      23883   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.478E-04   AT NODE     101529   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.301E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.250    
 STEP TIME COMPLETED        179.    ,  TOTAL TIME COMPLETED        1.979E+03


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   126.29    
       ELAPSED SYSTEM TIME (SEC)    =  0.54000    
       ELAPSED TOTAL CPU TIME (SEC) =   126.83    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.804       TIME AVG. FORCE       0.420    
 LARGEST RESIDUAL FORCE            -2.568E-03   AT NODE     126957   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.139E-08   AT NODE      23778   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.410E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.633E-18   AT NODE      58056   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.122E-02   AT NODE      23251   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.016E-02   AT NODE      90730   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   126.85    
       ELAPSED SYSTEM TIME (SEC)    =  0.54000    
       ELAPSED TOTAL CPU TIME (SEC) =   127.39    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.804       TIME AVG. FORCE       0.420    
 LARGEST RESIDUAL FORCE             4.744E-05   AT NODE      29096   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.880E-10   AT NODE      62402   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.410E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.177E-19   AT NODE      20451   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.110E-02   AT NODE      23251   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.537E-04   AT NODE      37254   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.110E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.260    
 STEP TIME COMPLETED        186.    ,  TOTAL TIME COMPLETED        1.986E+03


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.89    
       ELAPSED SYSTEM TIME (SEC)    =  0.56000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.45    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.832       TIME AVG. FORCE       0.435    
 LARGEST RESIDUAL FORCE             2.824E-03   AT NODE      73266   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.212E-08   AT NODE      23363   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.401E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.902E-18   AT NODE     122684   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.980E-02   AT NODE      23252   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.105E-02   AT NODE      94279   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.57    
       ELAPSED SYSTEM TIME (SEC)    =  0.54000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.11    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.832       TIME AVG. FORCE       0.435    
 LARGEST RESIDUAL FORCE             5.691E-05   AT NODE      36430   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.915E-10   AT NODE      83589   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.401E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -8.497E-20   AT NODE      22287   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.969E-02   AT NODE      23252   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.638E-04   AT NODE      96597   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.969E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.270    
 STEP TIME COMPLETED        193.    ,  TOTAL TIME COMPLETED        1.993E+03


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.64    
       ELAPSED SYSTEM TIME (SEC)    =  0.59000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.23    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.859       TIME AVG. FORCE       0.450    
 LARGEST RESIDUAL FORCE            -2.752E-03   AT NODE      67646   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.274E-08   AT NODE      24572   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.392E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.182E-18   AT NODE      96434   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.724E-02   AT NODE      23879   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.545E-02   AT NODE      21730   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.71    
       ELAPSED SYSTEM TIME (SEC)    =  0.53000    
       ELAPSED TOTAL CPU TIME (SEC) =   129.24    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.859       TIME AVG. FORCE       0.450    
 LARGEST RESIDUAL FORCE             6.501E-05   AT NODE      31571   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.783E-10   AT NODE      62983   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.392E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -8.293E-20   AT NODE      22365   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.713E-02   AT NODE      23879   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.780E-04   AT NODE      94318   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.713E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.280    
 STEP TIME COMPLETED        200.    ,  TOTAL TIME COMPLETED        2.000E+03


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   129.09    
       ELAPSED SYSTEM TIME (SEC)    =  0.56000    
       ELAPSED TOTAL CPU TIME (SEC) =   129.65    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.885       TIME AVG. FORCE       0.465    
 LARGEST RESIDUAL FORCE            -2.467E-03   AT NODE     126771   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.347E-08   AT NODE      84671   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.383E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.420E-18   AT NODE      91045   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.631E-02   AT NODE      23878   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.429E-02   AT NODE      30993   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.17    
       ELAPSED SYSTEM TIME (SEC)    =  0.54000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.71    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.885       TIME AVG. FORCE       0.465    
 LARGEST RESIDUAL FORCE             3.150E-05   AT NODE     101404   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.733E-10   AT NODE      57378   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.383E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.357E-19   AT NODE      20455   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.619E-02   AT NODE      23878   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.906E-04   AT NODE      96708   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.619E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.290    
 STEP TIME COMPLETED        208.    ,  TOTAL TIME COMPLETED        2.008E+03


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.11    
       ELAPSED SYSTEM TIME (SEC)    =  0.53000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.64    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.912       TIME AVG. FORCE       0.480    
 LARGEST RESIDUAL FORCE             3.012E-03   AT NODE     132026   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.410E-08   AT NODE      24004   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.374E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.510E-18   AT NODE      32325   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.527E-02   AT NODE      23256   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.392E-02   AT NODE      35267   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.29    
       ELAPSED SYSTEM TIME (SEC)    =  0.52000    
       ELAPSED TOTAL CPU TIME (SEC) =   127.81    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.912       TIME AVG. FORCE       0.480    
 LARGEST RESIDUAL FORCE            -4.701E-05   AT NODE      96449   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.213E-10   AT NODE      68382   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.374E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.129E-19   AT NODE      32714   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.516E-02   AT NODE      23256   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.089E-04   AT NODE      32353   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.516E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.300    
 STEP TIME COMPLETED        215.    ,  TOTAL TIME COMPLETED        2.015E+03


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.90    
       ELAPSED SYSTEM TIME (SEC)    =  0.59000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.49    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.939       TIME AVG. FORCE       0.495    
 LARGEST RESIDUAL FORCE            -3.351E-03   AT NODE      73091   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.516E-08   AT NODE      23784   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.365E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.312E-18   AT NODE      32461   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.379E-02   AT NODE      23252   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.334E-02   AT NODE      74587   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.54    
       ELAPSED SYSTEM TIME (SEC)    =  0.53000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.07    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.939       TIME AVG. FORCE       0.495    
 LARGEST RESIDUAL FORCE            -6.316E-05   AT NODE     100725   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.400E-10   AT NODE      37957   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.365E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.355E-19   AT NODE      37945   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.369E-02   AT NODE      23252   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.288E-04   AT NODE      37951   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.369E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.310    
 STEP TIME COMPLETED        222.    ,  TOTAL TIME COMPLETED        2.022E+03


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   119.91    
       ELAPSED SYSTEM TIME (SEC)    =  0.53000    
       ELAPSED TOTAL CPU TIME (SEC) =   120.44    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.965       TIME AVG. FORCE       0.510    
 LARGEST RESIDUAL FORCE            -3.559E-03   AT NODE      72897   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.591E-08   AT NODE      83070   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.356E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.361E-18   AT NODE      37695   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.434E-02   AT NODE      23252   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.640E-02   AT NODE      89307   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   123.42    
       ELAPSED SYSTEM TIME (SEC)    =  0.50000    
       ELAPSED TOTAL CPU TIME (SEC) =   123.92    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.965       TIME AVG. FORCE       0.510    
 LARGEST RESIDUAL FORCE            -5.295E-05   AT NODE      31582   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.618E-10   AT NODE      68392   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.356E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.409E-19   AT NODE     116481   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.422E-02   AT NODE      23252   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.489E-04   AT NODE      38185   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.422E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.320    
 STEP TIME COMPLETED        229.    ,  TOTAL TIME COMPLETED        2.029E+03


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   120.77    
       ELAPSED SYSTEM TIME (SEC)    =  0.45000    
       ELAPSED TOTAL CPU TIME (SEC) =   121.22    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.990       TIME AVG. FORCE       0.524    
 LARGEST RESIDUAL FORCE            -3.135E-03   AT NODE      73101   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.666E-08   AT NODE      27170   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.347E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.952E-18   AT NODE      38291   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.465E-02   AT NODE      23253   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.076E-02   AT NODE     105168   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   124.76    
       ELAPSED SYSTEM TIME (SEC)    =  0.48000    
       ELAPSED TOTAL CPU TIME (SEC) =   125.24    
       ELAPSED WALLCLOCK TIME (SEC) =           15

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.990       TIME AVG. FORCE       0.524    
 LARGEST RESIDUAL FORCE            -8.129E-05   AT NODE      83276   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.965E-10   AT NODE      83817   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.347E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.953E-19   AT NODE      91731   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.453E-02   AT NODE      23253   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.595E-04   AT NODE      40884   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.453E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.330    
 STEP TIME COMPLETED        236.    ,  TOTAL TIME COMPLETED        2.036E+03


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   122.77    
       ELAPSED SYSTEM TIME (SEC)    =  0.53000    
       ELAPSED TOTAL CPU TIME (SEC) =   123.30    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.02       TIME AVG. FORCE       0.539    
 LARGEST RESIDUAL FORCE            -3.937E-03   AT NODE     126992   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.758E-08   AT NODE      26177   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.338E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.756E-18   AT NODE      33152   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.478E-02   AT NODE      23253   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.751E-02   AT NODE      43313   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   126.42    
       ELAPSED SYSTEM TIME (SEC)    =  0.58000    
       ELAPSED TOTAL CPU TIME (SEC) =   127.00    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.02       TIME AVG. FORCE       0.539    
 LARGEST RESIDUAL FORCE            -1.724E-04   AT NODE     127187   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.635E-10   AT NODE      85393   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.338E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.682E-19   AT NODE      32215   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.465E-02   AT NODE      23253   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.573E-04   AT NODE      97501   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.465E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.340    
 STEP TIME COMPLETED        243.    ,  TOTAL TIME COMPLETED        2.043E+03


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.84    
       ELAPSED SYSTEM TIME (SEC)    =  0.51000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.35    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.04       TIME AVG. FORCE       0.553    
 LARGEST RESIDUAL FORCE            -4.681E-03   AT NODE     132051   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.993E-08   AT NODE      35736   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.329E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.676E-18   AT NODE      33247   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.473E-02   AT NODE      23253   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.001E-02   AT NODE      48549   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.38    
       ELAPSED SYSTEM TIME (SEC)    =  0.57000    
       ELAPSED TOTAL CPU TIME (SEC) =   127.95    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.04       TIME AVG. FORCE       0.553    
 LARGEST RESIDUAL FORCE             1.253E-04   AT NODE      73304   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.661E-10   AT NODE      26565   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.329E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.755E-19   AT NODE      20622   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.460E-02   AT NODE      23253   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.557E-04   AT NODE     102671   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.460E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.350    
 STEP TIME COMPLETED        251.    ,  TOTAL TIME COMPLETED        2.051E+03


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.01    
       ELAPSED SYSTEM TIME (SEC)    =  0.59000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.60    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.07       TIME AVG. FORCE       0.567    
 LARGEST RESIDUAL FORCE            -4.390E-03   AT NODE      72918   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.273E-08   AT NODE      33466   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.321E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -8.720E-18   AT NODE      92347   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.516E-02   AT NODE     106678   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.988E-02   AT NODE     110493   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.28    
       ELAPSED SYSTEM TIME (SEC)    =  0.54000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.82    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.07       TIME AVG. FORCE       0.567    
 LARGEST RESIDUAL FORCE             1.735E-04   AT NODE      94762   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.787E-10   AT NODE      85761   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.321E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.721E-19   AT NODE      22273   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.522E-02   AT NODE     106678   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.426E-04   AT NODE     110458   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.522E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.360    
 STEP TIME COMPLETED        258.    ,  TOTAL TIME COMPLETED        2.058E+03


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   129.43    
       ELAPSED SYSTEM TIME (SEC)    =  0.60000    
       ELAPSED TOTAL CPU TIME (SEC) =   130.03    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.581    
 LARGEST RESIDUAL FORCE            -3.969E-03   AT NODE      97928   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.642E-08   AT NODE      38814   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.312E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -9.081E-18   AT NODE      92469   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.483E-02   AT NODE      47620   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.556E-02   AT NODE     110520   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.36    
       ELAPSED SYSTEM TIME (SEC)    =  0.61000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.97    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.581    
 LARGEST RESIDUAL FORCE             2.146E-04   AT NODE      92628   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -6.353E-10   AT NODE      85738   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.312E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.820E-19   AT NODE      92256   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.489E-02   AT NODE      47620   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.351E-04   AT NODE      49089   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.489E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.370    
 STEP TIME COMPLETED        265.    ,  TOTAL TIME COMPLETED        2.065E+03


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   129.02    
       ELAPSED SYSTEM TIME (SEC)    =  0.55000    
       ELAPSED TOTAL CPU TIME (SEC) =   129.57    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.12       TIME AVG. FORCE       0.595    
 LARGEST RESIDUAL FORCE             5.591E-03   AT NODE      39002   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.235E-08   AT NODE      38916   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.303E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.299E-17   AT NODE      33486   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.866E-02   AT NODE     106840   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.326E-02   AT NODE     101572   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   127.02    
       ELAPSED SYSTEM TIME (SEC)    =  0.58000    
       ELAPSED TOTAL CPU TIME (SEC) =   127.60    
       ELAPSED WALLCLOCK TIME (SEC) =           16

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.12       TIME AVG. FORCE       0.595    
 LARGEST RESIDUAL FORCE             2.217E-04   AT NODE      85857   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.421E-10   AT NODE      59560   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.303E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.398E-19   AT NODE     118719   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.870E-02   AT NODE     106840   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.414E-04   AT NODE      19667   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.870E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.380    
 STEP TIME COMPLETED        272.    ,  TOTAL TIME COMPLETED        2.072E+03


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   128.13    
       ELAPSED SYSTEM TIME (SEC)    =  0.60000    
       ELAPSED TOTAL CPU TIME (SEC) =   128.73    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.14       TIME AVG. FORCE       0.609    
 LARGEST RESIDUAL FORCE            -5.489E-03   AT NODE      30670   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.480E-08   AT NODE      59938   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.295E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.799E-17   AT NODE      89693   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.940E-02   AT NODE     112090   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.419E-02   AT NODE     115758   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   130.60    
       ELAPSED SYSTEM TIME (SEC)    =  0.51000    
       ELAPSED TOTAL CPU TIME (SEC) =   131.11    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.14       TIME AVG. FORCE       0.609    
 LARGEST RESIDUAL FORCE            -3.527E-04   AT NODE     129164   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.118E-09   AT NODE      17536   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.295E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -9.044E-19   AT NODE     102717   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.944E-02   AT NODE     112090   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.011E-04   AT NODE      17594   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.944E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.390    
 STEP TIME COMPLETED        279.    ,  TOTAL TIME COMPLETED        2.079E+03


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   132.48    
       ELAPSED SYSTEM TIME (SEC)    =  0.54000    
       ELAPSED TOTAL CPU TIME (SEC) =   133.02    
       ELAPSED WALLCLOCK TIME (SEC) =           17

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.16       TIME AVG. FORCE       0.623    
 LARGEST RESIDUAL FORCE            -1.270E-02   AT NODE      70141   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.915E-08   AT NODE      18000   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.286E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.683E-17   AT NODE      44116   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.811E-02   AT NODE     117340   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.570E-02   AT NODE      80708   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   112.72    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   113.11    
       ELAPSED WALLCLOCK TIME (SEC) =           14

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.16       TIME AVG. FORCE       0.623    
 LARGEST RESIDUAL FORCE            -4.497E-04   AT NODE     110437   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.956E-09   AT NODE      77454   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.286E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.994E-18   AT NODE     108245   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.819E-02   AT NODE     117340   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.204E-04   AT NODE      44279   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.819E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.400    
 STEP TIME COMPLETED        286.    ,  TOTAL TIME COMPLETED        2.086E+03


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   105.38    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.76    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.18       TIME AVG. FORCE       0.637    
 LARGEST RESIDUAL FORCE            -1.623E-02   AT NODE     136506   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.187E-08   AT NODE      28332   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.278E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.091E-17   AT NODE      49525   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.913E-02   AT NODE     111940   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.515E-02   AT NODE     106840   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   100.37    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   100.74    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.18       TIME AVG. FORCE       0.637    
 LARGEST RESIDUAL FORCE             8.977E-04   AT NODE      97952   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.898E-09   AT NODE      98160   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.278E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.856E-18   AT NODE      94506   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.920E-02   AT NODE     111940   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.442E-04   AT NODE      46674   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.920E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.410    
 STEP TIME COMPLETED        294.    ,  TOTAL TIME COMPLETED        2.094E+03


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   101.28    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   101.66    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.20       TIME AVG. FORCE       0.650    
 LARGEST RESIDUAL FORCE            -2.085E-02   AT NODE     134928   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.923E-08   AT NODE      66578   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.269E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.249E-17   AT NODE      34514   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.841E-02   AT NODE      52699   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.764E-02   AT NODE     112090   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   96.650    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   97.050    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.20       TIME AVG. FORCE       0.650    
 LARGEST RESIDUAL FORCE             1.702E-03   AT NODE      70201   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.207E-09   AT NODE      18335   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.269E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.744E-18   AT NODE      44662   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.847E-02   AT NODE      52699   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.438E-04   AT NODE      88083   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.847E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.420    
 STEP TIME COMPLETED        301.    ,  TOTAL TIME COMPLETED        2.101E+03


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   96.260    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.630    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.22       TIME AVG. FORCE       0.663    
 LARGEST RESIDUAL FORCE            -2.151E-02   AT NODE     129408   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.146E-07   AT NODE      28415   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.261E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.050E-17   AT NODE     130170   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.965E-02   AT NODE     111781   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.512E-02   AT NODE     122590   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   96.530    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.900    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.22       TIME AVG. FORCE       0.663    
 LARGEST RESIDUAL FORCE             1.578E-03   AT NODE      76765   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.922E-09   AT NODE      71749   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.261E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.639E-18   AT NODE     125293   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.969E-02   AT NODE     111781   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.203E-04   AT NODE      28683   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.969E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.430    
 STEP TIME COMPLETED        308.    ,  TOTAL TIME COMPLETED        2.108E+03


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   96.180    
       ELAPSED SYSTEM TIME (SEC)    =  0.43000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.610    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.23       TIME AVG. FORCE       0.676    
 LARGEST RESIDUAL FORCE            -3.013E-02   AT NODE      25042   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.582E-07   AT NODE      18202   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.253E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.097E-16   AT NODE      34860   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.195E-02   AT NODE     117031   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.810E-02   AT NODE      77837   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   96.160    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.550    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.23       TIME AVG. FORCE       0.676    
 LARGEST RESIDUAL FORCE            -2.829E-03   AT NODE      89566   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.785E-09   AT NODE      12081   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.253E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -9.554E-18   AT NODE     121084   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.204E-02   AT NODE     117031   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -9.253E-04   AT NODE      28683   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.204E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.440    
 STEP TIME COMPLETED        315.    ,  TOTAL TIME COMPLETED        2.115E+03


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   96.660    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   97.030    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.24       TIME AVG. FORCE       0.689    
 LARGEST RESIDUAL FORCE            -8.111E-02   AT NODE      11277   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.365E-07   AT NODE      11431   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.245E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.972E-16   AT NODE      14062   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.174E-02   AT NODE     111617   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.506E-02   AT NODE      52867   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   96.000    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.390    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.24       TIME AVG. FORCE       0.689    
 LARGEST RESIDUAL FORCE            -8.461E-03   AT NODE      12676   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.711E-08   AT NODE      11393   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.245E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.458E-17   AT NODE      15449   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.180E-02   AT NODE     111617   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.551E-03   AT NODE      88083   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   96.100    
       ELAPSED SYSTEM TIME (SEC)    =  0.42000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.520    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.24       TIME AVG. FORCE       0.689    
 LARGEST RESIDUAL FORCE            -1.381E-04   AT NODE      16775   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.834E-10   AT NODE      15450   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.245E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.083E-18   AT NODE      11636   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.180E-02   AT NODE     111617   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -9.243E-05   AT NODE      23591   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.180E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.450    
 STEP TIME COMPLETED        322.    ,  TOTAL TIME COMPLETED        2.122E+03


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   96.300    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.690    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.26       TIME AVG. FORCE       0.701    
 LARGEST RESIDUAL FORCE            -0.833       AT NODE         49   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.938E-07   AT NODE       9887   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.236E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.380E-15   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.385E-02   AT NODE      63009   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.627E-02   AT NODE      19742   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.670    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.050    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.26       TIME AVG. FORCE       0.701    
 LARGEST RESIDUAL FORCE            -3.218E-02   AT NODE       9943   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.012E-07   AT NODE       6696   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.236E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -8.227E-16   AT NODE       9685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.402E-02   AT NODE      63009   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.378E-03   AT NODE      23888   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   96.440    
       ELAPSED SYSTEM TIME (SEC)    =  0.41000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.850    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.26       TIME AVG. FORCE       0.701    
 LARGEST RESIDUAL FORCE            -9.222E-04   AT NODE      15784   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.163E-09   AT NODE       9820   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.236E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.202E-17   AT NODE       9703   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.405E-02   AT NODE      63009   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.368E-04   AT NODE      28684   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.405E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.460    
 STEP TIME COMPLETED        329.    ,  TOTAL TIME COMPLETED        2.129E+03


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   96.290    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.660    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.27       TIME AVG. FORCE       0.713    
 LARGEST RESIDUAL FORCE             0.538       AT NODE      10670   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.279E-06   AT NODE       6412   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.228E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.287E-14   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.870E-02   AT NODE     116686   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.781E-02   AT NODE      57933   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.870    
       ELAPSED SYSTEM TIME (SEC)    =  0.35000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.220    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.27       TIME AVG. FORCE       0.713    
 LARGEST RESIDUAL FORCE             4.647E-02   AT NODE      10819   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.070E-07   AT NODE      10225   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.228E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.687E-15   AT NODE      10840   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.903E-02   AT NODE     116686   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -8.700E-03   AT NODE      23591   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   96.070    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.430    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.27       TIME AVG. FORCE       0.713    
 LARGEST RESIDUAL FORCE            -7.712E-04   AT NODE      12271   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.248E-09   AT NODE      12821   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.228E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.267E-17   AT NODE      10841   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.911E-02   AT NODE     116686   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.159E-04   AT NODE      23591   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.911E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.470    
 STEP TIME COMPLETED        337.    ,  TOTAL TIME COMPLETED        2.137E+03


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   97.840    
       ELAPSED SYSTEM TIME (SEC)    =  0.35000    
       ELAPSED TOTAL CPU TIME (SEC) =   98.190    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.29       TIME AVG. FORCE       0.726    
 LARGEST RESIDUAL FORCE            -0.554       AT NODE       6382   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.743E-07   AT NODE       4926   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.221E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.021E-14   AT NODE        158   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.057E-02   AT NODE     111237   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.481E-02   AT NODE     106181   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.030    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.400    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.29       TIME AVG. FORCE       0.726    
 LARGEST RESIDUAL FORCE             1.868E-02   AT NODE      10587   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.054E-08   AT NODE      12821   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.221E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.786E-16   AT NODE      10824   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.075E-02   AT NODE     111237   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.996E-03   AT NODE      23591   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.700    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.090    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.29       TIME AVG. FORCE       0.726    
 LARGEST RESIDUAL FORCE            -2.090E-03   AT NODE      19894   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.171E-09   AT NODE      12813   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.221E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.081E-17   AT NODE      10574   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.079E-02   AT NODE     111237   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.812E-04   AT NODE      22331   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.079E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.480    
 STEP TIME COMPLETED        344.    ,  TOTAL TIME COMPLETED        2.144E+03


  INCREMENT    49 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.380    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   93.760    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.31       TIME AVG. FORCE       0.737    
 LARGEST RESIDUAL FORCE            -0.617       AT NODE        686   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.893E-07   AT NODE       6394   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.213E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.741E-15   AT NODE        122   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.759E-02   AT NODE     116494   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.776E-02   AT NODE     111232   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.590    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   93.970    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.31       TIME AVG. FORCE       0.737    
 LARGEST RESIDUAL FORCE            -1.225E-02   AT NODE      10227   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.618E-08   AT NODE      13283   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.213E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.433E-16   AT NODE         82   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.786E-02   AT NODE     116494   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.641E-03   AT NODE      23888   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.950    
       ELAPSED SYSTEM TIME (SEC)    =  0.41000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.360    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.31       TIME AVG. FORCE       0.737    
 LARGEST RESIDUAL FORCE            -5.863E-04   AT NODE      19891   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -8.092E-10   AT NODE      79153   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.213E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.071E-18   AT NODE      10824   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.790E-02   AT NODE     116494   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.752E-04   AT NODE      18229   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.790E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.490    
 STEP TIME COMPLETED        351.    ,  TOTAL TIME COMPLETED        2.151E+03


  INCREMENT    50 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.780    
       ELAPSED SYSTEM TIME (SEC)    =  0.35000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.130    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.32       TIME AVG. FORCE       0.749    
 LARGEST RESIDUAL FORCE            -0.432       AT NODE       8350   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.563E-07   AT NODE       6892   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.205E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.677E-15   AT NODE       2630   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.936E-02   AT NODE     132244   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.062E-02   AT NODE     105988   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.840    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.210    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.32       TIME AVG. FORCE       0.749    
 LARGEST RESIDUAL FORCE             1.643E-02   AT NODE       9577   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.432E-08   AT NODE      23524   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.205E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.457E-16   AT NODE       3956   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.959E-02   AT NODE     132244   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.181E-03   AT NODE      23591   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.900    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.270    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.32       TIME AVG. FORCE       0.749    
 LARGEST RESIDUAL FORCE            -1.557E-04   AT NODE      80984   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.904E-10   AT NODE      21610   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.205E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.511E-18   AT NODE       9576   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.960E-02   AT NODE     132244   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -9.287E-05   AT NODE      60327   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.960E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.500    
 STEP TIME COMPLETED        358.    ,  TOTAL TIME COMPLETED        2.158E+03


  INCREMENT    51 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.840    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.210    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.34       TIME AVG. FORCE       0.761    
 LARGEST RESIDUAL FORCE            -0.372       AT NODE       9573   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.044E-07   AT NODE       4892   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.197E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.162E-15   AT NODE       3956   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.879E-02   AT NODE     105784   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.389E-02   AT NODE     116284   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.770    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.140    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.34       TIME AVG. FORCE       0.761    
 LARGEST RESIDUAL FORCE             6.774E-03   AT NODE      10705   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.529E-09   AT NODE      23527   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.197E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.195E-16   AT NODE       3956   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.866E-02   AT NODE     105784   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.139E-03   AT NODE      23591   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.330    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.730    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.34       TIME AVG. FORCE       0.761    
 LARGEST RESIDUAL FORCE            -2.779E-05   AT NODE      98224   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.878E-10   AT NODE     135112   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.197E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.425E-18   AT NODE       3956   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.865E-02   AT NODE     105784   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.094E-05   AT NODE     129671   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.865E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.510    
 STEP TIME COMPLETED        365.    ,  TOTAL TIME COMPLETED        2.165E+03


  INCREMENT    52 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.270    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.650    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.35       TIME AVG. FORCE       0.772    
 LARGEST RESIDUAL FORCE            -0.162       AT NODE       6143   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.595E-07   AT NODE      10155   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.190E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.617E-15   AT NODE       8352   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.212E-02   AT NODE      29676   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.764E-02   AT NODE      89499   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.240    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.600    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.35       TIME AVG. FORCE       0.772    
 LARGEST RESIDUAL FORCE             3.111E-03   AT NODE       9782   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.530E-09   AT NODE      81876   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.190E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.857E-17   AT NODE       8352   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.186E-02   AT NODE      29676   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.196E-03   AT NODE      23887   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   4.186E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.520    
 STEP TIME COMPLETED        372.    ,  TOTAL TIME COMPLETED        2.172E+03


  INCREMENT    53 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.270    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.640    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.36       TIME AVG. FORCE       0.783    
 LARGEST RESIDUAL FORCE            -0.339       AT NODE       5295   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.939E-07   AT NODE       4925   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.182E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.610E-15   AT NODE       7529   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.469E-02   AT NODE      88933   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.680E-02   AT NODE      40554   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.140    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.510    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.36       TIME AVG. FORCE       0.783    
 LARGEST RESIDUAL FORCE            -9.701E-02   AT NODE       5284   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -6.098E-09   AT NODE       5115   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.182E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.184E-16   AT NODE       7526   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.458E-02   AT NODE      88933   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.088E-03   AT NODE      23592   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.200    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.570    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.36       TIME AVG. FORCE       0.783    
 LARGEST RESIDUAL FORCE             1.525E-05   AT NODE      60047   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.875E-10   AT NODE       5283   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.182E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.281E-17   AT NODE       7526   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.458E-02   AT NODE      88933   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        5.440E-05   AT NODE      23888   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   4.458E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.530    
 STEP TIME COMPLETED        379.    ,  TOTAL TIME COMPLETED        2.179E+03


  INCREMENT    54 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.040    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.410    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.37       TIME AVG. FORCE       0.794    
 LARGEST RESIDUAL FORCE            -0.570       AT NODE       5249   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.259E-07   AT NODE      26195   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.175E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.288E-15   AT NODE       4638   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.830E-02   AT NODE      88921   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.211E-02   AT NODE      56256   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.120    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.490    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.37       TIME AVG. FORCE       0.794    
 LARGEST RESIDUAL FORCE             8.057E-04   AT NODE     129499   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -6.119E-09   AT NODE       5080   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.175E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.478E-16   AT NODE       7421   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.825E-02   AT NODE      88921   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.743E-03   AT NODE      23592   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   4.825E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.540    
 STEP TIME COMPLETED        387.    ,  TOTAL TIME COMPLETED        2.187E+03


  INCREMENT    55 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.820    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.180    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.38       TIME AVG. FORCE       0.805    
 LARGEST RESIDUAL FORCE            -0.370       AT NODE       5237   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.362E-07   AT NODE      29601   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.167E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.013E-15   AT NODE       4701   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.591E-02   AT NODE      71958   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.999E-02   AT NODE      19373   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.000    
       ELAPSED SYSTEM TIME (SEC)    =  0.33000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.330    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.38       TIME AVG. FORCE       0.805    
 LARGEST RESIDUAL FORCE             2.467E-02   AT NODE       4629   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.381E-09   AT NODE      88667   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.167E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.443E-16   AT NODE       4536   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.586E-02   AT NODE      71958   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.754E-03   AT NODE      23887   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.800    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.170    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.38       TIME AVG. FORCE       0.805    
 LARGEST RESIDUAL FORCE             3.408E-06   AT NODE       4626   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.064E-11   AT NODE       1343   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.167E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.181E-18   AT NODE       4665   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.586E-02   AT NODE      71958   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.729E-05   AT NODE      23592   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   5.586E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.550    
 STEP TIME COMPLETED        394.    ,  TOTAL TIME COMPLETED        2.194E+03


  INCREMENT    56 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.750    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.110    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.38       TIME AVG. FORCE       0.815    
 LARGEST RESIDUAL FORCE            -0.674       AT NODE       5361   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.309E-07   AT NODE       1760   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.160E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.722E-15   AT NODE       4566   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.089E-02   AT NODE      19373   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.033E-02   AT NODE      45773   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.960    
       ELAPSED SYSTEM TIME (SEC)    =  0.34000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.300    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.39       TIME AVG. FORCE       0.815    
 LARGEST RESIDUAL FORCE            -8.759E-02   AT NODE       1373   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.088E-08   AT NODE        207   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.160E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.163E-16   AT NODE       4629   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.087E-02   AT NODE      19373   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.333E-03   AT NODE      23592   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.860    
       ELAPSED SYSTEM TIME (SEC)    =  0.44000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.300    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.39       TIME AVG. FORCE       0.815    
 LARGEST RESIDUAL FORCE             2.600E-03   AT NODE       4852   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.417E-09   AT NODE       1759   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.160E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.691E-17   AT NODE       4629   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.087E-02   AT NODE      19373   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.242E-04   AT NODE      23591   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   6.087E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.560    
 STEP TIME COMPLETED        401.    ,  TOTAL TIME COMPLETED        2.201E+03


  INCREMENT    57 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.520    
       ELAPSED SYSTEM TIME (SEC)    =  0.35000    
       ELAPSED TOTAL CPU TIME (SEC) =   93.870    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.39       TIME AVG. FORCE       0.825    
 LARGEST RESIDUAL FORCE            -0.308       AT NODE       6065   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.291E-06   AT NODE        342   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.153E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.825E-14   AT NODE        119   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.981E-02   AT NODE      51007   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.227E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.740    
       ELAPSED SYSTEM TIME (SEC)    =  0.41000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.150    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.39       TIME AVG. FORCE       0.825    
 LARGEST RESIDUAL FORCE            -2.153E-02   AT NODE        785   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.862E-08   AT NODE       2188   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.153E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.754E-16   AT NODE       9669   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.990E-02   AT NODE      51007   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.599E-02   AT NODE      23591   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.880    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.270    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.39       TIME AVG. FORCE       0.825    
 LARGEST RESIDUAL FORCE             6.160E-06   AT NODE      10043   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -4.921E-10   AT NODE      29429   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.153E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.975E-18   AT NODE       3128   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.990E-02   AT NODE      51007   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.406E-04   AT NODE      29720   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   5.990E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.570    
 STEP TIME COMPLETED        408.    ,  TOTAL TIME COMPLETED        2.208E+03


  INCREMENT    58 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.440    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.830    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.39       TIME AVG. FORCE       0.835    
 LARGEST RESIDUAL FORCE            -0.563       AT NODE       2067   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.496E-07   AT NODE        480   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.145E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         5.312E-15   AT NODE        119   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.083E-02   AT NODE      66709   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.023E-02   AT NODE      30128   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.530    
       ELAPSED SYSTEM TIME (SEC)    =  0.41000    
       ELAPSED TOTAL CPU TIME (SEC) =   93.940    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.39       TIME AVG. FORCE       0.835    
 LARGEST RESIDUAL FORCE            -7.163E-03   AT NODE       3537   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.903E-09   AT NODE       1305   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.145E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.473E-16   AT NODE       4435   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.085E-02   AT NODE      66709   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        4.616E-03   AT NODE      23591   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.320    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   93.720    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.39       TIME AVG. FORCE       0.835    
 LARGEST RESIDUAL FORCE            -6.115E-07   AT NODE       3534   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.307E-11   AT NODE        446   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.145E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.481E-18   AT NODE       3537   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.085E-02   AT NODE      66709   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.024E-05   AT NODE      20051   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   6.085E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.580    
 STEP TIME COMPLETED        415.    ,  TOTAL TIME COMPLETED        2.215E+03


  INCREMENT    59 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.860    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.250    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.39       TIME AVG. FORCE       0.844    
 LARGEST RESIDUAL FORCE            -0.137       AT NODE       4437   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.061E-07   AT NODE       1760   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.138E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.739E-15   AT NODE       4671   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.978E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.207E-03   AT NODE      35305   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.040    
       ELAPSED SYSTEM TIME (SEC)    =  0.43000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.470    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.39       TIME AVG. FORCE       0.844    
 LARGEST RESIDUAL FORCE            -9.443E-05   AT NODE       3324   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.984E-09   AT NODE      88835   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.138E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.710E-17   AT NODE       4437   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.978E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -9.037E-04   AT NODE      23592   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   5.978E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.590    
 STEP TIME COMPLETED        422.    ,  TOTAL TIME COMPLETED        2.222E+03


  INCREMENT    60 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.250    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.620    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.40       TIME AVG. FORCE       0.854    
 LARGEST RESIDUAL FORCE            -0.203       AT NODE       3438   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.309E-07   AT NODE       3132   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.131E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.786E-15   AT NODE       3224   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.795E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.458E-03   AT NODE      35305   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.500    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.870    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.40       TIME AVG. FORCE       0.854    
 LARGEST RESIDUAL FORCE            -5.922E-04   AT NODE       6245   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.122E-09   AT NODE      29595   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.131E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.183E-17   AT NODE       3335   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.796E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        5.792E-04   AT NODE      29412   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   5.796E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.600    
 STEP TIME COMPLETED        430.    ,  TOTAL TIME COMPLETED        2.230E+03


  INCREMENT    61 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.340    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.710    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.40       TIME AVG. FORCE       0.862    
 LARGEST RESIDUAL FORCE            -0.196       AT NODE       4996   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.495E-07   AT NODE        373   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.123E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         3.965E-15   AT NODE        158   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.523E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.475E-03   AT NODE      63821   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.710    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.070    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.40       TIME AVG. FORCE       0.862    
 LARGEST RESIDUAL FORCE            -3.947E-05   AT NODE       7560   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.206E-09   AT NODE      27300   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.123E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.993E-17   AT NODE       7322   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.525E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.871E-03   AT NODE      23591   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   5.525E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.610    
 STEP TIME COMPLETED        437.    ,  TOTAL TIME COMPLETED        2.237E+03


  INCREMENT    62 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.750    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.140    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.40       TIME AVG. FORCE       0.871    
 LARGEST RESIDUAL FORCE             3.429E-02   AT NODE       5033   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.770E-08   AT NODE       7156   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.116E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.569E-15   AT NODE       3335   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.115E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.426E-03   AT NODE      56241   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.670    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.060    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.40       TIME AVG. FORCE       0.871    
 LARGEST RESIDUAL FORCE             1.329E-05   AT NODE      81619   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.127E-09   AT NODE      29434   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.116E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.401E-17   AT NODE       4641   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.115E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.149E-04   AT NODE      23592   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   5.115E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.620    
 STEP TIME COMPLETED        444.    ,  TOTAL TIME COMPLETED        2.244E+03


  INCREMENT    63 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.590    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.950    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.40       TIME AVG. FORCE       0.880    
 LARGEST RESIDUAL FORCE            -2.047E-03   AT NODE       5368   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.731E-08   AT NODE       6876   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.109E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.034E-15   AT NODE       3326   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.781E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.953E-03   AT NODE     131697   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   4.781E-02
          TIME INCREMENT REDUCED TO    2.92     TO MEET NEXT OUTPUT TIME POINT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.630    
 STEP TIME COMPLETED        451.    ,  TOTAL TIME COMPLETED        2.251E+03


  INCREMENT    64 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   2.92    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.410    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.780    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.41       TIME AVG. FORCE       0.888    
 LARGEST RESIDUAL FORCE             4.090E-04   AT NODE       7526   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         7.304E-06   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.909E-08   AT NODE        372   DOF  2
   INSTANCE: CHD2                                                                            
 ESTIMATE OF DISP.    CORRECTION   -9.288E-11
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE HEAT FLUX                  1.106E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.232E-15   AT NODE       3956   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.024E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.854E-03   AT NODE      87798   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.024E-02
          TIME INCREMENT REDUCED TO    4.24     TO MEET NEXT OUTPUT TIME POINT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   2.92    ,  FRACTION OF STEP COMPLETED  0.634    
 STEP TIME COMPLETED        454.    ,  TOTAL TIME COMPLETED        2.254E+03


  INCREMENT    65 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   4.24    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.590    
       ELAPSED SYSTEM TIME (SEC)    =  0.33000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.920    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.41       TIME AVG. FORCE       0.896    
 LARGEST RESIDUAL FORCE             1.085E-02   AT NODE       1238   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.061E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.665E-08   AT NODE       5122   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.102E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.492E-16   AT NODE       4656   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.714E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.254E-03   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.360    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.760    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.41       TIME AVG. FORCE       0.896    
 LARGEST RESIDUAL FORCE            -3.051E-06   AT NODE      81991   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.061E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.679E-10   AT NODE      88836   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.102E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         5.589E-18   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.714E-02   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.087E-04   AT NODE      23591   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.714E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   4.24    ,  FRACTION OF STEP COMPLETED  0.640    
 STEP TIME COMPLETED        458.    ,  TOTAL TIME COMPLETED        2.258E+03


  INCREMENT    66 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.360    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.720    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.41       TIME AVG. FORCE       0.904    
 LARGEST RESIDUAL FORCE            -2.452E-02   AT NODE       5311   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -7.056E-08   AT NODE        340   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.095E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.787E-16   AT NODE       9057   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.235E-02   AT NODE      28568   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.092E-03   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.430    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.800    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.41       TIME AVG. FORCE       0.904    
 LARGEST RESIDUAL FORCE             2.093E-05   AT NODE      21468   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.893E-09   AT NODE      29601   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.095E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         9.278E-18   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.235E-02   AT NODE      28568   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.928E-04   AT NODE      23592   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   4.235E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.650    
 STEP TIME COMPLETED        465.    ,  TOTAL TIME COMPLETED        2.265E+03


  INCREMENT    67 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.650    
       ELAPSED SYSTEM TIME (SEC)    =  0.35000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.000    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.41       TIME AVG. FORCE       0.911    
 LARGEST RESIDUAL FORCE            -3.689E-02   AT NODE       6279   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.186E-08   AT NODE       7153   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.088E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.859E-16   AT NODE       4874   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.094E-02   AT NODE      28568   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.271E-03   AT NODE      56241   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.800    
       ELAPSED SYSTEM TIME (SEC)    =  0.42000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.220    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.41       TIME AVG. FORCE       0.911    
 LARGEST RESIDUAL FORCE             2.502E-05   AT NODE      21343   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.574E-09   AT NODE      29435   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.088E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.243E-17   AT NODE       8859   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.095E-02   AT NODE      28568   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.541E-04   AT NODE      23887   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   4.095E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.660    
 STEP TIME COMPLETED        473.    ,  TOTAL TIME COMPLETED        2.273E+03


  INCREMENT    68 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.040    
       ELAPSED SYSTEM TIME (SEC)    =  0.41000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.450    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.41       TIME AVG. FORCE       0.919    
 LARGEST RESIDUAL FORCE             1.561E-03   AT NODE      10471   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.323E-08   AT NODE       6876   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.081E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.992E-16   AT NODE       3236   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.967E-02   AT NODE      28568   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.153E-03   AT NODE      66709   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.967E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.670    
 STEP TIME COMPLETED        480.    ,  TOTAL TIME COMPLETED        2.280E+03


  INCREMENT    69 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.450    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.850    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.41       TIME AVG. FORCE       0.926    
 LARGEST RESIDUAL FORCE            -0.120       AT NODE       8961   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.829E-08   AT NODE       6877   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.074E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.887E-16   AT NODE       3338   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.838E-02   AT NODE      28568   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.738E-03   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.690    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.050    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.41       TIME AVG. FORCE       0.926    
 LARGEST RESIDUAL FORCE            -3.571E-05   AT NODE       8965   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.128E-09   AT NODE      29435   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.074E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.346E-17   AT NODE       8958   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.838E-02   AT NODE      28568   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -8.192E-04   AT NODE      23592   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.838E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.680    
 STEP TIME COMPLETED        487.    ,  TOTAL TIME COMPLETED        2.287E+03


  INCREMENT    70 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.890    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.280    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.42       TIME AVG. FORCE       0.933    
 LARGEST RESIDUAL FORCE            -2.076E-02   AT NODE       8759   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.713E-08   AT NODE      29761   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.067E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.214E-16   AT NODE       8958   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.732E-02   AT NODE      87798   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.305E-03   AT NODE      19358   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.740    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.100    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.42       TIME AVG. FORCE       0.933    
 LARGEST RESIDUAL FORCE             5.281E-05   AT NODE      21469   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.183E-09   AT NODE      29435   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.067E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.685E-17   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.733E-02   AT NODE      87798   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.732E-04   AT NODE      23592   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.733E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.690    
 STEP TIME COMPLETED        494.    ,  TOTAL TIME COMPLETED        2.294E+03


  INCREMENT    71 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.280    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.640    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.42       TIME AVG. FORCE       0.940    
 LARGEST RESIDUAL FORCE             8.431E-04   AT NODE      21378   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.242E-08   AT NODE      29885   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.060E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.033E-16   AT NODE       3221   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.712E-02   AT NODE      87798   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.994E-03   AT NODE      41325   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.712E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.700    
 STEP TIME COMPLETED        501.    ,  TOTAL TIME COMPLETED        2.301E+03


  INCREMENT    72 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.420    
       ELAPSED SYSTEM TIME (SEC)    =  0.41000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.830    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.42       TIME AVG. FORCE       0.946    
 LARGEST RESIDUAL FORCE             1.251E-03   AT NODE       9963   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.317E-07   AT NODE       6877   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.053E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -8.603E-16   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.666E-02   AT NODE      87798   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.911E-03   AT NODE      51793   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.666E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.710    
 STEP TIME COMPLETED        508.    ,  TOTAL TIME COMPLETED        2.308E+03


  INCREMENT    73 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.370    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.730    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.42       TIME AVG. FORCE       0.953    
 LARGEST RESIDUAL FORCE            -2.164E-03   AT NODE       9685   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.199E-07   AT NODE      29086   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.046E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.795E-16   AT NODE       9685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.527E-02   AT NODE      87798   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.380E-03   AT NODE      77982   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.527E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.720    
 STEP TIME COMPLETED        516.    ,  TOTAL TIME COMPLETED        2.316E+03


  INCREMENT    74 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.120    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.480    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.42       TIME AVG. FORCE       0.959    
 LARGEST RESIDUAL FORCE             7.273E-03   AT NODE       4877   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.340E-08   AT NODE      30299   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.040E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         8.789E-16   AT NODE       4668   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.396E-02   AT NODE      87798   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.158E-03   AT NODE      88691   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.000    
       ELAPSED SYSTEM TIME (SEC)    =  0.31000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.310    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.42       TIME AVG. FORCE       0.959    
 LARGEST RESIDUAL FORCE             9.690E-05   AT NODE     132023   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.646E-09   AT NODE      29435   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.040E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         3.236E-17   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.396E-02   AT NODE      87798   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.789E-04   AT NODE      23887   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.396E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.730    
 STEP TIME COMPLETED        523.    ,  TOTAL TIME COMPLETED        2.323E+03


  INCREMENT    75 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.080    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.470    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.42       TIME AVG. FORCE       0.965    
 LARGEST RESIDUAL FORCE            -1.320E-02   AT NODE       4644   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.553E-07   AT NODE      88493   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.033E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.068E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.281E-02   AT NODE      87798   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.468E-03   AT NODE      88323   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.870    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.240    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.42       TIME AVG. FORCE       0.965    
 LARGEST RESIDUAL FORCE             1.327E-04   AT NODE     126773   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -6.259E-09   AT NODE      29435   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.033E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         3.690E-17   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.281E-02   AT NODE      87798   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.035E-03   AT NODE      23592   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.281E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.740    
 STEP TIME COMPLETED        530.    ,  TOTAL TIME COMPLETED        2.330E+03


  INCREMENT    76 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.820    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.200    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.42       TIME AVG. FORCE       0.971    
 LARGEST RESIDUAL FORCE             8.516E-04   AT NODE      21250   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.796E-07   AT NODE      29085   DOF  2
   INSTANCE: CHD2                                                                            
 ESTIMATE OF DISP.    CORRECTION    3.991E-10
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE HEAT FLUX                  1.026E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.020E-15   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.307E-02   AT NODE      29293   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.628E-03   AT NODE      29107   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.307E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.750    
 STEP TIME COMPLETED        537.    ,  TOTAL TIME COMPLETED        2.337E+03


  INCREMENT    77 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.590    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.950    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.42       TIME AVG. FORCE       0.977    
 LARGEST RESIDUAL FORCE            -9.952E-04   AT NODE      16112   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.731E-07   AT NODE      29086   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.020E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.018E-15   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.533E-02   AT NODE      29293   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.967E-03   AT NODE      29102   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.533E-02
          TIME INCREMENT REDUCED TO    5.84     TO MEET NEXT OUTPUT TIME POINT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.760    
 STEP TIME COMPLETED        544.    ,  TOTAL TIME COMPLETED        2.344E+03


  INCREMENT    78 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   5.84    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.890    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.260    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE       0.983    
 LARGEST RESIDUAL FORCE            -4.909E-04   AT NODE        753   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.461E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.416E-07   AT NODE      29265   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.014E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.050E-15   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.177E-02   AT NODE      88691   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.427E-03   AT NODE      88509   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.177E-02
          TIME INCREMENT REDUCED TO    1.32     TO MEET NEXT OUTPUT TIME POINT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   5.84    ,  FRACTION OF STEP COMPLETED  0.768    
 STEP TIME COMPLETED        550.    ,  TOTAL TIME COMPLETED        2.350E+03


  INCREMENT    79 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   1.32    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.580    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.940    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE       0.988    
 LARGEST RESIDUAL FORCE             9.536E-05   AT NODE      15791   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         3.302E-06   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.111E-08   AT NODE       7027   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.013E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.056E-16   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -8.541E-03   AT NODE      88691   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.698E-03   AT NODE      28916   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -8.541E-03
          TIME INCREMENT REDUCED TO    2.68     TO MEET NEXT OUTPUT TIME POINT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   1.32    ,  FRACTION OF STEP COMPLETED  0.770    
 STEP TIME COMPLETED        551.    ,  TOTAL TIME COMPLETED        2.351E+03


  INCREMENT    80 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   2.68    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.460    
       ELAPSED SYSTEM TIME (SEC)    =  0.35000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.810    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE       0.994    
 LARGEST RESIDUAL FORCE            -2.340E-04   AT NODE      73077   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         6.704E-06   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.873E-08   AT NODE      29594   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.010E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.438E-16   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.669E-02   AT NODE      88691   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.509E-03   AT NODE      88107   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -1.669E-02
          TIME INCREMENT REDUCED TO    4.48     TO MEET NEXT OUTPUT TIME POINT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   2.68    ,  FRACTION OF STEP COMPLETED  0.774    
 STEP TIME COMPLETED        554.    ,  TOTAL TIME COMPLETED        2.354E+03


  INCREMENT    81 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   4.48    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.160    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.550    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE       0.999    
 LARGEST RESIDUAL FORCE            -1.793E-03   AT NODE       9170   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.121E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.377E-08   AT NODE      29587   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.006E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.280E-16   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.693E-02   AT NODE      88691   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.690E-03   AT NODE      88107   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.693E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   4.48    ,  FRACTION OF STEP COMPLETED  0.780    
 STEP TIME COMPLETED        558.    ,  TOTAL TIME COMPLETED        2.358E+03


  INCREMENT    82 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.520    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.900    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.00    
 LARGEST RESIDUAL FORCE             2.644E-03   AT NODE       3239   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.448E-07   AT NODE      29595   DOF  2
   INSTANCE: CHD2                                                                            
 ESTIMATE OF DISP.    CORRECTION   -3.606E-09
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE HEAT FLUX                  9.999E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.321E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.182E-02   AT NODE      88691   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.572E-03   AT NODE      28916   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.182E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.790    
 STEP TIME COMPLETED        566.    ,  TOTAL TIME COMPLETED        2.366E+03


  INCREMENT    83 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.120    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.480    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.01    
 LARGEST RESIDUAL FORCE            -1.487E-03   AT NODE      10311   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.362E-07   AT NODE      88829   DOF  2
   INSTANCE: CHD2                                                                            
 ESTIMATE OF DISP.    CORRECTION   -2.044E-09
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE HEAT FLUX                  9.935E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.770E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.568E-02   AT NODE      29288   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.912E-03   AT NODE      80064   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.568E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.800    
 STEP TIME COMPLETED        573.    ,  TOTAL TIME COMPLETED        2.373E+03


  INCREMENT    84 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.140    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.520    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.01    
 LARGEST RESIDUAL FORCE             4.051E-03   AT NODE       9777   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.055E-07   AT NODE      29429   DOF  2
   INSTANCE: CHD2                                                                            
 ESTIMATE OF DISP.    CORRECTION   -9.634E-09
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE HEAT FLUX                  9.871E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.802E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.973E-02   AT NODE      88690   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.151E-03   AT NODE      88311   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.973E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.810    
 STEP TIME COMPLETED        580.    ,  TOTAL TIME COMPLETED        2.380E+03


  INCREMENT    85 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.050    
       ELAPSED SYSTEM TIME (SEC)    =  0.34000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.390    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.02    
 LARGEST RESIDUAL FORCE            -1.891E-02   AT NODE       8962   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        9.060E-07   AT NODE      29086   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.807E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.840E-15   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -5.453E-02   AT NODE      88690   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.241E-03   AT NODE      19151   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.160    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.520    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.02    
 LARGEST RESIDUAL FORCE            -1.665E-03   AT NODE       3340   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.138E-08   AT NODE      29435   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.807E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.234E-16   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -5.439E-02   AT NODE      88690   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.165E-02   AT NODE      23592   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -5.439E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.820    
 STEP TIME COMPLETED        587.    ,  TOTAL TIME COMPLETED        2.387E+03


  INCREMENT    86 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.180    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.570    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.02    
 LARGEST RESIDUAL FORCE             8.552E-03   AT NODE       9721   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.329E-06   AT NODE      29434   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.743E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.807E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -6.210E-02   AT NODE      88685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.101E-02   AT NODE      88504   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.940    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.330    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.02    
 LARGEST RESIDUAL FORCE             8.101E-04   AT NODE     132403   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -4.996E-08   AT NODE      29434   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.743E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         3.846E-16   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -6.189E-02   AT NODE      88685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.723E-02   AT NODE      23886   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -6.189E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.830    
 STEP TIME COMPLETED        594.    ,  TOTAL TIME COMPLETED        2.394E+03


  INCREMENT    87 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.090    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.490    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.03    
 LARGEST RESIDUAL FORCE             -1.45       AT NODE       6143   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.247E-06   AT NODE      29595   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.680E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.087E-14   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -7.432E-02   AT NODE      29282   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.671E-02   AT NODE      29096   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.040    
       ELAPSED SYSTEM TIME (SEC)    =  0.34000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.380    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.03    
 LARGEST RESIDUAL FORCE            -0.106       AT NODE       9680   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -7.014E-08   AT NODE      29266   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.680E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         7.448E-16   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -7.404E-02   AT NODE      29282   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.900E-02   AT NODE      23593   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.120    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.500    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.03    
 LARGEST RESIDUAL FORCE            -1.524E-02   AT NODE       6356   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.961E-08   AT NODE      28894   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.680E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.739E-16   AT NODE       3542   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -7.406E-02   AT NODE      29282   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.182E-04   AT NODE      89135   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.930    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.300    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.03    
 LARGEST RESIDUAL FORCE             6.215E-04   AT NODE       5960   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.791E-05   AT NODE       4937   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.860E-10   AT NODE       7158   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.680E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.442E-17   AT NODE       3542   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -7.407E-02   AT NODE      29282   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.428E-04   AT NODE      23592   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -7.407E-02

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.840    
 STEP TIME COMPLETED        601.    ,  TOTAL TIME COMPLETED        2.401E+03


  INCREMENT    88 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.110    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.470    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.45       TIME AVG. FORCE        1.03    
 LARGEST RESIDUAL FORCE             -12.5       AT NODE       1355   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.381E-05   AT NODE      23700   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.167E-06   AT NODE      29887   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.621E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.837E-14   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -9.033E-02   AT NODE      29282   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.440E-02   AT NODE      88442   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.930    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.290    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.03    
 LARGEST RESIDUAL FORCE            -0.142       AT NODE       2934   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.822E-05   AT NODE      23699   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.603E-06   AT NODE      29440   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.620E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.903E-14   AT NODE      10637   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.124       AT NODE      22371   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.339E-02   AT NODE      23883   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.060    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.420    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.03    
 LARGEST RESIDUAL FORCE             2.025E-03   AT NODE      10788   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.823E-05   AT NODE      23699   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.519E-07   AT NODE      23810   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.616E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.855E-16   AT NODE       9685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -8.993E-02   AT NODE      29282   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        6.322E-02   AT NODE      23883   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -8.993E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.850    
 STEP TIME COMPLETED        609.    ,  TOTAL TIME COMPLETED        2.409E+03


  INCREMENT    89 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.930    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.300    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.04    
 LARGEST RESIDUAL FORCE            -0.190       AT NODE       2233   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.858E-05   AT NODE      23572   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.704E-07   AT NODE      29747   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.553E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         6.002E-15   AT NODE        158   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.106       AT NODE      29445   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        9.649E-02   AT NODE      23888   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.990    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.370    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.04    
 LARGEST RESIDUAL FORCE             3.147E-02   AT NODE      10807   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.853E-05   AT NODE      23571   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.732E-08   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.553E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.494E-16   AT NODE         79   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.106       AT NODE      29445   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.826E-02   AT NODE     132186   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.870    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.260    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.04    
 LARGEST RESIDUAL FORCE            -7.864E-03   AT NODE       9690   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.853E-05   AT NODE      23571   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.374E-09   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.553E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.885E-17   AT NODE        158   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.106       AT NODE      29445   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.143E-03   AT NODE     126737   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.740    
       ELAPSED SYSTEM TIME (SEC)    =  0.34000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.080    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.04    
 LARGEST RESIDUAL FORCE            -3.878E-04   AT NODE      10454   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.853E-05   AT NODE      23571   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.130E-09   AT NODE      80130   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.553E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.504E-17   AT NODE        118   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.106       AT NODE      29445   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.461E-03   AT NODE     132378   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.106    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.860    
 STEP TIME COMPLETED        616.    ,  TOTAL TIME COMPLETED        2.416E+03


  INCREMENT    90 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.680    
       ELAPSED SYSTEM TIME (SEC)    =  0.42000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.100    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.04    
 LARGEST RESIDUAL FORCE            -0.223       AT NODE        958   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.885E-05   AT NODE      29745   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.533E-07   AT NODE      29756   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.490E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.966E-15   AT NODE        158   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.120       AT NODE      29440   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.809E-02   AT NODE      22134   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   93.840    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.230    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.04    
 LARGEST RESIDUAL FORCE            -2.568E-02   AT NODE      11267   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.879E-05   AT NODE      29745   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -9.012E-08   AT NODE      23767   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.490E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.492E-16   AT NODE        158   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.119       AT NODE      29440   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.919E-02   AT NODE      23884   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.050    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.430    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.04    
 LARGEST RESIDUAL FORCE            -6.177E-03   AT NODE       9878   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.879E-05   AT NODE      29601   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.609E-09   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.490E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.513E-17   AT NODE         79   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.119       AT NODE      29440   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.038E-02   AT NODE      28898   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.140    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.540    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.04    
 LARGEST RESIDUAL FORCE            -4.600E-04   AT NODE      10009   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.879E-05   AT NODE      29601   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.783E-10   AT NODE      79312   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.490E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.267E-18   AT NODE         79   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.119       AT NODE      29440   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        7.862E-05   AT NODE      28898   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.119    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.870    
 STEP TIME COMPLETED        623.    ,  TOTAL TIME COMPLETED        2.423E+03


  INCREMENT    91 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.070    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.430    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.05    
 LARGEST RESIDUAL FORCE            -0.157       AT NODE        720   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.912E-05   AT NODE      30036   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.699E-07   AT NODE      29607   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.428E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.487E-15   AT NODE        158   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.131       AT NODE      89000   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.696E-02   AT NODE      22131   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.300    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.700    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.05    
 LARGEST RESIDUAL FORCE            -4.764E-02   AT NODE       8558   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.906E-05   AT NODE      29888   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -9.248E-08   AT NODE      29747   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.428E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.642E-16   AT NODE        158   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.131       AT NODE      89000   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.318E-02   AT NODE      22289   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.330    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.700    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.05    
 LARGEST RESIDUAL FORCE            -2.308E-03   AT NODE         86   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.906E-05   AT NODE      29894   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.270E-09   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.428E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.733E-17   AT NODE         79   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.131       AT NODE      89000   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.109E-02   AT NODE      28708   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.131    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.880    
 STEP TIME COMPLETED        630.    ,  TOTAL TIME COMPLETED        2.430E+03


  INCREMENT    92 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   100.27    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   100.64    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.05    
 LARGEST RESIDUAL FORCE            -0.224       AT NODE        992   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.939E-05   AT NODE      19589   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.717E-07   AT NODE      29437   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.365E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         3.972E-15   AT NODE        158   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.141       AT NODE      29738   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.533E-02   AT NODE      23253   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.590    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.980    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.05    
 LARGEST RESIDUAL FORCE            -2.335E-02   AT NODE      14043   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.932E-05   AT NODE      19589   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        9.924E-08   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.365E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.684E-16   AT NODE        158   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.140       AT NODE      29738   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.587E-02   AT NODE      51930   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.800    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.170    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.43       TIME AVG. FORCE        1.05    
 LARGEST RESIDUAL FORCE             4.020E-03   AT NODE       9803   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.932E-05   AT NODE      19589   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.440E-09   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.365E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.263E-16   AT NODE     126761   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.140       AT NODE      29738   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.188E-02   AT NODE      28714   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.140    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.890    
 STEP TIME COMPLETED        637.    ,  TOTAL TIME COMPLETED        2.437E+03


  INCREMENT    93 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.490    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.890    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.42       TIME AVG. FORCE        1.05    
 LARGEST RESIDUAL FORCE            -0.174       AT NODE         41   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.963E-05   AT NODE      30425   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.861E-07   AT NODE      29447   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.303E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.209E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.155       AT NODE      30066   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.600E-02   AT NODE      28714   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.470    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.830    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.42       TIME AVG. FORCE        1.05    
 LARGEST RESIDUAL FORCE            -1.904E-02   AT NODE       9725   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.956E-05   AT NODE      30425   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.087E-07   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.303E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.838E-16   AT NODE        158   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.154       AT NODE      30066   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.093E-02   AT NODE      28719   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.380    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.740    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.42       TIME AVG. FORCE        1.05    
 LARGEST RESIDUAL FORCE            -1.581E-03   AT NODE     134919   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.955E-05   AT NODE      30425   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.048E-09   AT NODE      81967   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.303E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.048E-17   AT NODE      10824   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.154       AT NODE      30066   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.203E-03   AT NODE      51936   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.154    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.900    
 STEP TIME COMPLETED        644.    ,  TOTAL TIME COMPLETED        2.444E+03


  INCREMENT    94 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.460    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.830    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.42       TIME AVG. FORCE        1.06    
 LARGEST RESIDUAL FORCE             0.203       AT NODE       8454   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.987E-05   AT NODE      30293   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.212E-07   AT NODE      29453   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.242E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.366E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.161       AT NODE      30066   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.536E-02   AT NODE      21966   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.880    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.250    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.42       TIME AVG. FORCE        1.06    
 LARGEST RESIDUAL FORCE            -4.958E-02   AT NODE       9679   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.979E-05   AT NODE      16892   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.194E-07   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.242E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.443E-16   AT NODE        158   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.161       AT NODE      30066   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.237E-02   AT NODE      22363   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.480    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.860    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.42       TIME AVG. FORCE        1.06    
 LARGEST RESIDUAL FORCE            -1.731E-03   AT NODE     134342   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         1.979E-05   AT NODE      16892   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.812E-09   AT NODE      81967   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.242E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.157E-17   AT NODE         79   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.161       AT NODE      30066   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.189E-03   AT NODE      73079   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.161    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.910    
 STEP TIME COMPLETED        652.    ,  TOTAL TIME COMPLETED        2.452E+03


  INCREMENT    95 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.950    
       ELAPSED SYSTEM TIME (SEC)    =  0.34000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.290    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.41       TIME AVG. FORCE        1.06    
 LARGEST RESIDUAL FORCE            -0.260       AT NODE        618   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.009E-05   AT NODE      30297   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.282E-07   AT NODE      29281   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.181E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.061E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.171       AT NODE      30290   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.403E-02   AT NODE      20297   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.480    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.860    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.41       TIME AVG. FORCE        1.06    
 LARGEST RESIDUAL FORCE            -9.585E-02   AT NODE         42   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.000E-05   AT NODE      30295   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.364E-07   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.181E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.400E-16   AT NODE        118   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.170       AT NODE      30290   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.059E-02   AT NODE      23259   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.250    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.610    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.41       TIME AVG. FORCE        1.06    
 LARGEST RESIDUAL FORCE             4.707E-03   AT NODE      10332   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.000E-05   AT NODE      30295   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.991E-09   AT NODE      81966   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.181E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.225E-16   AT NODE         79   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.171       AT NODE      30290   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.399E-02   AT NODE      28727   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.171    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.920    
 STEP TIME COMPLETED        659.    ,  TOTAL TIME COMPLETED        2.459E+03


  INCREMENT    96 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.480    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.840    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.41       TIME AVG. FORCE        1.07    
 LARGEST RESIDUAL FORCE            -0.229       AT NODE         62   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.031E-05   AT NODE      30167   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.770E-07   AT NODE      29291   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.120E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.440E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.180       AT NODE      30423   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.390E-02   AT NODE      46716   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.650    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   96.020    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.41       TIME AVG. FORCE        1.07    
 LARGEST RESIDUAL FORCE            -2.021E-02   AT NODE        115   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.021E-05   AT NODE      30163   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.514E-07   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.120E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.339E-16   AT NODE       9685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.179       AT NODE      30423   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.256E-02   AT NODE      23260   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.240    
       ELAPSED SYSTEM TIME (SEC)    =  0.33000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.570    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.41       TIME AVG. FORCE        1.07    
 LARGEST RESIDUAL FORCE            -3.234E-03   AT NODE      10398   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.021E-05   AT NODE      30163   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.247E-09   AT NODE      81966   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.120E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.178E-17   AT NODE     122236   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.179       AT NODE      30423   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.200E-02   AT NODE      28934   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.179    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.930    
 STEP TIME COMPLETED        666.    ,  TOTAL TIME COMPLETED        2.466E+03


  INCREMENT    97 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.270    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.630    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.40       TIME AVG. FORCE        1.07    
 LARGEST RESIDUAL FORCE            -0.222       AT NODE      10398   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.052E-05   AT NODE      30037   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.184E-07   AT NODE      29083   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.060E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.203E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.183       AT NODE      30424   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.232E-02   AT NODE      23872   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.120    
       ELAPSED SYSTEM TIME (SEC)    =  0.35000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.470    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.40       TIME AVG. FORCE        1.07    
 LARGEST RESIDUAL FORCE            -1.904E-02   AT NODE      10178   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.041E-05   AT NODE      30170   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.686E-07   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.060E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.734E-16   AT NODE       9685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.183       AT NODE      30424   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.393E-02   AT NODE      23871   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.780    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.170    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.40       TIME AVG. FORCE        1.07    
 LARGEST RESIDUAL FORCE             1.488E-03   AT NODE      64801   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.041E-05   AT NODE      30170   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.027E-08   AT NODE      81965   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.060E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.220E-17   AT NODE     127499   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.183       AT NODE      30424   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.016E-03   AT NODE      20460   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.183    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.940    
 STEP TIME COMPLETED        673.    ,  TOTAL TIME COMPLETED        2.473E+03


  INCREMENT    98 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.140    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.540    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.39       TIME AVG. FORCE        1.07    
 LARGEST RESIDUAL FORCE            -0.146       AT NODE       2233   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.072E-05   AT NODE      29891   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.708E-07   AT NODE      29083   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.001E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.133E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.193       AT NODE      19577   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.125E-02   AT NODE      20302   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   95.180    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.540    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.40       TIME AVG. FORCE        1.07    
 LARGEST RESIDUAL FORCE            -2.462E-02   AT NODE        158   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.061E-05   AT NODE      30039   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.886E-07   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.001E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         6.169E-16   AT NODE       9685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.193       AT NODE      19577   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.319E-02   AT NODE      88143   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.180    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.550    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.40       TIME AVG. FORCE        1.07    
 LARGEST RESIDUAL FORCE            -1.908E-03   AT NODE      74874   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.061E-05   AT NODE      30039   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.237E-08   AT NODE      23894   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.001E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.475E-16   AT NODE     100728   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.193       AT NODE      19577   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.400E-02   AT NODE      88147   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.193    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.950    
 STEP TIME COMPLETED        680.    ,  TOTAL TIME COMPLETED        2.480E+03


  INCREMENT    99 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.500    
       ELAPSED SYSTEM TIME (SEC)    =  0.41000    
       ELAPSED TOTAL CPU TIME (SEC) =   94.910    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.39       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE            -0.144       AT NODE      10365   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.094E-05   AT NODE      29895   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        9.848E-07   AT NODE      28882   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.942E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.076E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.194       AT NODE      30165   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.145E-02   AT NODE      88147   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   94.600    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   95.000    
       ELAPSED WALLCLOCK TIME (SEC) =           12

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.39       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE            -3.877E-02   AT NODE          8   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.081E-05   AT NODE      29895   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.100E-07   AT NODE      22433   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.942E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         8.135E-16   AT NODE       9685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.194       AT NODE      30165   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.095E-02   AT NODE      88149   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   102.82    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   103.18    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.39       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE             2.998E-03   AT NODE      74969   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.080E-05   AT NODE      29895   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.478E-08   AT NODE      23893   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  8.942E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.873E-16   AT NODE      95286   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.194       AT NODE      30165   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.588E-02   AT NODE      88149   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.194    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.960    
 STEP TIME COMPLETED        687.    ,  TOTAL TIME COMPLETED        2.487E+03


  INCREMENT   100 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   104.94    
       ELAPSED SYSTEM TIME (SEC)    =  0.41000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.35    
       ELAPSED WALLCLOCK TIME (SEC) =           14

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.38       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE             0.178       AT NODE         11   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.116E-05   AT NODE      29748   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.160E-06   AT NODE      29162   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.884E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         3.951E-15   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.202       AT NODE      30037   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        7.853E-02   AT NODE      23867   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   106.84    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   107.23    
       ELAPSED WALLCLOCK TIME (SEC) =           14

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.38       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE             2.061E-02   AT NODE      10857   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.101E-05   AT NODE      29748   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.392E-07   AT NODE      28892   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.884E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.100E-15   AT NODE       9685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.201       AT NODE      30037   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.694E-02   AT NODE      72914   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   104.59    
       ELAPSED SYSTEM TIME (SEC)    =  0.41000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.00    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.38       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE             5.759E-03   AT NODE      10857   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.101E-05   AT NODE      29746   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.089E-08   AT NODE      23896   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.884E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.408E-16   AT NODE     111433   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.201       AT NODE      30037   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.334E-02   AT NODE      88153   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   104.27    
       ELAPSED SYSTEM TIME (SEC)    =  0.45000    
       ELAPSED TOTAL CPU TIME (SEC) =   104.72    
       ELAPSED WALLCLOCK TIME (SEC) =           14

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       1.38       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE            -3.490E-04   AT NODE     118428   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.101E-05   AT NODE      29746   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.261E-09   AT NODE      88213   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  8.884E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.069E-17   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.201       AT NODE      30037   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        5.170E-03   AT NODE     111433   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.201    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.970    
 STEP TIME COMPLETED        695.    ,  TOTAL TIME COMPLETED        2.495E+03


  INCREMENT   101 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   105.32    
       ELAPSED SYSTEM TIME (SEC)    =  0.41000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.73    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.37       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE             0.297       AT NODE      10857   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.141E-05   AT NODE      29598   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.551E-06   AT NODE      28991   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.827E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.129E-15   AT NODE      14056   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.209       AT NODE      29891   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        7.818E-02   AT NODE      88155   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   105.46    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.86    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.37       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE             3.786E-02   AT NODE      10306   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.124E-05   AT NODE      29598   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.268E-07   AT NODE      29184   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.827E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.536E-15   AT NODE       9685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.208       AT NODE      29891   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.361E-02   AT NODE      28757   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   104.97    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.36    
       ELAPSED WALLCLOCK TIME (SEC) =           14

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.37       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE            -5.070E-03   AT NODE      10371   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.124E-05   AT NODE      29598   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.453E-08   AT NODE      29013   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  8.827E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.024E-16   AT NODE      41703   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.208       AT NODE      29891   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.246E-02   AT NODE      31417   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.208    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.980    
 STEP TIME COMPLETED        702.    ,  TOTAL TIME COMPLETED        2.502E+03


  INCREMENT   102 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   104.74    
       ELAPSED SYSTEM TIME (SEC)    =  0.40000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.14    
       ELAPSED WALLCLOCK TIME (SEC) =           14

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.36       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE             0.344       AT NODE      10858   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.170E-05   AT NODE      29436   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.109E-06   AT NODE      28806   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.771E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.538E-15   AT NODE      14056   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.217       AT NODE      29746   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        7.727E-02   AT NODE      28759   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   104.91    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.29    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.36       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE             7.574E-02   AT NODE      10856   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.150E-05   AT NODE      29436   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -4.462E-07   AT NODE      29001   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.771E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.046E-15   AT NODE       9685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.217       AT NODE      29746   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.300E-02   AT NODE      88358   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   105.28    
       ELAPSED SYSTEM TIME (SEC)    =  0.39000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.67    
       ELAPSED WALLCLOCK TIME (SEC) =           14

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.36       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE             1.599E-02   AT NODE      10377   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.150E-05   AT NODE      29436   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.961E-08   AT NODE      28811   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.770E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.170E-16   AT NODE      20467   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.217       AT NODE      29746   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.508E-02   AT NODE      88358   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   104.91    
       ELAPSED SYSTEM TIME (SEC)    =  0.38000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.29    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       1.36       TIME AVG. FORCE        1.08    
 LARGEST RESIDUAL FORCE            -2.631E-03   AT NODE      10379   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.150E-05   AT NODE      29436   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.183E-08   AT NODE      88213   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  8.770E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.148E-17   AT NODE      10858   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.217       AT NODE      29746   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        6.051E-03   AT NODE      20467   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.217    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED  0.990    
 STEP TIME COMPLETED        709.    ,  TOTAL TIME COMPLETED        2.509E+03


  INCREMENT   103 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.16    
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   104.97    
       ELAPSED SYSTEM TIME (SEC)    =  0.37000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.34    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.36       TIME AVG. FORCE        1.09    
 LARGEST RESIDUAL FORCE            -0.599       AT NODE      10379   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.211E-05   AT NODE      10200   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.882E-06   AT NODE      23512   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.715E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.433E-15   AT NODE      14056   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.228       AT NODE      29598   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -8.686E-02   AT NODE      88834   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   105.42    
       ELAPSED SYSTEM TIME (SEC)    =  0.36000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.78    
       ELAPSED WALLCLOCK TIME (SEC) =           14

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.36       TIME AVG. FORCE        1.09    
 LARGEST RESIDUAL FORCE             0.194       AT NODE      10104   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.191E-05   AT NODE      10200   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.959E-07   AT NODE      28806   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.715E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.686E-15   AT NODE       9685   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.228       AT NODE      29598   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.251E-02   AT NODE      88364   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   104.67    
       ELAPSED SYSTEM TIME (SEC)    =  0.41000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.08    
       ELAPSED WALLCLOCK TIME (SEC) =           14

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.36       TIME AVG. FORCE        1.09    
 LARGEST RESIDUAL FORCE            -4.775E-02   AT NODE      10858   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.190E-05   AT NODE      10200   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.631E-07   AT NODE      28807   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.715E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         3.669E-16   AT NODE      14056   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.228       AT NODE      29598   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.245E-02   AT NODE      90779   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   105.71    
       ELAPSED SYSTEM TIME (SEC)    =  0.42000    
       ELAPSED TOTAL CPU TIME (SEC) =   106.13    
       ELAPSED WALLCLOCK TIME (SEC) =           13

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       1.36       TIME AVG. FORCE        1.09    
 LARGEST RESIDUAL FORCE             1.074E-02   AT NODE       9881   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.190E-05   AT NODE      10200   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.180E-08   AT NODE      88209   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  8.715E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.430E-16   AT NODE     106200   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.228       AT NODE      29598   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.509E-02   AT NODE      88364   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 8 THREADS
 	NUMBER OF EQUATIONS:     547080
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.252e+12

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   104.63    
       ELAPSED SYSTEM TIME (SEC)    =  0.44000    
       ELAPSED TOTAL CPU TIME (SEC) =   105.07    
       ELAPSED WALLCLOCK TIME (SEC) =           14

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       1.36       TIME AVG. FORCE        1.09    
 LARGEST RESIDUAL FORCE             1.070E-03   AT NODE      15010   DOF  3
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF DISP.         2.190E-05   AT NODE      10200   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.235E-09   AT NODE      28801   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  8.715E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.145E-16   AT NODE      73639   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.228       AT NODE      29598   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.336E-03   AT NODE      47317   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.228    

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.16    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED        716.    ,  TOTAL TIME COMPLETED        2.516E+03


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF        108  INCREMENTS
                       0  CUTBACKS IN AUTOMATIC INCREMENTATION
                     214  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     214  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                     214  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       6  WARNING MESSAGES DURING USER INPUT PROCESSING
                23489949  WARNING MESSAGES DURING ANALYSIS
                23489735  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                       0  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =     2.97E+04
       SYSTEM TIME (SEC)    =     5.16E+02
       TOTAL CPU TIME (SEC) =     3.02E+04
       WALLCLOCK TIME (SEC) =         4444
       MEMORY PEAK (GB)     =           17
